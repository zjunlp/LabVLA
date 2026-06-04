"""RoboInter-VQA dataset adapter.

Reads LLaVA-format JSON (image + question + answer) from
`/all-flash-data/Pretrain_Data/RoboInter-VQA/` and emits LabVLA sample dicts
that flow through the existing transform chain alongside action samples.

Design choices:
  * Single-loader heterogeneous-batch approach (π0.5 + OpenTau pattern). VQA
    samples have NO action / state — we emit zero-filled placeholders + set
    `action_is_pad` all True so the model's flow-matching MSE is auto-skipped.
    Only `annotation.vqa_answer` provides the supervision signal (CE via
    `vlm.lm_head`, identical machinery to RoboInter's `annotation.unified`).
  * `has_real_state=False` flag is read by `DiscretizeStateTransformFn` to
    skip discretization — VQA samples must NOT have a fake "<state>128 128
    ...</state>" string injected into their question prompt.
  * Single image per sample for V1 (Generation, Understanding). Task_planning
    multi-frame samples are reduced to the LAST frame in V1 (full multi-frame
    support is V2 — requires Qwen3-VL multi-image packing). The adapter
    accepts but ignores extra image paths.
  * Image path resolution: relative to `image_root` (typically
    `/all-flash-data/Pretrain_Data/RoboInter-VQA`).
  * Conversations format: `[{from: "human" or "huamn", value: ...},
    {from: "gpt", value: ...}]`. Typo "huamn" appears in some val files —
    handled.

Schema requirement: a paired schema must declare
  AnnotationLossSpec(field="annotation.vqa_answer",
                     loss_type="ce_text",
                     weight=1.0,
                     max_length=...).
See `schemas/robointer_vqa.py`.
"""
from __future__ import annotations

import io
import json
import logging
import os
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from src.dataset.adapters.robointer_token_budget import ROBOINTER_BUDGET, RoboInterTokenBudget
from utils.storage_retry import run_with_storage_retry, storage_path_exists


# Conversations role tags. The dataset has a typo "huamn" in some val files.
_USER_ROLES = {"human", "huamn", "user"}
_ASSISTANT_ROLES = {"gpt", "assistant"}
_JSONL_OFFSETS_SUFFIX = ".offsets.npy"
_VQA_SKIP_RETRY_STRIDES = (1, 37, 1009)

# Process-wide bad-record counter. The per-dataset
# `_missing_image_fallback_count` resets when a dataset object is rebuilt (e.g.
# on resume) and does not aggregate across VQA dataset instances in the same
# worker process. This module-level counter keeps the silent dilution of the
# batch (bad records replaced by zero-supervision sentinels) observable. It does
# not change the default sentinel behavior and does not hard-fail.
_VQA_BAD_RECORD_COUNT = 0
# Rate-limit policy: always log the first N occurrences, then every Nth, so a
# steady stream of bad records keeps surfacing instead of going silent after
# the initial burst.
_VQA_BAD_RECORD_LOG_FIRST_N = 10
_VQA_BAD_RECORD_LOG_EVERY = 1000


def _note_vqa_bad_record() -> int:
    """Increment the process-wide bad-record counter and return the new total.

    Returns the post-increment count so the caller can decide (via the
    rate-limit policy) whether to emit a warning for this occurrence.
    """
    global _VQA_BAD_RECORD_COUNT
    _VQA_BAD_RECORD_COUNT += 1
    return _VQA_BAD_RECORD_COUNT


def _should_log_vqa_bad_record(total: int) -> bool:
    return total <= _VQA_BAD_RECORD_LOG_FIRST_N or (total % _VQA_BAD_RECORD_LOG_EVERY == 0)

# Per-task-family answer length cap (sourced from ROBOINTER_BUDGET).
_TASK_FAMILY_TO_ANSWER_LEN: dict[str, str] = {
    "understanding":     "vqa_understanding_answer_max_length",
    "generation_short":  "vqa_generation_short_answer_max_length",
    "generation_traj":   "vqa_generation_traj_answer_max_length",
    "task_planning":     "vqa_task_planning_answer_max_length",
}


def _resolve_answer_max_length(task_family: str, budget: RoboInterTokenBudget) -> int:
    if task_family not in _TASK_FAMILY_TO_ANSWER_LEN:
        raise ValueError(
            f"unknown task_family {task_family!r}; valid: "
            f"{sorted(_TASK_FAMILY_TO_ANSWER_LEN)}"
        )
    return getattr(budget, _TASK_FAMILY_TO_ANSWER_LEN[task_family])


def _vqa_skip_max_attempts(dataset_len: int) -> int:
    raw = os.environ.get("LABVLA_VQA_SKIP_MAX_ATTEMPTS", "auto").strip().lower()
    if raw in {"", "auto", "default"}:
        return max(64, min(1024, max(1, int(dataset_len)) // 1000))
    try:
        return max(1, int(raw))
    except ValueError:
        logging.warning(
            "Invalid LABVLA_VQA_SKIP_MAX_ATTEMPTS=%r; using auto policy.", raw
        )
        return max(64, min(1024, max(1, int(dataset_len)) // 1000))


def _vqa_skip_offset(attempt: int) -> int:
    if int(attempt) <= 0:
        return 0
    stride = _VQA_SKIP_RETRY_STRIDES[(int(attempt) - 1) % len(_VQA_SKIP_RETRY_STRIDES)]
    multiplier = ((int(attempt) - 1) // len(_VQA_SKIP_RETRY_STRIDES)) + 1
    return stride * multiplier


def _infer_family_from_path(json_path: str) -> str:
    """Path-based family inference for ``task_family='mixed'`` manifests.

    Mapping (matches RoboInter-VQA's directory layout):
      ``Task_planning/...``                       → ``task_planning``
      ``Understanding/...``                       → ``understanding``
      ``Generation/.../*traj_qa*.json``           → ``generation_traj``
      ``Generation/...``                          → ``generation_short``
    """
    p = json_path.replace("\\", "/")
    # Match either absolute (/Task_planning/) or relative (Task_planning/) prefix.
    parts = p.split("/")
    if "Task_planning" in parts:
        return "task_planning"
    if "Understanding" in parts:
        return "understanding"
    if "Generation" in parts:
        # traj_qa serializes 16-keypoint waypoints (answers up to 256 tokens);
        # everything else (current_box, final_box, contact_box, contact_point,
        # gripper_det) is short.
        if "traj_qa" in p:
            return "generation_traj"
        return "generation_short"
    raise ValueError(
        f"_infer_family_from_path: cannot infer task family from path "
        f"{json_path!r}. Expected one of Task_planning/, Understanding/, "
        f"Generation/ in the path."
    )


def _load_image(path: str) -> torch.Tensor:
    """Open + RGB-convert a JPEG/PNG, return as [3, H, W] float tensor in
    [0, 1]. Matches the format LeRobot adapters emit, so the downstream
    `ResizeImagesWithPadFn` (which calls `image.ndim`) works on both VQA and
    action samples uniformly."""
    if not storage_path_exists(path):
        raise FileNotFoundError(f"VQA image missing: {path}")

    def _read_image_bytes() -> bytes:
        with open(path, "rb") as f:
            return f.read()

    try:
        pil = Image.open(
            io.BytesIO(
                run_with_storage_retry(
                    _read_image_bytes,
                    path=path,
                    description="read VQA image",
                )
            )
        ).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise OSError(f"VQA image unreadable: {path}: {exc}") from exc
    # PIL (H, W, 3) uint8 -> torch (3, H, W) float32 in [0, 1].
    import numpy as _np
    arr = _np.asarray(pil, dtype=_np.uint8).copy()
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return tensor


@dataclass
class _VQAMeta:
    """Tiny stand-in for LabVLA's adapter `.meta` interface so the
    TransformedLeRobotDataset hydration path can find a schema. Most fields
    not used by VQA — only `schema` and `stats` matter for hydrate_all."""
    schema: Any
    stats: dict
    fps: float = 1.0
    video_keys: tuple = ()
    features: dict = None

    def __post_init__(self):
        if self.features is None:
            self.features = {}


@dataclass
class _VQARecordSource:
    """One backing record source.

    Small legacy ``.json`` files are held in memory. Large ``.jsonl`` files are
    accessed through a sidecar numpy offset index so multi-rank training does
    not expand multi-GB JSON arrays into Python objects in every process.
    """

    path: str
    start: int
    length: int
    records: list[dict] | None = None
    offsets: Any = None
    family: str | None = None
    answer_max_length: int | None = None


def _jsonl_offsets_path(jsonl_path: str) -> str:
    return f"{jsonl_path}{_JSONL_OFFSETS_SUFFIX}"


def _load_jsonl_offsets(jsonl_path: str) -> Any:
    offsets_path = _jsonl_offsets_path(jsonl_path)
    if not storage_path_exists(offsets_path):
        raise FileNotFoundError(
            f"VQA jsonl offset index missing: {offsets_path}. "
            "Generate it before distributed training; otherwise every rank "
            "would have to scan/load the large metadata file."
        )

    import numpy as _np

    offsets = _np.load(offsets_path, mmap_mode="r")
    if getattr(offsets, "ndim", None) != 1:
        raise ValueError(
            f"VQA jsonl offset index must be 1-D: {offsets_path}, "
            f"got shape={getattr(offsets, 'shape', None)}"
        )
    if len(offsets) <= 0:
        raise ValueError(f"VQA jsonl offset index is empty: {offsets_path}")
    return offsets


def _read_jsonl_record(jsonl_path: str, offset: int) -> dict:
    def _read_line() -> bytes:
        with open(jsonl_path, "rb") as f:
            f.seek(int(offset))
            return f.readline()

    raw = run_with_storage_retry(
        _read_line,
        path=jsonl_path,
        description="read VQA jsonl record",
    )
    if not raw.strip():
        raise ValueError(f"empty VQA jsonl record at {jsonl_path}:{offset}")
    rec = json.loads(raw.decode("utf-8"))
    if not isinstance(rec, dict):
        raise ValueError(
            f"VQA jsonl record at {jsonl_path}:{offset} has type "
            f"{type(rec).__name__}; expected dict"
        )
    return rec


class RoboInterVQADataset(Dataset):
    """Iterable over LLaVA-format VQA records from one or more metadata files.

    Args:
        json_paths: list of .json files (each is a JSON list of records) or
            .jsonl files. JSONL files must have a sidecar
            ``<file>.offsets.npy`` int64 offset index and are read lazily.
        image_root: absolute base dir for resolving relative image paths.
            Records' "images" field is joined onto this root.
        task_family: one of "understanding", "generation_short",
            "generation_traj", "task_planning". Selects the answer-length cap
            from ROBOINTER_BUDGET. (Caller is expected to know which json
            corresponds to which family — there's no auto-classifier.)
        schema: a DatasetSchema (typically `schemas.robointer_vqa.SCHEMA`)
            attached to .meta for hydrate_all consumption.
        budget: RoboInterTokenBudget (default = module singleton).
        max_state_dim / max_action_dim / chunk_size: shape of the zero-filled
            placeholders. MUST match LabVLAConfig values; default = budget.
        primary_image_key: name under which the loaded image is stored. Maps
            to schema.image_mapping for downstream RemapImageKeyTransformFn.
    """

    def __init__(
        self,
        json_paths: Sequence[str],
        image_root: str,
        task_family: str,
        schema: Any,
        budget: RoboInterTokenBudget = ROBOINTER_BUDGET,
        primary_image_key: str = "observation.images.primary",
    ):
        self.image_root = image_root
        self.task_family = task_family
        self.budget = budget
        self.primary_image_key = primary_image_key
        # task_family='mixed' enables per-record family inference from source
        # path. Otherwise a single answer_max_length is shared across all
        # records (V1 single-family behavior).
        self._mixed = (task_family == "mixed")
        if not self._mixed:
            self.answer_max_length = _resolve_answer_max_length(task_family, budget)
        else:
            # Sentinel — never read in mixed mode (per-record _family decides).
            self.answer_max_length = budget.vqa_generation_traj_answer_max_length

        self.records: list[dict] = []
        self._sources: list[_VQARecordSource] = []
        self._source_starts: list[int] = []
        self._num_records = 0
        for jp in json_paths:
            if not storage_path_exists(jp):
                raise FileNotFoundError(f"VQA metadata missing: {jp}")
            fam = _infer_family_from_path(jp) if self._mixed else None
            ans_cap = (
                _resolve_answer_max_length(fam, budget)
                if fam is not None else None
            )
            if jp.endswith(".jsonl"):
                offsets = _load_jsonl_offsets(jp)
                self._append_source(
                    _VQARecordSource(
                        path=jp,
                        start=self._num_records,
                        length=int(len(offsets)),
                        offsets=offsets,
                        family=fam,
                        answer_max_length=ans_cap,
                    )
                )
                continue
            if not jp.endswith(".json"):
                raise ValueError(
                    f"VQA metadata file must end with .json or .jsonl: {jp}"
                )

            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(
                    f"VQA json {jp} top-level type is {type(data).__name__}; "
                    "expected list[dict]"
                )
            if self._mixed:
                # Tag each record with its family + per-family answer cap so
                # __getitem__ doesn't have to re-classify on every call.
                for rec in data:
                    if isinstance(rec, dict):
                        rec.setdefault("_family", fam)
                        rec.setdefault("_answer_max_length", int(ans_cap))
                self.records.extend(data)
            else:
                self.records.extend(data)
            self._append_source(
                _VQARecordSource(
                    path=jp,
                    start=self._num_records,
                    length=len(data),
                    records=data,
                    family=fam,
                    answer_max_length=ans_cap,
                )
            )
        if self._num_records <= 0:
            raise ValueError(
                "RoboInterVQADataset loaded zero records from "
                f"{list(json_paths)!r}"
            )
        # Pre-scanning all images for existence is too slow on the DPC filesystem
        # (1.6M+ stat() calls). Bad records are handled lazily in __getitem__;
        # default is fail-loud so corruption is not hidden by substituting a
        # different training sample.
        # Stand-in `.meta` so TransformedLeRobotDataset / _hydrate_unified finds
        # a schema. stats is empty: VQA samples never go through
        # NormalizeTransformFn (schema declares no state/action keys -> no-op).
        self.meta = _VQAMeta(schema=schema, stats={"_vqa_no_norm": {}})
        # TransformedLeRobotDataset late-discovery uses obj.root.
        self.root = image_root
        self._missing_image_fallback_count = 0

    def __len__(self) -> int:
        return self._num_records

    def _append_source(self, source: _VQARecordSource) -> None:
        if source.length <= 0:
            return
        self._source_starts.append(source.start)
        self._sources.append(source)
        self._num_records += source.length

    def _record_at(self, idx: int) -> dict:
        if idx < 0 or idx >= self._num_records:
            raise IndexError(
                f"VQA record index out of range: idx={idx}, len={self._num_records}"
            )
        source_pos = bisect_right(self._source_starts, idx) - 1
        source = self._sources[source_pos]
        local_idx = idx - source.start
        if source.records is not None:
            rec = source.records[local_idx]
            if not isinstance(rec, dict):
                raise ValueError(
                    f"VQA record at {source.path}[{local_idx}] has type "
                    f"{type(rec).__name__}; expected dict"
                )
        else:
            rec = _read_jsonl_record(source.path, int(source.offsets[local_idx]))

        if source.family is not None:
            rec.setdefault("_family", source.family)
            rec.setdefault("_answer_max_length", int(source.answer_max_length))
        return rec

    @staticmethod
    def _extract_qa(rec: dict) -> tuple[str, str]:
        """Pull (question, answer) text from a LLaVA-format record."""
        question, answer = "", ""
        for turn in rec.get("conversations", []):
            who = str(turn.get("from", "")).lower()
            val = turn.get("value", "")
            if who in _USER_ROLES and not question:
                question = str(val) if val is not None else ""
            elif who in _ASSISTANT_ROLES and not answer:
                answer = str(val) if val is not None else ""
        # LLaVA convention: <image> placeholder marks where the image goes.
        # We embed the image separately via Qwen3-VL processor, so strip the
        # placeholder from the prompt text.
        question = question.replace("<image>\n", "").replace("<image>", "").strip()
        return question, answer

    def __getitem__(self, idx: int) -> dict:
        # VQA metadata can contain stale paths or damaged images. Keep
        # __getitem__(idx) deterministic: do not redirect idx to a neighbor here.
        # If skip is enabled, return a zero-loss sentinel for this same idx; the
        # outer SkipBadSamplesDataset remains responsible for real neighbor retry
        # on non-VQA datasets.
        skip_errors = os.environ.get(
            "LABVLA_VQA_SKIP_BAD_RECORDS", "true"
        ).lower() == "true"
        dataset_len = len(self)
        if dataset_len <= 0:
            raise IndexError("RoboInterVQADataset is empty")
        try:
            return self._getitem_inner(int(idx) % dataset_len)
        except (
            FileNotFoundError,
            OSError,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
            json.JSONDecodeError,
        ) as e:
            if not skip_errors:
                raise
            self._missing_image_fallback_count += 1
            # Rate-limited warning so the silent dilution of the batch (this idx
            # now contributes zero gradient) stays observable.
            total = _note_vqa_bad_record()
            if _should_log_vqa_bad_record(total):
                logging.warning(
                    "RoboInterVQADataset bad record at idx=%s; returning "
                    "zero-loss sentinel for the same idx (process bad-record "
                    "count=%d, this-dataset count=%d). Set "
                    "LABVLA_VQA_SKIP_BAD_RECORDS=false to fail loud. error=%s",
                    idx,
                    total,
                    self._missing_image_fallback_count,
                    e,
                )
            return self._bad_record_sentinel(int(idx), e)

    def _bad_record_sentinel(self, idx: int, exc: BaseException) -> dict:
        """Structurally valid sample with all supervision masks disabled."""
        return {
            "task": "Answer.",
            self.primary_image_key: torch.zeros(3, 224, 224, dtype=torch.float32),
            "observation.state": torch.zeros(self.budget.max_state_dim, dtype=torch.float32),
            "action": torch.zeros(self.budget.chunk_size, self.budget.max_action_dim, dtype=torch.float32),
            "action_is_pad": torch.ones(self.budget.chunk_size, dtype=torch.bool),
            "has_real_state": False,
            "annotation.vqa_answer": "",
            "annotation.vqa_answer_max_length": 1,
            "vqa_bad_record_idx": int(idx),
            "vqa_bad_record_error": f"{type(exc).__name__}: {exc}",
        }

    def _getitem_inner(self, idx: int) -> dict:
        rec = self._record_at(idx)
        question, answer = self._extract_qa(rec)

        image_paths = rec.get("images", [])
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        if not image_paths:
            raise ValueError(
                f"VQA record at idx={idx} has no images. id={rec.get('id', '?')!r}"
            )
        # V1: single-frame only. For task_planning's multi-frame samples, use the
        # last frame (most informative for next-step prediction); V2 will pack
        # all frames into a multi-image Qwen3-VL prompt. Per-record family in
        # mixed mode falls back to self.task_family for legacy manifests.
        record_family = rec.get("_family", self.task_family)
        if record_family == "task_planning" and len(image_paths) > 1:
            chosen = image_paths[-1]
        else:
            chosen = image_paths[0]
        image_path = os.path.join(self.image_root, chosen)
        try:
            image = _load_image(image_path)
        except Exception as exc:
            raise type(exc)(
                f"{exc} (vqa_idx={idx}, id={rec.get('id', '?')!r}, image={chosen!r})"
            ) from exc

        sample: dict[str, Any] = {
            # Prompt for Qwen3VL processor.
            "task": question if question else "Answer.",
            # Single image — RemapImageKeyTransformFn will remap to image0.
            self.primary_image_key: image,
            # Zero-fill state + action. VQA has no proprioception; downstream
            # MSE is skipped via action_is_pad=True.
            "observation.state": torch.zeros(self.budget.max_state_dim, dtype=torch.float32),
            "action": torch.zeros(self.budget.chunk_size, self.budget.max_action_dim, dtype=torch.float32),
            "action_is_pad": torch.ones(self.budget.chunk_size, dtype=torch.bool),
            # Skip state discretization (no real proprioception).
            "has_real_state": False,
            # Supervision target — tokenized by AnnotationTokenizeTransformFn.
            "annotation.vqa_answer": answer,
            # Per-sample max_length override. The schema declares the union upper
            # bound (256, sized for generation_traj); AnnotationTokenizeTransformFn
            # honors this task-family-specific cap via the f"{spec.field}_max_length"
            # convention, clamping real-token CE to the family's answer length. In
            # mixed mode this is the per-record cap; otherwise the constructor cap.
            "annotation.vqa_answer_max_length": int(
                rec.get("_answer_max_length", self.answer_max_length)
            ),
        }
        return sample
