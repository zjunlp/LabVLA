"""LeRobot v3.0 dataset adapter.

v3.0 layout (vs v2.1):
  <root>/meta/info.json        (codebase_version="v3.0")
  <root>/meta/episodes/chunk-XXX/file-NNN.parquet   (per-episode indexing + stats)
  <root>/meta/tasks.parquet    (task_index → task string)
  <root>/meta/stats.json       (global dataset stats — same format as v2.1)
  <root>/data/chunk-XXX/file-NNN.parquet   (shard: many episodes per file)
  <root>/videos/<cam>/chunk-XXX/file-NNN.mp4   (shard: many episodes per file)

Key difference: v3 packs many episodes into each shard. Per-episode row
range is in the episodes meta parquet as `dataset_from_index / dataset_to_index`
— the adapter opens the shard, slices to those rows, and hands a standard
per-episode `pd.DataFrame` to the shared transformation logic.

Shares `LeRobotAdapterBase`'s per-frame transformation pipeline (padding,
task resolution, delta-timestamp expansion) via inheritance; only the
three format-specific methods are defined locally:
  1. __init__ — reads episodes.parquet + tasks.parquet
  2. _load_ep_parquet — slices a shard
  3. _read_video_frame — opens a packed mp4 with from_timestamp offset
"""
from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .base import DatasetMeta
from .lerobot_base import LeRobotAdapterBase, _get_shared_video_cache
from .lerobot_v21 import _ep_starts_lens
from utils.storage_retry import (
    read_parquet_with_storage_retry,
    run_with_storage_retry,
    storage_path_exists,
)

logger = logging.getLogger(__name__)


def _is_missing_scalar(value) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _read_episodes_parquet_v3(meta_root: Path) -> list[dict]:
    """Concat meta/episodes/chunk-*/file-*.parquet → list of per-episode dicts.

    Kept columns (everything we need for slicing + video seek):
      episode_index, length, tasks,
      data/chunk_index, data/file_index, dataset_from_index, dataset_to_index,
      videos/<cam>/chunk_index, videos/<cam>/file_index,
      videos/<cam>/from_timestamp, videos/<cam>/to_timestamp
    """
    def _maybe_decode_ascii_byte_array(seq):
        """Return decoded string if `seq` looks like a NUL-padded UTF-8 byte
        array (TFDS-style fixed-length string encoding used by OXE
        `language_table_*` sub-repos), else None.

        Heuristic tightened to avoid accepting low-id tokenizer output as text:
          (a) every element in [0, 127] (ASCII range);
          (b) NUL padding (trailing zeros) present, OR the array is fully
              printable;
          (c) after stripping NULs, >= 90% printable ASCII / whitespace;
          (d) result is at least 2 chars long.

        Returns None on any failure — caller falls through to the tokenized-task
        path (fail-loud unless LABVLA_ALLOW_TOKENIZED_TASK_COERCION is set).
        """
        if not seq:
            return None
        try:
            ints = [int(x) for x in seq]
        except (TypeError, ValueError):
            return None
        if any(not (0 <= b <= 127) for b in ints):
            return None
        # (b) require NUL-padding evidence (proves TFDS fixed-length encoding)
        # OR a fully printable array (unambiguously a string).
        n_trailing_zeros = 0
        for b in reversed(ints):
            if b == 0:
                n_trailing_zeros += 1
            else:
                break
        printable_or_ws = lambda c: (0x20 <= c <= 0x7E) or c in (0x09, 0x0A, 0x0D)
        non_pad = ints[: len(ints) - n_trailing_zeros] if n_trailing_zeros > 0 else ints
        if not non_pad:
            return None
        all_printable = all(printable_or_ws(c) for c in non_pad)
        if n_trailing_zeros == 0 and not all_printable:
            # No padding AND not fully printable: likely tokenizer output.
            return None
        # (c) >=90% printable within the unpadded prefix.
        n_printable = sum(1 for c in non_pad if printable_or_ws(c))
        if n_printable < 0.9 * len(non_pad):
            return None
        decoded = bytes(non_pad).decode("ascii", errors="replace").rstrip()
        # (d) single-char results are too ambiguous (could be any low-id token).
        if len(decoded) < 2:
            return None
        return decoded

    ep_root = meta_root / "episodes"
    if not ep_root.is_dir():
        raise FileNotFoundError(f"v3 meta/episodes dir missing at {ep_root}")
    paths = sorted(ep_root.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode parquets under {ep_root}")

    # Go through pandas for heterogeneous-schema-safe concat: merged
    # oxe-auge_clean has differing camera-key-sets and two `tasks` column types
    # (list<string> vs list<list<int>>). pa.concat_tables can't union these;
    # pd.concat(sort=False) can.
    dfs_with_paths = [
        (
            p,
            run_with_storage_retry(
                lambda _p=p: pq.read_table(_p).to_pandas(),
                path=p,
                description="read v3 episodes parquet",
            ),
        )
        for p in paths
    ]
    dfs = [df for _, df in dfs_with_paths]

    def _normalize_tasks(val):
        # Normalize the `tasks` column so it's always VLM-tokenizable.
        # list<string> / scalar pass through unchanged. Tokenized
        # (list<list<int>>) tasks fail-loud by default (coercing to repr would
        # train the VLM on digit strings); LABVLA_ALLOW_TOKENIZED_TASK_COERCION=1
        # restores the legacy coercion for inspection only.
        #
        # Exception: OXE `language_table_*` stores strings as NUL-padded ASCII
        # byte arrays (TFDS idiom), not tokenizer output. _maybe_decode_ascii_
        # byte_array decodes those back to text BEFORE the raise.
        if val is None:
            return None
        if isinstance(val, (list, np.ndarray)):
            allow_coerce = os.environ.get(
                "LABVLA_ALLOW_TOKENIZED_TASK_COERCION") == "1"
            out = []
            for item in val:
                if isinstance(item, (list, np.ndarray)):
                    item_list = list(item) if isinstance(item, np.ndarray) else item
                    decoded = _maybe_decode_ascii_byte_array(item_list)
                    if decoded is not None:
                        out.append(decoded)
                    elif allow_coerce:
                        out.append(str(list(item_list)))
                    else:
                        raise ValueError(
                            "v30 adapter: encountered a tokenized (list<int>) "
                            "task element. Coercing it to str() would train the "
                            "VLM on digit strings rather than language. Either "
                            "decode the tokens back to text in the upstream "
                            "dataset, exclude tokenized-task shards from "
                            "training, or set "
                            "LABVLA_ALLOW_TOKENIZED_TASK_COERCION=1 for a "
                            "best-effort inspection-only run."
                        )
                else:
                    out.append(str(item) if item is not None else "")
            return out
        return val

    # Warn loud when heterogeneous `tasks` column types are detected, naming the
    # shards with tokenized (list<list<int>>) tasks so the operator can
    # spot-check before training.
    tokenized_paths: list[Path] = []
    for p, df in dfs_with_paths:
        if "tasks" not in df.columns:
            continue
        sample = next((v for v in df["tasks"].values if v is not None), None)
        if isinstance(sample, (list, np.ndarray)) and len(sample) > 0:
            first = sample[0]
            if isinstance(first, (list, np.ndarray)):
                tokenized_paths.append(p)
    if tokenized_paths:
        from utils.logging_utils import warn_once
        # Heads-up that _normalize_tasks will raise at first read unless
        # LABVLA_ALLOW_TOKENIZED_TASK_COERCION=1.
        warn_once(
            logger,
            ("v30_tasks_heterogeneous", tuple(str(p) for p in tokenized_paths)),
            "[v30-adapter] Heterogeneous tasks column type detected — %d "
            "shard(s) contain tokenized (list<list<int>>) tasks while others "
            "contain list<string>. These will FAIL LOUD on first read; set "
            "LABVLA_ALLOW_TOKENIZED_TASK_COERCION=1 to fall back to legacy "
            "str() coercion (inspection-only — VLM should not be trained on "
            "the resulting digit strings). Affected paths: %s",
            len(tokenized_paths),
            ", ".join(str(p) for p in tokenized_paths[:5])
            + (" ..." if len(tokenized_paths) > 5 else ""),
        )

    for df in dfs:
        if "tasks" in df.columns:
            df["tasks"] = df["tasks"].map(_normalize_tasks)

    df = pd.concat(dfs, axis=0, ignore_index=True, sort=False)
    table = pa.Table.from_pandas(df, preserve_index=False)

    # Keep `videos/<cam>/…` columns verbatim for _read_video_frame; drop
    # `stats/<feature>/*` (per-episode stats the adapter doesn't use).
    keep_prefix = (
        "episode_index", "length", "tasks",
        "data/", "dataset_from_index", "dataset_to_index",
        "videos/",
    )
    keep_cols = [c for c in table.schema.names if any(
        c == p or c.startswith(p) for p in keep_prefix
    )]
    sub = table.select(keep_cols).to_pandas()
    out = [row.to_dict() for _, row in sub.iterrows()]
    # tasks is list<str> (np.ndarray from pandas); take the first (per-ep task).
    for e in out:
        tk = e.get("tasks")
        if isinstance(tk, (list, np.ndarray)) and len(tk) > 0:
            e["task"] = str(tk[0])
        else:
            e["task"] = ""
    out.sort(key=lambda e: int(e["episode_index"]))
    return out


def _read_tasks_parquet_v3(tasks_path: Path) -> dict[int, str]:
    """Parse meta/tasks.parquet → {task_index: task_string}.

    Observed conversion variants:
      (a) columns = {"task_index": int64, "task": string}
      (b) task string as the DataFrame index (`__index_level_0__` in pyarrow),
          `task_index: int64` as a column.
    Handle both.
    """
    if not storage_path_exists(tasks_path):
        return {}
    table = run_with_storage_retry(
        lambda: pq.read_table(tasks_path),
        path=tasks_path,
        description="read v3 tasks parquet",
    )
    df = table.to_pandas()
    if "task" in df.columns and "task_index" in df.columns:
        return {int(r["task_index"]): str(r["task"]) for _, r in df.iterrows()}
    # Variant (b): task string is df.index, task_index is a column.
    if "task_index" in df.columns:
        return {int(r["task_index"]): str(idx) for idx, r in df.iterrows()}
    logger.warning("[v30-adapter] unexpected tasks.parquet schema: cols=%s", list(df.columns))
    return {}


class LeRobotV30Adapter(LeRobotAdapterBase):
    """Adapter for LeRobot v3.0 shard-packed datasets.

    Inherits the shared per-frame transformation pipeline
    (``__getitem__``, row-padding, task resolution, delta-timestamp
    expansion, CRIT-05 ``_is_pad`` postcondition, zero-frame fallback)
    from ``LeRobotAdapterBase`` and overrides only the format-specific
    I/O hooks: ``_load_ep_parquet`` (slices a shard) and
    ``_read_video_frame`` (reads a packed mp4 with ``from_timestamp``
    offset).
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        delta_timestamps: dict | None = None,
        image_transforms=None,
        external_stats: dict | None = None,
        override_schema=None,
        video_backend: str = "pyav",
        episode_filter: list[int] | tuple[int, ...] | None = None,
    ):
        """See LeRobotV21Adapter for episode_filter semantics."""
        self.repo_id = repo_id
        root_p = Path(root)
        if root_p.name != Path(repo_id).name and (root_p / repo_id).exists():
            root_p = root_p / repo_id
        self.root = root_p
        self.meta_root = self.root / "meta"
        self.data_root = self.root / "data"
        self.video_root = self.root / "videos"
        self.delta_timestamps = delta_timestamps or {}
        self.image_transforms = image_transforms
        self.video_backend = video_backend

        with open(self.meta_root / "info.json") as _f:
            info = json.load(_f)
        v = info.get("codebase_version", "")
        if not v.startswith("v3"):
            raise ValueError(
                f"LeRobotV30Adapter: {self.root} is codebase_version={v!r}, "
                f"expected v3.x."
            )
        self._info = info
        self._chunks_size = int(info.get("chunks_size", 1000))
        self._episodes = _read_episodes_parquet_v3(self.meta_root)
        # Preserve unfiltered episodes for shard offsets: the offset is the
        # minimum dataset_from_index of ANY episode in the shard, since parquet
        # rows are numbered 0..N across all packed episodes (not just filtered).
        self._unfiltered_episodes = list(self._episodes)
        self._ep_starts, self._ep_lens = _ep_starts_lens(self._episodes)

        # tasks.parquet → {int: str} (v21 uses tasks.jsonl; same lookup path).
        self._tasks_by_idx = _read_tasks_parquet_v3(self.meta_root / "tasks.parquet")

        # stats.json — same format across v2.1 and v3.
        stats_path = self.meta_root / "stats.json"
        if stats_path.exists():
            from dataset.utils import cast_stats_to_numpy
            with open(stats_path) as _f:
                stats: dict = cast_stats_to_numpy(json.load(_f))
        else:
            stats = {}
        if external_stats:
            stats = {**stats, **external_stats}

        # Schema discovery (same chain as v21).
        from schema import discover_schema, SchemaDiscoveryError
        try:
            schema = discover_schema(
                self.root,
                robot_type=info.get("robot_type"),
                override=override_schema,
            )
        except SchemaDiscoveryError as e:
            logger.warning("[v30-adapter] schema discovery failed: %s", e)
            schema = override_schema

        stats = self.patch_stats_for_next_frame_actions(stats, schema)

        # Filter episodes whose data shard lacks schema-required columns. In v3
        # a shard shares one schema, so check each unique (chunk, file) once and
        # drop episodes pointing at a failed shard.
        if schema is not None:
            read_state_keys = (
                tuple(getattr(schema, "source_state_keys", ()) or ())
                or tuple(getattr(schema, "state_keys", ()) or ())
            )
            read_action_keys = (
                tuple(getattr(schema, "source_action_keys", ()) or ())
                or tuple(getattr(schema, "action_keys", ()) or ())
            )
            needed = list(dict.fromkeys(
                self._required_parquet_key(k)
                for k in (list(read_state_keys) + list(read_action_keys))
            ))
            alt = {"observation.state": "state", "action": "actions"}

            def _shard_ok(ci: int, fi: int) -> bool:
                p = self.data_root / f"chunk-{ci:03d}" / f"file-{fi:03d}.parquet"
                if not p.exists():
                    return False
                try:
                    cols = set(pq.ParquetFile(str(p)).schema_arrow.names)
                except Exception:
                    return False
                for k in needed:
                    if k in cols or alt.get(k, None) in cols:
                        continue
                    return False
                return True

            # Disk-persistent scan cache (v3 variant — keyed by ok_shards).
            # Reuses the v21 infrastructure; v3 has no chunks_size, so -1 sentinel.
            from .lerobot_v21 import (
                _scan_cache_key, _load_scan_cache, _save_scan_cache,
            )
            cache_key = _scan_cache_key(
                self.meta_root,
                schema_id=getattr(schema, "schema_id", "unknown"),
                chunks_size=-1,   # v3 sentinel
                data_root=self.data_root,
            )
            cache_key["required_columns"] = list(needed)
            cached = _load_scan_cache(self.meta_root, cache_key)

            unique_shards = {
                (int(e["data/chunk_index"]), int(e["data/file_index"]))
                for e in self._episodes
            }

            # The cache packs (ci, fi) as ``ci*_PACK_BASE + fi`` — unambiguous
            # only while fi < _PACK_BASE. Assert loudly so >1M-shards-per-chunk
            # datasets fail at launch instead of mis-filtering silently.
            _PACK_BASE = 1_000_000
            max_fi = max((fi for (_, fi) in unique_shards), default=0)
            assert max_fi < _PACK_BASE, (
                f"[v30-adapter] shard fi={max_fi} ≥ cache pack base {_PACK_BASE}; "
                f"cache collision risk — update encoding before reusing."
            )

            if cached is not None:
                # Reuse the v21 cache's `existing_episodes` slot as our
                # ok_shards set (packed ints).
                _packed_ok = cached[1]
                ok_shards = {(p // _PACK_BASE, p % _PACK_BASE) for p in _packed_ok}
                # Intersect with the current episode set, which can drift
                # without a schema change (which would invalidate the cache).
                ok_shards = ok_shards & unique_shards
            else:
                ok_shards = {s for s in unique_shards if _shard_ok(*s)}
                _packed_ok = {ci * _PACK_BASE + fi for (ci, fi) in ok_shards}
                _save_scan_cache(
                    self.meta_root, cache_key, chunk_ok={}, existing_episodes=_packed_ok,
                )

            if ok_shards != unique_shards:
                keep = [e for e in self._episodes
                        if (int(e["data/chunk_index"]), int(e["data/file_index"])) in ok_shards]
                dropped = len(self._episodes) - len(keep)
                logger.warning(
                    "[v30-adapter] %s: dropped %d/%d episodes in %d shards "
                    "lacking schema-required columns (%s)",
                    repo_id, dropped, len(self._episodes),
                    len(unique_shards) - len(ok_shards), needed,
                )
                self._episodes = keep
                self._ep_starts, self._ep_lens = _ep_starts_lens(self._episodes)

        # Task-uniform support: restrict to a user-supplied episode subset.
        if episode_filter is not None:
            allowed = {int(i) for i in episode_filter}
            before = len(self._episodes)
            self._episodes = [
                ep for ep in self._episodes
                if int(ep.get("episode_index", -1)) in allowed
            ]
            self._ep_starts, self._ep_lens = _ep_starts_lens(self._episodes)
            logger.info(
                "[v30-adapter] %s: episode_filter kept %d/%d episodes",
                repo_id, len(self._episodes), before,
            )

        self._drop_terminal_samples_for_next_frame_actions(schema)

        feats = info.get("features", {}) or {}
        video_keys = [
            k for k, val in feats.items()
            if isinstance(val, dict) and val.get("dtype") == "video"
        ]
        # dtype=image columns are PNG-in-parquet (HF Image feature), NOT mp4.
        # Datasets like LabUtopia/Level3_open ship images this way with no
        # videos/ dir; the base __getitem__ decodes them inline via
        # self._image_keys (else the video loop trains on all-black frames).
        image_keys = [
            k for k, val in feats.items()
            if isinstance(val, dict) and val.get("dtype") == "image"
        ]
        self._image_keys = tuple(image_keys)

        _ep_ends = self._ep_starts + self._ep_lens
        if "fps" not in info:
            raise ValueError(
                f"{self.meta_root}/info.json is missing 'fps'. v3.0 adapter "
                "requires an explicit fps to build delta_timestamps; a silent "
                "default of 10 caused action-vs-video drift when the real fps "
                "differs (e.g. 30)."
            )
        self.meta = DatasetMeta(
            robot_type=info.get("robot_type"),
            stats=stats if stats else None,
            fps=float(info["fps"]),
            total_episodes=len(self._episodes),
            total_frames=int(self._ep_lens.sum()),
            features=feats,
            video_keys=video_keys,
            # camera_keys = full set (mp4 video + PNG-in-parquet image);
            # video_keys alone is empty for image-only datasets.
            camera_keys=list(video_keys) + list(image_keys),
            schema=schema,
            episodes={
                "dataset_from_index": self._ep_starts.astype(int).tolist(),
                "dataset_to_index":   _ep_ends.astype(int).tolist(),
            },
        )

        # episode_index → meta row lookup, used by _load_ep_parquet /
        # _read_video_frame to recover shard + timestamp info O(1).
        self._ep_by_idx = {int(e["episode_index"]): e for e in self._episodes}

        # Per-shard global→local offset. `dataset_from_index` is cumulative
        # across the dataset, but each shard's parquet rows are numbered
        # 0..N_shard-1; _load_ep_parquet subtracts this offset before iloc.
        # Computed from UNFILTERED episodes: if episode_filter drops the first
        # episode in a shard, the original min dataset_from_index still applies
        # (parquet rows don't shift).
        self._shard_offsets: dict[tuple[int, int], int] = {}
        for _e in self._unfiltered_episodes:
            _key = (int(_e["data/chunk_index"]), int(_e["data/file_index"]))
            _a = int(_e["dataset_from_index"])
            _cur = self._shard_offsets.get(_key)
            if _cur is None or _a < _cur:
                self._shard_offsets[_key] = _a

        # Minimal column set for `_load_shard`: oxe-auge shards have ~49 columns
        # but we only need schema state/action + task-resolution fields.
        # Projecting cuts I/O ~10x and keeps LRU entries compact.
        needed: set[str] = set()
        if schema is not None:
            read_state_keys = (
                tuple(getattr(schema, "source_state_keys", ()) or ())
                or tuple(getattr(schema, "state_keys", ()) or ())
            )
            read_action_keys = (
                tuple(getattr(schema, "source_action_keys", ()) or ())
                or tuple(getattr(schema, "action_keys", ()) or ())
            )
            for k in read_state_keys:
                needed.add(self._required_parquet_key(k))
            for k in read_action_keys:
                needed.add(self._required_parquet_key(k))
            # Canonical siblings so the pluralization normalization can reach them.
            needed.update({"observation.state", "state", "action", "actions"})
        # Task resolution needs task_index (→ tasks.parquet) or
        # natural_language_instruction; include both.
        needed.update({"task_index", "natural_language_instruction", "task"})
        if schema is not None:
            for spec in getattr(schema, "annotation_losses", ()) or ():
                field = getattr(spec, "field", None)
                if field:
                    needed.add(str(field))
        # Minimal metadata for any downstream transform that inspects index.
        needed.update({"timestamp", "frame_index", "episode_index", "index"})
        # Project dtype=image PNG columns too, else _load_shard drops them and
        # __getitem__ falls through to all-black _zero_frame tensors.
        needed.update(image_keys)
        self._shard_load_columns: tuple[str, ...] | None = (
            tuple(needed) if needed else None
        )

        logger.info(
            "[v30-adapter] %s: %d episodes, %d frames, %d video_cameras, "
            "%d image_cameras, schema=%s, shard_load_columns=%d",
            repo_id, self.meta.total_episodes, self.meta.total_frames,
            len(video_keys), len(image_keys),
            getattr(schema, "schema_id", None),
            len(self._shard_load_columns) if self._shard_load_columns else -1,
        )

        # Warn once per (repo, horizon) if the longest offset clips past most
        # of the shortest episode. See base class.
        self._validate_delta_timestamps_vs_episode_lens()

        # Process-wide PyAV container LRU (shared with v21): frames within a
        # sample share a packed mp4, so caching avoids a fresh av.open+seek per
        # frame. Keys namespaced by adapter id to avoid (vkey, ci, fi) collisions.
        self._video_cache = _get_shared_video_cache()
        self._video_cache_owner_id = id(self)

    def _close_video_containers(self) -> None:
        """Best-effort flush of cached PyAV containers (used at adapter teardown).

        Now drops only entries owned by this adapter from the shared cache;
        other adapters in the same worker keep their entries.
        """
        try:
            self._video_cache.drop_owner(self._video_cache_owner_id)
        except Exception:
            pass

    def __del__(self):
        try:
            self._close_video_containers()
        except Exception:
            pass

    # ---- shard-sliced parquet loader (overrides v21) ----

    @lru_cache(maxsize=64)
    def _load_shard(self, ci: int, fi: int) -> pd.DataFrame:
        """Whole-shard DataFrame, LRU-cached at 64 shards (a BS=64 batch spans
        many short oxe-auge episodes).

        Projects onto `self._shard_load_columns` (schema state/action + task
        cols): a full oxe-auge shard is ~50MB / 49 cols; projection drops it to
        ~5-10MB (~10x I/O reduction).
        """
        p = self.data_root / f"chunk-{ci:03d}" / f"file-{fi:03d}.parquet"
        cols = self._shard_load_columns
        if cols is None:
            return read_parquet_with_storage_retry(p)
        # Sub-repos differ in columns (e.g. natural_language_instruction only
        # in some OXE sources); read only the subset on disk to avoid ArrowInvalid.
        try:
            pf = run_with_storage_retry(
                lambda: pq.ParquetFile(str(p)),
                path=p,
                description="open parquet metadata",
            )
            avail = set(pf.schema_arrow.names)
            proj = [c for c in cols if c in avail]
            return run_with_storage_retry(
                lambda: pf.read(columns=proj).to_pandas(),
                path=p,
                description="read parquet projection",
            )
        except Exception:
            # Column-metadata read failed: full-column read (slow but functional).
            return read_parquet_with_storage_retry(p)

    def _load_ep_parquet(self, ep_idx: int) -> pd.DataFrame:
        ep = self._ep_by_idx[ep_idx]
        ci = int(ep["data/chunk_index"])
        fi = int(ep["data/file_index"])
        a = int(ep["dataset_from_index"])
        b = int(ep["dataset_to_index"])
        # Global cumulative frame indices → shard-local rows: subtract the
        # smallest dataset_from_index of any episode in this shard.
        offset = self._shard_offsets.get((ci, fi), 0)
        return (
            self._load_shard(ci, fi)
            .iloc[a - offset:b - offset]
            .copy()
            .reset_index(drop=True)
        )

    # ---- packed mp4 video loader (overrides v21) ----

    def _read_video_frame(self, ep_idx: int, vkey: str, frame: int) -> torch.Tensor:
        """Read frame N of episode `ep_idx` from a packed mp4 shard.

        v3 packs many episodes into a single mp4. The episode's sub-window
        is given by `videos/<cam>/from_timestamp` and `to_timestamp` in
        the episodes meta. Target pts = from_ts + frame/fps.
        """
        ep = self._ep_by_idx.get(ep_idx)
        if ep is None:
            return self._zero_frame(reason="missing_episode")
        vci_key = f"videos/{vkey}/chunk_index"
        vfi_key = f"videos/{vkey}/file_index"
        vfrom_key = f"videos/{vkey}/from_timestamp"
        if vci_key not in ep or vfi_key not in ep or vfrom_key not in ep:
            # Schema asked for a camera v3 info doesn't declare.
            return self._zero_frame(reason="schema_camera_missing")
        if any(_is_missing_scalar(ep.get(k)) for k in (vci_key, vfi_key, vfrom_key)):
            return self._zero_frame(reason="video_metadata_missing")
        ci = int(ep[vci_key])
        fi = int(ep[vfi_key])
        from_ts = float(ep[vfrom_key])
        p = self.video_root / vkey / f"chunk-{ci:03d}" / f"file-{fi:03d}.mp4"
        if not storage_path_exists(p):
            return self._zero_frame(reason="missing_file")

        import av
        full_key = (self._video_cache_owner_id, vkey, ci, fi)
        video_cache = self._video_cache

        def _drop_cached_container(_exc, _attempt) -> None:
            cached_container = video_cache.pop(full_key)
            if cached_container is not None:
                try:
                    cached_container[0].close()
                except Exception:
                    pass

        def _decode_once():
            cached = video_cache.get(full_key)
            if cached is None:
                container = av.open(str(p))
                stream = container.streams.video[0]
                # "AUTO" hits a libav internal futex deadlock at 128-process
                # scale; "NONE" + thread_count=1 is the only safe production default.
                stream.thread_type = "NONE"
                stream.thread_count = 1
                # LRU eviction (closes evicted container) handled inside .put().
                video_cache.put(full_key, (container, stream))
            else:
                container, stream = cached

            fps_frac = Fraction(stream.average_rate) if stream.average_rate else Fraction(30)
            fps = float(fps_frac)
            tb = stream.time_base

            # Target = episode window start + local frame offset.
            target_sec = from_ts + frame / fps
            target_pts = int(target_sec / tb) if tb else 0
            try:
                container.seek(target_pts, stream=stream)
            except av.AVError:
                pass

            _start_pts = round(Fraction(from_ts) / tb) if tb else 0
            target_global_frame = round(_start_pts * tb * fps_frac) + frame
            for f in container.decode(stream):
                if f.pts is None:
                    continue
                cur = round(f.pts * tb * fps_frac)
                if cur < target_global_frame:
                    continue
                if cur > target_global_frame:
                    break
                img = f.to_ndarray(format="rgb24")
                t = torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0
                if self.image_transforms is not None:
                    apply_seeded = getattr(self.image_transforms, "apply_with_seed", None)
                    if callable(apply_seeded):
                        t = apply_seeded(t, seed_parts=(self.repo_id, ep_idx, vkey, frame))
                    else:
                        t = self.image_transforms(t)
                return t
            return None

        try:
            decoded = run_with_storage_retry(
                _decode_once,
                path=p,
                description="v30 video decode",
                on_retry=_drop_cached_container,
            )
            if decoded is not None:
                return decoded
            cached = video_cache.pop(full_key)
            if cached is not None:
                try:
                    cached[0].close()
                except Exception:
                    pass
            return self._zero_frame(reason="frame_overshoot")
        except Exception as e:
            # Drop the cached container on error; it may be in a bad state.
            cached = video_cache.pop(full_key)
            if cached is not None:
                try:
                    cached[0].close()
                except Exception:
                    pass
            logger.warning("[v30-adapter] video decode failed for %s (%s); zero frame.", p, e)
            return self._zero_frame(reason="decode_error")
