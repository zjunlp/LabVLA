"""Shared base class for LeRobot v21 / v30 adapters.

Factors common per-frame transformation logic into ``LeRobotAdapterBase``.
Subclasses plug in two format-specific methods:

  * ``_load_ep_parquet(ep_idx) -> pd.DataFrame``
        Return a flat, per-episode DataFrame. v21 reads one parquet per
        episode; v30 slices a shard using the episodes meta.
  * ``_read_video_frame(ep_idx, vkey, frame) -> torch.Tensor``
        Return one RGB frame as a ``(3, H, W) float32`` tensor in [0, 1].

Kept separate from ``base.py`` (a minimal abstract boundary) so other
adapter families (e.g. raw HDF5) can speak the same ``BaseAdapter``
interface without inheriting LeRobot plumbing.
"""
from __future__ import annotations

import io
import logging
import os
import threading
from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .base import BaseAdapter

logger = logging.getLogger(__name__)


def _is_missing_scalar_value(value) -> bool:
    """True if ``value`` is a pandas/NumPy missing scalar (NaN/NaT/None).

    Used by task resolution to distinguish a real instruction string from a
    NaN-filled cell (heterogeneous v3 shards leave absent columns as NaN).
    Guarded so array-like inputs (which would make ``pd.isna`` return an
    array) never raise here.
    """
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


# Process-wide PyAV container cache shared across ALL LeRobot adapters (v2.1
# and v3.0). Per-adapter caches summed to 60+ adapters × 64 slots = 3840
# decoders/worker in the 4ds mixture and OOM'd the host; one process-wide LRU
# (cap=LABVLA_VIDEO_CACHE_MAX, default 128) self-tunes via LRU eviction, with
# keys namespaced by ``id(adapter)`` so adapters cannot collide. Also fixes the
# v21 O(av.open per frame) bottleneck: consecutive frames in one chunk reuse a
# single cached container. Each DataLoader worker fork gets its own empty cache.

_VIDEO_CONTAINER_CACHE_MAX = int(os.environ.get("LABVLA_VIDEO_CACHE_MAX", "128"))


class _SharedVideoContainerCache:
    """Process-wide LRU of opened ``av.Container`` instances.

    Shared by both ``LeRobotV21Adapter`` and ``LeRobotV30Adapter``. Replaces
    per-adapter ``OrderedDict[(vkey,...) -> (container, stream)]`` with a
    single shared ``OrderedDict[(adapter_id, *cache_key) -> ...]`` LRU.
    """

    def __init__(self, max_size: int):
        self._cache: "OrderedDict[tuple, tuple]" = OrderedDict()
        self._max = max(1, int(max_size))
        # Precautionary: DataLoader workers run __getitem__ single-threaded.
        self._lock = threading.Lock()

    def get(self, key: tuple):
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                self._cache.move_to_end(key)
            return entry

    def put(self, key: tuple, value: tuple) -> None:
        """Insert; evict LRU and close its container if over cap."""
        to_close = []
        with self._lock:
            self._cache[key] = value
            while len(self._cache) > self._max:
                _evicted_key, evicted = self._cache.popitem(last=False)
                _old_container = evicted[0] if evicted else None
                if _old_container is not None:
                    to_close.append(_old_container)
        for container in to_close:
            try:
                container.close()
            except Exception:
                pass

    def pop(self, key: tuple):
        with self._lock:
            return self._cache.pop(key, None)

    def drop_owner(self, owner_id: int) -> None:
        """Close all entries owned by ``owner_id`` (called from adapter teardown)."""
        to_close = []
        with self._lock:
            stale_keys = [k for k in self._cache.keys() if k and k[0] == owner_id]
            for k in stale_keys:
                entry = self._cache.pop(k, None)
                if entry is not None:
                    to_close.append(entry[0])
        for container in to_close:
            try:
                container.close()
            except Exception:
                pass


# Lazy module-level singleton (one per worker process via fork).
_SHARED_VIDEO_CACHE: "_SharedVideoContainerCache | None" = None


def _get_shared_video_cache() -> _SharedVideoContainerCache:
    global _SHARED_VIDEO_CACHE
    if _SHARED_VIDEO_CACHE is None:
        _SHARED_VIDEO_CACHE = _SharedVideoContainerCache(_VIDEO_CONTAINER_CACHE_MAX)
    return _SHARED_VIDEO_CACHE


class LeRobotAdapterBase(BaseAdapter):
    """Shared transformation pipeline for LeRobot v21 / v30 adapters.

    Concrete subclasses own ``__init__`` (version-specific meta parsing)
    and override the two abstract I/O hooks below. Everything else — row
    → tensor conversion, schema-driven padding, task resolution,
    delta_timestamp expansion, video-frame dispatch, and the CRIT-05
    ``_is_pad`` postcondition — is provided here.
    """

    # Canonical column-name aliases. v2.1 ships two pluralizations of the
    # state/action columns; normalize to the canonical (v3.0) names so
    # downstream transforms see a consistent key set. {canonical: raw_alias};
    # when canonical is missing AND alias is present, alias is copied under
    # canonical. A class constant so subclasses can override/extend.
    CANONICAL_ALT_KEYS: dict[str, str] = {
        "observation.state": "state",
        "action": "actions",
    }

    # Virtual action columns derived from the following frame's observation.
    #
    # robointer_droid stores the gripper command as
    # `other_information.action_gripper_velocity`, but training should predict
    # the next absolute gripper pose. Rather than rewriting 130k+ parquet files,
    # expose a schema-visible synthetic key whose value at frame t is read from
    # the source observation at frame t+1. Samples that would need T+1 are
    # marked padded and contribute no action loss.
    NEXT_FRAME_ACTION_SOURCES: dict[str, str] = {
        "other_information.action_gripper_position":
            "other_information.observation_gripper_position",
    }

    # PNG-in-parquet support. Subclasses populate from info.json features whose
    # ``dtype == "image"`` (PNG bytes in a parquet struct column, e.g.
    # LabUtopia/Level3_open). Empty default: mp4-only datasets fall through to
    # ``_read_video_frame``. When populated, these columns are projected at read
    # time and decoded inline by ``_decode_image_cell`` per row.
    _image_keys: tuple[str, ...] = ()

    @staticmethod
    def _decode_image_cell(val) -> torch.Tensor:
        """Decode a parquet image cell to ``(C, H, W) float32`` in ``[0, 1]``.

        HuggingFace ``datasets`` Image feature stores cells as a struct
        ``{"bytes": <encoded image bytes>, "path": <str|None>}``; the bytes
        are typically PNG (LabUtopia/Level3_open) but JPG also decodes via
        Pillow. This helper handles those two shapes plus a few defensive
        fallbacks (raw bytes, np.ndarray HWC uint8, pre-decoded torch.Tensor).

        Args:
            val: a parquet image cell from the per-frame DataFrame row.

        Returns:
            ``torch.Tensor`` of shape ``(C, H, W)``, dtype ``float32``,
            range ``[0, 1]``. Always a freshly-allocated tensor — callers
            may mutate without affecting the cache.
        """
        from PIL import Image  # local import: PIL is heavy & only needed here

        if isinstance(val, dict):  # HF Image feature: {"bytes": ..., "path": ...}
            b = val.get("bytes")
            if b is None:
                raise ValueError(
                    "image cell dict missing 'bytes' key (HF Image feature "
                    "with only 'path' is unsupported — adapter cannot read "
                    "external files mid-batch)"
                )
        elif isinstance(val, (bytes, bytearray)):
            b = bytes(val)
        elif isinstance(val, np.ndarray):
            arr = val
            if arr.ndim == 3 and arr.shape[-1] == 3:  # HWC uint8 / float
                arr = arr.transpose(2, 0, 1)
            t = torch.from_numpy(np.ascontiguousarray(arr).copy())
            return t.float() / 255.0 if t.dtype == torch.uint8 else t.float()
        elif isinstance(val, torch.Tensor):
            return val.float() / 255.0 if val.dtype == torch.uint8 else val.float()
        else:
            raise TypeError(f"unsupported image cell type: {type(val)!r}")

        img = Image.open(io.BytesIO(b)).convert("RGB")
        arr = np.asarray(img)  # H, W, 3 uint8
        # `.copy()` → writable buffer, avoids the read-only-buffer UserWarning.
        return (
            torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous().float()
            / 255.0
        )

    # ---- format-specific hooks (must be overridden) -----------------------

    @abstractmethod
    def _load_ep_parquet(self, ep_idx: int) -> pd.DataFrame:
        """Return a flat DataFrame holding every frame of episode ``ep_idx``."""
        raise NotImplementedError

    @abstractmethod
    def _read_video_frame(
        self, ep_idx: int, vkey: str, frame: int
    ) -> torch.Tensor:
        """Return the ``(3, H, W) float32`` RGB frame at ``frame`` index."""
        raise NotImplementedError

    # ---- flat-idx helpers (subclass must set _ep_starts/_episodes/meta) ----

    def __len__(self) -> int:
        return int(self._ep_lens.sum())

    def _validate_delta_timestamps_vs_episode_lens(self) -> None:
        """Warn once per repo if the longest ``delta_timestamps`` offset clips
        past the tail of many episodes.

        ``__getitem__`` clips ``idxs`` to ``[0, T-1]`` and records ``_is_pad``,
        so training does not crash. But when ``max(deltas) * fps`` is a large
        fraction of the SHORTEST episode, every sample from those episodes is
        mostly-padded — MSE effectively trains on the last frame repeated K
        times. Surface this at init. Warn (not hard-fail) above an 80%
        worst-case clipped-frame ratio; clip semantics are preserved.
        """
        if not self.delta_timestamps or not len(self._ep_lens):
            return
        try:
            max_delta = max(
                (max(v) for v in self.delta_timestamps.values() if len(v) > 0),
                default=0.0,
            )
        except (TypeError, ValueError):
            return
        if max_delta <= 0:
            return

        fps = float(getattr(self.meta, "fps", 0.0) or 0.0)
        if fps <= 0:
            return

        max_offset_frames = int(round(max_delta * fps))
        min_ep_len = int(self._ep_lens.min())
        if min_ep_len <= 0:
            return

        if max_offset_frames > int(0.80 * min_ep_len):
            from utils.logging_utils import warn_once

            repo_id = getattr(self, "repo_id", "<unknown>")
            warn_once(
                logger,
                ("delta_ts_exceeds_ep_len", repo_id, max_offset_frames, min_ep_len),
                "[adapter] %s: delta_timestamps max offset (%d frames @ fps=%.1f) "
                "exceeds 80%% of shortest episode length (%d frames). Samples "
                "drawn from short episodes will be heavily padded — training "
                "MSE may silently degrade. Consider shortening action_horizon "
                "or filtering episodes with min_len > max_offset.",
                repo_id, max_offset_frames, fps, min_ep_len,
            )

    def _flat_to_ep(self, flat_idx: int) -> tuple[int, int]:
        if flat_idx < 0 or flat_idx >= self.meta.total_frames:
            raise IndexError(
                f"flat_idx {flat_idx} out of [0, {self.meta.total_frames})"
            )
        ep_i = int(np.searchsorted(self._ep_starts, flat_idx, side="right") - 1)
        ep_idx = int(self._episodes[ep_i]["episode_index"])
        return ep_idx, int(flat_idx - self._ep_starts[ep_i])

    # ---- next-frame virtual action helpers --------------------------------

    @classmethod
    def _next_frame_action_source(cls, key: str) -> Optional[str]:
        return cls.NEXT_FRAME_ACTION_SOURCES.get(str(key))

    @classmethod
    def _required_parquet_key(cls, key: str) -> str:
        """Map a schema key to the physical parquet column it requires."""
        return cls._next_frame_action_source(key) or str(key)

    @classmethod
    def _schema_uses_next_frame_actions(cls, schema) -> bool:
        if schema is None:
            return False
        return any(
            cls._next_frame_action_source(k) is not None
            for k in getattr(schema, "action_keys", ())
        )

    @staticmethod
    def _field_offset(key: str, keys, dims) -> tuple[int, int] | None:
        offset = 0
        for k, d in zip(keys, dims):
            d = int(d)
            if str(k) == str(key):
                return offset, d
            offset += d
        return None

    @staticmethod
    def _has_sized_stats(value) -> bool:
        if value is None or isinstance(value, (str, bytes)):
            return False
        try:
            len(value)
        except TypeError:
            return False
        return True

    @classmethod
    def patch_stats_for_next_frame_actions(cls, stats: dict, schema) -> dict:
        """Patch canonical action stats for synthetic next-frame action keys.

        `data_process stats` was computed from the physical parquet columns.
        For robointer_droid that means canonical `stats["action"][7]` still
        describes the old velocity command. Once the schema action key becomes
        `other_information.action_gripper_position`, normalization must use the
        source observation-position distribution instead. We copy the matching
        slice from `stats["observation.state"]` into both `action` and
        `action_abs`; arm stats are left unchanged.
        """
        if not stats or schema is None:
            return stats

        action_keys = tuple(getattr(schema, "action_keys", ()) or ())
        state_keys = tuple(getattr(schema, "state_keys", ()) or ())
        action_dims = tuple(getattr(schema, "action_dims", ()) or ())
        state_dims = tuple(getattr(schema, "state_dims", ()) or ())

        patched = dict(stats)
        for action_key in action_keys:
            source_key = cls._next_frame_action_source(action_key)
            if source_key is None:
                continue

            action_loc = cls._field_offset(action_key, action_keys, action_dims)
            source_loc = cls._field_offset(source_key, state_keys, state_dims)
            if action_loc is None or source_loc is None:
                logger.warning(
                    "[adapter] cannot patch stats for derived action %r from %r: "
                    "schema offsets not found",
                    action_key, source_key,
                )
                continue
            action_offset, action_dim = action_loc
            source_offset, source_dim = source_loc
            if action_dim != source_dim:
                logger.warning(
                    "[adapter] cannot patch stats for derived action %r from %r: "
                    "dim mismatch action_dim=%d source_dim=%d",
                    action_key, source_key, action_dim, source_dim,
                )
                continue

            state_stats = patched.get("observation.state")
            if not isinstance(state_stats, dict):
                logger.warning(
                    "[adapter] cannot patch stats for derived action %r: "
                    "stats['observation.state'] missing",
                    action_key,
                )
                continue

            for action_stats_key in ("action", "action_abs"):
                action_stats = patched.get(action_stats_key)
                if not isinstance(action_stats, dict):
                    continue
                action_stats = dict(action_stats)
                for stat_name, action_value in list(action_stats.items()):
                    if stat_name == "count" or not cls._has_sized_stats(action_value):
                        continue
                    source_value = state_stats.get(stat_name)
                    if not cls._has_sized_stats(source_value):
                        continue
                    if (
                        len(action_value) < action_offset + action_dim
                        or len(source_value) < source_offset + source_dim
                    ):
                        logger.warning(
                            "[adapter] cannot patch stats[%r][%r] for derived "
                            "action %r: action_len=%d source_len=%d",
                            action_stats_key, stat_name, action_key,
                            len(action_value), len(source_value),
                        )
                        continue
                    arr = np.asarray(action_value).copy()
                    src = np.asarray(source_value)[
                        source_offset:source_offset + source_dim
                    ]
                    arr[action_offset:action_offset + action_dim] = src
                    action_stats[stat_name] = arr
                patched[action_stats_key] = action_stats

        return patched

    def _drop_terminal_samples_for_next_frame_actions(self, schema) -> None:
        """Exclude each episode's terminal frame when next-frame labels exist.

        The terminal frame has no t+1 observation, so it cannot produce a valid
        next-frame gripper action. We keep the physical episode parquet intact
        (so frame T-2 can still read T-1 as its label) but shrink the sampling
        index by one frame per episode. Very short episodes are dropped.
        """
        if not self._schema_uses_next_frame_actions(schema):
            return

        before_eps = len(self._episodes)
        before_frames = int(self._ep_lens.sum()) if len(self._ep_lens) else 0
        kept: list[dict] = []
        for ep in self._episodes:
            length = int(ep.get("length", 0))
            if length <= 1:
                continue
            ep_view = dict(ep)
            ep_view["length"] = length - 1
            kept.append(ep_view)

        self._episodes = kept
        self._ep_lens = np.array(
            [int(e["length"]) for e in self._episodes], dtype=np.int64
        )
        self._ep_starts = (
            np.concatenate([[0], np.cumsum(self._ep_lens)[:-1]])
            if len(self._ep_lens)
            else np.array([], dtype=np.int64)
        )
        after_frames = int(self._ep_lens.sum()) if len(self._ep_lens) else 0
        logger.info(
            "[adapter] next-frame action schema=%s: dropped terminal samples "
            "for %d episodes (%d -> %d trainable frames; dropped episodes=%d)",
            getattr(schema, "schema_id", None),
            len(self._episodes),
            before_frames,
            after_frames,
            before_eps - len(self._episodes),
        )

    def _resolve_df_key(self, key: str, df: pd.DataFrame) -> Optional[str]:
        if key in df.columns:
            return key
        if key == "action" and "actions" in df.columns:
            return "actions"
        if key == "observation.state" and "state" in df.columns:
            return "state"
        return None

    def _materialize_next_frame_actions(
        self,
        out: dict,
        df: pd.DataFrame,
        frame_in_ep: int,
    ) -> None:
        sch = self.meta.schema
        if sch is None:
            return
        T = len(df)
        for action_key in getattr(sch, "action_keys", ()):
            source_key = self._next_frame_action_source(action_key)
            if source_key is None:
                continue
            src_key = self._resolve_df_key(source_key, df)
            target = self._schema_key_target_dim(action_key)
            valid = src_key is not None and frame_in_ep + 1 < T
            if valid:
                arr = np.atleast_1d(
                    np.asarray(df[src_key].iloc[frame_in_ep + 1], dtype=np.float32)
                )
            else:
                arr = np.zeros((target or 1,), dtype=np.float32)
            if target is not None:
                arr = self._pad_row(arr, target)
            if not arr.flags.writeable:
                arr = np.array(arr, copy=True)
            out[action_key] = torch.as_tensor(arr)
            out[f"{action_key}_is_pad"] = torch.as_tensor(
                [not valid], dtype=torch.bool
            )

    # ---- row-cell / per-key padding helpers (hoisted from v21) ------------

    @staticmethod
    def _cell_to_tensor(val):
        if isinstance(val, np.ndarray):
            # pyarrow-backed views are read-only AND contiguous, so
            # ascontiguousarray is a no-op; np.array(copy=True) forces a fresh
            # writable buffer (one small memcpy/row) to avoid the UserWarning.
            if not val.flags.writeable:
                val = np.array(val, copy=True)
            return torch.as_tensor(val)
        if isinstance(val, (list, tuple)):
            return torch.as_tensor(np.asarray(val))
        if isinstance(val, (int, np.integer)):
            return torch.tensor(int(val), dtype=torch.int64)
        if isinstance(val, (float, np.floating)):
            return torch.tensor(float(val), dtype=torch.float32)
        if isinstance(val, (bool, np.bool_)):
            return torch.tensor(bool(val))
        return val  # bytes/dict/str left as-is

    def _schema_target_dim(self, kind: str) -> Optional[int]:
        """Legacy canonical-key target dim. Kept for callers that assume
        a single (state_dim, action_dim) pair exists. Multi-key schemas
        must use ``_schema_key_target_dim`` for per-column lookup.
        """
        sch = self.meta.schema
        if sch is None:
            return None
        if kind == "state":
            keys, dims = sch.state_keys, sch.state_dims
        else:
            keys, dims = sch.action_keys, sch.action_dims
        if len(keys) == 1 and len(dims) == 1:
            return int(dims[0])
        return None

    def _schema_key_target_dim(self, col: str) -> Optional[int]:
        """Per-schema-key target dim lookup by raw column name.

        For multi-key schemas (e.g. robocoin: ``action`` +
        ``gripper_open_scale_action`` with dims ``(30, 2)``) each column
        must be padded/truncated INDEPENDENTLY to its declared dim before
        concat — otherwise variable-dim source robots (49-dim
        ``observation.state`` in one of the bimanual episodes) let the
        cat'd vector explode past the schema's ``delta_mask`` length and
        crash ``DeltaActionTransformFn``'s broadcast. Returns None for
        columns not declared in the schema (``task_index``, ``timestamp``,
        etc.).
        """
        sch = self.meta.schema
        if sch is None:
            return None
        for k, d in zip(sch.state_keys, sch.state_dims):
            if col == k:
                return int(d)
        for k, d in zip(sch.action_keys, sch.action_dims):
            if col == k:
                return int(d)
        return None

    # Per-(repo, cur, target) dedupe set for the LABVLA_ALLOW_TRUNCATE
    # escape-hatch warning. Keyed on repo_id so multi-repo mixes reveal WHICH
    # repo is being truncated (OXE packed-canonical is safe; others may mask a
    # real data bug). Emits once per unique mismatch across all ranks/workers.
    _truncate_warned: set = set()

    @classmethod
    def _emit_truncate_warning_once_cls(
        cls, cur: int, target_dim: int, repo_id: str
    ) -> None:
        key = (repo_id, cur, target_dim)
        if key in cls._truncate_warned:
            return
        cls._truncate_warned.add(key)
        logger.warning(
            "LABVLA_ALLOW_TRUNCATE=1: repo=%r truncating last-dim from %d to %d. "
            "This bypasses the P0-03 guard and is unsafe for joint/action data. "
            "If you only intended this for a specific repo (e.g. oxe-auge_clean_v2's "
            "packed-canonical observation.state), check whether other repos in "
            "this run are silently affected too.",
            repo_id, cur, target_dim,
        )

    def _pad_row(self, arr: np.ndarray, target_dim: int) -> np.ndarray:
        """Zero-pad last dim up to ``target_dim``; refuse to silently truncate.

        Truncating when ``cur > target_dim`` would silently drop real
        joint/action dims (dual-arm or gripper state), so this raises; data
        wider than the schema declares is a schema bug to fix at the source.
        ``LABVLA_ALLOW_TRUNCATE=1`` opts back into the legacy truncation.
        """
        cur = arr.shape[-1]
        if cur == target_dim:
            return arr
        if cur > target_dim:
            if os.environ.get("LABVLA_ALLOW_TRUNCATE") == "1":
                _repo = str(getattr(self, "repo_id", "<unknown>"))
                self._emit_truncate_warning_once_cls(cur, target_dim, _repo)
                return arr[..., :target_dim]
            raise ValueError(
                f"_pad_row: source last-dim={cur} > schema target_dim={target_dim}; "
                "refusing silent truncation. Either widen the schema's "
                "state_dims/action_dims or set LABVLA_ALLOW_TRUNCATE=1 to "
                "retain the legacy truncating behavior (not recommended)."
            )
        pad_shape = list(arr.shape)
        pad_shape[-1] = target_dim - cur
        return np.concatenate(
            [arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=-1
        )

    # ---- shared __getitem__ pipeline --------------------------------------

    def __getitem__(self, flat_idx: int) -> dict:
        ep_idx, frame_in_ep = self._flat_to_ep(flat_idx)
        df = self._load_ep_parquet(ep_idx)

        # Schema-driven target dims for per-row padding (multi-robot merges).
        state_target = self._schema_target_dim("state")
        action_target = self._schema_target_dim("action")

        out: dict = {}
        image_keys_set = set(self._image_keys)
        # Write image cells under the CANONICAL prefixed key
        # (``observation.images.<x>``) regardless of the parquet column name
        # (v3.0 uses prefixed feature names, v2.1 unprefixed like
        # ``camera_1_rgb``) so adapter writes and the downstream
        # RemapImageKeyTransformFn stay on the same canonical shape.
        from schema.camera_mapping import expand_camera_source as _expand_src
        _canonical_image_key = {col: _expand_src(col) for col in image_keys_set}
        for col in df.columns:
            # PNG-in-parquet (dtype=image) columns are decoded inline here;
            # dtype=video columns go through the mp4 path below. Without this
            # branch the image cell is skipped and the video loop falls through
            # to _zero_frame, training the model on all-black frames.
            if col in image_keys_set:
                val = df[col].iloc[frame_in_ep]
                out[_canonical_image_key[col]] = self._decode_image_cell(val)
                continue
            if col in self.meta.video_keys:
                continue
            val = df[col].iloc[frame_in_ep]
            # Per-schema-key padding: multi-key schemas (robocoin) pad each key
            # to its declared schema.state_dims[i]/action_dims[i] before cat.
            key_target = self._schema_key_target_dim(col)
            # Same source_*_keys guard as the chunked path below: if a
            # downstream canonical transform owns the raw-N -> canonical-8
            # mapping, don't pre-truncate here.
            sch_here = self.meta.schema
            if sch_here is not None and key_target is not None:
                act_set = set(getattr(sch_here, "action_keys", ()) or ()) | {"action", "actions"}
                st_set = set(getattr(sch_here, "state_keys", ()) or ()) | {"observation.state", "state"}
                if col in act_set and getattr(sch_here, "source_action_keys", ()):
                    key_target = None
                elif col in st_set and getattr(sch_here, "source_state_keys", ()):
                    key_target = None
            if key_target is not None and isinstance(val, (np.ndarray, list)):
                arr = np.atleast_1d(np.asarray(val, dtype=np.float32))
                arr = self._pad_row(arr, key_target)
                # Force a writable copy (see _cell_to_tensor).
                if not arr.flags.writeable:
                    arr = np.array(arr, copy=True)
                out[col] = torch.as_tensor(arr)
                continue
            out[col] = self._cell_to_tensor(val)

        # Normalize v2.1 pluralization quirks; keep both legacy and canonical
        # keys when they diverge. CANONICAL_ALT_KEYS is overridable by subclasses.
        for canonical, alias in self.CANONICAL_ALT_KEYS.items():
            if canonical not in out and alias in out:
                out[canonical] = out[alias]

        # Synthetic next-frame action columns are not stored in parquet. Expose
        # them under their schema action key so non-chunked paths still receive
        # a valid tensor + `_is_pad`; the chunking loop below overwrites this
        # with a (K, D) horizon tensor when delta_timestamps are configured.
        self._materialize_next_frame_actions(out, df, frame_in_ep)

        # Resolve the language instruction string for the VLM processor.
        # Priority: task -> task_index -> natural_language_instruction (some v3
        # OXE sub-repos ship only the last). Warn-loud once if still empty so
        # empty-instruction training surfaces instead of passing silently.
        if "task" not in out:
            resolved_task = ""
            if "task_index" in df.columns:
                ti = int(df["task_index"].iloc[frame_in_ep])
                resolved_task = self._tasks_by_idx.get(ti, "")
            if not resolved_task and "natural_language_instruction" in df.columns:
                nli = df["natural_language_instruction"].iloc[frame_in_ep]
                if nli is not None and not _is_missing_scalar_value(nli):
                    resolved_task = str(nli)
            out["task"] = resolved_task

        # Warn-loud (once per repo) if the final instruction is empty: the
        # sample carries real images but no language supervision.
        _final_task = out.get("task")
        if _final_task is None or (
            isinstance(_final_task, str) and _final_task == ""
        ):
            from utils.logging_utils import warn_once

            repo_id = getattr(self, "repo_id", "<unknown>")
            warn_once(
                logger,
                ("empty_task_instruction", repo_id),
                "[adapter] %s: resolved an EMPTY task/instruction string "
                "(task -> task_index -> natural_language_instruction all "
                "missing/empty). These samples train on an empty language "
                "instruction. Check that the dataset has `task`, a usable "
                "`task_index` (+ tasks.jsonl/tasks.parquet), or a "
                "`natural_language_instruction` column.",
                repo_id,
            )

        # delta_timestamps: caller wants K future frames of a key (e.g.
        # actions). Expand canonical "action"/"observation.state" specs to the
        # schema's actual action_keys/state_keys so schemas like
        # robointer_droid (e.g. other_information.action_joint_position) chunk.
        dt_items: list[tuple[str, list[float]]] = []
        sch = self.meta.schema
        for key, deltas in self.delta_timestamps.items():
            dt_items.append((key, deltas))
            # Expand to every schema action_key, INCLUDING "action"/"actions":
            # single-key schemas (labutopia action_keys=("actions",)) need the
            # phase-2 loop to overwrite the single-frame value with a (K, D)
            # chunk, else DeltaActionTransformFn fails to broadcast against
            # (1, D) state. Dedup since the top-level loop already added it.
            if key in ("action", "actions") and sch is not None:
                action_expand_keys = (
                    tuple(getattr(sch, "source_action_keys", ()) or ())
                    or tuple(getattr(sch, "action_keys", ()) or ())
                )
                for ak in action_expand_keys:
                    if ak != key:
                        dt_items.append((ak, deltas))
            # NOTE: state is intentionally NOT expanded —
            # DeltaActionTransformFn asserts state.ndim == 1 (single frame
            # per sample).

        for key, deltas in dt_items:
            next_frame_source = self._next_frame_action_source(key)
            lookup_key = next_frame_source or key
            src_key = self._resolve_df_key(lookup_key, df)
            if src_key is None:
                continue
            T = len(df)
            source_shift = 1 if next_frame_source is not None else 0
            idxs = [
                frame_in_ep + source_shift + int(round(dt * self.meta.fps))
                for dt in deltas
            ]
            clipped = np.clip(idxs, 0, T - 1)
            is_pad = np.array(
                [i != c for i, c in zip(idxs, clipped)], dtype=bool
            )
            col_vals = np.stack([
                np.atleast_1d(np.asarray(df[src_key].iloc[int(c)])) for c in clipped
            ]).astype(np.float32)
            # Per-key schema target dim (multi-key schemas like robocoin pad
            # each column to its own declared dim before cat).
            target = self._schema_key_target_dim(key)
            # If the schema declares source_*_keys, a downstream transform owns
            # the raw-N -> canonical-8 mapping, so the adapter MUST NOT
            # pre-truncate here: doing so drops the real gripper (e.g.
            # LabEmbodied UR/Festo raw 11, Rizon4 raw 12 → first 8 dims) and
            # replaces it with a mirror joint. Must run REGARDLESS of whether
            # the canonical key matched, since the data column is the raw source.
            if sch is not None and target is not None:
                action_keys_set = set(getattr(sch, "action_keys", ()) or ()) | {"action", "actions"}
                state_keys_set = set(getattr(sch, "state_keys", ()) or ()) | {"observation.state", "state"}
                if key in action_keys_set and getattr(sch, "source_action_keys", ()):
                    target = None
                elif key in state_keys_set and getattr(sch, "source_state_keys", ()):
                    target = None
            if target is None:
                # Fall back to canonical single-key path.
                target = action_target if key in ("action", "actions") else (
                    state_target if key in ("observation.state", "state") else None)
                if sch is not None:
                    if key in ("action", "actions") and getattr(sch, "source_action_keys", ()):
                        target = None
                    elif key in ("observation.state", "state") and getattr(sch, "source_state_keys", ()):
                        target = None
            if target is not None:
                col_vals = self._pad_row(col_vals, target)
            out[key] = torch.as_tensor(col_vals)
            out[f"{key}_is_pad"] = torch.as_tensor(is_pad)

        # Video frame reads — only cameras the schema actually uses. Iterating
        # info.json cameras that don't exist on disk or aren't in image_mapping
        # wastes a full mp4 decode + NFS stat per extra camera, which dominates
        # data loading under multi-rank contention.
        sch = self.meta.schema
        target_cams = (list(sch.image_mapping.keys())
                       if sch is not None and getattr(sch, "image_mapping", None)
                       else self.meta.video_keys)
        # Track cameras whose read fell back to a black _zero_frame; otherwise
        # RemapImageKeyTransformFn marks the slot mask=True and the attention
        # layer treats the fallback as a real observation.
        for vkey in target_cams:
            # dtype=image cameras were already decoded from parquet above; skip
            # the mp4 fallback to avoid clobbering the real frame and doing
            # wasted stat()s. image_mapping order yields vkey canonical-prefixed.
            if vkey in out:
                continue
            self._last_read_was_zero_frame = False
            frame_t = self._read_video_frame(ep_idx, vkey, frame_in_ep)
            out[vkey] = frame_t
            if self._last_read_was_zero_frame:
                out[f"{vkey}_invalid"] = True

        # Defensive fill: guarantee every schema action_key has a matching
        # `{key}_is_pad`. The dt_items loop skips keys absent from
        # delta_timestamps (inference/ablation), which otherwise leaves
        # `action_is_pad` absent and forces pad-masked MSE to unmasked mean on
        # ~9% of padded trailing frames. Fill zeros (no pad) where missing.
        if sch is not None:
            for ak in getattr(sch, "action_keys", ()):
                if ak in out and f"{ak}_is_pad" not in out:
                    val = out[ak]
                    chunk_len = int(val.shape[0]) if val.ndim >= 1 else 1
                    out[f"{ak}_is_pad"] = torch.zeros(chunk_len, dtype=torch.bool)

        return out

    # ---- zero-frame fallback (shared) -------------------------------------

    _last_read_was_zero_frame: bool = False

    def _zero_frame(
        self, h: int = 480, w: int = 640, reason: str = "missing_file"
    ) -> torch.Tensor:
        """Return a cached black frame for missing-video fallback.

        Never runs image_transforms on the cached zero tensor: the random
        jitter would destroy the 'this camera is invalid' semantic and cache
        one random outcome forever (same tensor returned for every sample).
        """
        # Count reasons per failure mode so silent I/O rot is discoverable.
        if not hasattr(self, "_zero_frame_reasons"):
            import collections as _collections
            self._zero_frame_reasons = _collections.Counter()
        self._zero_frame_reasons[reason] += 1
        self._last_read_was_zero_frame = True
        return torch.zeros(3, h, w, dtype=torch.float32)

    def video_fallback_summary(self) -> dict:
        """Snapshot of how many times each fallback reason fired in this
        adapter's lifetime, so callers can log silent I/O rot ("why did MSE
        never drop on that repo").
        """
        if not hasattr(self, "_zero_frame_reasons"):
            return {}
        return dict(self._zero_frame_reasons)
