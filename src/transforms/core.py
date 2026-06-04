from __future__ import annotations

from typing import Any, Optional, runtime_checkable
from dataclasses import dataclass, field, replace

import abc
import os
import draccus
import logging
import torch
import torchvision
import torch.nn.functional as F
import numpy as np

from transforms.utils import resize_with_pad, resize_center_crop
from utils.constants import OBS_IMAGE, OBS_IMAGES, OBS_STATE, ACTION, NUM_IMAGE_SLOTS

# Legacy dict imports deliberately removed in Phase 4.
# All hydration goes through hydrate_all() with a DatasetSchema.


DataDict = dict[str, Any]


_SOURCE_FAST_MAX_LENGTHS: dict[str, int] = {
    # Keep schema-explicit budget hooks, but do not tighten 4ds action sources
    # below the launcher union cap. Memory savings come from ``trim_to_mask``
    # emitting the real per-sample token length; lowering the tokenizer cap
    # itself can truncate FAST labels, which changes the CE target.
    "robointer_droid_v2": 224,
    "robointer_droid_anno_v1": 224,
    "oxe_auge_v1": 224,
    "agibot_dual_arm_v1": 224,
}


def source_fast_max_length(schema_id: str, configured_max_length: int) -> int:
    """Return the Phase-C FAST cap for a schema without widening user config."""
    budget = _SOURCE_FAST_MAX_LENGTHS.get(str(schema_id))
    if budget is None:
        return int(configured_max_length)
    return min(int(configured_max_length), int(budget))


def _dual_arm_canonical_indices(layout, source_dims: tuple[int, ...]) -> list[int] | None:
    """Raw [left joints, right joints, left grip, right grip] → 14-dim indices.

    Returns indices into the raw concatenated vector for the non-padding slots of
    the canonical dual-arm layout:
      0..5  left joints, 6 left gripper, 7..12 right joints, 13 right gripper.
    """
    if layout is None:
        return None
    try:
        from schema.arm_layout import ArmCount
    except Exception:
        return None
    if getattr(layout, "arm_count", None) != ArmCount.DUAL:
        return None
    left_dof = int(layout.left_arm_dof)
    right_dof = int(layout.right_arm_dof)
    raw_width = sum(int(d) for d in source_dims)
    if raw_width <= 0:
        return None
    left_grip = int(layout.left_gripper_index_in_raw)
    right_grip = int(layout.right_gripper_index_in_raw)
    right_start = left_dof
    indices = (
        list(range(0, min(left_dof, 6)))
        + [left_grip]
        + list(range(right_start, right_start + min(right_dof, 6)))
        + [right_grip]
    )
    if max(indices) >= raw_width:
        return None
    return indices


def _canonicalize_dual_arm_stats(stats: dict | None, schema) -> dict | None:
    """Reorder raw dual-arm stats to the schema's 14-dim canonical layout."""
    if not stats or schema is None:
        return stats
    layout = getattr(schema, "arm_layout", None)
    state_idx = _dual_arm_canonical_indices(
        layout, tuple(getattr(schema, "source_state_dims", ()) or ())
    )
    action_idx = _dual_arm_canonical_indices(
        layout, tuple(getattr(schema, "source_action_dims", ()) or ())
    )
    if state_idx is None and action_idx is None:
        return stats

    def _remap_entry(entry: dict, indices: list[int] | None) -> dict:
        if indices is None:
            return entry
        out = dict(entry)
        max_idx = max(indices)
        for stat_name, value in entry.items():
            try:
                if len(value) <= max_idx:
                    continue
            except TypeError:
                continue
            arr = np.asarray(value)
            if arr.ndim != 1:
                continue
            out[stat_name] = arr[np.asarray(indices, dtype=np.int64)].tolist()
        return out

    canonicalized = dict(stats)
    for key, indices in (
        (OBS_STATE, state_idx),
        (ACTION, action_idx),
        ("action_abs", action_idx),
    ):
        entry = stats.get(key)
        if isinstance(entry, dict):
            canonicalized[key] = _remap_entry(entry, indices)
    return canonicalized


_LAYOUT_CANON_MARKER = "_layout_canonicalization"
_GRIPPER_CANON_MARKER = "_gripper_canonicalization"


def _canonicalize_single_arm_stats(stats: dict | None, schema) -> dict | None:
    """Remap raw single-arm stats to the canonical 8-dim layout.

    Mirror of ``_canonicalize_dual_arm_stats`` for the single-arm path
    that ``CanonicalSingleArmLayoutTransformFn`` builds at training time.
    Raw layout: dims 0..arm_dof-1 are arm joints, dim ``gripper_index_in_raw``
    is the scalar gripper, any trailing dims are redundant mirrors.
    Canonical layout: dims 0..6 = arm (zero-padded if arm_dof<7),
    dim 7 = gripper.

    For zero-pad dims (e.g. UR/festo 6-DoF → canonical dim 6) there is
    no raw index to copy from; the transform fills dim 6 with 0.0 at
    every frame. The stats for that dim are set to a neutral sentinel
    ``mean=0, std=1, q01=-1, q99=+1`` so q01_q99 normalization of the
    constant 0 input yields a stable 0 (without div-by-zero).

    No-op when:
      - schema has no arm_layout or arm_count != SINGLE
      - schema has no ``source_state_keys`` (raw == canonical already, no
        remap needed; e.g. Franka 8-dim)
      - stats already carry ``_layout_canonicalization`` marker (pre-canonicalized
        on disk by tools/alias_and_canonicalize_labembodied_stats.py).
    """
    if not stats or schema is None:
        return stats
    # Idempotency: skip if stats were already layout-canonicalized on disk.
    if stats.get(_LAYOUT_CANON_MARKER):
        return stats
    layout = getattr(schema, "arm_layout", None)
    if layout is None:
        return stats
    try:
        from schema.arm_layout import ArmCount
    except Exception:
        return stats
    if getattr(layout, "arm_count", None) != ArmCount.SINGLE:
        return stats
    # If schema doesn't declare a raw source layout, stats are already
    # in canonical orientation (e.g. Franka 8-dim).
    if not getattr(schema, "source_state_keys", None) and not getattr(
        schema, "source_action_keys", None
    ):
        return stats

    arm_dof = int(getattr(layout, "arm_dof", 0))
    grip_idx_raw = getattr(layout, "gripper_index_in_raw", None)
    if grip_idx_raw is None or arm_dof <= 0:
        return stats
    grip_idx = int(grip_idx_raw)
    arm_keep = min(arm_dof, 7)
    pad_count = 7 - arm_keep

    # Per-slot index into the raw vector. None = zero-pad slot (no raw idx).
    # Length is always 8 (canonical width).
    indices: list[int | None] = (
        list(range(arm_keep))   # canonical 0..arm_keep-1 ← raw 0..arm_keep-1
        + [None] * pad_count    # canonical arm_keep..6 (zero-pad — sentinel stats)
        + [grip_idx]            # canonical 7 ← raw[grip_idx]
    )

    # Sentinel stats for zero-pad dim. Keep std=1 to avoid div-by-zero in
    # the eventual (x-mean)/std path; q01/q99 spans 2 so (0 - q01)/(q99-q01)
    # = 0.5 normalized — neutral.
    _SENTINEL = {"mean": 0.0, "std": 1.0, "q01": -1.0, "q99": 1.0}

    def _remap_entry(entry: dict, raw_width: int) -> dict:
        out = dict(entry)
        for stat_name, value in entry.items():
            try:
                if len(value) <= max(i for i in indices if i is not None):
                    # raw stats too narrow to slice with our grip_idx — bail
                    continue
            except TypeError:
                continue
            arr = np.asarray(value)
            if arr.ndim != 1:
                continue
            new = []
            for idx in indices:
                if idx is None:
                    new.append(float(_SENTINEL.get(stat_name, 0.0)))
                else:
                    if idx >= arr.shape[0]:
                        # Stats narrower than expected raw — bail without
                        # corrupting; the dataset will fail its strict checks
                        # downstream with a clearer error.
                        return entry
                    new.append(float(arr[idx]))
            out[stat_name] = new
        return out

    canonicalized = dict(stats)
    for key in (OBS_STATE, ACTION, "action_abs"):
        entry = stats.get(key)
        if isinstance(entry, dict):
            # raw_width here is informational — we let _remap_entry inspect
            # each stat array's length individually.
            canonicalized[key] = _remap_entry(entry, raw_width=0)
    return canonicalized


class DataTransformFn(draccus.ChoiceRegistry, abc.ABC):
    @abc.abstractmethod
    def __call__(self, data: DataDict) -> DataDict: ...


@dataclass(frozen=True)
class TransformGroup:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: list[DataTransformFn] = field(default_factory=list)

    # Transforms that are applied to the model output data.
    outputs: list[DataTransformFn] = field(default_factory=list)

    def push(self,
             *,
             inputs: list[DataTransformFn] = None,
             outputs: list[DataTransformFn] = None) -> TransformGroup:
        """Append transforms to the group and return a new group.

        Both inputs and outputs are appended to the *end* (FIFO order):
        earlier pushes run first.

        Returns:
            A new group with the appended transforms.
        """
        if inputs is None: inputs = []
        if outputs is None: outputs = []
        return TransformGroup(
            inputs=[*self.inputs, *inputs],
            outputs=[*self.outputs, *outputs],
        )


@DataTransformFn.register_subclass("composite")
@dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: list[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: list[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@DataTransformFn.register_subclass("identity")
@dataclass(frozen=True)
class IdentityTransformFn(DataTransformFn):
    def __call__(self, data: DataDict) -> DataDict:
        return data


@DataTransformFn.register_subclass("gripper_semantic_canonicalize")
@dataclass
class GripperSemanticCanonicalizeFn(DataTransformFn):
    """Convert gripper dim from source semantic (width / position) to the
    canonical continuous ``open_fraction`` ∈ [0, 1] target.

    Mathematical mapping (per PLAN-02 of the 2026-05-22 LabEmbodied audit):
        open_fraction = clip((x - closed) / (open - closed), 0, 1)
        if direction == -1: open_fraction = 1 - open_fraction

    where ``closed``, ``open``, ``direction`` are schema-level calibration
    constants for the source robot family. After this transform every
    source's gripper dim shares the SAME normalized semantic — fully
    closed = 0.0, fully open = 1.0 — so cross-source posttrain MSE on
    the gripper dim is no longer a contradictory target. The downstream
    ``NormalizeTransformFn`` should then use open_fraction-domain stats
    (q01=0, q99=1) on the gripper dim.

    Enabled only when the schema declares a non-empty
    ``source_semantic`` distinct from ``open_fraction``. For
    ``open_fraction`` sources (e.g. OXE-Auge) this transform is a no-op.
    """
    enabled: bool = False
    source_semantic: str = ""
    target_semantic: str = "open_fraction"
    closed: float = 0.0
    open: float = 1.0
    direction: int = 1  # +1: x↑ → of↑;  -1: x↑ → of↓
    gripper_dim: int = 7
    state_keys: tuple = ()
    action_keys: tuple = ()

    def __call__(self, data: DataDict) -> DataDict:
        if not self.enabled:
            return data
        if self.source_semantic == self.target_semantic:
            return data
        span = float(self.open) - float(self.closed)
        if not (span > 0 or span < 0):
            return data
        for key in (*self.state_keys, *self.action_keys):
            tensor = data.get(key)
            if tensor is None or not hasattr(tensor, "shape"):
                continue
            if tensor.shape[-1] <= self.gripper_dim:
                continue
            new_tensor = tensor.clone() if hasattr(tensor, "clone") else tensor.copy()
            raw = tensor[..., self.gripper_dim]
            of = (raw - float(self.closed)) / span
            if int(self.direction) < 0:
                of = 1.0 - of
            of = of.clamp(0.0, 1.0) if hasattr(of, "clamp") else of.clip(0.0, 1.0)
            new_tensor[..., self.gripper_dim] = of
            data[key] = new_tensor
        return data


def _canonicalize_gripper_semantic_stats(
    stats: dict | None, schema, calibration: dict | None
) -> dict | None:
    """Rescale gripper-dim stats from source semantic to open_fraction.

    Mirrors what GripperSemanticCanonicalizeFn does to the data, but on
    the q01/q99/mean/std arrays. Must run AFTER any layout canonicalization
    so the gripper is already at canonical dim 7.

    For source_semantic="width" (LabEmbodied/LabUtopia Franka):
        new[7] = (old[7] - closed) / (open - closed)
    For source_semantic="open_fraction" (OXE-Auge): no-op
    """
    if not stats or schema is None or not calibration:
        return stats
    if not calibration.get("enabled"):
        return stats
    # Idempotency: skip if stats were already gripper-canonicalized on disk.
    if stats.get(_GRIPPER_CANON_MARKER):
        return stats
    src = calibration.get("source_semantic", "")
    tgt = calibration.get("target_semantic", "open_fraction")
    if src == tgt or not src:
        return stats
    closed = float(calibration["closed"])
    open_v = float(calibration["open"])
    direction = int(calibration.get("direction", 1))
    span = open_v - closed
    if span == 0:
        return stats
    grip_dim = int(calibration.get("gripper_dim", 7))

    def _remap_value(v):
        of = (float(v) - closed) / span
        if direction < 0:
            of = 1.0 - of
        return of

    def _remap_entry(entry: dict) -> dict:
        out = dict(entry)
        for stat_name in ("q01", "q99", "mean"):
            arr = entry.get(stat_name)
            if not isinstance(arr, (list, tuple)) or len(arr) <= grip_dim:
                continue
            new = list(arr)
            new[grip_dim] = _remap_value(new[grip_dim])
            out[stat_name] = new
        # std scales by |1/span| (affine multiplier); direction sign drops
        std = entry.get("std")
        if isinstance(std, (list, tuple)) and len(std) > grip_dim:
            new = list(std)
            new[grip_dim] = float(new[grip_dim]) / abs(span)
            out["std"] = new
        return out

    canon = dict(stats)
    for key in (OBS_STATE, ACTION, "action_abs"):
        e = stats.get(key)
        if isinstance(e, dict):
            canon[key] = _remap_entry(e)
    return canon


@DataTransformFn.register_subclass("canonical_arm_layout")
@dataclass
class CanonicalArmLayoutTransformFn(DataTransformFn):
    """Materialize canonical state/action vectors from raw source columns.

    AgiBot stores raw dual-arm action as 14 joints plus 2 effectors:
    ``[left_7, right_7] + [left_gripper, right_gripper]``. LabVLA's existing
    dual-arm canonical layout is 14 dims:
    ``[left_6, left_gripper, right_6, right_gripper]``. This transform creates
    the canonical ``observation.state`` and ``action`` tensors before
    normalization, state discretization, FAST tokenization, and padding.
    """

    enabled: bool = False
    left_arm_dof: int = 0
    right_arm_dof: int = 0
    left_gripper_index_in_raw: int = -1
    right_gripper_index_in_raw: int = -1
    state_source_keys: tuple[str, ...] = ()
    action_source_keys: tuple[str, ...] = ()

    def __call__(self, data: DataDict) -> DataDict:
        if not self.enabled:
            return data
        self._materialize(data, self.state_source_keys, OBS_STATE)
        self._materialize(data, self.action_source_keys, ACTION)
        pad_keys = [f"{k}_is_pad" for k in self.action_source_keys if f"{k}_is_pad" in data]
        if ACTION in data and pad_keys and "action_is_pad" not in data:
            merged = data[pad_keys[0]].clone()
            for key in pad_keys[1:]:
                merged = merged | data[key]
            data["action_is_pad"] = merged
        return data

    def _materialize(self, data: DataDict, source_keys: tuple[str, ...], target_key: str) -> None:
        if not source_keys or any(k not in data for k in source_keys):
            return
        pieces = []
        for key in source_keys:
            value = data[key]
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value, dtype=torch.float32)
            pieces.append(value.to(dtype=torch.float32))
        raw = torch.cat(pieces, dim=-1)
        left_grip = int(self.left_gripper_index_in_raw)
        right_grip = int(self.right_gripper_index_in_raw)
        if raw.shape[-1] <= max(left_grip, right_grip):
            raise ValueError(
                f"CanonicalArmLayoutTransformFn raw width {raw.shape[-1]} is "
                f"too small for gripper indices ({left_grip}, {right_grip}) "
                f"from source_keys={source_keys!r}"
            )
        left_keep = min(self.left_arm_dof, 6)
        right_keep = min(self.right_arm_dof, 6)
        right_start = int(self.left_arm_dof)
        right_end = right_start + right_keep
        if raw.shape[-1] < right_end:
            raise ValueError(
                f"CanonicalArmLayoutTransformFn raw width {raw.shape[-1]} is "
                f"too small for right arm slice [{right_start}:{right_end}] "
                f"from source_keys={source_keys!r}"
            )
        out_shape = raw.shape[:-1] + (14,)
        out = raw.new_zeros(out_shape)
        out[..., :left_keep] = raw[..., :left_keep]
        out[..., 6] = raw[..., left_grip]
        out[..., 7:7 + right_keep] = raw[..., right_start:right_end]
        out[..., 13] = raw[..., right_grip]
        data[target_key] = out


@DataTransformFn.register_subclass("canonical_single_arm_layout")
@dataclass
class CanonicalSingleArmLayoutTransformFn(DataTransformFn):
    """Materialize canonical 8-dim single-arm state/action from raw N-dim source.

    Counterpart to ``CanonicalArmLayoutTransformFn`` for single-arm datasets
    whose raw layout has either a sub-7-DoF arm (e.g. UR/festo 6-DoF) or
    trailing redundant multi-finger gripper joints that aggregate to one
    scalar (e.g. LabEmbodied festo/UR 11-dim, rizon4 12-dim where dims after
    the gripper are perfectly correlated mirror copies of the single
    open-width signal).

    Canonical 8-dim layout (per src/schema/arm_layout.py):
      dim 0..6 = arm joints (zero-padded if raw arm DoF < 7)
      dim 7    = gripper (scalar copied from raw[..., raw_gripper_index_in_raw])

    Trailing raw dims past ``raw_gripper_index_in_raw`` are intentionally
    DROPPED — they are duplicate finger-joint copies of the single gripper
    scalar (verified by cross-correlation == ±1.0 against dim
    ``raw_gripper_index_in_raw``).

    Mirrors the dual-arm transform's pad-aggregation behavior: if any
    ``<source_action_key>_is_pad`` tensors are present, they are OR-merged
    into ``action_is_pad`` so downstream MSE masking stays correct.
    """

    enabled: bool = False
    raw_arm_dof: int = 0
    raw_gripper_index_in_raw: int = -1
    state_source_keys: tuple[str, ...] = ()
    action_source_keys: tuple[str, ...] = ()

    def __call__(self, data: DataDict) -> DataDict:
        if not self.enabled:
            return data
        self._materialize(data, self.state_source_keys, OBS_STATE)
        self._materialize(data, self.action_source_keys, ACTION)
        pad_keys = [
            f"{k}_is_pad" for k in self.action_source_keys
            if f"{k}_is_pad" in data
        ]
        if ACTION in data and pad_keys and "action_is_pad" not in data:
            merged = data[pad_keys[0]].clone()
            for key in pad_keys[1:]:
                merged = merged | data[key]
            data["action_is_pad"] = merged
        return data

    def _materialize(
        self,
        data: DataDict,
        source_keys: tuple[str, ...],
        target_key: str,
    ) -> None:
        if not source_keys or any(k not in data for k in source_keys):
            return
        pieces = []
        for key in source_keys:
            value = data[key]
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value, dtype=torch.float32)
            pieces.append(value.to(dtype=torch.float32))
        raw = torch.cat(pieces, dim=-1)
        grip = int(self.raw_gripper_index_in_raw)
        if raw.shape[-1] <= grip:
            raise ValueError(
                f"CanonicalSingleArmLayoutTransformFn raw width {raw.shape[-1]} is "
                f"too small for gripper index {grip} from source_keys={source_keys!r}"
            )
        arm_keep = min(int(self.raw_arm_dof), 7)
        if raw.shape[-1] < arm_keep:
            raise ValueError(
                f"CanonicalSingleArmLayoutTransformFn raw width {raw.shape[-1]} is "
                f"too small for arm slice [:{arm_keep}] from source_keys={source_keys!r}"
            )
        out_shape = raw.shape[:-1] + (8,)
        out = raw.new_zeros(out_shape)
        out[..., :arm_keep] = raw[..., :arm_keep]
        out[..., 7] = raw[..., grip]
        data[target_key] = out


@DataTransformFn.register_subclass("pad_state_and_action")
@dataclass
class PadStateAndActionTransformFn(DataTransformFn):
    max_state_dim: int = 32
    max_action_dim: int = 32

    def __call__(self, data: DataDict) -> DataDict: 
        data[OBS_STATE] = self._pad_vector(data[OBS_STATE], self.max_state_dim)
        data[ACTION] = self._pad_vector(data[ACTION], self.max_action_dim)
        return data

    def _pad_vector(self, vector: torch.Tensor, new_dim: int):
        # Fail loud when the incoming vector already exceeds the
        # padded target size. Silently returning as-is would
        # let downstream models receive state/action wider than max_*_dim
        # and either crash at a later layer or drop dims. Treat this as a
        # schema mismatch, not a condition to paper over.
        cur = vector.shape[-1]
        if cur > new_dim:
            raise ValueError(
                f"Vector dim {cur} > pad target {new_dim}. Schema declares "
                f"fewer dims than the runtime tensor — bump max_state_dim / "
                f"max_action_dim or prune schema keys."
            )
        if cur == new_dim:
            return vector
        return F.pad(vector, (0, new_dim - cur))


@DataTransformFn.register_subclass("snap_gripper_to_endpoints")
@dataclass
class SnapGripperToEndpointsFn(DataTransformFn):
    """Threshold gripper width signal to bimodal endpoints {0, max_width}.

    Use case: align LabUtopia continuous gripper width [0, 0.04m] to a binary
    {0=closed, 0.04m=open} representation that matches the OXE pretrain
    canonical convention (which is binary {0, 1} after gripper_canonicalize).

    Pipeline ordering: AFTER ComposeFieldsTransform (which builds canonical
    OBS_STATE / ACTION arrays) and BEFORE NormalizeTransformFn (so the snap
    to {0, max_width} happens on raw width values, then normalize maps to
    {-1, +1} cleanly when stats q01=0, q99=max_width).

    Stats override: caller MUST ensure stats[OBS_STATE][q01][gripper_dim]=0
    and stats[OBS_STATE][q99][gripper_dim]=max_width (same for ACTION) so
    that q01/q99 normalize maps {0, max_width} → {-1, +1}. Otherwise the
    snap-to-binary values get mapped through the wrong stats range and
    deploy gets garbage gripper output.

    Default Franka Panda config: gripper_dim=7, max_width=0.04m, threshold=0.5.
    """
    gripper_dim: int = 7
    max_width: float = 0.04
    threshold_ratio: float = 0.5  # > threshold_ratio*max_width → max_width, else 0
    state_keys: tuple[str, ...] = ()
    action_keys: tuple[str, ...] = ()
    state_dims: tuple[int, ...] = ()
    action_dims: tuple[int, ...] = ()

    def __call__(self, data: DataDict) -> DataDict:
        threshold = self.threshold_ratio * self.max_width
        applied = False
        for key in (OBS_STATE, ACTION):
            if key not in data:
                continue
            applied = self._snap_key(data, key, self.gripper_dim, threshold) or applied
        if applied:
            return data

        self._snap_split_fields(
            data,
            keys=self.state_keys,
            dims=self.state_dims,
            threshold=threshold,
        )
        self._snap_split_fields(
            data,
            keys=self.action_keys,
            dims=self.action_dims,
            threshold=threshold,
        )
        return data

    def _snap_split_fields(
        self,
        data: DataDict,
        *,
        keys: tuple[str, ...],
        dims: tuple[int, ...],
        threshold: float,
    ) -> bool:
        offset = 0
        for key, width in zip(keys, dims):
            width = int(width)
            if offset <= self.gripper_dim < offset + width:
                return self._snap_key(data, key, self.gripper_dim - offset, threshold)
            offset += width
        return False

    def _snap_key(
        self,
        data: DataDict,
        key: str,
        local_dim: int,
        threshold: float,
    ) -> bool:
        if key not in data:
            return False
        v = data[key]
        if isinstance(v, torch.Tensor):
            # Avoid in-place on shared tensor
            v = v.clone()
            if v.ndim == 0:
                if int(local_dim) != 0:
                    return False
                grip = v
                v = torch.where(
                    grip > threshold,
                    torch.full_like(grip, self.max_width),
                    torch.zeros_like(grip),
                )
            else:
                grip = v[..., local_dim]
                v[..., local_dim] = torch.where(
                    grip > threshold,
                    torch.full_like(grip, self.max_width),
                    torch.zeros_like(grip),
                )
        elif isinstance(v, np.ndarray):
            v = v.copy()
            if v.ndim == 0:
                if int(local_dim) != 0:
                    return False
                v = np.asarray(self.max_width if bool(v > threshold) else 0.0, dtype=v.dtype)
            else:
                v[..., local_dim] = np.where(
                    v[..., local_dim] > threshold,
                    self.max_width,
                    0.0,
                ).astype(v.dtype)
        else:
            return False
        data[key] = v
        return True


@DataTransformFn.register_subclass("totensor")
@dataclass
class ToTensorTransformFn(DataTransformFn):
    def __post_init__(self):
        self.img2tensor_fn = torchvision.transforms.ToTensor()
    
    def __call__(self, data: DataDict) -> DataDict:
        for key in data.keys():
            # Tight image-key guard: a loose ``"image" in key`` substring
            # match would misfire on unrelated keys like
            # ``action_image_mask`` or ``task.image_target``. Match only explicit
            # image slots: OBS_IMAGES prefix or the singleton OBS_IMAGE key. This
            # matches the tighter guards already used by ResizeImagesWithPadFn
            # (line 143) and ResizeShortestCenterCropFn (line 157).
            if key.startswith(OBS_IMAGES) or key == OBS_IMAGE:
                data[key] = self.img2tensor_fn(data[key])
            elif isinstance(data[key], list):
                data[key] = torch.tensor(data[key])
            elif isinstance(data[key], np.ndarray):
                data[key] = torch.from_numpy(data[key])
        return data


# Sentinel/companion suffixes carried alongside image tensors. Keys ending in
# these MUST NOT enter the resize path even though they include "image" as a
# substring (e.g. observation.images.primary_invalid is a bool, _is_pad is a
# bool tensor, _mask is an int per-vision-token vector). A loose
# `or "image" in k` would match these and crash at resize_with_pad.
_NON_IMAGE_KEY_SUFFIXES = ("_invalid", "_is_pad", "_mask")


def _is_image_tensor_key(k: str) -> bool:
    if any(k.endswith(suffix) for suffix in _NON_IMAGE_KEY_SUFFIXES):
        return False
    return k.startswith(OBS_IMAGES) or k == OBS_IMAGE or "image" in k


@DataTransformFn.register_subclass("resize_with_pad")
@dataclass
class ResizeImagesWithPadFn(DataTransformFn):
    height: int
    width: int
    mode: str = "bilinear"

    def __call__(self, data: DataDict) -> DataDict:
        for k, v in data.items():
            if _is_image_tensor_key(k):
                data[k] = resize_with_pad(v, self.height, self.width, self.mode)
        return data


@DataTransformFn.register_subclass("resize_center_crop")
@dataclass
class ResizeShortestCenterCropFn(DataTransformFn):
    height: int
    width: int
    mode: str = "bilinear"

    def __call__(self, data: DataDict) -> DataDict:
        for k, v in data.items():
            if _is_image_tensor_key(k):
                data[k] = resize_center_crop(v, self.height, self.width, self.mode)
        return data


@DataTransformFn.register_subclass("compose_fields")
@dataclass
class ComposeFieldsTransform(DataTransformFn):
    """
    Merge multiple keys' values into a single new key.

    Example:
        mapping = {
            "observation.state": [
                "observation.states.joint.position",
                "observation.states.effector.position",
            ]
            "action": [
                "actions.joint.position", 
                "actions.effector.position", 
            ]
        }
    """
    mapping: dict[str, list[str]] = field(default_factory=dict)

    def __call__(self, data: DataDict) -> DataDict:
        # Two-pass: first read + compute all merges, then write + pop. This
        # handles the case where the SAME src_key appears in multiple mappings
        # (e.g. oxe-auge joints-as-action: state_keys == action_keys ==
        # ("observation.joints",) — without two-phase order, iteration 1 pops
        # the key and iteration 2 KeyErrors).
        merged_by_new_key: dict[str, torch.Tensor] = {}
        keys_to_pop: set[str] = set()
        for new_key, src_keys in self.mapping.items():
            if len(src_keys) == 1 and src_keys[0] == new_key:
                continue
            merge_list = self._align_for_cat([data[k] for k in src_keys])
            merged_by_new_key[new_key] = torch.cat(merge_list, dim=-1)
            for k in src_keys:
                if k != new_key:
                    keys_to_pop.add(k)
        # Write merged values. new_keys override src keys; don't pop any
        # src_key that is itself a new_key (we just wrote it).
        for new_key, merged in merged_by_new_key.items():
            data[new_key] = merged
        for k in keys_to_pop:
            if k not in merged_by_new_key:
                data.pop(k, None)
        return data
    
    def _align_for_cat(self, tensors: list[torch.Tensor], dim=-1) -> list[torch.Tensor]:
        max_ndim = max((t.ndim for t in tensors))
        out = []
        for t in tensors:
            t = t if t.ndim == max_ndim else t.unsqueeze(dim)
            out.append(t)
        return out


@DataTransformFn.register_subclass("remap_image_key")
@dataclass
class RemapImageKeyTransformFn(DataTransformFn):
    """Remap image keys to unified `observation.images.imageN` slots.

    Downstream `Qwen3_VLProcessorTransformFn` iterates over exactly
    ``num_image_slots`` slots. If the schema declares fewer cameras, we fill
    the missing slots with zero-tensors AND `_mask=False` so normalization
    doesn't corrupt real data with artificial zero-centered padding signal.

    CRIT-04 fix: previously `num_image_slots` was hardcoded to 3, silently
    dropping the 4th+ camera when the schema declared more, and silently
    fabricating dummy slots when it declared fewer. Now the slot count is
    explicit and mismatches fail loud at hydrate_all time.

    Example::

        mapping = {
            "images.rgb.head":       f"{OBS_IMAGES}.image0",
            "images.rgb.hand_left":  f"{OBS_IMAGES}.image1",
            "images.rgb.hand_right": f"{OBS_IMAGES}.image2",
        }
    """
    mapping: dict[str, str] = field(default_factory=dict)
    # Hard upper bound on image slots consumed by Qwen3_VLProcessorTransformFn.
    # Authored in lockstep with the processor's `for i in range(3):` loop; if
    # the processor is extended to N slots, bump this (+ hydrate_all guard).
    num_image_slots: int = NUM_IMAGE_SLOTS

    def __call__(self, data: DataDict) -> DataDict:
        if len(self.mapping) > self.num_image_slots:
            raise ValueError(
                f"RemapImageKeyTransformFn: schema declares "
                f"{len(self.mapping)} cameras but only {self.num_image_slots} "
                f"slots are supported by the VLM processor. Drop the extras "
                f"from the schema's image_mapping or extend the processor."
            )
        for old_key, new_key in self.mapping.items():
            data[new_key] = data.pop(old_key)
            # Respect the adapter's `<old_key>_invalid` sentinel.
            # When _read_video_frame fell back to _zero_frame() (missing
            # mp4, decode error, index overshoot), the upstream adapter
            # tags the camera as invalid. Mark the slot mask=False so the
            # VLM processor treats it as padded — never as a real frame.
            invalid_key = f"{old_key}_invalid"
            is_valid = not bool(data.pop(invalid_key, False))
            data[f"{new_key}_mask"] = torch.tensor(is_valid, dtype=torch.bool)
        # Pad missing slots with zero tensors + False masks. Zero-images (not
        # white) keep normalization statistics benign: after standardization
        # a zero tensor maps to -mean/std, which the model quickly learns to
        # ignore via the mask=False signal.
        for i in range(len(self.mapping), self.num_image_slots):
            slot = f"{OBS_IMAGES}.image{i}"
            if slot not in data:
                # Safe reference image: first populated slot (image0 if the
                # schema declared ≥1 camera — guaranteed by upstream validation
                # that rejects zero-camera schemas in hydrate_all).
                ref_slot = f"{OBS_IMAGES}.image0"
                data[slot] = torch.zeros_like(data[ref_slot])
                data[f"{slot}_mask"] = torch.tensor(False, dtype=torch.bool)
        return data


def _resolve_mode(key: str, default_mode: str, mode_overrides: dict[str, str]) -> str:
    """Pick normalization mode for a key. If any override-substring matches the key,
    that mode wins; otherwise returns default_mode.

    L-03 fast-path: check for an exact key match first (O(1) dict lookup) — the
    hot path is callers pre-computing mode_overrides keyed by full-key, and the
    substring scan only exists for legacy "gripper"-style partial overrides.
    Exact match wins over substring; among substrings, first match (insertion
    order) wins.
    """
    # O(1) fast path: exact key match.
    exact = mode_overrides.get(key)
    if exact is not None:
        return exact
    # Legacy substring fallback (e.g. {"gripper": "q01_q99"}).
    for substr, override_mode in mode_overrides.items():
        if substr in key:
            return override_mode
    return default_mode


def _stats_has_q01_q99(stats: dict) -> bool:
    return "q01" in stats and "q99" in stats


# Plan-F: module-level _QUANTILE_FALLBACK_WARNED set removed in favor of
# utils.logging_utils.DedupeFilter installed at process startup.
def _warn_quantile_fallback_once(key: str, fallback: str, default_mode: str) -> None:
    from utils.logging_utils import warn_once
    warn_once(
        logging.getLogger(__name__),
        ("quantile_fallback", key, fallback),
        "[NormalizeTransformFn] Key %r mode=q01_q99 but stats lack q01/q99 — "
        "falling back to %r. Run compute_stats.py to regenerate stats "
        "with quantiles. (further occurrences for this key are suppressed)",
        key, fallback,
    )


@DataTransformFn.register_subclass("normalize")
@dataclass
class NormalizeTransformFn(DataTransformFn):
    """
    Normalize specified keys in a DataDict using precomputed statistics.

    Args:
        selected_keys: list of keys to normalize (e.g. ["observation.state", "actions"]).
            If None, will normalize all keys that exist in norm_stats.
        mode: default normalization mode — "mean_std" | "min_max" | "q01_q99".
        mode_overrides: per-key substring → mode override. E.g. {"gripper": "q01_q99"}
            makes any key containing "gripper" use [q01, q99] percentile normalization
            (robust to outliers; recommended for near-binary signals like gripper).
            If q01_q99 is requested but stats lack q01/q99, falls back to default mode.
        norm_stats: dictionary containing normalization parameters.

    Example:
        norm_stats = {
            "observation.state": {"mean": ..., "std": ..., "min": ..., "max": ...},
            "action": {"mean": ..., "std": ..., "q01": ..., "q99": ...},
        }
    """

    selected_keys: Optional[list[str]] = None
    mode: str = "mean_std"  # "mean_std" | "min_max" | "q01_q99"
    mode_overrides: dict[str, str] = field(default_factory=dict)
    # Plan A+ (V6): per-dim mode override within a single key. Format:
    #   dim_overrides = {key: {dim_idx: mode}}
    # Example: {"action": {7: "q01_q99"}, "observation.state": {7: "q01_q99"}}
    # — for the 8-dim "action" / "observation.state" key, dims 0..6 use the
    # default `mode` (typically mean_std for arm joints, preserves natural
    # z-score range), dim 7 uses q01_q99 (gripper width — bimodal/binary
    # distribution, q01/q99 matches better than mean_std). Empty dict
    # (default) preserves the legacy whole-key fast path bit-identically.
    # `hydrate_all` auto-injects these from `schema.gripper_action_dims` so
    # callers normally don't set this by hand.
    dim_overrides: dict[str, dict[int, str]] = field(default_factory=dict)
    norm_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Config-tunable for stats with tiny std values (near-constant channels);
    # default preserves the previous hardcoded 1e-6.
    eps: float = 1e-6

    def _resolve_with_fallback(self, mode: str, stats: dict, key_label: str) -> Optional[str]:
        """Resolve q01_q99 → mean_std → min_max fallback chain.

        ``q01_q99_strict`` is intentionally separate from the ordinary
        ``q01_q99`` fallback path. It is used for π0.5 state discretization:
        those bins are uniformly spaced over q01/q99-normalized [-1, 1], so
        falling back to z-score normalization silently reintroduces the
        boundary-bin saturation bug.
        """
        # Explicit "noop" mode passes through unchanged — used by hydrate_all
        # when --normalize_arm_joints=false or --normalize_gripper=false to
        # disable normalization for those dims without rewriting the chain.
        if mode == "noop":
            return "noop"
        if mode == "q01_q99_strict":
            if not _stats_has_q01_q99(stats):
                raise ValueError(
                    f"[NormalizeTransformFn] Key '{key_label}' requires "
                    "q01/q99 stats for state discretization, but they are "
                    "missing. Regenerate stats.json with q01/q99 fields."
                )
            return "q01_q99"
        if mode == "q01_q99" and not _stats_has_q01_q99(stats):
            if "mean" in stats and "std" in stats:
                fallback = "mean_std"
            elif "min" in stats and "max" in stats:
                fallback = "min_max"
            else:
                return None
            _warn_quantile_fallback_once(key_label, fallback, self.mode)
            return fallback
        return mode

    def _apply_mode(self, x: torch.Tensor, stats: dict, mode: str, key_label: str) -> torch.Tensor:
        """Apply one of the three normalization formulas to ``x`` using
        ``stats``. ``stats`` 1-D array fields must already be sliced to
        match ``x.shape[-1]`` — the caller (per-dim path) handles slicing
        via ``_slice_stats``; the whole-key path passes the original stats
        whose 1-D arrays line up with the full key dim already.
        """
        if mode == "mean_std":
            mean = torch.from_numpy(np.asarray(stats["mean"])).to(x)
            std = torch.from_numpy(np.asarray(stats["std"])).to(x)
            # Reject non-finite stats up-front — they silently
            # propagate NaN through every downstream op and surface as
            # "why did the loss become NaN at step 0" much later.
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
                raise ValueError(
                    f"[NormalizeTransformFn] non-finite mean/std for key "
                    f"'{key_label}'. Stats file is corrupted; re-run data_process stats."
                )
            return (x - mean) / (std + self.eps)
        if mode == "min_max":
            min_v = torch.from_numpy(np.asarray(stats["min"])).to(x)
            max_v = torch.from_numpy(np.asarray(stats["max"])).to(x)
            if not torch.isfinite(min_v).all() or not torch.isfinite(max_v).all():
                raise ValueError(
                    f"[NormalizeTransformFn] non-finite min/max for key '{key_label}'."
                )
            return (x - min_v) / (max_v - min_v + self.eps)
        if mode == "q01_q99":
            q01 = torch.from_numpy(np.asarray(stats["q01"])).to(x)
            q99 = torch.from_numpy(np.asarray(stats["q99"])).to(x)
            if not torch.isfinite(q01).all() or not torch.isfinite(q99).all():
                raise ValueError(
                    f"[NormalizeTransformFn] non-finite q01/q99 for key '{key_label}'."
                )
            # pi0/OpenVLA-style: rescale q01..q99 → [-1, 1] without clipping
            return 2 * (x - q01) / (q99 - q01 + self.eps) - 1
        if mode == "noop":
            # Identity for ablation toggles. dim_overrides assigns this when
            # --normalize_arm_joints=false or --normalize_gripper=false.
            return x
        raise ValueError(f"Unknown normalization mode: {mode}")

    @staticmethod
    def _slice_stats(stats: dict, dims: list[int]) -> dict:
        """Slice every 1-D array in ``stats`` by the given dim indices. Scalars
        (e.g. ``count``) and short arrays pass through unchanged so finite-
        check in ``_apply_mode`` can still error cleanly on bad input.
        """
        idx = np.asarray(dims, dtype=np.int64)
        out = {}
        for k, v in stats.items():
            try:
                n = len(v)
            except TypeError:
                out[k] = v
                continue
            if n > int(idx.max()):
                out[k] = np.asarray(v)[idx]
            else:
                out[k] = v
        return out

    def __call__(self, data: DataDict) -> DataDict:
        keys = self.selected_keys if self.selected_keys is not None else list(self.norm_stats.keys())

        for key in keys:
            if key not in data:
                logging.warning(
                    f"[NormalizeTransformFn] Key '{key}' not found in data — skipping normalization."
                )
                continue
            if key not in self.norm_stats:
                logging.warning(
                    f"[NormalizeTransformFn] No normalization stats found for key '{key}' — skipping."
                )
                continue

            x = data[key]
            stats = self.norm_stats[key]
            default_mode = _resolve_mode(key, self.mode, self.mode_overrides)
            per_dim = self.dim_overrides.get(key) or {}

            if not per_dim:
                # Fast path: whole-key normalization. Bit-identical to the
                # legacy single-mode body when `dim_overrides` is empty —
                # this preserves backward compat for every existing caller.
                resolved = self._resolve_with_fallback(default_mode, stats, key)
                if resolved is None:
                    logging.warning(
                        f"[NormalizeTransformFn] Key '{key}' mode=q01_q99 but stats lack "
                        f"q01/q99 AND mean/std AND min/max — skipping."
                    )
                    continue
                data[key] = self._apply_mode(x, stats, resolved, key)
                continue

            # Per-dim path: group dims by effective mode (one tensor op per
            # mode rather than per-dim), normalize each group with sliced
            # stats, write back into a clone.
            #
            # 0-dim (scalar) handling: some adapters store single-feature keys
            # (e.g. robointer_droid `*_gripper_position` with state_dims=(1,))
            # as a 0-dim tensor — torch.tensor(0.04) rather than
            # torch.tensor([0.04]). `x.shape[-1]` on a 0-dim tensor raises
            # IndexError. Upgrade to 1-dim before per-dim slicing and restore the
            # original rank after so callers see the same shape they handed in.
            was_scalar = (x.ndim == 0)
            if was_scalar:
                x = x.unsqueeze(0)

            D = x.shape[-1]
            mode_to_dims: dict[str, list[int]] = {}
            for d in range(D):
                m = per_dim.get(d, default_mode)
                mode_to_dims.setdefault(m, []).append(d)

            out = x.clone()
            for raw_mode, dims in mode_to_dims.items():
                resolved = self._resolve_with_fallback(raw_mode, stats, key)
                if resolved is None:
                    logging.warning(
                        f"[NormalizeTransformFn] Key '{key}' dims={dims} mode={raw_mode} "
                        f"but stats lack required fields — skipping these dims."
                    )
                    continue
                sub_stats = self._slice_stats(stats, dims)
                idx_t = torch.tensor(dims, dtype=torch.long, device=x.device)
                sub_x = x.index_select(-1, idx_t)
                sub_out = self._apply_mode(sub_x, sub_stats, resolved, f"{key}[dims={dims}]")
                out.index_copy_(-1, idx_t, sub_out.to(out.dtype))

            if was_scalar:
                out = out.squeeze(0)
            data[key] = out
        return data


@DataTransformFn.register_subclass("unnormalize")
@dataclass
class UnNormalizeTransformFn(DataTransformFn):
    """
    Unnormalize specified keys in a DataDict using precomputed statistics.

    Args:
        selected_keys: list of keys to unnormalize (e.g. ["observation.state", "actions"]).
            If None, will unnormalize all keys that exist in norm_stats.
        mode: default unnormalization mode — "mean_std" | "min_max" | "q01_q99".
        mode_overrides: per-key substring → mode override. Mirrors
            NormalizeTransformFn's semantics; pair the two transforms identically.
        norm_stats: dictionary containing unnormalization parameters.

    Example:
        norm_stats = {
            "observation.state": {"mean": ..., "std": ..., "min": ..., "max": ...},
            "action": {"mean": ..., "std": ..., "q01": ..., "q99": ...},
        }
    """

    selected_keys: Optional[list[str]] = None
    # Default q01_q99 to mirror NormalizeTransformFn (π0.5 / openpi convention).
    mode: str = "q01_q99"
    mode_overrides: dict[str, str] = field(default_factory=dict)
    # Symmetric per-dim mode override, mirroring
    # NormalizeTransformFn.dim_overrides. ``hydrate_all`` injects the SAME
    # dim_overrides into both transforms, so a key normalized with mixed modes
    # (e.g. arm joints mean_std / gripper q01_q99 / noop ablation) is
    # un-normalized as the exact per-dim inverse. Empty dict (default)
    # preserves the legacy whole-key fast path bit-identically.
    dim_overrides: dict[str, dict[int, str]] = field(default_factory=dict)
    norm_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Mirror NormalizeTransformFn.eps. Keep these two in sync when tuning: a
    # forward/inverse eps mismatch produces slow drift that is hard to isolate.
    eps: float = 1e-6

    # Reuse NormalizeTransformFn's mode resolution so the forward/inverse pair
    # resolve every per-dim mode identically (q01_q99→mean_std fallback, noop
    # passthrough, q01_q99_strict). This is the same per-dim mode resolution
    # the forward transform uses, applied to the inverse formulas below.
    _resolve_with_fallback = NormalizeTransformFn._resolve_with_fallback

    def _apply_inverse_mode(
        self, x: torch.Tensor, stats: dict, mode: str, key_label: str
    ) -> torch.Tensor:
        """Inverse of ``NormalizeTransformFn._apply_mode`` for one mode.

        ``stats`` 1-D array fields must already be sliced to match
        ``x.shape[-1]`` — the per-dim path slices via
        ``NormalizeTransformFn._slice_stats``; the whole-key path passes
        original stats whose 1-D arrays line up with the full key dim.
        """
        if mode == "mean_std":
            mean = torch.from_numpy(np.asarray(stats["mean"])).to(x)
            std = torch.from_numpy(np.asarray(stats["std"])).to(x)
            return x * (std + self.eps) + mean
        if mode == "min_max":
            min_v = torch.from_numpy(np.asarray(stats["min"])).to(x)
            max_v = torch.from_numpy(np.asarray(stats["max"])).to(x)
            return x * (max_v - min_v + self.eps) + min_v
        if mode == "q01_q99":
            q01 = torch.from_numpy(np.asarray(stats["q01"])).to(x)
            q99 = torch.from_numpy(np.asarray(stats["q99"])).to(x)
            # inverse of (2*(x-q01)/(q99-q01)-1) → x = (x+1)/2 * (q99-q01) + q01
            return (x + 1) / 2 * (q99 - q01 + self.eps) + q01
        if mode == "noop":
            return x
        raise ValueError(f"Unknown unnormalization mode: {mode}")

    def __call__(self, data: DataDict) -> DataDict:
        keys = self.selected_keys if self.selected_keys else list(self.norm_stats.keys())

        for key in keys:
            if key not in data:
                logging.warning(
                    f"[UnNormalizeTransformFn] Key '{key}' not found in data — skipping unnormalization."
                )
                continue
            if key not in self.norm_stats:
                logging.warning(
                    f"[UnNormalizeTransformFn] No stats found for key '{key}' — skipping unnormalization."
                )
                continue

            x = data[key]
            stats = self.norm_stats[key]
            default_mode = _resolve_mode(key, self.mode, self.mode_overrides)
            per_dim = self.dim_overrides.get(key) or {}

            if not per_dim:
                # Fast path: whole-key un-normalization. Bit-identical to the
                # legacy single-mode body when `dim_overrides` is empty —
                # preserves backward compat for every existing caller.
                resolved = self._resolve_with_fallback(default_mode, stats, key)
                if resolved is None:
                    logging.warning(
                        f"[UnNormalizeTransformFn] Key '{key}' mode=q01_q99 but stats lack "
                        f"q01/q99 AND mean/std AND min/max — skipping."
                    )
                    continue
                data[key] = self._apply_inverse_mode(x, stats, resolved, key)
                continue

            # Per-dim path: group dims by effective mode (one tensor op per
            # mode), un-normalize each group with sliced stats, write back into
            # a clone. Mirrors NormalizeTransformFn.__call__ exactly so the
            # inverse is the exact per-dim inverse of the forward transform.
            was_scalar = (x.ndim == 0)
            if was_scalar:
                x = x.unsqueeze(0)

            D = x.shape[-1]
            mode_to_dims: dict[str, list[int]] = {}
            for d in range(D):
                m = per_dim.get(d, default_mode)
                mode_to_dims.setdefault(m, []).append(d)

            out = x.clone()
            for raw_mode, dims in mode_to_dims.items():
                resolved = self._resolve_with_fallback(raw_mode, stats, key)
                if resolved is None:
                    logging.warning(
                        f"[UnNormalizeTransformFn] Key '{key}' dims={dims} mode={raw_mode} "
                        f"but stats lack required fields — skipping these dims."
                    )
                    continue
                sub_stats = NormalizeTransformFn._slice_stats(stats, dims)
                idx_t = torch.tensor(dims, dtype=torch.long, device=x.device)
                sub_x = x.index_select(-1, idx_t)
                sub_out = self._apply_inverse_mode(sub_x, sub_stats, resolved, f"{key}[dims={dims}]")
                out.index_copy_(-1, idx_t, sub_out.to(out.dtype))

            if was_scalar:
                out = out.squeeze(0)
            data[key] = out

        return data


@DataTransformFn.register_subclass("delta_action")
@dataclass
class DeltaActionTransformFn(DataTransformFn):
    # Every construction site leaves `mask=None` at ctor time and relies on
    # ``hydrate_all()`` to inject ``schema.to_bool_mask()`` before the transform
    # runs. The `mask is None` raise below is the authoritative guard. A new
    # construction site MUST either route through hydrate_all or pass
    # mask=schema.to_bool_mask() explicitly — never silently broadcast a default.

    mask: Optional[list[bool]] = None
    mapping: dict[str, list[str]] = field(default_factory=dict)

    def __call__(self, data: DataDict) -> DataDict:
        """Convert absolute action chunks → delta-from-state chunks.

        Contract:
          1. ``data[state_keys[i]]`` is a single-frame state. If it has a
             leading time axis (e.g. when state_keys overlaps action_keys
             and delta_timestamps prepends a chunk), the FIRST row
             ``state[0]`` is taken as the t=0 reference. This requires the
             caller to guarantee ``delta_timestamps[<state-key>][0] == 0.0``
             (LabVLADatasetConfig builds delta_timestamps with
             ``[i / fps for i in range(chunk_size)]`` so index 0 is t=0).
          2. ``data[action_keys[i]]`` is a (chunk, action_dim) tensor whose
             chunk[0] aligns with the t=0 state.
          3. ``self.mask`` MUST be hydrated by ``hydrate_all`` to the
             schema-derived bool tensor; no silent fallback.
        """
        state_keys = self.mapping[OBS_STATE]
        state_list, _ = self._align_for_cat([data[k] for k in state_keys])
        state = torch.cat(state_list, dim=-1)
        action_keys = self.mapping[ACTION]
        action_list, size = self._align_for_cat([data[k] for k in action_keys])
        action = torch.cat(action_list, dim=-1)
        # When state_keys overlap action_keys (e.g. oxe-auge joints-as-action
        # schemas where state_keys == action_keys == ("observation.joints",)),
        # the adapter's delta_timestamps expansion writes the chunked future
        # window to that key AFTER placing the single-frame state there. By
        # the time we read it here, `state` is (chunk, state_dim). The current
        # frame is always chunk[0] because delta_timestamps starts at 0.0, so
        # we can recover the intended 1-D state by slicing [0].
        if state.ndim > 1:
            state = state[0]
        # Guard against true dimension bugs (3+ axes, 0-dim, etc.).
        assert state.ndim == 1, (
            f"DeltaActionTransformFn expects 1-D state per sample "
            f"(state_dim,), got shape {tuple(state.shape)}. If you need "
            f"chunked state, update this transform's broadcasting logic."
        )
        # No silent fallback: a `mask = [True] * state.shape[-1]` default would
        # accept a missing mask and could mis-align when state_dim != action_dim.
        # hydrate_all always injects schema.to_bool_mask() whose length equals
        # sum(action_dims). Require it explicitly — no silent fallback.
        if self.mask is None:
            raise ValueError(
                "DeltaActionTransformFn.mask is None. hydrate_all must inject "
                "schema.to_bool_mask() before this transform runs. If constructing "
                "manually (tests, ablations), pass mask=schema.to_bool_mask()."
            )
        if self.mask.shape[-1] != action.shape[-1]:
            raise ValueError(
                f"DeltaActionTransformFn.mask length {self.mask.shape[-1]} does "
                f"not equal concatenated action width {action.shape[-1]}. This "
                f"usually means schema.delta_mask length disagrees with "
                f"sum(schema.action_dims); re-check the schema author-side."
            )
        # When state and action widths differ (e.g. robocoin's state has
        # 32 dims but action has 32 dims — equal — vs a future robot with
        # state_dim=14 but action_dim=32), subtraction of (state_dim,) from
        # (..., action_dim) requires explicit padding. Pad state with zeros
        # on the right so only the first `state_dim` mask bits effect a delta.
        if state.shape[-1] != action.shape[-1]:
            if state.shape[-1] > action.shape[-1]:
                raise ValueError(
                    f"DeltaActionTransformFn: state width {state.shape[-1]} "
                    f"exceeds action width {action.shape[-1]}; cannot align."
                )
            pad = action.shape[-1] - state.shape[-1]
            state = torch.cat([state, state.new_zeros(pad)], dim=-1)
        action -= torch.where(self.mask, state, 0)[None]
        sid, eid = 0, 0
        for i, key in enumerate(action_keys):
            eid += size[i]
            data[key] = action[..., sid:eid]
            sid = eid
        return data
    
    def _align_for_cat(self, tensors: list[torch.Tensor], dim=-1) -> list[torch.Tensor]:
        max_ndim = max((t.ndim for t in tensors))
        out, size = [], []
        for t in tensors:
            t = t if t.ndim == max_ndim else t.unsqueeze(dim)
            out.append(t)
            size.append(t.shape[-1])
        return out, size


# Phase 4: hydrate_normalize_transform / hydrate_compose_field_transform /
# hydrate_delta_action_transform / hydrate_remap_image_key_transform /
# filter_image_features were deleted. Every caller now uses hydrate_all(schema=...).
# See doc/schema_manifest.md for migration guidance if you hit an import error.


# ============== New unified hydrate interface using DatasetSchema ==============

def hydrate_all(
    transforms: list[DataTransformFn],
    schema,
    stats: dict | None = None,
    action_mode: str = "delta",
    normalize_arm_joints: bool = True,
    normalize_gripper: bool | str = True,
) -> list[DataTransformFn]:
    """Hydrate all transforms using a DatasetSchema.

    action_mode:
        "delta" → action normalization uses stats["action"] (delta-transformed,
                  what compute_stats wrote when delta_mask-True dims were
                  subtracted from state).
        "abs"   → action normalization uses stats["action_abs"] (raw absolute
                  action values, no delta). Requires compute_stats to have
                  produced the "action_abs" key; otherwise a clear error is
                  raised.

    Usage: hydrate_all(transforms, schema=dataset_schema, stats=stats).
    """
    if action_mode not in ("delta", "abs"):
        raise ValueError(f"action_mode must be 'delta' or 'abs', got {action_mode!r}")
    if isinstance(normalize_gripper, str):
        gripper_norm_mode = normalize_gripper.strip().lower().replace("-", "_")
        bool_aliases = {
            "true": "q01_q99",
            "1": "q01_q99",
            "yes": "q01_q99",
            "false": "noop",
            "0": "noop",
            "no": "noop",
            "none": "noop",
        }
        gripper_norm_mode = bool_aliases.get(gripper_norm_mode, gripper_norm_mode)
    else:
        gripper_norm_mode = "q01_q99" if normalize_gripper else "noop"
    if gripper_norm_mode not in ("q01_q99", "mean_std", "noop"):
        raise ValueError(
            "normalize_gripper must be bool or one of "
            f"'q01_q99', 'mean_std', 'noop'; got {normalize_gripper!r}"
        )
    if schema is None:
        raise ValueError(
            "hydrate_all: `schema` is required. Pass "
            "adapter.meta.schema (populated by schema.discover_schema)."
        )

    # Qwen3_VLProcessorTransformFn iterates image0..imageN slots.
    # If schema declares more, silently dropping them is a correctness hazard;
    # raise so the user partitions cameras or drops a few in the manifest.
    _MAX_CAMERAS = NUM_IMAGE_SLOTS
    if len(schema.image_mapping) > _MAX_CAMERAS:
        raise ValueError(
            f"hydrate_all: schema {schema.schema_id!r} has "
            f"{len(schema.image_mapping)} cameras but the current VLM "
            f"processor only consumes {_MAX_CAMERAS} (image0..image{_MAX_CAMERAS-1}). "
            f"Trim the images dict in the manifest to the {_MAX_CAMERAS} most "
            f"informative cameras, or extend Qwen3_VLProcessorTransformFn "
            f"+ UnifyLabVLAInputsTransformFn to consume more slots."
        )

    # Core loop: every transform pulls what it needs from the schema.
    feature_map = {
        OBS_STATE: list(schema.state_keys),
        ACTION: list(schema.action_keys),
    }
    bool_mask = schema.to_bool_mask()
    selected_keys = list(schema.state_keys) + list(schema.action_keys)
    stats = _canonicalize_dual_arm_stats(stats, schema)
    # Single-arm equivalent: remap raw 11/12-dim UR/festo/rizon4 stats to
    # the canonical 8-dim layout that CanonicalSingleArmLayoutTransformFn
    # builds at training time. No-op for canonical 8-dim Franka schemas
    # (those have empty source_*_keys).
    stats = _canonicalize_single_arm_stats(stats, schema)

    # Lazy import to avoid a cycle: transform_labvla imports from transforms.core.
    from policies.LabVLA.transform_labvla import UnifyLabVLAInputsTransformFn
    # Annotation tokenize transform needs schema.annotation_losses injected.
    from transforms.annotation_tokenize import AnnotationTokenizeTransformFn
    # State-discretize (vlm_pretrain) needs schema.state_keys for split-state
    # schemas where the transform runs BEFORE ComposeFieldsTransform.
    from transforms.state_discretize import DiscretizeStateTransformFn
    # AgiBot subtask builder: enabled iff schema.annotation_losses declares
    # annotation.subtask. Otherwise stays no-op.
    from transforms.agibot_subtask import BuildAgiBotSubtaskTransformFn
    # FAST action tokenizer can tighten its source-local max_length once the
    # dataset schema is known.
    from transforms.fast_action import FastActionEncodeTransformFn

    _has_subtask_loss = any(
        getattr(spec, "field", None) == "annotation.subtask"
        for spec in (schema.annotation_losses or ())
    )

    hydrated = []
    for t in transforms:
        if isinstance(t, CanonicalArmLayoutTransformFn):
            layout = getattr(schema, "arm_layout", None)
            try:
                from schema.arm_layout import ArmCount
                is_dual_arm_layout = getattr(layout, "arm_count", None) == ArmCount.DUAL
            except Exception:
                is_dual_arm_layout = False
            left_gripper_index = getattr(layout, "left_gripper_index_in_raw", None)
            right_gripper_index = getattr(layout, "right_gripper_index_in_raw", None)
            enabled = bool(
                layout is not None
                and is_dual_arm_layout
                and getattr(schema, "source_state_keys", ())
                and getattr(schema, "source_action_keys", ())
                and left_gripper_index is not None
                and right_gripper_index is not None
            )
            t = replace(
                t,
                enabled=enabled,
                left_arm_dof=int(getattr(layout, "left_arm_dof", 0) or 0),
                right_arm_dof=int(getattr(layout, "right_arm_dof", 0) or 0),
                left_gripper_index_in_raw=int(left_gripper_index or -1),
                right_gripper_index_in_raw=int(right_gripper_index or -1),
                state_source_keys=tuple(getattr(schema, "source_state_keys", ()) or ()),
                action_source_keys=tuple(getattr(schema, "source_action_keys", ()) or ()),
            )
            logging.info(
                f"Hydrated {t.__class__.__name__} enabled={t.enabled} "
                f"state_sources={t.state_source_keys} action_sources={t.action_source_keys} "
                f"({schema.schema_id})"
            )
        elif isinstance(t, CanonicalSingleArmLayoutTransformFn):
            layout = getattr(schema, "arm_layout", None)
            try:
                from schema.arm_layout import ArmCount
                is_single_arm_layout = (
                    getattr(layout, "arm_count", None) == ArmCount.SINGLE
                )
            except Exception:
                is_single_arm_layout = False
            grip_idx = getattr(layout, "gripper_index_in_raw", None)
            arm_dof = getattr(layout, "arm_dof", None)
            enabled = bool(
                layout is not None
                and is_single_arm_layout
                and getattr(schema, "source_state_keys", ())
                and getattr(schema, "source_action_keys", ())
                and grip_idx is not None
                and arm_dof is not None
            )
            t = replace(
                t,
                enabled=enabled,
                raw_arm_dof=int(arm_dof or 0),
                raw_gripper_index_in_raw=int(grip_idx if grip_idx is not None else -1),
                state_source_keys=tuple(
                    getattr(schema, "source_state_keys", ()) or ()
                ),
                action_source_keys=tuple(
                    getattr(schema, "source_action_keys", ()) or ()
                ),
            )
            logging.info(
                f"Hydrated {t.__class__.__name__} enabled={t.enabled} "
                f"raw_arm_dof={t.raw_arm_dof} raw_grip_idx={t.raw_gripper_index_in_raw} "
                f"state_sources={t.state_source_keys} action_sources={t.action_source_keys} "
                f"({schema.schema_id})"
            )
        elif isinstance(t, GripperSemanticCanonicalizeFn):
            # Convert source gripper semantic (width / position) to the
            # canonical continuous open_fraction target so multi-source
            # posttrain MSE sees a unified gripper signal across all repos.
            src_sem = getattr(schema, "gripper_semantic", None)
            grip_dims = tuple(getattr(schema, "gripper_action_dims", ()) or ())
            grip_dim = int(grip_dims[0]) if grip_dims else 7
            # Hardcoded calibration tables (extend here as new sources
            # come in with calibrated endpoints):
            #   width  → open_fraction with Franka spec [0, 0.04]
            #   open_fraction → open_fraction is no-op (enabled=False)
            _CALIBRATION = {
                "width":         dict(closed=0.0, open=0.04, direction=+1),
                "open_fraction": None,  # already canonical
            }
            cal = _CALIBRATION.get(src_sem)
            enabled = bool(src_sem and cal is not None and src_sem != "open_fraction")
            if enabled:
                t = replace(
                    t,
                    enabled=True,
                    source_semantic=src_sem,
                    target_semantic="open_fraction",
                    closed=float(cal["closed"]),
                    open=float(cal["open"]),
                    direction=int(cal["direction"]),
                    gripper_dim=grip_dim,
                    state_keys=tuple(schema.state_keys),
                    action_keys=tuple(schema.action_keys),
                )
                # Apply the same affine to gripper-dim stats so normalize
                # downstream uses open_fraction-domain q01/q99/mean/std.
                _calibration_for_stats = dict(
                    enabled=True,
                    source_semantic=src_sem,
                    target_semantic="open_fraction",
                    closed=float(cal["closed"]),
                    open=float(cal["open"]),
                    direction=int(cal["direction"]),
                    gripper_dim=grip_dim,
                )
                stats = _canonicalize_gripper_semantic_stats(
                    stats, schema, _calibration_for_stats
                )
            else:
                t = replace(t, enabled=False, source_semantic=src_sem or "")
            logging.info(
                f"Hydrated {t.__class__.__name__} enabled={t.enabled} "
                f"source_semantic={t.source_semantic} target=open_fraction "
                f"({schema.schema_id})"
            )
        elif isinstance(t, DeltaActionTransformFn):
            t = replace(t, mapping=feature_map, mask=bool_mask)
            logging.info(
                f"Hydrated {t.__class__.__name__} with mask ({schema.schema_id})"
            )
        elif isinstance(t, UnifyLabVLAInputsTransformFn):
            # Supply action_keys so UnifyLabVLAInputsTransformFn can locate
            # per-sub-key `_is_pad` tensors (OR-aggregation → action_is_pad).
            t = replace(t, action_keys=tuple(schema.action_keys))
            logging.info(
                f"Hydrated {t.__class__.__name__} with {len(t.action_keys)} "
                f"action_keys ({schema.schema_id})"
            )
        elif isinstance(t, NormalizeTransformFn):
            # Structural guard: `stats is None` AND empty-dict both count
            # as "stats missing". Without this second check a dataset whose
            # meta/stats.json is missing would pass an empty dict, then
            # NormalizeTransformFn would log a per-key warning and skip — i.e.
            # silently train on un-normalized data.
            if not stats:
                raise FileNotFoundError(
                    "NormalizeTransformFn requires non-empty stats but got "
                    f"{stats!r}. Run: python -m data_process stats --dataset "
                    "<dataset_root> --schema <schema_name>"
                )
            # `data_process stats` emits canonical-concatenated entries keyed as
            # `observation.state` and `action` — the state/action vectors AFTER
            # ComposeFieldsTransform would concatenate schema.state_keys /
            # action_keys. But NormalizeTransformFn runs BEFORE Compose in the
            # chain, so it sees per-schema-key tensors and looks stats up by the
            # schema's own key names. Build a per-schema-key view by slicing the
            # canonical stats along each key's dim offset.
            expanded = dict(stats)  # preserve the canonical entries for compat
            # action_mode chooses which canonical action stats to slice per-key.
            action_canonical_key = "action_abs" if action_mode == "abs" else "action"
            if action_mode == "abs" and "action_abs" not in stats:
                raise KeyError(
                    f"hydrate_all: action_mode='abs' requires stats['action_abs'] "
                    f"but it is missing in stats.json. Re-run `data_process stats` "
                    f"(the newer version writes both 'action' and 'action_abs'). "
                    f"stats keys present: {list(stats.keys())}"
                )
            for canonical, keys, dims in [
                ("observation.state",    list(schema.state_keys),  list(schema.state_dims)),
                (action_canonical_key,   list(schema.action_keys), list(schema.action_dims)),
            ]:
                base = stats.get(canonical)
                if base is None or not isinstance(base, dict):
                    continue
                total_expected = sum(dims)
                offset = 0
                for k, d in zip(keys, dims):
                    sub = {}
                    for sk, v in base.items():
                        # Slice array-like fields; copy metadata (count, …) as-is.
                        # ``count`` may arrive as a 1-element list (legacy v8
                        # stats_delta.json shape) or a scalar — both must
                        # bypass the per-dim slice/length check.
                        if sk == "count":
                            sub[sk] = v
                            continue
                        try:
                            _len = len(v)
                        except TypeError:
                            sub[sk] = v
                            continue
                        if _len == 0:
                            # Empty stats array — treat as corrupted but keep
                            # the original (caller may have intentionally
                            # cleared the field). Warn once with enough
                            # context (schema_id + key + canonical + the
                            # NormalizeTransformFn offset) to debug without
                            # re-opening the stats file by hand.
                            logging.warning(
                                "[hydrate_all] empty stats array: "
                                "stats[%r][%r] is len=0; schema_id=%s, "
                                "canonical=%r, per-schema-key=%r, "
                                "expected_dim=%d, slice_offset=%d, "
                                "total_expected=%d. Passing through as-is — "
                                "NormalizeTransformFn will see an empty field "
                                "and likely skip normalization for this key. "
                                "Re-run `python -m data_process stats` to "
                                "regenerate a full stats.json if this is "
                                "unexpected.",
                                canonical, sk, schema.schema_id,
                                canonical, k, d, offset, total_expected,
                            )
                            sub[sk] = v
                        elif offset + d > _len:
                            # Do NOT silently return the whole (wrongly-sized)
                            # array here: that would cause downstream
                            # NormalizeTransformFn to normalize a `d`-dim key
                            # against ``_len``-dim stats — wrong numerics with
                            # NO error. Now fail loud.
                            raise ValueError(
                                f"[hydrate_all] stats[{canonical!r}][{sk!r}] has "
                                f"length {_len}, but schema {schema.schema_id!r} "
                                f"requires slice [{offset}:{offset + d}] for key "
                                f"{k!r}. Expected total = sum({canonical}_dims) "
                                f"= {total_expected}. Regenerate stats.json "
                                f"(`python -m data_process stats ...`) or check "
                                f"that the schema matches this dataset's layout."
                            )
                        else:
                            sub[sk] = v[offset:offset + d]
                    expanded[k] = sub
                    offset += d

            # Plan A+ (V6): auto-build per-dim mode overrides.
            #
            # VLM-pretrain discretization exception:
            #   π0.5 bins are uniform over q01/q99-normalized [-1, 1]. If dims
            #   stay on mean_std, roughly 32% of near-Gaussian joint values fall
            #   outside [-1, 1] and collapse into boundary bins. Therefore,
            #   whenever DiscretizeStateTransformFn is present in the input
            #   chain, every state AND action dim is forced to q01/q99 strictly.
            #   Posttrain / π0-style knowledge-isolation runs do not include
            #   DiscretizeStateTransformFn, so they keep the mixed policy below:
            #   arm joints mean_std, gripper q01/q99.
            #
            # Action/gripper policy:
            #   The canonical gripper dim (post-Compose, in the concatenated
            #   action vector) is declared by schema.gripper_action_dims (and
            #   arm_layout.gripper_indices_canonical which mirrors it). Map each
            #   canonical gripper dim back to the *local* dim within whichever
            #   schema-key contains it, then request q01_q99 on those local dims.
            #   Default mode (mean_std, set at config-time) applies to remaining
            #   action arm dims.
            #
            # Why this is the right shape for action data:
            #   - arm joints span a wide near-Gaussian range → mean_std
            #     preserves natural z-score precision; q01_q99 would
            #     compress the working region by ~3× and erode spatial
            #     accuracy (this is the regression that hurt v11.x).
            #   - gripper width is bimodal (mostly fully open + brief
            #     closure events) → q01_q99 [-1, 1] matches the bimodal
            #     boundaries; mean_std gives huge mean-distance tails on
            #     the closed mode.
            #
            # Concretely for LabUtopia single-arm Franka:
            #   schema.state_keys = ('observation.state',), state_dims = (8,)
            #   schema.action_keys = ('action',), action_dims = (8,)
            #   schema.gripper_action_dims = (7,)
            #   → dim_overrides = {
            #       "observation.state": {7: "q01_q99"},
            #       "action":            {7: "q01_q99"},
            #     }
            # For multi-key schemas (robointer_droid: separate
            # joint/gripper keys), the gripper canonical dim falls inside
            # the gripper-specific key at local dim 0, so the override
            # fires on the whole 1-dim gripper key — semantically equivalent
            # to v8_ok's `mode_overrides={"gripper": "q99"}`.
            gripper_canonical = tuple(getattr(schema, "gripper_action_dims", ()) or ())
            dim_overrides: dict[str, dict[int, str]] = {}
            has_state_discretize = any(
                isinstance(candidate, DiscretizeStateTransformFn)
                for candidate in transforms
            )
            # Per-segment normalization toggle. By default both arm joints and
            # gripper are normalized. Setting one to False makes those dims pass
            # through identity ("noop" mode), letting the model see raw values
            # for that segment — matches V8 robointer_droid pretrain behavior
            # where a stats-key naming bug accidentally skipped normalization.
            for canonical, keys, dims in [
                ("observation.state",  list(schema.state_keys),  list(schema.state_dims)),
                (action_canonical_key, list(schema.action_keys), list(schema.action_dims)),
            ]:
                base = stats.get(canonical)
                if base is None or not isinstance(base, dict):
                    continue
                offset = 0
                for k, d in zip(keys, dims):
                    for local in range(d):
                        global_dim = offset + local
                        is_gripper = global_dim in gripper_canonical
                        if has_state_discretize:
                            mode = "q01_q99_strict"
                            dim_overrides.setdefault(k, {})[local] = mode
                        elif is_gripper:
                            mode = gripper_norm_mode
                            # only emit override when it differs from default ("mean_std")
                            if mode != "mean_std":
                                dim_overrides.setdefault(k, {})[local] = mode
                        else:
                            if not normalize_arm_joints:
                                dim_overrides.setdefault(k, {})[local] = "noop"
                            # else: keep default mean_std via fast-path (no override)
                    offset += d

            # Fail LOUD at hydrate time when a normalized key
            # has no stats, instead of NormalizeTransformFn silently warning +
            # skipping at runtime. A missing canonical entry
            # ('observation.state' / action_canonical_key) leaves the per-key
            # slices absent from `expanded`, so those keys would be trained
            # UN-normalized while other keys are normalized — a silent
            # train-correctness break that is very hard to notice in a long run.
            # Escape hatch for intentional edge configs (VQA-only / fully-noop
            # ablations / partial stats): LABVLA_ALLOW_MISSING_NORM_STATS=1.
            _missing_norm_keys = [
                k for k in selected_keys if not isinstance(expanded.get(k), dict)
            ]
            if _missing_norm_keys:
                _nm_msg = (
                    f"[hydrate_all] NormalizeTransformFn has NO stats for key(s) "
                    f"{_missing_norm_keys} (schema {schema.schema_id!r}). The canonical "
                    f"stats entry ('observation.state' / {action_canonical_key!r}) was "
                    f"absent or malformed, so these keys would be SILENTLY left "
                    f"un-normalized at runtime while other keys ARE normalized — a "
                    f"train-correctness break. Regenerate stats.json via "
                    f"`python -m data_process stats ...`. Present stats keys: "
                    f"{list(stats.keys())}."
                )
                if os.environ.get("LABVLA_ALLOW_MISSING_NORM_STATS") == "1":
                    logging.error(_nm_msg + " [BYPASSED via LABVLA_ALLOW_MISSING_NORM_STATS=1]")
                else:
                    raise ValueError(
                        _nm_msg + " Set LABVLA_ALLOW_MISSING_NORM_STATS=1 to bypass intentionally."
                    )

            t = replace(
                t,
                norm_stats=expanded,
                selected_keys=selected_keys,
                dim_overrides=dim_overrides,
            )
            logging.info(
                f"Hydrated {t.__class__.__name__} with {len(selected_keys)} keys "
                f"({schema.schema_id}); action_mode={action_mode}; sliced from "
                f"[state='observation.state' ({'observation.state' in stats}), "
                f"action={action_canonical_key!r} ({action_canonical_key in stats})]; "
                f"dim_overrides={dim_overrides if dim_overrides else 'none'}"
            )
        elif isinstance(t, ComposeFieldsTransform):
            t = replace(t, mapping=feature_map)
            logging.info(f"Hydrated {t.__class__.__name__} ({schema.schema_id})")
        elif isinstance(t, SnapGripperToEndpointsFn):
            t = replace(
                t,
                state_keys=tuple(schema.state_keys),
                action_keys=tuple(schema.action_keys),
                state_dims=tuple(schema.state_dims),
                action_dims=tuple(schema.action_dims),
            )
            logging.info(
                f"Hydrated {t.__class__.__name__} gripper_dim={t.gripper_dim} "
                f"({schema.schema_id})"
            )
            # snap_gripper_to_binary snaps raw gripper width to {0, max_width}
            # BEFORE NormalizeTransformFn. For the intended exact {-1,+1}
            # mapping, the gripper-dim stats MUST already be q01=0, q99=max_width.
            # This is NOT done automatically — it requires running
            # data_process/labutopia_canonicalize_stats.py on stats.json first.
            # Verify and fail loud so a forgotten canonicalize step doesn't
            # silently train a mis-scaled gripper. Bypass: LABVLA_ALLOW_UNPATCHED_SNAP_STATS=1.
            _grip = int(getattr(t, "gripper_dim", -1))
            _maxw = float(getattr(t, "max_width", 0.0))
            _ack = "action_abs" if action_mode == "abs" else "action"
            _tol = max(1e-4, abs(_maxw) * 0.05)
            _bad_snap = []
            for _can in ("observation.state", _ack):
                _b = stats.get(_can)
                if not isinstance(_b, dict):
                    continue
                _q01 = _b.get("q01")
                _q99 = _b.get("q99")
                # snap-to-binary REQUIRES q01/q99 (it maps {0,max_width}->{-1,+1}
                # via q01/q99 normalization). If they are absent/short,
                # NormalizeTransformFn falls back to mean_std (core.py
                # _resolve_with_fallback) → the snapped values are mis-scaled and
                # the guard must NOT silently pass. Treat absent/short/unparseable
                # q01/q99 as a failure too.
                if _q01 is None or _q99 is None:
                    _bad_snap.append((_can, "q01/q99 absent (snap needs quantile stats)"))
                    continue
                try:
                    if 0 <= _grip < len(_q01) and 0 <= _grip < len(_q99):
                        if abs(float(_q01[_grip])) > _tol or abs(float(_q99[_grip]) - _maxw) > _tol:
                            _bad_snap.append((_can, float(_q01[_grip]), float(_q99[_grip])))
                    else:
                        _bad_snap.append((_can, f"q01/q99 too short for gripper dim {_grip}"))
                except (TypeError, ValueError, IndexError):
                    _bad_snap.append((_can, "q01/q99 unparseable"))
            if _bad_snap:
                _snap_msg = (
                    f"[hydrate_all] snap_gripper_to_binary=True but gripper-dim ({_grip}) "
                    f"stats are NOT canonicalized to q01=0 / q99={_maxw} (found "
                    f"(canonical, q01, q99) = {_bad_snap}). The snapped {{0, max_width}} "
                    f"values would be normalized through the dataset's raw q01/q99 instead "
                    f"of to exactly {{-1, +1}}, mis-scaling the gripper at train AND deploy. "
                    f"Run data_process/labutopia_canonicalize_stats.py on this dataset's "
                    f"stats.json before training."
                )
                if os.environ.get("LABVLA_ALLOW_UNPATCHED_SNAP_STATS") == "1":
                    logging.error(_snap_msg + " [BYPASSED via LABVLA_ALLOW_UNPATCHED_SNAP_STATS=1]")
                else:
                    raise ValueError(
                        _snap_msg + " Set LABVLA_ALLOW_UNPATCHED_SNAP_STATS=1 to bypass intentionally."
                    )
        elif isinstance(t, RemapImageKeyTransformFn):
            t = replace(t, mapping=dict(schema.image_mapping))
            logging.info(
                f"Hydrated {t.__class__.__name__} with {len(schema.image_mapping)} "
                f"cameras ({schema.schema_id})"
            )
        elif isinstance(t, AnnotationTokenizeTransformFn):
            # Inject this dataset's annotation specs. Empty tuple → no-op path,
            # zero tokenizer load, zero overhead for OXE-style datasets.
            t = replace(t, annotation_specs=tuple(schema.annotation_losses))
            logging.info(
                f"Hydrated {t.__class__.__name__} with "
                f"{len(t.annotation_specs)} annotation_specs "
                f"dynamic_shape={getattr(t, 'dynamic_shape', False)} "
                f"({schema.schema_id})"
            )
        elif isinstance(t, FastActionEncodeTransformFn):
            source_action_dim = int(sum(schema.action_dims or ()))
            original_max_length = int(t.max_length)
            effective_max_length = original_max_length
            if getattr(t, "source_shape_convergence", False):
                effective_max_length = source_fast_max_length(
                    schema.schema_id,
                    original_max_length,
                )
            if effective_max_length != original_max_length:
                t = replace(t, max_length=effective_max_length)
            logging.info(
                f"Hydrated {t.__class__.__name__} max_length={t.max_length} "
                f"source_action_dim={source_action_dim} "
                f"trim_to_mask={getattr(t, 'trim_to_mask', False)} "
                f"source_shape_convergence={getattr(t, 'source_shape_convergence', False)} "
                f"({schema.schema_id})"
            )
        elif isinstance(t, BuildAgiBotSubtaskTransformFn):
            # Gate on an EXPLICIT schema identity, not just the presence of an
            # `annotation.subtask` loss field: gating on the field alone would
            # fire for ANY schema reusing that field name, clobbering its `task`
            # with the generic prompt and overwriting its `annotation.subtask`.
            # The task→subtask backfill + prompt rewrite is AgiBot-specific (one
            # task description per task dir), so require the AgiBot schema
            # identity. robot_type / schema_id carry the family tag
            # ("agibot_dual_arm" / "agibot_dual_arm_v1").
            _robot_type = str(getattr(schema, "robot_type", "") or "")
            _schema_id = str(getattr(schema, "schema_id", "") or "")
            _is_agibot_schema = (
                _robot_type.startswith("agibot") or _schema_id.startswith("agibot")
            )
            _agibot_enabled = bool(_has_subtask_loss and _is_agibot_schema)
            t = replace(t, enabled=_agibot_enabled)
            if _has_subtask_loss and not _is_agibot_schema:
                logging.info(
                    f"{t.__class__.__name__} left disabled: schema "
                    f"{_schema_id!r} declares annotation.subtask but is not an "
                    f"AgiBot schema — skipping task→subtask backfill/prompt rewrite."
                )
            if _agibot_enabled:
                logging.info(
                    f"Hydrated {t.__class__.__name__} enabled=True ({schema.schema_id})"
                )
        elif isinstance(t, DiscretizeStateTransformFn):
            # Inject schema.state_keys so the transform can transiently
            # concat per-key state tensors when OBS_STATE has not yet been
            # built by ComposeFieldsTransform (split-state schemas like
            # robointer_droid). For single-key schemas where state_keys
            # already equals (OBS_STATE,) this is a no-op fall-through.
            t = replace(t, state_keys=tuple(schema.state_keys))
            logging.info(
                f"Hydrated {t.__class__.__name__} with state_keys="
                f"{tuple(schema.state_keys)} ({schema.schema_id})"
            )
        hydrated.append(t)
    return hydrated
