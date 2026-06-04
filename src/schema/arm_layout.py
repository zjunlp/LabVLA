"""Canonical arm-count layouts for LabVLA action/state vectors.

Design principle (openpi-aligned, simpler than InternVLA-A1's per-robot mapping):

- **Single-arm** data → **8-dim canonical**: arm joints at dim 0..6 (pad 0 at
  dim 6 for 6-DoF arms like UR5/WidowX/Jaco), gripper at dim 7.
- **Dual-arm** data → **14-dim canonical**: left arm at dim 0..5 (6 joints,
  pad if 5-DoF), left gripper at dim 6, right arm at dim 7..12, right
  gripper at dim 13. Matches openpi Aloha's `state[[6, 13]]` gripper
  indexing (aloha_policy.py:161,186).

Both layouts are then right-padded to `max_action_dim=32` by
`PadStateAndActionTransformFn` downstream.

Pretrain, fine-tune, and deployment MUST all use the same
`ArmLayoutSpec` for a given robot, so the model sees a consistent
gripper position throughout its life. This module is the single source
of truth.

References:
  - openpi DROID (Franka 7-DoF single-arm, gripper dim 7):
    /all-flash-data/openpi/src/openpi/policies/droid_policy.py:40
  - openpi Aloha (dual-arm, grippers at [6, 13]):
    /all-flash-data/openpi/src/openpi/policies/aloha_policy.py:161,186
  - ABot gripper binarize at deploy:
    /all-flash-data/ABot-Manipulation/ABot/model/framework/base_framework.py:189
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


SINGLE_ARM_CANONICAL_DIM = 8
SINGLE_ARM_GRIPPER_INDEX = 7

DUAL_ARM_CANONICAL_DIM = 14
DUAL_ARM_GRIPPER_INDICES = (6, 13)


class ArmCount(str, Enum):
    SINGLE = "single"
    DUAL = "dual"


GRIPPER_BINARIZE_THRESHOLD_DEFAULT = 0.5


@dataclass(frozen=True)
class ArmLayoutSpec:
    """Describes how a raw robot's joint+gripper data maps to canonical form.

    Used by:
      - Data merge / preprocessing: forward_map_* writes canonical shards
      - Adapter: can apply forward_map_* on-the-fly if data isn't pre-merged
      - Deployment: reverse_map_* converts canonical output back to raw
      - Schema discovery: stored in `DatasetSchema.arm_layout`

    Example (single-arm, Franka 7-DoF):
        ArmLayoutSpec(arm_count=ArmCount.SINGLE, arm_dof=7, gripper_index_in_raw=7)
    Example (single-arm, UR5 6-DoF, gripper at raw index 6):
        ArmLayoutSpec(arm_count=ArmCount.SINGLE, arm_dof=6, gripper_index_in_raw=6)
    Example (dual-arm Aloha, both 6-DoF):
        ArmLayoutSpec(arm_count=ArmCount.DUAL,
                      left_arm_dof=6, right_arm_dof=6,
                      left_gripper_index_in_raw=6,
                      right_gripper_index_in_raw=13)
    """
    arm_count: ArmCount

    # Single-arm fields.
    arm_dof: Optional[int] = None
    gripper_index_in_raw: Optional[int] = None

    # Dual-arm fields (future: robocoin_clean / Aloha / etc.).
    left_arm_dof: Optional[int] = None
    right_arm_dof: Optional[int] = None
    left_gripper_index_in_raw: Optional[int] = None
    right_gripper_index_in_raw: Optional[int] = None

    # Per-robot gripper-binarization threshold (on q01/q99-normalized
    # [-1, 1] action space). Default 0.5 matches starVLA/openpi convention — a
    # conservative "open only when confidently positive" policy that
    # corresponds to the 75% quantile of the raw gripper distribution. Robots
    # with asymmetric gripper stats (e.g. a distribution with most mass in the
    # open state) may override; always read via the schema at deploy time,
    # never hard-code in the inference code path.
    gripper_binarize_threshold: float = GRIPPER_BINARIZE_THRESHOLD_DEFAULT

    def __post_init__(self) -> None:
        # Structural validation lives in schema/validate.py (SSOT).
        from .validate import validate_arm_layout
        validate_arm_layout(self)

    @property
    def canonical_dim(self) -> int:
        return (SINGLE_ARM_CANONICAL_DIM if self.arm_count == ArmCount.SINGLE
                else DUAL_ARM_CANONICAL_DIM)

    @property
    def gripper_indices_canonical(self) -> tuple[int, ...]:
        return ((SINGLE_ARM_GRIPPER_INDEX,) if self.arm_count == ArmCount.SINGLE
                else DUAL_ARM_GRIPPER_INDICES)

    def to_dict(self) -> dict:
        """Serialize to JSON-friendly dict for labvla_manifest.json.

        HIGH-21: ``gripper_binarize_threshold`` is always emitted (even when
        equal to the default) so the on-disk record is self-describing —
        consumers downstream (e.g. deploy) should read the threshold from
        the schema rather than assume a hard-coded 0.5.
        """
        base: dict = {"arm_count": self.arm_count.value}
        if self.arm_count == ArmCount.SINGLE:
            base["arm_dof"] = self.arm_dof
            base["gripper_index_in_raw"] = self.gripper_index_in_raw
        else:
            base["left_arm_dof"] = self.left_arm_dof
            base["right_arm_dof"] = self.right_arm_dof
            base["left_gripper_index_in_raw"] = self.left_gripper_index_in_raw
            base["right_gripper_index_in_raw"] = self.right_gripper_index_in_raw
        base["gripper_binarize_threshold"] = float(self.gripper_binarize_threshold)
        return base

    @classmethod
    def from_dict(cls, d: dict) -> "ArmLayoutSpec":
        arm_count = ArmCount(d["arm_count"])
        # Backward compatible — older manifests without the threshold
        # key fall back to the module default. New manifests always carry it.
        threshold = float(d.get(
            "gripper_binarize_threshold", GRIPPER_BINARIZE_THRESHOLD_DEFAULT
        ))
        if arm_count == ArmCount.SINGLE:
            return cls(
                arm_count=arm_count,
                arm_dof=d["arm_dof"],
                gripper_index_in_raw=d["gripper_index_in_raw"],
                gripper_binarize_threshold=threshold,
            )
        return cls(
            arm_count=arm_count,
            left_arm_dof=d["left_arm_dof"],
            right_arm_dof=d["right_arm_dof"],
            left_gripper_index_in_raw=d["left_gripper_index_in_raw"],
            right_gripper_index_in_raw=d["right_gripper_index_in_raw"],
            gripper_binarize_threshold=threshold,
        )


# ====================== Single-arm forward / reverse ======================

def forward_map_single(
    raw: np.ndarray,
    arm_dof: int,
    gripper_index_in_raw: int,
) -> np.ndarray:
    """Raw single-arm [arm + gripper, possibly 7 or 8 dim] → 8-dim canonical.

    The canonical layout:
        dim 0..6 = arm joints (pad 0 at dim 6 if arm_dof=6)
        dim 7   = gripper

    Works on a 1-D vector of shape (raw_dim,) or a batched (..., raw_dim).
    Returns (..., 8) with the last axis permuted.

    Examples:
        UR5 raw = [j0, j1, j2, j3, j4, j5, grip]  (7 dim, dof=6)
          → [j0, j1, j2, j3, j4, j5, 0.0, grip]   (8 dim, gripper at dim 7)

        Franka raw = [j0, j1, j2, j3, j4, j5, j6, grip]  (8 dim, dof=7)
          → unchanged (8 dim, gripper at dim 7)
    """
    raw = np.asarray(raw)
    raw_last_dim = raw.shape[-1]
    out_shape = raw.shape[:-1] + (SINGLE_ARM_CANONICAL_DIM,)
    out = np.zeros(out_shape, dtype=raw.dtype)

    if arm_dof == 7:
        # Expected raw layout: [j0..j6, gripper] — gripper_index_in_raw == 7
        if raw_last_dim != 8 or gripper_index_in_raw != 7:
            raise ValueError(
                f"forward_map_single(7-DoF): expected raw 8-dim with gripper at idx 7, "
                f"got raw_dim={raw_last_dim}, gripper_idx={gripper_index_in_raw}"
            )
        # Already canonical.
        out[...] = raw
    elif arm_dof == 6:
        # Expected raw layout: [j0..j5, gripper] — gripper_index_in_raw == 6
        if raw_last_dim != 7 or gripper_index_in_raw != 6:
            raise ValueError(
                f"forward_map_single(6-DoF): expected raw 7-dim with gripper at idx 6, "
                f"got raw_dim={raw_last_dim}, gripper_idx={gripper_index_in_raw}"
            )
        # arm joints at dim 0..5, pad 0 at dim 6, gripper at dim 7.
        out[..., :6] = raw[..., :6]
        out[..., 6] = 0.0
        out[..., 7] = raw[..., 6]
    else:
        raise ValueError(f"arm_dof must be 6 or 7, got {arm_dof}")

    return out


def reverse_map_single(
    canonical: np.ndarray,
    arm_dof: int,
    gripper_index_in_raw: int,
) -> np.ndarray:
    """8-dim canonical → raw robot format (for deployment).

    Inverse of `forward_map_single`. Drops the 0-pad at dim 6 for 6-DoF arms.
    """
    canonical = np.asarray(canonical)
    if canonical.shape[-1] != SINGLE_ARM_CANONICAL_DIM:
        raise ValueError(
            f"reverse_map_single expects canonical dim {SINGLE_ARM_CANONICAL_DIM}, "
            f"got {canonical.shape[-1]}"
        )
    if arm_dof == 7:
        # No-op: canonical already matches raw.
        return canonical.copy()
    if arm_dof == 6:
        out_shape = canonical.shape[:-1] + (7,)
        out = np.zeros(out_shape, dtype=canonical.dtype)
        out[..., :6] = canonical[..., :6]
        out[..., 6] = canonical[..., 7]  # gripper
        return out
    raise ValueError(f"arm_dof must be 6 or 7, got {arm_dof}")


# ====================== Dual-arm forward / reverse ======================
# Used by dual-arm schema canonicalization (AgiBot/Aloha-style layouts).

def forward_map_dual(
    raw_left_arm: np.ndarray,
    raw_left_gripper: np.ndarray,
    raw_right_arm: np.ndarray,
    raw_right_gripper: np.ndarray,
    left_arm_dof: int,
    right_arm_dof: int,
) -> np.ndarray:
    """Assemble 14-dim canonical from per-component raw arrays.

    Canonical layout:
        dim 0..5  = left arm (pad 0 at dim 5 if left_arm_dof=5)
        dim    6  = left gripper
        dim 7..12 = right arm (pad 0 at dim 12 if right_arm_dof=5)
        dim   13  = right gripper

    Note: the 14-dim canonical reserves six joint slots per side. Aloha
    6-DoF × 2 fits directly; AgiBot-style 7-DoF × 2 inputs are accepted by
    explicitly keeping joints 0..5 on each side and dropping joint 6.
    """
    if left_arm_dof not in (5, 6, 7) or right_arm_dof not in (5, 6, 7):
        raise ValueError(
            f"dual-arm canonical 14-dim requires 5, 6, or 7 DoF per arm, "
            f"got left={left_arm_dof} right={right_arm_dof}"
        )
    left_arm = np.asarray(raw_left_arm)
    right_arm = np.asarray(raw_right_arm)
    left_g = np.asarray(raw_left_gripper)
    right_g = np.asarray(raw_right_gripper)

    out_shape = left_arm.shape[:-1] + (DUAL_ARM_CANONICAL_DIM,)
    out = np.zeros(out_shape, dtype=left_arm.dtype)

    left_keep = min(left_arm_dof, 6)
    right_keep = min(right_arm_dof, 6)
    out[..., :left_keep] = left_arm[..., :left_keep]
    out[..., 6] = left_g[..., 0] if left_g.ndim > 0 else left_g
    out[..., 7:7 + right_keep] = right_arm[..., :right_keep]
    out[..., 13] = right_g[..., 0] if right_g.ndim > 0 else right_g
    return out


def reverse_map_dual(
    canonical: np.ndarray,
    left_arm_dof: int,
    right_arm_dof: int,
) -> dict[str, np.ndarray]:
    """Split 14-dim canonical back into per-component raw arrays.

    This reverse map is exact only for dual-arm layouts with at most six joint
    slots per side. AgiBot-style 7-DoF inputs are lossy in the forward map
    because joint 6 from each side is deliberately dropped to fit the canonical
    14-dim layout; pretending to reconstruct those missing joints would mix the
    gripper slots into arm joints.
    """
    if canonical.shape[-1] != DUAL_ARM_CANONICAL_DIM:
        raise ValueError(
            f"reverse_map_dual expects canonical dim {DUAL_ARM_CANONICAL_DIM}, "
            f"got {canonical.shape[-1]}"
        )
    if left_arm_dof > 6 or right_arm_dof > 6:
        raise ValueError(
            "reverse_map_dual cannot reconstruct lossy 7-DoF dual-arm inputs; "
            "the canonical 14-dim layout keeps only six joints per side and "
            "stores grippers at dims 6 and 13."
        )
    return {
        "left_arm": canonical[..., :left_arm_dof].copy(),
        "left_gripper": canonical[..., 6:7].copy(),
        "right_arm": canonical[..., 7:7 + right_arm_dof].copy(),
        "right_gripper": canonical[..., 13:14].copy(),
    }
