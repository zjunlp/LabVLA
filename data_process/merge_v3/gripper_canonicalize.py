"""Per-subset gripper value canonicalization for oxe-auge → oxe-auge_clean_v2 merge.

All 12 included subsets are mapped onto dim 7 ∈ [0, 1] with the OPEN direction
fixed at 1 (1 = more open, 0 = more closed):
  - 7 binary subsets: clean {0, 1} — endpoints are exactly {0, 1}.
  - 3 constant-1.0 subsets: constant 0.0 (no gripper variation in those tasks).
  - 2 continuous subsets (bridge, fractal20220817_data): gradation in [0, 1].

Endpoint-alignment caveat: direction is unified for every subset, but the CLOSED
endpoint is NOT rebased to exactly 0.0 for the continuous `bridge` subset.
`bridge` uses `_clip_keep` (clip-only), which preserves widowX's native
continuous width whose closed end sits near ~0.06, not 0.0. Treat the convention
as "1 = open direction, range ⊆ [0, 1]" rather than "0 = hard-closed endpoint"
for `bridge`. The binary and constant subsets DO land on exact {0, 1}.

4 ambiguous subsets are excluded from the merge (jaco_play, taco_play,
ucsd_kitchen_dataset, iamlab_cmu_pickup_insert).

Evidence chain:
  - AugE-Toolkit/core/processing_utils.py: per-source `gripper_states` extraction
    (most are `is_gripper_closed` bool — 1=close in oxe-auge convention).
  - AugE-Toolkit/core/gripper_utils.py: GRIPPER_RANGES per robot.
  - Empirical trajectory pattern audit (30 episodes/subset): direction confirmed
    by first→mid→last gripper signal (rises = 1=close → invert; falls = 1=open).

See doc/oxe_subset_gripper_audit.md for the full audit table.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# fractal20220817_data continuous scaling divides by this sampled-max bound. Raw
# values above it saturate at the canonical endpoint; we count and warn so
# out-of-bound tail / future-increment data is surfaced, not silenced.
_FRACTAL_RAW_MAX = 0.80


# === Per-rule canonicalize functions ===

def _invert_clip(x: np.ndarray) -> np.ndarray:
    """Bool source: oxe-auge convention 1=close → canonical 1=open.

    Used by:
      - 7 binary subsets: austin_buds_dataset, austin_sailor_dataset,
        berkeley_autolab_ur5, nyu_franka_play_dataset, utaustin_mutex,
        utokyo_xarm_pick_and_place, viola.
      - 3 constant-1.0 subsets: kaist_nonprehensile, language_table, toto.

    Empirical evidence (Δmid-first > +0.20 in trajectory audit):
        austin_buds          0.00 → 0.54 → 0.87  Δ +0.54
        austin_sailor        0.26 → 0.46 → 0.61  Δ +0.20
        berkeley_autolab_ur5 0.00 → 0.53 → 0.00  Δ +0.53 (pick-and-release)
        nyu_franka_play      0.00 → 0.25 → 0.43  Δ +0.25
        utaustin_mutex       0.00 → 0.57 → 0.84  Δ +0.57
        utokyo_xarm_pap      0.00 → 0.84 → 0.00  Δ +0.84 (pick-and-release)
        viola                0.00 → 0.30 → 0.51  Δ +0.30
    """
    return (1.0 - np.clip(x, 0.0, 1.0)).astype(np.float32)


def _scale_invert_fractal(x: np.ndarray) -> np.ndarray:
    """fractal20220817_data (google_robot, continuous):
    raw ∈ [0, 0.80] → divide by 0.80 → invert → continuous [0, 1] with 1=open.

    Direction confirmed empirically (Δmid-first = +0.40 in audit).
    Scale factor 0.80 = max raw value observed across all sampled episodes
    (matches google_robot's typical actuator range in oxe-auge).

    Note `0.80` is only a SAMPLED max, not a physical actuator limit: raw values
    > 0.80 saturate at the canonical closed endpoint after the clip, truncating
    continuous gripper semantics. We count and warn on such out-of-bound values
    to surface unsampled tails; saturation behavior is unchanged for raw ≤ 0.80.
    """
    arr = np.asarray(x)
    over = arr > _FRACTAL_RAW_MAX
    n_over = int(np.count_nonzero(over))
    if n_over:
        logger.warning(
            "[gripper-canonicalize] fractal20220817_data: %d raw gripper "
            "value(s) > %.2f (sampled-max bound); max raw=%.4f. These saturate "
            "at the canonical endpoint — the 0.80 scale factor may undercover "
            "the true actuator range.",
            n_over,
            _FRACTAL_RAW_MAX,
            float(arr.max()) if arr.size else float("nan"),
        )
    return (1.0 - np.clip(arr / _FRACTAL_RAW_MAX, 0.0, 1.0)).astype(np.float32)


def _clip_keep(x: np.ndarray) -> np.ndarray:
    """bridge (widowX state[-1] continuous):
    raw ∈ [0.06, 1.01] already 1=open convention → safe clip to [0, 1].

    Direction confirmed empirically (Δmid-first = -0.29 in audit, gripper
    drops in mid-trajectory, confirming 1=open). Clip removes 1.01 overshoot.
    """
    return np.clip(x, 0.0, 1.0).astype(np.float32)


# === Per-subset rule registry ===

# Map base subset name → canonicalize function.
SUBSET_GRIPPER_RULES = {
    # 7 binary INVERT (bool source: 1=close → output {0, 1}, 1=open)
    "austin_buds_dataset":          _invert_clip,
    "austin_sailor_dataset":        _invert_clip,
    "berkeley_autolab_ur5":         _invert_clip,
    "nyu_franka_play_dataset":      _invert_clip,
    "utaustin_mutex":               _invert_clip,
    "utokyo_xarm_pick_and_place":   _invert_clip,
    "viola":                        _invert_clip,
    # 3 constant INVERT (constant 1.0 → constant 0.0, "task has no gripper variation")
    "kaist_nonprehensile":          _invert_clip,
    "language_table":               _invert_clip,
    "toto":                         _invert_clip,
    # 1 continuous SCALE+INVERT (google_robot raw [0, 0.80])
    "fractal20220817_data":         _scale_invert_fractal,
    # 1 continuous CLIP (widowX state[-1] [0.06, 1.01])
    "bridge":                       _clip_keep,
}

INCLUDED_SUBSETS = frozenset(SUBSET_GRIPPER_RULES.keys())

# 4 subsets explicitly excluded by the audit (ambiguous direction or anomalous range).
EXCLUDED_SUBSETS = frozenset({
    "iamlab_cmu_pickup_insert",  # Δmid-first = +0.05, no clear pick pattern
    "taco_play",                 # Δmid-first = +0.06, conflicting signal
    "ucsd_kitchen_dataset",      # drop pattern (drawer/cabinet, non-pick task)
    "jaco_play",                 # raw range [0, 1.34] anomalous, scale unreliable
})


# === Subset name parsing ===

def get_base_subset_name(dir_name: str) -> str | None:
    """Extract base subset name from oxe-auge directory name.

    oxe-auge subdir naming convention: <base>{_train_X_Y|_test_X_Y|_val_X_Y}?_augmented.

    Examples:
      "austin_buds_dataset_augmented"             → "austin_buds_dataset"
      "bridge_train_0_5000_augmented"             → "bridge"
      "fractal20220817_data_train_0_2500_augmented" → "fractal20220817_data"
      "berkeley_autolab_ur5_test_augmented"       → "berkeley_autolab_ur5"
      "jaco_play_train_0_976_augmented"           → "jaco_play"

    Returns:
      Base name if it matches an INCLUDED or EXCLUDED subset; None otherwise.
    """
    name = dir_name
    # Strip _train_X_Y / _test_X_Y / _val_X_Y suffixes (numbers may be _<digits>_<digits>)
    for split_marker in ("_train_", "_test_", "_val_"):
        if split_marker in name:
            name = name.split(split_marker)[0]
            break
    # Strip _augmented (and _train / _test / _val without numbers)
    for tail in ("_augmented", "_train", "_test", "_val"):
        if name.endswith(tail):
            name = name[: -len(tail)]
            break
    if name in INCLUDED_SUBSETS or name in EXCLUDED_SUBSETS:
        return name
    return None


def is_included_subset(dir_name: str) -> bool:
    """Whether subset directory should be merged into oxe-auge_clean_v2."""
    base = get_base_subset_name(dir_name)
    return base is not None and base in INCLUDED_SUBSETS


# === Public canonicalize entry ===

def canonicalize_gripper(values: np.ndarray, dir_name: str) -> np.ndarray:
    """Apply per-subset gripper canonicalize.

    Args:
        values:   1-D array of raw gripper values (dim 7 of observation.joints).
        dir_name: oxe-auge subset directory name (used to look up rule).

    Returns:
        1-D array of canonicalized values in [0, 1] with 1=open, 0=close.
        Same shape and dtype-friendly (float32) as input.

    Raises:
        ValueError: if dir_name maps to an EXCLUDED subset (caller bug — those
                    should be filtered before reaching this function).
        KeyError:   if dir_name doesn't map to any known subset.
    """
    base = get_base_subset_name(dir_name)
    if base is None:
        raise KeyError(
            f"Unknown subset directory {dir_name!r} — not in INCLUDED_SUBSETS "
            f"({sorted(INCLUDED_SUBSETS)}) or EXCLUDED_SUBSETS "
            f"({sorted(EXCLUDED_SUBSETS)})."
        )
    if base in EXCLUDED_SUBSETS:
        raise ValueError(
            f"Subset {base!r} (from dir {dir_name!r}) is EXCLUDED by audit; "
            f"caller must filter these out at scan/whitelist stage BEFORE "
            f"reaching canonicalize_gripper."
        )
    rule = SUBSET_GRIPPER_RULES[base]
    return rule(np.asarray(values))


__all__ = [
    "SUBSET_GRIPPER_RULES",
    "INCLUDED_SUBSETS",
    "EXCLUDED_SUBSETS",
    "canonicalize_gripper",
    "get_base_subset_name",
    "is_included_subset",
]
