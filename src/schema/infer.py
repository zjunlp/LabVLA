"""Tier 2: auto-infer a schema from info.json `features[*].names`."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from .arm_layout import ArmCount, ArmLayoutSpec
from .camera_mapping import infer_image_mapping_from_info
from .dataset_schema import DatasetSchema


logger = logging.getLogger(__name__)

# Case-insensitive substring tokens that mark a dimension as "gripper-like"
# (absolute target instead of delta). "finger" alone is too permissive — it
# matches "fingertip_force_sensor" and "finger_joint_position" which are
# arm/sensor dims, not grippers. We only accept "finger" if it's paired with
# a closure/position indicator.
GRIPPER_TOKENS = ("gripper", "effector")
GRIPPER_FINGER_CONTEXTS = ("finger_position", "finger_pose", "finger_angle",
                           "finger_open", "finger_close", "finger_width")

# Substrings that *suggest* a dim is gripper/finger-related even if it did not
# satisfy the strict ``_is_gripper_name`` whitelist above. Used by the
# gripper-inference fallback to tell apart two failure modes when no strict
# match is found:
#   (a) genuinely grippered-but-mislabeled data ("finger" without a recognized
#       closure/position context, or an exotic "grip"/"claw"/"hand" spelling)
#       → fail-loud, because silently treating an absolute gripper dim as delta
#       corrupts the delta transform / stats / supervision.
#   (b) a robot that genuinely has no gripper channel (e.g. dex-hand robots
#       whose dims are individual finger joints) → all-delta is correct.
# A name containing any of these tokens is treated as "looks grippered" (case a).
GRIPPER_HINT_TOKENS = ("gripper", "effector", "finger", "grip", "claw", "hand",
                       "jaw", "pinch")


def _match_names_to_list(names: Any) -> list[str]:
    """Normalize the various shapes `features[k].names` can take.

    LeRobot v2.1 / v3.0 `names` may be:
      - None (unlabeled)
      - list[str]        — per-dim labels
      - dict[str, list]  — e.g. {"motors": ["joint_0", ..., "gripper"]}
    """
    if names is None:
        return []
    if isinstance(names, list):
        return [str(n) for n in names]
    if isinstance(names, dict):
        # pick the first inner list (LeRobot convention: {"motors": [...]})
        for v in names.values():
            if isinstance(v, list):
                return [str(x) for x in v]
    return []


def _is_gripper_name(name: str) -> bool:
    """Return True if the dim name unambiguously denotes a gripper channel.

    Rejects "fingertip_force_sensor" / "finger_joint_position" — those have
    "finger" but no gripper-action context.
    """
    low = name.lower()
    if any(tok in low for tok in GRIPPER_TOKENS):
        return True
    if "finger" in low and any(ctx in low for ctx in GRIPPER_FINGER_CONTEXTS):
        return True
    return False


def _infer_arm_layout(
    gripper_action_dims: tuple[int, ...],
) -> Optional[ArmLayoutSpec]:
    """Best-effort ArmLayoutSpec inference from gripper indices.

    Heuristic (compatible with the canonical single/dual-arm layouts):

      - One gripper dim at index ``g ∈ {6, 7}`` → SINGLE arm, ``arm_dof = g``,
        gripper at raw index ``g``. Matches 7-DoF Franka (grip @ 7) and
        6-DoF UR5/WidowX (grip @ 6) with the zero-pad at dim 6.

      - Two gripper dims ``(g0, g1)`` where ``left = g0 ∈ {5, 6}`` AND
        ``right = g1 - g0 - 1 ∈ {5, 6}`` → DUAL arm, layout ``[arm_l, grip_l,
        arm_r, grip_r]`` with ``left_gripper_index_in_raw = g0``,
        ``right_gripper_index_in_raw = g1``.

      - Anything else (no gripper, or exotic placement) → ``None``. The
        caller records arm_layout=None, which disables canonical remapping
        for this dataset (the gripper_action_dims metadata is still used).

    Returns:
        An ``ArmLayoutSpec`` if the heuristic succeeds, else ``None``.
    """
    if len(gripper_action_dims) == 1:
        g = gripper_action_dims[0]
        if g in (6, 7):
            return ArmLayoutSpec(
                arm_count=ArmCount.SINGLE,
                arm_dof=g,
                gripper_index_in_raw=g,
            )
        return None
    if len(gripper_action_dims) == 2:
        g0, g1 = gripper_action_dims
        left_arm_dof = g0
        right_arm_dof = g1 - g0 - 1
        if left_arm_dof in (5, 6) and right_arm_dof in (5, 6):
            return ArmLayoutSpec(
                arm_count=ArmCount.DUAL,
                left_arm_dof=left_arm_dof,
                right_arm_dof=right_arm_dof,
                left_gripper_index_in_raw=g0,
                right_gripper_index_in_raw=g1,
            )
        return None
    return None


def try_infer_from_info(
    info: dict,
    robot_type: Optional[str],
    root: Path,
) -> Optional[DatasetSchema]:
    """Tier 2 discovery. Returns None if names are too opaque — caller falls back."""
    features = info.get("features") or {}
    action_feat = features.get("action") or features.get("actions")
    state_feat = features.get("state") or features.get("observation.state")

    if not (action_feat and state_feat):
        logger.debug(
            "try_infer_from_info: missing 'action' / 'state' feature for %s", root
        )
        return None

    a_names = _match_names_to_list(action_feat.get("names"))
    s_names = _match_names_to_list(state_feat.get("names"))

    # Tier B/C detection: opaque single-label vectors like names=["action"].
    if len(a_names) <= 1 or len(s_names) <= 1:
        logger.debug(
            "try_infer_from_info: names too opaque for %s (|a|=%d |s|=%d)",
            root, len(a_names), len(s_names),
        )
        return None

    # Derive delta mask from per-dim names.
    delta_mask = tuple(not _is_gripper_name(n) for n in a_names)
    gripper_action_dims = tuple(i for i, d in enumerate(delta_mask) if not d)
    if not gripper_action_dims:
        # No dim satisfied the strict gripper whitelist. Two sub-cases:
        #   (a) The robot genuinely has no gripper (e.g. dex-hand robots like
        #       AIRBOT_MMK2 where all 36 dims are individual finger joints).
        #       In that case all-delta is the semantically correct schema.
        #   (b) The data DOES have a gripper dim, but it was named weirdly
        #       enough that the strict matcher missed it (e.g. "finger" with no
        #       recognized closure context, or an exotic "grip"/"claw"/"jaw"
        #       spelling).
        #
        # Do NOT blanket fall back to all-delta: that silently mislabels an
        # absolute gripper dim as delta in case (b) and corrupts the delta
        # transform / stats / supervision. Action names are detailed here
        # (|a_names| > 1 established above), so only continue for the
        # unambiguous case (a) — no action name even hints at a gripper. If any
        # name looks grippered but failed the strict whitelist, fail loud and
        # demand an explicit manifest.
        looks_grippered = any(
            any(tok in n.lower() for tok in GRIPPER_HINT_TOKENS)
            for n in a_names
        )
        if looks_grippered:
            from .dataset_schema import SchemaDiscoveryError
            raise SchemaDiscoveryError(
                f"[schema] auto-infer for {root}: action names look like they "
                f"contain a gripper/finger channel but none matched the strict "
                f"gripper-name whitelist, so the gripper dim cannot be "
                f"identified. Treating it as delta would corrupt the delta "
                f"transform, stats and supervision. Write a "
                f"meta/labvla_manifest.json declaring gripper_action_dims "
                f"explicitly (or rename the dim to a recognized form). "
                f"Action names: {list(a_names)}"
            )
        # Case (a): genuinely no gripper. all-delta is correct.
        logger.warning(
            "try_infer_from_info: no gripper channel detected in action names "
            "for %s (no name hints at a gripper/finger) — using an all-delta "
            "schema (every dim treated as delta). If this dataset actually has "
            "an absolute gripper/finger-position target, write a "
            "meta/labvla_manifest.json to override. First 5 action names: %s",
            root, list(a_names)[:5],
        )
        delta_mask = tuple([True] * len(a_names))
        gripper_action_dims = tuple()

    # Determine actual feature keys to use — prefer the exact key we read
    # (may be 'action' or 'actions' depending on dataset).
    action_key = "actions" if "actions" in features else "action"
    state_key = "observation.state" if "observation.state" in features else "state"

    action_shape = action_feat.get("shape") or [len(a_names)]
    state_shape = state_feat.get("shape") or [len(s_names)]

    schema_id = f"{robot_type or 'unknown'}:{root.name}"

    # Best-effort canonical arm layout — lets Tier-2 datasets reuse the
    # deploy-time forward/reverse mapper without needing a manifest. Falls
    # back to None when the gripper placement is ambiguous (caller then
    # trains/deploys using gripper_action_dims metadata directly).
    arm_layout = _infer_arm_layout(gripper_action_dims)

    return DatasetSchema(
        schema_id=schema_id,
        robot_type=robot_type or "unknown",
        state_keys=(state_key,),
        action_keys=(action_key,),
        state_dims=(int(state_shape[0]),),
        action_dims=(int(action_shape[0]),),
        delta_mask=delta_mask,
        gripper_action_dims=gripper_action_dims,
        image_mapping=infer_image_mapping_from_info(features),
        source="info_names",
        source_path=str(root / "meta" / "info.json"),
        arm_layout=arm_layout,
    )
