"""Franka presets.

`franka8`       — single state/action tensor of 8 dims (7 joints + 1 gripper),
                  gripper absolute, arm delta. Covers LabUtopia franka tasks.

`franka_split`  — split state/action keys (joints + gripper in separate
                  feature columns). Covers robointer_droid / droid v2.1-style
                  datasets.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Iterable, Sequence

from ..camera_mapping import expand_camera_target
from ..dataset_schema import DatasetSchema


def _caller_source_path() -> str:
    """Return the absolute path of the Python file that called into the preset,
    so `schema.source_path` is a debuggable file rather than `"<preset>"`.
    """
    stack = inspect.stack()
    for frame in stack[1:]:
        fn = frame.filename
        # Skip our own module + stdlib wrappers; return the first outside caller.
        if "schema/presets/" not in fn and "schema\\presets\\" not in fn:
            return fn
    return "<unknown>"


def _expand_images(
    cams: Sequence[str],
    image_prefix: str = "observation.images",
) -> dict[str, str]:
    """Map ordered raw camera keys -> canonical `observation.images.imageN` targets.

    The LHS (source key) controls how the camera is read from disk:
      - With ``image_prefix="observation.images"`` (default, v3 convention) the
        source key is prefixed if not already qualified.
      - With ``image_prefix=""`` (v2.1 convention) the source key is kept raw
        (e.g. ``camera_1_rgb``), because that is literally the parquet /
        info.json feature name.

    C10/C19 fix: the RHS (target slot) is ALWAYS the fully qualified canonical
    ``observation.images.imageN`` form, regardless of ``image_prefix``.
    Previously ``image_prefix=""`` emitted a bare ``image0`` target, which
    ``schema/validate.py`` rejects (it requires ``observation.images.imageN``),
    so the resulting schema failed in ``DatasetSchema.__post_init__``. Routing
    the target through ``expand_camera_target`` keeps the source-side v2.1
    behavior while producing a validator-accepted target.
    """
    out: dict[str, str] = {}
    for i, cam in enumerate(cams):
        if image_prefix == "":
            raw = cam
        else:
            raw = cam if cam.startswith(image_prefix + ".") else f"{image_prefix}.{cam}"
        out[raw] = expand_camera_target(raw, f"image{i}")
    return out


def franka8(
    schema_id: str,
    *,
    cams: Sequence[str],
    robot_type: str = "franka",
    state_key: str = "observation.state",
    action_key: str = "action",
    gripper_dim: int = 7,
    image_prefix: str = "observation.images",
) -> DatasetSchema:
    """8-dim franka preset: 7 arm-joint delta + 1 gripper absolute.

    Args:
        schema_id:   Stable id (appears in checkpoints/log).
        cams:        Ordered list of raw camera keys; mapped to image0/1/2...
                     Typical: `("camera_1_rgb", "camera_2_rgb", "camera_3_rgb")`.
        robot_type:  Attribution string (default "franka").
        state_key:   info.json feature key for state (default "observation.state").
        action_key:  info.json feature key for action (default "action").
        gripper_dim: Index (0..7) of gripper in the 8-dim action vector; the
                     rest are delta. Default 7 matches LabUtopia/panda layout.
        image_prefix: Prefix added to each `cams` entry to form the info.json
                     feature key. Default `"observation.images"` matches v3
                     convention; pass `""` for v2.1 datasets whose camera
                     features are named directly (e.g. `camera_1_rgb`).

    Returns:
        Frozen DatasetSchema with `source="manifest"` and `source_path` pointing
        to the caller's .py file.
    """
    if not (0 <= gripper_dim < 8):
        raise ValueError(f"gripper_dim must be in [0, 8), got {gripper_dim}")
    delta_mask = tuple(i != gripper_dim for i in range(8))
    return DatasetSchema(
        schema_id=schema_id,
        robot_type=robot_type,
        state_keys=(state_key,),
        action_keys=(action_key,),
        state_dims=(8,),
        action_dims=(8,),
        delta_mask=delta_mask,
        gripper_action_dims=(gripper_dim,),
        image_mapping=_expand_images(cams, image_prefix=image_prefix),
        source="manifest",
        source_path=_caller_source_path(),
    )


def franka_split(
    schema_id: str,
    *,
    state_keys: Sequence[str],
    state_dims: Sequence[int],
    action_keys: Sequence[str],
    action_dims: Sequence[int],
    images: dict[str, str] | Iterable[str],
    gripper_action_dim: int,
    robot_type: str = "franka_robotiq",
) -> DatasetSchema:
    """Split-key franka preset: arm joints and gripper in separate feature keys.

    Example (robointer_droid):

        franka_split(
            schema_id="robointer_droid_v1",
            robot_type="franka_robotiq",
            state_keys=("other_information.observation_joint_position",
                        "other_information.observation_gripper_position"),
            state_dims=(7, 1),
            action_keys=("other_information.action_joint_position",
                         "other_information.action_gripper_velocity"),
            action_dims=(7, 1),
            gripper_action_dim=7,
            images={"observation.images.primary": "observation.images.image0",
                    "observation.images.wrist":   "observation.images.image1"},
        )

    `gripper_action_dim` is the flat-concatenated index (i.e. 7 for the
    typical 7-joint + 1-gripper layout). Delta mask is automatically
    arm-delta + gripper-absolute.

    The ``images`` mapping's target slots may be given either as a short
    ``imageN`` alias or the fully qualified ``observation.images.imageN`` key;
    both are normalized to the canonical form here (C10/C19).
    """
    total = sum(action_dims)
    if not (0 <= gripper_action_dim < total):
        raise ValueError(
            f"gripper_action_dim={gripper_action_dim} out of range for "
            f"sum(action_dims)={total}"
        )
    delta_mask = tuple(i != gripper_action_dim for i in range(total))
    if isinstance(images, dict):
        # C10/C19 fix: normalize each target slot to the canonical
        # ``observation.images.imageN`` form so a short ``image0`` RHS (as the
        # docstring example historically used) does not reach the validator and
        # get rejected. expand_camera_target also rejects non-imageN aliases.
        image_mapping = {
            raw: expand_camera_target(raw, tgt) for raw, tgt in images.items()
        }
    else:
        image_mapping = _expand_images(list(images))
    return DatasetSchema(
        schema_id=schema_id,
        robot_type=robot_type,
        state_keys=tuple(state_keys),
        action_keys=tuple(action_keys),
        state_dims=tuple(state_dims),
        action_dims=tuple(action_dims),
        delta_mask=delta_mask,
        gripper_action_dims=(gripper_action_dim,),
        image_mapping=image_mapping,
        source="manifest",
        source_path=_caller_source_path(),
    )


__all__ = ["franka8", "franka_split"]
