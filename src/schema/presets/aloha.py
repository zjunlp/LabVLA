"""ALOHA bimanual presets."""
from __future__ import annotations

from typing import Sequence

from ..dataset_schema import DatasetSchema
from .franka import _caller_source_path, _expand_images


def aloha14(
    schema_id: str,
    *,
    cams: Sequence[str] = ("cam_high", "cam_low", "cam_left_wrist"),
    robot_type: str = "aloha",
    state_key: str = "observation.state",
    action_key: str = "action",
) -> DatasetSchema:
    """14-dim ALOHA preset: two 7-DoF arms, each ending in an absolute gripper.

    Convention: dims 0..5 left arm joints (delta), dim 6 left gripper (abs),
    dims 7..12 right arm joints (delta), dim 13 right gripper (abs).
    """
    delta_mask = tuple(i not in (6, 13) for i in range(14))
    return DatasetSchema(
        schema_id=schema_id,
        robot_type=robot_type,
        state_keys=(state_key,),
        action_keys=(action_key,),
        state_dims=(14,),
        action_dims=(14,),
        delta_mask=delta_mask,
        gripper_action_dims=(6, 13),
        image_mapping=_expand_images(cams),
        source="manifest",
        source_path=_caller_source_path(),
    )


__all__ = ["aloha14"]
