"""AgiBot / Genie-1 humanoid presets."""
from __future__ import annotations

from typing import Sequence

from ..dataset_schema import DatasetSchema
from .franka import _caller_source_path, _expand_images


def agibot_g1(
    schema_id: str,
    *,
    cams: Sequence[str] = ("head_color", "hand_left_color", "hand_right_color"),
    robot_type: str = "agibot_g1",
    state_key: str = "observation.state",
    action_key: str = "action",
) -> DatasetSchema:
    """AgiBot Genie-1 preset: 14 arm joints (7 per arm, delta) + 2 grippers (abs).

    Layout mirrors ALOHA-style bimanual but with AgiBot's camera naming.
    Gripper indices 7 and 15 in the flat 16-dim action vector.
    """
    delta_mask = tuple(i not in (7, 15) for i in range(16))
    return DatasetSchema(
        schema_id=schema_id,
        robot_type=robot_type,
        state_keys=(state_key,),
        action_keys=(action_key,),
        state_dims=(16,),
        action_dims=(16,),
        delta_mask=delta_mask,
        gripper_action_dims=(7, 15),
        image_mapping=_expand_images(cams),
        source="manifest",
        source_path=_caller_source_path(),
    )


__all__ = ["agibot_g1"]
