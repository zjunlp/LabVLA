"""robointer_droid — franka_robotiq, split state/action keys, 2 cameras (v2.1).

Dataset layout (per-frame parquet columns):
  other_information.observation_joint_position    float32[7]  arm joint positions
  other_information.observation_gripper_position  float32[1]  gripper finger pos
  other_information.action_joint_position         float32[7]  arm joint targets
  other_information.action_gripper_velocity       float32[1]  raw velocity cmd (unused)
  other_information.action_gripper_position       float32[1]  virtual next-frame gripper pos

Replaces the old `from schema.presets import franka_split` factory so that all
fields (delta_mask, gripper_action_dims, image_mapping) are directly visible
in this file — matching the robocoin.py / labutopia_level3.py convention.

The gripper action key is virtual: LeRobotAdapterBase derives
`other_information.action_gripper_position[t]` from
`other_information.observation_gripper_position[t+1]`. The final physical frame
has no t+1 gripper observation and is excluded from the trainable sample index.
"""
from schema import DatasetSchema
from schema.annotation_loss import AnnotationLossSpec
from schema.arm_layout import ArmLayoutSpec, ArmCount

T = True
F = False

SCHEMA = DatasetSchema(
    schema_id="robointer_droid_v2",
    robot_type="franka_robotiq",
    state_keys=(
        "other_information.observation_joint_position",
        "other_information.observation_gripper_position",
    ),
    state_dims=(7, 1),
    action_keys=(
        "other_information.action_joint_position",
        "other_information.action_gripper_position",
    ),
    action_dims=(7, 1),
    # arm (7 dims) → delta (relative to t0 state); gripper (1 dim) → absolute
    # next-frame position target, not a delta.
    delta_mask=(T, T, T, T, T, T, T, F),
    # Flat index of gripper after concat: dims 0..6 = joints, dim 7 = gripper.
    gripper_action_dims=(7,),
    image_mapping={
        "observation.images.primary": "observation.images.image0",
        "observation.images.wrist":   "observation.images.image1",
    },
    source="manifest",
    source_path=__file__,
    # Same canonical single-arm layout as oxe-auge_clean: Franka 7-DoF arm +
    # 1 gripper → gripper at dim 7. Used by deployment to reverse-map the
    # model's canonical output back to Franka raw joint format.
    arm_layout=ArmLayoutSpec(
        arm_count=ArmCount.SINGLE,
        arm_dof=7,
        gripper_index_in_raw=7,
    ),
    # Annotation CE intentionally disabled: pure trajectory MSE keeps this a
    # clean pi0-objective comparison point and avoids heterogeneous-batch
    # complexity from mixing annotated robointer with un-annotated oxe-auge.
    # Re-enable by restoring an AnnotationLossSpec entry here.
    annotation_losses=(),
    # M-A-4 update: gripper supervision no longer uses DROID's velocity
    # command. It is a synthetic absolute actuator/finger position target
    # derived from the next observation frame.
    gripper_semantic="position",
)
