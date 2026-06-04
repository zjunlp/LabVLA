"""RoboCOIN merged pretrain dataset — multi-channel schema via Blueprint.

/all-flash-data/Pretrain_Data/robocoin is merged from 16+ source robots
(AIRBOT-MMK2, Kinova, Piper, xArm, UR5, dual-arm combos, ...). Each source
has a different native action layout — the merge pads `action` to 30 dim
(arm + embedded gripper, variable per source) and exposes a separate
2-dim dual-arm gripper signal via `gripper_open_scale_action` (0-1
open-scale, slot [0]=left, slot [1]=right; single-arm robots leave slot
[1] inactive).

Two independent channels, different dims and delta modes:

  observation.state              float32[≤30]   arm proprioception
  action                         float32[≤30]   arm command (+ embedded gripper, noisy)
  gripper_open_scale_state       float32[2]     dual-arm 0-1 grip open scale
  gripper_open_scale_action      float32[2]     dual-arm 0-1 grip target

The dedicated gripper channel is the clean signal the model should learn;
the embedded gripper in `action` is kept for backward compat but trained
as part of the delta arm channel (noise the model will marginalize).
"""
from schema.blueprint import MultiChannelBlueprint, StateActionChannel


SCHEMA = MultiChannelBlueprint(
    schema_id="robocoin_merged_v2",
    robot_type="multi_robot",
    channels=(
        # Channel 0: arm proprioception + arm command (gripper embedded but
        # handled as delta — the model learns to denoise).
        StateActionChannel(
            state_col="observation.state",
            action_col="action",
            dim=30,
            mode="delta",
            is_gripper=False,
        ),
        # Channel 1: dedicated dual-arm gripper open scale (abs; π0.5 /
        # openpi convention for near-binary gripper signals).
        StateActionChannel(
            state_col="gripper_open_scale_state",
            action_col="gripper_open_scale_action",
            dim=2,
            mode="abs",
            is_gripper=True,
        ),
    ),
    cameras={
        "observation.images.cam_head_rgb":        "image0",
        "observation.images.cam_left_wrist_rgb":  "image1",
        "observation.images.cam_right_wrist_rgb": "image2",
    },
    # Merged info.json declares 29 cams; we pick 3, the rest are benign.
    allow_extra_cameras=True,
    source_path=__file__,
    # Robocoin sources span single-arm and dual-arm robots; the canonical
    # layout isn't a single ArmLayoutSpec. Leave None — deployment uses the
    # gripper_action_dims metadata directly (dims 30, 31).
    arm_layout_canonical=None,
    # M-A-4: gripper_open_scale_* is a 0-1 normalized open scale (already
    # canonicalized by upstream merge); semantically open_fraction.
    gripper_semantic="open_fraction",
).build()
