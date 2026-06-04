"""agibot_dual_arm — AgiBot World v3.0 dual-arm canonical schema.

Per-task LeRobot v3.0 dataset under
``/all-flash-data/Pretrain_Data/agibot_world_beta_lerobot/agibotworld/task_NNN``.
Each task dir is one behavior with one global instruction (1-row tasks.parquet),
so the "subtask" prediction target is the per-frame ``task`` string the v30
adapter resolves from ``task_index → tasks.parquet``.

Raw source layout:
  joint.position     dim 0..6  left arm, dim 7..13 right arm
  effector.position  dim 0 left gripper, dim 1 right gripper

Training canonical layout:
  dim 0..5   left arm joints 0..5
  dim 6      left gripper
  dim 7..12  right arm joints 0..5
  dim 13     right gripper

The raw 16-dim source is canonicalized to 14 dims by
``CanonicalArmLayoutTransformFn`` before normalization / state discretization /
FAST tokenization, then right-padded to ``max_action_dim=32`` by
``PadStateAndActionTransformFn``.

action_mode=abs is the natural mode (π0.5 §App B.1 "predict target joint and
end-effector poses"); agibot stores absolute joint targets and absolute
gripper positions in the source data.

Cameras: agibot has 8 raw cameras (head + 4 fisheye + 2 hand + back-fisheye).
We pick **3** (head + hand_left + hand_right) per the
``Qwen3_VLProcessorTransformFn`` 3-camera limit (hydrate_all line 825-834).

Annotation losses: only ``annotation.subtask`` (CE on the 1-per-task
description string). ``BuildAgiBotSubtaskTransformFn`` (src/transforms/
agibot_subtask.py) copies ``data["task"]`` → ``data["annotation.subtask"]`` and
rewrites ``data["task"]`` to a generic prompt so the model is not trivially
copying its input.
"""
from schema import DatasetSchema
from schema.annotation_loss import AnnotationLossSpec
from schema.arm_layout import ArmCount, ArmLayoutSpec
from utils.constants import ACTION, OBS_STATE

T = True
F = False

SCHEMA = DatasetSchema(
    schema_id="agibot_dual_arm_v1",
    robot_type="agibot_dual_arm",
    state_keys=(
        OBS_STATE,
    ),
    state_dims=(14,),
    action_keys=(
        ACTION,
    ),
    action_dims=(14,),
    # Canonical 14-dim dual arm: 6 joints + gripper per side.
    # Joint dims are delta-capable; grippers are absolute targets.
    delta_mask=(T,) * 6 + (F,) + (T,) * 6 + (F,),
    # Flat canonical gripper indices.
    gripper_action_dims=(6, 13),
    # 3-camera selection — matches Qwen3VL processor limit.
    # Format: <raw_key_in_parquet/video> → <observation.images.imageN slot>.
    image_mapping={
        "observation.images.head":       "observation.images.image0",
        "observation.images.hand_left":  "observation.images.image1",
        "observation.images.hand_right": "observation.images.image2",
    },
    source="manifest",
    source_path=__file__,
    # AgiBot info.json declares 8 videos; this schema intentionally keeps the
    # 3 Qwen3VL-supported cameras above and treats the rest as extra inputs.
    allow_extra_cameras=True,
    arm_layout=ArmLayoutSpec(
        arm_count=ArmCount.DUAL,
        left_arm_dof=7,
        right_arm_dof=7,
        left_gripper_index_in_raw=14,
        right_gripper_index_in_raw=15,
    ),
    source_state_keys=(
        "observation.states.joint.position",
        "observation.states.effector.position",
    ),
    source_state_dims=(14, 2),
    source_action_keys=(
        "actions.joint.position",
        "actions.effector.position",
    ),
    source_action_dims=(14, 2),
    annotation_losses=(
        AnnotationLossSpec(
            field="annotation.subtask",
            loss_type="ce_text",
            weight=1.0,
            max_length=64,  # agibot task strings are 30-120 chars typical
        ),
    ),
    # Effector position dim 1 is a 0..1 open scale (verified from per-task
    # stats.json: actions.effector.position[1].max=1.0, q99=0.9999). Slot 0
    # is essentially zero in this dataset. Treat as open_fraction (compatible
    # with oxe-auge open_fraction at the canonicalization boundary).
    gripper_semantic="open_fraction",
)
