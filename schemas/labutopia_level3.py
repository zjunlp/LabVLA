"""LabUtopia Level3 franka tasks — all variants declared via SingleArmBlueprint.

Every task shares the same 8-dim franka layout (7 arm joints + 1 gripper)
and 3 cameras; only `schema_id` differs per task. Add a new task by
appending a line to `_TASKS` — the module auto-registers it.

Dataset layout (v2.1 raw parquet columns):
  state             float32[8]   (7 joint + 1 gripper; singular "state")
  actions           float32[8]   (plural "actions" — v2.1 quirk)
  camera_1_rgb      video        top-down
  camera_2_rgb      video        wrist
  camera_3_rgb      video        side
"""
from schema import register
from schema.blueprint import SingleArmBlueprint, SingleArm


# (stem used as --dataset_schema name, schema_id for ckpts/logs).
# Add a new Level3 task here to extend the schema registry.
_TASKS = [
    ("labutopia_level3_heat_liquid",      "labutopia_level3_heat_liquid_v1"),
    ("labutopia_level3_open",             "labutopia_level3_open_v1"),
    ("labutopia_level3_pick",             "labutopia_level3_pick_v1"),
    ("labutopia_level3_pour_liquid",      "labutopia_level3_pour_liquid_v1"),
    ("labutopia_level3_press",            "labutopia_level3_press_v1"),
    # `transportbeaker` (no underscore) kept for legacy ckpt compatibility.
    ("labutopia_level3_transport_beaker", "labutopia_level3_transportbeaker_v1"),
]

for _stem, _sid in _TASKS:
    _schema = SingleArmBlueprint(
        schema_id=_sid,
        robot_type="franka",
        arm=SingleArm(
            dof=7,
            # v2.1 shares one column for arm joints + gripper, singular
            # "state" / plural "actions" (quirk of this dataset's raw layout).
            joint_state_col="state",
            joint_action_col="actions",
            gripper_state_col="state",
            gripper_action_col="actions",
            arm_mode="delta",
            gripper_mode="abs",
        ),
        cameras={
            "camera_1_rgb": "image0",
            "camera_2_rgb": "image1",
            "camera_3_rgb": "image2",
        },
        source_path=__file__,
        # M-A-4: LabUtopia gripper is metric width (m) in ~[0, 0.04];
        # see the 2026-04-30 cross-dataset gripper-semantics analysis doc for evidence.
        gripper_semantic="width",
    ).build()
    register(_schema, name=_stem, aliases=[_sid])


# v3.0 variants. The `Level3_*` (no `_old` suffix) datasets on disk are
# v3.0-format and use the canonical `observation.state` / `action` columns,
# not v2.1's `state` / `actions`. Register parallel `*_v3` schemas so the
# same task can be finetuned in either flavor by passing
# `--dataset_schema labutopia_level3_<task>_v3` against the v3 dataset path.
_TASKS_V3 = [
    ("labutopia_level3_heat_liquid_v3",      "labutopia_level3_heat_liquid_v3"),
    ("labutopia_level3_open_v3",             "labutopia_level3_open_v3"),
    ("labutopia_level3_pick_v3",             "labutopia_level3_pick_v3"),
    ("labutopia_level3_pour_liquid_v3",      "labutopia_level3_pour_liquid_v3"),
    ("labutopia_level3_press_v3",            "labutopia_level3_press_v3"),
    ("labutopia_level3_transport_beaker_v3", "labutopia_level3_transportbeaker_v3"),
]

for _stem, _sid in _TASKS_V3:
    _schema = SingleArmBlueprint(
        schema_id=_sid,
        robot_type="franka",
        arm=SingleArm(
            dof=7,
            joint_state_col="observation.state",
            joint_action_col="action",
            gripper_state_col="observation.state",
            gripper_action_col="action",
            arm_mode="delta",
            gripper_mode="abs",
        ),
        cameras={
            "camera_1_rgb": "image0",
            "camera_2_rgb": "image1",
            "camera_3_rgb": "image2",
        },
        source_path=__file__,
        # M-A-4: LabUtopia gripper is metric width (m) in ~[0, 0.04];
        # see the 2026-04-30 cross-dataset gripper-semantics analysis doc for evidence.
        gripper_semantic="width",
    ).build()
    register(_schema, name=_stem, aliases=[_sid])


# The `schemas` package auto-loader also looks for a module-level `SCHEMA`
# attribute; registration above is the real wiring, so expose None to opt out
# of the implicit single-schema register.
SCHEMA = None
