"""OXE-AugE merged schema (oxe-auge_clean).

Single-arm, 7 DoF + 1 gripper layout across all 180 sub-sources after merge.
Authored via `SingleArmBlueprint` — the blueprint compiler derives
`state_keys`, `action_keys`, `delta_mask`, `gripper_action_dims`, and
`arm_layout` from the one `SingleArm` spec below.

Merge-time handling of source heterogeneity (some sources are 6-DoF raw; we
promote them to 7-DoF canonical by zero-padding dim 6 and moving the raw
gripper to dim 7) is described declaratively by `MERGE_RECIPE`, which
`data_process/merge_v3/build_oxe_auge_clean.py` consumes.
"""
from schema.blueprint import SingleArmBlueprint, SingleArm
from schema.merge_recipe import MergeRecipe, SourceArmLayout
from schema.arm_layout import ArmLayoutSpec, ArmCount


SCHEMA = SingleArmBlueprint(
    schema_id="oxe_auge_v1",
    robot_type="oxe_auge_merged_single_arm",
    arm=SingleArm(
        dof=7,
        # After merge all shards use `observation.joints` (8-dim: 7 arm + 1
        # grip) and `observation.state` (same 8-dim). Joint column and
        # gripper column are the same parquet column — blueprint collapses
        # to one state_key / one action_key automatically.
        #
        # Intentionally NO source_state_keys/dims: every merged source's
        # `observation.state` is already canonical 8-dim, so runtime state
        # canonicalization stays off. If a future audit proves a source ships a
        # non-canonical `observation.state`, declare source_state_keys/dims here
        # (mirroring the action side) rather than relying on lerobot_base's
        # silent tail zero-pad.
        joint_state_col="observation.state",
        joint_action_col="observation.joints",
        gripper_state_col="observation.state",
        gripper_action_col="observation.joints",
        arm_mode="delta",
        gripper_mode="abs",
    ),
    cameras={
        "observation.images.image": "image0",
    },
    allow_extra_cameras=True,
    source_path=__file__,
    # M-A-4 / oxe-auge_clean_v2 invariant: gripper dim 7 has been canonicalized
    # to open_fraction ∈ [0, 1] (0=closed, 1=open) by data_process/merge_v3/
    # gripper_canonicalize.py during the v2 merge. v1 (oxe-auge_clean) was
    # NOT canonicalized — but v1 is deprecated, only v2 should be used for new
    # training. The cross-dataset semantic guard in scripts/train.py will use
    # this annotation to gate any future multi-repo mix that includes oxe-auge.
    gripper_semantic="open_fraction",
).build()


# Declarative source → canonical layout promotion rules. Consumed by the
# merge script at data prep time (not at training time).
MERGE_RECIPE = MergeRecipe(
    canonical=ArmLayoutSpec(
        arm_count=ArmCount.SINGLE,
        arm_dof=7,
        gripper_index_in_raw=7,
    ),
    source_layouts={
        # 6-DoF sources — zero-pad arm dim 6, move raw grip from 6 to 7.
        "berkeley_autolab_ur5": SourceArmLayout(arm_dof=6, raw_gripper_idx=6),
        "bridge":               SourceArmLayout(arm_dof=6, raw_gripper_idx=6),
        "jaco_play":            SourceArmLayout(arm_dof=6, raw_gripper_idx=6),
    },
    # All other sources are 7-DoF with grip at raw index 7 → no rewrite.
    default_layout=SourceArmLayout(arm_dof=7, raw_gripper_idx=7),
)
