"""LabEmbodied_data_beta schemas — 4 robot families, all canonicalize to
LabVLA's single-arm 8-dim layout (7 arm joints + 1 gripper at dim 7).

Dataset root: /all-flash-data/Pretrain_Data/LabEmbodied_data_beta/<task>/
Per-task v3.0 LeRobot layout; parquet column names follow v2.1 convention
(``state``, ``actions``, ``camera_<N>_rgb``) because the dataset was
produced via JPEG-passthrough v2.1 writer + lerobot's official v21→v30
upgrade (preserves column names, only reshapes sharding).

Raw layouts observed (verified 2026-05-20):
  - Franka  (8-dim) : dims 0..6 = 7 arm joints, dim 7 = 1 scalar gripper.
    76 datasets (A0X / B0X / atomic_* — 56 with 3 cams, 20 with 2 cams).
    Already matches canonical → CanonicalSingleArmLayoutTransformFn is a
    no-op for these (source dims == canonical dims).
  - UR (5e / 16e, 6-DoF) 11-dim : dims 0..5 = arm, dim 6 = gripper scalar,
    dims 7..10 = perfectly correlated mirror copies of dim 6 (verified
    corr=±1.0). 16 datasets (ur16e_*, ur5e_*), use camera_4_rgb instead
    of camera_3_rgb.
  - festo (6-DoF) 11-dim : same dim layout as UR but festo robot.
    10 datasets (festo_*), 2 cams.
  - rizon4 (Flexiv 7-DoF) 12-dim : dims 0..6 = arm, dim 7 = gripper scalar,
    dims 8..11 = mirror-redundancy copies of dim 7. 5 datasets, 2 cams.

Canonical 8-dim mapping (CanonicalSingleArmLayoutTransformFn handles this
at training time, no parquet rewrite needed):
  out[..., :arm_dof]  = raw[..., :arm_dof]      # arm joints, dim 0..arm_dof-1
  out[..., arm_dof:7] = 0                       # zero-pad up to dim 6 inclusive
  out[..., 7]         = raw[..., gripper_index_in_raw]   # scalar gripper
  # raw dims past gripper_index_in_raw are dropped (mirror redundancy)

Gripper semantics: LabEmbodied data was captured in sim with metric width
on dim 7 of the canonical 8-dim — matches LabUtopia/Level3 ``width``
convention, not the OXE ``open_fraction`` convention.
"""
from schema import DatasetSchema, register
from schema.annotation_loss import AnnotationLossSpec
from schema.arm_layout import ArmCount, ArmLayoutSpec
from schema.camera_mapping import expand_camera_mapping
from utils.constants import ACTION, OBS_STATE


T = True
F = False


def _make_schema(
    schema_id: str,
    robot_type: str,
    raw_arm_dof: int,
    raw_gripper_index_in_raw: int,
    raw_dim: int,
    cameras: dict,
) -> DatasetSchema:
    """Build one LabEmbodied single-arm schema.

    Args:
        schema_id: Unique stable id (e.g. ``"labembodied_franka_8"``).
        robot_type: Source robot name used in info.json / log messages.
        raw_arm_dof: Number of arm joints in the RAW parquet state vector
            (6 for UR/festo, 7 for Franka/rizon4).
        raw_gripper_index_in_raw: 0-indexed position of the scalar gripper
            signal in the raw state vector (6 for 11-dim, 7 for 8/12-dim).
        raw_dim: Total length of the raw state/action vector on disk.
        cameras: Raw camera key → canonical ``observation.images.imageN``
            slot mapping.
    """
    # Canonical 8-dim: 7 arm + 1 grip, all delta except grip absolute.
    delta_mask = (T,) * 7 + (F,)
    needs_canonicalization = (raw_dim != 8) or (raw_gripper_index_in_raw != 7)
    return DatasetSchema(
        schema_id=schema_id,
        robot_type=robot_type,
        # Post-transform canonical: a single 8-dim concat key. ``OBS_STATE``
        # / ``ACTION`` resolve to ``observation.state`` / ``action`` per
        # ``utils.constants``.
        state_keys=(OBS_STATE,),
        action_keys=(ACTION,),
        state_dims=(8,),
        action_dims=(8,),
        delta_mask=delta_mask,
        gripper_action_dims=(7,),
        image_mapping=expand_camera_mapping(cameras),
        source="manifest",
        source_path=__file__,
        # LabEmbodied tasks have variable camera count (2 or 3) — let
        # multi-cam datasets coexist via this flag.
        allow_extra_cameras=True,
        arm_layout=ArmLayoutSpec(
            arm_count=ArmCount.SINGLE,
            # NOTE: arm_dof / gripper_index_in_raw describe the RAW layout
            # (matches the dual-arm convention where these are RAW values
            # consumed by the canonicalization transform — see
            # CanonicalSingleArmLayoutTransformFn hydration in
            # src/transforms/core.py).
            arm_dof=raw_arm_dof,
            gripper_index_in_raw=raw_gripper_index_in_raw,
        ),
        # When raw layout matches canonical exactly (Franka 8-dim) we still
        # set source_*_keys so the transform path is exercised uniformly;
        # the materialize step then just copies raw[:8] → out unchanged.
        # For raw != canonical we MUST set these so the transform's
        # hydration enables it (per core.py hydration block).
        source_state_keys=("state",) if needs_canonicalization else (),
        source_action_keys=("actions",) if needs_canonicalization else (),
        source_state_dims=(raw_dim,) if needs_canonicalization else (),
        source_action_dims=(raw_dim,) if needs_canonicalization else (),
        # LabUtopia-style metric gripper width (see CLAUDE.md memory
        # ``labvla_arm_layout_canonical_2026_04_23``).
        gripper_semantic="width",
    )


# ---------------------------------------------------------------------------
# 4 robot-family schemas registered for direct use via --dataset_schema.
# ---------------------------------------------------------------------------

# Franka 8-dim (already canonical) — covers 76 LabEmbodied tasks across two
# camera-count variants. The 3-cam variant is the primary; downstream
# allow_extra_cameras=True permits 2-cam tasks to bind to the same schema
# (Qwen3VL processor consumes only the cameras present).
_FRANKA_3CAM = {
    "camera_1_rgb": "image0",
    "camera_2_rgb": "image1",
    "camera_3_rgb": "image2",
}
_FRANKA_2CAM = {
    "camera_1_rgb": "image0",
    "camera_2_rgb": "image1",
}
_UR_3CAM = {
    "camera_1_rgb": "image0",
    "camera_2_rgb": "image1",
    # ur5e/ur16e datasets have camera_4_rgb (not 3); see survey 2026-05-20.
    "camera_4_rgb": "image2",
}

register(
    _make_schema(
        schema_id="labembodied_franka_8",
        robot_type="franka_single_arm",
        raw_arm_dof=7,
        raw_gripper_index_in_raw=7,
        raw_dim=8,
        cameras=_FRANKA_3CAM,
    ),
    name="labembodied_franka_8",
    aliases=["labembodied_franka_v1"],
)

register(
    _make_schema(
        schema_id="labembodied_franka_8_2cam",
        robot_type="franka_single_arm_2cam",
        raw_arm_dof=7,
        raw_gripper_index_in_raw=7,
        raw_dim=8,
        cameras=_FRANKA_2CAM,
    ),
    name="labembodied_franka_8_2cam",
    aliases=["labembodied_franka_2cam_v1"],
)

# UR (5e / 16e) 6-DoF, 11-dim raw with mirror-redundant gripper joints in
# dims 7..10. Uses camera_4_rgb (no camera_3_rgb on this robot family).
register(
    _make_schema(
        schema_id="labembodied_ur_11_6dof",
        robot_type="ur_single_arm_6dof",
        raw_arm_dof=6,
        raw_gripper_index_in_raw=6,
        raw_dim=11,
        cameras=_UR_3CAM,
    ),
    name="labembodied_ur_11_6dof",
    aliases=["labembodied_ur_v1"],
)

# festo 6-DoF, 11-dim raw with mirror-redundant gripper joints in dims
# 7..10. Two cameras only.
register(
    _make_schema(
        schema_id="labembodied_festo_11_6dof",
        robot_type="festo_single_arm_6dof",
        raw_arm_dof=6,
        raw_gripper_index_in_raw=6,
        raw_dim=11,
        cameras=_FRANKA_2CAM,
    ),
    name="labembodied_festo_11_6dof",
    aliases=["labembodied_festo_v1"],
)

# rizon4 7-DoF (Flexiv Rizon 4), 12-dim raw with mirror-redundant gripper
# joints in dims 8..11. 3-cam variant (camera_1/2/4_rgb — same camera_4
# convention as UR family, no camera_3_rgb on this rig).
register(
    _make_schema(
        schema_id="labembodied_rizon4_12_7dof",
        robot_type="rizon4_single_arm_7dof",
        raw_arm_dof=7,
        raw_gripper_index_in_raw=7,
        raw_dim=12,
        cameras=_UR_3CAM,
    ),
    name="labembodied_rizon4_12_7dof",
    aliases=["labembodied_rizon4_v1"],
)

# rizon4 7-DoF 2-cam variant — 5 round-2 tasks (pick_beaker, place, pour,
# shake, stir) ship only camera_1_rgb + camera_2_rgb. Same raw layout
# (12-dim, gripper at dim 7) as the 3-cam variant.
register(
    _make_schema(
        schema_id="labembodied_rizon4_12_7dof_2cam",
        robot_type="rizon4_single_arm_7dof_2cam",
        raw_arm_dof=7,
        raw_gripper_index_in_raw=7,
        raw_dim=12,
        cameras=_FRANKA_2CAM,
    ),
    name="labembodied_rizon4_12_7dof_2cam",
    aliases=["labembodied_rizon4_2cam_v1"],
)


# The schemas package auto-loader also looks for a module-level `SCHEMA`
# attribute; this module registers via the explicit ``register()`` API
# above, so opt out of the SCHEMA fallback.
SCHEMA = None
