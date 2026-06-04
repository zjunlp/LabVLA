"""Dataset Blueprint — declarative single-source-of-truth for authoring schemas.

Users describe a dataset with high-level constructs (arm DoF, parquet columns,
camera roles, annotations). `.build()` compiles these into a `DatasetSchema` —
the same internal dataclass that every downstream module (stats, merge,
preflight, adapter, transform chain, model, deploy) already consumes.

Two top-level blueprints cover every dataset we have:

    SingleArmBlueprint      — arm + gripper layout (oxe_auge, labutopia_level3,
                              future droid-family datasets).
    MultiChannelBlueprint   — arbitrary list of (state_col, action_col, dim,
                              mode) channels (robocoin's action + dedicated
                              gripper_open_scale_action).

Anything a DatasetSchema needs but the blueprint can't infer (e.g. canonical
dual-arm layout on a multi-channel robocoin) is passed explicitly as an
argument. The blueprint compiler is a pure function — running it twice
yields bit-identical DatasetSchema.to_dict() — which makes stats.json /
labvla_manifest.json / checkpoint compatibility self-verifying.

Normalization mode
------------------
The project-wide default is ``"mean_std"`` — z-score, set at the transform
chain level in ``policies/LabVLA/configuration_labvla.py``. ``hydrate_all``
in ``transforms/core.py`` then auto-injects per-dim ``"q01_q99"`` overrides
on the canonical gripper indices declared by ``schema.gripper_action_dims``,
so the effective policy is "arm dims → mean_std, gripper dims → q01_q99".
Schemas don't carry a per-dim normalization field; overrides flow through
``schema.gripper_action_dims`` (and ``arm_layout.gripper_indices_canonical``,
which mirrors it).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

from .annotation_loss import AnnotationLossSpec
from .arm_layout import ArmLayoutSpec, ArmCount
from .camera_mapping import expand_camera_mapping
from .dataset_schema import DatasetSchema


# --------------------------------------------------------------------------- #
# Authoring primitives                                                        #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AnnotationSpec:
    """Blueprint-level shorthand for an annotation CE loss declaration.

    Compiled to `schema.annotation_loss.AnnotationLossSpec` at build() time.
    """

    field: str
    weight: float = 0.5
    max_length: int = 32
    loss_type: str = "ce_text"

    def to_loss_spec(self) -> AnnotationLossSpec:
        return AnnotationLossSpec(
            field=self.field,
            loss_type=self.loss_type,
            weight=self.weight,
            max_length=self.max_length,
        )


@dataclass(frozen=True)
class SingleArm:
    """Single-arm spec: arm joints + one gripper channel.

    The four parquet columns fully describe the per-frame layout. Gripper
    state/action may live in the *same* column as arm joints (e.g. oxe_auge's
    `observation.joints` holds both) — in that case the user sets the same
    column name for both `joint_*` and `gripper_*`; the blueprint infers
    that only one state/action key is needed.
    """

    dof: int
    joint_state_col: str
    joint_action_col: str
    gripper_state_col: str
    gripper_action_col: str
    arm_mode: str = "delta"       # "delta" | "abs"
    gripper_mode: str = "abs"     # "delta" | "abs"

    def __post_init__(self) -> None:
        if self.dof <= 0:
            raise ValueError(f"SingleArm.dof must be > 0, got {self.dof}")
        for name, mode in (("arm_mode", self.arm_mode),
                           ("gripper_mode", self.gripper_mode)):
            if mode not in ("delta", "abs"):
                raise ValueError(
                    f"SingleArm.{name} must be 'delta' or 'abs', got {mode!r}"
                )


@dataclass(frozen=True)
class StateActionChannel:
    """One independent (state_col, action_col) channel with its own dim/mode.

    Used by MultiChannelBlueprint for datasets whose action vector is split
    across multiple parquet columns (robocoin: `action` for arm +
    `gripper_open_scale_action` for gripper, with different dims and modes).
    """

    state_col: str
    action_col: str
    dim: int
    mode: str = "delta"           # "delta" | "abs"
    is_gripper: bool = False      # marks all `dim` flat indices as gripper dims

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError(f"StateActionChannel.dim must be > 0, got {self.dim}")
        if self.mode not in ("delta", "abs"):
            raise ValueError(
                f"StateActionChannel.mode must be 'delta' or 'abs', got {self.mode!r}"
            )
        if self.is_gripper and self.mode == "delta":
            raise ValueError(
                f"is_gripper=True channel must use mode='abs' (gripper never delta). "
                f"Got mode={self.mode!r}, state_col={self.state_col!r}"
            )


# --------------------------------------------------------------------------- #
# Top-level blueprints                                                        #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SingleArmBlueprint:
    """Author a single-arm dataset in ~15 lines.

    Given the SingleArm spec, the blueprint derives:
        state_keys  = (joint_state_col,) or (joint_state_col, gripper_state_col)
        state_dims  = (dof,) or (dof, 1)        [merged when cols are identical]
        action_keys = (joint_action_col,) or (joint_action_col, gripper_action_col)
        action_dims = (dof,) or (dof, 1)
        delta_mask  = arm_mode × dof  + gripper_mode × 1
        gripper_action_dims = (dof,)
        arm_layout  = ArmLayoutSpec(SINGLE, arm_dof=dof, gripper_index_in_raw=dof)
    """

    schema_id: str
    robot_type: str
    arm: SingleArm
    cameras: Mapping[str, str]
    annotations: tuple[AnnotationSpec, ...] = ()
    source_path: Optional[str] = None
    # Consumed POST-build by `discover_schema` (only relaxes the reverse
    # "info.json has extra cameras" check). Blueprint authoring does NOT read
    # this flag — it only forwards the value into the built DatasetSchema.
    allow_extra_cameras: bool = False
    # M-A-4: physical semantics of the gripper action channel — see
    # DatasetSchema.gripper_semantic. Forwarded into the built schema; default
    # None preserves legacy authoring (no cross-dataset semantic guard).
    gripper_semantic: Optional[str] = None

    def build(self) -> DatasetSchema:
        a = self.arm

        # Same column for joint + gripper (e.g. oxe_auge observation.joints
        # holds [j0..j6, grip] in a single column) → collapse to one key.
        state_collapsed = a.joint_state_col == a.gripper_state_col
        action_collapsed = a.joint_action_col == a.gripper_action_col

        if state_collapsed:
            state_keys = (a.joint_state_col,)
            state_dims = (a.dof + 1,)
        else:
            state_keys = (a.joint_state_col, a.gripper_state_col)
            state_dims = (a.dof, 1)

        if action_collapsed:
            action_keys = (a.joint_action_col,)
            action_dims = (a.dof + 1,)
        else:
            action_keys = (a.joint_action_col, a.gripper_action_col)
            action_dims = (a.dof, 1)

        delta_mask = (
            tuple([a.arm_mode == "delta"] * a.dof)
            + (a.gripper_mode == "delta",)
        )
        gripper_action_dims = (a.dof,)

        arm_layout = ArmLayoutSpec(
            arm_count=ArmCount.SINGLE,
            arm_dof=a.dof,
            gripper_index_in_raw=a.dof,
        )

        annotation_losses = tuple(s.to_loss_spec() for s in self.annotations)

        image_mapping = expand_camera_mapping(self.cameras)

        return DatasetSchema(
            schema_id=self.schema_id,
            robot_type=self.robot_type,
            state_keys=state_keys,
            action_keys=action_keys,
            state_dims=state_dims,
            action_dims=action_dims,
            delta_mask=delta_mask,
            gripper_action_dims=gripper_action_dims,
            image_mapping=image_mapping,
            source="manifest",
            source_path=self.source_path,
            allow_extra_cameras=self.allow_extra_cameras,
            arm_layout=arm_layout,
            annotation_losses=annotation_losses,
            gripper_semantic=self.gripper_semantic,
        )


@dataclass(frozen=True)
class MultiChannelBlueprint:
    """Author a dataset whose state/action spans multiple independent channels.

    Use when the robot-native layout isn't ``arm + 1 gripper`` — e.g. robocoin
    merges 16+ source robots with a dedicated dual-arm `gripper_open_scale_*`
    channel separate from the main `action` column.

    Because `channels` may not map onto any standard ArmLayout, the caller
    passes `arm_layout_canonical` explicitly (or leaves it None if the dataset
    is not "arm-shaped" at all).
    """

    schema_id: str
    robot_type: str
    channels: tuple[StateActionChannel, ...]
    cameras: Mapping[str, str]
    annotations: tuple[AnnotationSpec, ...] = ()
    arm_layout_canonical: Optional[ArmLayoutSpec] = None
    source_path: Optional[str] = None
    # Consumed POST-build by `discover_schema`. See SingleArmBlueprint
    # for the full note; same semantics apply here.
    allow_extra_cameras: bool = False
    # See SingleArmBlueprint.gripper_semantic.
    gripper_semantic: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.channels:
            raise ValueError("MultiChannelBlueprint.channels must be non-empty")

    def build(self) -> DatasetSchema:
        state_keys = tuple(c.state_col for c in self.channels)
        action_keys = tuple(c.action_col for c in self.channels)
        state_dims = tuple(c.dim for c in self.channels)
        action_dims = tuple(c.dim for c in self.channels)

        # delta_mask: concatenate per-channel (dim × mode flag).
        delta_mask: tuple[bool, ...] = ()
        gripper_idx_list: list[int] = []
        offset = 0
        for c in self.channels:
            delta_mask = delta_mask + tuple([c.mode == "delta"] * c.dim)
            if c.is_gripper:
                gripper_idx_list.extend(range(offset, offset + c.dim))
            offset += c.dim
        gripper_action_dims = tuple(gripper_idx_list)

        annotation_losses = tuple(s.to_loss_spec() for s in self.annotations)
        image_mapping = expand_camera_mapping(self.cameras)

        return DatasetSchema(
            schema_id=self.schema_id,
            robot_type=self.robot_type,
            state_keys=state_keys,
            action_keys=action_keys,
            state_dims=state_dims,
            action_dims=action_dims,
            delta_mask=delta_mask,
            gripper_action_dims=gripper_action_dims,
            image_mapping=image_mapping,
            source="manifest",
            source_path=self.source_path,
            allow_extra_cameras=self.allow_extra_cameras,
            arm_layout=self.arm_layout_canonical,
            annotation_losses=annotation_losses,
            gripper_semantic=self.gripper_semantic,
        )
