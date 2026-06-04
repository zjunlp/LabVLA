"""DatasetSchema dataclass — single source of truth for dataset layout.

Everything downstream (transforms, stats remapping, compute_stats,
deployment) consumes this object instead of dispatching on `robot_type`.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional

import torch

from utils.constants import ACTION, OBS_STATE

from .annotation_loss import AnnotationLossSpec
from .arm_layout import ArmLayoutSpec
from .errors import SchemaDiscoveryError  # noqa: F401  # re-export for backward compat


def _freeze_mapping(m: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType(dict(m))


@dataclass(frozen=True)
class DatasetSchema:
    """Frozen description of a LeRobot v3.0 dataset's concrete layout.

    Fields:
        schema_id:
            Stable, human-readable identifier, e.g. "labutopia_level3_press_v1".
            Used in stats keying and log messages.
        robot_type:
            The `robot_type` string from info.json. Kept for attribution;
            NOT used for dispatch.
        state_keys / action_keys:
            Concat order of feature keys that make up the state and action
            vectors after any schema-declared canonicalization. Tuples because
            the schema is hashable + worker-pickleable.
        state_dims / action_dims:
            Per-key dimensionalities (parallel to state_keys / action_keys).
            `sum(action_dims) == len(delta_mask)` is enforced.
        source_state_keys / source_action_keys:
            Optional raw parquet columns required to construct the canonical
            state/action keys. Empty means state_keys/action_keys are already
            present on disk. This supports datasets such as AgiBot, whose raw
            16-dim layout is remapped to the 14-dim dual-arm canonical layout
            before normalization and tokenization.
        delta_mask:
            Length `sum(action_dims)` bool tuple. True means "this action dim
            is delta (relative to t0 state)"; False means "absolute" (gripper
            dims typically).
        gripper_action_dims:
            Indices (into the flat concatenated action vector) that correspond
            to gripper/effector channels. Must all be False in delta_mask.
        image_mapping:
            Raw camera key -> unified "observation.images.imageN" target.
            Frozen to MappingProxyType in __post_init__.
        source:
            Which discovery tier produced this schema: "manifest" | "info_names".
        source_path:
            Absolute path of the manifest / info.json that produced this
            schema. Debugging only.
    """

    schema_id: str
    robot_type: str
    state_keys: tuple[str, ...]
    action_keys: tuple[str, ...]
    state_dims: tuple[int, ...]
    action_dims: tuple[int, ...]
    delta_mask: tuple[bool, ...]
    gripper_action_dims: tuple[int, ...]
    image_mapping: Mapping[str, str]
    source: str
    source_path: Optional[str] = None
    # When True, `discover_schema` skips the reverse check that flags info.json
    # video features absent from image_mapping. Needed for multi-robot merged
    # datasets (e.g. robocoin) where info.json declares all source cameras but
    # the schema intentionally picks only a few.
    #
    # This flag is checked POST-build by ``discover_schema``, not during
    # Blueprint construction. Setting it on a blueprint propagates through
    # ``.build()`` to the resulting DatasetSchema; downstream validators read it
    # after the schema object already exists. It does NOT relax any checks
    # performed during Blueprint authoring itself (e.g. dim arithmetic).
    allow_extra_cameras: bool = False

    # Optional canonical arm-count layout (single-arm 8-dim / dual-arm 14-dim).
    # When set, pretrain/finetune/deploy all use this layout to position
    # gripper(s) at stable canonical indices — see src/schema/arm_layout.py.
    # None = legacy behavior (no canonical remap; gripper_action_dims is the
    # only layout hint).
    arm_layout: Optional[ArmLayoutSpec] = None

    # Per-dataset auxiliary annotation losses. Empty tuple = pure MSE path
    # (oxe-auge, plain OXE, etc.). Non-empty = additional text-CE losses on
    # the named parquet columns. See src/schema/annotation_loss.py.
    # Decoupled from knowledge_isolation: these CEs always flow to the VLM
    # (that is the whole point — supervise the VLM on high-level labels).
    annotation_losses: tuple[AnnotationLossSpec, ...] = ()

    # M-A-4: Gripper physical semantics — what the gripper action channel
    # actually represents in the source data. Used by the cross-dataset
    # semantic guard in scripts/train.py:build_dataset to fail-loud when a
    # multi-repo training set mixes incompatible representations (e.g.
    # robointer's velocity command with LabUtopia's metric-width target).
    # q01/q99 normalization aligns scale but not semantics; mixing them
    # without a canonicalizer trains the model on contradictory targets.
    #
    # Values:
    #   "velocity"      — gripper command is rate-of-change in [-1, +1]
    #                     (e.g. DROID / robointer convention).
    #   "width"         — gripper position as physical width (meters or
    #                     similar metric); LabUtopia / many sim datasets.
    #   "position"      — gripper position as joint angle / actuator pose
    #                     (units depend on robot).
    #   "open_fraction" — normalized 0..1 where 0=closed, 1=fully open.
    #   "binary"        — discrete open/close (0 or 1).
    #   None            — semantics not declared (legacy schemas; the
    #                     cross-dataset guard skips comparison and only
    #                     warns once per launch). Set to a real value to
    #                     opt into the guard.
    gripper_semantic: Optional[str] = None

    # Optional raw source columns used to build the canonical state/action
    # vectors before NormalizeTransformFn / DeltaActionTransformFn. These are
    # deliberately separate from state_keys/action_keys so downstream model code
    # sees only the canonical contract while adapters still know which parquet
    # columns to project from disk.
    source_state_keys: tuple[str, ...] = ()
    source_action_keys: tuple[str, ...] = ()
    source_state_dims: tuple[int, ...] = ()
    source_action_dims: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        # Normalize mutable inputs into their canonical frozen forms BEFORE
        # validation — otherwise an un-normalized list of AnnotationLossSpec
        # objects could pass the validator but later silently mutate.
        if not isinstance(self.image_mapping, MappingProxyType) and self.image_mapping:
            object.__setattr__(
                self, "image_mapping", _freeze_mapping(self.image_mapping)
            )
        if not isinstance(self.annotation_losses, tuple):
            object.__setattr__(
                self, "annotation_losses", tuple(self.annotation_losses)
            )
        # Structural validation lives in schema/validate.py (SSOT).
        from .validate import validate_schema
        validate_schema(self)

    def to_feature_mapping(self) -> dict[str, list[str]]:
        return {
            OBS_STATE: list(self.state_keys),
            ACTION: list(self.action_keys),
        }

    def to_bool_mask(self) -> torch.BoolTensor:
        return torch.tensor(self.delta_mask, dtype=torch.bool)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "schema_id": self.schema_id,
            "robot_type": self.robot_type,
            "state_keys": list(self.state_keys),
            "action_keys": list(self.action_keys),
            "state_dims": list(self.state_dims),
            "action_dims": list(self.action_dims),
            "delta_mask": list(self.delta_mask),
            "gripper_action_dims": list(self.gripper_action_dims),
            "image_mapping": dict(self.image_mapping),
            "source": self.source,
            "source_path": self.source_path,
        }
        if self.arm_layout is not None:
            out["arm_layout"] = self.arm_layout.to_dict()
        if self.annotation_losses:
            out["annotation_losses"] = [s.to_dict() for s in self.annotation_losses]
        if self.gripper_semantic is not None:
            out["gripper_semantic"] = self.gripper_semantic
        if self.source_state_keys or self.source_action_keys:
            out["source_state_keys"] = list(self.source_state_keys)
            out["source_action_keys"] = list(self.source_action_keys)
            out["source_state_dims"] = list(self.source_state_dims)
            out["source_action_dims"] = list(self.source_action_dims)
        return out
