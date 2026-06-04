"""Dataset schema discovery — single source of truth for dataset layout.

Replaces the fragile `robot_type` -> constants dict lookup with a two-tier
chain (explicit meta/labvla_manifest.json > info.json `names` auto-infer)
that produces a frozen `DatasetSchema` object consumed by all downstream code.

See doc/schema_manifest.md for the manifest format and when each tier kicks in.
"""

from utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from .dataset_schema import DatasetSchema
from .discovery import discover_schema
from .errors import (
    SchemaDiscoveryError,
    SchemaError,
    SchemaSpecError,
    SchemaValidationError,
)
from .registry import register, resolve, registered_names

# Blueprint: declarative authoring layer. New schemas should be written as
# SingleArmBlueprint(...).build() or MultiChannelBlueprint(...).build() —
# the compiler emits a DatasetSchema identical to hand-authored low-level
# form, but derives delta_mask / gripper_action_dims / arm_layout
# automatically from the high-level SingleArm spec.
from .annotation_loss import AnnotationLossSpec
from .arm_layout import ArmCount, ArmLayoutSpec
from .blueprint import (
    AnnotationSpec,
    MultiChannelBlueprint,
    SingleArm,
    SingleArmBlueprint,
    StateActionChannel,
)
from .camera_mapping import (
    expand_camera_mapping,
    expand_camera_source,
    expand_camera_target,
    infer_image_mapping_from_info,
    validate_camera_mapping,
)
from .merge_recipe import MergeRecipe, SourceArmLayout
from .validate import validate_arm_layout, validate_schema

__all__ = [
    # Constants re-exported from utils.constants so `from schema import
    # OBS_IMAGES` works without pulling the whole utils package.
    "ACTION",
    "OBS_IMAGES",
    "OBS_STATE",
    # Internal normalized form (still public for rare low-level authoring).
    "DatasetSchema",
    "SchemaError",
    "SchemaDiscoveryError",
    "SchemaSpecError",
    "SchemaValidationError",
    "discover_schema",
    "register",
    "resolve",
    "registered_names",
    # Declarative authoring (preferred entrypoint for new datasets).
    "AnnotationSpec",
    "AnnotationLossSpec",
    "ArmCount",
    "ArmLayoutSpec",
    "MergeRecipe",
    "MultiChannelBlueprint",
    "SingleArm",
    "SingleArmBlueprint",
    "SourceArmLayout",
    "StateActionChannel",
    # Camera mapping SSOT helpers.
    "expand_camera_mapping",
    "expand_camera_source",
    "expand_camera_target",
    "infer_image_mapping_from_info",
    "validate_camera_mapping",
    # Validators.
    "validate_arm_layout",
    "validate_schema",
]
