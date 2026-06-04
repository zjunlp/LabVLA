"""`meta/labvla_manifest.json` parser + validator."""

from __future__ import annotations

import json
from pathlib import Path

from .camera_mapping import expand_camera_mapping
from .dataset_schema import DatasetSchema
from .errors import SchemaValidationError


MANIFEST_NAME = "labvla_manifest.json"
SUPPORTED_VERSION = 1


def load_manifest(path: str | Path) -> DatasetSchema:
    """Parse and validate a labvla_manifest.json file into a DatasetSchema.

    Raises ValueError on malformed manifests.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    version = data.get("version")
    if version != SUPPORTED_VERSION:
        raise ValueError(
            f"Unsupported manifest version {version!r} at {path} "
            f"(expected {SUPPORTED_VERSION})"
        )

    schema_id = data.get("schema_id")
    if not schema_id or not isinstance(schema_id, str):
        raise ValueError(f"manifest at {path} missing non-empty schema_id")

    robot_type = data.get("robot_type", "unknown")

    state = data.get("state") or {}
    action = data.get("action") or {}
    images = data.get("images") or {}
    allow_extra_cameras = bool(data.get("allow_extra_cameras", False))

    arm_layout = None
    if data.get("arm_layout") is not None:
        from .arm_layout import ArmLayoutSpec
        arm_layout = ArmLayoutSpec.from_dict(data["arm_layout"])

    # Optional auxiliary losses on annotation columns. Missing / empty list →
    # empty tuple → schema behaves exactly as before.
    annotation_losses: tuple = ()
    if data.get("annotation_losses"):
        from .annotation_loss import AnnotationLossSpec
        annotation_losses = tuple(
            AnnotationLossSpec.from_dict(d) for d in data["annotation_losses"]
        )

    state_keys = tuple(state.get("keys") or ())
    state_dims = tuple(state.get("dims") or ())
    action_keys = tuple(action.get("keys") or ())
    action_dims = tuple(action.get("dims") or ())
    delta_mask = tuple(bool(v) for v in (action.get("delta") or ()))
    gripper_action_dims = tuple(action.get("gripper_dims") or ())
    # Optional gripper physical semantic ("width" / "open_fraction" /
    # "position" / ...). Drives gripper canonicalization + cross-repo
    # compatibility guards in dataset_helpers. Absent in legacy manifests
    # → None → registry-based fallback keyed off schema_id still applies.
    gripper_semantic_raw = data.get("gripper_semantic")
    gripper_semantic = (
        str(gripper_semantic_raw) if gripper_semantic_raw else None
    )
    source_state = data.get("source_state") or {}
    source_action = data.get("source_action") or {}
    source_state_keys = tuple(source_state.get("keys") or ())
    source_state_dims = tuple(source_state.get("dims") or ())
    source_action_keys = tuple(source_action.get("keys") or ())
    source_action_dims = tuple(source_action.get("dims") or ())

    # Expand short image aliases ("image0") to the full unified key.
    expanded_images = expand_camera_mapping(images)

    # DatasetSchema's __post_init__ runs schema/validate.py::validate_schema,
    # which enforces every structural invariant. Wrap the construction so the
    # user sees the manifest path in the error message (instead of a terse
    # constructor trace) — otherwise authoring mistakes are hard to locate.
    try:
        return DatasetSchema(
            schema_id=schema_id,
            robot_type=robot_type,
            state_keys=state_keys,
            action_keys=action_keys,
            state_dims=state_dims,
            action_dims=action_dims,
            delta_mask=delta_mask,
            gripper_action_dims=gripper_action_dims,
            gripper_semantic=gripper_semantic,
            image_mapping=expanded_images,
            allow_extra_cameras=allow_extra_cameras,
            arm_layout=arm_layout,
            annotation_losses=annotation_losses,
            source_state_keys=source_state_keys,
            source_state_dims=source_state_dims,
            source_action_keys=source_action_keys,
            source_action_dims=source_action_dims,
            source="manifest",
            source_path=str(path),
        )
    except SchemaValidationError as e:
        raise SchemaValidationError(f"manifest {path}: {e}") from e
