"""Two-tier schema discovery chain.

Order: override > manifest > info.json names auto-infer > raise.

The previous third tier (RobotConfig registry under src/robots/) was removed:
dispatching on `robot_type` was fragile (same robot_type in different datasets
can have different state/action layouts). Every
dataset must now either (1) ship a meta/labvla_manifest.json, or (2) have
rich enough info.json `features[*].names` for the auto-inferrer to identify
gripper dims.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .dataset_schema import DatasetSchema, SchemaDiscoveryError
from .manifest import MANIFEST_NAME, load_manifest
from .infer import try_infer_from_info


logger = logging.getLogger(__name__)


# Annotation-loss fields that are *synthesized at runtime* by a transform
# (or a VQA adapter) rather than read directly from a parquet column, so they
# are legitimately absent from info.json ``features``:
#   - "annotation.unified"  — built by BuildUnifiedAnnotationTransformFn from
#                             the per-annotation.* columns.
#   - "annotation.subtask"  — copied from data["task"] by
#                             BuildAgiBotSubtaskTransformFn.
#   - "annotation.vqa_answer" — written into the sample dict by the VQA adapter.
# Any other annotation field is expected to correspond to a real data column;
# if it is absent from info.json features it is almost certainly a config typo.
_SYNTHESIZED_ANNOTATION_FIELDS = frozenset({
    "annotation.unified",
    "annotation.subtask",
    "annotation.vqa_answer",
})


def _load_info_json(root: Path) -> Optional[dict]:
    """Best-effort load of ``<root>/meta/info.json``. Returns None if missing or
    unreadable so callers can skip schema-vs-info cross-checks gracefully (the
    same posture as ``_warn_if_image_mapping_missing``)."""
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return None
    try:
        with open(info_path) as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        logger.debug("_load_info_json: could not read %s (%s)", info_path, e)
        return None


def _validate_annotation_fields(schema, info: dict, root: Path) -> None:
    """Fail loud on an annotation-loss field that references a non-existent column.

    T16: ``annotation_losses`` previously only had its type/duplicate checks in
    ``validate_schema``; a typo'd ``field`` (e.g. ``annotation.vqa_anwser``)
    sailed through, then tokenization read ``data.get(field, "")`` → empty
    string → all-zero mask → silent 0-CE, so the annotation supervision was
    effectively off while training "succeeded". Here, once info.json is
    available, we reject a field that is neither a known transform-synthesized
    field nor present in info.json ``features``.

    Note: the authoritative runtime guard (warn-once when a configured field is
    absent from the *materialized* sample dict) belongs in the tokenize
    transform; this is the schema-layer best effort that does not false-positive
    on the runtime-synthesized fields above.
    """
    losses = getattr(schema, "annotation_losses", ()) or ()
    if not losses:
        return
    features = info.get("features") or {}
    feature_keys = set(features.keys()) if isinstance(features, dict) else set()
    missing = []
    for spec in losses:
        field = getattr(spec, "field", None)
        if not field or field in _SYNTHESIZED_ANNOTATION_FIELDS:
            continue
        if field not in feature_keys:
            missing.append(field)
    if missing:
        raise SchemaDiscoveryError(
            f"[schema] schema_id={schema.schema_id!r} declares annotation_losses "
            f"on field(s) {missing} that are neither a known runtime-synthesized "
            f"annotation field {sorted(_SYNTHESIZED_ANNOTATION_FIELDS)} nor present "
            f"in info.json features at {root}. This is almost certainly a typo — "
            f"it would otherwise tokenize an empty string and train a silent "
            f"all-zero (0-CE) annotation loss. Available feature keys: "
            f"{sorted(feature_keys)}"
        )


def _warn_if_image_mapping_missing(schema, root: Path) -> None:
    """Raise SchemaDiscoveryError if manifest image keys don't match info.json.

    Forward direction (M-9 fix): any schema image key absent from info.json
    features is a hard error — the dataset doesn't carry what we promised.

    Reverse direction (M-5 add): any video feature in info.json that the
    schema doesn't map to an image slot is a hard error too. A dataset with
    3 cameras but a manifest that only declares 1 would silently drop 2
    modalities during training, producing a ckpt that at deploy time expects
    1 camera while the robot hardware sends 3 — classic silent skew.
    """
    try:
        import json as _json
        from .camera_mapping import expand_camera_source as _expand_src
        with open(root / "meta" / "info.json") as _f:
            info = _json.load(_f)
        features = info.get("features", {})
        # Apr-2026 fix: schema.image_mapping LHS is now uniformly the
        # canonical ``observation.images.<x>`` form (Fix 1 — both halves
        # of expand_camera_mapping). info.json features still use whatever
        # the source dataset declares (v3.0: prefixed; v2.1: often raw
        # ``camera_1_rgb``). Build a reverse lookup so the same physical
        # camera is recognized under either spelling.
        feature_canonical = {_expand_src(k): k for k in features.keys()}
        missing = []
        for src in schema.image_mapping:
            if src in features:
                continue
            if src in feature_canonical:
                continue
            missing.append(src)
        if missing:
            raise SchemaDiscoveryError(
                f"[schema] manifest {schema.source_path} references image keys "
                f"not present in info.json features: {missing}. Either the "
                f"manifest is stale or the info.json was regenerated — reconcile "
                f"them before training."
            )
        # Reverse check: find all image/video features in info.json and confirm
        # every one appears in schema.image_mapping. Only treat dtype=video or
        # keys with an obvious image prefix as camera features, so non-image
        # arrays (state, action, etc.) don't trip the check.
        # schema.allow_extra_cameras=True opts out — used by merged multi-robot
        # datasets (e.g. robocoin) where info.json declares the union of all
        # source cameras but the schema intentionally picks a 3-cam subset.
        if not getattr(schema, "allow_extra_cameras", False):
            declared = set(schema.image_mapping)
            extra_cams = []
            for key, entry in features.items():
                if not isinstance(entry, dict):
                    continue
                dtype = entry.get("dtype", "")
                # Apr-2026: v2.1 datasets set dtype="image" with raw key
                # like "camera_1_rgb"; v3.0 sets dtype="image" with the
                # prefixed "observation.images.camera_1_rgb". Both shapes
                # are valid camera features for the reverse check.
                is_camera = dtype == "video" or dtype == "image"
                if not is_camera:
                    continue
                # Compare in canonical-prefixed form so a feature key like
                # ``camera_1_rgb`` matches a schema entry
                # ``observation.images.camera_1_rgb`` and vice versa.
                canonical_key = _expand_src(key)
                if canonical_key not in declared and key not in declared:
                    extra_cams.append(key)
            if extra_cams:
                raise SchemaDiscoveryError(
                    f"[schema] dataset at {root} has camera features not declared "
                    f"in the schema's image_mapping: {extra_cams}. Declared cameras: "
                    f"{sorted(declared)!r}. Training would silently drop the extra "
                    f"modalities and produce a ckpt incompatible with the robot's "
                    f"actual camera count at deploy time. Either add the missing "
                    f"cameras to the manifest/schema, set "
                    f"allow_extra_cameras=True on the schema (for merged multi-"
                    f"robot datasets), or explicitly exclude them."
                )
    except SchemaDiscoveryError:
        raise
    except Exception as e:
        logger.debug("_warn_if_image_mapping_missing: skipped (%s)", e)


def discover_schema(
    root: str | Path,
    robot_type: Optional[str] = None,
    override: Optional[DatasetSchema] = None,
) -> DatasetSchema:
    """Resolve a DatasetSchema for a dataset on disk.

    Priority order:
      1. `override` argument (for CLI/config injection)
      2. `<root>/meta/labvla_manifest.json` (preferred, explicit)
      3. Auto-infer from `<root>/meta/info.json` `features[*].names`
      4. Raise SchemaDiscoveryError with guidance

    Args:
        root: dataset root directory (contains `meta/info.json`)
        robot_type: value of `info["robot_type"]` (kept for attribution/logging,
            not for dispatch)
        override: skip discovery, use this schema verbatim (debugging/tests)
    """
    root = Path(root)

    if override is not None:
        logger.info(
            "[schema] using override schema_id=%s source=%s",
            override.schema_id, override.source,
        )
        # Still validate image_mapping against info.json — same guard we
        # apply to Tier 1 manifests, so a stale Python schema referencing
        # non-existent camera keys fails loudly instead of KeyError'ing
        # deep inside a DataLoader worker.
        _warn_if_image_mapping_missing(override, root)
        # Reject annotation-loss fields that reference non-existent columns.
        _info = _load_info_json(root)
        if _info is not None:
            _validate_annotation_fields(override, _info, root)
        return override

    # Tier 1: explicit manifest
    manifest_path = root / "meta" / MANIFEST_NAME
    if manifest_path.exists():
        schema = load_manifest(manifest_path)
        logger.info(
            "[schema] loaded manifest schema_id=%s robot_type=%s from %s",
            schema.schema_id, schema.robot_type, manifest_path,
        )
        _warn_if_image_mapping_missing(schema, root)
        # Reject annotation-loss fields that reference non-existent columns.
        _info = _load_info_json(root)
        if _info is not None:
            _validate_annotation_fields(schema, _info, root)
        return schema

    # Parse info.json once for tier 2.
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise SchemaDiscoveryError(
            f"[schema] info.json missing at {info_path} — cannot discover schema "
            f"for dataset at {root}"
        )
    with open(info_path) as f:
        info = json.load(f)

    inferred_robot_type = robot_type or info.get("robot_type")

    # Tier 2: auto-infer from info.json names
    inferred = try_infer_from_info(info, inferred_robot_type, root)
    if inferred is not None:
        logger.info(
            "[schema] auto-inferred schema_id=%s robot_type=%s from info.json names",
            inferred.schema_id, inferred.robot_type,
        )
        return inferred

    raise SchemaDiscoveryError(
        f"[schema] could not discover schema for dataset at {root}. Tried:\n"
        f"  1. {manifest_path} — missing\n"
        f"  2. auto-infer from info.json names — insufficient "
        f"(features['action']/['state'].names lacked gripper tokens or were opaque)\n"
        f"To fix: create {manifest_path} describing this dataset's layout. "
        f"See doc/schema_manifest.md for the template."
    )
