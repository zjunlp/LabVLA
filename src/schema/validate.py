"""Schema-layer validators — single source of truth.

Before Plan-B, validation logic was duplicated inside
``DatasetSchema.__post_init__``, ``ArmLayoutSpec.__post_init__``, and the
manifest parser. This module extracts every structural check into two
functions so callers (constructors, manifest loaders, future CLI verifiers)
share one code path.

Both validators raise ``SchemaValidationError`` on failure. That class
inherits from ``ValueError`` for backward compatibility with code that
does ``except ValueError``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

from .annotation_loss import AnnotationLossSpec
from .arm_layout import ArmCount, ArmLayoutSpec
from .errors import SchemaValidationError
from utils.constants import NUM_IMAGE_SLOTS

if TYPE_CHECKING:  # avoid import cycle — DatasetSchema imports validate_schema
    from .dataset_schema import DatasetSchema


# Hard upper bound on the number of `observation.images.imageN` slots the VLM
# processor consumes. Authored in lockstep with
# ``RemapImageKeyTransformFn.num_image_slots`` — bump both together if the
# processor is extended to N slots.
# An image-mapping RHS must look like ``observation.images.imageN`` where N is a
# non-negative integer. Validated against ``_NUM_IMAGE_SLOTS`` range below; this
# regex only enforces shape.
_IMAGE_TARGET_RE = re.compile(r"^observation\.images\.image(\d+)$")


def validate_arm_layout(layout: ArmLayoutSpec) -> None:
    """Structural validation for an ``ArmLayoutSpec``.

    Checks:
      - SINGLE → ``arm_dof`` + ``gripper_index_in_raw`` set, ``arm_dof ∈ {6, 7}``.
      - DUAL → all four left/right fields set, each ``arm_dof ∈ {5, 6, 7}``.
      - Otherwise → unknown arm_count.

    For DUAL, 7-DoF raw arms are accepted as inputs to the 14-dim canonical
    layout: the canonical mapper keeps the first six joints per side and places
    grippers at dims 6 and 13.
    """
    if layout.arm_count == ArmCount.SINGLE:
        if layout.arm_dof is None or layout.gripper_index_in_raw is None:
            raise SchemaValidationError(
                "ArmLayoutSpec(SINGLE) requires arm_dof and gripper_index_in_raw"
            )
        if layout.arm_dof not in (6, 7):
            raise SchemaValidationError(
                f"arm_dof must be 6 or 7 for single-arm canonical 8-dim, "
                f"got {layout.arm_dof}"
            )
    elif layout.arm_count == ArmCount.DUAL:
        if (layout.left_arm_dof is None or layout.right_arm_dof is None
                or layout.left_gripper_index_in_raw is None
                or layout.right_gripper_index_in_raw is None):
            raise SchemaValidationError(
                "ArmLayoutSpec(DUAL) requires left_/right_arm_dof and "
                "left_/right_gripper_index_in_raw"
            )
        if layout.left_arm_dof not in (5, 6, 7):
            raise SchemaValidationError(
                f"left_arm_dof must be 5, 6, or 7 for dual-arm canonical 14-dim, "
                f"got {layout.left_arm_dof}"
            )
        if layout.right_arm_dof not in (5, 6, 7):
            raise SchemaValidationError(
                f"right_arm_dof must be 5, 6, or 7 for dual-arm canonical 14-dim, "
                f"got {layout.right_arm_dof}"
            )
    else:
        raise SchemaValidationError(f"unknown arm_count {layout.arm_count!r}")

    # gripper_binarize_threshold must be a finite float. The threshold
    # is applied in q01/q99-normalized [-1, 1] action space at deploy time, so
    # any value slightly outside [-1, 1] still has a mathematical meaning
    # (degenerate "always open" / "always closed" policy) — but NaN/inf or a
    # string leaked through JSON loading would produce silent all-zero / all-
    # one gripper behavior. Catch that here.
    threshold = layout.gripper_binarize_threshold
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise SchemaValidationError(
            f"gripper_binarize_threshold must be a real number, "
            f"got {type(threshold).__name__}={threshold!r}"
        )
    threshold_f = float(threshold)
    # Reject NaN / inf explicitly.
    if threshold_f != threshold_f or threshold_f in (float("inf"), float("-inf")):
        raise SchemaValidationError(
            f"gripper_binarize_threshold must be finite, got {threshold_f!r}"
        )


def validate_schema(schema: "DatasetSchema", context: Optional[str] = None) -> None:
    """Structural validation for a fully-constructed ``DatasetSchema``.

    Enforces every invariant the rest of the codebase relies on:
      - non-empty ``schema_id`` / ``image_mapping``
      - parallel ``(keys, dims)`` arrays
      - ``sum(action_dims) == len(delta_mask)``
      - every ``gripper_action_dims`` index is in-range and absolute
      - ``source in {"manifest", "info_names"}``
      - no duplicate annotation-loss ``field``s

    Args:
        schema: the DatasetSchema under validation.
        context: optional prefix for error messages (e.g. ``"manifest foo.json"``)
            — helps pinpoint the failing config when the same validator runs
            for both the constructor and an external file loader.
    """
    prefix = f"{context}: " if context else ""
    sid = schema.schema_id

    if not sid or not isinstance(sid, str):
        raise SchemaValidationError(
            f"{prefix}schema_id must be a non-empty string, got {sid!r}"
        )
    if not schema.image_mapping:
        raise SchemaValidationError(
            f"{prefix}schema_id={sid!r}: image_mapping must have at least one "
            f"camera entry."
        )

    # image_mapping target slots must be unique. A duplicate RHS means
    # two source cameras would silently overwrite each other when
    # ``RemapImageKeyTransformFn`` pops the source key into the same target —
    # the second write wins and the first camera's frames disappear without
    # error. Catch it loud here.
    if len(set(schema.image_mapping.values())) != len(schema.image_mapping):
        seen: dict[str, str] = {}
        for src, tgt in schema.image_mapping.items():
            if tgt in seen:
                raise SchemaValidationError(
                    f"{prefix}schema_id={sid!r}: image_mapping has duplicate "
                    f"target slot {tgt!r} — sources {seen[tgt]!r} and {src!r} "
                    f"both map to it. Each camera must claim a unique "
                    f"observation.images.imageN slot."
                )
            seen[tgt] = src

    # image_mapping target slots must look like
    # ``observation.images.imageN`` where N ∈ [0, num_image_slots-1]. Anything
    # else would either crash the downstream Qwen3VL processor (which iterates
    # exactly ``num_image_slots`` named slots) or produce unreachable cameras
    # (slot index never read).
    declared_slots: set[int] = set()
    for src, tgt in schema.image_mapping.items():
        m = _IMAGE_TARGET_RE.match(tgt)
        if m is None:
            raise SchemaValidationError(
                f"{prefix}schema_id={sid!r}: image_mapping target {tgt!r} "
                f"(source {src!r}) must match "
                f"'observation.images.imageN' for N ∈ "
                f"[0, {NUM_IMAGE_SLOTS})."
            )
        idx = int(m.group(1))
        if not (0 <= idx < NUM_IMAGE_SLOTS):
            raise SchemaValidationError(
                f"{prefix}schema_id={sid!r}: image_mapping target {tgt!r} "
                f"(source {src!r}) declares slot {idx} but only "
                f"[0, {NUM_IMAGE_SLOTS}) slots are supported by the VLM "
                f"processor."
            )
        declared_slots.add(idx)

    # Claim 11 fix: image_mapping must cover the slots {0, 1, ..., k-1} for
    # some k ≥ 1 — i.e. start at image0 and have no holes. The transform
    # pipeline (RemapImageKeyTransformFn at src/transforms/core.py:~290 and
    # transform_labvla.py:~81) hard-references `observation.images.image0`
    # as the zero-frame template for padded slots, and iterates
    # `range(num_image_slots)` reading `<key>_mask` for each i. A schema
    # that declares only `image1` would pass the per-target checks above
    # but crash with `KeyError: observation.images.image0` at transform
    # time. Reject it here so the contract between schema validation and
    # the transform pipeline is end-to-end.
    if declared_slots:
        max_slot = max(declared_slots)
        expected = set(range(max_slot + 1))
        if declared_slots != expected:
            missing = sorted(expected - declared_slots)
            raise SchemaValidationError(
                f"{prefix}schema_id={sid!r}: image_mapping must cover slots "
                f"observation.images.image0..image{max_slot} contiguously; "
                f"declared slots are {sorted(declared_slots)} but slot(s) "
                f"{missing} are missing. The transform pipeline references "
                f"image0 as the zero-frame pad template — start at image0."
            )

    if len(schema.state_keys) != len(schema.state_dims):
        raise SchemaValidationError(
            f"{prefix}state_keys length {len(schema.state_keys)} != state_dims "
            f"length {len(schema.state_dims)} for schema_id={sid!r}"
        )
    if len(schema.action_keys) != len(schema.action_dims):
        raise SchemaValidationError(
            f"{prefix}action_keys length {len(schema.action_keys)} != "
            f"action_dims length {len(schema.action_dims)} for schema_id={sid!r}"
        )
    # Canonical state/action dims must be strictly positive.
    # Previously only source_state_dims/source_action_dims were checked; a
    # schema with action_dims=(0,) and delta_mask=() could pass validation
    # (since 0==0 in the delta_mask length check below) and silently produce a
    # policy with zero-width action output / no action supervision.
    for label, dims in (("state", schema.state_dims), ("action", schema.action_dims)):
        if not dims:
            raise SchemaValidationError(
                f"{prefix}{label}_dims must be non-empty for schema_id={sid!r}"
            )
        if any(int(d) <= 0 for d in dims):
            raise SchemaValidationError(
                f"{prefix}{label}_dims must all be positive for schema_id={sid!r}, "
                f"got {tuple(dims)}"
            )
    total_action_dim = sum(schema.action_dims)
    if total_action_dim <= 0:
        raise SchemaValidationError(
            f"{prefix}sum(action_dims) must be > 0 for schema_id={sid!r}, "
            f"got {total_action_dim}"
        )
    if sum(schema.state_dims) <= 0:
        raise SchemaValidationError(
            f"{prefix}sum(state_dims) must be > 0 for schema_id={sid!r}, "
            f"got {sum(schema.state_dims)}"
        )
    if len(schema.delta_mask) != total_action_dim:
        raise SchemaValidationError(
            f"{prefix}delta_mask length {len(schema.delta_mask)} != "
            f"sum(action_dims) {total_action_dim} for schema_id={sid!r}"
        )
    for label, keys, dims in (
        ("source_state", schema.source_state_keys, schema.source_state_dims),
        ("source_action", schema.source_action_keys, schema.source_action_dims),
    ):
        if bool(keys) != bool(dims):
            raise SchemaValidationError(
                f"{prefix}{label}_keys and {label}_dims must either both be "
                f"empty or both be populated for schema_id={sid!r}"
            )
        if keys and len(keys) != len(dims):
            raise SchemaValidationError(
                f"{prefix}{label}_keys length {len(keys)} != {label}_dims "
                f"length {len(dims)} for schema_id={sid!r}"
            )
        if any(int(d) <= 0 for d in dims):
            raise SchemaValidationError(
                f"{prefix}{label}_dims must be positive for schema_id={sid!r}, "
                f"got {dims}"
            )
    for idx in schema.gripper_action_dims:
        if not (0 <= idx < total_action_dim):
            raise SchemaValidationError(
                f"{prefix}gripper_action_dims entry {idx} outside "
                f"[0, {total_action_dim}) for schema_id={sid!r}"
            )
        if schema.delta_mask[idx]:
            raise SchemaValidationError(
                f"{prefix}gripper_action_dims entry {idx} marked delta in "
                f"delta_mask for schema_id={sid!r} (gripper must be absolute)"
            )
    if schema.source not in ("manifest", "info_names"):
        raise SchemaValidationError(
            f"{prefix}source must be one of manifest|info_names, got "
            f"{schema.source!r}"
        )

    seen_fields: set[str] = set()
    for spec in schema.annotation_losses:
        if not isinstance(spec, AnnotationLossSpec):
            raise TypeError(
                f"{prefix}annotation_losses entries must be AnnotationLossSpec, "
                f"got {type(spec).__name__} for schema_id={sid!r}"
            )
        if spec.field in seen_fields:
            raise SchemaValidationError(
                f"{prefix}annotation_losses has duplicate field {spec.field!r} "
                f"for schema_id={sid!r}"
            )
        seen_fields.add(spec.field)

    # M-A-4: gripper_semantic, when provided, must be one of the recognized
    # values. Typos like "veloctiy" would silently disable the cross-dataset
    # semantic guard otherwise.
    if schema.gripper_semantic is not None:
        allowed = ("velocity", "width", "position", "open_fraction", "binary")
        if schema.gripper_semantic not in allowed:
            raise SchemaValidationError(
                f"{prefix}gripper_semantic must be one of {allowed} or None, "
                f"got {schema.gripper_semantic!r} for schema_id={sid!r}"
            )

    # Cross-check gripper_action_dims against arm_layout.
    # The coordinate system depends on whether canonicalization is active:
    # - If source_action_keys is set, the schema's action_keys are canonical,
    #   so gripper_action_dims should match gripper_indices_canonical.
    # - If source_action_keys is NOT set, action_keys are raw coordinates,
    #   so gripper_action_dims should include gripper_index_in_raw.
    if schema.arm_layout is not None and schema.gripper_action_dims:
        has_canonicalization = bool(
            getattr(schema, "source_action_keys", None)
        )
        declared_gripper = set(schema.gripper_action_dims)
        if has_canonicalization:
            canonical_gripper = set(schema.arm_layout.gripper_indices_canonical)
            if declared_gripper != canonical_gripper:
                raise SchemaValidationError(
                    f"{prefix}gripper_action_dims {declared_gripper} does not match "
                    f"arm_layout.gripper_indices_canonical {canonical_gripper} "
                    f"(schema uses canonicalization via source_action_keys) "
                    f"for schema_id={sid!r}"
                )
        elif schema.arm_layout.arm_count == ArmCount.DUAL:
            # C26 fix: a DUAL arm_layout has no single-arm gripper_index_in_raw
            # (it is None). Reading it here used to make a dual-arm schema that
            # carries a legitimate ArmLayoutSpec(DUAL, ...) but no
            # source_action_keys fail the check unconditionally. A dual-arm
            # schema with no canonicalization already exposes its action vector
            # in canonical order, so its gripper dims must match the canonical
            # dual-arm gripper indices (6, 13).
            canonical_gripper = set(schema.arm_layout.gripper_indices_canonical)
            if declared_gripper != canonical_gripper:
                raise SchemaValidationError(
                    f"{prefix}gripper_action_dims {declared_gripper} does not match "
                    f"arm_layout.gripper_indices_canonical {canonical_gripper} "
                    f"(dual-arm schema, no canonicalization) for schema_id={sid!r}"
                )
        else:
            raw_gripper = schema.arm_layout.gripper_index_in_raw
            if raw_gripper not in declared_gripper:
                raise SchemaValidationError(
                    f"{prefix}gripper_action_dims {declared_gripper} does not include "
                    f"arm_layout.gripper_index_in_raw={raw_gripper} "
                    f"(schema has no canonicalization) for schema_id={sid!r}"
                )


__all__ = [
    "validate_arm_layout",
    "validate_schema",
]
