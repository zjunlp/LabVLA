"""Schema exception hierarchy — single source of truth.

All schema-layer errors derive from `SchemaError` so callers can catch the
family with one `except` when that's desired. The leaf classes additionally
inherit from the semantically appropriate built-in (`ValueError` /
`RuntimeError`) so pre-existing `except ValueError` / `except RuntimeError`
code paths in the codebase keep working — this is backward-compat by design.

Historical note
---------------
Before Plan-B unification:
  - `SchemaDiscoveryError` lived in `schema/dataset_schema.py` as a
    `RuntimeError` subclass.
  - `SchemaSpecError` lived in `schema/registry.py` as a `ValueError`
    subclass.
Both are now re-exported from those modules for backward compat, but the
canonical definition is here.
"""

from __future__ import annotations


class SchemaError(Exception):
    """Base class for every schema-layer exception.

    Catch this when you want to handle any schema failure uniformly (bad
    manifest, missing info.json, unresolved --dataset_schema spec, invalid
    layout, etc.). For finer control, catch one of the leaves below.
    """


class SchemaValidationError(SchemaError, ValueError):
    """Raised when a DatasetSchema / ArmLayoutSpec fails structural checks.

    Structural checks include: dim/length mismatches, gripper index out of
    bounds, gripper marked delta, unsupported arm DoF, duplicate annotation
    fields, etc. Anything caught by `validate_schema` / `validate_arm_layout`.

    Multiple inheritance with `ValueError` is intentional — existing callers
    that do `except ValueError` (a few places in adapters/config) continue to
    work.
    """


class SchemaDiscoveryError(SchemaError, RuntimeError):
    """Raised when schema discovery fails (manifest missing, info.json
    opaque, image_mapping mismatched, etc.).

    Multiple inheritance with `RuntimeError` preserves the pre-Plan-B
    contract — `schema/dataset_schema.py` previously defined this as a
    `RuntimeError` subclass, and adapters catch it explicitly.
    """


class SchemaSpecError(SchemaError, ValueError):
    """Raised by `schema.registry.resolve(spec)` when `spec` cannot be
    mapped to any registered schema / dotted module / file path.

    Multiple inheritance with `ValueError` preserves the pre-Plan-B
    contract — `schema/registry.py` previously defined this as a
    `ValueError` subclass.
    """


__all__ = [
    "SchemaError",
    "SchemaValidationError",
    "SchemaDiscoveryError",
    "SchemaSpecError",
]
