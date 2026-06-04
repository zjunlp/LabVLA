"""Camera-mapping SSOT — expand short alias → full `observation.images.imageN`.

Before Plan-B, the same "alias expansion" logic was reimplemented three times
(blueprint compiler, manifest parser, Tier-2 info.json auto-inferrer) and the
info.json reverse-check/validation was spread across a fourth file
(discovery.py). Subtle drift between copies caused silent key-mismatch bugs
at adapter time. This module is the single source of truth.

Conventions
-----------
A camera "target" is a string. It is either:
  - A short numeric-slot alias like ``"image0"`` / ``"image1"``. Expanded to
    ``"{OBS_IMAGES}.{alias}"`` here.
  - Already a fully qualified slot key like ``"observation.images.image0"``.
    Kept as-is.

C34 note: the only valid target slot form is ``imageN`` (N a non-negative
integer). Non-numeric aliases such as ``"image_left"`` are NOT accepted —
the schema validator (``schema/validate.py::_IMAGE_TARGET_RE``) only admits
``observation.images.imageN``, and the downstream Qwen3VL processor iterates a
fixed set of numbered slots. ``expand_camera_target`` rejects non-``imageN``
aliases at authoring time so the camera-mapping contract matches the validator
instead of failing later inside ``DatasetSchema.__post_init__``.

A camera "mapping" is ``{raw_camera_key: target}`` where raw_camera_key is
whatever the source dataset's parquet / info.json calls the camera (e.g.
``"camera_1_rgb"``, ``"observation.images.cam_head_rgb"``).

Tier-2 discovery
----------------
``infer_image_mapping_from_info`` produces a default mapping from an
info.json ``features`` block when no manifest / explicit schema exists.
It uses a semantic ordering (head/top/front cameras first, then
hand/wrist, then alphabetical) so ``image0`` consistently matches the
third-person/primary view that most pretrained policies expect.
"""

from __future__ import annotations

import re
from typing import Mapping

from utils.constants import OBS_IMAGES


# C34: a target slot alias must be ``imageN`` (N a non-negative integer). This
# mirrors ``schema/validate.py::_IMAGE_TARGET_RE`` (which matches the fully
# qualified ``observation.images.imageN``) so the authoring contract and the
# validator agree.
_IMAGE_SLOT_ALIAS_RE = re.compile(r"^image(\d+)$")


# Tokens that mark a camera as the primary / third-person / scene view.
# Order within a tuple breaks ties between multiple matches.
_PRIMARY_TOKENS = ("head", "top", "front", "primary", "base", "exterior")

# Tokens that mark a camera as arm-mounted / wrist / ego view — placed
# after primary cameras so ``image0`` is almost always the scene cam.
_SECONDARY_TOKENS = ("hand", "wrist", "effector", "ego")


def expand_camera_target(raw: str, target: str) -> str:
    """Expand a possibly short camera target to the fully qualified form.

    Args:
        raw: The source camera key (used only in error messages).
        target: Either a short numeric-slot alias (``"image0"``) or a full
            ``observation.images.image0`` key. Non-string / empty values
            raise ``ValueError``. Non-``imageN`` short aliases (e.g.
            ``"image_left"``) are rejected (C34).

    Returns:
        The fully qualified ``observation.images.imageN`` key.
    """
    if not isinstance(target, str) or not target:
        raise ValueError(
            f"camera target for {raw!r} must be a non-empty string, got {target!r}"
        )
    if target.startswith(OBS_IMAGES + "."):
        # Already fully qualified — pass through verbatim. The schema validator
        # still enforces the ``imageN`` slot form on it.
        return target
    # C34 fix: a short alias must be ``imageN``. Previously any alias was
    # expanded (e.g. ``image_left`` → ``observation.images.image_left``), which
    # the validator then rejected deep inside ``DatasetSchema.__post_init__``.
    # Reject it here, at authoring time, with an actionable message.
    if _IMAGE_SLOT_ALIAS_RE.match(target) is None:
        raise ValueError(
            f"camera target for {raw!r} must be a numbered slot alias "
            f"'imageN' (e.g. 'image0') or a fully qualified "
            f"'{OBS_IMAGES}.imageN' key, got {target!r}. Non-numeric aliases "
            f"like 'image_left' are not supported — the VLM processor consumes "
            f"a fixed set of numbered camera slots."
        )
    return f"{OBS_IMAGES}.{target}"


# The legacy single-image-prefix some older datasets use. Treat as
# already-qualified to keep backward compatibility with manifests that
# happen to declare e.g. ``observation.image.head`` instead of
# ``observation.images.head``.
_OBS_IMAGE_LEGACY = "observation.image"


def expand_camera_source(raw: str) -> str:
    """Expand a possibly short camera SOURCE key to its fully qualified form.

    Mirror of :func:`expand_camera_target` for source keys (the LHS of
    ``schema.image_mapping``). Many schemas were authored with just the
    parquet column name (``"camera_1_rgb"``) on the LHS, then expanded the
    target on the RHS only. Adapters iterate ``image_mapping.keys()`` to
    decide which cameras to read, so a partial expansion left source keys
    in two different shapes downstream and led to the
    ``RemapImageKeyTransformFn`` silently popping a key that the adapter
    never wrote (the all-zero training-vs-inference bug).

    Behavior:
      - Already-prefixed (``observation.images.<x>`` or
        ``observation.image.<x>``) → returned verbatim.
      - Otherwise → ``f"{OBS_IMAGES}.{raw}"``.

    Idempotent on already-prefixed keys, so existing schemas authored
    with full keys (oxe_auge ``observation.images.image``,
    robointer_droid ``observation.images.primary``) continue to work
    unchanged.
    """
    if not isinstance(raw, str) or not raw:
        raise ValueError(
            f"camera source key must be a non-empty string, got {raw!r}"
        )
    if raw.startswith(OBS_IMAGES + ".") or raw.startswith(_OBS_IMAGE_LEGACY + "."):
        return raw
    return f"{OBS_IMAGES}.{raw}"


def expand_camera_mapping(mapping: Mapping[str, str]) -> dict[str, str]:
    """Expand every key AND value of a raw→target camera mapping.

    Short aliases are expanded via ``expand_camera_source`` (LHS) and
    ``expand_camera_target`` (RHS); already fully qualified keys/values
    are preserved verbatim. The returned dict is a fresh copy — callers
    may mutate freely.

    Note: prior to the Apr-2026 fix, only the value was expanded. Source
    keys for legacy schemas (e.g. labutopia_level3 with
    ``cameras={"camera_1_rgb": "image0"}``) stayed unprefixed, leaving
    adapters and downstream transforms iterating two different key shapes.
    Both sides are now normalized to the unified ``observation.images.<x>``
    form so adapter writes and ``RemapImageKeyTransformFn`` reads agree.
    """
    return {
        expand_camera_source(raw): expand_camera_target(raw, target)
        for raw, target in mapping.items()
    }


def validate_camera_mapping(
    mapping: Mapping[str, str],
    num_slots: int,
) -> None:
    """Raise ``ValueError`` if the mapping declares more cameras than slots.

    Most policies are pretrained with a fixed number of camera slots
    (typical ``image0`` / ``image1`` / ``image2``). Declaring more source
    cameras than the policy can consume is a user error, and usually means
    the schema author forgot to consolidate or drop a camera.

    Args:
        mapping: raw → fully-qualified-target.
        num_slots: max number of distinct camera slots the policy accepts.
    """
    if num_slots < 0:
        raise ValueError(f"num_slots must be >= 0, got {num_slots}")
    if len(mapping) > num_slots:
        raise ValueError(
            f"camera mapping declares {len(mapping)} cameras but the policy "
            f"only has {num_slots} slot(s). Mapping keys: {sorted(mapping)!r}"
        )


def _bucket_for_key(key: str) -> tuple[int, int]:
    """Return a (major, minor) sort key — lower buckets come first."""
    low = key.lower()
    for i, tok in enumerate(_PRIMARY_TOKENS):
        if tok in low:
            return (0, i)
    for i, tok in enumerate(_SECONDARY_TOKENS):
        if tok in low:
            return (1, i)
    return (2, 0)


def infer_image_mapping_from_info(features: dict) -> dict[str, str]:
    """Tier-2 discovery: derive a camera mapping from an info.json features block.

    Scans ``features`` for entries whose ``dtype`` is ``"video"`` or
    ``"image"`` and produces an ordered mapping to ``image0``,
    ``image1``, ... using semantic priority: head/top/front first,
    then hand/wrist, then alphabetical among the remainder.

    Args:
        features: the ``info["features"]`` dict from LeRobot v2.1/v3.0.

    Returns:
        ``{raw_camera_key: "observation.images.imageN"}`` — may be empty
        if ``features`` declares no image/video entries.
    """
    visual = [
        k for k, v in features.items()
        if isinstance(v, dict) and v.get("dtype") in ("video", "image")
    ]
    ordered = sorted(visual, key=lambda k: (_bucket_for_key(k), k))
    return {k: f"{OBS_IMAGES}.image{i}" for i, k in enumerate(ordered)}


__all__ = [
    "expand_camera_target",
    "expand_camera_mapping",
    "validate_camera_mapping",
    "infer_image_mapping_from_info",
]
