"""Convert legacy meta/labvla_manifest.json files into Python schema modules.

Usage:
  python -m data_process migrate \\
    /all-flash-data/Pretrain_Data/robointer_droid \\
    /all-flash-data/lerobot/.cache/LabUtopia/Level3_TransportBeaker \\
    --out schemas/

For each dataset root it:
  1. Reads meta/labvla_manifest.json
  2. Validates it via schema.manifest.load_manifest()
  3. Detects the closest preset shape (franka8 / franka_split / aloha14 / raw)
  4. Writes an equivalent Python file to the output directory

The original JSON manifest is left on disk; schema.discovery still reads it as
Tier 1 fallback if --dataset_schema is not passed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# __file__ = <repo>/data_process/migrate/manifest_to_py.py
# need: <repo>/src  →  parent.parent.parent / "src"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from schema.annotation_loss import AnnotationLossSpec  # noqa: E402
from schema.arm_layout import ArmCount, ArmLayoutSpec  # noqa: E402
from schema.dataset_schema import DatasetSchema  # noqa: E402
from schema.manifest import MANIFEST_NAME, load_manifest  # noqa: E402


def _reverse_image_mapping(schema: DatasetSchema) -> list[str]:
    """Extract camera keys in image0, image1, ... order."""
    by_target = {v: k for k, v in schema.image_mapping.items()}
    cams: list[str] = []
    i = 0
    while True:
        target = f"observation.images.image{i}"
        if target not in by_target:
            break
        cams.append(by_target[target])
        i += 1
    return cams


def _format_arm_layout(layout: ArmLayoutSpec | None) -> str:
    """Render an ArmLayoutSpec as a Python source-code expression.

    Always emits a value (``None`` when the schema has no canonical layout) so
    the generated module is unambiguous about which fields the manifest
    declared. ``ArmCount`` is referenced by qualified-name to match the
    in-repo schema authoring style (see schemas/robointer_droid.py).
    """
    if layout is None:
        return "None"
    if layout.arm_count == ArmCount.SINGLE:
        return (
            "ArmLayoutSpec(\n"
            "        arm_count=ArmCount.SINGLE,\n"
            f"        arm_dof={layout.arm_dof!r},\n"
            f"        gripper_index_in_raw={layout.gripper_index_in_raw!r},\n"
            f"        gripper_binarize_threshold={float(layout.gripper_binarize_threshold)!r},\n"
            "    )"
        )
    # DUAL
    return (
        "ArmLayoutSpec(\n"
        "        arm_count=ArmCount.DUAL,\n"
        f"        left_arm_dof={layout.left_arm_dof!r},\n"
        f"        right_arm_dof={layout.right_arm_dof!r},\n"
        f"        left_gripper_index_in_raw={layout.left_gripper_index_in_raw!r},\n"
        f"        right_gripper_index_in_raw={layout.right_gripper_index_in_raw!r},\n"
        f"        gripper_binarize_threshold={float(layout.gripper_binarize_threshold)!r},\n"
        "    )"
    )


def _format_annotation_losses(
    losses: tuple[AnnotationLossSpec, ...],
) -> str:
    """Render a tuple of AnnotationLossSpec as a Python source-code expression.

    Empty tuple → ``"()"`` so the generated file matches schemas/robointer_droid.py
    which explicitly sets ``annotation_losses=()`` to disable auxiliary CE.
    """
    if not losses:
        return "()"
    parts: list[str] = []
    for spec in losses:
        parts.append(
            "        AnnotationLossSpec(\n"
            f"            field={spec.field!r},\n"
            f"            loss_type={spec.loss_type!r},\n"
            f"            weight={float(spec.weight)!r},\n"
            f"            max_length={int(spec.max_length)!r},\n"
            "        ),"
        )
    return "(\n" + "\n".join(parts) + "\n    )"


def _has_extended_fields(schema: DatasetSchema) -> bool:
    """True if any manifest-declared field carries a non-default value that the
    franka8 / franka_split presets cannot express. Such schemas must round-trip
    through the raw ``DatasetSchema(...)`` branch so no manifest-declared field
    is silently dropped.

    ``gripper_semantic`` and the ``source_state_*`` / ``source_action_*``
    canonicalization columns are also load_manifest()-parsed but unsupported by
    the preset factories, so a schema declaring any of them MUST fall through to
    the raw branch, else migration silently downgrades the schema semantics
    (gripper compatibility guard, raw→canonical state/action remap).
    """
    return (
        schema.arm_layout is not None
        or bool(schema.allow_extra_cameras)
        or bool(schema.annotation_losses)
        or schema.gripper_semantic is not None
        or bool(schema.source_state_keys)
        or bool(schema.source_state_dims)
        or bool(schema.source_action_keys)
        or bool(schema.source_action_dims)
    )


def _detect_preset(schema: DatasetSchema) -> tuple[str, str]:
    """Return (preset_name, generated_body_code)."""
    # The franka8 / franka_split factories can't express non-default arm_layout
    # / allow_extra_cameras / annotation_losses, so such schemas fall through to
    # the raw DatasetSchema(...) branch to preserve every manifest field.
    extended = _has_extended_fields(schema)

    # 8-dim single action, one gripper at a single index → franka8
    if (
        not extended
        and len(schema.action_keys) == 1
        and schema.action_dims == (8,)
        and len(schema.gripper_action_dims) == 1
    ):
        cams = _reverse_image_mapping(schema)
        if cams:
            gripper = schema.gripper_action_dims[0]
            gripper_arg = f", gripper_dim={gripper}" if gripper != 7 else ""
            state_arg = (f", state_key={schema.state_keys[0]!r}"
                         if schema.state_keys[0] != "observation.state" else "")
            action_arg = (f", action_key={schema.action_keys[0]!r}"
                          if schema.action_keys[0] != "action" else "")
            robot_arg = (f", robot_type={schema.robot_type!r}"
                         if schema.robot_type != "franka" else "")
            body = (
                f"from schema.presets import franka8\n\n"
                f"SCHEMA = franka8(\n"
                f"    {schema.schema_id!r},\n"
                f"    cams={tuple(cams)!r}"
                f"{robot_arg}{state_arg}{action_arg}{gripper_arg},\n"
                f")\n"
            )
            return "franka8", body

    # Multi-key action (robointer-style split) → franka_split
    if (
        not extended
        and len(schema.action_keys) >= 2
        and len(schema.gripper_action_dims) == 1
    ):
        body = (
            "from schema.presets import franka_split\n\n"
            "SCHEMA = franka_split(\n"
            f"    schema_id={schema.schema_id!r},\n"
            f"    robot_type={schema.robot_type!r},\n"
            f"    state_keys={tuple(schema.state_keys)!r},\n"
            f"    state_dims={tuple(schema.state_dims)!r},\n"
            f"    action_keys={tuple(schema.action_keys)!r},\n"
            f"    action_dims={tuple(schema.action_dims)!r},\n"
            f"    gripper_action_dim={schema.gripper_action_dims[0]},\n"
            f"    images={dict(schema.image_mapping)!r},\n"
            ")\n"
        )
        return "franka_split", body

    # Fallback: emit a raw DatasetSchema(...) call. Every field (arm_layout,
    # allow_extra_cameras, annotation_losses, gripper_semantic, source_state_*,
    # source_action_*) is always rendered — None/False/() for the trivial case —
    # so the migrated .py is a faithful round-trip of the source manifest.
    arm_layout_code = _format_arm_layout(schema.arm_layout)
    annotation_losses_code = _format_annotation_losses(schema.annotation_losses)
    body = (
        "from schema import DatasetSchema\n"
        "from schema.annotation_loss import AnnotationLossSpec\n"
        "from schema.arm_layout import ArmLayoutSpec, ArmCount\n\n"
        "SCHEMA = DatasetSchema(\n"
        f"    schema_id={schema.schema_id!r},\n"
        f"    robot_type={schema.robot_type!r},\n"
        f"    state_keys={tuple(schema.state_keys)!r},\n"
        f"    action_keys={tuple(schema.action_keys)!r},\n"
        f"    state_dims={tuple(schema.state_dims)!r},\n"
        f"    action_dims={tuple(schema.action_dims)!r},\n"
        f"    delta_mask={tuple(schema.delta_mask)!r},\n"
        f"    gripper_action_dims={tuple(schema.gripper_action_dims)!r},\n"
        f"    image_mapping={dict(schema.image_mapping)!r},\n"
        f"    allow_extra_cameras={bool(schema.allow_extra_cameras)!r},\n"
        f"    arm_layout={arm_layout_code},\n"
        f"    annotation_losses={annotation_losses_code},\n"
        f"    gripper_semantic={schema.gripper_semantic!r},\n"
        f"    source_state_keys={tuple(schema.source_state_keys)!r},\n"
        f"    source_state_dims={tuple(schema.source_state_dims)!r},\n"
        f"    source_action_keys={tuple(schema.source_action_keys)!r},\n"
        f"    source_action_dims={tuple(schema.source_action_dims)!r},\n"
        '    source="manifest",\n'
        "    source_path=__file__,\n"
        ")\n"
    )
    return "raw", body


def _py_filename(schema_id: str) -> str:
    # schema_id → snake_case filename, drop trailing version tag
    stem = schema_id.lower().replace(" ", "_")
    if stem.endswith("_v1") or stem.endswith("_v2") or stem.endswith("_v3"):
        stem = stem.rsplit("_", 1)[0]
    return f"{stem}.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", type=Path,
                    help="Dataset root dirs (each must contain meta/labvla_manifest.json)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory for generated .py files")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    for root in args.roots:
        manifest_path = root / "meta" / MANIFEST_NAME
        if not manifest_path.exists():
            print(f"[skip] no manifest at {manifest_path}", file=sys.stderr)
            continue
        schema = load_manifest(manifest_path)
        preset, body = _detect_preset(schema)
        out_path = args.out / _py_filename(schema.schema_id)
        header = (
            f'"""Auto-migrated from {manifest_path}.\n'
            f"schema_id: {schema.schema_id}\n"
            f"preset   : {preset}\n"
            '"""\n'
        )
        out_path.write_text(header + body)
        print(f"[ok] {manifest_path} -> {out_path} (preset={preset})")


if __name__ == "__main__":
    main()
