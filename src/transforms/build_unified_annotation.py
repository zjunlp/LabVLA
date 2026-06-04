"""Pack the 12 RoboInter `annotation.<field>` parquet columns into a single
`annotation.unified` text field, ready for tokenization by
`AnnotationTokenizeTransformFn`.

Why one unified field instead of 12 per-field CE heads?
- π0.5 Eq (1) computes ONE CE over all output text tokens (paper App B.1-B.2).
- OpenTau modeling_pi05.py:1333 sums response_ce + discrete_action_ce — also
  one combined CE.
- 12 per-field CE = 12× lm_head matmul calls per step; one unified CE = 1 call.
- Per-field weight tuning is a 12-D Cartesian space; per-sample sampling weight
  (PI0Mixture n^0.43) is the simpler control surface.

Pipeline placement:
- After ResizeImagesWithPadFn / RemapImageKeyTransformFn (image side ready).
- Before AnnotationTokenizeTransformFn (so the unified text field exists when
  the tokenizer runs).
- The schema declares `AnnotationLossSpec(field="annotation.unified", ...)` and
  `AnnotationTokenizeTransformFn` picks it up via `hydrate_all` injection.

V1 design: raw text passthrough with simple `<tag>...</tag>` wrapping per
field. Coordinates are kept as-is from the parquet (pixel-space, model learns
the convention). V2 can normalize bbox to Qwen3-VL `<|box_start|>(x,y),(x,y)
<|box_end|>` integer-1000 format for tighter grounding alignment — left as a
follow-up to keep the first co-train run shippable.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Sequence

from transforms.core import DataDict, DataTransformFn


# Default field order. annotation.segmentation is intentionally OMITTED —
# dense pixel mask is too heavy to tokenize as text (would blow seq budget).
#
# bbox / object-location fields (object_box / gripper_box / affordance_box) are
# emitted BEFORE annotation.substask. π0.5 App B.2 prescribes "locate first,
# then subtask": grounding the boxes first constrains the subtask token
# prediction (next-token CE sees boxes in the prefix when it predicts subtask).
_DEFAULT_FIELDS: tuple[str, ...] = (
    "annotation.object_box",
    "annotation.gripper_box",
    "annotation.affordance_box",
    "annotation.substask",
    "annotation.primitive_skill",
    "annotation.instruction_add",
    "annotation.placement_proposal",
    "annotation.state_affordance",
    "annotation.contact_frame",
    "annotation.contact_points",
    "annotation.trace",
    # annotation.time_clip / annotation.origin_shape kept available but lower-
    # signal — included for completeness, model will learn to skip if useless.
    "annotation.time_clip",
)


# Short tag per field (keeps token count low — full names like
# "annotation_state_affordance" would burn ~6 BPE tokens per tag).
_FIELD_TAGS: dict[str, str] = {
    "annotation.substask":            "subtask",
    "annotation.primitive_skill":     "skill",
    "annotation.instruction_add":     "inst",
    "annotation.object_box":          "obj_box",
    "annotation.gripper_box":         "grip_box",
    "annotation.affordance_box":      "aff_box",
    "annotation.placement_proposal":  "place",
    "annotation.state_affordance":    "aff",
    "annotation.contact_frame":       "ctc_f",
    "annotation.contact_points":      "ctc_p",
    "annotation.trace":               "trace",
    "annotation.time_clip":           "tspan",
}


def _coerce_to_str(value) -> str:
    """Coerce parquet-cell value to a clean string. Returns '' for missing /
    null / 'None' / 'nan' sentinels so the unified text doesn't get polluted."""
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8", errors="ignore")
        except Exception:
            return ""
    if not isinstance(value, str):
        try:
            value = str(value)
        except Exception:
            return ""
    s = value.strip()
    # Common parquet null-stringification artifacts.
    if not s or s.lower() in {"none", "nan", "null", "[]", "{}", "[[]]", "''", '""'}:
        return ""
    return s


@DataTransformFn.register_subclass("build_unified_annotation")
@dataclass
class BuildUnifiedAnnotationTransformFn(DataTransformFn):
    """Concatenate per-field annotation columns into one `annotation.unified`
    string, tag-wrapped for the tokenizer.

    Args:
        fields: ordered list of `annotation.<X>` column names to include.
            Order matters: model sees fields in this order, so put highest-
            signal first (subtask before low-signal time_clip).
        unified_key: output key in the sample dict.
        max_chars: optional safety cap on total characters (prevents a
            runaway trace string from blowing past the tokenizer budget).
    """

    fields: tuple[str, ...] = dc_field(default_factory=lambda: _DEFAULT_FIELDS)
    unified_key: str = "annotation.unified"
    max_chars: int = 4096   # ~256 tokens at ~16 chars/token (BPE worst case)

    def __call__(self, data: DataDict) -> DataDict:
        segments = []
        budget = self.max_chars
        for f in self.fields:
            if f not in data:
                continue
            val = _coerce_to_str(data[f])
            if not val:
                continue
            tag = _FIELD_TAGS.get(f, f.replace("annotation.", ""))
            seg = f"<{tag}>{val}</{tag}>"
            # Char-budget guard — drop overflowing tail rather than silently
            # truncate mid-token (which would corrupt the JSON-like coord lists).
            if budget - len(seg) - 1 < 0:
                break
            segments.append(seg)
            budget -= (len(seg) + 1)  # +1 for newline separator
        data[self.unified_key] = "\n".join(segments)
        return data
