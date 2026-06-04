"""robointer_droid_anno — same physical dataset as `robointer_droid` (the
`robointer_droid_clean` repo), but with `annotation_losses` populated to
turn on the unified-annotation CE branch.

Use:
    --dataset_schema robointer_droid_clean=robointer_droid_anno

The `robointer_droid_clean` parquet already carries all 12
`annotation.<X>` columns inline (verified 2026-04-26 — see plan
/root/.claude/plans/cheerful-munching-bumblebee.md, Phase 0). The transform
chain runs `BuildUnifiedAnnotationTransformFn` to pack the 12 columns into
a single `annotation.unified` text field, then
`AnnotationTokenizeTransformFn` (with this schema's annotation_losses
injected via `hydrate_all`) tokenizes it for next-token CE.
"""
from dataclasses import replace

from schemas.robointer_droid import SCHEMA as _BASE
from schema.annotation_loss import AnnotationLossSpec
from src.dataset.adapters.robointer_token_budget import ROBOINTER_BUDGET


SCHEMA = replace(
    _BASE,
    schema_id="robointer_droid_anno_v1",
    annotation_losses=(
        AnnotationLossSpec(
            field="annotation.unified",
            loss_type="ce_text",
            weight=1.0,                                            # full weight; sampling controls mix
            max_length=ROBOINTER_BUDGET.annotation_unified_max_length,  # 256
        ),
    ),
)
