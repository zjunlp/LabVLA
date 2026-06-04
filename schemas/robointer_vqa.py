"""robointer_vqa — VQA-only schema (no robot, but declares canonical
state/action keys with zero data so the existing transform chain can flow
without per-transform empty-mapping guards).

Used by `RoboInterVQADataset` for samples that carry image + question +
answer but no proprioceptive trajectory. The adapter fills `observation.
state` and `action` with zero tensors and sets `action_is_pad=all True` so
the model's flow-matching MSE is auto-masked. Only the annotation CE on
`annotation.vqa_answer` provides supervision.

Why declare canonical keys instead of empty tuples:
  Empty state/action keys would crash `ComposeFieldsTransform` (empty
  `src_keys` → `torch.cat([], ...)`) and `DeltaActionTransformFn` (same).
  Declaring single-key canonical state/action is the smallest-diff path —
  identity-stats normalization, identity compose, zero-deltas.
"""
import numpy as np

from schema import DatasetSchema
from schema.annotation_loss import AnnotationLossSpec
from src.dataset.adapters.robointer_token_budget import ROBOINTER_BUDGET


# Identity stats for the zero-filled state/action so NormalizeTransformFn
# is a numerical no-op (subtract 0, divide by 1).
def _identity_stats(dim: int) -> dict:
    return {
        "mean":  np.zeros(dim, dtype=np.float32),
        "std":   np.ones(dim, dtype=np.float32),
        "min":   -np.ones(dim, dtype=np.float32),
        "max":   np.ones(dim, dtype=np.float32),
        "q01":   -np.ones(dim, dtype=np.float32),
        "q99":   np.ones(dim, dtype=np.float32),
        "count": 1,
    }


# Surface a stats blob the adapter can attach to its `meta.stats`. Both the
# delta and abs canonical keys are populated so `hydrate_all`'s
# `action_canonical_key` lookup (line 808) succeeds in either action_mode.
VQA_STATS: dict = {
    "observation.state": _identity_stats(ROBOINTER_BUDGET.max_state_dim),
    "action":            _identity_stats(ROBOINTER_BUDGET.max_action_dim),
    "action_abs":        _identity_stats(ROBOINTER_BUDGET.max_action_dim),
}


SCHEMA = DatasetSchema(
    schema_id="robointer_vqa_v1",
    robot_type="vqa_no_robot",
    # Single canonical state/action keys — adapter pre-fills both with zeros.
    state_keys=("observation.state",),
    state_dims=(ROBOINTER_BUDGET.max_state_dim,),
    action_keys=("action",),
    action_dims=(ROBOINTER_BUDGET.max_action_dim,),
    # All-False delta_mask (length = max_action_dim): no delta arithmetic
    # performed (state and action are both zero, but the mask suppresses any
    # accidental subtraction).
    delta_mask=(False,) * ROBOINTER_BUDGET.max_action_dim,
    # No gripper dims declared (no gripper).
    gripper_action_dims=(),
    image_mapping={
        "observation.images.primary": "observation.images.image0",
    },
    source="manifest",
    source_path=__file__,
    arm_layout=None,                                # no arm — legacy/unset
    annotation_losses=(
        AnnotationLossSpec(
            field="annotation.vqa_answer",
            loss_type="ce_text",
            weight=1.0,
            # 256 covers the longest case (Generation/traj_qa with 16 keypoints).
            # Per-task-family stricter caps are enforced in the adapter side
            # (RoboInterVQADataset.answer_max_length) — this is the tokenizer
            # truncation cap, set to the union upper bound.
            max_length=ROBOINTER_BUDGET.vqa_generation_traj_answer_max_length,
        ),
    ),
)
