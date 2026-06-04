"""π0.5 §B.1 — robot proprioceptive state discretized into bins, prepended to
the language prompt as text tokens.

Equivalent to OpenTau `prepare_discrete_state` (modeling_pi05.py:693-705):
  state ∈ [-1, 1] → bin_idx = floor((state + 1) * num_bins / 2) clamped to
  [0, num_bins - 1] → space-separated digit string.

Pipeline placement:
  Runs BEFORE Qwen3_VLProcessorTransformFn so the resulting prefix string is
  tokenized by the standard VLM tokenizer alongside the task instruction.
  This means the transform runs BEFORE ComposeFieldsTransform, so for
  split-state schemas (e.g. robointer_droid with separate
  ``other_information.observation_joint_position`` and
  ``other_information.observation_gripper_position`` keys) the canonical
  ``observation.state`` key is NOT yet present in the sample dict. In that
  case we transiently concatenate the per-key tensors in schema-declared
  order to compute the discretization — without writing the merged vector
  back into the sample (ComposeFieldsTransform later does that authoritatively).

VQA samples: VQA datasets emit a sentinel `has_real_state: bool = False` in
  the sample dict. When that key is False (or missing for VQA samples that
  forgot to set it), the transform is a no-op for that sample — VQA samples
  do not have proprioceptive state.

Hydration:
  `transforms.core.hydrate_all` injects ``state_keys`` (the schema-declared
  per-key concat order) into this transform when
  `LabVLADatasetConfig.discretize_state_in_vlm_pretrain` is True
  (auto-true when `training_phase == "vlm_pretrain"`). This is what makes
  the transform schema-aware for split-state datasets.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from utils.constants import OBS_STATE
from utils.logging_utils import warn_once
from transforms.core import DataDict, DataTransformFn


logger = logging.getLogger(__name__)


@DataTransformFn.register_subclass("state_discretize")
@dataclass
class DiscretizeStateTransformFn(DataTransformFn):
    """Discretize observation.state into `num_bins` bins, serialize as text,
    prepend to `task` prompt.

    Args:
        num_bins: bin count for each state dim. π0.5 spec = 256.
        wrap_open: opening tag inserted around the state digits in the prompt.
        wrap_close: closing tag.
        state_keys: schema-declared state sub-key concat order. Hydrated by
            ``transforms.core.hydrate_all`` from ``schema.state_keys``. When
            ``OBS_STATE`` is absent from the sample (split-state schemas
            running BEFORE ComposeFieldsTransform), the transform reads
            these sub-keys, concatenates them along the last dim in order,
            and uses the result for discretization. Empty default preserves
            backward compat for callers that route OBS_STATE in directly.
    """

    num_bins: int = 256
    wrap_open: str = "<state>"
    wrap_close: str = "</state>"
    # Hydrated by transforms.core.hydrate_all from schema.state_keys.
    # Tuple (not list) so the transform stays hashable / pickle-safe across
    # DataLoader workers, matching every other schema-driven field.
    state_keys: tuple[str, ...] = field(default_factory=tuple)

    def _gather_state(self, data: DataDict) -> tuple[torch.Tensor | None, str | None]:
        """Return ``(state_tensor, missing_key)``.

        ``state_tensor`` is the tensor to discretize, or None if unavailable.
        ``missing_key`` names the first declared ``state_keys`` sub-key that was
        absent (only set on the split-state path), so the caller can warn-once
        when ``has_real_state`` is True instead of a silent no-op that drops the
        ``<state>`` prefix.

        Resolution order:
          1. ``data[OBS_STATE]`` if present (post-Compose, or single-key
             schemas where state_keys == (OBS_STATE,)).
          2. ``data[k]`` concatenated for each k in self.state_keys (split
             state, pre-Compose).
        """
        if OBS_STATE in data:
            return data[OBS_STATE], None
        if not self.state_keys:
            return None, None
        parts: list[torch.Tensor] = []
        for k in self.state_keys:
            if k not in data:
                # Sample is missing a declared sub-key. Report which key so the
                # caller can fail loud / warn-once for real-state samples rather
                # than silently dropping the <state> prefix.
                return None, k
            v = data[k]
            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v, dtype=torch.float32)
            # Promote 0-dim scalar (e.g. robointer_droid gripper width
            # arrives as a single-element scalar at this stage) to 1-dim
            # so cat along last axis works uniformly.
            if v.ndim == 0:
                v = v.unsqueeze(0)
            parts.append(v)
        # Align ndim for cat: if some parts are (D,) and others are
        # (T, D) (e.g. chunked state from delta_timestamps), take first
        # timestep from temporal parts to flatten to (D,). We deliberately
        # don't write the merged tensor back into `data` — that's
        # ComposeFieldsTransform's job downstream.
        aligned = [p[0] if p.ndim > 1 else p for p in parts]
        return torch.cat(aligned, dim=-1), None

    def __call__(self, data: DataDict) -> DataDict:
        # VQA samples mark themselves with has_real_state=False (or omit
        # observation.state entirely). Skip discretization for them.
        has_real_state = bool(data.get("has_real_state", True))
        if not has_real_state:
            return data

        state, missing_key = self._gather_state(data)
        if state is None:
            # has_real_state is True but a declared state_keys sub-key is
            # missing. Warn-once so this schema/sample drift is visible: a
            # silent no-op would drop the <state>...</state> prefix and conflate
            # "real robot sample with drift" with "VQA sample with no state".
            if missing_key is not None:
                warn_once(
                    logger,
                    ("state_discretize_missing_key", missing_key),
                    "[DiscretizeStateTransformFn] has_real_state=True but "
                    "declared state_keys sub-key %r is missing from the sample "
                    "(state_keys=%r) — dropping the <state> prefix for these "
                    "samples. This indicates schema/sample drift; the model "
                    "will train WITHOUT discretized state for them. (further "
                    "occurrences for this key are suppressed)",
                    missing_key,
                    tuple(self.state_keys),
                )
            return data

        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)
        else:
            state = state.detach().to(dtype=torch.float32, device="cpu")
        if not torch.isfinite(state).all():
            raise ValueError(
                "DiscretizeStateTransformFn received non-finite state values; "
                "skipping this sample is safer than serializing corrupted bins."
            )

        # Clamp to [-1, 1] and quantize (matches OpenTau modeling_pi05.py:701).
        state = torch.clamp(state, -1.0, 1.0)
        bin_indices = ((state + 1.0) * (self.num_bins / 2.0)).long().clamp(0, self.num_bins - 1)
        # Flatten (state may be 1D after PadStateAndActionTransformFn or before).
        flat = bin_indices.flatten().tolist()
        state_text = " ".join(str(int(b)) for b in flat)

        # Prepend to task instruction.
        old_prompt = data.get("task", "")
        if not isinstance(old_prompt, str):
            old_prompt = str(old_prompt) if old_prompt is not None else ""
        data["task"] = f"{self.wrap_open}{state_text}{self.wrap_close}\n{old_prompt}"
        return data
