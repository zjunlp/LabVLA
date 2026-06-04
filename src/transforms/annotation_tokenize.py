"""Tokenize free-text annotation columns (e.g. `annotation.substask`) into
fixed-length token tensors keyed by `annotation_tokens__<field>` /
`annotation_mask__<field>`.

The model-side CE loss branch consumes these keys; see
`modeling_labvla._compute_annotation_ce`.

Placed late in the transform chain but BEFORE the final
`UnifyLabVLAInputsTransformFn` so that:
  - the adapter has already put the raw annotation string into the sample dict
  - the downstream Unify step can pass `annotation_tokens__*` through to the
    collated batch

This transform is a no-op when `annotation_specs` is empty, so datasets
without annotation_losses incur zero cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from schema.annotation_loss import AnnotationLossSpec
from transforms.core import DataDict, DataTransformFn


@DataTransformFn.register_subclass("annotation_tokenize")
@dataclass
class AnnotationTokenizeTransformFn(DataTransformFn):
    """Tokenize each configured annotation field with the VLM tokenizer.

    Args:
        tokenizer_path:
            HF model id or local path used to instantiate a fresh tokenizer.
            Should match the VLM tokenizer used by
            `Qwen3_VLProcessorTransformFn` to ensure token-id consistency.
        annotation_specs:
            Tuple of AnnotationLossSpec; one tokenization pass per entry.
    """

    tokenizer_path: str = "Qwen/Qwen3-VL-4B-Instruct"
    # Populated by hydrate_all from schema.annotation_losses. Default empty
    # tuple → this transform is a cheap passthrough (no-op).
    annotation_specs: tuple[AnnotationLossSpec, ...] = ()
    # When true, emit tensors at the effective per-sample cap instead of the
    # schema union cap. The LabVLA collator pads variable 1D token tensors back
    # to the batch-local max, preserving collatability.
    dynamic_shape: bool = False

    # Instantiated lazily in __post_init__; not a dataclass init arg.
    tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.annotation_specs:
            # No-op mode: skip tokenizer load entirely. Keeps OXE-only runs
            # (no annotations) free of unused HF downloads / CPU cost.
            self.tokenizer = None
            return
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

    def __call__(self, data: DataDict) -> DataDict:
        if not self.annotation_specs or self.tokenizer is None:
            return data
        for spec in self.annotation_specs:
            raw = data.get(spec.field, "")
            if not isinstance(raw, str):
                # Some annotations are stored as bytes or json strings;
                # coerce to plain text. Non-coercible → empty string (fully
                # masked below).
                try:
                    raw = str(raw) if raw is not None else ""
                except Exception:
                    raw = ""
            # Per-sample max_length override. Adapters that
            # know a tighter cap for the current sample (e.g. RoboInter VQA's
            # task-family budgets: understanding=8, planning=64) can attach
            # `f"{spec.field}_max_length"` to the sample dict. We always clamp
            # against the schema-level `spec.max_length` (the schema value is
            # the union upper bound; per-sample can only TIGHTEN it).
            #
            # Tensor SHAPE is preserved at `(spec.max_length,)` so the default
            # collator can stack heterogeneous task-family samples into one
            # batch. The cap manifests as: real tokens are truncated to
            # `effective_max_length`; mask positions in
            # `[effective_max_length, spec.max_length)` are forced to 0; ID
            # positions in that range are zero-filled (so even if the LM ever
            # ran on them, CE would be ignored via mask).
            per_sample_key = f"{spec.field}_max_length"
            per_sample_cap = data.get(per_sample_key, None)
            if isinstance(per_sample_cap, int) and per_sample_cap > 0:
                effective_max_length = min(per_sample_cap, spec.max_length)
            else:
                effective_max_length = spec.max_length
            tensor_max_length = effective_max_length if self.dynamic_shape else spec.max_length
            enc = self.tokenizer(
                raw if raw else "",
                max_length=tensor_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",        # direct tensor, skip list round-trip
                add_special_tokens=False,   # never pollute annotation with BOS/EOS
            )
            # return_tensors="pt" yields shape (1, L); squeeze to (L,).
            ids = enc["input_ids"][0].long()
            mask = enc["attention_mask"][0].bool()
            # Samples with empty annotation string still get non-zero length
            # (BOS/EOS depending on tokenizer) but attention_mask zeros out
            # every position — guarantees zero CE contribution.
            if not raw:
                mask = torch.zeros_like(mask)
            # Apply per-sample cap by zeroing IDs and mask beyond
            # `effective_max_length`. Done in-place on the freshly-allocated
            # squeezed tensors (no aliasing concern). When per-sample cap ==
            # spec.max_length (the default no-override path), this is a no-op.
            if not self.dynamic_shape and effective_max_length < spec.max_length:
                ids[effective_max_length:] = 0
                mask[effective_max_length:] = False
            data[spec.batch_key_tokens()] = ids
            data[spec.batch_key_mask()] = mask
            # Per-field weight carried inside the batch so the model can
            # compose the loss without a cross-cutting schema reference. All
            # samples from the same dataset carry the same scalar — this is
            # a per-sample-shape (not per-value) carrier for a schema-level
            # constant.
            data[spec.batch_key_weight()] = torch.tensor(
                float(spec.weight), dtype=torch.float32
            )
        return data
