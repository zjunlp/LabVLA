"""LabVLA DataLoader collation helpers.

This module is imported by ``scripts/train.py`` but kept outside the main
training entrypoint so DataLoader worker pickling can resolve collator symbols
without bloating the train script.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate


def last_valid_token_length(mask: torch.Tensor) -> int | None:
    """Return the last valid sequence position + 1 for a ``(B, L)`` mask.

    Keeps at least one column for all-empty masks so downstream code that
    expects rank-2 token tensors remains shape-safe while still avoiding the
    full fixed max_length tail.
    """
    if not isinstance(mask, torch.Tensor) or mask.ndim != 2:
        return None
    if mask.shape[1] <= 1:
        return int(mask.shape[1])
    valid_by_pos = mask.to(torch.bool).any(dim=0)
    if not bool(valid_by_pos.any().item()):
        return 1
    return int(torch.nonzero(valid_by_pos, as_tuple=False).max().item()) + 1


def trim_token_pair_to_mask(batch: dict, tokens_key: str, mask_key: str) -> bool:
    tokens = batch.get(tokens_key)
    mask = batch.get(mask_key)
    if not isinstance(tokens, torch.Tensor) or not isinstance(mask, torch.Tensor):
        return False
    if tokens.ndim != 2 or mask.ndim != 2 or tokens.shape[:2] != mask.shape[:2]:
        return False
    keep_len = last_valid_token_length(mask)
    if keep_len is None or keep_len >= tokens.shape[1]:
        return False
    batch[tokens_key] = tokens[:, :keep_len].contiguous()
    batch[mask_key] = mask[:, :keep_len].contiguous()
    return True


def trim_token_padding_to_batch(batch: dict) -> dict:
    """Remove collated all-pad token tails without changing supervised tokens.

    The transforms intentionally emit fixed-shape prompt / FAST / annotation
    tensors so vanilla collation works. Once samples are collated, trailing
    positions whose masks are false for every row can be removed without
    changing any supervised token or loss mask. This reduces LM sequence length
    but preserves forward/loss semantics.
    """
    trim_token_pair_to_mask(batch, "observation.input_ids", "observation.attention_mask")
    trim_token_pair_to_mask(batch, "fast_action_tokens", "fast_action_mask")

    for tokens_key in list(batch.keys()):
        if not tokens_key.startswith("annotation_tokens__"):
            continue
        field_name = tokens_key[len("annotation_tokens__"):]
        trim_token_pair_to_mask(
            batch,
            tokens_key,
            f"annotation_mask__{field_name}",
        )
    return batch


def token_mask_key_pairs(keys: set[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if "observation.input_ids" in keys or "observation.attention_mask" in keys:
        pairs.append(("observation.input_ids", "observation.attention_mask"))
    if "fast_action_tokens" in keys or "fast_action_mask" in keys:
        pairs.append(("fast_action_tokens", "fast_action_mask"))
    annotation_fields = {
        k[len("annotation_tokens__"):]
        for k in keys
        if k.startswith("annotation_tokens__")
    }
    annotation_fields.update(
        k[len("annotation_mask__"):]
        for k in keys
        if k.startswith("annotation_mask__")
    )
    for field_name in sorted(annotation_fields):
        pairs.append((
            f"annotation_tokens__{field_name}",
            f"annotation_mask__{field_name}",
        ))
    return pairs


def validate_token_mask_pair_completeness(samples: list[dict], all_keys: set[str]) -> None:
    """Reject samples that carry only one side of a token/mask supervision pair."""
    for sample_index, sample in enumerate(samples):
        for tokens_key, mask_key in token_mask_key_pairs(all_keys):
            has_tokens = tokens_key in sample
            has_mask = mask_key in sample
            if has_tokens == has_mask:
                continue
            present_key = tokens_key if has_tokens else mask_key
            missing_key = mask_key if has_tokens else tokens_key
            raise ValueError(
                "Incomplete token/mask supervision pair before collation: "
                f"sample={sample_index}, present={present_key!r}, missing={missing_key!r}"
            )


def pad_variable_token_pairs(samples: list[dict], all_keys: set[str]) -> None:
    """Pad variable-length 1D token/mask pairs to the batch-local max length."""
    for tokens_key, mask_key in token_mask_key_pairs(all_keys):
        max_len = 0
        compatible = True
        for sample in samples:
            tokens = sample.get(tokens_key)
            mask = sample.get(mask_key)
            if not isinstance(tokens, torch.Tensor) or not isinstance(mask, torch.Tensor):
                compatible = False
                break
            if tokens.ndim != 1 or mask.ndim != 1 or tokens.shape[0] != mask.shape[0]:
                compatible = False
                break
            max_len = max(max_len, int(tokens.shape[0]))
        if not compatible or max_len <= 0:
            continue
        for sample in samples:
            current_len = int(sample[tokens_key].shape[0])
            pad_len = max_len - current_len
            if pad_len <= 0:
                continue
            sample[tokens_key] = F.pad(sample[tokens_key], (0, pad_len), value=0)
            sample[mask_key] = F.pad(sample[mask_key], (0, pad_len), value=0)


class LabVLACollator:
    """Pickle-safe DataLoader collator for LabVLA batches."""

    def __init__(self, trim_token_padding_to_batch: bool = False) -> None:
        self.trim_token_padding_to_batch = bool(trim_token_padding_to_batch)

    def __call__(self, batch):
        return labvla_collate_fn(
            batch,
            trim_token_padding_to_batch=self.trim_token_padding_to_batch,
        )


def labvla_collate_fn(batch, *, trim_token_padding_to_batch: bool = False):
    """Heterogeneous-batch collate for pi0-mixture multi-dataset training.

    When pi0-mixture mixes datasets with different schemas, torch's
    ``default_collate`` raises ``KeyError`` because each sample dict has a
    different key set. Missing keys are backfilled from a present sample's
    shape/dtype, using zero tensors for tensor fields.

    Zero-fill is semantically correct for the annotation pipeline:
      - ``annotation_mask__*``: 0 = ignore in CE gather
      - ``annotation_tokens__*``: 0 IDs are masked out by CE
      - ``annotation_weight__*``: 0 scales the loss to 0
    A sample whose schema declares no annotation specs therefore contributes
    no annotation CE after zero-fill.
    """
    if not batch:
        return {}

    normalized_batch = [dict(sample) for sample in batch]

    all_keys: set = set()
    for sample in normalized_batch:
        all_keys.update(sample.keys())

    validate_token_mask_pair_completeness(normalized_batch, all_keys)

    ref: dict = {}
    for key in all_keys:
        for sample in normalized_batch:
            if key in sample:
                ref[key] = sample[key]
                break

    for sample in normalized_batch:
        for key in all_keys:
            if key in sample:
                continue
            value = ref[key]
            if isinstance(value, torch.Tensor):
                sample[key] = torch.zeros_like(value)
            else:
                sample[key] = value

    pad_variable_token_pairs(normalized_batch, all_keys)
    collated = default_collate(normalized_batch)
    if trim_token_padding_to_batch:
        collated = trim_token_padding_to_batch_fn(collated)
    return collated


# Avoid shadowing the public function name with the boolean parameter above.
trim_token_padding_to_batch_fn = trim_token_padding_to_batch
