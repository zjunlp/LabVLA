"""DataLoader-worker transform: encode normalized action chunk into FAST tokens.

Must run AFTER:
  - NormalizeTransformFn (action in [-1, 1])
  - ComposeFieldsTransformFn / source-local action field assembly

Must run BEFORE:
  - PadStateAndActionTransformFn

Running after pad would encode zero-filled trailing action dimensions into
FAST tokens and pollute the CE target.

Output keys added in-place into sample dict:
  - fast_action_tokens: LongTensor(max_length,)
  - fast_action_mask:   BoolTensor(max_length,)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from transforms.core import DataTransformFn


_TOKENIZER_CACHE: dict = {}  # process-local


def _get_tokenizer(path: str, vocab_size: int, max_length: int):
    key = (path, int(vocab_size), int(max_length))
    if key not in _TOKENIZER_CACHE:
        from policies.LabVLA.ki.fast_tokenizer import FastTokenizerWrapper
        _TOKENIZER_CACHE[key] = FastTokenizerWrapper(path, vocab_size, max_length)
    return _TOKENIZER_CACHE[key]


@DataTransformFn.register_subclass("fast_action_encode")
@dataclass
class FastActionEncodeTransformFn(DataTransformFn):
    path: str = "/all-flash-data/Embodied_models/fast"
    vocab_size: int = 2048
    max_length: int = 256
    source_shape_convergence: bool = False
    trim_to_mask: bool = False

    def __post_init__(self):
        # Eagerly preload in the parent process so all fork'd DataLoader workers
        # inherit the loaded tokenizer via copy-on-write, avoiding a 192-worker
        # scipy/BPE cold-load storm on first sample pull.
        _get_tokenizer(self.path, self.vocab_size, self.max_length)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "action" not in data:
            return data
        a = data["action"]
        if isinstance(a, torch.Tensor):
            a_np = a.detach().cpu().numpy()
        else:
            a_np = np.asarray(a)
        if a_np.ndim != 2:
            # Default-strict: a non-chunk action means an earlier transform
            # reshaped action to 1D/0D, which would silently drop FAST CE
            # supervision while training continues (weaker vlm_pretrain run).
            # Opt out only when intentionally mixing un-chunked action (rare).
            if os.environ.get("LABVLA_FAST_ACTION_NON_CHUNK_SKIP") == "1":
                return data
            raise ValueError(
                f"FastActionEncodeTransformFn: expected 2-D action chunk "
                f"(T, D) but got ndim={a_np.ndim}, shape={a_np.shape}. An "
                "earlier transform stripped the chunk dim, or the adapter "
                "emitted a single-frame action without chunking. Set "
                "LABVLA_FAST_ACTION_NON_CHUNK_SKIP=1 to fall back to the "
                "legacy silent-skip behaviour."
            )

        # Zero-fill padded frames before FAST encoding. FAST = DCT + BPE;
        # feeding clipped-to-last-frame action through DCT produces a distinct
        # "padded-tail" coefficient pattern that BPE then codes into specific
        # tokens. The CE branch would learn to *predict* those tokens, i.e.
        # learn to emit padded action chunks at inference — a leaked training
        # artifact. Masking padded frames to zero keeps the pre-pad signal
        # intact while removing the padded tail's distinctive DCT fingerprint.
        is_pad = data.get("action_is_pad")
        pad_np = None
        if is_pad is not None:
            pad_np = is_pad.detach().cpu().numpy() if isinstance(is_pad, torch.Tensor) else np.asarray(is_pad)
            # action shape may be (T, D) after adapter unwraps per-sample batch;
            # is_pad can arrive as (T,) or (1, T). Align to (T,) then broadcast.
            if pad_np.ndim == 2 and pad_np.shape[0] == 1:
                pad_np = pad_np[0]
            # VQA-only samples (robointer_vqa_adapter) emit
            # ``action=zeros`` + ``action_is_pad=ones`` (every frame padded).
            # If we tokenize anyway, FAST encodes the all-zero chunk into a
            # specific BPE pattern and the model learns to predict it for VQA
            # samples — fabricated supervision on samples that have no real
            # action. Detect "every frame is padded" and skip emitting
            # fast_action_tokens / fast_action_mask entirely. Downstream
            # _forward_vlm_pretrain treats the absence of these keys as
            # "this sample contributes no FAST CE term".
            if pad_np.shape[0] == a_np.shape[0] and bool(pad_np.astype(bool).all()):
                return data
            if pad_np.shape[0] == a_np.shape[0]:
                a_np = a_np.copy()
                a_np[pad_np.astype(bool)] = 0.0
            else:
                # action_is_pad length does not match the action chunk length T.
                # A length-mismatched mask means upstream drift; fail loud rather
                # than encode the raw action, whose padded tail's DCT/BPE
                # fingerprint would leak into the FAST CE target (the same
                # fabricated-supervision artifact the zero-fill branch prevents).
                raise ValueError(
                    f"FastActionEncodeTransformFn: action_is_pad length "
                    f"{pad_np.shape[0]} != action chunk length {a_np.shape[0]} "
                    f"(action shape={a_np.shape}, pad shape={pad_np.shape}). "
                    "An upstream transform desynced the pad mask from the "
                    "action chunk; refusing to encode the raw action because a "
                    "padded tail would leak into the FAST CE target."
                )

        tok = _get_tokenizer(self.path, self.vocab_size, self.max_length)
        tokens, mask = tok.encode(a_np)
        if self.trim_to_mask:
            valid = np.asarray(mask).astype(bool)
            if valid.ndim == 1 and valid.size > 1:
                if bool(valid.any()):
                    keep_len = int(np.nonzero(valid)[0].max()) + 1
                else:
                    keep_len = 1
                if keep_len < tokens.shape[0]:
                    tokens = tokens[:keep_len].copy()
                    mask = mask[:keep_len].copy()
        data["fast_action_tokens"] = torch.from_numpy(tokens)
        data["fast_action_mask"] = torch.from_numpy(mask)
        return data


@DataTransformFn.register_subclass("drop_fast_action_supervision")
@dataclass
class DropFastActionSupervisionTransformFn(DataTransformFn):
    """Remove FAST CE targets after they were built by the normal pipeline.

    This is an explicit per-repository escape hatch for VLM-pretrain mixes that
    should keep text/annotation supervision from a source but must not train on
    its action-token targets. Missing FAST keys are already treated downstream
    as "no action CE for this sample".
    """

    reason: str = ""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        data.pop("fast_action_tokens", None)
        data.pop("fast_action_mask", None)
        return data
