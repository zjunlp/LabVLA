"""Thin wrapper around HuggingFace-format FAST action tokenizer.

FAST (Pertsch et al., Physical Intelligence) tokenizes continuous action chunks
via DCT + quantization + BPE. Input must be in [-1, 1] range (post-normalization).

Local path (default): /data/rbc/Embodied_models/fast
The underlying processor is a UniversalActionProcessor (HuggingFace ProcessorMixin).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from transformers import AutoProcessor


class FastTokenizerWrapper:
    """Load once, encode many. Stateless after load; safe for DataLoader workers.

    Encode path:
      actions (K, D) in [-1, 1] -> list[int] tokens -> zero-padded np.int64 (max_length,)
                                                    -> mask (max_length,) bool
    """

    # Tolerance for the out-of-range diagnostic in encode(). See encode()'s
    # docstring for why FAST's required [-1, 1] clip is asymmetric with the
    # continuous MSE branch and why we only warn rather than clip the MSE target.
    _OOR_TOLERANCE: float = 1e-3

    def __init__(self, path: str, vocab_size: int, max_length: int):
        self.path = Path(path)
        self.vocab_size = int(vocab_size)
        self.max_length = int(max_length)
        # trust_remote_code needed because the processor class is defined locally
        # in processing_action_tokenizer.py (not in transformers upstream).
        self._processor = AutoProcessor.from_pretrained(
            str(self.path), trust_remote_code=True,
        )
        self._truncated = 0  # count how many encode() calls hit max_length
        self._out_of_range = 0  # count encode() calls with |action| > 1+tol

    def encode(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encode a single (K, D) action chunk.

        Args:
            actions: (K, D) normalized to [-1, 1], float32/float64 ok

        Returns:
            tokens: (max_length,) int64, zero-padded
            mask:   (max_length,) bool, True for valid positions

        Note: FAST's DCT+BPE pipeline requires inputs in [-1, 1], so this method
        clips `actions` before tokenizing (intentional, part of the FAST
        contract). This is ASYMMETRIC with the continuous flow-matching MSE
        branch, which trains on the UNCLIPPED normalized actions — so any element
        with |action| > 1 gets a clipped discrete CE target but an unclipped MSE
        target. We warn-once (no behavioural change) past a small tolerance,
        since it usually means stats drift or outliers, but deliberately do NOT
        clip the continuous MSE target (that would discard precision).
        """
        assert actions.ndim == 2, f"expected (K,D), got {actions.shape}"
        a_f32 = actions.astype(np.float32)
        # Diagnostic: detect normalized actions outside the FAST-required
        # [-1, 1] range (signals stats drift / outliers / norm mismatch).
        max_abs = float(np.abs(a_f32).max()) if a_f32.size else 0.0
        if max_abs > 1.0 + self._OOR_TOLERANCE:
            self._out_of_range += 1
            if self._out_of_range == 1 or self._out_of_range % 5000 == 0:
                import logging
                logging.warning(
                    "[FastTokenizerWrapper] %d action chunks had normalized "
                    "values outside [-1, 1] (most recent max|a|=%.4f) and were "
                    "clipped for FAST tokenization. The continuous flow-matching "
                    "MSE branch trains on the UNCLIPPED values, so the discrete "
                    "(FAST CE) and continuous (MSE) targets diverge for those "
                    "elements. This usually indicates normalization-stats drift "
                    "or outliers — re-check the dataset stats / canonicalization.",
                    self._out_of_range, max_abs,
                )
        a = np.clip(a_f32, -1.0, 1.0)
        # UniversalActionProcessor.__call__ handles (K,D) -> wraps to (1,K,D) internally
        # and returns list[list[int]] of length 1
        batched = self._processor(a)
        token_ids = batched[0] if isinstance(batched, list) and len(batched) > 0 else []

        tokens = np.zeros(self.max_length, dtype=np.int64)
        mask = np.zeros(self.max_length, dtype=bool)
        raw_len = len(token_ids)
        L = min(raw_len, self.max_length)
        if raw_len > self.max_length:
            self._truncated += 1
            # Sample-log on 1st and every 5000th truncation to avoid flooding but
            # still surface systematic truncation without needing a separate counter
            # report.
            if self._truncated == 1 or self._truncated % 5000 == 0:
                import logging
                logging.warning(
                    "[FastTokenizerWrapper] %d samples truncated to max_length=%d "
                    "(most recent raw_len=%d). If this count grows, consider "
                    "raising max_length.", self._truncated, self.max_length, raw_len,
                )
        if L > 0:
            tokens[:L] = np.asarray(token_ids[:L], dtype=np.int64)
            mask[:L] = True
        return tokens, mask
