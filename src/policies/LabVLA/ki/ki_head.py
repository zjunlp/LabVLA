"""Discrete Action Head for Knowledge Isolation / π0.5 VLM pretrain.

Responsibilities:
  - Embed FAST discrete action tokens into VLM hidden dim (for feeding into VLM forward)
  - Project VLM last hidden state back to FAST vocab (for next-token CE loss)
  - Compute masked CE loss over FAST action token positions

Design: fully decoupled from VLM/DiT architecture. Takes only the VLM hidden size
as a construction param. Used in BOTH vlm_pretrain and posttrain+KI modes.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DiscreteActionHead(nn.Module):
    """FAST token embedding (vocab -> hidden) + classifier (hidden -> vocab)."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)

        # Embedding stays in bf16 (matches VLM input-embedding scale; quantization
        # of ±0.02-std weights is harmless).
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        # Classifier kept in fp32 for logits precision. bf16's 7-bit mantissa
        # loses ~0.07 CE nats to quantization in a softmax over 2048 classes at
        # logits ~O(20). Cost of fp32: ~20 MB params + an fp32
        # (B·L, 2560)·(2560, 2048) matmul per step — negligible vs VLM forward.
        self.classifier = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.kaiming_uniform_(self.classifier.weight, a=5 ** 0.5)

        # Split-dtype: embedding bf16 to match VLM; classifier fp32 for precise CE.
        self.embedding.to(dtype=dtype)
        self.classifier.to(dtype=torch.float32)

    def assert_classifier_fp32_invariant(self) -> None:
        """Raise AssertionError if classifier weight is not fp32.

        CE precision (π0.5 "no precision tradeoffs" red line) requires the
        classifier weight to be fp32. Three redundant defenses keep CE
        numerically identical regardless of DeepSpeed ZeRO master-shard behavior:

          1. `__init__` casts classifier to fp32 (this file, ctor).
          2. `scripts/train.py` re-casts classifier to fp32 AFTER
             `accelerator.prepare` because DS bfloat16() recasts everything.
          3. `compute_ce_loss` defensively casts the matmul operands to fp32
             on every call.

        If defenses 1 + 2 are removed AND defense 3 is removed, classifier
        silently downgrades to bf16 -> CE quantization. Call this assert on
        a known-trained (>= step 1) model to detect regression. Cheap dtype
        check (no allocation).
        """
        assert self.classifier.weight.dtype == torch.float32, (
            f"ki_head.classifier.weight.dtype={self.classifier.weight.dtype} "
            f"(expected torch.float32). The fp32 invariant is set in "
            f"DiscreteActionHead.__init__ and re-asserted in scripts/train.py "
            f"after accelerator.prepare (DeepSpeed bfloat16() recasts every "
            f"param). If you removed either, CE precision regresses (red-team "
            f"H5: bf16 quantization adds ~0.07 nat to CE loss over 2048 "
            f"classes). Restore the cast or accept the precision loss."
        )

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, L) int64 -> (B, L, hidden) same dtype as weights."""
        return self.embedding(tokens)

    def compute_ce_loss(
        self,
        hidden: torch.Tensor,         # (B, L, hidden) — VLM output aligned with target_tokens
        target_tokens: torch.Tensor,  # (B, L) int64, ground truth FAST tokens
        mask: torch.Tensor,           # (B, L) bool, True = valid
    ) -> torch.Tensor:
        """Next-token CE loss, masked averaged to scalar.

        Caller is responsible for the one-position shift (hidden[i] predicts
        target_tokens[i]). This function just applies the classifier + masked CE.

        DeepSpeed ZeRO-2's bf16 master shard can write classifier.weight back to
        bf16 on optimizer.step() even after we re-cast to fp32 post-prepare. Run
        the matmul in fp32 regardless of the stored weight dtype to keep CE
        precision invariant; the classifier is tiny (~20MB) so the fp32 cast is
        cheap and CE is numerically identical whether the weight is bf16 or fp32.
        """
        w = self.classifier.weight
        # Explicit `.to(torch.float32)` (not `.float()`) so the fp32-CE intent is
        # obvious regardless of whatever dtype DeepSpeed's master shard last wrote.
        logits = F.linear(hidden.to(torch.float32), w.to(torch.float32), None)
        loss_flat = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            target_tokens.view(-1).to(hidden.device).long(),
            reduction="none",
        )
        mask_flat = mask.view(-1).to(loss_flat.device).float()
        denom = mask_flat.sum().clamp(min=1.0)
        return (loss_flat * mask_flat).sum() / denom
