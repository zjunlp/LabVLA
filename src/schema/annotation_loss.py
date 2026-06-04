"""Per-dataset annotation loss specification.

A schema can declare one or more `AnnotationLossSpec` entries to request
auxiliary losses computed on annotation columns present in the parquet data.
Dataset adapters pass through arbitrary string columns (e.g. RoboInter's
`annotation.substask`); this spec tells the training pipeline which ones to
tokenize and supervise with a text next-token CE loss.

Only datasets whose parquet actually contains the named `field` should declare
the spec. Datasets with no annotation_losses (e.g. oxe-auge) take the fast
pure-MSE path — completely decoupled from this feature.

Decoupling from Knowledge Isolation
-----------------------------------
Annotation CE is orthogonal to the KI toggle:

- KI=false: annotation CE flows to VLM (standard π0 joint-grad, same as MSE).
- KI=true : annotation CE still flows to VLM (MSE detach does not affect it —
            KI only detaches the MSE path into DiT).

The whole point of annotation CE is to *train the VLM on semantic labels*, so
it must always be un-detached. Only MSE detaching is governed by KI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_SUPPORTED_LOSS_TYPES = ("ce_text",)


@dataclass(frozen=True)
class AnnotationLossSpec:
    """Frozen spec for a single annotation-driven auxiliary loss.

    Fields:
        field:
            Parquet column name, e.g. "annotation.substask". Must be a
            string-typed column; the adapter passes its raw value through to
            the sample dict unchanged.
        loss_type:
            Only "ce_text" in v1 — tokenize the string with the VLM tokenizer
            and apply next-token cross entropy on VLM hidden states at the
            annotation positions.
        weight:
            Multiplier applied to this CE term when added to the total loss.
            Must be > 0. Typical range [0.1, 1.0].
        max_length:
            Max token count after tokenization. Shorter strings are padded;
            longer ones are truncated. Padding positions are excluded from the
            CE computation via an attention-like mask.
    """

    field: str
    loss_type: str = "ce_text"
    weight: float = 0.5
    max_length: int = 32

    def __post_init__(self) -> None:
        if not self.field or not isinstance(self.field, str):
            raise ValueError(
                f"AnnotationLossSpec.field must be a non-empty string, got {self.field!r}"
            )
        if self.loss_type not in _SUPPORTED_LOSS_TYPES:
            raise ValueError(
                f"AnnotationLossSpec.loss_type must be one of {_SUPPORTED_LOSS_TYPES}, "
                f"got {self.loss_type!r}"
            )
        # C36 fix: bool is a subclass of int, so ``isinstance(True, (int, float))``
        # is True. Without the explicit bool guard a config typo like
        # ``weight: true`` / ``max_length: false`` would be silently coerced to
        # 1.0 / 1 / 0 and change the loss semantics instead of being rejected.
        if (isinstance(self.weight, bool)
                or not isinstance(self.weight, (int, float))
                or self.weight <= 0):
            raise ValueError(
                f"AnnotationLossSpec.weight must be a real number > 0, got "
                f"{type(self.weight).__name__}={self.weight!r}"
            )
        if (isinstance(self.max_length, bool)
                or not isinstance(self.max_length, int)
                or self.max_length <= 0):
            raise ValueError(
                f"AnnotationLossSpec.max_length must be a positive int, got "
                f"{type(self.max_length).__name__}={self.max_length!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "loss_type": self.loss_type,
            "weight": float(self.weight),
            "max_length": int(self.max_length),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AnnotationLossSpec":
        # C36 fix: validate the RAW types before float()/int() coercion. Otherwise
        # ``"weight": true`` round-trips through float(True)==1.0 (and
        # ``"max_length": false`` through int(False)==0) and silently changes
        # semantics instead of raising. bool is a subclass of int, so it is
        # excluded explicitly here as well as in __post_init__.
        weight_raw = d.get("weight", 0.5)
        if isinstance(weight_raw, bool) or not isinstance(weight_raw, (int, float)):
            raise ValueError(
                f"AnnotationLossSpec.weight must be a real number, got "
                f"{type(weight_raw).__name__}={weight_raw!r}"
            )
        max_length_raw = d.get("max_length", 32)
        if isinstance(max_length_raw, bool) or not isinstance(max_length_raw, int):
            raise ValueError(
                f"AnnotationLossSpec.max_length must be an int, got "
                f"{type(max_length_raw).__name__}={max_length_raw!r}"
            )
        return cls(
            field=d["field"],
            loss_type=d.get("loss_type", "ce_text"),
            weight=float(weight_raw),
            max_length=int(max_length_raw),
        )

    def batch_key_tokens(self) -> str:
        """Sample-dict / batch-dict key for this annotation's token IDs."""
        return f"annotation_tokens__{self.field}"

    def batch_key_mask(self) -> str:
        """Sample-dict / batch-dict key for this annotation's valid-token mask."""
        return f"annotation_mask__{self.field}"

    def batch_key_weight(self) -> str:
        """Sample-dict / batch-dict key carrying `spec.weight` as a scalar
        tensor. Per-sample scalar lets the model compose loss without a
        schema reference plumbed through the training loop."""
        return f"annotation_weight__{self.field}"

    @staticmethod
    def field_from_tokens_key(key: str) -> str | None:
        """Reverse of `batch_key_tokens` — extract field name or None."""
        prefix = "annotation_tokens__"
        return key[len(prefix):] if key.startswith(prefix) else None
