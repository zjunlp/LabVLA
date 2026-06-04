"""Exponential Moving Average (EMA) of model parameters.

Aligned with openpi (`/all-flash-data/openpi/scripts/train.py:169-173`):

    ema_params = ema_decay * ema_params + (1 - ema_decay) * params

Update is unconditional every step, no warmup. EMA is initialized from the
current params on first call (mirrors openpi `ema_params=params` at init).

Parameter alignment with openpi:
  - pretrain configs (pi0_base / pi05 / pi0_fast):  ema_decay = 0.999
  - finetune default                              :  ema_decay = 0.99
  - LoRA finetune                                 :  ema_decay = None  (disabled)

The implementation tracks ONLY trainable parameters (matches openpi's
`config.trainable_filter`). Buffer dtype is fp32 to avoid bf16/fp16
accumulation drift. Each DDP / ZeRO-2 rank keeps its own EMA copy; since
ZeRO-2 does not shard parameters and Accelerate keeps weights in sync, all
ranks see the same param values and produce identical EMA buffers without
explicit synchronization.

Usage:

    ema = ModelEMA(model, decay=0.99)
    for step in range(num_steps):
        loss.backward()
        optimizer.step()
        ema.update(model)          # call AFTER optimizer.step

    # Evaluate using EMA weights (in-place swap, then restore)
    backup = ema.apply_to(model)
    eval(model)
    ema.restore(model, backup)

    # Checkpointing
    torch.save(ema.state_dict(), ckpt_dir / "ema.pt")
    ema.load_state_dict(torch.load(ckpt_dir / "ema.pt"))
"""
from __future__ import annotations

import logging
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelEMA:
    """Exponential Moving Average over trainable parameters of a nn.Module.

    Args:
        model: The model to track. Only parameters with `requires_grad=True`
            are tracked. EMA buffer is initialized to a fp32 copy of the
            current parameter values (matches openpi init).
        decay: EMA decay factor in (0, 1). openpi default is 0.99 for
            finetune, 0.999 for pretrain. Higher decay → smoother trajectory
            but slower adaptation (effective averaging window ≈ 1 / (1 - decay)
            steps; 0.99 → ~100 steps, 0.999 → ~1000 steps).
        device: Device for the EMA buffer. Defaults to the parameter device.
            Set to "cpu" to save GPU memory at the cost of slower update.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.99,
        device: torch.device | str | None = None,
    ):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self._device_override = device
        # name -> fp32 buffer on the chosen device
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                buf = param.detach().to(torch.float32)
                if device is not None:
                    buf = buf.to(device)
                self.shadow[name] = buf.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """In-place EMA update: shadow = decay * shadow + (1 - decay) * param.

        Skips parameters that are not in the shadow (e.g. dynamically added
        modules or parameters whose `requires_grad` flipped after init).
        """
        one_minus_decay = 1.0 - self.decay
        for name, param in model.named_parameters():
            buf = self.shadow.get(name)
            if buf is None:
                continue
            # Cast param to fp32 on the buffer's device for the linear combination.
            new_value = param.detach().to(buf.device, dtype=torch.float32)
            buf.mul_(self.decay).add_(new_value, alpha=one_minus_decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Copy EMA buffers into the model in-place. Returns a backup of the
        original parameter tensors so callers can ``restore()`` afterward.

        The model's original dtype is preserved (EMA buffer is cast back from
        fp32 to each parameter's native dtype on copy).
        """
        backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            buf = self.shadow.get(name)
            if buf is None:
                continue
            backup[name] = param.detach().clone()
            param.data.copy_(buf.to(device=param.device, dtype=param.dtype))
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        """Restore parameters from a backup produced by ``apply_to``."""
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return the EMA buffer as a plain ``{name: fp32 tensor}`` dict.

        Suitable for ``torch.save``. The tensors share storage with the live
        EMA buffer — callers should not mutate them.
        """
        return {"decay": torch.tensor(self.decay), **self.shadow}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True) -> None:
        """Load EMA buffer from a state dict produced by ``state_dict``.

        The ``decay`` entry is honored if present (overrides the constructor
        argument), so a resumed run uses the same decay it was created with.

        Validates the key sets to avoid silently resuming a "part historical
        EMA + part current params" mix (e.g. after a param rename or LoRA
        toggle):

        - ``strict=True`` (default): raise ``RuntimeError`` if the sets differ.
        - ``strict=False``: WARN but load the intersecting keys.

        The saved ``decay`` scalar is metadata and excluded from the comparison.
        """
        decay = state_dict.get("decay")
        if isinstance(decay, torch.Tensor):
            self.decay = float(decay.item())

        saved_keys = {k for k in state_dict.keys() if k != "decay"}
        shadow_keys = set(self.shadow.keys())
        missing = shadow_keys - saved_keys      # tracked but not in checkpoint
        unexpected = saved_keys - shadow_keys    # in checkpoint but not tracked

        if missing or unexpected:
            def _sample(s: set[str], n: int = 5) -> list[str]:
                return sorted(s)[:n]

            detail = (
                f"EMA state_dict key mismatch: "
                f"{len(missing)} missing (tracked param absent from checkpoint, "
                f"would stay at current params), "
                f"{len(unexpected)} unexpected (checkpoint key not tracked). "
                f"missing_sample={_sample(missing)} "
                f"unexpected_sample={_sample(unexpected)}"
            )
            if strict:
                raise RuntimeError(
                    detail
                    + " — refusing to silently resume a part-historical-EMA / "
                    "part-current-params mix. Pass strict=False to load the "
                    "intersection anyway."
                )
            logger.warning("%s — loading intersection only (strict=False).", detail)

        for name, buf in self.shadow.items():
            saved = state_dict.get(name)
            if saved is None:
                continue
            buf.copy_(saved.to(device=buf.device, dtype=buf.dtype))

    def __len__(self) -> int:
        return len(self.shadow)

    def __repr__(self) -> str:
        n_params = sum(b.numel() for b in self.shadow.values())
        return (
            f"ModelEMA(decay={self.decay}, tracked_params={len(self.shadow)}, "
            f"total_elements={n_params:,})"
        )
