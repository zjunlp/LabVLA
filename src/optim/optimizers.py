#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import abc
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import draccus
import torch
from safetensors.torch import load_file, save_file

from dataset.utils import flatten_dict, unflatten_dict, write_json
from utils.constants import (
    OPTIMIZER_PARAM_GROUPS,
    OPTIMIZER_STATE,
)
from utils.io_utils import deserialize_json_into_object

logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig(draccus.ChoiceRegistry, abc.ABC):
    lr: float
    weight_decay: float
    grad_clip_norm: float

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @classmethod
    def default_choice_name(cls) -> str | None:
        return "adam"

    @abc.abstractmethod
    def build(self) -> torch.optim.Optimizer | dict[str, torch.optim.Optimizer]:
        """Build the optimizer for this config.

        Most subclasses return a single ``torch.optim.Optimizer``; only
        ``MultiAdamConfig`` returns a ``dict[str, torch.optim.Optimizer]``. The
        union return type keeps interface compatibility with multi-optimizer
        setups (e.g. policy + value in RL).
        """
        raise NotImplementedError


@OptimizerConfig.register_subclass("adam")
@dataclass
class AdamConfig(OptimizerConfig):
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0

    def build(self, params: dict) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        kwargs.pop("grad_clip_norm")
        return torch.optim.Adam(params, **kwargs)


@OptimizerConfig.register_subclass("adamw")
@dataclass
class AdamWConfig(OptimizerConfig):
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    grad_clip_norm: float = 10.0

    def build(self, params: dict) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        kwargs.pop("grad_clip_norm")
        return torch.optim.AdamW(params, **kwargs)


@OptimizerConfig.register_subclass("sgd")
@dataclass
class SGDConfig(OptimizerConfig):
    lr: float = 1e-3
    momentum: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0

    def build(self, params: dict) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        kwargs.pop("grad_clip_norm")
        return torch.optim.SGD(params, **kwargs)


@OptimizerConfig.register_subclass("labvla-adamw")
@dataclass
class LabVLAAdamWConfig(OptimizerConfig):
    """AdamW optimizer with grouped learning rates for LabVLA DiT architecture.

    Parameter Groups:
        - Group 0 (vlm): VLM backbone parameters at vlm_lr
        - Group 1 (dit_action_head): DiT action head at dit_lr
        - Group 2 (other): Projections, compressor, etc. at full lr
    """

    lr: float = 5e-5
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    vlm_lr: float = 5e-5
    dit_lr: float = 1e-4

    def build(self, params: dict) -> torch.optim.Optimizer:
        assert isinstance(params, dict), "LabVLAAdamW requires named_parameters() dict as input."

        # Within each of the 4 semantic groups (vlm/dit/ki/other), split into
        # decay/no_decay. ndim<=1 params (biases, norm weights, scalar
        # embeddings) get NO weight decay (GPT/pi0/Qwen convention) — decaying
        # norm/bias measurably hurts long-run stability.
        def _no_decay(name: str, p: torch.nn.Parameter) -> bool:
            if p.ndim <= 1:
                return True
            nl = name.lower()
            return nl.endswith(".bias") or "norm" in nl

        buckets: dict[str, dict[str, list[torch.nn.Parameter]]] = {
            "vlm": {"decay": [], "no_decay": []},
            "dit_action_head": {"decay": [], "no_decay": []},
            "ki": {"decay": [], "no_decay": []},
            "other": {"decay": [], "no_decay": []},
        }

        dit_side_modules = {
            "dit_action_head",
            "proj_vlm_to_dit",
            "state_proj",
            "action_in_proj",
            "action_out_proj",
        }
        ki_modules = {"ki_head", "state_vlm_proj"}

        for name, p in params.items():
            if not p.requires_grad:
                continue
            root = name.split(".", 1)[0]
            if root == "model" and "." in name:
                root = name.split(".", 2)[1]
            if root in dit_side_modules:
                bucket = "dit_action_head"
            elif root in ki_modules:
                # π0.5 / KI randomly-initialized modules: use dit_lr for fast convergence.
                bucket = "ki"
            elif root == "vlm":
                bucket = "vlm"
            else:
                bucket = "other"
            subkey = "no_decay" if _no_decay(name, p) else "decay"
            buckets[bucket][subkey].append(p)

        group_lrs = {
            "vlm": self.vlm_lr,
            "dit_action_head": self.dit_lr,
            "ki": self.dit_lr,
            "other": self.lr,
        }

        param_groups = []
        for gname, sub in buckets.items():
            lr = group_lrs[gname]
            if sub["decay"]:
                param_groups.append({
                    "params": sub["decay"],
                    "lr": lr,
                    "weight_decay": self.weight_decay,
                    "name": gname,
                })
            if sub["no_decay"]:
                param_groups.append({
                    "params": sub["no_decay"],
                    "lr": lr,
                    "weight_decay": 0.0,
                    "name": f"{gname}_no_decay",
                })

        return torch.optim.AdamW(
            param_groups,
            betas=self.betas,
            eps=self.eps,
        )


@OptimizerConfig.register_subclass("xvla-adamw")
@dataclass
class XVLAAdamWConfig(OptimizerConfig):
    """Custom AdamW optimizer for XVLA with differential learning rates.

    The Vision-Language Model (VLM) is trained with 1/10 of the base learning rate
    for stable optimization, while all other components use the full LR.

    This LR ratio is crucial for achieving strong and stable finetuning performance.

    Soft-prompts can optionally use a separate learning rate with warm-up support.
    Set `soft_prompt_lr_scale` to a value < 1.0 (e.g., 0.1) to start soft-prompts
    at a lower LR. Combine with a warmup scheduler for optimal results.

    Note:
        Completely matching official reported performance may require an additional
        warm-up LR schedule for soft-prompts, which can bring minor improvements.
        When `soft_prompt_warmup_lr_scale` is set, soft-prompts start at
        `lr * soft_prompt_warmup_lr_scale` and should be warmed up via the scheduler.

    Parameter Groups:
        - Group 0 (vlm): VLM parameters at lr * 0.1, weight_decay * 0.1
        - Group 1 (soft_prompts): Soft-prompt parameters at lr * soft_prompt_lr_scale
        - Group 2 (other): All other parameters at full lr
    """

    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0
    # Soft-prompt specific settings
    soft_prompt_lr_scale: float = 1.0  # Scale factor for soft-prompt LR (1.0 = same as base LR)
    soft_prompt_warmup_lr_scale: float | None = None  # If set, start soft-prompts at this scale (e.g., 0.01)

    def build(self, params: dict) -> torch.optim.Optimizer:
        """
        Build AdamW optimizer with differential learning rates.

        Expects `named_parameters()` as input (dict of name -> param).
        Applies:
        - lr * 0.1 for all VLM-related parameters
        - lr * soft_prompt_lr_scale for soft-prompt parameters (with optional warmup)
        - full lr for all other parameters

        Args:
            params: Dictionary of parameter names to parameters (from named_parameters())

        Returns:
            AdamW optimizer with parameter groups for VLM, soft-prompts, and other components
        """
        assert isinstance(params, dict), "Custom LR optimizer requires `named_parameters()` as inputs."

        vlm_group, soft_prompt_group, other_group = [], [], []
        for name, p in params.items():
            if not p.requires_grad:
                continue
            if "vlm" in name.lower():
                vlm_group.append(p)
            elif "soft_prompt" in name.lower():
                soft_prompt_group.append(p)
            else:
                other_group.append(p)

        # Determine soft-prompt LR
        soft_prompt_lr = self.lr * self.soft_prompt_lr_scale
        if self.soft_prompt_warmup_lr_scale is not None:
            # Start at warmup scale, scheduler will warm up to soft_prompt_lr
            soft_prompt_lr = self.lr * self.soft_prompt_warmup_lr_scale

        param_groups = [
            {
                "params": vlm_group,
                "lr": self.lr * 0.1,
                "weight_decay": self.weight_decay * 0.1,
                "name": "vlm",
            },
            {
                "params": soft_prompt_group,
                "lr": soft_prompt_lr,
                "weight_decay": self.weight_decay,
                "name": "soft_prompts",
            },
            {
                "params": other_group,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "name": "other",
            },
        ]

        # Filter out empty groups
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        return torch.optim.AdamW(
            param_groups,
            betas=self.betas,
            eps=self.eps,
        )


@OptimizerConfig.register_subclass("multi_adam")
@dataclass
class MultiAdamConfig(OptimizerConfig):
    """Configuration for multiple Adam optimizers with different parameter groups.

    This creates a dictionary of Adam optimizers, each with its own hyperparameters.

    Args:
        lr: Default learning rate (used if not specified for a group)
        weight_decay: Default weight decay (used if not specified for a group)
        optimizer_groups: Dictionary mapping parameter group names to their hyperparameters
        grad_clip_norm: Gradient clipping norm
    """

    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0
    optimizer_groups: dict[str, dict[str, Any]] = field(default_factory=dict)

    def build(self, params_dict: dict[str, list]) -> dict[str, torch.optim.Optimizer]:
        """Build multiple Adam optimizers.

        Args:
            params_dict: Dictionary mapping parameter group names to lists of parameters
                         The keys should match the keys in optimizer_groups

        Returns:
            Dictionary mapping parameter group names to their optimizers
        """
        # Require params_dict.keys() == optimizer_groups.keys() exactly. Failing
        # open let a typo'd group silently run with default hyperparameters (or
        # a declared group be dropped with no optimizer and no error).
        if self.optimizer_groups:
            declared = set(self.optimizer_groups.keys())
            provided = set(params_dict.keys())
            undeclared = provided - declared           # params with no config
            unused = declared - provided               # config with no params
            if undeclared or unused:
                raise ValueError(
                    "MultiAdamConfig group-name mismatch: "
                    f"params without optimizer_groups config={sorted(undeclared)}, "
                    f"optimizer_groups declared but no params provided={sorted(unused)}. "
                    "params_dict.keys() must match optimizer_groups.keys() exactly."
                )

        optimizers = {}

        for name, params in params_dict.items():
            # Reject declared-but-empty groups (can't build an optimizer).
            if not params:
                raise ValueError(
                    f"MultiAdamConfig group '{name}' has no parameters; "
                    "cannot build an optimizer for an empty parameter group."
                )

            group_config = self.optimizer_groups.get(name, {})
            optimizer_kwargs = {
                "lr": group_config.get("lr", self.lr),
                "betas": group_config.get("betas", (0.9, 0.999)),
                "eps": group_config.get("eps", 1e-5),
                "weight_decay": group_config.get("weight_decay", self.weight_decay),
            }

            optimizers[name] = torch.optim.Adam(params, **optimizer_kwargs)

        return optimizers


def save_optimizer_state(
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer], save_dir: Path
) -> None:
    """Save optimizer state to disk.

    Args:
        optimizer: Either a single optimizer or a dictionary of optimizers.
        save_dir: Directory to save the optimizer state.
    """
    if isinstance(optimizer, dict):
        for name, opt in optimizer.items():
            optimizer_dir = save_dir / name
            optimizer_dir.mkdir(exist_ok=True, parents=True)
            _save_single_optimizer_state(opt, optimizer_dir)
    else:
        _save_single_optimizer_state(optimizer, save_dir)


def _save_single_optimizer_state(optimizer: torch.optim.Optimizer, save_dir: Path) -> None:
    """Save a single optimizer's state to disk."""
    state = optimizer.state_dict()
    param_groups = state.pop("param_groups")
    flat_state = flatten_dict(state)
    save_file(flat_state, save_dir / OPTIMIZER_STATE)
    write_json(param_groups, save_dir / OPTIMIZER_PARAM_GROUPS)


def load_optimizer_state(
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer],
    save_dir: Path,
    strict: bool = True,
) -> torch.optim.Optimizer | dict[str, torch.optim.Optimizer]:
    """Load optimizer state from disk.

    Args:
        optimizer: Either a single optimizer or a dictionary of optimizers.
        save_dir: Directory to load the optimizer state from.
        strict: When ``True`` (default), a declared sub-optimizer whose state
            directory is missing is a hard error. When ``False``, the missing
            sub-optimizer is left freshly-built and a WARNING is logged.

    Returns:
        The updated optimizer(s) with loaded state.
    """
    if isinstance(optimizer, dict):
        loaded_optimizers = {}
        for name, opt in optimizer.items():
            optimizer_dir = save_dir / name
            if optimizer_dir.exists():
                loaded_optimizers[name] = _load_single_optimizer_state(opt, optimizer_dir)
            else:
                # Fail loud by default: a missing sub-optimizer dir would
                # otherwise resume "part state restored, part from zero"
                # silently. strict=False falls back to a fresh optimizer.
                if strict:
                    raise FileNotFoundError(
                        f"Missing optimizer state for declared sub-optimizer "
                        f"'{name}' (expected {optimizer_dir}). Refusing to silently "
                        f"resume with it freshly-initialized. Pass strict=False to "
                        f"allow a partial resume."
                    )
                logger.warning(
                    "Missing optimizer state for sub-optimizer '%s' (expected %s); "
                    "starting it from a freshly-built optimizer (strict=False).",
                    name,
                    optimizer_dir,
                )
                loaded_optimizers[name] = opt
        return loaded_optimizers
    else:
        return _load_single_optimizer_state(optimizer, save_dir)


def _load_single_optimizer_state(optimizer: torch.optim.Optimizer, save_dir: Path) -> torch.optim.Optimizer:
    """Load a single optimizer's state from disk."""
    current_state_dict = optimizer.state_dict()
    flat_state = load_file(save_dir / OPTIMIZER_STATE)
    state = unflatten_dict(flat_state)

    # 'state' may be absent for newly created optimizers.
    if "state" in state:
        loaded_state_dict = {"state": {int(k): v for k, v in state["state"].items()}}
    else:
        loaded_state_dict = {"state": {}}

    if "param_groups" in current_state_dict:
        param_groups = deserialize_json_into_object(
            save_dir / OPTIMIZER_PARAM_GROUPS, current_state_dict["param_groups"]
        )
        loaded_state_dict["param_groups"] = param_groups

    optimizer.load_state_dict(loaded_state_dict)
    return optimizer
