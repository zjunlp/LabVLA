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
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import draccus
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from dataset.utils import write_json
from utils.constants import SCHEDULER_STATE
from utils.io_utils import deserialize_json_into_object


@dataclass
class LRSchedulerConfig(draccus.ChoiceRegistry, abc.ABC):
    num_warmup_steps: int

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractmethod
    def build(self, optimizer: Optimizer, num_training_steps: int) -> LRScheduler | None:
        raise NotImplementedError


@LRSchedulerConfig.register_subclass("diffuser")
@dataclass
class DiffuserSchedulerConfig(LRSchedulerConfig):
    name: str = "cosine"
    num_warmup_steps: int | None = None

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        from diffusers.optimization import get_scheduler

        kwargs = {**asdict(self), "num_training_steps": num_training_steps, "optimizer": optimizer}
        return get_scheduler(**kwargs)


@LRSchedulerConfig.register_subclass("vqbet")
@dataclass
class VQBeTSchedulerConfig(LRSchedulerConfig):
    num_warmup_steps: int
    num_vqvae_training_steps: int
    num_cycles: float = 0.5

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        def lr_lambda(current_step):
            if current_step < self.num_vqvae_training_steps:
                return float(1)
            else:
                adjusted_step = current_step - self.num_vqvae_training_steps
                if adjusted_step < self.num_warmup_steps:
                    return float(adjusted_step) / float(max(1, self.num_warmup_steps))
                progress = float(adjusted_step - self.num_warmup_steps) / float(
                    max(1, num_training_steps - self.num_warmup_steps)
                )
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)))

        return LambdaLR(optimizer, lr_lambda, -1)


@LRSchedulerConfig.register_subclass("cosine_decay_with_warmup")
@dataclass
class CosineDecayWithWarmupSchedulerConfig(LRSchedulerConfig):
    """Used by Physical Intelligence to train pi0 (adopted by LabVLA).

    When ``num_training_steps < num_decay_steps`` the configured recipe cannot
    run as written. By default (``allow_auto_scale=False``) this is treated as a
    configuration error and raises, rather than silently rewriting the schedule
    into a different training recipe. Set ``allow_auto_scale=True`` to opt into
    the historical behavior of scaling warmup/decay to fit the available steps;
    when allowed, a >10% rewrite still escalates to a WARNING.
    """

    num_warmup_steps: int
    num_decay_steps: int
    decay_lr: float
    peak_lr: float | None = None  # unused — per-group base lr is authoritative
    allow_auto_scale: bool = False  # C31 fix: fail-closed unless explicitly opted in

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        actual_warmup_steps = self.num_warmup_steps
        actual_decay_steps = self.num_decay_steps
        self._auto_scaled = False

        if num_training_steps < self.num_decay_steps:
            # Refuse to silently rewrite the recipe unless explicitly allowed:
            # a short run / post-resume step change would otherwise become a
            # different schedule without the caller's consent.
            if not self.allow_auto_scale:
                raise ValueError(
                    "CosineDecayWithWarmup: num_training_steps "
                    f"({num_training_steps}) < num_decay_steps ({self.num_decay_steps}). "
                    "This would auto-rewrite the LR recipe. Refusing by default; set "
                    "allow_auto_scale=True to opt into scaling warmup/decay to fit "
                    "the available training steps."
                )

            # Scale the schedule to fit the available training steps.
            scale_factor = num_training_steps / self.num_decay_steps
            actual_warmup_steps = int(self.num_warmup_steps * scale_factor)
            actual_decay_steps = num_training_steps
            self._auto_scaled = True

            # <=10% drift is typical (shaving total_steps to fit a job slot) and
            # logs at INFO; beyond 10% it's effectively a different recipe, so
            # escalate to a once-per-(config,run) WARNING.
            drift_frac = abs(num_training_steps - self.num_decay_steps) / self.num_decay_steps
            if drift_frac > 0.10:
                from utils.logging_utils import warn_once
                warn_once(
                    logging.getLogger(__name__),
                    ("cosine_autoscale", num_training_steps, self.num_decay_steps),
                    "Auto-scaling LR scheduler with >10%% drift: "
                    "num_training_steps (%d) vs configured num_decay_steps (%d) "
                    "— this is effectively a different training recipe. "
                    "Scaling warmup: %d → %d, decay: %d → %d (scale factor: %.3f)",
                    num_training_steps, self.num_decay_steps,
                    self.num_warmup_steps, actual_warmup_steps,
                    self.num_decay_steps, actual_decay_steps, scale_factor,
                )
            else:
                logging.info(
                    f"Auto-scaling LR scheduler: "
                    f"num_training_steps ({num_training_steps}) < num_decay_steps ({self.num_decay_steps}). "
                    f"Scaling warmup: {self.num_warmup_steps} → {actual_warmup_steps}, "
                    f"decay: {self.num_decay_steps} → {actual_decay_steps} "
                    f"(scale factor: {scale_factor:.3f})"
                )

        # Keep the cosine formula identical to scripts/train.py:lr_lambda (the
        # production curve); a step-0-indexed variant would diverge from it.
        decay_lr = self.decay_lr
        warmup_steps = actual_warmup_steps
        decay_steps = actual_decay_steps

        # Per-group lambdas so every param_group's cosine decay lands at
        # decay_lr ABSOLUTELY. A single shared lambda would end DiT/KI groups
        # (base_lr != args.lr) at the wrong final lr.
        def make_lambda(alpha: float):
            def lr_lambda(step):
                if step < warmup_steps:
                    if step <= 0:
                        return 1.0 / (warmup_steps + 1)
                    frac = 1.0 - step / warmup_steps
                    return (1.0 / (warmup_steps + 1) - 1.0) * frac + 1.0

                decay_progress = min(step - warmup_steps, decay_steps - warmup_steps)
                decay_total = max(1, decay_steps - warmup_steps)
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_progress / decay_total))
                return (1.0 - alpha) * cosine_decay + alpha
            return lr_lambda

        lr_lambdas = [
            make_lambda(decay_lr / max(float(g["lr"]), 1e-12))
            for g in optimizer.param_groups
        ]

        return LambdaLR(optimizer, lr_lambdas, -1)


def save_scheduler_state(scheduler: LRScheduler, save_dir: Path) -> None:
    state_dict = scheduler.state_dict()
    write_json(state_dict, save_dir / SCHEDULER_STATE)


def load_scheduler_state(scheduler: LRScheduler, save_dir: Path) -> LRScheduler:
    state_dict = deserialize_json_into_object(save_dir / SCHEDULER_STATE, scheduler.state_dict())
    scheduler.load_state_dict(state_dict)
    return scheduler
