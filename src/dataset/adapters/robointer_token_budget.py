"""Centralized token-length budget for RoboInter VQA + annotation co-training.

All sequence-length related hyperparameters live here. Adapters / transforms /
schemas / launch scripts MUST read from this dataclass — do NOT hardcode their
own copies. Drift between scripts is the #1 source of silent regressions
(e.g. `discrete_action_max_length=128` in launch script vs `=256` in
scripts/train.py argparse default — the bug we hit on 2026-04-26).

Reference:
- π0.5 paper §B.1 "Robot proprioceptive state is discretized and input to the
  model as text tokens" → state_num_bins=256.
- OpenTau modeling_pi05.py:701 `((state + 1.0) * 128.0).long().clamp(0, 255)`
  → matches our state_num_bins=256.
- FAST tokenizer `physical-intelligence/fast` → vocab_size=2048 (verified via
  AutoProcessor.from_pretrained).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoboInterTokenBudget:
    # ---- State discretization (π0.5 §B.1) ----
    state_num_bins: int = 256

    # ---- VQA — per task family answer length ----
    # Understanding: MCQ "Yes"/"No"/"A-D"; very short.
    vqa_understanding_answer_max_length: int = 8
    # Generation: bbox / single point / contact frame.
    vqa_generation_short_answer_max_length: int = 48
    # Generation/traj_qa: 16 (x,y) waypoints serialized as text.
    vqa_generation_traj_answer_max_length: int = 256
    # Task_planning: free-form short subtask sentence.
    vqa_task_planning_answer_max_length: int = 64
    vqa_question_max_length: int = 128
    # V1 trains on a single selected frame because the Qwen3-VL processor is
    # wired for three fixed image slots shared with robot cameras, not arbitrary
    # multi-image VQA packing.
    vqa_task_planning_num_frames: int = 1

    # ---- Annotation (12 RoboInter types unified into 1 text field) ----
    annotation_unified_max_length: int = 256
    # Q_trace can be 100+ pixel waypoints — downsample to bound token count.
    annotation_trace_max_points: int = 16
    # Skip Q_segmentation: dense pixel mask, too heavy to tokenize as text.
    annotation_skip_fields: tuple[str, ...] = ("annotation.segmentation",)
    # Qwen3-VL grounding format uses integer coords in [0, 1000].
    annotation_coord_scale: int = 1000

    # ---- Action / FAST (sourced from existing LabVLA defaults; do not change) ----
    chunk_size: int = 50
    # Must match (or exceed) configuration_labvla.py runtime default, else this
    # budget under-counts FAST token length. robointer p99=70, robocoin p99=124;
    # legacy 128 silently truncated 3-5% of FAST CE targets.
    discrete_action_max_length: int = 240
    discrete_action_vocab_size: int = 2048  # FAST processor (physical-intelligence/fast)
    max_state_dim: int = 32
    max_action_dim: int = 32

    # ---- Hard caps for the full LM input sequence ----
    # Single-frame action sample budget: 256 image + 80 prompt + 256 ann + 240 FAST
    # + 50 action expert ≈ 882. VQA Task_planning V1 uses one selected frame:
    # 256 image + 128 question + 64 answer ≈ 448. Budget 2048 leaves headroom.
    total_seq_len_hard_cap: int = 2048

    def estimate_action_sample_length(self, num_cams: int = 2) -> int:
        """Estimate full prefix+suffix length for an action+annotation sample."""
        return (
            256 * num_cams                        # vision tokens (Qwen3-VL @ 224x224)
            + 80                                  # state-text + instruction
            + self.annotation_unified_max_length  # unified annotation text
            + self.discrete_action_max_length     # FAST tokens
            + self.chunk_size                     # action-expert tokens (suffix)
        )

    def estimate_vqa_sample_length(self, task_family: str, num_frames: int = 1) -> int:
        """Estimate length for a VQA-only sample (no action, no annotation)."""
        if task_family == "understanding":
            ans = self.vqa_understanding_answer_max_length
        elif task_family == "generation_short":
            ans = self.vqa_generation_short_answer_max_length
        elif task_family == "generation_traj":
            ans = self.vqa_generation_traj_answer_max_length
        elif task_family == "task_planning":
            ans = self.vqa_task_planning_answer_max_length
            num_frames = min(num_frames, self.vqa_task_planning_num_frames)
        else:
            raise ValueError(f"unknown task_family {task_family!r}")
        return 256 * num_frames + self.vqa_question_max_length + ans + 50  # + chat-template overhead

    def __post_init__(self) -> None:
        # Worst-case sanity: every supported sample type must fit under the hard
        # cap. Fail at import time if a hyperparameter drifts in the future.
        for nc in (1, 2, 4):
            est = self.estimate_action_sample_length(nc)
            if est > self.total_seq_len_hard_cap:
                raise ValueError(
                    f"action sample at num_cams={nc} estimated {est} tokens, "
                    f"exceeds total_seq_len_hard_cap={self.total_seq_len_hard_cap}"
                )
        for fam, nf in [
            ("understanding", 1),
            ("generation_short", 1),
            ("generation_traj", 1),
            ("task_planning", self.vqa_task_planning_num_frames),
        ]:
            est = self.estimate_vqa_sample_length(fam, nf)
            if est > self.total_seq_len_hard_cap:
                raise ValueError(
                    f"VQA {fam} at {nf} frames estimated {est} tokens, "
                    f"exceeds total_seq_len_hard_cap={self.total_seq_len_hard_cap}"
                )


# Module-level singleton.
ROBOINTER_BUDGET = RoboInterTokenBudget()
