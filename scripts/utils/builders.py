"""Config and optimizer/scheduler builders.

`build_config` mirrors the LabVLAConfig argument wiring; `build_optimizer`
keeps the grouped-LR + cosine-decay setup, including the per-group
absolute-decay-lr behavior.
"""
from __future__ import annotations

import logging

from policies.LabVLA.configuration_labvla import LabVLAConfig


def build_config(args) -> LabVLAConfig:
    config = LabVLAConfig(
        vlm_pretrained_path=args.vlm_pretrained_path,
        dtype=args.dtype,
        chunk_size=args.chunk_size,
        # Lock n_action_steps to chunk_size at train-time (deploy horizon is set
        # separately via serve_labvla.py --output_chunk_size). Without this the
        # dataclass default (50) would trip __post_init__'s
        # n_action_steps <= chunk_size invariant when chunk_size < 50.
        n_action_steps=args.chunk_size,
        max_state_dim=args.max_state_dim,
        max_action_dim=args.max_action_dim,
        # DiT Action Head
        dit_num_layers=args.dit_num_layers,
        dit_num_heads=args.dit_num_heads,
        dit_head_dim=args.dit_head_dim,
        dit_dropout=args.dit_dropout,
        dit_interleave_self_attention=args.dit_interleave_self_attention,
        dit_layerwise_vlm_features=args.dit_layerwise_vlm_features,
        # Freeze control
        freeze_vision_encoder=args.freeze_vision_encoder,
        train_expert_only=args.train_expert_only,
        train_vlm_only=args.train_vlm_only,
        gradient_checkpointing=args.gradient_checkpointing,
        gc_visual_encoder=args.gc_visual_encoder,
        gc_language_model=args.gc_language_model,
        gc_dit=args.gc_dit,
        compile_model=args.compile_model,
        attn_implementation=args.attn_implementation,
        # Optimizer (grouped learning rates)
        optimizer_lr=args.lr,
        optimizer_weight_decay=args.weight_decay,
        optimizer_grad_clip_norm=args.grad_clip_norm,
        optimizer_vlm_lr=args.vlm_lr,
        optimizer_dit_lr=args.dit_lr,
        # Scheduler
        scheduler_warmup_steps=args.warmup_steps,
        scheduler_decay_steps=args.decay_steps,
        scheduler_decay_lr=args.decay_lr,
        # Opt-in cosine warmup/decay auto-scaling (default fail-closed).
        scheduler_allow_auto_scale=getattr(args, "allow_lr_auto_scale", False),
        image_resolution=(args.image_height, args.image_width),
        # π0.5 / KI (opt-in)
        training_phase=args.training_phase,
        action_mode=args.action_mode,
        knowledge_isolation=args.knowledge_isolation,
        use_fast_tokenizer=args.use_fast_tokenizer,
        fast_tokenizer_path=args.fast_tokenizer_path,
        discrete_action_vocab_size=args.discrete_action_vocab_size,
        discrete_action_max_length=args.discrete_action_max_length,
        ki_mse_weight=args.ki_mse_weight,
        discretize_state_in_vlm_pretrain=args.discretize_state_in_vlm_pretrain,
        # Forward the CLI value so resume/ablation runs can opt into the
        # pi0.5 block-wise mask. Parser default stays False (FA2 + causal is
        # the throughput main line).
        pi05_block_attention_mask=args.pi05_block_attention_mask,
        # Mirror gripper_loss_weight here so build_config remains the single
        # source of LabVLAConfig wiring (no post-construction patch in train.py).
        gripper_loss_weight=getattr(args, "gripper_loss_weight", 1.0),
    )
    return config


def build_optimizer(policy, config, num_training_steps: int | None = None):
    """Build LabVLA grouped learning rate optimizer + cosine decay scheduler.

    Parameter groups (implemented by LabVLAAdamWConfig):
      - vlm: VLM backbone -> vlm_lr
      - dit_action_head: DiT -> dit_lr (faster convergence of randomly-init module)
      - other: projection layers/compressors etc. -> lr
    """
    opt_config = config.get_optimizer_preset()
    named_params = dict(
        (policy.model if hasattr(policy, 'model') else policy).named_parameters()
    )
    optimizer = opt_config.build(named_params)

    trainable_count = sum(p.numel() for p in named_params.values() if p.requires_grad)
    if trainable_count == 0:
        raise ValueError("No trainable parameters found. Check freeze settings.")

    for g in optimizer.param_groups:
        n_params = sum(p.numel() for p in g["params"]) / 1e6
        logging.info(f"  Optimizer group '{g.get('name', '?')}': {n_params:.1f}M params, lr={g['lr']:.2e}")

    # Cosine decay with warmup. Builds per-group lambdas so every param_group
    # lands at decay_lr absolutely. A single global alpha = decay_lr/optimizer_lr
    # would let dit/ki end at decay_lr * (group_lr/optimizer_lr) instead.
    from optim.schedulers import CosineDecayWithWarmupSchedulerConfig
    sched_config = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=config.scheduler_warmup_steps,
        num_decay_steps=config.scheduler_decay_steps,
        peak_lr=config.optimizer_lr,
        decay_lr=config.scheduler_decay_lr,
        # Opt-in; default False keeps fail-closed behavior.
        allow_auto_scale=getattr(config, "scheduler_allow_auto_scale", False),
    )
    scheduler = sched_config.build(
        optimizer,
        int(num_training_steps or config.scheduler_decay_steps),
    )
    return optimizer, scheduler
