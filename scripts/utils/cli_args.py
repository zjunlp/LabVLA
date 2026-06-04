"""Argparse wiring for scripts/train.py."""
from __future__ import annotations

import argparse


def _strict_bool(value: str) -> bool:
    """Strict bool parser: only accepts {"true", "false"} (case-insensitive).

    A lenient parser would coerce "1"/"yes"/"on" or a typo like "ture" to
    False, silently flipping a training-semantic flag (knowledge_isolation,
    freeze_vision_encoder, ...) on a single bad character. Reject up front so
    operator mistakes surface at argparse rather than in a mis-trained ckpt.
    """
    s = str(value).strip().lower()
    if s == "true":
        return True
    if s == "false":
        return False
    raise argparse.ArgumentTypeError(
        f"expected 'true' or 'false' (case-insensitive), got {value!r}. "
        f"Strict to avoid silent flip of training-semantic flags."
    )


def _positive_int(value: str) -> int:
    """Strict positive-int parser (value >= 1).

    Used for --gradient_accumulation_steps: 0 or negative produces undefined
    training semantics (global_batch = micro * 0 * N = 0, DeepSpeed
    train_batch_size of 0 divides-by-zero in engine init). Reject at argparse.
    """
    iv = int(value)
    if iv < 1:
        raise argparse.ArgumentTypeError(f"expected an integer >= 1, got {iv}")
    return iv


def parse_args():
    parser = argparse.ArgumentParser(description="LabVLA Training")

    # Model config
    parser.add_argument("--vlm_pretrained_path", type=str,
                        default="/all-flash-data/vlm/Qwen3-VL-4B-Instruct")
    parser.add_argument("--dtype", type=str, default="bfloat16")

    # DiT Action Head config
    parser.add_argument("--dit_num_layers", type=int, default=18)
    parser.add_argument("--dit_num_heads", type=int, default=8)
    parser.add_argument("--dit_head_dim", type=int, default=128)
    parser.add_argument("--dit_dropout", type=float, default=0.0)
    parser.add_argument("--dit_interleave_self_attention", type=_strict_bool, default=True,
                        help="Spirit-style interleaved self-attn in DiT (odd-idx blocks self-attn, "
                             "even-idx cross-attn). Default true (2026-05-13).")
    parser.add_argument("--dit_layerwise_vlm_features", type=_strict_bool, default=False,
                        help="starVLA-style layerwise VLM features: each cross-attn DiT block reads a "
                             "DIFFERENT VLM hidden-state layer (last-N of VLM, one per cross-attn block) "
                             "instead of all cross-attn blocks sharing VLM's last layer. Adds ~1.7 GB / GPU "
                             "(retains all VLM hidden states for cross-attn). posttrain+KI=False only.")

    # Training config
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--max_state_dim", type=int, default=32)
    parser.add_argument("--max_action_dim", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    # Default 2 (not 8) so launchers that omit this don't get workers x prefetch
    # = tens of in-flight batches per rank and the host RAM/SHM pressure that
    # follows. Runs that want deep prefetch set this explicitly.
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=2,
                        help="DataLoader workers' per-worker prefetch. Default 2 keeps "
                             "host RAM/SHM bounded; raise to 4-8 only when you have "
                             "verified data_s>0 in logs and free RAM headroom.")
    parser.add_argument("--prefetch_buffer", type=_strict_bool, default=True)
    parser.add_argument("--prefetch_buffer_size", type=int, default=5)
    parser.add_argument("--steps", type=int, default=280000)
    parser.add_argument("--save_freq", type=int, default=10000)
    # Default 8 (not 0=unbounded): each ckpt carries DeepSpeed ZeRO-2 sharded
    # optimizer state + EMA + model, so unbounded retention fills tens of TB.
    # Multiples of 50000 are kept as long-run anchors; pass --max_keep_ckpts=0
    # to opt back into unbounded retention.
    parser.add_argument("--max_keep_ckpts", type=int, default=8,
                        help="Keep the newest N non-sticky checkpoints; 0 disables rotation. "
                             "Multiples of 50000 are kept as long-run anchors. R9 M9-02 "
                             "raised the default 0→8 to bound disk usage; legacy callers "
                             "wanting infinite retention must pass --max_keep_ckpts=0.")
    parser.add_argument("--log_freq", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_height", type=int, default=224)
    parser.add_argument("--image_width", type=int, default=224)

    # Optimizer
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--decay_steps", type=int, default=280000)
    parser.add_argument("--decay_lr", type=float, default=2.5e-6)
    # Opt-in for the cosine scheduler to auto-rewrite warmup/decay when the
    # optimizer-step horizon (--steps // gradient_accumulation_steps) is shorter
    # than --decay_steps. Default false = fail-closed (raise) so a decay length
    # that can't run as written surfaces loudly.
    parser.add_argument("--allow_lr_auto_scale", type=_strict_bool, default=False)
    parser.add_argument("--vlm_lr", type=float, default=5e-5,
                        help="VLM backbone learning rate (directly specified)")
    parser.add_argument("--dit_lr", type=float, default=1e-4,
                        help="DiT action head learning rate")

    # EMA of trainable params, aligned with openpi (finetune 0.99, pretrain
    # 0.999, LoRA finetune None). Pass 0 or negative to disable.
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.0,
        help=(
            "Exponential Moving Average decay over trainable parameters. "
            "0 (default) disables EMA. openpi-aligned values: 0.99 for finetune "
            "(default in their TrainConfig), 0.999 for pretrain. EMA buffer "
            "is fp32 and lives on the parameter device. Saved to "
            "<ckpt>/training_state/ema_state.pt every save_freq steps."
        ),
    )

    # Gradient accumulation; _positive_int enforces >= 1.
    parser.add_argument("--gradient_accumulation_steps", type=_positive_int, default=1)

    # Freeze control
    parser.add_argument("--freeze_vision_encoder", type=_strict_bool, default=True)
    parser.add_argument("--train_expert_only", type=_strict_bool, default=False)
    parser.add_argument("--train_vlm_only", type=_strict_bool, default=False)
    parser.add_argument("--gradient_checkpointing", type=_strict_bool, default=False)
    parser.add_argument("--gc_visual_encoder", type=_strict_bool, default=True,
                        help="Vision encoder gradient checkpointing (only effective when gradient_checkpointing=true)")
    parser.add_argument("--gc_language_model", type=_strict_bool, default=True,
                        help="VLM Language Model gradient checkpointing (only effective when gradient_checkpointing=true)")
    parser.add_argument("--gc_dit", type=_strict_bool, default=False,
                        help="DiT Action Head gradient checkpointing (only effective when gradient_checkpointing=true)")
    parser.add_argument(
        "--compile_model", type=_strict_bool, default=False,
        help=(
            "Enables torch.compile on `sample_actions` (INFERENCE path) ONLY. "
            "Training `forward()` stays under `@dynamo.disable` for "
            "DeepSpeed + flash_attn compatibility — this flag does NOT speed "
            "up training. See modeling_labvla.py L224-235."
        ),
    )
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                        choices=["eager", "sdpa", "flash_attention_2"])

    # ---- pi0.5 VLM pretrain + Knowledge Isolation (opt-in) ----
    parser.add_argument("--training_phase", type=str, default="posttrain",
                        choices=["posttrain", "vlm_pretrain"],
                        help="posttrain=current VLM+DiT+MSE (default); "
                             "vlm_pretrain=π0.5 VLM-only with FAST token CE (no DiT)")
    parser.add_argument("--knowledge_isolation", type=_strict_bool, default=False,
                        help="posttrain only: detach VLM→DiT grad + add FAST CE branch")
    parser.add_argument("--use_fast_tokenizer", type=_strict_bool, default=False,
                        help="DataLoader computes FAST action tokens (required for vlm_pretrain / KI)")
    parser.add_argument("--fast_tokenizer_path", type=str,
                        default="/all-flash-data/Embodied_models/fast")
    parser.add_argument("--discrete_action_vocab_size", type=int, default=2048)
    parser.add_argument("--discrete_action_max_length", type=int, default=240,
                        help="FAST token sequence upper bound (truncated above). "
                             "T10 (2026-06-04): raised default 224→240 so it sits "
                             "above the historical upper bound (max observed=229) and "
                             "stays consistent with configuration_labvla.py / "
                             "robointer_token_budget.py (both 240); the CLI default "
                             "otherwise silently overrides the config default. "
                             "Production 4ds CPU sample p99≈175. Launchers that need "
                             "smaller sequences (smoke tests, single-task finetune) "
                             "can still pass --discrete_action_max_length=128.")
    parser.add_argument("--ki_mse_weight", type=float, default=10.0,
                        help="α weight on MSE per π0.5 Eq (1): total = α·MSE + CE (α=10 per App B.3)")
    parser.add_argument("--discretize_state_in_vlm_pretrain", type=_strict_bool,
                        default=True,
                        help="π0.5 §B.1: discretize observation.state into 256 bins → text "
                             "tokens prepended to the language prompt (vlm_pretrain only). "
                             "Default True for paper-faithful pretrain. Posttrain ignores "
                             "this flag (state stays continuous via state_proj into DiT).")
    # Expose pi05_block_attention_mask via CLI so resume from a ckpt trained
    # with it can re-enable the same attention pattern. Default False keeps the
    # FA2 + causal-prefix throughput main line (4D-mask + SDPA regresses
    # throughput and carries 4D-mask memory overhead); paper-faithful ablation
    # opt-in only.
    parser.add_argument("--pi05_block_attention_mask",
                        type=_strict_bool, default=False,
                        help="OPT-IN paper-faithful π0.5 block attention "
                             "(Appendix B.1 + Figure 11): prefix full "
                             "bidirectional + FAST/ann causal-with-prefix. "
                             "Default False — keeps FA2 + causal throughput "
                             "main line. When True, must also set "
                             "--attn_implementation sdpa (FA2 silently "
                             "ignores arbitrary 4D patterns and the config "
                             "guard rejects the combo).")

    # Dataset (LeRobot v2.1)
    parser.add_argument("--repo_ids", type=str, required=True,
                        help="Comma-separated list of LeRobot v2.1 repo IDs.")
    parser.add_argument("--dataset_schema", type=str, default=None,
                        help="Schema spec for --repo_ids. Scalar: applies to all repos; "
                             "or comma-map 'repoA=name1,repoB=/path/to/x.py:SCHEMA' "
                             "for per-repo override. Each spec is resolved via "
                             "schema.resolve() (registered name | dotted module | "
                             "absolute .py path). If omitted, falls back to the "
                             "dataset's meta/labvla_manifest.json then info.json auto-infer.")
    # --data_root is required: dataset_build reads
    # Path(root or data_root)/"meta/info.json" before any adapter resolution,
    # so a None root + None data_root raises deep in dataset build. Marking it
    # required surfaces the contract at argparse time.
    parser.add_argument("--data_root", type=str, required=True,
                        help="Local dataset root directory. Must be set — dataset_build "
                             "reads <data_root>/<repo>/meta/info.json before any adapter "
                             "resolution. (R9 M9-05: was optional, now required.)")
    parser.add_argument("--external_stats_path", type=str, default=None,
                        help="Path to pre-computed stats.json for normalization (optional, "
                             "applied globally to all repos). Superseded by --external_stats_map "
                             "per-repo entries when both provided.")
    parser.add_argument("--external_stats_map", type=str, default=None,
                        help="Per-repo stats override: 'repo_id=path,repo_id=path'. "
                             "Used when a repo (e.g. Collection topology like RoboCOIN) "
                             "has no stats.json of its own but needs one from elsewhere "
                             "(e.g. from a pre-merged flat version). Falls back to "
                             "fallback chain: repo own > parent > external_stats_map.")
    parser.add_argument("--dataset_weights", type=str, default=None,
                        help="Comma-separated weights for multi-dataset training (optional)")
    parser.add_argument("--homogeneous_mixture_batches", type=_strict_bool, default=False,
                        help="For PI0MixtureDataset, sample each mini-batch from one source dataset. "
                             "This avoids heterogeneous-schema collate field-union activation blow-up.")
    parser.add_argument("--trim_token_padding_to_batch", type=_strict_bool, default=False,
                        help="Trim all-pad prompt/FAST/annotation token tails after collation. "
                             "Preserves masks and supervised tokens while reducing per-batch LM sequence length.")
    parser.add_argument("--source_shape_convergence", type=_strict_bool, default=False,
                        help="Phase-C: allow FAST/annotation transforms to emit source/sample-local "
                             "token lengths; the LabVLA collator pads them to the batch-local max.")
    parser.add_argument("--data_error_skip", type=_strict_bool, default=True,
                        help="Skip/retry bad dataset samples instead of crashing DataLoader workers.")
    parser.add_argument("--data_error_skip_max_attempts", type=int, default=64,
                        help="Per-sample adjacent retry window when --data_error_skip is true.")
    parser.add_argument("--data_error_log_first", type=int, default=20,
                        help="Log only the first N skipped-sample warnings per dataset worker copy.")
    parser.add_argument("--action_supervision_skip_repos", type=str, default="",
                        help="Comma-separated repo IDs whose FAST action CE targets are dropped in "
                             "vlm_pretrain. Use only when keeping annotation/text supervision from "
                             "a repo whose action gripper semantics are intentionally excluded from "
                             "the action-target mix.")
    parser.add_argument("--streaming", type=_strict_bool, default=False,
                        help="Reserved (currently no-op). PI0MixtureDataset (the active "
                             "multi-repo dataset) honors --dataset_weights natively in "
                             "non-streaming mode. This flag is kept for future wiring "
                             "of StreamingLeRobotDataset (IterableDataset) when needed.")
    parser.add_argument("--image_augmentation", type=_strict_bool, default=True,
                        help="Enable random image augmentation")
    parser.add_argument("--action_mode", type=str, default="delta",
                        choices=["abs", "delta"],
                        help="Action mode: 'abs' for absolute, 'delta' for delta increments")
    # Per-segment normalization toggles (A/B testing). False = raw passthrough
    # for those dims. Default true preserves mixed-norm behavior
    # (arm mean_std + gripper q01_q99).
    parser.add_argument("--normalize_arm_joints", type=_strict_bool, default=True,
                        help="Apply mean_std normalization to arm joint dims. False → raw passthrough.")
    parser.add_argument("--normalize_gripper", type=_strict_bool, default=True,
                        help="Apply q01_q99 normalization to gripper dims. False → raw passthrough.")
    parser.add_argument("--gripper_norm_mode", type=str, default=None,
                        choices=["q01_q99", "mean_std", "noop"],
                        help="Explicit gripper normalization mode. Overrides --normalize_gripper when set.")

    # LabUtopia-side gripper canonicalize (for OXE-pretrain warmstart). When
    # True, raw gripper width is snapped to {0, max_width} before
    # NormalizeTransformFn, aligning LabUtopia to the OXE ckpt's binary {0, 1}
    # convention. Requires stats overrides so q01[grip]=0, q99[grip]=max_width.
    parser.add_argument("--snap_gripper_to_binary", type=_strict_bool, default=False,
                        help="Snap LabUtopia gripper width to {0, max_width} before normalize.")
    parser.add_argument("--gripper_max_width", type=float, default=0.04,
                        help="Physical max gripper width in meters (Franka Panda: 0.04).")
    parser.add_argument("--gripper_canonical_dim", type=int, default=7,
                        help="Canonical index of gripper in the state/action vector (default 7).")

    # Gripper loss weighting. In a uniform action-MSE average a single gripper
    # scalar gets ~1/D weight, so the 7 joint dims outweigh the gripper dim 7x;
    # up-weighting helps tasks that fail on "grip then drop". Default 1.0 is a
    # no-op; >1.0 scales gripper-dim MSE before the mean.
    parser.add_argument("--gripper_loss_weight", type=float, default=1.0,
                        help="Per-dim multiplier on gripper action dims in flow-matching MSE "
                             "(applied to mse_per_elem before averaging). 1.0 = no-op (default); "
                             "5.0 is the analysis report's suggested first-try value for "
                             "LabUtopia TransportBeaker. Only takes effect when schema declares "
                             "gripper_action_dims (auto-populated from active schema).")

    # Output
    parser.add_argument("--output_dir", type=str,
                        default="/data/rbc/LabVLA/outputs")
    parser.add_argument("--job_name", type=str, default=None)

    # wandb
    parser.add_argument("--wandb_enable", type=_strict_bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="labvla_pretrain")
    parser.add_argument("--wandb_mode", type=str, default="offline")

    # Diagnostics
    parser.add_argument("--mem_trace", type=_strict_bool, default=False,
                        help="Enable GPU memory growth tracing log (every 50 steps for first 600 steps)")
    parser.add_argument("--dist_timeout_seconds", type=int, default=3600,
                        help="PyTorch distributed process-group timeout.")

    # Resume
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--load_weights_only",
        type=_strict_bool,
        default=False,
        help=(
            "Controls how --resume is interpreted. "
            "When true: load ONLY model weights from the checkpoint (no optimizer state, "
            "no scheduler state, no step counter, no RNG state); training starts from "
            "step 0 with fresh optimizer/scheduler/RNG. Use this for fine-tuning from a "
            "pretrained checkpoint (weights-only warmstart). "
            "When false (default): full resume of a prior training run — optimizer state, "
            "scheduler state, step counter, and RNG state are all restored from the "
            "checkpoint's training_state/training_state.pt. Use this to resume an "
            "interrupted training run exactly where it left off."
        ),
    )
    parser.add_argument(
        "--allow_legacy_partial_resume",
        type=_strict_bool,
        default=False,
        help=(
            "Allow pre-fix checkpoints without accelerate_state to resume with "
            "best-effort model/scheduler/RNG loading. Optimizer state may be "
            "partial under DeepSpeed ZeRO; default false prevents silent "
            "non-full resume."
        ),
    )
    parser.add_argument(
        "--resumable_dataloader",
        type=_strict_bool,
        default=True,
        help="Use deterministic batch samplers so future full resume can continue data order.",
    )

    args = parser.parse_args()

    # The training loop computes current_step % save_freq; save_freq <= 0 would
    # ZeroDivisionError mid-training. Reject at argparse; pass save_freq > steps
    # to effectively disable periodic saving.
    if int(getattr(args, "save_freq", 1)) <= 0:
        parser.error(
            f"--save_freq must be a positive integer (got {args.save_freq}). "
            f"Pass --save_freq > --steps to effectively disable periodic "
            f"saving."
        )

    # gripper_loss_weight is a per-dim MSE multiplier: 0 zeros out gripper
    # supervision, negatives flip the gradient sign (optimizing the gripper to
    # be maximally wrong). The flag exists to boost gripper learning, so <= 0 is
    # always operator error.
    if float(getattr(args, "gripper_loss_weight", 1.0)) <= 0:
        parser.error(
            f"--gripper_loss_weight must be > 0 (got {args.gripper_loss_weight}). "
            f"Pass 1.0 for no-op weighting. 0 zeros out gripper supervision; "
            f"negative values flip the gradient sign on gripper MSE."
        )

    # BackgroundBatchBuffer uses queue.Queue(maxsize=prefetch_buffer_size), and
    # maxsize=0 is UNBOUNDED (an uncapped producer that can OOM the host), not
    # "no buffer". Reject explicitly; use --prefetch_buffer false to disable.
    if int(getattr(args, "prefetch_buffer_size", 5)) < 1:
        parser.error(
            f"--prefetch_buffer_size must be >= 1 (got {args.prefetch_buffer_size}). "
            f"Python queue.Queue(maxsize=0) is UNBOUNDED (not disabled). "
            f"Use --prefetch_buffer false to fully disable the buffer."
        )

    # float() accepts nan/inf/-inf and negatives for hyperparameters where they
    # are meaningless (negative lr flips gradient direction; NaN clip norm
    # disables clipping mid-train). Reject up front so misconfigured launchers
    # fail at argparse, not midway through CUDA warmup.
    #
    # Per-param semantics:
    #   strict_positive=True  -> must be > 0 (lr family; 0 makes every
    #                           optimizer.step() a no-op).
    #   strict_positive=False -> 0 allowed (weight_decay=0 disables L2;
    #                           grad_clip_norm=0 disables clipping;
    #                           ki_mse_weight=0 disables alpha scaling;
    #                           ema_decay=0 disables EMA).
    #   max=None -> unbounded above. max=1.0 -> must be in [0, 1].
    import math as _math
    _float_hp_constraints = [
        # (arg_name, strict_positive, max_inclusive)
        ("lr",              True,  None),
        ("vlm_lr",          True,  None),
        ("dit_lr",          True,  None),
        ("decay_lr",        True,  None),
        ("weight_decay",    False, None),
        ("grad_clip_norm",  False, None),
        ("ki_mse_weight",   False, None),
        ("dit_dropout",     False, 1.0),
        ("ema_decay",       False, 1.0),
    ]
    for _name, _strict_pos, _hi in _float_hp_constraints:
        _v = getattr(args, _name, None)
        if _v is None:
            continue
        try:
            _vf = float(_v)
        except (TypeError, ValueError):
            parser.error(f"--{_name} must be a number (got {_v!r}).")
        if not _math.isfinite(_vf):
            parser.error(
                f"--{_name} must be finite (got {_vf!r}). NaN/Inf "
                f"training hyperparameters silently corrupt optimizer state."
            )
        if _strict_pos:
            if _vf <= 0:
                parser.error(
                    f"--{_name} must be > 0 (got {_vf}). "
                    f"0 makes every optimizer step a no-op; negative "
                    f"values flip the gradient direction."
                )
        else:
            if _vf < 0:
                parser.error(
                    f"--{_name} must be >= 0 (got {_vf}). Negative "
                    f"values flip or invert the corresponding signal."
                )
        if _hi is not None and _vf > _hi:
            parser.error(
                f"--{_name} must be <= {_hi} (got {_vf}). "
                f"Values out of [0, {_hi}] break the parameter's "
                f"defined semantics."
            )

    return args
