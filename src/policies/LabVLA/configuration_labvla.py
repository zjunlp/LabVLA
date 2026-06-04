#!/usr/bin/env python

# LabVLA Configuration
# DiT Cross-Attention action head + Flow Matching
# Independent VLM forward + DiT cross-attention denoising

from dataclasses import dataclass, field, replace

from config.default import DatasetConfig
from config.policies import PreTrainedConfig
from config.types import FeatureType, NormalizationMode, PolicyFeature
from optim.optimizers import AdamWConfig, LabVLAAdamWConfig
from utils.constants import OBS_IMAGES
from transforms.core import *
from transforms.annotation_tokenize import AnnotationTokenizeTransformFn
from transforms.build_unified_annotation import BuildUnifiedAnnotationTransformFn
from transforms.agibot_subtask import BuildAgiBotSubtaskTransformFn
from policies.LabVLA.transform_labvla import Qwen3_VLProcessorTransformFn, UnifyLabVLAInputsTransformFn


@DatasetConfig.register_subclass("labvla")
@dataclass
class LabVLADatasetConfig(DatasetConfig):
    height: int = 224
    width: int = 224
    max_state_dim: int = 32
    max_action_dim: int = 32

    # π0.5 VLM pretrain / KI — FAST action tokenizer (default off, transform chain unchanged)
    use_fast_tokenizer: bool = False
    fast_tokenizer_path: str = "/all-flash-data/Embodied_models/fast"
    discrete_action_vocab_size: int = 2048
    discrete_action_max_length: int = 240  # above the historical max observed (229) so FAST CE targets aren't truncated (4ds p99 ≈ 175); encode() still warns past max_length. Smoke tests may pass an explicit 128.
    source_shape_convergence: bool = False

    # π0.5 §B.1: state discretized into 256 bins, prepended to language prompt as text.
    # Mirror of LabVLAConfig.discretize_state_in_vlm_pretrain — kept here too so the
    # transform chain can be specialized at config-build time. Only takes effect when
    # training_phase == "vlm_pretrain". Posttrain keeps state as a continuous tensor
    # (state_proj into DiT) — out of scope for this feature.
    training_phase: str = "posttrain"
    # Persist the data-side action target mode so deployment can decide whether
    # to add current state back to predicted arm deltas.
    action_mode: str = "delta"
    discretize_state_in_vlm_pretrain: bool = True
    # When discretization is on, the language prompt becomes much longer (32 dim ×
    # ~3 chars per bin idx ≈ 100+ tokens). Bump max_length to fit. 192 covers
    # 32-dim discretized state + the original instruction tokens with headroom.
    tokenizer_max_length_with_discretized_state: int = 192

    # The Qwen3VL processor and AnnotationTokenize tokenizer both default to the
    # HF repo id 'Qwen/Qwen3-VL-4B-Instruct', which needs that repo cached under
    # HF_HUB_OFFLINE. Plumb the actual `--vlm_pretrained_path` (typically a local
    # dir) through so the transform chain stays self-contained and offline-safe.
    vlm_pretrained_path: str = "/all-flash-data/vlm/Qwen3-VL-4B-Instruct"

    # LabUtopia-side gripper canonicalize. When True, snap raw gripper width to
    # {0, gripper_max_width} BEFORE NormalizeTransformFn (continuous [0, 0.04m] →
    # bimodal {0, 0.04m}). Use for LabUtopia finetune from an OXE pretrain ckpt
    # that saw gripper as binary {0,1}.
    #
    # IMPORTANT: the gripper-dim stats override is NOT automatic. For the snapped
    # {0, max_width} values to normalize to exactly {-1, +1}, the stats.json
    # gripper dim MUST already have q01=0, q99=max_width — pre-patch it with
    # `data_process/labutopia_canonicalize_stats.py`, otherwise the snapped values
    # are normalized through the dataset's RAW q01/q99 (mis-scaled gripper).
    # `hydrate_all` fails loud when snap_gripper_to_binary=True but the stats are
    # not canonicalized (bypass: LABVLA_ALLOW_UNPATCHED_SNAP_STATS=1).
    snap_gripper_to_binary: bool = False
    gripper_max_width: float = 0.04         # Franka Panda max gripper width (m)
    gripper_canonical_dim: int = 7          # canonical gripper index (7-DoF arm + grip)

    data_transforms: TransformGroup | None = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        if (
            self.source_shape_convergence
            and self.use_fast_tokenizer
            and self.discrete_action_max_length < 192
        ):
            raise ValueError(
                "source_shape_convergence=True requires "
                "discrete_action_max_length>=192 for current Phase-C FAST "
                "budgets; pass --discrete_action_max_length 192 or disable "
                "source shape convergence."
            )
        # When π0.5 VLM-pretrain + state discretization are both on, build the
        # chain in an order that lets DiscretizeStateTransformFn read normalized
        # state values BEFORE Qwen3_VLProcessorTransformFn tokenizes the task
        # prompt: [...image transforms..., NormalizeTransformFn,
        # DiscretizeStateTransformFn, Qwen3_VLProcessorTransformFn,
        # AnnotationTokenizeTransformFn, ComposeFieldsTransform, ...].
        # Otherwise (posttrain or discretize off) keep the original chain order.
        do_discretize = (
            self.training_phase == "vlm_pretrain"
            and self.discretize_state_in_vlm_pretrain
        )
        # Build transforms using instance attributes (not class-level defaults)
        # to ensure custom height/width/max_state_dim/max_action_dim are respected.
        if self.data_transforms is None:
            qwen_processor = Qwen3_VLProcessorTransformFn(
                pretrained_model_name_or_path=self.vlm_pretrained_path,
                max_length=(
                    self.tokenizer_max_length_with_discretized_state
                    if do_discretize else 48
                ),
            )
            normalize_xform = NormalizeTransformFn(mode="mean_std")
            base_inputs = [
                CanonicalArmLayoutTransformFn(),
                CanonicalSingleArmLayoutTransformFn(),
                DeltaActionTransformFn(),
                # GripperSemanticCanonicalizeFn() is disabled: q01/q99
                # NormalizeTransformFn already maps each repo's gripper to [-1, +1]
                # equivalently across width / open_fraction / data-driven sources,
                # and per-robot max_width is unknown for UR/Festo/Rizon4 (Franka
                # 0.04m doesn't generalize). Re-enable only with per-robot
                # calibration metadata.
                ResizeImagesWithPadFn(height=self.height, width=self.width),
                RemapImageKeyTransformFn(),
            ]
            # BuildUnifiedAnnotationTransformFn packs the 12 RoboInter
            # `annotation.<X>` parquet columns into a single
            # `annotation.unified` text field. No-op for datasets without
            # those columns (cheap absent-key skip).
            unify_anno = BuildUnifiedAnnotationTransformFn()
            # Agibot subtask builder: copies data['task'] → data['annotation.subtask']
            # and rewrites prompt for agibot. Must run BEFORE
            # DiscretizeStateTransformFn — that transform PREPENDS
            # "<state>...</state>\n" to data['task'], so reading data['task']
            # afterwards would put discretized digits into the CE target.
            # No-op for other schemas (enabled=False default; flipped True by
            # hydrate_all when schema declares annotation.subtask in annotation_losses).
            agibot_subtask = BuildAgiBotSubtaskTransformFn()
            if do_discretize:
                from transforms.state_discretize import DiscretizeStateTransformFn
                from src.dataset.adapters.robointer_token_budget import ROBOINTER_BUDGET
                # Order: ... → AgibotSubtask → Normalize → Discretize →
                # BuildUnifiedAnno → Qwen3VL processor → AnnotationTokenize → ...
                # AgibotSubtask runs early so it sees the clean task string,
                # then Discretize prepends the <state>...</state> bin digits
                # to the rewritten generic prompt.
                input_transforms = base_inputs + [
                    agibot_subtask,
                    normalize_xform,
                    DiscretizeStateTransformFn(num_bins=ROBOINTER_BUDGET.state_num_bins),
                    unify_anno,
                    qwen_processor,
                    AnnotationTokenizeTransformFn(
                        tokenizer_path=self.vlm_pretrained_path,
                        dynamic_shape=self.source_shape_convergence,
                    ),
                    ComposeFieldsTransform(),
                    PadStateAndActionTransformFn(
                        max_state_dim=self.max_state_dim,
                        max_action_dim=self.max_action_dim,
                    ),
                    UnifyLabVLAInputsTransformFn(),
                ]
            else:
                # Original order + unified-anno builder before tokenize. Posttrain
                # stays bit-identical for datasets with no annotation_losses
                # (BuildUnified is a no-op then). Optional SnapGripperToEndpointsFn
                # runs just before NormalizeTransformFn for LabUtopia finetune from
                # an OXE ckpt that saw binary gripper; the q01[grip]=0 /
                # q99[grip]=max_width stats override is NOT applied here (pre-patch
                # stats.json — hydrate_all fails loud if snap is on but stats
                # aren't canonicalized).
                pre_normalize: list = []
                if self.snap_gripper_to_binary:
                    pre_normalize.append(SnapGripperToEndpointsFn(
                        gripper_dim=self.gripper_canonical_dim,
                        max_width=self.gripper_max_width,
                        threshold_ratio=0.5,
                    ))
                input_transforms = base_inputs + [
                    agibot_subtask,
                    unify_anno,
                    qwen_processor,
                    AnnotationTokenizeTransformFn(
                        tokenizer_path=self.vlm_pretrained_path,
                        dynamic_shape=self.source_shape_convergence,
                    ),
                    *pre_normalize,
                    normalize_xform,
                    ComposeFieldsTransform(),
                    PadStateAndActionTransformFn(
                        max_state_dim=self.max_state_dim,
                        max_action_dim=self.max_action_dim,
                    ),
                    UnifyLabVLAInputsTransformFn(),
                ]
            self.data_transforms = TransformGroup(inputs=input_transforms, outputs=[])
        inputs = list(self.data_transforms.inputs)
        has_delta = any(isinstance(t, DeltaActionTransformFn) for t in inputs)
        if self.action_mode == "delta":
            if not has_delta:
                inputs = [DeltaActionTransformFn(), *inputs]
                self.data_transforms = replace(self.data_transforms, inputs=inputs)
        else:
            if has_delta:
                # Warn loudly when stripping a user-declared Delta in abs mode: a
                # silent removal (e.g. a typo action_mode='abs' on a chunk-delta
                # dataset) would drop delta semantics and train on wrong targets.
                import logging as _logging
                _logging.warning(
                    "[LabVLADatasetConfig] action_mode='abs' but user-provided "
                    "data_transforms includes DeltaActionTransformFn. Stripping "
                    "it. If you meant delta, set action_mode='delta'."
                )
                inputs = [t for t in inputs if not isinstance(t, DeltaActionTransformFn)]
                self.data_transforms = replace(self.data_transforms, inputs=inputs)

        # π0.5 / KI: optionally insert FAST action tokenizer AFTER
        # NormalizeTransformFn + ComposeFieldsTransform but BEFORE
        # PadStateAndActionTransformFn. Running before pad is critical — otherwise
        # DCT+BPE encodes the zero-padded trailing dims into the FAST token
        # sequence and pollutes the CE signal (up to 24 of 32 dims are zeros for
        # robointer_droid).
        if self.use_fast_tokenizer:
            from transforms.fast_action import FastActionEncodeTransformFn
            has_fast = any(isinstance(t, FastActionEncodeTransformFn) for t in inputs)
            if not has_fast:
                # Find the Pad transform; insert FAST immediately before it.
                pad_idx = next(
                    (i for i, t in enumerate(inputs)
                     if isinstance(t, PadStateAndActionTransformFn)),
                    len(inputs),
                )
                inputs = [
                    *inputs[:pad_idx],
                    FastActionEncodeTransformFn(
                        path=self.fast_tokenizer_path,
                        vocab_size=self.discrete_action_vocab_size,
                        max_length=self.discrete_action_max_length,
                        source_shape_convergence=self.source_shape_convergence,
                        trim_to_mask=self.source_shape_convergence,
                    ),
                    *inputs[pad_idx:],
                ]
                self.data_transforms = replace(self.data_transforms, inputs=inputs)


@PreTrainedConfig.register_subclass("labvla")
@dataclass
class LabVLAConfig(PreTrainedConfig):
    dtype: str = "bfloat16"

    # DiT Action Head config (replaces Qwen3 Transformer action expert)
    # Default params match Spirit-v1.5: 18 layers, hidden=1024, 8 heads, head_dim=128
    dit_num_layers: int = 18
    dit_num_heads: int = 8
    dit_head_dim: int = 128
    dit_dropout: float = 0.0
    dit_cross_attention_dim: int | None = None  # None = same as dit_hidden_size
    # Spirit-v1.5-style interleaved self-attention in the DiT. When True
    # (default), odd-indexed DiT blocks are self-attention only so the prepended
    # state token and action tokens exchange information layer-by-layer;
    # even-indexed blocks remain cross-attention to VLM features. Persisted in the
    # config so deployment and resume can't silently mismatch the Python default.
    dit_interleave_self_attention: bool = True
    # starVLA-style layerwise VLM features. When True, each cross-attn DiT block
    # reads a DIFFERENT VLM hidden-state layer (last-N of VLM, one per cross-attn
    # block) instead of all sharing VLM's last layer. Costs ~1.7 GB/GPU extra
    # (retains all VLM hidden states). Only supported on the posttrain+KI=False
    # path.
    dit_layerwise_vlm_features: bool = False

    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.0  # Flipped convention (t=0=noise, t=1=clean): include pure-noise end, exclude pure-clean end (ill-defined target)
    image_resolution: tuple[int, int] = (224, 224)

    empty_cameras: int = 0

    # VLM pretrained weights path
    vlm_pretrained_path: str = "/all-flash-data/vlm/Qwen3-VL-4B-Instruct"

    # Attention implementation
    attn_implementation: str = "flash_attention_2"

    # Normalization -- Note: set to IDENTITY because normalization is done inside data_processors
    # (pretrain_processor.py / lerobot_v30_processor.py hardcode mean_std normalization),
    # not through the standard lerobot Transform chain's NormalizeTransformFn.
    # Denormalization at deployment time is manually implemented in serve_labvla.py.
    # If normalization is migrated to the Transform chain, this must be updated to the corresponding NormalizationMode.
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # Training control
    gradient_checkpointing: bool = False
    gc_visual_encoder: bool = True   # Vision encoder gradient checkpointing (only effective when gradient_checkpointing=True)
    gc_language_model: bool = True   # VLM Language Model gradient checkpointing (only effective when gradient_checkpointing=True)
    gc_dit: bool = False             # DiT Action Head gradient checkpointing (only effective when gradient_checkpointing=True)
    # compile_model applies torch.compile to `sample_actions` (inference) ONLY.
    # Training `forward()` stays under @dynamo.disable for DeepSpeed + flash_attn
    # compatibility, so this flag does NOT speed up training.
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    device: str | None = None

    # Optimizer
    optimizer_lr: float = 5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Grouped learning rates (starVLA style)
    # VLM uses low LR to protect pretrained weights, DiT uses high LR to accelerate convergence of randomly initialized modules
    optimizer_vlm_lr: float = 5e-5       # VLM backbone lr (same as base lr, no scaling)
    optimizer_dit_lr: float = 1e-4       # DiT action head lr

    # Scheduler
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 100_000
    scheduler_decay_lr: float = 2.5e-6
    # Opt-in for the cosine scheduler to auto-rewrite warmup/decay when the
    # optimizer-step horizon (num_training_steps) is shorter than
    # scheduler_decay_steps. Default False = fail-closed (raise), so a
    # misconfigured decay length surfaces instead of silently becoming a different
    # LR recipe. With gradient_accumulation>1 the horizon is steps//GA, so a
    # launcher that sets decay_steps==steps must express decay_steps in
    # optimizer-step units or set this True.
    scheduler_allow_auto_scale: bool = False

    tokenizer_max_length: int = 48

    # Freeze control
    freeze_vision_encoder: bool = True
    train_expert_only: bool = False
    train_vlm_only: bool = False

    # -------- π0.5 VLM pretrain + Knowledge Isolation (opt-in, default = current behavior) --------
    # training_phase:
    #   "posttrain"    (default) current VLM + DiT + Flow Matching MSE pipeline (bit-identical).
    #   "vlm_pretrain" π0.5 Sec 3.3 pretrain style: VLM-only + FAST action token CE.
    #                  DiT not instantiated. Sequence: [state_tok | image | lang | FAST tokens].
    training_phase: str = "posttrain"
    # Delta mode predicts action deltas for delta_mask dims; abs mode predicts
    # target poses directly. VLM pretrain launchers set this to "abs".
    action_mode: str = "delta"

    # knowledge_isolation: only meaningful in posttrain (vlm_pretrain implies True-like).
    #   True  -> a) vlm_features.detach() blocks DiT -> VLM gradients
    #            b) adds FAST CE loss alongside MSE: total = ki_mse_weight * MSE + CE
    #   False -> preserves current bit-identical posttrain behavior
    knowledge_isolation: bool = False

    # use_fast_tokenizer: whether the DataLoader computes FAST action tokens.
    #   vlm_pretrain requires True; posttrain only when knowledge_isolation=True.
    use_fast_tokenizer: bool = False

    fast_tokenizer_path: str = "/all-flash-data/Embodied_models/fast"
    discrete_action_vocab_size: int = 2048
    discrete_action_max_length: int = 240  # above the historical max observed (229) so FAST CE targets aren't truncated (4ds p99 ≈ 175); encode() still warns past max_length. Smoke tests may pass an explicit 128.

    # KI loss mixing (posttrain+KI): total = ki_mse_weight * MSE + CE, per π0.5
    # Eq (1) with α=10.0 (App B.3 p.19). vlm_pretrain ignores this (α=0 there).
    # Renamed from ki_ce_weight to match paper: α multiplies MSE, not CE.
    ki_mse_weight: float = 10.0

    # Gripper loss weighting. When >1.0, the per-element flow-matching MSE for
    # dims in `gripper_action_dims` is scaled before the mean — counteracting the
    # structural dilution of a single gripper scalar against 7 joint dims in
    # uniform-average MSE.
    # Default 1.0 keeps current behaviour. `gripper_action_dims` is populated
    # from the active dataset schema in scripts/train.py before model
    # construction; an empty tuple disables the weighting branch entirely.
    gripper_loss_weight: float = 1.0
    gripper_action_dims: tuple[int, ...] = ()

    # π0.5 §B.1: discretize proprioceptive state into 256 bins → text tokens
    # prepended to language prompt (vlm_pretrain only). Default True so π0.5
    # paradigm pretrain runs are paper-faithful out-of-the-box. Posttrain
    # ignores this flag (state continues to flow through state_proj into DiT).
    discretize_state_in_vlm_pretrain: bool = True

    # π0.5 Appendix B.1 + Figure 11 specify a block-wise attention mask: the
    # prefix (images + state + prompt) is FULLY bidirectional, FAST tokens see
    # prefix fully and are causal among themselves, and annotation/text tokens are
    # causal on themselves but see prefix fully. The default HF Qwen3-VL forward
    # uses pure decoder-only causal attention. Enable this to build and pass a
    # (B, 1, L, L) 4D block-wise mask to `self.vlm.language_model(...)`; HF
    # `create_causal_mask()` early-exits on a 4D mask and uses it as-is.
    #
    # WARNING: changing this flag mid-training (ckpt trained with one value,
    # continued with the other) causes a train/deploy distribution mismatch. Set
    # True before step 0 for paper-faithful new training; keep False to deploy
    # legacy ckpts trained with causal-only prefix attention. Default False
    # preserves bit-identical loading of pre-flag ckpts.
    pi05_block_attention_mask: bool = False

    def __post_init__(self):
        super().__post_init__()
        # n_action_steps governs deploy-time queue length / how many predicted
        # actions are executed per inference; it CANNOT exceed chunk_size (the
        # model only predicts chunk_size frames per call). Historical default
        # was 50 == chunk_size; when chunk_size shrinks we silently clamp
        # n_action_steps so a single `--chunk_size N` flag in launchers is
        # sufficient (no need to also remember `--n_action_steps`). If callers
        # ever want a smaller deploy horizon (e.g. execute only 5 of 32 chunk
        # actions) they can construct LabVLAConfig directly with an explicit
        # smaller n_action_steps.
        if self.n_action_steps > self.chunk_size:
            self.n_action_steps = self.chunk_size

        # Double gradient checkpointing is a HARD RULE violation (~2x step time).
        if (
            self.gradient_checkpointing
            and self.gc_visual_encoder
            and self.gc_language_model
        ):
            raise ValueError(
                "HARD RULE: gc_visual_encoder=True and gc_language_model=True cannot "
                "both be enabled when gradient_checkpointing=True — double-GC causes "
                "~2× per-step time. Pick exactly one. See CLAUDE.md for details."
            )
        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")
        # train_expert_only freezes the VLM and train_vlm_only freezes the DiT +
        # projections; both at once freezes every trainable param. Catch it at
        # config-time instead of a confusing "No trainable parameters" later.
        if self.train_expert_only and self.train_vlm_only:
            raise ValueError(
                "train_expert_only and train_vlm_only are mutually exclusive — "
                "both freeze 100% of the model. Pick exactly one."
            )
        # Mirror the CLI's gripper_loss_weight>0 guard at the config layer so
        # direct LabVLAConfig(...) paths (tests, notebooks, deploy wrapper) can't
        # pass 0 (zeros gripper supervision) or negative (flips MSE gradient sign).
        if not (self.gripper_loss_weight > 0):
            raise ValueError(
                f"gripper_loss_weight must be > 0 (got "
                f"{self.gripper_loss_weight!r}); 0 zeros gripper "
                f"supervision and negative values flip the MSE "
                f"gradient sign. Pass 1.0 for no-op weighting."
            )
        if self.training_phase not in ("posttrain", "vlm_pretrain"):
            raise ValueError(
                f"training_phase must be 'posttrain' or 'vlm_pretrain', got {self.training_phase!r}"
            )
        if self.action_mode not in ("delta", "abs"):
            raise ValueError(
                f"action_mode must be 'delta' or 'abs', got {self.action_mode!r}"
            )
        if self.training_phase == "vlm_pretrain" and not self.use_fast_tokenizer:
            raise ValueError(
                "training_phase='vlm_pretrain' requires use_fast_tokenizer=True "
                "(VLM pretrain's only training signal is FAST action token CE)"
            )
        if self.knowledge_isolation and not self.use_fast_tokenizer:
            raise ValueError(
                "knowledge_isolation=True requires use_fast_tokenizer=True "
                "(KI's CE branch needs FAST tokens as target)"
            )
        if self.knowledge_isolation and self.train_expert_only:
            raise ValueError(
                "knowledge_isolation=True is incompatible with train_expert_only=True. "
                "KI needs CE gradient to flow through the VLM backbone, but "
                "train_expert_only freezes the entire VLM. Pick exactly one."
            )
        # Reject (posttrain, KI=False, fast_tok=True): that path early-returns MSE
        # before touching ki_head, so fast_tok only creates dead ki_head params
        # that pollute optimizer state + checkpoints.
        if (self.training_phase == "posttrain"
                and not self.knowledge_isolation
                and self.use_fast_tokenizer):
            raise ValueError(
                "use_fast_tokenizer=True is only meaningful in vlm_pretrain "
                "or posttrain+knowledge_isolation. The current config "
                "(posttrain, KI=False, fast_tok=True) would build ki_head "
                "without ever using it — pick vlm_pretrain, KI=True, or "
                "set use_fast_tokenizer=False."
            )

        # vlm_pretrain + train_vlm_only is broken: train_vlm_only freezes
        # everything but the VLM backbone, including ki_head — yet vlm_pretrain's
        # only signal is FAST CE through ki_head, so the VLM would train against a
        # frozen random classifier. Refuse to start.
        if (self.training_phase == "vlm_pretrain"
                and self.train_vlm_only):
            raise ValueError(
                "training_phase='vlm_pretrain' is incompatible with "
                "train_vlm_only=True. vlm_pretrain trains the VLM via FAST "
                "CE through ki_head; train_vlm_only freezes ki_head along "
                "with the DiT/projection, leaving the VLM to train against "
                "a randomly-initialised classifier head. Either turn off "
                "train_vlm_only or switch to posttrain (where the DiT/MSE "
                "path is the supervision signal)."
            )

        # Symmetric trap to the above: train_expert_only freezes the entire VLM
        # backbone, but vlm_pretrain's only signal is FAST CE through that
        # backbone, so CE would only update ki_head over frozen random features.
        if (self.training_phase == "vlm_pretrain"
                and self.train_expert_only):
            raise ValueError(
                "training_phase='vlm_pretrain' is incompatible with "
                "train_expert_only=True. vlm_pretrain trains the VLM via "
                "FAST CE; train_expert_only freezes the entire VLM, so the "
                "only thing left to train would be the ki_head classifier "
                "over random VLM features. Either turn off train_expert_only "
                "or switch to posttrain."
            )

        # vlm_pretrain + knowledge_isolation=True is incoherent: KI is
        # posttrain-only (CE via ki_head + MSE via DiT with detached flow between
        # them), but vlm_pretrain already runs CE-only through ki_head with no DiT,
        # so KI=True has nothing to isolate and just adds a trainable
        # state_vlm_proj (never invoked when state is discretized) that trips DDP
        # find_unused_parameters / pollutes optimizer state.
        if (self.training_phase == "vlm_pretrain"
                and self.knowledge_isolation):
            raise ValueError(
                "training_phase='vlm_pretrain' is incompatible with "
                "knowledge_isolation=True. KI is a posttrain-only construct "
                "(CE + Flow-Matching MSE simultaneously with information "
                "isolation between them). vlm_pretrain is already CE-only "
                "via ki_head — KI=True here builds unused trainable params "
                "(state_vlm_proj) when state is discretized into the prompt."
            )

        # posttrain + KI + train_vlm_only freezes ki_head/DiT/projections —
        # everything but the VLM. KI co-trains the VLM (CE through ki_head) against
        # a *trained* ki_head, so a frozen random ki_head gives no useful signal.
        # Dangerous even with --resume because the heads are silently frozen; force
        # the operator to confront it.
        if (self.training_phase == "posttrain"
                and self.knowledge_isolation
                and self.train_vlm_only):
            raise ValueError(
                "training_phase='posttrain' + knowledge_isolation=True + "
                "train_vlm_only=True is unsafe: train_vlm_only freezes "
                "ki_head/DiT/projections, but KI's CE branch needs the VLM "
                "to learn against a *trained* ki_head classifier — a "
                "randomly-initialised frozen ki_head produces no useful "
                "signal. If you want VLM-only finetuning under KI, do it "
                "from a checkpoint that already has a trained ki_head and "
                "instead lower vlm_lr."
            )

        # π0.5 block-wise attention passes a custom 4D mask to the LM forward, but
        # HF's flash_attention_2 honours only the padding part of attention_mask
        # (FA2 uses cu_seqlens, not bias matrices). Setting both would silently
        # fall back to FA2 causal+padding and ignore the block structure, so force
        # SDPA.
        if (self.pi05_block_attention_mask
                and self.attn_implementation == "flash_attention_2"):
            raise ValueError(
                "pi05_block_attention_mask=True is incompatible with "
                "attn_implementation='flash_attention_2'. FA2 only honours "
                "padding portions of attention_mask and silently ignores "
                "arbitrary 4D attention patterns, so the π0.5 block-wise "
                "mask would have no effect on attention. Switch to "
                "attn_implementation='sdpa' (slight perf cost) to get "
                "paper-faithful π0.5 prefix-bidirectional attention."
            )

        # KI + layerwise VLM features is unsupported: the KI posttrain branch reads
        # the LM output as a single Tensor and slices/detaches the prefix portion,
        # while layerwise returns a List[Tensor] (one per cross-attn block) that
        # this code would misinterpret. Refuse at config time.
        if (self.knowledge_isolation
                and self.dit_layerwise_vlm_features):
            raise ValueError(
                "knowledge_isolation=True is incompatible with "
                "dit_layerwise_vlm_features=True. The KI posttrain branch "
                "reads a single-Tensor last hidden state (and slices its "
                "prefix portion before detaching), while layerwise expects "
                "a List[Tensor] of per-block VLM hiddens. Pick exactly "
                "one. (If you need both in the future, the KI branch's "
                "prefix-slice + detach logic has to be extended to handle "
                "the per-layer list explicitly.)"
            )

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )
            self.input_features[key] = empty_camera

        if "observation.state" not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
            self.input_features["observation.state"] = state_feature

        if "action" not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )
            self.output_features["action"] = action_feature

    def get_optimizer_preset(self) -> LabVLAAdamWConfig:
        return LabVLAAdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
            vlm_lr=self.optimizer_vlm_lr,
            dit_lr=self.optimizer_dit_lr,
        )

    def get_scheduler_preset(self):
        # train.py builds its own LambdaLR scheduler inline, so this is not used.
        # Kept to satisfy PreTrainedConfig abstract method requirement.
        return None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def image_delta_indices(self) -> list | None:
        return None
