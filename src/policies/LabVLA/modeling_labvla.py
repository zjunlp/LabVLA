#!/usr/bin/env python

# LabVLA - Vision-Language-Action Model
# Qwen3-VL VLM backbone + DiT Cross-Attention action head
# Continuous action generation via Flow Matching
# Decoupled architecture: independent VLM forward + DiT cross-attention denoising

import logging
from collections import deque

import torch
import torch.nn.functional as F
import torch._dynamo as dynamo
from torch import Tensor, nn
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration

# Liger-Kernel: fused Triton ops replacing HuggingFace native implementations.
# Must be applied before model instantiation; patches module-level classes in modeling_qwen3_vl.
#
# Version gate: Liger 0.7.x declares an `swiglu=True` kwarg on
# apply_liger_kernel_to_qwen3_vl but does not branch on it, so we manually
# rebind the fused SwiGLU below for <=0.7.x; 0.8+ is assumed to support it
# natively. Bump _LIGER_SWIGLU_NEEDS_MANUAL_PATCH_MAX_VER after re-verifying.
_LIGER_SWIGLU_NEEDS_MANUAL_PATCH_MAX_VER = (0, 7)  # <= 0.7.x needs manual patch
try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3_vl
    apply_liger_kernel_to_qwen3_vl(
        rms_norm=True,     # Fused RMSNorm (eliminates fp32 upcasting overhead)
        rope=True,         # Fused RoPE (reduces sin/cos intermediate tensors)
        swiglu=False,      # SwiGLU disabled by default (Qwen3 MLP structure may not fully match)
        cross_entropy=False,
        fused_linear_cross_entropy=False,  # LabVLA uses MSE loss, no CE fusion needed
    )
    _LIGER_AVAILABLE = True

    # Detect Liger version so we only run the manual SwiGLU rebind on versions
    # where `apply_liger_kernel_to_qwen3_vl(swiglu=True)` is a known no-op.
    try:
        from importlib.metadata import version as _pkg_version
        _LIGER_VERSION_STR = _pkg_version("liger_kernel")
        _parts = _LIGER_VERSION_STR.split(".")
        _LIGER_VERSION = (int(_parts[0]), int(_parts[1]))
    except Exception:
        # Unknown version — assume we need the manual patch (safer for 0.7.x).
        _LIGER_VERSION_STR = "unknown"
        _LIGER_VERSION = (0, 0)
    _LIGER_SWIGLU_NEEDS_MANUAL_PATCH = (
        _LIGER_VERSION <= _LIGER_SWIGLU_NEEDS_MANUAL_PATCH_MAX_VER
    )
except ImportError:
    _LIGER_AVAILABLE = False
    _LIGER_VERSION_STR = "not_installed"
    _LIGER_VERSION = (0, 0)
    _LIGER_SWIGLU_NEEDS_MANUAL_PATCH = False

from policies.LabVLA.configuration_labvla import LabVLAConfig
from policies.LabVLA.dit_action_head import DiTActionHead
from policies.pretrained import PreTrainedPolicy
from utils.utils import format_big_number
from utils.constants import (
    ACTION,
    OBS_STATE,
    OBS_PREFIX,
)

logger = logging.getLogger(__name__)


def pad_vector(vector, new_dim):
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


class LabVLAModel(nn.Module):
    """LabVLA model — VLM + DiT action head, Flow Matching for action generation.

    Architecture: Qwen3-VL backbone (frozen vision, trainable LM) + DiT cross-attention action head.
    The VLM processes vision+language, DiT cross-attends to VLM features for action denoising.
    """

    def __init__(self, config: LabVLAConfig):
        super().__init__()
        self.config = config

        # ---- VLM Backbone ----
        _load_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            config.vlm_pretrained_path,
            dtype=_load_dtype,
            attn_implementation=config.attn_implementation,
        )
        self.vlm_hidden_size = self.vlm.config.text_config.hidden_size  # 2560

        # Liger SwiGLU patch — version-gated.
        # On <=0.7.x apply_liger_kernel_to_qwen3_vl(swiglu=True) is a no-op, so
        # manually rebind the fused forward on each LM MLP (weights unchanged,
        # math identical). On 0.8+ assume native support and skip to avoid a
        # double-rebind.
        if _LIGER_AVAILABLE and _LIGER_SWIGLU_NEEDS_MANUAL_PATCH:
            try:
                from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
                from liger_kernel.transformers.monkey_patch import _patch_swiglu_module
                _layers = self.vlm.model.language_model.layers
                for _layer in _layers:
                    _patch_swiglu_module(_layer.mlp, LigerSwiGLUMLP)
                logger.info(
                    f"LigerSwiGLUMLP manually patched on {len(_layers)} Qwen3VL "
                    f"text decoder layers (liger_kernel={_LIGER_VERSION_STR})"
                )
            except Exception as _swiglu_err:
                logger.warning(
                    f"Skipping LigerSwiGLUMLP patch ({type(_swiglu_err).__name__}: {_swiglu_err}); "
                    f"falling back to stock Qwen3VLTextMLP (numerically identical, slower)"
                )
        elif _LIGER_AVAILABLE:
            from utils.logging_utils import warn_once
            warn_once(
                logger,
                ("liger_swiglu_native_path_assumed", _LIGER_VERSION_STR),
                "liger_kernel %s > 0.7.x: skipping manual SwiGLU rebind "
                "(assuming native apply_liger_kernel_to_qwen3_vl(swiglu=True) "
                "support). If SwiGLU is not actually fused on this version, "
                "bump _LIGER_SWIGLU_NEEDS_MANUAL_PATCH_MAX_VER in "
                "modeling_labvla.py.",
                _LIGER_VERSION_STR,
            )

        # ---- DiT Action Head (posttrain only) ----
        # In vlm_pretrain mode DiT and its projection layers are not instantiated,
        # saving ~268M params + optimizer state. π0.5 Sec 3.3: action expert is
        # added only in posttrain (randomly initialized at that point).
        if config.training_phase == "posttrain":
            dit_hidden_size = config.dit_num_heads * config.dit_head_dim  # default: 8*128=1024
            self.dit_hidden_size = dit_hidden_size
            self.dit_action_head = DiTActionHead(
                num_attention_heads=config.dit_num_heads,
                attention_head_dim=config.dit_head_dim,
                num_layers=config.dit_num_layers,
                dropout=config.dit_dropout,
                attention_bias=True,
                activation_fn="gelu-approximate",
                cross_attention_dim=config.dit_cross_attention_dim,
                interleave_self_attention=config.dit_interleave_self_attention,
            )

            # VLM hidden → DiT hidden (if dimensions differ and no cross_attention_dim)
            if config.dit_cross_attention_dim is not None:
                self.proj_vlm_to_dit = nn.Identity()
            elif self.vlm_hidden_size != dit_hidden_size:
                self.proj_vlm_to_dit = nn.Linear(self.vlm_hidden_size, dit_hidden_size)
            else:
                self.proj_vlm_to_dit = nn.Identity()

            self.state_proj = nn.Linear(config.max_state_dim, dit_hidden_size)
            self.action_in_proj = nn.Linear(config.max_action_dim, dit_hidden_size)
            self.action_out_proj = nn.Linear(dit_hidden_size, config.max_action_dim)

            # Dtype alignment
            _target_dtype = _load_dtype
            for module in [self.state_proj, self.action_in_proj, self.action_out_proj]:
                module.to(dtype=_target_dtype)
            if not isinstance(self.proj_vlm_to_dit, nn.Identity):
                self.proj_vlm_to_dit.to(dtype=_target_dtype)
            self.dit_action_head.to(dtype=_target_dtype)
        else:
            # vlm_pretrain mode: no DiT, no DiT-side projection layers
            self.dit_hidden_size = None
            self.dit_action_head = None
            self.proj_vlm_to_dit = None
            self.state_proj = None
            self.action_in_proj = None
            self.action_out_proj = None

        # ---- KI head (DiscreteActionHead) — instantiated whenever FAST CE loss is needed ----
        if config.use_fast_tokenizer or config.training_phase == "vlm_pretrain":
            from policies.LabVLA.ki.ki_head import DiscreteActionHead
            self.ki_head = DiscreteActionHead(
                vocab_size=config.discrete_action_vocab_size,
                hidden_size=self.vlm_hidden_size,
                dtype=_load_dtype,
            )
        else:
            self.ki_head = None

        # ---- state_vlm_proj: project state into VLM hidden dim as a single soft token ----
        # Instantiate ONLY on code paths that actually consume it in forward;
        # building it otherwise leaves a dead trainable Linear in the optimizer's
        # "ki" group that trips DDP find_unused_parameters on non-DeepSpeed
        # launchers. Live consumers:
        #   - posttrain + KI=True            : π0.5 Fig.1 injects state into VLM prefix
        #   - vlm_pretrain + discretize=False: continuous state goes through proj
        # Other combos either don't use it or are config-rejected.
        _needs_state_vlm_proj = (
            config.knowledge_isolation
            or (
                config.training_phase == "vlm_pretrain"
                and not getattr(config, "discretize_state_in_vlm_pretrain", False)
            )
        )
        if _needs_state_vlm_proj:
            self.state_vlm_proj = nn.Linear(config.max_state_dim, self.vlm_hidden_size)
            self.state_vlm_proj.to(dtype=_load_dtype)
        else:
            self.state_vlm_proj = None

        self._attn_implementation = config.attn_implementation

        # π0.5 / KI prescribe a block-wise prefix attention mask (prefix
        # bidirectional; FAST tokens see prefix fully + causal among themselves).
        # `pi05_block_attention_mask` defaults to False only to keep bit-identical
        # loading of checkpoints trained before the flag existed; flipping it
        # would change attention semantics for old ckpts. For a FRESH
        # paper-faithful run that trains the FAST-token representation
        # (vlm_pretrain, or posttrain + knowledge_isolation), leaving it off makes
        # the model learn plain decoder-only causal attention instead of the
        # paper's prefix-bidirectional pattern. Advise the operator once.
        _trains_fast_representation = (
            config.training_phase == "vlm_pretrain"
            or (config.training_phase == "posttrain" and config.knowledge_isolation)
        )
        if _trains_fast_representation and not config.pi05_block_attention_mask:
            from utils.logging_utils import warn_once
            warn_once(
                logger,
                ("pi05_block_attention_mask_off", config.training_phase),
                "[LabVLA] training_phase=%r with FAST-token training but "
                "pi05_block_attention_mask=False. The model will use HF "
                "decoder-only causal attention, NOT the π0.5/KI block-wise "
                "prefix-bidirectional mask. This default is kept only for "
                "bit-identical loading of legacy checkpoints. For a fresh, "
                "paper-faithful vlm_pretrain/posttrain+KI run set "
                "pi05_block_attention_mask=True (also requires "
                "attn_implementation='sdpa').",
                config.training_phase,
            )

        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            # Compile is scoped to inference only. Training's forward() calls
            # `_vlm_forward_impl` (marked @dynamo.disable below), so compile and
            # @dynamo.disable never conflict: the compiled path (sample_actions)
            # never enters the VLM forward under dynamo, and training never
            # enters compile. If a future caller runs `sample_actions` inside the
            # training loop (e.g. eval-every-N-steps), revisit this — mixing
            # compile with DeepSpeed + flash_attn under dynamo has been fragile.
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            # Surface the inference-only scope so operators don't expect
            # training-step speedups from compile_model=True.
            logger.info(
                "[compile_model=True] torch.compile applied to sample_actions "
                "(INFERENCE path) ONLY; training forward() unaffected — see "
                "modeling_labvla.py L224-235."
            )

        self.set_requires_grad()

    # ----------------------------- Freeze / Train control -----------------------------

    def set_requires_grad(self):
        if self.config.freeze_vision_encoder:
            self.vlm.visual.eval()
            for p in self.vlm.visual.parameters():
                p.requires_grad = False

        if self.config.train_expert_only:
            self.vlm.eval()
            for p in self.vlm.parameters():
                p.requires_grad = False

        if self.config.train_vlm_only:
            for module in [
                self.dit_action_head,
                self.state_proj,
                self.action_in_proj,
                self.action_out_proj,
                self.state_vlm_proj,
                self.ki_head,
            ]:
                if module is None:
                    continue
                if hasattr(module, "eval"):
                    module.eval()
                for p in module.parameters():
                    p.requires_grad = False
            # proj_vlm_to_dit is part of the DiT input pipeline — freeze it with
            # dit_action_head + state/action projections. It's an Identity (no
            # params) when vlm_hidden_size == dit_hidden_size, so guard with
            # hasattr + None + isinstance(nn.Identity) checks.
            if (hasattr(self, "proj_vlm_to_dit")
                    and self.proj_vlm_to_dit is not None
                    and not isinstance(self.proj_vlm_to_dit, nn.Identity)):
                for p in self.proj_vlm_to_dit.parameters():
                    p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_vision_encoder:
            self.vlm.visual.eval()
        if self.config.train_expert_only:
            self.vlm.eval()
        if self.config.train_vlm_only and self.dit_action_head is not None:
            self.dit_action_head.eval()
        if self.config.train_vlm_only:
            if self.state_vlm_proj is not None:
                self.state_vlm_proj.eval()
            if self.ki_head is not None:
                self.ki_head.eval()
        return self

    def gradient_checkpointing_enable(self, gc_visual_encoder=True, gc_language_model=True, gc_dit=False):
        """Enable gradient checkpointing with independent control for vision encoder, Language Model, and DiT.

        Must call HuggingFace's gradient_checkpointing_enable() instead of setting flags manually,
        because GradientCheckpointingLayer.__call__() requires _gradient_checkpointing_func to work.
        """
        # When DiT is fully frozen (train_vlm_only=True), GC on DiT buys nothing:
        # no params receive gradients so there's no activation memory to save,
        # and re-running forward in backward is pure waste (~10% step-time).
        # Override gc_dit=False with a notice rather than silently ignoring it.
        if (gc_dit
                and self.config.train_vlm_only
                and self.dit_action_head is not None):
            logger.info(
                "[gc_dit] DiT is fully frozen under train_vlm_only=True; "
                "skipping gc_dit checkpointing to avoid recompute waste "
                "(MED-V5-16). Pass gc_dit=False explicitly to suppress this "
                "info line."
            )
            gc_dit = False

        # First set flag + _gradient_checkpointing_func on all sub-modules via HF method
        # use_reentrant=False is the PyTorch-recommended mode; matches DiT's torch_checkpoint setting.
        self.vlm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Disable checkpointing for vision encoder if not requested
        if not gc_visual_encoder:
            for module in self.vlm.visual.modules():
                if hasattr(module, 'gradient_checkpointing'):
                    module.gradient_checkpointing = False

        # Disable checkpointing for Language Model if not requested
        if not gc_language_model:
            for module in self.vlm.language_model.modules():
                if hasattr(module, 'gradient_checkpointing'):
                    module.gradient_checkpointing = False

        # DiT Action Head checkpointing (only if DiT exists — vlm_pretrain has no DiT)
        if self.dit_action_head is not None:
            self.dit_action_head.gradient_checkpointing = gc_dit

        parts = []
        if gc_visual_encoder:
            parts.append("visual_encoder")
        if gc_language_model:
            parts.append("language_model")
        if gc_dit and self.dit_action_head is not None:
            parts.append("dit_action_head")
        logger.info(f"Gradient checkpointing enabled for: {', '.join(parts) if parts else 'none'}")

        # Self-check: count per-layer active flags for observability
        def _count(root, name):
            total = sum(1 for m in root.modules() if hasattr(m, "gradient_checkpointing"))
            active = sum(1 for m in root.modules() if getattr(m, "gradient_checkpointing", False))
            return name, active, total
        v = _count(self.vlm.visual, "visual")
        l = _count(self.vlm.language_model, "language_model")
        dit_on = int(getattr(self.dit_action_head, "gradient_checkpointing", False)) if self.dit_action_head is not None else 0
        logger.info(
            f"GC layers active: visual={v[1]}/{v[2]}  language_model={l[1]}/{l[2]}  dit={dit_on}"
        )

    def gradient_checkpointing_disable(self):
        self.vlm.gradient_checkpointing_disable()
        # Symmetric with enable(): clear DiT flag too, otherwise caller can't
        # actually turn off checkpointing once enabled.
        if self.dit_action_head is not None:
            self.dit_action_head.gradient_checkpointing = False

    # ----------------------------- Noise / Time sampling -----------------------------

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        # Note: the swapped parameter order is intentional.
        # LabVLA time convention: t=0 is noise, t=1 is clean action (opposite to original pi0/InternVLA-A1 where t=1 is noise).
        # Beta(beta_beta=1.0, beta_alpha=1.5) = Beta(1.0, 1.5) biases toward t~0 (noise end),
        # equivalent to InternVLA-A1's Beta(1.5, 1.0) biasing toward t~1 (noise end).
        time_beta = sample_beta(
            self.config.time_sampling_beta_beta,
            self.config.time_sampling_beta_alpha,
            bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    # ----------------------------- VLM Forward (shared impl) -----------------------------

    @dynamo.disable
    def _build_prefix_embeds(self, pixel_values, image_grid_thw, lang_tokens, lang_masks):
        """Build vision-merged input embeddings + 3D M-RoPE position_ids WITHOUT running the LM.

        Shared by `_vlm_forward_impl` (posttrain default path) and the single-forward
        KI / vlm_pretrain paths that need to concat `[state | prefix | fast]` and run
        the LM once over the composite. Factoring this out removes the 2× LM forward
        in earlier KI code.

        Returns:
            embs:         (B, L, H)  vision-merged token embeddings
            position_ids: (3, B, L)  3D M-RoPE position ids
            attention_mask: (B, L)   padding mask (long dtype)
        """
        image_token_id = self.vlm.config.image_token_id
        D1 = pixel_values.shape[-1]
        pixel_values_flat = pixel_values.view(-1, D1)
        image_grid_thw_flat = image_grid_thw.view(-1, 3)
        image_embs, _ = self.vlm.visual(pixel_values_flat, image_grid_thw_flat)

        embs = self.vlm.get_input_embeddings()(lang_tokens)
        B, L, D2 = embs.shape
        embs = embs.view(-1, D2)
        lang_tokens_flat = lang_tokens.view(-1)
        embs[lang_tokens_flat == image_token_id] = image_embs
        embs = embs.view(B, L, D2)

        attention_mask = lang_masks.to(torch.long)
        position_ids, _ = self.vlm.model.get_rope_index(
            lang_tokens, image_grid_thw_flat,
            attention_mask=attention_mask,
        )
        return embs, position_ids, attention_mask

    @dynamo.disable
    def _build_pi05_block_attention_mask(
        self,
        L_state: int,
        prefix_pad_mask: torch.Tensor,
        ann_pad_mask: "torch.Tensor | None",
        fast_pad_mask: "torch.Tensor | None",
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """π0.5 paper-faithful block-wise attention mask (Appendix B.1 + Fig. 11).

        Constructs a (B, 1, L_total, L_total) 4D additive float mask following
        the π0.5 spec: ``[state | prefix | ann | fast]``:

        - Prefix block (state + images + lang prompt): **full bidirectional**.
        - Annotation tokens: see prefix fully, **causal** among themselves.
        - FAST tokens: see prefix + ann fully, **causal** among themselves.
        - No row in (state, prefix) attends to ann or fast (one-way info flow,
          matching the paper's "prefix doesn't see future" requirement).
        - Padded key positions get masked out of every query's view; padded
          query positions can't attend to anything (their output is unused).

        Without this, the LM falls back to pure decoder-only causal attention,
        which makes vision patches only see prior vision patches and lang tokens
        only see prior lang tokens within the prefix — a deviation from π0.5's
        "prefix is one contiguous bidirectional block" spec.

        HF's ``create_causal_mask`` early-exits when given a 4D mask and
        uses it as-is (transformers/masking_utils.py:711). So routing this
        mask through ``self.vlm.language_model(attention_mask=...)`` bypasses
        the auto-causal logic entirely.

        Returns:
            A (B, 1, L_total, L_total) tensor where 0.0 = allow attention
            and ``torch.finfo(dtype).min`` (~-inf) = block. Compatible with
            both SDPA and eager attention; FA2 doesn't honour arbitrary
            4D patterns — the config guard rejects that combo.
        """
        B, L_prefix = prefix_pad_mask.shape
        L_ann = int(ann_pad_mask.shape[1]) if ann_pad_mask is not None else 0
        L_fast = int(fast_pad_mask.shape[1]) if fast_pad_mask is not None else 0
        L_total = L_state + L_prefix + L_ann + L_fast

        # Slice boundaries.
        prefix_end = L_state + L_prefix
        ann_end = prefix_end + L_ann
        fast_end = ann_end + L_fast

        # 1. Build the LxL "static" allow matrix (block structure, no per-batch
        #    info). bool dtype keeps memory small (1 byte per cell).
        allow = torch.zeros((L_total, L_total), dtype=torch.bool, device=device)
        # state + prefix block: full bidirectional among themselves.
        allow[:prefix_end, :prefix_end] = True
        # Annotation: see prefix fully + causal among annotation tokens.
        if L_ann > 0:
            allow[prefix_end:ann_end, :prefix_end] = True
            ann_causal = torch.tril(torch.ones(L_ann, L_ann, dtype=torch.bool, device=device))
            allow[prefix_end:ann_end, prefix_end:ann_end] = ann_causal
        # FAST: see prefix + ann fully + causal among FAST.
        if L_fast > 0:
            allow[ann_end:fast_end, :ann_end] = True
            fast_causal = torch.tril(torch.ones(L_fast, L_fast, dtype=torch.bool, device=device))
            allow[ann_end:fast_end, ann_end:fast_end] = fast_causal

        # 2. Broadcast to per-batch (B, L, L) so per-sample padding can be
        #    applied below.
        allow_b = allow.unsqueeze(0).expand(B, L_total, L_total).contiguous()

        # 3. Apply per-batch padding to KEY columns only. Build key_mask
        #    (B, L_total) bool: True = real, False = pad. A padded KEY
        #    position is masked out of every query's view.
        #
        # Mask KEY columns only, never QUERY rows: an all-False query row makes
        # softmax over an all-(-inf) row produce NaN, which propagates through any
        # downstream slice (FAST/annotation CE). It also matters that CE picks the
        # first-FAST-token predictor at fixed index `fast_start - 1`, which under
        # right-side prefix padding lands in the pad region — masking key columns
        # only keeps that position computable (it attends to the same prefix a
        # valid query would). The per-sample first-target shift is a separate,
        # more invasive optimization not implemented here.
        state_key = torch.ones(B, L_state, dtype=torch.bool, device=device)
        prefix_key = prefix_pad_mask.to(torch.bool)
        pieces = [state_key, prefix_key]
        if L_ann > 0:
            pieces.append(ann_pad_mask.to(torch.bool))
        if L_fast > 0:
            pieces.append(fast_pad_mask.to(torch.bool))
        key_mask = torch.cat(pieces, dim=1)  # (B, L_total)
        # Mask KEY columns only — padded queries still receive valid
        # attention output (their row in `allow_b` keeps at least the
        # block-structure True entries). Downstream slices avoid pad
        # query positions via the CE mask, not via the attention mask.
        allow_b = allow_b & key_mask.unsqueeze(1)

        # 4. Convert to additive float mask. HF accepts 4D additive masks
        #    directly (early-exit in create_causal_mask).
        add_mask = torch.zeros((B, L_total, L_total), dtype=dtype, device=device)
        add_mask.masked_fill_(~allow_b, torch.finfo(dtype).min)

        # 5. Insert head dim → (B, 1, L_total, L_total).
        return add_mask.unsqueeze(1)

    @dynamo.disable
    def _vlm_forward_impl(self, pixel_values, image_grid_thw, lang_tokens, lang_masks):
        """Run VLM forward pass and return hidden states + prefix attention mask.

        Also returns the prefix `attention_mask` so downstream `_denoise_step` can
        pass it as `encoder_attention_mask` to DiT cross-attention; without it,
        padded text tokens / masked-out camera slots in the VLM prefix silently
        contribute to action-token cross-attention.

        Layerwise VLM features: when
        ``config.dit_layerwise_vlm_features`` is True, ``output_hidden_states=True``
        forces the VLM to retain every LM layer's hidden state. The returned
        first element is then the FULL tuple of hidden states (one per VLM
        layer, length = num_vlm_layers + 1 including embeddings) — downstream
        `_project_vlm_for_dit` slices the last `num_cross_attn_blocks` and
        maps them one-to-one to DiT cross-attention blocks. Default False
        keeps the single-last-layer behaviour bit-identical.
        """
        embs, position_ids, attention_mask = self._build_prefix_embeds(
            pixel_values, image_grid_thw, lang_tokens, lang_masks,
        )

        # Wire the π0.5 block mask into the prefix-only forward too (the path used
        # by posttrain non-KI + no-annotation training and non-KI `sample_actions`
        # inference). The mask reduces to a single (state=0, ann=0, fast=0) block,
        # i.e. pure prefix, which π0.5 specifies as full bidirectional. The legacy
        # 1D padding mask path is kept for `pi05_block_attention_mask=False` so
        # existing ckpts deploy/resume bit-identically.
        if self.config.pi05_block_attention_mask:
            lm_attention_mask = self._build_pi05_block_attention_mask(
                L_state=0,
                prefix_pad_mask=attention_mask,
                ann_pad_mask=None,
                fast_pad_mask=None,
                dtype=embs.dtype,
                device=embs.device,
            )
        else:
            lm_attention_mask = attention_mask

        # Run VLM language model (use_cache=False must be passed explicitly; otherwise gradient checkpointing cannot save memory properly)
        layerwise = bool(getattr(self.config, "dit_layerwise_vlm_features", False))
        vlm_output = self.vlm.language_model(
            inputs_embeds=embs,
            attention_mask=lm_attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=layerwise,
        )
        if layerwise:
            # Return the FULL tuple of LM hidden states (includes embeddings at
            # index 0 and the final hidden at index -1). _project_vlm_for_dit
            # will slice the last `num_cross_attn_blocks` and project each.
            return vlm_output.hidden_states, attention_mask
        # (hidden_state (B, L, H), attention_mask (B, L) long with 1=keep / 0=pad)
        return vlm_output.last_hidden_state, attention_mask

    def _project_vlm_for_dit(self, vlm_hidden):
        """Apply ``proj_vlm_to_dit`` to one VLM hidden state, or to each of
        the last ``num_cross_attn_blocks`` VLM hidden states (layerwise).

        Returns:
            - When ``vlm_hidden`` is a Tensor: a Tensor of shape (B, L, dit_hidden).
            - When ``vlm_hidden`` is a tuple/list (layerwise): a Python list
              of length ``num_cross_attn_blocks``, each (B, L, dit_hidden),
              ordered so that ``[i]`` is the i-th cross-attn block's KV input
              from shallow→deep VLM. The same ``proj_vlm_to_dit`` (or
              ``nn.Identity``) is shared across all layers — no per-layer
              projection params introduced, so existing checkpoints load
              unchanged.
        """
        if isinstance(vlm_hidden, (tuple, list)):
            num_cross = getattr(self.dit_action_head, "_num_cross_attn_blocks", None)
            if num_cross is None:
                raise RuntimeError(
                    "_project_vlm_for_dit: layerwise input requires DiTActionHead "
                    "to expose `_num_cross_attn_blocks` (added 2026-05-13)."
                )
            # Slice last `num_cross` VLM hidden states. `vlm_hidden` from
            # transformers' Qwen3VL.language_model output is
            # (embeddings, layer_0_out, layer_1_out, ..., layer_{L-1}_out)
            # so we take the deepest `num_cross`.
            sliced = list(vlm_hidden[-num_cross:])
            return [self.proj_vlm_to_dit(h) for h in sliced]
        return self.proj_vlm_to_dit(vlm_hidden)

    @torch.no_grad()
    @dynamo.disable
    def _run_vlm_forward(self, pixel_values, image_grid_thw, lang_tokens, lang_masks):
        """VLM forward without gradient (for inference or train_expert_only).

        Returns (hidden_state, prefix_attention_mask) — see _vlm_forward_impl.

        @dynamo.disable matches the safeguard on the inner `_vlm_forward_impl`.
        Without it, when `config.compile_model=True` and
        `knowledge_isolation=False`, `sample_actions()` lets torch.compile trace
        this wrapper, hits the dynamo.disable boundary on `_vlm_forward_impl`, and
        produces a graph break every inference step — a slow path masquerading as
        compiled, plus first-step latency spikes.
        """
        return self._vlm_forward_impl(pixel_values, image_grid_thw, lang_tokens, lang_masks)

    @torch.no_grad()
    @dynamo.disable
    def _run_vlm_forward_with_state_token(
        self, pixel_values, image_grid_thw, lang_tokens, lang_masks, state,
    ):
        """KI-path inference VLM forward that mirrors training's state-prepend.

        Training (`_forward_posttrain` KI branch) prepends `state_vlm_proj(state)`
        at position 0 of the VLM sequence — `[state | prefix | ann | fast]` — so
        the prefix slice sent to DiT is state-conditioned via causal attention.
        Inference must mirror this, otherwise the deploy-time prefix is
        state-blind and DiT cross-attention sees a train/deploy mismatch.

        This helper recreates the training layout for inference: prepend
        one state soft token to `[prefix]`, run the LM over the composite,
        then return the prefix slice (drop the state token) plus the
        prefix-only attention mask — so downstream DiT cross-attn sees the
        same shape it would see from `_run_vlm_forward` but with prefix
        hidden states that are now state-conditioned.

        @dynamo.disable mirrors the safeguard on _vlm_forward_impl /
        _run_vlm_forward. Without it, KI inference under
        `config.compile_model=True` lets torch.compile trace the Qwen3-VL
        language_model + flash-attn + M-RoPE forward, breaking the compile
        contract ("never enters the VLM forward under dynamo") and causing
        graph-break storms or compile failures.
        """
        assert self.state_vlm_proj is not None, (
            "_run_vlm_forward_with_state_token requires state_vlm_proj. "
            "This path is only used under knowledge_isolation=True."
        )
        dtype = next(self.vlm.parameters()).dtype
        prefix_embs, prefix_pos_ids, prefix_attn = self._build_prefix_embeds(
            pixel_values, image_grid_thw, lang_tokens, lang_masks,
        )
        B = prefix_embs.shape[0]
        L_prefix = prefix_embs.shape[1]
        # Align state with the VLM-embed device explicitly: if state arrives on a
        # different device (CPU staging buffer, gradient-checkpoint reshard), the
        # state_vlm_proj call would otherwise move data via implicit copy.
        state = state.to(device=prefix_embs.device, dtype=dtype)
        state_emb = self.state_vlm_proj(state)[:, None, :]
        state_mask = torch.ones(
            B, 1, dtype=prefix_attn.dtype, device=prefix_attn.device,
        )
        full_embeds = torch.cat([state_emb, prefix_embs], dim=1)
        full_attn = torch.cat([state_mask, prefix_attn], dim=1)
        full_pos_ids = self._extend_position_ids_for_extra_tokens(
            prefix_pos_ids, n_prepend=1, n_append=0,
        )

        # π0.5 block-wise mask for `[state | prefix]` at inference — both blocks
        # are "prefix" in the paper sense and form one bidirectional region. The
        # legacy 1D mask would impose pure causal attention (state blind to
        # prefix, prefix-prefix causal); mirror the training-time block mask so
        # the inference distribution matches.
        if self.config.pi05_block_attention_mask:
            lm_attention_mask = self._build_pi05_block_attention_mask(
                L_state=1,
                prefix_pad_mask=prefix_attn,
                ann_pad_mask=None,
                fast_pad_mask=None,
                dtype=dtype,
                device=prefix_embs.device,
            )
        else:
            lm_attention_mask = full_attn

        lm_out = self.vlm.language_model(
            inputs_embeds=full_embeds,
            attention_mask=lm_attention_mask,
            position_ids=full_pos_ids,
            use_cache=False,
        )
        # Drop the state token at position 0; return prefix-only slice.
        vlm_hidden = lm_out.last_hidden_state[:, 1:1 + L_prefix, :]
        return vlm_hidden, prefix_attn

    def _run_vlm_forward_with_grad(self, pixel_values, image_grid_thw, lang_tokens, lang_masks):
        """VLM forward with gradient (for joint training).

        Returns (hidden_state, prefix_attention_mask) — see _vlm_forward_impl.
        """
        return self._vlm_forward_impl(pixel_values, image_grid_thw, lang_tokens, lang_masks)

    # ----------------------------- DiT Action Embedding & Denoising -----------------------------

    def _embed_suffix(self, state, noisy_actions):
        """Embed state + noisy_actions -> (B, 1+chunk_size, dit_hidden).

        State token is prepended at position 0; action tokens occupy positions
        1..chunk_size. With Spirit-style interleaved self-attention enabled in
        the DiT (``interleave_self_attention=True`` in DiTActionHead, default
        on 2026-05-13), odd-indexed transformer blocks run self-attention over
        ``hidden_states`` so the state token and action tokens exchange
        information layer-by-layer. The trailing state-token slice in
        ``_denoise_step`` then drops state's own *output* — but the action
        tokens have already absorbed state information through every self-attn
        block.
        """
        _proj_dtype = self.state_proj.weight.dtype
        state = state.to(dtype=_proj_dtype)
        noisy_actions = noisy_actions.to(dtype=_proj_dtype)

        state_emb = self.state_proj(state)  # (B, state_dim) -> (B, dit_hidden)
        if state_emb.ndim == 2:
            state_emb = state_emb.unsqueeze(1)  # (B, 1, dit_hidden)
        action_emb = self.action_in_proj(noisy_actions)  # (B, T, dit_hidden)
        return torch.cat([state_emb, action_emb], dim=1)  # (B, 1+T, dit_hidden)

    def _denoise_step(self, state, vlm_features, x_t, timestep, vlm_attention_mask=None):
        """Single denoising step: embed suffix -> DiT -> predict velocity v_t.

        State routing (Spirit-clone, 2026-05-13). The DiT's odd-indexed blocks
        are now self-attention layers (see ``DiTActionHead.interleave_self_attention``).
        Pipeline:

          1. ``_embed_suffix`` prepends ``state_proj(state)`` to action embeddings,
             yielding hidden_states shape ``(B, 1+chunk_size, dit_hidden)``.
          2. Cross-attn blocks (even idx) attend to ``vlm_features`` on the KV
             side; state and action tokens each look up VLM features
             independently in these blocks.
          3. Self-attn blocks (odd idx) ignore the encoder and instead mix
             ``hidden_states`` among themselves — this is the path that lets
             action tokens read the state token AND each other. Without these
             interleaved self-attn layers ``state_proj`` would receive zero
             gradient (the original "dead state_proj" bug) and adjacent action
             tokens would never share information (the suspected cause of
             temporal incoherence / grasp-then-release gripper oscillation).
          4. Slice off position 0 (state token's own output is unused; its
             contribution to action positions has already happened through
             every self-attn layer).

        ``vlm_attention_mask`` continues to mask padded VLM positions on the
        cross-attn path (M2). Self-attn blocks see no mask — all suffix
        positions are valid by construction.
        """
        suffix_embs = self._embed_suffix(state, x_t)  # (B, 1+T, dit_hidden)
        suffix_out = self.dit_action_head(
            hidden_states=suffix_embs,
            encoder_hidden_states=vlm_features,
            timestep=timestep,
            encoder_attention_mask=vlm_attention_mask,
        )
        # Drop the state-token output at position 0; keep action tokens only.
        # Action tokens have already absorbed state via self-attn layers above.
        suffix_out = suffix_out[:, -self.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out).float()

    # ----------------------------- Training Forward -----------------------------

    def forward(self, *args, **kwargs):
        """Training forward dispatcher based on training_phase.

        - posttrain  (default): current VLM + DiT + Flow Matching MSE pipeline.
        - vlm_pretrain       : π0.5 VLM-only + FAST action token CE loss.
        """
        if self.config.training_phase == "vlm_pretrain":
            return self._forward_vlm_pretrain(*args, **kwargs)
        return self._forward_posttrain(*args, **kwargs)

    @staticmethod
    def _extract_annotation_fields(annotation_bundle):
        """From a dict of batch keys (`annotation_tokens__<field>`,
        `annotation_mask__<field>`, `annotation_weight__<field>`), derive a
        stable list of (field, tokens, mask, weight) tuples sorted by field
        name — deterministic across ranks / steps (avoids nondeterminism in
        dict iteration order across Python versions, even though Python 3.7+
        guarantees insertion order).

        Weight is preserved as a per-sample tensor (or python float when scalar)
        rather than collapsed to its first element, so `_compute_annotation_ces`
        can broadcast it against the loss and mask and handle heterogeneous
        batches (samples from different schemas with different weights) correctly:
          - same weight across batch → behaviour bit-identical to legacy
          - different weights / weight=0 for some samples → those samples'
            loss contributions scale proportionally; weight=0 samples drop
            out of both numerator and denominator (no silent dilution).
        """
        fields = []
        for k in annotation_bundle:
            if k.startswith("annotation_tokens__"):
                fields.append(k[len("annotation_tokens__"):])
        fields.sort()
        out = []
        for f in fields:
            tkey = f"annotation_tokens__{f}"
            mkey = f"annotation_mask__{f}"
            wkey = f"annotation_weight__{f}"
            if tkey in annotation_bundle and mkey in annotation_bundle:
                w_t = annotation_bundle.get(wkey)
                # Pass weight through as-is. Tensor weights stay tensor; None
                # / scalar collapse to python float 1.0 / float(w).
                if w_t is None:
                    w = 1.0
                elif torch.is_tensor(w_t):
                    w = w_t  # (B,) or 0-d tensor; do NOT take [0]
                else:
                    w = float(w_t)
                out.append((f, annotation_bundle[tkey], annotation_bundle[mkey], w))
        return out

    def _embed_annotations(self, annotation_bundle, target_dtype, device):
        """Embed each annotation field's tokens via the VLM input embedding
        layer, concatenate along sequence dim in a stable (field-name-sorted)
        order, and return (embs, attn_mask, per_field_info).

        The VLM input embedding (same matrix whose transpose acts as lm_head
        when tied) gives the right bf16-aligned distribution; no extra
        trainable parameter is added. Per-field info `(field, tokens, mask,
        weight, L_field)` is kept for later slicing + weighted CE.
        """
        fields = self._extract_annotation_fields(annotation_bundle)
        if not fields:
            # Warn once: a non-empty bundle with no extractable fields usually
            # means the DataLoader produced `annotation_weight__*` / partial keys
            # without matching tokens — a plumbing bug. Surface it instead of
            # silently falling back to pure MSE.
            if annotation_bundle and not getattr(
                self, "_warned_empty_annotation_bundle", False
            ):
                logger.warning(
                    "_embed_annotations: annotation_bundle has %d keys but "
                    "no (tokens, mask) pair was extractable. Keys: %s. "
                    "Silently falling back to MSE-only. "
                    "(further occurrences suppressed)",
                    len(annotation_bundle), sorted(annotation_bundle)[:6],
                )
                self._warned_empty_annotation_bundle = True
            return None, None, []
        get_in_embed = self.vlm.get_input_embeddings()
        per_field_embs = []
        per_field_masks = []
        per_field_info = []
        for field_name, tokens, mask, weight in fields:
            ids = tokens.to(device)
            m = mask.to(device)
            emb = get_in_embed(ids).to(target_dtype)
            per_field_embs.append(emb)
            per_field_masks.append(m.to(torch.long))
            # Preserve tensor weights (move to device); float weights stay python
            # scalars. _compute_annotation_ces handles both forms.
            if torch.is_tensor(weight):
                weight_kept = weight.to(device)
            else:
                weight_kept = float(weight)
            per_field_info.append(
                (field_name, ids, m, weight_kept, int(emb.shape[1]))
            )
        embs = torch.cat(per_field_embs, dim=1)
        masks = torch.cat(per_field_masks, dim=1)
        return embs, masks, per_field_info

    def _compute_annotation_ces(self, last_hidden, ann_block_start, per_field_info):
        """Next-token CE for each annotation field, computed on the VLM's own
        vocabulary via `self.vlm.lm_head`.

        Args:
            last_hidden:     (B, L_total, H) output of the composite LM forward.
            ann_block_start: index in L_total where the annotation block begins.
                             In `[prefix | ann]` this is L_prefix; in the
                             KI composite `[state | prefix | ann | fast]` it is
                             L_state + L_prefix.
            per_field_info:  list of (field_name, ids, mask, weight, L_field),
                             parallel-ordered with the concatenated embeds.

        Returns:
            {field_name: weighted_ce_scalar} — already multiplied by weight so
            the training loop can simply sum them.

        Next-token shift: hidden at absolute position `ann_block_start - 1 + k`
        predicts `ann_tokens[k]` for k in [0, L_field).
        """
        # Use Liger fused linear-cross-entropy instead of `F.linear(.float()) +
        # F.cross_entropy`. The naive path materializes a (B, L_field, V≈152K)
        # fp32 logits tensor (~9.96 GB at B=64, L=256) that OOMs on an 80 GB H100
        # alongside the 4.7B-param model; FLCE chunk-fuses matmul + softmax + CE so
        # the big logits tensor is never created. fp32 numerics are preserved
        # (lm_head_w cast to fp32 here; FLCE accumulates CE in fp32 internally) —
        # not bit-identical to F.cross_entropy but fp32-equivalent.
        #
        # Must use reduction="sum" (FLCE's reduction="none" backward is
        # incomplete) and apply the weighted mask outside the kernel:
        #   - per-token mask  → ignore_index=-100 on target ids
        #   - per-sample weight → group rows by equal weight, scale each grouped
        #     loss_sum by its weight, accumulate num/denom manually.
        #
        # Memory invariant: each FLCE call saves a dense grad_weight tensor shaped
        # like lm_head (V≈152K, H=2560), ~1.45 GiB fp32. Calling it once per
        # sample stored O(B) copies and OOM'd at BS=64; grouping rows by
        # field/weight keeps the common heterogeneous-mixture case to one call per
        # annotation field.
        #
        # Import is wrapped with an actionable error: an unconditional import
        # would ImportError mid-training (after the first non-empty
        # annotation_bundle) on environments without `liger_kernel`, burning GPU
        # hours up to that point.
        try:
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        except ImportError as e:
            raise RuntimeError(
                "[M8-03] _compute_annotation_ces requires liger_kernel "
                "(LigerFusedLinearCrossEntropyLoss). Install via "
                "`pip install liger-kernel>=0.7` to enable annotation "
                "supervision (FAST CE through lm_head), or remove "
                "`annotation_losses` from the active schema to disable "
                "this code path entirely. The Liger fused FLCE is "
                "load-bearing for memory: a non-fused F.linear + "
                "F.cross_entropy fallback would OOM at BS=64 because "
                "it materializes a (B, V, H) gradient tensor."
            ) from e
        if not hasattr(self, "_flce_sum_loss"):
            self._flce_sum_loss = LigerFusedLinearCrossEntropyLoss(
                reduction="sum",
                ignore_index=-100,
            )

        weighted_ces = {}
        cum_offset = 0
        lm_head_w_fp32 = self.vlm.lm_head.weight.to(torch.float32)
        for field_name, ids, mask, weight, L_field in per_field_info:
            start = ann_block_start - 1 + cum_offset
            end = start + L_field
            ce_hidden = last_hidden[:, start:end, :].to(torch.float32)
            B = ids.shape[0]
            H = ce_hidden.shape[-1]
            ids_2d = ids.to(device=ce_hidden.device).long()
            mask_bool = mask.to(device=ce_hidden.device).bool()

            # Per-sample weight: scalar in homogeneous batch, (B,) tensor in
            # heterogeneous (PI0Mixture) batches.
            is_tensor_weight = torch.is_tensor(weight)
            if is_tensor_weight:
                w_b = weight.to(device=ce_hidden.device, dtype=torch.float32).reshape(-1)
                if w_b.numel() == 1:
                    w_b = w_b.expand(B)
            else:
                w_val = float(weight)

            num = ce_hidden.new_zeros(())
            denom = ce_hidden.new_zeros(())

            def _accumulate_group(group_hidden, group_ids, group_mask, weight_value):
                nonlocal num, denom
                target = group_ids.masked_fill(~group_mask, -100).reshape(-1)
                loss_sum = self._flce_sum_loss(
                    lm_head_w_fp32,
                    group_hidden.reshape(-1, H),
                    target,
                )
                num = num + loss_sum * weight_value
                denom = denom + group_mask.sum().to(torch.float32)

            if is_tensor_weight:
                # Heterogeneous PI0Mixture batches zero-fill fields absent from
                # a sample's schema. Most real fields therefore have one positive
                # schema weight and many zeros; grouping collapses that from
                # O(nonzero rows) FLCE calls to O(unique positive weights).
                for weight_value in torch.unique(w_b.detach()):
                    if float(weight_value.item()) == 0.0:
                        continue
                    group_idx = torch.nonzero(w_b == weight_value, as_tuple=False).flatten()
                    if group_idx.numel() == 0:
                        continue
                    _accumulate_group(
                        ce_hidden.index_select(0, group_idx),
                        ids_2d.index_select(0, group_idx),
                        mask_bool.index_select(0, group_idx),
                        weight_value.to(device=ce_hidden.device, dtype=torch.float32),
                    )
            elif w_val != 0.0:
                _accumulate_group(
                    ce_hidden,
                    ids_2d,
                    mask_bool,
                    ce_hidden.new_tensor(w_val, dtype=torch.float32),
                )

            weighted_ces[field_name] = num / denom.clamp(min=1.0)
            cum_offset += L_field
        return weighted_ces

    def _forward_posttrain(
        self, pixel_values, image_grid_thw, lang_tokens, lang_masks, state, actions,
        noise=None, time=None, actions_is_pad=None,
        fast_action_tokens=None, fast_action_mask=None,
        annotation_bundle=None,
    ):
        """Posttrain forward: VLM -> DiT -> flow matching MSE loss.

        When knowledge_isolation=False (default), returns Tensor (B, K, D) of
        per-element MSE — bit-identical to the pre-KI code path.

        When knowledge_isolation=True, additionally:
          - detaches VLM features before feeding DiT
          - computes FAST token CE loss on VLM hidden
          - returns dict {"mse", "mse_per_elem", "ce"}
        """
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        # Flow Matching: x_t = t*actions + (1-t)*noise, u_t = actions - noise
        time_expanded = time[:, None, None]
        x_t = time_expanded * actions + (1 - time_expanded) * noise
        u_t = actions - noise

        has_annotations = bool(annotation_bundle) and any(
            k.startswith("annotation_tokens__") for k in annotation_bundle
        )

        # Non-KI path: regular single VLM forward (bit-identical to pre-KI code
        # when no annotation_losses are configured — this preserves the fast
        # pure-MSE path for OXE-style datasets).
        if not self.config.knowledge_isolation and not has_annotations:
            if self.config.train_expert_only:
                vlm_hidden, vlm_attn = self._run_vlm_forward(
                    pixel_values, image_grid_thw, lang_tokens, lang_masks
                )
            else:
                vlm_hidden, vlm_attn = self._run_vlm_forward_with_grad(
                    pixel_values, image_grid_thw, lang_tokens, lang_masks
                )
            # `vlm_hidden` is either a Tensor (legacy: VLM last layer) or a
            # tuple of Tensors (layerwise mode, one per VLM layer). The
            # `_project_vlm_for_dit` helper handles both → returns a Tensor or
            # List[Tensor] that DiTActionHead understands.
            vlm_features_for_dit = self._project_vlm_for_dit(vlm_hidden)
            # Drop the layerwise tuple (29 entries, 28+ activation tensors; only
            # the last 9 are referenced via vlm_features_for_dit) so autograd
            # cleanup and memory profilers see this scope's footprint clearly.
            del vlm_hidden
            v_t = self._denoise_step(state, vlm_features_for_dit, x_t, time, vlm_attn)
            v_t = v_t.float(); u_t = u_t.float()
            return F.mse_loss(u_t, v_t, reduction="none")

        # Non-KI + annotations: single VLM forward over `[prefix | annotation]`
        # composite. Prefix portion is bit-identical to the forward used above
        # (causal attention: annotation tokens cannot leak into prefix hidden),
        # so MSE is computed the same way, flowing through VLM (π0 style — no
        # detach). Annotation CE is computed on the annotation portion via
        # self.vlm.lm_head, flowing through VLM (the whole point).
        if not self.config.knowledge_isolation and has_annotations:
            # This manual LM forward does NOT request output_hidden_states, so
            # under `dit_layerwise_vlm_features=True` annotation batches would feed
            # only the last layer to DiT cross-attention while no-annotation
            # batches use the full per-layer tuple — the two halves of an ablation
            # become incomparable. Fail loud rather than mix semantics silently.
            if self.config.dit_layerwise_vlm_features:
                raise RuntimeError(
                    "[H8-02] training_phase='posttrain' + "
                    "knowledge_isolation=False + dit_layerwise_vlm_features=True + "
                    "schema annotation supervision is unsupported. This branch "
                    "would silently feed only the LAST VLM layer to DiT cross-"
                    "attention on annotation batches while no-annotation batches "
                    "use the full per-layer tuple, polluting any layerwise "
                    "ablation. Either disable dit_layerwise_vlm_features, "
                    "switch to knowledge_isolation=True, or remove "
                    "annotation_losses from the schema."
                )
            _dtype = next(self.vlm.parameters()).dtype
            device = lang_tokens.device

            prefix_embs, prefix_pos_ids, prefix_attn = self._build_prefix_embeds(
                pixel_values, image_grid_thw, lang_tokens, lang_masks,
            )
            L_prefix = prefix_embs.shape[1]

            ann_embs, ann_mask, per_field_info = self._embed_annotations(
                annotation_bundle, _dtype, device,
            )
            if ann_embs is None:
                # No configured annotation field had tokens in this batch
                # (shouldn't happen, but defensive): fall through to pure-MSE.
                vlm_hidden, vlm_attn = self._run_vlm_forward_with_grad(
                    pixel_values, image_grid_thw, lang_tokens, lang_masks
                )
                # Route through the helper so this rare fallback stays compatible
                # with layerwise mode (`proj_vlm_to_dit(tuple)` would TypeError).
                vlm_features_for_dit = self._project_vlm_for_dit(vlm_hidden)
                v_t = self._denoise_step(state, vlm_features_for_dit, x_t, time, vlm_attn)
                v_t = v_t.float(); u_t = u_t.float()
                return F.mse_loss(u_t, v_t, reduction="none")

            L_ann = ann_embs.shape[1]
            full_embeds = torch.cat([prefix_embs, ann_embs], dim=1)
            full_attn = torch.cat([prefix_attn, ann_mask.to(prefix_attn.dtype)], dim=1)
            full_pos_ids = self._extend_position_ids_for_extra_tokens(
                prefix_pos_ids, n_prepend=0, n_append=L_ann,
            )

            # π0.5 block-wise mask for `[prefix | ann]` — prefix bidirectional,
            # ann causal-with-prefix-visible. L_state=0 (no state soft token in
            # non-KI).
            if self.config.pi05_block_attention_mask:
                lm_attention_mask = self._build_pi05_block_attention_mask(
                    L_state=0,
                    prefix_pad_mask=prefix_attn,
                    ann_pad_mask=ann_mask.to(prefix_attn.dtype),
                    fast_pad_mask=None,
                    dtype=_dtype,
                    device=device,
                )
            else:
                lm_attention_mask = full_attn

            # When train_expert_only=True the entire VLM is frozen, but PyTorch
            # still builds the autograd graph (wasting activation memory) unless
            # we wrap in no_grad. Downstream MSE then reads `last` as
            # detached-equivalent (prefix_hidden is sliced from no_grad output),
            # and annotation CE produces zero gradient anyway, so wrapping is
            # semantically equivalent for both branches.
            if self.config.train_expert_only:
                with torch.no_grad():
                    lm_out = self.vlm.language_model(
                        inputs_embeds=full_embeds,
                        attention_mask=lm_attention_mask,
                        position_ids=full_pos_ids,
                        use_cache=False,
                    )
            else:
                lm_out = self.vlm.language_model(
                    inputs_embeds=full_embeds,
                    attention_mask=lm_attention_mask,
                    position_ids=full_pos_ids,
                    use_cache=False,
                )
            last = lm_out.last_hidden_state  # (B, L_prefix + L_ann, H)

            # MSE path — prefix portion, NOT detached (non-KI, π0 joint grad).
            prefix_hidden = last[:, :L_prefix, :]
            vlm_features_for_dit = self.proj_vlm_to_dit(prefix_hidden)
            # DiT only attends to the prefix slice → mask is the corresponding
            # slice of the composite attention mask.
            prefix_attn_for_dit = full_attn[:, :L_prefix]
            v_t = self._denoise_step(state, vlm_features_for_dit, x_t, time, prefix_attn_for_dit)
            v_t = v_t.float(); u_t = u_t.float()
            mse_per_elem = F.mse_loss(u_t, v_t, reduction="none")

            # Under train_expert_only=True the VLM is frozen and the LM forward
            # ran under no_grad, so annotation CE here would cost lm_head FLOPs and
            # emit a scalar that enters total loss but carries zero gradient — a
            # silent dead-loss trap. Skip it and emit an empty dict (downstream sum
            # treats {} as zero).
            if self.config.train_expert_only:
                if not getattr(self, "_warned_ann_ce_dead_loss", False):
                    logger.warning(
                        "[R7-F2] train_expert_only=True with annotation "
                        "supervision: skipping annotation CE — VLM is frozen "
                        "so this loss term has zero gradient and would "
                        "appear in logs without contributing to learning. "
                        "Either turn off train_expert_only or remove "
                        "annotation_losses from the schema."
                    )
                    self._warned_ann_ce_dead_loss = True
                annotation_ces = {}
            else:
                # Annotation CE(s) — block starts right after the prefix.
                annotation_ces = self._compute_annotation_ces(
                    last, ann_block_start=L_prefix,
                    per_field_info=per_field_info,
                )

            return {
                "mse": mse_per_elem.mean(),
                "mse_per_elem": mse_per_elem,
                "annotation_ces": annotation_ces,
            }

        # KI path: single-pass VLM forward over [prefix | fast]; extract prefix portion
        # as vlm_hidden for DiT cross-attn (detached), extract fast portion for CE.
        # This replaces the earlier 2-pass (run VLM, then re-run LM on concat) — same
        # semantics, ~40% less LM compute per step.
        assert fast_action_tokens is not None and fast_action_mask is not None, (
            "knowledge_isolation=True requires FAST tokens in batch "
            "(set use_fast_tokenizer=True in config)"
        )
        assert self.ki_head is not None
        assert self.state_vlm_proj is not None, (
            "posttrain+KI requires state_vlm_proj; it is now built "
            "unconditionally when use_fast_tokenizer or knowledge_isolation "
            "is set — check modeling_labvla.py __init__."
        )

        _dtype = next(self.vlm.parameters()).dtype
        device = lang_tokens.device

        prefix_embs, prefix_pos_ids, prefix_attn = self._build_prefix_embeds(
            pixel_values, image_grid_thw, lang_tokens, lang_masks,
        )

        # Inject state as a soft token at the FRONT of the VLM sequence.
        #
        # Sequence order vs π0.5 Fig 1: the paper depicts `[prompt | state |
        # action]` (state between prompt and action), but LabVLA's composite is
        # `[state | prefix(vision+language) | fast_action_tokens]` (state first).
        # This is a deliberate legacy choice kept for ckpt bit-compatibility:
        # every posttrain+KI ckpt learned state-first conditioning.
        #
        # Consequences:
        #   - M-RoPE position 0 is occupied by the state token, shifting
        #     vision-token `(t, h, w)` positions by 1 (see
        #     `_extend_position_ids_for_extra_tokens`). Qwen3-VL's pos-0 bias
        #     for vision is marginal because the state projection produces a
        #     narrow distribution that doesn't materially displace attention
        #     weights.
        #   - Loading official π0.5 checkpoints would require reordering the
        #     position embeddings; we don't currently do that.
        #
        # Without this prepended token the VLM backbone would see only
        # `[vision+lang | FAST]` and lack state context when predicting
        # state-conditional FAST tokens.
        B = lang_tokens.shape[0]
        # Explicit device alignment, matching `_run_vlm_forward_with_state_token`:
        # state usually arrives on prefix_embs' device, but a CPU-staging buffer or
        # gradient-checkpoint reshard could deliver it elsewhere, and without the
        # `.to(device=...)` the state_vlm_proj forward would move data via implicit
        # copy.
        state = state.to(device=prefix_embs.device, dtype=_dtype)
        state_emb = self.state_vlm_proj(state)[:, None, :]  # (B, 1, H)
        state_mask = torch.ones(B, 1, dtype=prefix_attn.dtype, device=device)

        fast_action_tokens = fast_action_tokens.to(device)
        fast_emb = self.ki_head.embed(fast_action_tokens).to(_dtype)
        L_fast = fast_action_tokens.shape[1]
        fast_mask = fast_action_mask.to(prefix_attn.dtype).to(device)

        # Optional annotation tokens — inserted BEFORE fast in the composite.
        # Zero impact when the batch/schema has no annotation_losses (L_ann=0
        # reduces the composite to the plain [state | prefix | fast] shape).
        ann_embs, ann_mask, per_field_info = self._embed_annotations(
            annotation_bundle or {}, _dtype, device,
        )
        L_ann = 0 if ann_embs is None else ann_embs.shape[1]

        # Annotation block sits BEFORE FAST. Under causal attention the layout
        # `[state | prefix | fast | ann]` would let each annotation token attend
        # leftward to ground-truth FAST tokens — a teacher-forcing leak that lets
        # the annotation head read the real action sequence instead of predicting
        # from prefix/scene alone. `[state | prefix | ann | fast]` keeps FAST able
        # to condition on annotations (low-level action depends on high-level
        # semantics) while blocking annotation's access to FAST.
        L_state = 1
        L_prefix = prefix_embs.shape[1]
        pieces_embeds = [state_emb, prefix_embs]
        pieces_attn = [state_mask, prefix_attn]
        if ann_embs is not None:
            pieces_embeds.append(ann_embs)
            pieces_attn.append(ann_mask.to(prefix_attn.dtype))
        pieces_embeds.append(fast_emb)
        pieces_attn.append(fast_mask)
        full_embeds = torch.cat(pieces_embeds, dim=1)
        full_attn = torch.cat(pieces_attn, dim=1)
        full_pos_ids = self._extend_position_ids_for_extra_tokens(
            prefix_pos_ids, n_prepend=L_state, n_append=L_ann + L_fast,
        )

        # Under config.pi05_block_attention_mask, swap the 1D padding mask for a
        # paper-faithful (B, 1, L, L) block mask: prefix bidirectional, FAST/ann
        # causal-with-prefix-visible. The 1D path is kept for ckpts trained before
        # this flag.
        if self.config.pi05_block_attention_mask:
            lm_attention_mask = self._build_pi05_block_attention_mask(
                L_state=L_state,
                prefix_pad_mask=prefix_attn,
                ann_pad_mask=ann_mask.to(prefix_attn.dtype) if ann_embs is not None else None,
                fast_pad_mask=fast_mask,
                dtype=_dtype,
                device=device,
            )
        else:
            lm_attention_mask = full_attn

        lm_out = self.vlm.language_model(
            inputs_embeds=full_embeds,
            attention_mask=lm_attention_mask,
            position_ids=full_pos_ids,
            use_cache=False,
        )
        last = lm_out.last_hidden_state  # (B, L_state + L_prefix + L_ann + L_fast, H)

        # Prefix hidden → DiT. KI detaches BEFORE the projection so the VLM
        # backbone gets no MSE gradient (KI intent) while proj_vlm_to_dit still
        # does (learnable projection). Detaching AFTER proj would leave
        # proj_vlm_to_dit a dead randomly-init'd parameter under KI.
        vlm_hidden = last[:, L_state:L_state + L_prefix, :]
        vlm_features_for_dit = self.proj_vlm_to_dit(vlm_hidden.detach())
        # KI's composite is [state | prefix | (ann) | fast]; DiT only attends to
        # the prefix slice, so its attention mask is the matching slice of
        # full_attn (skipping the state token at position 0).
        prefix_attn_for_dit = full_attn[:, L_state:L_state + L_prefix]
        v_t = self._denoise_step(state, vlm_features_for_dit, x_t, time, prefix_attn_for_dit)
        v_t = v_t.float(); u_t = u_t.float()
        mse_per_elem = F.mse_loss(u_t, v_t, reduction="none")

        # Slice offsets for the layout `[state | prefix | ann | fast]`: FAST starts
        # after annotation, so its shift-1 CE slice is at index
        # (L_state + L_prefix + L_ann - 1) for length L_fast, and the annotation CE
        # block is at index (L_state + L_prefix) for length L_ann.
        fast_start = L_state + L_prefix + L_ann - 1
        ce_hidden = last[:, fast_start:fast_start + L_fast, :]
        ce_loss = self.ki_head.compute_ce_loss(ce_hidden, fast_action_tokens, fast_action_mask)

        out_dict = {
            "mse": mse_per_elem.mean(),
            "mse_per_elem": mse_per_elem,
            "ce": ce_loss,
        }
        if ann_embs is not None:
            ann_block_start = L_state + L_prefix
            out_dict["annotation_ces"] = self._compute_annotation_ces(
                last, ann_block_start=ann_block_start,
                per_field_info=per_field_info,
            )
        return out_dict

    def _extend_position_ids_for_extra_tokens(self, base_pos_ids, n_prepend, n_append):
        """Extend a (3, B, L) M-RoPE position_ids tensor with consecutive integer
        positions for `n_prepend` tokens on the left and `n_append` tokens on the
        right. Each prepend/append step advances the position by 1 on all 3
        (temporal / height / width) RoPE dimensions — same pattern as Qwen3VL
        handles plain text tokens.

        Args:
            base_pos_ids: (3, B, L) position IDs from get_rope_index for the prefix.
            n_prepend:    int, e.g. 1 for the `state` soft token.
            n_append:     int, e.g. L_fast for FAST action tokens.
        Returns:
            (3, B, n_prepend + L + n_append) tensor on the same device/dtype.
        """
        dev = base_pos_ids.device
        dty = base_pos_ids.dtype
        B = base_pos_ids.shape[1]
        # Prepend: single (or few) token(s) placed at positions -n_prepend..-1 of the
        # base sequence's position range. Cleanest: shift base by n_prepend, put
        # prepend at 0..n_prepend-1.
        if n_prepend > 0:
            pre = torch.arange(n_prepend, device=dev, dtype=dty).view(1, 1, n_prepend).expand(3, B, n_prepend)
            base_pos_ids = base_pos_ids + n_prepend
        # Append: continue after base's last position per batch item.
        if n_append > 0:
            # Use the maximum of base_pos_ids across the 3 rope dims at the last
            # position as the starting point. Since text tokens have t==h==w, this
            # equals base_pos_ids[0, b, -1] for non-vision-ending prefixes. To be
            # safe, use the max over all 3 components.
            last = base_pos_ids.max(dim=2).values  # (3, B)
            offsets = torch.arange(1, n_append + 1, device=dev, dtype=dty)  # (n_append,)
            # (3, B, n_append) = last[:, :, None] + offsets[None, None, :]
            post = last[:, :, None] + offsets[None, None, :]
        # Concat
        pieces = []
        if n_prepend > 0:
            pieces.append(pre)
        pieces.append(base_pos_ids)
        if n_append > 0:
            pieces.append(post)
        return torch.cat(pieces, dim=2)

    def _forward_vlm_pretrain(
        self, pixel_values, image_grid_thw, lang_tokens, lang_masks, state, actions,
        noise=None, time=None, actions_is_pad=None,
        fast_action_tokens=None, fast_action_mask=None,
        annotation_bundle=None,  # dict with annotation_tokens__/_mask__/_weight__ keys
                                 # (same convention as posttrain — see
                                 # _extract_annotation_fields). Empty/None →
                                 # only FAST CE is computed.
    ):
        """π0.5 VLM pretrain forward — single-pass over a composite sequence.

        Sequence layout (when discretize_state_in_vlm_pretrain=True, π0.5 spec):
            [vision + lang (state-text inside lang)] [annotation_segs] [FAST]

        Sequence layout (legacy soft state token, discretize=False):
            [state_emb] [vision + lang] [annotation_segs] [FAST]

        Loss = ce_fast + Σ (per-spec annotation CE × weight).
        Reuses self._embed_annotations + self._compute_annotation_ces helpers
        so the path is bit-identical to posttrain's annotation handling.

        VQA-only co-train:
          When the *entire* batch is VQA samples (every sample's
          ``action_is_pad`` was True so ``FastActionEncodeTransformFn`` did
          not emit fast tokens, and the collate's zero-fill therefore
          produced ``fast_action_mask.sum() == 0``), the FAST sequence is
          dropped from the LM forward entirely and ``ce`` is returned as a
          zero tensor. This avoids both fabricated CE supervision on
          all-zero "action" tokens and the wasted compute of running a FAST
          embed through the language model only to mask the result to 0.
          Mixed batches (some real, some VQA) still run the FAST path —
          per-sample masking inside ``ki_head.compute_ce_loss`` correctly
          excludes the VQA samples (mask=False everywhere in their row)
          while keeping the real-action rows in the gradient.
        """
        # Don't hard-assert FAST availability — VQA co-train can legitimately have
        # no FAST tokens for some batches. Gate the FAST branch on (a) keys being
        # present and (b) the batch having ANY valid FAST positions to learn from.
        assert self.ki_head is not None
        has_fast_keys = fast_action_tokens is not None and fast_action_mask is not None
        if has_fast_keys:
            # Move mask once for both the "any valid?" check and the LM forward.
            fast_mask_local = fast_action_mask.to(lang_tokens.device)
            # `int()` on a 0-d tensor pulls the value to host once. The
            # alternative `bool(... .any())` triggers a full sync per call.
            run_fast_branch = bool(fast_mask_local.any().item())
        else:
            fast_mask_local = None
            run_fast_branch = False

        B = lang_tokens.shape[0]
        _dtype = next(self.vlm.parameters()).dtype
        device = lang_tokens.device

        # (1) Build vision-merged prefix embeds + 3D M-RoPE position ids
        prefix_embs, prefix_pos_ids, prefix_attn = self._build_prefix_embeds(
            pixel_values, image_grid_thw, lang_tokens, lang_masks,
        )  # (B, L_pref, H); (3, B, L_pref); (B, L_pref)
        L_pref = prefix_embs.shape[1]

        # (2) Optional state soft token (skipped when discretization is on —
        #     state already lives inside lang_tokens as digit text).
        use_state_proj = not getattr(
            self.config, "discretize_state_in_vlm_pretrain", False
        )
        if use_state_proj:
            assert self.state_vlm_proj is not None, (
                "discretize_state_in_vlm_pretrain=False requires state_vlm_proj"
            )
            state_emb = self.state_vlm_proj(state.to(_dtype))[:, None, :]  # (B, 1, H)
            state_mask = torch.ones(B, 1, dtype=prefix_attn.dtype, device=device)
            n_prepend = 1
        else:
            state_emb = None
            state_mask = None
            n_prepend = 0

        # (3) Optional annotation segments — embedded via shared helper
        #     (mirrors posttrain's _embed_annotations contract).
        ann_embs, ann_attn, per_field_info = self._embed_annotations(
            annotation_bundle or {}, _dtype, device,
        )  # ann_embs/(ann_attn): (B, L_ann_total, H)/(B, L_ann_total) or (None, None, [])
        L_ann_total = 0 if ann_embs is None else ann_embs.shape[1]

        # (4) FAST action tokens — only embedded when at least one batch row
        #     has a valid FAST position. All-VQA batches drop the FAST block
        #     from the LM forward entirely.
        if run_fast_branch:
            fast_emb = self.ki_head.embed(fast_action_tokens.to(device)).to(_dtype)
            L_fast = fast_action_tokens.shape[1]
            fast_mask = fast_mask_local.to(prefix_attn.dtype)
        else:
            fast_emb = None
            L_fast = 0
            fast_mask = None

        # (5) Assemble composite sequence: [state?] | prefix | annotations* | fast?
        pieces_emb = []
        pieces_attn = []
        if state_emb is not None:
            pieces_emb.append(state_emb)
            pieces_attn.append(state_mask)
        pieces_emb.append(prefix_embs)
        pieces_attn.append(prefix_attn)
        if ann_embs is not None:
            pieces_emb.append(ann_embs)
            pieces_attn.append(ann_attn.to(prefix_attn.dtype))
        if fast_emb is not None:
            pieces_emb.append(fast_emb)
            pieces_attn.append(fast_mask)

        full_embeds = torch.cat(pieces_emb, dim=1)
        full_attn = torch.cat(pieces_attn, dim=1)
        full_pos_ids = self._extend_position_ids_for_extra_tokens(
            prefix_pos_ids,
            n_prepend=n_prepend,
            n_append=L_ann_total + L_fast,
        )

        # Block-wise attention mask for the composite
        # `[state? | prefix | ann | fast]`. State soft token is optional (skipped
        # when discretize_state_in_vlm_pretrain=True, the default). The 1D-mask
        # fallback is kept for config.pi05_block_attention_mask=False so existing
        # pretrained checkpoints keep their original attention semantics.
        if self.config.pi05_block_attention_mask:
            lm_attention_mask = self._build_pi05_block_attention_mask(
                L_state=n_prepend,
                prefix_pad_mask=prefix_attn,
                ann_pad_mask=(
                    ann_attn.to(prefix_attn.dtype) if ann_embs is not None else None
                ),
                fast_pad_mask=fast_mask if fast_emb is not None else None,
                dtype=_dtype,
                device=device,
            )
        else:
            lm_attention_mask = full_attn

        # (6) SINGLE LM forward
        lm_out = self.vlm.language_model(
            inputs_embeds=full_embeds,
            attention_mask=lm_attention_mask,
            position_ids=full_pos_ids,
            use_cache=False,
        )
        last = lm_out.last_hidden_state  # (B, total_len, H)

        # Layout offsets in `last`:
        #   [0           : n_prepend]                — state (if any)
        #   [n_prepend   : n_prepend+L_pref]         — prefix (vision + lang)
        #   [ann_start   : ann_start+L_ann_total]    — annotation block
        #   [fast_start  : fast_start+L_fast]        — FAST tokens (if present)
        ann_start = n_prepend + L_pref
        fast_start = ann_start + L_ann_total

        # FAST CE — at the end of the sequence (next-token shift). Produces
        # a real gradient term only when at least one batch row has a True
        # mask position. All-VQA batches emit a 0 loss that is still connected
        # to every ki_head parameter, preventing DDP unused-parameter failures
        # without fabricating FAST supervision.
        if run_fast_branch:
            ce_fast_hidden = last[:, fast_start - 1 : fast_start + L_fast - 1, :]
            fast_ce = self.ki_head.compute_ce_loss(
                ce_fast_hidden, fast_action_tokens, fast_mask_local
            )
        else:
            fast_ce = torch.zeros((), device=device, dtype=torch.float32)
            for p in self.ki_head.parameters():
                if p.numel() > 0:
                    fast_ce = fast_ce + p.reshape(-1)[0].to(torch.float32) * 0.0

        # Annotation CE (one weighted term per spec) via shared helper.
        annotation_ces = {}
        if per_field_info:
            annotation_ces = self._compute_annotation_ces(
                last, ann_start, per_field_info,
            )

        # Return dict matches the `forward` dispatcher contract: "ce" = FAST CE
        # only; "annotation_ces" = per-field weighted CE dict; dispatcher sums them
        # into total loss.
        return {
            "ce": fast_ce,
            "mse": torch.zeros((), device=fast_ce.device, dtype=fast_ce.dtype),
            "annotation_ces": annotation_ces,
        }

    # ----------------------------- Inference Sampling -----------------------------

    @torch.no_grad()
    def sample_actions(
        self, pixel_values, image_grid_thw, lang_tokens, lang_masks, state,
        noise=None, num_steps=None
    ) -> Tensor:
        """Inference: VLM forward once, then iterative DiT denoising."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        # Fail fast on an empty batch: shape[0]==0 silently NaN's downstream
        # shape-sensitive ops (e.g. proj_vlm_to_dit flattens leading dims).
        if state.shape[0] == 0:
            raise ValueError(
                "sample_actions called with batch_size=0. An empty batch "
                "reached inference — this is always a caller bug (e.g. an "
                "outer collate_fn dropped every sample). Fix upstream rather "
                "than returning empty tensors that propagate NaN."
            )

        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        # Phase 1: VLM forward (single pass, no grad)
        # When KI is on, training puts a `state_vlm_proj(state)` soft token at
        # position 0 of the VLM sequence (see `_forward_posttrain` KI branch), so
        # the prefix slice passed to DiT is state-conditioned. Mirror that
        # state-prepend here, otherwise the deploy-time prefix is state-blind — a
        # train/deploy mismatch. Non-KI runs use the plain VLM forward (state flows
        # only through DiT's own `state_proj` in `_denoise_step`).
        if self.config.knowledge_isolation and self.state_vlm_proj is not None:
            vlm_hidden, vlm_attn = self._run_vlm_forward_with_state_token(
                pixel_values, image_grid_thw, lang_tokens, lang_masks, state
            )
        else:
            vlm_hidden, vlm_attn = self._run_vlm_forward(
                pixel_values, image_grid_thw, lang_tokens, lang_masks
            )
        # `vlm_features` is either Tensor (legacy last-layer) or List[Tensor]
        # (layerwise — one per cross-attn DiT block).
        vlm_features = self._project_vlm_for_dit(vlm_hidden)

        # Phase 2: Iterative denoising (t: 0 -> 1, LabVLA convention)
        dt = 1.0 / num_steps
        x_t = noise
        time_val = 0.0
        while time_val < 1.0 - dt / 2:
            expanded_time = torch.full(
                (bsize,), time_val, dtype=torch.float32, device=device
            )
            v_t = self._denoise_step(state, vlm_features, x_t.to(dtype), expanded_time, vlm_attn)
            x_t = x_t + dt * v_t
            time_val += dt

        return x_t


class LabVLAPolicy(PreTrainedPolicy):
    """LabVLA Policy for LeRobot framework.

    VLM + DiT Cross-Attention action head, continuous action generation via Flow Matching.
    """

    config_class = LabVLAConfig
    name = "labvla"

    def __init__(self, config: LabVLAConfig):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = LabVLAModel(config)

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gc_visual_encoder=config.gc_visual_encoder,
                gc_language_model=config.gc_language_model,
                gc_dit=config.gc_dit,
            )

        self.model.to(config.device)

        self.reset()

    def __str__(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append(f"Policy: {self.__class__.__name__}")
        lines.append("")

        num_total_params = sum(p.numel() for p in self.parameters())
        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_vlm = sum(p.numel() for p in self.model.vlm.parameters())
        num_dit = (sum(p.numel() for p in self.model.dit_action_head.parameters())
                   if self.model.dit_action_head is not None else 0)
        num_ki = (sum(p.numel() for p in self.model.ki_head.parameters())
                  if getattr(self.model, "ki_head", None) is not None else 0)

        lines.append("Parameter statistics:")
        lines.append(f"  - Total params        : {num_total_params} ({format_big_number(num_total_params)})")
        lines.append(f"  - Trainable params    : {num_trainable_params} ({format_big_number(num_trainable_params)})")
        lines.append(f"  - VLM params          : {num_vlm} ({format_big_number(num_vlm)})")
        lines.append(f"  - DiT Action Head     : {num_dit} ({format_big_number(num_dit)})")
        if num_ki > 0:
            lines.append(f"  - KI Head (FAST CE)   : {num_ki} ({format_big_number(num_ki)})")
        lines.append(f"  - Training phase      : {self.config.training_phase}"
                     + (f"  (KI on, α={self.config.ki_mse_weight})" if self.config.knowledge_isolation else ""))
        lines.append("=" * 60)

        return "\n".join(lines)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self

    def get_optim_params(self) -> dict:
        """Return named parameters dict for grouped LR optimizer."""
        return dict(self.named_parameters())

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def prepare_state(self, batch):
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        pixel_values = batch[f"{OBS_PREFIX}pixel_values"]
        image_grid_thw = batch[f"{OBS_PREFIX}image_grid_thw"]
        lang_tokens = batch[f"{OBS_PREFIX}input_ids"]
        lang_masks = batch[f"{OBS_PREFIX}attention_mask"]
        state = self.prepare_state(batch)

        actions = self.model.sample_actions(
            pixel_values, image_grid_thw, lang_tokens, lang_masks, state
        )

        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    def _apply_gripper_loss_weighting(self, mse_per_elem: Tensor) -> Tensor:
        """Scale gripper-dim elements of a per-element MSE tensor.

        A single gripper scalar gets ~1/D weight in a uniform-average
        flow-matching MSE; the 7 joint dims outweigh the 1 gripper dim by 7×.
        Tasks that fail on "grip and drop" (e.g. LabUtopia TransportBeaker)
        benefit from up-weighting the gripper-dim per-element MSE BEFORE the mean.

        Returns the input unchanged when no gripper dims are declared, the
        weight is exactly 1.0, or the dims are out of range — so a default
        config (weight=1.0, gripper_action_dims=()) is bit-identical to the
        pre-fix behaviour.
        """
        w = float(getattr(self.config, "gripper_loss_weight", 1.0))
        dims = tuple(getattr(self.config, "gripper_action_dims", ()) or ())
        if not dims or w == 1.0 or mse_per_elem.shape[-1] == 0:
            return mse_per_elem
        D = mse_per_elem.shape[-1]
        valid_dims = [int(d) for d in dims if 0 <= int(d) < D]
        if not valid_dims:
            return mse_per_elem
        weights = torch.ones(D, device=mse_per_elem.device, dtype=mse_per_elem.dtype)
        for d in valid_dims:
            weights[d] = w
        return mse_per_elem * weights

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward pass.

        Return contract:
          - posttrain + KI=False (default): model returns Tensor -> we pad-mask-mean -> loss_action.
          - posttrain + KI=True           : model returns dict{"mse","ce","mse_per_elem"}.
          - vlm_pretrain                  : model returns dict{"mse"=0,"ce"}.
        """
        pixel_values = batch[f"{OBS_PREFIX}pixel_values"]
        image_grid_thw = batch[f"{OBS_PREFIX}image_grid_thw"]
        lang_tokens = batch[f"{OBS_PREFIX}input_ids"]
        lang_masks = batch[f"{OBS_PREFIX}attention_mask"]

        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad", None)
        fast_tokens = batch.get("fast_action_tokens", None)
        fast_mask = batch.get("fast_action_mask", None)

        # Per-dataset annotation CE: gather batch keys emitted by
        # `AnnotationTokenizeTransformFn` when the sample's schema declares
        # annotation_losses. OXE-style samples carry no such keys → empty
        # bundle → model takes the pure-MSE path. Weights ride along in the
        # `annotation_weight__<field>` entries (per-dataset scalar).
        annotation_bundle = {
            k: v for k, v in batch.items()
            if (k.startswith("annotation_tokens__")
                or k.startswith("annotation_mask__")
                or k.startswith("annotation_weight__"))
        }

        model_out = self.model.forward(
            pixel_values, image_grid_thw, lang_tokens, lang_masks, state, actions,
            actions_is_pad=actions_is_pad,
            fast_action_tokens=fast_tokens, fast_action_mask=fast_mask,
            annotation_bundle=annotation_bundle,
        )

        original_action_dim = self.config.output_features[ACTION].shape[0]

        # --- Default (backward compat) path: model returned Tensor per-elem MSE ---
        if isinstance(model_out, torch.Tensor):
            losses_action = model_out[:, :, :original_action_dim]
            # Up-weight gripper-dim per-element MSE when configured (default 1.0 =
            # no-op). Must run AFTER the original_action_dim slice so
            # gripper_action_dims indices are valid against the real-action-dim
            # subset, not max_action_dim.
            losses_action = self._apply_gripper_loss_weighting(losses_action)
            if actions_is_pad is not None:
                valid_mask = (~actions_is_pad).unsqueeze(-1).expand_as(losses_action)
                valid_mask = valid_mask.to(losses_action.device)
                valid_elements = losses_action[valid_mask]
                if valid_elements.numel() == 0:
                    # When a local batch has no valid action elements (every frame
                    # is `action_is_pad=True`, e.g. a VQA-only batch with zero
                    # action + all-pad), do NOT fall back to losses_action.mean()
                    # (a mean over fabricated zero-target padded frames). Return a
                    # zero loss connected to the losses_action graph via *0 so the
                    # model gets no gradient from padded samples while DDP keeps a
                    # valid backward path through the action head.
                    loss_action = (losses_action.sum() * 0.0)
                else:
                    loss_action = valid_elements.mean()
            else:
                loss_action = losses_action.mean()
            loss = loss_action

            loss_dict = {
                "loss": loss.item(),
                "loss_action": loss_action.item(),
            }
        else:
            # --- Dict path: vlm_pretrain (CE only) / posttrain+KI (MSE+FAST_CE)
            # / posttrain+annotations (MSE + annotation_CE) ---
            mse = model_out["mse"]
            fast_ce = model_out.get("ce", None)             # only KI / vlm_pretrain
            annotation_ces = model_out.get("annotation_ces", {}) or {}
            # Compute ann_total ONLY when non-empty, so the graph in
            # KI-no-annotation stays bit-identical to the pre-change path.
            ann_total = sum(annotation_ces.values()) if annotation_ces else None

            if self.config.training_phase == "vlm_pretrain":
                # VLM pretrain: FAST CE is the primary signal; annotation CE
                # (if any) adds on top.
                loss_action = fast_ce
                loss = fast_ce if ann_total is None else (fast_ce + ann_total)
            elif self.config.knowledge_isolation:
                # Pad-masked MSE from per-element tensor.
                if "mse_per_elem" in model_out and actions_is_pad is not None:
                    mse_pe = model_out["mse_per_elem"][:, :, :original_action_dim]
                    mse_pe = self._apply_gripper_loss_weighting(mse_pe)
                    valid_mask = (~actions_is_pad).unsqueeze(-1).expand_as(mse_pe).to(mse_pe.device)
                    valid = mse_pe[valid_mask]
                    # Zero-loss when no valid elements (mse_pe.mean() would average
                    # over padded zero targets = fabricated supervision). Connect to
                    # the mse_pe graph via sum*0 to keep a valid DDP backward path.
                    mse_scalar = valid.mean() if valid.numel() > 0 else (mse_pe.sum() * 0.0)
                else:
                    mse_scalar = mse
                loss_action = mse_scalar
                # π0.5 Eq (1): total = α·MSE + CE, α=10 per App B.3.
                loss = self.config.ki_mse_weight * mse_scalar + fast_ce
                if ann_total is not None:
                    loss = loss + ann_total
            else:
                # Non-KI + annotations: flow-matching MSE (not α-scaled; α is
                # KI-specific) plus annotation CEs.
                if "mse_per_elem" in model_out and actions_is_pad is not None:
                    mse_pe = model_out["mse_per_elem"][:, :, :original_action_dim]
                    mse_pe = self._apply_gripper_loss_weighting(mse_pe)
                    valid_mask = (~actions_is_pad).unsqueeze(-1).expand_as(mse_pe).to(mse_pe.device)
                    valid = mse_pe[valid_mask]
                    # Same zero-loss fallback as above.
                    mse_scalar = valid.mean() if valid.numel() > 0 else (mse_pe.sum() * 0.0)
                else:
                    mse_scalar = mse
                loss_action = mse_scalar
                loss = mse_scalar if ann_total is None else (mse_scalar + ann_total)

            loss_dict = {
                "loss": loss.item(),
                "loss_action": loss_action.item(),
                "loss_mse": (mse.item() if mse.ndim == 0 else mse.mean().item()),
            }
            if fast_ce is not None:
                loss_dict["loss_ce"] = fast_ce.item()
            if ann_total is not None:
                loss_dict["loss_annotation_total"] = ann_total.item()
                for fname, v in annotation_ces.items():
                    short = fname[len("annotation."):] if fname.startswith("annotation.") else fname
                    loss_dict[f"loss_ann_{short}"] = v.item()
            # Populate losses_action-shaped tensor for the downstream per-dim
            # logging path (loop expects (B, K, D) tensor). Fill with zeros in
            # vlm_pretrain (no MSE to report); pass through per-elem MSE in KI.
            if "mse_per_elem" in model_out:
                losses_action = model_out["mse_per_elem"][:, :, :original_action_dim]
            else:
                losses_action = torch.zeros(
                    actions.shape[0], self.config.chunk_size, original_action_dim,
                    device=loss.device, dtype=torch.float32,
                )

        # Keep per-dim loss as a GPU tensor and let the logger sync to CPU only
        # when it logs (log_freq cadence); a per-step .cpu().numpy().tolist() wastes
        # 9/10 syncs under log_freq=10. Detached so no autograd trace is retained.
        loss_dict["per_dim_action_losses"] = (
            losses_action.mean(dim=[0, 1]).detach()
        )
        loss_dict["per_dim_action_dim"] = int(original_action_dim)

        return loss, loss_dict
