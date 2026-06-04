"""DiT (Diffusion Transformer) Action Head for LabVLA.

Ported from Spirit-v1.5 implementation. Uses cross-attention to VLM features
with AdaLayerNorm timestep conditioning for flow matching action generation.

Architecture: TimestepEncoder + N x BasicTransformerBlock(CrossAttn + FFN) + output projection
~268M params with default config (18 layers, hidden=1024, 8 heads)
"""

from typing import Optional
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from torch import nn


class TimestepEncoder(nn.Module):
    """Encode scalar timestep to embedding vector via sinusoidal + MLP."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        return self.timestep_embedder(timesteps_proj)


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization conditioned on timestep embedding.

    output = LayerNorm(x) * (1 + scale) + shift
    where (scale, shift) = Linear(SiLU(temb))
    """

    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, norm_eps, norm_elementwise_affine)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class BasicTransformerBlock(nn.Module):
    """Single DiT transformer block with cross-attention and timestep conditioning.

    Structure:
        1. AdaLayerNorm(x, temb) -> CrossAttention(query=x, kv=encoder_hidden_states) -> residual
        2. LayerNorm -> FeedForward (GEGLU) -> residual
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
    ):
        super().__init__()
        # 1. AdaLayerNorm + Cross-Attention
        self.norm1 = AdaLayerNorm(dim, norm_elementwise_affine, norm_eps)

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
        )

        # 2. LayerNorm + FeedForward
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Cross-attention with timestep conditioning.
        # Forward `encoder_attention_mask` to the diffusers Attention so padded
        # text tokens / masked-out camera slots in the VLM prefix don't contribute
        # to action-token cross-attention. `Attention.forward(attention_mask=...)`
        # accepts shape (B, S_kv) with 1=keep / 0=mask and converts internally,
        # keeping this compatible with both SDPA and flash backends.
        norm_hidden_states = self.norm1(hidden_states, temb)
        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = attn_output + hidden_states

        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states

        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class DiTActionHead(nn.Module):
    """DiT-based action head for flow matching action generation.

    Replaces the Qwen3 Transformer action expert (~760M) with a lightweight
    DiT (~268M). VLM features are used as encoder_hidden_states for
    cross-attention conditioning.

    Spirit-v1.5 interleave: when ``interleave_self_attention=True`` (default),
    odd-indexed blocks are *self-attention only* (no encoder KV) so query tokens
    (state + action) mix among themselves before the next cross-attention layer
    reads VLM features. This mirrors Spirit-v1.5
    (``use_self_attn = idx % 2 == 1 and interleave_self_attention``) and is the
    *only* path by which the state token (prepended to hidden_states by
    ``_embed_suffix``) influences action tokens before the trailing slice in
    ``_denoise_step``. Without it, state→action gradient is dead (the original
    bug) and action↔action interactions are absent (suspected cause of a
    mid-training regression on TransportBeaker).

    Config defaults (matching Spirit-v1.5):
        num_attention_heads=8, attention_head_dim=128, num_layers=18
        inner_dim = 8 * 128 = 1024
    """

    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 128,
        num_layers: int = 18,
        dropout: float = 0.0,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        positional_embeddings: Optional[str] = "sinusoidal",
        max_num_positional_embeddings: int = 512,
        cross_attention_dim: Optional[int] = None,
        interleave_self_attention: bool = True,
    ):
        super().__init__()
        self.inner_dim = num_attention_heads * attention_head_dim
        self.num_layers = num_layers
        self.interleave_self_attention = bool(interleave_self_attention)
        self.gradient_checkpointing = False
        # Store dropout so forward() can decide whether to preserve RNG state
        # across gradient checkpointing: with dit_dropout > 0 and
        # preserve_rng_state=False the recompute pass would use a different
        # dropout mask than the forward pass, producing wrong gradients.
        self._dropout = float(dropout)

        # Timestep encoder
        self.timestep_encoder = TimestepEncoder(embedding_dim=self.inner_dim)

        # Transformer blocks. Per Spirit, odd-indexed blocks are self-attention
        # when interleave is on (constructed with ``cross_attention_dim=None``
        # so diffusers' ``Attention`` runs Q/K/V over ``hidden_states``).
        blocks = []
        for idx in range(num_layers):
            use_self_attn = bool(self.interleave_self_attention and (idx % 2 == 1))
            block_cross_dim = None if use_self_attn else cross_attention_dim
            blocks.append(
                BasicTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=block_cross_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=max_num_positional_embeddings,
                )
            )
        self.transformer_blocks = nn.ModuleList(blocks)
        # Cache per-block self-attn-ness so forward doesn't re-check each step.
        self._block_is_self_attn = [
            bool(self.interleave_self_attention and (i % 2 == 1)) for i in range(num_layers)
        ]
        # Cache cross-attn block count for layerwise VLM-features routing
        # (used by `LabVLAModel._project_vlm_for_dit` to slice the matching
        # number of VLM hidden states from the last-N).
        self._num_cross_attn_blocks = sum(1 for is_self in self._block_is_self_attn if not is_self)

        # Output: AdaLayerNorm-style scale/shift + projection
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,        # (B, T_q, inner_dim) — state+action suffix
        encoder_hidden_states,                # (B, S, D) Tensor OR List[Tensor] (layerwise)
        timestep: torch.Tensor,              # (B,) — flow matching timestep
        encoder_attention_mask: Optional[torch.Tensor] = None,  # (B, S) — 1=keep, 0=pad
    ) -> torch.Tensor:
        """
        `encoder_attention_mask` (B, S) — 1 for valid VLM tokens, 0 for padding
        (text padding, masked-out camera slots). When None, cross-attention attends
        to all S positions. Diffusers Attention takes the mask as `attention_mask`
        with the same convention.

        Layerwise VLM features: ``encoder_hidden_states`` may be a
        Python list of length ``self._num_cross_attn_blocks``. When it is,
        cross-attn block i consumes ``encoder_hidden_states[i]`` (mapped in
        construction order — i.e. shallow→deep across cross-attn blocks); the
        same VLM-position attention mask is reused for every layer (VLM
        padding/camera mask is shared across LM depths). Self-attn blocks
        always receive ``encoder_hidden_states=None`` regardless of list/Tensor.

        Returns:
            hidden_states: (B, T, inner_dim) — denoised action embeddings
        """
        temb = self.timestep_encoder(timestep)

        hidden_states = hidden_states.contiguous()
        layerwise = isinstance(encoder_hidden_states, (list, tuple))
        if layerwise:
            if len(encoder_hidden_states) != self._num_cross_attn_blocks:
                raise ValueError(
                    f"Layerwise encoder_hidden_states has length "
                    f"{len(encoder_hidden_states)} but DiT has "
                    f"{self._num_cross_attn_blocks} cross-attn blocks."
                )
            encoder_hidden_states = [e.contiguous() for e in encoder_hidden_states]
        else:
            encoder_hidden_states = encoder_hidden_states.contiguous()
        if encoder_attention_mask is not None:
            # Diffusers Attention forwards `attention_mask` to F.scaled_dot_product_attention,
            # which only accepts `bool` or `float` matching query dtype. The VLM-side mask
            # is typically `torch.long` (1=keep, 0=pad), so convert to bool here once and
            # let downstream broadcasting handle the (B, S) → (B, n_heads, T_q, S) reshape.
            if encoder_attention_mask.dtype != torch.bool:
                encoder_attention_mask = encoder_attention_mask.to(torch.bool)
            encoder_attention_mask = encoder_attention_mask.contiguous()

        # When dit_dropout > 0 the gradient-checkpointing recompute MUST use the
        # same dropout mask as the forward pass: preserve_rng_state=True restores
        # torch's CPU+GPU RNG so Bernoulli draws stay bit-identical, whereas False
        # silently gives wrong gradients. dropout=0 keeps the cheaper False path.
        _preserve_rng = self._dropout > 0.0
        cross_attn_counter = 0
        for idx, block in enumerate(self.transformer_blocks):
            # Spirit interleave: self-attn blocks must receive
            # encoder_hidden_states=None (so diffusers' Attention computes
            # Q/K/V from hidden_states) and a matching encoder_attention_mask=None.
            # Cross-attn blocks pass both through unchanged.
            if self._block_is_self_attn[idx]:
                block_encoder_states = None
                block_encoder_mask = None
            else:
                if layerwise:
                    block_encoder_states = encoder_hidden_states[cross_attn_counter]
                else:
                    block_encoder_states = encoder_hidden_states
                block_encoder_mask = encoder_attention_mask
                cross_attn_counter += 1

            if self.gradient_checkpointing and self.training:
                # Pass kwargs to block so adding future params to
                # BasicTransformerBlock.forward doesn't silently shift positions.
                hidden_states = torch_checkpoint(
                    block,
                    hidden_states,
                    encoder_hidden_states=block_encoder_states,
                    temb=temb,
                    encoder_attention_mask=block_encoder_mask,
                    use_reentrant=False,
                    preserve_rng_state=_preserve_rng,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=block_encoder_states,
                    temb=temb,
                    encoder_attention_mask=block_encoder_mask,
                )

        # Output shift/scale conditioning. Keep this order for compatibility
        # with existing LabVLA checkpoints trained before 2026-05-30: their
        # proj_out_1 weights were learned with the first half as shift and
        # the second half as scale.
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=-1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]

        return hidden_states
