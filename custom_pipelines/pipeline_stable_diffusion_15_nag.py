"""
Stable Diffusion 1.5 - Normalized Attention Guidance (NAG)

This file defines a custom attention processor that plugs into Diffusers' UNet
cross-attention modules. It implements NAG in the attention-feature space
using the *batched* unconditional/conditional trick (CFG-style 2x batch):

  encoder_hidden_states = cat([neg_embeds, pos_embeds], dim=0)
  latent_model_input    = cat([latents, latents], dim=0)

Then each cross-attention call produces two attention outputs Z- and Z+ in the
same forward pass. NAG replaces the *positive* branch attention output with a
normalized extrapolation computed from (Z+, Z-).

Designed for Diffusers v0.36.x and the default PyTorch 2.0 attention path
(AttnProcessor2_0 / scaled_dot_product_attention).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class NAGConfig:
    """Parameters from the NAG paper (defaults tuned for SD1.5 in the paper)."""

    enabled: bool = True
    # Extrapolation strength in attention-feature space: Z~ = Z+ + s*(Z+ - Z-)
    scale: float = 2.0
    # L1-norm ratio clipping threshold.
    tau: float = 2.5
    # Refinement blend toward the original positive attention features.
    alpha: float = 0.375
    # Numerical stability for ratios and norms.
    eps: float = 1e-6


class NAGAttnProcessor2_0:
    """
    Drop-in replacement for diffusers.models.attention_processor.AttnProcessor2_0.

    To pass parameters through Diffusers, provide them via `cross_attention_kwargs`
    when calling the pipeline (Diffusers filters kwargs by this signature).

    Supported kwargs:
      - nag_enabled: bool
      - nag_scale: float
      - nag_tau: float
      - nag_alpha: float
      - nag_eps: float
      - nag_apply_to_self_attn: bool (default False)
    """

    def __init__(self, config: Optional[NAGConfig] = None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "NAGAttnProcessor2_0 requires PyTorch 2.0+ (scaled_dot_product_attention)."
            )
        self.config = config or NAGConfig()

    def __call__(
        self,
        attn,  # diffusers.models.attention_processor.Attention
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *,
        nag_enabled: Optional[bool] = None,
        nag_scale: Optional[float] = None,
        nag_tau: Optional[float] = None,
        nag_alpha: Optional[float] = None,
        nag_eps: Optional[float] = None,
        nag_apply_to_self_attn: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        # Keep compatibility with Diffusers, which may still pass unrelated kwargs.
        # Those will be filtered out before reaching here, but guard anyway.
        _ = kwargs

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape:
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        is_cross_attn = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Attention output: (batch, heads, seq_len, head_dim)
        attn_out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # Apply NAG only when we have the 2-branch batch available.
        cfg = self.config
        enabled = cfg.enabled if nag_enabled is None else bool(nag_enabled)
        if enabled and (is_cross_attn or nag_apply_to_self_attn) and attn_out.shape[0] % 2 == 0:
            s = cfg.scale if nag_scale is None else float(nag_scale)
            tau = cfg.tau if nag_tau is None else float(nag_tau)
            alpha = cfg.alpha if nag_alpha is None else float(nag_alpha)
            eps = cfg.eps if nag_eps is None else float(nag_eps)

            # Split unconditional/negative (first half) and positive (second half).
            z_neg, z_pos = attn_out.chunk(2, dim=0)

            # Extrapolate in attention-feature space: Z~ = Z+ + s*(Z+ - Z-)
            z_tilde = z_pos + s * (z_pos - z_neg)

            # L1 norm ratio per head & position (over feature dimension).
            # Shapes: (b, h, q, 1)
            n_pos = z_pos.abs().sum(dim=-1, keepdim=True).clamp_min(eps)
            n_tilde = z_tilde.abs().sum(dim=-1, keepdim=True).clamp_min(eps)
            r = n_tilde / n_pos

            # Clip ratio to tau, then rescale Z~ to keep the clipped ratio.
            # z_hat = (min(r, tau) / r) * z_tilde
            r_clip = r.clamp(max=tau)
            z_hat = (r_clip / (r + eps)) * z_tilde

            # Refinement: blend back toward Z+.
            z_nag = alpha * z_hat + (1.0 - alpha) * z_pos

            attn_out = torch.cat([z_neg, z_nag], dim=0)

        hidden_states = attn_out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Output projection + dropout (same as Diffusers).
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def set_unet_nag_attn_processor(unet, config: Optional[NAGConfig] = None) -> None:
    """
    Convenience: install NAGAttnProcessor2_0 for every attention module in the UNet.

    Note: This affects *both* self-attn and cross-attn modules, but NAG itself only
    triggers on cross-attn by default (encoder_hidden_states is not None).
    """

    proc = NAGAttnProcessor2_0(config=config)
    # UNet2DConditionModel exposes a dict-like setter for all attention processors.
    # We mirror Diffusers patterns and reuse the same processor instance everywhere.
    unet.set_attn_processor(proc)

