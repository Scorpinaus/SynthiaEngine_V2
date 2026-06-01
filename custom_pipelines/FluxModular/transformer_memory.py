"""Allocation-reduction helpers for Diffusers Flux transformers.

The stock Flux transformer repeatedly concatenates large token tensors inside
the denoise loop. These helpers keep the same public modules and math, but
replace selected `torch.cat` calls with reusable workspace tensors during
inference. The workspace is intentionally not registered as module buffers so it
does not affect state dicts, saving, or parameter offload.
"""

from __future__ import annotations

import types
from typing import Any

import torch

from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux import (
    FluxAttnProcessor,
    FluxIPAdapterAttnProcessor,
    FluxSingleTransformerBlock,
    _get_qkv_projections,
)


class FluxTransformerWorkspace:
    """Reusable scratch tensors for one Flux transformer inference stream."""

    def __init__(self) -> None:
        self._buffers: dict[str, torch.Tensor] = {}

    def get(
        self,
        name: str,
        shape: tuple[int, ...] | torch.Size,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        shape = tuple(int(dim) for dim in shape)
        device = torch.device(device)
        buffer = self._buffers.get(name)
        if buffer is None or buffer.shape != shape or buffer.dtype != dtype or buffer.device != device:
            buffer = torch.empty(shape, dtype=dtype, device=device)
            self._buffers[name] = buffer
        return buffer

    def clear(self) -> None:
        self._buffers.clear()


class LowMemoryFluxAttnProcessor(FluxAttnProcessor):
    """Flux attention processor that reuses Q/K/V concat storage."""

    def __init__(self, workspace: FluxTransformerWorkspace) -> None:
        super().__init__()
        self.workspace = workspace

    def _concat_added_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        encoder_query: torch.Tensor,
        encoder_key: torch.Tensor,
        encoder_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, image_tokens, heads, head_dim = query.shape
        text_tokens = encoder_query.shape[1]
        total_tokens = text_tokens + image_tokens
        shape = (batch, total_tokens, heads, head_dim)

        query_buffer = self.workspace.get("added_query", shape, dtype=query.dtype, device=query.device)
        key_buffer = self.workspace.get("added_key", shape, dtype=key.dtype, device=key.device)
        value_buffer = self.workspace.get("added_value", shape, dtype=value.dtype, device=value.device)

        query_buffer[:, :text_tokens].copy_(encoder_query)
        query_buffer[:, text_tokens:].copy_(query)
        key_buffer[:, :text_tokens].copy_(encoder_key)
        key_buffer[:, text_tokens:].copy_(key)
        value_buffer[:, :text_tokens].copy_(encoder_value)
        value_buffer[:, text_tokens:].copy_(value)
        return query_buffer, key_buffer, value_buffer

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query, key, value = self._concat_added_qkv(
                query, key, value, encoder_query, encoder_key, encoder_value
            )

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states.contiguous())
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states.contiguous())

            return hidden_states, encoder_hidden_states

        return hidden_states


def _copy_processor_runtime_options(source: Any, target: LowMemoryFluxAttnProcessor) -> LowMemoryFluxAttnProcessor:
    target._attention_backend = getattr(source, "_attention_backend", None)
    target._parallel_config = getattr(source, "_parallel_config", None)
    return target


def enable_low_memory_flux_attention_processors(
    transformer: torch.nn.Module,
    workspace: FluxTransformerWorkspace | None = None,
) -> int:
    """Replace plain Flux attention processors with workspace-backed processors."""
    workspace = workspace or getattr(transformer, "_fluxmodular_workspace", None) or FluxTransformerWorkspace()
    setattr(transformer, "_fluxmodular_workspace", workspace)

    originals = getattr(transformer, "_fluxmodular_original_attn_processors", None)
    if originals is None:
        originals = {}
        setattr(transformer, "_fluxmodular_original_attn_processors", originals)

    patched = 0
    for name, module in transformer.named_modules():
        if not hasattr(module, "get_processor") or not hasattr(module, "set_processor"):
            continue

        processor = module.get_processor()
        if isinstance(processor, FluxIPAdapterAttnProcessor):
            continue
        if isinstance(processor, LowMemoryFluxAttnProcessor):
            processor.workspace = workspace
            patched += 1
            continue
        if processor.__class__ is not FluxAttnProcessor:
            continue

        originals.setdefault(name, processor)
        module.set_processor(_copy_processor_runtime_options(processor, LowMemoryFluxAttnProcessor(workspace)))
        patched += 1

    return patched


def _make_single_block_forward(block: FluxSingleTransformerBlock, workspace: FluxTransformerWorkspace):
    original_forward = block.forward

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.is_grad_enabled():
            return original_forward(
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        text_seq_len = encoder_hidden_states.shape[1]
        image_seq_len = hidden_states.shape[1]
        token_shape = (
            hidden_states.shape[0],
            text_seq_len + image_seq_len,
            hidden_states.shape[2],
        )
        token_buffer = workspace.get(
            "single_block_tokens",
            token_shape,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        token_buffer[:, :text_seq_len].copy_(encoder_hidden_states)
        token_buffer[:, text_seq_len:].copy_(hidden_states)

        residual = token_buffer
        norm_hidden_states, gate = self.norm(token_buffer, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        attention_width = attn_output.shape[2]
        feature_shape = (
            attn_output.shape[0],
            attn_output.shape[1],
            attention_width + mlp_hidden_states.shape[2],
        )
        feature_buffer = workspace.get(
            "single_block_features",
            feature_shape,
            dtype=attn_output.dtype,
            device=attn_output.device,
        )
        feature_buffer[:, :, :attention_width].copy_(attn_output)
        feature_buffer[:, :, attention_width:].copy_(mlp_hidden_states)

        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(feature_buffer)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
        return encoder_hidden_states, hidden_states

    return original_forward, types.MethodType(forward, block)


def enable_low_memory_flux_single_block_buffers(
    transformer: torch.nn.Module,
    workspace: FluxTransformerWorkspace | None = None,
) -> int:
    """Patch Flux single blocks to reuse token and MLP concat storage."""
    workspace = workspace or getattr(transformer, "_fluxmodular_workspace", None) or FluxTransformerWorkspace()
    setattr(transformer, "_fluxmodular_workspace", workspace)

    patched = 0
    for block in getattr(transformer, "single_transformer_blocks", []):
        if not isinstance(block, FluxSingleTransformerBlock):
            continue
        if hasattr(block, "_fluxmodular_original_forward"):
            patched += 1
            continue
        original_forward, patched_forward = _make_single_block_forward(block, workspace)
        setattr(block, "_fluxmodular_original_forward", original_forward)
        block.forward = patched_forward
        patched += 1

    return patched


def enable_low_memory_flux_transformer_buffers(
    transformer: torch.nn.Module | None,
    *,
    attention_processors: bool = True,
    single_blocks: bool = True,
) -> dict[str, int]:
    """Enable inference-only allocation reductions on a Flux transformer."""
    if transformer is None:
        return {"attention_processors": 0, "single_blocks": 0}

    workspace = getattr(transformer, "_fluxmodular_workspace", None)
    if workspace is None:
        workspace = FluxTransformerWorkspace()
        setattr(transformer, "_fluxmodular_workspace", workspace)

    stats = {"attention_processors": 0, "single_blocks": 0}
    if attention_processors:
        stats["attention_processors"] = enable_low_memory_flux_attention_processors(transformer, workspace)
    if single_blocks:
        stats["single_blocks"] = enable_low_memory_flux_single_block_buffers(transformer, workspace)
    setattr(transformer, "_fluxmodular_low_memory_buffer_stats", stats)
    return stats


def clear_low_memory_flux_transformer_buffers(transformer: torch.nn.Module | None) -> None:
    """Drop scratch tensors while keeping low-memory patches installed."""
    workspace = getattr(transformer, "_fluxmodular_workspace", None)
    if workspace is not None:
        workspace.clear()


def disable_low_memory_flux_transformer_buffers(transformer: torch.nn.Module | None) -> None:
    """Restore original Flux attention processors and single block forwards."""
    if transformer is None:
        return

    originals = getattr(transformer, "_fluxmodular_original_attn_processors", None)
    if originals is not None:
        for name, processor in list(originals.items()):
            module = transformer.get_submodule(name) if name else transformer
            if hasattr(module, "set_processor"):
                module.set_processor(processor)
        originals.clear()

    for block in getattr(transformer, "single_transformer_blocks", []):
        original_forward = getattr(block, "_fluxmodular_original_forward", None)
        if original_forward is not None:
            block.forward = original_forward
            delattr(block, "_fluxmodular_original_forward")

    clear_low_memory_flux_transformer_buffers(transformer)
    if hasattr(transformer, "_fluxmodular_workspace"):
        delattr(transformer, "_fluxmodular_workspace")
    if hasattr(transformer, "_fluxmodular_low_memory_buffer_stats"):
        delattr(transformer, "_fluxmodular_low_memory_buffer_stats")
