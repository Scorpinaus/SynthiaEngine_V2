from __future__ import annotations

import types
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import logging


logger = logging.get_logger(__name__)

PLACEMENT_EVENTS_ATTR = "_fluxmodular_device_placement_events"
ORIGINAL_FORWARD_ATTR = "_fluxmodular_original_forward"
STREAM_CONFIG_ATTR = "_fluxmodular_block_stream_config"


@dataclass(frozen=True)
class BlockStreamConfig:
    device: torch.device
    offload_device: torch.device
    blocks_per_group: int
    clear_cache: bool = True


def parse_memory_bytes(value: str | int | None) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return max(0, value)
    text = str(value).strip().upper().replace(" ", "")
    if not text:
        return 0
    units = (
        ("GIB", 1024**3),
        ("GB", 1000**3),
        ("MIB", 1024**2),
        ("MB", 1000**2),
        ("KIB", 1024),
        ("KB", 1000),
    )
    for suffix, scale in units:
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * scale)
    return int(float(text))


def module_nbytes(module: torch.nn.Module | None) -> int:
    if module is None:
        return 0
    total = 0
    for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
        total += tensor.numel() * tensor.element_size()
    return int(total)


def module_device(module: torch.nn.Module | None) -> torch.device | None:
    if module is None:
        return None
    for tensor in module.parameters(recurse=True):
        return tensor.device
    for tensor in module.buffers(recurse=True):
        return tensor.device
    return None


def cuda_free_bytes(device: torch.device | str = "cuda") -> tuple[int, int]:
    if not torch.cuda.is_available():
        return 0, 0
    with torch.cuda.device(torch.device(device)):
        return torch.cuda.mem_get_info()


def clear_device_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


def record_placement_event(components: Any, event: dict[str, Any]) -> None:
    events = getattr(components, PLACEMENT_EVENTS_ATTR, None)
    if events is None:
        events = []
        setattr(components, PLACEMENT_EVENTS_ATTR, events)
    events.append(event)


def get_placement_events(components: Any) -> list[dict[str, Any]]:
    return list(getattr(components, PLACEMENT_EVENTS_ATTR, []))


def _iter_groups(items: list[torch.nn.Module], group_size: int) -> Iterable[list[torch.nn.Module]]:
    group_size = max(1, int(group_size))
    for start in range(0, len(items), group_size):
        yield items[start : start + group_size]


def _to_device(module: torch.nn.Module | None, device: torch.device) -> bool:
    if module is None:
        return False
    current = module_device(module)
    if current is not None and current == device:
        return False
    module.to(device)
    return True


def _maybe_offload(module: torch.nn.Module | None, device: torch.device, clear_cache: bool) -> None:
    if module is None:
        return
    module.to(device)
    if clear_cache:
        clear_device_cache()


def choose_transformer_stream_group_size(
    transformer: torch.nn.Module,
    *,
    device: torch.device,
    reserve_margin: str | int | None,
    requested: str | int = "auto",
) -> int:
    if str(requested).lower() != "auto":
        return max(1, int(requested))

    free_bytes, _total_bytes = cuda_free_bytes(device)
    usable_bytes = max(0, free_bytes - parse_memory_bytes(reserve_margin))
    blocks = list(getattr(transformer, "transformer_blocks", [])) + list(
        getattr(transformer, "single_transformer_blocks", [])
    )
    if not blocks:
        return 1

    largest = max(module_nbytes(block) for block in blocks)
    if largest <= 0:
        return 1

    # Leave room for activations and allocator fragmentation. The estimator is
    # intentionally conservative; users can override with an explicit integer.
    per_group_budget = max(largest, usable_bytes // 3)
    group_bytes = 0
    group_size = 0
    for block in blocks:
        next_bytes = module_nbytes(block)
        if group_size > 0 and group_bytes + next_bytes > per_group_budget:
            break
        group_bytes += next_bytes
        group_size += 1
    return max(1, group_size)


def _streaming_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: dict[str, Any] | None = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
) -> torch.Tensor | Transformer2DModelOutput:
    config: BlockStreamConfig = getattr(self, STREAM_CONFIG_ATTR)
    device = config.device
    offload_device = config.offload_device

    hidden_states = hidden_states.to(device)
    encoder_hidden_states = encoder_hidden_states.to(device)
    pooled_projections = pooled_projections.to(device)
    timestep = timestep.to(device)
    if guidance is not None:
        guidance = guidance.to(device)
    txt_ids = txt_ids.to(device)
    img_ids = img_ids.to(device)

    staged_prelude = [
        self.x_embedder,
        self.time_text_embed,
        self.context_embedder,
        self.pos_embed,
    ]
    for module in staged_prelude:
        _to_device(module, device)

    hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        joint_attention_kwargs = dict(joint_attention_kwargs)
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds").to(device)
        _to_device(self.encoder_hid_proj, device)
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})
        _maybe_offload(self.encoder_hid_proj, offload_device, config.clear_cache)

    for module in staged_prelude:
        _maybe_offload(module, offload_device, config.clear_cache)

    transformer_blocks = list(self.transformer_blocks)
    for block_offset, block_group in enumerate(_iter_groups(transformer_blocks, config.blocks_per_group)):
        for block in block_group:
            _to_device(block, device)
        for group_index, block in enumerate(block_group):
            index_block = block_offset * config.blocks_per_group + group_index
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)].to(device)
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control].to(device)
        for block in block_group:
            _maybe_offload(block, offload_device, config.clear_cache)

    single_blocks = list(self.single_transformer_blocks)
    for block_offset, block_group in enumerate(_iter_groups(single_blocks, config.blocks_per_group)):
        for block in block_group:
            _to_device(block, device)
        for group_index, block in enumerate(block_group):
            index_block = block_offset * config.blocks_per_group + group_index
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control].to(
                    device
                )
        for block in block_group:
            _maybe_offload(block, offload_device, config.clear_cache)

    for module in (self.norm_out, self.proj_out):
        _to_device(module, device)
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)
    for module in (self.norm_out, self.proj_out):
        _maybe_offload(module, offload_device, config.clear_cache)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def enable_transformer_block_streaming(
    transformer: torch.nn.Module,
    *,
    device: torch.device | str,
    blocks_per_group: int,
    offload_device: torch.device | str = "cpu",
    clear_cache: bool = True,
) -> None:
    if not hasattr(transformer, ORIGINAL_FORWARD_ATTR):
        setattr(transformer, ORIGINAL_FORWARD_ATTR, transformer.forward)
    config = BlockStreamConfig(
        device=torch.device(device),
        offload_device=torch.device(offload_device),
        blocks_per_group=max(1, int(blocks_per_group)),
        clear_cache=clear_cache,
    )
    setattr(transformer, STREAM_CONFIG_ATTR, config)
    transformer.forward = types.MethodType(_streaming_forward, transformer)


def disable_transformer_block_streaming(transformer: torch.nn.Module | None) -> None:
    if transformer is None or not hasattr(transformer, ORIGINAL_FORWARD_ATTR):
        return
    transformer.forward = getattr(transformer, ORIGINAL_FORWARD_ATTR)
    delattr(transformer, ORIGINAL_FORWARD_ATTR)
    if hasattr(transformer, STREAM_CONFIG_ATTR):
        delattr(transformer, STREAM_CONFIG_ATTR)


def transformer_streaming_enabled(transformer: torch.nn.Module | None) -> bool:
    return transformer is not None and hasattr(transformer, STREAM_CONFIG_ATTR)


def prepare_transformer_for_denoise(
    components: Any,
    *,
    placement: str,
    reserve_margin: str | int | None,
    stream_blocks: str | int,
    device: torch.device | str = "cuda",
) -> torch.device:
    transformer = getattr(components, "transformer", None)
    if transformer is None:
        return torch.device("cpu")

    placement = (placement or "auto").replace("_", "-").lower()
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        record_placement_event(
            components,
            {"component": "transformer", "mode": placement, "device": "cpu", "reason": "cuda_unavailable"},
        )
        disable_transformer_block_streaming(transformer)
        transformer.to("cpu")
        return torch.device("cpu")

    if placement == "cpu":
        record_placement_event(components, {"component": "transformer", "mode": placement, "device": "cpu"})
        disable_transformer_block_streaming(transformer)
        transformer.to("cpu")
        return torch.device("cpu")

    free_bytes, _total_bytes = cuda_free_bytes(target)
    reserve_bytes = parse_memory_bytes(reserve_margin)
    usable_bytes = max(0, free_bytes - reserve_bytes)
    transformer_bytes = module_nbytes(transformer)
    full_fits = placement == "cuda" or (placement == "auto" and transformer_bytes < usable_bytes)

    if full_fits:
        disable_transformer_block_streaming(transformer)
        try:
            transformer.to(target)
            record_placement_event(
                components,
                {
                    "component": "transformer",
                    "mode": "whole-component",
                    "device": str(target),
                    "module_bytes": transformer_bytes,
                    "free_bytes": free_bytes,
                    "reserve_bytes": reserve_bytes,
                },
            )
            return target
        except RuntimeError as exc:
            if placement == "cuda" or "out of memory" not in str(exc).lower():
                raise
            transformer.to("cpu")
            clear_device_cache()

    blocks_per_group = choose_transformer_stream_group_size(
        transformer,
        device=target,
        reserve_margin=reserve_margin,
        requested=stream_blocks,
    )
    enable_transformer_block_streaming(
        transformer,
        device=target,
        blocks_per_group=blocks_per_group,
        offload_device="cpu",
        clear_cache=True,
    )
    transformer.to("cpu")
    record_placement_event(
        components,
        {
            "component": "transformer",
            "mode": "block-stream",
            "device": str(target),
            "blocks_per_group": blocks_per_group,
            "module_bytes": transformer_bytes,
            "free_bytes": free_bytes,
            "reserve_bytes": reserve_bytes,
        },
    )
    return target


def prepare_component_for_cuda(
    components: Any,
    name: str,
    *,
    placement: str,
    reserve_margin: str | int | None,
    device: torch.device | str = "cuda",
) -> torch.device:
    module = getattr(components, name, None)
    if not isinstance(module, torch.nn.Module):
        return torch.device("cpu")

    placement = (placement or "auto").replace("_", "-").lower()
    target = torch.device(device)
    if placement == "cpu" or target.type != "cuda" or not torch.cuda.is_available():
        module.to("cpu")
        return torch.device("cpu")

    free_bytes, _total_bytes = cuda_free_bytes(target)
    usable_bytes = max(0, free_bytes - parse_memory_bytes(reserve_margin))
    module_bytes = module_nbytes(module)
    if placement == "auto" and module_bytes >= usable_bytes:
        record_placement_event(
            components,
            {
                "component": name,
                "mode": "cpu-fallback",
                "device": "cpu",
                "module_bytes": module_bytes,
                "free_bytes": free_bytes,
            },
        )
        module.to("cpu")
        return torch.device("cpu")

    try:
        module.to(target)
        record_placement_event(
            components,
            {
                "component": name,
                "mode": "whole-component",
                "device": str(target),
                "module_bytes": module_bytes,
                "free_bytes": free_bytes,
            },
        )
        return target
    except RuntimeError as exc:
        if placement == "cuda" or "out of memory" not in str(exc).lower():
            raise
        module.to("cpu")
        clear_device_cache()
        return torch.device("cpu")


def denoise_execution_device(components: Any) -> torch.device:
    transformer = getattr(components, "transformer", None)
    if transformer_streaming_enabled(transformer):
        return getattr(transformer, STREAM_CONFIG_ATTR).device
    device = module_device(transformer)
    if device is not None:
        return device
    return getattr(components, "_execution_device", torch.device("cpu"))
