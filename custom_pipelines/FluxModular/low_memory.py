"""Low-memory Flux Modular Diffusers blocks.

This module keeps the public Modular Diffusers block shape, but replaces the
highest-churn steps with lower-allocation variants:

* T5 prompt embeddings use dynamic padding by default instead of always 512.
* Precomputed prompt embeddings can skip both text encoders.
* Text-to-image noise is generated directly in Flux's packed latent layout.
* VAE/image intermediates are pruned once they are no longer needed.
* Transformer/text encoders are eagerly moved back to CPU before VAE decode.
* Kontext image-conditioned denoise reuses a concatenation buffer per step.
* Flux transformer attention/single-block concat buffers are reused per denoise
  pass, then cleared before VAE decode.
"""

from __future__ import annotations

import gc
from typing import Any

import numpy as np
import PIL.Image
import torch

from diffusers.modular_pipelines.modular_pipeline import (
    AutoPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
    SequentialPipelineBlocks,
)
from diffusers.modular_pipelines.modular_pipeline_utils import InputParam, InsertableDict, OutputParam
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import randn_tensor

from .before_denoise import (
    FluxImg2ImgPrepareLatentsStep,
    FluxImg2ImgSetTimestepsStep,
    FluxPrepareLatentsStep,
    FluxRoPEInputsStep,
    FluxSetTimestepsStep,
    FluxKontextRoPEInputsStep,
    _get_initial_timesteps_and_optionals,
)
from .decoders import FluxDecodeStep, _unpack_latents
from .denoise import (
    FluxDenoiseLoopWrapper,
    FluxDenoiseStep,
    FluxLoopAfterDenoiser,
    FluxLoopDenoiser,
    FluxKontextLoopDenoiser,
)
from .encoders import (
    FluxProcessImagesInputStep,
    FluxTextEncoderStep,
    FluxVaeEncoderStep,
    FluxKontextProcessImagesInputStep,
)
from .inputs import (
    FluxAdditionalInputsStep,
    FluxKontextAdditionalInputsStep,
    FluxKontextSetResolutionStep,
    FluxTextInputStep,
)
from .device_placement import (
    denoise_execution_device,
    prepare_component_for_cuda,
    prepare_transformer_for_denoise,
)


logger = logging.get_logger(__name__)


def _clear_device_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


def _collect_memory() -> None:
    gc.collect()
    _clear_device_cache()


def _offload_module_to_cpu(module: torch.nn.Module | None) -> None:
    if module is None:
        return
    hook = getattr(module, "_hf_hook", None)
    if hook is not None and hasattr(hook, "init_hook"):
        hook.init_hook(module)
    else:
        module.to("cpu")


def _has_runtime_offload_hook(module: torch.nn.Module | None) -> bool:
    hook = getattr(module, "_hf_hook", None)
    return hook is not None and hasattr(hook, "pre_forward")


def _ensure_module_device(module: torch.nn.Module | None, device: torch.device | str | None) -> None:
    if module is None or device is None or _has_runtime_offload_hook(module):
        return
    module.to(torch.device(device))


def offload_components_to_cpu(components: Any, *names: str, clear_cache: bool = True) -> None:
    """Move named module components to CPU without requiring pipeline-level offload APIs."""
    moved = False
    for name in names:
        module = getattr(components, name, None)
        if isinstance(module, torch.nn.Module):
            _offload_module_to_cpu(module)
            moved = True
    if moved and clear_cache:
        _clear_device_cache()


def _move_block_state_tensors(block_state: Any, device: torch.device, *names: str) -> None:
    for name in names:
        value = getattr(block_state, name, None)
        if torch.is_tensor(value) and value.device != device:
            setattr(block_state, name, value.to(device))


def enable_flux_vae_memory_savers(components: Any) -> None:
    vae = getattr(components, "vae", None)
    if vae is None:
        return
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()


def _clear_transformer_workspace(components: Any) -> None:
    transformer = getattr(components, "transformer", None)
    if transformer is None:
        return
    from .transformer_memory import clear_low_memory_flux_transformer_buffers

    clear_low_memory_flux_transformer_buffers(transformer)


def enable_low_memory_flux_modular(
    pipe: Any,
    *,
    device: str | int | torch.device | None = None,
    memory_reserve_margin: str = "3GB",
    enable_vae_savers: bool = True,
    layerwise_casting: bool = False,
    storage_dtype: torch.dtype | None = None,
    compute_dtype: torch.dtype | None = None,
) -> str:
    """Configure a local Flux modular pipeline for lower memory use.

    Returns the main offload mode applied. Quantization is intentionally not
    automatic here because the best backend depends on installed optional
    packages and desired quality/speed tradeoffs.
    """
    if enable_vae_savers:
        enable_flux_vae_memory_savers(pipe)

    if layerwise_casting:
        from diffusers.hooks import apply_layerwise_casting

        storage_dtype = storage_dtype or torch.float8_e4m3fn
        compute_dtype = compute_dtype or getattr(getattr(pipe, "transformer", None), "dtype", torch.bfloat16)
        transformer = getattr(pipe, "transformer", None)
        text_encoder_2 = getattr(pipe, "text_encoder_2", None)
        if transformer is not None:
            apply_layerwise_casting(transformer, storage_dtype=storage_dtype, compute_dtype=compute_dtype)
        if text_encoder_2 is not None:
            apply_layerwise_casting(text_encoder_2, storage_dtype=storage_dtype, compute_dtype=compute_dtype)

    manager = getattr(pipe, "_components_manager", None)
    if manager is not None:
        manager.enable_auto_cpu_offload(device=device, memory_reserve_margin=memory_reserve_margin)
        return "components_manager_auto_cpu_offload"

    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
        return "model_cpu_offload"

    return "manual_block_offload"


class LowMemoryFluxTextEncoderStep(FluxTextEncoderStep):
    """Flux text encoder with dynamic T5 padding and embedding passthrough."""

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("prompt"),
            InputParam("prompt_2"),
            InputParam(
                "prompt_embeds",
                type_hint=torch.Tensor | None,
                kwargs_type="denoiser_input_fields",
                description="Optional precomputed T5 prompt embeddings. Skips text encoders when paired with pooled_prompt_embeds.",
            ),
            InputParam(
                "pooled_prompt_embeds",
                type_hint=torch.Tensor | None,
                kwargs_type="denoiser_input_fields",
                description="Optional precomputed CLIP pooled prompt embeddings.",
            ),
            InputParam(
                "max_sequence_length",
                type_hint=int | None,
                default=None,
                required=False,
                description="Maximum T5 length. None uses the longest actual prompt length capped at 512.",
            ),
            InputParam("joint_attention_kwargs"),
            InputParam("low_memory_eager_offload", type_hint=bool, default=True),
            InputParam("low_memory_cuda_placement", type_hint=str, default="auto"),
            InputParam("low_memory_vram_reserve_margin", type_hint=str, default="3GB"),
        ]

    @staticmethod
    def _resolve_t5_max_length(components, prompt: list[str], max_sequence_length: int | None) -> int:
        if max_sequence_length is not None:
            return int(max_sequence_length)
        tokenized = components.tokenizer_2(prompt, padding=False, truncation=False)
        lengths = [len(ids) for ids in tokenized.input_ids]
        return max(1, min(max(lengths), 512))

    @staticmethod
    def _get_t5_prompt_embeds(components, prompt: str | list[str], max_sequence_length: int | None, device: torch.device):
        dtype = components.text_encoder_2.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt

        if hasattr(components, "maybe_convert_prompt"):
            prompt = components.maybe_convert_prompt(prompt, components.tokenizer_2)

        resolved_length = LowMemoryFluxTextEncoderStep._resolve_t5_max_length(
            components, prompt, max_sequence_length
        )
        text_inputs = components.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=resolved_length,
            truncation=True,
            return_attention_mask=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask

        untruncated_ids = components.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] > text_input_ids.shape[-1]:
            removed_text = components.tokenizer_2.batch_decode(untruncated_ids[:, resolved_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f"{resolved_length} tokens: {removed_text}"
            )

        prompt_embeds = components.text_encoder_2(
            text_input_ids.to(device),
            attention_mask=attention_mask.to(device),
            output_hidden_states=False,
        )[0]
        return prompt_embeds.to(dtype=dtype, device=device)

    @staticmethod
    def encode_prompt(
        components,
        prompt: str | list[str],
        prompt_2: str | list[str] | None,
        device: torch.device | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        pooled_prompt_embeds: torch.FloatTensor | None = None,
        max_sequence_length: int | None = None,
        lora_scale: float | None = None,
        low_memory_cuda_placement: str = "auto",
        low_memory_vram_reserve_margin: str = "3GB",
        low_memory_eager_offload: bool = True,
    ):
        device = device or components._execution_device

        if prompt_embeds is not None or pooled_prompt_embeds is not None:
            if prompt_embeds is None or pooled_prompt_embeds is None:
                raise ValueError("Pass both `prompt_embeds` and `pooled_prompt_embeds`, or neither.")
            return prompt_embeds.to(device=device), pooled_prompt_embeds.to(device=device)

        if lora_scale is not None:
            components._lora_scale = lora_scale
        if lora_scale is not None:
            if components.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(components.text_encoder, lora_scale)
            if components.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(components.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        clip_device = prepare_component_for_cuda(
            components,
            "text_encoder",
            placement=low_memory_cuda_placement,
            reserve_margin=low_memory_vram_reserve_margin,
        )
        if clip_device.type == "cpu":
            clip_device = torch.device("cpu")
        pooled_prompt_embeds = FluxTextEncoderStep._get_clip_prompt_embeds(
            components,
            prompt=prompt,
            device=clip_device,
        )
        if low_memory_eager_offload:
            offload_components_to_cpu(components, "text_encoder")

        t5_device = prepare_component_for_cuda(
            components,
            "text_encoder_2",
            placement=low_memory_cuda_placement,
            reserve_margin=low_memory_vram_reserve_margin,
        )
        if t5_device.type == "cpu":
            t5_device = torch.device("cpu")
        prompt_embeds = LowMemoryFluxTextEncoderStep._get_t5_prompt_embeds(
            components,
            prompt=prompt_2,
            max_sequence_length=max_sequence_length,
            device=t5_device,
        )
        if low_memory_eager_offload:
            offload_components_to_cpu(components, "text_encoder_2")

        if lora_scale is not None:
            if components.text_encoder is not None and USE_PEFT_BACKEND:
                unscale_lora_layers(components.text_encoder, lora_scale)
            if components.text_encoder_2 is not None and USE_PEFT_BACKEND:
                unscale_lora_layers(components.text_encoder_2, lora_scale)

        return prompt_embeds, pooled_prompt_embeds

    @staticmethod
    def check_inputs(block_state):
        prompt_embeds = getattr(block_state, "prompt_embeds", None)
        pooled_prompt_embeds = getattr(block_state, "pooled_prompt_embeds", None)
        if prompt_embeds is not None or pooled_prompt_embeds is not None:
            if prompt_embeds is None or pooled_prompt_embeds is None:
                raise ValueError("Pass both `prompt_embeds` and `pooled_prompt_embeds`, or neither.")
            return
        if getattr(block_state, "prompt", None) is None:
            raise ValueError("Provide `prompt`, or precomputed `prompt_embeds` and `pooled_prompt_embeds`.")
        FluxTextEncoderStep.check_inputs(block_state)

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        block_state.device = components._execution_device
        block_state.text_encoder_lora_scale = (
            block_state.joint_attention_kwargs.get("scale", None)
            if block_state.joint_attention_kwargs is not None
            else None
        )
        block_state.prompt_embeds, block_state.pooled_prompt_embeds = self.encode_prompt(
            components,
            prompt=block_state.prompt,
            prompt_2=block_state.prompt_2,
            prompt_embeds=block_state.prompt_embeds,
            pooled_prompt_embeds=block_state.pooled_prompt_embeds,
            device=block_state.device,
            max_sequence_length=block_state.max_sequence_length,
            lora_scale=block_state.text_encoder_lora_scale,
            low_memory_cuda_placement=block_state.low_memory_cuda_placement,
            low_memory_vram_reserve_margin=block_state.low_memory_vram_reserve_margin,
            low_memory_eager_offload=block_state.low_memory_eager_offload,
        )

        self.set_block_state(state, block_state)
        if block_state.low_memory_eager_offload:
            offload_components_to_cpu(components, "text_encoder", "text_encoder_2")
        return components, state


class LowMemoryFluxTextInputStep(FluxTextInputStep):
    """Avoids copying prompt embeddings when num_images_per_prompt is one."""

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(components, block_state)

        block_state.batch_size = block_state.prompt_embeds.shape[0]
        block_state.dtype = block_state.prompt_embeds.dtype

        if block_state.num_images_per_prompt != 1:
            _, seq_len, _ = block_state.prompt_embeds.shape
            block_state.prompt_embeds = block_state.prompt_embeds.repeat(1, block_state.num_images_per_prompt, 1)
            block_state.prompt_embeds = block_state.prompt_embeds.view(
                block_state.batch_size * block_state.num_images_per_prompt, seq_len, -1
            )
            pooled_prompt_embeds = block_state.pooled_prompt_embeds.repeat(1, block_state.num_images_per_prompt)
            block_state.pooled_prompt_embeds = pooled_prompt_embeds.view(
                block_state.batch_size * block_state.num_images_per_prompt, -1
            )

        self.set_block_state(state, block_state)
        return components, state


class LowMemoryFluxVaeEncoderStep(FluxVaeEncoderStep):
    """Prunes preprocessed image tensors and offloads VAE after encoding."""

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        components, state = super().__call__(components, state)
        state.values.pop(self._image_input_name, None)
        if state.get("low_memory_eager_offload", True):
            offload_components_to_cpu(components, "vae")
        return components, state


class LowMemoryFluxPrepareLatentsStep(FluxPrepareLatentsStep):
    """Generate text-to-image latents directly in packed Flux layout."""

    @staticmethod
    def prepare_latents(
        comp,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = 2 * (int(height) // (comp.vae_scale_factor * 2))
        width = 2 * (int(width) // (comp.vae_scale_factor * 2))

        if latents is not None:
            latents = latents.to(device=device, dtype=dtype)
            if latents.ndim == 3:
                return latents
            if latents.ndim == 4:
                return LowMemoryFluxPrepareLatentsStep._pack_existing_latents(
                    latents, batch_size, num_channels_latents, height, width
                )
            raise ValueError("`latents` must be packed [B, tokens, C*4] or unpacked [B, C, H, W].")

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        packed_shape = (batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return randn_tensor(packed_shape, generator=generator, device=device, dtype=dtype)

    @staticmethod
    def _pack_existing_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.height = block_state.height or components.default_height
        block_state.width = block_state.width or components.default_width
        block_state.device = denoise_execution_device(components)
        block_state.num_channels_latents = components.num_channels_latents

        self.check_inputs(components, block_state)
        batch_size = block_state.batch_size * block_state.num_images_per_prompt
        block_state.latents = self.prepare_latents(
            components,
            batch_size,
            block_state.num_channels_latents,
            block_state.height,
            block_state.width,
            block_state.dtype,
            block_state.device,
            block_state.generator,
            block_state.latents,
        )

        self.set_block_state(state, block_state)
        return components, state


class LowMemoryFluxImg2ImgPrepareLatentsStep(FluxImg2ImgPrepareLatentsStep):
    @property
    def inputs(self) -> list[InputParam]:
        inputs = list(super().inputs)
        inputs.append(InputParam("low_memory_prune_intermediates", type_hint=bool, default=True))
        return inputs

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        denoise_device = denoise_execution_device(components)
        block_state.device = denoise_device
        _move_block_state_tensors(
            block_state,
            denoise_device,
            "image_latents",
            "latents",
            "timesteps",
            "guidance",
        )

        self.check_inputs(image_latents=block_state.image_latents, latents=block_state.latents)

        latent_timestep = block_state.timesteps[:1].repeat(block_state.latents.shape[0])
        block_state.initial_noise = block_state.latents
        block_state.latents = components.scheduler.scale_noise(
            block_state.image_latents, latent_timestep, block_state.latents
        )

        self.set_block_state(state, block_state)
        if state.get("low_memory_prune_intermediates", True):
            state.values.pop("image_latents", None)
        return components, state


class LowMemoryFluxTransformerBufferSetupStep(ModularPipelineBlocks):
    """Install inference-only Flux transformer allocation reducers."""

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("low_memory_transformer_buffers", type_hint=bool, default=True),
            InputParam("low_memory_transformer_attention_buffers", type_hint=bool, default=True),
            InputParam("low_memory_transformer_single_block_buffers", type_hint=bool, default=True),
            InputParam("low_memory_cuda_placement", type_hint=str, default="auto"),
            InputParam("low_memory_vram_reserve_margin", type_hint=str, default="3GB"),
            InputParam("low_memory_transformer_stream_blocks", type_hint=str, default="auto"),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        transformer = getattr(components, "transformer", None)
        denoise_device = prepare_transformer_for_denoise(
            components,
            placement=block_state.low_memory_cuda_placement,
            reserve_margin=block_state.low_memory_vram_reserve_margin,
            stream_blocks=block_state.low_memory_transformer_stream_blocks,
        )
        _move_block_state_tensors(
            block_state,
            denoise_device,
            "prompt_embeds",
            "pooled_prompt_embeds",
            "image_latents",
            "latents",
        )
        self.set_block_state(state, block_state)

        enabled = (
            True
            if block_state.low_memory_transformer_buffers is None
            else block_state.low_memory_transformer_buffers
        )
        if not enabled:
            from .transformer_memory import disable_low_memory_flux_transformer_buffers

            disable_low_memory_flux_transformer_buffers(transformer)
            return components, state

        attention_buffers = (
            True
            if block_state.low_memory_transformer_attention_buffers is None
            else block_state.low_memory_transformer_attention_buffers
        )
        single_block_buffers = (
            True
            if block_state.low_memory_transformer_single_block_buffers is None
            else block_state.low_memory_transformer_single_block_buffers
        )
        if not attention_buffers or not single_block_buffers:
            from .transformer_memory import disable_low_memory_flux_transformer_buffers

            disable_low_memory_flux_transformer_buffers(transformer)
        if attention_buffers or single_block_buffers:
            from .transformer_memory import enable_low_memory_flux_transformer_buffers

            enable_low_memory_flux_transformer_buffers(
                transformer,
                attention_processors=attention_buffers,
                single_blocks=single_block_buffers,
            )
        return components, state


class LowMemoryFluxDenoiseStep(FluxDenoiseStep):
    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        components, state = super().__call__(components, state)
        if state.get("low_memory_transformer_buffers", True):
            _clear_transformer_workspace(components)
        if state.get("low_memory_eager_offload", True):
            offload_components_to_cpu(components, "transformer")
        return components, state


class LowMemoryFluxKontextLoopDenoiser(FluxKontextLoopDenoiser):
    """Reuses a latent+image concatenation buffer across denoising steps."""

    @torch.no_grad()
    def __call__(self, components, block_state, i: int, t: torch.Tensor):
        latents = block_state.latents
        latent_model_input = latents
        image_latents = block_state.image_latents
        if image_latents is not None:
            expected_shape = (
                latents.shape[0],
                latents.shape[1] + image_latents.shape[1],
                latents.shape[2],
            )
            buffer = getattr(block_state, "_latent_model_input_buffer", None)
            if (
                buffer is None
                or buffer.shape != expected_shape
                or buffer.device != latents.device
                or buffer.dtype != latents.dtype
            ):
                buffer = torch.empty(expected_shape, device=latents.device, dtype=latents.dtype)
                block_state._latent_model_input_buffer = buffer
            buffer[:, : latents.shape[1]].copy_(latents)
            buffer[:, latents.shape[1] :].copy_(image_latents)
            latent_model_input = buffer

        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        noise_pred = components.transformer(
            hidden_states=latent_model_input,
            timestep=timestep / 1000,
            guidance=block_state.guidance,
            encoder_hidden_states=block_state.prompt_embeds,
            pooled_projections=block_state.pooled_prompt_embeds,
            joint_attention_kwargs=block_state.joint_attention_kwargs,
            txt_ids=block_state.txt_ids,
            img_ids=block_state.img_ids,
            return_dict=False,
        )[0]
        block_state.noise_pred = noise_pred[:, : latents.size(1)]
        return components, block_state


class LowMemoryFluxKontextDenoiseStep(FluxDenoiseLoopWrapper):
    model_name = "flux-kontext"
    block_classes = [LowMemoryFluxKontextLoopDenoiser, FluxLoopAfterDenoiser]
    block_names = ["denoiser", "after_denoiser"]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        components, state = super().__call__(components, state)
        if state.get("low_memory_transformer_buffers", True):
            _clear_transformer_workspace(components)
        if state.get("low_memory_eager_offload", True):
            offload_components_to_cpu(components, "transformer")
        return components, state


class LowMemoryBeforeDecodeCleanupStep(ModularPipelineBlocks):
    """Drop denoise-only state and offload heavy modules before VAE decode."""

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="Denoised packed latents that must survive cleanup for decode.",
            ),
            InputParam("output_type", default="pil"),
            InputParam("low_memory_eager_offload", type_hint=bool, default=True),
            InputParam("low_memory_prune_intermediates", type_hint=bool, default=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("latents", type_hint=torch.Tensor, description="Denoised packed latents.")]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.latents = block_state.latents.contiguous()

        if state.get("low_memory_transformer_buffers", True):
            _clear_transformer_workspace(components)

        if block_state.low_memory_eager_offload:
            offload_components_to_cpu(components, "transformer", "text_encoder", "text_encoder_2")

        if block_state.low_memory_prune_intermediates:
            for name in (
                "prompt",
                "prompt_2",
                "prompt_embeds",
                "pooled_prompt_embeds",
                "text_encoder_lora_scale",
                "txt_ids",
                "img_ids",
                "timesteps",
                "guidance",
                "sigmas",
                "num_inference_steps",
                "initial_noise",
                "noise_pred",
                "image_latents",
                "processed_image",
                "image_height",
                "image_width",
                "dtype",
                "batch_size",
                "num_channels_latents",
                "joint_attention_kwargs",
                "generator",
            ):
                state.values.pop(name, None)

        self.set_block_state(state, block_state)
        _collect_memory()
        return components, state


class LowMemoryFluxDecodeStep(FluxDecodeStep):
    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("output_type", default="pil"),
            InputParam("height", default=1024),
            InputParam("width", default=1024),
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The denoised packed latents from the denoising step",
            ),
            InputParam("decode_chunk_size", type_hint=int, default=1),
            InputParam(
                "vae_decode_device",
                description="Optional device for VAE decode. Use 'cpu' for minimum VRAM or leave None for the pipeline execution device.",
            ),
            InputParam("low_memory_eager_offload", type_hint=bool, default=True),
            InputParam("low_memory_prune_intermediates", type_hint=bool, default=True),
            InputParam("low_memory_cuda_placement", type_hint=str, default="auto"),
            InputParam("low_memory_vram_reserve_margin", type_hint=str, default="3GB"),
        ]

    @property
    def intermediate_outputs(self) -> list[str]:
        return [
            OutputParam(
                "images",
                type_hint=list[PIL.Image.Image] | torch.Tensor | np.ndarray,
                description="Generated images, tensor, ndarray, or packed latents.",
            )
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        if block_state.low_memory_eager_offload:
            offload_components_to_cpu(components, "transformer", "text_encoder", "text_encoder_2")
        enable_flux_vae_memory_savers(components)

        if block_state.output_type == "latent":
            block_state.images = block_state.latents
        else:
            vae = components.vae
            if block_state.vae_decode_device and block_state.vae_decode_device != "auto":
                decode_device = torch.device(block_state.vae_decode_device)
                _ensure_module_device(vae, decode_device)
            else:
                decode_device = prepare_component_for_cuda(
                    components,
                    "vae",
                    placement=block_state.low_memory_cuda_placement,
                    reserve_margin=block_state.low_memory_vram_reserve_margin,
                )
            _ensure_module_device(vae, decode_device)
            chunk_size = max(1, int(block_state.decode_chunk_size or 1))
            image_chunks = []
            for start in range(0, block_state.latents.shape[0], chunk_size):
                packed_chunk = block_state.latents[start : start + chunk_size].to(decode_device)
                latents = _unpack_latents(
                    packed_chunk, block_state.height, block_state.width, components.vae_scale_factor
                )
                latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
                decoded = vae.decode(latents, return_dict=False)[0]
                processed = components.image_processor.postprocess(decoded, output_type=block_state.output_type)
                if block_state.output_type == "pil":
                    image_chunks.extend(processed)
                else:
                    image_chunks.append(processed)
                del latents, decoded, processed, packed_chunk
                _clear_device_cache()

            if block_state.output_type == "pil":
                block_state.images = image_chunks
            elif block_state.output_type == "pt":
                block_state.images = torch.cat(image_chunks, dim=0)
            elif block_state.output_type == "np":
                block_state.images = np.concatenate(image_chunks, axis=0)
            else:
                block_state.images = image_chunks

        self.set_block_state(state, block_state)
        if block_state.low_memory_prune_intermediates and block_state.output_type != "latent":
            for name in (
                "latents",
                "prompt_embeds",
                "pooled_prompt_embeds",
                "txt_ids",
                "img_ids",
                "initial_noise",
                "image_latents",
            ):
                state.values.pop(name, None)
        return components, state


class LowMemoryFluxImg2ImgVaeEncoderStep(SequentialPipelineBlocks):
    model_name = "flux"
    block_classes = [FluxProcessImagesInputStep(), LowMemoryFluxVaeEncoderStep()]
    block_names = ["preprocess", "encode"]


class LowMemoryFluxAutoVaeEncoderStep(AutoPipelineBlocks):
    model_name = "flux"
    block_classes = [LowMemoryFluxImg2ImgVaeEncoderStep]
    block_names = ["img2img"]
    block_trigger_inputs = ["image"]


class LowMemoryFluxSetTimestepsStep(FluxSetTimestepsStep):
    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.device = denoise_execution_device(components)

        scheduler = components.scheduler
        transformer = components.transformer
        batch_size = block_state.batch_size * block_state.num_images_per_prompt
        timesteps, num_inference_steps, sigmas, guidance = _get_initial_timesteps_and_optionals(
            transformer,
            scheduler,
            batch_size,
            block_state.height,
            block_state.width,
            components.vae_scale_factor,
            block_state.num_inference_steps,
            block_state.guidance_scale,
            block_state.sigmas,
            block_state.device,
        )
        block_state.timesteps = timesteps
        block_state.num_inference_steps = num_inference_steps
        block_state.sigmas = sigmas
        block_state.guidance = guidance
        components.scheduler.set_begin_index(0)

        self.set_block_state(state, block_state)
        return components, state


class LowMemoryFluxImg2ImgSetTimestepsStep(FluxImg2ImgSetTimestepsStep):
    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.device = denoise_execution_device(components)
        block_state.height = block_state.height or components.default_height
        block_state.width = block_state.width or components.default_width

        scheduler = components.scheduler
        transformer = components.transformer
        batch_size = block_state.batch_size * block_state.num_images_per_prompt
        timesteps, num_inference_steps, sigmas, guidance = _get_initial_timesteps_and_optionals(
            transformer,
            scheduler,
            batch_size,
            block_state.height,
            block_state.width,
            components.vae_scale_factor,
            block_state.num_inference_steps,
            block_state.guidance_scale,
            block_state.sigmas,
            block_state.device,
        )
        timesteps, num_inference_steps = self.get_timesteps(
            scheduler, num_inference_steps, block_state.strength, block_state.device
        )
        block_state.timesteps = timesteps
        block_state.num_inference_steps = num_inference_steps
        block_state.sigmas = sigmas
        block_state.guidance = guidance

        self.set_block_state(state, block_state)
        return components, state


class LowMemoryFluxBeforeDenoiseStep(SequentialPipelineBlocks):
    model_name = "flux"
    block_classes = [LowMemoryFluxPrepareLatentsStep(), LowMemoryFluxSetTimestepsStep(), FluxRoPEInputsStep()]
    block_names = ["prepare_latents", "set_timesteps", "prepare_rope_inputs"]


class LowMemoryFluxImg2ImgBeforeDenoiseStep(SequentialPipelineBlocks):
    model_name = "flux"
    block_classes = [
        LowMemoryFluxPrepareLatentsStep(),
        LowMemoryFluxImg2ImgSetTimestepsStep(),
        LowMemoryFluxImg2ImgPrepareLatentsStep(),
        FluxRoPEInputsStep(),
    ]
    block_names = ["prepare_latents", "set_timesteps", "prepare_img2img_latents", "prepare_rope_inputs"]


class LowMemoryFluxAutoBeforeDenoiseStep(AutoPipelineBlocks):
    model_name = "flux"
    block_classes = [LowMemoryFluxImg2ImgBeforeDenoiseStep, LowMemoryFluxBeforeDenoiseStep]
    block_names = ["img2img", "text2image"]
    block_trigger_inputs = ["image_latents", None]


class LowMemoryFluxImg2ImgInputStep(SequentialPipelineBlocks):
    model_name = "flux"
    block_classes = [LowMemoryFluxTextInputStep(), FluxAdditionalInputsStep()]
    block_names = ["text_inputs", "additional_inputs"]


class LowMemoryFluxAutoInputStep(AutoPipelineBlocks):
    model_name = "flux"
    block_classes = [LowMemoryFluxImg2ImgInputStep, LowMemoryFluxTextInputStep]
    block_names = ["img2img", "text2image"]
    block_trigger_inputs = ["image_latents", None]


class LowMemoryFluxCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "flux"
    block_classes = [
        LowMemoryFluxTransformerBufferSetupStep(),
        LowMemoryFluxAutoInputStep,
        LowMemoryFluxAutoBeforeDenoiseStep,
        LowMemoryFluxDenoiseStep,
    ]
    block_names = ["transformer_buffers", "input", "before_denoise", "denoise"]

    @property
    def outputs(self):
        return [OutputParam.template("latents")]


LOW_MEMORY_AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", LowMemoryFluxTextEncoderStep()),
        ("vae_encoder", LowMemoryFluxAutoVaeEncoderStep()),
        ("denoise", LowMemoryFluxCoreDenoiseStep()),
        ("cleanup", LowMemoryBeforeDecodeCleanupStep()),
        ("decode", LowMemoryFluxDecodeStep()),
    ]
)


class LowMemoryFluxAutoBlocks(SequentialPipelineBlocks):
    model_name = "flux"
    block_classes = LOW_MEMORY_AUTO_BLOCKS.values()
    block_names = LOW_MEMORY_AUTO_BLOCKS.keys()
    _workflow_map = {
        "text2image": {"prompt": True},
        "embeds2image": {"prompt_embeds": True, "pooled_prompt_embeds": True},
        "image2image": {"image": True, "prompt": True},
        "image2image_embeds": {"image": True, "prompt_embeds": True, "pooled_prompt_embeds": True},
    }

    @property
    def description(self):
        return "Low-memory local Modular pipeline for Flux text-to-image and image-to-image."

    @property
    def outputs(self):
        return [OutputParam.template("images")]


class LowMemoryFluxKontextVaeEncoderStep(SequentialPipelineBlocks):
    model_name = "flux-kontext"
    block_classes = [FluxKontextProcessImagesInputStep(), LowMemoryFluxVaeEncoderStep(sample_mode="argmax")]
    block_names = ["preprocess", "encode"]


class LowMemoryFluxKontextAutoVaeEncoderStep(AutoPipelineBlocks):
    model_name = "flux-kontext"
    block_classes = [LowMemoryFluxKontextVaeEncoderStep]
    block_names = ["image_conditioned"]
    block_trigger_inputs = ["image"]


class LowMemoryFluxKontextBeforeDenoiseStep(SequentialPipelineBlocks):
    model_name = "flux-kontext"
    block_classes = [LowMemoryFluxPrepareLatentsStep(), LowMemoryFluxSetTimestepsStep(), FluxRoPEInputsStep()]
    block_names = ["prepare_latents", "set_timesteps", "prepare_rope_inputs"]


class LowMemoryFluxKontextImageConditionedBeforeDenoiseStep(SequentialPipelineBlocks):
    model_name = "flux-kontext"
    block_classes = [LowMemoryFluxPrepareLatentsStep(), LowMemoryFluxSetTimestepsStep(), FluxKontextRoPEInputsStep()]
    block_names = ["prepare_latents", "set_timesteps", "prepare_rope_inputs"]


class LowMemoryFluxKontextAutoBeforeDenoiseStep(AutoPipelineBlocks):
    model_name = "flux-kontext"
    block_classes = [LowMemoryFluxKontextImageConditionedBeforeDenoiseStep, LowMemoryFluxKontextBeforeDenoiseStep]
    block_names = ["image_conditioned", "text2image"]
    block_trigger_inputs = ["image_latents", None]


class LowMemoryFluxKontextInputStep(SequentialPipelineBlocks):
    model_name = "flux-kontext"
    block_classes = [FluxKontextSetResolutionStep(), LowMemoryFluxTextInputStep(), FluxKontextAdditionalInputsStep()]
    block_names = ["set_resolution", "text_inputs", "additional_inputs"]


class LowMemoryFluxKontextAutoInputStep(AutoPipelineBlocks):
    model_name = "flux-kontext"
    block_classes = [LowMemoryFluxKontextInputStep, LowMemoryFluxTextInputStep]
    block_names = ["image_conditioned", "text2image"]
    block_trigger_inputs = ["image_latents", None]


class LowMemoryFluxKontextCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "flux-kontext"
    block_classes = [
        LowMemoryFluxTransformerBufferSetupStep(),
        LowMemoryFluxKontextAutoInputStep,
        LowMemoryFluxKontextAutoBeforeDenoiseStep,
        LowMemoryFluxKontextDenoiseStep,
    ]
    block_names = ["transformer_buffers", "input", "before_denoise", "denoise"]

    @property
    def outputs(self):
        return [OutputParam.template("latents")]


LOW_MEMORY_AUTO_BLOCKS_KONTEXT = InsertableDict(
    [
        ("text_encoder", LowMemoryFluxTextEncoderStep()),
        ("vae_encoder", LowMemoryFluxKontextAutoVaeEncoderStep()),
        ("denoise", LowMemoryFluxKontextCoreDenoiseStep()),
        ("cleanup", LowMemoryBeforeDecodeCleanupStep()),
        ("decode", LowMemoryFluxDecodeStep()),
    ]
)


class LowMemoryFluxKontextAutoBlocks(SequentialPipelineBlocks):
    model_name = "flux-kontext"
    block_classes = LOW_MEMORY_AUTO_BLOCKS_KONTEXT.values()
    block_names = LOW_MEMORY_AUTO_BLOCKS_KONTEXT.keys()
    _workflow_map = {
        "image_conditioned": {"image": True, "prompt": True},
        "image_conditioned_embeds": {"image": True, "prompt_embeds": True, "pooled_prompt_embeds": True},
        "text2image": {"prompt": True},
        "embeds2image": {"prompt_embeds": True, "pooled_prompt_embeds": True},
    }

    @property
    def description(self):
        return "Low-memory local Modular pipeline for Flux Kontext."

    @property
    def outputs(self):
        return [OutputParam.template("images")]
