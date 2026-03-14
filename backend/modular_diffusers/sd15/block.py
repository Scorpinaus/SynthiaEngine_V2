"""Custom Modular Diffusers blocks for SD1.5 text-to-image testing."""

import inspect
from typing import Any

import PIL.Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.modular_pipelines import ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from diffusers.schedulers import PNDMScheduler
from diffusers.utils.torch_utils import randn_tensor


def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
):
    """Copied from Diffusers Stable Diffusion pipeline to support custom schedules."""
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")

    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                " timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accepts_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                " sigma schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps


class SD15Text2ImgBlocks(ModularPipelineBlocks):
    """Minimal SD1.5 text-to-image workflow for Modular Diffusers."""

    model_name = "sd15"

    @property
    def description(self) -> str:
        return "Single-block SD1.5 text-to-image workflow for local Modular Diffusers testing."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("tokenizer", CLIPTokenizer),
            ComponentSpec("text_encoder", CLIPTextModel),
            ComponentSpec("unet", UNet2DConditionModel),
            ComponentSpec("vae", AutoencoderKL),
            ComponentSpec("scheduler", PNDMScheduler),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("prompt", type_hint=str | list[str], required=True),
            InputParam("negative_prompt", type_hint=str | list[str]),
            InputParam("height", type_hint=int),
            InputParam("width", type_hint=int),
            InputParam("num_inference_steps", type_hint=int, default=50),
            InputParam("guidance_scale", type_hint=float, default=7.5),
            InputParam("num_images_per_prompt", type_hint=int, default=1),
            InputParam("eta", type_hint=float, default=0.0),
            InputParam("generator", type_hint=torch.Generator | list[torch.Generator] | None),
            InputParam("latents", type_hint=torch.Tensor | None),
            InputParam("timesteps", type_hint=list[int] | None),
            InputParam("sigmas", type_hint=list[float] | None),
            InputParam("output_type", type_hint=str, default="pil"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("images", type_hint=list[PIL.Image.Image] | torch.Tensor),
            OutputParam("latents", type_hint=torch.Tensor),
            OutputParam("timesteps", type_hint=torch.Tensor),
        ]

    @staticmethod
    def _execution_device(components) -> torch.device:
        device = getattr(components, "_execution_device", None)
        if device is not None:
            return torch.device(device)
        if getattr(components, "unet", None) is not None:
            return next(components.unet.parameters()).device
        return torch.device("cpu")

    @staticmethod
    def _default_size(components) -> int:
        sample_size = 64
        if getattr(components, "unet", None) is not None:
            sample_size = getattr(components.unet.config, "sample_size", sample_size)
        vae_scale_factor = SD15Text2ImgBlocks._vae_scale_factor(components)
        return int(sample_size) * vae_scale_factor

    @staticmethod
    def _vae_scale_factor(components) -> int:
        if getattr(components, "vae", None) is None:
            return 8
        return 2 ** (len(components.vae.config.block_out_channels) - 1)

    @staticmethod
    def _check_prompt_batch(prompt: str | list[str], negative_prompt: str | list[str] | None, batch_size: int) -> None:
        if isinstance(negative_prompt, list) and len(negative_prompt) != batch_size:
            raise ValueError(
                "When providing `negative_prompt` as a list, it must have the same batch size as `prompt`."
            )

    @staticmethod
    def _encode_prompt(
        components,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None,
        device: torch.device,
        num_images_per_prompt: int,
        guidance_scale: float,
    ) -> torch.Tensor:
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_batch = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt_batch)

        text_inputs = components.tokenizer(
            prompt_batch,
            padding="max_length",
            max_length=components.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeds = components.text_encoder(text_inputs.input_ids.to(device), return_dict=False)[0]
        prompt_embeds = prompt_embeds.to(device=device, dtype=components.text_encoder.dtype)
        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        if not do_classifier_free_guidance:
            return prompt_embeds

        if negative_prompt is None:
            negative_prompt_batch = [""] * batch_size
        elif isinstance(negative_prompt, str):
            negative_prompt_batch = [negative_prompt] * batch_size
        else:
            negative_prompt_batch = negative_prompt

        SD15Text2ImgBlocks._check_prompt_batch(prompt, negative_prompt_batch, batch_size)

        uncond_inputs = components.tokenizer(
            negative_prompt_batch,
            padding="max_length",
            max_length=text_inputs.input_ids.shape[-1],
            truncation=True,
            return_tensors="pt",
        )
        negative_prompt_embeds = components.text_encoder(uncond_inputs.input_ids.to(device), return_dict=False)[0]
        negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=components.text_encoder.dtype)
        negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        return torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    @staticmethod
    def _prepare_latents(
        components,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None,
    ) -> torch.Tensor:
        shape = (
            batch_size,
            components.unet.config.in_channels,
            height // SD15Text2ImgBlocks._vae_scale_factor(components),
            width // SD15Text2ImgBlocks._vae_scale_factor(components),
        )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents * components.scheduler.init_noise_sigma

    @staticmethod
    def _prepare_extra_step_kwargs(components, generator, eta: float) -> dict[str, Any]:
        extra_step_kwargs: dict[str, Any] = {}
        accepts_eta = "eta" in inspect.signature(components.scheduler.step).parameters
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        accepts_generator = "generator" in inspect.signature(components.scheduler.step).parameters
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @staticmethod
    def _decode_latents(components, latents: torch.Tensor, output_type: str):
        if output_type == "latent":
            return latents

        needs_upcast = bool(getattr(components.vae.config, "force_upcast", False))
        decode_latents = latents
        if needs_upcast:
            components.vae.to(dtype=torch.float32)
            decode_latents = decode_latents.float()

        decode_latents = decode_latents / components.vae.config.scaling_factor
        image = components.vae.decode(decode_latents, return_dict=False)[0]
        if needs_upcast:
            components.vae.to(dtype=latents.dtype)

        image_processor = VaeImageProcessor(vae_scale_factor=SD15Text2ImgBlocks._vae_scale_factor(components))
        return image_processor.postprocess(image, output_type=output_type)

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)

        device = self._execution_device(components)
        default_size = self._default_size(components)
        height = block_state.height or default_size
        width = block_state.width or default_size

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("`height` and `width` must be divisible by 8 for SD1.5 latents.")

        prompt_batch = [block_state.prompt] if isinstance(block_state.prompt, str) else block_state.prompt
        batch_size = len(prompt_batch)
        image_batch_size = batch_size * block_state.num_images_per_prompt

        prompt_embeds = self._encode_prompt(
            components=components,
            prompt=block_state.prompt,
            negative_prompt=block_state.negative_prompt,
            device=device,
            num_images_per_prompt=block_state.num_images_per_prompt,
            guidance_scale=block_state.guidance_scale,
        )

        timesteps, num_inference_steps = retrieve_timesteps(
            components.scheduler,
            num_inference_steps=block_state.num_inference_steps,
            device=device,
            timesteps=block_state.timesteps,
            sigmas=block_state.sigmas,
        )
        latents = self._prepare_latents(
            components=components,
            batch_size=image_batch_size,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=block_state.generator,
            latents=block_state.latents,
        )
        extra_step_kwargs = self._prepare_extra_step_kwargs(
            components=components,
            generator=block_state.generator,
            eta=block_state.eta,
        )

        do_classifier_free_guidance = block_state.guidance_scale > 1.0
        for timestep in timesteps:
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            if hasattr(components.scheduler, "scale_model_input"):
                latent_model_input = components.scheduler.scale_model_input(latent_model_input, timestep)

            noise_pred = components.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + block_state.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = components.scheduler.step(
                noise_pred,
                timestep,
                latents,
                return_dict=False,
                **extra_step_kwargs,
            )[0]

        block_state.timesteps = timesteps
        block_state.latents = latents
        block_state.images = self._decode_latents(components, latents, block_state.output_type)

        # Keep the resolved dimensions in state for debugging follow-up calls.
        block_state.height = height
        block_state.width = width
        block_state.num_inference_steps = num_inference_steps

        self.set_block_state(state, block_state)
        return components, state
