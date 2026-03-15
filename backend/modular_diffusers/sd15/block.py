"""Custom Modular Diffusers blocks for SD1.5 text-to-image testing."""

import inspect
from typing import Any

import PIL.Image
import torch
import torch.nn.functional as F
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
            InputParam("prompt", type_hint=str | list[str]),
            InputParam("negative_prompt", type_hint=str | list[str]),
            InputParam("prompt_embeds", type_hint=torch.Tensor | None),
            InputParam("negative_prompt_embeds", type_hint=torch.Tensor | None),
            InputParam("image", type_hint=PIL.Image.Image | list[PIL.Image.Image] | torch.Tensor | None),
            InputParam("mask_image", type_hint=PIL.Image.Image | list[PIL.Image.Image] | torch.Tensor | None),
            InputParam("height", type_hint=int),
            InputParam("width", type_hint=int),
            InputParam("num_inference_steps", type_hint=int, default=50),
            InputParam("guidance_scale", type_hint=float, default=7.5),
            InputParam("num_images_per_prompt", type_hint=int, default=1),
            InputParam("strength", type_hint=float, default=0.8),
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
    def _validate_inputs(block_state) -> None:
        has_prompt = block_state.prompt is not None
        has_prompt_embeds = block_state.prompt_embeds is not None

        if not has_prompt and not has_prompt_embeds:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")

        if has_prompt and has_prompt_embeds:
            raise ValueError("Pass either `prompt` or `prompt_embeds`, not both.")

        if block_state.prompt_embeds is not None and block_state.prompt_embeds.ndim != 3:
            raise ValueError("`prompt_embeds` must be a 3D tensor of shape [batch, sequence, hidden].")

        if block_state.negative_prompt_embeds is not None and block_state.negative_prompt_embeds.ndim != 3:
            raise ValueError("`negative_prompt_embeds` must be a 3D tensor of shape [batch, sequence, hidden].")

        if (
            block_state.prompt_embeds is not None
            and block_state.negative_prompt_embeds is not None
            and block_state.prompt_embeds.shape != block_state.negative_prompt_embeds.shape
        ):
            raise ValueError("`negative_prompt_embeds` must have the same shape as `prompt_embeds`.")

        if isinstance(block_state.prompt, list):
            SD15Text2ImgBlocks._check_prompt_batch(
                block_state.prompt,
                block_state.negative_prompt,
                len(block_state.prompt),
            )

        if block_state.mask_image is not None and block_state.image is None:
            raise ValueError("`mask_image` requires `image` for inpaint.")

        if block_state.image is not None and not 0 < block_state.strength <= 1:
            raise ValueError("`strength` must be greater than 0 and less than or equal to 1 for img2img/inpaint.")

        if (
            block_state.image is not None
            and block_state.mask_image is not None
            and isinstance(block_state.image, PIL.Image.Image)
            and isinstance(block_state.mask_image, PIL.Image.Image)
            and block_state.image.size != block_state.mask_image.size
        ):
            raise ValueError("`image` and `mask_image` must have the same size for inpaint.")

    @staticmethod
    def _batch_size(block_state) -> int:
        if isinstance(block_state.prompt, list):
            return len(block_state.prompt)
        if block_state.prompt is not None:
            return 1
        if block_state.prompt_embeds is not None:
            return int(block_state.prompt_embeds.shape[0])
        raise ValueError("Unable to determine batch size from inputs.")

    @staticmethod
    def _prepare_prompt_embeds(
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
        num_images_per_prompt: int,
        guidance_scale: float,
    ) -> torch.Tensor:
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        if guidance_scale <= 1.0:
            return prompt_embeds

        if negative_prompt_embeds is None:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds[: prompt_embeds.shape[0] // num_images_per_prompt])
        negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
        negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        return torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    @staticmethod
    def _get_batch_size_for_image(image: PIL.Image.Image | list[PIL.Image.Image] | torch.Tensor) -> int:
        if isinstance(image, list):
            return len(image)
        if isinstance(image, torch.Tensor):
            return int(image.shape[0])
        return 1

    @staticmethod
    def _validate_image_batch(block_state, batch_size: int) -> None:
        if block_state.image is None:
            return

        image_batch_size = SD15Text2ImgBlocks._get_batch_size_for_image(block_state.image)
        if image_batch_size not in (1, batch_size):
            raise ValueError(
                "`image` batch size must be 1 or match the prompt batch size for img2img."
            )

        if block_state.mask_image is not None:
            mask_batch_size = SD15Text2ImgBlocks._get_batch_size_for_image(block_state.mask_image)
            if mask_batch_size not in (1, batch_size):
                raise ValueError("`mask_image` batch size must be 1 or match the prompt batch size for inpaint.")

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
    def _validate_latents(
        components,
        latents: torch.Tensor | None,
        batch_size: int,
        height: int,
        width: int,
    ) -> None:
        if latents is None:
            return

        in_channels = 4
        if getattr(components, "unet", None) is not None:
            in_channels = components.unet.config.in_channels
        expected_shape = (
            batch_size,
            in_channels,
            height // SD15Text2ImgBlocks._vae_scale_factor(components),
            width // SD15Text2ImgBlocks._vae_scale_factor(components),
        )
        if tuple(latents.shape) != expected_shape:
            raise ValueError(f"`latents` must have shape {expected_shape}, got {tuple(latents.shape)}.")

    @staticmethod
    def _preprocess_image(
        components,
        image: PIL.Image.Image | list[PIL.Image.Image] | torch.Tensor,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        image_processor = VaeImageProcessor(vae_scale_factor=SD15Text2ImgBlocks._vae_scale_factor(components))
        image_tensor = image_processor.preprocess(image, height=height, width=width)
        return image_tensor.to(device=device, dtype=dtype)

    @staticmethod
    def _preprocess_mask(
        components,
        mask_image: PIL.Image.Image | list[PIL.Image.Image] | torch.Tensor,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mask_processor = VaeImageProcessor(
            vae_scale_factor=SD15Text2ImgBlocks._vae_scale_factor(components),
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        mask = mask_processor.preprocess(mask_image, height=height, width=width)
        latent_height = height // SD15Text2ImgBlocks._vae_scale_factor(components)
        latent_width = width // SD15Text2ImgBlocks._vae_scale_factor(components)
        if mask.shape[-2:] != (latent_height, latent_width):
            mask = F.interpolate(mask, size=(latent_height, latent_width), mode="nearest")
        return mask.to(device=device, dtype=dtype)

    @staticmethod
    def _get_timesteps_img2img(
        scheduler,
        num_inference_steps: int,
        strength: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, int]:
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        init_timestep = max(init_timestep, 1)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = scheduler.timesteps[t_start:]
        return timesteps, len(timesteps)

    @staticmethod
    def _encode_image_latents(
        components,
        image_tensor: torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        needs_upcast = bool(getattr(components.vae.config, "force_upcast", False))
        if needs_upcast:
            image_tensor = image_tensor.float()
            components.vae.to(dtype=torch.float32)

        if isinstance(generator, list):
            image_latents = []
            for index in range(image_tensor.shape[0]):
                latent = components.vae.encode(image_tensor[index : index + 1]).latent_dist.sample(generator[index])
                image_latents.append(latent)
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = components.vae.encode(image_tensor).latent_dist.sample(generator)

        if needs_upcast:
            components.vae.to(dtype=dtype)

        image_latents = image_latents.to(dtype=dtype)
        return image_latents * components.vae.config.scaling_factor

    @staticmethod
    def _prepare_img2img_latents(
        components,
        image: PIL.Image.Image | list[PIL.Image.Image] | torch.Tensor,
        batch_size: int,
        num_images_per_prompt: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_tensor = SD15Text2ImgBlocks._preprocess_image(
            components=components,
            image=image,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
        )
        image_latents = SD15Text2ImgBlocks._encode_image_latents(
            components=components,
            image_tensor=image_tensor,
            generator=generator,
            dtype=dtype,
        )

        effective_batch_size = batch_size * num_images_per_prompt
        if image_latents.shape[0] == 1 and effective_batch_size > 1:
            image_latents = image_latents.repeat(effective_batch_size, 1, 1, 1)
        elif image_latents.shape[0] != effective_batch_size:
            if effective_batch_size % image_latents.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate image latents of batch size {image_latents.shape[0]} to {effective_batch_size}."
                )
            image_latents = image_latents.repeat(effective_batch_size // image_latents.shape[0], 1, 1, 1)

        noise = randn_tensor(image_latents.shape, generator=generator, device=device, dtype=dtype)
        return components.scheduler.add_noise(image_latents, noise, timestep), image_latents, noise

    @staticmethod
    def _prepare_mask_latents(
        components,
        mask_image: PIL.Image.Image | list[PIL.Image.Image] | torch.Tensor,
        batch_size: int,
        num_images_per_prompt: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        mask = SD15Text2ImgBlocks._preprocess_mask(
            components=components,
            mask_image=mask_image,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
        )
        effective_batch_size = batch_size * num_images_per_prompt
        if mask.shape[0] == 1 and effective_batch_size > 1:
            mask = mask.repeat(effective_batch_size, 1, 1, 1)
        elif mask.shape[0] != effective_batch_size:
            if effective_batch_size % mask.shape[0] != 0:
                raise ValueError(f"Cannot duplicate mask batch size {mask.shape[0]} to {effective_batch_size}.")
            mask = mask.repeat(effective_batch_size // mask.shape[0], 1, 1, 1)
        return mask

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
        self._validate_inputs(block_state)

        device = self._execution_device(components)
        default_size = self._default_size(components)
        height = block_state.height or default_size
        width = block_state.width or default_size

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("`height` and `width` must be divisible by 8 for SD1.5 latents.")

        batch_size = self._batch_size(block_state)
        self._validate_image_batch(block_state, batch_size)
        image_batch_size = batch_size * block_state.num_images_per_prompt
        self._validate_latents(
            components=components,
            latents=block_state.latents,
            batch_size=image_batch_size,
            height=height,
            width=width,
        )

        text_encoder_dtype = getattr(components.text_encoder, "dtype", torch.float32)
        if block_state.prompt_embeds is not None:
            prompt_embeds = self._prepare_prompt_embeds(
                prompt_embeds=block_state.prompt_embeds,
                negative_prompt_embeds=block_state.negative_prompt_embeds,
                device=device,
                dtype=text_encoder_dtype,
                num_images_per_prompt=block_state.num_images_per_prompt,
                guidance_scale=block_state.guidance_scale,
            )
        else:
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

        image_latents = None
        latent_noise = None
        mask = None
        if block_state.image is not None:
            timesteps, num_inference_steps = self._get_timesteps_img2img(
                scheduler=components.scheduler,
                num_inference_steps=num_inference_steps,
                strength=block_state.strength,
                device=device,
            )
            latent_timestep = timesteps[:1].repeat(image_batch_size)
            if block_state.latents is not None:
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
            else:
                latents, image_latents, latent_noise = self._prepare_img2img_latents(
                    components=components,
                    image=block_state.image,
                    batch_size=batch_size,
                    num_images_per_prompt=block_state.num_images_per_prompt,
                    height=height,
                    width=width,
                    dtype=prompt_embeds.dtype,
                    device=device,
                    generator=block_state.generator,
                    timestep=latent_timestep,
                )
            if block_state.mask_image is not None:
                mask = self._prepare_mask_latents(
                    components=components,
                    mask_image=block_state.mask_image,
                    batch_size=batch_size,
                    num_images_per_prompt=block_state.num_images_per_prompt,
                    height=height,
                    width=width,
                    dtype=prompt_embeds.dtype,
                    device=device,
                )
        else:
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
        for timestep_index, timestep in enumerate(timesteps):
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

            if mask is not None and image_latents is not None and latent_noise is not None:
                if timestep_index < len(timesteps) - 1:
                    next_timestep = timesteps[timestep_index + 1]
                    init_latents_proper = components.scheduler.add_noise(image_latents, latent_noise, next_timestep)
                else:
                    init_latents_proper = image_latents
                latents = init_latents_proper * (1 - mask) + latents * mask

        block_state.timesteps = timesteps
        block_state.latents = latents
        block_state.images = self._decode_latents(components, latents, block_state.output_type)

        # Keep the resolved dimensions in state for debugging follow-up calls.
        block_state.height = height
        block_state.width = width
        block_state.num_inference_steps = num_inference_steps

        self.set_block_state(state, block_state)
        return components, state
