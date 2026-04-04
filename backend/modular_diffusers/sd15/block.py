"""Custom Modular Diffusers blocks for SD1.5 workflows."""

import inspect
from typing import Any

import PIL.Image
import torch
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.modular_pipelines import AutoPipelineBlocks, ModularPipelineBlocks, PipelineState, SequentialPipelineBlocks
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


class SD15WorkflowUtils:
    @staticmethod
    def expected_components() -> list[ComponentSpec]:
        return [
            ComponentSpec("tokenizer", CLIPTokenizer),
            ComponentSpec("text_encoder", CLIPTextModel),
            ComponentSpec("unet", UNet2DConditionModel),
            ComponentSpec("vae", AutoencoderKL),
            ComponentSpec("scheduler", PNDMScheduler),
        ]

    @staticmethod
    def execution_device(components) -> torch.device:
        device = getattr(components, "_execution_device", None)
        if device is not None:
            return torch.device(device)
        if getattr(components, "unet", None) is not None:
            return next(components.unet.parameters()).device
        return torch.device("cpu")

    @staticmethod
    def vae_scale_factor(components) -> int:
        if getattr(components, "vae", None) is None:
            return 8
        return 2 ** (len(components.vae.config.block_out_channels) - 1)

    @staticmethod
    def default_size(components) -> int:
        sample_size = 64
        if getattr(components, "unet", None) is not None:
            sample_size = getattr(components.unet.config, "sample_size", sample_size)
        return int(sample_size) * SD15WorkflowUtils.vae_scale_factor(components)

    @staticmethod
    def validate_prompt_inputs(block_state) -> None:
        prompt = getattr(block_state, "prompt", None)
        prompt_embeds = getattr(block_state, "prompt_embeds", None)
        negative_prompt_embeds = getattr(block_state, "negative_prompt_embeds", None)

        has_prompt = prompt is not None
        has_prompt_embeds = prompt_embeds is not None
        if not has_prompt and not has_prompt_embeds:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        if has_prompt and has_prompt_embeds:
            raise ValueError("Pass either `prompt` or `prompt_embeds`, not both.")
        if prompt_embeds is not None and prompt_embeds.ndim != 3:
            raise ValueError("`prompt_embeds` must be a 3D tensor of shape [batch, sequence, hidden].")
        if negative_prompt_embeds is not None and negative_prompt_embeds.ndim != 3:
            raise ValueError("`negative_prompt_embeds` must be a 3D tensor of shape [batch, sequence, hidden].")
        if (
            prompt_embeds is not None
            and negative_prompt_embeds is not None
            and prompt_embeds.shape != negative_prompt_embeds.shape
        ):
            raise ValueError("`negative_prompt_embeds` must have the same shape as `prompt_embeds`.")
        if isinstance(prompt, list):
            negative_prompt = getattr(block_state, "negative_prompt", None)
            if isinstance(negative_prompt, list) and len(negative_prompt) != len(prompt):
                raise ValueError("When providing `negative_prompt` as a list, it must have the same batch size as `prompt`.")

    @staticmethod
    def validate_dimensions(height: int, width: int) -> None:
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("`height` and `width` must be divisible by 8 for SD1.5 latents.")

    @staticmethod
    def batch_size_from_state(block_state) -> int:
        state_batch_size = getattr(block_state, "batch_size", None)
        if state_batch_size is not None:
            return int(state_batch_size)
        prompt = getattr(block_state, "prompt", None)
        prompt_embeds = getattr(block_state, "prompt_embeds", None)
        num_images_per_prompt = getattr(block_state, "num_images_per_prompt", 1) or 1
        guidance_scale = getattr(block_state, "guidance_scale", 7.5)
        if isinstance(prompt, list):
            return len(prompt)
        if prompt is not None:
            return 1
        if prompt_embeds is not None:
            num_conditions = 2 if guidance_scale > 1.0 else 1
            effective_batch_size = int(prompt_embeds.shape[0]) // num_conditions
            return max(1, effective_batch_size // num_images_per_prompt)
        raise ValueError("Unable to determine batch size from inputs.")

    @staticmethod
    def prompt_input_params() -> list[InputParam]:
        return [
            InputParam("prompt", type_hint=str | list[str]),
            InputParam("negative_prompt", type_hint=str | list[str]),
            InputParam("prompt_embeds", type_hint=torch.Tensor | None),
            InputParam("negative_prompt_embeds", type_hint=torch.Tensor | None),
            InputParam("num_images_per_prompt", type_hint=int, default=1),
            InputParam("guidance_scale", type_hint=float, default=7.5),
        ]

    @staticmethod
    def common_generation_inputs(include_image: bool = False, include_mask: bool = False) -> list[InputParam]:
        inputs = [
            InputParam("prompt_embeds", type_hint=torch.Tensor, required=True),
            InputParam("batch_size", type_hint=int),
            InputParam("num_images_per_prompt", type_hint=int, default=1),
            InputParam("height", type_hint=int),
            InputParam("width", type_hint=int),
            InputParam("num_inference_steps", type_hint=int, default=50),
            InputParam("eta", type_hint=float, default=0.0),
            InputParam("generator", type_hint=torch.Generator | list[torch.Generator] | None),
            InputParam("latents", type_hint=torch.Tensor | None),
            InputParam("timesteps", type_hint=list[int] | None),
            InputParam("sigmas", type_hint=list[float] | None),
        ]
        if include_image:
            inputs.insert(0, InputParam("image", type_hint=PIL.Image.Image | list[PIL.Image.Image] | torch.Tensor | None, required=True))
            inputs.insert(1, InputParam("strength", type_hint=float, default=0.8))
        if include_mask:
            inputs.insert(1, InputParam("mask_image", type_hint=PIL.Image.Image | list[PIL.Image.Image] | torch.Tensor | None, required=True))
        return inputs

    @staticmethod
    def output_input_param() -> InputParam:
        return InputParam("output_type", type_hint=str, default="pil")

    @staticmethod
    def prompt_encoding_outputs() -> list[OutputParam]:
        return [
            OutputParam("prompt_embeds", type_hint=torch.Tensor),
            OutputParam("batch_size", type_hint=int),
        ]

    @staticmethod
    def latent_outputs(include_mask: bool = False) -> list[OutputParam]:
        outputs = [
            OutputParam("latents", type_hint=torch.Tensor),
            OutputParam("timesteps", type_hint=torch.Tensor),
            OutputParam("num_inference_steps", type_hint=int),
        ]
        if include_mask:
            outputs.extend(
                [
                    OutputParam("image_latents", type_hint=torch.Tensor),
                    OutputParam("latent_noise", type_hint=torch.Tensor),
                    OutputParam("mask", type_hint=torch.Tensor),
                ]
            )
        return outputs

    @staticmethod
    def decode_outputs() -> list[OutputParam]:
        return [OutputParam("images", type_hint=list[PIL.Image.Image] | torch.Tensor)]

    @staticmethod
    def prepare_prompt_embeds(
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
    def encode_prompt(
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

        if isinstance(negative_prompt_batch, list) and len(negative_prompt_batch) != batch_size:
            raise ValueError("When providing `negative_prompt` as a list, it must have the same batch size as `prompt`.")

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
    def validate_and_resolve_image_inputs(block_state, require_mask: bool = False) -> tuple[Any, Any]:
        image = getattr(block_state, "image", None)
        mask_image = getattr(block_state, "mask_image", None)
        strength = getattr(block_state, "strength", None)
        if image is None:
            raise ValueError("`image` is required for img2img/inpaint.")
        if strength is None or not 0 < strength <= 1:
            raise ValueError("`strength` must be greater than 0 and less than or equal to 1 for img2img/inpaint.")
        if require_mask and mask_image is None:
            raise ValueError("`mask_image` requires `image` for inpaint.")
        if (
            image is not None
            and mask_image is not None
            and isinstance(image, PIL.Image.Image)
            and isinstance(mask_image, PIL.Image.Image)
            and image.size != mask_image.size
        ):
            raise ValueError("`image` and `mask_image` must have the same size for inpaint.")
        return image, mask_image

    @staticmethod
    def get_batch_size_for_image(image: PIL.Image.Image | list[PIL.Image.Image] | torch.Tensor) -> int:
        if isinstance(image, list):
            return len(image)
        if isinstance(image, torch.Tensor):
            return int(image.shape[0])
        return 1

    @staticmethod
    def validate_image_batch(image, batch_size: int, label: str) -> None:
        image_batch_size = SD15WorkflowUtils.get_batch_size_for_image(image)
        if image_batch_size not in (1, batch_size):
            raise ValueError(f"`{label}` batch size must be 1 or match the prompt batch size.")

    @staticmethod
    def validate_latents(components, latents: torch.Tensor | None, batch_size: int, height: int, width: int) -> None:
        if latents is None:
            return
        in_channels = 4 if getattr(components, "unet", None) is None else components.unet.config.in_channels
        expected_shape = (
            batch_size,
            in_channels,
            height // SD15WorkflowUtils.vae_scale_factor(components),
            width // SD15WorkflowUtils.vae_scale_factor(components),
        )
        if tuple(latents.shape) != expected_shape:
            raise ValueError(f"`latents` must have shape {expected_shape}, got {tuple(latents.shape)}.")

    @staticmethod
    def preprocess_image(components, image, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        image_processor = VaeImageProcessor(vae_scale_factor=SD15WorkflowUtils.vae_scale_factor(components))
        image_tensor = image_processor.preprocess(image, height=height, width=width)
        return image_tensor.to(device=device, dtype=dtype)

    @staticmethod
    def preprocess_mask(components, mask_image, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask_processor = VaeImageProcessor(
            vae_scale_factor=SD15WorkflowUtils.vae_scale_factor(components),
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        mask = mask_processor.preprocess(mask_image, height=height, width=width)
        latent_height = height // SD15WorkflowUtils.vae_scale_factor(components)
        latent_width = width // SD15WorkflowUtils.vae_scale_factor(components)
        if mask.shape[-2:] != (latent_height, latent_width):
            mask = F.interpolate(mask, size=(latent_height, latent_width), mode="nearest")
        return mask.to(device=device, dtype=dtype)


class SD15PromptEncodingStep(ModularPipelineBlocks):
    model_name = "sd15"

    @property
    def description(self) -> str:
        return "Encode text prompts or reuse precomputed prompt embeddings."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return SD15WorkflowUtils.expected_components()

    @property
    def inputs(self) -> list[InputParam]:
        return SD15WorkflowUtils.prompt_input_params()

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return SD15WorkflowUtils.prompt_encoding_outputs()

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        SD15WorkflowUtils.validate_prompt_inputs(block_state)
        device = SD15WorkflowUtils.execution_device(components)
        text_encoder_dtype = getattr(components.text_encoder, "dtype", torch.float32)
        if getattr(block_state, "prompt_embeds", None) is not None:
            block_state.batch_size = int(block_state.prompt_embeds.shape[0])
            block_state.prompt_embeds = SD15WorkflowUtils.prepare_prompt_embeds(
                prompt_embeds=block_state.prompt_embeds,
                negative_prompt_embeds=getattr(block_state, "negative_prompt_embeds", None),
                device=device,
                dtype=text_encoder_dtype,
                num_images_per_prompt=block_state.num_images_per_prompt,
                guidance_scale=block_state.guidance_scale,
            )
        else:
            block_state.batch_size = SD15WorkflowUtils.batch_size_from_state(block_state)
            block_state.prompt_embeds = SD15WorkflowUtils.encode_prompt(
                components=components,
                prompt=block_state.prompt,
                negative_prompt=getattr(block_state, "negative_prompt", None),
                device=device,
                num_images_per_prompt=block_state.num_images_per_prompt,
                guidance_scale=block_state.guidance_scale,
            )
        self.set_block_state(state, block_state)
        return components, state


def _attach_remaining_utils():
    def get_timesteps_img2img(scheduler, num_inference_steps: int, strength: float) -> tuple[torch.Tensor, int]:
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        init_timestep = max(init_timestep, 1)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = scheduler.timesteps[t_start:]
        return timesteps, len(timesteps)

    def encode_image_latents(components, image_tensor: torch.Tensor, generator, dtype: torch.dtype) -> torch.Tensor:
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

    def expand_batch(tensor: torch.Tensor, effective_batch_size: int, label: str) -> torch.Tensor:
        if tensor.shape[0] == 1 and effective_batch_size > 1:
            return tensor.repeat(effective_batch_size, 1, 1, 1)
        if tensor.shape[0] != effective_batch_size:
            if effective_batch_size % tensor.shape[0] != 0:
                raise ValueError(f"Cannot duplicate {label} batch size {tensor.shape[0]} to {effective_batch_size}.")
            return tensor.repeat(effective_batch_size // tensor.shape[0], 1, 1, 1)
        return tensor

    def prepare_img2img_latents(components, image, batch_size, num_images_per_prompt, height, width, dtype, device, generator, timestep):
        image_tensor = SD15WorkflowUtils.preprocess_image(components, image, height, width, device, dtype)
        image_latents = encode_image_latents(components, image_tensor, generator, dtype)
        effective_batch_size = batch_size * num_images_per_prompt
        image_latents = expand_batch(image_latents, effective_batch_size, "image latents")
        noise = randn_tensor(image_latents.shape, generator=generator, device=device, dtype=dtype)
        return components.scheduler.add_noise(image_latents, noise, timestep), image_latents, noise

    def prepare_mask_latents(components, mask_image, batch_size, num_images_per_prompt, height, width, dtype, device):
        mask = SD15WorkflowUtils.preprocess_mask(components, mask_image, height, width, device, dtype)
        return expand_batch(mask, batch_size * num_images_per_prompt, "mask")

    def prepare_text2img_latents(components, batch_size, height, width, dtype, device, generator, latents):
        shape = (
            batch_size,
            components.unet.config.in_channels,
            height // SD15WorkflowUtils.vae_scale_factor(components),
            width // SD15WorkflowUtils.vae_scale_factor(components),
        )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)
        return latents * components.scheduler.init_noise_sigma

    def prepare_extra_step_kwargs(components, generator, eta: float) -> dict[str, Any]:
        extra_step_kwargs: dict[str, Any] = {}
        if "eta" in inspect.signature(components.scheduler.step).parameters:
            extra_step_kwargs["eta"] = eta
        if "generator" in inspect.signature(components.scheduler.step).parameters:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def decode_latents(components, latents: torch.Tensor, output_type: str):
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
        image_processor = VaeImageProcessor(vae_scale_factor=SD15WorkflowUtils.vae_scale_factor(components))
        return image_processor.postprocess(image, output_type=output_type)

    SD15WorkflowUtils.get_timesteps_img2img = staticmethod(get_timesteps_img2img)
    SD15WorkflowUtils.encode_image_latents = staticmethod(encode_image_latents)
    SD15WorkflowUtils.expand_batch = staticmethod(expand_batch)
    SD15WorkflowUtils.prepare_img2img_latents = staticmethod(prepare_img2img_latents)
    SD15WorkflowUtils.prepare_mask_latents = staticmethod(prepare_mask_latents)
    SD15WorkflowUtils.prepare_text2img_latents = staticmethod(prepare_text2img_latents)
    SD15WorkflowUtils.prepare_extra_step_kwargs = staticmethod(prepare_extra_step_kwargs)
    SD15WorkflowUtils.decode_latents = staticmethod(decode_latents)


_attach_remaining_utils()


class SD15Text2ImgLatentsStep(ModularPipelineBlocks):
    model_name = "sd15"

    @property
    def description(self) -> str:
        return "Prepare text-to-image timesteps and random latents."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return SD15WorkflowUtils.expected_components()

    @property
    def inputs(self) -> list[InputParam]:
        return SD15WorkflowUtils.common_generation_inputs()

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return SD15WorkflowUtils.latent_outputs()

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        device = SD15WorkflowUtils.execution_device(components)
        default_size = SD15WorkflowUtils.default_size(components)
        height = block_state.height or default_size
        width = block_state.width or default_size
        SD15WorkflowUtils.validate_dimensions(height, width)
        batch_size = SD15WorkflowUtils.batch_size_from_state(block_state)
        image_batch_size = batch_size * block_state.num_images_per_prompt
        SD15WorkflowUtils.validate_latents(components, block_state.latents, image_batch_size, height, width)
        timesteps, num_inference_steps = retrieve_timesteps(
            components.scheduler,
            num_inference_steps=block_state.num_inference_steps,
            device=device,
            timesteps=getattr(block_state, "timesteps", None),
            sigmas=getattr(block_state, "sigmas", None),
        )
        block_state.timesteps = timesteps
        block_state.num_inference_steps = num_inference_steps
        block_state.latents = SD15WorkflowUtils.prepare_text2img_latents(
            components, image_batch_size, height, width, block_state.prompt_embeds.dtype, device, block_state.generator, block_state.latents
        )
        block_state.height = height
        block_state.width = width
        self.set_block_state(state, block_state)
        return components, state


class SD15Img2ImgLatentsStep(ModularPipelineBlocks):
    model_name = "sd15"

    @property
    def description(self) -> str:
        return "Prepare img2img timesteps and noised image latents."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return SD15WorkflowUtils.expected_components()

    @property
    def inputs(self) -> list[InputParam]:
        return SD15WorkflowUtils.common_generation_inputs(include_image=True)

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return SD15WorkflowUtils.latent_outputs(include_mask=True)

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        device = SD15WorkflowUtils.execution_device(components)
        default_size = SD15WorkflowUtils.default_size(components)
        height = block_state.height or default_size
        width = block_state.width or default_size
        SD15WorkflowUtils.validate_dimensions(height, width)
        batch_size = SD15WorkflowUtils.batch_size_from_state(block_state)
        image, _ = SD15WorkflowUtils.validate_and_resolve_image_inputs(block_state, require_mask=False)
        SD15WorkflowUtils.validate_image_batch(image, batch_size, "image")
        image_batch_size = batch_size * block_state.num_images_per_prompt
        SD15WorkflowUtils.validate_latents(components, block_state.latents, image_batch_size, height, width)
        timesteps, num_inference_steps = retrieve_timesteps(
            components.scheduler,
            num_inference_steps=block_state.num_inference_steps,
            device=device,
            timesteps=getattr(block_state, "timesteps", None),
            sigmas=getattr(block_state, "sigmas", None),
        )
        timesteps, num_inference_steps = SD15WorkflowUtils.get_timesteps_img2img(components.scheduler, num_inference_steps, block_state.strength)
        block_state.timesteps = timesteps
        block_state.num_inference_steps = num_inference_steps
        block_state.height = height
        block_state.width = width
        if block_state.latents is not None:
            block_state.latents = SD15WorkflowUtils.prepare_text2img_latents(
                components, image_batch_size, height, width, block_state.prompt_embeds.dtype, device, block_state.generator, block_state.latents
            )
            block_state.image_latents = None
            block_state.latent_noise = None
        else:
            latent_timestep = timesteps[:1].repeat(image_batch_size)
            latents, image_latents, latent_noise = SD15WorkflowUtils.prepare_img2img_latents(
                components,
                image,
                batch_size,
                block_state.num_images_per_prompt,
                height,
                width,
                block_state.prompt_embeds.dtype,
                device,
                block_state.generator,
                latent_timestep,
            )
            block_state.latents = latents
            block_state.image_latents = image_latents
            block_state.latent_noise = latent_noise
        block_state.mask = None
        self.set_block_state(state, block_state)
        return components, state


class SD15InpaintLatentsStep(ModularPipelineBlocks):
    model_name = "sd15"

    @property
    def description(self) -> str:
        return "Prepare inpaint timesteps, masks, and noised image latents."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return SD15WorkflowUtils.expected_components()

    @property
    def inputs(self) -> list[InputParam]:
        return SD15WorkflowUtils.common_generation_inputs(include_image=True, include_mask=True)

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return SD15WorkflowUtils.latent_outputs(include_mask=True)

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        device = SD15WorkflowUtils.execution_device(components)
        default_size = SD15WorkflowUtils.default_size(components)
        height = block_state.height or default_size
        width = block_state.width or default_size
        SD15WorkflowUtils.validate_dimensions(height, width)
        batch_size = SD15WorkflowUtils.batch_size_from_state(block_state)
        image, mask_image = SD15WorkflowUtils.validate_and_resolve_image_inputs(block_state, require_mask=True)
        SD15WorkflowUtils.validate_image_batch(image, batch_size, "image")
        SD15WorkflowUtils.validate_image_batch(mask_image, batch_size, "mask_image")
        image_batch_size = batch_size * block_state.num_images_per_prompt
        SD15WorkflowUtils.validate_latents(components, block_state.latents, image_batch_size, height, width)
        timesteps, num_inference_steps = retrieve_timesteps(
            components.scheduler,
            num_inference_steps=block_state.num_inference_steps,
            device=device,
            timesteps=getattr(block_state, "timesteps", None),
            sigmas=getattr(block_state, "sigmas", None),
        )
        timesteps, num_inference_steps = SD15WorkflowUtils.get_timesteps_img2img(components.scheduler, num_inference_steps, block_state.strength)
        block_state.timesteps = timesteps
        block_state.num_inference_steps = num_inference_steps
        block_state.height = height
        block_state.width = width
        if block_state.latents is not None:
            block_state.latents = SD15WorkflowUtils.prepare_text2img_latents(
                components, image_batch_size, height, width, block_state.prompt_embeds.dtype, device, block_state.generator, block_state.latents
            )
            block_state.image_latents = None
            block_state.latent_noise = None
        else:
            latent_timestep = timesteps[:1].repeat(image_batch_size)
            latents, image_latents, latent_noise = SD15WorkflowUtils.prepare_img2img_latents(
                components,
                image,
                batch_size,
                block_state.num_images_per_prompt,
                height,
                width,
                block_state.prompt_embeds.dtype,
                device,
                block_state.generator,
                latent_timestep,
            )
            block_state.latents = latents
            block_state.image_latents = image_latents
            block_state.latent_noise = latent_noise
        block_state.mask = SD15WorkflowUtils.prepare_mask_latents(
            components,
            mask_image,
            batch_size,
            block_state.num_images_per_prompt,
            height,
            width,
            block_state.prompt_embeds.dtype,
            device,
        )
        self.set_block_state(state, block_state)
        return components, state


class SD15DenoiseStep(ModularPipelineBlocks):
    model_name = "sd15"

    @property
    def description(self) -> str:
        return "Run the SD1.5 denoising loop."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return SD15WorkflowUtils.expected_components()

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("prompt_embeds", type_hint=torch.Tensor, required=True),
            InputParam("guidance_scale", type_hint=float, default=7.5),
            InputParam("latents", type_hint=torch.Tensor, required=True),
            InputParam("timesteps", type_hint=torch.Tensor, required=True),
            InputParam("eta", type_hint=float, default=0.0),
            InputParam("generator", type_hint=torch.Generator | list[torch.Generator] | None),
            InputParam("image_latents", type_hint=torch.Tensor | None),
            InputParam("latent_noise", type_hint=torch.Tensor | None),
            InputParam("mask", type_hint=torch.Tensor | None),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("latents", type_hint=torch.Tensor)]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        extra_step_kwargs = SD15WorkflowUtils.prepare_extra_step_kwargs(components, block_state.generator, block_state.eta)
        do_classifier_free_guidance = block_state.guidance_scale > 1.0
        image_latents = getattr(block_state, "image_latents", None)
        latent_noise = getattr(block_state, "latent_noise", None)
        mask = getattr(block_state, "mask", None)

        for timestep_index, timestep in enumerate(block_state.timesteps):
            latent_model_input = torch.cat([block_state.latents] * 2) if do_classifier_free_guidance else block_state.latents
            if hasattr(components.scheduler, "scale_model_input"):
                latent_model_input = components.scheduler.scale_model_input(latent_model_input, timestep)
            noise_pred = components.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=block_state.prompt_embeds,
                return_dict=False,
            )[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + block_state.guidance_scale * (noise_pred_text - noise_pred_uncond)
            block_state.latents = components.scheduler.step(
                noise_pred,
                timestep,
                block_state.latents,
                return_dict=False,
                **extra_step_kwargs,
            )[0]
            if mask is not None and image_latents is not None and latent_noise is not None:
                if timestep_index < len(block_state.timesteps) - 1:
                    next_timestep = block_state.timesteps[timestep_index + 1]
                    init_latents_proper = components.scheduler.add_noise(image_latents, latent_noise, next_timestep)
                else:
                    init_latents_proper = image_latents
                block_state.latents = init_latents_proper * (1 - mask) + block_state.latents * mask

        self.set_block_state(state, block_state)
        return components, state


class SD15DecodeStep(ModularPipelineBlocks):
    model_name = "sd15"

    @property
    def description(self) -> str:
        return "Decode SD1.5 latents into the requested output format."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return SD15WorkflowUtils.expected_components()

    @property
    def inputs(self) -> list[InputParam]:
        return [InputParam("latents", type_hint=torch.Tensor, required=True), SD15WorkflowUtils.output_input_param()]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return SD15WorkflowUtils.decode_outputs()

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        block_state.images = SD15WorkflowUtils.decode_latents(components, block_state.latents, block_state.output_type)
        self.set_block_state(state, block_state)
        return components, state


class SD15Text2ImgBlocks(SequentialPipelineBlocks):
    block_classes = [SD15PromptEncodingStep, SD15Text2ImgLatentsStep, SD15DenoiseStep, SD15DecodeStep]
    block_names = ["prompt_encode", "prepare_latents", "denoise", "decode"]

    @property
    def description(self) -> str:
        return "Sequential SD1.5 text-to-image workflow."

    def get_execution_blocks(self, **kwargs):
        return self


class SD15Img2ImgBlocks(SequentialPipelineBlocks):
    block_classes = [SD15PromptEncodingStep, SD15Img2ImgLatentsStep, SD15DenoiseStep, SD15DecodeStep]
    block_names = ["prompt_encode", "prepare_latents", "denoise", "decode"]

    @property
    def description(self) -> str:
        return "Sequential SD1.5 img2img workflow."

    def get_execution_blocks(self, **kwargs):
        return self


class SD15InpaintBlocks(SequentialPipelineBlocks):
    block_classes = [SD15PromptEncodingStep, SD15InpaintLatentsStep, SD15DenoiseStep, SD15DecodeStep]
    block_names = ["prompt_encode", "prepare_latents", "denoise", "decode"]

    @property
    def description(self) -> str:
        return "Sequential SD1.5 inpaint workflow."

    def get_execution_blocks(self, **kwargs):
        return self


class SD15AutoBlocks(AutoPipelineBlocks):
    block_classes = [SD15InpaintBlocks, SD15Img2ImgBlocks, SD15Text2ImgBlocks]
    block_names = ["inpaint", "img2img", "text2img"]
    block_trigger_inputs = ["mask_image", "image", None]

    @property
    def description(self) -> str:
        return (
            "Auto-routing SD1.5 workflow blocks.\n"
            " - Uses `SD15InpaintBlocks` when `mask_image` is provided.\n"
            " - Uses `SD15Img2ImgBlocks` when `image` is provided.\n"
            " - Uses `SD15Text2ImgBlocks` otherwise."
        )
