"""Shared helpers and component specs for SD1.5 modular workflows."""

import inspect
from typing import Any

import PIL.Image
import torch
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.modular_pipelines import AutoPipelineBlocks, ModularPipelineBlocks, PipelineState, SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from diffusers.schedulers import PNDMScheduler
from diffusers.utils.torch_utils import randn_tensor


SD15_BASE_MODEL = "runwayml/stable-diffusion-v1-5"
SD15_LATENT_CHANNELS = 4
SD15_DEFAULT_VAE_SCALE_FACTOR = 8


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
    def tokenizer_spec() -> ComponentSpec:
        return ComponentSpec(
            "tokenizer",
            CLIPTokenizer,
            pretrained_model_name_or_path=SD15_BASE_MODEL,
            subfolder="tokenizer",
        )

    @staticmethod
    def text_encoder_spec() -> ComponentSpec:
        return ComponentSpec(
            "text_encoder",
            CLIPTextModel,
            pretrained_model_name_or_path=SD15_BASE_MODEL,
            subfolder="text_encoder",
        )

    @staticmethod
    def unet_spec() -> ComponentSpec:
        return ComponentSpec(
            "unet",
            UNet2DConditionModel,
            pretrained_model_name_or_path=SD15_BASE_MODEL,
            subfolder="unet",
        )

    @staticmethod
    def vae_spec() -> ComponentSpec:
        return ComponentSpec(
            "vae",
            AutoencoderKL,
            pretrained_model_name_or_path=SD15_BASE_MODEL,
            subfolder="vae",
        )

    @staticmethod
    def scheduler_spec() -> ComponentSpec:
        return ComponentSpec(
            "scheduler",
            PNDMScheduler,
            pretrained_model_name_or_path=SD15_BASE_MODEL,
            subfolder="scheduler",
        )

    @staticmethod
    def image_processor_spec() -> ComponentSpec:
        return ComponentSpec(
            "image_processor",
            VaeImageProcessor,
            config={"vae_scale_factor": SD15_DEFAULT_VAE_SCALE_FACTOR},
            default_creation_method="from_config",
        )

    @staticmethod
    def mask_processor_spec() -> ComponentSpec:
        return ComponentSpec(
            "mask_processor",
            VaeImageProcessor,
            config={
                "vae_scale_factor": SD15_DEFAULT_VAE_SCALE_FACTOR,
                "do_normalize": False,
                "do_binarize": True,
                "do_convert_grayscale": True,
            },
            default_creation_method="from_config",
        )

    @staticmethod
    def prompt_components() -> list[ComponentSpec]:
        return [SD15WorkflowUtils.tokenizer_spec(), SD15WorkflowUtils.text_encoder_spec()]

    @staticmethod
    def latent_components(include_image: bool = False, include_mask: bool = False) -> list[ComponentSpec]:
        components = [SD15WorkflowUtils.unet_spec(), SD15WorkflowUtils.scheduler_spec()]
        if include_image:
            components.extend([SD15WorkflowUtils.vae_spec(), SD15WorkflowUtils.image_processor_spec()])
        if include_mask:
            components.append(SD15WorkflowUtils.mask_processor_spec())
        return components

    @staticmethod
    def denoise_components() -> list[ComponentSpec]:
        return [SD15WorkflowUtils.unet_spec(), SD15WorkflowUtils.scheduler_spec()]

    @staticmethod
    def decode_components() -> list[ComponentSpec]:
        return [SD15WorkflowUtils.vae_spec(), SD15WorkflowUtils.image_processor_spec()]

    @staticmethod
    def expected_components() -> list[ComponentSpec]:
        return [
            SD15WorkflowUtils.tokenizer_spec(),
            SD15WorkflowUtils.text_encoder_spec(),
            SD15WorkflowUtils.unet_spec(),
            SD15WorkflowUtils.vae_spec(),
            SD15WorkflowUtils.scheduler_spec(),
            SD15WorkflowUtils.image_processor_spec(),
            SD15WorkflowUtils.mask_processor_spec(),
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
            return SD15_DEFAULT_VAE_SCALE_FACTOR
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
            inputs.insert(0, InputParam("image", type_hint=PipelineImageInput, required=True))
            inputs.insert(1, InputParam("strength", type_hint=float, default=0.8))
        if include_mask:
            inputs.insert(1, InputParam("mask_image", type_hint=PipelineImageInput, required=True))
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
        if require_mask and mask_image is None:
            raise ValueError("`mask_image` is required for inpaint.")
        if mask_image is not None and image is None:
            raise ValueError("`mask_image` requires `image` for inpaint.")
        if image is None:
            raise ValueError("`image` is required for img2img/inpaint.")
        if strength is None or not 0 < strength <= 1:
            raise ValueError("`strength` must be greater than 0 and less than or equal to 1 for img2img/inpaint.")
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
    def validate_latents_without_components(
        latents: torch.Tensor | None,
        batch_size: int,
        height: int,
        width: int,
    ) -> None:
        if latents is None:
            return
        expected_shape = (
            batch_size,
            SD15_LATENT_CHANNELS,
            height // SD15_DEFAULT_VAE_SCALE_FACTOR,
            width // SD15_DEFAULT_VAE_SCALE_FACTOR,
        )
        if tuple(latents.shape) != expected_shape:
            raise ValueError(f"`latents` must have shape {expected_shape}, got {tuple(latents.shape)}.")

    @staticmethod
    def get_timesteps_img2img(scheduler, num_inference_steps: int, strength: float) -> tuple[torch.Tensor, int]:
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        init_timestep = max(init_timestep, 1)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = scheduler.timesteps[t_start:]
        return timesteps, len(timesteps)

    @staticmethod
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

    @staticmethod
    def expand_batch(tensor: torch.Tensor, effective_batch_size: int, label: str) -> torch.Tensor:
        if tensor.shape[0] == 1 and effective_batch_size > 1:
            return tensor.repeat(effective_batch_size, 1, 1, 1)
        if tensor.shape[0] != effective_batch_size:
            if effective_batch_size % tensor.shape[0] != 0:
                raise ValueError(f"Cannot duplicate {label} batch size {tensor.shape[0]} to {effective_batch_size}.")
            return tensor.repeat(effective_batch_size // tensor.shape[0], 1, 1, 1)
        return tensor

    @staticmethod
    def prepare_img2img_latents(components, image, batch_size, num_images_per_prompt, height, width, dtype, device, generator, timestep):
        image_tensor = SD15WorkflowUtils.preprocess_image(components, image, height, width, device, dtype)
        image_latents = SD15WorkflowUtils.encode_image_latents(components, image_tensor, generator, dtype)
        effective_batch_size = batch_size * num_images_per_prompt
        image_latents = SD15WorkflowUtils.expand_batch(image_latents, effective_batch_size, "image latents")
        noise = randn_tensor(image_latents.shape, generator=generator, device=device, dtype=dtype)
        return components.scheduler.add_noise(image_latents, noise, timestep), image_latents, noise

    @staticmethod
    def prepare_mask_latents(components, mask_image, batch_size, num_images_per_prompt, height, width, dtype, device):
        mask = SD15WorkflowUtils.preprocess_mask(components, mask_image, height, width, device, dtype)
        return SD15WorkflowUtils.expand_batch(mask, batch_size * num_images_per_prompt, "mask")

    @staticmethod
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

    @staticmethod
    def prepare_extra_step_kwargs(components, generator, eta: float) -> dict[str, Any]:
        extra_step_kwargs: dict[str, Any] = {}
        if "eta" in inspect.signature(components.scheduler.step).parameters:
            extra_step_kwargs["eta"] = eta
        if "generator" in inspect.signature(components.scheduler.step).parameters:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @staticmethod
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
        image_processor = getattr(components, "image_processor", None) or VaeImageProcessor(
            vae_scale_factor=SD15WorkflowUtils.vae_scale_factor(components)
        )
        return image_processor.postprocess(image, output_type=output_type)

    @staticmethod
    def preprocess_image(components, image, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        image_processor = getattr(components, "image_processor", None) or VaeImageProcessor(
            vae_scale_factor=SD15WorkflowUtils.vae_scale_factor(components)
        )
        image_tensor = image_processor.preprocess(image, height=height, width=width)
        return image_tensor.to(device=device, dtype=dtype)

    @staticmethod
    def preprocess_mask(components, mask_image, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask_processor = getattr(components, "mask_processor", None) or VaeImageProcessor(
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


