"""SD1.5 ModularPipeline container and shared input/output schemas."""

import numpy as np
import PIL.Image
import torch

from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.modular_pipelines import ModularPipeline
from diffusers.modular_pipelines.modular_pipeline_utils import InputParam, OutputParam
from diffusers.pipelines.pipeline_utils import StableDiffusionMixin


class SD15ModularPipeline(
    ModularPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    StableDiffusionLoraLoaderMixin,
):
    """A lightweight ModularPipeline container for Stable Diffusion 1.5."""

    default_blocks_name = "SD15AutoBlocks"

    @property
    def default_height(self) -> int:
        return self.default_sample_size * self.vae_scale_factor

    @property
    def default_width(self) -> int:
        return self.default_sample_size * self.vae_scale_factor

    @property
    def default_sample_size(self) -> int:
        default_sample_size = 64
        if hasattr(self, "unet") and self.unet is not None:
            default_sample_size = self.unet.config.sample_size
        return int(default_sample_size)

    @property
    def vae_scale_factor(self) -> int:
        vae_scale_factor = 8
        if hasattr(self, "vae") and self.vae is not None:
            vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        return int(vae_scale_factor)

    @property
    def num_channels_unet(self) -> int:
        num_channels_unet = 4
        if hasattr(self, "unet") and self.unet is not None:
            num_channels_unet = self.unet.config.in_channels
        return int(num_channels_unet)

    @property
    def num_channels_latents(self) -> int:
        num_channels_latents = 4
        if hasattr(self, "vae") and self.vae is not None:
            num_channels_latents = self.vae.config.latent_channels
        return int(num_channels_latents)


SD15_INPUTS_SCHEMA = {
    "prompt": InputParam("prompt", type_hint=str | list[str], description="The prompt or prompts to guide generation"),
    "negative_prompt": InputParam(
        "negative_prompt", type_hint=str | list[str], description="The prompt or prompts not to guide generation"
    ),
    "prompt_embeds": InputParam(
        "prompt_embeds", type_hint=torch.Tensor | None, description="Precomputed text embeddings"
    ),
    "negative_prompt_embeds": InputParam(
        "negative_prompt_embeds", type_hint=torch.Tensor | None, description="Precomputed negative text embeddings"
    ),
    "image": InputParam(
        "image", type_hint=PipelineImageInput, required=True, description="Image input for img2img or inpainting"
    ),
    "mask_image": InputParam(
        "mask_image", type_hint=PipelineImageInput, required=True, description="Mask image for inpainting"
    ),
    "padding_mask_crop": InputParam(
        "padding_mask_crop",
        type_hint=int | None,
        description="Optional padding around the detected mask crop for inpainting",
    ),
    "height": InputParam("height", type_hint=int | None, description="Output height in pixels"),
    "width": InputParam("width", type_hint=int | None, description="Output width in pixels"),
    "num_images_per_prompt": InputParam(
        "num_images_per_prompt", type_hint=int, default=1, description="Number of images per prompt"
    ),
    "num_inference_steps": InputParam(
        "num_inference_steps", type_hint=int, default=50, description="Number of denoising steps"
    ),
    "guidance_scale": InputParam(
        "guidance_scale", type_hint=float, default=7.5, description="Classifier-free guidance scale"
    ),
    "strength": InputParam(
        "strength", type_hint=float, default=0.8, description="Amount of reference image transformation"
    ),
    "eta": InputParam("eta", type_hint=float, default=0.0, description="DDIM eta parameter when supported"),
    "generator": InputParam(
        "generator",
        type_hint=torch.Generator | list[torch.Generator] | None,
        description="Generator or generators for deterministic sampling",
    ),
    "latents": InputParam(
        "latents", type_hint=torch.Tensor | None, description="Pre-generated noisy latents"
    ),
    "timesteps": InputParam(
        "timesteps", type_hint=list[int] | None, description="Custom denoising timestep schedule"
    ),
    "sigmas": InputParam("sigmas", type_hint=list[float] | None, description="Custom denoising sigma schedule"),
    "output_type": InputParam("output_type", type_hint=str, default="pil", description="Output format"),
}


SD15_INTERMEDIATE_OUTPUTS_SCHEMA = {
    "prompt_embeds": OutputParam("prompt_embeds", type_hint=torch.Tensor, description="Text embeddings"),
    "batch_size": OutputParam("batch_size", type_hint=int, description="Prompt batch size"),
    "latents": OutputParam("latents", type_hint=torch.Tensor, description="Current latent tensor"),
    "timesteps": OutputParam("timesteps", type_hint=torch.Tensor, description="Denoising timesteps"),
    "num_inference_steps": OutputParam(
        "num_inference_steps", type_hint=int, description="Resolved number of denoising steps"
    ),
    "image_latents": OutputParam(
        "image_latents", type_hint=torch.Tensor, description="Encoded reference image latents"
    ),
    "latent_noise": OutputParam("latent_noise", type_hint=torch.Tensor, description="Noise added to image latents"),
    "mask": OutputParam("mask", type_hint=torch.Tensor, description="Prepared inpaint mask latents"),
    "masked_image_latents": OutputParam(
        "masked_image_latents", type_hint=torch.Tensor, description="Encoded masked reference image latents"
    ),
    "crops_coords": OutputParam(
        "crops_coords", type_hint=tuple[int, int, int, int] | None, description="Inpaint crop coordinates"
    ),
    "images": OutputParam(
        "images",
        type_hint=list[PIL.Image.Image] | torch.Tensor | np.ndarray,
        description="Decoded images or tensor/array output",
    ),
}


SD15_OUTPUTS_SCHEMA = {
    "images": OutputParam(
        "images",
        type_hint=list[PIL.Image.Image] | torch.Tensor | np.ndarray,
        description="Final generated images or latent/tensor output",
    )
}
