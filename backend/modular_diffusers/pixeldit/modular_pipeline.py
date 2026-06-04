"""PixelDiT ModularPipeline container and schema definitions."""

from __future__ import annotations

import numpy as np
import PIL.Image
import torch
from diffusers.modular_pipelines import ModularPipeline
from diffusers.modular_pipelines.modular_pipeline_utils import InputParam, OutputParam


class PixelDiTModularPipeline(ModularPipeline):
    """Research ModularPipeline container for PixelDiT text-to-image."""

    default_blocks_name = "PixelDiTText2ImgBlocks"


PIXELDIT_INPUTS_SCHEMA = {
    "prompt": InputParam("prompt", type_hint=str | list[str], description="Prompt text"),
    "negative_prompt": InputParam("negative_prompt", type_hint=str | list[str], description="Negative prompt text"),
    "prompt_embeds": InputParam("prompt_embeds", type_hint=torch.Tensor | None, description="Precomputed text embeds"),
    "negative_prompt_embeds": InputParam(
        "negative_prompt_embeds", type_hint=torch.Tensor | None, description="Precomputed negative text embeds"
    ),
    "attention_mask": InputParam("attention_mask", type_hint=torch.Tensor | None, description="Prompt attention mask"),
    "negative_attention_mask": InputParam(
        "negative_attention_mask", type_hint=torch.Tensor | None, description="Negative prompt attention mask"
    ),
    "use_chi_prompt": InputParam(
        "use_chi_prompt", type_hint=bool, default=False, description="Prepend PixelDiT CHI prompt template"
    ),
    "chi_prompt": InputParam(
        "chi_prompt", type_hint=str | list[str] | None, default=None, description="Custom CHI prompt template"
    ),
    "height": InputParam("height", type_hint=int | None, default=None, description="Output height in pixels"),
    "width": InputParam("width", type_hint=int | None, default=None, description="Output width in pixels"),
    "num_inference_steps": InputParam("num_inference_steps", type_hint=int, default=50),
    "guidance_scale": InputParam("guidance_scale", type_hint=float, default=2.75),
    "num_images_per_prompt": InputParam("num_images_per_prompt", type_hint=int, default=1),
    "generator": InputParam("generator", type_hint=torch.Generator | None, description="Optional random generator"),
    "latents": InputParam("latents", type_hint=torch.Tensor | None, description="Optional RGB pixel noise tensor"),
    "sampling_algo": InputParam("sampling_algo", type_hint=str, default="flow_dpm-solver"),
    "flow_shift": InputParam("flow_shift", type_hint=float | None, default=None),
    "interval_guidance": InputParam("interval_guidance", type_hint=tuple[float, float], default=(0.0, 1.0)),
    "output_type": InputParam("output_type", type_hint=str, default="pil"),
}


PIXELDIT_INTERMEDIATE_OUTPUTS_SCHEMA = {
    "prompt_embeds": OutputParam("prompt_embeds", type_hint=torch.Tensor),
    "negative_prompt_embeds": OutputParam("negative_prompt_embeds", type_hint=torch.Tensor | None),
    "attention_mask": OutputParam("attention_mask", type_hint=torch.Tensor | None),
    "negative_attention_mask": OutputParam("negative_attention_mask", type_hint=torch.Tensor | None),
    "batch_size": OutputParam("batch_size", type_hint=int),
    "latents": OutputParam("latents", type_hint=torch.Tensor),
    "images": OutputParam("images", type_hint=list[PIL.Image.Image] | torch.Tensor | np.ndarray),
}


PIXELDIT_OUTPUTS_SCHEMA = {
    "images": OutputParam("images", type_hint=list[PIL.Image.Image] | torch.Tensor | np.ndarray)
}
