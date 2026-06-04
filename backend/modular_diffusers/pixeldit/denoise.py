"""Latent preparation, denoising, and image postprocessing blocks for PixelDiT."""

from __future__ import annotations

import numpy as np
import PIL.Image
import torch
from diffusers.modular_pipelines import ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec, InputParam, OutputParam

from .pixeldit_transformer import PixelDiTTransformer2DModel
from .sampling import flow_dpm_sample


class PixelDiTPrepareNoiseStep(ModularPipelineBlocks):
    model_name = "pixeldit"

    @property
    def description(self) -> str:
        return "Prepare RGB pixel-space noise latents."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", type_hint=torch.Tensor | None),
            InputParam("height", type_hint=int, required=True),
            InputParam("width", type_hint=int, required=True),
            InputParam("batch_size", type_hint=int, required=True),
            InputParam("generator", type_hint=torch.Generator | None),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("latents", type_hint=torch.Tensor)]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        device = next(components.transformer.parameters()).device
        dtype = components.transformer.dtype
        if block_state.latents is None:
            block_state.latents = torch.randn(
                int(block_state.batch_size),
                int(getattr(components.transformer.config, "in_channels", 3)),
                int(block_state.height),
                int(block_state.width),
                generator=block_state.generator,
                device=device,
                dtype=dtype,
            )
        else:
            block_state.latents = block_state.latents.to(device=device, dtype=dtype)
        self.set_block_state(state, block_state)
        return components, state


class PixelDiTDenoiseStep(ModularPipelineBlocks):
    model_name = "pixeldit"

    @property
    def description(self) -> str:
        return "Run the PixelDiT pixel-space denoising loop."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", type_hint=PixelDiTTransformer2DModel, default_creation_method="from_pretrained")]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", type_hint=torch.Tensor, required=True),
            InputParam("prompt_embeds", type_hint=torch.Tensor, required=True),
            InputParam("negative_prompt_embeds", type_hint=torch.Tensor | None),
            InputParam("attention_mask", type_hint=torch.Tensor | None),
            InputParam("negative_attention_mask", type_hint=torch.Tensor | None),
            InputParam("num_inference_steps", type_hint=int, default=50),
            InputParam("guidance_scale", type_hint=float, default=2.75),
            InputParam("flow_shift", type_hint=float | None),
            InputParam("interval_guidance", type_hint=tuple[float, float], default=(0.0, 1.0)),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("latents", type_hint=torch.Tensor)]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        flow_shift = block_state.flow_shift
        if flow_shift is None:
            flow_shift = float(getattr(components.transformer.config, "flow_shift", 4.0))
        block_state.latents = flow_dpm_sample(
            components.transformer,
            block_state.latents,
            block_state.prompt_embeds,
            block_state.negative_prompt_embeds,
            block_state.attention_mask,
            block_state.negative_attention_mask,
            num_inference_steps=int(block_state.num_inference_steps),
            guidance_scale=float(block_state.guidance_scale),
            flow_shift=float(flow_shift),
            interval_guidance=tuple(block_state.interval_guidance or (0.0, 1.0)),
        )
        self.set_block_state(state, block_state)
        return components, state


class PixelDiTPostprocessStep(ModularPipelineBlocks):
    model_name = "pixeldit"

    @property
    def description(self) -> str:
        return "Convert PixelDiT RGB tensors into PIL/NumPy/tensor outputs."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", type_hint=torch.Tensor, required=True),
            InputParam("output_type", type_hint=str, default="pil"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("images", type_hint=list[PIL.Image.Image] | torch.Tensor | np.ndarray)]

    @property
    def outputs(self) -> list[OutputParam]:
        return [OutputParam("images", type_hint=list[PIL.Image.Image] | torch.Tensor | np.ndarray)]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        images = (block_state.latents.detach().float().clamp(-1, 1) + 1.0) / 2.0
        if block_state.output_type == "pt":
            block_state.images = images
        elif block_state.output_type == "np":
            block_state.images = images.permute(0, 2, 3, 1).cpu().numpy()
        elif block_state.output_type == "pil":
            arrays = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype("uint8")
            block_state.images = [PIL.Image.fromarray(array) for array in arrays]
        else:
            raise ValueError("output_type must be one of 'pil', 'np', or 'pt'.")
        self.set_block_state(state, block_state)
        return components, state
