"""Local SDXL text-to-image Modular Diffusers blocks."""

from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.stable_diffusion_xl.before_denoise import StableDiffusionXLInputStep
from diffusers.modular_pipelines.stable_diffusion_xl.decoders import StableDiffusionXLDecodeStep
from diffusers.modular_pipelines.stable_diffusion_xl.denoise import StableDiffusionXLDenoiseStep
from diffusers.modular_pipelines.stable_diffusion_xl.encoders import StableDiffusionXLTextEncoderStep
from diffusers.modular_pipelines.stable_diffusion_xl.modular_blocks_stable_diffusion_xl import (
    StableDiffusionXLBeforeDenoiseStep,
)


class SDXLText2ImageBlocks(SequentialPipelineBlocks):
    """A minimal SDXL text-to-image workflow built from the official modular SDXL steps."""

    block_classes = [
        StableDiffusionXLTextEncoderStep,
        StableDiffusionXLInputStep,
        StableDiffusionXLBeforeDenoiseStep,
        StableDiffusionXLDenoiseStep,
        StableDiffusionXLDecodeStep,
    ]
    block_names = [
        "text_encoder",
        "input",
        "before_denoise",
        "denoise",
        "decode",
    ]
    _workflow_map = {
        "text2image": {"prompt": True},
    }

    @property
    def description(self) -> str:
        return (
            "Local SDXL text-to-image workflow composed from the official Diffusers modular SDXL steps.\n"
            + "Supported inputs match SDXL text generation inputs such as `prompt`, `negative_prompt`,\n"
            + "`num_inference_steps`, `guidance_scale`, `height`, `width`, and `generator`.\n"
            + "This first-pass repo intentionally supports text-to-image only."
        )
