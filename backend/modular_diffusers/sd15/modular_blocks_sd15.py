"""SDXL-style block composition for SD1.5 modular workflows."""

from diffusers.modular_pipelines import AutoPipelineBlocks, SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import OutputParam

from .before_denoise import SD15Img2ImgLatentsStep, SD15InpaintLatentsStep, SD15Text2ImgLatentsStep
from .decoders import SD15DecodeStep
from .denoise import SD15DenoiseStep
from .encoders import SD15InputValidationStep, SD15PromptEncodingStep


class SD15Text2ImgBlocks(SequentialPipelineBlocks):
    block_classes = [SD15InputValidationStep, SD15PromptEncodingStep, SD15Text2ImgLatentsStep, SD15DenoiseStep, SD15DecodeStep]
    block_names = ["validate_inputs", "prompt_encode", "prepare_latents", "denoise", "decode"]

    @property
    def description(self) -> str:
        return "Sequential SD1.5 text-to-image workflow."

    def get_execution_blocks(self, **kwargs):
        return self


class SD15Img2ImgBlocks(SequentialPipelineBlocks):
    block_classes = [SD15InputValidationStep, SD15PromptEncodingStep, SD15Img2ImgLatentsStep, SD15DenoiseStep, SD15DecodeStep]
    block_names = ["validate_inputs", "prompt_encode", "prepare_latents", "denoise", "decode"]

    @property
    def description(self) -> str:
        return "Sequential SD1.5 img2img workflow."

    def get_execution_blocks(self, **kwargs):
        return self


class SD15InpaintBlocks(SequentialPipelineBlocks):
    block_classes = [SD15InputValidationStep, SD15PromptEncodingStep, SD15InpaintLatentsStep, SD15DenoiseStep, SD15DecodeStep]
    block_names = ["validate_inputs", "prompt_encode", "prepare_latents", "denoise", "decode"]

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
