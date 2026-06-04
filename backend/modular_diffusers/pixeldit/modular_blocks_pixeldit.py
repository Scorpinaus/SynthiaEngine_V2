"""PixelDiT modular block composition."""

from diffusers.modular_pipelines import SequentialPipelineBlocks

from .denoise import PixelDiTDenoiseStep, PixelDiTPostprocessStep, PixelDiTPrepareNoiseStep
from .encoders import PixelDiTInputValidationStep, PixelDiTPromptEncodingStep


class PixelDiTText2ImgBlocks(SequentialPipelineBlocks):
    block_classes = [
        PixelDiTInputValidationStep,
        PixelDiTPromptEncodingStep,
        PixelDiTPrepareNoiseStep,
        PixelDiTDenoiseStep,
        PixelDiTPostprocessStep,
    ]
    block_names = ["validate_inputs", "prompt_encode", "prepare_noise", "denoise", "postprocess"]

    @property
    def description(self) -> str:
        return (
            "Sequential PixelDiT text-to-image workflow: validate inputs, encode Gemma prompt embeddings, "
            "prepare RGB pixel noise, denoise with PixelDiT, then postprocess image tensors."
        )

    def get_execution_blocks(self, **kwargs):
        return self
