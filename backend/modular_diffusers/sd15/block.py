"""Compatibility shim for SD1.5 Modular Diffusers blocks.

The implementation is split across SDXL-style modules. This file keeps the
legacy ``block.SD15AutoBlocks`` dynamic-loading path working.
"""

from ._common import SD15_BASE_MODEL, SD15_DEFAULT_VAE_SCALE_FACTOR, SD15_LATENT_CHANNELS, SD15WorkflowUtils, retrieve_timesteps
from .before_denoise import SD15Img2ImgLatentsStep, SD15InpaintLatentsStep, SD15Text2ImgLatentsStep
from .decoders import SD15DecodeStep
from .denoise import SD15DenoiseStep
from .encoders import SD15InputValidationStep, SD15PromptEncodingStep
from .modular_blocks_sd15 import SD15AutoBlocks, SD15Img2ImgBlocks, SD15InpaintBlocks, SD15Text2ImgBlocks

__all__ = [
    "SD15_BASE_MODEL",
    "SD15_DEFAULT_VAE_SCALE_FACTOR",
    "SD15_LATENT_CHANNELS",
    "SD15WorkflowUtils",
    "retrieve_timesteps",
    "SD15InputValidationStep",
    "SD15PromptEncodingStep",
    "SD15Text2ImgLatentsStep",
    "SD15Img2ImgLatentsStep",
    "SD15InpaintLatentsStep",
    "SD15DenoiseStep",
    "SD15DecodeStep",
    "SD15Text2ImgBlocks",
    "SD15Img2ImgBlocks",
    "SD15InpaintBlocks",
    "SD15AutoBlocks",
]
