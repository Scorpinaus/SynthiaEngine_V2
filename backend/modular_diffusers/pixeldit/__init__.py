"""PixelDiT Modular Diffusers research pipeline."""

from .modular_blocks_pixeldit import PixelDiTText2ImgBlocks
from .modular_pipeline import PixelDiTModularPipeline
from .pixeldit_transformer import PixelDiTTransformer2DModel

try:
    import diffusers as _diffusers

    if not hasattr(_diffusers, "PixelDiTText2ImgBlocks"):
        setattr(_diffusers, "PixelDiTText2ImgBlocks", PixelDiTText2ImgBlocks)
except Exception:
    pass

__all__ = [
    "PixelDiTModularPipeline",
    "PixelDiTText2ImgBlocks",
    "PixelDiTTransformer2DModel",
]
