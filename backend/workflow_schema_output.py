from __future__ import annotations

from pydantic import BaseModel, Field


class ArtifactInfo(BaseModel):
    artifact_id: str
    url: str
    path: str


class ImagesOutput(BaseModel):
    """Standard output for image-generating tasks."""

    images: list[str] = Field(
        ...,
        description='List of output image URLs ("/outputs/...").',
    )


class ImagesWithBatchOutput(BaseModel):
    """Image-generating output that includes a batch id."""

    batch_id: str = Field(..., description="Batch identifier used to group outputs on disk.")
    images: list[str] = Field(
        ...,
        description='List of output image URLs ("/outputs/...").',
    )


class VideosWithBatchOutput(BaseModel):
    """Video-generating output that includes a batch id."""

    batch_id: str = Field(..., description="Batch identifier used to group outputs on disk.")
    videos: list[str] = Field(
        ...,
        description='List of output video URLs ("/outputs/...").',
    )


class Sd15ControlNetText2ImgOutput(ImagesWithBatchOutput):
    """SD1.5 ControlNet output with optional compatibility warnings."""

    warnings: list[str] | None = Field(
        default=None,
        description="Optional non-fatal compatibility warnings.",
    )


class Sd15Img2ImgOutput(ImagesWithBatchOutput):
    """SD1.5 img2img output with optional compatibility warnings."""

    warnings: list[str] | None = Field(
        default=None,
        description="Optional non-fatal compatibility warnings.",
    )


class Sd15InpaintOutput(ImagesWithBatchOutput):
    """SD1.5 inpaint output with optional compatibility warnings."""

    warnings: list[str] | None = Field(
        default=None,
        description="Optional non-fatal compatibility warnings.",
    )


class ControlNetPreprocessOutput(BaseModel):
    """Output of controlnet.preprocess (produces a new artifact)."""

    artifact: ArtifactInfo


class SdxlControlNetText2ImgOutput(ImagesOutput):
    """SDXL ControlNet output with optional compatibility warnings."""

    warnings: list[str] | None = Field(
        default=None,
        description="Optional non-fatal compatibility warnings.",
    )


class SdxlImg2ImgOutput(ImagesOutput):
    """SDXL img2img output with optional compatibility warnings."""

    warnings: list[str] | None = Field(
        default=None,
        description="Optional non-fatal compatibility warnings.",
    )


class SdxlInpaintOutput(ImagesOutput):
    """SDXL inpaint output with optional compatibility warnings."""

    warnings: list[str] | None = Field(
        default=None,
        description="Optional non-fatal compatibility warnings.",
    )
