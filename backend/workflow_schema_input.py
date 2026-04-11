from __future__ import annotations

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field

_DEFAULT_SD15_CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_canny"
_DEFAULT_SDXL_CONTROLNET_MODEL = "diffusers/controlnet-canny-sdxl-1.0"


class ArtifactRef(BaseModel):
    # Schema for artifact references used in workflow inputs/outputs; enforces
    # the API artifact id format (`a|p` prefix + 32 lowercase hex characters).
    artifact_id: str = Field(
        ...,
        description="Artifact id returned by POST /api/artifacts.",
        pattern=r"^[ap][0-9a-f]{32}$",
        examples=["a0123456789abcdef0123456789abcdef"],
    )


ImageRef: TypeAlias = ArtifactRef | str


class Sd15UnifiedLoraContract(BaseModel):
    lora_enabled: bool = True
    lora_adapters: list[dict[str, Any]] = Field(default_factory=list)


class Sd15HiresContract(BaseModel):
    hiresEnabled: bool = False
    hires_scale: float = 1.0


class Sd15EffectiveControlNetItem(BaseModel):
    control_image: ImageRef
    model_id: str | None = None
    conditioning_scale: float | None = Field(default=None, ge=0.0, le=2.0)
    preprocessor_id: str | None = None


class Sd15Text2ImgInputs(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = 20
    cfg: float = 7.5
    width: int = 512
    height: int = 512
    seed: int | None = None
    scheduler: str = "euler"
    model: str | None = None
    num_images: int = 1
    clip_skip: int = 1
    lora: Sd15UnifiedLoraContract | None = None
    hires_enabled: bool = False
    hires_scale: float = 1.0
    weighting_policy: str = "diffusers-like"
    batch_id: str | None = None
    controlNetEnabled: bool = False
    hires: Sd15HiresContract | None = None


class Sd15AnimateDiffText2VideoInputs(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = 25
    cfg: float = 7.5
    width: int = 512
    height: int = 512
    seed: int | None = None
    scheduler: str = "ddim"
    model: str | None = None
    motion_adapter: str = "guoyww/animatediff-motion-adapter-v1-5-2"
    num_frames: int = 16
    fps: int = 8
    num_videos: int = 1
    clip_skip: int = 1
    lora: Sd15UnifiedLoraContract | None = None
    weighting_policy: str = "diffusers-like"
    batch_id: str | None = None


class Sd15Img2ImgInputs(BaseModel):
    initial_image: ImageRef = Field(
        ...,
        description='Image reference: {"artifact_id":"..."} OR "@artifact:..." OR "/outputs/...".',
    )
    prompt: str
    negative_prompt: str = ""
    strength: float = 0.75
    steps: int = 20
    cfg: float = 7.5
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    scheduler: str = "euler"
    model: str | None = None
    num_images: int = 1
    clip_skip: int = 1
    control_image: ImageRef | None = Field(
        default=None,
        description='Optional ControlNet image reference: {"artifact_id":"..."} OR "@artifact:..." OR "/outputs/...".',
    )
    control_images: list[ImageRef] | None = None
    controlnet_model: str = _DEFAULT_SD15_CONTROLNET_MODEL
    controlnet_models: list[str] | None = None
    controlnet_preprocessor_id: str | None = None
    controlnet_preprocessor_ids: list[str] | None = None
    controlnet_compat_mode: Literal["warn", "error", "off"] = "warn"
    controlnet_conditioning_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    controlnet_conditioning_scales: list[float] | None = None
    controlnet_guess_mode: bool = False
    control_guidance_start: float = Field(default=0.0, ge=0.0, le=1.0)
    control_guidance_end: float = Field(default=1.0, ge=0.0, le=1.0)
    lora: Sd15UnifiedLoraContract | None = None
    batch_id: str | None = None


class Sd15InpaintInputs(BaseModel):
    initial_image: ImageRef = Field(
        ...,
        description='Image reference: {"artifact_id":"..."} OR "@artifact:..." OR "/outputs/...".',
    )
    mask_image: ImageRef = Field(
        ...,
        description='Mask reference: {"artifact_id":"..."} OR "@artifact:..." OR "/outputs/...".',
    )
    prompt: str
    negative_prompt: str = ""
    strength: float = 0.5
    steps: int = 20
    cfg: float = 7.5
    seed: int | None = None
    scheduler: str = "euler"
    model: str | None = None
    num_images: int = 1
    padding_mask_crop: int = 32
    clip_skip: int = 1
    control_image: ImageRef | None = Field(
        default=None,
        description='Optional ControlNet image reference: {"artifact_id":"..."} OR "@artifact:..." OR "/outputs/...".',
    )
    control_images: list[ImageRef] | None = None
    controlnet_model: str = _DEFAULT_SD15_CONTROLNET_MODEL
    controlnet_models: list[str] | None = None
    controlnet_preprocessor_id: str | None = None
    controlnet_preprocessor_ids: list[str] | None = None
    controlnet_compat_mode: Literal["warn", "error", "off"] = "warn"
    controlnet_conditioning_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    controlnet_conditioning_scales: list[float] | None = None
    controlnet_guess_mode: bool = False
    control_guidance_start: float = Field(default=0.0, ge=0.0, le=1.0)
    control_guidance_end: float = Field(default=1.0, ge=0.0, le=1.0)
    lora: Sd15UnifiedLoraContract | None = None
    batch_id: str | None = None


class Sd15ControlNetText2ImgInputs(BaseModel):
    control_image: ImageRef | None = Field(
        default=None,
        description='Control image reference: {"artifact_id":"..."} OR "@artifact:..." OR "/outputs/...".',
    )
    prompt: str
    negative_prompt: str = ""
    steps: int = 20
    cfg: float = 7.5
    width: int = 512
    height: int = 512
    seed: int | None = None
    scheduler: str = "euler"
    model: str | None = None
    num_images: int = 1
    clip_skip: int = 1
    controlnet_model: str = _DEFAULT_SD15_CONTROLNET_MODEL
    controlnet_models: list[str] | None = None
    controlnet_preprocessor_id: str | None = None
    controlnet_preprocessor_ids: list[str] | None = None
    controlnet_compat_mode: Literal["warn", "error", "off"] = "warn"
    control_images: list[ImageRef] | None = None
    controlnet_conditioning_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    controlnet_conditioning_scales: list[float] | None = None
    controlnet_guess_mode: bool = False
    control_guidance_start: float = Field(default=0.0, ge=0.0, le=1.0)
    control_guidance_end: float = Field(default=1.0, ge=0.0, le=1.0)
    lora: Sd15UnifiedLoraContract | None = None
    batch_id: str | None = None
    controlNetEnabled: bool = True
    effectiveItems: list[Sd15EffectiveControlNetItem] | None = None
    hires: Sd15HiresContract | None = None


class ControlNetPreprocessInputs(BaseModel):
    image: ImageRef = Field(
        ...,
        description='Source image reference: {"artifact_id":"..."} OR "@artifact:..." OR "/outputs/...".',
    )
    preprocessor_id: str
    params: dict[str, Any] = Field(default_factory=dict)


class Sd15HiresFixInputs(BaseModel):
    images: list[ImageRef] = Field(
        ...,
        description='List of image references (usually from @t1.images).',
    )
    prompt: str
    negative_prompt: str = ""
    steps: int = 20
    cfg: float = 7.5
    seed: int | None = None
    scheduler: str = "euler"
    model: str | None = None
    clip_skip: int = 1
    hires_scale: float = 1.0
    hires_strength: float = 0.35
    lora: Sd15UnifiedLoraContract | None = None
    weighting_policy: str = "diffusers-like"
    batch_id: str | None = None
    hires: Sd15HiresContract | None = None


class SdxlText2ImgInputs(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    steps: int = 20
    guidance_scale: float = 7.5
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    model: str | None = None
    num_images: int = 1
    clip_skip: int = 1
    scheduler: str = "euler"
    lora_adapters: Any | None = None


class SdxlControlNetText2ImgInputs(BaseModel):
    control_image: ImageRef = Field(
        ...,
        description='Control image reference: {"artifact_id":"..."} OR "@artifact:..." OR "/outputs/...".',
    )
    prompt: str = ""
    negative_prompt: str = ""
    steps: int = 20
    guidance_scale: float = 7.5
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    model: str | None = None
    num_images: int = 1
    clip_skip: int = 1
    scheduler: str = "euler"
    controlnet_model: str = _DEFAULT_SDXL_CONTROLNET_MODEL
    controlnet_models: list[str] | None = None
    controlnet_preprocessor_id: str | None = None
    controlnet_preprocessor_ids: list[str] | None = None
    controlnet_compat_mode: Literal["warn", "error", "off"] = "warn"
    control_images: list[ImageRef] | None = None
    controlnet_conditioning_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    controlnet_conditioning_scales: list[float] | None = None
    controlnet_guess_mode: bool = False
    control_guidance_start: float = Field(default=0.0, ge=0.0, le=1.0)
    control_guidance_end: float = Field(default=1.0, ge=0.0, le=1.0)


class SdxlImg2ImgInputs(BaseModel):
    initial_image: ImageRef
    prompt: str
    negative_prompt: str = ""
    strength: float = 0.75
    steps: int = 20
    guidance_scale: float = 7.5
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    scheduler: str = "euler"
    model: str | None = None
    num_images: int = 1
    clip_skip: int = 1
    lora_adapters: Any | None = None
    control_image: ImageRef | None = Field(
        default=None,
        description='Optional ControlNet image reference: {"artifact_id":"..."} OR "@artifact:..." OR "/outputs/...".',
    )
    control_images: list[ImageRef] | None = None
    controlnet_model: str = _DEFAULT_SDXL_CONTROLNET_MODEL
    controlnet_models: list[str] | None = None
    controlnet_preprocessor_id: str | None = None
    controlnet_preprocessor_ids: list[str] | None = None
    controlnet_compat_mode: Literal["warn", "error", "off"] = "warn"
    controlnet_conditioning_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    controlnet_conditioning_scales: list[float] | None = None
    controlnet_guess_mode: bool = False
    control_guidance_start: float = Field(default=0.0, ge=0.0, le=1.0)
    control_guidance_end: float = Field(default=1.0, ge=0.0, le=1.0)


class SdxlInpaintInputs(BaseModel):
    initial_image: ImageRef
    mask_image: ImageRef
    prompt: str
    negative_prompt: str = ""
    strength: float = 0.5
    steps: int = 20
    guidance_scale: float = 7.5
    seed: int | None = None
    scheduler: str = "euler"
    model: str | None = None
    num_images: int = 1
    padding_mask_crop: int = 32
    clip_skip: int = 1
    lora_adapters: Any | None = None
    control_image: ImageRef | None = Field(
        default=None,
        description='Optional ControlNet image reference: {"artifact_id":"..."} OR "@artifact:..." OR "/outputs/...".',
    )
    control_images: list[ImageRef] | None = None
    controlnet_model: str = _DEFAULT_SDXL_CONTROLNET_MODEL
    controlnet_models: list[str] | None = None
    controlnet_preprocessor_id: str | None = None
    controlnet_preprocessor_ids: list[str] | None = None
    controlnet_compat_mode: Literal["warn", "error", "off"] = "warn"
    controlnet_conditioning_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    controlnet_conditioning_scales: list[float] | None = None
    controlnet_guess_mode: bool = False
    control_guidance_start: float = Field(default=0.0, ge=0.0, le=1.0)
    control_guidance_end: float = Field(default=1.0, ge=0.0, le=1.0)


class FluxText2ImgInputs(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    steps: int = 20
    guidance_scale: float = 0.0
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    model: str | None = None
    num_images: int = 1
    scheduler: str = "euler"
    lora_adapters: Any | None = None


class FluxImg2ImgInputs(BaseModel):
    initial_image: ImageRef
    prompt: str
    negative_prompt: str = ""
    strength: float = 0.75
    steps: int = 20
    guidance_scale: float = 0.0
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    scheduler: str = "euler"
    model: str | None = None
    num_images: int = 1
    lora_adapters: Any | None = None


class FluxInpaintInputs(BaseModel):
    initial_image: ImageRef
    mask_image: ImageRef
    prompt: str
    negative_prompt: str = ""
    strength: float = 0.5
    steps: int = 20
    guidance_scale: float = 0.0
    seed: int | None = None
    scheduler: str = "euler"
    model: str | None = None
    num_images: int = 1
    lora_adapters: Any | None = None


class QwenImageText2ImgInputs(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    steps: int = 30
    true_cfg_scale: float = 4.0
    guidance_scale: float = 7.5
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    model: str | None = None
    num_images: int = 1
    scheduler: str = "euler"
    lora_adapters: Any | None = None


class QwenImageImg2ImgInputs(BaseModel):
    initial_image: ImageRef
    prompt: str
    negative_prompt: str = ""
    strength: float = 0.75
    steps: int = 30
    true_cfg_scale: float = 4.0
    guidance_scale: float = 7.5
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    scheduler: str = "euler"
    model: str | None = None
    num_images: int = 1
    lora_adapters: Any | None = None


class QwenImageInpaintInputs(BaseModel):
    initial_image: ImageRef
    mask_image: ImageRef
    prompt: str
    negative_prompt: str = ""
    strength: float = 0.5
    steps: int = 30
    true_cfg_scale: float = 4.0
    guidance_scale: float = 7.5
    seed: int | None = None
    scheduler: str = "euler"
    model: str | None = None
    num_images: int = 1
    lora_adapters: Any | None = None


class ZImageText2ImgInputs(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    steps: int = 8
    guidance_scale: float = 0.0
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    model: str | None = None
    num_images: int = 1
    scheduler: str = "euler"
    lora_adapters: Any | None = None


class ZImageImg2ImgInputs(BaseModel):
    initial_image: ImageRef
    prompt: str
    negative_prompt: str = ""
    strength: float = 0.75
    steps: int = 8
    guidance_scale: float = 0.0
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    scheduler: str = "euler"
    model: str | None = None
    num_images: int = 1
    lora_adapters: Any | None = None


class ZImageInpaintInputs(BaseModel):
    initial_image: ImageRef
    mask_image: ImageRef
    prompt: str
    negative_prompt: str = ""
    strength: float = 0.5
    steps: int = 8
    guidance_scale: float = 0.0
    seed: int | None = None
    scheduler: str = "euler"
    model: str | None = None
    num_images: int = 1
    lora_adapters: Any | None = None
