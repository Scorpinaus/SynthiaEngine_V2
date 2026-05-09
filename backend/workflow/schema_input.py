from __future__ import annotations

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field, field_validator, model_validator

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


class EmbedArtifactRef(BaseModel):
    artifact_id: str = Field(
        ...,
        description="Embed artifact id produced by an IP-Adapter encode task.",
        pattern=r"^e[0-9a-f]{32}$",
        examples=["e0123456789abcdef0123456789abcdef"],
    )


class VideoArtifactRef(BaseModel):
    artifact_id: str = Field(
        ...,
        description="Video artifact id returned by POST /api/artifacts.",
        pattern=r"^v[0-9a-f]{32}$",
        examples=["v0123456789abcdef0123456789abcdef"],
    )


ImageRef: TypeAlias = ArtifactRef | str
EmbedRef: TypeAlias = EmbedArtifactRef | str
VideoRef: TypeAlias = VideoArtifactRef | str


class Sd15UnifiedLoraContract(BaseModel):
    lora_enabled: bool = True
    lora_adapters: list[dict[str, Any]] = Field(default_factory=list)


class Sd15HiresContract(BaseModel):
    hiresEnabled: bool = False
    hires_scale: float = 1.0


class Sd15LcmContract(BaseModel):
    enabled: bool = False


class Sd15IpAdapterContract(BaseModel):
    enabled: bool = False
    image: ImageRef | None = Field(
        default=None,
        description='IP-Adapter reference image: {"artifact_id":"..."} OR "@artifact:..." OR "/outputs/...".',
    )
    mask_image: ImageRef | None = Field(
        default=None,
        description='Optional IP-Adapter influence mask: {"artifact_id":"..."} OR "@artifact:..." OR "/outputs/...". White applies the image prompt, black suppresses it.',
    )
    image_embeds: EmbedRef | None = Field(
        default=None,
        description='Precomputed IP-Adapter embeds from sd15.ip_adapter.encode: {"artifact_id":"..."} OR "@artifact:...".',
    )
    scale: float = Field(default=0.6, ge=0.0, le=1.0)
    model: str = "h94/IP-Adapter"
    subfolder: str = "models"
    weight_name: str = "ip-adapter_sd15.bin"


class SdxlIpAdapterContract(BaseModel):
    enabled: bool = False
    image: ImageRef | None = Field(
        default=None,
        description='IP-Adapter reference image: {"artifact_id":"..."} OR "@artifact:..." OR "/outputs/...".',
    )
    image_embeds: EmbedRef | None = Field(
        default=None,
        description='Precomputed IP-Adapter embeds from sdxl.ip_adapter.encode: {"artifact_id":"..."} OR "@artifact:...".',
    )
    scale: float = Field(default=0.6, ge=0.0, le=1.0)
    model: str = "h94/IP-Adapter"
    subfolder: str = "sdxl_models"
    weight_name: str = "ip-adapter_sdxl.bin"


class Sd15EffectiveControlNetItem(BaseModel):
    control_image: ImageRef
    model_id: str | None = None
    conditioning_scale: float | None = Field(default=None, ge=0.0, le=2.0)
    guidance_start: float | None = Field(default=None, ge=0.0, le=1.0)
    guidance_end: float | None = Field(default=None, ge=0.0, le=1.0)
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
    lcm: Sd15LcmContract | None = None
    ip_adapter: Sd15IpAdapterContract | None = None


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
    free_noise_enabled: bool = False
    free_noise_context_length: int = Field(default=16, ge=1)
    free_noise_context_stride: int = Field(default=4, ge=1)
    free_init_enabled: bool = False
    free_init_num_iters: int = Field(default=3, ge=1)
    free_init_use_fast_sampling: bool = False
    free_init_method: Literal["butterworth", "ideal", "gaussian"] = "butterworth"
    free_init_order: int = Field(default=4, ge=1)
    free_init_spatial_stop_frequency: float = Field(default=0.25, ge=0.0, le=1.0)
    free_init_temporal_stop_frequency: float = Field(default=0.25, ge=0.0, le=1.0)
    clip_skip: int = 1
    lora: Sd15UnifiedLoraContract | None = None
    weighting_policy: str = "diffusers-like"
    batch_id: str | None = None


class WanText2VideoInputs(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = Field(default=30, ge=1, le=200)
    guidance_scale: float = Field(default=6.0, ge=0.0, le=30.0)
    width: int = Field(default=832, ge=64, le=2048)
    height: int = Field(default=480, ge=64, le=2048)
    seed: int | None = None
    model: str = r"D:\diffusion\diffusers\Wan2.1-T2V-1.3B-Diffusers"
    num_frames: int = 49
    fps: int = Field(default=16, ge=1, le=60)
    num_videos: int = Field(default=1, ge=1, le=1)
    memory_preset: Literal["safe"] = "safe"
    quantization: Literal["none", "bnb_8bit"] = "none"
    reference_image: ImageRef | None = Field(
        default=None,
        description='Optional Wan VACE reference image: {"artifact_id":"..."} OR "@artifact:..." OR "/outputs/...".',
    )
    mask_image: ImageRef | None = Field(
        default=None,
        description='Optional Wan VACE mask image. Black conditions/preserves; white generates.',
    )
    conditioning_video: VideoRef | None = Field(
        default=None,
        description='Optional Wan VACE conditioning video: {"artifact_id":"..."} OR "@artifact:..." OR "/outputs/...".',
    )
    conditioning_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    batch_id: str | None = None

    @field_validator("num_frames")
    @classmethod
    def _validate_num_frames(cls, value: int) -> int:
        if value not in {33, 49, 81}:
            raise ValueError("num_frames must be one of 33, 49, 81 for wan.text2video")
        return value

    @model_validator(mode="after")
    def _validate_resolution(self) -> "WanText2VideoInputs":
        if (self.width, self.height) not in {(832, 480), (512, 512)}:
            raise ValueError("wan.text2video supports only 832x480 or 512x512 output.")
        return self


class WanImage2VideoInputs(BaseModel):
    image: ImageRef = Field(
        ...,
        description='Input image for WAN I2V: {"artifact_id":"..."} OR "@artifact:..." OR "/outputs/...".',
    )
    prompt: str
    negative_prompt: str = ""
    steps: int = Field(default=50, ge=1, le=200)
    guidance_scale: float = Field(default=5.0, ge=0.0, le=30.0)
    width: int = Field(default=832, ge=64, le=2048)
    height: int = Field(default=480, ge=64, le=2048)
    seed: int | None = None
    model: str = r"D:\diffusion\diffusers\Wan2.1-I2V-14B-480P-Diffusers"
    num_frames: int = 81
    fps: int = Field(default=16, ge=1, le=60)
    num_videos: int = Field(default=1, ge=1, le=1)
    memory_preset: Literal["offload", "group_offload"] = "offload"
    quantization: Literal["none", "bnb_8bit"] = "none"
    experimental_ack: bool = True
    batch_id: str | None = None

    @field_validator("num_frames")
    @classmethod
    def _validate_num_frames(cls, value: int) -> int:
        if value not in {33, 49, 81}:
            raise ValueError("num_frames must be one of 33, 49, 81 for wan.image2video")
        return value

    @model_validator(mode="after")
    def _validate_resolution(self) -> "WanImage2VideoInputs":
        if (self.width, self.height) != (832, 480):
            raise ValueError("wan.image2video supports only 832x480 output.")
        return self


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
    lcm: Sd15LcmContract | None = None
    ip_adapter: Sd15IpAdapterContract | None = None
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
    lcm: Sd15LcmContract | None = None
    ip_adapter: Sd15IpAdapterContract | None = None
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
    control_guidance_starts: list[float] | None = None
    control_guidance_ends: list[float] | None = None
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
    ip_adapter: SdxlIpAdapterContract | None = None


class Sd15IpAdapterEncodeInputs(BaseModel):
    image: ImageRef = Field(
        ...,
        description='IP-Adapter reference image: {"artifact_id":"..."} OR "@artifact:..." OR "/outputs/...".',
    )
    model: str | None = None
    guidance_scale: float = 7.5
    ip_adapter_model: str = "h94/IP-Adapter"
    ip_adapter_subfolder: str = "models"
    ip_adapter_weight_name: str = "ip-adapter_sd15.bin"
    ip_adapter_scale: float = Field(default=0.6, ge=0.0, le=1.0)


class SdxlIpAdapterEncodeInputs(BaseModel):
    image: ImageRef = Field(
        ...,
        description='IP-Adapter reference image: {"artifact_id":"..."} OR "@artifact:..." OR "/outputs/...".',
    )
    model: str | None = None
    guidance_scale: float = 7.5
    ip_adapter_model: str = "h94/IP-Adapter"
    ip_adapter_subfolder: str = "sdxl_models"
    ip_adapter_weight_name: str = "ip-adapter_sdxl.bin"
    ip_adapter_scale: float = Field(default=0.6, ge=0.0, le=1.0)


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
    ip_adapter: SdxlIpAdapterContract | None = None
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
    ip_adapter: SdxlIpAdapterContract | None = None
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


class ErnieImageText2ImgInputs(BaseModel):
    prompt: str = ""
    steps: int = Field(default=8, ge=1, le=50)
    guidance_scale: float = Field(default=1.0, ge=0.0, le=30.0)
    width: int = Field(default=768, ge=64, le=1536)
    height: int = Field(default=768, ge=64, le=1536)
    seed: int | None = None
    model: str | None = None
    num_images: int = Field(default=1, ge=1, le=1)
    use_pe: bool = False
    load_pe: bool = False
    memory_preset: Literal["model_offload", "sequential_offload"] = "sequential_offload"

    @model_validator(mode="after")
    def _validate_pe_loading(self) -> "ErnieImageText2ImgInputs":
        if self.use_pe and not self.load_pe:
            raise ValueError("use_pe=true requires load_pe=true")
        return self
