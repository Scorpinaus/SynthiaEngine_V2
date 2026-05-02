"""WAN text-to-video pipeline helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from diffusers.utils import export_to_video, load_video

from backend.config import OUTPUT_DIR
from backend.utilities.logging import configure_logging
from backend.utilities.pipeline import (
    build_batch_output_relpath,
    get_batch_output_dir,
    make_batch_id,
)

logger = logging.getLogger(__name__)
configure_logging()

_DEFAULT_MODEL_ID = r"D:\diffusion\diffusers\Wan2.1-T2V-1.3B-Diffusers"
_DEFAULT_VACE_MODEL_ID = r"D:\diffusion\diffusers\Wan2.1-VACE-1.3B-diffusers"
_SUPPORTED_FRAME_COUNTS = {33, 49, 81}
_SUPPORTED_RESOLUTIONS = {(832, 480), (512, 512)}
_DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


def _wan_video_metadata_path(batch_output_dir: Path, batch_id: str) -> Path:
    return batch_output_dir / f"video_{batch_id}.mp4.json"


def _json_safe_metadata(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_metadata(item) for item in value]
    return str(value)


def _write_wan_video_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe_metadata(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _validate_wan_frame_count(num_frames: int) -> None:
    if num_frames not in _SUPPORTED_FRAME_COUNTS:
        raise ValueError("num_frames must be one of 33, 49, 81 for wan.text2video")


def _validate_wan_resolution(width: int, height: int) -> None:
    if (width, height) not in _SUPPORTED_RESOLUTIONS:
        raise ValueError("wan.text2video supports only 832x480 or 512x512 output.")


def _make_wan_generator(seed: int) -> torch.Generator:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.Generator(device=device).manual_seed(seed)


def load_text2video_pipeline(model_id: str, *, memory_preset: str = "safe"):
    """Load a WAN text-to-video pipeline using the safe memory preset."""
    if memory_preset != "safe":
        raise ValueError("memory_preset must be 'safe' for wan.text2video")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for wan.text2video generation.")

    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    pipe = WanPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=3.0,
    )
    pipe.enable_model_cpu_offload()
    return pipe


def load_vace_pipeline(model_id: str, *, memory_preset: str = "safe"):
    """Load a WAN VACE pipeline using the safe memory preset."""
    if memory_preset != "safe":
        raise ValueError("memory_preset must be 'safe' for wan.text2video")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for wan.text2video generation.")

    from diffusers import AutoencoderKLWan, WanVACEPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    pipe = WanVACEPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=3.0,
    )
    pipe.enable_model_cpu_offload()
    return pipe


def _fit_frame_list(frames: list[Image.Image], *, width: int, height: int, num_frames: int) -> list[Image.Image]:
    if not frames:
        raise ValueError("conditioning_video must contain at least one frame.")
    fitted = [frame.convert("RGB").resize((width, height), Image.Resampling.LANCZOS) for frame in frames]
    if len(fitted) >= num_frames:
        return fitted[:num_frames]
    return fitted + [fitted[-1].copy() for _ in range(num_frames - len(fitted))]


def _prepare_vace_conditions(
    *,
    conditioning_video: Path,
    mask_image: Image.Image,
    reference_image: Image.Image | None,
    width: int,
    height: int,
    num_frames: int,
) -> tuple[list[Image.Image], list[Image.Image], list[Image.Image] | None]:
    video_frames = _fit_frame_list(
        load_video(str(conditioning_video)),
        width=width,
        height=height,
        num_frames=num_frames,
    )
    mask = mask_image.convert("L").resize((width, height), Image.Resampling.NEAREST)
    masks = [mask.copy() for _ in range(num_frames)]
    reference_images = [reference_image.convert("RGB")] if reference_image is not None else None
    return video_frames, masks, reference_images


@torch.inference_mode()
def generate_text2video(params: dict[str, object]) -> list[str]:
    """Generate WAN text-to-video MP4 files and return relative output paths."""
    prompt = str(params["prompt"])
    negative_prompt = str(params.get("negative_prompt") or _DEFAULT_NEGATIVE_PROMPT)
    steps = int(params.get("steps") or 30)
    guidance_scale = float(params.get("guidance_scale") or 6.0)
    width = int(params.get("width") or 832)
    height = int(params.get("height") or 480)
    seed = params.get("seed")
    model = str(params.get("model") or _DEFAULT_MODEL_ID)
    num_frames = int(params.get("num_frames") or 49)
    fps = int(params.get("fps") or 16)
    num_videos = int(params.get("num_videos") or 1)
    memory_preset = str(params.get("memory_preset") or "safe")
    reference_image = params.get("reference_image")
    mask_image = params.get("mask_image")
    conditioning_video = params.get("conditioning_video")
    conditioning_scale = float(params.get("conditioning_scale") or 1.0)
    batch_id = params.get("batch_id")
    is_vace = (
        "vace" in model.lower()
        or reference_image is not None
        or mask_image is not None
        or conditioning_video is not None
    )

    _validate_wan_frame_count(num_frames)
    if fps < 1:
        raise ValueError("fps must be >= 1")
    if num_videos != 1:
        raise ValueError("num_videos must be 1 for wan.text2video")
    _validate_wan_resolution(width, height)
    if conditioning_scale < 0 or conditioning_scale > 2:
        raise ValueError("conditioning_scale must be within [0, 2] for wan.text2video")
    if is_vace:
        model = model or _DEFAULT_VACE_MODEL_ID
        if "vace" not in model.lower():
            model = _DEFAULT_VACE_MODEL_ID
        if conditioning_video is None:
            raise ValueError("conditioning_video is required for Wan VACE generation.")
        if mask_image is None:
            raise ValueError("mask_image is required when conditioning_video is provided.")
        if reference_image is None:
            raise ValueError("reference_image is required for Wan VACE generation.")
        if not isinstance(conditioning_video, Path):
            raise ValueError("conditioning_video must resolve to a local video path.")
        if not isinstance(mask_image, Image.Image):
            raise ValueError("mask_image must resolve to an image.")
        if not isinstance(reference_image, Image.Image):
            raise ValueError("reference_image must resolve to an image.")

    logger.info("WAN seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)

    if batch_id is None:
        batch_id = make_batch_id()
    batch_id = str(batch_id)
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    logger.info(
        "Generate WAN %s: model=%s seed=%s steps=%s guidance_scale=%s size=%sx%s "
        "num_frames=%s fps=%s memory_preset=%s",
        "VACE" if is_vace else "T2V",
        model,
        base_seed,
        steps,
        guidance_scale,
        width,
        height,
        num_frames,
        fps,
        memory_preset,
    )

    output_name = f"{batch_id}_{base_seed}.mp4"
    relative_path = build_batch_output_relpath(batch_id, output_name)
    metadata_path = _wan_video_metadata_path(batch_output_dir, batch_id)
    metadata: dict[str, Any] = {
        "mode": "wan.text2video",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "width": width,
        "height": height,
        "model": model,
        "num_frames": num_frames,
        "fps": fps,
        "num_videos": num_videos,
        "memory_preset": memory_preset,
        "vace": {
            "enabled": is_vace,
            "has_reference_image": reference_image is not None,
            "has_mask_image": mask_image is not None,
            "has_conditioning_video": conditioning_video is not None,
            "conditioning_scale": conditioning_scale,
        },
        "batch_id": batch_id,
        "base_seed": base_seed,
        "seed": base_seed,
        "videos": [],
    }

    if is_vace:
        pipe = load_vace_pipeline(model, memory_preset=memory_preset)
        video, mask, reference_images = _prepare_vace_conditions(
            conditioning_video=conditioning_video,
            mask_image=mask_image,
            reference_image=reference_image,
            width=width,
            height=height,
            num_frames=num_frames,
        )
        result = pipe(
            video=video,
            mask=mask,
            reference_images=reference_images,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            conditioning_scale=conditioning_scale,
            generator=_make_wan_generator(base_seed),
        )
    else:
        pipe = load_text2video_pipeline(model, memory_preset=memory_preset)
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=_make_wan_generator(base_seed),
        )
    export_to_video(result.frames[0], batch_output_dir / output_name, fps=fps)
    metadata["videos"].append(
        {
            "filename": output_name,
            "path": relative_path,
            "seed": base_seed,
            "index": 0,
        }
    )
    _write_wan_video_metadata(metadata_path, metadata)
    logger.info("WAN video saved to %s", output_name)
    return [relative_path]
