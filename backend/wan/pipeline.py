"""WAN text-to-video pipeline helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from diffusers.utils import export_to_video

from backend.config import OUTPUT_DIR
from backend.utilities.logging import configure_logging
from backend.utilities.pipeline import (
    build_batch_output_relpath,
    get_batch_output_dir,
    make_batch_id,
)

logger = logging.getLogger(__name__)
configure_logging()

_DEFAULT_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
_SUPPORTED_FRAME_COUNTS = {33, 49, 81}
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
    batch_id = params.get("batch_id")

    _validate_wan_frame_count(num_frames)
    if fps < 1:
        raise ValueError("fps must be >= 1")
    if num_videos != 1:
        raise ValueError("num_videos must be 1 for wan.text2video")
    if width != 832 or height != 480:
        raise ValueError("wan.text2video currently supports only 832x480 output.")

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
        "Generate WAN T2V: model=%s seed=%s steps=%s guidance_scale=%s size=%sx%s "
        "num_frames=%s fps=%s memory_preset=%s",
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

    pipe = load_text2video_pipeline(model, memory_preset=memory_preset)
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
        "batch_id": batch_id,
        "base_seed": base_seed,
        "seed": base_seed,
        "videos": [],
    }

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
