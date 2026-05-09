"""WAN text-to-video pipeline helpers."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from diffusers.utils import export_to_video, load_video
from diffusers import AutoencoderKLWan, WanPipeline, WanImageToVideoPipeline, WanVACEPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.hooks.group_offloading import apply_group_offloading

from transformers import CLIPVisionModel

from backend.config import OUTPUT_DIR
from backend.quantization import build_diffusers_pipeline_quantization_config
from backend.utilities.logging import configure_logging
from backend.utilities.pipeline import (
    build_batch_output_relpath,
    get_batch_output_dir,
    make_batch_id,
    release_pipeline,
)
from backend.wan.subprocess_io import serialize_params_for_subprocess

logger = logging.getLogger(__name__)
configure_logging()

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WAN_SUBPROCESS_SEMAPHORE = threading.Semaphore(1)
_DEFAULT_MODEL_ID = r"D:\diffusion\diffusers\Wan2.1-T2V-1.3B-Diffusers"
_DEFAULT_VACE_MODEL_ID = r"D:\diffusion\diffusers\Wan2.1-VACE-1.3B-diffusers"
_DEFAULT_I2V_MODEL_ID = r"D:\diffusion\diffusers\Wan2.1-I2V-14B-480P-Diffusers"
_SUPPORTED_FRAME_COUNTS = {33, 49, 81}
_SUPPORTED_RESOLUTIONS = {(832, 480), (512, 512)}
_SUPPORTED_I2V_RESOLUTION = (832, 480)
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


def _validate_wan_i2v_frame_count(num_frames: int) -> None:
    if num_frames not in _SUPPORTED_FRAME_COUNTS:
        raise ValueError("num_frames must be one of 33, 49, 81 for wan.image2video")


def _validate_wan_resolution(width: int, height: int) -> None:
    if (width, height) not in _SUPPORTED_RESOLUTIONS:
        raise ValueError("wan.text2video supports only 832x480 or 512x512 output.")


def _validate_wan_i2v_resolution(width: int, height: int) -> None:
    if (width, height) != _SUPPORTED_I2V_RESOLUTION:
        raise ValueError("wan.image2video supports only 832x480 output.")


def _make_wan_generator(seed: int) -> torch.Generator:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.Generator(device=device).manual_seed(seed)


def load_text2video_pipeline(
    model_id: str,
    *,
    memory_preset: str = "safe",
    quantization: str = "none",
):
    """Load a WAN text-to-video pipeline using the safe memory preset."""
    if memory_preset != "safe":
        raise ValueError("memory_preset must be 'safe' for wan.text2video")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for wan.text2video generation.")

    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    quantization_config = build_diffusers_pipeline_quantization_config(
        quantization,
        components_to_quantize=["transformer", "text_encoder"],
        task_type="wan.text2video",
    )
    kwargs: dict[str, Any] = {}
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    pipe = WanPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch.bfloat16,
        **kwargs,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=3.0,
    )
    pipe.enable_model_cpu_offload()
    return pipe


def _apply_wan_i2v_memory_preset(pipe: object, *, memory_preset: str) -> None:
    if memory_preset == "offload":
        pipe.enable_model_cpu_offload()
        return

    if memory_preset != "group_offload":
        raise ValueError("memory_preset must be 'offload' or 'group_offload' for wan.image2video")

    onload_device = torch.device("cuda")
    offload_device = torch.device("cpu")
    text_encoder = getattr(pipe, "text_encoder", None)
    transformer = getattr(pipe, "transformer", None)
    if text_encoder is not None:
        apply_group_offloading(
            text_encoder,
            onload_device=onload_device,
            offload_device=offload_device,
            offload_type="block_level",
            num_blocks_per_group=4,
        )
    if transformer is not None and hasattr(transformer, "enable_group_offload"):
        transformer.enable_group_offload(
            onload_device=onload_device,
            offload_device=offload_device,
            offload_type="leaf_level",
            use_stream=True,
        )
    pipe.to("cuda")


def load_image2video_pipeline(
    model_id: str,
    *,
    memory_preset: str = "offload",
    quantization: str = "none",
):
    """Load a WAN 14B image-to-video pipeline with explicit experimental memory controls."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for wan.image2video generation.")

    logger.warning(
        "WAN I2V 14B 480P is slow and experimental. Expect heavy CPU offload, high RAM use, and long runtimes."
    )
    quantization_config = build_diffusers_pipeline_quantization_config(
        quantization,
        components_to_quantize=["transformer", "text_encoder"],
        task_type="wan.image2video",
    )
    image_encoder = CLIPVisionModel.from_pretrained(
        model_id,
        subfolder="image_encoder",
        torch_dtype=torch.float32,
    )
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    kwargs: dict[str, Any] = {}
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id,
        vae=vae,
        image_encoder=image_encoder,
        torch_dtype=torch.bfloat16,
        **kwargs,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=3.0,
    )
    _apply_wan_i2v_memory_preset(pipe, memory_preset=memory_preset)
    return pipe


def load_vace_pipeline(
    model_id: str,
    *,
    memory_preset: str = "safe",
    quantization: str = "none",
):
    """Load a WAN VACE pipeline using the safe memory preset."""
    if memory_preset != "safe":
        raise ValueError("memory_preset must be 'safe' for wan.text2video")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for wan.text2video generation.")

    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    quantization_config = build_diffusers_pipeline_quantization_config(
        quantization,
        components_to_quantize=["transformer", "text_encoder"],
        task_type="wan.text2video",
    )
    kwargs: dict[str, Any] = {}
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    pipe = WanVACEPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch.bfloat16,
        **kwargs,
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


def _run_wan_subprocess(operation: str, params: dict[str, object]) -> list[str]:

    with tempfile.TemporaryDirectory(prefix="wan_") as tmpdir:
        tmp_path = Path(tmpdir)
        input_path = tmp_path / "input.json"
        output_path = tmp_path / "output.json"
        payload = {
            "operation": operation,
            "params": serialize_params_for_subprocess(params, tmp_path),
        }
        input_path.write_text(
            json.dumps(payload, separators=(",", ": ")),
            encoding="utf-8",
        )

        cmd = [
            sys.executable,
            "-m",
            "backend.wan.subprocess_runner",
            str(input_path),
            str(output_path),
        ]
        with _WAN_SUBPROCESS_SEMAPHORE:
            completed = subprocess.run(cmd, cwd=str(_REPO_ROOT))

        if not output_path.exists():
            raise RuntimeError("WAN subprocess failed: No subprocess result was written.")

        result_payload = json.loads(output_path.read_text(encoding="utf-8"))
        if completed.returncode != 0 or not result_payload.get("ok"):
            detail = result_payload.get("error") or "Unknown subprocess failure."
            error_type = result_payload.get("error_type")
            if error_type:
                detail = f"{error_type}: {detail}"
            raise RuntimeError(f"WAN subprocess failed: {detail}")

        result = result_payload.get("result")
        if not isinstance(result, list):
            raise RuntimeError("WAN subprocess returned an invalid result.")
        return [str(path) for path in result]


def generate_text2video(params: dict[str, object]) -> list[str]:
    return _run_wan_subprocess("text2video", params)


def generate_image2video(params: dict[str, object]) -> list[str]:
    return _run_wan_subprocess("image2video", params)


@torch.inference_mode()
def generate_text2video_in_process(params: dict[str, object]) -> list[str]:
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
    quantization = str(params.get("quantization") or "none")
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
        "num_frames=%s fps=%s memory_preset=%s quantization=%s",
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
        quantization,
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
        "quantization": quantization,
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

    pipe = None
    try:
        if is_vace:
            pipe = load_vace_pipeline(
                model,
                memory_preset=memory_preset,
                quantization=quantization,
            )
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
            pipe = load_text2video_pipeline(
                model,
                memory_preset=memory_preset,
                quantization=quantization,
            )
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
    finally:
        release_pipeline(pipe, logger=logger)
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


@torch.inference_mode()
def generate_image2video_in_process(params: dict[str, object]) -> list[str]:
    """Generate WAN image-to-video MP4 files and return relative output paths."""
    image = params["image"]
    prompt = str(params["prompt"])
    negative_prompt = str(params.get("negative_prompt") or _DEFAULT_NEGATIVE_PROMPT)
    steps = int(params.get("steps") or 50)
    guidance_scale = float(params.get("guidance_scale") or 5.0)
    width = int(params.get("width") or 832)
    height = int(params.get("height") or 480)
    seed = params.get("seed")
    model = str(params.get("model") or _DEFAULT_I2V_MODEL_ID)
    num_frames = int(params.get("num_frames") or 81)
    fps = int(params.get("fps") or 16)
    num_videos = int(params.get("num_videos") or 1)
    memory_preset = str(params.get("memory_preset") or "offload")
    quantization = str(params.get("quantization") or "none")
    experimental_ack = bool(params.get("experimental_ack", False))
    batch_id = params.get("batch_id")

    if not experimental_ack:
        raise ValueError("experimental_ack must be true for wan.image2video")
    if not isinstance(image, Image.Image):
        raise ValueError("image must resolve to an image for wan.image2video")
    _validate_wan_i2v_frame_count(num_frames)
    _validate_wan_i2v_resolution(width, height)
    if fps < 1:
        raise ValueError("fps must be >= 1")
    if num_videos != 1:
        raise ValueError("num_videos must be 1 for wan.image2video")
    if memory_preset not in {"offload", "group_offload"}:
        raise ValueError("memory_preset must be 'offload' or 'group_offload' for wan.image2video")
    if quantization not in {"none", "bnb_8bit"}:
        raise ValueError("quantization must be 'none' or 'bnb_8bit' for wan.image2video")

    logger.warning(
        "Generating WAN I2V 14B 480P: slow / experimental path enabled; memory_preset=%s quantization=%s",
        memory_preset,
        quantization,
    )
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)

    if batch_id is None:
        batch_id = make_batch_id()
    batch_id = str(batch_id)
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    output_name = f"{batch_id}_{base_seed}.mp4"
    relative_path = build_batch_output_relpath(batch_id, output_name)
    metadata_path = _wan_video_metadata_path(batch_output_dir, batch_id)
    metadata: dict[str, Any] = {
        "mode": "wan.image2video",
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
        "quantization": quantization,
        "experimental_ack": experimental_ack,
        "batch_id": batch_id,
        "base_seed": base_seed,
        "seed": base_seed,
        "videos": [],
    }

    pipe = None
    try:
        pipe = load_image2video_pipeline(
            model,
            memory_preset=memory_preset,
            quantization=quantization,
        )
        source_image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        result = pipe(
            image=source_image,
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
    finally:
        release_pipeline(pipe, logger=logger)

    metadata["videos"].append(
        {
            "filename": output_name,
            "path": relative_path,
            "seed": base_seed,
            "index": 0,
        }
    )
    _write_wan_video_metadata(metadata_path, metadata)
    logger.info("WAN I2V video saved to %s", output_name)
    return [relative_path]
