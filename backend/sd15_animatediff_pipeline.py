"""
SD1.5 AnimateDiff text-to-video pipeline helpers.

This module mirrors the workflow-facing conventions of ``backend.sd15_pipeline``
while keeping AnimateDiff loading/generation isolated from the existing image
pipelines.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from diffusers import AnimateDiffPipeline
from diffusers.models import MotionAdapter
from diffusers.schedulers import DDIMScheduler
from diffusers.utils import export_to_video

from backend.config import OUTPUT_DIR
from backend.logging_utils import configure_logging
from backend.lora_utils import apply_lora_adapters_with_validation, write_lora_coverage_report
from backend.model_registry import get_model_entry
from backend.pipeline_utils import build_batch_output_relpath, get_batch_output_dir, make_batch_id, resolve_model_source
from backend.prompt_utils import build_prompt_embeddings
from backend.schedulers import create_scheduler

logger = logging.getLogger(__name__)
configure_logging()

_DEFAULT_MOTION_ADAPTER = "guoyww/animatediff-motion-adapter-v1-5-2"
_FREE_INIT_METHODS = {"butterworth", "ideal", "gaussian"}


def _animatediff_video_metadata_path(batch_output_dir: Path, batch_id: str) -> Path:
    return batch_output_dir / f"video_{batch_id}.mp4.json"


def _json_safe_metadata(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_metadata(item) for item in value]
    return str(value)


def _write_animatediff_video_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe_metadata(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _int_param(params: dict[str, object], key: str, default: int) -> int:
    value = params.get(key, default)
    if value is None:
        return default
    return int(value)


def _float_param(params: dict[str, object], key: str, default: float) -> float:
    value = params.get(key, default)
    if value is None:
        return default
    return float(value)


def _motion_adapter_max_seq_length(adapter: MotionAdapter) -> int | None:
    config = getattr(adapter, "config", None)
    raw_value = None
    if isinstance(config, dict):
        raw_value = config.get("motion_max_seq_length")
    elif config is not None:
        raw_value = getattr(config, "motion_max_seq_length", None)
    try:
        max_seq_length = int(raw_value)
    except (TypeError, ValueError):
        return None
    return max_seq_length if max_seq_length > 0 else None


def _validate_animatediff_frame_settings(
    *,
    num_frames: int,
    free_noise_enabled: bool,
    free_noise_context_length: int,
    free_noise_context_stride: int,
    motion_max_seq_length: int | None,
) -> None:
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    if free_noise_context_length < 1:
        raise ValueError("free_noise_context_length must be >= 1")
    if free_noise_context_stride < 1:
        raise ValueError("free_noise_context_stride must be >= 1")
    if free_noise_context_stride > free_noise_context_length:
        raise ValueError("free_noise_context_stride must be <= free_noise_context_length")
    if motion_max_seq_length is None:
        return
    if num_frames > motion_max_seq_length and not free_noise_enabled:
        raise ValueError(
            f"num_frames={num_frames} exceeds motion adapter temporal limit "
            f"{motion_max_seq_length}. Enable FreeNoise or use num_frames <= "
            f"{motion_max_seq_length}."
        )
    if free_noise_enabled and min(num_frames, free_noise_context_length) > motion_max_seq_length:
        raise ValueError(
            f"free_noise_context_length={free_noise_context_length} exceeds motion adapter "
            f"temporal limit {motion_max_seq_length}. Use a context length <= "
            f"{motion_max_seq_length}."
        )


def _enable_free_noise(
    pipe: AnimateDiffPipeline,
    *,
    context_length: int,
    context_stride: int,
) -> None:
    if not hasattr(pipe, "enable_free_noise"):
        raise RuntimeError("This diffusers version does not support AnimateDiff FreeNoise.")
    pipe.enable_free_noise(context_length=context_length, context_stride=context_stride)

    if hasattr(pipe, "enable_free_noise_split_inference"):
        try:
            pipe.enable_free_noise_split_inference(temporal_split_size=context_length)
        except TypeError:
            pipe.enable_free_noise_split_inference()

    unet = getattr(pipe, "unet", None)
    if unet is not None and hasattr(unet, "enable_forward_chunking"):
        unet.enable_forward_chunking(min(16, context_length))


def _validate_free_init_settings(
    *,
    num_iters: int,
    method: str,
    order: int,
    spatial_stop_frequency: float,
    temporal_stop_frequency: float,
) -> None:
    if num_iters < 1:
        raise ValueError("free_init_num_iters must be >= 1")
    if method not in _FREE_INIT_METHODS:
        raise ValueError(
            "free_init_method must be one of butterworth, ideal, gaussian"
        )
    if order < 1:
        raise ValueError("free_init_order must be >= 1")
    if spatial_stop_frequency < 0.0 or spatial_stop_frequency > 1.0:
        raise ValueError("free_init_spatial_stop_frequency must be within [0, 1]")
    if temporal_stop_frequency < 0.0 or temporal_stop_frequency > 1.0:
        raise ValueError("free_init_temporal_stop_frequency must be within [0, 1]")


def _enable_free_init(
    pipe: AnimateDiffPipeline,
    *,
    num_iters: int,
    use_fast_sampling: bool,
    method: str,
    order: int,
    spatial_stop_frequency: float,
    temporal_stop_frequency: float,
) -> None:
    if not hasattr(pipe, "enable_free_init"):
        raise RuntimeError("This diffusers version does not support AnimateDiff FreeInit.")
    pipe.enable_free_init(
        num_iters=num_iters,
        use_fast_sampling=use_fast_sampling,
        method=method,
        order=order,
        spatial_stop_frequency=spatial_stop_frequency,
        temporal_stop_frequency=temporal_stop_frequency,
    )


def _disable_free_init(pipe: AnimateDiffPipeline) -> None:
    if not hasattr(pipe, "disable_free_init"):
        return
    try:
        pipe.disable_free_init()
    except Exception:
        logger.exception("Failed to disable AnimateDiff FreeInit cleanly.")


def _prepare_animatediff_prompt_inputs(
    pipe: AnimateDiffPipeline,
    *,
    prompt: str,
    negative_prompt: str,
    clip_skip: int,
    weighting_policy: str,
    free_noise_enabled: bool,
) -> tuple[str | None, str | None, torch.Tensor | None, torch.Tensor | None]:
    if free_noise_enabled:
        return prompt, negative_prompt, None, None

    prompt_embeds, negative_prompt_embeds, use_prompt_embeds = build_prompt_embeddings(
        pipe,
        prompt,
        negative_prompt,
        clip_skip=clip_skip,
        weighting_policy=weighting_policy,
    )
    return (
        None if use_prompt_embeds else prompt,
        None if use_prompt_embeds else negative_prompt,
        prompt_embeds if use_prompt_embeds else None,
        negative_prompt_embeds if use_prompt_embeds else None,
    )


def _make_animatediff_generator(
    *,
    seed: int,
    free_noise_enabled: bool,
) -> torch.Generator:
    device = "cpu" if free_noise_enabled else "cuda"
    return torch.Generator(device=device).manual_seed(seed)


def _load_motion_adapter(motion_adapter: str | None) -> MotionAdapter:
    adapter_source = str(motion_adapter or _DEFAULT_MOTION_ADAPTER).strip()
    if not adapter_source:
        adapter_source = _DEFAULT_MOTION_ADAPTER

    local_path = Path(adapter_source).expanduser()
    if local_path.is_file():
        return MotionAdapter.from_single_file(
            str(local_path),
            torch_dtype=torch.float16,
        )
    return MotionAdapter.from_pretrained(adapter_source, torch_dtype=torch.float16)


def _cleanup_lora_adapters(pipe, adapter_names: list[str]) -> None:
    if not adapter_names:
        return
    if hasattr(pipe, "unload_lora_weights"):
        try:
            pipe.unload_lora_weights()
        except Exception:
            logger.exception("Failed to unload AnimateDiff LoRA weights cleanly.")
    for component_name in ("unet", "text_encoder"):
        component = getattr(pipe, component_name, None)
        if component is None or not hasattr(component, "delete_adapters"):
            continue
        try:
            component.delete_adapters(adapter_names)
        except Exception:
            logger.debug(
                "Skipping AnimateDiff adapter cleanup for %s; delete_adapters failed.",
                component_name,
                exc_info=True,
            )


def _apply_animatediff_scheduler(pipe: AnimateDiffPipeline, scheduler_name: str) -> None:
    normalized = str(scheduler_name or "ddim").lower()
    if normalized == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )
        return
    pipe.scheduler = create_scheduler(normalized, pipe)


def load_text2video_pipeline(
    model_name: str | None,
    motion_adapter: str | None,
) -> AnimateDiffPipeline:
    """Load an AnimateDiff SD1.5 text-to-video pipeline on CUDA fp16."""
    entry = get_model_entry(model_name)
    source = resolve_model_source(entry)
    adapter = _load_motion_adapter(motion_adapter)
    motion_max_seq_length = _motion_adapter_max_seq_length(adapter)

    logger.info("AnimateDiff base model: %s", source)
    logger.info("AnimateDiff motion adapter: %s", motion_adapter or _DEFAULT_MOTION_ADAPTER)
    if entry.model_type == "diffusers":
        pipe = AnimateDiffPipeline.from_pretrained(
            source,
            motion_adapter=adapter,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        pipe = AnimateDiffPipeline.from_single_file(
            source,
            motion_adapter=adapter,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    pipe.enable_vae_slicing()
    setattr(pipe, "_syntha_motion_max_seq_length", motion_max_seq_length)
    pipe.to("cuda")
    return pipe


@torch.inference_mode()
def generate_videos_text2video(params: dict[str, object]) -> list[str]:
    """Generate SD1.5 AnimateDiff videos, write MP4 files, and return relative paths."""
    prompt = str(params["prompt"])
    negative_prompt = str(params.get("negative_prompt") or "")
    steps = int(params.get("steps") or 25)
    cfg = float(params.get("cfg") or 7.5)
    width = int(params.get("width") or 512)
    height = int(params.get("height") or 512)
    seed = params.get("seed")
    scheduler = str(params.get("scheduler") or "ddim")
    model = params.get("model")
    motion_adapter = str(params.get("motion_adapter") or _DEFAULT_MOTION_ADAPTER)
    num_frames = int(params.get("num_frames") or 16)
    fps = int(params.get("fps") or 8)
    num_videos = int(params.get("num_videos") or 1)
    free_noise_enabled = _coerce_bool(params.get("free_noise_enabled", False))
    free_noise_context_length = _int_param(params, "free_noise_context_length", 16)
    free_noise_context_stride = _int_param(params, "free_noise_context_stride", 4)
    free_init_enabled = _coerce_bool(params.get("free_init_enabled", False))
    free_init_num_iters = _int_param(params, "free_init_num_iters", 3)
    free_init_use_fast_sampling = _coerce_bool(
        params.get("free_init_use_fast_sampling", False)
    )
    free_init_method = str(params.get("free_init_method") or "butterworth").lower()
    free_init_order = _int_param(params, "free_init_order", 4)
    free_init_spatial_stop_frequency = _float_param(
        params, "free_init_spatial_stop_frequency", 0.25
    )
    free_init_temporal_stop_frequency = _float_param(
        params, "free_init_temporal_stop_frequency", 0.25
    )
    clip_skip = int(params.get("clip_skip") or 1)
    lora_adapters = params.get("lora_adapters")
    weighting_policy = str(params.get("weighting_policy") or "diffusers-like")
    batch_id = params.get("batch_id")

    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    if fps < 1:
        raise ValueError("fps must be >= 1")
    if num_videos < 1:
        raise ValueError("num_videos must be >= 1")
    _validate_free_init_settings(
        num_iters=free_init_num_iters,
        method=free_init_method,
        order=free_init_order,
        spatial_stop_frequency=free_init_spatial_stop_frequency,
        temporal_stop_frequency=free_init_temporal_stop_frequency,
    )

    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)

    if batch_id is None:
        batch_id = make_batch_id()
    batch_id = str(batch_id)
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    pipe = load_text2video_pipeline(
        str(model) if model is not None else None,
        motion_adapter,
    )
    motion_max_seq_length = getattr(pipe, "_syntha_motion_max_seq_length", None)
    _validate_animatediff_frame_settings(
        num_frames=num_frames,
        free_noise_enabled=free_noise_enabled,
        free_noise_context_length=free_noise_context_length,
        free_noise_context_stride=free_noise_context_stride,
        motion_max_seq_length=motion_max_seq_length,
    )
    if free_noise_enabled:
        active_context_length = min(num_frames, free_noise_context_length)
        active_context_stride = min(free_noise_context_stride, active_context_length)
        _enable_free_noise(
            pipe,
            context_length=active_context_length,
            context_stride=active_context_stride,
        )
    _apply_animatediff_scheduler(pipe, scheduler)
    if free_init_enabled:
        _enable_free_init(
            pipe,
            num_iters=free_init_num_iters,
            use_fast_sampling=free_init_use_fast_sampling,
            method=free_init_method,
            order=free_init_order,
            spatial_stop_frequency=free_init_spatial_stop_frequency,
            temporal_stop_frequency=free_init_temporal_stop_frequency,
        )
    logger.info(
        "Generate AnimateDiff: model=%s motion_adapter=%s seed=%s scheduler=%s "
        "steps=%s cfg=%s size=%sx%s num_frames=%s fps=%s num_videos=%s "
        "free_noise=%s free_noise_context_length=%s free_noise_context_stride=%s "
        "free_init=%s free_init_num_iters=%s free_init_use_fast_sampling=%s "
        "free_init_method=%s free_init_order=%s free_init_spatial_stop_frequency=%s "
        "free_init_temporal_stop_frequency=%s",
        model,
        motion_adapter,
        base_seed,
        scheduler,
        steps,
        cfg,
        width,
        height,
        num_frames,
        fps,
        num_videos,
        free_noise_enabled,
        free_noise_context_length,
        free_noise_context_stride,
        free_init_enabled,
        free_init_num_iters,
        free_init_use_fast_sampling,
        free_init_method,
        free_init_order,
        free_init_spatial_stop_frequency,
        free_init_temporal_stop_frequency,
    )

    adapter_names, lora_coverage = apply_lora_adapters_with_validation(
        pipe,
        lora_adapters,
        expected_family="sd15",
        validate=True,
    )
    report_path = write_lora_coverage_report(batch_output_dir, batch_id, lora_coverage)
    if report_path is not None:
        logger.info("LoRA coverage report saved to %s", report_path)

    (
        prompt_input,
        negative_prompt_input,
        prompt_embeds,
        negative_prompt_embeds,
    ) = _prepare_animatediff_prompt_inputs(
        pipe,
        prompt=prompt,
        negative_prompt=negative_prompt,
        clip_skip=clip_skip,
        weighting_policy=weighting_policy,
        free_noise_enabled=free_noise_enabled,
    )

    filenames: list[str] = []
    metadata_path = _animatediff_video_metadata_path(batch_output_dir, batch_id)
    metadata: dict[str, Any] = {
        "mode": "sd15.animatediff.text2video",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "cfg": cfg,
        "width": width,
        "height": height,
        "scheduler": scheduler,
        "model": model,
        "motion_adapter": motion_adapter,
        "num_frames": num_frames,
        "fps": fps,
        "num_videos": num_videos,
        "free_noise_enabled": free_noise_enabled,
        "free_noise_context_length": free_noise_context_length,
        "free_noise_context_stride": free_noise_context_stride,
        "free_init_enabled": free_init_enabled,
        "free_init_num_iters": free_init_num_iters,
        "free_init_use_fast_sampling": free_init_use_fast_sampling,
        "free_init_method": free_init_method,
        "free_init_order": free_init_order,
        "free_init_spatial_stop_frequency": free_init_spatial_stop_frequency,
        "free_init_temporal_stop_frequency": free_init_temporal_stop_frequency,
        "clip_skip": clip_skip,
        "lora_adapters": lora_adapters,
        "weighting_policy": weighting_policy,
        "batch_id": batch_id,
        "base_seed": base_seed,
        "videos": [],
    }
    try:
        for i in range(num_videos):
            current_seed = base_seed + i
            generator = _make_animatediff_generator(
                seed=current_seed,
                free_noise_enabled=free_noise_enabled,
            )
            result = pipe(
                prompt=prompt_input,
                negative_prompt=negative_prompt_input,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                num_inference_steps=steps,
                guidance_scale=cfg,
                width=width,
                height=height,
                num_frames=num_frames,
                clip_skip=clip_skip,
                generator=generator,
                decode_chunk_size=8,
            )
            output_name = f"{batch_id}_{current_seed}.mp4"
            export_to_video(result.frames[0], batch_output_dir / output_name, fps=fps)
            relative_path = build_batch_output_relpath(batch_id, output_name)
            metadata["videos"].append(
                {
                    "filename": output_name,
                    "path": relative_path,
                    "seed": current_seed,
                    "index": i,
                }
            )
            if num_videos == 1:
                metadata["seed"] = current_seed
            _write_animatediff_video_metadata(metadata_path, metadata)
            logger.info("Video %s saved to %s", i, output_name)
            filenames.append(relative_path)
    finally:
        if free_init_enabled:
            _disable_free_init(pipe)
        _cleanup_lora_adapters(pipe, adapter_names)

    return filenames
