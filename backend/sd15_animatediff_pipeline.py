"""
SD1.5 AnimateDiff text-to-video pipeline helpers.

This module mirrors the workflow-facing conventions of ``backend.sd15_pipeline``
while keeping AnimateDiff loading/generation isolated from the existing image
pipelines.
"""

from __future__ import annotations

import logging
from pathlib import Path

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
    _apply_animatediff_scheduler(pipe, scheduler)
    logger.info(
        "Generate AnimateDiff: model=%s motion_adapter=%s seed=%s scheduler=%s "
        "steps=%s cfg=%s size=%sx%s num_frames=%s fps=%s num_videos=%s",
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

    prompt_embeds, negative_prompt_embeds, use_prompt_embeds = build_prompt_embeddings(
        pipe,
        prompt,
        negative_prompt,
        clip_skip=clip_skip,
        weighting_policy=weighting_policy,
    )

    filenames: list[str] = []
    try:
        for i in range(num_videos):
            current_seed = base_seed + i
            generator = torch.Generator(device="cuda").manual_seed(current_seed)
            result = pipe(
                prompt=None if use_prompt_embeds else prompt,
                negative_prompt=None if use_prompt_embeds else negative_prompt,
                prompt_embeds=prompt_embeds if use_prompt_embeds else None,
                negative_prompt_embeds=negative_prompt_embeds if use_prompt_embeds else None,
                num_inference_steps=steps,
                guidance_scale=cfg,
                width=width,
                height=height,
                num_frames=num_frames,
                clip_skip=clip_skip,
                generator=generator,
            )
            output_name = f"{batch_id}_{current_seed}.mp4"
            export_to_video(result.frames[0], batch_output_dir / output_name, fps=fps)
            logger.info("Video %s saved to %s", i, output_name)
            filenames.append(build_batch_output_relpath(batch_id, output_name))
    finally:
        _cleanup_lora_adapters(pipe, adapter_names)

    return filenames
