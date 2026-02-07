"""
Stable Diffusion 1.5 (SD1.5) pipeline helpers.

This module is responsible for:
- Loading Diffusers pipelines for txt2img, img2img, inpaint, and ControlNet.
- Running inference (CUDA / fp16) and writing PNG outputs + embedded metadata.
- Optional LoRA adapter application and pipeline-layer logging/diagnostics.

The functions here are used by workflow tasks (e.g. `sd15.text2img`), so they
aim to be deterministic (seeded) and side-effectful only in well-defined ways
(writing files under `OUTPUT_DIR`).
"""

import torch
import logging
import math
from pathlib import Path
from PIL import ImageFilter, Image

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)

from backend.config import OUTPUT_DIR
from backend.logging_utils import configure_logging
from backend.model_registry import get_model_entry
from backend.resource_logging import resource_logger
# from testing.pipeline_stable_diffusion import(StableDiffusionPipeline)
from backend.pipeline_utils import (
    build_fixed_step_timesteps,
    build_png_metadata,
    build_batch_output_relpath,
    get_batch_output_dir,
    make_batch_id,
    resolve_model_source,
)
from backend.schedulers import create_scheduler
from backend.prompt_utils import build_prompt_embeddings
from backend import config
from backend.pipeline_layer_logging import (
    append_layers_report,
    capture_runtime_used_layers,
    collect_pipeline_layers,
)
from backend.lora_utils import apply_lora_adapters_with_validation, write_lora_coverage_report

logger = logging.getLogger(__name__)
configure_logging()


## Helper functions

def create_blur_mask(mask_image, blur_factor: int):
    """
    Return a blurred copy of `mask_image` with a bounded Gaussian blur radius.

    Args:
        mask_image: PIL image used as an inpaint mask.
        blur_factor: Requested blur radius. Values are clamped to ``[0, 128]``.

    Returns:
        The original image when blur is ``0``; otherwise a blurred copy.
    """
    blur_factor = max(0, min(blur_factor, 128))
    if blur_factor == 0:
        return mask_image
    return mask_image.filter(ImageFilter.GaussianBlur(radius=blur_factor))


def _resource_metadata(bound_args):
    """
    Build resource-logging metadata from a function's bound arguments.

    This keeps the logging payload small and consistent across generation calls.
    """
    return {
        "batch_id": bound_args.arguments.get("batch_id"),
        "model": bound_args.arguments.get("model"),
        "num_images": bound_args.arguments.get("num_images"),
    }


def _snap_dimension(value: int, multiple: int = 8) -> int:
    """Round a dimension up to the next multiple (SD models commonly prefer multiples of 8)."""
    if multiple <= 0:
        return value
    return max(multiple, int(math.ceil(value / multiple)) * multiple)


def _upscale_image(image: Image.Image, scale: float) -> Image.Image:
    """Upscale an image by `scale` using Lanczos, snapping size to SD-friendly dimensions."""
    if scale <= 1.0:
        return image
    target_width = _snap_dimension(int(round(image.width * scale)))
    target_height = _snap_dimension(int(round(image.height * scale)))
    return image.resize((target_width, target_height), resample=Image.LANCZOS)


def apply_hires_fix(
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    steps: int,
    cfg: float,
    seed: int | None,
    scheduler: str,
    model: str | None,
    clip_skip: int,
    hires_scale: float,
    hires_strength: float = 0.35,
    lora_adapters: list[object] | None = None,
    prompt_embeds: torch.Tensor | None = None,
    negative_prompt_embeds: torch.Tensor | None = None,
    lora_scale: float | None = None,
) -> Image.Image:
    """
    Run a hires-fix pass by upscaling, then refining with SD1.5 img2img.

    Args:
        image: Input image to refine.
        prompt: Positive prompt text.
        negative_prompt: Negative prompt text.
        steps: Number of denoising steps for img2img refinement.
        cfg: Classifier-free guidance scale.
        seed: Optional seed for deterministic output.
        scheduler: Scheduler identifier used by ``create_scheduler``.
        model: Optional model registry key.
        clip_skip: CLIP skip value passed to Diffusers.
        hires_scale: Upscale factor. Values ``<= 1.0`` skip hires-fix.
        hires_strength: Img2img strength used during refinement.
        lora_adapters: Optional LoRA adapter specs.
        prompt_embeds: Optional precomputed positive prompt embeddings.
        negative_prompt_embeds: Optional precomputed negative prompt embeddings.
        lora_scale: Optional LoRA cross-attention scale.

    Returns:
        Refined image. If ``hires_scale <= 1.0``, returns input image unchanged.
    """
    if hires_scale <= 1.0:
        return image

    upscaled = _upscale_image(image, hires_scale)
    pipe = load_img2img_pipeline(model)
    pipe.scheduler = create_scheduler(scheduler, pipe)
    adapter_names = _apply_lora_adapters(pipe, lora_adapters)

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    try:
        return pipe(
            prompt=None if prompt_embeds is not None else prompt,
            negative_prompt=None if negative_prompt_embeds is not None else negative_prompt,
            image=upscaled,
            strength=hires_strength,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
            clip_skip=clip_skip,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            cross_attention_kwargs={"scale": lora_scale} if lora_scale is not None else None,
        ).images[0]
    finally:
        if adapter_names and hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()


def _apply_lora_adapters(
    pipe,
    lora_adapters: list[object] | None,
    *,
    validate: bool = False,
) -> list[str]:
    """
    Apply requested LoRA adapters to a pipeline.

    Returns:
        A list of adapter names actually loaded into the pipeline.
    """
    adapter_names, _ = apply_lora_adapters_with_validation(
        pipe,
        lora_adapters,
        expected_family="sd15",
        validate=validate,
    )
    return adapter_names

@torch.inference_mode()
def run_sd15_hires_fix(
    *,
    images: list[Image.Image],
    prompt: str,
    negative_prompt: str,
    steps: int,
    cfg: float,
    seed: int | None,
    scheduler: str,
    model: str | None,
    clip_skip: int,
    hires_scale: float,
    hires_strength: float = 0.35,
    lora_adapters: list[object] | None = None,
    weighting_policy: str = "diffusers-like",
    lora_scale: float | None = None,
    output_dir: Path | None = None,
    batch_id: str | None = None,
) -> list[str]:
    """
    Apply SD1.5 hires-fix to each input image and write PNGs to disk.

    Args:
        images: Source images to upscale/refine.
        prompt: Positive prompt text.
        negative_prompt: Negative prompt text.
        steps: Number of denoising steps.
        cfg: Classifier-free guidance scale.
        seed: Optional base seed. ``None`` or ``0`` selects a random base seed.
        scheduler: Scheduler name.
        model: Optional model registry key.
        clip_skip: CLIP skip value.
        hires_scale: Upscale factor. Must be ``> 1.0``.
        hires_strength: Img2img strength for refinement.
        lora_adapters: Optional LoRA adapter specs.
        weighting_policy: Prompt-weighting policy for embedding construction.
        lora_scale: Optional LoRA cross-attention scale.
        output_dir: Optional output root. Defaults to batch folder under ``OUTPUT_DIR``.
        batch_id: Optional batch identifier.

    Returns:
        List of output PNG paths relative to ``OUTPUT_DIR``.

    Raises:
        ValueError: If ``hires_scale <= 1.0``.
    """
    if hires_scale <= 1.0:
        raise ValueError("hires_scale must be > 1.0 for sd15.hires_fix")
    if not images:
        return []

    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)

    if batch_id is None:
        batch_id = make_batch_id()
    batch_output_dir = output_dir or get_batch_output_dir(OUTPUT_DIR, batch_id)

    pipe = load_img2img_pipeline(model)
    pipe.scheduler = create_scheduler(scheduler, pipe)
    adapter_names = _apply_lora_adapters(pipe, lora_adapters, validate=False)

    prompt_embeds, negative_prompt_embeds, use_prompt_embeds = build_prompt_embeddings(
        pipe,
        prompt,
        negative_prompt,
        clip_skip=clip_skip,
        lora_scale=lora_scale,
        weighting_policy=weighting_policy,
    )

    relpaths: list[str] = []
    try:
        for idx, image in enumerate(images):
            # Offset the seed per image to make batch outputs deterministic and distinct.
            current_seed = base_seed + idx
            generator = torch.Generator(device="cuda").manual_seed(current_seed)

            upscaled = _upscale_image(image, hires_scale)
            out_image = pipe(
                prompt=None if use_prompt_embeds else prompt,
                negative_prompt=None if use_prompt_embeds else negative_prompt,
                image=upscaled,
                strength=hires_strength,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
                clip_skip=clip_skip,
                prompt_embeds=prompt_embeds if use_prompt_embeds else None,
                negative_prompt_embeds=negative_prompt_embeds if use_prompt_embeds else None,
                cross_attention_kwargs={"scale": lora_scale} if lora_scale is not None else None,
            ).images[0]

            filename = batch_output_dir / f"{batch_id}_{current_seed}.png"
            # Store prompt/settings inside the PNG for later reproduction/debugging.
            pnginfo = build_png_metadata(
                {
                    "mode": "hires_fix",
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "steps": steps,
                    "cfg": cfg,
                    "seed": current_seed,
                    "scheduler": scheduler,
                    "model": model,
                    "clip_skip": clip_skip,
                    "hires_scale": hires_scale,
                    "hires_strength": hires_strength,
                    "batch_id": batch_id,
                }
            )
            out_image.save(filename, pnginfo=pnginfo)
            relpaths.append(build_batch_output_relpath(batch_id, filename.name))
    finally:
        if adapter_names and hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()

    return relpaths

## Load pipelines

def load_text2img_pipeline(model_name: str | None):
    """
    Load the base SD1.5 txt2img pipeline on CUDA fp16.

    ``model_name`` is resolved via the model registry and may point to a
    Diffusers directory model or a single-file checkpoint.

    Side effects:
        Moves the pipeline to GPU (``cuda``) and disables the safety checker.
    """
    entry = get_model_entry(model_name)

    source = resolve_model_source(entry)
    logger.info("URL: %s", source)
    
    if entry.model_type == "diffusers":
        pipe = StableDiffusionPipeline.from_pretrained(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,  # keep simple; can re-enable later
        )
    elif entry.model_type == "single-file":
        pipe = StableDiffusionPipeline.from_single_file(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    # Run on CUDA in fp16 for performance. Safety checker is disabled by design here.
    pipe.to("cuda")
    return pipe


def load_img2img_pipeline(model_name: str | None):
    """
    Load the SD1.5 img2img pipeline on CUDA fp16.

    Side effects:
        Moves the pipeline to GPU (``cuda``) and disables the safety checker.
    """
    entry = get_model_entry(model_name)

    source = resolve_model_source(entry)
    logger.info("URL: %s", source)
    if entry.model_type == "diffusers":
        img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        img2img_pipe = StableDiffusionImg2ImgPipeline.from_single_file(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    img2img_pipe.to("cuda")
    return img2img_pipe


def load_inpaint_pipeline(model_name: str | None):
    """
    Load the SD1.5 inpainting pipeline on CUDA fp16.

    Side effects:
        Moves the pipeline to GPU (``cuda``) and disables the safety checker.
    """
    entry = get_model_entry(model_name)

    source = resolve_model_source(entry)
    logger.info("URL: %s", source)
    if entry.model_type == "diffusers":
        inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        inpaint_pipe = StableDiffusionInpaintPipeline.from_single_file(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    inpaint_pipe.to("cuda")
    return inpaint_pipe


def load_controlnet_pipeline(model_name: str | None, controlnet_model: str | list[str]):
    """
    Load a ControlNet-enabled SD1.5 pipeline on CUDA fp16.

    Args:
        model_name: Optional base model registry key.
        controlnet_model: Diffusers ControlNet model id/path or list of ids/paths.

    Side effects:
        Loads both base and ControlNet weights and moves the pipeline to GPU.
    """
    entry = get_model_entry(model_name)

    source = resolve_model_source(entry)
    logger.info("Base model: %s", source)
    controlnet: ControlNetModel | list[ControlNetModel]
    if isinstance(controlnet_model, list):
        controlnet = [
            ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)
            for model_id in controlnet_model
        ]
    else:
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model,
            torch_dtype=torch.float16,
        )

    if entry.model_type == "diffusers":
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        pipe = StableDiffusionControlNetPipeline.from_single_file(
            source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    pipe.to("cuda")
    return pipe

## Generate and render images

def generate_images_controlnet(
    prompt: str,
    negative_prompt: str,
    steps: int,
    cfg: float,
    width: int,
    height: int,
    seed: int | None,
    scheduler: str,
    model: str | None,
    num_images: int,
    clip_skip: int,
    controlnet_model: str | list[str],
    control_image: Image.Image | list[Image.Image],
    controlnet_conditioning_scale: float | list[float] = 1.0,
    controlnet_guess_mode: bool = False,
    control_guidance_start: float = 0.0,
    control_guidance_end: float = 1.0,
    batch_id: str | None = None,
) -> list[str]:
    """
    Generate SD1.5 + ControlNet images and write PNG outputs to disk.

    This function optionally captures pipeline layer-usage diagnostics based on
    runtime configuration and embeds generation settings into PNG metadata.

    Returns:
        Output PNG paths relative to ``OUTPUT_DIR``.
    """
    if not batch_id:
        batch_id = make_batch_id()

    pipe = load_controlnet_pipeline(model, controlnet_model)
    pipe.scheduler = create_scheduler(scheduler, pipe)
    pipe.safety_checker = None
    pipe.enable_xformers_memory_efficient_attention()

    if clip_skip > 1:
        # Diffusers exposes clip-skip by effectively reducing the text encoder depth.
        pipe.text_encoder.config.num_hidden_layers = (
            pipe.text_encoder.config.num_hidden_layers - (clip_skip - 1)
        )

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    arch_layers = None
    used_layer_names = None
    name_to_type = None

    if config.PIPELINE_LAYER_LOGGING_ENABLED:
        # Optionally capture which layers run (useful for debugging pipeline variants).
        arch_layers = collect_pipeline_layers(
            pipe,
            leaf_only=config.PIPELINE_LAYER_LOGGING_LEAF_ONLY,
        )
        with capture_runtime_used_layers(
            pipe,
            leaf_only=config.PIPELINE_LAYER_LOGGING_LEAF_ONLY,
        ) as (used_layer_names, name_to_type, name_to_inputs, name_to_calls):
            results = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg,
                width=width,
                height=height,
                image=control_image,
                num_images_per_prompt=num_images,
                generator=generator,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                guess_mode=controlnet_guess_mode,
                control_guidance_start=control_guidance_start,
                control_guidance_end=control_guidance_end,
            )
    else:
        results = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            width=width,
            height=height,
            image=control_image,
            num_images_per_prompt=num_images,
            generator=generator,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guess_mode=controlnet_guess_mode,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
        )

    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    if config.PIPELINE_LAYER_LOGGING_ENABLED:
        append_layers_report(
            output_dir=batch_output_dir,
            batch_id=batch_id,
            label="sd15_controlnet",
            pipeline_name=pipe.__class__.__name__,
            architecture_layers=arch_layers,
            runtime_used_layer_names=used_layer_names,
            runtime_name_to_type=name_to_type,
            runtime_name_to_input_summary=(name_to_inputs if config.PIPELINE_LAYER_LOGGING_CAPTURE_INPUTS else None),
            runtime_name_to_call_count=(name_to_calls if config.PIPELINE_LAYER_LOGGING_CAPTURE_INPUTS else None),
        )

    # Embed settings into PNG metadata for reproducibility/debugging.
    metadata = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "cfg": cfg,
        "width": width,
        "height": height,
        "seed": seed,
        "scheduler": scheduler,
        "model": model,
        "controlnet_model": controlnet_model,
        "controlnet_conditioning_scale": controlnet_conditioning_scale,
        "controlnet_guess_mode": controlnet_guess_mode,
        "control_guidance_start": control_guidance_start,
        "control_guidance_end": control_guidance_end,
        "batch_id": batch_id,
    }
    if isinstance(controlnet_model, list):
        metadata["controlnet_models"] = controlnet_model
    if isinstance(controlnet_conditioning_scale, list):
        metadata["controlnet_conditioning_scales"] = controlnet_conditioning_scale
    png_info = build_png_metadata(metadata)

    filenames = []
    for idx, image in enumerate(results.images):
        name = f"{batch_id}_controlnet_{idx}.png"
        image.save(batch_output_dir / name, pnginfo=png_info)
        filenames.append(build_batch_output_relpath(batch_id, name))

    return filenames

@torch.inference_mode()
def generate_images(
    prompt: str,
    negative_prompt: str,
    steps: int,
    cfg: float,
    width: int,
    height: int,
    seed: int,
    scheduler: str,
    model: str | None,
    num_images:int,
    clip_skip: int,
    lora_adapters: list[object] | None = None,
    hires_scale: float = 1.0,
    hires_enabled: bool = False,
    weighting_policy: str = "diffusers-like",
    batch_id: str | None = None,
    lora_scale: float | None = None,
):
    """
    Generate SD1.5 txt2img images, write PNG outputs, and return relative paths.

    Features:
        - Optional LoRA adapter loading with coverage report output.
        - Optional prompt embedding path for prompt-weighting/clip-skip policies.
        - Optional runtime layer logging on the first generated image.
        - Embedded PNG metadata for reproducibility.

    Notes:
        ``hires_enabled``/``hires_scale`` are currently recorded in metadata for
        downstream usage; this function itself performs txt2img generation only.
    """
    # 1. Check and set seed number(if not present, set random seed)
    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = seed
    
    # 2. Set batch_id for output folder
    if batch_id is None:
        batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)
    filenames = []
    
    # 3. Load pipeline and chosen scheduler
    pipe = load_pipeline(model)
    pipe.scheduler = create_scheduler(scheduler, pipe)
    logger.info("Generate: model=%s seed=%s scheduler=%s steps=%s cfg=%s size=%sx%s num_images=%s", model, base_seed, scheduler, steps, cfg, width, height, num_images,)
    
    # 4. Apply lora to pipeline and generate lora coverage report
    adapter_names, lora_coverage = apply_lora_adapters_with_validation(
        pipe,
        lora_adapters,
        expected_family="sd15",
        validate=True,
    )
    report_path = write_lora_coverage_report(batch_output_dir, batch_id, lora_coverage)
    if report_path is not None:
        logger.info("LoRA coverage report saved to %s", report_path)

    arch_layers = None
    if config.PIPELINE_LAYER_LOGGING_ENABLED:
        arch_layers = collect_pipeline_layers(pipe, leaf_only=config.PIPELINE_LAYER_LOGGING_LEAF_ONLY,)

    # 5. Build prompt embeddings
    prompt_embeds = None
    negative_prompt_embeds = None
    use_prompt_embeds = False
    prompt_embeds_ready = False
    if not config.PIPELINE_LAYER_LOGGING_ENABLED:
        prompt_embeds, negative_prompt_embeds, use_prompt_embeds = build_prompt_embeddings(
            pipe,
            prompt,
            negative_prompt,
            clip_skip=clip_skip,
            lora_scale=lora_scale,
            weighting_policy=weighting_policy,
        )
        prompt_embeds_ready = True

    # 6. Loop around image generation per image
    try:
        for i in range(num_images):
            # Offset seed per image so batches are deterministic and distinct.
            current_seed = base_seed + i
            generator = torch.Generator(device="cuda").manual_seed(current_seed)

            # Capture used layers during rendering
            if config.PIPELINE_LAYER_LOGGING_ENABLED and i == 0:
                with capture_runtime_used_layers(
                    pipe,
                    leaf_only=config.PIPELINE_LAYER_LOGGING_LEAF_ONLY,
                ) as (used_layer_names, name_to_type, name_to_inputs, name_to_calls):
                    prompt_embeds, negative_prompt_embeds, use_prompt_embeds = build_prompt_embeddings(
                        pipe,
                        prompt,
                        negative_prompt,
                        clip_skip=clip_skip,
                        lora_scale=lora_scale,
                        weighting_policy=weighting_policy,
                    )
                    prompt_embeds_ready = True
                     
                    # Generate image
                    image = pipe(
                        prompt=None if use_prompt_embeds else prompt,
                        negative_prompt=None if use_prompt_embeds else negative_prompt,
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        width=width,
                        height=height,
                        generator=generator,
                        clip_skip=clip_skip,
                        prompt_embeds=prompt_embeds if use_prompt_embeds else None,
                        negative_prompt_embeds=negative_prompt_embeds if use_prompt_embeds else None,
                        cross_attention_kwargs={"scale": lora_scale} if lora_scale is not None else None,
                    ).images[0]

                # Log layers to report
                append_layers_report(
                    output_dir=batch_output_dir,
                    batch_id=batch_id,
                    label="sd15_txt2img",
                    pipeline_name=pipe.__class__.__name__,
                    architecture_layers=arch_layers,
                    runtime_used_layer_names=used_layer_names,
                    runtime_name_to_type=name_to_type,
                    runtime_name_to_input_summary=(name_to_inputs if config.PIPELINE_LAYER_LOGGING_CAPTURE_INPUTS else None),
                    runtime_name_to_call_count=(name_to_calls if config.PIPELINE_LAYER_LOGGING_CAPTURE_INPUTS else None),
                )
            else:
                # If prompt embeds not present, generate them
                if not prompt_embeds_ready:
                    prompt_embeds, negative_prompt_embeds, use_prompt_embeds = build_prompt_embeddings(
                        pipe,
                        prompt,
                        negative_prompt,
                        clip_skip=clip_skip,
                        lora_scale=lora_scale,
                        weighting_policy=weighting_policy,
                    )
                    prompt_embeds_ready = True
                # Generate image
                image = pipe(
                    prompt=None if use_prompt_embeds else prompt,
                    negative_prompt=None if use_prompt_embeds else negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    width=width,
                    height=height,
                    generator=generator,
                    clip_skip=clip_skip,
                    prompt_embeds=prompt_embeds if use_prompt_embeds else None,
                    negative_prompt_embeds=negative_prompt_embeds if use_prompt_embeds else None,
                    cross_attention_kwargs={"scale": lora_scale} if lora_scale is not None else None,
                ).images[0]

            # Write the PNG and embed all inputs/settings for later inspection.
            filename = batch_output_dir / f"{batch_id}_{current_seed}.png"
            pnginfo = build_png_metadata({
                "mode": "txt2img",
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "cfg": cfg,
                "width": width,
                "height": height,
                "seed": current_seed,
                "scheduler": scheduler,
                "model": model,
                "clip_skip": clip_skip,
                "hires_enabled": hires_enabled,
                "hires_scale": hires_scale,
                "batch_id": batch_id,
            })
            image.save(filename, pnginfo=pnginfo)
            logger.info("Image %s saved to %s", i, filename.name)

            filenames.append(build_batch_output_relpath(batch_id, filename.name))
    finally:
        # Unload lora_weights at the end
        if adapter_names and hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()
    # Return list of filenames
    return filenames

@torch.inference_mode()
def generate_images_img2img(
    initial_image,
    strength: float,
    prompt: str,
    negative_prompt: str,
    steps: int,
    cfg: float,
    width: int,
    height: int,
    seed: int,
    scheduler: str,
    model: str | None,
    num_images: int,
    clip_skip: int,
    lora_adapters: list[object] | None = None,
    batch_id: str | None = None,
):
    """
    Generate SD1.5 img2img outputs from an initial image and write PNG files.

    Args:
        initial_image: Source image for img2img.
        strength: Img2img denoise strength.
        prompt: Positive prompt text.
        negative_prompt: Negative prompt text.
        steps: Number of denoising steps.
        cfg: Classifier-free guidance scale.
        width: Requested width (used for logging/compatibility).
        height: Requested height (used for logging/compatibility).
        seed: Base seed; ``None``/``0`` selects a random base seed.
        scheduler: Scheduler name.
        model: Optional model registry key.
        num_images: Number of images to generate.
        clip_skip: CLIP skip value.
        lora_adapters: Optional LoRA adapter specs.
        batch_id: Optional batch identifier.

    Returns:
        Output PNG paths relative to ``OUTPUT_DIR``.
    """
    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = seed
    if batch_id is None:
        batch_id = make_batch_id()

    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    pipe = load_img2img_pipeline(model)
    pipe.scheduler = create_scheduler(scheduler, pipe)
    logger.info(
        "Img2Img: model=%s seed=%s scheduler=%s steps=%s cfg=%s size=%sx%s strength=%s num_images=%s",
        model,
        base_seed,
        scheduler,
        steps,
        cfg,
        width,
        height,
        strength,
        num_images,
    )

    filenames = []
    adapter_names = _apply_lora_adapters(pipe, lora_adapters)

    try:
        for i in range(num_images):
            # Offset seed per image so batches are deterministic and distinct.
            current_seed = base_seed + i
            generator = torch.Generator(device="cuda").manual_seed(current_seed)

            if config.PIPELINE_LAYER_LOGGING_ENABLED and i == 0:
                arch_layers = collect_pipeline_layers(
                    pipe,
                    leaf_only=config.PIPELINE_LAYER_LOGGING_LEAF_ONLY,
                )
                with capture_runtime_used_layers(
                    pipe,
                    leaf_only=config.PIPELINE_LAYER_LOGGING_LEAF_ONLY,
                ) as (used_layer_names, name_to_type, name_to_inputs, name_to_calls):
                    image = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=initial_image,
                        strength=strength,
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        generator=generator,
                        clip_skip=clip_skip,
                    ).images[0]
                append_layers_report(
                    output_dir=batch_output_dir,
                    batch_id=batch_id,
                    label="sd15_img2img",
                    pipeline_name=pipe.__class__.__name__,
                    architecture_layers=arch_layers,
                    runtime_used_layer_names=used_layer_names,
                    runtime_name_to_type=name_to_type,
                    runtime_name_to_input_summary=(name_to_inputs if config.PIPELINE_LAYER_LOGGING_CAPTURE_INPUTS else None),
                    runtime_name_to_call_count=(name_to_calls if config.PIPELINE_LAYER_LOGGING_CAPTURE_INPUTS else None),
                )
            else:
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=initial_image,
                    strength=strength,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=generator,
                    clip_skip=clip_skip,
                ).images[0]

            filename = batch_output_dir / f"{batch_id}_{current_seed}.png"
            image_width, image_height = initial_image.size
            pnginfo = build_png_metadata({
                "mode": "img2img",
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "cfg": cfg,
                "width": image_width,
                "height": image_height,
                "seed": current_seed,
                "scheduler": scheduler,
                "model": model,
                "strength": strength,
                "clip_skip": clip_skip,
                "batch_id": batch_id,
            })
            image.save(filename, pnginfo=pnginfo)
            logger.info("Image %s saved to %s", i, filename.name)

            filenames.append(build_batch_output_relpath(batch_id, filename.name))
    finally:
        if adapter_names and hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()

    return filenames

@torch.inference_mode()
def generate_images_inpaint(
    initial_image,
    mask_image,
    prompt: str,
    negative_prompt: str,
    steps: int,
    cfg: float,
    seed: int,
    scheduler: str,
    model: str | None,
    num_images: int,
    strength: float,
    padding_mask_crop: int,
    clip_skip: int,
    batch_id: str | None = None,
):
    """
    Generate SD1.5 inpaint outputs from an initial image and mask.

    This function writes PNG files to the batch directory, stores generation
    settings in PNG metadata, and optionally captures layer-usage diagnostics.

    Returns:
        Output PNG paths relative to ``OUTPUT_DIR``.
    """
    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = seed
    if batch_id is None:
        batch_id = make_batch_id()

    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    pipe = load_inpaint_pipeline(model)
    pipe.scheduler = create_scheduler(scheduler, pipe)
    width, height = initial_image.size
    logger.info(
        "Inpaint: model=%s seed=%s scheduler=%s steps=%s cfg=%s size=%sx%s num_images=%s strength=%s, padding_mask_crop=%s",
        model, base_seed, scheduler, steps, cfg,
        width, height, num_images, strength, padding_mask_crop
    )

    filenames = []

    for i in range(num_images):
        # Offset seed per image so batches are deterministic and distinct.
        current_seed = base_seed + i
        generator = torch.Generator(device="cuda").manual_seed(current_seed)

        if config.PIPELINE_LAYER_LOGGING_ENABLED and i == 0:
            arch_layers = collect_pipeline_layers(
                pipe,
                leaf_only=config.PIPELINE_LAYER_LOGGING_LEAF_ONLY,
            )
            with capture_runtime_used_layers(
                pipe,
                leaf_only=config.PIPELINE_LAYER_LOGGING_LEAF_ONLY,
            ) as (used_layer_names, name_to_type, name_to_inputs, name_to_calls):
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=initial_image,
                    mask_image=mask_image,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=generator,
                    strength=strength,
                    padding_mask_crop=padding_mask_crop,
                    clip_skip=clip_skip,
                ).images[0]
            append_layers_report(
                output_dir=batch_output_dir,
                batch_id=batch_id,
                label="sd15_inpaint",
                pipeline_name=pipe.__class__.__name__,
                architecture_layers=arch_layers,
                runtime_used_layer_names=used_layer_names,
                runtime_name_to_type=name_to_type,
                runtime_name_to_input_summary=(name_to_inputs if config.PIPELINE_LAYER_LOGGING_CAPTURE_INPUTS else None),
                runtime_name_to_call_count=(name_to_calls if config.PIPELINE_LAYER_LOGGING_CAPTURE_INPUTS else None),
            )
        else:
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=initial_image,
                mask_image=mask_image,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
                strength=strength,
                padding_mask_crop=padding_mask_crop,
                clip_skip=clip_skip,
            ).images[0]

        filename = batch_output_dir / f"{batch_id}_{current_seed}.png"
        # Embed all settings into the PNG for later reproduction/debugging.
        pnginfo = build_png_metadata({
            "mode": "inpaint",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg": cfg,
            "width": width,
            "height": height,
            "seed": current_seed,
            "scheduler": scheduler,
            "model": model,
            "strength": strength,
            "padding_mask_crop": padding_mask_crop,
            "clip_skip": clip_skip,
            "batch_id": batch_id,
        })
        image.save(filename, pnginfo=pnginfo)
        logger.info("Image %s saved to %s", i, filename.name)

        filenames.append(build_batch_output_relpath(batch_id, filename.name))

    return filenames
