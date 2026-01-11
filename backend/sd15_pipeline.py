import torch
import logging
import math
from PIL import ImageFilter, Image

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)
from pathlib import Path

from backend.logging_utils import configure_logging
from backend.model_registry import get_model_entry
from backend.lora_registry import get_lora_entry
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

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PIPELINE_CACHE: dict[str, StableDiffusionPipeline] = {}
IMG2IMG_PIPELINE_CACHE: dict[str, StableDiffusionImg2ImgPipeline] = {}
INPAINT_PIPELINE_CACHE: dict[str, StableDiffusionInpaintPipeline] = {}
CONTROLNET_PIPELINE_CACHE: dict[str, StableDiffusionControlNetPipeline] = {}

logger = logging.getLogger(__name__)
configure_logging()


## Helper functions

def create_blur_mask(mask_image, blur_factor: int):
    blur_factor = max(0, min(blur_factor, 128))
    if blur_factor == 0:
        return mask_image
    return mask_image.filter(ImageFilter.GaussianBlur(radius=blur_factor))


def _resource_metadata(bound_args):
    return {
        "batch_id": bound_args.arguments.get("batch_id"),
        "model": bound_args.arguments.get("model"),
        "num_images": bound_args.arguments.get("num_images"),
    }


def _snap_dimension(value: int, multiple: int = 8) -> int:
    if multiple <= 0:
        return value
    return max(multiple, int(math.ceil(value / multiple)) * multiple)


def _upscale_image(image: Image.Image, scale: float) -> Image.Image:
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
) -> Image.Image:
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
        ).images[0]
    finally:
        if adapter_names and hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()


def _apply_lora_adapters(pipe, lora_adapters: list[object] | None) -> list[str]:
    if not lora_adapters:
        return []

    adapter_names: list[str] = []
    weights: list[float] = []
    for adapter in lora_adapters:
        if isinstance(adapter, dict):
            lora_id = adapter.get("lora_id")
            strength = adapter.get("strength", 1.0)
        else:
            lora_id = getattr(adapter, "lora_id", None)
            strength = getattr(adapter, "strength", 1.0)

        if lora_id is None:
            raise ValueError("LoRA adapter missing lora_id.")
        entry = get_lora_entry(int(lora_id))

        if entry.lora_model_family.lower() != "sd15":
            raise ValueError(f"LoRA {entry.name} is not compatible with SD1.5.")

        adapter_name = f"lora_{entry.name}"
        source = entry.file_path
        pipe.load_lora_weights(source, adapter_name=adapter_name)
        adapter_names.append(adapter_name)
        weights.append(float(strength))
        logger.info(
            "lora_name: %s , lora_id: %s, lora_weight: %s",
            adapter_name,
            entry.lora_id,
            strength,
        )
    if hasattr(pipe, "set_adapters"):
        pipe.set_adapters(adapter_names, adapter_weights=weights)

    return adapter_names

## Load pipelines

def load_pipeline(model_name: str | None):
    entry = get_model_entry(model_name)

    if entry.name in PIPELINE_CACHE:
        return PIPELINE_CACHE[entry.name]

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

    pipe.to("cuda")
    PIPELINE_CACHE[entry.name] = pipe

    return pipe


def load_img2img_pipeline(model_name: str | None):
    entry = get_model_entry(model_name)

    if entry.name in IMG2IMG_PIPELINE_CACHE:
        return IMG2IMG_PIPELINE_CACHE[entry.name]

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
    IMG2IMG_PIPELINE_CACHE[entry.name] = img2img_pipe

    return img2img_pipe


def load_inpaint_pipeline(model_name: str | None):
    entry = get_model_entry(model_name)

    if entry.name in INPAINT_PIPELINE_CACHE:
        return INPAINT_PIPELINE_CACHE[entry.name]

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
    INPAINT_PIPELINE_CACHE[entry.name] = inpaint_pipe

    return inpaint_pipe


def load_controlnet_pipeline(model_name: str | None, controlnet_model: str):
    entry = get_model_entry(model_name)
    cache_key = f"{entry.name}::{controlnet_model}"
    if cache_key in CONTROLNET_PIPELINE_CACHE:
        return CONTROLNET_PIPELINE_CACHE[cache_key]

    source = resolve_model_source(entry)
    logger.info("Base model: %s", source)
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
    CONTROLNET_PIPELINE_CACHE[cache_key] = pipe
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
    controlnet_model: str,
    control_image: Image.Image,
    batch_id: str | None = None,
) -> list[str]:
    if not batch_id:
        batch_id = make_batch_id()

    pipe = load_controlnet_pipeline(model, controlnet_model)
    pipe.scheduler = create_scheduler(scheduler, pipe)
    pipe.safety_checker = None
    pipe.enable_xformers_memory_efficient_attention()

    if clip_skip > 1:
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
        "batch_id": batch_id,
    }
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
):
    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = seed
    if batch_id is None:
        batch_id = make_batch_id()

    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)
    
    pipe = load_pipeline(model)
    pipe.scheduler = create_scheduler(scheduler, pipe)
    logger.info(
        "Generate: model=%s seed=%s scheduler=%s steps=%s cfg=%s size=%sx%s num_images=%s",
        model, base_seed, scheduler, steps, cfg, width, height, num_images,)
        
    filenames = []
    adapter_names = _apply_lora_adapters(pipe, lora_adapters)

    arch_layers = None
    if config.PIPELINE_LAYER_LOGGING_ENABLED:
        arch_layers = collect_pipeline_layers(
            pipe,
            leaf_only=config.PIPELINE_LAYER_LOGGING_LEAF_ONLY,
        )

    prompt_embeds = None
    negative_prompt_embeds = None
    use_prompt_embeds = False
    prompt_embeds_ready = False
    if not config.PIPELINE_LAYER_LOGGING_ENABLED:
        prompt_embeds, negative_prompt_embeds, use_prompt_embeds = build_prompt_embeddings(
            pipe,
            prompt,
            negative_prompt,
            weighting_policy=weighting_policy,
        )
        prompt_embeds_ready = True

    try:
        for i in range(num_images):
            current_seed = base_seed + i
            generator = torch.Generator(device="cuda").manual_seed(current_seed)

            if config.PIPELINE_LAYER_LOGGING_ENABLED and i == 0:
                with capture_runtime_used_layers(
                    pipe,
                    leaf_only=config.PIPELINE_LAYER_LOGGING_LEAF_ONLY,
                ) as (used_layer_names, name_to_type, name_to_inputs, name_to_calls):
                    prompt_embeds, negative_prompt_embeds, use_prompt_embeds = build_prompt_embeddings(
                        pipe,
                        prompt,
                        negative_prompt,
                        weighting_policy=weighting_policy,
                    )
                    prompt_embeds_ready = True
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
                    ).images[0]

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
                if not prompt_embeds_ready:
                    prompt_embeds, negative_prompt_embeds, use_prompt_embeds = build_prompt_embeddings(
                        pipe,
                        prompt,
                        negative_prompt,
                        weighting_policy=weighting_policy,
                    )
                    prompt_embeds_ready = True
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
                ).images[0]

            if hires_enabled:
                image = apply_hires_fix(
                    image=image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    steps=steps,
                    cfg=cfg,
                    seed=current_seed,
                    scheduler=scheduler,
                    model=model,
                    clip_skip=clip_skip,
                    hires_scale=hires_scale,
                    lora_adapters=lora_adapters,
                    prompt_embeds=prompt_embeds if use_prompt_embeds else None,
                    negative_prompt_embeds=negative_prompt_embeds if use_prompt_embeds else None,
                )

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
        if adapter_names and hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()

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
