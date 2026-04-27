"""
Docstring for backend.flux.pipeline
"""
import logging
import threading
from typing import Any

import torch
from diffusers import FluxImg2ImgPipeline, FluxInpaintPipeline, FluxPipeline
try:
    from diffusers import FluxFillPipeline
except ImportError:  # pragma: no cover - depends on installed diffusers version
    FluxFillPipeline = None
from custom_pipelines.Flux.pipeline_flux import FluxPipeline as CustomFluxPipeline

from backend.config import OUTPUT_DIR
from backend.utilities.logging import configure_logging
from backend.lora.utils import apply_lora_adapters_with_validation, write_lora_coverage_report
from backend.registries.model import get_model_entry
from backend.utilities.pipeline import (
    build_png_metadata,
    build_batch_output_relpath,
    cleanup_memory,
    get_batch_output_dir,
    make_batch_id,
    release_pipeline,
    resolve_model_source,
)
from backend.utilities.schedulers import create_scheduler

"""
    Static Variables and Logging
"""
GEN_LOCK = threading.Lock()

logger = logging.getLogger(__name__)
configure_logging()


def _should_use_flux_fill_pipeline(model_name: str | None, source: str, version: str) -> bool:
    joined = " ".join([model_name or "", source or "", version or ""]).lower()
    return "flux" in joined and "fill" in joined


"""
    Methods for loading flux pipelines
"""

def load_text2img_pipeline(model_name: str | None) -> Any:
    # 1. Check input model_name is valid and load valid path
    entry = get_model_entry(model_name)
    source = resolve_model_source(entry)
    logger.info("Flux model source: %s", source)

    # 2. Check if diffusers multi-folder or single-file checkpoint
    if entry.model_type == "diffusers":
        pipe = CustomFluxPipeline.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
        )
    elif entry.model_type == "single-file":
        pipe = CustomFluxPipeline.from_single_file(
            source,
            torch_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")
    
    # if entry.model_type == "diffusers":
    #     pipe = FluxPipeline.from_pretrained(
    #         source,
    #         torch_dtype=torch.bfloat16,
    #     )
    # elif entry.model_type == "single-file":
    #     pipe = FluxPipeline.from_single_file(
    #         source,
    #         torch_dtype=torch.bfloat16,
    #     )
    # else:
    #     raise ValueError(f"Unsupported model type: {entry.model_type}")

    #3. Set pipeline settings to prevent OOM
    # pipe.enable_attention_slicing("max")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.enable_sequential_cpu_offload()

    return pipe

def load_img2img_pipeline(model_name: str | None) -> FluxImg2ImgPipeline:
    
    #1. Check input model_name is valid and load valid path
    entry = get_model_entry(model_name)
    source = resolve_model_source(entry)
    logger.info("Flux img2img model source: %s", source)

    #2. Check if diffusers multi-folder or single-file checkpoint
    if entry.model_type == "diffusers":
        pipe = FluxImg2ImgPipeline.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
        )
    elif entry.model_type == "single-file":
        pipe = FluxImg2ImgPipeline.from_single_file(
            source,
            torch_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    #3. Set pipeline settings: enable vae slicing and tiling to reduce vram and sequential cpu offload
    pipe.enable_attention_slicing("max")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.enable_sequential_cpu_offload()

    return pipe

def load_inpaint_pipeline(model_name: str | None) -> Any:
    #1. Check input model_name is valid and load valid path
    entry = get_model_entry(model_name)
    source = resolve_model_source(entry)
    logger.info("Flux inpaint model source: %s", source)
    
    if _should_use_flux_fill_pipeline(entry.name, source, entry.version):
        if FluxFillPipeline is None:
            raise ValueError(
                "Flux Fill model selected but FluxFillPipeline is unavailable in the installed diffusers package. "
                "Install a diffusers build with Flux Fill support."
            )
        pipeline_cls = FluxFillPipeline
        pipeline_name = "FluxFillPipeline"
    else:
        pipeline_cls: Any = FluxInpaintPipeline
        pipeline_name = "FluxInpaintPipeline"        
    logger.info("Flux inpaint pipeline class: %s", pipeline_name)

    #2. Check if diffusers multi-folder or single-file checkpoint
    if entry.model_type == "diffusers":
        pipe = pipeline_cls.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
        )
    elif entry.model_type == "single-file":
        pipe = pipeline_cls.from_single_file(
            source,
            torch_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    #3. Set pipeline settings:
    pipe.enable_attention_slicing("max")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.enable_sequential_cpu_offload()

    return pipe

"""
    Methods which generates and renders image using Flux-related Pipelines
"""

@torch.inference_mode()
def generate_text2img(params: dict[str, object]) -> dict[str, list[str]]:
    # 1. Load and create local method variables + ensure correct formatting from input dict
    prompt = str(params["prompt"])
    negative_prompt = str(params["negative_prompt"])
    steps = int(params["steps"])
    guidance_scale = float(params["guidance_scale"])
    width = int(params["width"])
    height = int(params["height"])
    seed = params["seed"]
    model = params["model"]
    num_images = int(params["num_images"])
    scheduler = str(params["scheduler"])
    lora_adapters = params["lora_adapters"]

    # 2. Check and set seed value
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)
    logger.info(
        "Flux Text2Image: model=%s, seed=%s, steps=%s, guidance_scale=%s, size=%sx%s, num_images=%s",
        model, base_seed, steps, guidance_scale, width, height, num_images,
    )

    # 3. Create batch_id and output directory
    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    # 4. Load and create pipeline and scheduler
    pipe = load_text2img_pipeline(model)
    pipe.scheduler = create_scheduler(scheduler, pipe)
    
    # 5. Load lora into pipeline
    adapter_names: list[str] = []
    adapter_names, lora_coverage = apply_lora_adapters_with_validation(
        pipe,
        lora_adapters,
        expected_family="flux",
        validate=True,
    )
    report_path = write_lora_coverage_report(batch_output_dir, batch_id, lora_coverage)
    if report_path is not None:
        logger.info("LoRA coverage report saved to %s", report_path)

    #6. Create list of filenames
    filenames: list[str] = []
    
    #7. Render image
    try:
        with GEN_LOCK:
            for i in range(num_images):
                # Define current seed
                current_seed = base_seed + i
                generator = torch.Generator(device="cpu").manual_seed(current_seed)

                # Render image
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    call_kwargs: dict[str, object] = {
                        "prompt": prompt,
                        "num_inference_steps": steps,
                        "guidance_scale": guidance_scale,
                        "width": width,
                        "height": height,
                        "generator": generator,
                    }
                    if negative_prompt:
                        call_kwargs["negative_prompt"] = negative_prompt

                    image = pipe(**call_kwargs).images[0]

                # Set filename and create image_params metadata dioct
                filename = batch_output_dir / f"{batch_id}_{current_seed}.png"
                image_params = dict(params)
                image_params.update({
                    "mode": "txt2img",
                    "pipeline": "flux",
                    "seed": current_seed,
                    "batch_id": batch_id,
                })
                pnginfo = build_png_metadata(image_params)
                image.save(filename, pnginfo=pnginfo)
                logger.info("Image %s saved to %s", i, filename.name)

                # Save filename to rendered image
                filenames.append(build_batch_output_relpath(batch_id, filename.name))
                del image
                cleanup_memory()
    finally:
        #8. Load pipeline + clean memory
        if pipe is not None and adapter_names and hasattr(pipe, "unload_lora_weights"):
            try:
                pipe.unload_lora_weights()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                logger.warning("Failed to unload LoRA weights: %s", exc)

        release_pipeline(pipe, logger=logger)
        pipe = None

    #9.  Return output
    return {"images": [f"/outputs/{name}" for name in filenames]}


@torch.inference_mode()
def generate_img2img(params: dict[str, object]) -> dict[str, list[str]]:
    #1. Load and create local method variables + ensure correct formatting from input dict
    initial_image = params["initial_image"]
    strength = float(params["strength"])
    prompt = str(params["prompt"])
    negative_prompt = str(params["negative_prompt"])
    steps = int(params["steps"])
    guidance_scale = float(params["guidance_scale"])
    width = int(params["width"])
    height = int(params["height"])
    seed = params["seed"]
    model = params["model"]
    num_images = int(params["num_images"])
    scheduler = str(params["scheduler"])
    lora_adapters = params["lora_adapters"]

    #2. Check and set seed value
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)
    logger.info(
        "Flux Img2Img: model=%s, seed=%s, steps=%s, guidance_scale=%s, size=%sx%s, strength=%s, num_images=%s",
        model, base_seed, steps, guidance_scale, width, height, strength,num_images,
    )
    
    #3. Create batch_id and output directory
    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    #4. Load and create pipeline and scheduler
    pipe = load_img2img_pipeline(model)
    pipe.scheduler = create_scheduler(scheduler, pipe)
    
    #5. Load lora into pipeline
    adapter_names: list[str] = []
    adapter_names, lora_coverage = apply_lora_adapters_with_validation(
        pipe,
        lora_adapters,
        expected_family="flux",
        validate=True,
    )
    report_path = write_lora_coverage_report(batch_output_dir, batch_id, lora_coverage)
    if report_path is not None:
        logger.info("LoRA coverage report saved to %s", report_path)

    #6. Create list of filenames
    filenames: list[str] = []
    
    #7. Render images one by one
    try:
        with GEN_LOCK:
            for i in range(num_images):
                #Set current seed
                current_seed = base_seed + i
                generator = torch.Generator(device="cpu").manual_seed(current_seed)
                
                #Render image
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    call_kwargs = dict(
                        prompt=prompt,
                        image=initial_image,
                        strength=strength,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                        generator=generator,
                    )
                    if negative_prompt:
                        call_kwargs["negative_prompt"] = negative_prompt

                    image = pipe(**call_kwargs).images[0]

                # define filenames and create image_params dict to save as image metadata
                filename = batch_output_dir / f"{batch_id}_{current_seed}.png"
                image_width, image_height = initial_image.size
                image_params = dict(params)
                image_params.pop("initial_image", None)
                image_params.update({
                    "mode": "img2img",
                    "pipeline": "flux",
                    "width": image_width,
                    "height": image_height,
                    "seed": current_seed,
                    "batch_id": batch_id,
                })
                pnginfo = build_png_metadata(image_params)
                image.save(filename, pnginfo=pnginfo)
                logger.info("Image %s saved to %s", i, filename.name)
                filenames.append(build_batch_output_relpath(batch_id, filename.name))

                del image
                cleanup_memory()
    finally:
        #8. Unload lora weights & clean memory
        if pipe is not None and adapter_names and hasattr(pipe, "unload_lora_weights"):
            try:
                pipe.unload_lora_weights()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                logger.warning("Failed to unload LoRA weights: %s", exc)

        release_pipeline(pipe, logger=logger)
        pipe = None

    #9. Return output
    return {"images": [f"/outputs/{name}" for name in filenames]}


@torch.inference_mode()
def generate_inpaint(params: dict[str, object]) -> dict[str, list[str]]:
    #1. Load and create local method variables + ensure correct formatting from input dict
    initial_image = params["initial_image"]
    mask_image = params["mask_image"]
    strength = float(params["strength"])
    prompt = str(params["prompt"])
    negative_prompt = str(params["negative_prompt"])
    steps = int(params["steps"])
    guidance_scale = float(params["guidance_scale"])
    seed = params["seed"]
    model = params["model"]
    num_images = int(params["num_images"])
    scheduler = str(params["scheduler"])
    lora_adapters = params["lora_adapters"]

    #2. Check and set seed value
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)
    logger.info(
        "Flux Inpaint: model=%s, seed=%s, steps=%s, guidance_scale=%s, strength=%s, num_images=%s",
        model, base_seed, steps, guidance_scale, strength, num_images,
    )

    #3. Create batch_id and output directory
    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    #4. Load and create pipeline and scheduler
    pipe = load_inpaint_pipeline(model)
    pipe.scheduler = create_scheduler(scheduler, pipe)

    #5. Load lora into pipeline
    adapter_names: list[str] = []
    adapter_names, lora_coverage = apply_lora_adapters_with_validation(
        pipe,
        lora_adapters,
        expected_family="flux",
        validate=True,
    )
    report_path = write_lora_coverage_report(batch_output_dir, batch_id, lora_coverage)
    if report_path is not None:
        logger.info("LoRA coverage report saved to %s", report_path)

    #6. Create list of filenames
    filenames: list[str] = []
    
    #7. Render image one by one
    try:
        with GEN_LOCK:
            for i in range(num_images):
                # Set current seed
                current_seed = base_seed + i
                generator = torch.Generator(device="cpu").manual_seed(current_seed)

                # Render image
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    call_kwargs = dict(
                        prompt=prompt,
                        image=initial_image,
                        mask_image=mask_image,
                        strength=strength,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    )
                    if negative_prompt:
                        call_kwargs["negative_prompt"] = negative_prompt

                    image = pipe(**call_kwargs).images[0]

                # Define filenames & Create image_params metadata dict
                filename = batch_output_dir / f"{batch_id}_{current_seed}.png"
                image_width, image_height = initial_image.size
                image_params = dict(params)
                image_params.pop("initial_image", None)
                image_params.pop("mask_image", None)
                image_params.update({
                    "mode": "inpaint",
                    "pipeline": "flux",
                    "width": image_width,
                    "height": image_height,
                    "seed": current_seed,
                    "batch_id": batch_id,
                })
                pnginfo = build_png_metadata(image_params)
                image.save(filename, pnginfo=pnginfo)
                logger.info("Image %s saved to %s", i, filename.name)
                
                # Save filename to rendered image
                filenames.append(build_batch_output_relpath(batch_id, filename.name))

                del image
                cleanup_memory()
    finally:
        #8. Unload lora weights & clean memory
        if pipe is not None and adapter_names and hasattr(pipe, "unload_lora_weights"):
            try:
                pipe.unload_lora_weights()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                logger.warning("Failed to unload LoRA weights: %s", exc)

        release_pipeline(pipe, logger=logger)
        pipe = None

    # 9. Return output
    return {"images": [f"/outputs/{name}" for name in filenames]}
