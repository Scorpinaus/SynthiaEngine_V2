import logging
import threading
from pathlib import Path

import torch
from diffusers import QwenImageImg2ImgPipeline, QwenImageInpaintPipeline, QwenImagePipeline

from backend.logging_utils import configure_logging
from backend.model_registry import get_model_entry
from backend.pipeline_utils import (
    build_png_metadata,
    build_batch_output_relpath,
    get_batch_output_dir,
    make_batch_id,
    resolve_model_source,
)
from backend.schedulers import create_scheduler

GEN_LOCK = threading.Lock()
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PIPELINE_CACHE: dict[str, QwenImagePipeline] = {}
IMG2IMG_PIPELINE_CACHE: dict[str, QwenImageImg2ImgPipeline] = {}
INPAINT_PIPELINE_CACHE: dict[str, QwenImageInpaintPipeline] = {}

logger = logging.getLogger(__name__)
configure_logging()


def load_qwen_image_pipeline(model_name: str | None) -> QwenImagePipeline:
    entry = get_model_entry(model_name)

    pipe = PIPELINE_CACHE.get(entry.name)
    if pipe is not None:
        return pipe

    source = resolve_model_source(entry)
    logger.info("Qwen-Image model source: %s", source)

    if entry.model_type == "diffusers":
        pipe = QwenImagePipeline.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
        )
    elif entry.model_type == "single-file":
        pipe = QwenImagePipeline.from_single_file(
            source,
            torch_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing("max")
    if getattr(pipe, "vae", None) is not None:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    pipe.enable_sequential_cpu_offload()

    PIPELINE_CACHE[entry.name] = pipe
    return pipe


def load_qwen_image_img2img_pipeline(model_name: str | None) -> QwenImageImg2ImgPipeline:
    entry = get_model_entry(model_name)

    pipe = IMG2IMG_PIPELINE_CACHE.get(entry.name)
    if pipe is not None:
        return pipe

    source = resolve_model_source(entry)
    logger.info("Qwen-Image img2img model source: %s", source)

    if entry.model_type == "diffusers":
        pipe = QwenImageImg2ImgPipeline.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
        )
    elif entry.model_type == "single-file":
        pipe = QwenImageImg2ImgPipeline.from_single_file(
            source,
            torch_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing("max")
    if getattr(pipe, "vae", None) is not None:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    pipe.enable_sequential_cpu_offload()

    IMG2IMG_PIPELINE_CACHE[entry.name] = pipe
    return pipe


def load_qwen_image_inpaint_pipeline(model_name: str | None) -> QwenImageInpaintPipeline:
    entry = get_model_entry(model_name)

    pipe = INPAINT_PIPELINE_CACHE.get(entry.name)
    if pipe is not None:
        return pipe

    source = resolve_model_source(entry)
    logger.info("Qwen-Image inpaint model source: %s", source)

    if entry.model_type == "diffusers":
        pipe = QwenImageInpaintPipeline.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
        )
    elif entry.model_type == "single-file":
        pipe = QwenImageInpaintPipeline.from_single_file(
            source,
            torch_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing("max")
    if getattr(pipe, "vae", None) is not None:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    pipe.enable_sequential_cpu_offload()

    INPAINT_PIPELINE_CACHE[entry.name] = pipe
    return pipe


@torch.inference_mode()
def run_qwen_image_text2img(payload: dict[str, object]) -> dict[str, list[str]]:
    prompt = str(payload.get("prompt") or "")
    negative_prompt = str(payload.get("negative_prompt") or "").strip()
    steps = int(payload.get("steps", 30))
    true_cfg_scale = float(payload.get("true_cfg_scale", 4.0))
    guidance_scale = float(payload.get("guidance_scale", 7.5))
    width = int(payload.get("width", 1024))
    height = int(payload.get("height", 1024))
    seed = payload.get("seed")
    model = payload.get("model")
    num_images = int(payload.get("num_images", 1))
    scheduler = str(payload.get("scheduler") or "euler")

    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)

    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    pipe = load_qwen_image_pipeline(model)
    logger.info(
        "Qwen-Image Generate: model=%s seed=%s steps=%s true_cfg_scale=%s guidance_scale=%s size=%sx%s num_images=%s",
        model,
        base_seed,
        steps,
        true_cfg_scale,
        guidance_scale,
        width,
        height,
        num_images,
    )

    filenames: list[str] = []
    pipe.scheduler = create_scheduler(scheduler, pipe)

    with GEN_LOCK:
        for i in range(num_images):
            current_seed = base_seed + i
            generator = torch.Generator(device="cpu").manual_seed(current_seed)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                call_kwargs: dict[str, object] = {
                    "prompt": prompt,
                    "num_inference_steps": steps,
                    "true_cfg_scale": true_cfg_scale,
                    "guidance_scale": guidance_scale,
                    "width": width,
                    "height": height,
                    "generator": generator,
                }
                if negative_prompt:
                    call_kwargs["negative_prompt"] = negative_prompt

                image = pipe(**call_kwargs).images[0]

            filename = batch_output_dir / f"{batch_id}_{current_seed}.png"
            pnginfo = build_png_metadata(
                {
                    "mode": "txt2img",
                    "pipeline": "qwen-image",
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "steps": steps,
                    "true_cfg_scale": true_cfg_scale,
                    "guidance_scale": guidance_scale,
                    "width": width,
                    "height": height,
                    "seed": current_seed,
                    "model": model,
                    "scheduler": scheduler,
                    "batch_id": batch_id,
                }
            )
            image.save(filename, pnginfo=pnginfo)
            logger.info("Image %s saved to %s", i, filename.name)

            filenames.append(build_batch_output_relpath(batch_id, filename.name))

    return {"images": [f"/outputs/{name}" for name in filenames]}


@torch.inference_mode()
def run_qwen_image_img2img(
    initial_image,
    strength: float,
    prompt: str,
    negative_prompt: str,
    steps: int,
    true_cfg_scale: float,
    guidance_scale: float,
    width: int,
    height: int,
    seed: int | None,
    model: str | None,
    num_images: int,
    scheduler: str,
) -> dict[str, list[str]]:
    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)

    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    pipe = load_qwen_image_img2img_pipeline(model)
    logger.info(
        "Qwen-Image Img2Img: model=%s seed=%s steps=%s true_cfg_scale=%s guidance_scale=%s size=%sx%s strength=%s num_images=%s",
        model,
        base_seed,
        steps,
        true_cfg_scale,
        guidance_scale,
        width,
        height,
        strength,
        num_images,
    )

    filenames: list[str] = []
    pipe.scheduler = create_scheduler(scheduler, pipe)

    with GEN_LOCK:
        for i in range(num_images):
            current_seed = base_seed + i
            generator = torch.Generator(device="cpu").manual_seed(current_seed)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                call_kwargs: dict[str, object] = {
                    "prompt": prompt,
                    "image": initial_image,
                    "strength": strength,
                    "num_inference_steps": steps,
                    "true_cfg_scale": true_cfg_scale,
                    "guidance_scale": guidance_scale,
                    "width": width,
                    "height": height,
                    "generator": generator,
                }
                if negative_prompt:
                    call_kwargs["negative_prompt"] = negative_prompt

                image = pipe(**call_kwargs).images[0]

            filename = batch_output_dir / f"{batch_id}_{current_seed}.png"
            image_width, image_height = initial_image.size
            pnginfo = build_png_metadata(
                {
                    "mode": "img2img",
                    "pipeline": "qwen-image",
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "steps": steps,
                    "true_cfg_scale": true_cfg_scale,
                    "guidance_scale": guidance_scale,
                    "width": image_width,
                    "height": image_height,
                    "seed": current_seed,
                    "model": model,
                    "strength": strength,
                    "scheduler": scheduler,
                    "batch_id": batch_id,
                }
            )
            image.save(filename, pnginfo=pnginfo)
            logger.info("Image %s saved to %s", i, filename.name)

            filenames.append(build_batch_output_relpath(batch_id, filename.name))

    return {"images": [f"/outputs/{name}" for name in filenames]}


@torch.inference_mode()
def run_qwen_image_inpaint(
    initial_image,
    mask_image,
    strength: float,
    prompt: str,
    negative_prompt: str,
    steps: int,
    true_cfg_scale: float,
    guidance_scale: float,
    seed: int | None,
    model: str | None,
    num_images: int,
    scheduler: str,
) -> dict[str, list[str]]:
    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)

    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    pipe = load_qwen_image_inpaint_pipeline(model)
    width, height = initial_image.size
    logger.info(
        "Qwen-Image Inpaint: model=%s seed=%s steps=%s true_cfg_scale=%s guidance_scale=%s size=%sx%s strength=%s num_images=%s",
        model,
        base_seed,
        steps,
        true_cfg_scale,
        guidance_scale,
        width,
        height,
        strength,
        num_images,
    )

    filenames: list[str] = []
    pipe.scheduler = create_scheduler(scheduler, pipe)

    with GEN_LOCK:
        for i in range(num_images):
            current_seed = base_seed + i
            generator = torch.Generator(device="cpu").manual_seed(current_seed)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                call_kwargs: dict[str, object] = {
                    "prompt": prompt,
                    "image": initial_image,
                    "mask_image": mask_image,
                    "strength": strength,
                    "num_inference_steps": steps,
                    "true_cfg_scale": true_cfg_scale,
                    "guidance_scale": guidance_scale,
                    "generator": generator,
                }
                if negative_prompt:
                    call_kwargs["negative_prompt"] = negative_prompt

                image = pipe(**call_kwargs).images[0]

            filename = batch_output_dir / f"{batch_id}_{current_seed}.png"
            pnginfo = build_png_metadata(
                {
                    "mode": "inpaint",
                    "pipeline": "qwen-image",
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "steps": steps,
                    "true_cfg_scale": true_cfg_scale,
                    "guidance_scale": guidance_scale,
                    "width": width,
                    "height": height,
                    "seed": current_seed,
                    "model": model,
                    "strength": strength,
                    "scheduler": scheduler,
                    "batch_id": batch_id,
                }
            )
            image.save(filename, pnginfo=pnginfo)
            logger.info("Image %s saved to %s", i, filename.name)

            filenames.append(build_batch_output_relpath(batch_id, filename.name))

    return {"images": [f"/outputs/{name}" for name in filenames]}
