import gc
import logging
import random
import threading
import time
from pathlib import Path

import torch
from PIL.PngImagePlugin import PngInfo
from diffusers import FluxImg2ImgPipeline, FluxInpaintPipeline, FluxPipeline

from backend.model_registry import ModelRegistryEntry, get_model_entry

GEN_LOCK = threading.Lock()
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PIPELINE_CACHE: dict[str, FluxPipeline] = {}
IMG2IMG_PIPELINE_CACHE: dict[str, FluxImg2ImgPipeline] = {}
INPAINT_PIPELINE_CACHE: dict[str, FluxInpaintPipeline] = {}

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def _resolve_model_source(entry: ModelRegistryEntry) -> str:
    if entry.location_type == "hub":
        return entry.link

    return str(Path(entry.link).expanduser())


def _make_batch_id() -> str:
    return f"b{int(time.time())}_{random.randint(1000, 9999)}"


def _build_png_metadata(metadata: dict[str, object]) -> PngInfo:
    info = PngInfo()
    for key, value in metadata.items():
        if value is None:
            continue
        info.add_text(key, str(value))
    return info


def load_flux_pipeline(model_name: str | None) -> FluxPipeline:
    entry = get_model_entry(model_name)

    pipe = PIPELINE_CACHE.get(entry.name)
    if pipe is not None:
        return pipe

    source = _resolve_model_source(entry)
    logger.info("Flux model source: %s", source)

    if entry.model_type == "diffusers":
        pipe = FluxPipeline.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
        )
    elif entry.model_type == "single-file":
        pipe = FluxPipeline.from_single_file(
            source,
            torch_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    pipe.enable_attention_slicing("max")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.enable_sequential_cpu_offload()

    PIPELINE_CACHE[entry.name] = pipe
    return pipe


def load_flux_img2img_pipeline(model_name: str | None) -> FluxImg2ImgPipeline:
    entry = get_model_entry(model_name)

    pipe = IMG2IMG_PIPELINE_CACHE.get(entry.name)
    if pipe is not None:
        return pipe

    source = _resolve_model_source(entry)
    logger.info("Flux img2img model source: %s", source)

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

    pipe.enable_model_cpu_offload()

    IMG2IMG_PIPELINE_CACHE[entry.name] = pipe
    return pipe


def load_flux_inpaint_pipeline(model_name: str | None) -> FluxInpaintPipeline:
    entry = get_model_entry(model_name)

    pipe = INPAINT_PIPELINE_CACHE.get(entry.name)
    if pipe is not None:
        return pipe

    source = _resolve_model_source(entry)
    logger.info("Flux inpaint model source: %s", source)

    if entry.model_type == "diffusers":
        pipe = FluxInpaintPipeline.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
        )
    elif entry.model_type == "single-file":
        pipe = FluxInpaintPipeline.from_single_file(
            source,
            torch_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    pipe.enable_model_cpu_offload()

    INPAINT_PIPELINE_CACHE[entry.name] = pipe
    return pipe


@torch.inference_mode()
def run_flux_text2img(payload: dict[str, object]) -> dict[str, list[str]]:
    prompt = str(payload.get("prompt") or "")
    negative_prompt = str(payload.get("negative_prompt") or "").strip()
    steps = int(payload.get("steps", 20))
    guidance_scale = float(payload.get("guidance_scale", 0.0))
    width = int(payload.get("width", 1024))
    height = int(payload.get("height", 1024))
    seed = payload.get("seed")
    model = payload.get("model")
    num_images = int(payload.get("num_images", 1))

    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)

    batch_id = _make_batch_id()

    pipe = load_flux_pipeline(model)
    logger.info(
        "Flux Generate: model=%s seed=%s steps=%s guidance_scale=%s size=%sx%s num_images=%s",
        model,
        base_seed,
        steps,
        guidance_scale,
        width,
        height,
        num_images,
    )

    filenames: list[str] = []

    with GEN_LOCK:
        for i in range(num_images):
            current_seed = base_seed + i
            generator = torch.Generator(device="cpu").manual_seed(current_seed)

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

            filename = OUTPUT_DIR / f"{batch_id}_{current_seed}.png"
            pnginfo = _build_png_metadata({
                "mode": "txt2img",
                "pipeline": "flux",
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
                "seed": current_seed,
                "model": model,
                "batch_id": batch_id,
            })
            image.save(filename, pnginfo=pnginfo)
            logger.info("Image %s saved to %s", i, filename.name)

            filenames.append(filename.name)

            del image
            gc.collect()
            torch.cuda.empty_cache()

    return {"images": [f"/outputs/{name}" for name in filenames]}


@torch.inference_mode()
def run_flux_img2img(
    initial_image,
    strength: float,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    seed: int | None,
    model: str | None,
    num_images: int,
) -> dict[str, list[str]]:
    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)

    batch_id = _make_batch_id()

    pipe = load_flux_img2img_pipeline(model)
    logger.info(
        "Flux Img2Img: model=%s seed=%s steps=%s guidance_scale=%s size=%sx%s strength=%s num_images=%s",
        model,
        base_seed,
        steps,
        guidance_scale,
        width,
        height,
        strength,
        num_images,
    )

    filenames: list[str] = []

    with GEN_LOCK:
        for i in range(num_images):
            current_seed = base_seed + i
            generator = torch.Generator(device="cpu").manual_seed(current_seed)

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

            filename = OUTPUT_DIR / f"{batch_id}_{current_seed}.png"
            image_width, image_height = initial_image.size
            pnginfo = _build_png_metadata({
                "mode": "img2img",
                "pipeline": "flux",
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "width": image_width,
                "height": image_height,
                "seed": current_seed,
                "model": model,
                "strength": strength,
                "batch_id": batch_id,
            })
            image.save(filename, pnginfo=pnginfo)
            logger.info("Image %s saved to %s", i, filename.name)

            filenames.append(filename.name)

            del image
            gc.collect()
            torch.cuda.empty_cache()

    return {"images": [f"/outputs/{name}" for name in filenames]}


@torch.inference_mode()
def run_flux_inpaint(
    initial_image,
    mask_image,
    strength: float,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    seed: int | None,
    model: str | None,
    num_images: int,
) -> dict[str, list[str]]:
    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)

    batch_id = _make_batch_id()

    pipe = load_flux_inpaint_pipeline(model)
    logger.info(
        "Flux Inpaint: model=%s seed=%s steps=%s guidance_scale=%s strength=%s num_images=%s",
        model,
        base_seed,
        steps,
        guidance_scale,
        strength,
        num_images,
    )

    filenames: list[str] = []

    with GEN_LOCK:
        for i in range(num_images):
            current_seed = base_seed + i
            generator = torch.Generator(device="cpu").manual_seed(current_seed)

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

            filename = OUTPUT_DIR / f"{batch_id}_{current_seed}.png"
            image_width, image_height = initial_image.size
            pnginfo = _build_png_metadata({
                "mode": "inpaint",
                "pipeline": "flux",
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "width": image_width,
                "height": image_height,
                "seed": current_seed,
                "model": model,
                "strength": strength,
                "batch_id": batch_id,
            })
            image.save(filename, pnginfo=pnginfo)
            logger.info("Image %s saved to %s", i, filename.name)

            filenames.append(filename.name)

            del image
            gc.collect()
            torch.cuda.empty_cache()

    return {"images": [f"/outputs/{name}" for name in filenames]}
