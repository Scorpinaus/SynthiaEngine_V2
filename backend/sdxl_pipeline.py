import logging
import random
import time
from pathlib import Path

import torch
from PIL.PngImagePlugin import PngInfo
from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLInpaintPipeline,
)

from backend.model_registry import ModelRegistryEntry, get_model_entry

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PIPELINE_CACHE: dict[str, StableDiffusionXLPipeline] = {}
IMG2IMG_PIPELINE_CACHE: dict[str, StableDiffusionXLImg2ImgPipeline] = {}
INPAINT_PIPELINE_CACHE: dict[str, StableDiffusionXLInpaintPipeline] = {}

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


def load_sdxl_pipeline(model_name: str | None) -> StableDiffusionXLPipeline:
    entry = get_model_entry(model_name)

    if entry.name in PIPELINE_CACHE:
        return PIPELINE_CACHE[entry.name]

    source = _resolve_model_source(entry)
    logger.info("SDXL model source: %s", source)

    if entry.model_type == "diffusers":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        pipe = StableDiffusionXLPipeline.from_single_file(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    pipe.to("cuda")
    PIPELINE_CACHE[entry.name] = pipe

    return pipe


def load_sdxl_img2img_pipeline(model_name: str | None) -> StableDiffusionXLImg2ImgPipeline:
    entry = get_model_entry(model_name)

    if entry.name in IMG2IMG_PIPELINE_CACHE:
        return IMG2IMG_PIPELINE_CACHE[entry.name]

    source = _resolve_model_source(entry)
    logger.info("SDXL img2img model source: %s", source)

    if entry.model_type == "diffusers":
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    pipe.to("cuda")
    IMG2IMG_PIPELINE_CACHE[entry.name] = pipe

    return pipe


def load_sdxl_inpaint_pipeline(model_name: str | None) -> StableDiffusionXLInpaintPipeline:
    entry = get_model_entry(model_name)

    if entry.name in INPAINT_PIPELINE_CACHE:
        return INPAINT_PIPELINE_CACHE[entry.name]

    source = _resolve_model_source(entry)
    logger.info("SDXL inpaint model source: %s", source)

    if entry.model_type == "diffusers":
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        pipe = StableDiffusionXLInpaintPipeline.from_single_file(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    pipe.to("cuda")
    INPAINT_PIPELINE_CACHE[entry.name] = pipe

    return pipe


@torch.inference_mode()
def run_sdxl_text2img(payload: dict[str, object]) -> dict[str, list[str]]:
    prompt = str(payload.get("prompt") or "")
    negative_prompt = str(payload.get("negative_prompt") or "")
    steps = int(payload.get("steps", 20))
    guidance_scale = float(payload.get("guidance_scale", 7.5))
    width = int(payload.get("width", 1024))
    height = int(payload.get("height", 1024))
    seed = payload.get("seed")
    model = payload.get("model")
    num_images = int(payload.get("num_images", 1))
    clip_skip = int(payload.get("clip_skip", 1))

    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)

    batch_id = _make_batch_id()

    pipe = load_sdxl_pipeline(model)
    logger.info(
        "SDXL Generate: model=%s seed=%s steps=%s guidance_scale=%s size=%sx%s num_images=%s",
        model, base_seed, steps, guidance_scale, width, height, num_images,
    )

    filenames: list[str] = []

    for i in range(num_images):
        current_seed = base_seed + i
        generator = torch.Generator(device="cuda").manual_seed(current_seed)

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            clip_skip=clip_skip,
        ).images[0]

        filename = OUTPUT_DIR / f"{batch_id}_{current_seed}.png"
        pnginfo = _build_png_metadata({
            "mode": "txt2img",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "seed": current_seed,
            "model": model,
            "clip_skip": clip_skip,
            "batch_id": batch_id,
        })
        image.save(filename, pnginfo=pnginfo)
        logger.info("Image %s saved to %s", i, filename.name)

        filenames.append(filename.name)

    return {"images": [f"/outputs/{name}" for name in filenames]}


@torch.inference_mode()
def run_sdxl_img2img(
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
    clip_skip: int,
) -> dict[str, list[str]]:
    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)

    batch_id = _make_batch_id()

    pipe = load_sdxl_img2img_pipeline(model)
    logger.info(
        "SDXL Img2Img: model=%s seed=%s steps=%s guidance_scale=%s size=%sx%s strength=%s num_images=%s",
        model, base_seed, steps, guidance_scale, width, height, strength, num_images,
    )

    filenames: list[str] = []

    for i in range(num_images):
        current_seed = base_seed + i
        generator = torch.Generator(device="cuda").manual_seed(current_seed)

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=initial_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            clip_skip=clip_skip,
        ).images[0]

        filename = OUTPUT_DIR / f"{batch_id}_{current_seed}.png"
        image_width, image_height = initial_image.size
        pnginfo = _build_png_metadata({
            "mode": "img2img",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "width": image_width,
            "height": image_height,
            "seed": current_seed,
            "model": model,
            "strength": strength,
            "clip_skip": clip_skip,
            "batch_id": batch_id,
        })
        image.save(filename, pnginfo=pnginfo)
        logger.info("Image %s saved to %s", i, filename.name)

        filenames.append(filename.name)

    return {"images": [f"/outputs/{name}" for name in filenames]}


@torch.inference_mode()
def run_sdxl_inpaint(
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
    padding_mask_crop: int,
    clip_skip: int,
) -> dict[str, list[str]]:
    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)

    batch_id = _make_batch_id()

    pipe = load_sdxl_inpaint_pipeline(model)
    width, height = initial_image.size
    logger.info(
        "SDXL Inpaint: model=%s seed=%s steps=%s guidance_scale=%s size=%sx%s strength=%s num_images=%s padding_mask_crop=%s",
        model, base_seed, steps, guidance_scale, width, height, strength, num_images, padding_mask_crop,
    )

    filenames: list[str] = []

    for i in range(num_images):
        current_seed = base_seed + i
        generator = torch.Generator(device="cuda").manual_seed(current_seed)

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=initial_image,
            mask_image=mask_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            padding_mask_crop=padding_mask_crop,
            clip_skip=clip_skip,
        ).images[0]

        filename = OUTPUT_DIR / f"{batch_id}_{current_seed}.png"
        pnginfo = _build_png_metadata({
            "mode": "inpaint",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "seed": current_seed,
            "model": model,
            "strength": strength,
            "padding_mask_crop": padding_mask_crop,
            "clip_skip": clip_skip,
            "batch_id": batch_id,
        })
        image.save(filename, pnginfo=pnginfo)
        logger.info("Image %s saved to %s", i, filename.name)

        filenames.append(filename.name)

    return {"images": [f"/outputs/{name}" for name in filenames]}
