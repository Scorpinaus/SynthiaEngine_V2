import logging
import random
import time
from pathlib import Path

import torch
from PIL.PngImagePlugin import PngInfo
from diffusers import ZImagePipeline

from backend.model_registry import ModelRegistryEntry, get_model_entry

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PIPELINE_CACHE: dict[str, ZImagePipeline] = {}

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


def load_z_image_pipeline(model_name: str | None) -> ZImagePipeline:
    entry = get_model_entry(model_name)

    if entry.name in PIPELINE_CACHE:
        return PIPELINE_CACHE[entry.name]

    source = _resolve_model_source(entry)
    logger.info("Z-Image model source: %s", source)

    if entry.model_type == "diffusers":
        pipe = ZImagePipeline.from_pretrained(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        pipe = ZImagePipeline.from_single_file(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    pipe.to("cuda")
    PIPELINE_CACHE[entry.name] = pipe

    return pipe


@torch.inference_mode()
def run_z_image_text2img(payload: dict[str, object]) -> dict[str, list[str]]:
    prompt = str(payload.get("prompt") or "")
    negative_prompt = str(payload.get("negative_prompt") or "")
    steps = int(payload.get("steps", 20))
    guidance_scale = float(payload.get("guidance_scale", 7.5))
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

    pipe = load_z_image_pipeline(model)
    logger.info(
        "Z-Image Generate: model=%s seed=%s steps=%s guidance_scale=%s size=%sx%s num_images=%s",
        model,
        base_seed,
        steps,
        guidance_scale,
        width,
        height,
        num_images,
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
        ).images[0]

        filename = OUTPUT_DIR / f"{batch_id}_{current_seed}.png"
        pnginfo = _build_png_metadata({
            "mode": "txt2img",
            "pipeline": "z-image",
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

    return {"images": [f"/outputs/{name}" for name in filenames]}
