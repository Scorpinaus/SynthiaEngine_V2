import logging
from pathlib import Path

import torch, gc
from diffusers import ZImageImg2ImgPipeline, ZImagePipeline

import threading

from backend.logging_utils import configure_logging
from backend.model_registry import get_model_entry
from backend.pipeline_utils import (
    build_fixed_step_timesteps,
    build_png_metadata,
    make_batch_id,
    resolve_model_source,
)
from backend.schedulers import create_scheduler

GEN_LOCK = threading.Lock()
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PIPELINE_CACHE: dict[str, ZImagePipeline] = {}
IMG2IMG_PIPELINE_CACHE: dict[str, ZImageImg2ImgPipeline] = {}

logger = logging.getLogger(__name__)
configure_logging()


# def _align_pad_token_dtype(pipe: ZImagePipeline | ZImageImg2ImgPipeline) -> None:
#     transformer = pipe.transformer
#     try:
#         first_param = next(transformer.parameters())
#     except StopIteration:
#         return
#     target_dtype = first_param.dtype
#     target_device = first_param.device
#     for attr in ("x_pad_token", "cap_pad_token"):
#         token = getattr(transformer, attr, None)
#         if token is None:
#             continue
#         if token.dtype != target_dtype or token.device != target_device:
#             token.data = token.data.to(dtype=target_dtype, device=target_device)

## Load Pipelines

def load_z_image_pipeline(model_name: str | None) -> ZImagePipeline:
    entry = get_model_entry(model_name)

    pipe = PIPELINE_CACHE.get(entry.name)
    if pipe is not None:
        return pipe

    source = resolve_model_source(entry)
    logger.info("Z-Image model source: %s", source)

    if entry.model_type == "diffusers":
        pipe = ZImagePipeline.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    elif entry.model_type == "single-file":
        pipe = ZImagePipeline.from_single_file(
            source,
            config="Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    dtypes = set(p.dtype for p in pipe.transformer.parameters())
    logger.info("Transformer dtypes: %s", dtypes)
    
    logger.info("Allocated GB: %s", torch.cuda.memory_allocated() / 1024**3)
    logger.info("Reserved GB: %s", torch.cuda.memory_reserved() / 1024**3)
    
    pipe.enable_sequential_cpu_offload()

    # Cleanup any transient allocations after load
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    PIPELINE_CACHE[entry.name] = pipe
    return pipe


def load_z_image_img2img_pipeline(model_name: str | None) -> ZImageImg2ImgPipeline:
    entry = get_model_entry(model_name)

    pipe = IMG2IMG_PIPELINE_CACHE.get(entry.name)
    if pipe is not None:
        return pipe

    source = resolve_model_source(entry)
    logger.info("Z-Image img2img model source: %s", source)

    if entry.model_type == "diffusers":
        pipe = ZImageImg2ImgPipeline.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    elif entry.model_type == "single-file":
        pipe = ZImageImg2ImgPipeline.from_single_file(
            source,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    dtypes = set(p.dtype for p in pipe.transformer.parameters())
    logger.info("Transformer dtypes: %s", dtypes)

    logger.info("Allocated GB: %s", torch.cuda.memory_allocated() / 1024**3)
    logger.info("Reserved GB: %s", torch.cuda.memory_reserved() / 1024**3)

    pipe.enable_sequential_cpu_offload()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    IMG2IMG_PIPELINE_CACHE[entry.name] = pipe
    return pipe

## Run and generate renders

@torch.inference_mode()
def run_z_image_text2img(payload: dict[str, object]) -> dict[str, list[str]]:
    
    prompt = str(payload.get("prompt") or "")
    negative_prompt = str(payload.get("negative_prompt") or "").strip()
    steps = int(payload.get("steps", 8))
    guidance_scale = float(payload.get("guidance_scale", 0.0))
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
    pipe.scheduler = create_scheduler(scheduler, pipe)

    with GEN_LOCK:
        for i in range(num_images):
            current_seed = base_seed + i
            
            generator = torch.Generator(device="cpu").manual_seed(current_seed)
            
            print("Allocated GB:", torch.cuda.memory_allocated()/1024**3)
            print("Reserved GB:", torch.cuda.memory_reserved()/1024**3)
            
            with torch.autocast("cuda", dtype=torch.bfloat16):
                call_kwargs = dict(
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator,
                )
                # Only include negative_prompt if user actually provided one
                if negative_prompt:
                    call_kwargs["negative_prompt"] = negative_prompt

                image = pipe(**call_kwargs).images[0]

            filename = OUTPUT_DIR / f"{batch_id}_{current_seed}.png"
            pnginfo = build_png_metadata({
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
                "scheduler": scheduler,
                "batch_id": batch_id,
            })
            image.save(filename, pnginfo=pnginfo)
            logger.info("Image %s saved to %s", i, filename.name)

            filenames.append(filename.name)
            
            # ✅ release per-image intermediates
            del image
            gc.collect()
            torch.cuda.empty_cache()

    return {"images": [f"/outputs/{name}" for name in filenames]}


@torch.inference_mode()
def run_z_image_img2img(
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
    scheduler: str,
) -> dict[str, list[str]]:
    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)

    batch_id = make_batch_id()

    pipe = load_z_image_img2img_pipeline(model)
    logger.info(
        "Z-Image Img2Img: model=%s seed=%s steps=%s guidance_scale=%s size=%sx%s strength=%s num_images=%s",
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
    pipe.scheduler = create_scheduler(scheduler, pipe)

    with GEN_LOCK:
        for i in range(num_images):
            current_seed = base_seed + i

            generator = torch.Generator(device="cpu").manual_seed(current_seed)

            print("Allocated GB:", torch.cuda.memory_allocated()/1024**3)
            print("Reserved GB:", torch.cuda.memory_reserved()/1024**3)

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
            pnginfo = build_png_metadata({
                "mode": "img2img",
                "pipeline": "z-image",
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "width": image_width,
                "height": image_height,
                "seed": current_seed,
                "model": model,
                "strength": strength,
                "scheduler": scheduler,
                "batch_id": batch_id,
            })
            image.save(filename, pnginfo=pnginfo)
            logger.info("Image %s saved to %s", i, filename.name)

            filenames.append(filename.name)

            del image
            gc.collect()
            torch.cuda.empty_cache()

    return {"images": [f"/outputs/{name}" for name in filenames]}
