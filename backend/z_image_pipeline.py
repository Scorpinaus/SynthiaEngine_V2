import logging
import re
from typing import Any

import torch
from diffusers import ZImageImg2ImgPipeline, ZImagePipeline
try:
    from diffusers import ZImageInpaintPipeline
except ImportError:  # pragma: no cover - depends on installed diffusers version
    ZImageInpaintPipeline = None

import threading

from backend.config import OUTPUT_DIR
from backend.logging_utils import configure_logging
from backend.lora_registry import get_lora_entry
from backend.model_registry import get_model_entry
from backend.pipeline_utils import (
    build_fixed_step_timesteps,
    build_png_metadata,
    build_batch_output_relpath,
    cleanup_memory,
    get_batch_output_dir,
    make_batch_id,
    resolve_model_source,
)
from backend.schedulers import create_scheduler

GEN_LOCK = threading.Lock()

logger = logging.getLogger(__name__)
configure_logging()
_ADAPTER_NAME_SANITIZE_RE = re.compile(r"[^0-9A-Za-z_-]+")


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


def _sanitize_adapter_fragment(raw_name: str | None) -> str:
    if not raw_name:
        return ""
    sanitized = _ADAPTER_NAME_SANITIZE_RE.sub("_", raw_name).strip("_")
    return re.sub(r"_+", "_", sanitized)


def _build_adapter_name(
    lora_id: int,
    display_name: str | None,
    used_names: set[str],
) -> str:
    fragment = _sanitize_adapter_fragment(display_name) or f"id_{lora_id}"
    base_name = f"lora_{fragment}"
    candidate = base_name
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate

    candidate = f"{base_name}_{lora_id}"
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate

    suffix = 2
    while True:
        candidate = f"{base_name}_{lora_id}_{suffix}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        suffix += 1


def _apply_z_image_lora_adapters(
    pipe: Any,
    lora_adapters: list[object] | None,
) -> list[str]:
    if not lora_adapters:
        return []

    adapter_names: list[str] = []
    adapter_weights: list[float] = []
    used_adapter_names: set[str] = set()

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
        if entry.lora_model_family.lower() != "z-image":
            raise ValueError(f"LoRA {entry.name} is not compatible with z-image.")

        adapter_name = _build_adapter_name(entry.lora_id, entry.name, used_adapter_names)
        adapter_weight = float(strength)
        pipe.load_lora_weights(entry.file_path, adapter_name=adapter_name)
        adapter_names.append(adapter_name)
        adapter_weights.append(adapter_weight)

        logger.info(
            "z-image lora_name=%s lora_id=%s lora_weight=%s",
            adapter_name,
            entry.lora_id,
            adapter_weight,
        )

    if hasattr(pipe, "set_adapters"):
        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

    return adapter_names

## Load Pipelines

def load_z_image_pipeline(model_name: str | None) -> ZImagePipeline:
    entry = get_model_entry(model_name)

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
    cleanup_memory()
    
    return pipe


def load_z_image_img2img_pipeline(model_name: str | None) -> ZImageImg2ImgPipeline:
    entry = get_model_entry(model_name)

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

    cleanup_memory()

    return pipe


def load_z_image_inpaint_pipeline(model_name: str | None) -> Any:
    entry = get_model_entry(model_name)

    source = resolve_model_source(entry)
    logger.info("Z-Image inpaint model source: %s", source)

    if ZImageInpaintPipeline is None:
        raise ValueError(
            "ZImageInpaintPipeline is unavailable in the installed diffusers package. "
            "Install a diffusers build with Z-Image inpaint support."
        )

    if entry.model_type == "diffusers":
        pipe = ZImageInpaintPipeline.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    elif entry.model_type == "single-file":
        pipe = ZImageInpaintPipeline.from_single_file(
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
    cleanup_memory()

    return pipe

## Run and generate renders

@torch.inference_mode()
def run_z_image_text2img(params: dict[str, object]) -> dict[str, list[str]]:
    prompt = str(params.get("prompt") or "")
    negative_prompt = str(params.get("negative_prompt") or "").strip()
    steps = int(params.get("steps", 8))
    guidance_scale = float(params.get("guidance_scale", 0.0))
    width = int(params.get("width", 1024))
    height = int(params.get("height", 1024))
    seed = params.get("seed")
    model = params.get("model")
    num_images = int(params.get("num_images", 1))
    scheduler = str(params.get("scheduler") or "euler")
    lora_adapters = params.get("lora_adapters")

    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)

    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

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
    adapter_names = _apply_z_image_lora_adapters(pipe, lora_adapters)

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

            filename = batch_output_dir / f"{batch_id}_{current_seed}.png"
            image_params = dict(params)
            image_params.update({
                "mode": "txt2img",
                "pipeline": "z-image",
                "seed": current_seed,
                "batch_id": batch_id,
            })
            pnginfo = build_png_metadata(image_params)
            image.save(filename, pnginfo=pnginfo)
            logger.info("Image %s saved to %s", i, filename.name)

            filenames.append(build_batch_output_relpath(batch_id, filename.name))
            
            # ✅ release per-image intermediates
            del image
            cleanup_memory()

    if adapter_names and hasattr(pipe, "unload_lora_weights"):
        pipe.unload_lora_weights()

    return {"images": [f"/outputs/{name}" for name in filenames]}


@torch.inference_mode()
def run_z_image_img2img(params: dict[str, object]) -> dict[str, list[str]]:
    initial_image = params.get("initial_image")
    if initial_image is None:
        raise ValueError("initial_image is required")
    strength = float(params.get("strength", 0.75))
    prompt = str(params.get("prompt") or "")
    negative_prompt = str(params.get("negative_prompt") or "").strip()
    steps = int(params.get("steps", 8))
    guidance_scale = float(params.get("guidance_scale", 0.0))
    width = int(params.get("width", 1024))
    height = int(params.get("height", 1024))
    seed = params.get("seed")
    model = params.get("model")
    num_images = int(params.get("num_images", 1))
    scheduler = str(params.get("scheduler") or "euler")
    lora_adapters = params.get("lora_adapters")

    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)

    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

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
    adapter_names = _apply_z_image_lora_adapters(pipe, lora_adapters)

    try:
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

                filename = batch_output_dir / f"{batch_id}_{current_seed}.png"
                image_params = dict(params)
                image_params.update(
                    {
                        "mode": "img2img",
                        "pipeline": "z-image",
                        "seed": current_seed,
                        "batch_id": batch_id,
                    }
                )
                pnginfo = build_png_metadata(image_params)
                image.save(filename, pnginfo=pnginfo)
                logger.info("Image %s saved to %s", i, filename.name)

                filenames.append(build_batch_output_relpath(batch_id, filename.name))

                del image
                cleanup_memory()
    finally:
        if adapter_names and hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()

    return {"images": [f"/outputs/{name}" for name in filenames]}


@torch.inference_mode()
def run_z_image_inpaint(params: dict[str, object]) -> dict[str, list[str]]:
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

    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)

    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    pipe = load_z_image_inpaint_pipeline(model)
    width, height = initial_image.size
    logger.info(
        "Z-Image Inpaint: model=%s seed=%s steps=%s guidance_scale=%s size=%sx%s strength=%s num_images=%s",
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
    adapter_names = _apply_z_image_lora_adapters(pipe, lora_adapters)

    try:
        with GEN_LOCK:
            for i in range(num_images):
                current_seed = base_seed + i
                generator = torch.Generator(device="cpu").manual_seed(current_seed)

                print("Allocated GB:", torch.cuda.memory_allocated() / 1024**3)
                print("Reserved GB:", torch.cuda.memory_reserved() / 1024**3)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    call_kwargs: dict[str, object] = {
                        "prompt": prompt,
                        "image": initial_image,
                        "mask_image": mask_image,
                        "strength": strength,
                        "num_inference_steps": steps,
                        "guidance_scale": guidance_scale,
                        "generator": generator,
                    }
                    if negative_prompt:
                        call_kwargs["negative_prompt"] = negative_prompt

                    image = pipe(**call_kwargs).images[0]

                filename = batch_output_dir / f"{batch_id}_{current_seed}.png"
                image_params = dict(params)
                image_params.pop("initial_image", None)
                image_params.pop("mask_image", None)
                image_params.update(
                    {
                        "mode": "inpaint",
                        "pipeline": "z-image",
                        "width": width,
                        "height": height,
                        "seed": current_seed,
                        "batch_id": batch_id,
                    }
                )
                pnginfo = build_png_metadata(image_params)
                image.save(filename, pnginfo=pnginfo)
                logger.info("Image %s saved to %s", i, filename.name)

                filenames.append(build_batch_output_relpath(batch_id, filename.name))

                del image
                cleanup_memory()
    finally:
        if adapter_names and hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()

    return {"images": [f"/outputs/{name}" for name in filenames]}
