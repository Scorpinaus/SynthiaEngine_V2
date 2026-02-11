import logging
import re
import threading

import torch
from diffusers import QwenImageImg2ImgPipeline, QwenImageInpaintPipeline, QwenImagePipeline

from backend.config import OUTPUT_DIR
from backend.logging_utils import configure_logging
from backend.lora_registry import get_lora_entry
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

logger = logging.getLogger(__name__)
configure_logging()
_ADAPTER_NAME_SANITIZE_RE = re.compile(r"[^0-9A-Za-z_-]+")


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


def _apply_qwen_image_lora_adapters(
    pipe: QwenImagePipeline,
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
        if entry.lora_model_family.lower() != "qwen-image":
            raise ValueError(f"LoRA {entry.name} is not compatible with qwen-image.")

        adapter_name = _build_adapter_name(entry.lora_id, entry.name, used_adapter_names)
        adapter_weight = float(strength)
        pipe.load_lora_weights(entry.file_path, adapter_name=adapter_name)
        adapter_names.append(adapter_name)
        adapter_weights.append(adapter_weight)

        logger.info(
            "qwen-image lora_name=%s lora_id=%s lora_weight=%s",
            adapter_name,
            entry.lora_id,
            adapter_weight,
        )

    if hasattr(pipe, "set_adapters"):
        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

    return adapter_names


def load_qwen_image_pipeline(model_name: str | None) -> QwenImagePipeline:
    entry = get_model_entry(model_name)

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

    return pipe


def load_qwen_image_img2img_pipeline(model_name: str | None) -> QwenImageImg2ImgPipeline:
    entry = get_model_entry(model_name)

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

    return pipe


def load_qwen_image_inpaint_pipeline(model_name: str | None) -> QwenImageInpaintPipeline:
    entry = get_model_entry(model_name)

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
    lora_adapters = payload.get("lora_adapters")

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
    adapter_names = _apply_qwen_image_lora_adapters(pipe, lora_adapters)

    try:
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
    finally:
        if adapter_names and hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()

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
    lora_adapters: list[object] | None = None,
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
    adapter_names = _apply_qwen_image_lora_adapters(pipe, lora_adapters)

    try:
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
    finally:
        if adapter_names and hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()

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
    lora_adapters: list[object] | None = None,
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
    adapter_names = _apply_qwen_image_lora_adapters(pipe, lora_adapters)

    try:
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
    finally:
        if adapter_names and hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()

    return {"images": [f"/outputs/{name}" for name in filenames]}
