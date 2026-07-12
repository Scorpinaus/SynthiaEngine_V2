import logging
import json
import re
import subprocess
import sys
import tempfile
import torch
import threading
from typing import Any
from pathlib import Path

from diffusers import ZImageImg2ImgPipeline, ZImagePipeline, ZImageInpaintPipeline


from backend.config import OUTPUT_DIR
from backend.utilities.logging import configure_logging
from backend.lora.registry import get_lora_entry
from backend.registries.model import get_model_entry
from backend.utilities.pipeline import (
    build_fixed_step_timesteps,
    cleanup_memory,
    get_batch_output_dir,
    make_batch_id,
    release_pipeline,
    resolve_base_seed,
    resolve_model_source,
    save_generated_image,
)
from backend.utilities.schedulers import create_scheduler
from backend.z_image.subprocess_io import serialize_params_for_subprocess

_REPO_ROOT = Path(__file__).resolve().parents[2]
_Z_IMAGE_SUBPROCESS_SEMAPHORE = threading.Semaphore(1)

logger = logging.getLogger(__name__)
configure_logging()

_ADAPTER_NAME_SANITIZE_RE = re.compile(r"[^0-9A-Za-z_-]+")

"""
    Helper Functions
"""

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
    # Checks for empty or Null input
    if not raw_name:
        return ""
    # Checks for spaces, punctuation and special symbols - replaces them with _. Trims leading or trailing underscopes
    sanitized = _ADAPTER_NAME_SANITIZE_RE.sub("_", raw_name).strip("_")
    # Replaces multiple underscores with 1 underscore and returns final output
    return re.sub(r"_+", "_", sanitized)


def _build_adapter_name(lora_id: int, display_name: str | None,used_names: set[str],) -> str:
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


def _apply_lora_adapters(
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


def _cleanup_lora_adapters(pipe: Any | None, adapter_names: list[str]) -> None:
    if pipe is None or not adapter_names or not hasattr(pipe, "unload_lora_weights"):
        return
    try:
        pipe.unload_lora_weights()
    except Exception:
        logger.exception("Failed to unload Z-Image LoRA weights.")


def _run_z_image_subprocess(operation: str, params: dict[str, object]) -> dict[str, list[str]]:

    with tempfile.TemporaryDirectory(prefix="z_image_") as tmpdir:
        tmp_path = Path(tmpdir)
        input_path = tmp_path / "input.json"
        output_path = tmp_path / "output.json"
        payload = {
            "operation": operation,
            "params": serialize_params_for_subprocess(params, tmp_path),
        }
        input_path.write_text(
            json.dumps(payload, separators=(",", ": ")),
            encoding="utf-8",
        )

        cmd = [
            sys.executable,
            "-m",
            "backend.z_image.subprocess_runner",
            str(input_path),
            str(output_path),
        ]
        with _Z_IMAGE_SUBPROCESS_SEMAPHORE:
            completed = subprocess.run(cmd, cwd=str(_REPO_ROOT))

        if not output_path.exists():
            raise RuntimeError("Z-Image subprocess failed: No subprocess result was written.")

        result_payload = json.loads(output_path.read_text(encoding="utf-8"))
        if completed.returncode != 0 or not result_payload.get("ok"):
            detail = result_payload.get("error") or "Unknown subprocess failure."
            error_type = result_payload.get("error_type")
            if error_type:
                detail = f"{error_type}: {detail}"
            raise RuntimeError(f"Z-Image subprocess failed: {detail}")

        result = result_payload.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("images"), list):
            raise RuntimeError("Z-Image subprocess returned an invalid result.")
        return {"images": [str(path) for path in result["images"]]}


def generate_text2img(params: dict[str, object]) -> dict[str, list[str]]:
    return _run_z_image_subprocess("text2img", params)


def generate_img2img(params: dict[str, object]) -> dict[str, list[str]]:
    return _run_z_image_subprocess("img2img", params)


def generate_inpaint(params: dict[str, object]) -> dict[str, list[str]]:
    return _run_z_image_subprocess("inpaint", params)

"""
    Load Z-Image Pipelines
"""

def load_text2img_pipeline(model_name: str | None) -> ZImagePipeline:
    # 1. Check input model_name is valid and load valid path
    entry = get_model_entry(model_name)
    source = resolve_model_source(entry)
    logger.info("Z-Image model source: %s", source)

    #2. Check if diffusers multi-folder or single-file checkpoint
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
    
    #3. Set pipeline settings
    pipe.enable_sequential_cpu_offload()

    #Clean-up memory & Return ready pipeline
    cleanup_memory()    
    return pipe


def load_img2img_pipeline(model_name: str | None) -> ZImageImg2ImgPipeline:
    #1. Check input model_name is valid and load valid path
    entry = get_model_entry(model_name)
    source = resolve_model_source(entry)
    logger.info("Z-Image img2img model source: %s", source)

    #2. Check if diffusers multi-folder or single-file checkpoint
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

    #3. Set pipeline settings
    pipe.enable_sequential_cpu_offload()

    #4. Clean-up memory & Return ready pipeline
    cleanup_memory()
    return pipe


def load_inpaint_pipeline(model_name: str | None) -> Any:
    #1. Check input model_name is valid and load valid path
    entry = get_model_entry(model_name)
    source = resolve_model_source(entry)
    logger.info("Z-Image inpaint model source: %s", source)

    if ZImageInpaintPipeline is None:
        raise ValueError(
            "ZImageInpaintPipeline is unavailable in the installed diffusers package. "
            "Install a diffusers build with Z-Image inpaint support."
        )

    #2. Check if diffusers multi-folder or single-file checkpoint
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

    #3. Set pipeline settings:
    pipe.enable_sequential_cpu_offload()
    
    #4. Clean-up memory & Return ready pipeline
    cleanup_memory()
    return pipe


"""
    Image Rendering Functions
"""

@torch.inference_mode()
def generate_text2img_in_process(params: dict[str, object]) -> dict[str, list[str]]:
    #1. Load and create local method variables + ensure correct formatting from input dict
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

    base_seed = resolve_base_seed(seed)
    logger.info(
        "Z-Image Generate: model=%s seed=%s steps=%s guidance_scale=%s size=%sx%s num_images=%s",
        model, base_seed, steps, guidance_scale, width, height, num_images,
    )
    
    #3. Create batch_id and output directory
    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    #6. Create list of filenames
    filenames: list[str] = []
    pipe = None
    adapter_names: list[str] = []
    try:
        #4. Load and create pipeline and scheduler
        pipe = load_text2img_pipeline(model)
        pipe.scheduler = create_scheduler(scheduler, pipe)

        #5. Load lora into pipeline
        adapter_names = _apply_lora_adapters(pipe, lora_adapters)

        #7. Render image one by one
        for i in range(num_images):
            # Set current seed
            current_seed = base_seed + i
            generator = torch.Generator(device="cpu").manual_seed(current_seed)

            # Render image
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

            relpath = save_generated_image(
                image,
                batch_output_dir,
                batch_id,
                current_seed,
                params,
                mode="txt2img",
                pipeline="z-image",
            )
            logger.info("Image %s saved to %s", i, relpath)
            filenames.append(relpath)

            # Clean-up memory to prevent OOM
            del image
            cleanup_memory()
    finally:
        # 8. Unload lora weights & clean memory
        _cleanup_lora_adapters(pipe, adapter_names)
        release_pipeline(pipe, logger=logger)
        pipe = None

    # 9. Return output
    return {"images": [f"/outputs/{name}" for name in filenames]}


@torch.inference_mode()
def generate_img2img_in_process(params: dict[str, object]) -> dict[str, list[str]]:
    #1. Load and create local method variables + ensure correct formatting from input dict
    initial_image = params.get("initial_image")
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

    base_seed = resolve_base_seed(seed)
    logger.info(
        "Z-Image Img2Img: model=%s seed=%s steps=%s guidance_scale=%s size=%sx%s strength=%s num_images=%s",
        model, base_seed, steps, guidance_scale, width, height, strength,
        num_images,)

    #3. Create batch_id and output directory
    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    #6. Create list of filenames
    filenames: list[str] = []
    pipe = None
    adapter_names: list[str] = []
    
    #7. Render image one by one
    try:
        #4. Load and create pipeline and scheduler
        pipe = load_img2img_pipeline(model)
        pipe.scheduler = create_scheduler(scheduler, pipe)

        #5. Load lora into pipeline
        adapter_names = _apply_lora_adapters(pipe, lora_adapters)

        for i in range(num_images):
            # Set current seed
            current_seed = base_seed + i
            generator = torch.Generator(device="cpu").manual_seed(current_seed)

            # Render image
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

            relpath = save_generated_image(
                image,
                batch_output_dir,
                batch_id,
                current_seed,
                params,
                mode="img2img",
                pipeline="z-image",
            )
            logger.info("Image %s saved to %s", i, relpath)
            filenames.append(relpath)

            #Clean-up intermediate memory to prevent OOM
            del image
            cleanup_memory()
    finally:
        #8. Unload lora weights & clean memory
        _cleanup_lora_adapters(pipe, adapter_names)
        release_pipeline(pipe, logger=logger)
        pipe = None

    #9. Return output back to workflow calling method
    return {"images": [f"/outputs/{name}" for name in filenames]}


@torch.inference_mode()
def generate_inpaint_in_process(params: dict[str, object]) -> dict[str, list[str]]:
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
    width, height = initial_image.size

    base_seed = resolve_base_seed(seed)
    logger.info(
        "Z-Image Inpaint: model=%s seed=%s steps=%s guidance_scale=%s size=%sx%s strength=%s num_images=%s",
        model, base_seed, steps, guidance_scale, width, height, strength,
        num_images,
    )

    #3. Create batch_id and output directory
    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    #6. Create list of filenames
    filenames: list[str] = []
    pipe = None
    adapter_names: list[str] = []
    
    #7. Render image one by one
    try:
        #4. Load and create pipeline and scheduler
        pipe = load_inpaint_pipeline(model)
        pipe.scheduler = create_scheduler(scheduler, pipe)

        #5. Load lora into pipeline
        adapter_names = _apply_lora_adapters(pipe, lora_adapters)

        for i in range(num_images):
            # Set current seed
            current_seed = base_seed + i
            generator = torch.Generator(device="cpu").manual_seed(current_seed)

            # Render image
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

            relpath = save_generated_image(
                image,
                batch_output_dir,
                batch_id,
                current_seed,
                params,
                mode="inpaint",
                pipeline="z-image",
                remove_params=("initial_image", "mask_image"),
                size=(width, height),
            )
            logger.info("Image %s saved to %s", i, relpath)
            filenames.append(relpath)

            # Memory cleanup
            del image
            cleanup_memory()
    finally:
        #8. Unload loras and final memory clean-up
        _cleanup_lora_adapters(pipe, adapter_names)
        release_pipeline(pipe, logger=logger)
        pipe = None

    #9. Return output back to workflow calling method
    return {"images": [f"/outputs/{name}" for name in filenames]}
