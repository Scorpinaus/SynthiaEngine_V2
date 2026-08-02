import json
import logging
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import torch
from diffusers import QwenImageImg2ImgPipeline, QwenImageInpaintPipeline, QwenImagePipeline

from backend.config import OUTPUT_DIR
from backend.settings import REPOSITORY_ROOT
from backend.utilities.logging import configure_logging
from backend.lora.utils import apply_lora_adapters_with_validation, write_lora_coverage_report
from backend.registries.model import get_model_entry
from backend.utilities.pipeline import (
    cleanup_memory,
    get_batch_output_dir,
    make_batch_id,
    release_pipeline,
    resolve_base_seed,
    resolve_model_source,
    save_generated_image,
)
from backend.utilities.schedulers import create_scheduler
from backend.qwen_image.subprocess_io import serialize_params_for_subprocess

_QWEN_IMAGE_SUBPROCESS_SEMAPHORE = threading.Semaphore(1)

logger = logging.getLogger(__name__)
configure_logging()

""" Methods involving loading of pipelines"""

def load_text2img_pipeline(model_name: str | None) -> QwenImagePipeline:
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


def load_img2img_pipeline(model_name: str | None) -> QwenImageImg2ImgPipeline:
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


def load_inpaint_pipeline(model_name: str | None) -> QwenImageInpaintPipeline:
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


""" Methods involving generation using Qwen_Image related pipelines """

def _run_qwen_image_subprocess(operation: str, params: dict[str, object]) -> dict[str, list[str]]:


    with tempfile.TemporaryDirectory(prefix="qwen_image_") as tmpdir:
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
            "backend.qwen_image.subprocess_runner",
            str(input_path),
            str(output_path),
        ]
        with _QWEN_IMAGE_SUBPROCESS_SEMAPHORE:
            completed = subprocess.run(cmd, cwd=str(REPOSITORY_ROOT))

        if not output_path.exists():
            raise RuntimeError("Qwen-Image subprocess failed: No subprocess result was written.")

        result_payload = json.loads(output_path.read_text(encoding="utf-8"))
        if completed.returncode != 0 or not result_payload.get("ok"):
            detail = result_payload.get("error") or "Unknown subprocess failure."
            error_type = result_payload.get("error_type")
            if error_type:
                detail = f"{error_type}: {detail}"
            raise RuntimeError(f"Qwen-Image subprocess failed: {detail}")

        result = result_payload.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("images"), list):
            raise RuntimeError("Qwen-Image subprocess returned an invalid result.")
        return {"images": [str(path) for path in result["images"]]}


def generate_text2img(params: dict[str, object]) -> dict[str, list[str]]:
    return _run_qwen_image_subprocess("text2img", params)


def generate_img2img(params: dict[str, object]) -> dict[str, list[str]]:
    return _run_qwen_image_subprocess("img2img", params)


def generate_inpaint(params: dict[str, object]) -> dict[str, list[str]]:
    return _run_qwen_image_subprocess("inpaint", params)


@torch.inference_mode()
def generate_text2img_in_process(params: dict[str, object]) -> dict[str, list[str]]:
    prompt = str(params.get("prompt") or "")
    negative_prompt = str(params.get("negative_prompt") or "").strip()
    steps = int(params.get("steps", 30))
    true_cfg_scale = float(params.get("true_cfg_scale", 4.0))
    guidance_scale = float(params.get("guidance_scale", 7.5))
    width = int(params.get("width", 1024))
    height = int(params.get("height", 1024))
    seed = params.get("seed")
    model = params.get("model")
    num_images = int(params.get("num_images", 1))
    scheduler = str(params.get("scheduler") or "euler")
    lora_adapters = params.get("lora_adapters")

    logger.info("seed=%s", seed)
    base_seed = resolve_base_seed(seed)

    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

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
    pipe = None
    adapter_names: list[str] = []
    try:
        pipe = load_text2img_pipeline(model)
        pipe.scheduler = create_scheduler(scheduler, pipe)
        adapter_names, lora_coverage = apply_lora_adapters_with_validation(
            pipe,
            lora_adapters,
            expected_family="qwen-image",
            validate=True,
        )
        report_path = write_lora_coverage_report(batch_output_dir, batch_id, lora_coverage)
        if report_path is not None:
            logger.info("LoRA coverage report saved to %s", report_path)

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

            relpath = save_generated_image(
                image, batch_output_dir, batch_id, current_seed, params,
                mode="txt2img", pipeline="qwen-image",
            )
            logger.info("Image %s saved to %s", i, relpath)
            filenames.append(relpath)

            del image
            cleanup_memory()
    finally:
        if pipe is not None and adapter_names and hasattr(pipe, "unload_lora_weights"):
            try:
                pipe.unload_lora_weights()
            except Exception:
                logger.exception("Failed to unload Qwen-Image LoRA weights.")
        release_pipeline(pipe, logger=logger)
        pipe = None

    return {"images": [f"/outputs/{name}" for name in filenames]}


@torch.inference_mode()
def generate_img2img_in_process(params: dict[str, object]) -> dict[str, list[str]]:
    initial_image = params.get("initial_image")
    if initial_image is None:
        raise ValueError("initial_image is required")
    strength = float(params.get("strength", 0.75))
    prompt = str(params.get("prompt") or "")
    negative_prompt = str(params.get("negative_prompt") or "").strip()
    steps = int(params.get("steps", 30))
    true_cfg_scale = float(params.get("true_cfg_scale", 4.0))
    guidance_scale = float(params.get("guidance_scale", 7.5))
    width = int(params.get("width", 1024))
    height = int(params.get("height", 1024))
    seed = params.get("seed")
    model = params.get("model")
    num_images = int(params.get("num_images", 1))
    scheduler = str(params.get("scheduler") or "euler")
    lora_adapters = params.get("lora_adapters")

    logger.info("seed=%s", seed)
    base_seed = resolve_base_seed(seed)

    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

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
    pipe = None
    adapter_names: list[str] = []
    try:
        pipe = load_img2img_pipeline(model)
        pipe.scheduler = create_scheduler(scheduler, pipe)
        adapter_names, lora_coverage = apply_lora_adapters_with_validation(
            pipe,
            lora_adapters,
            expected_family="qwen-image",
            validate=True,
        )
        report_path = write_lora_coverage_report(batch_output_dir, batch_id, lora_coverage)
        if report_path is not None:
            logger.info("LoRA coverage report saved to %s", report_path)

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

            image_width, image_height = initial_image.size
            relpath = save_generated_image(
                image, batch_output_dir, batch_id, current_seed, params,
                mode="img2img", pipeline="qwen-image",
                remove_params=("initial_image",),
                size=(image_width, image_height),
            )
            logger.info("Image %s saved to %s", i, relpath)
            filenames.append(relpath)

            del image
            cleanup_memory()
    finally:
        if pipe is not None and adapter_names and hasattr(pipe, "unload_lora_weights"):
            try:
                pipe.unload_lora_weights()
            except Exception:
                logger.exception("Failed to unload Qwen-Image LoRA weights.")
        release_pipeline(pipe, logger=logger)
        pipe = None

    return {"images": [f"/outputs/{name}" for name in filenames]}


@torch.inference_mode()
def generate_inpaint_in_process(params: dict[str, object]) -> dict[str, list[str]]:
    initial_image = params.get("initial_image")
    if initial_image is None:
        raise ValueError("initial_image is required")
    mask_image = params.get("mask_image")
    if mask_image is None:
        raise ValueError("mask_image is required")
    strength = float(params.get("strength", 0.5))
    prompt = str(params.get("prompt") or "")
    negative_prompt = str(params.get("negative_prompt") or "").strip()
    steps = int(params.get("steps", 30))
    true_cfg_scale = float(params.get("true_cfg_scale", 4.0))
    guidance_scale = float(params.get("guidance_scale", 7.5))
    seed = params.get("seed")
    model = params.get("model")
    num_images = int(params.get("num_images", 1))
    scheduler = str(params.get("scheduler") or "euler")
    lora_adapters = params.get("lora_adapters")

    logger.info("seed=%s", seed)
    base_seed = resolve_base_seed(seed)

    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

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
    pipe = None
    adapter_names: list[str] = []
    try:
        pipe = load_inpaint_pipeline(model)
        pipe.scheduler = create_scheduler(scheduler, pipe)
        adapter_names, lora_coverage = apply_lora_adapters_with_validation(
            pipe,
            lora_adapters,
            expected_family="qwen-image",
            validate=True,
        )
        report_path = write_lora_coverage_report(batch_output_dir, batch_id, lora_coverage)
        if report_path is not None:
            logger.info("LoRA coverage report saved to %s", report_path)

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

            relpath = save_generated_image(
                image, batch_output_dir, batch_id, current_seed, params,
                mode="inpaint", pipeline="qwen-image",
                remove_params=("initial_image", "mask_image"),
                size=(width, height),
            )
            logger.info("Image %s saved to %s", i, relpath)
            filenames.append(relpath)

            del image
            cleanup_memory()
    finally:
        if pipe is not None and adapter_names and hasattr(pipe, "unload_lora_weights"):
            try:
                pipe.unload_lora_weights()
            except Exception:
                logger.exception("Failed to unload Qwen-Image LoRA weights.")
        release_pipeline(pipe, logger=logger)
        pipe = None

    return {"images": [f"/outputs/{name}" for name in filenames]}
