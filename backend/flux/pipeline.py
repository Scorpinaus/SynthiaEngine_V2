"""
Docstring for backend.flux.pipeline
"""
import logging
import json
import os
import subprocess
import sys
import tempfile
import threading
import inspect
import time
from pathlib import Path
from typing import Any

import torch

try:
    from diffusers import FluxFillPipeline
except ImportError:  # pragma: no cover - depends on installed diffusers version
    FluxFillPipeline = None  # type: ignore[assignment]

from backend.config import OUTPUT_DIR
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
from backend.flux.subprocess_io import serialize_params_for_subprocess
from backend.utilities.pipeline_cache import PipelineCache

"""
    Static Variables and Logging
"""
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FLUX_SUBPROCESS_SEMAPHORE = threading.Semaphore(1)
_FLUX_PIPELINE_CACHE = PipelineCache.from_env()

logger = logging.getLogger(__name__)
configure_logging()


def _should_use_flux_fill_pipeline(model_name: str | None, source: str, version: str) -> bool:
    joined = " ".join([model_name or "", source or "", version or ""]).lower()
    return "flux" in joined and "fill" in joined


def _flux_low_memory_mode() -> str:
    return os.getenv("SYNTHA_FLUX_OFFLOAD", "auto")


def _configure_flux_pipeline(pipe: Any) -> Any:
    from custom_pipelines.Flux.memory import enable_low_memory_flux

    mode = _flux_low_memory_mode()
    if hasattr(pipe, "enable_low_memory_flux"):
        applied_mode = pipe.enable_low_memory_flux(mode=mode)
    else:
        applied_mode = enable_low_memory_flux(pipe, mode=mode)
    logger.info("Flux low-memory mode requested=%s applied=%s", mode, applied_mode)
    return pipe


def _run_flux_subprocess(operation: str, params: dict[str, object]) -> dict[str, list[str]]:

    with tempfile.TemporaryDirectory(prefix="flux_") as tmpdir:
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
            "backend.flux.subprocess_runner",
            str(input_path),
            str(output_path),
        ]
        with _FLUX_SUBPROCESS_SEMAPHORE:
            completed = subprocess.run(cmd, cwd=str(_REPO_ROOT))

        if not output_path.exists():
            raise RuntimeError("Flux subprocess failed: No subprocess result was written.")

        result_payload = json.loads(output_path.read_text(encoding="utf-8"))
        if completed.returncode != 0 or not result_payload.get("ok"):
            detail = result_payload.get("error") or "Unknown subprocess failure."
            error_type = result_payload.get("error_type")
            if error_type:
                detail = f"{error_type}: {detail}"
            raise RuntimeError(f"Flux subprocess failed: {detail}")

        result = result_payload.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("images"), list):
            raise RuntimeError("Flux subprocess returned an invalid result.")
        normalized: dict[str, Any] = {"images": [str(path) for path in result["images"]]}
        if isinstance(result.get("runtime_profile"), dict):
            normalized["runtime_profile"] = result["runtime_profile"]
        return normalized


def generate_text2img(params: dict[str, object]) -> dict[str, list[str]]:
    if _FLUX_PIPELINE_CACHE.enabled:
        return generate_text2img_in_process(params)
    return _run_flux_subprocess("text2img", params)


def generate_img2img(params: dict[str, object]) -> dict[str, list[str]]:
    if _FLUX_PIPELINE_CACHE.enabled:
        return generate_img2img_in_process(params)
    return _run_flux_subprocess("img2img", params)


def generate_inpaint(params: dict[str, object]) -> dict[str, list[str]]:
    if _FLUX_PIPELINE_CACHE.enabled:
        return generate_inpaint_in_process(params)
    return _run_flux_subprocess("inpaint", params)


def _acquire_flux_pipeline(operation: str, model: object, loader):
    estimated_mb = int(os.getenv("SYNTHA_FLUX_PIPELINE_ESTIMATED_MB", "12000"))
    key = ("flux", operation, str(model or "default"), _flux_low_memory_mode())
    hits_before = _FLUX_PIPELINE_CACHE.hits
    pipe, cache_owned = _FLUX_PIPELINE_CACHE.acquire(
        key,
        loader,
        cost_mb=estimated_mb,
        release=lambda value: release_pipeline(value, logger=logger),
    )
    logger.info("Flux pipeline cache operation=%s owned=%s stats=%s", operation, cache_owned, _FLUX_PIPELINE_CACHE.stats())
    return pipe, cache_owned, _FLUX_PIPELINE_CACHE.hits > hits_before


def _timed_pipeline_call(pipe: Any, call_kwargs: dict[str, object]):
    started = time.perf_counter()
    last_denoise_step: list[float | None] = [None]
    kwargs = dict(call_kwargs)
    try:
        supports_step_callback = "callback_on_step_end" in inspect.signature(pipe.__call__).parameters
    except (TypeError, ValueError):
        supports_step_callback = False
    if supports_step_callback and "callback_on_step_end" not in kwargs:
        def _on_step_end(_pipe, _step, _timestep, callback_kwargs):
            last_denoise_step[0] = time.perf_counter()
            return callback_kwargs
        kwargs["callback_on_step_end"] = _on_step_end
    output = pipe(**kwargs)
    finished = time.perf_counter()
    final_step = last_denoise_step[0]
    return output, {
        "inference_seconds": round(finished - started, 6),
        "denoise_seconds": round(final_step - started, 6) if final_step else None,
        "decode_seconds": round(finished - final_step, 6) if final_step else None,
    }


"""
    Methods for loading flux pipelines
"""

def load_text2img_pipeline(model_name: str | None) -> Any:
    from custom_pipelines.Flux.pipeline_flux import FluxPipeline as CustomFluxPipeline

    # 1. Check input model_name is valid and load valid path
    entry = get_model_entry(model_name)
    source = resolve_model_source(entry)
    logger.info("Flux model source: %s", source)

    # 2. Check if diffusers multi-folder or single-file checkpoint
    if entry.model_type == "diffusers":
        pipe = CustomFluxPipeline.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    elif entry.model_type == "single-file":
        pipe = CustomFluxPipeline.from_single_file(
            source,
            torch_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")
    return _configure_flux_pipeline(pipe)

def load_img2img_pipeline(model_name: str | None) -> Any:
    from custom_pipelines.Flux.pipeline_flux_img2img import FluxImg2ImgPipeline as CustomFluxImg2ImgPipeline

    
    #1. Check input model_name is valid and load valid path
    entry = get_model_entry(model_name)
    source = resolve_model_source(entry)
    logger.info("Flux img2img model source: %s", source)

    #2. Check if diffusers multi-folder or single-file checkpoint
    if entry.model_type == "diffusers":
        pipe = CustomFluxImg2ImgPipeline.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    elif entry.model_type == "single-file":
        pipe = CustomFluxImg2ImgPipeline.from_single_file(
            source,
            torch_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    return _configure_flux_pipeline(pipe)

def load_inpaint_pipeline(model_name: str | None) -> Any:
    from custom_pipelines.Flux.pipeline_flux_inpaint import FluxInpaintPipeline as CustomFluxInpaintPipeline

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
        pipeline_cls: Any = CustomFluxInpaintPipeline
        pipeline_name = "FluxInpaintPipeline"        
    logger.info("Flux inpaint pipeline class: %s", pipeline_name)

    #2. Check if diffusers multi-folder or single-file checkpoint
    if entry.model_type == "diffusers":
        pipe = pipeline_cls.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    elif entry.model_type == "single-file":
        pipe = pipeline_cls.from_single_file(
            source,
            torch_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    if pipeline_cls is FluxFillPipeline:
        from custom_pipelines.Flux.memory import enable_flux_vae_memory_savers

        enable_flux_vae_memory_savers(pipe)
    return _configure_flux_pipeline(pipe)

"""
    Methods which generates and renders image using Flux-related Pipelines
"""

@torch.inference_mode()
def generate_text2img_in_process(params: dict[str, object]) -> dict[str, list[str]]:
    from backend.utilities.schedulers import create_scheduler

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
    base_seed = resolve_base_seed(seed)
    logger.info(
        "Flux Text2Image: model=%s, seed=%s, steps=%s, guidance_scale=%s, size=%sx%s, num_images=%s",
        model, base_seed, steps, guidance_scale, width, height, num_images,
    )

    # 3. Create batch_id and output directory
    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    # 4. Load and create pipeline and scheduler
    pipe = None
    pipe_cached = False
    filenames: list[str] = []
    adapter_names: list[str] = []
    stage_profile: dict[str, object] = {"inference": [], "output_save_seconds": 0.0}
    pipeline_healthy = False
    
    #7. Render image
    try:
        load_started = time.perf_counter()
        pipe, pipe_cached, cache_hit = _acquire_flux_pipeline(
            "text2img", model, lambda: load_text2img_pipeline(model)
        )
        stage_profile["pipeline_acquire_seconds"] = round(time.perf_counter() - load_started, 6)
        stage_profile["cache_hit"] = cache_hit
        prepare_started = time.perf_counter()
        pipe.scheduler = create_scheduler(scheduler, pipe)

        adapter_names, lora_coverage = apply_lora_adapters_with_validation(
            pipe,
            lora_adapters,
            expected_family="flux",
            validate=True,
        )
        report_path = write_lora_coverage_report(batch_output_dir, batch_id, lora_coverage)
        if report_path is not None:
            logger.info("LoRA coverage report saved to %s", report_path)
        stage_profile["device_adapter_prepare_seconds"] = round(time.perf_counter() - prepare_started, 6)

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

                pipeline_output, inference_timing = _timed_pipeline_call(pipe, call_kwargs)
                image = pipeline_output.images[0]
                stage_profile["inference"].append(inference_timing)

            save_started = time.perf_counter()
            relpath = save_generated_image(
                image, batch_output_dir, batch_id, current_seed, params,
                mode="txt2img", pipeline="flux",
            )
            stage_profile["output_save_seconds"] += time.perf_counter() - save_started
            logger.info("Image %s saved to %s", i, relpath)
            filenames.append(relpath)
            del image
            cleanup_memory()
        pipeline_healthy = True
    finally:
        #8. Load pipeline + clean memory
        if pipe is not None and adapter_names and hasattr(pipe, "unload_lora_weights"):
            try:
                pipe.unload_lora_weights()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                logger.warning("Failed to unload LoRA weights: %s", exc)
                pipeline_healthy = False

        if pipe_cached and not pipeline_healthy:
            _FLUX_PIPELINE_CACHE.discard(pipe)
            pipe = None
        elif not pipe_cached:
            release_pipeline(pipe, logger=logger)
        pipe = None

    #9.  Return output
    stage_profile["output_save_seconds"] = round(float(stage_profile["output_save_seconds"]), 6)
    return {"images": [f"/outputs/{name}" for name in filenames], "runtime_profile": stage_profile}


@torch.inference_mode()
def generate_img2img_in_process(params: dict[str, object]) -> dict[str, list[str]]:
    from backend.utilities.schedulers import create_scheduler

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
    base_seed = resolve_base_seed(seed)
    logger.info(
        "Flux Img2Img: model=%s, seed=%s, steps=%s, guidance_scale=%s, size=%sx%s, strength=%s, num_images=%s",
        model, base_seed, steps, guidance_scale, width, height, strength,num_images,
    )
    
    #3. Create batch_id and output directory
    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    #4. Load and create pipeline and scheduler
    pipe = None
    pipe_cached = False
    filenames: list[str] = []
    adapter_names: list[str] = []
    stage_profile: dict[str, object] = {"inference": [], "output_save_seconds": 0.0}
    pipeline_healthy = False
    
    #7. Render images one by one
    try:
        load_started = time.perf_counter()
        pipe, pipe_cached, cache_hit = _acquire_flux_pipeline(
            "img2img", model, lambda: load_img2img_pipeline(model)
        )
        stage_profile["pipeline_acquire_seconds"] = round(time.perf_counter() - load_started, 6)
        stage_profile["cache_hit"] = cache_hit
        prepare_started = time.perf_counter()
        pipe.scheduler = create_scheduler(scheduler, pipe)

        adapter_names, lora_coverage = apply_lora_adapters_with_validation(
            pipe,
            lora_adapters,
            expected_family="flux",
            validate=True,
        )
        report_path = write_lora_coverage_report(batch_output_dir, batch_id, lora_coverage)
        if report_path is not None:
            logger.info("LoRA coverage report saved to %s", report_path)
        stage_profile["device_adapter_prepare_seconds"] = round(time.perf_counter() - prepare_started, 6)

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

                pipeline_output, inference_timing = _timed_pipeline_call(pipe, call_kwargs)
                image = pipeline_output.images[0]
                stage_profile["inference"].append(inference_timing)

            image_width, image_height = initial_image.size
            save_started = time.perf_counter()
            relpath = save_generated_image(
                image, batch_output_dir, batch_id, current_seed, params,
                mode="img2img", pipeline="flux",
                remove_params=("initial_image",),
                size=(image_width, image_height),
            )
            stage_profile["output_save_seconds"] += time.perf_counter() - save_started
            logger.info("Image %s saved to %s", i, relpath)
            filenames.append(relpath)

            del image
            cleanup_memory()
        pipeline_healthy = True
    finally:
        #8. Unload lora weights & clean memory
        if pipe is not None and adapter_names and hasattr(pipe, "unload_lora_weights"):
            try:
                pipe.unload_lora_weights()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                logger.warning("Failed to unload LoRA weights: %s", exc)
                pipeline_healthy = False

        if pipe_cached and not pipeline_healthy:
            _FLUX_PIPELINE_CACHE.discard(pipe)
            pipe = None
        elif not pipe_cached:
            release_pipeline(pipe, logger=logger)
        pipe = None

    #9. Return output
    stage_profile["output_save_seconds"] = round(float(stage_profile["output_save_seconds"]), 6)
    return {"images": [f"/outputs/{name}" for name in filenames], "runtime_profile": stage_profile}


@torch.inference_mode()
def generate_inpaint_in_process(params: dict[str, object]) -> dict[str, list[str]]:
    from backend.utilities.schedulers import create_scheduler

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
    base_seed = resolve_base_seed(seed)
    logger.info(
        "Flux Inpaint: model=%s, seed=%s, steps=%s, guidance_scale=%s, strength=%s, num_images=%s",
        model, base_seed, steps, guidance_scale, strength, num_images,
    )

    #3. Create batch_id and output directory
    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    #4. Load and create pipeline and scheduler
    pipe = None
    pipe_cached = False
    filenames: list[str] = []
    adapter_names: list[str] = []
    stage_profile: dict[str, object] = {"inference": [], "output_save_seconds": 0.0}
    pipeline_healthy = False
    
    #7. Render image one by one
    try:
        load_started = time.perf_counter()
        pipe, pipe_cached, cache_hit = _acquire_flux_pipeline(
            "inpaint", model, lambda: load_inpaint_pipeline(model)
        )
        stage_profile["pipeline_acquire_seconds"] = round(time.perf_counter() - load_started, 6)
        stage_profile["cache_hit"] = cache_hit
        prepare_started = time.perf_counter()
        pipe.scheduler = create_scheduler(scheduler, pipe)

        adapter_names, lora_coverage = apply_lora_adapters_with_validation(
            pipe,
            lora_adapters,
            expected_family="flux",
            validate=True,
        )
        report_path = write_lora_coverage_report(batch_output_dir, batch_id, lora_coverage)
        if report_path is not None:
            logger.info("LoRA coverage report saved to %s", report_path)
        stage_profile["device_adapter_prepare_seconds"] = round(time.perf_counter() - prepare_started, 6)

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

                pipeline_output, inference_timing = _timed_pipeline_call(pipe, call_kwargs)
                image = pipeline_output.images[0]
                stage_profile["inference"].append(inference_timing)

            image_width, image_height = initial_image.size
            save_started = time.perf_counter()
            relpath = save_generated_image(
                image, batch_output_dir, batch_id, current_seed, params,
                mode="inpaint", pipeline="flux",
                remove_params=("initial_image", "mask_image"),
                size=(image_width, image_height),
            )
            stage_profile["output_save_seconds"] += time.perf_counter() - save_started
            logger.info("Image %s saved to %s", i, relpath)
            filenames.append(relpath)

            del image
            cleanup_memory()
        pipeline_healthy = True
    finally:
        #8. Unload lora weights & clean memory
        if pipe is not None and adapter_names and hasattr(pipe, "unload_lora_weights"):
            try:
                pipe.unload_lora_weights()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                logger.warning("Failed to unload LoRA weights: %s", exc)
                pipeline_healthy = False

        if pipe_cached and not pipeline_healthy:
            _FLUX_PIPELINE_CACHE.discard(pipe)
            pipe = None
        elif not pipe_cached:
            release_pipeline(pipe, logger=logger)
        pipe = None

    # 9. Return output
    stage_profile["output_save_seconds"] = round(float(stage_profile["output_save_seconds"]), 6)
    return {"images": [f"/outputs/{name}" for name in filenames], "runtime_profile": stage_profile}
