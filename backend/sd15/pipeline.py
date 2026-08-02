"""
Stable Diffusion 1.5 (SD1.5) pipeline helpers.

This module is responsible for:
- Loading Diffusers pipelines for txt2img, img2img, inpaint, and ControlNet.
- Running inference (CUDA / fp16) and writing PNG outputs + embedded metadata.
- Optional LoRA adapter application and pipeline-layer logging/diagnostics.

The functions here are used by workflow tasks (e.g. `sd15.text2img`), so they
aim to be deterministic (seeded) and side-effectful only in well-defined ways
(writing files under `OUTPUT_DIR`).
"""

import torch
import json
import logging
import math
import subprocess
import sys
import tempfile
import threading
import numpy as np
from contextlib import contextmanager
from pathlib import Path
from typing import cast
from PIL import ImageFilter, Image

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
)

from backend.config import OUTPUT_DIR
from backend.settings import REPOSITORY_ROOT
from backend.utilities.logging import configure_logging
from backend.registries.model import get_model_entry
from backend.utilities.resource_logging import resource_logger
from backend.adapters.ip_adapter import IpAdapterManager
from backend.adapters.ip_adapter_embeds import (
    load_ip_adapter_embeds_artifact,
    validate_ip_adapter_embeds_metadata,
)
# from testing.pipeline_stable_diffusion import(StableDiffusionPipeline)
from backend.utilities.pipeline import (
    build_fixed_step_timesteps,
    build_png_metadata,
    build_batch_output_relpath,
    get_batch_output_dir,
    make_batch_id,
    release_pipeline,
    resolve_model_source,
)
from backend.utilities.schedulers import create_scheduler
from backend.utilities.prompt import build_prompt_embeddings
from backend import config
from backend.utilities.pipeline_layer_logging import (
    append_layers_report,
    capture_runtime_used_layers,
    collect_pipeline_layers,
)
from backend.lora.utils import apply_lora_adapters_with_validation, write_lora_coverage_report
from backend.sd15.subprocess_io import serialize_params_for_subprocess

logger = logging.getLogger(__name__)
configure_logging()

_LCM_LORA_MODEL_ID = "latent-consistency/lcm-lora-sdv1-5"
_LCM_LORA_ADAPTER_NAME = "lcm_lora_sd15"
_LCM_DEFAULT_STEPS = 4
_LCM_DEFAULT_CFG = 0.0
_DEFAULT_IP_ADAPTER_MODEL = "h94/IP-Adapter"
_DEFAULT_IP_ADAPTER_SUBFOLDER = "models"
_DEFAULT_IP_ADAPTER_WEIGHT_NAME = "ip-adapter_sd15.bin"
_DEFAULT_IP_ADAPTER_SCALE = 0.6
_SD15_SUBPROCESS_SEMAPHORE = threading.Semaphore(1)

"""
    Helper functions
"""
def create_blur_mask(mask_image, blur_factor: int):
    """
    Return a blurred copy of `mask_image` with a bounded Gaussian blur radius.

    Args:
        mask_image: PIL image used as an inpaint mask.
        blur_factor: Requested blur radius. Values are clamped to ``[0, 128]``.

    Returns:
        The original image when blur is ``0``; otherwise a blurred copy.
    """
    blur_factor = max(0, min(blur_factor, 128))
    if blur_factor == 0:
        return mask_image
    return mask_image.filter(ImageFilter.GaussianBlur(radius=blur_factor))


def _build_sd15_prompt_call_kwargs(
    pipe,
    prompt: str,
    negative_prompt: str,
    *,
    clip_skip: int | None,
    weighting_policy: str = "diffusers-like",
) -> dict[str, object]:
    """Build mutually exclusive raw-prompt or precomputed-embedding kwargs."""
    prompt_embeds, negative_prompt_embeds, use_prompt_embeds = build_prompt_embeddings(
        pipe,
        prompt,
        negative_prompt,
        clip_skip=clip_skip,
        weighting_policy=weighting_policy,
    )
    if use_prompt_embeds:
        return {
            "prompt": None,
            "negative_prompt": None,
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            # Embeddings already include clip-skip. Keeping this unset also
            # avoids Diffusers 0.39's incompatible Transformers 5.x lookup.
            "clip_skip": None,
        }
    return {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "prompt_embeds": None,
        "negative_prompt_embeds": None,
        "clip_skip": None,
    }


def _resource_metadata(bound_args):
    """
    Build resource-logging metadata from a function's bound arguments.

    This keeps the logging payload small and consistent across generation calls.
    """
    return {
        "batch_id": bound_args.arguments.get("batch_id"),
        "model": bound_args.arguments.get("model"),
        "num_images": bound_args.arguments.get("num_images"),
    }


def _snap_dimension(value: int, multiple: int = 8) -> int:
    """Round a dimension up to the next multiple (SD models commonly prefer multiples of 8)."""
    if multiple <= 0:
        return value
    return max(multiple, int(math.ceil(value / multiple)) * multiple)


def _upscale_image(image: Image.Image, scale: float) -> Image.Image:
    """Upscale an image by `scale` using Lanczos, snapping size to SD-friendly dimensions."""
    if scale <= 1.0:
        return image
    target_width = _snap_dimension(int(round(image.width * scale)))
    target_height = _snap_dimension(int(round(image.height * scale)))
    return image.resize((target_width, target_height), resample=Image.LANCZOS)


def _resize_control_image_to_target(
    control_image: Image.Image | list[Image.Image],
    *,
    target_width: int,
    target_height: int,
) -> Image.Image | list[Image.Image]:
    """Resize ControlNet image(s) to exactly match the rendered output size."""

    def _resize_single(image: Image.Image, index: int | None = None) -> Image.Image:
        source_width, source_height = image.size
        if source_width == target_width and source_height == target_height:
            return image

        if source_width != target_width and source_height != target_height:
            resize_case = "resize_width_and_height"
        elif source_height != target_height:
            resize_case = "resize_height_only"
        else:
            resize_case = "resize_width_only"

        if index is None:
            logger.info(
                "Resizing ControlNet control_image (%s): %sx%s -> %sx%s",
                resize_case,
                source_width,
                source_height,
                target_width,
                target_height,
            )
        else:
            logger.info(
                "Resizing ControlNet control_image[%s] (%s): %sx%s -> %sx%s",
                index,
                resize_case,
                source_width,
                source_height,
                target_width,
                target_height,
            )
        return image.resize((target_width, target_height), resample=Image.LANCZOS)

    if isinstance(control_image, list):
        return [_resize_single(image, index=i) for i, image in enumerate(control_image)]
    return _resize_single(control_image)


def _make_inpaint_controlnet_condition(
    image: Image.Image,
    mask_image: Image.Image,
) -> torch.Tensor:
    """
    Build the special conditioning tensor expected by SD1.5 inpaint ControlNet.

    The ControlNet v1.1 inpaint checkpoint is conditioned on the original image
    with masked pixels set to -1.0, matching the Diffusers model-card example.
    """
    rgb_image = image.convert("RGB")
    mask = mask_image.convert("L").resize(rgb_image.size)
    image_array = np.array(rgb_image).astype(np.float32) / 255.0
    mask_array = np.array(mask).astype(np.float32) / 255.0
    image_array[mask_array > 0.5] = -1.0
    image_array = np.expand_dims(image_array, 0).transpose(0, 3, 1, 2)
    return torch.from_numpy(image_array)


def _enable_xformers_memory_efficient_attention_if_available(pipe) -> bool:
    """
    Enable xFormers attention when the optional dependency is installed.

    xFormers is a performance optimization, not a functional requirement. Some
    Windows/local installs do not include it, so generation should keep running
    with Diffusers' default attention path when it is unavailable.
    """
    if not hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        logger.debug(
            "Pipeline %s does not expose xFormers memory efficient attention.",
            pipe.__class__.__name__,
        )
        return False

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except (ImportError, ModuleNotFoundError) as exc:
        logger.warning(
            "xFormers memory efficient attention is unavailable; continuing without it. %s",
            exc,
        )
        return False

    logger.info("Enabled xFormers memory efficient attention.")
    return True


def _apply_lora_adapters(
    pipe,
    lora_adapters: list[object] | None,
    *,
    validate: bool = False,
) -> list[str]:
    """
    Apply requested LoRA adapters to a pipeline.

    Returns:
        A list of adapter names actually loaded into the pipeline.
    """
    adapter_names, _ = apply_lora_adapters_with_validation(
        pipe,
        lora_adapters,
        expected_family="sd15",
        validate=validate,
    )
    return adapter_names


def _apply_lcm_lora(pipe) -> str:
    """Load the hard-coded SD1.5 LCM LoRA adapter."""
    logger.info("Loading SD1.5 LCM LoRA adapter: %s", _LCM_LORA_MODEL_ID)
    pipe.load_lora_weights(_LCM_LORA_MODEL_ID, adapter_name=_LCM_LORA_ADAPTER_NAME)
    return _LCM_LORA_ADAPTER_NAME


def _cleanup_lora_adapters(pipe, adapter_names: list[str]) -> None:
    """Best-effort cleanup for both pipeline-level and component-level LoRA adapters."""
    if not adapter_names:
        return
    logger.info("Cleaning up %s LoRA adapter(s): %s", len(adapter_names), adapter_names)

    if hasattr(pipe, "unload_lora_weights"):
        try:
            logger.debug("Attempting pipeline-level LoRA unload via unload_lora_weights().")
            pipe.unload_lora_weights()
            logger.debug("Pipeline-level LoRA unload completed.")
        except Exception:
            logger.exception("Failed to unload pipeline LoRA weights cleanly.")

    for component_name in ("unet", "text_encoder", "text_encoder_2", "transformer"):
        component = getattr(pipe, component_name, None)
        if component is None or not hasattr(component, "delete_adapters"):
            continue
        try:
            logger.debug("Attempting adapter deletion on component '%s'.", component_name)
            component.delete_adapters(adapter_names)
            logger.debug("Adapter deletion succeeded on component '%s'.", component_name)
        except Exception:
            logger.debug(
                "Skipping component LoRA adapter cleanup for %s; delete_adapters failed.",
                component_name,
                exc_info=True,
            )


def _metadata_without_runtime_images(params: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in params.items()
        if key not in {"ip_adapter_image", "ip_adapter_mask_image"}
    }


def _build_ip_adapter_kwargs(
    *,
    enabled: bool,
    image_embeds: list[torch.Tensor] | None,
    masks: list[torch.Tensor] | None,
) -> dict[str, object]:
    if not enabled:
        return {}

    kwargs: dict[str, object] = {"ip_adapter_image_embeds": image_embeds}
    if masks is not None:
        kwargs["cross_attention_kwargs"] = {"ip_adapter_masks": masks}
    return kwargs


def _run_sd15_subprocess(operation: str, params: dict[str, object]) -> list[str]:

    with tempfile.TemporaryDirectory(prefix="sd15_") as tmpdir:
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
            "backend.sd15.subprocess_runner",
            str(input_path),
            str(output_path),
        ]
        with _SD15_SUBPROCESS_SEMAPHORE:
            completed = subprocess.run(cmd, cwd=str(REPOSITORY_ROOT))

        if not output_path.exists():
            raise RuntimeError("SD1.5 subprocess failed: No subprocess result was written.")

        result_payload = json.loads(output_path.read_text(encoding="utf-8"))
        if completed.returncode != 0 or not result_payload.get("ok"):
            detail = result_payload.get("error") or "Unknown subprocess failure."
            error_type = result_payload.get("error_type")
            if error_type:
                detail = f"{error_type}: {detail}"
            raise RuntimeError(f"SD1.5 subprocess failed: {detail}")

        result = result_payload.get("result")
        if not isinstance(result, list):
            raise RuntimeError("SD1.5 subprocess returned an invalid result.")
        return [str(path) for path in result]


def generate_images_controlnet(params: dict[str, object]) -> list[str]:
    return _run_sd15_subprocess("controlnet_text2img", params)


def generate_images(params: dict[str, object]) -> list[str]:
    return _run_sd15_subprocess("text2img", params)


def generate_images_img2img(params: dict[str, object]) -> list[str]:
    return _run_sd15_subprocess("img2img", params)


def generate_images_img2img_controlnet(params: dict[str, object]) -> list[str]:
    return _run_sd15_subprocess("img2img_controlnet", params)


def generate_images_inpaint(params: dict[str, object]) -> list[str]:
    return _run_sd15_subprocess("inpaint", params)


def generate_images_inpaint_controlnet(params: dict[str, object]) -> list[str]:
    return _run_sd15_subprocess("inpaint_controlnet", params)


@contextmanager
def _hide_image_encoder_while_using_ip_adapter_embeds(pipe, *, enabled: bool):
    if not enabled or pipe is None or not hasattr(pipe, "image_encoder"):
        yield
        return

    image_encoder = pipe.image_encoder
    pipe.image_encoder = None
    try:
        yield
    finally:
        pipe.image_encoder = image_encoder


"""
    Load Pipelines
"""

def load_text2img_pipeline(model_name: str | None):
    """
    Load the base SD1.5 txt2img pipeline on CUDA fp16.

    ``model_name`` is resolved via the model registry and may point to a
    Diffusers directory model or a single-file checkpoint.

    Side effects:
        Moves the pipeline to GPU (``cuda``) and disables the safety checker.
    """
    entry = get_model_entry(model_name)

    source = resolve_model_source(entry)
    logger.info("URL: %s", source)
    
    if entry.model_type == "diffusers":
        pipe = StableDiffusionPipeline.from_pretrained(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,  # keep simple; can re-enable later
        )
    elif entry.model_type == "single-file":
        pipe = StableDiffusionPipeline.from_single_file(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    # Run on CUDA in fp16 for performance. Safety checker is disabled by design here.
    pipe.to("cuda")
    return pipe


def load_img2img_pipeline(model_name: str | None):
    """
    Load the SD1.5 img2img pipeline on CUDA fp16.

    Side effects:
        Moves the pipeline to GPU (``cuda``) and disables the safety checker.
    """
    entry = get_model_entry(model_name)

    source = resolve_model_source(entry)
    logger.info("URL: %s", source)
    if entry.model_type == "diffusers":
        img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        img2img_pipe = StableDiffusionImg2ImgPipeline.from_single_file(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    img2img_pipe.to("cuda")
    return img2img_pipe


def load_inpaint_pipeline(model_name: str | None):
    """
    Load the SD1.5 inpainting pipeline on CUDA fp16.

    Side effects:
        Moves the pipeline to GPU (``cuda``) and disables the safety checker.
    """
    entry = get_model_entry(model_name)

    source = resolve_model_source(entry)
    logger.info("URL: %s", source)
    if entry.model_type == "diffusers":
        inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        inpaint_pipe = StableDiffusionInpaintPipeline.from_single_file(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    inpaint_pipe.to("cuda")
    return inpaint_pipe


def load_controlnet_pipeline(model_name: str | None, controlnet_model: str | list[str]):
    """
    Load a ControlNet-enabled SD1.5 pipeline on CUDA fp16.

    Args:
        model_name: Optional base model registry key.
        controlnet_model: Diffusers ControlNet model id/path or list of ids/paths.

    Side effects:
        Loads both base and ControlNet weights and moves the pipeline to GPU.
    """
    entry = get_model_entry(model_name)

    source = resolve_model_source(entry)
    logger.info("Base model: %s", source)
    controlnet: ControlNetModel | list[ControlNetModel]
    if isinstance(controlnet_model, list):
        controlnet = [
            ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)
            for model_id in controlnet_model
        ]
    else:
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model,
            torch_dtype=torch.float16,
        )

    if entry.model_type == "diffusers":
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        pipe = StableDiffusionControlNetPipeline.from_single_file(
            source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    pipe.to("cuda")
    return pipe


def load_controlnet_img2img_pipeline(model_name: str | None, controlnet_model: str | list[str]):
    """
    Load a ControlNet-enabled SD1.5 img2img pipeline on CUDA fp16.

    Args:
        model_name: Optional base model registry key.
        controlnet_model: Diffusers ControlNet model id/path or list of ids/paths.

    Side effects:
        Loads both base and ControlNet weights and moves the pipeline to GPU.
    """
    entry = get_model_entry(model_name)

    source = resolve_model_source(entry)
    logger.info("Base model: %s", source)
    controlnet: ControlNetModel | list[ControlNetModel]
    if isinstance(controlnet_model, list):
        controlnet = [
            ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)
            for model_id in controlnet_model
        ]
    else:
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model,
            torch_dtype=torch.float16,
        )

    if entry.model_type == "diffusers":
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
            source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    pipe.to("cuda")
    return pipe


def load_controlnet_inpaint_pipeline(model_name: str | None, controlnet_model: str | list[str]):
    """
    Load a ControlNet-enabled SD1.5 inpaint pipeline on CUDA fp16.

    Args:
        model_name: Optional base model registry key.
        controlnet_model: Diffusers ControlNet model id/path or list of ids/paths.

    Side effects:
        Loads both base and ControlNet weights and moves the pipeline to GPU.
    """
    entry = get_model_entry(model_name)

    source = resolve_model_source(entry)
    logger.info("Base model: %s", source)
    controlnet: ControlNetModel | list[ControlNetModel]
    if isinstance(controlnet_model, list):
        controlnet = [
            ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)
            for model_id in controlnet_model
        ]
    else:
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model,
            torch_dtype=torch.float16,
        )

    if entry.model_type == "diffusers":
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        pipe = StableDiffusionControlNetInpaintPipeline.from_single_file(
            source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    pipe.to("cuda")
    return pipe

"""
    Generate and render images
"""

@torch.inference_mode()
def generate_images_controlnet_in_process(params: dict[str, object],) -> list[str]:
    """
    Generate SD1.5 + ControlNet images and write PNG outputs to disk.

    This function optionally captures pipeline layer-usage diagnostics based on
    runtime configuration and embeds generation settings into PNG metadata.

    Returns:
        Output PNG paths relative to ``OUTPUT_DIR``.
    """
    # Base text2image inputs
    prompt = str(params["prompt"])
    negative_prompt = str(params.get("negative_prompt") or "")
    steps = int(params.get("steps") or 20)
    cfg = float(params.get("cfg") or 7.5)
    width = int(params.get("width") or 512)
    height = int(params.get("height") or 512)
    seed = params.get("seed")
    scheduler = str(params.get("scheduler") or "euler")
    model = params.get("model")
    num_images = int(params.get("num_images") or 1)
    clip_skip = int(params.get("clip_skip") or 1)
    
    # Controlnet inputs
    controlnet_model = cast(str | list[str], params["controlnet_model"])
    control_image = cast(Image.Image | list[Image.Image], params["control_image"])
    controlnet_conditioning_scale = cast(
        float | list[float],
        params.get("controlnet_conditioning_scale", 1.0),
    )
    controlnet_guess_mode = bool(params.get("controlnet_guess_mode", False))
    control_guidance_start = cast(
        float | list[float],
        params.get("control_guidance_start", 0.0),
    )
    control_guidance_end = cast(
        float | list[float],
        params.get("control_guidance_end", 1.0),
    )
    batch_id = cast(str | None, params.get("batch_id"))

    if not batch_id:
        batch_id = make_batch_id()
    params["batch_id"] = batch_id

    control_image = _resize_control_image_to_target(
        control_image,
        target_width=width,
        target_height=height,
    )

    pipe = load_controlnet_pipeline(model, controlnet_model)
    pipe.scheduler = create_scheduler(scheduler, pipe)
    pipe.safety_checker = None
    _enable_xformers_memory_efficient_attention_if_available(pipe)

    if clip_skip > 1:
        # Diffusers exposes clip-skip by effectively reducing the text encoder depth.
        pipe.text_encoder.config.num_hidden_layers = (
            pipe.text_encoder.config.num_hidden_layers - (clip_skip - 1)
        )

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    arch_layers = None
    used_layer_names = None
    name_to_type = None

    if config.PIPELINE_LAYER_LOGGING_ENABLED:
        # Optionally capture which layers run (useful for debugging pipeline variants).
        arch_layers = collect_pipeline_layers(
            pipe,
            leaf_only=config.PIPELINE_LAYER_LOGGING_LEAF_ONLY,
        )
        with capture_runtime_used_layers(
            pipe,
            leaf_only=config.PIPELINE_LAYER_LOGGING_LEAF_ONLY,
        ) as (used_layer_names, name_to_type, name_to_inputs, name_to_calls):
            results = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg,
                width=width,
                height=height,
                image=control_image,
                num_images_per_prompt=num_images,
                generator=generator,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                guess_mode=controlnet_guess_mode,
                control_guidance_start=control_guidance_start,
                control_guidance_end=control_guidance_end,
            )
    else:
        results = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            width=width,
            height=height,
            image=control_image,
            num_images_per_prompt=num_images,
            generator=generator,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guess_mode=controlnet_guess_mode,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
        )

    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    if config.PIPELINE_LAYER_LOGGING_ENABLED:
        append_layers_report(
            output_dir=batch_output_dir,
            batch_id=batch_id,
            label="sd15_controlnet",
            pipeline_name=pipe.__class__.__name__,
            architecture_layers=arch_layers,
            runtime_used_layer_names=used_layer_names,
            runtime_name_to_type=name_to_type,
            runtime_name_to_input_summary=(name_to_inputs if config.PIPELINE_LAYER_LOGGING_CAPTURE_INPUTS else None),
            runtime_name_to_call_count=(name_to_calls if config.PIPELINE_LAYER_LOGGING_CAPTURE_INPUTS else None),
        )

    png_info = build_png_metadata(params)

    filenames = []
    for idx, image in enumerate(results.images):
        name = f"{batch_id}_controlnet_{idx}.png"
        image.save(batch_output_dir / name, pnginfo=png_info)
        filenames.append(build_batch_output_relpath(batch_id, name))

    return filenames

@torch.inference_mode()
def generate_images_in_process(params: dict[str, object],) -> list[str]:
    """
    Generate SD1.5 txt2img images, write PNG outputs, and return relative paths.

    Features:
        - Optional LoRA adapter loading with coverage report output.
        - Optional prompt embedding path for prompt-weighting/clip-skip policies.
        - Optional runtime layer logging on the first generated image.
        - Embedded PNG metadata for reproducibility.

    Notes:
        ``hires_enabled``/``hires_scale`` are currently recorded in metadata for
        downstream usage; this function itself performs txt2img generation only.
    """
    # Normalize all txt2img inputs in one place for easier maintenance and tracing.
    prompt = str(params["prompt"])
    negative_prompt = str(params.get("negative_prompt") or "")
    steps = int(params.get("steps") or 20)
    cfg = float(params.get("cfg") or 7.5)
    width = int(params.get("width") or 512)
    height = int(params.get("height") or 512)
    seed = params.get("seed")
    scheduler = str(params.get("scheduler") or "euler")
    model = params.get("model")
    num_images = int(params.get("num_images") or 1)
    clip_skip = int(params.get("clip_skip") or 1)
    lora_adapters = params.get("lora_adapters")
    lcm_enabled = bool(params.get("lcm_enabled", False)) or scheduler.lower() == "lcm"
    weighting_policy = str(params.get("weighting_policy") or "diffusers-like")

    ip_adapter_image = cast(Image.Image | None, params.get("ip_adapter_image"))
    ip_adapter_image_embeds_ref = params.get("ip_adapter_image_embeds_ref")
    if ip_adapter_image is not None and ip_adapter_image_embeds_ref is not None:
        raise ValueError("Provide either ip_adapter_image or ip_adapter_image_embeds_ref, not both.")
    ip_adapter_enabled = ip_adapter_image is not None or ip_adapter_image_embeds_ref is not None
    ip_adapter_mask_image = cast(Image.Image | None, params.get("ip_adapter_mask_image"))
    ip_adapter_model = str(params.get("ip_adapter_model") or _DEFAULT_IP_ADAPTER_MODEL)
    ip_adapter_subfolder = str(
        params.get("ip_adapter_subfolder") or _DEFAULT_IP_ADAPTER_SUBFOLDER
    )
    ip_adapter_weight_name = str(
        params.get("ip_adapter_weight_name") or _DEFAULT_IP_ADAPTER_WEIGHT_NAME
    )
    ip_adapter_scale_raw = params.get("ip_adapter_scale")
    ip_adapter_scale = (
        _DEFAULT_IP_ADAPTER_SCALE
        if ip_adapter_scale_raw is None
        else float(ip_adapter_scale_raw)
    )
    batch_id = params.get("batch_id")

    # 1. Check and set seed number(if not present, set random seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = seed
    
    # 2. Set batch_id for output folder
    if batch_id is None:
        batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)
    
    # 3. Load pipeline and chosen scheduler
    pipe = load_text2img_pipeline(model)
    pipe.scheduler = create_scheduler(scheduler, pipe)
    logger.info("Generate: model=%s, seed=%s, scheduler=%s, steps=%s, cfg=%s, size= %sx%s, num_images=%s", model, base_seed, scheduler, steps, cfg, width, height, num_images,)

    if ip_adapter_enabled:
        if ip_adapter_image_embeds_ref is not None:
            embeds_payload = load_ip_adapter_embeds_artifact(ip_adapter_image_embeds_ref)
            validate_ip_adapter_embeds_metadata(
                embeds_payload,
                expected_model=ip_adapter_model,
                expected_subfolder=ip_adapter_subfolder,
                expected_weight_name=ip_adapter_weight_name,
                do_classifier_free_guidance=cfg > 1.0,
                expected_family="SD15",
            )
            ip_adapter_image_embeds = embeds_payload["embeds"]
            IpAdapterManager.load(
                pipe,
                model=ip_adapter_model,
                subfolder=ip_adapter_subfolder,
                weight_name=ip_adapter_weight_name,
                scale=ip_adapter_scale,
                family="SD1.5",
                image_encoder_folder=None,
            )
        else:
            IpAdapterManager.load(
                pipe,
                model=ip_adapter_model,
                subfolder=ip_adapter_subfolder,
                weight_name=ip_adapter_weight_name,
                scale=ip_adapter_scale,
                family="SD1.5",
            )
            ip_adapter_image_embeds = IpAdapterManager.prepare_image_embeds(
                pipe,
                ip_adapter_image,
                do_classifier_free_guidance=cfg > 1.0,
            )
        ip_adapter_masks = (
            IpAdapterManager.prepare_masks(
                ip_adapter_mask_image,
                height=height,
                width=width,
            )
            if ip_adapter_mask_image is not None
            else None
        )
    else:
        ip_adapter_image_embeds = None
        ip_adapter_masks = None
    
    # 4. Apply lora to pipeline and generate lora coverage report
    adapter_names = []
    lora_coverage = {}
    if lcm_enabled:
        lcm_adapter_name = _apply_lcm_lora(pipe)
        adapter_names, lora_coverage = apply_lora_adapters_with_validation(
            pipe,
            lora_adapters,
            expected_family="sd15",
            validate=True,
            preloaded_adapters=[(lcm_adapter_name, 1.0)],
        )
    else:
        adapter_names, lora_coverage = apply_lora_adapters_with_validation(
            pipe,
            lora_adapters,
            expected_family="sd15",
            validate=True,
        )
    report_path = write_lora_coverage_report(batch_output_dir, batch_id, lora_coverage)
    if report_path is not None:
        logger.info("LoRA coverage report saved to %s", report_path)

    arch_layers = None
    if config.PIPELINE_LAYER_LOGGING_ENABLED:
        arch_layers = collect_pipeline_layers(pipe, leaf_only=config.PIPELINE_LAYER_LOGGING_LEAF_ONLY,)

    # 5. Build prompt embeddings
    prompt_embeds = None
    negative_prompt_embeds = None
    use_prompt_embeds = False
    prompt_embeds_ready = False
    if not config.PIPELINE_LAYER_LOGGING_ENABLED:
        prompt_embeds, negative_prompt_embeds, use_prompt_embeds = build_prompt_embeddings(
            pipe,
            prompt,
            negative_prompt,
            clip_skip=clip_skip,
            weighting_policy=weighting_policy,
        )
        prompt_embeds_ready = True
    
    filenames = []
    ip_adapter_kwargs = _build_ip_adapter_kwargs(
        enabled=ip_adapter_enabled,
        image_embeds=ip_adapter_image_embeds,
        masks=ip_adapter_masks,
    )
    # 6. Loop around image generation per image
    try:
        for i in range(num_images):
            # Offset seed per image so batches are deterministic and distinct.
            current_seed = base_seed + i
            generator = torch.Generator(device="cuda").manual_seed(current_seed)

            # Capture used layers during rendering
            if config.PIPELINE_LAYER_LOGGING_ENABLED and i == 0:
                with capture_runtime_used_layers(
                    pipe,
                    leaf_only=config.PIPELINE_LAYER_LOGGING_LEAF_ONLY,
                ) as (used_layer_names, name_to_type, name_to_inputs, name_to_calls):
                    prompt_embeds, negative_prompt_embeds, use_prompt_embeds = build_prompt_embeddings(
                        pipe,
                        prompt,
                        negative_prompt,
                        clip_skip=clip_skip,
                        weighting_policy=weighting_policy,
                    )
                    prompt_embeds_ready = True
                    # Generate image
                    with _hide_image_encoder_while_using_ip_adapter_embeds(
                        pipe,
                        enabled=ip_adapter_image_embeds is not None,
                    ):
                        image = pipe(
                            prompt=None if use_prompt_embeds else prompt,
                            negative_prompt=None if use_prompt_embeds else negative_prompt,
                            num_inference_steps=steps,
                            guidance_scale=cfg,
                            width=width,
                            height=height,
                            generator=generator,
                            clip_skip=clip_skip,
                            prompt_embeds=prompt_embeds if use_prompt_embeds else None,
                            negative_prompt_embeds=negative_prompt_embeds if use_prompt_embeds else None,
                            **ip_adapter_kwargs,
                        ).images[0]

                # Log layers to report
                append_layers_report(
                    output_dir=batch_output_dir,
                    batch_id=batch_id,
                    label="sd15_txt2img",
                    pipeline_name=pipe.__class__.__name__,
                    architecture_layers=arch_layers,
                    runtime_used_layer_names=used_layer_names,
                    runtime_name_to_type=name_to_type,
                    runtime_name_to_input_summary=(name_to_inputs if config.PIPELINE_LAYER_LOGGING_CAPTURE_INPUTS else None),
                    runtime_name_to_call_count=(name_to_calls if config.PIPELINE_LAYER_LOGGING_CAPTURE_INPUTS else None),
                )
            else:
                # If prompt embeds not present, generate them
                if not prompt_embeds_ready:
                    prompt_embeds, negative_prompt_embeds, use_prompt_embeds = build_prompt_embeddings(
                        pipe,
                        prompt,
                        negative_prompt,
                        clip_skip=clip_skip,
                        weighting_policy=weighting_policy,
                    )
                    prompt_embeds_ready = True
                # Generate image
                with _hide_image_encoder_while_using_ip_adapter_embeds(
                    pipe,
                    enabled=ip_adapter_image_embeds is not None,
                ):
                    image = pipe(
                        prompt=None if use_prompt_embeds else prompt,
                        negative_prompt=None if use_prompt_embeds else negative_prompt,
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        width=width,
                        height=height,
                        generator=generator,
                        clip_skip=clip_skip,
                        prompt_embeds=prompt_embeds if use_prompt_embeds else None,
                        negative_prompt_embeds=negative_prompt_embeds if use_prompt_embeds else None,
                        **ip_adapter_kwargs,
                    ).images[0]

            # Write the PNG and embed all inputs/settings for later inspection.
            filename = batch_output_dir / f"{batch_id}_{current_seed}.png"
            metadata = {
                **_metadata_without_runtime_images(params),
                "seed": current_seed,
                "ip_adapter_enabled": ip_adapter_enabled,
            }
            pnginfo = build_png_metadata(metadata)
            image.save(filename, pnginfo=pnginfo)
            logger.info("Image %s saved to %s", i, filename.name)

            filenames.append(build_batch_output_relpath(batch_id, filename.name))
    finally:
        IpAdapterManager.cleanup(pipe, ip_adapter_enabled)
        _cleanup_lora_adapters(pipe, adapter_names)
        release_pipeline(pipe, logger=logger)
    # Return list of filenames
    return filenames

@torch.inference_mode()
def generate_images_img2img_in_process(params: dict[str, object],) -> list[str]:
    """
    Generate SD1.5 img2img outputs from an initial image and write PNG files.

    Args:
        initial_image: Source image for img2img.
        strength: Img2img denoise strength.
        prompt: Positive prompt text.
        negative_prompt: Negative prompt text.
        steps: Number of denoising steps.
        cfg: Classifier-free guidance scale.
        width: Requested width (used for logging/compatibility).
        height: Requested height (used for logging/compatibility).
        seed: Base seed; ``None``/``0`` selects a random base seed.
        scheduler: Scheduler name.
        model: Optional model registry key.
        num_images: Number of images to generate.
        clip_skip: CLIP skip value.
        lora_adapters: Optional LoRA adapter specs.
        batch_id: Optional batch identifier.

    Returns:
        Output PNG paths relative to ``OUTPUT_DIR``.
    """
    # Normalize all img2img inputs in one place for easier maintenance and tracing.
    initial_image = params["initial_image"]
    strength = float(params.get("strength") or 0.75)
    prompt = str(params["prompt"])
    negative_prompt = str(params.get("negative_prompt") or "")
    scheduler = str(params.get("scheduler") or "euler")
    lcm_enabled = bool(params.get("lcm_enabled", False)) or scheduler.lower() == "lcm"
    if lcm_enabled:
        steps = int(
            params["steps"]
            if "steps" in params and params.get("steps") is not None
            else _LCM_DEFAULT_STEPS
        )
        cfg = float(
            params["cfg"]
            if "cfg" in params and params.get("cfg") is not None
            else _LCM_DEFAULT_CFG
        )
    else:
        steps = int(params.get("steps") or 20)
        cfg = float(params.get("cfg") or 7.5)
    width = int(params.get("width") or getattr(initial_image, "width", 0))
    height = int(params.get("height") or getattr(initial_image, "height", 0))
    seed = params.get("seed")
    model = params.get("model")
    num_images = int(params.get("num_images") or 1)
    clip_skip = int(params.get("clip_skip") or 1)
    weighting_policy = str(params.get("weighting_policy") or "diffusers-like")
    lora_adapters = params.get("lora_adapters")
    ip_adapter_image = cast(Image.Image | None, params.get("ip_adapter_image"))
    ip_adapter_image_embeds_ref = params.get("ip_adapter_image_embeds_ref")
    if ip_adapter_image is not None and ip_adapter_image_embeds_ref is not None:
        raise ValueError("Provide either ip_adapter_image or ip_adapter_image_embeds_ref, not both.")
    ip_adapter_enabled = ip_adapter_image is not None or ip_adapter_image_embeds_ref is not None
    ip_adapter_mask_image = cast(Image.Image | None, params.get("ip_adapter_mask_image"))
    ip_adapter_model = str(params.get("ip_adapter_model") or _DEFAULT_IP_ADAPTER_MODEL)
    ip_adapter_subfolder = str(
        params.get("ip_adapter_subfolder") or _DEFAULT_IP_ADAPTER_SUBFOLDER
    )
    ip_adapter_weight_name = str(
        params.get("ip_adapter_weight_name") or _DEFAULT_IP_ADAPTER_WEIGHT_NAME
    )
    ip_adapter_scale_raw = params.get("ip_adapter_scale")
    ip_adapter_scale = (
        _DEFAULT_IP_ADAPTER_SCALE
        if ip_adapter_scale_raw is None
        else float(ip_adapter_scale_raw)
    )
    batch_id = params.get("batch_id")

    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = seed
    if batch_id is None:
        batch_id = make_batch_id()

    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    pipe = load_img2img_pipeline(model)
    pipe.scheduler = create_scheduler(scheduler, pipe)
    logger.info(
        "Img2Img: model=%s seed=%s scheduler=%s steps=%s cfg=%s size=%sx%s strength=%s num_images=%s",
        model,
        base_seed,
        scheduler,
        steps,
        cfg,
        width,
        height,
        strength,
        num_images,
    )
    if ip_adapter_enabled:
        if ip_adapter_image_embeds_ref is not None:
            embeds_payload = load_ip_adapter_embeds_artifact(ip_adapter_image_embeds_ref)
            validate_ip_adapter_embeds_metadata(
                embeds_payload,
                expected_model=ip_adapter_model,
                expected_subfolder=ip_adapter_subfolder,
                expected_weight_name=ip_adapter_weight_name,
                do_classifier_free_guidance=cfg > 1.0,
                expected_family="SD15",
            )
            ip_adapter_image_embeds = embeds_payload["embeds"]
            IpAdapterManager.load(
                pipe,
                model=ip_adapter_model,
                subfolder=ip_adapter_subfolder,
                weight_name=ip_adapter_weight_name,
                scale=ip_adapter_scale,
                family="SD1.5",
                image_encoder_folder=None,
            )
        else:
            IpAdapterManager.load(
                pipe,
                model=ip_adapter_model,
                subfolder=ip_adapter_subfolder,
                weight_name=ip_adapter_weight_name,
                scale=ip_adapter_scale,
                family="SD1.5",
            )
            ip_adapter_image_embeds = IpAdapterManager.prepare_image_embeds(
                pipe,
                ip_adapter_image,
                do_classifier_free_guidance=cfg > 1.0,
            )
        ip_adapter_masks = (
            IpAdapterManager.prepare_masks(
                ip_adapter_mask_image,
                height=height,
                width=width,
            )
            if ip_adapter_mask_image is not None
            else None
        )
    else:
        ip_adapter_image_embeds = None
        ip_adapter_masks = None

    filenames = []
    adapter_names = []
    if lcm_enabled:
        lcm_adapter_name = _apply_lcm_lora(pipe)
        adapter_names, lora_coverage = apply_lora_adapters_with_validation(
            pipe,
            lora_adapters,
            expected_family="sd15",
            validate=True,
            preloaded_adapters=[(lcm_adapter_name, 1.0)],
        )
        report_path = write_lora_coverage_report(batch_output_dir, batch_id, lora_coverage)
        if report_path is not None:
            logger.info("LoRA coverage report saved to %s", report_path)
    else:
        adapter_names = _apply_lora_adapters(pipe, lora_adapters)
    image_width, image_height = initial_image.size
    metadata_base = {
        "mode": "img2img",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "cfg": cfg,
        "width": image_width,
        "height": image_height,
        "scheduler": scheduler,
        "model": model,
        "strength": strength,
        "clip_skip": clip_skip,
        "lcm_enabled": lcm_enabled,
        "lcm_lora_model": _LCM_LORA_MODEL_ID if lcm_enabled else None,
        "batch_id": batch_id,
    }
    ip_adapter_kwargs = _build_ip_adapter_kwargs(
        enabled=ip_adapter_enabled,
        image_embeds=ip_adapter_image_embeds,
        masks=ip_adapter_masks,
    )

    try:
        for i in range(num_images):
            # Offset seed per image so batches are deterministic and distinct.
            current_seed = base_seed + i
            generator = torch.Generator(device="cuda").manual_seed(current_seed)

            if config.PIPELINE_LAYER_LOGGING_ENABLED and i == 0:
                arch_layers = collect_pipeline_layers(
                    pipe,
                    leaf_only=config.PIPELINE_LAYER_LOGGING_LEAF_ONLY,
                )
                with capture_runtime_used_layers(
                    pipe,
                    leaf_only=config.PIPELINE_LAYER_LOGGING_LEAF_ONLY,
                ) as (used_layer_names, name_to_type, name_to_inputs, name_to_calls):
                    prompt_kwargs = _build_sd15_prompt_call_kwargs(
                        pipe,
                        prompt,
                        negative_prompt,
                        clip_skip=clip_skip,
                        weighting_policy=weighting_policy,
                    )
                    with _hide_image_encoder_while_using_ip_adapter_embeds(
                        pipe,
                        enabled=ip_adapter_image_embeds is not None,
                    ):
                        image = pipe(
                            **prompt_kwargs,
                            image=initial_image,
                            strength=strength,
                            num_inference_steps=steps,
                            guidance_scale=cfg,
                            generator=generator,
                            **ip_adapter_kwargs,
                        ).images[0]
                append_layers_report(
                    output_dir=batch_output_dir,
                    batch_id=batch_id,
                    label="sd15_img2img",
                    pipeline_name=pipe.__class__.__name__,
                    architecture_layers=arch_layers,
                    runtime_used_layer_names=used_layer_names,
                    runtime_name_to_type=name_to_type,
                    runtime_name_to_input_summary=(name_to_inputs if config.PIPELINE_LAYER_LOGGING_CAPTURE_INPUTS else None),
                    runtime_name_to_call_count=(name_to_calls if config.PIPELINE_LAYER_LOGGING_CAPTURE_INPUTS else None),
                )
            else:
                prompt_kwargs = _build_sd15_prompt_call_kwargs(
                    pipe,
                    prompt,
                    negative_prompt,
                    clip_skip=clip_skip,
                    weighting_policy=weighting_policy,
                )
                with _hide_image_encoder_while_using_ip_adapter_embeds(
                    pipe,
                    enabled=ip_adapter_image_embeds is not None,
                ):
                    image = pipe(
                        **prompt_kwargs,
                        image=initial_image,
                        strength=strength,
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        generator=generator,
                        **ip_adapter_kwargs,
                    ).images[0]

            filename = batch_output_dir / f"{batch_id}_{current_seed}.png"
            metadata = {
                **metadata_base,
                "seed": current_seed,
                "ip_adapter_enabled": ip_adapter_enabled,
                "ip_adapter_model": ip_adapter_model if ip_adapter_enabled else None,
                "ip_adapter_subfolder": ip_adapter_subfolder if ip_adapter_enabled else None,
                "ip_adapter_weight_name": ip_adapter_weight_name if ip_adapter_enabled else None,
                "ip_adapter_scale": ip_adapter_scale if ip_adapter_enabled else None,
            }
            pnginfo = build_png_metadata(metadata)
            image.save(filename, pnginfo=pnginfo)
            logger.info("Image %s saved to %s", i, filename.name)

            filenames.append(build_batch_output_relpath(batch_id, filename.name))
    finally:
        IpAdapterManager.cleanup(pipe, ip_adapter_enabled)
        _cleanup_lora_adapters(pipe, adapter_names)
        release_pipeline(pipe, logger=logger)

    return filenames

@torch.inference_mode()
def generate_images_img2img_controlnet_in_process(params: dict[str, object],) -> list[str]:
    """
    Generate SD1.5 img2img + ControlNet outputs and write PNG files.

    Returns:
        Output PNG paths relative to ``OUTPUT_DIR``.
    """
    # Normalize all img2img controlnet inputs in one place for easier maintenance and tracing.
    initial_image = cast(Image.Image, params["initial_image"])
    strength = float(params.get("strength") or 0.75)
    prompt = str(params["prompt"])
    negative_prompt = str(params.get("negative_prompt") or "")
    steps = int(params.get("steps") or 20)
    cfg = float(params.get("cfg") or 7.5)
    width = int(params.get("width") or initial_image.width)
    height = int(params.get("height") or initial_image.height)
    seed = params.get("seed")
    scheduler = str(params.get("scheduler") or "euler")
    model = params.get("model")
    num_images = int(params.get("num_images") or 1)
    clip_skip = int(params.get("clip_skip") or 1)
    controlnet_model = cast(str | list[str], params["controlnet_model"])
    control_image = cast(Image.Image | list[Image.Image], params["control_image"])
    controlnet_conditioning_scale = cast(
        float | list[float],
        params.get("controlnet_conditioning_scale", 1.0),
    )
    controlnet_guess_mode = bool(params.get("controlnet_guess_mode", False))
    control_guidance_start = float(params.get("control_guidance_start", 0.0))
    control_guidance_end = float(params.get("control_guidance_end", 1.0))
    lora_adapters = params.get("lora_adapters")
    batch_id = cast(str | None, params.get("batch_id"))

    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = seed
    if batch_id is None:
        batch_id = make_batch_id()
    params["batch_id"] = batch_id

    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)
    image_width, image_height = initial_image.size
    control_image = _resize_control_image_to_target(
        control_image,
        target_width=image_width,
        target_height=image_height,
    )

    pipe = load_controlnet_img2img_pipeline(model, controlnet_model)
    pipe.scheduler = create_scheduler(scheduler, pipe)
    logger.info(
        "ControlNet Img2Img: model=%s seed=%s scheduler=%s steps=%s cfg=%s size=%sx%s strength=%s num_images=%s",
        model,
        base_seed,
        scheduler,
        steps,
        cfg,
        width,
        height,
        strength,
        num_images,
    )
    adapter_names, lora_coverage = apply_lora_adapters_with_validation(
        pipe,
        lora_adapters,
        expected_family="sd15",
        validate=True,
    )
    report_path = write_lora_coverage_report(batch_output_dir, batch_id, lora_coverage)
    if report_path is not None:
        logger.info("LoRA coverage report saved to %s", report_path)

    filenames = []
    try:
        for i in range(num_images):
            # Offset seed per image so batches are deterministic and distinct.
            current_seed = base_seed + i
            generator = torch.Generator(device="cuda").manual_seed(current_seed)

            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=initial_image,
                control_image=control_image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
                clip_skip=clip_skip,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                guess_mode=controlnet_guess_mode,
                control_guidance_start=control_guidance_start,
                control_guidance_end=control_guidance_end,
            ).images[0]

            filename = batch_output_dir / f"{batch_id}_controlnet_{current_seed}.png"
            image_params = {
                **params,
                "mode": "img2img_controlnet",
                "width": image_width,
                "height": image_height,
                "seed": current_seed,
                "batch_id": batch_id,
            }
            pnginfo = build_png_metadata(image_params)
            image.save(filename, pnginfo=pnginfo)
            logger.info("Image %s saved to %s", i, filename.name)
            filenames.append(build_batch_output_relpath(batch_id, filename.name))
    finally:
        _cleanup_lora_adapters(pipe, adapter_names)
        release_pipeline(pipe, logger=logger)

    return filenames

@torch.inference_mode()
def generate_images_inpaint_in_process(params: dict[str, object],) -> list[str]:
    """
    Generate SD1.5 inpaint outputs from an initial image and mask.

    This function writes PNG files to the batch directory, stores generation
    settings in PNG metadata, and optionally captures layer-usage diagnostics.

    Returns:
        Output PNG paths relative to ``OUTPUT_DIR``.
    """
    # Normalize all inpaint inputs in one place for easier maintenance and tracing.
    initial_image = params["initial_image"]
    mask_image = params["mask_image"]
    prompt = str(params["prompt"])
    negative_prompt = str(params.get("negative_prompt") or "")
    seed = params.get("seed")
    scheduler = str(params.get("scheduler") or "euler")
    lcm_enabled = bool(params.get("lcm_enabled", False)) or scheduler.lower() == "lcm"
    scheduler = "lcm" if lcm_enabled else scheduler
    if lcm_enabled:
        steps = int(
            params["steps"]
            if "steps" in params and params.get("steps") is not None
            else _LCM_DEFAULT_STEPS
        )
        cfg = float(
            params["cfg"]
            if "cfg" in params and params.get("cfg") is not None
            else _LCM_DEFAULT_CFG
        )
    else:
        steps = int(params.get("steps") or 20)
        cfg = float(params.get("cfg") or 7.5)
    model = params.get("model")
    num_images = int(params.get("num_images") or 1)
    strength = float(params.get("strength") or 0.5)
    padding_mask_crop = int(params.get("padding_mask_crop") or 32)
    clip_skip = int(params.get("clip_skip") or 1)
    weighting_policy = str(params.get("weighting_policy") or "diffusers-like")
    lora_adapters = params.get("lora_adapters")
    ip_adapter_image = cast(Image.Image | None, params.get("ip_adapter_image"))
    ip_adapter_image_embeds_ref = params.get("ip_adapter_image_embeds_ref")
    if ip_adapter_image is not None and ip_adapter_image_embeds_ref is not None:
        raise ValueError("Provide either ip_adapter_image or ip_adapter_image_embeds_ref, not both.")
    ip_adapter_enabled = ip_adapter_image is not None or ip_adapter_image_embeds_ref is not None
    ip_adapter_mask_image = cast(Image.Image | None, params.get("ip_adapter_mask_image"))
    ip_adapter_model = str(params.get("ip_adapter_model") or _DEFAULT_IP_ADAPTER_MODEL)
    ip_adapter_subfolder = str(
        params.get("ip_adapter_subfolder") or _DEFAULT_IP_ADAPTER_SUBFOLDER
    )
    ip_adapter_weight_name = str(
        params.get("ip_adapter_weight_name") or _DEFAULT_IP_ADAPTER_WEIGHT_NAME
    )
    ip_adapter_scale_raw = params.get("ip_adapter_scale")
    ip_adapter_scale = (
        _DEFAULT_IP_ADAPTER_SCALE
        if ip_adapter_scale_raw is None
        else float(ip_adapter_scale_raw)
    )
    batch_id = params.get("batch_id")

    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = seed
    if batch_id is None:
        batch_id = make_batch_id()

    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    pipe = load_inpaint_pipeline(model)
    pipe.scheduler = create_scheduler(scheduler, pipe)
    width, height = initial_image.size
    logger.info(
        "Inpaint: model=%s seed=%s scheduler=%s steps=%s cfg=%s size=%sx%s num_images=%s strength=%s, padding_mask_crop=%s",
        model, base_seed, scheduler, steps, cfg,
        width, height, num_images, strength, padding_mask_crop
    )
    if ip_adapter_enabled:
        if ip_adapter_image_embeds_ref is not None:
            embeds_payload = load_ip_adapter_embeds_artifact(ip_adapter_image_embeds_ref)
            validate_ip_adapter_embeds_metadata(
                embeds_payload,
                expected_model=ip_adapter_model,
                expected_subfolder=ip_adapter_subfolder,
                expected_weight_name=ip_adapter_weight_name,
                do_classifier_free_guidance=cfg > 1.0,
                expected_family="SD15",
            )
            ip_adapter_image_embeds = embeds_payload["embeds"]
            IpAdapterManager.load(
                pipe,
                model=ip_adapter_model,
                subfolder=ip_adapter_subfolder,
                weight_name=ip_adapter_weight_name,
                scale=ip_adapter_scale,
                family="SD1.5",
                image_encoder_folder=None,
            )
        else:
            IpAdapterManager.load(
                pipe,
                model=ip_adapter_model,
                subfolder=ip_adapter_subfolder,
                weight_name=ip_adapter_weight_name,
                scale=ip_adapter_scale,
                family="SD1.5",
            )
            ip_adapter_image_embeds = IpAdapterManager.prepare_image_embeds(
                pipe,
                ip_adapter_image,
                do_classifier_free_guidance=cfg > 1.0,
            )
        ip_adapter_masks = (
            IpAdapterManager.prepare_masks(
                ip_adapter_mask_image,
                height=initial_image.height,
                width=initial_image.width,
            )
            if ip_adapter_mask_image is not None
            else None
        )
    else:
        ip_adapter_image_embeds = None
        ip_adapter_masks = None

    filenames = []
    adapter_names = []
    if lcm_enabled:
        lcm_adapter_name = _apply_lcm_lora(pipe)
        adapter_names, lora_coverage = apply_lora_adapters_with_validation(
            pipe,
            lora_adapters,
            expected_family="sd15",
            validate=True,
            preloaded_adapters=[(lcm_adapter_name, 1.0)],
        )
        report_path = write_lora_coverage_report(batch_output_dir, batch_id, lora_coverage)
        if report_path is not None:
            logger.info("LoRA coverage report saved to %s", report_path)
    else:
        adapter_names = _apply_lora_adapters(pipe, lora_adapters)
    metadata_base = {
        "mode": "inpaint",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "cfg": cfg,
        "width": width,
        "height": height,
        "scheduler": scheduler,
        "model": model,
        "strength": strength,
        "padding_mask_crop": padding_mask_crop,
        "clip_skip": clip_skip,
        "lcm_enabled": lcm_enabled,
        "lcm_lora_model": _LCM_LORA_MODEL_ID if lcm_enabled else None,
        "batch_id": batch_id,
    }
    ip_adapter_kwargs = _build_ip_adapter_kwargs(
        enabled=ip_adapter_enabled,
        image_embeds=ip_adapter_image_embeds,
        masks=ip_adapter_masks,
    )

    try:
        for i in range(num_images):
            # Offset seed per image so batches are deterministic and distinct.
            current_seed = base_seed + i
            generator = torch.Generator(device="cuda").manual_seed(current_seed)

            if config.PIPELINE_LAYER_LOGGING_ENABLED and i == 0:
                arch_layers = collect_pipeline_layers(
                    pipe,
                    leaf_only=config.PIPELINE_LAYER_LOGGING_LEAF_ONLY,
                )
                with capture_runtime_used_layers(
                    pipe,
                    leaf_only=config.PIPELINE_LAYER_LOGGING_LEAF_ONLY,
                ) as (used_layer_names, name_to_type, name_to_inputs, name_to_calls):
                    prompt_kwargs = _build_sd15_prompt_call_kwargs(
                        pipe,
                        prompt,
                        negative_prompt,
                        clip_skip=clip_skip,
                        weighting_policy=weighting_policy,
                    )
                    with _hide_image_encoder_while_using_ip_adapter_embeds(
                        pipe,
                        enabled=ip_adapter_image_embeds is not None,
                    ):
                        image = pipe(
                            **prompt_kwargs,
                            image=initial_image,
                            mask_image=mask_image,
                            num_inference_steps=steps,
                            guidance_scale=cfg,
                            generator=generator,
                            strength=strength,
                            padding_mask_crop=padding_mask_crop,
                            **ip_adapter_kwargs,
                        ).images[0]
                append_layers_report(
                    output_dir=batch_output_dir,
                    batch_id=batch_id,
                    label="sd15_inpaint",
                    pipeline_name=pipe.__class__.__name__,
                    architecture_layers=arch_layers,
                    runtime_used_layer_names=used_layer_names,
                    runtime_name_to_type=name_to_type,
                    runtime_name_to_input_summary=(name_to_inputs if config.PIPELINE_LAYER_LOGGING_CAPTURE_INPUTS else None),
                    runtime_name_to_call_count=(name_to_calls if config.PIPELINE_LAYER_LOGGING_CAPTURE_INPUTS else None),
                )
            else:
                prompt_kwargs = _build_sd15_prompt_call_kwargs(
                    pipe,
                    prompt,
                    negative_prompt,
                    clip_skip=clip_skip,
                    weighting_policy=weighting_policy,
                )
                with _hide_image_encoder_while_using_ip_adapter_embeds(
                    pipe,
                    enabled=ip_adapter_image_embeds is not None,
                ):
                    image = pipe(
                        **prompt_kwargs,
                        image=initial_image,
                        mask_image=mask_image,
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        generator=generator,
                        strength=strength,
                        padding_mask_crop=padding_mask_crop,
                        **ip_adapter_kwargs,
                    ).images[0]

            filename = batch_output_dir / f"{batch_id}_{current_seed}.png"
            metadata = {
                **metadata_base,
                "seed": current_seed,
                "ip_adapter_enabled": ip_adapter_enabled,
                "ip_adapter_model": ip_adapter_model if ip_adapter_enabled else None,
                "ip_adapter_subfolder": ip_adapter_subfolder if ip_adapter_enabled else None,
                "ip_adapter_weight_name": ip_adapter_weight_name if ip_adapter_enabled else None,
                "ip_adapter_scale": ip_adapter_scale if ip_adapter_enabled else None,
            }
            pnginfo = build_png_metadata(metadata)
            image.save(filename, pnginfo=pnginfo)
            logger.info("Image %s saved to %s", i, filename.name)

            filenames.append(build_batch_output_relpath(batch_id, filename.name))
    finally:
        IpAdapterManager.cleanup(pipe, ip_adapter_enabled)
        _cleanup_lora_adapters(pipe, adapter_names)
        release_pipeline(pipe, logger=logger)

    return filenames

@torch.inference_mode()
def generate_images_inpaint_controlnet_in_process(params: dict[str, object],) -> list[str]:
    """
    Generate SD1.5 inpaint + ControlNet outputs and write PNG files.

    Returns:
        Output PNG paths relative to ``OUTPUT_DIR``.
    """
    # Normalize all inpaint controlnet inputs in one place for easier maintenance and tracing.
    initial_image = cast(Image.Image, params["initial_image"])
    mask_image = cast(Image.Image, params["mask_image"])
    prompt = str(params["prompt"])
    negative_prompt = str(params.get("negative_prompt") or "")
    steps = int(params.get("steps") or 20)
    cfg = float(params.get("cfg") or 7.5)
    seed = params.get("seed")
    scheduler = str(params.get("scheduler") or "euler")
    model = params.get("model")
    num_images = int(params.get("num_images") or 1)
    strength = float(params.get("strength") or 0.5)
    padding_mask_crop = int(params.get("padding_mask_crop") or 32)
    clip_skip = int(params.get("clip_skip") or 1)
    controlnet_model = cast(str | list[str], params["controlnet_model"])
    control_image = cast(Image.Image | list[Image.Image], params["control_image"])
    controlnet_conditioning_scale = cast(
        float | list[float],
        params.get("controlnet_conditioning_scale", 1.0),
    )
    controlnet_guess_mode = bool(params.get("controlnet_guess_mode", False))
    control_guidance_start = float(params.get("control_guidance_start", 0.0))
    control_guidance_end = float(params.get("control_guidance_end", 1.0))
    controlnet_inpaint_condition = bool(params.get("controlnet_inpaint_condition", False))
    lora_adapters = params.get("lora_adapters")
    batch_id = cast(str | None, params.get("batch_id"))

    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = seed
    if batch_id is None:
        batch_id = make_batch_id()
    params["batch_id"] = batch_id

    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)
    width, height = initial_image.size
    if controlnet_inpaint_condition:
        control_image = _make_inpaint_controlnet_condition(initial_image, mask_image)
    else:
        control_image = _resize_control_image_to_target(
            cast(Image.Image | list[Image.Image], control_image),
            target_width=width,
            target_height=height,
        )

    pipe = load_controlnet_inpaint_pipeline(model, controlnet_model)
    pipe.scheduler = create_scheduler(scheduler, pipe)
    logger.info(
        "ControlNet Inpaint: model=%s seed=%s scheduler=%s steps=%s cfg=%s size=%sx%s num_images=%s strength=%s padding_mask_crop=%s",
        model,
        base_seed,
        scheduler,
        steps,
        cfg,
        width,
        height,
        num_images,
        strength,
        padding_mask_crop,
    )

    filenames = []
    adapter_names, lora_coverage = apply_lora_adapters_with_validation(
        pipe,
        lora_adapters,
        expected_family="sd15",
        validate=True,
    )
    report_path = write_lora_coverage_report(batch_output_dir, batch_id, lora_coverage)
    if report_path is not None:
        logger.info("LoRA coverage report saved to %s", report_path)

    try:
        for i in range(num_images):
            current_seed = base_seed + i
            generator = torch.Generator(device="cuda").manual_seed(current_seed)

            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=initial_image,
                mask_image=mask_image,
                control_image=control_image,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
                strength=strength,
                padding_mask_crop=padding_mask_crop,
                clip_skip=clip_skip,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                guess_mode=controlnet_guess_mode,
                control_guidance_start=control_guidance_start,
                control_guidance_end=control_guidance_end,
            ).images[0]

            filename = batch_output_dir / f"{batch_id}_controlnet_{current_seed}.png"
            image_params = {
                **params,
                "mode": "inpaint_controlnet",
                "width": width,
                "height": height,
                "seed": current_seed,
                "batch_id": batch_id,
            }
            pnginfo = build_png_metadata(image_params)
            image.save(filename, pnginfo=pnginfo)
            logger.info("Image %s saved to %s", i, filename.name)

            filenames.append(build_batch_output_relpath(batch_id, filename.name))
    finally:
        _cleanup_lora_adapters(pipe, adapter_names)
        release_pipeline(pipe, logger=logger)

    return filenames

@torch.inference_mode()
def run_sd15_hires_fix(
    *,
    images: list[Image.Image],
    prompt: str,
    negative_prompt: str,
    steps: int,
    cfg: float,
    seed: int | None,
    scheduler: str,
    model: str | None,
    clip_skip: int,
    hires_scale: float,
    hires_strength: float = 0.35,
    lora_adapters: list[object] | None = None,
    weighting_policy: str = "diffusers-like",
    output_dir: Path | None = None,
    batch_id: str | None = None,
) -> list[str]:
    """
    Apply SD1.5 hires-fix to each input image and write PNGs to disk.

    Args:
        images: Source images to upscale/refine.
        prompt: Positive prompt text.
        negative_prompt: Negative prompt text.
        steps: Number of denoising steps.
        cfg: Classifier-free guidance scale.
        seed: Optional base seed. ``None`` or ``0`` selects a random base seed.
        scheduler: Scheduler name.
        model: Optional model registry key.
        clip_skip: CLIP skip value.
        hires_scale: Upscale factor. Must be ``> 1.0``.
        hires_strength: Img2img strength for refinement.
        lora_adapters: Optional LoRA adapter specs.
        weighting_policy: Prompt-weighting policy for embedding construction.
        output_dir: Optional output root. Defaults to batch folder under ``OUTPUT_DIR``.
        batch_id: Optional batch identifier.

    Returns:
        List of output PNG paths relative to ``OUTPUT_DIR``.

    Raises:
        ValueError: If ``hires_scale <= 1.0``.
    """
    if hires_scale <= 1.0:
        raise ValueError("hires_scale must be > 1.0 for sd15.hires_fix")
    if not images:
        return []

    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)

    if batch_id is None:
        batch_id = make_batch_id()
    batch_output_dir = output_dir or get_batch_output_dir(OUTPUT_DIR, batch_id)

    pipe = load_img2img_pipeline(model)
    pipe.scheduler = create_scheduler(scheduler, pipe)
    adapter_names = _apply_lora_adapters(pipe, lora_adapters, validate=False)

    prompt_embeds, negative_prompt_embeds, use_prompt_embeds = build_prompt_embeddings(
        pipe,
        prompt,
        negative_prompt,
        clip_skip=clip_skip,
        weighting_policy=weighting_policy,
    )

    relpaths: list[str] = []
    try:
        for idx, image in enumerate(images):
            # Offset the seed per image to make batch outputs deterministic and distinct.
            current_seed = base_seed + idx
            generator = torch.Generator(device="cuda").manual_seed(current_seed)

            upscaled = _upscale_image(image, hires_scale)
            out_image = pipe(
                prompt=None if use_prompt_embeds else prompt,
                negative_prompt=None if use_prompt_embeds else negative_prompt,
                image=upscaled,
                strength=hires_strength,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
                clip_skip=clip_skip,
                prompt_embeds=prompt_embeds if use_prompt_embeds else None,
                negative_prompt_embeds=negative_prompt_embeds if use_prompt_embeds else None,
            ).images[0]

            filename = batch_output_dir / f"{batch_id}_{current_seed}.png"
            # Store prompt/settings inside the PNG for later reproduction/debugging.
            pnginfo = build_png_metadata(
                {
                    "mode": "hires_fix",
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "steps": steps,
                    "cfg": cfg,
                    "seed": current_seed,
                    "scheduler": scheduler,
                    "model": model,
                    "clip_skip": clip_skip,
                    "hires_scale": hires_scale,
                    "hires_strength": hires_strength,
                    "batch_id": batch_id,
                }
            )
            out_image.save(filename, pnginfo=pnginfo)
            relpaths.append(build_batch_output_relpath(batch_id, filename.name))
    finally:
        _cleanup_lora_adapters(pipe, adapter_names)
        release_pipeline(pipe, logger=logger)

    return relpaths
