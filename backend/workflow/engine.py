from __future__ import annotations

import importlib
import json
import logging
from typing import Any, Callable

from PIL import Image
from pydantic import BaseModel

from backend.config import OUTPUT_DIR
from backend.workflow.catalog import build_workflow_catalog as _build_workflow_catalog
from backend.adapters.controlnet_preprocessor_registry import CONTROLNET_PREPROCESSOR_REGISTRY
from backend.adapters.controlnet_preprocessors import get_preprocessor
from backend.utilities.pipeline import get_batch_output_dir, make_batch_id
from backend.workflow.schema_input import (
    AnimaText2ImgInputs,
    ArtifactRef,
    ControlNetPreprocessInputs,
    ErnieImageText2ImgInputs,
    FluxImg2ImgInputs,
    FluxInpaintInputs,
    FluxText2ImgInputs,
    ImageRef,
    Sd15AnimateDiffText2VideoInputs,
    QwenImageImg2ImgInputs,
    QwenImageInpaintInputs,
    QwenImageText2ImgInputs,
    Sd15ControlNetText2ImgInputs,
    Sd15EffectiveControlNetItem,
    Sd15HiresFixInputs,
    Sd15HiresContract,
    Sd15IpAdapterContract,
    Sd15IpAdapterEncodeInputs,
    Sd15Img2ImgInputs,
    Sd15InpaintInputs,
    Sd15Text2ImgInputs,
    Sd15UnifiedLoraContract,
    SdxlControlNetText2ImgInputs,
    SdxlImg2ImgInputs,
    SdxlInpaintInputs,
    SdxlIpAdapterEncodeInputs,
    SdxlText2ImgInputs,
    WanImage2VideoInputs,
    WanText2VideoInputs,
    ZImageImg2ImgInputs,
    ZImageInpaintInputs,
    ZImageText2ImgInputs,
    _DEFAULT_SD15_CONTROLNET_MODEL,
    _DEFAULT_SDXL_CONTROLNET_MODEL,
)
from backend.workflow.schema_output import (
    ArtifactInfo,
    ControlNetPreprocessOutput,
    ImagesOutput,
    ImagesWithBatchOutput,
    Sd15ControlNetText2ImgOutput,
    Sd15Img2ImgOutput,
    Sd15InpaintOutput,
    Sd15IpAdapterEncodeOutput,
    SdxlControlNetText2ImgOutput,
    SdxlImg2ImgOutput,
    SdxlInpaintOutput,
    SdxlIpAdapterEncodeOutput,
    VideosWithBatchOutput,
)
from backend.workflow.types import (
    TaskType,
    WorkflowCanceled,
    WorkflowContext,
    WorkflowRequest,
    WorkflowTask,
)
from backend.workflow.utility import (
    _ARTIFACT_ID_RE,
    _artifact_dir,
    _load_image_from_outputs_url,
    _normalize_sd15_controlnet_contract_inputs,
    _normalized_hires_settings,
    _normalized_lora_adapters,
    _normalized_sd15_lora_adapters,
    _open_image_ref,
    _open_video_ref,
    _remap_img2img_strength,
    _resolve_refs,
    _validate_artifact_id,
    cleanup_artifacts,
    collect_artifact_ids,
    save_artifact_png,
)
from backend.workflow.sd15 import (
    run_sd15_animatediff_text2video as _run_sd15_animatediff_text2video,
    run_sd15_controlnet_text2img as _run_sd15_controlnet_text2img,
    run_sd15_hires_fix as _run_sd15_hires_fix,
    run_sd15_img2img as _run_sd15_img2img,
    run_sd15_inpaint as _run_sd15_inpaint,
    run_sd15_ip_adapter_encode_task as _run_sd15_ip_adapter_encode,
    run_sd15_text2img as _run_sd15_text2img,
)
from backend.workflow.sdxl import (
    run_sdxl_controlnet_text2img_task as _run_sdxl_controlnet_text2img,
    run_sdxl_img2img_task as _run_sdxl_img2img,
    run_sdxl_inpaint_task as _run_sdxl_inpaint,
    run_sdxl_ip_adapter_encode_task as _run_sdxl_ip_adapter_encode,
    run_sdxl_text2img_task as _run_sdxl_text2img,
)
from backend.workflow.flux import (
    run_flux_text2img_task as _run_flux_text2img,
    run_flux_img2img_task as _run_flux_img2img,
    run_flux_inpaint_task as _run_flux_inpaint,
)
from backend.workflow.z_image import (
    run_z_image_text2img_task as _run_z_image_text2img,
    run_z_image_img2img_task as _run_z_image_img2img,
    run_z_image_inpaint_task as _run_z_image_inpaint,
)
from backend.workflow.qwen_image import (
    run_qwen_image_text2img_task as _run_qwen_image_text2img,
    run_qwen_image_img2img_task as _run_qwen_image_img2img,
    run_qwen_image_inpaint_task as _run_qwen_image_inpaint,
)
from backend.workflow.ernie_image import (
    run_ernie_image_text2img_task as _run_ernie_image_text2img,
)
from backend.workflow.anima import (
    run_anima_text2img_task as _run_anima_text2img,
)
from backend.sd15.pipeline import (
    generate_images,
    generate_images_controlnet,
    generate_images_img2img,
    generate_images_img2img_controlnet,
    generate_images_inpaint,
    generate_images_inpaint_controlnet,
    run_sd15_hires_fix,
)
from backend.sd15.animatediff_pipeline import generate_videos_text2video
from backend.wan.pipeline import generate_image2video as generate_wan_image2video
from backend.wan.pipeline import generate_text2video as generate_wan_text2video
import backend.sdxl.pipeline as sdxl_pipeline_module

logger = logging.getLogger(__name__)

_CONTROLNET_PREPROCESSOR_REGISTRY_BY_ID = {
    entry.id: entry for entry in CONTROLNET_PREPROCESSOR_REGISTRY
}

_MAX_CONTROLNET_MODELS = 2

TASK_INPUT_MODELS: dict[str, type[BaseModel]] = {
    # SD15 tasks
    "sd15.text2img": Sd15Text2ImgInputs,
    "sd15.animatediff.text2video": Sd15AnimateDiffText2VideoInputs,
    "sd15.img2img": Sd15Img2ImgInputs,
    "sd15.inpaint": Sd15InpaintInputs,
    "sd15.controlnet.text2img": Sd15ControlNetText2ImgInputs,
    "sd15.hires_fix": Sd15HiresFixInputs,
    "sd15.ip_adapter.encode": Sd15IpAdapterEncodeInputs,
    # WAN tasks
    "wan.text2video": WanText2VideoInputs,
    "wan.image2video": WanImage2VideoInputs,
    # ControlNet utility tasks
    "controlnet.preprocess": ControlNetPreprocessInputs,
    # SDXL tasks
    "sdxl.ip_adapter.encode": SdxlIpAdapterEncodeInputs,
    "sdxl.text2img": SdxlText2ImgInputs,
    "sdxl.controlnet.text2img": SdxlControlNetText2ImgInputs,
    "sdxl.img2img": SdxlImg2ImgInputs,
    "sdxl.inpaint": SdxlInpaintInputs,
    # Flux tasks
    "flux.text2img": FluxText2ImgInputs,
    "flux.img2img": FluxImg2ImgInputs,
    "flux.inpaint": FluxInpaintInputs,
    # Qwen-Image tasks
    "qwen-image.text2img": QwenImageText2ImgInputs,
    "qwen-image.img2img": QwenImageImg2ImgInputs,
    "qwen-image.inpaint": QwenImageInpaintInputs,
    # Z-Image tasks
    "z-image.text2img": ZImageText2ImgInputs,
    "z-image.img2img": ZImageImg2ImgInputs,
    "z-image.inpaint": ZImageInpaintInputs,
    "ernie-image.text2img": ErnieImageText2ImgInputs,
    "anima.text2img": AnimaText2ImgInputs,
}


TASK_OUTPUT_MODELS: dict[str, type[BaseModel]] = {
    # SD15 tasks
    "sd15.text2img": ImagesWithBatchOutput,
    "sd15.animatediff.text2video": VideosWithBatchOutput,
    "sd15.img2img": Sd15Img2ImgOutput,
    "sd15.inpaint": Sd15InpaintOutput,
    "sd15.controlnet.text2img": Sd15ControlNetText2ImgOutput,
    "sd15.hires_fix": ImagesWithBatchOutput,
    "sd15.ip_adapter.encode": Sd15IpAdapterEncodeOutput,
    # WAN tasks
    "wan.text2video": VideosWithBatchOutput,
    "wan.image2video": VideosWithBatchOutput,
    # ControlNet utility tasks
    "controlnet.preprocess": ControlNetPreprocessOutput,
    # SDXL tasks
    "sdxl.ip_adapter.encode": SdxlIpAdapterEncodeOutput,
    "sdxl.text2img": ImagesOutput,
    "sdxl.controlnet.text2img": SdxlControlNetText2ImgOutput,
    "sdxl.img2img": SdxlImg2ImgOutput,
    "sdxl.inpaint": SdxlInpaintOutput,
    # Flux tasks
    "flux.text2img": ImagesOutput,
    "flux.img2img": ImagesOutput,
    "flux.inpaint": ImagesOutput,
    # Qwen-Image tasks
    "qwen-image.text2img": ImagesOutput,
    "qwen-image.img2img": ImagesOutput,
    "qwen-image.inpaint": ImagesOutput,
    # Z-Image tasks
    "z-image.text2img": ImagesOutput,
    "z-image.img2img": ImagesOutput,
    "z-image.inpaint": ImagesOutput,
    "ernie-image.text2img": ImagesOutput,
    "anima.text2img": ImagesOutput,
}



def build_workflow_catalog() -> dict[str, Any]:
    return _build_workflow_catalog(TASK_INPUT_MODELS, TASK_OUTPUT_MODELS)



# SD15 task handlers and dependencies
def _sd15_runtime_deps() -> dict[str, Any]:
    return {
        "normalized_hires_settings": _normalized_hires_settings,
        "normalized_lora_adapters": _normalized_sd15_lora_adapters,
        "normalize_sd15_controlnet_contract_inputs": _normalize_sd15_controlnet_contract_inputs,
        "remap_img2img_strength": _remap_img2img_strength,
        "open_image_ref": _open_image_ref,
        "make_batch_id": make_batch_id,
        "generate_images": generate_images,
        "generate_videos_text2video": generate_videos_text2video,
        "generate_images_img2img": generate_images_img2img,
        "generate_images_img2img_controlnet": generate_images_img2img_controlnet,
        "generate_images_inpaint": generate_images_inpaint,
        "generate_images_inpaint_controlnet": generate_images_inpaint_controlnet,
        "generate_images_controlnet": generate_images_controlnet,
        "generate_ip_adapter_image_embeds": importlib.import_module(
            "backend.sd15.ip_adapter_pipeline"
        ).generate_ip_adapter_image_embeds,
        "default_sd15_controlnet_model": _DEFAULT_SD15_CONTROLNET_MODEL,
        "max_controlnet_models": _MAX_CONTROLNET_MODELS,
        "controlnet_preprocessor_registry_by_id": _CONTROLNET_PREPROCESSOR_REGISTRY_BY_ID,
        "logger": logger,
        "get_batch_output_dir": get_batch_output_dir,
        "output_dir": OUTPUT_DIR,
        "run_sd15_hires_fix": run_sd15_hires_fix,
    }

def _sd15_text2img(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    return _run_sd15_text2img(inputs, _sd15_runtime_deps())


def _sd15_animatediff_text2video(
    inputs: dict[str, Any],
    _ctx: WorkflowContext,
) -> dict[str, Any]:
    return _run_sd15_animatediff_text2video(inputs, _sd15_runtime_deps())



def _sd15_img2img(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    return _run_sd15_img2img(inputs, _sd15_runtime_deps())



def _sd15_inpaint(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    return _run_sd15_inpaint(inputs, _sd15_runtime_deps())



def _sd15_controlnet_text2img(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    return _run_sd15_controlnet_text2img(inputs, _sd15_runtime_deps())


def _sd15_ip_adapter_encode(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    return _run_sd15_ip_adapter_encode(inputs, _sd15_runtime_deps())



# ControlNet utility task handlers
def _controlnet_preprocess(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    source = _open_image_ref(inputs["image"]).convert("RGB")
    preprocessor_id = str(inputs["preprocessor_id"])
    preprocessor = get_preprocessor(preprocessor_id)

    params = inputs.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("params must be an object")

    for key in ("low_threshold", "high_threshold"):
        if inputs.get(key) is not None:
            params[key] = inputs[key]

    try:
        processed = preprocessor.process(source, params)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    artifact = save_artifact_png(processed, prefix="p")
    return {"artifact": artifact}


def _sd15_hires_fix(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    return _run_sd15_hires_fix(inputs, _sd15_runtime_deps())


def _wan_runtime_deps() -> dict[str, Any]:
    return {
        "make_batch_id": make_batch_id,
        "generate_text2video": generate_wan_text2video,
        "generate_image2video": generate_wan_image2video,
        "open_image_ref": _open_image_ref,
        "open_video_ref": _open_video_ref,
    }


def _wan_text2video(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    deps = _wan_runtime_deps()
    generate_text2video = deps["generate_text2video"]
    batch_id = str(inputs.get("batch_id") or make_batch_id())
    generation_params = {
        "prompt": str(inputs["prompt"]),
        "negative_prompt": str(inputs.get("negative_prompt") or ""),
        "steps": int(inputs.get("steps") or 30),
        "guidance_scale": float(inputs.get("guidance_scale") or 6.0),
        "width": int(inputs.get("width") or 832),
        "height": int(inputs.get("height") or 480),
        "seed": inputs.get("seed"),
        "model": str(inputs.get("model") or "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"),
        "num_frames": int(inputs.get("num_frames") or 49),
        "fps": int(inputs.get("fps") or 16),
        "num_videos": int(inputs.get("num_videos") or 1),
        "memory_preset": str(inputs.get("memory_preset") or "safe"),
        "quantization": str(inputs.get("quantization") or "none"),
        "reference_image": (
            deps["open_image_ref"](inputs["reference_image"])
            if inputs.get("reference_image") is not None
            else None
        ),
        "mask_image": (
            deps["open_image_ref"](inputs["mask_image"])
            if inputs.get("mask_image") is not None
            else None
        ),
        "conditioning_video": (
            deps["open_video_ref"](inputs["conditioning_video"])
            if inputs.get("conditioning_video") is not None
            else None
        ),
        "conditioning_scale": float(inputs.get("conditioning_scale") or 1.0),
        "batch_id": batch_id,
    }
    filenames = generate_text2video(generation_params)
    return {"batch_id": batch_id, "videos": [f"/outputs/{name}" for name in filenames]}


def _wan_image2video(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    deps = _wan_runtime_deps()
    generate_image2video = deps["generate_image2video"]
    batch_id = str(inputs.get("batch_id") or make_batch_id())
    generation_params = {
        "image": deps["open_image_ref"](inputs["image"]),
        "prompt": str(inputs["prompt"]),
        "negative_prompt": str(inputs.get("negative_prompt") or ""),
        "steps": int(inputs.get("steps") or 50),
        "guidance_scale": float(inputs.get("guidance_scale") or 5.0),
        "width": int(inputs.get("width") or 832),
        "height": int(inputs.get("height") or 480),
        "seed": inputs.get("seed"),
        "model": str(
            inputs.get("model")
            or r"D:\diffusion\diffusers\Wan2.1-I2V-14B-480P-Diffusers"
        ),
        "num_frames": int(inputs.get("num_frames") or 81),
        "fps": int(inputs.get("fps") or 16),
        "num_videos": int(inputs.get("num_videos") or 1),
        "memory_preset": str(inputs.get("memory_preset") or "offload"),
        "quantization": str(inputs.get("quantization") or "none"),
        "experimental_ack": bool(inputs.get("experimental_ack", True)),
        "batch_id": batch_id,
    }
    filenames = generate_image2video(generation_params)
    return {"batch_id": batch_id, "videos": [f"/outputs/{name}" for name in filenames]}





# SDXL task handlers and dependencies
def _sdxl_runtime_deps() -> dict[str, Any]:
    return {
        "open_image_ref": _open_image_ref,
        "remap_img2img_strength": _remap_img2img_strength,
        "default_sdxl_controlnet_model": _DEFAULT_SDXL_CONTROLNET_MODEL,
        "max_controlnet_models": _MAX_CONTROLNET_MODELS,
        "controlnet_preprocessor_registry_by_id": _CONTROLNET_PREPROCESSOR_REGISTRY_BY_ID,
        "logger": logger,
        "generate_text2img": sdxl_pipeline_module.generate_text2img,
        "generate_ip_adapter_image_embeds": importlib.import_module(
            "backend.sdxl.ip_adapter_pipeline"
        ).generate_ip_adapter_image_embeds,
        "generate_controlnet_text2img": sdxl_pipeline_module.generate_controlnet_text2img,
        "generate_img2img": sdxl_pipeline_module.generate_img2img,
        "generate_img2img_controlnet": sdxl_pipeline_module.generate_img2img_controlnet,
        "generate_inpaint": sdxl_pipeline_module.generate_inpaint,
        "generate_inpaint_controlnet": sdxl_pipeline_module.generate_inpaint_controlnet,
    }

def _sdxl_text2img(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    return _run_sdxl_text2img(inputs, _sdxl_runtime_deps())


def _sdxl_ip_adapter_encode(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    return _run_sdxl_ip_adapter_encode(inputs, _sdxl_runtime_deps())



def _sdxl_controlnet_text2img(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    return _run_sdxl_controlnet_text2img(inputs, _sdxl_runtime_deps())



def _sdxl_img2img(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    return _run_sdxl_img2img(inputs, _sdxl_runtime_deps())



def _sdxl_inpaint(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    return _run_sdxl_inpaint(inputs, _sdxl_runtime_deps())



# Flux task handlers and dependencies
def _flux_runtime_deps() -> dict[str, Any]:
    flux_pipeline_module = importlib.import_module("backend.flux.pipeline")
    deps: dict[str, Any] = {"open_image_ref": _open_image_ref}
    for name in ("generate_text2img", "generate_img2img", "generate_inpaint"):
        func = getattr(flux_pipeline_module, name, None)
        if func is not None:
            deps[name] = func
    return deps


def _flux_text2img(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    return _run_flux_text2img(inputs, _flux_runtime_deps())


def _flux_img2img(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    return _run_flux_img2img(inputs, _flux_runtime_deps())


def _flux_inpaint(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    return _run_flux_inpaint(inputs, _flux_runtime_deps())


# Qwen-Image task handlers and dependencies
def _qwen_image_runtime_deps() -> dict[str, Any]:
    qwen_image_pipeline_module = importlib.import_module("backend.qwen_image.pipeline")
    deps: dict[str, Any] = {
        "open_image_ref": _open_image_ref,
        "remap_img2img_strength": _remap_img2img_strength,
        "normalized_lora_adapters": _normalized_lora_adapters,
    }
    for name in ("generate_text2img", "generate_img2img", "generate_inpaint"):
        func = getattr(qwen_image_pipeline_module, name, None)
        if func is not None:
            deps[name] = func
    return deps


def _qwen_image_text2img(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    return _run_qwen_image_text2img(inputs, _qwen_image_runtime_deps())


def _qwen_image_img2img(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    return _run_qwen_image_img2img(inputs, _qwen_image_runtime_deps())


def _qwen_image_inpaint(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    return _run_qwen_image_inpaint(inputs, _qwen_image_runtime_deps())


# Z-Image task handlers and dependencies
def _z_image_runtime_deps() -> dict[str, Any]:
    z_image_pipeline_module = importlib.import_module("backend.z_image.pipeline")
    deps: dict[str, Any] = {
        "open_image_ref": _open_image_ref,
        "remap_img2img_strength": _remap_img2img_strength,
    }
    for name in ("generate_text2img", "generate_img2img", "generate_inpaint"):
        func = getattr(z_image_pipeline_module, name, None)
        if func is not None:
            deps[name] = func
    return deps


def _z_image_text2img(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    return _run_z_image_text2img(inputs, _z_image_runtime_deps())


def _z_image_img2img(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    return _run_z_image_img2img(inputs, _z_image_runtime_deps())


def _z_image_inpaint(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    return _run_z_image_inpaint(inputs, _z_image_runtime_deps())


def _ernie_image_runtime_deps() -> dict[str, Any]:
    ernie_image_pipeline_module = importlib.import_module("backend.ernie_image.pipeline")
    return {
        "generate_text2img": ernie_image_pipeline_module.generate_text2img,
        "normalized_lora_adapters": _normalized_lora_adapters,
    }


def _ernie_image_text2img(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    return _run_ernie_image_text2img(inputs, _ernie_image_runtime_deps())


def _anima_runtime_deps() -> dict[str, Any]:
    anima_pipeline_module = importlib.import_module("backend.anima.pipeline")
    return {
        "generate_text2img": anima_pipeline_module.generate_text2img,
    }


def _anima_text2img(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    return _run_anima_text2img(inputs, _anima_runtime_deps())


TASK_REGISTRY: dict[str, Callable[[dict[str, Any], WorkflowContext], dict[str, Any]]] = {
    # SD15 tasks
    "sd15.text2img": _sd15_text2img,
    "sd15.animatediff.text2video": _sd15_animatediff_text2video,
    "sd15.img2img": _sd15_img2img,
    "sd15.inpaint": _sd15_inpaint,
    "sd15.controlnet.text2img": _sd15_controlnet_text2img,
    "sd15.hires_fix": _sd15_hires_fix,
    "sd15.ip_adapter.encode": _sd15_ip_adapter_encode,
    # WAN tasks
    "wan.text2video": _wan_text2video,
    "wan.image2video": _wan_image2video,
    # ControlNet utility tasks
    "controlnet.preprocess": _controlnet_preprocess,
    # SDXL tasks
    "sdxl.ip_adapter.encode": _sdxl_ip_adapter_encode,
    "sdxl.text2img": _sdxl_text2img,
    "sdxl.controlnet.text2img": _sdxl_controlnet_text2img,
    "sdxl.img2img": _sdxl_img2img,
    "sdxl.inpaint": _sdxl_inpaint,
    # Flux tasks
    "flux.text2img": _flux_text2img,
    "flux.img2img": _flux_img2img,
    "flux.inpaint": _flux_inpaint,
    # Qwen-Image tasks
    "qwen-image.text2img": _qwen_image_text2img,
    "qwen-image.img2img": _qwen_image_img2img,
    "qwen-image.inpaint": _qwen_image_inpaint,
    # Z-Image tasks
    "z-image.text2img": _z_image_text2img,
    "z-image.img2img": _z_image_img2img,
    "z-image.inpaint": _z_image_inpaint,
    "ernie-image.text2img": _ernie_image_text2img,
    "anima.text2img": _anima_text2img,
}


def execute_workflow(payload: dict[str, Any], *, ctx: WorkflowContext | None = None) -> dict[str, Any]:
    wf = WorkflowRequest.model_validate(payload)
    context = ctx or WorkflowContext()

    task_results: dict[str, dict[str, Any]] = {}
    created_artifacts: set[str] = set()
    try:
        for idx, task in enumerate(wf.tasks):
            if context.should_cancel and context.should_cancel():
                raise WorkflowCanceled("Cancel requested")
            if task.id in task_results:
                raise ValueError(f"Duplicate task id: {task.id}")

            resolved_inputs = _resolve_refs(task.inputs, task_results)
            handler = TASK_REGISTRY.get(task.type)
            if handler is None:
                raise ValueError(f"Unsupported task type: {task.type}")

            if context.update_progress:
                context.update_progress(
                    {
                        "current_task": task.id,
                        "current_task_index": idx,
                        "total_tasks": len(wf.tasks),
                        "phase": "running",
                    }
                )

            result = handler(resolved_inputs, context)
            if not isinstance(result, dict):
                raise ValueError(f"Task {task.id} must return an object")
            created_artifacts |= collect_artifact_ids(result)
            task_results[task.id] = result

            if context.update_progress:
                context.update_progress(
                    {
                        "current_task": task.id,
                        "current_task_index": idx,
                        "total_tasks": len(wf.tasks),
                        "phase": "completed_task",
                    }
                )
    except Exception as exc:
        setattr(exc, "_workflow_created_artifacts", created_artifacts)
        raise

    if wf.return_value is None:
        final_value: Any = task_results[wf.tasks[-1].id] if wf.tasks else {}
    else:
        final_value = _resolve_refs(wf.return_value, task_results)

    return {"outputs": final_value, "tasks": task_results, "created_artifacts": sorted(created_artifacts)}
