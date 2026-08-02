from __future__ import annotations

"""Runtime dependency binding and authoritative workflow task assembly."""

import importlib
import logging
from typing import Any

from backend.config import OUTPUT_DIR
from backend.workflow.catalog import build_workflow_catalog as _build_workflow_catalog
from backend.adapters.controlnet_preprocessor_registry import CONTROLNET_PREPROCESSOR_REGISTRY
from backend.adapters.controlnet_preprocessors import get_preprocessor
from backend.utilities.pipeline import get_batch_output_dir, make_batch_id
from backend.workflow.schema_input import (
    ControlNetPreprocessInputs,
    _DEFAULT_SD15_CONTROLNET_MODEL,
    _DEFAULT_SDXL_CONTROLNET_MODEL,
)
from backend.workflow.schema_output import ControlNetPreprocessOutput
from backend.workflow.registry import TaskDefinition, merge_task_definitions
from backend.workflow.types import WorkflowContext
from backend.workflow.utility import (
    _normalize_sd15_controlnet_contract_inputs,
    _normalized_hires_settings,
    _normalized_lora_adapters,
    _normalized_sd15_lora_adapters,
    _open_image_ref,
    _open_video_ref,
    _remap_img2img_strength,
    save_artifact_png,
)
from backend.workflow.sd15 import (
    task_definitions as _sd15_task_definitions,
    run_sd15_animatediff_text2video as _run_sd15_animatediff_text2video,
    run_sd15_controlnet_text2img as _run_sd15_controlnet_text2img,
    run_sd15_hires_fix as _run_sd15_hires_fix,
    run_sd15_img2img as _run_sd15_img2img,
    run_sd15_inpaint as _run_sd15_inpaint,
    run_sd15_ip_adapter_encode_task as _run_sd15_ip_adapter_encode,
    run_sd15_text2img as _run_sd15_text2img,
)
from backend.workflow.sdxl import (
    task_definitions as _sdxl_task_definitions,
    run_sdxl_controlnet_text2img_task as _run_sdxl_controlnet_text2img,
    run_sdxl_img2img_task as _run_sdxl_img2img,
    run_sdxl_inpaint_task as _run_sdxl_inpaint,
    run_sdxl_ip_adapter_encode_task as _run_sdxl_ip_adapter_encode,
    run_sdxl_text2img_task as _run_sdxl_text2img,
)
from backend.workflow.flux import (
    task_definitions as _flux_task_definitions,
    run_flux_text2img_task as _run_flux_text2img,
    run_flux_img2img_task as _run_flux_img2img,
    run_flux_inpaint_task as _run_flux_inpaint,
)
from backend.workflow.z_image import (
    task_definitions as _z_image_task_definitions,
    run_z_image_text2img_task as _run_z_image_text2img,
    run_z_image_img2img_task as _run_z_image_img2img,
    run_z_image_inpaint_task as _run_z_image_inpaint,
)
from backend.workflow.qwen_image import (
    task_definitions as _qwen_image_task_definitions,
    run_qwen_image_text2img_task as _run_qwen_image_text2img,
    run_qwen_image_img2img_task as _run_qwen_image_img2img,
    run_qwen_image_inpaint_task as _run_qwen_image_inpaint,
)
from backend.workflow.ernie_image import (
    task_definitions as _ernie_image_task_definitions,
    run_ernie_image_text2img_task as _run_ernie_image_text2img,
)
from backend.workflow.anima import (
    task_definitions as _anima_task_definitions,
    run_anima_text2img_task as _run_anima_text2img,
)
from backend.workflow.wan import (
    run_wan_image2video_task as _run_wan_image2video,
    run_wan_text2video_task as _run_wan_text2video,
    task_definitions as _wan_task_definitions,
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
    return _run_wan_text2video(inputs, _wan_runtime_deps())


def _wan_image2video(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    return _run_wan_image2video(inputs, _wan_runtime_deps())

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


TASK_DEFINITIONS = merge_task_definitions(
    {
        "controlnet.preprocess": TaskDefinition(
            ControlNetPreprocessInputs,
            ControlNetPreprocessOutput,
            _controlnet_preprocess,
        )
    },
    _sd15_task_definitions(
        {
            "sd15.text2img": _sd15_text2img,
            "sd15.animatediff.text2video": _sd15_animatediff_text2video,
            "sd15.img2img": _sd15_img2img,
            "sd15.inpaint": _sd15_inpaint,
            "sd15.controlnet.text2img": _sd15_controlnet_text2img,
            "sd15.hires_fix": _sd15_hires_fix,
            "sd15.ip_adapter.encode": _sd15_ip_adapter_encode,
        }
    ),
    _sdxl_task_definitions(
        {
            "sdxl.ip_adapter.encode": _sdxl_ip_adapter_encode,
            "sdxl.text2img": _sdxl_text2img,
            "sdxl.controlnet.text2img": _sdxl_controlnet_text2img,
            "sdxl.img2img": _sdxl_img2img,
            "sdxl.inpaint": _sdxl_inpaint,
        }
    ),
    _wan_task_definitions(
        {
            "wan.text2video": _wan_text2video,
            "wan.image2video": _wan_image2video,
        }
    ),
    _flux_task_definitions(
        {
            "flux.text2img": _flux_text2img,
            "flux.img2img": _flux_img2img,
            "flux.inpaint": _flux_inpaint,
        }
    ),
    _qwen_image_task_definitions(
        {
            "qwen-image.text2img": _qwen_image_text2img,
            "qwen-image.img2img": _qwen_image_img2img,
            "qwen-image.inpaint": _qwen_image_inpaint,
        }
    ),
    _z_image_task_definitions(
        {
            "z-image.text2img": _z_image_text2img,
            "z-image.img2img": _z_image_img2img,
            "z-image.inpaint": _z_image_inpaint,
        }
    ),
    _ernie_image_task_definitions({"ernie-image.text2img": _ernie_image_text2img}),
    _anima_task_definitions({"anima.text2img": _anima_text2img}),
)

# Compatibility views for existing callers. They are derived from the single
# authoritative definition map and must never be edited independently.
TASK_REGISTRY = {name: definition.handler for name, definition in TASK_DEFINITIONS.items()}
TASK_INPUT_MODELS = {name: definition.input_model for name, definition in TASK_DEFINITIONS.items()}
TASK_OUTPUT_MODELS = {name: definition.output_model for name, definition in TASK_DEFINITIONS.items()}


def build_workflow_catalog() -> dict[str, Any]:
    return _build_workflow_catalog(TASK_INPUT_MODELS, TASK_OUTPUT_MODELS)
