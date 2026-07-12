from __future__ import annotations

from typing import Any

from backend.workflow.image_tasks import ImageTaskDefaults, run_img2img, run_inpaint, run_text2img
from backend.workflow.registry import TaskDefinition, TaskHandler, bind_task
from backend.workflow.schema_input import FluxImg2ImgInputs, FluxInpaintInputs, FluxText2ImgInputs
from backend.workflow.schema_output import ImagesWithRuntimeProfileOutput


DEFAULTS = ImageTaskDefaults(steps=20, guidance_scale=0.0)


def task_definitions(handlers: dict[str, TaskHandler]) -> dict[str, TaskDefinition]:
    contracts = {
        "flux.text2img": FluxText2ImgInputs,
        "flux.img2img": FluxImg2ImgInputs,
        "flux.inpaint": FluxInpaintInputs,
    }
    return {
        name: bind_task(handlers, name, model, ImagesWithRuntimeProfileOutput)
        for name, model in contracts.items()
    }


def run_flux_text2img_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    return run_text2img("flux.text2img", inputs, deps)


def run_flux_img2img_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    return run_img2img("flux.img2img", inputs, deps, DEFAULTS)


def run_flux_inpaint_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    return run_inpaint("flux.inpaint", inputs, deps, DEFAULTS)
