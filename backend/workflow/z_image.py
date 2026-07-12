from __future__ import annotations

from typing import Any

from backend.workflow.image_tasks import ImageTaskDefaults, run_img2img, run_inpaint, run_text2img
from backend.workflow.registry import TaskDefinition, TaskHandler, bind_task
from backend.workflow.schema_input import ZImageImg2ImgInputs, ZImageInpaintInputs, ZImageText2ImgInputs
from backend.workflow.schema_output import ImagesOutput


DEFAULTS = ImageTaskDefaults(steps=8, guidance_scale=0.0)


def task_definitions(handlers: dict[str, TaskHandler]) -> dict[str, TaskDefinition]:
    contracts = {
        "z-image.text2img": ZImageText2ImgInputs,
        "z-image.img2img": ZImageImg2ImgInputs,
        "z-image.inpaint": ZImageInpaintInputs,
    }
    return {name: bind_task(handlers, name, model, ImagesOutput) for name, model in contracts.items()}


def run_z_image_text2img_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    return run_text2img("z-image.text2img", inputs, deps)


def run_z_image_img2img_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    return run_img2img("z-image.img2img", inputs, deps, DEFAULTS, remap_strength=True)


def run_z_image_inpaint_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    return run_inpaint("z-image.inpaint", inputs, deps, DEFAULTS)
