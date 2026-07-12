from __future__ import annotations

from typing import Any

from backend.workflow.registry import TaskDefinition, TaskHandler, bind_task
from backend.workflow.schema_input import AnimaText2ImgInputs
from backend.workflow.schema_output import ImagesOutput


def task_definitions(handlers: dict[str, TaskHandler]) -> dict[str, TaskDefinition]:
    name = "anima.text2img"
    return {name: bind_task(handlers, name, AnimaText2ImgInputs, ImagesOutput)}


def run_anima_text2img_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    generate_text2img = deps["generate_text2img"]

    result = generate_text2img(dict(inputs))
    if not isinstance(result, dict):
        raise ValueError("anima.text2img must return an object")
    return result
