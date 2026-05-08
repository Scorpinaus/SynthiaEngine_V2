from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

# Canonical set of task identifiers accepted by workflow validation/dispatch.
# Keeping this as a Literal enables static type checking and autocomplete.
TaskType = Literal[
    "sd15.text2img",
    "sd15.animatediff.text2video",
    "sd15.img2img",
    "sd15.inpaint",
    "sd15.controlnet.text2img",
    "sd15.hires_fix",
    "sd15.ip_adapter.encode",
    "wan.text2video",
    "wan.image2video",
    "controlnet.preprocess",
    "sdxl.ip_adapter.encode",
    "sdxl.text2img",
    "sdxl.controlnet.text2img",
    "sdxl.img2img",
    "sdxl.inpaint",
    "flux.text2img",
    "flux.img2img",
    "flux.inpaint",
    "qwen-image.text2img",
    "qwen-image.img2img",
    "qwen-image.inpaint",
    "z-image.text2img",
    "z-image.img2img",
    "z-image.inpaint",
    "ernie-image.text2img",
]


class WorkflowTask(BaseModel):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    type: TaskType
    inputs: dict[str, Any] = Field(default_factory=dict)


class WorkflowRequest(BaseModel):
    tasks: list[WorkflowTask] = Field(max_length=64)
    return_value: Any | None = Field(default=None, alias="return")


@dataclass(frozen=True)
class WorkflowContext:
    update_progress: Callable[[dict[str, Any]], None] | None = None
    should_cancel: Callable[[], bool] | None = None


class WorkflowCanceled(Exception):
    pass
