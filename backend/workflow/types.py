from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, Field

# Canonical set of task identifiers accepted by workflow validation/dispatch.
# Keeping this as a Literal enables static type checking and autocomplete.
# Task identifiers are validated against the authoritative runtime registry by
# the workflow engine. Keeping a second Literal list here caused registered
# tasks to become unreachable when the two declarations drifted.
TaskType = str


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
