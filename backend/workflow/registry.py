from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from pydantic import BaseModel

from backend.workflow.types import WorkflowContext

TaskHandler = Callable[[dict[str, Any], WorkflowContext], dict[str, Any]]


@dataclass(frozen=True)
class TaskDefinition:
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: TaskHandler


def bind_task(
    handlers: Mapping[str, TaskHandler],
    task_type: str,
    input_model: type[BaseModel],
    output_model: type[BaseModel],
) -> TaskDefinition:
    try:
        handler = handlers[task_type]
    except KeyError as exc:
        raise RuntimeError(f"No runtime handler was provided for task type: {task_type}") from exc
    return TaskDefinition(input_model, output_model, handler)


def merge_task_definitions(
    *definition_groups: Mapping[str, TaskDefinition],
) -> dict[str, TaskDefinition]:
    """Merge explicit task groups and reject ambiguous registrations."""
    merged: dict[str, TaskDefinition] = {}
    for definitions in definition_groups:
        overlap = merged.keys() & definitions.keys()
        if overlap:
            duplicates = ", ".join(sorted(overlap))
            raise RuntimeError(f"Duplicate workflow task registrations: {duplicates}")
        merged.update(definitions)
    return merged
