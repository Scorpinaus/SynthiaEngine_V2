from __future__ import annotations

"""Workflow validation, DAG ordering, dispatch, and result aggregation."""

from typing import Any

from backend.workflow.assembly import TASK_DEFINITIONS
from backend.workflow.types import (
    WorkflowCanceled,
    WorkflowContext,
    WorkflowRequest,
    WorkflowTask,
)
from backend.workflow.utility import (
    _resolve_refs,
    collect_artifact_ids,
    collect_task_refs,
)


def _execution_order(tasks: list[WorkflowTask], return_value: Any | None) -> list[WorkflowTask]:
    """Validate references and return a stable topological task order."""
    by_id: dict[str, WorkflowTask] = {}
    declared_index: dict[str, int] = {}
    for index, task in enumerate(tasks):
        if task.id in by_id:
            raise ValueError(f"Duplicate task id: {task.id}")
        by_id[task.id] = task
        declared_index[task.id] = index

    dependencies: dict[str, set[str]] = {}
    for task in tasks:
        refs = collect_task_refs(task.inputs)
        unknown = refs - set(by_id)
        if unknown:
            raise ValueError(
                f"Task {task.id} references unknown task id(s): {', '.join(sorted(unknown))}"
            )
        dependencies[task.id] = refs

    return_unknown = collect_task_refs(return_value) - set(by_id)
    if return_unknown:
        raise ValueError(
            f"Workflow return references unknown task id(s): {', '.join(sorted(return_unknown))}"
        )

    remaining = {task_id: set(refs) for task_id, refs in dependencies.items()}
    ordered: list[WorkflowTask] = []
    while remaining:
        ready = sorted(
            (task_id for task_id, refs in remaining.items() if not refs),
            key=declared_index.__getitem__,
        )
        if not ready:
            cycle_ids = sorted(remaining, key=declared_index.__getitem__)
            raise ValueError(f"Workflow task reference cycle detected: {', '.join(cycle_ids)}")
        for task_id in ready:
            ordered.append(by_id[task_id])
            remaining.pop(task_id)
        completed = set(ready)
        for refs in remaining.values():
            refs.difference_update(completed)
    return ordered


def execute_workflow(payload: dict[str, Any], *, ctx: WorkflowContext | None = None) -> dict[str, Any]:
    wf = WorkflowRequest.model_validate(payload)
    context = ctx or WorkflowContext()
    ordered_tasks = _execution_order(wf.tasks, wf.return_value)

    task_results: dict[str, dict[str, Any]] = {}
    created_artifacts: set[str] = set()
    try:
        for idx, task in enumerate(ordered_tasks):
            if context.should_cancel and context.should_cancel():
                raise WorkflowCanceled("Cancel requested")
            resolved_inputs = _resolve_refs(task.inputs, task_results)
            definition = TASK_DEFINITIONS.get(task.type)
            if definition is None:
                raise ValueError(f"Unsupported task type: {task.type}")

            if context.update_progress:
                context.update_progress(
                    {
                        "current_task": task.id,
                        "current_task_index": idx,
                        "total_tasks": len(ordered_tasks),
                        "phase": "running",
                    }
                )

            # The registry is the executable contract: schemas are enforced at
            # the dispatch boundary instead of being catalog-only metadata.
            # Progress is published first so validation failures are persisted
            # against the task that actually failed.
            validated_inputs = definition.input_model.model_validate(resolved_inputs)

            task_context = context
            if context.update_progress:
                def _task_progress(patch: dict[str, Any]) -> None:
                    context.update_progress(
                        {
                            **patch,
                            "current_task": task.id,
                            "current_task_index": idx,
                            "total_tasks": len(ordered_tasks),
                        }
                    )

                task_context = WorkflowContext(
                    update_progress=_task_progress,
                    should_cancel=context.should_cancel,
                )

            result = definition.handler(
                validated_inputs.model_dump(by_alias=True),
                task_context,
            )
            if not isinstance(result, dict):
                raise ValueError(f"Task {task.id} must return an object")
            definition.output_model.model_validate(result)
            created_artifacts |= collect_artifact_ids(result)
            task_results[task.id] = result

            if context.update_progress:
                context.update_progress(
                    {
                        "current_task": task.id,
                        "current_task_index": idx,
                        "total_tasks": len(ordered_tasks),
                        "phase": "completed_task",
                    }
                )
    except Exception as exc:
        setattr(exc, "_workflow_created_artifacts", created_artifacts)
        raise

    if wf.return_value is None:
        final_value: Any = task_results[ordered_tasks[-1].id] if ordered_tasks else {}
    else:
        final_value = _resolve_refs(wf.return_value, task_results)

    return {"outputs": final_value, "tasks": task_results, "created_artifacts": sorted(created_artifacts)}
