from __future__ import annotations

from backend.workflow.registry import TaskDefinition, TaskHandler, bind_task
from backend.workflow.schema_input import WanImage2VideoInputs, WanText2VideoInputs
from backend.workflow.schema_output import VideosWithBatchOutput


def task_definitions(handlers: dict[str, TaskHandler]) -> dict[str, TaskDefinition]:
    contracts = {
        "wan.text2video": WanText2VideoInputs,
        "wan.image2video": WanImage2VideoInputs,
    }
    return {
        name: bind_task(handlers, name, input_model, VideosWithBatchOutput)
        for name, input_model in contracts.items()
    }
