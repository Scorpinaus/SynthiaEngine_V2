from unittest.mock import patch

import pytest
from pydantic import ValidationError

from backend.workflow import (
    TASK_DEFINITIONS,
    TASK_INPUT_MODELS,
    TASK_OUTPUT_MODELS,
    TASK_REGISTRY,
    WorkflowRequest,
    build_workflow_catalog,
    execute_workflow,
)


def test_task_contract_views_are_derived_from_authoritative_definitions():
    expected = set(TASK_DEFINITIONS)
    assert set(TASK_REGISTRY) == expected
    assert set(TASK_INPUT_MODELS) == expected
    assert set(TASK_OUTPUT_MODELS) == expected
    assert set(build_workflow_catalog()["tasks"]) == expected


def test_every_registered_task_is_accepted_by_workflow_request_validation():
    for task_type in TASK_DEFINITIONS:
        request = WorkflowRequest.model_validate(
            {"tasks": [{"id": "task", "type": task_type, "inputs": {}}]}
        )
        assert request.tasks[0].type == task_type


def test_anima_task_is_reachable_through_dispatch():
    with patch("backend.workflow.engine._anima_text2img", return_value={"images": ["/outputs/a.png"]}):
        # Patch the immutable definition through a replacement compatibility-safe
        # definition so this remains a dispatch test without loading a model.
        definition = TASK_DEFINITIONS["anima.text2img"]
        replacement = type(definition)(definition.input_model, definition.output_model, lambda _i, _c: {"images": ["/outputs/a.png"]})
        with patch.dict(TASK_DEFINITIONS, {"anima.text2img": replacement}):
            result = execute_workflow(
                {"tasks": [{"id": "a", "type": "anima.text2img", "inputs": {"prompt": "test"}}]}
            )
    assert result["outputs"]["images"] == ["/outputs/a.png"]


def test_registered_input_schema_is_enforced_before_handler_runs():
    with pytest.raises(ValidationError):
        execute_workflow(
            {"tasks": [{"id": "a", "type": "anima.text2img", "inputs": {"width": 1}}]}
        )
