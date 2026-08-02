from pydantic import BaseModel
import pytest

from backend.workflow.engine import TASK_DEFINITIONS, execute_workflow
from backend.workflow.registry import TaskDefinition
from backend.workflow.utility import _resolve_refs


class _Inputs(BaseModel):
    value: object | None = None


class _Output(BaseModel):
    value: object | None = None


def _definition(handler):
    return TaskDefinition(_Inputs, _Output, handler)


def test_nested_and_indexed_task_references_resolve():
    results = {"source": {"images": ["one.png", "two.png"], "nested": {"seed": 42}}}
    assert _resolve_refs("@source.images[1]", results) == "two.png"
    assert _resolve_refs("@source.nested.seed", results) == 42


def test_forward_references_execute_in_stable_topological_order(monkeypatch):
    calls = []
    definitions = {
        "test.source": _definition(lambda inputs, _ctx: calls.append("source") or {"value": "ready"}),
        "test.consumer": _definition(lambda inputs, _ctx: calls.append("consumer") or {"value": inputs["value"]}),
        "test.independent": _definition(lambda inputs, _ctx: calls.append("independent") or {"value": "other"}),
    }
    monkeypatch.setattr("backend.workflow.engine.TASK_DEFINITIONS", definitions)

    result = execute_workflow(
        {
            "tasks": [
                {"id": "consumer", "type": "test.consumer", "inputs": {"value": "@source.value"}},
                {"id": "independent", "type": "test.independent", "inputs": {}},
                {"id": "source", "type": "test.source", "inputs": {}},
            ],
            "return": "@consumer.value",
        }
    )
    assert calls == ["independent", "source", "consumer"]
    assert result["outputs"] == "ready"


def test_unknown_reference_fails_before_any_task_executes(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "backend.workflow.engine.TASK_DEFINITIONS",
        {"test.task": _definition(lambda _inputs, _ctx: calls.append("called") or {"value": None})},
    )
    with pytest.raises(ValueError, match="references unknown task id"):
        execute_workflow(
            {"tasks": [{"id": "task", "type": "test.task", "inputs": {"value": "@missing.value"}}]}
        )
    assert calls == []


def test_reference_cycle_reports_all_involved_tasks(monkeypatch):
    monkeypatch.setattr(
        "backend.workflow.engine.TASK_DEFINITIONS",
        {"test.task": _definition(lambda inputs, _ctx: {"value": inputs.get("value")})},
    )
    with pytest.raises(ValueError, match="reference cycle detected: first, second"):
        execute_workflow(
            {
                "tasks": [
                    {"id": "first", "type": "test.task", "inputs": {"value": "@second.value"}},
                    {"id": "second", "type": "test.task", "inputs": {"value": "@first.value"}},
                ]
            }
        )
