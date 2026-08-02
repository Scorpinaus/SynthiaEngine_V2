from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from backend.api.jobs import serialize_job
from backend.jobs.execution import WorkflowJobExecutor
from backend.utilities import resource_logging
from backend.workflow import engine as workflow_engine
from backend.workflow import utility as workflow_utility


class FakeExecutionStore:
    def __init__(self, *, cancel_requested: bool = False) -> None:
        self.cancel_requested = cancel_requested
        self.partial_updates: list[tuple[str, dict[str, Any]]] = []

    def is_cancel_requested(self, job_id: str) -> bool:
        return self.cancel_requested

    def record_progress(self, job_id: str, progress: dict[str, Any]) -> None:
        self.partial_updates.append((job_id, {"progress": progress}))

    def merge_partial_result(self, job_id: str, patch: dict[str, Any]) -> None:
        self.partial_updates.append((job_id, patch))


def test_serialize_job_timestamps_include_timezone_for_sqlite_naive_datetimes():
    job = SimpleNamespace(
        id="job-1",
        idempotency_key=None,
        cancel_requested=False,
        kind="workflow",
        status="queued",
        payload={"tasks": []},
        result=None,
        error=None,
        created_at=datetime(2026, 4, 25, 5, 52, 47, 541776),
        updated_at=datetime(2026, 4, 25, 5, 52, 59, 197138),
        started_at=datetime(2026, 4, 25, 5, 52, 47, 888335),
        finished_at=datetime(2026, 4, 25, 5, 52, 59, 197138, tzinfo=timezone.utc),
    )

    payload = serialize_job(job).model_dump()

    assert payload["created_at"] == "2026-04-25T05:52:47.541776+00:00"
    assert payload["updated_at"] == "2026-04-25T05:52:59.197138+00:00"
    assert payload["started_at"] == "2026-04-25T05:52:47.888335+00:00"
    assert payload["finished_at"] == "2026-04-25T05:52:59.197138+00:00"


def test_execute_workflow_job_attaches_summary_profile(monkeypatch):
    store = FakeExecutionStore()
    cleanup_calls = []
    monkeypatch.setattr(workflow_utility, "collect_artifact_ids", lambda payload: set())
    monkeypatch.setattr(
        workflow_utility,
        "cleanup_artifacts",
        lambda artifact_ids: cleanup_calls.append(set(artifact_ids)),
    )

    class FakeSummaryProfiler:
        def __init__(self, *, on_update=None):
            self.on_update = on_update
            self.profile = None

        def __enter__(self):
            if self.on_update is not None:
                self.on_update(
                    {
                        "schema_version": 1,
                        "elapsed_seconds": 0.1,
                        "rss_current_mb": 10.0,
                        "nvml_used_current_mb": 20.0,
                    }
                )
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.profile = {
                "schema_version": 1,
                "elapsed_seconds": 1.0,
                "rss_before_mb": 10.0,
                "rss_current_mb": 12.0,
                "rss_after_mb": 12.0,
                "rss_peak_sampled_mb": 12.0,
                "cuda_available": False,
                "cuda_allocated_current_mb": None,
                "cuda_reserved_current_mb": None,
                "cuda_peak_allocated_mb": None,
                "cuda_peak_reserved_mb": None,
                "nvml_available": False,
                "nvml_device_index": None,
                "nvml_used_start_mb": None,
                "nvml_used_current_mb": None,
                "nvml_used_end_mb": None,
                "nvml_used_peak_sampled_mb": None,
            }
            if self.on_update is not None:
                self.on_update(self.profile)

    monkeypatch.setattr(resource_logging, "SummaryProfiler", FakeSummaryProfiler)

    def fake_execute_workflow(payload, *, ctx=None):
        assert payload == {"tasks": []}
        assert ctx is not None
        assert ctx.should_cancel() is False
        ctx.update_progress({"phase": "running"})
        return {
            "outputs": {"ok": True},
            "tasks": {},
            "created_artifacts": ["a123"],
        }

    monkeypatch.setattr(workflow_engine, "execute_workflow", fake_execute_workflow)

    result = WorkflowJobExecutor(store).execute(
        job_id="job-1",
        kind="workflow",
        payload={"tasks": []},
    )

    assert result["outputs"] == {"ok": True}
    assert result["tasks"] == {}
    assert "created_artifacts" not in result
    assert ("job-1", {"progress": {"phase": "running"}}) in store.partial_updates
    profile_updates = [
        patch["profile"] for _, patch in store.partial_updates if "profile" in patch
    ]
    assert profile_updates[0]["rss_current_mb"] == 10.0
    assert profile_updates[-1]["elapsed_seconds"] == 1.0
    assert cleanup_calls == [{"a123"}]

    profile = result["profile"]
    assert profile["schema_version"] == 1
    assert profile["elapsed_seconds"] == 1.0
    assert profile["rss_before_mb"] == 10.0
    assert profile["rss_current_mb"] == 12.0
    assert profile["rss_after_mb"] == 12.0
    assert profile["rss_peak_sampled_mb"] == 12.0
    assert isinstance(profile["cuda_available"], bool)
    assert "cuda_allocated_current_mb" in profile
    assert "cuda_reserved_current_mb" in profile
    assert "cuda_peak_allocated_mb" in profile
    assert "cuda_peak_reserved_mb" in profile
    assert isinstance(profile["nvml_available"], bool)
    assert "nvml_used_start_mb" in profile
    assert "nvml_used_current_mb" in profile
    assert "nvml_used_end_mb" in profile
    assert "nvml_used_peak_sampled_mb" in profile


def test_execute_workflow_job_cleans_input_and_created_artifacts_after_failure(monkeypatch):
    store = FakeExecutionStore()
    cleanup_calls = []

    class FakeSummaryProfiler:
        profile = None

        def __init__(self, *, on_update=None):
            self.on_update = on_update

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    def fail_workflow(_payload, *, ctx=None):
        error = RuntimeError("synthetic generation failure")
        error._workflow_created_artifacts = {"created-artifact"}
        raise error

    monkeypatch.setattr(resource_logging, "SummaryProfiler", FakeSummaryProfiler)
    monkeypatch.setattr(
        workflow_utility,
        "collect_artifact_ids",
        lambda _payload: {"input-artifact"},
    )
    monkeypatch.setattr(
        workflow_utility,
        "cleanup_artifacts",
        lambda ids: cleanup_calls.append(set(ids)),
    )
    monkeypatch.setattr(workflow_engine, "execute_workflow", fail_workflow)

    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        WorkflowJobExecutor(store).execute(
            job_id="job-1",
            kind="workflow",
            payload={"tasks": []},
        )

    assert cleanup_calls == [{"input-artifact", "created-artifact"}]


def test_execution_failure_takes_precedence_when_artifact_cleanup_also_fails(
    monkeypatch,
    caplog,
):
    class FakeSummaryProfiler:
        profile = None

        def __init__(self, *, on_update=None):
            self.on_update = on_update

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    def fail_workflow(_payload, *, ctx=None):
        raise RuntimeError("render failed first")

    def fail_cleanup(_artifact_ids):
        raise OSError("cleanup failed second")

    monkeypatch.setattr(resource_logging, "SummaryProfiler", FakeSummaryProfiler)
    monkeypatch.setattr(workflow_utility, "collect_artifact_ids", lambda _payload: set())
    monkeypatch.setattr(workflow_engine, "execute_workflow", fail_workflow)

    with pytest.raises(RuntimeError, match="render failed first"):
        WorkflowJobExecutor(
            FakeExecutionStore(),
            artifact_cleanup=fail_cleanup,
        ).execute(
            job_id="job-1",
            kind="workflow",
            payload={"tasks": []},
        )

    assert "Artifact cleanup also failed" in caplog.text
    assert "cleanup failed second" in caplog.text
