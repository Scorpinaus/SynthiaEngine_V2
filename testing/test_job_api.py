from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import backend.workflow as workflow
from backend.jobs import queue as job_queue
from backend.main import _serialize_job
from backend.utilities import resource_logging


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

    payload = _serialize_job(job).model_dump()

    assert payload["created_at"] == "2026-04-25T05:52:47.541776+00:00"
    assert payload["updated_at"] == "2026-04-25T05:52:59.197138+00:00"
    assert payload["started_at"] == "2026-04-25T05:52:47.888335+00:00"
    assert payload["finished_at"] == "2026-04-25T05:52:59.197138+00:00"


def test_execute_workflow_job_attaches_summary_profile(monkeypatch):
    partial_updates = []
    cleanup_calls = []

    monkeypatch.setattr(
        job_queue,
        "update_job_partial_result",
        lambda SessionLocal, job_id, patch: partial_updates.append((job_id, patch)),
    )
    monkeypatch.setattr(job_queue, "is_cancel_requested", lambda SessionLocal, job_id: False)
    monkeypatch.setattr(workflow, "collect_artifact_ids", lambda payload: set())
    monkeypatch.setattr(workflow, "cleanup_artifacts", lambda artifact_ids: cleanup_calls.append(set(artifact_ids)))

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

    monkeypatch.setattr(workflow, "execute_workflow", fake_execute_workflow)

    result = job_queue.execute_job(
        job_id="job-1",
        kind="workflow",
        payload={"tasks": []},
        SessionLocal=object(),
    )

    assert result["outputs"] == {"ok": True}
    assert result["tasks"] == {}
    assert "created_artifacts" not in result
    assert ("job-1", {"progress": {"phase": "running"}}) in partial_updates
    profile_updates = [patch["profile"] for _, patch in partial_updates if "profile" in patch]
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
    monkeypatch.setattr(workflow, "collect_artifact_ids", lambda _payload: {"input-artifact"})
    monkeypatch.setattr(workflow, "cleanup_artifacts", lambda ids: cleanup_calls.append(set(ids)))
    monkeypatch.setattr(workflow, "execute_workflow", fail_workflow)

    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        job_queue.execute_job(
            job_id="job-1",
            kind="workflow",
            payload={"tasks": []},
            SessionLocal=object(),
        )

    assert cleanup_calls == [{"input-artifact", "created-artifact"}]
