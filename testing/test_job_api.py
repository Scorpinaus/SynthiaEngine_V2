from datetime import datetime, timezone
from types import SimpleNamespace

import backend.workflow as workflow
from backend.jobs import queue as job_queue
from backend.main import _serialize_job


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
    progress_patches = []
    cleanup_calls = []

    monkeypatch.setattr(
        job_queue,
        "update_job_partial_result",
        lambda SessionLocal, job_id, patch: progress_patches.append((job_id, patch)),
    )
    monkeypatch.setattr(job_queue, "is_cancel_requested", lambda SessionLocal, job_id: False)
    monkeypatch.setattr(workflow, "collect_artifact_ids", lambda payload: set())
    monkeypatch.setattr(workflow, "cleanup_artifacts", lambda artifact_ids: cleanup_calls.append(set(artifact_ids)))

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
    assert progress_patches == [("job-1", {"progress": {"phase": "running"}})]
    assert cleanup_calls == [{"a123"}]

    profile = result["profile"]
    assert profile["schema_version"] == 1
    assert profile["elapsed_seconds"] >= 0
    assert profile["rss_before_mb"] is None or profile["rss_before_mb"] > 0
    assert profile["rss_after_mb"] is None or profile["rss_after_mb"] > 0
    assert isinstance(profile["cuda_available"], bool)
    assert "cuda_peak_allocated_mb" in profile
    assert "cuda_peak_reserved_mb" in profile
    assert isinstance(profile["nvml_available"], bool)
    assert "nvml_used_start_mb" in profile
    assert "nvml_used_end_mb" in profile
    assert "nvml_used_peak_sampled_mb" in profile
