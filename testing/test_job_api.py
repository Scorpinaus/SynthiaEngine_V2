from datetime import datetime, timezone
from types import SimpleNamespace

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
