from datetime import timedelta

from sqlalchemy import update

from backend.jobs.models import Job, utcnow
from backend.jobs.queue import (
    JobQueueConfig,
    _claim_next_job,
    create_job_queue,
    enqueue_job,
    renew_job_lease,
    requeue_expired_jobs,
)


def _queue(tmp_path):
    db_url = f"sqlite:///{(tmp_path / 'jobs.sqlite3').as_posix()}"
    return create_job_queue(JobQueueConfig(db_url=db_url))


def test_active_lease_prevents_a_second_worker_claim(tmp_path):
    engine, sessions, _worker = _queue(tmp_path)
    try:
        enqueue_job(sessions, kind="workflow", payload={"tasks": []})
        with sessions() as session:
            claimed = _claim_next_job(session, worker_id="worker-a", lease_duration_s=30)
        assert claimed is not None
        assert claimed.worker_id == "worker-a"
        assert claimed.attempt == 1

        with sessions() as session:
            assert _claim_next_job(session, worker_id="worker-b", lease_duration_s=30) is None
        assert requeue_expired_jobs(sessions) == 0
    finally:
        engine.dispose()


def test_only_expired_leases_are_requeued_and_reclaimed(tmp_path):
    engine, sessions, _worker = _queue(tmp_path)
    try:
        queued, _ = enqueue_job(sessions, kind="workflow", payload={"tasks": []})
        with sessions() as session:
            _claim_next_job(session, worker_id="worker-a", lease_duration_s=30)
            session.execute(
                update(Job)
                .where(Job.id == queued.id)
                .values(lease_expires_at=utcnow() - timedelta(seconds=1))
            )
            session.commit()

        assert requeue_expired_jobs(sessions) == 1
        with sessions() as session:
            reclaimed = _claim_next_job(session, worker_id="worker-b", lease_duration_s=30)
        assert reclaimed is not None
        assert reclaimed.worker_id == "worker-b"
        assert reclaimed.attempt == 2
    finally:
        engine.dispose()


def test_heartbeat_requires_matching_worker_ownership(tmp_path):
    engine, sessions, _worker = _queue(tmp_path)
    try:
        job, _ = enqueue_job(sessions, kind="workflow", payload={"tasks": []})
        with sessions() as session:
            _claim_next_job(session, worker_id="worker-a", lease_duration_s=30)
        assert renew_job_lease(
            sessions, job_id=job.id, worker_id="worker-b", lease_duration_s=30
        ) is False
        assert renew_job_lease(
            sessions, job_id=job.id, worker_id="worker-a", lease_duration_s=30
        ) is True
    finally:
        engine.dispose()


def test_worker_skips_jobs_above_its_configured_vram_capacity(tmp_path):
    engine, sessions, _worker = _queue(tmp_path)
    try:
        large, _ = enqueue_job(
            sessions,
            kind="workflow",
            payload={"tasks": [{"id": "video", "type": "wan.text2video", "inputs": {"num_frames": 81}}]},
        )
        small, _ = enqueue_job(
            sessions,
            kind="workflow",
            payload={"tasks": [{"id": "image", "type": "sd15.text2img", "inputs": {"width": 512, "height": 512}}]},
        )
        assert large.resource_requirements["estimated_vram_mb"] == 20_000
        assert small.resource_requirements["estimated_vram_mb"] == 6_000

        with sessions() as session:
            claimed = _claim_next_job(
                session,
                worker_id="worker-a",
                lease_duration_s=30,
                worker_vram_mb=8_000,
            )
        assert claimed is not None
        assert claimed.id == small.id
    finally:
        engine.dispose()


def test_sqlite_queue_uses_wal_and_busy_timeout(tmp_path):
    engine, _sessions, _worker = _queue(tmp_path)
    try:
        with engine.connect() as connection:
            assert connection.exec_driver_sql("PRAGMA journal_mode").scalar_one().lower() == "wal"
            assert connection.exec_driver_sql("PRAGMA busy_timeout").scalar_one() == 5000
    finally:
        engine.dispose()
