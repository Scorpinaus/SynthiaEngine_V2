from datetime import timedelta

from sqlalchemy import update

from backend.jobs.models import Job, utcnow
from backend.jobs.store import (
    create_job_store,
    enqueue_job,
    get_job,
)


def _store(tmp_path):
    db_url = f"sqlite:///{(tmp_path / 'jobs.sqlite3').as_posix()}"
    return create_job_store(db_url)


def test_active_lease_prevents_a_second_worker_claim(tmp_path):
    engine, sessions, store = _store(tmp_path)
    try:
        enqueue_job(sessions, kind="workflow", payload={"tasks": []})
        claimed = store.claim_next(worker_id="worker-a", lease_duration_s=30)
        assert claimed is not None
        assert claimed.worker_id == "worker-a"
        assert claimed.attempt == 1

        assert store.claim_next(worker_id="worker-b", lease_duration_s=30) is None
        assert store.requeue_expired() == 0
    finally:
        engine.dispose()


def test_only_expired_leases_are_requeued_and_reclaimed(tmp_path):
    engine, sessions, store = _store(tmp_path)
    try:
        queued, _ = enqueue_job(sessions, kind="workflow", payload={"tasks": []})
        assert store.claim_next(worker_id="worker-a", lease_duration_s=30) is not None
        with sessions() as session:
            session.execute(
                update(Job)
                .where(Job.id == queued.id)
                .values(lease_expires_at=utcnow() - timedelta(seconds=1))
            )
            session.commit()

        assert store.requeue_expired() == 1
        reclaimed = store.claim_next(worker_id="worker-b", lease_duration_s=30)
        assert reclaimed is not None
        assert reclaimed.worker_id == "worker-b"
        assert reclaimed.attempt == 2
    finally:
        engine.dispose()


def test_heartbeat_requires_matching_worker_ownership(tmp_path):
    engine, sessions, store = _store(tmp_path)
    try:
        job, _ = enqueue_job(sessions, kind="workflow", payload={"tasks": []})
        assert store.claim_next(worker_id="worker-a", lease_duration_s=30) is not None
        assert store.renew_lease(
            job_id=job.id, worker_id="worker-b", lease_duration_s=30
        ) is False
        assert store.renew_lease(
            job_id=job.id, worker_id="worker-a", lease_duration_s=30
        ) is True
    finally:
        engine.dispose()


def test_terminal_state_requires_matching_worker_ownership(tmp_path):
    engine, sessions, store = _store(tmp_path)
    try:
        job, _ = enqueue_job(sessions, kind="workflow", payload={"tasks": []})
        assert store.claim_next(worker_id="worker-a", lease_duration_s=30) is not None

        assert store.mark_succeeded(job.id, {"outputs": {}}, worker_id="worker-b") is False
        assert get_job(sessions, job.id).status == "running"

        assert store.mark_succeeded(job.id, {"outputs": {}}, worker_id="worker-a") is True
        assert get_job(sessions, job.id).status == "succeeded"
    finally:
        engine.dispose()


def test_worker_skips_jobs_above_its_configured_vram_capacity(tmp_path):
    engine, sessions, store = _store(tmp_path)
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

        claimed = store.claim_next(
            worker_id="worker-a",
            lease_duration_s=30,
            worker_vram_mb=8_000,
        )
        assert claimed is not None
        assert claimed.id == small.id
    finally:
        engine.dispose()


def test_sqlite_queue_uses_wal_and_busy_timeout(tmp_path):
    engine, _sessions, _store_instance = _store(tmp_path)
    try:
        with engine.connect() as connection:
            assert connection.exec_driver_sql("PRAGMA journal_mode").scalar_one().lower() == "wal"
            assert connection.exec_driver_sql("PRAGMA busy_timeout").scalar_one() == 5000
    finally:
        engine.dispose()
