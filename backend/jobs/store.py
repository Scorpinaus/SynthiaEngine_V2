from __future__ import annotations

import logging
import threading
import uuid
from datetime import timedelta
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from backend.jobs.contracts import ClaimedJob
from backend.jobs.db import JobDbConfig, create_job_engine, create_sessionmaker
from backend.jobs.models import Base, Job, JobTask, utcnow


logger = logging.getLogger(__name__)
JOB_RESULT_UPDATE_LOCK = threading.Lock()


class JobNotFoundError(Exception):
    pass


class IdempotencyConflictError(Exception):
    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


def _sqlite_column_exists(engine: Engine, *, table: str, column: str) -> bool:
    try:
        with engine.connect() as conn:
            rows = conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
    except Exception:
        return False
    return any(str(row[1]) == column for row in rows)


def _sqlite_table_exists(engine: Engine, *, table: str) -> bool:
    try:
        with engine.connect() as conn:
            row = conn.exec_driver_sql(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
                (table,),
            ).fetchone()
            return row is not None
    except Exception:
        return False


def _sqlite_ensure_idempotency_schema(engine: Engine) -> None:
    if not str(engine.url).startswith("sqlite:") or not _sqlite_table_exists(engine, table="jobs"):
        return
    with engine.begin() as conn:
        if not _sqlite_column_exists(engine, table="jobs", column="idempotency_key"):
            conn.exec_driver_sql("ALTER TABLE jobs ADD COLUMN idempotency_key VARCHAR(128)")
        conn.exec_driver_sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_jobs_idempotency_key "
            "ON jobs(idempotency_key) WHERE idempotency_key IS NOT NULL"
        )


def _sqlite_ensure_cancel_schema(engine: Engine) -> None:
    if not str(engine.url).startswith("sqlite:") or not _sqlite_table_exists(engine, table="jobs"):
        return
    with engine.begin() as conn:
        if not _sqlite_column_exists(engine, table="jobs", column="cancel_requested"):
            conn.exec_driver_sql(
                "ALTER TABLE jobs ADD COLUMN cancel_requested BOOLEAN NOT NULL DEFAULT 0"
            )
        conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_jobs_cancel_requested ON jobs(cancel_requested)"
        )


def _sqlite_ensure_worker_lease_schema(engine: Engine) -> None:
    if not str(engine.url).startswith("sqlite:") or not _sqlite_table_exists(engine, table="jobs"):
        return
    columns = {
        "worker_id": "VARCHAR(64)",
        "heartbeat_at": "DATETIME",
        "lease_expires_at": "DATETIME",
        "attempt": "INTEGER NOT NULL DEFAULT 0",
        "resource_requirements": "JSON NOT NULL DEFAULT '{}'",
    }
    with engine.begin() as conn:
        for name, declaration in columns.items():
            if not _sqlite_column_exists(engine, table="jobs", column=name):
                conn.exec_driver_sql(f"ALTER TABLE jobs ADD COLUMN {name} {declaration}")
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_jobs_worker_id ON jobs(worker_id)")
        conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_jobs_lease_expires_at ON jobs(lease_expires_at)"
        )


def init_job_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)
    try:
        _sqlite_ensure_idempotency_schema(engine)
        _sqlite_ensure_cancel_schema(engine)
        _sqlite_ensure_worker_lease_schema(engine)
    except OperationalError as exc:
        logger.warning("Failed to ensure sqlite schema: %s", exc)


_FAMILY_VRAM_ESTIMATES_MB = {
    "sd15": 6_000,
    "sdxl": 10_000,
    "flux": 12_000,
    "qwen-image": 16_000,
    "z-image": 12_000,
    "ernie-image": 16_000,
    "anima": 12_000,
    "wan": 20_000,
}


def estimate_job_resources(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    if kind != "workflow":
        return {}
    families: set[str] = set()
    max_pixels = 0
    max_frames = 1
    requires_video = False
    tasks = payload.get("tasks")
    if isinstance(tasks, list):
        for task in tasks:
            if not isinstance(task, dict):
                continue
            task_type = str(task.get("type") or "")
            family = task_type.split(".", 1)[0]
            if family in _FAMILY_VRAM_ESTIMATES_MB:
                families.add(family)
            inputs = task.get("inputs") if isinstance(task.get("inputs"), dict) else {}
            width = inputs.get("width")
            height = inputs.get("height")
            if isinstance(width, (int, float)) and isinstance(height, (int, float)):
                max_pixels = max(max_pixels, int(width) * int(height))
            frames = inputs.get("num_frames")
            if isinstance(frames, (int, float)):
                max_frames = max(max_frames, int(frames))
            requires_video = requires_video or "video" in task_type
    estimated_vram_mb = max((_FAMILY_VRAM_ESTIMATES_MB[f] for f in families), default=0)
    return {
        "device": "cuda" if families else "cpu",
        "families": sorted(families),
        "estimated_vram_mb": estimated_vram_mb,
        "max_pixels": max_pixels,
        "max_frames": max_frames,
        "requires_video": requires_video,
    }


def enqueue_job(
    SessionLocal: sessionmaker,
    *,
    kind: str,
    payload: dict[str, Any],
    idempotency_key: str | None = None,
) -> tuple[Job, bool]:
    now = utcnow()
    with SessionLocal() as session:
        if idempotency_key:
            existing = session.execute(
                select(Job).where(Job.idempotency_key == idempotency_key).limit(1)
            ).scalar_one_or_none()
            if existing is not None:
                if existing.kind != kind or dict(existing.payload or {}) != payload:
                    raise IdempotencyConflictError(idempotency_key)
                return existing, False

        job = Job(
            id=str(uuid.uuid4()),
            idempotency_key=idempotency_key,
            kind=kind,
            status="queued",
            payload=payload,
            resource_requirements=estimate_job_resources(kind, payload),
            created_at=now,
            updated_at=now,
        )
        session.add(job)
        if kind == "workflow" and isinstance(payload.get("tasks"), list):
            for index, task in enumerate(payload["tasks"]):
                if not isinstance(task, dict):
                    continue
                task_id = task.get("id")
                task_type = task.get("type")
                if not isinstance(task_id, str) or not isinstance(task_type, str):
                    continue
                inputs = task.get("inputs")
                session.add(
                    JobTask(
                        job_id=job.id,
                        task_id=task_id,
                        task_type=task_type,
                        task_index=index,
                        status="queued",
                        inputs=dict(inputs) if isinstance(inputs, dict) else {},
                        created_at=now,
                    )
                )
        session.commit()
        session.refresh(job)
        return job, True


def get_job(SessionLocal: sessionmaker, job_id: str) -> Job:
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        return job


def list_jobs(SessionLocal: sessionmaker, *, limit: int = 50) -> list[Job]:
    with SessionLocal() as session:
        return list(
            session.execute(select(Job).order_by(Job.created_at.desc()).limit(limit)).scalars().all()
        )


def list_job_tasks(SessionLocal: sessionmaker, job_id: str) -> list[JobTask]:
    with SessionLocal() as session:
        if session.get(Job, job_id) is None:
            raise JobNotFoundError(job_id)
        return list(
            session.execute(
                select(JobTask)
                .where(JobTask.job_id == job_id)
                .order_by(JobTask.task_index.asc())
            ).scalars().all()
        )


def update_job_task_progress(
    SessionLocal: sessionmaker,
    job_id: str,
    progress: dict[str, Any],
) -> None:
    task_id = progress.get("current_task")
    phase = progress.get("phase")
    if not isinstance(task_id, str) or phase not in {"running", "completed_task"}:
        return
    now = utcnow()
    values = (
        {"status": "running", "started_at": now, "error": None}
        if phase == "running"
        else {"status": "succeeded", "finished_at": now, "error": None}
    )
    with SessionLocal() as session:
        session.execute(
            update(JobTask)
            .where(JobTask.job_id == job_id, JobTask.task_id == task_id)
            .values(**values)
        )
        session.commit()


def finalize_job_tasks(
    session: Session,
    job_id: str,
    *,
    status: str,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    now = utcnow()
    if status == "succeeded":
        task_outputs = (result or {}).get("tasks")
        if isinstance(task_outputs, dict):
            for task_id, output in task_outputs.items():
                session.execute(
                    update(JobTask)
                    .where(JobTask.job_id == job_id, JobTask.task_id == str(task_id))
                    .values(
                        status="succeeded",
                        output=output if isinstance(output, dict) else {"value": output},
                        finished_at=now,
                        error=None,
                    )
                )
        return

    active_status = "canceled" if status == "canceled" else "failed"
    session.execute(
        update(JobTask)
        .where(JobTask.job_id == job_id, JobTask.status == "running")
        .values(status=active_status, error=error, finished_at=now)
    )
    session.execute(
        update(JobTask)
        .where(JobTask.job_id == job_id, JobTask.status == "queued")
        .values(status="canceled" if status == "canceled" else "skipped", finished_at=now)
    )


def cancel_job(SessionLocal: sessionmaker, job_id: str) -> Job:
    now = utcnow()
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        if job.status != "queued":
            return job
        job.status = "canceled"
        job.updated_at = now
        job.finished_at = now
        finalize_job_tasks(session, job_id, status="canceled")
        session.commit()
        session.refresh(job)
        return job


def request_cancel_job(SessionLocal: sessionmaker, job_id: str) -> Job:
    now = utcnow()
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        if job.status in {"succeeded", "failed", "canceled"}:
            return job
        job.cancel_requested = True
        job.updated_at = now
        if job.status == "queued":
            job.status = "canceled"
            job.finished_at = now
            finalize_job_tasks(session, job_id, status="canceled")
        session.commit()
        session.refresh(job)
        return job


def is_cancel_requested(SessionLocal: sessionmaker, job_id: str) -> bool:
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        return bool(getattr(job, "cancel_requested", False)) if job is not None else False


def requeue_expired_jobs(SessionLocal: sessionmaker) -> int:
    now = utcnow()
    with SessionLocal() as session:
        expired_ids = list(
            session.execute(
                select(Job.id)
                .where(Job.status == "running")
                .where(Job.lease_expires_at.is_not(None))
                .where(Job.lease_expires_at < now)
            ).scalars()
        )
        if not expired_ids:
            return 0
        updated = session.execute(
            update(Job)
            .where(Job.id.in_(expired_ids), Job.status == "running")
            .where(Job.lease_expires_at.is_not(None), Job.lease_expires_at < now)
            .values(
                status="queued",
                updated_at=now,
                started_at=None,
                worker_id=None,
                heartbeat_at=None,
                lease_expires_at=None,
            )
        )
        if updated.rowcount:
            session.execute(
                update(JobTask)
                .where(JobTask.job_id.in_(expired_ids))
                .where(JobTask.status.in_({"running", "succeeded", "failed", "skipped"}))
                .values(status="queued", output=None, error=None, started_at=None, finished_at=None)
            )
        session.commit()
        return int(updated.rowcount or 0)


def _claim_next_job(
    session: Session,
    *,
    worker_id: str,
    lease_duration_s: float,
    worker_vram_mb: int = 0,
) -> Job | None:
    candidates = session.execute(
        select(Job.id, Job.resource_requirements)
        .where(Job.status == "queued")
        .order_by(Job.created_at.asc())
        .limit(100)
    ).all()
    job_id = None
    for candidate_id, requirements in candidates:
        estimated = int((requirements or {}).get("estimated_vram_mb") or 0)
        if worker_vram_mb <= 0 or estimated <= worker_vram_mb:
            job_id = candidate_id
            break
    if job_id is None:
        return None

    now = utcnow()
    running_exists = select(Job.id).where(Job.status == "running").exists()
    try:
        updated = session.execute(
            update(Job)
            .where(Job.id == job_id, Job.status == "queued")
            .where(~running_exists)
            .values(
                status="running",
                started_at=now,
                updated_at=now,
                worker_id=worker_id,
                heartbeat_at=now,
                lease_expires_at=now + timedelta(seconds=lease_duration_s),
                attempt=Job.attempt + 1,
            )
        )
        if (updated.rowcount or 0) != 1:
            session.rollback()
            return None
        session.commit()
        return session.get(Job, job_id)
    except Exception:
        session.rollback()
        return None


def renew_job_lease(
    SessionLocal: sessionmaker,
    *,
    job_id: str,
    worker_id: str,
    lease_duration_s: float,
) -> bool:
    now = utcnow()
    with SessionLocal() as session:
        updated = session.execute(
            update(Job)
            .where(Job.id == job_id, Job.status == "running", Job.worker_id == worker_id)
            .values(
                heartbeat_at=now,
                lease_expires_at=now + timedelta(seconds=lease_duration_s),
                updated_at=now,
            )
        )
        session.commit()
        return (updated.rowcount or 0) == 1


def _owned_job_update(job_id: str, worker_id: str | None):
    statement = update(Job).where(Job.id == job_id)
    return statement.where(Job.worker_id == worker_id) if worker_id is not None else statement


def _mark_job_failed(
    session: Session,
    job_id: str,
    message: str,
    *,
    worker_id: str | None = None,
) -> bool:
    now = utcnow()
    updated = session.execute(
        _owned_job_update(job_id, worker_id).values(
            status="failed",
            error=message,
            updated_at=now,
            finished_at=now,
            worker_id=None,
            heartbeat_at=None,
            lease_expires_at=None,
        )
    )
    if (updated.rowcount or 0) != 1:
        session.rollback()
        return False
    finalize_job_tasks(session, job_id, status="failed", error=message)
    session.commit()
    return True


def _mark_job_canceled(
    session: Session,
    job_id: str,
    *,
    worker_id: str | None = None,
) -> bool:
    now = utcnow()
    updated = session.execute(
        _owned_job_update(job_id, worker_id).values(
            status="canceled",
            error=None,
            updated_at=now,
            finished_at=now,
            worker_id=None,
            heartbeat_at=None,
            lease_expires_at=None,
        )
    )
    if (updated.rowcount or 0) != 1:
        session.rollback()
        return False
    finalize_job_tasks(session, job_id, status="canceled")
    session.commit()
    return True


def _mark_job_succeeded(
    session: Session,
    job_id: str,
    result: dict[str, Any],
    *,
    worker_id: str | None = None,
) -> bool:
    now = utcnow()
    updated = session.execute(
        _owned_job_update(job_id, worker_id).values(
            status="succeeded",
            result=result,
            updated_at=now,
            finished_at=now,
            error=None,
            worker_id=None,
            heartbeat_at=None,
            lease_expires_at=None,
        )
    )
    if (updated.rowcount or 0) != 1:
        session.rollback()
        return False
    finalize_job_tasks(session, job_id, status="succeeded", result=result)
    session.commit()
    return True


def update_job_partial_result(
    SessionLocal: sessionmaker,
    job_id: str,
    patch: dict[str, Any],
) -> None:
    now = utcnow()
    with JOB_RESULT_UPDATE_LOCK:
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            if job is None:
                return
            current = dict(job.result or {})
            current.update(patch)
            job.result = current
            job.updated_at = now
            session.commit()


class SqlAlchemyJobStore:
    """SQLAlchemy implementation of the complete job persistence boundary."""

    def __init__(self, *, engine: Engine, SessionLocal: sessionmaker) -> None:
        self.engine = engine
        self.SessionLocal = SessionLocal

    def initialize(self) -> None:
        init_job_db(self.engine)

    def requeue_expired(self) -> int:
        return requeue_expired_jobs(self.SessionLocal)

    def claim_next(
        self,
        *,
        worker_id: str,
        lease_duration_s: float,
        worker_vram_mb: int = 0,
    ) -> ClaimedJob | None:
        with self.SessionLocal() as session:
            job = _claim_next_job(
                session,
                worker_id=worker_id,
                lease_duration_s=lease_duration_s,
                worker_vram_mb=worker_vram_mb,
            )
            if job is None:
                return None
            return ClaimedJob(
                id=job.id,
                kind=job.kind,
                payload=dict(job.payload or {}),
                cancel_requested=bool(job.cancel_requested),
                worker_id=str(job.worker_id),
                attempt=int(job.attempt),
                resource_requirements=dict(job.resource_requirements or {}),
            )

    def renew_lease(
        self,
        *,
        job_id: str,
        worker_id: str,
        lease_duration_s: float,
    ) -> bool:
        return renew_job_lease(
            self.SessionLocal,
            job_id=job_id,
            worker_id=worker_id,
            lease_duration_s=lease_duration_s,
        )

    def mark_succeeded(
        self,
        job_id: str,
        result: dict[str, Any],
        *,
        worker_id: str,
    ) -> bool:
        with self.SessionLocal() as session:
            return _mark_job_succeeded(session, job_id, result, worker_id=worker_id)

    def mark_failed(self, job_id: str, message: str, *, worker_id: str) -> bool:
        with self.SessionLocal() as session:
            return _mark_job_failed(session, job_id, message, worker_id=worker_id)

    def mark_canceled(self, job_id: str, *, worker_id: str) -> bool:
        with self.SessionLocal() as session:
            return _mark_job_canceled(session, job_id, worker_id=worker_id)

    def is_cancel_requested(self, job_id: str) -> bool:
        return is_cancel_requested(self.SessionLocal, job_id)

    def record_progress(self, job_id: str, progress: dict[str, Any]) -> None:
        update_job_task_progress(self.SessionLocal, job_id, progress)
        update_job_partial_result(self.SessionLocal, job_id, {"progress": progress})

    def merge_partial_result(self, job_id: str, patch: dict[str, Any]) -> None:
        update_job_partial_result(self.SessionLocal, job_id, patch)


def create_job_store(db_url: str) -> tuple[Engine, sessionmaker, SqlAlchemyJobStore]:
    engine = create_job_engine(JobDbConfig(url=db_url))
    SessionLocal = create_sessionmaker(engine)
    store = SqlAlchemyJobStore(engine=engine, SessionLocal=SessionLocal)
    store.initialize()
    return engine, SessionLocal, store

