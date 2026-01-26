from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import OperationalError

from backend.job_db import JobDbConfig, create_job_engine, create_sessionmaker
from backend.job_models import Base, Job, utcnow

logger = logging.getLogger(__name__)

EXECUTION_LOCK = threading.Lock()


class JobNotFoundError(Exception):
    pass


class IdempotencyConflictError(Exception):
    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


@dataclass(frozen=True)
class JobQueueConfig:
    db_url: str = "sqlite:///outputs/jobs.sqlite3"
    poll_interval_s: float = 0.5
    requeue_running_on_startup: bool = True


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
    if not str(engine.url).startswith("sqlite:"):
        return
    if not _sqlite_table_exists(engine, table="jobs"):
        return
    with engine.begin() as conn:
        if not _sqlite_column_exists(engine, table="jobs", column="idempotency_key"):
            conn.exec_driver_sql("ALTER TABLE jobs ADD COLUMN idempotency_key VARCHAR(128)")
        conn.exec_driver_sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_jobs_idempotency_key "
            "ON jobs(idempotency_key) WHERE idempotency_key IS NOT NULL"
        )


def _sqlite_ensure_cancel_schema(engine: Engine) -> None:
    if not str(engine.url).startswith("sqlite:"):
        return
    if not _sqlite_table_exists(engine, table="jobs"):
        return

    with engine.begin() as conn:
        if not _sqlite_column_exists(engine, table="jobs", column="cancel_requested"):
            conn.exec_driver_sql("ALTER TABLE jobs ADD COLUMN cancel_requested BOOLEAN NOT NULL DEFAULT 0")
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_jobs_cancel_requested ON jobs(cancel_requested)")


def init_job_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)
    try:
        _sqlite_ensure_idempotency_schema(engine)
        _sqlite_ensure_cancel_schema(engine)
    except OperationalError as exc:
        logger.warning("Failed to ensure sqlite schema: %s", exc)


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
                existing_payload = dict(existing.payload or {})
                if existing.kind != kind or existing_payload != payload:
                    raise IdempotencyConflictError(idempotency_key)
                return existing, False

        job = Job(
            id=str(uuid.uuid4()),
            idempotency_key=idempotency_key,
            kind=kind,
            status="queued",
            payload=payload,
            created_at=now,
            updated_at=now,
        )
        session.add(job)
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
        rows = session.execute(select(Job).order_by(Job.created_at.desc()).limit(limit)).scalars().all()
        return list(rows)


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
        session.commit()
        session.refresh(job)
        return job


def is_cancel_requested(SessionLocal: sessionmaker, job_id: str) -> bool:
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        return bool(getattr(job, "cancel_requested", False)) if job is not None else False


def requeue_running_jobs(SessionLocal: sessionmaker) -> None:
    now = utcnow()
    with SessionLocal() as session:
        session.execute(
            update(Job)
            .where(Job.status == "running")
            .values(status="queued", updated_at=now, started_at=None)
        )
        session.commit()


def _claim_next_job(session: Session) -> Job | None:
    job_id = session.execute(
        select(Job.id).where(Job.status == "queued").order_by(Job.created_at.asc()).limit(1)
    ).scalar_one_or_none()
    if job_id is None:
        return None

    now = utcnow()
    running_exists = select(Job.id).where(Job.status == "running").exists()
    try:
        updated = session.execute(
            update(Job)
            .where(Job.id == job_id)
            .where(Job.status == "queued")
            .where(~running_exists)
            .values(status="running", started_at=now, updated_at=now)
        )
        if (updated.rowcount or 0) != 1:
            session.rollback()
            return None

        session.commit()
        return session.get(Job, job_id)
    except Exception:
        session.rollback()
        return None


def _mark_job_failed(session: Session, job_id: str, message: str) -> None:
    now = utcnow()
    session.execute(
        update(Job)
        .where(Job.id == job_id)
        .values(status="failed", error=message, updated_at=now, finished_at=now)
    )
    session.commit()


def _mark_job_canceled(session: Session, job_id: str) -> None:
    now = utcnow()
    session.execute(
        update(Job)
        .where(Job.id == job_id)
        .values(status="canceled", error=None, updated_at=now, finished_at=now)
    )
    session.commit()


def _mark_job_succeeded(session: Session, job_id: str, result: dict[str, Any]) -> None:
    now = utcnow()
    session.execute(
        update(Job)
        .where(Job.id == job_id)
        .values(status="succeeded", result=result, updated_at=now, finished_at=now, error=None)
    )
    session.commit()


def update_job_partial_result(SessionLocal: sessionmaker, job_id: str, patch: dict[str, Any]) -> None:
    now = utcnow()
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        current = dict(job.result or {})
        current.update(patch)
        job.result = current
        job.updated_at = now
        session.commit()


def execute_job(
    *,
    job_id: str,
    kind: str,
    payload: dict[str, Any],
    SessionLocal: sessionmaker,
) -> dict[str, Any]:
    if kind == "workflow":
        from backend.workflow import WorkflowCanceled, WorkflowContext, cleanup_artifacts, collect_artifact_ids, execute_workflow

        def _progress(patch: dict[str, Any]) -> None:
            update_job_partial_result(SessionLocal, job_id, {"progress": patch})

        artifacts_to_cleanup = collect_artifact_ids(payload)
        try:
            result = execute_workflow(
                payload,
                ctx=WorkflowContext(
                    update_progress=_progress,
                    should_cancel=lambda: is_cancel_requested(SessionLocal, job_id),
                ),
            )
            created = result.pop("created_artifacts", None)
            if isinstance(created, list):
                artifacts_to_cleanup |= set(str(x) for x in created)
            return result
        except WorkflowCanceled:
            raise
        except Exception as exc:
            created = getattr(exc, "_workflow_created_artifacts", None)
            if isinstance(created, set):
                artifacts_to_cleanup |= set(str(x) for x in created)
            raise
        finally:
            cleanup_artifacts(artifacts_to_cleanup)

    raise ValueError(f"Unsupported job kind: {kind}")


class JobWorker:
    def __init__(self, *, engine: Engine, SessionLocal: sessionmaker, config: JobQueueConfig):
        self._engine = engine
        self._SessionLocal = SessionLocal
        self._config = config
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, name="job-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout_s)

    def _run_loop(self) -> None:
        from backend.workflow import WorkflowCanceled

        init_job_db(self._engine)
        if self._config.requeue_running_on_startup:
            requeue_running_jobs(self._SessionLocal)

        while not self._stop.is_set():
            with self._SessionLocal() as session:
                job = _claim_next_job(session)

            if job is None:
                time.sleep(self._config.poll_interval_s)
                continue

            logger.info("Running job id=%s kind=%s", job.id, job.kind)
            try:
                if getattr(job, "cancel_requested", False):
                    with self._SessionLocal() as session:
                        _mark_job_canceled(session, job.id)
                    continue
                with EXECUTION_LOCK:
                    result = execute_job(
                        job_id=job.id,
                        kind=job.kind,
                        payload=dict(job.payload or {}),
                        SessionLocal=self._SessionLocal,
                    )
                with self._SessionLocal() as session:
                    _mark_job_succeeded(session, job.id, result)
            except WorkflowCanceled:
                logger.info("Job canceled id=%s kind=%s", job.id, job.kind)
                with self._SessionLocal() as session:
                    _mark_job_canceled(session, job.id)
            except Exception as exc:
                logger.exception("Job failed id=%s kind=%s", job.id, job.kind)
                with self._SessionLocal() as session:
                    _mark_job_failed(session, job.id, str(exc))


def create_job_queue(config: JobQueueConfig) -> tuple[Engine, sessionmaker, JobWorker]:
    engine = create_job_engine(JobDbConfig(url=config.db_url))
    SessionLocal = create_sessionmaker(engine)
    init_job_db(engine)
    worker = JobWorker(engine=engine, SessionLocal=SessionLocal, config=config)
    return engine, SessionLocal, worker
