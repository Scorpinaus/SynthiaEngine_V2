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

from backend.job_db import JobDbConfig, create_job_engine, create_sessionmaker
from backend.job_models import Base, Job, utcnow

logger = logging.getLogger(__name__)

EXECUTION_LOCK = threading.Lock()


class JobNotFoundError(Exception):
    pass


@dataclass(frozen=True)
class JobQueueConfig:
    db_url: str = "sqlite:///outputs/jobs.sqlite3"
    poll_interval_s: float = 0.5
    requeue_running_on_startup: bool = True


def init_job_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def enqueue_job(SessionLocal: sessionmaker, *, kind: str, payload: dict[str, Any]) -> Job:
    now = utcnow()
    job = Job(
        id=str(uuid.uuid4()),
        kind=kind,
        status="queued",
        payload=payload,
        created_at=now,
        updated_at=now,
    )
    with SessionLocal() as session:
        session.add(job)
        session.commit()
        session.refresh(job)
        return job


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


def _mark_job_succeeded(session: Session, job_id: str, result: dict[str, Any]) -> None:
    now = utcnow()
    session.execute(
        update(Job)
        .where(Job.id == job_id)
        .values(status="succeeded", result=result, updated_at=now, finished_at=now, error=None)
    )
    session.commit()


def execute_job(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    if kind == "noop":
        return {"ok": True}

    if kind == "sdxl_text2img":
        from backend.sdxl_pipeline import run_sdxl_text2img

        return run_sdxl_text2img(payload)

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
                with EXECUTION_LOCK:
                    result = execute_job(job.kind, dict(job.payload or {}))
                with self._SessionLocal() as session:
                    _mark_job_succeeded(session, job.id, result)
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
