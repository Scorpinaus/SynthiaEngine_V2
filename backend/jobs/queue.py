"""Compatibility facade and composition factory for the job subsystem.

New code should import persistence from :mod:`backend.jobs.store`, execution
from :mod:`backend.jobs.execution`, and polling from :mod:`backend.jobs.worker`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from backend.jobs.db import DEFAULT_JOB_DB_URL
from backend.jobs.execution import WorkflowJobExecutor
from backend.jobs.store import (
    IdempotencyConflictError,
    JobNotFoundError,
    SqlAlchemyJobStore,
    _claim_next_job,
    _mark_job_canceled,
    _mark_job_failed,
    _mark_job_succeeded,
    cancel_job,
    create_job_store,
    enqueue_job,
    estimate_job_resources,
    finalize_job_tasks,
    get_job,
    init_job_db,
    is_cancel_requested,
    list_job_tasks,
    list_jobs,
    renew_job_lease,
    request_cancel_job,
    requeue_expired_jobs,
    update_job_partial_result,
    update_job_task_progress,
)
from backend.jobs.worker import EXECUTION_LOCK, JobWorker, JobWorkerConfig
from backend.settings import load_settings


@dataclass(frozen=True)
class JobQueueConfig:
    db_url: str = DEFAULT_JOB_DB_URL
    poll_interval_s: float = 0.5
    max_poll_interval_s: float = 5.0
    requeue_running_on_startup: bool = True
    lease_duration_s: float = 30.0
    heartbeat_interval_s: float = 5.0
    worker_vram_mb: int = field(default_factory=lambda: load_settings().worker.vram_mb)

    def worker_config(self) -> JobWorkerConfig:
        return JobWorkerConfig(
            poll_interval_s=self.poll_interval_s,
            max_poll_interval_s=self.max_poll_interval_s,
            requeue_running_on_startup=self.requeue_running_on_startup,
            lease_duration_s=self.lease_duration_s,
            heartbeat_interval_s=self.heartbeat_interval_s,
            worker_vram_mb=self.worker_vram_mb,
        )


class _LegacyExecutionStore:
    """Adapt the historic sessionmaker callbacks to the execution boundary."""

    def __init__(self, SessionLocal: sessionmaker) -> None:
        self._SessionLocal = SessionLocal

    def is_cancel_requested(self, job_id: str) -> bool:
        return is_cancel_requested(self._SessionLocal, job_id)

    def record_progress(self, job_id: str, progress: dict[str, Any]) -> None:
        update_job_task_progress(self._SessionLocal, job_id, progress)
        update_job_partial_result(self._SessionLocal, job_id, {"progress": progress})

    def merge_partial_result(self, job_id: str, patch: dict[str, Any]) -> None:
        update_job_partial_result(self._SessionLocal, job_id, patch)


def execute_job(
    *,
    job_id: str,
    kind: str,
    payload: dict[str, Any],
    SessionLocal: sessionmaker,
) -> dict[str, Any]:
    """Compatibility wrapper for callers of the pre-ARC-04 execution helper."""

    executor = WorkflowJobExecutor(_LegacyExecutionStore(SessionLocal))
    return executor.execute(job_id=job_id, kind=kind, payload=payload)


def create_job_queue(config: JobQueueConfig) -> tuple[Engine, sessionmaker, JobWorker]:
    engine, SessionLocal, store = create_job_store(config.db_url)
    worker = JobWorker(
        store=store,
        executor=WorkflowJobExecutor(store),
        config=config.worker_config(),
    )
    return engine, SessionLocal, worker


__all__ = [
    "EXECUTION_LOCK",
    "IdempotencyConflictError",
    "JobNotFoundError",
    "JobQueueConfig",
    "JobWorker",
    "SqlAlchemyJobStore",
    "_claim_next_job",
    "_mark_job_canceled",
    "_mark_job_failed",
    "_mark_job_succeeded",
    "cancel_job",
    "create_job_queue",
    "enqueue_job",
    "estimate_job_resources",
    "execute_job",
    "finalize_job_tasks",
    "get_job",
    "init_job_db",
    "is_cancel_requested",
    "list_job_tasks",
    "list_jobs",
    "renew_job_lease",
    "request_cancel_job",
    "requeue_expired_jobs",
    "update_job_partial_result",
    "update_job_task_progress",
]
