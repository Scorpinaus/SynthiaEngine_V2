from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol


TerminalJobStatus = Literal["succeeded", "failed", "canceled"]


@dataclass(frozen=True)
class ClaimedJob:
    """Detached job data passed from persistence to execution orchestration."""

    id: str
    kind: str
    payload: dict[str, Any]
    cancel_requested: bool
    worker_id: str
    attempt: int
    resource_requirements: dict[str, Any]


class JobExecutionStore(Protocol):
    """Persistence operations available while a job is executing."""

    def is_cancel_requested(self, job_id: str) -> bool: ...

    def record_progress(self, job_id: str, progress: dict[str, Any]) -> None: ...

    def merge_partial_result(self, job_id: str, patch: dict[str, Any]) -> None: ...


class JobStore(JobExecutionStore, Protocol):
    """Worker-facing queue, lease, and terminal-state boundary."""

    def initialize(self) -> None: ...

    def requeue_expired(self) -> int: ...

    def claim_next(
        self,
        *,
        worker_id: str,
        lease_duration_s: float,
        worker_vram_mb: int = 0,
    ) -> ClaimedJob | None: ...

    def renew_lease(
        self,
        *,
        job_id: str,
        worker_id: str,
        lease_duration_s: float,
    ) -> bool: ...

    def mark_succeeded(
        self,
        job_id: str,
        result: dict[str, Any],
        *,
        worker_id: str,
    ) -> bool: ...

    def mark_failed(self, job_id: str, message: str, *, worker_id: str) -> bool: ...

    def mark_canceled(self, job_id: str, *, worker_id: str) -> bool: ...


class JobExecutor(Protocol):
    """Worker-facing boundary for one claimed job execution."""

    def execute(
        self,
        *,
        job_id: str,
        kind: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]: ...


class ArtifactCleanup(Protocol):
    """Cleanup boundary invoked once with every artifact owned by a job."""

    def __call__(self, artifact_ids: set[str]) -> None: ...


class JobExecutionCanceled(Exception):
    """Execution-layer cancellation signal understood by the worker."""
