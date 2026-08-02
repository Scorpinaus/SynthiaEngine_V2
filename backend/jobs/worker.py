from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass

from backend.jobs.contracts import (
    JobExecutionCanceled,
    JobExecutor,
    JobStore,
    TerminalJobStatus,
)


logger = logging.getLogger(__name__)

# Rendering remains process-local serialized in addition to the database's
# cross-process single-running-row constraint.
EXECUTION_LOCK = threading.Lock()


@dataclass(frozen=True)
class JobWorkerConfig:
    poll_interval_s: float = 0.5
    max_poll_interval_s: float = 5.0
    requeue_running_on_startup: bool = True
    lease_duration_s: float = 30.0
    heartbeat_interval_s: float = 5.0
    worker_vram_mb: int = 0


class JobWorker:
    """Poll, heartbeat, and orchestrate jobs through injected boundaries."""

    def __init__(
        self,
        *,
        store: JobStore,
        executor: JobExecutor,
        config: JobWorkerConfig,
    ) -> None:
        self._store = store
        self._executor = executor
        self._config = config
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._worker_id = uuid.uuid4().hex

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
        self._store.initialize()
        if self._config.requeue_running_on_startup:
            recovered = self._store.requeue_expired()
            if recovered:
                logger.warning("Recovered %d job(s) with expired worker leases", recovered)

        idle_delay = max(0.05, self._config.poll_interval_s)
        while not self._stop.is_set():
            self._store.requeue_expired()
            job = self._store.claim_next(
                worker_id=self._worker_id,
                lease_duration_s=self._config.lease_duration_s,
                worker_vram_mb=self._config.worker_vram_mb,
            )

            if job is None:
                self._stop.wait(idle_delay)
                idle_delay = min(
                    max(idle_delay * 1.5, self._config.poll_interval_s),
                    max(self._config.poll_interval_s, self._config.max_poll_interval_s),
                )
                continue

            idle_delay = max(0.05, self._config.poll_interval_s)
            logger.info("Running job id=%s kind=%s", job.id, job.kind)
            heartbeat_stop = threading.Event()
            heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                args=(job.id, heartbeat_stop),
                name=f"job-heartbeat-{job.id}",
                daemon=True,
            )
            heartbeat_thread.start()
            try:
                if job.cancel_requested:
                    self._mark_or_log(
                        self._store.mark_canceled(job.id, worker_id=self._worker_id),
                        job.id,
                        "canceled",
                    )
                    continue
                with EXECUTION_LOCK:
                    result = self._executor.execute(
                        job_id=job.id,
                        kind=job.kind,
                        payload=job.payload,
                    )
                self._mark_or_log(
                    self._store.mark_succeeded(
                        job.id,
                        result,
                        worker_id=self._worker_id,
                    ),
                    job.id,
                    "succeeded",
                )
            except JobExecutionCanceled:
                logger.info("Job canceled id=%s kind=%s", job.id, job.kind)
                self._mark_or_log(
                    self._store.mark_canceled(job.id, worker_id=self._worker_id),
                    job.id,
                    "canceled",
                )
            except Exception as exc:
                logger.exception("Job failed id=%s kind=%s", job.id, job.kind)
                self._mark_or_log(
                    self._store.mark_failed(
                        job.id,
                        str(exc),
                        worker_id=self._worker_id,
                    ),
                    job.id,
                    "failed",
                )
            finally:
                heartbeat_stop.set()
                heartbeat_thread.join(timeout=max(1.0, self._config.heartbeat_interval_s * 2))

    def _heartbeat_loop(self, job_id: str, stop_event: threading.Event) -> None:
        interval = max(
            0.1,
            min(self._config.heartbeat_interval_s, self._config.lease_duration_s / 2),
        )
        while not stop_event.wait(interval):
            if not self._store.renew_lease(
                job_id=job_id,
                worker_id=self._worker_id,
                lease_duration_s=self._config.lease_duration_s,
            ):
                logger.error("Lost lease for running job id=%s worker=%s", job_id, self._worker_id)
                return

    def _mark_or_log(
        self,
        updated: bool,
        job_id: str,
        status: TerminalJobStatus,
    ) -> None:
        if not updated:
            logger.error(
                "Could not mark job id=%s status=%s because worker ownership was lost",
                job_id,
                status,
            )
