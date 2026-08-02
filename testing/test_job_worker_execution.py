import threading
from typing import Any

from backend.jobs.contracts import ClaimedJob
from backend.jobs.worker import JobWorker, JobWorkerConfig


class FakeWorkerStore:
    def __init__(self, job: ClaimedJob) -> None:
        self._job: ClaimedJob | None = job
        self.initialized = False
        self.terminal: tuple[str, str, Any] | None = None
        self.terminal_event = threading.Event()

    def initialize(self) -> None:
        self.initialized = True

    def requeue_expired(self) -> int:
        return 0

    def claim_next(self, **_kwargs) -> ClaimedJob | None:
        job, self._job = self._job, None
        return job

    def renew_lease(self, **_kwargs) -> bool:
        return True

    def mark_succeeded(
        self,
        job_id: str,
        result: dict[str, Any],
        *,
        worker_id: str,
    ) -> bool:
        self.terminal = ("succeeded", job_id, result)
        self.terminal_event.set()
        return True

    def mark_failed(self, job_id: str, message: str, *, worker_id: str) -> bool:
        self.terminal = ("failed", job_id, message)
        self.terminal_event.set()
        return True

    def mark_canceled(self, job_id: str, *, worker_id: str) -> bool:
        self.terminal = ("canceled", job_id, None)
        self.terminal_event.set()
        return True

    def is_cancel_requested(self, job_id: str) -> bool:
        return False

    def record_progress(self, job_id: str, progress: dict[str, Any]) -> None:
        pass

    def merge_partial_result(self, job_id: str, patch: dict[str, Any]) -> None:
        pass


class FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def execute(
        self,
        *,
        job_id: str,
        kind: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append((job_id, kind, payload))
        return {"outputs": {"ok": True}, "tasks": {}}


def test_worker_orchestration_uses_injected_store_and_executor():
    job = ClaimedJob(
        id="job-1",
        kind="workflow",
        payload={"tasks": []},
        cancel_requested=False,
        worker_id="worker-placeholder",
        attempt=1,
        resource_requirements={},
    )
    store = FakeWorkerStore(job)
    executor = FakeExecutor()
    worker = JobWorker(
        store=store,
        executor=executor,
        config=JobWorkerConfig(poll_interval_s=0.01, heartbeat_interval_s=1.0),
    )

    worker.start()
    try:
        assert store.terminal_event.wait(timeout=2.0)
    finally:
        worker.stop()

    assert store.initialized is True
    assert executor.calls == [("job-1", "workflow", {"tasks": []})]
    assert store.terminal == (
        "succeeded",
        "job-1",
        {"outputs": {"ok": True}, "tasks": {}},
    )
