from __future__ import annotations

import logging
from typing import Any

from backend.jobs.contracts import ArtifactCleanup, JobExecutionCanceled, JobExecutionStore


logger = logging.getLogger(__name__)


class WorkflowJobExecutor:
    """Execute workflow jobs without owning queue or lease persistence."""

    def __init__(
        self,
        store: JobExecutionStore,
        *,
        artifact_cleanup: ArtifactCleanup | None = None,
    ) -> None:
        self._store = store
        self._artifact_cleanup = artifact_cleanup

    def execute(
        self,
        *,
        job_id: str,
        kind: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if kind != "workflow":
            raise ValueError(f"Unsupported job kind: {kind}")

        from backend.utilities.resource_logging import SummaryProfiler
        from backend.utilities.subprocess_transport import SubprocessCanceled
        from backend.workflow.engine import execute_workflow
        from backend.workflow.types import WorkflowCanceled, WorkflowContext
        from backend.workflow.utility import cleanup_artifacts, collect_artifact_ids

        cleanup = self._artifact_cleanup or cleanup_artifacts

        def _progress(patch: dict[str, Any]) -> None:
            self._store.record_progress(job_id, patch)

        def _profile_update(profile: dict[str, Any]) -> None:
            self._store.merge_partial_result(job_id, {"profile": profile})

        artifacts_to_cleanup = collect_artifact_ids(payload)
        result: dict[str, Any] | None = None
        execution_error: Exception | None = None
        execution_traceback = None
        try:
            with SummaryProfiler(on_update=_profile_update) as profiler:
                result = execute_workflow(
                    payload,
                    ctx=WorkflowContext(
                        update_progress=_progress,
                        should_cancel=lambda: self._store.is_cancel_requested(job_id),
                    ),
                )
            if profiler.profile is not None:
                result["profile"] = profiler.profile
            from backend.utilities.pipeline_cache import pipeline_cache_stats

            result.setdefault("profile", {})["pipeline_caches"] = pipeline_cache_stats()
            created = result.pop("created_artifacts", None)
            if isinstance(created, list):
                artifacts_to_cleanup |= set(str(value) for value in created)
        except Exception as exc:
            created = getattr(exc, "_workflow_created_artifacts", None)
            if isinstance(created, set):
                artifacts_to_cleanup |= set(str(value) for value in created)
            execution_error = exc
            execution_traceback = exc.__traceback__

        cleanup_error: Exception | None = None
        try:
            cleanup(artifacts_to_cleanup)
        except Exception as exc:
            cleanup_error = exc

        if execution_error is not None:
            if cleanup_error is not None:
                logger.error(
                    "Artifact cleanup also failed after workflow execution failed for job id=%s",
                    job_id,
                    exc_info=(
                        type(cleanup_error),
                        cleanup_error,
                        cleanup_error.__traceback__,
                    ),
                )
            if isinstance(execution_error, (WorkflowCanceled, SubprocessCanceled)):
                raise JobExecutionCanceled(job_id) from execution_error
            raise execution_error.with_traceback(execution_traceback)

        if cleanup_error is not None:
            raise cleanup_error
        if result is None:
            raise RuntimeError(f"Workflow job {job_id} completed without a result.")
        return result
