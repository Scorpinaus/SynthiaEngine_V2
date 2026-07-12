from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.jobs.queue import (
    IdempotencyConflictError,
    JobNotFoundError,
    enqueue_job,
    get_job,
    list_job_tasks,
    list_jobs,
    request_cancel_job,
)
from backend.workflow import WorkflowRequest

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class WorkflowJobCreateRequest(BaseModel):
    kind: Literal["workflow"]
    payload: WorkflowRequest
    idempotency_key: str | None = None


JobCreateRequest = WorkflowJobCreateRequest


class JobResponse(BaseModel):
    id: str
    idempotency_key: str | None = None
    cancel_requested: bool | None = None
    kind: str
    status: str
    payload: dict[str, object]
    result: dict[str, object] | None = None
    error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    resource_requirements: dict[str, Any] = Field(default_factory=dict)


class JobTaskResponse(BaseModel):
    task_id: str
    task_type: str
    task_index: int
    status: str
    inputs: dict[str, Any]
    output: dict[str, Any] | None = None
    error: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


def _timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def serialize_job(job) -> JobResponse:
    return JobResponse(
        id=job.id,
        idempotency_key=getattr(job, "idempotency_key", None),
        cancel_requested=getattr(job, "cancel_requested", None),
        kind=job.kind,
        status=job.status,
        payload=dict(job.payload or {}),
        result=dict(job.result) if job.result else None,
        error=job.error,
        created_at=_timestamp(job.created_at),
        updated_at=_timestamp(job.updated_at),
        started_at=_timestamp(job.started_at),
        finished_at=_timestamp(job.finished_at),
        resource_requirements=dict(getattr(job, "resource_requirements", None) or {}),
    )


def serialize_job_task(task) -> JobTaskResponse:
    return JobTaskResponse(
        task_id=task.task_id,
        task_type=task.task_type,
        task_index=task.task_index,
        status=task.status,
        inputs=dict(task.inputs or {}),
        output=dict(task.output) if task.output else None,
        error=task.error,
        created_at=_timestamp(task.created_at),
        started_at=_timestamp(task.started_at),
        finished_at=_timestamp(task.finished_at),
    )


def _sessions(request: Request):
    sessions = getattr(request.app.state, "job_sessionmaker", None)
    if sessions is None:
        raise HTTPException(status_code=503, detail="Job queue not initialized.")
    return sessions


@router.post("", response_model=JobResponse, status_code=201)
async def submit_job(req: JobCreateRequest, response: Response, request: Request):
    sessions = _sessions(request)
    header_key = request.headers.get("Idempotency-Key")
    idempotency_key = req.idempotency_key or (header_key.strip() if header_key else None)
    try:
        job, created = enqueue_job(
            sessions,
            kind=req.kind,
            payload=req.payload.model_dump(by_alias=True),
            idempotency_key=idempotency_key,
        )
    except IdempotencyConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail="Idempotency key already used with a different request.",
        ) from exc
    if idempotency_key and not created:
        response.status_code = 200
    return serialize_job(job)


@router.get("/{job_id}", response_model=JobResponse)
async def fetch_job(job_id: str, request: Request):
    try:
        return serialize_job(get_job(_sessions(request), job_id))
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc


@router.get("/{job_id}/tasks", response_model=list[JobTaskResponse])
async def fetch_job_tasks(job_id: str, request: Request):
    try:
        tasks = list_job_tasks(_sessions(request), job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc
    return [serialize_job_task(task) for task in tasks]


@router.get("", response_model=list[JobResponse])
async def fetch_jobs(request: Request, limit: int = 50):
    jobs = list_jobs(_sessions(request), limit=max(1, min(500, int(limit))))
    return [serialize_job(job) for job in jobs]


@router.post("/{job_id}/cancel", response_model=JobResponse)
async def cancel_queued_job(job_id: str, request: Request):
    try:
        job = request_cancel_job(_sessions(request), job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc
    return serialize_job(job)


@router.get("/{job_id}/events")
async def stream_job_events(job_id: str, request: Request):
    sessions = _sessions(request)
    try:
        get_job(sessions, job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc

    async def event_generator():
        last_status = None
        last_updated_at = None
        while True:
            try:
                payload = serialize_job(get_job(sessions, job_id)).model_dump()
            except JobNotFoundError:
                yield f'data: {json.dumps({"error": "Job not found.", "status": "missing"})}\n\n'
                break
            status = payload.get("status")
            updated_at = payload.get("updated_at")
            if status != last_status or updated_at != last_updated_at:
                yield f"data: {json.dumps(payload)}\n\n"
                last_status, last_updated_at = status, updated_at
            if status in {"succeeded", "failed", "canceled"}:
                break
            await asyncio.sleep(1.0)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
