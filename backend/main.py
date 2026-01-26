"""
SynthiaEngine FastAPI application.

This module defines the HTTP API surface for:
- Job submission/status and server-sent event (SSE) polling for updates.
- Artifact upload and static serving from the `OUTPUT_DIR` directory.
- Lightweight registries for models, LoRAs, and ControlNet preprocessors.

Keep business logic in the `backend/*` modules; handlers here should remain thin
and focused on validation, serialization, and HTTP concerns.
"""
import logging
import re
import shutil
import tempfile
from datetime import datetime, timezone
from io import BytesIO
import json
import asyncio
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageFilter
from pydantic import BaseModel, Field

from backend.config import DEFAULTS, OUTPUT_DIR
from backend.controlnet_preprocessors import get_preprocessor, list_preprocessors
from backend.controlnet_preprocessor_registry import (
    CONTROLNET_PREPROCESSOR_REGISTRY,
    ControlNetPreprocessorModelEntry,
)
from backend.model_analysis import SUPPORTED_EXTS, analyze_model_file
from backend.model_registry import (
    ModelRegistryEntry,
    create_model_entry,
    list_model_entries,
)
from backend.lora_registry import (
    LORA_REGISTRY,
    LoraRegistryEntry,
    add_lora,
)
from backend.job_queue import (
    JobNotFoundError,
    JobQueueConfig,
    cancel_job,
    request_cancel_job,
    create_job_queue,
    IdempotencyConflictError,
    enqueue_job,
    get_job,
    list_jobs,
)

from backend.workflow import (
    TASK_REGISTRY,
    WorkflowRequest,
    WorkflowTask,
    build_workflow_catalog,
    save_artifact_png,
)

app = FastAPI(title="SynthiaEngine API")
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

ALLOWED_JOB_KINDS = {"workflow"}

## BaseModel references
class ModelCreateRequest(BaseModel):
    """Request payload used to register a new model in the local registry."""

    name: str
    family: str
    model_type: str
    location_type: str
    model_id: int
    version: str
    link: str


class LoraCreateRequest(BaseModel):
    """Request payload used to register a new LoRA entry in the local registry."""

    lora_id: int
    lora_model_family: str
    lora_type: str
    lora_location: str
    file_path: str
    name: str | None = None


class ControlNetPreprocessorInfo(BaseModel):
    """Serializable info about a ControlNet preprocessor implementation."""

    id: str
    name: str
    description: str
    defaults: dict[str, object]


class ModelLayerRow(BaseModel):
    """Single row in the model-layer analysis response."""

    key: str
    shape: str
    dtype: str


class ModelAnalysisResponse(BaseModel):
    """Response for model analysis endpoint (list of layers/weights)."""

    file_name: str
    loader: str
    total: int
    returned: int
    rows: list[ModelLayerRow]


class WorkflowJobCreateRequest(BaseModel):
    """Job creation request for workflow execution."""

    kind: Literal["workflow"]
    payload: WorkflowRequest
    idempotency_key: str | None = None


JobCreateRequest = WorkflowJobCreateRequest


class JobResponse(BaseModel):
    """Normalized, API-friendly job representation."""

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


class WorkflowTaskTypesResponse(BaseModel):
    """Response payload listing workflow task type identifiers."""

    task_types: list[str]


class WorkflowSchemaResponse(BaseModel):
    """Response payload exposing the JSON schema for workflow requests/tasks."""

    workflow_request_schema: dict[str, Any]
    workflow_task_schema: dict[str, Any]


class WorkflowCatalogTask(BaseModel):
    """A single task type entry in the workflow catalog."""

    input_schema: dict[str, Any]
    input_defaults: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    ui_hints: dict[str, Any] | None = None


class WorkflowCatalogResponse(BaseModel):
    """Response payload exposing per-task input schemas/defaults for workflow builders."""

    version: str
    tasks: dict[str, WorkflowCatalogTask]


def _serialize_job(job) -> JobResponse:
    """Convert a queue job object into the public `JobResponse` format."""
    return JobResponse(
        id=job.id,
        idempotency_key=getattr(job, "idempotency_key", None),
        cancel_requested=getattr(job, "cancel_requested", None),
        kind=job.kind,
        status=job.status,
        payload=dict(job.payload or {}),
        result=dict(job.result) if job.result else None,
        error=job.error,
        created_at=job.created_at.isoformat() if job.created_at else None,
        updated_at=job.updated_at.isoformat() if job.updated_at else None,
        started_at=job.started_at.isoformat() if job.started_at else None,
        finished_at=job.finished_at.isoformat() if job.finished_at else None,
    )


def _get_job_sessionmaker():
    """Return the SQLAlchemy sessionmaker stored on app state (or 503)."""
    sessionmaker = getattr(app.state, "job_sessionmaker", None)
    if sessionmaker is None:
        raise HTTPException(status_code=503, detail="Job queue not initialized.")
    return sessionmaker


@app.on_event("startup")
def _startup_job_queue() -> None:
    """Initialize the job queue and start the background worker thread."""
    engine, sessionmaker, worker = create_job_queue(JobQueueConfig())
    worker.start()
    app.state.job_engine = engine
    app.state.job_sessionmaker = sessionmaker
    app.state.job_worker = worker


@app.on_event("shutdown")
def _shutdown_job_queue() -> None:
    """Stop the background job worker (best effort)."""
    worker = getattr(app.state, "job_worker", None)
    if worker is not None:
        worker.stop()


def _extract_png_metadata(path: Path) -> dict[str, str]:
    """Extract embedded PNG text metadata in a safe, best-effort way."""
    try:
        with Image.open(path) as image:
            metadata: dict[str, str] = {}
            if hasattr(image, "text"):
                metadata.update(image.text)
            for key, value in (image.info or {}).items():
                if isinstance(value, str) and key not in metadata:
                    metadata[key] = value
            return metadata
    except Exception as exc:
        logger.warning("Failed to read metadata for %s: %s", path.name, exc)
        return {}


@app.get("/health")
async def health_check():
    """Basic liveness endpoint used by deployment/health checks."""
    return {"status": "ok"}


@app.post("/api/jobs", response_model=JobResponse, status_code=201)
async def submit_job(req: JobCreateRequest, response: Response, request: Request):
    """
    Enqueue a new job.

    Supports idempotent submissions via:
    - request body `idempotency_key`, or
    - `Idempotency-Key` HTTP header.

    If the idempotency key is present and the job already exists, this returns
    HTTP 200 with the existing job instead of creating a new one.
    """
    if req.kind not in ALLOWED_JOB_KINDS:
        raise HTTPException(status_code=400, detail=f"Unsupported job kind: {req.kind}")

    sessionmaker = _get_job_sessionmaker()
    header_key = request.headers.get("Idempotency-Key")
    idempotency_key = req.idempotency_key or (header_key.strip() if header_key else None)
    # Normalize payload into JSON-serializable primitives for storage/transport.
    payload = req.payload.model_dump(by_alias=True)
    try:
        job, created = enqueue_job(
            sessionmaker,
            kind=req.kind,
            payload=payload,
            idempotency_key=idempotency_key,
        )
    except IdempotencyConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail="Idempotency key already used with a different request.",
        ) from exc

    if idempotency_key and not created:
        response.status_code = 200
    return _serialize_job(job)


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
async def fetch_job(job_id: str):
    """Fetch a single job by id."""
    sessionmaker = _get_job_sessionmaker()
    try:
        job = get_job(sessionmaker, job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc
    return _serialize_job(job)


@app.get("/api/jobs", response_model=list[JobResponse])
async def fetch_jobs(limit: int = 50):
    """List recent jobs, bounded by a small server-side maximum."""
    sessionmaker = _get_job_sessionmaker()
    jobs = list_jobs(sessionmaker, limit=max(1, min(500, int(limit))))
    return [_serialize_job(job) for job in jobs]


@app.post("/api/jobs/{job_id}/cancel", response_model=JobResponse)
async def cancel_queued_job(job_id: str):
    """Request cancellation of a queued/running job (best effort)."""
    sessionmaker = _get_job_sessionmaker()
    try:
        job = request_cancel_job(sessionmaker, job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc
    return _serialize_job(job)


@app.get("/api/jobs/{job_id}/events")
async def stream_job_events(job_id: str):
    """
    Stream job status updates as Server-Sent Events (SSE).

    This implementation polls the job record periodically and emits a new event
    when the status/updated_at changes. It stops once the job reaches a terminal
    state or disappears.
    """
    sessionmaker = _get_job_sessionmaker()
    try:
        get_job(sessionmaker, job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc

    async def event_generator():
        """Yield SSE `data:` frames containing the serialized job payload."""
        last_status = None
        last_updated_at = None
        while True:
            try:
                job = get_job(sessionmaker, job_id)
            except JobNotFoundError:
                # If the job disappears mid-stream, send a final error frame.
                payload = {"error": "Job not found.", "status": "missing"}
                yield f"data: {json.dumps(payload)}\n\n"
                break

            job_response = _serialize_job(job)
            payload = job_response.model_dump()
            status = payload.get("status")
            updated_at = payload.get("updated_at")
            if status != last_status or updated_at != last_updated_at:
                # Only emit when something meaningful changes to reduce spam.
                yield f"data: {json.dumps(payload)}\n\n"
                last_status = status
                last_updated_at = updated_at

            if status in {"succeeded", "failed", "canceled"}:
                # Terminal states: end the stream cleanly.
                break

            # Polling interval for SSE consumers. Kept simple by design.
            await asyncio.sleep(1.0)

    headers = {
        # SSE best practices: prevent buffering and keep the connection open.
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


@app.get("/api/workflow/task-types", response_model=WorkflowTaskTypesResponse)
async def list_workflow_task_types():
    """Return the set of registered workflow task type keys."""
    return WorkflowTaskTypesResponse(task_types=sorted(TASK_REGISTRY.keys()))


@app.get("/api/workflow/schema", response_model=WorkflowSchemaResponse)
async def get_workflow_schema():
    """Expose workflow request/task JSON schemas for UI validation."""
    return WorkflowSchemaResponse(
        workflow_request_schema=WorkflowRequest.model_json_schema(by_alias=True),
        workflow_task_schema=WorkflowTask.model_json_schema(by_alias=True),
    )


@app.get("/api/workflow/catalog", response_model=WorkflowCatalogResponse)
async def get_workflow_catalog():
    """
    Return the workflow task catalog.

    Each task entry includes an input JSON Schema and a best-effort `input_defaults`
    dict so UIs can build/validate workflows without hardcoding values.
    """
    return WorkflowCatalogResponse(**build_workflow_catalog())


class ArtifactResponse(BaseModel):
    """Response payload describing a stored artifact in `OUTPUT_DIR`."""

    artifact_id: str
    url: str
    path: str


@app.post("/api/artifacts", response_model=ArtifactResponse, status_code=201)
async def upload_artifact(file: UploadFile = File(...)):
    """Upload an image artifact and persist it under `OUTPUT_DIR`."""
    file_bytes = await file.read()
    try:
        image = Image.open(BytesIO(file_bytes))
        # Force decode early to catch truncated/invalid image streams.
        image.load()
        if image.mode == "P":
            # Palette images don't carry alpha in a convenient way for later steps.
            image = image.convert("RGBA")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    artifact = save_artifact_png(image, prefix="a")
    return ArtifactResponse(**artifact)


@app.get("/models", response_model=list[ModelRegistryEntry])
async def list_models(family: str | None = None):
    """
    List registered models.

    If `family` is provided, it is matched loosely (case-insensitive), with a few
    common aliases mapped to friendlier patterns (e.g., "sd15", "sd1.5").
    """
    entries = list_model_entries()
    if not family:
        return entries

    family_value = family.strip().lower()
    if not family_value:
        return entries

    # Map common UI aliases to a more permissive regex to improve recall.
    if family_value in {"sd15", "sd1.5"}:
        pattern = re.compile(r"sd[\s_-]*1\.?5|sd15", re.IGNORECASE)
    elif family_value == "sdxl":
        pattern = re.compile(r"sdxl", re.IGNORECASE)
    elif family_value == "z-image-turbo":
        pattern = re.compile(r"z-image-turbo", re.IGNORECASE)
    elif family_value == "qwen-image":
        pattern = re.compile(r"qwen[-_\s]?image", re.IGNORECASE)
    elif family_value == "flux":
        pattern = re.compile(r"flux", re.IGNORECASE)
    else:
        pattern = re.compile(re.escape(family_value), re.IGNORECASE)

    return [entry for entry in entries if pattern.search(entry.family)]


@app.get("/lora-models", response_model=list[LoraRegistryEntry])
async def list_lora_models(family: str | None = None):
    """List registered LoRAs, optionally filtered by exact family (case-insensitive)."""
    if not family:
        return LORA_REGISTRY

    family_value = family.strip().lower()
    if not family_value:
        return LORA_REGISTRY

    return [
        entry
        for entry in LORA_REGISTRY
        if entry.lora_model_family.lower() == family_value
    ]


@app.post("/lora-models", response_model=LoraRegistryEntry)
async def create_lora_model(req: LoraCreateRequest):
    """Create a new LoRA registry entry."""
    try:
        entry = LoraRegistryEntry(**req.dict())
        return add_lora(entry)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/controlnet/preprocessors", response_model=list[ControlNetPreprocessorInfo])
async def list_controlnet_preprocessors():
    """Return available ControlNet preprocessors and their default params."""
    preprocessors = list_preprocessors()
    return [
        ControlNetPreprocessorInfo(
            id=preprocessor.id,
            name=preprocessor.name,
            description=preprocessor.description,
            defaults=preprocessor.defaults,
        )
        for preprocessor in preprocessors
    ]


@app.get("/api/controlnet/preprocessor-models",
    response_model=list[ControlNetPreprocessorModelEntry],
)
async def list_controlnet_preprocessor_models():
    """Return the list of ControlNet model entries (for UI selection)."""
    return CONTROLNET_PREPROCESSOR_REGISTRY


@app.post("/api/controlnet/preprocess")
async def run_controlnet_preprocessor(
    image: UploadFile = File(...),
    preprocessor_id: str = Form(...),
    params: str | None = Form(None),
    low_threshold: int | None = Form(None),
    high_threshold: int | None = Form(None),
):
    """
    Run a ControlNet preprocessor over an uploaded image and return a PNG.

    `params` is expected to be a JSON object encoded as a string. For
    convenience, threshold form fields override corresponding JSON keys when
    provided.
    """
    image_bytes = await image.read()
    try:
        source_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc


    try:
        preprocessor = get_preprocessor(preprocessor_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    parsed_params: dict[str, object] = {}
    if params:
        try:
            parsed_params = json.loads(params)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Invalid params JSON.") from exc

    # Allow simple threshold overrides without requiring clients to build JSON.
    if low_threshold is not None:
        parsed_params["low_threshold"] = low_threshold
    if high_threshold is not None:
        parsed_params["high_threshold"] = high_threshold

    processed = preprocessor.process(source_image, parsed_params)
    output = BytesIO()
    processed.save(output, format="PNG")
    return Response(content=output.getvalue(), media_type="image/png")


@app.post("/api/tools/analyze-model", response_model=ModelAnalysisResponse)
async def analyze_model_layers(
    file: UploadFile = File(...),
    limit: int | None = Form(None),
):
    """
    Analyze an uploaded model file and return a (possibly limited) layer list.

    The upload stream is copied to a temporary file to support loader APIs that
    require a filesystem path. The temporary file is always cleaned up.
    """
    filename = file.filename or "uploaded_model"
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension: {suffix or 'unknown'}.",
        )

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = Path(temp_file.name)

        rows, loader, total = analyze_model_file(temp_path, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        # Ensure we don't leak disk usage for failed analyses.
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)

    return ModelAnalysisResponse(
        file_name=filename,
        loader=loader,
        total=total,
        returned=len(rows),
        rows=[ModelLayerRow(key=k, shape=s, dtype=d) for k, s, d in rows],
    )


@app.post("/models", response_model=ModelRegistryEntry, status_code=201)
async def create_model(req: ModelCreateRequest):
    """Create a new model registry entry, enforcing unique names."""
    try:
        entry = ModelRegistryEntry(**req.dict())
        return create_model_entry(entry)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/history")
async def list_history():
    """List generated images from `OUTPUT_DIR` along with embedded metadata."""
    if not OUTPUT_DIR.exists():
        return []

    records: list[dict[str, object]] = []
    # Walk the outputs folder to produce a lightweight generation history feed.
    for image_path in OUTPUT_DIR.rglob("*.png"):
        stat = image_path.stat()
        timestamp = stat.st_mtime
        created_at = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        metadata = _extract_png_metadata(image_path)
        relative_path = image_path.relative_to(OUTPUT_DIR).as_posix()
        records.append(
            {
                "filename": relative_path,
                "url": f"/outputs/{relative_path}",
                "timestamp": timestamp,
                "created_at": created_at,
                "metadata": metadata,
            }
        )

    records.sort(key=lambda item: item.get("timestamp", 0), reverse=True)
    return records
## Inpainting related endpoints

def _create_blur_mask(mask_image: Image.Image, blur_factor: int) -> Image.Image:
    """Apply a configurable Gaussian blur to a mask image (clamped)."""
    blur_factor = max(0, min(int(blur_factor), 128))
    if blur_factor == 0:
        return mask_image
    return mask_image.filter(ImageFilter.GaussianBlur(radius=blur_factor))


@app.post("/create-blur-mask")
async def create_blur_mask_endpoint(
    mask_image: UploadFile = File(...),
    blur_factor: int = Form(8),
):
    """Generate a blurred version of a grayscale mask (used for inpainting)."""
    mask_bytes = await mask_image.read()
    try:
        mask = Image.open(BytesIO(mask_bytes)).convert("L")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid mask image file.") from exc

    blurred_mask = _create_blur_mask(mask, blur_factor)
    output = BytesIO()
    blurred_mask.save(output, format="PNG")
    return Response(content=output.getvalue(), media_type="image/png")
