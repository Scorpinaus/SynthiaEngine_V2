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
import os
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
from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    delete_model_entry,
    get_model_entry_exact,
    list_model_entries,
    update_model_entry,
)
from backend.lora_registry import (
    LoraRegistryEntry,
    add_lora,
    delete_lora_entry,
    get_lora_entry,
    list_lora_entries,
    update_lora_entry,
)
from backend.logging_utils import configure_logging
from backend.preset_registry import (
    PresetRegistryCreate,
    PresetRegistryEntry,
    create_preset_entry,
    delete_preset_entry,
    get_preset_entry,
    list_preset_entries,
    update_preset_entry,
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

configure_logging(role=os.getenv("SYNTHA_LOG_ROLE", "api"))

app = FastAPI(title="SynthiaEngine API")
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

ALLOWED_JOB_KINDS = {"workflow"}


def _validate_lora_name(name: str | None) -> str | None:
    if name is None:
        return None
    if "." in name:
        raise ValueError("LoRA name cannot contain '.'")
    return name

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


class ModelUpdateRequest(BaseModel):
    """Request payload used to update editable fields on a model entry."""

    model_config = ConfigDict(extra="forbid")

    family: str | None = None
    model_type: str | None = None
    location_type: str | None = None
    model_id: int | None = None
    version: str | None = None
    link: str | None = None


class LoraCreateRequest(BaseModel):
    """Request payload used to register a new LoRA entry in the local registry."""

    lora_id: int
    lora_model_family: str
    lora_type: str
    lora_location: str
    file_path: str
    name: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        return _validate_lora_name(value)


class LoraUpdateRequest(BaseModel):
    """Request payload used to update editable fields on a LoRA entry."""

    model_config = ConfigDict(extra="forbid")

    lora_model_family: str | None = None
    lora_type: str | None = None
    lora_location: str | None = None
    file_path: str | None = None
    name: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        return _validate_lora_name(value)


class PresetCreateRequest(BaseModel):
    """Request payload used to create a saved generation preset."""

    name: str
    family: str
    task_type: str
    settings: dict[str, Any] = Field(default_factory=dict)


class PresetUpdateRequest(BaseModel):
    """Request payload used to update editable fields on a saved preset."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    family: str | None = None
    task_type: str | None = None
    settings: dict[str, Any] | None = None


class ControlNetPreprocessorInfo(BaseModel):
    """Serializable info about a ControlNet preprocessor implementation."""

    id: str
    name: str
    description: str
    defaults: dict[str, object]
    param_schema: dict[str, dict[str, object]] = Field(default_factory=dict)
    recommended_sd15_control_models: list[str] = Field(default_factory=list)
    legacy_aliases: list[str] = Field(default_factory=list)


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


class WorkflowModelCapabilities(BaseModel):
    """Capability summary for a single model family."""

    label: str
    aliases: list[str] = Field(default_factory=list)
    task_types: list[str] = Field(default_factory=list)
    features: dict[str, bool] = Field(default_factory=dict)


class WorkflowCatalogResponse(BaseModel):
    """Response payload exposing per-task input schemas/defaults for workflow builders."""

    version: str
    tasks: dict[str, WorkflowCatalogTask]
    capabilities: dict[str, WorkflowModelCapabilities] = Field(default_factory=dict)


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


def _env_flag_enabled(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _api_embedded_worker_enabled() -> bool:
    return _env_flag_enabled("SYNTHA_API_START_WORKER", default=True)


@app.on_event("startup")
def _startup_job_queue() -> None:
    """Initialize the job queue and optionally start the embedded worker thread."""
    engine, sessionmaker, worker = create_job_queue(JobQueueConfig())
    app.state.job_engine = engine
    app.state.job_sessionmaker = sessionmaker
    app.state.job_worker = worker
    app.state.job_worker_started = False

    if _api_embedded_worker_enabled():
        worker.start()
        app.state.job_worker_started = True
        logger.info("Embedded API job worker started.")
    else:
        logger.info("Embedded API job worker disabled; external render worker expected.")


@app.on_event("shutdown")
def _shutdown_job_queue() -> None:
    """Stop the background job worker (best effort)."""
    worker = getattr(app.state, "job_worker", None)
    if worker is not None:
        worker.stop()
    engine = getattr(app.state, "job_engine", None)
    if engine is not None:
        engine.dispose()


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
    entries = list_lora_entries()
    if not family:
        return entries

    family_value = family.strip().lower()
    if not family_value:
        return entries

    return [
        entry
        for entry in entries
        if entry.lora_model_family.lower() == family_value
    ]


@app.post("/lora-models", response_model=LoraRegistryEntry)
async def create_lora_model(req: LoraCreateRequest):
    """Create a new LoRA registry entry."""
    try:
        entry = LoraRegistryEntry(**req.model_dump())
        return add_lora(entry)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/lora-models/{lora_id}", response_model=LoraRegistryEntry)
async def get_lora_model(lora_id: int):
    """Fetch a single LoRA registry entry by id."""
    try:
        return get_lora_entry(lora_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.patch("/lora-models/{lora_id}", response_model=LoraRegistryEntry)
async def patch_lora_model(lora_id: int, req: LoraUpdateRequest):
    """Update editable fields for a single LoRA registry entry."""
    updates = req.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="At least one editable field must be provided.")

    try:
        return update_lora_entry(lora_id, updates)
    except ValueError as exc:
        detail = str(exc)
        if detail.endswith("not found."):
            raise HTTPException(status_code=404, detail=detail) from exc
        raise HTTPException(status_code=400, detail=detail) from exc


@app.delete("/lora-models/{lora_id}", status_code=204)
async def remove_lora_model(lora_id: int):
    """Delete a single LoRA registry entry by id."""
    try:
        delete_lora_entry(lora_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(status_code=204)


@app.get("/api/presets", response_model=list[PresetRegistryEntry])
async def list_presets(family: str | None = None, task_type: str | None = None):
    """List saved generation presets, with optional family/task filters."""
    try:
        return list_preset_entries(family=family, task_type=task_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/presets", response_model=PresetRegistryEntry, status_code=201)
async def create_preset(req: PresetCreateRequest):
    """Create a new saved generation preset."""
    try:
        return create_preset_entry(PresetRegistryCreate(**req.model_dump()))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/presets/{preset_id}", response_model=PresetRegistryEntry)
async def get_preset(preset_id: int):
    """Fetch one saved generation preset by id."""
    try:
        return get_preset_entry(preset_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.patch("/api/presets/{preset_id}", response_model=PresetRegistryEntry)
async def patch_preset(preset_id: int, req: PresetUpdateRequest):
    """Update editable fields for one saved generation preset."""
    updates = req.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="At least one editable field must be provided.")
    try:
        return update_preset_entry(preset_id, updates)
    except ValueError as exc:
        detail = str(exc)
        if detail.endswith("not found."):
            raise HTTPException(status_code=404, detail=detail) from exc
        raise HTTPException(status_code=400, detail=detail) from exc


@app.delete("/api/presets/{preset_id}", status_code=204)
async def remove_preset(preset_id: int):
    """Delete one saved generation preset by id."""
    try:
        delete_preset_entry(preset_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(status_code=204)


@app.get("/api/controlnet/preprocessors", response_model=list[ControlNetPreprocessorInfo])
async def list_controlnet_preprocessors():
    """Return available ControlNet preprocessors and their default params."""
    preprocessors = list_preprocessors()
    registry_by_id = {entry.id: entry for entry in CONTROLNET_PREPROCESSOR_REGISTRY}
    infos: list[ControlNetPreprocessorInfo] = []
    for preprocessor in preprocessors:
        registry_entry = registry_by_id.get(preprocessor.id)
        infos.append(
            ControlNetPreprocessorInfo(
                id=preprocessor.id,
                name=preprocessor.name,
                description=preprocessor.description,
                defaults=preprocessor.defaults,
                param_schema={
                    key: {
                        "type": spec.type,
                        "description": spec.description,
                        "minimum": spec.minimum,
                        "maximum": spec.maximum,
                    }
                    for key, spec in preprocessor.param_schema.items()
                },
                recommended_sd15_control_models=(
                    registry_entry.recommended_sd15_control_models if registry_entry else []
                ),
                legacy_aliases=(registry_entry.legacy_aliases if registry_entry else []),
            )
        )
    return infos


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

    try:
        processed = preprocessor.process(source_image, parsed_params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        message = str(exc)
        if "controlnet-aux is required" in message:
            raise HTTPException(
                status_code=503,
                detail=(
                    "ControlNet preprocessors dependency is missing. "
                    "Install `controlnet-aux` and restart the backend."
                ),
            ) from exc
        raise HTTPException(status_code=500, detail=message) from exc
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
        entry = ModelRegistryEntry(**req.model_dump())
        return create_model_entry(entry)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/models/{model_name:path}", response_model=ModelRegistryEntry)
async def get_model(model_name: str):
    """Fetch a single model registry entry by exact name."""
    try:
        return get_model_entry_exact(model_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.patch("/models/{model_name:path}", response_model=ModelRegistryEntry)
async def patch_model(model_name: str, req: ModelUpdateRequest):
    """Update editable fields for a single model registry entry."""
    updates = req.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="At least one editable field must be provided.")

    try:
        return update_model_entry(model_name, updates)
    except ValueError as exc:
        detail = str(exc)
        if detail.endswith("not found."):
            raise HTTPException(status_code=404, detail=detail) from exc
        if detail == "Model name already exists.":
            raise HTTPException(status_code=409, detail=detail) from exc
        raise HTTPException(status_code=400, detail=detail) from exc


@app.delete("/models/{model_name:path}", status_code=204)
async def remove_model(model_name: str):
    """Delete a single model registry entry by exact name."""
    try:
        delete_model_entry(model_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(status_code=204)


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
