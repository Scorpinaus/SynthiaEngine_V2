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
import shutil
import tempfile
from io import BytesIO
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageFilter
from pydantic import BaseModel, Field

from backend.config import DEFAULTS, OUTPUT_DIR
from backend.adapters.controlnet_preprocessors import get_preprocessor, list_preprocessors
from backend.adapters.controlnet_preprocessor_registry import (
    CONTROLNET_PREPROCESSOR_REGISTRY,
    ControlNetPreprocessorModelEntry,
)
from backend.utilities.model_analysis import SUPPORTED_EXTS, analyze_model_file
from backend.utilities.logging import configure_logging
from backend.jobs.queue import (
    JobQueueConfig,
    create_job_queue,
)
from backend.api.jobs import (
    JobCreateRequest,
    JobResponse,
    JobTaskResponse,
    WorkflowJobCreateRequest,
    router as jobs_router,
    serialize_job as _serialize_job,
    serialize_job_task as _serialize_job_task,
)
from backend.api.workflow import router as workflow_router
from backend.api.history import router as history_router
from backend.api.presets import router as presets_router
from backend.api.models import router as models_router
from backend.api.loras import router as loras_router

from backend.workflow import (
    save_artifact_png,
)
from backend.workflow.utility import save_artifact_file

configure_logging(role=os.getenv("SYNTHA_LOG_ROLE", "api"))

app = FastAPI(title="SynthiaEngine API")
logger = logging.getLogger(__name__)


def _configured_cors_origins() -> list[str]:
    configured = os.getenv("SYNTHA_CORS_ORIGINS", "")
    if configured.strip():
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return ["http://127.0.0.1:4173", "http://localhost:4173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_configured_cors_origins(),
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Idempotency-Key"],
)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.include_router(jobs_router)
app.include_router(workflow_router)
app.include_router(history_router)
app.include_router(presets_router)
app.include_router(models_router)
app.include_router(loras_router)

ALLOWED_JOB_KINDS = {"workflow"}
UPLOAD_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".gif"}
MAX_ARTIFACT_UPLOAD_BYTES = int(os.getenv("SYNTHA_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
MAX_ARTIFACT_IMAGE_PIXELS = int(os.getenv("SYNTHA_MAX_IMAGE_PIXELS", str(64 * 1024 * 1024)))


class LocalPathSelectRequest(BaseModel):
    """Request payload used to open a local path picker on the API host."""

    selection_type: Literal["file", "folder"]


class LocalPathSelectResponse(BaseModel):
    """Selected local path returned by the API host picker."""

    path: str


class ControlNetPreprocessorInfo(BaseModel):
    """Serializable info about a ControlNet preprocessor implementation."""

    id: str
    name: str
    description: str
    defaults: dict[str, object]
    param_schema: dict[str, dict[str, object]] = Field(default_factory=dict)
    available: bool = True
    unavailable_reason: str | None = None
    install_hint: str | None = None
    recommended_sd15_control_models: list[str] = Field(default_factory=list)
    legacy_aliases: list[str] = Field(default_factory=list)


def _open_local_path_dialog(selection_type: Literal["file", "folder"]) -> str:
    """Open a native local picker on the backend host and return the path."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError("Local path picker is unavailable on this host.") from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if selection_type == "file":
            selected_path = filedialog.askopenfilename(parent=root, title="Select local model file")
        else:
            selected_path = filedialog.askdirectory(parent=root, title="Select local model folder")
    finally:
        root.destroy()

    return os.path.normpath(selected_path) if selected_path else ""


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
    architecture: str | None = None
    architecture_confidence: str = "unknown"
    metadata_available: bool = False
    safetensors_metadata: dict[str, str] = Field(default_factory=dict)
    metadata_keys: list[str] = Field(default_factory=list)
    architecture_evidence: list[str] = Field(default_factory=list)
    rows: list[ModelLayerRow]


def _env_flag_enabled(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _api_embedded_worker_enabled() -> bool:
    return _env_flag_enabled("SYNTHA_API_START_WORKER", default=True)


def _startup_job_queue() -> None:
    """Initialize queue state; retained as a directly testable lifecycle unit."""
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


def _shutdown_job_queue() -> None:
    worker = getattr(app.state, "job_worker", None)
    if worker is not None:
        worker.stop()
    engine = getattr(app.state, "job_engine", None)
    if engine is not None:
        engine.dispose()


@asynccontextmanager
async def _app_lifespan(_application: FastAPI):
    """Own queue initialization and renderer cleanup for the API lifespan."""
    _startup_job_queue()
    try:
        yield
    finally:
        _shutdown_job_queue()


app.router.lifespan_context = _app_lifespan


@app.get("/health")
async def health_check():
    """Basic liveness endpoint used by deployment/health checks."""
    return {"status": "ok"}


class ArtifactResponse(BaseModel):
    """Response payload describing a stored artifact in `OUTPUT_DIR`."""

    artifact_id: str
    url: str
    path: str


@app.post("/api/artifacts", response_model=ArtifactResponse, status_code=201)
async def upload_artifact(file: UploadFile = File(...)):
    """Upload an image or video artifact and persist it under `OUTPUT_DIR`."""
    file_bytes = await file.read(MAX_ARTIFACT_UPLOAD_BYTES + 1)
    if len(file_bytes) > MAX_ARTIFACT_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Artifact exceeds the configured upload size limit.")
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()
    content_type = (file.content_type or "").lower()
    if extension == ".gif" or content_type.startswith("video/"):
        try:
            artifact = save_artifact_file(file_bytes, extension=extension or ".mp4")
        except ValueError as save_exc:
            raise HTTPException(status_code=400, detail=str(save_exc)) from save_exc
        return ArtifactResponse(**artifact)

    try:
        image = Image.open(BytesIO(file_bytes))
        # Force decode early to catch truncated/invalid image streams.
        image.load()
        if image.width * image.height > MAX_ARTIFACT_IMAGE_PIXELS:
            raise HTTPException(status_code=413, detail="Image exceeds the configured pixel limit.")
        if image.mode == "P":
            # Palette images don't carry alpha in a convenient way for later steps.
            image = image.convert("RGBA")
        artifact = save_artifact_png(image, prefix="a")
        return ArtifactResponse(**artifact)
    except HTTPException:
        raise
    except Exception as exc:
        if extension in UPLOAD_VIDEO_EXTENSIONS:
            try:
                artifact = save_artifact_file(file_bytes, extension=extension or ".mp4")
            except ValueError as save_exc:
                raise HTTPException(status_code=400, detail=str(save_exc)) from save_exc
            return ArtifactResponse(**artifact)
        raise HTTPException(status_code=400, detail="Invalid image or video file.") from exc


@app.post("/api/local-path/select", response_model=LocalPathSelectResponse)
def select_local_path(req: LocalPathSelectRequest, request: Request):
    """Open a native local path picker on the API host."""
    client_host = request.client.host if request.client else ""
    allow_remote = os.getenv("SYNTHA_ALLOW_REMOTE_PATH_PICKER", "").strip().lower() in {"1", "true", "yes", "on"}
    if not allow_remote and client_host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        raise HTTPException(status_code=403, detail="Local path selection is restricted to loopback clients.")
    try:
        return LocalPathSelectResponse(path=_open_local_path_dialog(req.selection_type))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/controlnet/preprocessors", response_model=list[ControlNetPreprocessorInfo])
async def list_controlnet_preprocessors():
    """Return available ControlNet preprocessors and their default params."""
    preprocessors = list_preprocessors()
    registry_by_id = {entry.id: entry for entry in CONTROLNET_PREPROCESSOR_REGISTRY}
    infos: list[ControlNetPreprocessorInfo] = []
    for preprocessor in preprocessors:
        registry_entry = registry_by_id.get(preprocessor.id)
        implementation = get_preprocessor(preprocessor.id)
        available, unavailable_reason, install_hint = implementation.availability()
        infos.append(
            ControlNetPreprocessorInfo(
                id=preprocessor.id,
                name=preprocessor.name,
                description=preprocessor.description,
                defaults=preprocessor.defaults,
                available=available,
                unavailable_reason=unavailable_reason,
                install_hint=install_hint,
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
        if "is unavailable" in message:
            raise HTTPException(status_code=503, detail=message) from exc
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

        rows, loader, total, architecture = analyze_model_file(temp_path, limit=limit)
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
        architecture=architecture.architecture,
        architecture_confidence=architecture.confidence,
        metadata_available=architecture.metadata_available,
        safetensors_metadata=architecture.metadata,
        metadata_keys=architecture.metadata_keys,
        architecture_evidence=architecture.evidence,
        rows=[ModelLayerRow(key=k, shape=s, dtype=d) for k, s, d in rows],
    )


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
