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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageFilter
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.config import DEFAULTS, OUTPUT_DIR
from backend.adapters.controlnet_preprocessors import get_preprocessor, list_preprocessors
from backend.adapters.controlnet_preprocessor_registry import (
    CONTROLNET_PREPROCESSOR_REGISTRY,
    ControlNetPreprocessorModelEntry,
)
from backend.utilities.model_analysis import SUPPORTED_EXTS, analyze_model_file
from backend.registries.model import (
    ModelRegistryEntry,
    create_model_entry,
    delete_model_entry,
    get_model_entry_exact,
    list_model_entries,
    update_model_entry,
)
from backend.lora.registry import (
    LoraPromptPreset,
    LoraRegistryEntry,
    add_lora,
    delete_lora_entry,
    get_lora_entry,
    list_lora_entries,
    update_lora_entry,
)
from backend.utilities.logging import configure_logging
from backend.registries.preset import (
    PresetRegistryCreate,
    PresetRegistryEntry,
    create_preset_entry,
    delete_preset_entry,
    get_preset_entry,
    list_preset_entries,
    update_preset_entry,
)
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

ALLOWED_JOB_KINDS = {"workflow"}
HISTORY_IMAGE_EXTENSIONS = {".png"}
HISTORY_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov"}
UPLOAD_VIDEO_EXTENSIONS = HISTORY_VIDEO_EXTENSIONS | {".gif"}
MAX_ARTIFACT_UPLOAD_BYTES = int(os.getenv("SYNTHA_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
MAX_ARTIFACT_IMAGE_PIXELS = int(os.getenv("SYNTHA_MAX_IMAGE_PIXELS", str(64 * 1024 * 1024)))


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


class LocalPathSelectRequest(BaseModel):
    """Request payload used to open a local path picker on the API host."""

    selection_type: Literal["file", "folder"]


class LocalPathSelectResponse(BaseModel):
    """Selected local path returned by the API host picker."""

    path: str


class LoraCreateRequest(BaseModel):
    """Request payload used to register a new LoRA entry in the local registry."""

    lora_id: int
    lora_model_family: str
    lora_type: str
    lora_location: str
    file_path: str
    name: str | None = None
    prompt_presets: list[LoraPromptPreset] = Field(default_factory=list)

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
    prompt_presets: list[LoraPromptPreset] | None = None

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


def _extract_png_metadata(path: Path) -> dict[str, object]:
    """Extract embedded PNG text metadata in a safe, best-effort way."""
    try:
        with Image.open(path) as image:
            metadata: dict[str, object] = {}
            if hasattr(image, "text"):
                metadata.update(image.text)
            for key, value in (image.info or {}).items():
                if isinstance(value, str) and key not in metadata:
                    metadata[key] = value
            return metadata
    except Exception as exc:
        logger.warning("Failed to read metadata for %s: %s", path.name, exc)
        return {}


def _extract_json_metadata(path: Path) -> dict[str, object]:
    """Extract JSON sidecar metadata in a safe, best-effort way."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read metadata sidecar for %s: %s", path.name, exc)
        return {}
    if not isinstance(payload, dict):
        logger.warning("Ignoring metadata sidecar with non-object payload: %s", path.name)
        return {}
    return payload


def _infer_batch_id_from_outputs_path(relative_path: Path) -> str | None:
    """Infer a batch id from outputs/batch_<id>/... media paths."""
    for part in relative_path.parts[:-1]:
        if part.startswith("batch_") and len(part) > len("batch_"):
            return part.removeprefix("batch_")
    return None


def _video_metadata_sidecar_path(media_path: Path, relative_path: Path) -> Path | None:
    batch_id = _infer_batch_id_from_outputs_path(relative_path)
    if not batch_id:
        return None
    return media_path.parent / f"video_{batch_id}.mp4.json"


def _matching_video_sidecar_entry(
    videos: object,
    *,
    media_name: str,
    relative_path: str,
) -> dict[str, object]:
    if not isinstance(videos, list):
        return {}
    for entry in videos:
        if not isinstance(entry, dict):
            continue
        entry_filename = str(entry.get("filename") or "")
        entry_path = str(entry.get("path") or "")
        if entry_filename == media_name or entry_path == relative_path:
            return {
                key: value
                for key, value in entry.items()
                if key not in {"filename", "path"}
            }
    return {}


def _extract_video_metadata(media_path: Path, relative_path: Path) -> dict[str, object]:
    """Extract adjacent video sidecar metadata, falling back to path-derived batch id."""
    metadata: dict[str, object] = {}
    relative_path_text = relative_path.as_posix()
    sidecar_path = _video_metadata_sidecar_path(media_path, relative_path)
    if sidecar_path is not None and sidecar_path.exists():
        metadata.update(_extract_json_metadata(sidecar_path))
        videos = metadata.pop("videos", None)
        metadata.update(
            _matching_video_sidecar_entry(
                videos,
                media_name=media_path.name,
                relative_path=relative_path_text,
            )
        )

    batch_id = _infer_batch_id_from_outputs_path(relative_path)
    if batch_id and not metadata.get("batch_id"):
        metadata["batch_id"] = batch_id
    return metadata


def _history_media_type(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in HISTORY_IMAGE_EXTENSIONS:
        return "image"
    if suffix in HISTORY_VIDEO_EXTENSIONS:
        return "video"
    return None


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
    elif family_value == "ernie-image":
        pattern = re.compile(r"ernie[-_\s]?image", re.IGNORECASE)
    elif family_value == "anima":
        pattern = re.compile(r"anima", re.IGNORECASE)
    elif family_value == "flux":
        pattern = re.compile(r"flux", re.IGNORECASE)
    else:
        pattern = re.compile(re.escape(family_value), re.IGNORECASE)

    return [entry for entry in entries if pattern.search(entry.family)]


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
    """List generated media from `OUTPUT_DIR` along with embedded metadata."""
    if not OUTPUT_DIR.exists():
        return []

    records: list[dict[str, object]] = []
    # Walk the outputs folder to produce a lightweight generation history feed.
    for media_path in OUTPUT_DIR.rglob("*"):
        if not media_path.is_file():
            continue
        media_type = _history_media_type(media_path)
        if media_type is None:
            continue

        stat = media_path.stat()
        timestamp = stat.st_mtime
        created_at = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        relative_path_obj = media_path.relative_to(OUTPUT_DIR)
        if media_type == "image":
            metadata = _extract_png_metadata(media_path)
        else:
            metadata = _extract_video_metadata(media_path, relative_path_obj)
        relative_path = relative_path_obj.as_posix()
        records.append(
            {
                "filename": relative_path,
                "url": f"/outputs/{relative_path}",
                "timestamp": timestamp,
                "created_at": created_at,
                "media_type": media_type,
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
