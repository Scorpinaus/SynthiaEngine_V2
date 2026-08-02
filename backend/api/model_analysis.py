"""Uploaded model inspection endpoint."""

from pathlib import Path
import shutil
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.utilities.model_analysis import SUPPORTED_EXTS, analyze_model_file


router = APIRouter(prefix="/api/tools", tags=["tools"])


class ModelLayerRow(BaseModel):
    key: str
    shape: str
    dtype: str


class ModelAnalysisResponse(BaseModel):
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


@router.post("/analyze-model", response_model=ModelAnalysisResponse)
async def analyze_model_layers(
    file: UploadFile = File(...),
    limit: int | None = Form(None),
):
    """Analyze a model upload using a temporary file that is always removed."""
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
        rows=[ModelLayerRow(key=key, shape=shape, dtype=dtype) for key, shape, dtype in rows],
    )
