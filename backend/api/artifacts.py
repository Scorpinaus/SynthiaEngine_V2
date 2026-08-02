"""Artifact upload endpoints."""

from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from PIL import Image
from pydantic import BaseModel

from backend.api.dependencies import get_app_settings
from backend.artifacts import save_artifact_file, save_artifact_png


router = APIRouter(tags=["artifacts"])

UPLOAD_VIDEO_EXTENSIONS = frozenset({".mp4", ".webm", ".mov", ".gif"})


class ArtifactResponse(BaseModel):
    """Response payload describing a stored artifact."""

    artifact_id: str
    url: str
    path: str


@router.post("/api/artifacts", response_model=ArtifactResponse, status_code=201)
async def upload_artifact(request: Request, file: UploadFile = File(...)):
    """Upload an image or video artifact under the configured output directory."""
    settings = get_app_settings(request)
    upload_limit = settings.api.max_artifact_upload_bytes
    file_bytes = await file.read(upload_limit + 1)
    if len(file_bytes) > upload_limit:
        raise HTTPException(
            status_code=413,
            detail="Artifact exceeds the configured upload size limit.",
        )

    filename = file.filename or ""
    extension = Path(filename).suffix.lower()
    content_type = (file.content_type or "").lower()
    if extension == ".gif" or content_type.startswith("video/"):
        try:
            artifact = save_artifact_file(
                file_bytes,
                extension=extension or ".mp4",
                output_dir=settings.paths.output_dir,
            )
        except ValueError as save_exc:
            raise HTTPException(status_code=400, detail=str(save_exc)) from save_exc
        return ArtifactResponse(**artifact)

    try:
        image = Image.open(BytesIO(file_bytes))
        image.load()
        if image.width * image.height > settings.api.max_artifact_image_pixels:
            raise HTTPException(
                status_code=413,
                detail="Image exceeds the configured pixel limit.",
            )
        if image.mode == "P":
            image = image.convert("RGBA")
        artifact = save_artifact_png(
            image,
            output_dir=settings.paths.output_dir,
            prefix="a",
        )
        return ArtifactResponse(**artifact)
    except HTTPException:
        raise
    except Exception as exc:
        if extension in UPLOAD_VIDEO_EXTENSIONS:
            try:
                artifact = save_artifact_file(
                    file_bytes,
                    extension=extension or ".mp4",
                    output_dir=settings.paths.output_dir,
                )
            except ValueError as save_exc:
                raise HTTPException(status_code=400, detail=str(save_exc)) from save_exc
            return ArtifactResponse(**artifact)
        raise HTTPException(status_code=400, detail="Invalid image or video file.") from exc
