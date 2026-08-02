"""ControlNet preprocessor catalog and image preprocessing endpoints."""

from io import BytesIO
import json

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

from backend.adapters.controlnet_preprocessor_registry import (
    CONTROLNET_PREPROCESSOR_REGISTRY,
    ControlNetPreprocessorModelEntry,
)
from backend.adapters.controlnet_preprocessors import get_preprocessor, list_preprocessors


router = APIRouter(prefix="/api/controlnet", tags=["controlnet"])


class ControlNetPreprocessorInfo(BaseModel):
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


@router.get("/preprocessors", response_model=list[ControlNetPreprocessorInfo])
async def list_controlnet_preprocessors():
    """Return preprocessors, parameter schemas, and availability details."""
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
                legacy_aliases=registry_entry.legacy_aliases if registry_entry else [],
            )
        )
    return infos


@router.get(
    "/preprocessor-models",
    response_model=list[ControlNetPreprocessorModelEntry],
)
async def list_controlnet_preprocessor_models():
    """Return ControlNet model entries used by the UI."""
    return CONTROLNET_PREPROCESSOR_REGISTRY


@router.post("/preprocess")
async def run_controlnet_preprocessor(
    image: UploadFile = File(...),
    preprocessor_id: str = Form(...),
    params: str | None = Form(None),
    low_threshold: int | None = Form(None),
    high_threshold: int | None = Form(None),
):
    """Run a selected preprocessor over an uploaded image and return a PNG."""
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
