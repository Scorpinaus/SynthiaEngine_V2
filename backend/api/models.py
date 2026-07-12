from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, ConfigDict

from backend.registries.model import (
    ModelRegistryEntry,
    create_model_entry,
    delete_model_entry,
    get_model_entry_exact,
    list_model_entries,
    update_model_entry,
)

router = APIRouter(prefix="/models", tags=["models"])


class ModelCreateRequest(BaseModel):
    name: str
    family: str
    model_type: str
    location_type: str
    model_id: int
    version: str
    link: str


class ModelUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: str | None = None
    model_type: str | None = None
    location_type: str | None = None
    model_id: int | None = None
    version: str | None = None
    link: str | None = None


def _family_pattern(family: str) -> re.Pattern[str]:
    aliases = {
        "sd15": r"sd[\s_-]*1\.?5|sd15",
        "sd1.5": r"sd[\s_-]*1\.?5|sd15",
        "sdxl": r"sdxl",
        "z-image-turbo": r"z-image-turbo",
        "qwen-image": r"qwen[-_\s]?image",
        "ernie-image": r"ernie[-_\s]?image",
        "anima": r"anima",
        "flux": r"flux",
    }
    return re.compile(aliases.get(family, re.escape(family)), re.IGNORECASE)


@router.get("", response_model=list[ModelRegistryEntry])
async def list_models(family: str | None = None):
    entries = list_model_entries()
    family_value = (family or "").strip().lower()
    if not family_value:
        return entries
    pattern = _family_pattern(family_value)
    return [entry for entry in entries if pattern.search(entry.family)]


@router.post("", response_model=ModelRegistryEntry, status_code=201)
async def create_model(req: ModelCreateRequest):
    try:
        return create_model_entry(ModelRegistryEntry(**req.model_dump()))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/{model_name:path}", response_model=ModelRegistryEntry)
async def get_model(model_name: str):
    try:
        return get_model_entry_exact(model_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/{model_name:path}", response_model=ModelRegistryEntry)
async def patch_model(model_name: str, req: ModelUpdateRequest):
    updates = req.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="At least one editable field must be provided.")
    try:
        return update_model_entry(model_name, updates)
    except ValueError as exc:
        detail = str(exc)
        if detail.endswith("not found."):
            status = 404
        elif detail == "Model name already exists.":
            status = 409
        else:
            status = 400
        raise HTTPException(status_code=status, detail=detail) from exc


@router.delete("/{model_name:path}", status_code=204)
async def remove_model(model_name: str):
    try:
        delete_model_entry(model_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(status_code=204)
