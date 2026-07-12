from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from backend.registries.preset import (
    PresetRegistryCreate,
    PresetRegistryEntry,
    create_preset_entry,
    delete_preset_entry,
    get_preset_entry,
    list_preset_entries,
    update_preset_entry,
)

router = APIRouter(prefix="/api/presets", tags=["presets"])


class PresetCreateRequest(BaseModel):
    name: str
    family: str
    task_type: str
    settings: dict[str, Any] = Field(default_factory=dict)


class PresetUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    family: str | None = None
    task_type: str | None = None
    settings: dict[str, Any] | None = None


@router.get("", response_model=list[PresetRegistryEntry])
async def list_presets(family: str | None = None, task_type: str | None = None):
    try:
        return list_preset_entries(family=family, task_type=task_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("", response_model=PresetRegistryEntry, status_code=201)
async def create_preset(req: PresetCreateRequest):
    try:
        return create_preset_entry(PresetRegistryCreate(**req.model_dump()))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{preset_id}", response_model=PresetRegistryEntry)
async def get_preset(preset_id: int):
    try:
        return get_preset_entry(preset_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/{preset_id}", response_model=PresetRegistryEntry)
async def patch_preset(preset_id: int, req: PresetUpdateRequest):
    updates = req.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="At least one editable field must be provided.")
    try:
        return update_preset_entry(preset_id, updates)
    except ValueError as exc:
        detail = str(exc)
        status = 404 if detail.endswith("not found.") else 400
        raise HTTPException(status_code=status, detail=detail) from exc


@router.delete("/{preset_id}", status_code=204)
async def remove_preset(preset_id: int):
    try:
        delete_preset_entry(preset_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(status_code=204)
