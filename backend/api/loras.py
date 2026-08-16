from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.lora.registry import (
    LoraCompatibility,
    LoraPromptPreset,
    LoraRegistryEntry,
    LoraRuntimeProfile,
    add_lora,
    delete_lora_entry,
    get_lora_entry,
    list_lora_entries,
    update_lora_entry,
)

router = APIRouter(prefix="/lora-models", tags=["loras"])


def _validate_name(name: str | None) -> str | None:
    if name is not None and "." in name:
        raise ValueError("LoRA name cannot contain '.'")
    return name


class LoraCreateRequest(BaseModel):
    lora_id: int
    lora_model_family: str
    lora_type: str
    lora_location: str
    file_path: str
    name: str | None = None
    prompt_presets: list[LoraPromptPreset] = Field(default_factory=list)
    runtime_profile: LoraRuntimeProfile | None = None
    compatibility: LoraCompatibility | None = None
    weight_name: str | None = None
    subfolder: str | None = None
    revision: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        return _validate_name(value)


class LoraUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lora_model_family: str | None = None
    lora_type: str | None = None
    lora_location: str | None = None
    file_path: str | None = None
    name: str | None = None
    prompt_presets: list[LoraPromptPreset] | None = None
    runtime_profile: LoraRuntimeProfile | None = None
    compatibility: LoraCompatibility | None = None
    weight_name: str | None = None
    subfolder: str | None = None
    revision: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        return _validate_name(value)


@router.get("", response_model=list[LoraRegistryEntry])
async def list_lora_models(family: str | None = None):
    entries = list_lora_entries()
    family_value = (family or "").strip().lower()
    if not family_value:
        return entries
    return [entry for entry in entries if entry.lora_model_family.lower() == family_value]


@router.post("", response_model=LoraRegistryEntry)
async def create_lora_model(req: LoraCreateRequest):
    try:
        return add_lora(LoraRegistryEntry(**req.model_dump()))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{lora_id}", response_model=LoraRegistryEntry)
async def get_lora_model(lora_id: int):
    try:
        return get_lora_entry(lora_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/{lora_id}", response_model=LoraRegistryEntry)
async def patch_lora_model(lora_id: int, req: LoraUpdateRequest):
    updates = req.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="At least one editable field must be provided.")
    try:
        return update_lora_entry(lora_id, updates)
    except ValueError as exc:
        detail = str(exc)
        status = 404 if detail.endswith("not found.") else 400
        raise HTTPException(status_code=status, detail=detail) from exc


@router.delete("/{lora_id}", status_code=204)
async def remove_lora_model(lora_id: int):
    try:
        delete_lora_entry(lora_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(status_code=204)
