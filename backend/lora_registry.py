import json
from pathlib import Path

from pydantic import BaseModel


class LoraRegistryEntry(BaseModel):
    lora_id: int
    lora_model_family: str
    lora_type: str
    lora_location: str
    file_path: str
    name: str | None = None


REGISTRY_PATH = Path(__file__).with_name("lora_registry.json")


def load_lora_registry() -> list[LoraRegistryEntry]:
    if not REGISTRY_PATH.exists():
        return []

    raw_data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw_data, list):
        raise ValueError("LoRA registry JSON must be a list of entries.")

    return [LoraRegistryEntry(**entry) for entry in raw_data]


LORA_REGISTRY: list[LoraRegistryEntry] = load_lora_registry()


def save_lora_registry(entries: list[LoraRegistryEntry]) -> None:
    payload = [entry.dict() for entry in entries]
    REGISTRY_PATH.write_text(json.dumps(payload, indent=4), encoding="utf-8")


def add_lora(entry: LoraRegistryEntry) -> LoraRegistryEntry:
    if any(existing.lora_id == entry.lora_id for existing in LORA_REGISTRY):
        raise ValueError(f"LoRA with id {entry.lora_id} already exists.")

    LORA_REGISTRY.append(entry)
    save_lora_registry(LORA_REGISTRY)
    return entry


def get_lora_entry(lora_id: int) -> LoraRegistryEntry:
    for entry in LORA_REGISTRY:
        if entry.lora_id == lora_id:
            return entry
    raise ValueError(f"LoRA with id {lora_id} not found.")
