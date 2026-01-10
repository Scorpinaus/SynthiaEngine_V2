import json
import re
from pathlib import Path

from pydantic import BaseModel


class ModelRegistryEntry(BaseModel):
    name: str
    family: str
    model_type: str
    location_type: str
    model_id: int
    version: str
    link: str


REGISTRY_PATH = Path(__file__).with_name("model_registry.json")


def load_model_registry() -> list[ModelRegistryEntry]:
    if not REGISTRY_PATH.exists():
        return []

    raw_data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw_data, list):
        raise ValueError("Model registry JSON must be a list of entries.")

    return [ModelRegistryEntry(**entry) for entry in raw_data]


MODEL_REGISTRY: list[ModelRegistryEntry] = load_model_registry()


def save_model_registry(entries: list[ModelRegistryEntry]) -> None:
    payload = [entry.dict() for entry in entries]
    REGISTRY_PATH.write_text(json.dumps(payload, indent=4), encoding="utf-8")


def get_model_entry(model_name: str | None) -> ModelRegistryEntry:
    if model_name:
        for entry in MODEL_REGISTRY:
            if entry.name == model_name:
                return entry

    if not MODEL_REGISTRY:
        raise ValueError("Model registry is empty.")

    return MODEL_REGISTRY[0]


def get_model_family(model_name: str | None) -> str | None:
    if model_name:
        for entry in MODEL_REGISTRY:
            if entry.name == model_name:
                return entry.family

        lowered = model_name.lower()
        if re.search(r"flux", lowered):
            return "flux"
        if re.search(r"sdxl", lowered):
            return "sdxl"
        if re.search(r"qwen[-_\s]?image", lowered):
            return "qwen-image"
        if re.search(r"z[-_\s]?image|turbo", lowered):
            return "z-image-turbo"
        if re.search(r"sd[\s_-]*1\.?5|sd15", lowered):
            return "sd15"

    if MODEL_REGISTRY:
        return MODEL_REGISTRY[0].family

    return None
