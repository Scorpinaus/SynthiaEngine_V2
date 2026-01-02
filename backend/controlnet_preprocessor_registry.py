import json
from pathlib import Path

from pydantic import BaseModel


class ControlNetPreprocessorModelEntry(BaseModel):
    id: str
    name: str
    description: str
    repo_id: str | None = None
    revision: str | None = None


REGISTRY_PATH = Path(__file__).with_name("controlnet_preprocessor_registry.json")


def load_controlnet_preprocessor_registry() -> list[ControlNetPreprocessorModelEntry]:
    if not REGISTRY_PATH.exists():
        return []

    raw_data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw_data, list):
        raise ValueError("ControlNet preprocessor registry JSON must be a list of entries.")

    return [ControlNetPreprocessorModelEntry(**entry) for entry in raw_data]


CONTROLNET_PREPROCESSOR_REGISTRY: list[ControlNetPreprocessorModelEntry] = (
    load_controlnet_preprocessor_registry()
)


def save_controlnet_preprocessor_registry(
    entries: list[ControlNetPreprocessorModelEntry],
) -> None:
    payload = [entry.model_dump() for entry in entries]
    REGISTRY_PATH.write_text(json.dumps(payload, indent=4), encoding="utf-8")
