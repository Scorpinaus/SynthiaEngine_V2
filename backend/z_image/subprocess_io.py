from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

_IMAGE_MARKER = "__z_image_subprocess_image__"


def serialize_params_for_subprocess(params: dict[str, object], tmp_dir: Path) -> dict[str, Any]:
    image_index = 0

    def _serialize(value: Any) -> Any:
        nonlocal image_index
        if isinstance(value, Image.Image):
            image_path = tmp_dir / f"image_{image_index}.png"
            image_index += 1
            value.save(image_path)
            return {
                _IMAGE_MARKER: str(image_path),
                "mode": value.mode,
            }
        if isinstance(value, list):
            return [_serialize(item) for item in value]
        if isinstance(value, tuple):
            return [_serialize(item) for item in value]
        if isinstance(value, dict):
            return {str(key): _serialize(item) for key, item in value.items()}
        return value

    return {str(key): _serialize(value) for key, value in params.items()}


def deserialize_params_from_subprocess(params: dict[str, Any]) -> dict[str, object]:
    def _deserialize(value: Any) -> Any:
        if isinstance(value, dict):
            image_path = value.get(_IMAGE_MARKER)
            if isinstance(image_path, str):
                image = Image.open(image_path)
                image.load()
                mode = value.get("mode")
                if isinstance(mode, str) and image.mode != mode:
                    image = image.convert(mode)
                return image
            return {str(key): _deserialize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_deserialize(item) for item in value]
        return value

    return {str(key): _deserialize(value) for key, value in params.items()}
