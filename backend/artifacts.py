"""Shared artifact persistence independent of HTTP and workflow orchestration."""

from __future__ import annotations

from pathlib import Path
import uuid

from PIL import Image


VIDEO_ARTIFACT_EXTENSIONS = frozenset({".mp4", ".webm", ".mov", ".gif"})


def artifact_directory(output_dir: Path) -> Path:
    artifacts = output_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    return artifacts


def save_artifact_png(
    image: Image.Image,
    *,
    output_dir: Path,
    prefix: str = "a",
) -> dict[str, str]:
    artifact_id = f"{prefix}{uuid.uuid4().hex}"
    path = artifact_directory(output_dir) / f"{artifact_id}.png"
    image.save(path, format="PNG")
    relative_path = path.relative_to(output_dir).as_posix()
    return {
        "artifact_id": artifact_id,
        "path": relative_path,
        "url": f"/outputs/{relative_path}",
    }


def save_artifact_file(
    file_bytes: bytes,
    *,
    extension: str,
    output_dir: Path,
    prefix: str = "v",
) -> dict[str, str]:
    normalized_extension = extension.lower()
    if not normalized_extension.startswith("."):
        normalized_extension = f".{normalized_extension}"
    if normalized_extension not in VIDEO_ARTIFACT_EXTENSIONS:
        raise ValueError("Unsupported artifact file extension.")
    artifact_id = f"{prefix}{uuid.uuid4().hex}"
    path = artifact_directory(output_dir) / f"{artifact_id}{normalized_extension}"
    path.write_bytes(file_bytes)
    relative_path = path.relative_to(output_dir).as_posix()
    return {
        "artifact_id": artifact_id,
        "path": relative_path,
        "url": f"/outputs/{relative_path}",
    }
