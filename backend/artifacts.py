"""Shared artifact persistence independent of HTTP and workflow orchestration."""

from __future__ import annotations

from pathlib import Path
import re
import uuid

from PIL import Image


VIDEO_ARTIFACT_EXTENSIONS = frozenset({".mp4", ".webm", ".mov", ".gif"})
_ARTIFACT_ID_RE = re.compile(r"^[apev][0-9a-f]{32}$")


def validate_artifact_id(value: str) -> str:
    artifact_id = value.strip()
    if not _ARTIFACT_ID_RE.match(artifact_id):
        raise ValueError("Invalid artifact_id")
    return artifact_id


def artifact_directory(output_dir: Path) -> Path:
    artifacts = output_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    return artifacts


def artifact_path_for_id(artifact_id: str, *, output_dir: Path) -> Path:
    safe_id = validate_artifact_id(artifact_id)
    artifacts = artifact_directory(output_dir)
    if safe_id.startswith("e"):
        return (artifacts / f"{safe_id}.pt").resolve()
    if safe_id.startswith("v"):
        for path in artifacts.glob(f"{safe_id}.*"):
            if path.suffix.lower() in VIDEO_ARTIFACT_EXTENSIONS:
                return path.resolve()
        return (artifacts / f"{safe_id}.mp4").resolve()
    return (artifacts / f"{safe_id}.png").resolve()


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
