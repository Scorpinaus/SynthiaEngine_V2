from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter
from PIL import Image

from backend.config import OUTPUT_DIR

router = APIRouter(tags=["history"])
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov"}


def _read_png_metadata(path: Path) -> dict[str, object]:
    try:
        with Image.open(path) as image:
            metadata = dict(getattr(image, "text", {}))
            metadata.update(
                (key, value)
                for key, value in (image.info or {}).items()
                if isinstance(value, str) and key not in metadata
            )
            return metadata
    except Exception as exc:
        logger.warning("Failed to read metadata for %s: %s", path.name, exc)
        return {}


def _read_json_metadata(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read metadata sidecar for %s: %s", path.name, exc)
        return {}
    if isinstance(payload, dict):
        return payload
    logger.warning("Ignoring metadata sidecar with non-object payload: %s", path.name)
    return {}


def _batch_id(relative_path: Path) -> str | None:
    for part in relative_path.parts[:-1]:
        if part.startswith("batch_") and len(part) > len("batch_"):
            return part.removeprefix("batch_")
    return None


def _video_metadata(media_path: Path, relative_path: Path) -> dict[str, object]:
    batch_id = _batch_id(relative_path)
    metadata: dict[str, object] = {}
    if batch_id:
        sidecar = media_path.parent / f"video_{batch_id}.mp4.json"
        if sidecar.exists():
            metadata.update(_read_json_metadata(sidecar))
            videos = metadata.pop("videos", None)
            if isinstance(videos, list):
                for entry in videos:
                    if not isinstance(entry, dict):
                        continue
                    if entry.get("filename") == media_path.name or entry.get("path") == relative_path.as_posix():
                        metadata.update(
                            (key, value) for key, value in entry.items() if key not in {"filename", "path"}
                        )
                        break
        metadata.setdefault("batch_id", batch_id)
    return metadata


def _media_type(path: Path) -> str | None:
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return "image"
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        return "video"
    return None


@router.get("/history")
async def list_history():
    if not OUTPUT_DIR.exists():
        return []

    records: list[dict[str, object]] = []
    for media_path in OUTPUT_DIR.rglob("*"):
        media_type = _media_type(media_path) if media_path.is_file() else None
        if media_type is None:
            continue
        timestamp = media_path.stat().st_mtime
        relative_path = media_path.relative_to(OUTPUT_DIR)
        metadata = (
            _read_png_metadata(media_path)
            if media_type == "image"
            else _video_metadata(media_path, relative_path)
        )
        relative_path_text = relative_path.as_posix()
        records.append(
            {
                "filename": relative_path_text,
                "url": f"/outputs/{relative_path_text}",
                "timestamp": timestamp,
                "created_at": datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
                "media_type": media_type,
                "metadata": metadata,
            }
        )
    return sorted(records, key=lambda item: item.get("timestamp", 0), reverse=True)
