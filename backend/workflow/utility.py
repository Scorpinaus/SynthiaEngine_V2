from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PIL import Image

from backend.artifacts import (
    VIDEO_ARTIFACT_EXTENSIONS,
    artifact_path_for_id as shared_artifact_path_for_id,
    save_artifact_file as persist_artifact_file,
    save_artifact_png as persist_artifact_png,
    validate_artifact_id as shared_validate_artifact_id,
)
from backend.config import OUTPUT_DIR
from backend.workflow.schema_input import _DEFAULT_SD15_CONTROLNET_MODEL


def _artifact_dir() -> Path:
    artifacts = OUTPUT_DIR / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    return artifacts


_IMAGE_ARTIFACT_ID_RE = re.compile(r"^[ap][0-9a-f]{32}$")
_VIDEO_ARTIFACT_ID_RE = re.compile(r"^v[0-9a-f]{32}$")
_VIDEO_ARTIFACT_EXTENSIONS = VIDEO_ARTIFACT_EXTENSIONS


def _validate_artifact_id(value: str) -> str:
    return shared_validate_artifact_id(value)


def _validate_image_artifact_id(value: str) -> str:
    artifact_id = value.strip()
    if not _IMAGE_ARTIFACT_ID_RE.match(artifact_id):
        raise ValueError("Invalid image artifact_id")
    return artifact_id


def _validate_video_artifact_id(value: str) -> str:
    artifact_id = value.strip()
    if not _VIDEO_ARTIFACT_ID_RE.match(artifact_id):
        raise ValueError("Invalid video artifact_id")
    return artifact_id


def _artifact_path_for_id(artifact_id: str) -> Path:
    return shared_artifact_path_for_id(artifact_id, output_dir=OUTPUT_DIR)


def collect_artifact_ids(value: Any) -> set[str]:
    out: set[str] = set()

    def _walk(node: Any) -> None:
        if isinstance(node, str):
            if node.startswith("@artifact:"):
                try:
                    out.add(_validate_artifact_id(node.removeprefix("@artifact:")))
                except ValueError:
                    pass
            return

        if isinstance(node, dict):
            artifact_id = node.get("artifact_id")
            if isinstance(artifact_id, str):
                try:
                    out.add(_validate_artifact_id(artifact_id))
                except ValueError:
                    pass
            for v in node.values():
                _walk(v)
            return

        if isinstance(node, list):
            for item in node:
                _walk(item)
            return

    _walk(value)
    return out


def cleanup_artifacts(artifact_ids: set[str]) -> None:
    if not artifact_ids:
        return
    for artifact_id in artifact_ids:
        try:
            path = _artifact_path_for_id(artifact_id)
        except ValueError:
            continue
        if not str(path).startswith(str(_artifact_dir().resolve())):
            continue
        try:
            path.unlink(missing_ok=True)
        except Exception:
            # Best-effort cleanup; job success shouldn't depend on deletion.
            pass


def save_artifact_png(image: Image.Image, *, prefix: str = "a") -> dict[str, str]:
    return persist_artifact_png(image, output_dir=OUTPUT_DIR, prefix=prefix)


def save_artifact_file(file_bytes: bytes, *, extension: str, prefix: str = "v") -> dict[str, str]:
    return persist_artifact_file(
        file_bytes,
        extension=extension,
        output_dir=OUTPUT_DIR,
        prefix=prefix,
    )


def _load_image_from_outputs_url(url: str) -> Image.Image:
    if not url.startswith("/outputs/"):
        raise ValueError("Expected /outputs/ URL.")
    rel = url.removeprefix("/outputs/").lstrip("/")
    path = (OUTPUT_DIR / rel).resolve()
    if not str(path).startswith(str(OUTPUT_DIR.resolve())):
        raise ValueError("Invalid outputs path.")
    with Image.open(path) as img:
        return img.copy()


def _open_image_ref(value: Any) -> Image.Image:
    if isinstance(value, dict) and "artifact_id" in value:
        artifact_id = _validate_image_artifact_id(str(value["artifact_id"]))
        path = (_artifact_dir() / f"{artifact_id}.png").resolve()
        with Image.open(path) as img:
            return img.copy()
    if isinstance(value, str) and value.startswith("@artifact:"):
        artifact_id = _validate_image_artifact_id(value.removeprefix("@artifact:"))
        path = (_artifact_dir() / f"{artifact_id}.png").resolve()
        with Image.open(path) as img:
            return img.copy()
    if isinstance(value, str) and value.startswith("/outputs/"):
        return _load_image_from_outputs_url(value)
    raise ValueError("Unsupported image reference.")


def _video_path_from_outputs_url(url: str) -> Path:
    if not url.startswith("/outputs/"):
        raise ValueError("Expected /outputs/ URL.")
    rel = url.removeprefix("/outputs/").lstrip("/")
    path = (OUTPUT_DIR / rel).resolve()
    if not str(path).startswith(str(OUTPUT_DIR.resolve())):
        raise ValueError("Invalid outputs path.")
    if path.suffix.lower() not in _VIDEO_ARTIFACT_EXTENSIONS:
        raise ValueError("Unsupported video file extension.")
    return path


def _open_video_ref(value: Any) -> Path:
    if isinstance(value, dict) and "artifact_id" in value:
        artifact_id = _validate_video_artifact_id(str(value["artifact_id"]))
        return _artifact_path_for_id(artifact_id)
    if isinstance(value, str) and value.startswith("@artifact:"):
        artifact_id = _validate_video_artifact_id(value.removeprefix("@artifact:"))
        return _artifact_path_for_id(artifact_id)
    if isinstance(value, str) and value.startswith("/outputs/"):
        return _video_path_from_outputs_url(value)
    raise ValueError("Unsupported video reference.")


def _resolve_refs(value: Any, task_results: dict[str, dict[str, Any]]) -> Any:
    if isinstance(value, str) and value.startswith("@"):
        token = value[1:]
        if token.startswith("artifact:"):
            return {"artifact_id": token.removeprefix("artifact:").strip()}
        if "." in token:
            task_id, path = token.split(".", 1)
            if task_id not in task_results:
                raise KeyError(f"Unknown task id: {task_id}")
            current: Any = task_results[task_id]
            path_tokens = re.findall(r"(?:^|\.)([A-Za-z_][A-Za-z0-9_-]*)|\[(\d+)\]", path)
            if not path_tokens or "".join(
                (f".{key}" if index == "" else f"[{index}]")
                for key, index in path_tokens
            ).lstrip(".") != path:
                raise ValueError(f"Invalid reference path: {value}")
            for key, index in path_tokens:
                if key:
                    if not isinstance(current, dict) or key not in current:
                        raise KeyError(f"Reference field not found: {value}")
                    current = current[key]
                else:
                    position = int(index)
                    if not isinstance(current, (list, tuple)) or position >= len(current):
                        raise IndexError(f"Reference index out of range: {value}")
                    current = current[position]
            return current
        raise ValueError(f"Invalid reference: {value}")

    if isinstance(value, list):
        return [_resolve_refs(item, task_results) for item in value]
    if isinstance(value, dict):
        return {k: _resolve_refs(v, task_results) for k, v in value.items()}
    return value


def collect_task_refs(value: Any) -> set[str]:
    """Return task ids referenced anywhere in a workflow value."""
    refs: set[str] = set()
    if isinstance(value, str) and value.startswith("@"):
        token = value[1:]
        if not token.startswith("artifact:") and "." in token:
            refs.add(token.split(".", 1)[0])
    elif isinstance(value, list):
        for item in value:
            refs |= collect_task_refs(item)
    elif isinstance(value, dict):
        for item in value.values():
            refs |= collect_task_refs(item)
    return refs


def _normalized_lora_adapters(inputs: dict[str, Any]) -> Any:
    unified_lora = inputs.get("lora")
    if isinstance(unified_lora, dict):
        lora_enabled_raw = unified_lora.get("lora_enabled")
        if lora_enabled_raw is not None and not bool(lora_enabled_raw):
            return []

        unified_adapters = unified_lora.get("lora_adapters")
        if isinstance(unified_adapters, list):
            return unified_adapters

    lora_adapters = inputs.get("lora_adapters")
    if lora_adapters is not None:
        return lora_adapters

    lora_contract = inputs.get("Lora")
    if isinstance(lora_contract, dict):
        lora_status_raw = lora_contract.get("loraStatus")
        if lora_status_raw is not None and not bool(lora_status_raw):
            return []
        adapters = lora_contract.get("adapters")
        if isinstance(adapters, list):
            return adapters
    return None


def _normalized_sd15_lora_adapters(inputs: dict[str, Any]) -> Any:
    if "Lora" in inputs:
        raise ValueError(
            "Legacy SD1.5 LoRA field `Lora` is no longer supported. "
            "Use `lora` with `lora_enabled` and `lora_adapters`."
        )

    if "lora_adapters" in inputs:
        raise ValueError(
            "Top-level SD1.5 `lora_adapters` is no longer supported. "
            "Use `lora.lora_adapters`."
        )

    unified_lora = inputs.get("lora")
    if unified_lora is None:
        return None
    if not isinstance(unified_lora, dict):
        raise ValueError("`lora` must be an object with `lora_enabled` and `lora_adapters`.")

    lora_enabled_raw = unified_lora.get("lora_enabled")
    if lora_enabled_raw is not None and not bool(lora_enabled_raw):
        return []

    unified_adapters = unified_lora.get("lora_adapters")
    if unified_adapters is None:
        return []
    if not isinstance(unified_adapters, list):
        raise ValueError("`lora.lora_adapters` must be a list.")
    return unified_adapters


def _normalized_hires_settings(inputs: dict[str, Any]) -> tuple[bool, float]:
    hires_enabled = bool(inputs.get("hires_enabled") or False)
    hires_scale = float(inputs.get("hires_scale") or 1.0)

    hires_contract = inputs.get("hires")
    if isinstance(hires_contract, dict):
        if "hiresEnabled" in hires_contract and "hires_enabled" not in inputs:
            hires_enabled = bool(hires_contract.get("hiresEnabled"))
        if "hires_scale" in hires_contract and "hires_scale" not in inputs:
            hires_scale = float(hires_contract.get("hires_scale") or 1.0)

    return hires_enabled, hires_scale


def _normalize_sd15_controlnet_contract_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(inputs)
    effective_items = normalized.get("effectiveItems")
    if effective_items is not None and not isinstance(effective_items, list):
        raise ValueError("effectiveItems must be a list of objects.")

    if effective_items:
        control_images: list[Any] = []
        controlnet_models: list[str] = []
        controlnet_scales: list[float] = []
        control_guidance_starts: list[float | None] = []
        control_guidance_ends: list[float | None] = []
        controlnet_preprocessor_ids: list[str | None] = []

        for idx, item in enumerate(effective_items):
            if not isinstance(item, dict):
                raise ValueError(f"effectiveItems[{idx}] must be an object.")
            if "control_image" not in item:
                raise ValueError(f"effectiveItems[{idx}].control_image is required.")

            control_images.append(item["control_image"])
            controlnet_models.append(
                str(item.get("model_id") or _DEFAULT_SD15_CONTROLNET_MODEL)
            )
            controlnet_scales.append(float(item.get("conditioning_scale") or 1.0))
            guidance_start = item.get("guidance_start")
            guidance_end = item.get("guidance_end")
            control_guidance_starts.append(
                float(guidance_start) if guidance_start is not None else None
            )
            control_guidance_ends.append(
                float(guidance_end) if guidance_end is not None else None
            )
            preprocessor_id = item.get("preprocessor_id")
            controlnet_preprocessor_ids.append(
                str(preprocessor_id) if preprocessor_id is not None else None
            )

        if "control_image" not in normalized and control_images:
            normalized["control_image"] = control_images[0]
        if "control_images" not in normalized and len(control_images) > 1:
            normalized["control_images"] = control_images
        if "controlnet_models" not in normalized and len(controlnet_models) > 1:
            normalized["controlnet_models"] = controlnet_models
        if "controlnet_model" not in normalized and controlnet_models:
            normalized["controlnet_model"] = controlnet_models[0]
        if "controlnet_conditioning_scales" not in normalized and len(controlnet_scales) > 1:
            normalized["controlnet_conditioning_scales"] = controlnet_scales
        if "controlnet_conditioning_scale" not in normalized and controlnet_scales:
            normalized["controlnet_conditioning_scale"] = controlnet_scales[0]

        if any(value is not None for value in control_guidance_starts):
            default_start = float(normalized.get("control_guidance_start", 0.0))
            resolved_starts = [
                default_start if value is None else value for value in control_guidance_starts
            ]
            if "control_guidance_starts" not in normalized and len(resolved_starts) > 1:
                normalized["control_guidance_starts"] = resolved_starts
            if "control_guidance_start" not in normalized and resolved_starts:
                normalized["control_guidance_start"] = resolved_starts[0]

        if any(value is not None for value in control_guidance_ends):
            default_end = float(normalized.get("control_guidance_end", 1.0))
            resolved_ends = [
                default_end if value is None else value for value in control_guidance_ends
            ]
            if "control_guidance_ends" not in normalized and len(resolved_ends) > 1:
                normalized["control_guidance_ends"] = resolved_ends
            if "control_guidance_end" not in normalized and resolved_ends:
                normalized["control_guidance_end"] = resolved_ends[0]

        has_all_preprocessor_ids = all(
            isinstance(value, str) and len(value) > 0 for value in controlnet_preprocessor_ids
        )
        if has_all_preprocessor_ids:
            if "controlnet_preprocessor_ids" not in normalized and len(controlnet_preprocessor_ids) > 1:
                normalized["controlnet_preprocessor_ids"] = controlnet_preprocessor_ids
            if "controlnet_preprocessor_id" not in normalized and controlnet_preprocessor_ids:
                normalized["controlnet_preprocessor_id"] = controlnet_preprocessor_ids[0]

    if bool(normalized.get("controlNetEnabled")) and not (
        normalized.get("control_image") is not None
        or (isinstance(normalized.get("control_images"), list) and normalized.get("control_images"))
    ):
        raise ValueError(
            "controlNetEnabled is true but no control image references were provided."
        )

    return normalized


def _remap_img2img_strength(strength: float, *, min_strength: float = 0.0, gamma: float = 0.5) -> float:
    clamped = max(0.0, min(1.0, strength))
    if min_strength <= 0.0:
        remapped = clamped**gamma
    else:
        normalized = max(0.0, min(1.0, (clamped - min_strength) / (1.0 - min_strength)))
        remapped = min_strength + (normalized**gamma) * (1.0 - min_strength)
    return max(0.0, min(1.0, remapped))
