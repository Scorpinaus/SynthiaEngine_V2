from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from backend.config import OUTPUT_DIR
from backend.workflow_schema_input import _DEFAULT_SD15_CONTROLNET_MODEL


def _artifact_dir() -> Path:
    artifacts = OUTPUT_DIR / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    return artifacts


_ARTIFACT_ID_RE = re.compile(r"^[ap][0-9a-f]{32}$")


def _validate_artifact_id(value: str) -> str:
    artifact_id = value.strip()
    if not _ARTIFACT_ID_RE.match(artifact_id):
        raise ValueError("Invalid artifact_id")
    return artifact_id


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
    artifacts_dir = _artifact_dir().resolve()
    for artifact_id in artifact_ids:
        try:
            safe_id = _validate_artifact_id(artifact_id)
        except ValueError:
            continue
        path = (artifacts_dir / f"{safe_id}.png").resolve()
        if not str(path).startswith(str(artifacts_dir)):
            continue
        try:
            path.unlink(missing_ok=True)
        except Exception:
            # Best-effort cleanup; job success shouldn't depend on deletion.
            pass


def save_artifact_png(image: Image.Image, *, prefix: str = "a") -> dict[str, str]:
    artifact_id = f"{prefix}{uuid.uuid4().hex}"
    path = _artifact_dir() / f"{artifact_id}.png"
    image.save(path, format="PNG")
    rel = path.relative_to(OUTPUT_DIR).as_posix()
    return {"artifact_id": artifact_id, "path": rel, "url": f"/outputs/{rel}"}


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
        artifact_id = _validate_artifact_id(str(value["artifact_id"]))
        path = (_artifact_dir() / f"{artifact_id}.png").resolve()
        with Image.open(path) as img:
            return img.copy()
    if isinstance(value, str) and value.startswith("@artifact:"):
        artifact_id = _validate_artifact_id(value.removeprefix("@artifact:"))
        path = (_artifact_dir() / f"{artifact_id}.png").resolve()
        with Image.open(path) as img:
            return img.copy()
    if isinstance(value, str) and value.startswith("/outputs/"):
        return _load_image_from_outputs_url(value)
    raise ValueError("Unsupported image reference.")


def _resolve_refs(value: Any, task_results: dict[str, dict[str, Any]]) -> Any:
    if isinstance(value, str) and value.startswith("@"):
        token = value[1:]
        if token.startswith("artifact:"):
            return {"artifact_id": token.removeprefix("artifact:").strip()}
        if "." in token:
            task_id, key = token.split(".", 1)
            if task_id not in task_results:
                raise KeyError(f"Unknown task id: {task_id}")
            return task_results[task_id].get(key)
        raise ValueError(f"Invalid reference: {value}")

    if isinstance(value, list):
        return [_resolve_refs(item, task_results) for item in value]
    if isinstance(value, dict):
        return {k: _resolve_refs(v, task_results) for k, v in value.items()}
    return value


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
