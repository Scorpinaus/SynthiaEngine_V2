import json
import logging
import re
from pathlib import Path
from typing import Any

from backend.lora_registry import get_lora_entry

logger = logging.getLogger(__name__)
_ADAPTER_NAME_SANITIZE_RE = re.compile(r"[^0-9A-Za-z_-]+")


def _extract_lora_params(
    adapter: Any,
) -> tuple[int | None, float, float | None, float | None, Any, Any]:
    if isinstance(adapter, dict):
        lora_id = adapter.get("lora_id")
        strength = adapter.get("strength", 1.0)
        unet_strength = adapter.get("unet_strength")
        text_encoder_strength = adapter.get("text_encoder_strength")
        unet_scales = adapter.get("unet_scales")
        text_encoder_scales = adapter.get("text_encoder_scales")
    else:
        lora_id = getattr(adapter, "lora_id", None)
        strength = getattr(adapter, "strength", 1.0)
        unet_strength = getattr(adapter, "unet_strength", None)
        text_encoder_strength = getattr(adapter, "text_encoder_strength", None)
        unet_scales = getattr(adapter, "unet_scales", None)
        text_encoder_scales = getattr(adapter, "text_encoder_scales", None)

    return (
        lora_id,
        float(strength),
        _coerce_optional_float(unet_strength),
        _coerce_optional_float(text_encoder_strength),
        unet_scales,
        text_encoder_scales,
    )


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_unet_scales_value(value: Any, path: str) -> float | list[Any] | dict[str, Any]:
    if _is_number(value):
        return float(value)

    if isinstance(value, list):
        return [_validate_unet_scales_value(item, f"{path}[{index}]") for index, item in enumerate(value)]

    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{path} keys must be non-empty strings.")
            normalized[key] = _validate_unet_scales_value(nested, f"{path}.{key}")
        return normalized

    raise ValueError(f"{path} must contain only numbers, lists, and nested objects.")


def _normalize_unet_scales(unet_scales: Any, adapter_index: int) -> float | dict[str, Any] | None:
    if unet_scales is None:
        return None

    field_path = f"LoRA adapter at index {adapter_index} field 'unet_scales'"
    if _is_number(unet_scales):
        return float(unet_scales)
    if isinstance(unet_scales, dict):
        return _validate_unet_scales_value(unet_scales, field_path)

    raise ValueError(f"{field_path} must be a number or an object.")


def _normalize_text_encoder_scales(
    text_encoder_scales: Any,
    adapter_index: int,
) -> dict[str, float] | None:
    if text_encoder_scales is None:
        return None

    field_path = f"LoRA adapter at index {adapter_index} field 'text_encoder_scales'"
    if not isinstance(text_encoder_scales, dict):
        raise ValueError(f"{field_path} must be an object mapping module patterns to numbers.")

    normalized: dict[str, float] = {}
    for key, value in text_encoder_scales.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{field_path} keys must be non-empty strings.")
        if not _is_number(value):
            raise ValueError(f"{field_path}.{key} must be a number.")
        normalized[key] = float(value)

    return normalized


def _build_text_encoder_scales_map(
    pipe,
    *,
    adapter_name: str,
    adapter_index: int,
    default_scale: float,
    overrides: dict[str, float],
) -> dict[str, float]:
    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is None:
        raise ValueError(
            f"LoRA adapter at index {adapter_index} provides text_encoder_scales but this pipeline has no text_encoder."
        )

    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except Exception:
        return dict(overrides)

    module_scales: dict[str, float] = {}
    for module_name, module in text_encoder.named_modules():
        if not isinstance(module, BaseTunerLayer):
            continue
        lora_a = getattr(module, "lora_A", None)
        if isinstance(lora_a, dict) and adapter_name in lora_a:
            module_scales[module_name] = default_scale

    if not module_scales:
        return dict(overrides)

    for pattern, scale in overrides.items():
        for module_name in module_scales:
            if pattern in module_name:
                module_scales[module_name] = scale

    return module_scales


def _build_adapter_weight(
    pipe,
    *,
    adapter_name: str,
    adapter_index: int,
    strength: float,
    unet_strength: float | None,
    text_encoder_strength: float | None,
    unet_scales: Any = None,
    text_encoder_scales: dict[str, float] | None = None,
) -> float | dict[str, Any]:
    if (
        unet_strength is None
        and text_encoder_strength is None
        and unet_scales is None
        and text_encoder_scales is None
    ):
        return float(strength)

    default_unet_scale = float(unet_strength if unet_strength is not None else strength)
    default_text_encoder_scale = float(text_encoder_strength if text_encoder_strength is not None else strength)
    if unet_scales is None:
        unet_weight: float | dict[str, Any] = default_unet_scale
    else:
        unet_weight = unet_scales

    if text_encoder_scales is None:
        text_encoder_weight: float | dict[str, float] = default_text_encoder_scale
    else:
        text_encoder_weight = _build_text_encoder_scales_map(
            pipe,
            adapter_name=adapter_name,
            adapter_index=adapter_index,
            default_scale=default_text_encoder_scale,
            overrides=text_encoder_scales,
        )

    return {
        "unet": unet_weight,
        "text_encoder": text_encoder_weight,
    }


def _sanitize_adapter_fragment(raw_name: str | None) -> str:
    if not raw_name:
        return ""
    sanitized = _ADAPTER_NAME_SANITIZE_RE.sub("_", raw_name).strip("_")
    return re.sub(r"_+", "_", sanitized)


def _build_adapter_name(
    lora_id: int,
    display_name: str | None,
    used_names: set[str],
) -> str:
    fragment = _sanitize_adapter_fragment(display_name) or f"id_{lora_id}"
    base_name = f"lora_{fragment}"
    candidate = base_name
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate

    candidate = f"{base_name}_{lora_id}"
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate

    suffix = 2
    while True:
        candidate = f"{base_name}_{lora_id}_{suffix}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        suffix += 1


def _matches_target(module_name: str, target: str) -> bool:
    if not target:
        return False
    if module_name.endswith(target):
        return True
    return module_name.split(".")[-1] == target


def _summarize_lora_coverage(model, adapter_name: str, label: str) -> dict[str, object]:
    if not hasattr(model, "peft_config") or adapter_name not in model.peft_config:
        return {
            "adapter_present": False,
            "target_modules": None,
            "expected": 0,
            "present": 0,
            "missing": 0,
            "present_names": [],
            "missing_names": [],
        }

    target_modules = model.peft_config[adapter_name].target_modules
    if isinstance(target_modules, str):
        target_list = [target_modules]
    elif isinstance(target_modules, (list, tuple, set)):
        target_list = list(target_modules)
    else:
        target_list = None

    if not target_list:
        return {
            "adapter_present": True,
            "target_modules": target_modules,
            "expected": 0,
            "present": 0,
            "missing": 0,
            "present_names": [],
            "missing_names": [],
        }

    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except Exception:
        return {
            "adapter_present": True,
            "target_modules": target_list,
            "expected": 0,
            "present": 0,
            "missing": 0,
            "present_names": [],
            "missing_names": [],
        }

    expected = []
    present = []
    missing = []

    for name, module in model.named_modules():
        if not any(_matches_target(name, target) for target in target_list):
            continue
        expected.append(name)
        if isinstance(module, BaseTunerLayer) and adapter_name in getattr(module, "lora_A", {}):
            present.append(name)
        else:
            missing.append(name)

    return {
        "adapter_present": True,
        "target_modules": target_list,
        "expected": len(expected),
        "present": len(present),
        "missing": len(missing),
        "present_names": present,
        "missing_names": missing,
    }


def apply_lora_adapters_with_validation(
    pipe,
    lora_adapters: list[object] | None,
    expected_family: str,
    validate: bool = True,
) -> tuple[list[str], dict[str, dict[str, object]]]:
    if not lora_adapters:
        return [], {}

    adapter_names: list[str] = []
    adapter_weights: list[float | dict[str, Any]] = []
    coverage: dict[str, dict[str, object]] = {}
    used_adapter_names: set[str] = set()

    for adapter_index, adapter in enumerate(lora_adapters):
        (
            lora_id,
            strength,
            unet_strength,
            text_encoder_strength,
            unet_scales_raw,
            text_encoder_scales_raw,
        ) = _extract_lora_params(adapter)
        if lora_id is None:
            raise ValueError("LoRA adapter missing lora_id.")

        unet_scales = _normalize_unet_scales(unet_scales_raw, adapter_index)
        text_encoder_scales = _normalize_text_encoder_scales(text_encoder_scales_raw, adapter_index)

        entry = get_lora_entry(int(lora_id))
        if entry.lora_model_family.lower() != expected_family.lower():
            raise ValueError(f"LoRA {entry.name} is not compatible with {expected_family}.")

        adapter_name = _build_adapter_name(entry.lora_id, entry.name, used_adapter_names)
        pipe.load_lora_weights(entry.file_path, adapter_name=adapter_name)
        adapter_names.append(adapter_name)
        adapter_weights.append(
            _build_adapter_weight(
                pipe,
                adapter_name=adapter_name,
                adapter_index=adapter_index,
                strength=strength,
                unet_strength=unet_strength,
                text_encoder_strength=text_encoder_strength,
                unet_scales=unet_scales,
                text_encoder_scales=text_encoder_scales,
            )
        )

        if validate:
            coverage[adapter_name] = {
                "unet": _summarize_lora_coverage(pipe.unet, adapter_name, "unet"),
                "text_encoder": _summarize_lora_coverage(pipe.text_encoder, adapter_name, "text_encoder"),
            }

        logger.info(
            "lora_name: %s , lora_id: %s, lora_weight: %s",
            adapter_name, entry.lora_id, strength,)

    if hasattr(pipe, "set_adapters"):
        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

    if validate:
        for adapter_name, report in coverage.items():
            for label, summary in report.items():
                if not summary.get("adapter_present"):
                    continue
                expected = int(summary.get("expected", 0))
                missing = int(summary.get("missing", 0))
                if expected and missing:
                    missing_names = summary.get("missing_names", [])[:5]
                    logger.warning(
                        "LoRA adapter '%s' missing on %s: %s/%s targets not patched. Example: %s",
                        adapter_name, label, missing, expected, missing_names,)

    return adapter_names, coverage


def write_lora_coverage_report(
    output_dir: Path,
    batch_id: str | None,
    coverage: dict[str, dict[str, object]],
) -> Path | None:
    if not coverage:
        return None
    filename = f"{batch_id}_lora_coverage.json" if batch_id else "lora_coverage.json"
    report_path = output_dir / filename
    report_path.write_text(json.dumps(coverage, indent=2, sort_keys=True), encoding="utf-8")
    return report_path
