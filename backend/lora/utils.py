import json
import logging
import re
from pathlib import Path
from typing import Any, cast

from backend.lora.registry import get_lora_entry

logger = logging.getLogger(__name__)
_ADAPTER_NAME_SANITIZE_RE = re.compile(r"[^0-9A-Za-z_-]+")


def _extract_lora_params(
    adapter: Any,
) -> tuple[int | None, float, float | None, float | None, Any, Any, str]:
    if isinstance(adapter, dict):
        lora_id = adapter.get("lora_id")
        strength = adapter.get("strength", 1.0)
        unet_strength = adapter.get("unet_strength")
        text_encoder_strength = adapter.get("text_encoder_strength")
        unet_scales = adapter.get("unet_scales")
        text_encoder_scales = adapter.get("text_encoder_scales")
        target = adapter.get("target", "both")
    else:
        lora_id = getattr(adapter, "lora_id", None)
        strength = getattr(adapter, "strength", 1.0)
        unet_strength = getattr(adapter, "unet_strength", None)
        text_encoder_strength = getattr(adapter, "text_encoder_strength", None)
        unet_scales = getattr(adapter, "unet_scales", None)
        text_encoder_scales = getattr(adapter, "text_encoder_scales", None)
        target = getattr(adapter, "target", "both")

    return (
        lora_id,
        float(strength),
        _coerce_optional_float(unet_strength),
        _coerce_optional_float(text_encoder_strength),
        unet_scales,
        text_encoder_scales,
        _normalize_lora_target(target),
    )


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_lora_target(value: Any) -> str:
    target = str(value or "both").strip().lower().replace("-", "_")
    if target not in {"both", "unet", "text_encoder"}:
        raise ValueError("LoRA adapter field 'target' must be one of: both, unet, text_encoder.")
    return target


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


def _load_lora_into_text_encoder(pipe, file_path: str, adapter_name: str) -> None:
    text_encoder = getattr(pipe, "text_encoder", None)
    if (
        text_encoder is None
        or not hasattr(pipe, "lora_state_dict")
        or not hasattr(pipe, "load_lora_into_text_encoder")
    ):
        raise ValueError(
            "LoRA adapter target 'text_encoder' requires pipeline.lora_state_dict "
            "and pipeline.load_lora_into_text_encoder support."
        )

    low_cpu_mem_usage = getattr(pipe, "_lora_low_cpu_mem_usage", False)
    state_dict, network_alphas, metadata = pipe.lora_state_dict(
        file_path,
        return_lora_metadata=True,
    )
    pipe.load_lora_into_text_encoder(
        state_dict,
        network_alphas=network_alphas,
        text_encoder=text_encoder,
        lora_scale=getattr(pipe, "lora_scale", 1.0),
        adapter_name=adapter_name,
        _pipeline=pipe,
        metadata=metadata,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )


def _set_text_encoder_adapters(
    text_encoder,
    adapter_names: list[str],
    adapter_weights: list[float | dict[str, Any]],
) -> None:
    try:
        from diffusers.loaders.lora_base import set_adapters_for_text_encoder
    except Exception as exc:
        raise ValueError(
            "LoRA adapter target 'text_encoder' requires Diffusers text encoder adapter support."
        ) from exc

    set_adapters_for_text_encoder(
        adapter_names,
        text_encoder=text_encoder,
        text_encoder_weights=adapter_weights,
    )


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
    preloaded_adapters: list[tuple[str, float | dict[str, Any]]] | None = None,
    allowed_lora_types: tuple[str, ...] | None = None,
    allowed_targets: tuple[str, ...] | None = None,
    coverage_components: tuple[str, ...] = ("unet", "text_encoder"),
) -> tuple[list[str], dict[str, dict[str, object]]]:
    allowed_type_values = (
        None
        if allowed_lora_types is None
        else {str(value).strip().lower() for value in allowed_lora_types}
    )
    allowed_target_values = (
        None
        if allowed_targets is None
        else {_normalize_lora_target(value) for value in allowed_targets}
    )
    preloaded_adapter_items = list(preloaded_adapters or [])
    adapter_names: list[str] = [name for name, _weight in preloaded_adapter_items]
    adapter_weights: list[float | dict[str, Any]] = [
        weight for _name, weight in preloaded_adapter_items
    ]
    both_names: list[str] = [name for name, _weight in preloaded_adapter_items]
    unet_only_names: list[str] = []
    unet_only_weights: list[float | dict[str, Any]] = []
    text_only_names: list[str] = []
    text_only_weights: list[float | dict[str, Any]] = []
    coverage: dict[str, dict[str, object]] = {}
    used_adapter_names: set[str] = set(adapter_names)

    for adapter_index, adapter in enumerate(lora_adapters or []):
        (
            lora_id,
            strength,
            unet_strength,
            text_encoder_strength,
            unet_scales_raw,
            text_encoder_scales_raw,
            target,
        ) = _extract_lora_params(adapter)
        if lora_id is None:
            raise ValueError("LoRA adapter missing lora_id.")
        if allowed_target_values is not None and target not in allowed_target_values:
            allowed = ", ".join(sorted(allowed_target_values)) or "none"
            raise ValueError(
                f"LoRA adapter target '{target}' is not supported for "
                f"{expected_family}; allowed targets: {allowed}."
            )
        logger.debug(
            "Parsed LoRA adapter[%s]: lora_id=%s strength=%s target=%s",
            adapter_index,
            lora_id,
            strength,
            target,
        )

        unet_scales = _normalize_unet_scales(unet_scales_raw, adapter_index)
        text_encoder_scales = _normalize_text_encoder_scales(text_encoder_scales_raw, adapter_index)

        entry = get_lora_entry(int(lora_id))
        if entry.lora_model_family.lower() != expected_family.lower():
            raise ValueError(f"LoRA {entry.name} is not compatible with {expected_family}.")
        entry_lora_type = str(getattr(entry, "lora_type", "")).strip().lower()
        if allowed_type_values is not None and entry_lora_type not in allowed_type_values:
            allowed = ", ".join(sorted(allowed_type_values)) or "none"
            raise ValueError(
                f"LoRA {entry.name} has unsupported type '{entry_lora_type}' for "
                f"{expected_family}; allowed types: {allowed}."
            )

        adapter_name = _build_adapter_name(entry.lora_id, entry.name, used_adapter_names)
        if target == "both":
            logger.info("Loading LoRA '%s' via pipeline.load_lora_weights", adapter_name)
            pipe.load_lora_weights(entry.file_path, adapter_name=adapter_name)
        elif target == "unet":
            if not hasattr(pipe, "unet") or not hasattr(pipe.unet, "load_lora_adapter"):
                raise ValueError(
                    "LoRA adapter target 'unet' requires pipeline.unet.load_lora_adapter support."
                )
            logger.info("Loading LoRA '%s' via unet.load_lora_adapter", adapter_name)
            pipe.unet.load_lora_adapter(entry.file_path, adapter_name=adapter_name, prefix="unet")
        else:
            logger.info("Loading LoRA '%s' via pipeline.load_lora_into_text_encoder", adapter_name)
            _load_lora_into_text_encoder(pipe, entry.file_path, adapter_name)

        adapter_names.append(adapter_name)
        adapter_weight = _build_adapter_weight(
            pipe,
            adapter_name=adapter_name,
            adapter_index=adapter_index,
            strength=strength,
            unet_strength=unet_strength,
            text_encoder_strength=text_encoder_strength,
            unet_scales=unet_scales,
            text_encoder_scales=text_encoder_scales,
        )
        if target == "both":
            adapter_weights.append(adapter_weight)
            both_names.append(adapter_name)
        elif target == "unet":
            if isinstance(adapter_weight, dict):
                unet_weight = cast(float | dict[str, Any], adapter_weight.get("unet", strength))
            else:
                unet_weight = adapter_weight
            unet_only_names.append(adapter_name)
            unet_only_weights.append(unet_weight)
        else:
            if isinstance(adapter_weight, dict):
                text_weight = cast(float | dict[str, Any], adapter_weight.get("text_encoder", strength))
            else:
                text_weight = adapter_weight
            text_only_names.append(adapter_name)
            text_only_weights.append(text_weight)

        logger.info(
            "lora_name: %s , lora_id: %s, lora_weight: %s, target: %s",
            adapter_name,
            entry.lora_id,
            strength,
            target,
        )

        if validate:
            coverage[adapter_name] = {
                component_name: _summarize_lora_coverage(
                    getattr(pipe, component_name, None),
                    adapter_name,
                    component_name,
                )
                for component_name in coverage_components
            }

    if both_names and hasattr(pipe, "set_adapters"):
        logger.info("Activating %s pipeline-level LoRA adapters: %s", len(both_names), both_names)
        pipe.set_adapters(both_names, adapter_weights=adapter_weights)

    if unet_only_names:
        if not hasattr(pipe, "unet") or not hasattr(pipe.unet, "set_adapters"):
            raise ValueError("LoRA adapter target 'unet' requires pipeline.unet.set_adapters support.")
        active_unet_names = both_names + unet_only_names
        active_unet_weights = [
            cast(float | dict[str, Any], weight.get("unet", 1.0))
            if isinstance(weight, dict)
            else weight
            for weight in adapter_weights
        ] + unet_only_weights
        logger.info(
            "Activating %s UNet LoRA adapters: %s",
            len(active_unet_names),
            active_unet_names,
        )
        pipe.unet.set_adapters(active_unet_names, weights=active_unet_weights)

    if text_only_names:
        text_encoder = getattr(pipe, "text_encoder", None)
        if text_encoder is None:
            raise ValueError(
                "LoRA adapter target 'text_encoder' requires pipeline.text_encoder support."
            )
        active_text_names = both_names + text_only_names
        active_text_weights = [
            cast(float | dict[str, Any], weight.get("text_encoder", 1.0))
            if isinstance(weight, dict)
            else weight
            for weight in adapter_weights
        ] + text_only_weights
        logger.info(
            "Activating %s text encoder LoRA adapters: %s",
            len(active_text_names),
            active_text_names,
        )
        _set_text_encoder_adapters(text_encoder, active_text_names, active_text_weights)

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
