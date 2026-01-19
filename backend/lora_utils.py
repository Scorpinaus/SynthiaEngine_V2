import json
import logging
from pathlib import Path
from typing import Any

from backend.lora_registry import get_lora_entry

logger = logging.getLogger(__name__)


def _extract_lora_params(adapter: Any) -> tuple[int | None, float, float | None, float | None]:
    if isinstance(adapter, dict):
        lora_id = adapter.get("lora_id")
        strength = adapter.get("strength", 1.0)
        unet_strength = adapter.get("unet_strength")
        text_encoder_strength = adapter.get("text_encoder_strength")
    else:
        lora_id = getattr(adapter, "lora_id", None)
        strength = getattr(adapter, "strength", 1.0)
        unet_strength = getattr(adapter, "unet_strength", None)
        text_encoder_strength = getattr(adapter, "text_encoder_strength", None)

    return lora_id, float(strength), _coerce_optional_float(unet_strength), _coerce_optional_float(text_encoder_strength)


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_adapter_weight(
    strength: float,
    unet_strength: float | None,
    text_encoder_strength: float | None,
) -> float | dict[str, float]:
    if unet_strength is None and text_encoder_strength is None:
        return float(strength)
    return {
        "unet": float(unet_strength if unet_strength is not None else strength),
        "text_encoder": float(text_encoder_strength if text_encoder_strength is not None else strength),
    }


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
    adapter_weights: list[float | dict[str, float]] = []
    coverage: dict[str, dict[str, object]] = {}

    for adapter in lora_adapters:
        lora_id, strength, unet_strength, text_encoder_strength = _extract_lora_params(adapter)
        if lora_id is None:
            raise ValueError("LoRA adapter missing lora_id.")

        entry = get_lora_entry(int(lora_id))
        if entry.lora_model_family.lower() != expected_family.lower():
            raise ValueError(f"LoRA {entry.name} is not compatible with {expected_family}.")

        adapter_name = f"lora_{entry.name}"
        pipe.load_lora_weights(entry.file_path, adapter_name=adapter_name)
        adapter_names.append(adapter_name)
        adapter_weights.append(_build_adapter_weight(strength, unet_strength, text_encoder_strength))

        if validate:
            coverage[adapter_name] = {
                "unet": _summarize_lora_coverage(pipe.unet, adapter_name, "unet"),
                "text_encoder": _summarize_lora_coverage(pipe.text_encoder, adapter_name, "text_encoder"),
            }

        logger.info(
            "lora_name: %s , lora_id: %s, lora_weight: %s",
            adapter_name,
            entry.lora_id,
            strength,
        )

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
                        adapter_name,
                        label,
                        missing,
                        expected,
                        missing_names,
                    )

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
