from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType

from diffusers import FlowMatchEulerDiscreteScheduler

from backend.lora.registry import LoraRegistryEntry, LoraRuntimeProfile, get_lora_entry
from backend.registries.model import ModelRegistryEntry
from backend.utilities.schedulers import create_scheduler


_QWEN_IMAGE_TASKS = {
    "text2img": "text2img",
    "qwen-image.text2img": "text2img",
    "img2img": "img2img",
    "qwen-image.img2img": "img2img",
    "inpaint": "inpaint",
    "qwen-image.inpaint": "inpaint",
}

_QWEN_IMAGE_LIGHTNING_SCHEDULER_PROFILE = "qwen_image_lightning_shift3"
_QWEN_IMAGE_LIGHTNING_SCHEDULER_CONFIG = MappingProxyType(
    {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
)


@dataclass(frozen=True)
class ResolvedQwenImageLoraAdapter:
    lora_id: int
    strength: float
    target: str
    entry: LoraRegistryEntry


@dataclass(frozen=True)
class QwenImageLightningResolution:
    adapters: tuple[ResolvedQwenImageLoraAdapter, ...]
    model_entry: ModelRegistryEntry
    model_variant: str
    task: str
    steps: int
    true_cfg_scale: float
    lightning_profile: LoraRuntimeProfile | None


def get_qwen_image_lightning_scheduler_config() -> dict[str, object]:
    """Return a fresh fixed scheduler config for Qwen Image Lightning."""
    return dict(_QWEN_IMAGE_LIGHTNING_SCHEDULER_CONFIG)


def create_qwen_image_lightning_scheduler(
    profile: LoraRuntimeProfile,
) -> FlowMatchEulerDiscreteScheduler:
    """Construct the fixed shift-3 scheduler for a Lightning profile."""
    if profile.scheduler_profile != _QWEN_IMAGE_LIGHTNING_SCHEDULER_PROFILE:
        raise ValueError(
            "Qwen Image Lightning requires scheduler_profile "
            f"'{_QWEN_IMAGE_LIGHTNING_SCHEDULER_PROFILE}'."
        )
    return FlowMatchEulerDiscreteScheduler.from_config(
        get_qwen_image_lightning_scheduler_config()
    )


def select_qwen_image_scheduler(
    resolution: QwenImageLightningResolution,
    pipe: object,
) -> object:
    """Select a base or Lightning scheduler without mutating the pipeline."""
    if resolution.lightning_profile is None:
        return create_scheduler("flowmatch_euler", pipe)
    return create_qwen_image_lightning_scheduler(resolution.lightning_profile)


def get_qwen_image_base_variant(entry: ModelRegistryEntry) -> str:
    """Return the Qwen Image base variant required by acceleration profiles."""
    if entry.family.strip().lower() != "qwen-image":
        raise ValueError(f"Model '{entry.name}' is not a Qwen-Image model.")

    identity = " ".join((entry.name, entry.link)).lower()
    if "edit" in identity:
        return "qwen-image-edit"
    if "qwen-image-2512" in identity:
        return "qwen-image-2512"
    return "qwen-image-other"


def _normalize_task(task: str) -> str:
    normalized = task.strip().lower()
    try:
        return _QWEN_IMAGE_TASKS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported Qwen Image task '{task}'.") from exc


def _normalize_steps(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Qwen Image requested steps must be an integer.")
    return value


def _normalize_true_cfg_scale(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Qwen Image requested true_cfg_scale must be numeric.")
    return float(value)


def _adapter_mapping(adapter: object, index: int) -> Mapping[str, object]:
    if isinstance(adapter, Mapping):
        return adapter
    model_dump = getattr(adapter, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    raise ValueError(f"Qwen Image LoRA adapter at index {index} must be a mapping.")


def _normalize_adapter(
    adapter: object,
    index: int,
    lookup: Callable[[int], LoraRegistryEntry],
) -> ResolvedQwenImageLoraAdapter:
    raw = _adapter_mapping(adapter, index)
    lora_id = raw.get("lora_id")
    if isinstance(lora_id, bool) or not isinstance(lora_id, int):
        raise ValueError(f"Qwen Image LoRA adapter at index {index} has invalid lora_id.")

    strength_value = raw.get("strength", 0.8)
    if isinstance(strength_value, bool) or not isinstance(strength_value, (int, float)):
        raise ValueError(f"Qwen Image LoRA adapter {lora_id} strength must be numeric.")
    strength = float(strength_value)
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"Qwen Image LoRA adapter {lora_id} strength must be from 0.0 through 1.0.")

    target = raw.get("target", "both")
    if target != "both":
        raise ValueError(f"Qwen Image LoRA adapter {lora_id} target must be 'both'.")

    try:
        entry = lookup(lora_id)
    except ValueError as exc:
        raise ValueError(f"Qwen Image LoRA adapter {lora_id} could not be resolved: {exc}") from exc
    if entry.lora_model_family.strip().lower() != "qwen-image":
        raise ValueError(f"Qwen Image LoRA adapter {lora_id} has incompatible family '{entry.lora_model_family}'.")
    if entry.lora_type.strip().lower() != "lora":
        raise ValueError(f"Qwen Image LoRA adapter {lora_id} has unsupported type '{entry.lora_type}'.")

    return ResolvedQwenImageLoraAdapter(
        lora_id=lora_id,
        strength=strength,
        target="both",
        entry=entry,
    )


def resolve_qwen_image_lightning_profile(
    selected_lora_adapters: list[object] | None,
    model_entry: ModelRegistryEntry,
    task: str,
    requested_steps: int,
    requested_true_cfg_scale: float,
    *,
    lookup_lora_entry: Callable[[int], LoraRegistryEntry] = get_lora_entry,
) -> QwenImageLightningResolution:
    """Resolve Qwen adapters and validate a selected Lightning profile."""
    if selected_lora_adapters is None:
        adapters_input: list[object] = []
    elif isinstance(selected_lora_adapters, list):
        adapters_input = selected_lora_adapters
    else:
        raise ValueError("Qwen Image lora_adapters must be a list.")

    normalized_task = _normalize_task(task)
    steps = _normalize_steps(requested_steps)
    true_cfg_scale = _normalize_true_cfg_scale(requested_true_cfg_scale)
    model_variant = get_qwen_image_base_variant(model_entry)
    resolved_adapters = tuple(
        _normalize_adapter(adapter, index, lookup_lora_entry)
        for index, adapter in enumerate(adapters_input)
    )
    lightning_adapters = [
        adapter for adapter in resolved_adapters if adapter.entry.runtime_profile is not None
    ]
    standard_adapters = [
        adapter for adapter in resolved_adapters if adapter.entry.runtime_profile is None
    ]

    if len(lightning_adapters) > 1:
        adapter_ids = ", ".join(str(adapter.lora_id) for adapter in lightning_adapters)
        raise ValueError(f"Qwen Image Lightning supports one Lightning adapter; received IDs: {adapter_ids}.")

    lightning_profile = (
        lightning_adapters[0].entry.runtime_profile if lightning_adapters else None
    )
    if lightning_profile is not None:
        lightning_adapter = lightning_adapters[0]
        if len(standard_adapters) > 1:
            adapter_ids = ", ".join(str(adapter.lora_id) for adapter in standard_adapters)
            raise ValueError(
                "Qwen Image Lightning supports at most one standard companion LoRA; "
                f"received IDs: {adapter_ids}."
            )
        if normalized_task not in lightning_profile.supported_tasks:
            raise ValueError(
                f"Qwen Image Lightning adapter {lightning_adapter.lora_id} does not support "
                f"received '{normalized_task}'."
            )
        if model_variant != lightning_profile.base_variant:
            raise ValueError(
                f"Qwen Image Lightning adapter {lightning_adapter.lora_id} requires base variant "
                f"'{lightning_profile.base_variant}'; selected '{model_variant}'."
            )
        if steps != lightning_profile.steps:
            raise ValueError(
                f"Qwen Image Lightning adapter {lightning_adapter.lora_id} requires steps="
                f"{lightning_profile.steps}; received {steps}."
            )
        if true_cfg_scale != lightning_profile.true_cfg_scale:
            raise ValueError(
                f"Qwen Image Lightning adapter {lightning_adapter.lora_id} requires "
                f"true_cfg_scale={lightning_profile.true_cfg_scale}; received {true_cfg_scale}."
            )
        if lightning_adapter.strength != lightning_profile.adapter_strength:
            raise ValueError(
                f"Qwen Image Lightning adapter {lightning_adapter.lora_id} requires strength="
                f"{lightning_profile.adapter_strength}; received {lightning_adapter.strength}."
            )
        if standard_adapters:
            companion_adapter = standard_adapters[0]
            compatibility = companion_adapter.entry.compatibility
            if compatibility is None:
                raise ValueError(
                    f"Qwen Image Lightning companion LoRA {companion_adapter.lora_id} "
                    "has no declared compatibility metadata."
                )
            if model_variant not in compatibility.base_variants:
                raise ValueError(
                    f"Qwen Image Lightning companion LoRA {companion_adapter.lora_id} "
                    f"compatibility is missing base variant '{model_variant}'."
                )
            if lightning_profile.kind not in compatibility.runtime_profile_kinds:
                raise ValueError(
                    f"Qwen Image Lightning companion LoRA {companion_adapter.lora_id} "
                    f"compatibility is missing runtime profile kind '{lightning_profile.kind}'."
                )
            if normalized_task not in compatibility.supported_tasks:
                raise ValueError(
                    f"Qwen Image Lightning companion LoRA {companion_adapter.lora_id} "
                    f"compatibility is missing task '{normalized_task}'."
                )

    return QwenImageLightningResolution(
        adapters=resolved_adapters,
        model_entry=model_entry,
        model_variant=model_variant,
        task=normalized_task,
        steps=steps,
        true_cfg_scale=true_cfg_scale,
        lightning_profile=lightning_profile,
    )
