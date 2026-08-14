"""Shared mechanics for simple text2img, img2img, and inpaint tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from PIL import Image


@dataclass(frozen=True)
class ImageTaskDefaults:
    steps: int
    guidance_scale: float
    true_cfg_scale: float | None = None
    scheduler: str = "euler"


def run_text2img(
    task_type: str,
    inputs: dict[str, Any],
    deps: dict[str, Any],
    *,
    transform: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = transform(inputs, deps) if transform else dict(inputs)
    return _require_result(task_type, deps["generate_text2img"](payload))


def run_img2img(
    task_type: str,
    inputs: dict[str, Any],
    deps: dict[str, Any],
    defaults: ImageTaskDefaults,
    *,
    lora_adapters: Any = None,
    remap_strength: bool = False,
    passthrough_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    image = deps["open_image_ref"](inputs["initial_image"]).convert("RGB")
    width = int(inputs.get("width") or 1024)
    height = int(inputs.get("height") or 1024)
    strength = _strength(inputs, 0.75)
    if remap_strength:
        strength = deps["remap_img2img_strength"](strength)

    payload = _generation_inputs(inputs, defaults)
    payload.update(
        initial_image=image.resize((width, height)),
        strength=strength,
        width=width,
        height=height,
        lora_adapters=inputs.get("lora_adapters") if lora_adapters is None else lora_adapters,
    )
    payload.update({name: inputs.get(name) for name in passthrough_fields})
    return _require_result(task_type, deps["generate_img2img"](payload))


def run_inpaint(
    task_type: str,
    inputs: dict[str, Any],
    deps: dict[str, Any],
    defaults: ImageTaskDefaults,
    *,
    lora_adapters: Any = None,
    passthrough_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    image = deps["open_image_ref"](inputs["initial_image"]).convert("RGB")
    mask = deps["open_image_ref"](inputs["mask_image"]).convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, resample=Image.NEAREST)

    payload = _generation_inputs(inputs, defaults)
    payload.update(
        initial_image=image,
        mask_image=mask,
        strength=_strength(inputs, 0.5),
        lora_adapters=inputs.get("lora_adapters") if lora_adapters is None else lora_adapters,
    )
    payload.update({name: inputs.get(name) for name in passthrough_fields})
    return _require_result(task_type, deps["generate_inpaint"](payload))


def _generation_inputs(inputs: dict[str, Any], defaults: ImageTaskDefaults) -> dict[str, Any]:
    payload = {
        "prompt": str(inputs["prompt"]),
        "negative_prompt": str(inputs.get("negative_prompt") or ""),
        "steps": int(inputs.get("steps") or defaults.steps),
        "guidance_scale": float(inputs.get("guidance_scale") or defaults.guidance_scale),
        "seed": inputs.get("seed"),
        "scheduler": str(inputs.get("scheduler") or defaults.scheduler),
        "model": inputs.get("model"),
        "num_images": int(inputs.get("num_images") or 1),
    }
    if defaults.true_cfg_scale is not None:
        payload["true_cfg_scale"] = float(inputs.get("true_cfg_scale") or defaults.true_cfg_scale)
    return payload


def _strength(inputs: dict[str, Any], fallback: float) -> float:
    strength = float(inputs.get("strength") or fallback)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    return strength


def _require_result(task_type: str, result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError(f"{task_type} must return an object")
    return result
