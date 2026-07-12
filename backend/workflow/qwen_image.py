from __future__ import annotations

from typing import Any

from backend.workflow.image_tasks import ImageTaskDefaults, run_img2img, run_inpaint, run_text2img
from backend.workflow.registry import TaskDefinition, TaskHandler, bind_task
from backend.workflow.schema_input import QwenImageImg2ImgInputs, QwenImageInpaintInputs, QwenImageText2ImgInputs
from backend.workflow.schema_output import ImagesOutput


DEFAULTS = ImageTaskDefaults(steps=30, guidance_scale=7.5, true_cfg_scale=4.0)


def task_definitions(handlers: dict[str, TaskHandler]) -> dict[str, TaskDefinition]:
    contracts = {
        "qwen-image.text2img": QwenImageText2ImgInputs,
        "qwen-image.img2img": QwenImageImg2ImgInputs,
        "qwen-image.inpaint": QwenImageInpaintInputs,
    }
    return {name: bind_task(handlers, name, model, ImagesOutput) for name, model in contracts.items()}


def _lora_adapters(inputs: dict[str, Any], deps: dict[str, Any]) -> Any:
    contract = inputs.get("Lora")
    if isinstance(contract, dict) and "enabled" in contract:
        return contract.get("adapters", []) if contract.get("enabled") else []
    normalizer = deps.get("normalized_lora_adapters")
    if callable(normalizer):
        normalized = normalizer(inputs)
        if normalized is not None:
            return normalized
    if isinstance(contract, dict) and isinstance(contract.get("adapters"), list):
        return contract["adapters"]
    return inputs.get("lora_adapters")


def _text2img_inputs(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    payload = dict(inputs)
    adapters = _lora_adapters(payload, deps)
    if adapters is not None:
        payload["lora_adapters"] = adapters
    return payload


def run_qwen_image_text2img_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    return run_text2img("qwen-image.text2img", inputs, deps, transform=_text2img_inputs)


def run_qwen_image_img2img_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    return run_img2img(
        "qwen-image.img2img",
        inputs,
        deps,
        DEFAULTS,
        lora_adapters=_lora_adapters(inputs, deps),
        remap_strength=True,
    )


def run_qwen_image_inpaint_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    return run_inpaint(
        "qwen-image.inpaint",
        inputs,
        deps,
        DEFAULTS,
        lora_adapters=_lora_adapters(inputs, deps),
    )
