from __future__ import annotations

from typing import Any

from backend.workflow.registry import TaskDefinition, TaskHandler, bind_task
from backend.workflow.schema_input import QwenImageImg2ImgInputs, QwenImageInpaintInputs, QwenImageText2ImgInputs
from backend.workflow.schema_output import ImagesOutput


def task_definitions(handlers: dict[str, TaskHandler]) -> dict[str, TaskDefinition]:
    contracts = {
        "qwen-image.text2img": QwenImageText2ImgInputs,
        "qwen-image.img2img": QwenImageImg2ImgInputs,
        "qwen-image.inpaint": QwenImageInpaintInputs,
    }
    return {name: bind_task(handlers, name, model, ImagesOutput) for name, model in contracts.items()}

from PIL import Image


def _normalize_qwen_lora_adapters(inputs: dict[str, Any], deps: dict[str, Any]) -> Any:
    lora_contract = inputs.get("Lora")
    if isinstance(lora_contract, dict) and "enabled" in lora_contract:
        if not bool(lora_contract.get("enabled")):
            return []
        adapters = lora_contract.get("adapters")
        if isinstance(adapters, list):
            return adapters
        return []

    normalized_lora_adapters = deps.get("normalized_lora_adapters")
    if callable(normalized_lora_adapters):
        normalized = normalized_lora_adapters(inputs)
        if normalized is not None:
            return normalized

    if isinstance(lora_contract, dict):
        adapters = lora_contract.get("adapters")
        if isinstance(adapters, list):
            return adapters

    return inputs.get("lora_adapters")


def _with_qwen_lora_inputs(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    payload = dict(inputs)
    normalized_lora_adapters = _normalize_qwen_lora_adapters(payload, deps)
    if normalized_lora_adapters is not None:
        payload["lora_adapters"] = normalized_lora_adapters
    return payload


def run_qwen_image_text2img_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    generate_text2img = deps["generate_text2img"]

    payload = _with_qwen_lora_inputs(inputs, deps)
    result = generate_text2img(payload)
    if not isinstance(result, dict):
        raise ValueError("qwen-image.text2img must return an object")
    return result


def run_qwen_image_img2img_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    _open_image_ref = deps["open_image_ref"]
    _remap_img2img_strength = deps["remap_img2img_strength"]
    generate_img2img = deps["generate_img2img"]
    normalized_lora_adapters = _normalize_qwen_lora_adapters(inputs, deps)

    initial_image = _open_image_ref(inputs["initial_image"]).convert("RGB")
    width = int(inputs.get("width") or 1024)
    height = int(inputs.get("height") or 1024)
    initial_image = initial_image.resize((width, height))

    strength = float(inputs.get("strength") or 0.75)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    strength = _remap_img2img_strength(strength)

    result = generate_img2img(
        {
            "initial_image": initial_image,
            "strength": strength,
            "prompt": str(inputs["prompt"]),
            "negative_prompt": str(inputs.get("negative_prompt") or ""),
            "steps": int(inputs.get("steps") or 30),
            "true_cfg_scale": float(inputs.get("true_cfg_scale") or 4.0),
            "guidance_scale": float(inputs.get("guidance_scale") or 7.5),
            "width": width,
            "height": height,
            "seed": inputs.get("seed"),
            "scheduler": str(inputs.get("scheduler") or "euler"),
            "model": inputs.get("model"),
            "num_images": int(inputs.get("num_images") or 1),
            "lora_adapters": normalized_lora_adapters,
        }
    )
    if not isinstance(result, dict):
        raise ValueError("qwen-image.img2img must return an object")
    return result


def run_qwen_image_inpaint_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    _open_image_ref = deps["open_image_ref"]
    generate_inpaint = deps["generate_inpaint"]
    normalized_lora_adapters = _normalize_qwen_lora_adapters(inputs, deps)

    initial_image = _open_image_ref(inputs["initial_image"]).convert("RGB")
    mask_image = _open_image_ref(inputs["mask_image"]).convert("L")
    if mask_image.size != initial_image.size:
        mask_image = mask_image.resize(initial_image.size, resample=Image.NEAREST)

    strength = float(inputs.get("strength") or 0.5)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")

    result = generate_inpaint(
        {
            "initial_image": initial_image,
            "mask_image": mask_image,
            "strength": strength,
            "prompt": str(inputs["prompt"]),
            "negative_prompt": str(inputs.get("negative_prompt") or ""),
            "steps": int(inputs.get("steps") or 30),
            "true_cfg_scale": float(inputs.get("true_cfg_scale") or 4.0),
            "guidance_scale": float(inputs.get("guidance_scale") or 7.5),
            "seed": inputs.get("seed"),
            "scheduler": str(inputs.get("scheduler") or "euler"),
            "model": inputs.get("model"),
            "num_images": int(inputs.get("num_images") or 1),
            "lora_adapters": normalized_lora_adapters,
        }
    )
    if not isinstance(result, dict):
        raise ValueError("qwen-image.inpaint must return an object")
    return result
