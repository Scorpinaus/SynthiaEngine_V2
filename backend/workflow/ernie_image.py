from __future__ import annotations

from typing import Any


def run_ernie_image_text2img_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    generate_text2img = deps["generate_text2img"]

    payload = dict(inputs)
    normalized_lora_adapters = deps.get("normalized_lora_adapters")
    if callable(normalized_lora_adapters):
        normalized = normalized_lora_adapters(inputs)
        if normalized is not None:
            payload["lora_adapters"] = normalized

    result = generate_text2img(payload)
    if not isinstance(result, dict):
        raise ValueError("ernie-image.text2img must return an object")
    return result
