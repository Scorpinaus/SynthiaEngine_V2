from __future__ import annotations

from typing import Any


def run_anima_text2img_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    generate_text2img = deps["generate_text2img"]

    result = generate_text2img(dict(inputs))
    if not isinstance(result, dict):
        raise ValueError("anima.text2img must return an object")
    return result
