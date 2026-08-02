"""SDXL text-to-image workflow task adapter."""

from backend.workflow.sdxl_shared import *

def run_sdxl_text2img_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    _open_image_ref = deps["open_image_ref"]
    generate_text2img = deps["generate_text2img"]
    ip_adapter_settings = _normalized_ip_adapter_settings(
        inputs,
        _open_image_ref,
        allow_image_embeds=True,
    )

    pipeline_params: dict[str, Any] = {
        "prompt": str(inputs.get("prompt") or ""),
        "negative_prompt": str(inputs.get("negative_prompt") or ""),
        "steps": int(inputs.get("steps") or 20),
        "guidance_scale": float(inputs.get("guidance_scale") or inputs.get("cfg") or 7.5),
        "width": int(inputs.get("width") or 1024),
        "height": int(inputs.get("height") or 1024),
        "seed": inputs.get("seed"),
        "model": inputs.get("model"),
        "num_images": int(inputs.get("num_images") or 1),
        "clip_skip": int(inputs.get("clip_skip") or 1),
        "scheduler": str(inputs.get("scheduler") or "euler"),
        "lora_adapters": inputs.get("lora_adapters"),
    }
    if ip_adapter_settings is not None:
        pipeline_params.update(ip_adapter_settings)

    result = generate_text2img(pipeline_params)
    if not isinstance(result, dict):
        raise ValueError("sdxl.text2img must return an object")
    return result

