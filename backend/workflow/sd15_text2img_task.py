"""SD1.5 text-to-image workflow task adapter."""

from backend.workflow.sd15_shared import *

def run_sd15_text2img(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    _normalized_hires_settings = deps["normalized_hires_settings"]
    _normalized_lora_adapters = deps["normalized_lora_adapters"]
    _open_image_ref = deps["open_image_ref"]
    make_batch_id = deps["make_batch_id"]
    generate_images = deps["generate_images"]

    hires_enabled, hires_scale = _normalized_hires_settings(inputs)
    lora_adapters = _normalized_lora_adapters(inputs)
    lcm_enabled = _lcm_enabled(inputs)
    steps = int(
        inputs["steps"]
        if "steps" in inputs and inputs.get("steps") is not None
        else (_LCM_DEFAULT_STEPS if lcm_enabled else 20)
    )
    cfg = float(
        inputs["cfg"]
        if "cfg" in inputs and inputs.get("cfg") is not None
        else (_LCM_DEFAULT_CFG if lcm_enabled else 7.5)
    )
    scheduler = "lcm" if lcm_enabled else str(inputs.get("scheduler") or "euler")
    if lcm_enabled:
        _validate_lcm_text2img_settings(steps, cfg)
    ip_adapter_raw = inputs.get("ip_adapter")
    if (
        lcm_enabled
        and isinstance(ip_adapter_raw, dict)
        and bool(ip_adapter_raw.get("enabled", False))
    ):
        raise ValueError("sd15.text2img IP-Adapter cannot be combined with LCM mode.")
    ip_adapter_settings = _normalized_ip_adapter_settings(inputs, _open_image_ref)

    batch_id = str(inputs.get("batch_id") or make_batch_id())
    generation_params = {
        "prompt": str(inputs["prompt"]),
        "negative_prompt": str(inputs.get("negative_prompt") or ""),
        "steps": steps,
        "cfg": cfg,
        "width": int(inputs.get("width") or 512),
        "height": int(inputs.get("height") or 512),
        "seed": inputs.get("seed"),
        "scheduler": scheduler,
        "model": inputs.get("model"),
        "num_images": int(inputs.get("num_images") or 1),
        "clip_skip": int(inputs.get("clip_skip") or 1),
        "lora_adapters": lora_adapters,
        "lcm_enabled": lcm_enabled,
        "lcm_lora_model": _LCM_LORA_MODEL_ID if lcm_enabled else None,
        "hires_enabled": hires_enabled,
        "hires_scale": hires_scale,
        "weighting_policy": str(inputs.get("weighting_policy") or "diffusers-like"),
        "batch_id": batch_id,
    }
    if ip_adapter_settings is not None:
        generation_params.update(ip_adapter_settings)
    filenames = generate_images(generation_params)
    return {"batch_id": batch_id, "images": [f"/outputs/{name}" for name in filenames]}

