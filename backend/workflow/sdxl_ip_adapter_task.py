"""SDXL IP-Adapter workflow task adapter."""

from backend.workflow.sdxl_shared import *

def run_sdxl_ip_adapter_encode_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    _open_image_ref = deps["open_image_ref"]
    generate_ip_adapter_image_embeds = deps["generate_ip_adapter_image_embeds"]
    image = _open_image_ref(inputs["image"]).convert("RGB")
    result = generate_ip_adapter_image_embeds(
        {
            "image": image,
            "model": inputs.get("model"),
            "guidance_scale": float(inputs.get("guidance_scale") or 7.5),
            "ip_adapter_model": str(inputs.get("ip_adapter_model") or _DEFAULT_IP_ADAPTER_MODEL),
            "ip_adapter_subfolder": str(
                inputs.get("ip_adapter_subfolder") or _DEFAULT_IP_ADAPTER_SUBFOLDER
            ),
            "ip_adapter_weight_name": str(
                inputs.get("ip_adapter_weight_name") or _DEFAULT_IP_ADAPTER_WEIGHT_NAME
            ),
            "ip_adapter_scale": float(inputs.get("ip_adapter_scale") or _DEFAULT_IP_ADAPTER_SCALE),
        }
    )
    if not isinstance(result, dict):
        raise ValueError("sdxl.ip_adapter.encode must return an object")
    return result

