from __future__ import annotations

from typing import Any

from backend.workflow.registry import TaskDefinition, TaskHandler, bind_task
from backend.workflow.schema_input import (
    SdxlControlNetText2ImgInputs,
    SdxlImg2ImgInputs,
    SdxlInpaintInputs,
    SdxlIpAdapterEncodeInputs,
    SdxlText2ImgInputs,
)
from backend.workflow.schema_output import (
    ImagesOutput,
    SdxlControlNetText2ImgOutput,
    SdxlImg2ImgOutput,
    SdxlInpaintOutput,
    SdxlIpAdapterEncodeOutput,
)


def task_definitions(handlers: dict[str, TaskHandler]) -> dict[str, TaskDefinition]:
    contracts = {
        "sdxl.ip_adapter.encode": (SdxlIpAdapterEncodeInputs, SdxlIpAdapterEncodeOutput),
        "sdxl.text2img": (SdxlText2ImgInputs, ImagesOutput),
        "sdxl.controlnet.text2img": (SdxlControlNetText2ImgInputs, SdxlControlNetText2ImgOutput),
        "sdxl.img2img": (SdxlImg2ImgInputs, SdxlImg2ImgOutput),
        "sdxl.inpaint": (SdxlInpaintInputs, SdxlInpaintOutput),
    }
    return {name: bind_task(handlers, name, *models) for name, models in contracts.items()}

from PIL import Image

_DEFAULT_IP_ADAPTER_MODEL = "h94/IP-Adapter"
_DEFAULT_IP_ADAPTER_SUBFOLDER = "sdxl_models"
_DEFAULT_IP_ADAPTER_WEIGHT_NAME = "ip-adapter_sdxl.bin"
_DEFAULT_IP_ADAPTER_SCALE = 0.6


def _normalized_ip_adapter_settings(
    inputs: dict[str, Any],
    open_image_ref,
    *,
    allow_image_embeds: bool = False,
) -> dict[str, Any] | None:
    ip_adapter = inputs.get("ip_adapter")
    if ip_adapter is None:
        return None
    if not isinstance(ip_adapter, dict):
        raise ValueError("`ip_adapter` must be an object.")
    if not bool(ip_adapter.get("enabled", False)):
        return None

    image_ref = ip_adapter.get("image")
    image_embeds_ref = ip_adapter.get("image_embeds")
    if image_embeds_ref is not None and not allow_image_embeds:
        raise ValueError("ip_adapter.image_embeds is only supported for sdxl.text2img.")
    if image_ref is None and image_embeds_ref is None:
        if allow_image_embeds:
            raise ValueError(
                "ip_adapter.image or ip_adapter.image_embeds is required when IP-Adapter is enabled."
            )
        raise ValueError("ip_adapter.image is required when IP-Adapter is enabled.")
    if image_ref is not None and image_embeds_ref is not None:
        raise ValueError("Provide either ip_adapter.image or ip_adapter.image_embeds, not both.")

    scale_raw = ip_adapter.get("scale", _DEFAULT_IP_ADAPTER_SCALE)
    scale = _DEFAULT_IP_ADAPTER_SCALE if scale_raw is None else float(scale_raw)
    if scale < 0.0 or scale > 1.0:
        raise ValueError("ip_adapter.scale must be within [0, 1].")

    settings: dict[str, Any] = {
        "ip_adapter_scale": scale,
        "ip_adapter_model": str(ip_adapter.get("model") or _DEFAULT_IP_ADAPTER_MODEL),
        "ip_adapter_subfolder": str(
            ip_adapter.get("subfolder") or _DEFAULT_IP_ADAPTER_SUBFOLDER
        ),
        "ip_adapter_weight_name": str(
            ip_adapter.get("weight_name") or _DEFAULT_IP_ADAPTER_WEIGHT_NAME
        ),
    }
    if image_ref is not None:
        settings["ip_adapter_image"] = open_image_ref(image_ref).convert("RGB")
    else:
        settings["ip_adapter_image_embeds_ref"] = image_embeds_ref
    return settings

__all__ = [name for name in globals() if not name.startswith("__")]

