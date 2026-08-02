from __future__ import annotations

from typing import Any

from backend.workflow.registry import TaskDefinition, TaskHandler, bind_task
from backend.workflow.schema_input import (
    Sd15AnimateDiffText2VideoInputs,
    Sd15ControlNetText2ImgInputs,
    Sd15HiresFixInputs,
    Sd15IpAdapterEncodeInputs,
    Sd15Img2ImgInputs,
    Sd15InpaintInputs,
    Sd15Text2ImgInputs,
)
from backend.workflow.schema_output import (
    ImagesWithBatchOutput,
    Sd15ControlNetText2ImgOutput,
    Sd15Img2ImgOutput,
    Sd15InpaintOutput,
    Sd15IpAdapterEncodeOutput,
    VideosWithBatchOutput,
)


def task_definitions(handlers: dict[str, TaskHandler]) -> dict[str, TaskDefinition]:
    contracts = {
        "sd15.text2img": (Sd15Text2ImgInputs, ImagesWithBatchOutput),
        "sd15.animatediff.text2video": (Sd15AnimateDiffText2VideoInputs, VideosWithBatchOutput),
        "sd15.img2img": (Sd15Img2ImgInputs, Sd15Img2ImgOutput),
        "sd15.inpaint": (Sd15InpaintInputs, Sd15InpaintOutput),
        "sd15.controlnet.text2img": (Sd15ControlNetText2ImgInputs, Sd15ControlNetText2ImgOutput),
        "sd15.hires_fix": (Sd15HiresFixInputs, ImagesWithBatchOutput),
        "sd15.ip_adapter.encode": (Sd15IpAdapterEncodeInputs, Sd15IpAdapterEncodeOutput),
    }
    return {name: bind_task(handlers, name, *models) for name, models in contracts.items()}

from PIL import Image

_LCM_LORA_MODEL_ID = "latent-consistency/lcm-lora-sdv1-5"
_LCM_DEFAULT_STEPS = 4
_LCM_DEFAULT_CFG = 0.0
_SD15_INPAINT_CONTROLNET_MODEL_IDS = {
    "lllyasviel/control_v11p_sd15_inpaint",
}
_SD15_INPAINT_CONDITION_PREPROCESSOR_ID = "inpaint-condition"
_LCM_MIN_STEPS = 1
_LCM_MAX_STEPS = 8
_LCM_MIN_CFG = 0.0
_LCM_MAX_CFG = 2.0
_DEFAULT_IP_ADAPTER_MODEL = "h94/IP-Adapter"
_DEFAULT_IP_ADAPTER_SUBFOLDER = "models"
_DEFAULT_IP_ADAPTER_WEIGHT_NAME = "ip-adapter_sd15.bin"
_DEFAULT_IP_ADAPTER_SCALE = 0.6


def _lcm_enabled(inputs: dict[str, Any]) -> bool:
    lcm_contract = inputs.get("lcm")
    if isinstance(lcm_contract, dict) and bool(lcm_contract.get("enabled", False)):
        return True
    return str(inputs.get("scheduler") or "").lower() == "lcm"


def _validate_lcm_settings(task_type: str, steps: int, cfg: float) -> None:
    if steps < _LCM_MIN_STEPS or steps > _LCM_MAX_STEPS:
        raise ValueError(
            f"{task_type} LCM mode requires steps within [{_LCM_MIN_STEPS}, {_LCM_MAX_STEPS}]."
        )
    if cfg < _LCM_MIN_CFG or cfg > _LCM_MAX_CFG:
        raise ValueError(
            f"{task_type} LCM mode requires cfg within [{_LCM_MIN_CFG:g}, {_LCM_MAX_CFG:g}]."
        )


def _validate_lcm_text2img_settings(steps: int, cfg: float) -> None:
    _validate_lcm_settings("sd15.text2img", steps, cfg)


def _validate_lcm_img2img_settings(steps: int, cfg: float) -> None:
    _validate_lcm_settings("sd15.img2img", steps, cfg)


def _validate_lcm_inpaint_settings(steps: int, cfg: float) -> None:
    _validate_lcm_settings("sd15.inpaint", steps, cfg)


def _resolve_control_guidance_timings(
    inputs: dict[str, Any],
    *,
    controlnet_count: int,
) -> tuple[list[float], list[float]]:
    starts_raw = inputs.get("control_guidance_starts")
    if starts_raw is not None and not isinstance(starts_raw, list):
        raise ValueError("control_guidance_starts must be a list of numbers")
    if starts_raw is not None:
        control_guidance_starts = [float(item) for item in starts_raw]
        if len(control_guidance_starts) != controlnet_count:
            raise ValueError(
                "control_guidance_starts length must match controlnet_models length."
            )
    else:
        control_guidance_starts = [
            float(inputs.get("control_guidance_start", 0.0))
        ] * controlnet_count

    ends_raw = inputs.get("control_guidance_ends")
    if ends_raw is not None and not isinstance(ends_raw, list):
        raise ValueError("control_guidance_ends must be a list of numbers")
    if ends_raw is not None:
        control_guidance_ends = [float(item) for item in ends_raw]
        if len(control_guidance_ends) != controlnet_count:
            raise ValueError(
                "control_guidance_ends length must match controlnet_models length."
            )
    else:
        control_guidance_ends = [
            float(inputs.get("control_guidance_end", 1.0))
        ] * controlnet_count

    using_scalar_guidance = starts_raw is None and ends_raw is None
    for idx, (start, end) in enumerate(zip(control_guidance_starts, control_guidance_ends)):
        if start < 0.0 or start > 1.0:
            raise ValueError(f"control_guidance_starts[{idx}] must be within [0, 1].")
        if end < 0.0 or end > 1.0:
            raise ValueError(f"control_guidance_ends[{idx}] must be within [0, 1].")
        if start > end:
            if using_scalar_guidance:
                raise ValueError("control_guidance_start must be <= control_guidance_end")
            raise ValueError(
                f"control_guidance_starts[{idx}] must be <= control_guidance_ends[{idx}]."
            )

    return control_guidance_starts, control_guidance_ends


def _normalized_ip_adapter_settings(
    inputs: dict[str, Any],
    open_image_ref,
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
    if image_ref is None and image_embeds_ref is None:
        raise ValueError(
            "ip_adapter.image or ip_adapter.image_embeds is required when IP-Adapter is enabled."
        )
    if image_ref is not None and image_embeds_ref is not None:
        raise ValueError("Provide either ip_adapter.image or ip_adapter.image_embeds, not both.")

    mask_ref = ip_adapter.get("mask_image")
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
    if mask_ref is not None:
        settings["ip_adapter_mask_image"] = open_image_ref(mask_ref).convert("L")
    return settings

__all__ = [name for name in globals() if not name.startswith("__")]

