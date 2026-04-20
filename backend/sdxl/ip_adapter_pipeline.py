from __future__ import annotations

import logging
from typing import Any

import torch
from diffusers.models.modeling_utils import load_state_dict
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from backend.adapters.ip_adapter_embeds import save_ip_adapter_embeds_artifact
from backend.utilities.pipeline import cleanup_memory

logger = logging.getLogger(__name__)

_DEFAULT_IP_ADAPTER_MODEL = "h94/IP-Adapter"
_DEFAULT_IP_ADAPTER_SUBFOLDER = "sdxl_models"
_DEFAULT_IP_ADAPTER_WEIGHT_NAME = "ip-adapter_sdxl.bin"
_DEFAULT_IP_ADAPTER_SCALE = 0.6
_UNSUPPORTED_MINIMAL_ENCODER_ERROR = (
    "Only the base SDXL IP-Adapter is supported by the minimal encoder."
)


def _require_default_base_adapter(
    *,
    model: str,
    subfolder: str,
    weight_name: str,
) -> None:
    if (
        model != _DEFAULT_IP_ADAPTER_MODEL
        or subfolder != _DEFAULT_IP_ADAPTER_SUBFOLDER
        or weight_name != _DEFAULT_IP_ADAPTER_WEIGHT_NAME
    ):
        raise ValueError(
            "Only the default SDXL base IP-Adapter is supported by the minimal encoder."
        )


def _load_ip_adapter_state_dict(
    *,
    model: str,
    subfolder: str,
    weight_name: str,
) -> dict[str, Any]:
    model_file = hf_hub_download(
        repo_id=model,
        filename=weight_name,
        subfolder=subfolder,
    )
    state_dict = load_state_dict(model_file)
    if not isinstance(state_dict, dict):
        raise ValueError("IP-Adapter weights must contain a state dict.")
    return state_dict


def _validate_base_ip_adapter_state_dict(state_dict: dict[str, Any]) -> None:
    image_proj = state_dict.get("image_proj")
    if not isinstance(image_proj, dict) or "proj.weight" not in image_proj:
        raise ValueError(_UNSUPPORTED_MINIMAL_ENCODER_ERROR)


@torch.inference_mode()
def generate_ip_adapter_image_embeds(params: dict[str, Any]) -> dict[str, Any]:
    image = params["image"]
    if not isinstance(image, Image.Image):
        raise ValueError("image must be a PIL image.")

    base_model = params.get("model")
    guidance_scale = float(params.get("guidance_scale", 7.5))
    ip_adapter_model = str(params.get("ip_adapter_model") or _DEFAULT_IP_ADAPTER_MODEL)
    ip_adapter_subfolder = str(
        params.get("ip_adapter_subfolder") or _DEFAULT_IP_ADAPTER_SUBFOLDER
    )
    ip_adapter_weight_name = str(
        params.get("ip_adapter_weight_name") or _DEFAULT_IP_ADAPTER_WEIGHT_NAME
    )
    ip_adapter_scale_raw = params.get("ip_adapter_scale")
    ip_adapter_scale = (
        _DEFAULT_IP_ADAPTER_SCALE
        if ip_adapter_scale_raw is None
        else float(ip_adapter_scale_raw)
    )
    do_classifier_free_guidance = guidance_scale > 1.0

    _require_default_base_adapter(
        model=ip_adapter_model,
        subfolder=ip_adapter_subfolder,
        weight_name=ip_adapter_weight_name,
    )
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required for SDXL IP-Adapter minimal encode.")

    image_encoder = None
    image_processor = None
    pixel_values = None
    image_embeds = None
    negative_image_embeds = None
    single_image_embeds = None
    try:
        logger.info(
            "SDXL Custom IP-Adapter encode: base_model=%s, adapter_model=%s, subfolder=%s, weight_name=%s, cfg=%s",
            base_model,
            ip_adapter_model,
            ip_adapter_subfolder,
            ip_adapter_weight_name,
            do_classifier_free_guidance,
        )
        state_dict = _load_ip_adapter_state_dict(
            model=ip_adapter_model,
            subfolder=ip_adapter_subfolder,
            weight_name=ip_adapter_weight_name,
        )
        _validate_base_ip_adapter_state_dict(state_dict)

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            ip_adapter_model,
            subfolder=f"{ip_adapter_subfolder}/image_encoder",
            torch_dtype=torch.float16,
        ).to("cuda")
        clip_image_size = int(getattr(image_encoder.config, "image_size", 224))
        image_processor = CLIPImageProcessor(
            size=clip_image_size,
            crop_size=clip_image_size,
        )
        pixel_values = image_processor(
            image.convert("RGB"),
            return_tensors="pt",
        ).pixel_values
        pixel_values = pixel_values.to(device="cuda", dtype=torch.float16)

        image_embeds = image_encoder(pixel_values).image_embeds
        negative_image_embeds = torch.zeros_like(image_embeds)

        single_image_embeds = image_embeds[None, :]
        if do_classifier_free_guidance:
            single_negative_image_embeds = negative_image_embeds[None, :]
            single_image_embeds = torch.cat(
                [single_negative_image_embeds, single_image_embeds],
                dim=0,
            )
        embeds = [single_image_embeds.cpu()]

        artifact = save_ip_adapter_embeds_artifact(
            embeds,
            metadata={
                "base_model": base_model,
                "adapters": [
                    {
                        "model": ip_adapter_model,
                        "subfolder": ip_adapter_subfolder,
                        "weight_name": ip_adapter_weight_name,
                        "scale": ip_adapter_scale,
                    }
                ],
                "do_classifier_free_guidance": do_classifier_free_guidance,
                "num_images_per_prompt": 1,
            },
        )
        return {"image_embeds": artifact}
    finally:
        if image_encoder is not None:
            try:
                image_encoder.to("cpu")
            except Exception:
                logger.debug(
                    "Failed to move SDXL IP-Adapter image encoder to CPU.",
                    exc_info=True,
                )
        image_encoder = None
        image_processor = None
        pixel_values = None
        image_embeds = None
        negative_image_embeds = None
        single_image_embeds = None
        cleanup_memory()
