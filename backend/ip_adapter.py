from __future__ import annotations

import logging
from typing import Any

import torch
from PIL import Image

from backend.pipeline_utils import cleanup_memory

_UNSET = object()

logger = logging.getLogger(__name__)


class IpAdapterManager:
    """Shared lifecycle helpers for Diffusers IP-Adapter usage."""

    @staticmethod
    def pipe_device(pipe: Any) -> torch.device | str:
        return getattr(pipe, "_execution_device", None) or getattr(pipe, "device", "cuda")

    @staticmethod
    def load(
        pipe: Any,
        *,
        model: str,
        subfolder: str,
        weight_name: str,
        scale: float,
        family: str,
        image_encoder_folder: str | None | object = _UNSET,
    ) -> None:
        if not hasattr(pipe, "load_ip_adapter"):
            raise RuntimeError(
                "The installed Diffusers pipeline does not support IP-Adapter loading. "
                "Install a Diffusers version with load_ip_adapter support."
            )

        logger.info(
            "Loading %s IP-Adapter: model=%s, subfolder=%s, weight_name=%s ,scale=%s",
            family,
            model,
            subfolder,
            weight_name,
            scale,
        )
        load_kwargs: dict[str, object] = {
            "subfolder": subfolder,
            "weight_name": weight_name,
        }
        if image_encoder_folder is not _UNSET:
            load_kwargs["image_encoder_folder"] = image_encoder_folder

        pipe.load_ip_adapter(model, **load_kwargs)
        if hasattr(pipe, "set_ip_adapter_scale"):
            pipe.set_ip_adapter_scale(scale)
        pipe.to("cuda")

    @staticmethod
    def cleanup(pipe: Any, enabled: bool) -> None:
        if not enabled or not hasattr(pipe, "unload_ip_adapter"):
            return
        try:
            pipe.unload_ip_adapter()
        except Exception:
            logger.exception("Failed to unload IP-Adapter weights cleanly.")

    @classmethod
    def prepare_image_embeds(
        cls,
        pipe: Any,
        image: Image.Image,
        *,
        do_classifier_free_guidance: bool,
        num_images_per_prompt: int = 1,
        offload_image_encoder: bool = True,
    ) -> list[torch.Tensor]:
        if not hasattr(pipe, "prepare_ip_adapter_image_embeds"):
            raise RuntimeError(
                "The installed Diffusers pipeline does not support "
                "prepare_ip_adapter_image_embeds."
            )

        image_embeds = pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image=image,
            ip_adapter_image_embeds=None,
            device=cls.pipe_device(pipe),
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        if offload_image_encoder:
            image_encoder = getattr(pipe, "image_encoder", None)
            if image_encoder is not None and hasattr(image_encoder, "to"):
                image_encoder.to("cpu")
                cleanup_memory()

        return image_embeds
