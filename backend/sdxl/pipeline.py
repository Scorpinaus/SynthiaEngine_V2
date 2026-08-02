"""Stable public facade for the decomposed SDXL runtime."""

from __future__ import annotations

import functools
import logging
import sys
import threading
from collections.abc import Callable
from typing import Any

from backend.sdxl.runtime_common import *
from backend.sdxl.adapters import _cleanup_lora_adapters as _impl_cleanup_lora_adapters
from backend.sdxl.controlnet import (
    _resize_control_image_to_target as _impl_resize_control_image_to_target,
    generate_controlnet_text2img_in_process as _impl_generate_controlnet_text2img_in_process,
    generate_img2img_controlnet_in_process as _impl_generate_img2img_controlnet_in_process,
    generate_inpaint_controlnet_in_process as _impl_generate_inpaint_controlnet_in_process,
)
from backend.sdxl.img2img import generate_img2img_in_process as _impl_generate_img2img_in_process
from backend.sdxl.inpaint import generate_inpaint_in_process as _impl_generate_inpaint_in_process
from backend.sdxl.loaders import (
    load_controlnet_img2img_pipeline as _impl_load_controlnet_img2img_pipeline,
    load_controlnet_inpaint_pipeline as _impl_load_controlnet_inpaint_pipeline,
    load_controlnet_text2img_pipeline as _impl_load_controlnet_text2img_pipeline,
    load_img2img_pipeline as _impl_load_img2img_pipeline,
    load_inpaint_pipeline as _impl_load_inpaint_pipeline,
    load_text2img_pipeline as _impl_load_text2img_pipeline,
)
from backend.sdxl.preparation import (
    _LatentDecoder,
    _build_latent_decoder as _impl_build_latent_decoder,
    _decode_latents_to_pil as _impl_decode_latents_to_pil,
    _enable_vae_memory_savers as _impl_enable_vae_memory_savers,
    _get_module_device as _impl_get_module_device,
    _get_pipe_device as _impl_get_pipe_device,
    _hide_image_encoder_while_using_ip_adapter_embeds as _impl_hide_image_encoder,
    render_img2img_latents as _impl_render_img2img_latents,
    render_inpaint_image as _impl_render_inpaint_image,
    render_text2img_latents as _impl_render_text2img_latents,
)
from backend.sdxl.results import (
    _metadata_without_runtime_images as _impl_metadata_without_runtime_images,
    save_image as _impl_save_image,
)
from backend.sdxl.text2img import generate_text2img_in_process as _impl_generate_text2img_in_process
from backend.sdxl.transport import (
    _run_sdxl_subprocess as _impl_run_sdxl_subprocess,
    generate_controlnet_text2img as _impl_generate_controlnet_text2img,
    generate_img2img as _impl_generate_img2img,
    generate_img2img_controlnet as _impl_generate_img2img_controlnet,
    generate_inpaint as _impl_generate_inpaint,
    generate_inpaint_controlnet as _impl_generate_inpaint_controlnet,
    generate_text2img as _impl_generate_text2img,
)

logger = logging.getLogger(__name__)
_COMPAT_CALL_LOCK = threading.RLock()


def _call_implementation(implementation: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Call a split implementation while honoring patches on this facade."""

    implementation_module = sys.modules[implementation.__module__]
    facade_globals = globals()
    with _COMPAT_CALL_LOCK:
        original: dict[str, Any] = {}
        for name, value in facade_globals.items():
            if name.startswith("__") or name not in vars(implementation_module):
                continue
            original[name] = getattr(implementation_module, name)
            setattr(implementation_module, name, value)
        try:
            return implementation(*args, **kwargs)
        finally:
            for name, value in original.items():
                setattr(implementation_module, name, value)


def _compat(implementation: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(implementation)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return _call_implementation(implementation, *args, **kwargs)

    return wrapper


_get_pipe_device = _compat(_impl_get_pipe_device)
_get_module_device = _compat(_impl_get_module_device)
_enable_vae_memory_savers = _compat(_impl_enable_vae_memory_savers)
_build_latent_decoder = _compat(_impl_build_latent_decoder)
_hide_image_encoder_while_using_ip_adapter_embeds = _compat(_impl_hide_image_encoder)
_decode_latents_to_pil = _compat(_impl_decode_latents_to_pil)
render_text2img_latents = _compat(_impl_render_text2img_latents)
render_img2img_latents = _compat(_impl_render_img2img_latents)
render_inpaint_image = _compat(_impl_render_inpaint_image)
save_image = _compat(_impl_save_image)
_resize_control_image_to_target = _compat(_impl_resize_control_image_to_target)
_cleanup_lora_adapters = _compat(_impl_cleanup_lora_adapters)
_metadata_without_runtime_images = _compat(_impl_metadata_without_runtime_images)

_run_sdxl_subprocess = _compat(_impl_run_sdxl_subprocess)
generate_controlnet_text2img = _compat(_impl_generate_controlnet_text2img)
generate_img2img_controlnet = _compat(_impl_generate_img2img_controlnet)
generate_text2img = _compat(_impl_generate_text2img)
generate_img2img = _compat(_impl_generate_img2img)
generate_inpaint = _compat(_impl_generate_inpaint)
generate_inpaint_controlnet = _compat(_impl_generate_inpaint_controlnet)

load_text2img_pipeline = _compat(_impl_load_text2img_pipeline)
load_controlnet_text2img_pipeline = _compat(_impl_load_controlnet_text2img_pipeline)
load_img2img_pipeline = _compat(_impl_load_img2img_pipeline)
load_controlnet_img2img_pipeline = _compat(_impl_load_controlnet_img2img_pipeline)
load_inpaint_pipeline = _compat(_impl_load_inpaint_pipeline)
load_controlnet_inpaint_pipeline = _compat(_impl_load_controlnet_inpaint_pipeline)

generate_controlnet_text2img_in_process = _compat(_impl_generate_controlnet_text2img_in_process)
generate_img2img_controlnet_in_process = _compat(_impl_generate_img2img_controlnet_in_process)
generate_text2img_in_process = _compat(_impl_generate_text2img_in_process)
generate_img2img_in_process = _compat(_impl_generate_img2img_in_process)
generate_inpaint_in_process = _compat(_impl_generate_inpaint_in_process)
generate_inpaint_controlnet_in_process = _compat(_impl_generate_inpaint_controlnet_in_process)
