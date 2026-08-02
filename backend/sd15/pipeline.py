"""Stable public facade for the decomposed SD1.5 runtime.

Loading, preparation, adapter policy, subprocess transport, and each generation
operation live in focused sibling modules.  This module keeps the historical
imports and callable signatures used by workflows, subprocess runners, and
third-party callers.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Callable
from typing import Any

# Workflow assembly can be imported while adapter utilities are initializing.
# Publish the historical runtime hooks before importing the split modules so
# that this benign cycle sees a complete compatibility surface.
def _early_call(name: str, /, *args: Any, **kwargs: Any) -> Any:
    return _call_implementation(globals()[name], *args, **kwargs)


def _run_sd15_subprocess(operation: str, params: dict[str, object]) -> list[str]:
    return _early_call("_impl_run_sd15_subprocess", operation, params)


def generate_images_controlnet(params: dict[str, object]) -> list[str]:
    return _early_call("_impl_generate_images_controlnet", params)


def generate_images(params: dict[str, object]) -> list[str]:
    return _early_call("_impl_generate_images", params)


def generate_images_img2img(params: dict[str, object]) -> list[str]:
    return _early_call("_impl_generate_images_img2img", params)


def generate_images_img2img_controlnet(params: dict[str, object]) -> list[str]:
    return _early_call("_impl_generate_images_img2img_controlnet", params)


def generate_images_inpaint(params: dict[str, object]) -> list[str]:
    return _early_call("_impl_generate_images_inpaint", params)


def generate_images_inpaint_controlnet(params: dict[str, object]) -> list[str]:
    return _early_call("_impl_generate_images_inpaint_controlnet", params)


def run_sd15_hires_fix(*args: Any, **kwargs: Any) -> list[str]:
    return _early_call("_impl_run_sd15_hires_fix", *args, **kwargs)


from backend.sd15.runtime_common import *
from backend.sd15.adapters import (
    _apply_lcm_lora as _impl_apply_lcm_lora,
    _apply_lora_adapters as _impl_apply_lora_adapters,
    _build_ip_adapter_kwargs as _impl_build_ip_adapter_kwargs,
    _cleanup_lora_adapters as _impl_cleanup_lora_adapters,
    _hide_image_encoder_while_using_ip_adapter_embeds as _impl_hide_image_encoder,
    _metadata_without_runtime_images as _impl_metadata_without_runtime_images,
)
from backend.sd15.hires_fix import run_sd15_hires_fix as _impl_run_sd15_hires_fix
from backend.sd15.img2img import (
    generate_images_img2img_controlnet_in_process as _impl_generate_images_img2img_controlnet_in_process,
    generate_images_img2img_in_process as _impl_generate_images_img2img_in_process,
)
from backend.sd15.inpaint import (
    generate_images_inpaint_controlnet_in_process as _impl_generate_images_inpaint_controlnet_in_process,
    generate_images_inpaint_in_process as _impl_generate_images_inpaint_in_process,
)
from backend.sd15.loaders import (
    load_controlnet_img2img_pipeline as _impl_load_controlnet_img2img_pipeline,
    load_controlnet_inpaint_pipeline as _impl_load_controlnet_inpaint_pipeline,
    load_controlnet_pipeline as _impl_load_controlnet_pipeline,
    load_img2img_pipeline as _impl_load_img2img_pipeline,
    load_inpaint_pipeline as _impl_load_inpaint_pipeline,
    load_text2img_pipeline as _impl_load_text2img_pipeline,
)
from backend.sd15.preparation import (
    _build_sd15_prompt_call_kwargs as _impl_build_sd15_prompt_call_kwargs,
    _enable_xformers_memory_efficient_attention_if_available as _impl_enable_xformers,
    _make_inpaint_controlnet_condition as _impl_make_inpaint_controlnet_condition,
    _resize_control_image_to_target as _impl_resize_control_image_to_target,
    _resource_metadata,
    _snap_dimension,
    _upscale_image,
    create_blur_mask,
)
from backend.sd15.text2img import (
    generate_images_controlnet_in_process as _impl_generate_images_controlnet_in_process,
    generate_images_in_process as _impl_generate_images_in_process,
)
from backend.sd15.transport import (
    _run_sd15_subprocess as _impl_run_sd15_subprocess,
    generate_images as _impl_generate_images,
    generate_images_controlnet as _impl_generate_images_controlnet,
    generate_images_img2img as _impl_generate_images_img2img,
    generate_images_img2img_controlnet as _impl_generate_images_img2img_controlnet,
    generate_images_inpaint as _impl_generate_images_inpaint,
    generate_images_inpaint_controlnet as _impl_generate_images_inpaint_controlnet,
)

logger = logging.getLogger(__name__)
_COMPAT_CALL_LOCK = threading.RLock()


def _call_implementation(implementation: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Call a split implementation while honoring patches on this facade.

    Existing tests and integrations patch loader/helper names on
    ``backend.sd15.pipeline``.  The scoped synchronization keeps that behavior
    without making operation modules depend back on the compatibility facade.
    """

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


def _build_sd15_prompt_call_kwargs(pipe: object, prompt: str, negative_prompt: str, *, clip_skip: int, weighting_policy: str) -> dict[str, object]:
    return _call_implementation(_impl_build_sd15_prompt_call_kwargs, pipe, prompt, negative_prompt, clip_skip=clip_skip, weighting_policy=weighting_policy)


def _resize_control_image_to_target(control_image: Image.Image | list[Image.Image], *, target_width: int, target_height: int) -> Image.Image | list[Image.Image]:
    return _call_implementation(_impl_resize_control_image_to_target, control_image, target_width=target_width, target_height=target_height)


def _make_inpaint_controlnet_condition(initial_image: Image.Image, mask_image: Image.Image) -> Image.Image:
    return _call_implementation(_impl_make_inpaint_controlnet_condition, initial_image, mask_image)


def _enable_xformers_memory_efficient_attention_if_available(pipe: object) -> bool:
    return _call_implementation(_impl_enable_xformers, pipe)


def _apply_lora_adapters(pipe: object, lora_adapters: list[object] | None, *, validate: bool = False) -> list[str]:
    return _call_implementation(
        _impl_apply_lora_adapters,
        pipe,
        lora_adapters,
        validate=validate,
    )


def _apply_lcm_lora(pipe: object) -> str:
    return _call_implementation(_impl_apply_lcm_lora, pipe)


def _cleanup_lora_adapters(pipe: object, adapter_names: list[str]) -> None:
    _call_implementation(_impl_cleanup_lora_adapters, pipe, adapter_names)


def _metadata_without_runtime_images(params: dict[str, object]) -> dict[str, object]:
    return _call_implementation(_impl_metadata_without_runtime_images, params)


def _build_ip_adapter_kwargs(*, enabled: bool, image_embeds: list[torch.Tensor] | None, masks: list[torch.Tensor] | None) -> dict[str, object]:
    return _call_implementation(
        _impl_build_ip_adapter_kwargs,
        enabled=enabled,
        image_embeds=image_embeds,
        masks=masks,
    )


def _hide_image_encoder_while_using_ip_adapter_embeds(pipe: object, *, enabled: bool):
    return _call_implementation(_impl_hide_image_encoder, pipe, enabled=enabled)


def _run_sd15_subprocess(operation: str, params: dict[str, object]) -> list[str]:
    return _call_implementation(_impl_run_sd15_subprocess, operation, params)


def generate_images_controlnet(params: dict[str, object]) -> list[str]:
    return _call_implementation(_impl_generate_images_controlnet, params)


def generate_images(params: dict[str, object]) -> list[str]:
    return _call_implementation(_impl_generate_images, params)


def generate_images_img2img(params: dict[str, object]) -> list[str]:
    return _call_implementation(_impl_generate_images_img2img, params)


def generate_images_img2img_controlnet(params: dict[str, object]) -> list[str]:
    return _call_implementation(_impl_generate_images_img2img_controlnet, params)


def generate_images_inpaint(params: dict[str, object]) -> list[str]:
    return _call_implementation(_impl_generate_images_inpaint, params)


def generate_images_inpaint_controlnet(params: dict[str, object]) -> list[str]:
    return _call_implementation(_impl_generate_images_inpaint_controlnet, params)


def load_text2img_pipeline(model_name: str | None):
    return _call_implementation(_impl_load_text2img_pipeline, model_name)


def load_img2img_pipeline(model_name: str | None):
    return _call_implementation(_impl_load_img2img_pipeline, model_name)


def load_inpaint_pipeline(model_name: str | None):
    return _call_implementation(_impl_load_inpaint_pipeline, model_name)


def load_controlnet_pipeline(model_name: str | None, controlnet_model: str | list[str]):
    return _call_implementation(_impl_load_controlnet_pipeline, model_name, controlnet_model)


def load_controlnet_img2img_pipeline(model_name: str | None, controlnet_model: str | list[str]):
    return _call_implementation(_impl_load_controlnet_img2img_pipeline, model_name, controlnet_model)


def load_controlnet_inpaint_pipeline(model_name: str | None, controlnet_model: str | list[str]):
    return _call_implementation(_impl_load_controlnet_inpaint_pipeline, model_name, controlnet_model)


def generate_images_controlnet_in_process(params: dict[str, object]) -> list[str]:
    return _call_implementation(_impl_generate_images_controlnet_in_process, params)


def generate_images_in_process(params: dict[str, object]) -> list[str]:
    return _call_implementation(_impl_generate_images_in_process, params)


def generate_images_img2img_in_process(params: dict[str, object]) -> list[str]:
    return _call_implementation(_impl_generate_images_img2img_in_process, params)


def generate_images_img2img_controlnet_in_process(params: dict[str, object]) -> list[str]:
    return _call_implementation(_impl_generate_images_img2img_controlnet_in_process, params)


def generate_images_inpaint_in_process(params: dict[str, object]) -> list[str]:
    return _call_implementation(_impl_generate_images_inpaint_in_process, params)


def generate_images_inpaint_controlnet_in_process(params: dict[str, object]) -> list[str]:
    return _call_implementation(_impl_generate_images_inpaint_controlnet_in_process, params)


def run_sd15_hires_fix(*, images: list[Image.Image], prompt: str, negative_prompt: str, steps: int, cfg: float, seed: int | None, scheduler: str, model: str | None, clip_skip: int, hires_scale: float, hires_strength: float, lora_adapters: object | None, weighting_policy: str, output_dir: Path, batch_id: str) -> list[str]:
    return _call_implementation(_impl_run_sd15_hires_fix, images=images, prompt=prompt, negative_prompt=negative_prompt, steps=steps, cfg=cfg, seed=seed, scheduler=scheduler, model=model, clip_skip=clip_skip, hires_scale=hires_scale, hires_strength=hires_strength, lora_adapters=lora_adapters, weighting_policy=weighting_policy, output_dir=output_dir, batch_id=batch_id)
