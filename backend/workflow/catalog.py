from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

_SCHEDULER_OPTIONS: list[str] = [
    "euler",
    "euler_a",
    "lcm",
    "ddim",
    "dpm++2m",
    "dpm++2m_karras",
    "dpm++2m_sde",
    "dpm++2m_sde_karras",
    "dpm++_sde",
    "dpm++_sde_karras",
    "dpm2",
    "dpm2_karras",
    "dpm2_a",
    "dpm2_a_karras",
    "flowmatch_euler",
    "flowmatch_heun",
    "heun",
    "lms",
    "lms_karras",
    "deis",
    "unipc",
]


_WEIGHTING_POLICY_OPTIONS: list[str] = [
    "diffusers-like",
    "a1111-like",
    "comfyui-like",
]

_MODEL_FAMILY_METADATA: dict[str, dict[str, Any]] = {
    "sd15": {"label": "SD 1.5", "aliases": ["sd1.5"]},
    "sdxl": {"label": "SDXL", "aliases": []},
    "flux": {"label": "Flux", "aliases": []},
    "wan": {"label": "WAN", "aliases": ["wan2.1", "wan2.2"]},
    "qwen-image": {"label": "Qwen-Image", "aliases": ["qwen"]},
    "z-image": {"label": "Z-Image", "aliases": ["zimage"]},
}


def _infer_model_family(task_type: str) -> str | None:
    prefix = task_type.split(".", 1)[0]
    if prefix in {"sd15", "sdxl", "flux", "wan"}:
        return prefix
    if prefix == "qwen-image":
        return "qwen-image"
    if prefix == "z-image":
        return "z-image"
    return None


def _build_task_ui_hints(task_type: str, model_cls: type[BaseModel]) -> dict[str, Any]:
    # Minimal, stable contract for UIs/workflow builders. Everything here is optional
    # and should be treated as best-effort.
    family = _infer_model_family(task_type)

    title = task_type
    if task_type.endswith(".text2img"):
        title = f"{task_type} (Text to Image)"
    elif task_type.endswith(".text2video"):
        title = f"{task_type} (Text to Video)"
    elif task_type.endswith(".img2img"):
        title = f"{task_type} (Image to Image)"
    elif task_type.endswith(".inpaint"):
        title = f"{task_type} (Inpaint)"
    elif task_type == "controlnet.preprocess":
        title = "controlnet.preprocess (Preprocessor)"
    elif task_type == "sd15.hires_fix":
        title = "sd15.hires_fix (Hires Fix)"

    inputs: dict[str, Any] = {}
    input_order: list[str] = []

    common_numeric: dict[str, dict[str, Any]] = {
        "steps": {"min": 1, "max": 200, "step": 1, "integer": True},
        "cfg": {"min": 0, "max": 30, "step": 0.1},
        "guidance_scale": {"min": 0, "max": 30, "step": 0.1},
        "true_cfg_scale": {"min": 0, "max": 30, "step": 0.1},
        "strength": {"min": 0, "max": 1, "step": 0.01},
        "width": {"min": 64, "max": 2048, "step": 8, "integer": True},
        "height": {"min": 64, "max": 2048, "step": 8, "integer": True},
        "num_images": {"min": 1, "max": 8, "step": 1, "integer": True},
        "num_videos": {"min": 1, "max": 8, "step": 1, "integer": True},
        "num_frames": {"min": 1, "max": 256, "step": 1, "integer": True},
        "free_noise_context_length": {"min": 1, "max": 32, "step": 1, "integer": True},
        "free_noise_context_stride": {"min": 1, "max": 32, "step": 1, "integer": True},
        "free_init_num_iters": {"min": 1, "max": 8, "step": 1, "integer": True},
        "free_init_order": {"min": 1, "max": 16, "step": 1, "integer": True},
        "free_init_spatial_stop_frequency": {"min": 0, "max": 1, "step": 0.01},
        "free_init_temporal_stop_frequency": {"min": 0, "max": 1, "step": 0.01},
        "fps": {"min": 1, "max": 60, "step": 1, "integer": True},
        "clip_skip": {"min": 1, "max": 4, "step": 1, "integer": True},
        "padding_mask_crop": {"min": 0, "max": 128, "step": 1, "integer": True},
        "hires_scale": {"min": 1, "max": 4, "step": 0.05},
        "hires_strength": {"min": 0, "max": 1, "step": 0.01},
        "controlnet_conditioning_scale": {"min": 0, "max": 2, "step": 0.05},
        "control_guidance_start": {"min": 0, "max": 1, "step": 0.01},
        "control_guidance_end": {"min": 0, "max": 1, "step": 0.01},
        "seed": {"min": 0, "max": 2**31 - 1, "step": 1, "integer": True},
    }

    for field_name, field_info in model_cls.model_fields.items():
        input_order.append(field_name)
        hint: dict[str, Any] = {"label": field_name.replace("_", " ").title()}

        if field_name in {"prompt", "negative_prompt"}:
            hint.update(
                widget="textarea",
                placeholder="",
                multiline=True,
            )
            if field_name == "prompt":
                hint["placeholder"] = "Describe what you want to generate..."
            else:
                hint["placeholder"] = "Describe what to avoid..."

        if field_name in {"initial_image", "mask_image", "control_image", "image"}:
            hint.update(
                widget="image_ref",
                accepts=["artifact", "outputs", "task_ref"],
                help="Upload via /api/artifacts, or reference a prior task output (e.g. @t1.images[0]).",
            )

        if field_name == "reference_image":
            hint.update(
                widget="image_ref",
                accepts=["artifact", "outputs", "task_ref"],
                help="Upload via /api/artifacts, or reference a prior image output.",
            )

        if field_name == "conditioning_video":
            hint.update(
                widget="video_ref",
                accepts=["artifact", "outputs", "task_ref"],
                help="Upload via /api/artifacts, or reference a prior video output.",
            )

        if field_name in {"images", "control_images"}:
            hint.update(
                widget="image_list_ref",
                accepts=["artifact", "outputs", "task_ref"],
            )

        if field_name == "scheduler":
            hint.update(widget="select", options=_SCHEDULER_OPTIONS)

        if field_name == "weighting_policy":
            hint.update(widget="select", options=_WEIGHTING_POLICY_OPTIONS)

        if field_name == "free_init_method":
            hint.update(widget="select", options=["butterworth", "ideal", "gaussian"])

        if field_name == "model":
            if family:
                hint.update(
                    widget="model_select",
                    source={"type": "models", "params": {"family": family}},
                )
            else:
                hint.update(widget="text")

        if field_name in {"preprocessor_id", "controlnet_preprocessor_id"}:
            hint.update(
                widget="select",
                source={"type": "controlnet_preprocessors", "endpoint": "/api/controlnet/preprocessors"},
            )
        if field_name == "controlnet_preprocessor_ids":
            hint.update(widget="json")
        if field_name == "controlnet_compat_mode":
            hint.update(widget="select", options=["warn", "error", "off"])
        if field_name in {"controlnet_models", "controlnet_conditioning_scales"}:
            hint.update(widget="json")

        if field_name == "lora_adapters":
            hint.update(
                widget="json",
                advanced=True,
                help="List of LoRA adapter objects; UI may provide a dedicated editor.",
            )
        if field_name == "lora":
            hint.update(
                widget="json",
                advanced=True,
                help="Unified SD1.5 LoRA contract: { lora_enabled, lora_adapters }.",
            )

        if field_name == "lcm":
            hint.update(
                widget="json",
                advanced=True,
                help="SD1.5 LCM mode: { enabled }.",
            )

        if field_name == "ip_adapter":
            hint.update(
                widget="json",
                advanced=True,
                help="IP-Adapter contract: { enabled, image, image_embeds, mask_image, scale, model, subfolder, weight_name }.",
            )

        if field_name in common_numeric:
            hint.setdefault("widget", "number")
            hint.update(common_numeric[field_name])

        if field_info.annotation is bool or str(field_info.annotation) == "bool":
            hint.setdefault("widget", "checkbox")

        inputs[field_name] = hint

    return {
        "title": title,
        "task_type": task_type,
        "input_order": input_order,
        "inputs": inputs,
    }


def _build_model_capabilities(task_input_models: dict[str, type[BaseModel]]) -> dict[str, Any]:
    families: dict[str, dict[str, Any]] = {}
    for task_type, model_cls in task_input_models.items():
        family = _infer_model_family(task_type)
        if not family:
            continue
        meta = _MODEL_FAMILY_METADATA.get(family, {})
        entry = families.setdefault(
            family,
            {
                "label": meta.get("label", family),
                "aliases": list(meta.get("aliases", [])),
                "task_types": [],
                "features": {
                    "text2img": False,
                    "text2video": False,
                    "img2img": False,
                    "inpaint": False,
                    "controlnet": False,
                    "multi_controlnet": False,
                    "hires_fix": False,
                    "lora_adapters": False,
                    "ip_adapter": False,
                    "scheduler": False,
                    "true_cfg_scale": False,
                },
            },
        )
        if task_type not in entry["task_types"]:
            entry["task_types"].append(task_type)

        features = entry["features"]
        if task_type.endswith(".text2img"):
            features["text2img"] = True
        if task_type.endswith(".text2video"):
            features["text2video"] = True
        if task_type.endswith(".img2img"):
            features["img2img"] = True
        if task_type.endswith(".inpaint"):
            features["inpaint"] = True
        if task_type.endswith(".hires_fix"):
            features["hires_fix"] = True

        field_names = model_cls.model_fields.keys()
        if "scheduler" in field_names:
            features["scheduler"] = True
        if "true_cfg_scale" in field_names:
            features["true_cfg_scale"] = True
        if "lora_adapters" in field_names or "lora" in field_names:
            features["lora_adapters"] = True
        if "ip_adapter" in field_names:
            features["ip_adapter"] = True
        if (
            "controlnet_model" in field_names
            or "controlnet_models" in field_names
            or "control_image" in field_names
            or "control_images" in field_names
        ):
            features["controlnet"] = True
        if "controlnet_models" in field_names:
            features["multi_controlnet"] = True

    return {family: families[family] for family in sorted(families.keys())}


def build_workflow_catalog(
    task_input_models: dict[str, type[BaseModel]],
    task_output_models: dict[str, type[BaseModel]],
) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for task_type, model_cls in task_input_models.items():
        defaults: dict[str, Any] = {}
        for field_name, field_info in model_cls.model_fields.items():
            if field_info.default is not PydanticUndefined:
                defaults[field_name] = field_info.default
                continue
            if field_info.default_factory is not None:
                defaults[field_name] = field_info.default_factory()
        output_model = task_output_models.get(task_type)
        tasks[task_type] = {
            "input_schema": model_cls.model_json_schema(by_alias=True),
            "input_defaults": defaults,
            "output_schema": output_model.model_json_schema(by_alias=True) if output_model else None,
            "ui_hints": _build_task_ui_hints(task_type, model_cls),
        }
    return {
        "version": "v2",
        "tasks": tasks,
        "capabilities": _build_model_capabilities(task_input_models),
    }
