from __future__ import annotations

from typing import Any

from PIL import Image

_LCM_LORA_MODEL_ID = "latent-consistency/lcm-lora-sdv1-5"
_LCM_DEFAULT_STEPS = 4
_LCM_DEFAULT_CFG = 0.0
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


def run_sd15_ip_adapter_encode_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
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
        raise ValueError("sd15.ip_adapter.encode must return an object")
    return result


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


def run_sd15_animatediff_text2video(
    inputs: dict[str, Any],
    deps: dict[str, Any],
) -> dict[str, Any]:
    _normalized_lora_adapters = deps["normalized_lora_adapters"]
    make_batch_id = deps["make_batch_id"]
    generate_videos_text2video = deps["generate_videos_text2video"]

    lora_adapters = _normalized_lora_adapters(inputs)
    batch_id = str(inputs.get("batch_id") or make_batch_id())
    generation_params = {
        "prompt": str(inputs["prompt"]),
        "negative_prompt": str(inputs.get("negative_prompt") or ""),
        "steps": int(inputs.get("steps") or 25),
        "cfg": float(inputs.get("cfg") or 7.5),
        "width": int(inputs.get("width") or 512),
        "height": int(inputs.get("height") or 512),
        "seed": inputs.get("seed"),
        "scheduler": str(inputs.get("scheduler") or "ddim"),
        "model": inputs.get("model"),
        "motion_adapter": str(
            inputs.get("motion_adapter")
            or "guoyww/animatediff-motion-adapter-v1-5-2"
        ),
        "num_frames": int(inputs.get("num_frames") or 16),
        "fps": int(inputs.get("fps") or 8),
        "num_videos": int(inputs.get("num_videos") or 1),
        "free_noise_enabled": inputs.get("free_noise_enabled", False),
        "free_noise_context_length": int(inputs.get("free_noise_context_length") or 16),
        "free_noise_context_stride": int(inputs.get("free_noise_context_stride") or 4),
        "free_init_enabled": inputs.get("free_init_enabled", False),
        "free_init_num_iters": int(inputs.get("free_init_num_iters") or 3),
        "free_init_use_fast_sampling": inputs.get("free_init_use_fast_sampling", False),
        "free_init_method": str(inputs.get("free_init_method") or "butterworth"),
        "free_init_order": int(inputs.get("free_init_order") or 4),
        "free_init_spatial_stop_frequency": float(
            inputs.get("free_init_spatial_stop_frequency", 0.25)
        ),
        "free_init_temporal_stop_frequency": float(
            inputs.get("free_init_temporal_stop_frequency", 0.25)
        ),
        "clip_skip": int(inputs.get("clip_skip") or 1),
        "lora_adapters": lora_adapters,
        "weighting_policy": str(inputs.get("weighting_policy") or "diffusers-like"),
        "batch_id": batch_id,
    }
    filenames = generate_videos_text2video(generation_params)
    return {"batch_id": batch_id, "videos": [f"/outputs/{name}" for name in filenames]}

def run_sd15_img2img(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    _open_image_ref = deps["open_image_ref"]
    _remap_img2img_strength = deps["remap_img2img_strength"]
    _normalized_lora_adapters = deps["normalized_lora_adapters"]
    make_batch_id = deps["make_batch_id"]
    _DEFAULT_SD15_CONTROLNET_MODEL = deps["default_sd15_controlnet_model"]
    _MAX_CONTROLNET_MODELS = deps["max_controlnet_models"]
    _CONTROLNET_PREPROCESSOR_REGISTRY_BY_ID = deps["controlnet_preprocessor_registry_by_id"]
    logger = deps["logger"]
    generate_images_img2img_controlnet = deps["generate_images_img2img_controlnet"]
    generate_images_img2img = deps["generate_images_img2img"]

    initial_image = _open_image_ref(inputs["initial_image"]).convert("RGB")
    width = int(inputs.get("width") or initial_image.width)
    height = int(inputs.get("height") or initial_image.height)
    initial_image = initial_image.resize((width, height))

    strength = float(inputs.get("strength") or 0.75)
    strength = _remap_img2img_strength(strength)
    lora_adapters = _normalized_lora_adapters(inputs)
    lcm_enabled = _lcm_enabled(inputs)
    if lcm_enabled:
        steps = int(
            inputs["steps"]
            if "steps" in inputs and inputs.get("steps") is not None
            else _LCM_DEFAULT_STEPS
        )
        cfg = float(
            inputs["cfg"]
            if "cfg" in inputs and inputs.get("cfg") is not None
            else _LCM_DEFAULT_CFG
        )
    else:
        steps = int(inputs.get("steps") or 20)
        cfg = float(inputs.get("cfg") or 7.5)
    scheduler = "lcm" if lcm_enabled else str(inputs.get("scheduler") or "euler")
    if lcm_enabled:
        _validate_lcm_img2img_settings(steps, cfg)
    ip_adapter_raw = inputs.get("ip_adapter")
    if ip_adapter_raw is not None and not isinstance(ip_adapter_raw, dict):
        raise ValueError("`ip_adapter` must be an object.")
    ip_adapter_enabled = (
        isinstance(ip_adapter_raw, dict) and bool(ip_adapter_raw.get("enabled", False))
    )
    if lcm_enabled and ip_adapter_enabled:
        raise ValueError("sd15.img2img IP-Adapter cannot be combined with LCM mode.")

    batch_id = str(inputs.get("batch_id") or make_batch_id())
    controlnet_requested = any(
        key in inputs
        for key in (
            "control_image",
            "control_images",
            "controlnet_model",
            "controlnet_models",
            "controlnet_conditioning_scale",
            "controlnet_conditioning_scales",
            "controlnet_guess_mode",
            "control_guidance_start",
            "control_guidance_end",
            "controlnet_preprocessor_id",
            "controlnet_preprocessor_ids",
            "controlnet_compat_mode",
        )
    )
    if controlnet_requested:
        if ip_adapter_enabled:
            raise ValueError("sd15.img2img IP-Adapter cannot be combined with ControlNet.")
        if lcm_enabled:
            raise ValueError("sd15.img2img LCM mode cannot be combined with ControlNet.")

        control_image_input = inputs.get("control_image")
        control_images_raw = inputs.get("control_images")
        if control_images_raw is not None and not isinstance(control_images_raw, list):
            raise ValueError("control_images must be a list of image references")
        if control_image_input is None and not control_images_raw:
            raise ValueError("control_image is required when using ControlNet in sd15.img2img")

        control_guidance_start = float(inputs.get("control_guidance_start", 0.0))
        control_guidance_end = float(inputs.get("control_guidance_end", 1.0))
        if control_guidance_start > control_guidance_end:
            raise ValueError("control_guidance_start must be <= control_guidance_end")

        controlnet_model_single = str(
            inputs.get("controlnet_model") or _DEFAULT_SD15_CONTROLNET_MODEL
        )
        controlnet_models_raw = inputs.get("controlnet_models")
        if controlnet_models_raw is not None and not isinstance(controlnet_models_raw, list):
            raise ValueError("controlnet_models must be a list of model ids")
        controlnet_models: list[str] = (
            [str(item) for item in controlnet_models_raw] if controlnet_models_raw else []
        )
        if not controlnet_models:
            controlnet_models = [controlnet_model_single]

        control_images: list[Image.Image] = []
        if control_image_input is not None:
            control_images.append(
                _open_image_ref(control_image_input).convert("RGB").resize((width, height))
            )
        if control_images_raw:
            control_images.extend(
                _open_image_ref(image_ref).convert("RGB").resize((width, height))
                for image_ref in control_images_raw
            )

        if len(controlnet_models) > _MAX_CONTROLNET_MODELS:
            raise ValueError(
                f"At most {_MAX_CONTROLNET_MODELS} ControlNet models are supported per task."
            )

        if len(controlnet_models) != len(control_images):
            if len(controlnet_models) == 1 and len(control_images) > 1:
                controlnet_models = controlnet_models * len(control_images)
            elif len(control_images) == 1 and len(controlnet_models) > 1:
                control_images = control_images * len(controlnet_models)
            else:
                raise ValueError(
                    "controlnet_models and control_images must have the same length."
                )

        controlnet_count = len(controlnet_models)
        controlnet_conditioning_scales_raw = inputs.get("controlnet_conditioning_scales")
        if (
            controlnet_conditioning_scales_raw is not None
            and not isinstance(controlnet_conditioning_scales_raw, list)
        ):
            raise ValueError("controlnet_conditioning_scales must be a list of numbers")
        if controlnet_conditioning_scales_raw:
            controlnet_conditioning_scales = [
                float(item) for item in controlnet_conditioning_scales_raw
            ]
            if len(controlnet_conditioning_scales) != controlnet_count:
                raise ValueError(
                    "controlnet_conditioning_scales length must match controlnet_models length."
                )
        else:
            controlnet_conditioning_scales = [
                float(inputs.get("controlnet_conditioning_scale", 1.0))
            ] * controlnet_count
        for scale in controlnet_conditioning_scales:
            if scale < 0.0 or scale > 2.0:
                raise ValueError("controlnet conditioning scales must be within [0, 2].")

        controlnet_preprocessor_id_raw = inputs.get("controlnet_preprocessor_id")
        controlnet_preprocessor_id = (
            str(controlnet_preprocessor_id_raw)
            if controlnet_preprocessor_id_raw is not None
            else None
        )
        controlnet_preprocessor_ids_raw = inputs.get("controlnet_preprocessor_ids")
        if (
            controlnet_preprocessor_ids_raw is not None
            and not isinstance(controlnet_preprocessor_ids_raw, list)
        ):
            raise ValueError("controlnet_preprocessor_ids must be a list of preprocessor ids")
        if controlnet_preprocessor_ids_raw:
            controlnet_preprocessor_ids = [
                str(item) if item is not None else None
                for item in controlnet_preprocessor_ids_raw
            ]
            if len(controlnet_preprocessor_ids) != controlnet_count:
                raise ValueError(
                    "controlnet_preprocessor_ids length must match controlnet_models length."
                )
        else:
            controlnet_preprocessor_ids = [controlnet_preprocessor_id] * controlnet_count

        controlnet_compat_mode = str(inputs.get("controlnet_compat_mode") or "warn").lower()
        warnings: list[str] = []
        if controlnet_count > 1:
            perf_warning = (
                f"Using {controlnet_count} ControlNet models may significantly increase VRAM use "
                "and generation latency."
            )
            warnings.append(perf_warning)
            logger.warning(perf_warning)

        if controlnet_compat_mode != "off":
            for idx, (model_id, preprocessor_id) in enumerate(
                zip(controlnet_models, controlnet_preprocessor_ids)
            ):
                if not preprocessor_id:
                    continue
                entry = _CONTROLNET_PREPROCESSOR_REGISTRY_BY_ID.get(preprocessor_id)
                if entry is None:
                    message = (
                        f"Unknown controlnet_preprocessor_id '{preprocessor_id}' at index {idx}. "
                        "Compatibility check could not be applied."
                    )
                    if controlnet_compat_mode == "error":
                        raise ValueError(message)
                    warnings.append(message)
                    logger.warning(message)
                    continue
                compatible_models = set(entry.recommended_sd15_control_models) | set(
                    entry.legacy_aliases
                )
                if compatible_models and model_id not in compatible_models:
                    message = (
                        "ControlNet model/preprocessor pairing mismatch: "
                        f"preprocessor '{preprocessor_id}' with model '{model_id}' at index {idx}. "
                        f"Recommended models: {', '.join(entry.recommended_sd15_control_models)}."
                    )
                    if controlnet_compat_mode == "error":
                        raise ValueError(message)
                    warnings.append(message)
                    logger.warning(message)

        controlnet_model_arg: str | list[str]
        control_image_arg: Image.Image | list[Image.Image]
        controlnet_conditioning_scale_arg: float | list[float]
        if controlnet_count == 1:
            controlnet_model_arg = controlnet_models[0]
            control_image_arg = control_images[0]
            controlnet_conditioning_scale_arg = controlnet_conditioning_scales[0]
        else:
            controlnet_model_arg = controlnet_models
            control_image_arg = control_images
            controlnet_conditioning_scale_arg = controlnet_conditioning_scales

        generation_params = {
            "initial_image": initial_image,
            "strength": strength,
            "prompt": str(inputs["prompt"]),
            "negative_prompt": str(inputs.get("negative_prompt") or ""),
            "steps": steps,
            "cfg": cfg,
            "width": width,
            "height": height,
            "seed": inputs.get("seed"),
            "scheduler": scheduler,
            "model": inputs.get("model"),
            "num_images": int(inputs.get("num_images") or 1),
            "clip_skip": int(inputs.get("clip_skip") or 1),
            "control_image": control_image_arg,
            "controlnet_model": controlnet_model_arg,
            "controlnet_conditioning_scale": controlnet_conditioning_scale_arg,
            "controlnet_guess_mode": bool(inputs.get("controlnet_guess_mode", False)),
            "control_guidance_start": control_guidance_start,
            "control_guidance_end": control_guidance_end,
            "lora_adapters": lora_adapters,
            "batch_id": batch_id,
        }
        filenames = generate_images_img2img_controlnet(generation_params)
        result: dict[str, Any] = {
            "batch_id": batch_id,
            "images": [f"/outputs/{name}" for name in filenames],
        }
        if warnings:
            result["warnings"] = warnings
        return result

    generation_params = {
        "initial_image": initial_image,
        "strength": strength,
        "prompt": str(inputs["prompt"]),
        "negative_prompt": str(inputs.get("negative_prompt") or ""),
        "steps": steps,
        "cfg": cfg,
        "width": width,
        "height": height,
        "seed": inputs.get("seed"),
        "scheduler": scheduler,
        "model": inputs.get("model"),
        "num_images": int(inputs.get("num_images") or 1),
        "clip_skip": int(inputs.get("clip_skip") or 1),
        "lora_adapters": lora_adapters,
        "lcm_enabled": lcm_enabled,
        "lcm_lora_model": _LCM_LORA_MODEL_ID if lcm_enabled else None,
        "batch_id": batch_id,
    }
    ip_adapter_settings = _normalized_ip_adapter_settings(inputs, _open_image_ref)
    if ip_adapter_settings is not None:
        generation_params.update(ip_adapter_settings)
    filenames = generate_images_img2img(generation_params)
    return {"batch_id": batch_id, "images": [f"/outputs/{name}" for name in filenames]}

def run_sd15_inpaint(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    _open_image_ref = deps["open_image_ref"]
    _remap_img2img_strength = deps["remap_img2img_strength"]
    _normalized_lora_adapters = deps["normalized_lora_adapters"]
    make_batch_id = deps["make_batch_id"]
    _DEFAULT_SD15_CONTROLNET_MODEL = deps["default_sd15_controlnet_model"]
    _MAX_CONTROLNET_MODELS = deps["max_controlnet_models"]
    _CONTROLNET_PREPROCESSOR_REGISTRY_BY_ID = deps["controlnet_preprocessor_registry_by_id"]
    logger = deps["logger"]
    generate_images_inpaint_controlnet = deps["generate_images_inpaint_controlnet"]
    generate_images_inpaint = deps["generate_images_inpaint"]

    initial_image = _open_image_ref(inputs["initial_image"]).convert("RGB")
    mask_image = _open_image_ref(inputs["mask_image"]).convert("L")
    if mask_image.size != initial_image.size:
        mask_image = mask_image.resize(initial_image.size)

    strength = float(inputs.get("strength") or 0.5)
    strength = _remap_img2img_strength(strength)
    padding_mask_crop_input = inputs.get("padding_mask_crop")
    padding_mask_crop = 32 if padding_mask_crop_input is None else int(padding_mask_crop_input)
    lora_adapters = _normalized_lora_adapters(inputs)
    lcm_enabled = _lcm_enabled(inputs)
    if lcm_enabled:
        steps = int(
            inputs["steps"]
            if "steps" in inputs and inputs.get("steps") is not None
            else _LCM_DEFAULT_STEPS
        )
        cfg = float(
            inputs["cfg"]
            if "cfg" in inputs and inputs.get("cfg") is not None
            else _LCM_DEFAULT_CFG
        )
    else:
        steps = int(inputs.get("steps") or 20)
        cfg = float(inputs.get("cfg") or 7.5)
    scheduler = "lcm" if lcm_enabled else str(inputs.get("scheduler") or "euler")
    if lcm_enabled:
        _validate_lcm_inpaint_settings(steps, cfg)
    ip_adapter_raw = inputs.get("ip_adapter")
    if ip_adapter_raw is not None and not isinstance(ip_adapter_raw, dict):
        raise ValueError("`ip_adapter` must be an object.")
    ip_adapter_enabled = (
        isinstance(ip_adapter_raw, dict) and bool(ip_adapter_raw.get("enabled", False))
    )
    if lcm_enabled and ip_adapter_enabled:
        raise ValueError("sd15.inpaint IP-Adapter cannot be combined with LCM mode.")
    batch_id = str(inputs.get("batch_id") or make_batch_id())
    width, height = initial_image.size

    controlnet_requested = any(
        key in inputs
        for key in (
            "control_image",
            "control_images",
            "controlnet_model",
            "controlnet_models",
            "controlnet_conditioning_scale",
            "controlnet_conditioning_scales",
            "controlnet_guess_mode",
            "control_guidance_start",
            "control_guidance_end",
            "controlnet_preprocessor_id",
            "controlnet_preprocessor_ids",
            "controlnet_compat_mode",
        )
    )
    if controlnet_requested:
        if ip_adapter_enabled:
            raise ValueError("sd15.inpaint IP-Adapter cannot be combined with ControlNet.")
        if lcm_enabled:
            raise ValueError("sd15.inpaint LCM mode cannot be combined with ControlNet.")

        control_image_input = inputs.get("control_image")
        control_images_raw = inputs.get("control_images")
        if control_images_raw is not None and not isinstance(control_images_raw, list):
            raise ValueError("control_images must be a list of image references")
        if control_image_input is None and not control_images_raw:
            raise ValueError("control_image is required when using ControlNet in sd15.inpaint")

        control_guidance_start = float(inputs.get("control_guidance_start", 0.0))
        control_guidance_end = float(inputs.get("control_guidance_end", 1.0))
        if control_guidance_start > control_guidance_end:
            raise ValueError("control_guidance_start must be <= control_guidance_end")

        controlnet_model_single = str(
            inputs.get("controlnet_model") or _DEFAULT_SD15_CONTROLNET_MODEL
        )
        controlnet_models_raw = inputs.get("controlnet_models")
        if controlnet_models_raw is not None and not isinstance(controlnet_models_raw, list):
            raise ValueError("controlnet_models must be a list of model ids")
        controlnet_models: list[str] = (
            [str(item) for item in controlnet_models_raw] if controlnet_models_raw else []
        )
        if not controlnet_models:
            controlnet_models = [controlnet_model_single]

        control_images: list[Image.Image] = []
        if control_image_input is not None:
            control_images.append(
                _open_image_ref(control_image_input).convert("RGB").resize((width, height))
            )
        if control_images_raw:
            control_images.extend(
                _open_image_ref(image_ref).convert("RGB").resize((width, height))
                for image_ref in control_images_raw
            )

        if len(controlnet_models) > _MAX_CONTROLNET_MODELS:
            raise ValueError(
                f"At most {_MAX_CONTROLNET_MODELS} ControlNet models are supported per task."
            )

        if len(controlnet_models) != len(control_images):
            if len(controlnet_models) == 1 and len(control_images) > 1:
                controlnet_models = controlnet_models * len(control_images)
            elif len(control_images) == 1 and len(controlnet_models) > 1:
                control_images = control_images * len(controlnet_models)
            else:
                raise ValueError(
                    "controlnet_models and control_images must have the same length."
                )

        controlnet_count = len(controlnet_models)
        controlnet_conditioning_scales_raw = inputs.get("controlnet_conditioning_scales")
        if (
            controlnet_conditioning_scales_raw is not None
            and not isinstance(controlnet_conditioning_scales_raw, list)
        ):
            raise ValueError("controlnet_conditioning_scales must be a list of numbers")
        if controlnet_conditioning_scales_raw:
            controlnet_conditioning_scales = [
                float(item) for item in controlnet_conditioning_scales_raw
            ]
            if len(controlnet_conditioning_scales) != controlnet_count:
                raise ValueError(
                    "controlnet_conditioning_scales length must match controlnet_models length."
                )
        else:
            controlnet_conditioning_scales = [
                float(inputs.get("controlnet_conditioning_scale", 1.0))
            ] * controlnet_count
        for scale in controlnet_conditioning_scales:
            if scale < 0.0 or scale > 2.0:
                raise ValueError("controlnet conditioning scales must be within [0, 2].")

        controlnet_preprocessor_id_raw = inputs.get("controlnet_preprocessor_id")
        controlnet_preprocessor_id = (
            str(controlnet_preprocessor_id_raw)
            if controlnet_preprocessor_id_raw is not None
            else None
        )
        controlnet_preprocessor_ids_raw = inputs.get("controlnet_preprocessor_ids")
        if (
            controlnet_preprocessor_ids_raw is not None
            and not isinstance(controlnet_preprocessor_ids_raw, list)
        ):
            raise ValueError("controlnet_preprocessor_ids must be a list of preprocessor ids")
        if controlnet_preprocessor_ids_raw:
            controlnet_preprocessor_ids = [
                str(item) if item is not None else None
                for item in controlnet_preprocessor_ids_raw
            ]
            if len(controlnet_preprocessor_ids) != controlnet_count:
                raise ValueError(
                    "controlnet_preprocessor_ids length must match controlnet_models length."
                )
        else:
            controlnet_preprocessor_ids = [controlnet_preprocessor_id] * controlnet_count

        controlnet_compat_mode = str(inputs.get("controlnet_compat_mode") or "warn").lower()
        warnings: list[str] = []
        if controlnet_count > 1:
            perf_warning = (
                f"Using {controlnet_count} ControlNet models may significantly increase VRAM use "
                "and generation latency."
            )
            warnings.append(perf_warning)
            logger.warning(perf_warning)

        if controlnet_compat_mode != "off":
            for idx, (model_id, preprocessor_id) in enumerate(
                zip(controlnet_models, controlnet_preprocessor_ids)
            ):
                if not preprocessor_id:
                    continue
                entry = _CONTROLNET_PREPROCESSOR_REGISTRY_BY_ID.get(preprocessor_id)
                if entry is None:
                    message = (
                        f"Unknown controlnet_preprocessor_id '{preprocessor_id}' at index {idx}. "
                        "Compatibility check could not be applied."
                    )
                    if controlnet_compat_mode == "error":
                        raise ValueError(message)
                    warnings.append(message)
                    logger.warning(message)
                    continue
                compatible_models = set(entry.recommended_sd15_control_models) | set(
                    entry.legacy_aliases
                )
                if compatible_models and model_id not in compatible_models:
                    message = (
                        "ControlNet model/preprocessor pairing mismatch: "
                        f"preprocessor '{preprocessor_id}' with model '{model_id}' at index {idx}. "
                        f"Recommended models: {', '.join(entry.recommended_sd15_control_models)}."
                    )
                    if controlnet_compat_mode == "error":
                        raise ValueError(message)
                    warnings.append(message)
                    logger.warning(message)

        controlnet_model_arg: str | list[str]
        control_image_arg: Image.Image | list[Image.Image]
        controlnet_conditioning_scale_arg: float | list[float]
        if controlnet_count == 1:
            controlnet_model_arg = controlnet_models[0]
            control_image_arg = control_images[0]
            controlnet_conditioning_scale_arg = controlnet_conditioning_scales[0]
        else:
            controlnet_model_arg = controlnet_models
            control_image_arg = control_images
            controlnet_conditioning_scale_arg = controlnet_conditioning_scales

        generation_params = {
            "initial_image": initial_image,
            "mask_image": mask_image,
            "prompt": str(inputs["prompt"]),
            "negative_prompt": str(inputs.get("negative_prompt") or ""),
            "steps": steps,
            "cfg": cfg,
            "seed": inputs.get("seed"),
            "scheduler": scheduler,
            "model": inputs.get("model"),
            "num_images": int(inputs.get("num_images") or 1),
            "strength": strength,
            "padding_mask_crop": padding_mask_crop,
            "clip_skip": int(inputs.get("clip_skip") or 1),
            "control_image": control_image_arg,
            "controlnet_model": controlnet_model_arg,
            "controlnet_conditioning_scale": controlnet_conditioning_scale_arg,
            "controlnet_guess_mode": bool(inputs.get("controlnet_guess_mode", False)),
            "control_guidance_start": control_guidance_start,
            "control_guidance_end": control_guidance_end,
            "lora_adapters": lora_adapters,
            "batch_id": batch_id,
        }
        filenames = generate_images_inpaint_controlnet(generation_params)
        result: dict[str, Any] = {
            "batch_id": batch_id,
            "images": [f"/outputs/{name}" for name in filenames],
        }
        if warnings:
            result["warnings"] = warnings
        return result

    generation_params = {
        "initial_image": initial_image,
        "mask_image": mask_image,
        "prompt": str(inputs["prompt"]),
        "negative_prompt": str(inputs.get("negative_prompt") or ""),
        "steps": steps,
        "cfg": cfg,
        "seed": inputs.get("seed"),
        "scheduler": scheduler,
        "model": inputs.get("model"),
        "num_images": int(inputs.get("num_images") or 1),
        "strength": strength,
        "padding_mask_crop": padding_mask_crop,
        "clip_skip": int(inputs.get("clip_skip") or 1),
        "lora_adapters": lora_adapters,
        "lcm_enabled": lcm_enabled,
        "lcm_lora_model": _LCM_LORA_MODEL_ID if lcm_enabled else None,
        "batch_id": batch_id,
    }
    ip_adapter_settings = _normalized_ip_adapter_settings(inputs, _open_image_ref)
    if ip_adapter_settings is not None:
        generation_params.update(ip_adapter_settings)
    filenames = generate_images_inpaint(generation_params)
    return {"batch_id": batch_id, "images": [f"/outputs/{name}" for name in filenames]}

def run_sd15_controlnet_text2img(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    _normalize_sd15_controlnet_contract_inputs = deps["normalize_sd15_controlnet_contract_inputs"]
    _normalized_lora_adapters = deps["normalized_lora_adapters"]
    _open_image_ref = deps["open_image_ref"]
    _DEFAULT_SD15_CONTROLNET_MODEL = deps["default_sd15_controlnet_model"]
    _MAX_CONTROLNET_MODELS = deps["max_controlnet_models"]
    _CONTROLNET_PREPROCESSOR_REGISTRY_BY_ID = deps["controlnet_preprocessor_registry_by_id"]
    logger = deps["logger"]
    make_batch_id = deps["make_batch_id"]
    generate_images_controlnet = deps["generate_images_controlnet"]

    inputs = _normalize_sd15_controlnet_contract_inputs(inputs)
    lora_adapters = _normalized_lora_adapters(inputs)
    if inputs.get("control_image") is None and not inputs.get("control_images"):
        raise ValueError("control_image is required for sd15.controlnet.text2img")
    width = int(inputs.get("width") or 512)
    height = int(inputs.get("height") or 512)
    control_image_single = _open_image_ref(inputs["control_image"]).convert("RGB")
    control_image_single = control_image_single.resize((width, height))
    controlnet_model_single = str(
        inputs.get("controlnet_model") or _DEFAULT_SD15_CONTROLNET_MODEL
    )
    
    controlnet_models_raw = inputs.get("controlnet_models")
    if controlnet_models_raw is not None and not isinstance(controlnet_models_raw, list):
        raise ValueError("controlnet_models must be a list of model ids")
    controlnet_models: list[str] = (
        [str(item) for item in controlnet_models_raw] if controlnet_models_raw else []
    )
    if not controlnet_models:
        controlnet_models = [controlnet_model_single]

    control_images_raw = inputs.get("control_images")
    if control_images_raw is not None and not isinstance(control_images_raw, list):
        raise ValueError("control_images must be a list of image references")
    control_images: list[Image.Image] = []
    if control_images_raw:
        control_images = [
            _open_image_ref(image_ref).convert("RGB").resize((width, height))
            for image_ref in control_images_raw
        ]
    if not control_images:
        control_images = [control_image_single]

    if len(controlnet_models) > _MAX_CONTROLNET_MODELS:
        raise ValueError(
            f"At most {_MAX_CONTROLNET_MODELS} ControlNet models are supported per task."
        )

    if len(controlnet_models) != len(control_images):
        if len(controlnet_models) == 1 and len(control_images) > 1:
            controlnet_models = controlnet_models * len(control_images)
        elif len(control_images) == 1 and len(controlnet_models) > 1:
            control_images = control_images * len(controlnet_models)
        else:
            raise ValueError(
                "controlnet_models and control_images must have the same length."
            )

    controlnet_count = len(controlnet_models)
    control_guidance_starts, control_guidance_ends = _resolve_control_guidance_timings(
        inputs,
        controlnet_count=controlnet_count,
    )

    controlnet_conditioning_scales_raw = inputs.get("controlnet_conditioning_scales")
    if (
        controlnet_conditioning_scales_raw is not None
        and not isinstance(controlnet_conditioning_scales_raw, list)
    ):
        raise ValueError("controlnet_conditioning_scales must be a list of numbers")
    if controlnet_conditioning_scales_raw:
        controlnet_conditioning_scales = [
            float(item) for item in controlnet_conditioning_scales_raw
        ]
        if len(controlnet_conditioning_scales) != controlnet_count:
            raise ValueError(
                "controlnet_conditioning_scales length must match controlnet_models length."
            )
    else:
        controlnet_conditioning_scales = [
            float(inputs.get("controlnet_conditioning_scale", 1.0))
        ] * controlnet_count
    for scale in controlnet_conditioning_scales:
        if scale < 0.0 or scale > 2.0:
            raise ValueError("controlnet conditioning scales must be within [0, 2].")

    controlnet_preprocessor_id_raw = inputs.get("controlnet_preprocessor_id")
    controlnet_preprocessor_id = (
        str(controlnet_preprocessor_id_raw)
        if controlnet_preprocessor_id_raw is not None
        else None
    )
    controlnet_preprocessor_ids_raw = inputs.get("controlnet_preprocessor_ids")
    if (
        controlnet_preprocessor_ids_raw is not None
        and not isinstance(controlnet_preprocessor_ids_raw, list)
    ):
        raise ValueError("controlnet_preprocessor_ids must be a list of preprocessor ids")
    if controlnet_preprocessor_ids_raw:
        controlnet_preprocessor_ids = [
            str(item) if item is not None else None for item in controlnet_preprocessor_ids_raw
        ]
        if len(controlnet_preprocessor_ids) != controlnet_count:
            raise ValueError(
                "controlnet_preprocessor_ids length must match controlnet_models length."
            )
    else:
        controlnet_preprocessor_ids = [controlnet_preprocessor_id] * controlnet_count

    controlnet_compat_mode = str(inputs.get("controlnet_compat_mode") or "warn").lower()
    warnings: list[str] = []
    if controlnet_count > 1:
        perf_warning = (
            f"Using {controlnet_count} ControlNet models may significantly increase VRAM use "
            "and generation latency."
        )
        warnings.append(perf_warning)
        logger.warning(perf_warning)

    if controlnet_compat_mode != "off":
        for idx, (model_id, preprocessor_id) in enumerate(
            zip(controlnet_models, controlnet_preprocessor_ids)
        ):
            if not preprocessor_id:
                continue
            entry = _CONTROLNET_PREPROCESSOR_REGISTRY_BY_ID.get(preprocessor_id)
            if entry is None:
                message = (
                    f"Unknown controlnet_preprocessor_id '{preprocessor_id}' at index {idx}. "
                    "Compatibility check could not be applied."
                )
                if controlnet_compat_mode == "error":
                    raise ValueError(message)
                warnings.append(message)
                logger.warning(message)
                continue
            compatible_models = set(entry.recommended_sd15_control_models) | set(
                entry.legacy_aliases
            )
            if compatible_models and model_id not in compatible_models:
                message = (
                    "ControlNet model/preprocessor pairing mismatch: "
                    f"preprocessor '{preprocessor_id}' with model '{model_id}' at index {idx}. "
                    f"Recommended models: {', '.join(entry.recommended_sd15_control_models)}."
                )
                if controlnet_compat_mode == "error":
                    raise ValueError(message)
                warnings.append(message)
                logger.warning(message)
    batch_id = str(inputs.get("batch_id") or make_batch_id())

    controlnet_model_arg: str | list[str]
    control_image_arg: Image.Image | list[Image.Image]
    controlnet_conditioning_scale_arg: float | list[float]
    control_guidance_start_arg: float | list[float]
    control_guidance_end_arg: float | list[float]
    if controlnet_count == 1:
        controlnet_model_arg = controlnet_models[0]
        control_image_arg = control_images[0]
        controlnet_conditioning_scale_arg = controlnet_conditioning_scales[0]
        control_guidance_start_arg = control_guidance_starts[0]
        control_guidance_end_arg = control_guidance_ends[0]
    else:
        controlnet_model_arg = controlnet_models
        control_image_arg = control_images
        controlnet_conditioning_scale_arg = controlnet_conditioning_scales
        control_guidance_start_arg = control_guidance_starts
        control_guidance_end_arg = control_guidance_ends

    generation_params = {
        "prompt": str(inputs["prompt"]),
        "negative_prompt": str(inputs.get("negative_prompt") or ""),
        "steps": int(inputs.get("steps") or 20),
        "cfg": float(inputs.get("cfg") or 7.5),
        "width": width,
        "height": height,
        "seed": inputs.get("seed"),
        "scheduler": str(inputs.get("scheduler") or "euler"),
        "model": inputs.get("model"),
        "num_images": int(inputs.get("num_images") or 1),
        "clip_skip": int(inputs.get("clip_skip") or 1),
        "control_image": control_image_arg,
        "controlnet_model": controlnet_model_arg,
        "controlnet_conditioning_scale": controlnet_conditioning_scale_arg,
        "controlnet_guess_mode": bool(inputs.get("controlnet_guess_mode", False)),
        "control_guidance_start": control_guidance_start_arg,
        "control_guidance_end": control_guidance_end_arg,
        "lora_adapters": lora_adapters,
        "batch_id": batch_id,
    }
    filenames = generate_images_controlnet(generation_params)
    result: dict[str, Any] = {
        "batch_id": batch_id,
        "images": [f"/outputs/{name}" for name in filenames],
    }
    if warnings:
        result["warnings"] = warnings
    return result

def run_sd15_hires_fix(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    _open_image_ref = deps["open_image_ref"]
    make_batch_id = deps["make_batch_id"]
    get_batch_output_dir = deps["get_batch_output_dir"]
    OUTPUT_DIR = deps["output_dir"]
    _normalized_lora_adapters = deps["normalized_lora_adapters"]
    run_sd15_hires_fix = deps["run_sd15_hires_fix"]

    images_in = inputs["images"]
    if not isinstance(images_in, list):
        raise ValueError("images must be a list")
    images = [_open_image_ref(item).convert("RGB") for item in images_in]

    batch_id = str(inputs.get("batch_id") or make_batch_id())
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    lora_adapters = _normalized_lora_adapters(inputs)
    relpaths = run_sd15_hires_fix(
        images=images,
        prompt=str(inputs["prompt"]),
        negative_prompt=str(inputs.get("negative_prompt") or ""),
        steps=int(inputs.get("steps") or 20),
        cfg=float(inputs.get("cfg") or 7.5),
        seed=inputs.get("seed"),
        scheduler=str(inputs.get("scheduler") or "euler"),
        model=inputs.get("model"),
        clip_skip=int(inputs.get("clip_skip") or 1),
        hires_scale=float(inputs.get("hires_scale") or 1.0),
        hires_strength=float(inputs.get("hires_strength") or 0.35),
        lora_adapters=lora_adapters,
        weighting_policy=str(inputs.get("weighting_policy") or "diffusers-like"),
        output_dir=batch_output_dir,
        batch_id=batch_id,
    )
    return {"batch_id": batch_id, "images": [f"/outputs/{p}" for p in relpaths]}
