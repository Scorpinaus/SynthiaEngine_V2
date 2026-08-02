"""SD1.5 image-to-image workflow task adapter."""

from backend.workflow.sd15_shared import *

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

