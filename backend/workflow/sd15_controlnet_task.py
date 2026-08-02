"""SD1.5 ControlNet workflow task adapter."""

from backend.workflow.sd15_shared import *

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

