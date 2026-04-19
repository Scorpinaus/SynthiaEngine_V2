from __future__ import annotations

from typing import Any

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


def run_sdxl_ip_adapter_encode_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
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
        raise ValueError("sdxl.ip_adapter.encode must return an object")
    return result


def run_sdxl_text2img_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    _open_image_ref = deps["open_image_ref"]
    generate_text2img = deps["generate_text2img"]
    ip_adapter_settings = _normalized_ip_adapter_settings(
        inputs,
        _open_image_ref,
        allow_image_embeds=True,
    )

    pipeline_params: dict[str, Any] = {
        "prompt": str(inputs.get("prompt") or ""),
        "negative_prompt": str(inputs.get("negative_prompt") or ""),
        "steps": int(inputs.get("steps") or 20),
        "guidance_scale": float(inputs.get("guidance_scale") or inputs.get("cfg") or 7.5),
        "width": int(inputs.get("width") or 1024),
        "height": int(inputs.get("height") or 1024),
        "seed": inputs.get("seed"),
        "model": inputs.get("model"),
        "num_images": int(inputs.get("num_images") or 1),
        "clip_skip": int(inputs.get("clip_skip") or 1),
        "scheduler": str(inputs.get("scheduler") or "euler"),
        "lora_adapters": inputs.get("lora_adapters"),
    }
    if ip_adapter_settings is not None:
        pipeline_params.update(ip_adapter_settings)

    result = generate_text2img(pipeline_params)
    if not isinstance(result, dict):
        raise ValueError("sdxl.text2img must return an object")
    return result

def run_sdxl_controlnet_text2img_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    _open_image_ref = deps["open_image_ref"]
    _DEFAULT_SDXL_CONTROLNET_MODEL = deps["default_sdxl_controlnet_model"]
    _MAX_CONTROLNET_MODELS = deps["max_controlnet_models"]
    _CONTROLNET_PREPROCESSOR_REGISTRY_BY_ID = deps["controlnet_preprocessor_registry_by_id"]
    logger = deps["logger"]
    generate_controlnet_text2img = deps["generate_controlnet_text2img"]


    width = int(inputs.get("width") or 1024)
    height = int(inputs.get("height") or 1024)
    control_image_single = _open_image_ref(inputs["control_image"]).convert("RGB")
    control_image_single = control_image_single.resize((width, height))
    control_guidance_start = float(inputs.get("control_guidance_start", 0.0))
    control_guidance_end = float(inputs.get("control_guidance_end", 1.0))
    if control_guidance_start > control_guidance_end:
        raise ValueError("control_guidance_start must be <= control_guidance_end")

    controlnet_model_single = str(
        inputs.get("controlnet_model") or _DEFAULT_SDXL_CONTROLNET_MODEL
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
            raise ValueError("controlnet_models and control_images must have the same length.")

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
        for idx, preprocessor_id in enumerate(controlnet_preprocessor_ids):
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

    pipeline_params: dict[str, Any] = {
        "prompt": str(inputs.get("prompt") or ""),
        "negative_prompt": str(inputs.get("negative_prompt") or ""),
        "steps": int(inputs.get("steps") or 20),
        "guidance_scale": float(inputs.get("guidance_scale") or inputs.get("cfg") or 7.5),
        "width": width,
        "height": height,
        "seed": inputs.get("seed"),
        "scheduler": str(inputs.get("scheduler") or "euler"),
        "model": inputs.get("model"),
        "num_images": int(inputs.get("num_images") or 1),
        "clip_skip": int(inputs.get("clip_skip") or 1),
        "controlnet_model": controlnet_model_arg,
        "control_image": control_image_arg,
        "controlnet_conditioning_scale": controlnet_conditioning_scale_arg,
        "controlnet_guess_mode": bool(inputs.get("controlnet_guess_mode", False)),
        "control_guidance_start": control_guidance_start,
        "control_guidance_end": control_guidance_end,
    }

    result = generate_controlnet_text2img(pipeline_params)
    if not isinstance(result, dict):
        raise ValueError("sdxl.controlnet.text2img must return an object")
    if warnings:
        result["warnings"] = warnings
    return result

def run_sdxl_img2img_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    _open_image_ref = deps["open_image_ref"]
    _remap_img2img_strength = deps["remap_img2img_strength"]
    _DEFAULT_SDXL_CONTROLNET_MODEL = deps["default_sdxl_controlnet_model"]
    _MAX_CONTROLNET_MODELS = deps["max_controlnet_models"]
    _CONTROLNET_PREPROCESSOR_REGISTRY_BY_ID = deps["controlnet_preprocessor_registry_by_id"]
    logger = deps["logger"]
    generate_img2img = deps["generate_img2img"]
    generate_img2img_controlnet = deps["generate_img2img_controlnet"]

    initial_image = _open_image_ref(inputs["initial_image"]).convert("RGB")
    width = int(inputs.get("width") or 1024)
    height = int(inputs.get("height") or 1024)
    initial_image = initial_image.resize((width, height))

    strength = float(inputs.get("strength") or 0.75)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    strength = _remap_img2img_strength(strength)

    ip_adapter_raw = inputs.get("ip_adapter")
    if ip_adapter_raw is not None and not isinstance(ip_adapter_raw, dict):
        raise ValueError("`ip_adapter` must be an object.")
    ip_adapter_enabled = (
        isinstance(ip_adapter_raw, dict) and bool(ip_adapter_raw.get("enabled", False))
    )
    ip_adapter_settings = _normalized_ip_adapter_settings(inputs, _open_image_ref)

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
            raise ValueError("sdxl.img2img IP-Adapter cannot be combined with ControlNet.")
        control_image_input = inputs.get("control_image")
        control_images_raw = inputs.get("control_images")
        if control_images_raw is not None and not isinstance(control_images_raw, list):
            raise ValueError("control_images must be a list of image references")
        if control_image_input is None and not control_images_raw:
            raise ValueError("control_image is required when using ControlNet in sdxl.img2img")

        control_guidance_start = float(inputs.get("control_guidance_start", 0.0))
        control_guidance_end = float(inputs.get("control_guidance_end", 1.0))
        if control_guidance_start > control_guidance_end:
            raise ValueError("control_guidance_start must be <= control_guidance_end")

        controlnet_model_single = str(
            inputs.get("controlnet_model") or _DEFAULT_SDXL_CONTROLNET_MODEL
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
            for idx, preprocessor_id in enumerate(controlnet_preprocessor_ids):
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

        result = generate_img2img_controlnet(
            {
                "initial_image": initial_image,
                "strength": strength,
                "prompt": str(inputs["prompt"]),
                "negative_prompt": str(inputs.get("negative_prompt") or ""),
                "steps": int(inputs.get("steps") or 20),
                "guidance_scale": float(inputs.get("guidance_scale") or inputs.get("cfg") or 7.5),
                "width": width,
                "height": height,
                "seed": inputs.get("seed"),
                "scheduler": str(inputs.get("scheduler") or "euler"),
                "model": inputs.get("model"),
                "num_images": int(inputs.get("num_images") or 1),
                "clip_skip": int(inputs.get("clip_skip") or 1),
                "lora_adapters": inputs.get("lora_adapters"),
                "controlnet_model": controlnet_model_arg,
                "control_image": control_image_arg,
                "controlnet_conditioning_scale": controlnet_conditioning_scale_arg,
                "controlnet_guess_mode": bool(inputs.get("controlnet_guess_mode", False)),
                "control_guidance_start": control_guidance_start,
                "control_guidance_end": control_guidance_end,
            }
        )
        if not isinstance(result, dict):
            raise ValueError("sdxl.img2img must return an object")
        if warnings:
            result["warnings"] = warnings
        return result

    pipeline_params: dict[str, Any] = {
        "initial_image": initial_image,
        "strength": strength,
        "prompt": str(inputs["prompt"]),
        "negative_prompt": str(inputs.get("negative_prompt") or ""),
        "steps": int(inputs.get("steps") or 20),
        "guidance_scale": float(inputs.get("guidance_scale") or inputs.get("cfg") or 7.5),
        "width": width,
        "height": height,
        "seed": inputs.get("seed"),
        "scheduler": str(inputs.get("scheduler") or "euler"),
        "model": inputs.get("model"),
        "num_images": int(inputs.get("num_images") or 1),
        "clip_skip": int(inputs.get("clip_skip") or 1),
        "lora_adapters": inputs.get("lora_adapters"),
    }
    if ip_adapter_settings is not None:
        pipeline_params.update(ip_adapter_settings)

    result = generate_img2img(pipeline_params)
    if not isinstance(result, dict):
        raise ValueError("sdxl.img2img must return an object")
    return result

def run_sdxl_inpaint_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    _open_image_ref = deps["open_image_ref"]
    _remap_img2img_strength = deps["remap_img2img_strength"]
    _DEFAULT_SDXL_CONTROLNET_MODEL = deps["default_sdxl_controlnet_model"]
    _MAX_CONTROLNET_MODELS = deps["max_controlnet_models"]
    _CONTROLNET_PREPROCESSOR_REGISTRY_BY_ID = deps["controlnet_preprocessor_registry_by_id"]
    logger = deps["logger"]
    generate_inpaint = deps["generate_inpaint"]
    generate_inpaint_controlnet = deps["generate_inpaint_controlnet"]


    initial_image = _open_image_ref(inputs["initial_image"]).convert("RGB")
    mask_image = _open_image_ref(inputs["mask_image"]).convert("L")
    if mask_image.size != initial_image.size:
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
            raise ValueError(
                "mask_image dimensions must match initial_image dimensions when using ControlNet in sdxl.inpaint"
            )
        mask_image = mask_image.resize(initial_image.size, resample=Image.NEAREST)

    strength = float(inputs.get("strength") or 0.5)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    strength = _remap_img2img_strength(strength)
    padding_mask_crop_input = inputs.get("padding_mask_crop")
    padding_mask_crop = 32 if padding_mask_crop_input is None else int(padding_mask_crop_input)

    ip_adapter_raw = inputs.get("ip_adapter")
    if ip_adapter_raw is not None and not isinstance(ip_adapter_raw, dict):
        raise ValueError("`ip_adapter` must be an object.")
    ip_adapter_enabled = (
        isinstance(ip_adapter_raw, dict) and bool(ip_adapter_raw.get("enabled", False))
    )
    ip_adapter_settings = _normalized_ip_adapter_settings(inputs, _open_image_ref)

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
            raise ValueError("sdxl.inpaint IP-Adapter cannot be combined with ControlNet.")
        control_image_input = inputs.get("control_image")
        control_images_raw = inputs.get("control_images")
        if control_images_raw is not None and not isinstance(control_images_raw, list):
            raise ValueError("control_images must be a list of image references")
        if control_image_input is None and not control_images_raw:
            raise ValueError("control_image is required when using ControlNet in sdxl.inpaint")

        control_guidance_start = float(inputs.get("control_guidance_start", 0.0))
        control_guidance_end = float(inputs.get("control_guidance_end", 1.0))
        if control_guidance_start > control_guidance_end:
            raise ValueError("control_guidance_start must be <= control_guidance_end")

        controlnet_model_single = str(
            inputs.get("controlnet_model") or _DEFAULT_SDXL_CONTROLNET_MODEL
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
            first_control_image = _open_image_ref(control_image_input).convert("RGB")
            if first_control_image.size != initial_image.size:
                raise ValueError(
                    "control_image dimensions must match initial_image dimensions in sdxl.inpaint"
                )
            control_images.append(first_control_image)
        if control_images_raw:
            for idx, image_ref in enumerate(control_images_raw):
                image = _open_image_ref(image_ref).convert("RGB")
                if image.size != initial_image.size:
                    raise ValueError(
                        f"control_images[{idx}] dimensions must match initial_image dimensions in sdxl.inpaint"
                    )
                control_images.append(image)

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
            for idx, preprocessor_id in enumerate(controlnet_preprocessor_ids):
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

        result = generate_inpaint_controlnet(
            {
                "initial_image": initial_image,
                "mask_image": mask_image,
                "strength": strength,
                "prompt": str(inputs["prompt"]),
                "negative_prompt": str(inputs.get("negative_prompt") or ""),
                "steps": int(inputs.get("steps") or 20),
                "guidance_scale": float(inputs.get("guidance_scale") or inputs.get("cfg") or 7.5),
                "seed": inputs.get("seed"),
                "scheduler": str(inputs.get("scheduler") or "euler"),
                "model": inputs.get("model"),
                "num_images": int(inputs.get("num_images") or 1),
                "padding_mask_crop": padding_mask_crop,
                "clip_skip": int(inputs.get("clip_skip") or 1),
                "lora_adapters": inputs.get("lora_adapters"),
                "controlnet_model": controlnet_model_arg,
                "control_image": control_image_arg,
                "controlnet_conditioning_scale": controlnet_conditioning_scale_arg,
                "controlnet_guess_mode": bool(inputs.get("controlnet_guess_mode", False)),
                "control_guidance_start": control_guidance_start,
                "control_guidance_end": control_guidance_end,
            }
        )
        if not isinstance(result, dict):
            raise ValueError("sdxl.inpaint must return an object")
        if warnings:
            result["warnings"] = warnings
        return result

    pipeline_params: dict[str, Any] = {
        "initial_image": initial_image,
        "mask_image": mask_image,
        "strength": strength,
        "prompt": str(inputs["prompt"]),
        "negative_prompt": str(inputs.get("negative_prompt") or ""),
        "steps": int(inputs.get("steps") or 20),
        "guidance_scale": float(inputs.get("guidance_scale") or inputs.get("cfg") or 7.5),
        "seed": inputs.get("seed"),
        "scheduler": str(inputs.get("scheduler") or "euler"),
        "model": inputs.get("model"),
        "num_images": int(inputs.get("num_images") or 1),
        "padding_mask_crop": padding_mask_crop,
        "clip_skip": int(inputs.get("clip_skip") or 1),
        "lora_adapters": inputs.get("lora_adapters"),
    }
    if ip_adapter_settings is not None:
        pipeline_params.update(ip_adapter_settings)

    result = generate_inpaint(pipeline_params)
    if not isinstance(result, dict):
        raise ValueError("sdxl.inpaint must return an object")
    return result
