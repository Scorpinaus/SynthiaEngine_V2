"""SD1.5 inpaint generation operations, with optional ControlNet."""

from backend.sd15.runtime_common import *
from backend.sd15.adapters import (
    _apply_lcm_lora,
    _apply_lora_adapters,
    _build_ip_adapter_kwargs,
    _cleanup_lora_adapters,
    _hide_image_encoder_while_using_ip_adapter_embeds,
)
from backend.sd15.loaders import load_controlnet_inpaint_pipeline, load_inpaint_pipeline
from backend.sd15.preparation import (
    _build_sd15_prompt_call_kwargs,
    _make_inpaint_controlnet_condition,
    _resize_control_image_to_target,
)

@torch.inference_mode()
def generate_images_inpaint_in_process(params: dict[str, object],) -> list[str]:
    """
    Generate SD1.5 inpaint outputs from an initial image and mask.

    This function writes PNG files to the batch directory, stores generation
    settings in PNG metadata, and optionally captures layer-usage diagnostics.

    Returns:
        Output PNG paths relative to ``OUTPUT_DIR``.
    """
    # Normalize all inpaint inputs in one place for easier maintenance and tracing.
    initial_image = params["initial_image"]
    mask_image = params["mask_image"]
    prompt = str(params["prompt"])
    negative_prompt = str(params.get("negative_prompt") or "")
    seed = params.get("seed")
    scheduler = str(params.get("scheduler") or "euler")
    lcm_enabled = bool(params.get("lcm_enabled", False)) or scheduler.lower() == "lcm"
    scheduler = "lcm" if lcm_enabled else scheduler
    if lcm_enabled:
        steps = int(
            params["steps"]
            if "steps" in params and params.get("steps") is not None
            else _LCM_DEFAULT_STEPS
        )
        cfg = float(
            params["cfg"]
            if "cfg" in params and params.get("cfg") is not None
            else _LCM_DEFAULT_CFG
        )
    else:
        steps = int(params.get("steps") or 20)
        cfg = float(params.get("cfg") or 7.5)
    model = params.get("model")
    num_images = int(params.get("num_images") or 1)
    strength = float(params.get("strength") or 0.5)
    padding_mask_crop = int(params.get("padding_mask_crop") or 32)
    clip_skip = int(params.get("clip_skip") or 1)
    weighting_policy = str(params.get("weighting_policy") or "diffusers-like")
    lora_adapters = params.get("lora_adapters")
    ip_adapter_image = cast(Image.Image | None, params.get("ip_adapter_image"))
    ip_adapter_image_embeds_ref = params.get("ip_adapter_image_embeds_ref")
    if ip_adapter_image is not None and ip_adapter_image_embeds_ref is not None:
        raise ValueError("Provide either ip_adapter_image or ip_adapter_image_embeds_ref, not both.")
    ip_adapter_enabled = ip_adapter_image is not None or ip_adapter_image_embeds_ref is not None
    ip_adapter_mask_image = cast(Image.Image | None, params.get("ip_adapter_mask_image"))
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
    batch_id = params.get("batch_id")

    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = seed
    if batch_id is None:
        batch_id = make_batch_id()

    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    adapter_names: list[str] = []
    pipe = load_inpaint_pipeline(model)
    try:
        pipe.scheduler = create_scheduler(scheduler, pipe)
        width, height = initial_image.size
        logger.info(
            "Inpaint: model=%s seed=%s scheduler=%s steps=%s cfg=%s size=%sx%s num_images=%s strength=%s, padding_mask_crop=%s",
            model, base_seed, scheduler, steps, cfg,
            width, height, num_images, strength, padding_mask_crop
        )
        if ip_adapter_enabled:
            if ip_adapter_image_embeds_ref is not None:
                embeds_payload = load_ip_adapter_embeds_artifact(ip_adapter_image_embeds_ref)
                validate_ip_adapter_embeds_metadata(
                    embeds_payload,
                    expected_model=ip_adapter_model,
                    expected_subfolder=ip_adapter_subfolder,
                    expected_weight_name=ip_adapter_weight_name,
                    do_classifier_free_guidance=cfg > 1.0,
                    expected_family="SD15",
                )
                ip_adapter_image_embeds = embeds_payload["embeds"]
                IpAdapterManager.load(
                    pipe,
                    model=ip_adapter_model,
                    subfolder=ip_adapter_subfolder,
                    weight_name=ip_adapter_weight_name,
                    scale=ip_adapter_scale,
                    family="SD1.5",
                    image_encoder_folder=None,
                )
            else:
                IpAdapterManager.load(
                    pipe,
                    model=ip_adapter_model,
                    subfolder=ip_adapter_subfolder,
                    weight_name=ip_adapter_weight_name,
                    scale=ip_adapter_scale,
                    family="SD1.5",
                )
                ip_adapter_image_embeds = IpAdapterManager.prepare_image_embeds(
                    pipe,
                    ip_adapter_image,
                    do_classifier_free_guidance=cfg > 1.0,
                )
            ip_adapter_masks = (
                IpAdapterManager.prepare_masks(
                    ip_adapter_mask_image,
                    height=initial_image.height,
                    width=initial_image.width,
                )
                if ip_adapter_mask_image is not None
                else None
            )
        else:
            ip_adapter_image_embeds = None
            ip_adapter_masks = None

        filenames = []
        if lcm_enabled:
            lcm_adapter_name = _apply_lcm_lora(pipe)
            adapter_names, lora_coverage = apply_lora_adapters_with_validation(
                pipe,
                lora_adapters,
                expected_family="sd15",
                validate=True,
                preloaded_adapters=[(lcm_adapter_name, 1.0)],
            )
            report_path = write_lora_coverage_report(batch_output_dir, batch_id, lora_coverage)
            if report_path is not None:
                logger.info("LoRA coverage report saved to %s", report_path)
        else:
            adapter_names = _apply_lora_adapters(pipe, lora_adapters)
        metadata_base = {
            "mode": "inpaint",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg": cfg,
            "width": width,
            "height": height,
            "scheduler": scheduler,
            "model": model,
            "strength": strength,
            "padding_mask_crop": padding_mask_crop,
            "clip_skip": clip_skip,
            "lcm_enabled": lcm_enabled,
            "lcm_lora_model": _LCM_LORA_MODEL_ID if lcm_enabled else None,
            "batch_id": batch_id,
        }
        ip_adapter_kwargs = _build_ip_adapter_kwargs(
            enabled=ip_adapter_enabled,
            image_embeds=ip_adapter_image_embeds,
            masks=ip_adapter_masks,
        )

        for i in range(num_images):
            # Offset seed per image so batches are deterministic and distinct.
            current_seed = base_seed + i
            generator = torch.Generator(device="cuda").manual_seed(current_seed)

            if config.PIPELINE_LAYER_LOGGING_ENABLED and i == 0:
                arch_layers = collect_pipeline_layers(
                    pipe,
                    leaf_only=config.PIPELINE_LAYER_LOGGING_LEAF_ONLY,
                )
                with capture_runtime_used_layers(
                    pipe,
                    leaf_only=config.PIPELINE_LAYER_LOGGING_LEAF_ONLY,
                ) as (used_layer_names, name_to_type, name_to_inputs, name_to_calls):
                    prompt_kwargs = _build_sd15_prompt_call_kwargs(
                        pipe,
                        prompt,
                        negative_prompt,
                        clip_skip=clip_skip,
                        weighting_policy=weighting_policy,
                    )
                    with _hide_image_encoder_while_using_ip_adapter_embeds(
                        pipe,
                        enabled=ip_adapter_image_embeds is not None,
                    ):
                        image = pipe(
                            **prompt_kwargs,
                            image=initial_image,
                            mask_image=mask_image,
                            num_inference_steps=steps,
                            guidance_scale=cfg,
                            generator=generator,
                            strength=strength,
                            padding_mask_crop=padding_mask_crop,
                            **ip_adapter_kwargs,
                        ).images[0]
                append_layers_report(
                    output_dir=batch_output_dir,
                    batch_id=batch_id,
                    label="sd15_inpaint",
                    pipeline_name=pipe.__class__.__name__,
                    architecture_layers=arch_layers,
                    runtime_used_layer_names=used_layer_names,
                    runtime_name_to_type=name_to_type,
                    runtime_name_to_input_summary=(name_to_inputs if config.PIPELINE_LAYER_LOGGING_CAPTURE_INPUTS else None),
                    runtime_name_to_call_count=(name_to_calls if config.PIPELINE_LAYER_LOGGING_CAPTURE_INPUTS else None),
                )
            else:
                prompt_kwargs = _build_sd15_prompt_call_kwargs(
                    pipe,
                    prompt,
                    negative_prompt,
                    clip_skip=clip_skip,
                    weighting_policy=weighting_policy,
                )
                with _hide_image_encoder_while_using_ip_adapter_embeds(
                    pipe,
                    enabled=ip_adapter_image_embeds is not None,
                ):
                    image = pipe(
                        **prompt_kwargs,
                        image=initial_image,
                        mask_image=mask_image,
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        generator=generator,
                        strength=strength,
                        padding_mask_crop=padding_mask_crop,
                        **ip_adapter_kwargs,
                    ).images[0]

            filename = batch_output_dir / f"{batch_id}_{current_seed}.png"
            metadata = {
                **metadata_base,
                "seed": current_seed,
                "ip_adapter_enabled": ip_adapter_enabled,
                "ip_adapter_model": ip_adapter_model if ip_adapter_enabled else None,
                "ip_adapter_subfolder": ip_adapter_subfolder if ip_adapter_enabled else None,
                "ip_adapter_weight_name": ip_adapter_weight_name if ip_adapter_enabled else None,
                "ip_adapter_scale": ip_adapter_scale if ip_adapter_enabled else None,
            }
            pnginfo = build_png_metadata(metadata)
            image.save(filename, pnginfo=pnginfo)
            logger.info("Image %s saved to %s", i, filename.name)

            filenames.append(build_batch_output_relpath(batch_id, filename.name))
    finally:
        IpAdapterManager.cleanup(pipe, ip_adapter_enabled)
        _cleanup_lora_adapters(pipe, adapter_names)
        release_pipeline(pipe, logger=logger)

    return filenames

@torch.inference_mode()
def generate_images_inpaint_controlnet_in_process(params: dict[str, object],) -> list[str]:
    """
    Generate SD1.5 inpaint + ControlNet outputs and write PNG files.

    Returns:
        Output PNG paths relative to ``OUTPUT_DIR``.
    """
    # Normalize all inpaint controlnet inputs in one place for easier maintenance and tracing.
    initial_image = cast(Image.Image, params["initial_image"])
    mask_image = cast(Image.Image, params["mask_image"])
    prompt = str(params["prompt"])
    negative_prompt = str(params.get("negative_prompt") or "")
    steps = int(params.get("steps") or 20)
    cfg = float(params.get("cfg") or 7.5)
    seed = params.get("seed")
    scheduler = str(params.get("scheduler") or "euler")
    model = params.get("model")
    num_images = int(params.get("num_images") or 1)
    strength = float(params.get("strength") or 0.5)
    padding_mask_crop = int(params.get("padding_mask_crop") or 32)
    clip_skip = int(params.get("clip_skip") or 1)
    controlnet_model = cast(str | list[str], params["controlnet_model"])
    control_image = cast(Image.Image | list[Image.Image], params["control_image"])
    controlnet_conditioning_scale = cast(
        float | list[float],
        params.get("controlnet_conditioning_scale", 1.0),
    )
    controlnet_guess_mode = bool(params.get("controlnet_guess_mode", False))
    control_guidance_start = float(params.get("control_guidance_start", 0.0))
    control_guidance_end = float(params.get("control_guidance_end", 1.0))
    controlnet_inpaint_condition = bool(params.get("controlnet_inpaint_condition", False))
    lora_adapters = params.get("lora_adapters")
    batch_id = cast(str | None, params.get("batch_id"))

    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = seed
    if batch_id is None:
        batch_id = make_batch_id()
    params["batch_id"] = batch_id

    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)
    width, height = initial_image.size
    if controlnet_inpaint_condition:
        control_image = _make_inpaint_controlnet_condition(initial_image, mask_image)
    else:
        control_image = _resize_control_image_to_target(
            cast(Image.Image | list[Image.Image], control_image),
            target_width=width,
            target_height=height,
        )

    adapter_names: list[str] = []
    pipe = load_controlnet_inpaint_pipeline(model, controlnet_model)
    try:
        pipe.scheduler = create_scheduler(scheduler, pipe)
        logger.info(
            "ControlNet Inpaint: model=%s seed=%s scheduler=%s steps=%s cfg=%s size=%sx%s num_images=%s strength=%s padding_mask_crop=%s",
            model,
            base_seed,
            scheduler,
            steps,
            cfg,
            width,
            height,
            num_images,
            strength,
            padding_mask_crop,
        )

        filenames = []
        adapter_names, lora_coverage = apply_lora_adapters_with_validation(
            pipe,
            lora_adapters,
            expected_family="sd15",
            validate=True,
        )
        report_path = write_lora_coverage_report(batch_output_dir, batch_id, lora_coverage)
        if report_path is not None:
            logger.info("LoRA coverage report saved to %s", report_path)

        for i in range(num_images):
            current_seed = base_seed + i
            generator = torch.Generator(device="cuda").manual_seed(current_seed)

            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=initial_image,
                mask_image=mask_image,
                control_image=control_image,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
                strength=strength,
                padding_mask_crop=padding_mask_crop,
                clip_skip=clip_skip,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                guess_mode=controlnet_guess_mode,
                control_guidance_start=control_guidance_start,
                control_guidance_end=control_guidance_end,
            ).images[0]

            filename = batch_output_dir / f"{batch_id}_controlnet_{current_seed}.png"
            image_params = {
                **params,
                "mode": "inpaint_controlnet",
                "width": width,
                "height": height,
                "seed": current_seed,
                "batch_id": batch_id,
            }
            pnginfo = build_png_metadata(image_params)
            image.save(filename, pnginfo=pnginfo)
            logger.info("Image %s saved to %s", i, filename.name)

            filenames.append(build_batch_output_relpath(batch_id, filename.name))
    finally:
        _cleanup_lora_adapters(pipe, adapter_names)
        release_pipeline(pipe, logger=logger)

    return filenames
