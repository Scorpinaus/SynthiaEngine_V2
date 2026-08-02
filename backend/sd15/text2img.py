"""SD1.5 text-to-image and ControlNet generation operations."""

from backend.sd15.runtime_common import *
from backend.sd15.adapters import (
    _apply_lcm_lora,
    _build_ip_adapter_kwargs,
    _cleanup_lora_adapters,
    _hide_image_encoder_while_using_ip_adapter_embeds,
    _metadata_without_runtime_images,
)
from backend.sd15.loaders import load_controlnet_pipeline, load_text2img_pipeline
from backend.sd15.preparation import (
    _enable_xformers_memory_efficient_attention_if_available,
    _resize_control_image_to_target,
)

@torch.inference_mode()
def generate_images_controlnet_in_process(params: dict[str, object],) -> list[str]:
    """
    Generate SD1.5 + ControlNet images and write PNG outputs to disk.

    This function optionally captures pipeline layer-usage diagnostics based on
    runtime configuration and embeds generation settings into PNG metadata.

    Returns:
        Output PNG paths relative to ``OUTPUT_DIR``.
    """
    # Base text2image inputs
    prompt = str(params["prompt"])
    negative_prompt = str(params.get("negative_prompt") or "")
    steps = int(params.get("steps") or 20)
    cfg = float(params.get("cfg") or 7.5)
    width = int(params.get("width") or 512)
    height = int(params.get("height") or 512)
    seed = params.get("seed")
    scheduler = str(params.get("scheduler") or "euler")
    model = params.get("model")
    num_images = int(params.get("num_images") or 1)
    clip_skip = int(params.get("clip_skip") or 1)
    
    # Controlnet inputs
    controlnet_model = cast(str | list[str], params["controlnet_model"])
    control_image = cast(Image.Image | list[Image.Image], params["control_image"])
    controlnet_conditioning_scale = cast(
        float | list[float],
        params.get("controlnet_conditioning_scale", 1.0),
    )
    controlnet_guess_mode = bool(params.get("controlnet_guess_mode", False))
    control_guidance_start = cast(
        float | list[float],
        params.get("control_guidance_start", 0.0),
    )
    control_guidance_end = cast(
        float | list[float],
        params.get("control_guidance_end", 1.0),
    )
    batch_id = cast(str | None, params.get("batch_id"))

    if not batch_id:
        batch_id = make_batch_id()
    params["batch_id"] = batch_id

    control_image = _resize_control_image_to_target(
        control_image,
        target_width=width,
        target_height=height,
    )

    pipe = load_controlnet_pipeline(model, controlnet_model)
    try:
        pipe.scheduler = create_scheduler(scheduler, pipe)
        pipe.safety_checker = None
        _enable_xformers_memory_efficient_attention_if_available(pipe)
    
        if clip_skip > 1:
            # Diffusers exposes clip-skip by effectively reducing the text encoder depth.
            pipe.text_encoder.config.num_hidden_layers = (
                pipe.text_encoder.config.num_hidden_layers - (clip_skip - 1)
            )
    
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)
    
        arch_layers = None
        used_layer_names = None
        name_to_type = None
    
        if config.PIPELINE_LAYER_LOGGING_ENABLED:
            # Optionally capture which layers run (useful for debugging pipeline variants).
            arch_layers = collect_pipeline_layers(
                pipe,
                leaf_only=config.PIPELINE_LAYER_LOGGING_LEAF_ONLY,
            )
            with capture_runtime_used_layers(
                pipe,
                leaf_only=config.PIPELINE_LAYER_LOGGING_LEAF_ONLY,
            ) as (used_layer_names, name_to_type, name_to_inputs, name_to_calls):
                results = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    width=width,
                    height=height,
                    image=control_image,
                    num_images_per_prompt=num_images,
                    generator=generator,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    guess_mode=controlnet_guess_mode,
                    control_guidance_start=control_guidance_start,
                    control_guidance_end=control_guidance_end,
                )
        else:
            results = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg,
                width=width,
                height=height,
                image=control_image,
                num_images_per_prompt=num_images,
                generator=generator,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                guess_mode=controlnet_guess_mode,
                control_guidance_start=control_guidance_start,
                control_guidance_end=control_guidance_end,
            )
    
        batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)
    
        if config.PIPELINE_LAYER_LOGGING_ENABLED:
            append_layers_report(
                output_dir=batch_output_dir,
                batch_id=batch_id,
                label="sd15_controlnet",
                pipeline_name=pipe.__class__.__name__,
                architecture_layers=arch_layers,
                runtime_used_layer_names=used_layer_names,
                runtime_name_to_type=name_to_type,
                runtime_name_to_input_summary=(name_to_inputs if config.PIPELINE_LAYER_LOGGING_CAPTURE_INPUTS else None),
                runtime_name_to_call_count=(name_to_calls if config.PIPELINE_LAYER_LOGGING_CAPTURE_INPUTS else None),
            )
    
        png_info = build_png_metadata(params)
    
        filenames = []
        for idx, image in enumerate(results.images):
            name = f"{batch_id}_controlnet_{idx}.png"
            image.save(batch_output_dir / name, pnginfo=png_info)
            filenames.append(build_batch_output_relpath(batch_id, name))
    
        return filenames
    finally:
        release_pipeline(pipe, logger=logger)

@torch.inference_mode()
def generate_images_in_process(params: dict[str, object],) -> list[str]:
    """
    Generate SD1.5 txt2img images, write PNG outputs, and return relative paths.

    Features:
        - Optional LoRA adapter loading with coverage report output.
        - Optional prompt embedding path for prompt-weighting/clip-skip policies.
        - Optional runtime layer logging on the first generated image.
        - Embedded PNG metadata for reproducibility.

    Notes:
        ``hires_enabled``/``hires_scale`` are currently recorded in metadata for
        downstream usage; this function itself performs txt2img generation only.
    """
    # Normalize all txt2img inputs in one place for easier maintenance and tracing.
    prompt = str(params["prompt"])
    negative_prompt = str(params.get("negative_prompt") or "")
    steps = int(params.get("steps") or 20)
    cfg = float(params.get("cfg") or 7.5)
    width = int(params.get("width") or 512)
    height = int(params.get("height") or 512)
    seed = params.get("seed")
    scheduler = str(params.get("scheduler") or "euler")
    model = params.get("model")
    num_images = int(params.get("num_images") or 1)
    clip_skip = int(params.get("clip_skip") or 1)
    lora_adapters = params.get("lora_adapters")
    lcm_enabled = bool(params.get("lcm_enabled", False)) or scheduler.lower() == "lcm"
    weighting_policy = str(params.get("weighting_policy") or "diffusers-like")

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

    # 1. Check and set seed number(if not present, set random seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = seed
    
    # 2. Set batch_id for output folder
    if batch_id is None:
        batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)
    
    # 3. Load pipeline and chosen scheduler
    adapter_names: list[str] = []
    pipe = load_text2img_pipeline(model)
    try:
        pipe.scheduler = create_scheduler(scheduler, pipe)
        logger.info("Generate: model=%s, seed=%s, scheduler=%s, steps=%s, cfg=%s, size= %sx%s, num_images=%s", model, base_seed, scheduler, steps, cfg, width, height, num_images,)

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
                    height=height,
                    width=width,
                )
                if ip_adapter_mask_image is not None
                else None
            )
        else:
            ip_adapter_image_embeds = None
            ip_adapter_masks = None
        
        # 4. Apply lora to pipeline and generate lora coverage report
        lora_coverage = {}
        if lcm_enabled:
            lcm_adapter_name = _apply_lcm_lora(pipe)
            adapter_names, lora_coverage = apply_lora_adapters_with_validation(
                pipe,
                lora_adapters,
                expected_family="sd15",
                validate=True,
                preloaded_adapters=[(lcm_adapter_name, 1.0)],
            )
        else:
            adapter_names, lora_coverage = apply_lora_adapters_with_validation(
                pipe,
                lora_adapters,
                expected_family="sd15",
                validate=True,
            )
        report_path = write_lora_coverage_report(batch_output_dir, batch_id, lora_coverage)
        if report_path is not None:
            logger.info("LoRA coverage report saved to %s", report_path)

        arch_layers = None
        if config.PIPELINE_LAYER_LOGGING_ENABLED:
            arch_layers = collect_pipeline_layers(pipe, leaf_only=config.PIPELINE_LAYER_LOGGING_LEAF_ONLY,)

        # 5. Build prompt embeddings
        prompt_embeds = None
        negative_prompt_embeds = None
        use_prompt_embeds = False
        prompt_embeds_ready = False
        if not config.PIPELINE_LAYER_LOGGING_ENABLED:
            prompt_embeds, negative_prompt_embeds, use_prompt_embeds = build_prompt_embeddings(
                pipe,
                prompt,
                negative_prompt,
                clip_skip=clip_skip,
                weighting_policy=weighting_policy,
            )
            prompt_embeds_ready = True
        
        filenames = []
        ip_adapter_kwargs = _build_ip_adapter_kwargs(
            enabled=ip_adapter_enabled,
            image_embeds=ip_adapter_image_embeds,
            masks=ip_adapter_masks,
        )
        # 6. Loop around image generation per image
        for i in range(num_images):
            # Offset seed per image so batches are deterministic and distinct.
            current_seed = base_seed + i
            generator = torch.Generator(device="cuda").manual_seed(current_seed)
            # Capture used layers during rendering
            if config.PIPELINE_LAYER_LOGGING_ENABLED and i == 0:
                with capture_runtime_used_layers(
                    pipe,
                    leaf_only=config.PIPELINE_LAYER_LOGGING_LEAF_ONLY,
                ) as (used_layer_names, name_to_type, name_to_inputs, name_to_calls):
                    prompt_embeds, negative_prompt_embeds, use_prompt_embeds = build_prompt_embeddings(
                        pipe,
                        prompt,
                        negative_prompt,
                        clip_skip=clip_skip,
                        weighting_policy=weighting_policy,
                    )
                    prompt_embeds_ready = True
                    # Generate image
                    with _hide_image_encoder_while_using_ip_adapter_embeds(
                        pipe,
                        enabled=ip_adapter_image_embeds is not None,
                    ):
                        image = pipe(
                            prompt=None if use_prompt_embeds else prompt,
                            negative_prompt=None if use_prompt_embeds else negative_prompt,
                            num_inference_steps=steps,
                            guidance_scale=cfg,
                            width=width,
                            height=height,
                            generator=generator,
                            clip_skip=clip_skip,
                            prompt_embeds=prompt_embeds if use_prompt_embeds else None,
                            negative_prompt_embeds=negative_prompt_embeds if use_prompt_embeds else None,
                            **ip_adapter_kwargs,
                        ).images[0]

                # Log layers to report
                append_layers_report(
                    output_dir=batch_output_dir,
                    batch_id=batch_id,
                    label="sd15_txt2img",
                    pipeline_name=pipe.__class__.__name__,
                    architecture_layers=arch_layers,
                    runtime_used_layer_names=used_layer_names,
                    runtime_name_to_type=name_to_type,
                    runtime_name_to_input_summary=(name_to_inputs if config.PIPELINE_LAYER_LOGGING_CAPTURE_INPUTS else None),
                    runtime_name_to_call_count=(name_to_calls if config.PIPELINE_LAYER_LOGGING_CAPTURE_INPUTS else None),
                )
            else:
                # If prompt embeds not present, generate them
                if not prompt_embeds_ready:
                    prompt_embeds, negative_prompt_embeds, use_prompt_embeds = build_prompt_embeddings(
                        pipe,
                        prompt,
                        negative_prompt,
                        clip_skip=clip_skip,
                        weighting_policy=weighting_policy,
                    )
                    prompt_embeds_ready = True
                # Generate image
                with _hide_image_encoder_while_using_ip_adapter_embeds(
                    pipe,
                    enabled=ip_adapter_image_embeds is not None,
                ):
                    image = pipe(
                        prompt=None if use_prompt_embeds else prompt,
                        negative_prompt=None if use_prompt_embeds else negative_prompt,
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        width=width,
                        height=height,
                        generator=generator,
                        clip_skip=clip_skip,
                        prompt_embeds=prompt_embeds if use_prompt_embeds else None,
                        negative_prompt_embeds=negative_prompt_embeds if use_prompt_embeds else None,
                        **ip_adapter_kwargs,
                    ).images[0]

            # Write the PNG and embed all inputs/settings for later inspection.
            filename = batch_output_dir / f"{batch_id}_{current_seed}.png"
            metadata = {
                **_metadata_without_runtime_images(params),
                "seed": current_seed,
                "ip_adapter_enabled": ip_adapter_enabled,
            }
            pnginfo = build_png_metadata(metadata)
            image.save(filename, pnginfo=pnginfo)
            logger.info("Image %s saved to %s", i, filename.name)

            filenames.append(build_batch_output_relpath(batch_id, filename.name))
    finally:
        IpAdapterManager.cleanup(pipe, ip_adapter_enabled)
        _cleanup_lora_adapters(pipe, adapter_names)
        release_pipeline(pipe, logger=logger)
    # Return list of filenames
    return filenames
