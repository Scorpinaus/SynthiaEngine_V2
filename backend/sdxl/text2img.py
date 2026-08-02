"""SDXL text-to-image generation operation."""

from backend.sdxl.runtime_common import *
from backend.sdxl.adapters import _cleanup_lora_adapters
from backend.sdxl.loaders import load_text2img_pipeline
from backend.sdxl.preparation import _build_latent_decoder, _decode_latents_to_pil, render_text2img_latents
from backend.sdxl.results import _metadata_without_runtime_images, save_image

@torch.inference_mode()
def generate_text2img_in_process(payload: dict[str, object]) -> dict[str, list[str]]:
    
    #1. Load and create local method variables + ensure correct formatting from input dict
    prompt = str(payload["prompt"])
    negative_prompt = str(payload["negative_prompt"])
    steps = int(payload["steps"])
    guidance_scale = float(payload["guidance_scale"])
    width = int(payload["width"])
    height = int(payload["height"])
    seed = payload["seed"]
    model = payload["model"]
    num_images = int(payload["num_images"])
    clip_skip = int(payload["clip_skip"])
    scheduler = payload["scheduler"]
    
    lora_adapters = payload.get("lora_adapters")
    ip_adapter_image = payload.get("ip_adapter_image")
    ip_adapter_image_embeds_ref = payload.get("ip_adapter_image_embeds_ref")
    ip_adapter_enabled = isinstance(ip_adapter_image, Image.Image) or ip_adapter_image_embeds_ref is not None
    ip_adapter_was_enabled = ip_adapter_enabled
    if isinstance(ip_adapter_image, Image.Image) and ip_adapter_image_embeds_ref is not None:
        raise ValueError("Provide either ip_adapter_image or ip_adapter_image_embeds_ref, not both.")
    ip_adapter_model = str(payload.get("ip_adapter_model") or _DEFAULT_IP_ADAPTER_MODEL)
    ip_adapter_subfolder = str(
        payload.get("ip_adapter_subfolder") or _DEFAULT_IP_ADAPTER_SUBFOLDER
    )
    ip_adapter_weight_name = str(
        payload.get("ip_adapter_weight_name") or _DEFAULT_IP_ADAPTER_WEIGHT_NAME
    )
    ip_adapter_scale_raw = payload.get("ip_adapter_scale")
    ip_adapter_scale = (
        _DEFAULT_IP_ADAPTER_SCALE
        if ip_adapter_scale_raw is None
        else float(ip_adapter_scale_raw)
    )
    
    #2. Check and set seed value
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)
    logger.info("SDXL Text2Image: model=%s, seed=%s, steps=%s, guidance_scale=%s, size=%sx%s, num_images=%s", model, base_seed, steps, guidance_scale, width, height, num_images)

    #3. Create batch_id and output directory
    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    #6. Create list of filenames
    filenames: list[str] = []

    pipe = None
    adapter_names: list[str] = []
    #7. Render image one by one
    try:
        #4. Load and create pipeline and scheduler
        pipe = load_text2img_pipeline(model)
        pipe.scheduler = create_scheduler(scheduler, pipe)
        if ip_adapter_enabled:
            if ip_adapter_image_embeds_ref is not None:
                embeds_payload = load_ip_adapter_embeds_artifact(ip_adapter_image_embeds_ref)
                validate_ip_adapter_embeds_metadata(
                    embeds_payload,
                    expected_model=ip_adapter_model,
                    expected_subfolder=ip_adapter_subfolder,
                    expected_weight_name=ip_adapter_weight_name,
                    do_classifier_free_guidance=guidance_scale > 1.0,
                )
                ip_adapter_image_embeds = embeds_payload["embeds"]
            else:
                ip_adapter_image_embeds = None
            if ip_adapter_image_embeds_ref is not None:
                IpAdapterManager.load(
                    pipe,
                    model=ip_adapter_model,
                    subfolder=ip_adapter_subfolder,
                    weight_name=ip_adapter_weight_name,
                    scale=ip_adapter_scale,
                    family="SDXL",
                    image_encoder_folder=None,
                )
            else:
                IpAdapterManager.load(
                    pipe,
                    model=ip_adapter_model,
                    subfolder=ip_adapter_subfolder,
                    weight_name=ip_adapter_weight_name,
                    scale=ip_adapter_scale,
                    family="SDXL",
                )
            if ip_adapter_image_embeds_ref is None:
                ip_adapter_image_embeds = IpAdapterManager.prepare_image_embeds(
                    pipe,
                    ip_adapter_image,
                    do_classifier_free_guidance=guidance_scale > 1.0,
                )
        else:
            ip_adapter_image_embeds = None

        #5. Load lora into pipeline
        adapter_names, lora_coverage = apply_lora_adapters_with_validation(
            pipe, lora_adapters, expected_family="sdxl", validate=True,
        )
        report_path = write_lora_coverage_report(batch_output_dir, batch_id, lora_coverage)
        if report_path is not None:
            logger.info("LoRA coverage report saved to %s", report_path)

        # Render latent images. Create latent and seed batch list
        latents_batch: list[torch.Tensor] = []
        seed_batch: list[int] = []
        for i in range(num_images):
            current_seed = base_seed + i
            # Render latents and add to list
            latents = render_text2img_latents(
                pipe,
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                seed=current_seed,
                clip_skip=clip_skip,
                ip_adapter_image_embeds=ip_adapter_image_embeds,
            )
            latents_batch.append(latents.detach().cpu())
            seed_batch.append(current_seed)
            del latents

        latent_decoder = _build_latent_decoder(pipe)
        ip_adapter_image_embeds = None
        IpAdapterManager.cleanup(pipe, ip_adapter_enabled)
        ip_adapter_enabled = False
        _cleanup_lora_adapters(pipe, adapter_names)
        adapter_names = []
        release_pipeline(pipe, logger=logger)
        pipe = None

        # Decode latents to images
        images: list[Image.Image] = []
        for latents in latents_batch:
            images.append(_decode_latents_to_pil(latent_decoder, latents))
        del latents_batch

        # Generate image metadata and append to image
        for i, (image, current_seed) in enumerate(zip(images, seed_batch, strict=True)):
            image_params = _metadata_without_runtime_images(payload)
            image_params["mode"] = "txt2img"
            image_params["seed"] = current_seed
            image_params["batch_id"] = batch_id
            image_params["ip_adapter_enabled"] = ip_adapter_was_enabled
            relpath = save_image(
                image=image,
                batch_output_dir=batch_output_dir,
                batch_id=batch_id,
                seed=current_seed,
                metadata=image_params,
            )
            logger.info("Image %s saved to %s", i, Path(relpath).name)

            filenames.append(relpath)

        # Return images list with metadata
        return {"images": [f"/outputs/{name}" for name in filenames]}
    finally:
        if pipe is not None:
            IpAdapterManager.cleanup(pipe, ip_adapter_enabled)
            _cleanup_lora_adapters(pipe, adapter_names)
            release_pipeline(pipe, logger=logger)
            pipe = None

