"""SDXL image-to-image generation operation."""

from backend.sdxl.runtime_common import *
from backend.sdxl.adapters import _cleanup_lora_adapters
from backend.sdxl.loaders import load_img2img_pipeline
from backend.sdxl.preparation import _build_latent_decoder, _decode_latents_to_pil, render_img2img_latents
from backend.sdxl.results import _metadata_without_runtime_images, save_image

@torch.inference_mode()
def generate_img2img_in_process(params: dict[str, object],) -> dict[str, list[str]]:
    #1. Load and create local method variables + ensure correct formatting from input dict
    initial_image = params["initial_image"]
    strength = float(params["strength"])
    prompt = str(params["prompt"])
    negative_prompt = str(params["negative_prompt"])
    steps = int(params["steps"])
    guidance_scale = float(params["guidance_scale"])
    width = int(params["width"])
    height = int(params["height"])
    seed = params["seed"]
    model = params["model"]
    num_images = int(params["num_images"])
    clip_skip = int(params["clip_skip"])
    scheduler = str(params["scheduler"])
    lora_adapters = params["lora_adapters"]
    ip_adapter_image = params.get("ip_adapter_image")
    ip_adapter_enabled = isinstance(ip_adapter_image, Image.Image)
    ip_adapter_was_enabled = ip_adapter_enabled
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

    #2. Check and set seed value
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)
    logger.info("SDXL Img2Img: model=%s seed=%s steps=%s guidance_scale=%s size=%sx%s strength=%s num_images=%s",
        model, base_seed, steps, guidance_scale, width, height, strength, num_images,
    )

    #3. Create batch_id and output directory
    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    #6. Create list of filenames
    filenames: list[str] = []
    pipe = None
    adapter_names: list[str] = []
    try:
        #4. Load and create pipeline and scheduler
        pipe = load_img2img_pipeline(model)
        pipe.scheduler = create_scheduler(scheduler, pipe)
        if ip_adapter_enabled:
            IpAdapterManager.load(
                pipe,
                model=ip_adapter_model,
                subfolder=ip_adapter_subfolder,
                weight_name=ip_adapter_weight_name,
                scale=ip_adapter_scale,
                family="SDXL",
            )

        #5. Load lora into pipeline
        adapter_names, lora_coverage = apply_lora_adapters_with_validation(
            pipe,lora_adapters,expected_family="sdxl",validate=True,
        )
        report_path = write_lora_coverage_report(batch_output_dir, batch_id, lora_coverage)
        if report_path is not None:
            logger.info("LoRA coverage report saved to %s", report_path)

        #7. Render latent images one by one
        latents_batch: list[torch.Tensor] = []
        seed_batch: list[int] = []
        for i in range(num_images):
            current_seed = base_seed + i

            # timesteps = build_fixed_step_timesteps(pipe.scheduler, steps, strength, device=device)
            # Render latent images
            latents = render_img2img_latents(
                pipe,
                initial_image=initial_image,
                strength=strength,
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=current_seed,
                clip_skip=clip_skip,
                ip_adapter_image=ip_adapter_image if ip_adapter_enabled else None,
            )
            latents_batch.append(latents.detach().cpu())
            seed_batch.append(current_seed)
            del latents

        latent_decoder = _build_latent_decoder(pipe)
        IpAdapterManager.cleanup(pipe, ip_adapter_enabled)
        ip_adapter_enabled = False
        _cleanup_lora_adapters(pipe, adapter_names)
        adapter_names = []
        release_pipeline(pipe, logger=logger)
        pipe = None

        for i, (latents, current_seed) in enumerate(zip(latents_batch, seed_batch, strict=True)):
            # Decode latent to image after the render pipeline has been released.
            image = _decode_latents_to_pil(latent_decoder, latents)

            # Generate image metadata and append to image
            image_width, image_height = initial_image.size
            image_params = _metadata_without_runtime_images(params)
            image_params.pop("initial_image", None)
            image_params["mode"] = "img2img"
            image_params["seed"] = current_seed
            image_params["batch_id"] = batch_id
            image_params["width"] = image_width
            image_params["height"] = image_height
            image_params["ip_adapter_enabled"] = ip_adapter_was_enabled
            # Save filename to rendered image
            relpath = save_image(
                image=image,
                batch_output_dir=batch_output_dir,
                batch_id=batch_id,
                seed=current_seed,
                metadata=image_params,
            )
            logger.info("Image %s saved to %s", i, Path(relpath).name)

            filenames.append(relpath)
        del latents_batch
    finally:
        if pipe is not None:
            IpAdapterManager.cleanup(pipe, ip_adapter_enabled)
            _cleanup_lora_adapters(pipe, adapter_names)
            release_pipeline(pipe, logger=logger)
            pipe = None
    #9. Return output back to workflow calling method
    return {"images": [f"/outputs/{name}" for name in filenames]}

