"""SDXL inpaint generation operation."""

from backend.sdxl.runtime_common import *
from backend.sdxl.adapters import _cleanup_lora_adapters
from backend.sdxl.loaders import load_inpaint_pipeline
from backend.sdxl.preparation import render_inpaint_image
from backend.sdxl.results import _metadata_without_runtime_images, save_image

@torch.inference_mode()
def generate_inpaint_in_process(params: dict[str, object],) -> dict[str, list[str]]:
    #1. Load and create local method variables + ensure correct formatting from input dict
    initial_image = params["initial_image"]
    mask_image = params["mask_image"]
    strength = float(params["strength"])
    prompt = str(params["prompt"])
    negative_prompt = str(params["negative_prompt"])
    steps = int(params["steps"])
    guidance_scale = float(params["guidance_scale"])
    seed = params["seed"]
    model = params["model"]
    num_images = int(params["num_images"])
    padding_mask_crop = int(params["padding_mask_crop"])
    clip_skip = int(params["clip_skip"])
    scheduler = str(params["scheduler"])
    lora_adapters = params["lora_adapters"]
    ip_adapter_image = params.get("ip_adapter_image")
    ip_adapter_enabled = isinstance(ip_adapter_image, Image.Image)
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
    width, height = initial_image.size
    logger.info("SDXL Inpaint: model=%s seed=%s steps=%s guidance_scale=%s size=%sx%s strength=%s num_images=%s padding_mask_crop=%s",model, base_seed, steps, guidance_scale, width, height, strength, num_images, padding_mask_crop,
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
        pipe = load_inpaint_pipeline(model)
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
            pipe,
            lora_adapters,
            expected_family="sdxl",
            validate=True,
        )
        report_path = write_lora_coverage_report(batch_output_dir, batch_id, lora_coverage)
        if report_path is not None:
            logger.info("LoRA coverage report saved to %s", report_path)

        #7. Render image one by one
        for i in range(num_images):
            # Set current seed
            current_seed = base_seed + i

            # device = getattr(pipe, "_execution_device", None) or pipe.device
            # timesteps = build_fixed_step_timesteps(pipe.scheduler, steps, strength, device = device)
            # Render images
            image = render_inpaint_image(
                pipe,
                initial_image=initial_image,
                mask_image=mask_image,
                strength=strength,
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=current_seed,
                padding_mask_crop=padding_mask_crop,
                clip_skip=clip_skip,
                ip_adapter_image=ip_adapter_image if ip_adapter_enabled else None,
            )

            # Generate image metadata and append to image
            image_params = _metadata_without_runtime_images(params)
            image_params.pop("initial_image", None)
            image_params.pop("mask_image", None)
            image_params["mode"] = "inpaint"
            image_params["seed"] = current_seed
            image_params["batch_id"] = batch_id
            image_params["width"] = width
            image_params["height"] = height
            image_params["ip_adapter_enabled"] = ip_adapter_enabled
            relpath = save_image(
                image=image,
                batch_output_dir=batch_output_dir,
                batch_id=batch_id,
                seed=current_seed,
                metadata=image_params,
            )
            logger.info("Image %s saved to %s", i, Path(relpath).name)

            filenames.append(relpath)
    finally:
        IpAdapterManager.cleanup(pipe, ip_adapter_enabled)
        _cleanup_lora_adapters(pipe, adapter_names)
        release_pipeline(pipe, logger=logger)
        pipe = None

    # 9. Return output back to workflow calling method
    return {"images": [f"/outputs/{name}" for name in filenames]}

