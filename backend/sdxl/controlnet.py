"""SDXL ControlNet preparation and generation operations."""

from backend.sdxl.runtime_common import *
from backend.sdxl.adapters import _cleanup_lora_adapters
from backend.sdxl.loaders import (
    load_controlnet_img2img_pipeline,
    load_controlnet_inpaint_pipeline,
    load_controlnet_text2img_pipeline,
)
from backend.sdxl.preparation import _get_pipe_device
from backend.sdxl.results import save_image

def _resize_control_image_to_target(
    control_image: Image.Image | list[Image.Image],
    *,
    target_width: int,
    target_height: int,
) -> Image.Image | list[Image.Image]:
    def _resize_single(image: Image.Image) -> Image.Image:
        if image.size == (target_width, target_height):
            return image
        return image.resize((target_width, target_height), resample=Image.LANCZOS)

    if isinstance(control_image, list):
        return [_resize_single(image) for image in control_image]
    return _resize_single(control_image)

@torch.inference_mode()
def generate_controlnet_text2img_in_process(params: dict[str, object],) -> dict[str, list[str]]:
    #1. Load and create local method variables + ensure correct formatting from input dict
    prompt = str(params["prompt"])
    negative_prompt = str(params["negative_prompt"])
    steps = int(params["steps"])
    guidance_scale = float(params["guidance_scale"])
    width = int(params["width"])
    height = int(params["height"])
    seed = params["seed"]
    scheduler = str(params["scheduler"])
    model = params["model"]
    num_images = int(params["num_images"])
    clip_skip = int(params["clip_skip"])
    controlnet_model = params["controlnet_model"]
    control_image = params["control_image"]
    controlnet_conditioning_scale = params.get("controlnet_conditioning_scale", 1.0)
    controlnet_guess_mode = bool(params.get("controlnet_guess_mode", False))
    control_guidance_start = float(params.get("control_guidance_start", 0.0))
    control_guidance_end = float(params.get("control_guidance_end", 1.0))

    # 2. Check and set seed value
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)
    control_image = _resize_control_image_to_target(
        control_image,
        target_width=width,
        target_height=height,
    )
    logger.info(
        "SDXL ControlNet Generate: model=%s seed=%s steps=%s guidance_scale=%s size=%sx%s num_images=%s",
        model, base_seed, steps, guidance_scale, width, height, num_images,
    )

    #3. Create batch_id and output directory
    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    #6. Create list of filenames
    filenames: list[str] = []

    pipe = None
    try:
        #4. Load and create pipeline and scheduler
        pipe = load_controlnet_text2img_pipeline(model, controlnet_model)
        pipe.scheduler = create_scheduler(scheduler, pipe)

        #5. Load lora into pipeline
        # TBC
        
        for i in range(num_images):
            # Set current seed
            current_seed = base_seed + i
            generator = torch.Generator(device=_get_pipe_device(pipe)).manual_seed(current_seed)

            # Generate image
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
                clip_skip=clip_skip,
                image=control_image,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                guess_mode=controlnet_guess_mode,
                control_guidance_start=control_guidance_start,
                control_guidance_end=control_guidance_end,
            ).images[0]

            # Create meta-data dict
            image_params = dict(params)
            image_params.pop("control_image", None)
            image_params["mode"] = "txt2img_controlnet"
            image_params["seed"] = current_seed
            image_params["batch_id"] = batch_id

            # Save image with metadata
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
        release_pipeline(pipe, logger=logger)
        pipe = None

    #9. Return list of image names
    return {"images": [f"/outputs/{name}" for name in filenames]}
@torch.inference_mode()
def generate_img2img_controlnet_in_process(params: dict[str, object],) -> dict[str, list[str]]:
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
    lora_adapters = params.get("lora_adapters")
    controlnet_model = params["controlnet_model"]
    control_image = params["control_image"]
    controlnet_conditioning_scale = params.get("controlnet_conditioning_scale", 1.0)
    controlnet_guess_mode = bool(params.get("controlnet_guess_mode", False))
    control_guidance_start = float(params.get("control_guidance_start", 0.0))
    control_guidance_end = float(params.get("control_guidance_end", 1.0))

    #2. Check and set seed value
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)
        
    image_width, image_height = initial_image.size
    control_image = _resize_control_image_to_target(
        control_image,
        target_width=image_width,
        target_height=image_height,
    )
    logger.info(
        "SDXL ControlNet Img2Img: model=%s seed=%s steps=%s guidance_scale=%s size=%sx%s strength=%s num_images=%s",
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
        pipe = load_controlnet_img2img_pipeline(model, controlnet_model)
        pipe.scheduler = create_scheduler(scheduler, pipe)

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
            generator = torch.Generator(device=_get_pipe_device(pipe)).manual_seed(current_seed)
            
            # Render image
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=initial_image,
                control_image=control_image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                clip_skip=clip_skip,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                guess_mode=controlnet_guess_mode,
                control_guidance_start=control_guidance_start,
                control_guidance_end=control_guidance_end,
            ).images[0]
            
            # Generate image metadata and append to image
            image_params = dict(params)
            image_params.pop("initial_image", None)
            image_params.pop("control_image", None)
            image_params["mode"] = "img2img_controlnet"
            image_params["seed"] = current_seed
            image_params["batch_id"] = batch_id
            image_params["width"] = image_width
            image_params["height"] = image_height
            
            # Save image + metadata
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
        _cleanup_lora_adapters(pipe, adapter_names)
        release_pipeline(pipe, logger=logger)
        pipe = None

    # 9. Return output back to workflow calling method
    return {"images": [f"/outputs/{name}" for name in filenames]}
@torch.inference_mode()
def generate_inpaint_controlnet_in_process(params: dict[str, object],) -> dict[str, list[str]]:
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
    lora_adapters = params.get("lora_adapters")
    controlnet_model = params["controlnet_model"]
    control_image = params["control_image"]
    controlnet_conditioning_scale = params.get("controlnet_conditioning_scale", 1.0)
    controlnet_guess_mode = bool(params.get("controlnet_guess_mode", False))
    control_guidance_start = float(params.get("control_guidance_start", 0.0))
    control_guidance_end = float(params.get("control_guidance_end", 1.0))
    
    #2. Check and set seed value
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)

    width, height = initial_image.size
    control_image = _resize_control_image_to_target(
        control_image,
        target_width=width,
        target_height=height,
    )
    logger.info(
        "SDXL ControlNet Inpaint: model=%s seed=%s steps=%s guidance_scale=%s size=%sx%s strength=%s num_images=%s padding_mask_crop=%s",
        model, base_seed, steps, guidance_scale, width, height, strength, num_images, padding_mask_crop,)
    
    #3. Create batch_id and output directory
    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    #6. Create list of filenames
    filenames: list[str] = []
    pipe = None
    adapter_names: list[str] = []
    try:
        #4. Load and create pipeline and scheduler
        pipe = load_controlnet_inpaint_pipeline(model, controlnet_model)
        pipe.scheduler = create_scheduler(scheduler, pipe)

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

        # 7. Render image one by one
        for i in range(num_images):
            # Define current seed
            current_seed = base_seed + i
            generator = torch.Generator(device=_get_pipe_device(pipe)).manual_seed(current_seed)
            
            # Generate image
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=initial_image,
                mask_image=mask_image,
                control_image=control_image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                padding_mask_crop=padding_mask_crop,
                clip_skip=clip_skip,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                guess_mode=controlnet_guess_mode,
                control_guidance_start=control_guidance_start,
                control_guidance_end=control_guidance_end,
            ).images[0]

            # Generate image metadata and append to image
            image_params = dict(params)
            image_params.pop("initial_image", None)
            image_params.pop("mask_image", None)
            image_params.pop("control_image", None)
            image_params["mode"] = "inpaint_controlnet"
            image_params["seed"] = current_seed
            image_params["batch_id"] = batch_id
            image_params["width"] = width
            image_params["height"] = height
            
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
    finally:
        _cleanup_lora_adapters(pipe, adapter_names)
        release_pipeline(pipe, logger=logger)
        pipe = None

    # 9. Return output back to workflow calling method
    return {"images": [f"/outputs/{name}" for name in filenames]}

