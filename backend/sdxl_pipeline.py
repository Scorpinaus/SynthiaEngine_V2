import logging
from contextlib import contextmanager
from pathlib import Path

import torch
from PIL import Image
from diffusers import (
    ControlNetModel,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLInpaintPipeline,
)

from backend.config import OUTPUT_DIR
from backend.ip_adapter import IpAdapterManager
from backend.ip_adapter_embeds import (
    load_ip_adapter_embeds_artifact,
    validate_ip_adapter_embeds_metadata,
)
from backend.logging_utils import configure_logging
from backend.lora_utils import apply_lora_adapters_with_validation, write_lora_coverage_report
from backend.model_registry import get_model_entry
from backend.pipeline_utils import (
    build_fixed_step_timesteps,
    build_png_metadata,
    build_batch_output_relpath,
    cleanup_memory,
    get_batch_output_dir,
    make_batch_id,
    resolve_model_source,
)
from backend.schedulers import create_scheduler

logger = logging.getLogger(__name__)
configure_logging()

_DEFAULT_IP_ADAPTER_MODEL = "h94/IP-Adapter"
_DEFAULT_IP_ADAPTER_SUBFOLDER = "sdxl_models"
_DEFAULT_IP_ADAPTER_WEIGHT_NAME = "ip-adapter_sdxl.bin"
_DEFAULT_IP_ADAPTER_SCALE = 0.6

""" 
    Private Helper functions
"""
def _get_pipe_device(
    pipe: StableDiffusionXLPipeline | StableDiffusionXLImg2ImgPipeline | StableDiffusionXLInpaintPipeline | StableDiffusionXLControlNetPipeline | StableDiffusionXLControlNetImg2ImgPipeline | StableDiffusionXLControlNetInpaintPipeline,
) -> torch.device | str:
    return getattr(pipe, "_execution_device", None) or pipe.device


def _get_module_device(module, fallback: torch.device | str) -> torch.device | str:
    module_device = getattr(module, "device", None)
    if module_device is not None:
        return module_device
    try:
        return next(module.parameters()).device
    except (AttributeError, StopIteration):
        return fallback


def _enable_vae_memory_savers(pipe: object) -> None:
    enable_slicing = getattr(pipe, "enable_vae_slicing", None)
    if callable(enable_slicing):
        enable_slicing()
    else:
        vae = getattr(pipe, "vae", None)
        vae_enable_slicing = getattr(vae, "enable_slicing", None)
        if callable(vae_enable_slicing):
            vae_enable_slicing()

    enable_tiling = getattr(pipe, "enable_vae_tiling", None)
    if callable(enable_tiling):
        enable_tiling()
    else:
        vae = getattr(pipe, "vae", None)
        vae_enable_tiling = getattr(vae, "enable_tiling", None)
        if callable(vae_enable_tiling):
            vae_enable_tiling()


@contextmanager
def _hide_image_encoder_while_using_ip_adapter_embeds(pipe, *, enabled: bool):
    if not enabled or pipe is None or not hasattr(pipe, "image_encoder"):
        yield
        return

    image_encoder = pipe.image_encoder
    pipe.image_encoder = None
    try:
        yield
    finally:
        pipe.image_encoder = image_encoder


def _decode_latents_to_pil(
    pipe: StableDiffusionXLPipeline | StableDiffusionXLImg2ImgPipeline | StableDiffusionXLInpaintPipeline,
    latents: torch.Tensor,
) -> Image.Image:
    if latents.ndim == 3:
        latents = latents.unsqueeze(0)

    latents = latents.to(device=_get_module_device(pipe.vae, _get_pipe_device(pipe)), dtype=pipe.vae.dtype)
    latents = latents / pipe.vae.config.scaling_factor

    image = pipe.vae.decode(latents, return_dict=False)[0]

    if hasattr(pipe, "image_processor") and hasattr(pipe.image_processor, "postprocess"):
        return pipe.image_processor.postprocess(image, output_type="pil")[0]

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    return Image.fromarray((image[0] * 255).round().astype("uint8"))


def render_text2img_latents(
    pipe: StableDiffusionXLPipeline,
    *,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    seed: int,
    clip_skip: int,
    ip_adapter_image_embeds: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    ip_adapter_kwargs = (
        {"ip_adapter_image_embeds": ip_adapter_image_embeds}
        if ip_adapter_image_embeds is not None
        else {}
    )
    with _hide_image_encoder_while_using_ip_adapter_embeds(
        pipe,
        enabled=ip_adapter_image_embeds is not None,
    ):
        device = _get_pipe_device(pipe)
        generator = torch.Generator(device=device).manual_seed(seed)
        return pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            clip_skip=clip_skip,
            output_type="latent",
            **ip_adapter_kwargs,
        ).images[0]


def render_img2img_latents(
    pipe: StableDiffusionXLImg2ImgPipeline,
    *,
    initial_image: Image.Image,
    strength: float,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    seed: int,
    clip_skip: int,
    ip_adapter_image: Image.Image | None = None,
) -> torch.Tensor:
    device = _get_pipe_device(pipe)
    generator = torch.Generator(device=device).manual_seed(seed)
    ip_adapter_kwargs = (
        {"ip_adapter_image": ip_adapter_image} if ip_adapter_image is not None else {}
    )
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=initial_image,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        clip_skip=clip_skip,
        output_type="latent",
        **ip_adapter_kwargs,
    ).images[0]


def render_inpaint_image(
    pipe: StableDiffusionXLInpaintPipeline,
    *,
    initial_image: Image.Image,
    mask_image: Image.Image,
    strength: float,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    seed: int,
    padding_mask_crop: int,
    clip_skip: int,
    ip_adapter_image: Image.Image | None = None,
) -> Image.Image:
    device = _get_pipe_device(pipe)
    generator = torch.Generator(device=device).manual_seed(seed)
    ip_adapter_kwargs = (
        {"ip_adapter_image": ip_adapter_image} if ip_adapter_image is not None else {}
    )
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=initial_image,
        mask_image=mask_image,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        padding_mask_crop=padding_mask_crop,
        clip_skip=clip_skip,
        **ip_adapter_kwargs,
    ).images[0]


def save_image(
    *,
    image: Image.Image,
    batch_output_dir: Path,
    batch_id: str,
    seed: int,
    metadata: dict[str, object],
) -> str:
    filename = batch_output_dir / f"{batch_id}_{seed}.png"
    pnginfo = build_png_metadata(metadata)
    image.save(filename, pnginfo=pnginfo)
    return build_batch_output_relpath(batch_id, filename.name)


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


def _cleanup_lora_adapters(pipe, adapter_names: list[str]) -> None:
    if not adapter_names or not hasattr(pipe, "unload_lora_weights"):
        return
    try:
        pipe.unload_lora_weights()
    except Exception:
        logger.exception("Failed to unload LoRA weights cleanly.")


def _release_pipeline(pipe) -> None:
    if pipe is None:
        return

    if hasattr(pipe, "maybe_free_model_hooks"):
        try:
            pipe.maybe_free_model_hooks()
        except Exception:
            logger.exception("Failed to free SDXL pipeline model hooks.")

    if hasattr(pipe, "remove_all_hooks"):
        try:
            pipe.remove_all_hooks()
        except Exception:
            logger.exception("Failed to remove SDXL pipeline hooks.")


def _metadata_without_runtime_images(params: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in params.items()
        if key not in {"ip_adapter_image", "ip_adapter_image_embeds_ref"}
    }


"""
    Load SDXL Pipeline Functions
"""
def load_text2img_pipeline(model_name: str | None) -> StableDiffusionXLPipeline:
    entry = get_model_entry(model_name)

    source = resolve_model_source(entry)
    logger.info("SDXL model source: %s", source)

    if entry.model_type == "diffusers":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        pipe = StableDiffusionXLPipeline.from_single_file(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    _enable_vae_memory_savers(pipe)
    pipe.to("cuda")
    return pipe


def load_controlnet_text2img_pipeline(
    model_name: str | None,
    controlnet_model: str | list[str],
) -> StableDiffusionXLControlNetPipeline:
    
    entry = get_model_entry(model_name)
    source = resolve_model_source(entry)
    logger.info("SDXL ControlNet model source: %s", source)

    controlnet: ControlNetModel | list[ControlNetModel]
    if isinstance(controlnet_model, list):
        controlnet = [
            ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)
            for model_id in controlnet_model
        ]
    else:
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model,
            torch_dtype=torch.float16,
        )

    if entry.model_type == "diffusers":
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        pipe = StableDiffusionXLControlNetPipeline.from_single_file(
            source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    pipe.to("cuda")
    return pipe


def load_img2img_pipeline(model_name: str | None) -> StableDiffusionXLImg2ImgPipeline:
    entry = get_model_entry(model_name)
    source = resolve_model_source(entry)
    logger.info("SDXL img2img model source: %s", source)

    if entry.model_type == "diffusers":
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    pipe.to("cuda")
    return pipe


def load_controlnet_img2img_pipeline(
    model_name: str | None,
    controlnet_model: str | list[str],
) -> StableDiffusionXLControlNetImg2ImgPipeline:
    entry = get_model_entry(model_name)
    source = resolve_model_source(entry)
    logger.info("SDXL ControlNet img2img model source: %s", source)

    controlnet: ControlNetModel | list[ControlNetModel]
    if isinstance(controlnet_model, list):
        controlnet = [
            ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)
            for model_id in controlnet_model
        ]
    else:
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model,
            torch_dtype=torch.float16,
        )

    if entry.model_type == "diffusers":
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
            source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    pipe.to("cuda")
    return pipe


def load_inpaint_pipeline(model_name: str | None) -> StableDiffusionXLInpaintPipeline:
    entry = get_model_entry(model_name)
    source = resolve_model_source(entry)
    logger.info("SDXL inpaint model source: %s", source)

    if entry.model_type == "diffusers":
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        pipe = StableDiffusionXLInpaintPipeline.from_single_file(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    pipe.to("cuda")
    return pipe


def load_controlnet_inpaint_pipeline(
    model_name: str | None,
    controlnet_model: str | list[str],
) -> StableDiffusionXLControlNetInpaintPipeline:
    entry = get_model_entry(model_name)
    source = resolve_model_source(entry)
    logger.info("SDXL ControlNet inpaint model source: %s", source)

    controlnet: ControlNetModel | list[ControlNetModel]
    if isinstance(controlnet_model, list):
        controlnet = [
            ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)
            for model_id in controlnet_model
        ]
    else:
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model,
            torch_dtype=torch.float16,
        )

    if entry.model_type == "diffusers":
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_single_file(
            source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    pipe.to("cuda")
    return pipe


"""
    Generate and render images functions
"""

@torch.inference_mode()
def generate_controlnet_text2img(params: dict[str, object],) -> dict[str, list[str]]:
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
        _release_pipeline(pipe)
        pipe = None
        cleanup_memory()

    #9. Return list of image names
    return {"images": [f"/outputs/{name}" for name in filenames]}


@torch.inference_mode()
def generate_img2img_controlnet(params: dict[str, object],) -> dict[str, list[str]]:
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
        _release_pipeline(pipe)
        pipe = None
        cleanup_memory()

    # 9. Return output back to workflow calling method
    return {"images": [f"/outputs/{name}" for name in filenames]}


@torch.inference_mode()
def generate_text2img(payload: dict[str, object]) -> dict[str, list[str]]:
    
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

        # Decode latents to images
        images: list[Image.Image] = []
        for latents in latents_batch:
            images.append(_decode_latents_to_pil(pipe, latents))
        del latents_batch

        # Generate image metadata and append to image
        for i, (image, current_seed) in enumerate(zip(images, seed_batch, strict=True)):
            image_params = _metadata_without_runtime_images(payload)
            image_params["mode"] = "txt2img"
            image_params["seed"] = current_seed
            image_params["batch_id"] = batch_id
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

        # Return images list with metadata
        return {"images": [f"/outputs/{name}" for name in filenames]}
    finally:
        IpAdapterManager.cleanup(pipe, ip_adapter_enabled)
        _cleanup_lora_adapters(pipe, adapter_names)
        _release_pipeline(pipe)
        pipe = None
        cleanup_memory()


@torch.inference_mode()
def generate_img2img(params: dict[str, object],) -> dict[str, list[str]]:
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

        #7. Render image one by one
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

            # Decode latent to image and delete intermediate latents
            image = _decode_latents_to_pil(pipe, latents)
            del latents

            # Generate image metadata and append to image
            image_width, image_height = initial_image.size
            image_params = _metadata_without_runtime_images(params)
            image_params.pop("initial_image", None)
            image_params["mode"] = "img2img"
            image_params["seed"] = current_seed
            image_params["batch_id"] = batch_id
            image_params["width"] = image_width
            image_params["height"] = image_height
            image_params["ip_adapter_enabled"] = ip_adapter_enabled
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
        IpAdapterManager.cleanup(pipe, ip_adapter_enabled)
        _cleanup_lora_adapters(pipe, adapter_names)
        _release_pipeline(pipe)
        pipe = None
        cleanup_memory()
    #9. Return output back to workflow calling method
    return {"images": [f"/outputs/{name}" for name in filenames]}


@torch.inference_mode()
def generate_inpaint(params: dict[str, object],) -> dict[str, list[str]]:
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
        _release_pipeline(pipe)
        pipe = None
        cleanup_memory()

    # 9. Return output back to workflow calling method
    return {"images": [f"/outputs/{name}" for name in filenames]}


@torch.inference_mode()
def generate_inpaint_controlnet(params: dict[str, object],) -> dict[str, list[str]]:
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
        _release_pipeline(pipe)
        pipe = None
        cleanup_memory()

    # 9. Return output back to workflow calling method
    return {"images": [f"/outputs/{name}" for name in filenames]}
