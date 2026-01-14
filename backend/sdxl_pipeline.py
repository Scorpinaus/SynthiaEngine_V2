import logging
import math
from pathlib import Path

import torch
from PIL import Image
from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLInpaintPipeline,
)

from backend.logging_utils import configure_logging
from backend.model_registry import get_model_entry
from backend.pipeline_utils import (
    build_fixed_step_timesteps,
    build_png_metadata,
    build_batch_output_relpath,
    get_batch_output_dir,
    make_batch_id,
    resolve_model_source,
)
from backend.schedulers import create_scheduler

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PIPELINE_CACHE: dict[str, StableDiffusionXLPipeline] = {}
IMG2IMG_PIPELINE_CACHE: dict[str, StableDiffusionXLImg2ImgPipeline] = {}
INPAINT_PIPELINE_CACHE: dict[str, StableDiffusionXLInpaintPipeline] = {}

logger = logging.getLogger(__name__)
configure_logging()


def _get_pipe_device(
    pipe: StableDiffusionXLPipeline | StableDiffusionXLImg2ImgPipeline | StableDiffusionXLInpaintPipeline,
) -> torch.device | str:
    return getattr(pipe, "_execution_device", None) or pipe.device


def _decode_sdxl_latents_to_pil(
    pipe: StableDiffusionXLPipeline | StableDiffusionXLImg2ImgPipeline | StableDiffusionXLInpaintPipeline,
    latents: torch.Tensor,
) -> Image.Image:
    if latents.ndim == 3:
        latents = latents.unsqueeze(0)

    latents = latents.to(device=_get_pipe_device(pipe), dtype=pipe.vae.dtype)
    latents = latents / pipe.vae.config.scaling_factor

    image = pipe.vae.decode(latents, return_dict=False)[0]

    if hasattr(pipe, "image_processor") and hasattr(pipe.image_processor, "postprocess"):
        return pipe.image_processor.postprocess(image, output_type="pil")[0]

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    return Image.fromarray((image[0] * 255).round().astype("uint8"))


def render_sdxl_text2img_latents(
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
) -> torch.Tensor:
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
    ).images[0]


def render_sdxl_img2img_latents(
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
) -> torch.Tensor:
    device = _get_pipe_device(pipe)
    generator = torch.Generator(device=device).manual_seed(seed)
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
    ).images[0]


def render_sdxl_inpaint_latents(
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
) -> torch.Tensor:
    device = _get_pipe_device(pipe)
    generator = torch.Generator(device=device).manual_seed(seed)
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
        output_type="latent",
    ).images[0]


def decode_and_apply_sdxl_hires(
    pipe: StableDiffusionXLPipeline | StableDiffusionXLImg2ImgPipeline | StableDiffusionXLInpaintPipeline,
    *,
    latents: torch.Tensor | list[torch.Tensor],
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    seed: int | list[int] | None,
    scheduler: str,
    clip_skip: int,
    hires_enabled: bool,
    hires_scale: float,
    hires_pipe: StableDiffusionXLImg2ImgPipeline | None = None,
) -> Image.Image | list[Image.Image]:
    latents_list = latents if isinstance(latents, list) else [latents]
    if isinstance(seed, list):
        if len(seed) != len(latents_list):
            raise ValueError("seed list length must match latents list length")
        seed_list: list[int | None] = [int(s) for s in seed]
    else:
        if len(latents_list) != 1 and seed is not None:
            raise ValueError("seed must be a list when latents is a list")
        seed_list = [None if seed is None else int(seed)] * len(latents_list)

    if hires_enabled and hires_scale > 1.0 and torch.cuda.is_available():
        torch.cuda.empty_cache()

    images: list[Image.Image] = []
    for current_latents, current_seed in zip(latents_list, seed_list, strict=True):
        image = _decode_sdxl_latents_to_pil(pipe, current_latents)

        if not hires_enabled or hires_scale <= 1.0:
            images.append(image)
            continue

        if hires_pipe is not None:
            images.append(
                apply_hires_fix_with_img2img_pipe(
                    img2img_pipe=hires_pipe,
                    image=image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    seed=current_seed,
                    scheduler=scheduler,
                    clip_skip=clip_skip,
                    hires_scale=hires_scale,
                )
            )
        else:
            images.append(
                apply_hires_fix(
                    base_pipe=pipe,  # type: ignore[arg-type]
                    image=image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    seed=current_seed,
                    scheduler=scheduler,
                    clip_skip=clip_skip,
                    hires_scale=hires_scale,
                )
            )

    return images[0] if isinstance(latents, torch.Tensor) else images


def save_sdxl_image(
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


def _snap_dimension(value: int, multiple: int = 8) -> int:
    if multiple <= 0:
        return value
    return max(multiple, int(math.ceil(value / multiple)) * multiple)


def _upscale_image(image: Image.Image, scale: float) -> Image.Image:
    if scale <= 1.0:
        return image
    target_width = _snap_dimension(int(round(image.width * scale)))
    target_height = _snap_dimension(int(round(image.height * scale)))
    return image.resize((target_width, target_height), resample=Image.LANCZOS)


def apply_hires_fix(
    base_pipe: StableDiffusionXLPipeline,
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    seed: int | None,
    scheduler: str,
    clip_skip: int,
    hires_scale: float,
    hires_strength: float = 0.35,
) -> Image.Image:
    if hires_scale <= 1.0:
        return image

    upscaled = _upscale_image(image, hires_scale)
    img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pipe(base_pipe)
    img2img_pipe.scheduler = create_scheduler(scheduler, img2img_pipe)
    del base_pipe

    # Hi-res img2img tends to peak during VAE encode at high resolutions; tiling/slicing reduces peak VRAM.
    if hasattr(img2img_pipe, "enable_attention_slicing"):
        img2img_pipe.enable_attention_slicing()
    if hasattr(img2img_pipe, "enable_vae_slicing"):
        img2img_pipe.enable_vae_slicing()
    if hasattr(img2img_pipe, "enable_vae_tiling"):
        img2img_pipe.enable_vae_tiling()

    generator = None
    if seed is not None:
        device = _get_pipe_device(img2img_pipe)
        generator = torch.Generator(device=device).manual_seed(seed)

    device_type = "cuda" if str(_get_pipe_device(img2img_pipe)).startswith("cuda") else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == "cuda")):
        return img2img_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=upscaled,
            strength=hires_strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            clip_skip=clip_skip,
        ).images[0]


def apply_hires_fix_with_img2img_pipe(
    img2img_pipe: StableDiffusionXLImg2ImgPipeline,
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    seed: int | None,
    scheduler: str,
    clip_skip: int,
    hires_scale: float,
    hires_strength: float = 0.35,
) -> Image.Image:
    if hires_scale <= 1.0:
        return image

    upscaled = _upscale_image(image, hires_scale)
    img2img_pipe.scheduler = create_scheduler(scheduler, img2img_pipe)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=_get_pipe_device(img2img_pipe)).manual_seed(seed)

    device_type = "cuda" if str(_get_pipe_device(img2img_pipe)).startswith("cuda") else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == "cuda")):
        return img2img_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=upscaled,
            strength=hires_strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            clip_skip=clip_skip,
        ).images[0]


def load_sdxl_pipeline(model_name: str | None) -> StableDiffusionXLPipeline:
    entry = get_model_entry(model_name)

    if entry.name in PIPELINE_CACHE:
        return PIPELINE_CACHE[entry.name]

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

    pipe.to("cuda")
    PIPELINE_CACHE[entry.name] = pipe

    return pipe


def load_sdxl_img2img_pipeline(model_name: str | None) -> StableDiffusionXLImg2ImgPipeline:
    entry = get_model_entry(model_name)

    if entry.name in IMG2IMG_PIPELINE_CACHE:
        return IMG2IMG_PIPELINE_CACHE[entry.name]

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
    IMG2IMG_PIPELINE_CACHE[entry.name] = pipe

    return pipe


def load_sdxl_inpaint_pipeline(model_name: str | None) -> StableDiffusionXLInpaintPipeline:
    entry = get_model_entry(model_name)

    if entry.name in INPAINT_PIPELINE_CACHE:
        return INPAINT_PIPELINE_CACHE[entry.name]

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
    INPAINT_PIPELINE_CACHE[entry.name] = pipe

    return pipe


@torch.inference_mode()
def run_sdxl_text2img(payload: dict[str, object]) -> dict[str, list[str]]:
    prompt = str(payload.get("prompt") or "")
    negative_prompt = str(payload.get("negative_prompt") or "")
    steps = int(payload.get("steps", 20))
    guidance_scale = float(payload.get("guidance_scale", 7.5))
    width = int(payload.get("width", 1024))
    height = int(payload.get("height", 1024))
    seed = payload.get("seed")
    model = payload.get("model")
    num_images = int(payload.get("num_images", 1))
    clip_skip = int(payload.get("clip_skip", 1))
    scheduler = payload.get("scheduler")
    hires_enabled = bool(payload.get("hires_enabled", False))
    hires_scale = float(payload.get("hires_scale", 1.0))

    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)

    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    pipe = load_sdxl_pipeline(model)
    logger.info("SDXL Generate: model=%s seed=%s steps=%s guidance_scale=%s size=%sx%s num_images=%s",
        model, base_seed, steps, guidance_scale, width, height, num_images,
    )

    filenames: list[str] = []
    pipe.scheduler = create_scheduler(scheduler, pipe)

    latents_batch: list[torch.Tensor] = []
    seed_batch: list[int] = []
    for i in range(num_images):
        current_seed = base_seed + i

        latents = render_sdxl_text2img_latents(
            pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            seed=current_seed,
            clip_skip=clip_skip,
        )
        latents_batch.append(latents.detach().cpu())
        seed_batch.append(current_seed)
        del latents

    hires_pipe = None
    decode_pipe: StableDiffusionXLPipeline | StableDiffusionXLImg2ImgPipeline = pipe
    if hires_enabled and hires_scale > 1.0:
        hires_pipe = StableDiffusionXLImg2ImgPipeline(**pipe.components)
        hires_pipe.scheduler = create_scheduler(scheduler, hires_pipe)

        if hasattr(hires_pipe, "enable_attention_slicing"):
            hires_pipe.enable_attention_slicing()
        if hasattr(hires_pipe, "enable_vae_slicing"):
            hires_pipe.enable_vae_slicing()
        if hasattr(hires_pipe, "enable_vae_tiling"):
            hires_pipe.enable_vae_tiling()

        decode_pipe = hires_pipe
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    images = decode_and_apply_sdxl_hires(
        decode_pipe,
        latents=latents_batch,
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed_batch,
        scheduler=scheduler,
        clip_skip=clip_skip,
        hires_enabled=hires_enabled,
        hires_scale=hires_scale,
        hires_pipe=hires_pipe,
    )
    del latents_batch

    for i, (image, current_seed) in enumerate(zip(images, seed_batch, strict=True)):

        relpath = save_sdxl_image(
            image=image,
            batch_output_dir=batch_output_dir,
            batch_id=batch_id,
            seed=current_seed,
            metadata={
            "mode": "txt2img",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "seed": current_seed,
            "model": model,
            "clip_skip": clip_skip,
            "hires_enabled": hires_enabled,
            "hires_scale": hires_scale,
            "batch_id": batch_id,
            },
        )
        logger.info("Image %s saved to %s", i, Path(relpath).name)

        filenames.append(relpath)

    return {"images": [f"/outputs/{name}" for name in filenames]}


@torch.inference_mode()
def run_sdxl_img2img(
    initial_image,
    strength: float,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    seed: int | None,
    model: str | None,
    num_images: int,
    clip_skip: int,
    scheduler: str
) -> dict[str, list[str]]:
    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)

    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    pipe = load_sdxl_img2img_pipeline(model)
    logger.info("SDXL Img2Img: model=%s seed=%s steps=%s guidance_scale=%s size=%sx%s strength=%s num_images=%s",
        model, base_seed, steps, guidance_scale, width, height, strength, num_images,
    )

    filenames: list[str] = []
    pipe.scheduler = create_scheduler(scheduler, pipe)
    device = _get_pipe_device(pipe)

    for i in range(num_images):
        current_seed = base_seed + i

        timesteps = build_fixed_step_timesteps(pipe.scheduler, steps, strength, device=device)

        latents = render_sdxl_img2img_latents(
            pipe,
            initial_image=initial_image,
            strength=strength,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=current_seed,
            clip_skip=clip_skip,
        )

        image = _decode_sdxl_latents_to_pil(pipe, latents)
        del latents

        image_width, image_height = initial_image.size
        relpath = save_sdxl_image(
            image=image,
            batch_output_dir=batch_output_dir,
            batch_id=batch_id,
            seed=current_seed,
            metadata={
            "mode": "img2img",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "width": image_width,
            "height": image_height,
            "seed": current_seed,
            "model": model,
            "strength": strength,
            "clip_skip": clip_skip,
            "batch_id": batch_id,
            },
        )
        logger.info("Image %s saved to %s", i, Path(relpath).name)

        filenames.append(relpath)

    return {"images": [f"/outputs/{name}" for name in filenames]}


@torch.inference_mode()
def run_sdxl_inpaint(
    initial_image,
    mask_image,
    strength: float,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    seed: int | None,
    model: str | None,
    num_images: int,
    padding_mask_crop: int,
    clip_skip: int,
    scheduler: str
) -> dict[str, list[str]]:
    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)

    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    pipe = load_sdxl_inpaint_pipeline(model)
    width, height = initial_image.size
    logger.info(
        "SDXL Inpaint: model=%s seed=%s steps=%s guidance_scale=%s size=%sx%s strength=%s num_images=%s padding_mask_crop=%s",
        model, base_seed, steps, guidance_scale, width, height, strength, num_images, padding_mask_crop,
    )

    filenames: list[str] = []
    pipe.scheduler = create_scheduler(scheduler, pipe)

    for i in range(num_images):
        current_seed = base_seed + i

        # device = getattr(pipe, "_execution_device", None) or pipe.device
        # timesteps = build_fixed_step_timesteps(pipe.scheduler, steps, strength, device = device)

        latents = render_sdxl_inpaint_latents(
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
        )

        image = _decode_sdxl_latents_to_pil(pipe, latents)
        del latents

        relpath = save_sdxl_image(
            image=image,
            batch_output_dir=batch_output_dir,
            batch_id=batch_id,
            seed=current_seed,
            metadata={
            "mode": "inpaint",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "seed": current_seed,
            "model": model,
            "strength": strength,
            "padding_mask_crop": padding_mask_crop,
            "clip_skip": clip_skip,
            "batch_id": batch_id,
            },
        )
        logger.info("Image %s saved to %s", i, Path(relpath).name)

        filenames.append(relpath)

    return {"images": [f"/outputs/{name}" for name in filenames]}
