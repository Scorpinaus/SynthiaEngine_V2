import torch
import logging
import random
import time
from PIL import ImageFilter, Image
from PIL.PngImagePlugin import PngInfo
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)
from diffusers import (EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    DPMSolverSinglestepScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler, HeunDiscreteScheduler, LMSDiscreteScheduler, DEISMultistepScheduler, UniPCMultistepScheduler)
from pathlib import Path

from backend.model_registry import ModelRegistryEntry, get_model_entry
from backend.lora_registry import get_lora_entry
from backend.resource_logging import resource_logger
# from testing.pipeline_stable_diffusion import(StableDiffusionPipeline)
from backend.pipeline_utils import build_fixed_step_timesteps

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PIPELINE_CACHE: dict[str, StableDiffusionPipeline] = {}
IMG2IMG_PIPELINE_CACHE: dict[str, StableDiffusionImg2ImgPipeline] = {}
INPAINT_PIPELINE_CACHE: dict[str, StableDiffusionInpaintPipeline] = {}
CONTROLNET_PIPELINE_CACHE: dict[str, StableDiffusionControlNetPipeline] = {}

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def _resolve_model_source(entry: ModelRegistryEntry) -> str:
    if entry.location_type == "hub":
        return entry.link

    return str(Path(entry.link).expanduser())


def load_pipeline(model_name: str | None):
    entry = get_model_entry(model_name)

    if entry.name in PIPELINE_CACHE:
        return PIPELINE_CACHE[entry.name]

    source = _resolve_model_source(entry)
    logger.info("URL: %s", source)
    if entry.model_type == "diffusers":
        pipe = StableDiffusionPipeline.from_pretrained(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,  # keep simple; can re-enable later
        )
    elif entry.model_type == "single-file":
        pipe = StableDiffusionPipeline.from_single_file(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    pipe.to("cuda")
    PIPELINE_CACHE[entry.name] = pipe

    return pipe


def load_img2img_pipeline(model_name: str | None):
    entry = get_model_entry(model_name)

    if entry.name in IMG2IMG_PIPELINE_CACHE:
        return IMG2IMG_PIPELINE_CACHE[entry.name]

    source = _resolve_model_source(entry)
    logger.info("URL: %s", source)
    if entry.model_type == "diffusers":
        img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        img2img_pipe = StableDiffusionImg2ImgPipeline.from_single_file(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    img2img_pipe.to("cuda")
    IMG2IMG_PIPELINE_CACHE[entry.name] = img2img_pipe

    return img2img_pipe


def load_inpaint_pipeline(model_name: str | None):
    entry = get_model_entry(model_name)

    if entry.name in INPAINT_PIPELINE_CACHE:
        return INPAINT_PIPELINE_CACHE[entry.name]

    source = _resolve_model_source(entry)
    logger.info("URL: %s", source)
    if entry.model_type == "diffusers":
        inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        inpaint_pipe = StableDiffusionInpaintPipeline.from_single_file(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    inpaint_pipe.to("cuda")
    INPAINT_PIPELINE_CACHE[entry.name] = inpaint_pipe

    return inpaint_pipe


def load_controlnet_pipeline(model_name: str | None, controlnet_model: str):
    entry = get_model_entry(model_name)
    cache_key = f"{entry.name}::{controlnet_model}"
    if cache_key in CONTROLNET_PIPELINE_CACHE:
        return CONTROLNET_PIPELINE_CACHE[cache_key]

    source = _resolve_model_source(entry)
    logger.info("Base model: %s", source)
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model,
        torch_dtype=torch.float16,
    )

    if entry.model_type == "diffusers":
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        pipe = StableDiffusionControlNetPipeline.from_single_file(
            source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    pipe.to("cuda")
    CONTROLNET_PIPELINE_CACHE[cache_key] = pipe
    return pipe
def create_scheduler(name: str, pipe):
    name = name.lower()
    
    if name == "ddim":
        return DDIMScheduler.from_config(pipe.scheduler.config)
    
    if name == "dpm++2m":
        return DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if name == "dpm++2m_karras":
        return DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
        )    
    
    if name == "dpm++2m_sde":
        return DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
        )
    
    if name == "dpm++2m_sde_karras":
        return DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
        )        

    if name == "dpm++_sde":
        return DPMSolverSinglestepScheduler.from_config(
            pipe.scheduler.config,
        )

    if name == "dpm++_sde_karras":
        return DPMSolverSinglestepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas = True,
        )
        
    if name == "dpm2":
        return KDPM2DiscreteScheduler.from_config(
            pipe.scheduler.config,
        )
    
    if name == "dpm2_karras":
        return KDPM2DiscreteScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas = True,
        )

    if name == "dpm2_a":
        return KDPM2AncestralDiscreteScheduler.from_config(
            pipe.scheduler.config,
        )    

    if name == "dpm2_a_karras":
        return KDPM2AncestralDiscreteScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas = True,
        ) 
        
    if name == "euler":
        return EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    if name == "euler_a":
        return EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    if name == "heun":
        return HeunDiscreteScheduler.from_config(pipe.scheduler.config)
    
    if name == "lms":
        return LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    
    if name == "lms_karras":
        return LMSDiscreteScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas = True)

    if name == "deis":
        return DEISMultistepScheduler.from_config(pipe.scheduler.config)
    
    if name == "unipc":
        return UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    raise ValueError(f"Unknown scheduler: {name}")


def create_blur_mask(mask_image, blur_factor: int):
    blur_factor = max(0, min(blur_factor, 128))
    if blur_factor == 0:
        return mask_image
    return mask_image.filter(ImageFilter.GaussianBlur(radius=blur_factor))


def _make_batch_id() -> str:
    return f"b{int(time.time())}_{random.randint(1000, 9999)}"


def _resource_metadata(bound_args):
    return {
        "batch_id": bound_args.arguments.get("batch_id"),
        "model": bound_args.arguments.get("model"),
        "num_images": bound_args.arguments.get("num_images"),
    }


def _build_png_metadata(metadata: dict[str, object]) -> PngInfo:
    info = PngInfo()
    for key, value in metadata.items():
        if value is None:
            continue
        info.add_text(key, str(value))
    return info

def generate_images_controlnet(
    prompt: str,
    negative_prompt: str,
    steps: int,
    cfg: float,
    width: int,
    height: int,
    seed: int | None,
    scheduler: str,
    model: str | None,
    num_images: int,
    clip_skip: int,
    controlnet_model: str,
    control_image: Image.Image,
    batch_id: str | None = None,
) -> list[str]:
    if not batch_id:
        batch_id = _make_batch_id()

    pipe = load_controlnet_pipeline(model, controlnet_model)
    pipe.scheduler = create_scheduler(scheduler, pipe)
    pipe.safety_checker = None
    pipe.enable_xformers_memory_efficient_attention()

    if clip_skip > 1:
        pipe.text_encoder.config.num_hidden_layers = (
            pipe.text_encoder.config.num_hidden_layers - (clip_skip - 1)
        )

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

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
    )

    metadata = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "cfg": cfg,
        "width": width,
        "height": height,
        "seed": seed,
        "scheduler": scheduler,
        "model": model,
        "controlnet_model": controlnet_model,
        "batch_id": batch_id,
    }
    png_info = _build_png_metadata(metadata)

    filenames = []
    for idx, image in enumerate(results.images):
        name = f"{batch_id}_controlnet_{idx}.png"
        image.save(OUTPUT_DIR / name, pnginfo=png_info)
        filenames.append(name)

    return filenames

@torch.inference_mode()
def generate_images(
    prompt: str,
    negative_prompt: str,
    steps: int,
    cfg: float,
    width: int,
    height: int,
    seed: int,
    scheduler: str,
    model: str | None,
    num_images:int,
    clip_skip: int,
    lora_adapters: list[object] | None = None,
    batch_id: str | None = None,
):
    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = seed
    if batch_id is None:
        batch_id = _make_batch_id()
    
    pipe = load_pipeline(model)
    pipe.scheduler = create_scheduler(scheduler, pipe)
    logger.info(
        "Generate: model=%s seed=%s scheduler=%s steps=%s cfg=%s size=%sx%s num_images=%s",
        model, base_seed, scheduler, steps, cfg, width, height, num_images,)
        
    filenames = []
    adapter_names: list[str] = []

    if lora_adapters:
        weights: list[float] = []
        for adapter in lora_adapters:
            if isinstance(adapter, dict):
                lora_id = adapter.get("lora_id")
                strength = adapter.get("strength", 1.0)
            else:
                lora_id = getattr(adapter, "lora_id", None)
                strength = getattr(adapter, "strength", 1.0)
                
            if lora_id is None:
                raise ValueError("LoRA adapter missing lora_id.")
            entry = get_lora_entry(int(lora_id))
            
            if entry.lora_model_family.lower() != "sd15":
                raise ValueError(f"LoRA {entry.name} is not compatible with SD1.5.")
            
            adapter_name = f"lora_{entry.name}"
            source = entry.file_path
            pipe.load_lora_weights(source, adapter_name=adapter_name)
            adapter_names.append(adapter_name)
            weights.append(float(strength))
            logger.info('lora_name: %s , lora_id: %s, lora_weight: %s', adapter_name, entry.lora_id, strength)
        if hasattr(pipe, "set_adapters"):
            pipe.set_adapters(adapter_names, adapter_weights=weights)

    try:
        for i in range(num_images):
            current_seed = base_seed + i
            generator = torch.Generator(device="cuda").manual_seed(current_seed)

            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg,
                width=width,
                height=height,
                generator=generator,
                clip_skip = clip_skip,
            ).images[0]

            filename = OUTPUT_DIR / f"{batch_id}_{current_seed}.png"
            pnginfo = _build_png_metadata({
                "mode": "txt2img",
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "cfg": cfg,
                "width": width,
                "height": height,
                "seed": current_seed,
                "scheduler": scheduler,
                "model": model,
                "clip_skip": clip_skip,
                "batch_id": batch_id,
            })
            image.save(filename, pnginfo=pnginfo)
            logger.info("Image %s saved to %s", i, filename.name)

            filenames.append(filename.name)
    finally:
        if adapter_names and hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()

    return filenames

@torch.inference_mode()
def generate_images_img2img(
    initial_image,
    strength: float,
    prompt: str,
    negative_prompt: str,
    steps: int,
    cfg: float,
    width: int,
    height: int,
    seed: int,
    scheduler: str,
    model: str | None,
    num_images: int,
    clip_skip: int,
    batch_id: str | None = None,
):
    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = seed
    if batch_id is None:
        batch_id = _make_batch_id()

    pipe = load_img2img_pipeline(model)
    pipe.scheduler = create_scheduler(scheduler, pipe)
    logger.info(
        "Img2Img: model=%s seed=%s scheduler=%s steps=%s cfg=%s size=%sx%s strength=%s num_images=%s",
        model,
        base_seed,
        scheduler,
        steps,
        cfg,
        width,
        height,
        strength,
        num_images,
    )

    filenames = []

    for i in range(num_images):
        current_seed = base_seed + i
        generator = torch.Generator(device="cuda").manual_seed(current_seed)

        # device = getattr(pipe, "_execution_device", None) or pipe.device
        # timesteps = build_fixed_step_timesteps(pipe.scheduler, steps, strength, device=device)

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=initial_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
            clip_skip=clip_skip
        ).images[0]

        filename = OUTPUT_DIR / f"{batch_id}_{current_seed}.png"
        image_width, image_height = initial_image.size
        pnginfo = _build_png_metadata({
            "mode": "img2img",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg": cfg,
            "width": image_width,
            "height": image_height,
            "seed": current_seed,
            "scheduler": scheduler,
            "model": model,
            "strength": strength,
            "clip_skip": clip_skip,
            "batch_id": batch_id,
        })
        image.save(filename, pnginfo=pnginfo)
        logger.info("Image %s saved to %s", i, filename.name)

        filenames.append(filename.name)

    return filenames

@torch.inference_mode()
def generate_images_inpaint(
    initial_image,
    mask_image,
    prompt: str,
    negative_prompt: str,
    steps: int,
    cfg: float,
    seed: int,
    scheduler: str,
    model: str | None,
    num_images: int,
    strength: float,
    padding_mask_crop: int,
    clip_skip: int,
    batch_id: str | None = None,
):
    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = seed
    if batch_id is None:
        batch_id = _make_batch_id()

    pipe = load_inpaint_pipeline(model)
    pipe.scheduler = create_scheduler(scheduler, pipe)
    width, height = initial_image.size
    logger.info(
        "Inpaint: model=%s seed=%s scheduler=%s steps=%s cfg=%s size=%sx%s num_images=%s strength=%s, padding_mask_crop=%s",
        model,
        base_seed,
        scheduler,
        steps,
        cfg,
        width,
        height,
        num_images,
        strength,
        padding_mask_crop
    )

    filenames = []

    for i in range(num_images):
        current_seed = base_seed + i
        generator = torch.Generator(device="cuda").manual_seed(current_seed)

        # device = getattr(pipe, "_execution_device", None) or pipe.device
        # timesteps = build_fixed_step_timesteps(pipe.scheduler, steps, strength, device=device)

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=initial_image,
            mask_image=mask_image,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
            strength=strength,
            padding_mask_crop = padding_mask_crop,
            clip_skip= clip_skip
        ).images[0]

        filename = OUTPUT_DIR / f"{batch_id}_{current_seed}.png"
        pnginfo = _build_png_metadata({
            "mode": "inpaint",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg": cfg,
            "width": width,
            "height": height,
            "seed": current_seed,
            "scheduler": scheduler,
            "model": model,
            "strength": strength,
            "padding_mask_crop": padding_mask_crop,
            "clip_skip": clip_skip,
            "batch_id": batch_id,
        })
        image.save(filename, pnginfo=pnginfo)
        logger.info("Image %s saved to %s", i, filename.name)

        filenames.append(filename.name)

    return filenames
