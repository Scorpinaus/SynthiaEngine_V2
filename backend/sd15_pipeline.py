import torch
import logging
import random
import time
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)
from diffusers import (EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,)
from pathlib import Path

from backend.model_registry import ModelRegistryEntry, get_model_entry
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PIPELINE_CACHE: dict[str, StableDiffusionPipeline] = {}
IMG2IMG_PIPELINE_CACHE: dict[str, StableDiffusionImg2ImgPipeline] = {}
INPAINT_PIPELINE_CACHE: dict[str, StableDiffusionInpaintPipeline] = {}

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

def create_scheduler(name: str, pipe):
    name = name.lower()

    if name == "euler":
        return EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    if name == "euler_a":
        return EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    if name == "dpmpp_2m":
        return DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",
        )

    if name == "dpmpp_2m_karras":
        return DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )

    if name == "ddim":
        return DDIMScheduler.from_config(pipe.scheduler.config)

    raise ValueError(f"Unknown scheduler: {name}")


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
):
    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = seed
    batch_id = f"b{int(time.time())}_{random.randint(1000, 9999)}"
    
    pipe = load_pipeline(model)
    pipe.scheduler = create_scheduler(scheduler, pipe)
    logger.info(
        "Generate: model=%s seed=%s scheduler=%s steps=%s cfg=%s size=%sx%s num_images=%s",
        model,
        base_seed,
        scheduler,
        steps,
        cfg,
        width,
        height,
        num_images,
    )
        
    filenames = []

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
        ).images[0]
        
        filename = OUTPUT_DIR / f"{batch_id}_{current_seed}.png"
        image.save(filename)
        logger.info("Image %s saved to %s", i, filename.name)
        
        filenames.append(filename.name)

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
):
    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = seed
    batch_id = f"b{int(time.time())}_{random.randint(1000, 9999)}"

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

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=initial_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        ).images[0]

        filename = OUTPUT_DIR / f"{batch_id}_{current_seed}.png"
        image.save(filename)
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
    padding_mask_crop: int
):
    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = seed
    batch_id = f"b{int(time.time())}_{random.randint(1000, 9999)}"

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

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=initial_image,
            mask_image=mask_image,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
            strength=strength,
            padding_mask_crop = padding_mask_crop
        ).images[0]

        filename = OUTPUT_DIR / f"{batch_id}_{current_seed}.png"
        image.save(filename)
        logger.info("Image %s saved to %s", i, filename.name)

        filenames.append(filename.name)

    return filenames
