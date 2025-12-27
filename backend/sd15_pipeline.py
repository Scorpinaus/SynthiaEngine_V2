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

MODEL_ID = "runwayml/stable-diffusion-v1-5"

MODEL_PATH = Path(r"D:\diffusion\diffusers\mistoonruby3")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

pipe = None
img2img_pipe = None
inpaint_pipe = None

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def load_pipeline():
    global pipe

    if pipe is not None:
        return pipe

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        safety_checker=None,  # keep simple; can re-enable later
    )

    pipe.to("cuda")

    return pipe


def load_img2img_pipeline():
    global img2img_pipe

    if img2img_pipe is not None:
        return img2img_pipe

    img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    img2img_pipe.to("cuda")

    return img2img_pipe


def load_inpaint_pipeline():
    global inpaint_pipe

    if inpaint_pipe is not None:
        return inpaint_pipe

    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    inpaint_pipe.to("cuda")

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
    num_images:int,
):
    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = seed
    batch_id = f"b{int(time.time())}_{random.randint(1000, 9999)}"
    
    pipe = load_pipeline()
    pipe.scheduler = create_scheduler(scheduler, pipe)
    logger.info(
        "Generate: seed=%s scheduler=%s steps=%s cfg=%s size=%sx%s num_images=%s",
        base_seed, scheduler, steps, cfg, width, height, num_images,)
        
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
    num_images: int,
):
    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = seed
    batch_id = f"b{int(time.time())}_{random.randint(1000, 9999)}"

    pipe = load_img2img_pipeline()
    pipe.scheduler = create_scheduler(scheduler, pipe)
    logger.info(
        "Img2Img: seed=%s scheduler=%s steps=%s cfg=%s size=%sx%s strength=%s num_images=%s",
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
    strength: float,
    prompt: str,
    negative_prompt: str,
    steps: int,
    cfg: float,
    width: int,
    height: int,
    seed: int,
    scheduler: str,
    num_images: int,
):
    logger.info("seed=%s", seed)
    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = seed
    batch_id = f"b{int(time.time())}_{random.randint(1000, 9999)}"

    pipe = load_inpaint_pipeline()
    pipe.scheduler = create_scheduler(scheduler, pipe)
    logger.info(
        "Inpaint: seed=%s scheduler=%s steps=%s cfg=%s size=%sx%s strength=%s num_images=%s",
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
            mask_image=mask_image,
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
