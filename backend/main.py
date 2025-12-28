import logging
from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from backend.config import DEFAULTS
from backend.model_registry import MODEL_REGISTRY, ModelRegistryEntry
from backend.sd15_pipeline import (
    create_blur_mask,
    generate_images,
    generate_images_img2img,
    generate_images_inpaint,
)

app = FastAPI(title="SD 1.5 API")
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = DEFAULTS["negative_prompt"]
    steps: int = DEFAULTS["steps"]
    cfg: float = DEFAULTS["cfg"]
    width: int = DEFAULTS["width"]
    height: int = DEFAULTS["height"]
    seed: int | None = None
    scheduler: str = "euler"
    num_images: int = 1
    model: str | None = None


@app.get("/models", response_model=list[ModelRegistryEntry])
async def list_models():
    return MODEL_REGISTRY


@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    logger.info("Request JSON: %s", await request.json())
    logger.info("Parsed seed: %s", req.seed)
    filenames = generate_images(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        steps=req.steps,
        cfg=req.cfg,
        width=req.width,
        height=req.height,
        seed=req.seed,
        scheduler=req.scheduler,
        model=req.model,
        num_images = req.num_images,
    )

    return {
        "images": [f"/outputs/{name}" for name in filenames]
    }


@app.post("/generate-img2img")
async def generate_img2img(
    initial_image: UploadFile = File(...),
    strength: float = Form(0.75),
    prompt: str = Form(...),
    negative_prompt: str = Form(DEFAULTS["negative_prompt"]),
    steps: int = Form(DEFAULTS["steps"]),
    cfg: float = Form(DEFAULTS["cfg"]),
    width: int = Form(DEFAULTS["width"]),
    height: int = Form(DEFAULTS["height"]),
    seed: int | None = Form(None),
    scheduler: str = Form("euler"),
    num_images: int = Form(1),
    model: str | None = Form(None),
):
    if not 0 <= strength <= 1:
        raise HTTPException(status_code=400, detail="Strength must be between 0 and 1.")

    image_bytes = await initial_image.read()
    try:
        init_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    init_image = init_image.resize((width, height))

    filenames = generate_images_img2img(
        initial_image=init_image,
        strength=strength,
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        cfg=cfg,
        width=width,
        height=height,
        seed=seed,
        scheduler=scheduler,
        model=model,
        num_images=num_images,
    )

    return {
        "images": [f"/outputs/{name}" for name in filenames]
    }


@app.post("/generate-inpaint")
async def generate_inpaint(
    initial_image: UploadFile = File(...),
    mask_image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(DEFAULTS["negative_prompt"]),
    steps: int = Form(DEFAULTS["steps"]),
    cfg: float = Form(DEFAULTS["cfg"]),
    seed: int | None = Form(None),
    scheduler: str = Form("euler"),
    num_images: int = Form(1),
    model: str | None = Form(None),
    strength: float = Form(0.5),
    padding_mask_crop: int = Form(32)
):
    image_bytes = await initial_image.read()
    mask_bytes = await mask_image.read()
    try:
        init_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid initial image file.") from exc

    try:
        mask = Image.open(BytesIO(mask_bytes)).convert("L")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid mask image file.") from exc

    if mask.size != init_image.size:
        mask = mask.resize(init_image.size, resample=Image.NEAREST)

    filenames = generate_images_inpaint(
        initial_image=init_image,
        mask_image=mask,
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        cfg=cfg,
        seed=seed,
        scheduler=scheduler,
        model=model,
        num_images=num_images,
        strength=strength,
        padding_mask_crop = padding_mask_crop
    )

    return {
        "images": [f"/outputs/{name}" for name in filenames]
    }


@app.post("/create-blur-mask")
async def create_blur_mask_endpoint(
    mask_image: UploadFile = File(...),
    blur_factor: int = Form(8),
):
    mask_bytes = await mask_image.read()
    try:
        mask = Image.open(BytesIO(mask_bytes)).convert("L")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid mask image file.") from exc

    blurred_mask = create_blur_mask(mask, blur_factor)
    output = BytesIO()
    blurred_mask.save(output, format="PNG")
    return Response(content=output.getvalue(), media_type="image/png")
