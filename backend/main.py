import logging
import re
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from backend.config import DEFAULTS
from backend.model_cache import clear_all_pipelines, prepare_model
from backend.model_registry import MODEL_REGISTRY, ModelRegistryEntry, save_model_registry
from backend.sd15_pipeline import (
    create_blur_mask,
    generate_images,
    generate_images_img2img,
    generate_images_inpaint,
)
from backend.flux_pipeline import run_flux_img2img, run_flux_inpaint, run_flux_text2img
from backend.sdxl_pipeline import (
    run_sdxl_img2img,
    run_sdxl_inpaint,
    run_sdxl_text2img,
)
from backend.z_image_pipeline import run_z_image_img2img, run_z_image_text2img

app = FastAPI(title="SD 1.5 API")
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

OUTPUT_DIR = Path("outputs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

## BaseModel references
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
    clip_skip: int = 1


class SdxlGenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = DEFAULTS["negative_prompt"]
    steps: int = DEFAULTS["steps"]
    guidance_scale: float = DEFAULTS["cfg"]
    width: int = DEFAULTS["width"]
    height: int = DEFAULTS["height"]
    seed: int | None = None
    num_images: int = 1
    model: str | None = None
    clip_skip: int = 1


class ZImageGenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = DEFAULTS["negative_prompt"]
    steps: int = DEFAULTS["steps"]
    guidance_scale: float = DEFAULTS["cfg"]
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    num_images: int = 1
    model: str | None = None


class FluxGenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = DEFAULTS["negative_prompt"]
    steps: int = DEFAULTS["steps"]
    guidance_scale: float = DEFAULTS["cfg"]
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    num_images: int = 1
    model: str | None = None


class ModelCreateRequest(BaseModel):
    name: str
    family: str
    model_type: str
    location_type: str
    model_id: int
    version: str
    link: str


def _extract_png_metadata(path: Path) -> dict[str, str]:
    try:
        with Image.open(path) as image:
            metadata: dict[str, str] = {}
            if hasattr(image, "text"):
                metadata.update(image.text)
            for key, value in (image.info or {}).items():
                if isinstance(value, str) and key not in metadata:
                    metadata[key] = value
            return metadata
    except Exception as exc:
        logger.warning("Failed to read metadata for %s: %s", path.name, exc)
        return {}


@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/models", response_model=list[ModelRegistryEntry])
async def list_models(family: str | None = None):
    if not family:
        return MODEL_REGISTRY

    family_value = family.strip().lower()
    if not family_value:
        return MODEL_REGISTRY

    if family_value in {"sd15", "sd1.5"}:
        pattern = re.compile(r"sd[\s_-]*1\.?5|sd15", re.IGNORECASE)
    elif family_value == "sdxl":
        pattern = re.compile(r"sdxl", re.IGNORECASE)
    elif family_value == "z-image-turbo":
        pattern = re.compile(r"z-image-turbo", re.IGNORECASE)
    elif family_value == "flux":
        pattern = re.compile(r"flux", re.IGNORECASE)
    else:
        pattern = re.compile(re.escape(family_value), re.IGNORECASE)

    return [entry for entry in MODEL_REGISTRY if pattern.search(entry.family)]


@app.post("/models", response_model=ModelRegistryEntry, status_code=201)
async def create_model(req: ModelCreateRequest):
    if any(entry.name == req.name for entry in MODEL_REGISTRY):
        raise HTTPException(status_code=409, detail="Model name already exists.")

    entry = ModelRegistryEntry(**req.dict())
    MODEL_REGISTRY.append(entry)
    save_model_registry(MODEL_REGISTRY)
    return entry


@app.get("/history")
async def list_history():
    if not OUTPUT_DIR.exists():
        return []

    records: list[dict[str, object]] = []
    for image_path in OUTPUT_DIR.glob("*.png"):
        stat = image_path.stat()
        timestamp = stat.st_mtime
        created_at = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        metadata = _extract_png_metadata(image_path)
        records.append(
            {
                "filename": image_path.name,
                "url": f"/outputs/{image_path.name}",
                "timestamp": timestamp,
                "created_at": created_at,
                "metadata": metadata,
            }
        )

    records.sort(key=lambda item: item.get("timestamp", 0), reverse=True)
    return records

def remap_img2img_strength(strength:float, min_strength = 0.0, gamma: float=0.5) -> float:
    clamped = max(0.0, min(1.0, strength))
    if min_strength <= 0.0:
        # remapped = clamped
        remapped = clamped ** gamma
    else:
        normalized = max(0.0, min(1.0, (clamped - min_strength) / (1.0 - min_strength)))
        # remapped = min_strength + normalized * (1.0 - min_strength)
        remapped = min_strength + (normalized**gamma) * (1.0 - min_strength)        
    return max(0.0, min(1.0, remapped))

## SD1.5

@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    # logger.info("Request JSON: %s", await request.json())
    # logger.info("Parsed seed: %s", req.seed)
    prepare_model(req.model)
    
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
        clip_skip = req.clip_skip,
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
    clip_skip: int = Form(1)
):
    if not 0 <= strength <= 1:
        raise HTTPException(status_code=400, detail="Strength must be between 0 and 1.")

    image_bytes = await initial_image.read()
    try:
        init_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    remapped_strength = remap_img2img_strength(strength)
    init_image = init_image.resize((width, height))
    prepare_model(model)

    filenames = generate_images_img2img(
        initial_image=init_image,
        strength=remapped_strength,
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
        clip_skip=clip_skip
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
    padding_mask_crop: int = Form(32),
    clip_skip: int = Form(1)    
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
    prepare_model(model)
    remapped_strength = remap_img2img_strength(strength)

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
        strength=remapped_strength,
        padding_mask_crop = padding_mask_crop,
        clip_skip=clip_skip
    )

    return {
        "images": [f"/outputs/{name}" for name in filenames]
    }

## SDXL endpoints

@app.post("/api/sdxl/text2img")
async def generate_sdxl_text2img(req: SdxlGenerateRequest, request: Request):
    # logger.info("SDXL request JSON: %s", await request.json())
    # logger.info("Parsed SDXL seed: %s", req.seed)
    prepare_model(req.model)

    return run_sdxl_text2img(req.model_dump())

@app.post("/api/sdxl/img2img")
async def generate_sdxl_img2img(
    initial_image: UploadFile = File(...),
    strength: float = Form(0.75),
    prompt: str = Form(...),
    negative_prompt: str = Form(DEFAULTS["negative_prompt"]),
    steps: int = Form(DEFAULTS["steps"]),
    guidance_scale: float = Form(DEFAULTS["cfg"]),
    width: int = Form(1024),
    height: int = Form(1024),
    seed: int | None = Form(None),
    num_images: int = Form(1),
    model: str | None = Form(None),
    clip_skip: int = Form(1),
):
    if not 0 <= strength <= 1:
        raise HTTPException(status_code=400, detail="Strength must be between 0 and 1.")

    image_bytes = await initial_image.read()
    try:
        init_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    init_image = init_image.resize((width, height))
    remapped_strength = remap_img2img_strength(strength)
    prepare_model(model)

    return run_sdxl_img2img(
        initial_image=init_image,
        strength=remapped_strength,
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        seed=seed,
        model=model,
        num_images=num_images,
        clip_skip=clip_skip,
    )


@app.post("/api/sdxl/inpaint")
async def generate_sdxl_inpaint(
    initial_image: UploadFile = File(...),
    mask_image: UploadFile = File(...),
    strength: float = Form(0.5),
    prompt: str = Form(...),
    negative_prompt: str = Form(DEFAULTS["negative_prompt"]),
    steps: int = Form(DEFAULTS["steps"]),
    guidance_scale: float = Form(DEFAULTS["cfg"]),
    seed: int | None = Form(None),
    num_images: int = Form(1),
    model: str | None = Form(None),
    padding_mask_crop: int = Form(32),
    clip_skip: int = Form(1),
):
    if not 0 <= strength <= 1:
        raise HTTPException(status_code=400, detail="Strength must be between 0 and 1.")

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


    remapped_strength = remap_img2img_strength(strength)
    if mask.size != init_image.size:
        mask = mask.resize(init_image.size, resample=Image.NEAREST)
    prepare_model(model)

    return run_sdxl_inpaint(
        initial_image=init_image,
        mask_image=mask,
        strength=remapped_strength,
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        model=model,
        num_images=num_images,
        padding_mask_crop=padding_mask_crop,
        clip_skip=clip_skip,
    )
    
## Z-Image Endpoints
@app.post("/api/z-image/text2img")
async def generate_z_image_text2img(req: ZImageGenerateRequest, request: Request):
    # logger.info("Z-Image request JSON: %s", await request.json())
    # logger.info("Parsed Z-Image seed: %s", req.seed)
    prepare_model(req.model)

    return run_z_image_text2img(req.model_dump())


@app.post("/api/z-image/img2img")
async def generate_z_image_img2img(
    initial_image: UploadFile = File(...),
    strength: float = Form(0.75),
    prompt: str = Form(...),
    negative_prompt: str = Form(DEFAULTS["negative_prompt"]),
    steps: int = Form(DEFAULTS["steps"]),
    guidance_scale: float = Form(DEFAULTS["cfg"]),
    width: int = Form(1024),
    height: int = Form(1024),
    seed: int | None = Form(None),
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
    remapped_strength = remap_img2img_strength(strength)
    init_image = init_image.resize((width, height))
    prepare_model(model)

    return run_z_image_img2img(
        initial_image=init_image,
        strength=remapped_strength,
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        seed=seed,
        model=model,
        num_images=num_images,
    )

## Flux Endpoints
@app.post("/api/flux/text2img")
async def generate_flux_text2img(req: FluxGenerateRequest, request: Request):
    # logger.info("Flux request JSON: %s", await request.json())
    # logger.info("Parsed Flux seed: %s", req.seed)
    prepare_model(req.model)

    return run_flux_text2img(req.model_dump())


@app.post("/api/flux/img2img")
async def generate_flux_img2img(
    initial_image: UploadFile = File(...),
    strength: float = Form(0.75),
    prompt: str = Form(...),
    negative_prompt: str = Form(DEFAULTS["negative_prompt"]),
    steps: int = Form(DEFAULTS["steps"]),
    guidance_scale: float = Form(DEFAULTS["cfg"]),
    width: int = Form(1024),
    height: int = Form(1024),
    seed: int | None = Form(None),
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
    prepare_model(model)

    return run_flux_img2img(
        initial_image=init_image,
        strength=strength,
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        seed=seed,
        model=model,
        num_images=num_images,
    )


@app.post("/api/flux/inpaint")
async def generate_flux_inpaint(
    initial_image: UploadFile = File(...),
    mask_image: UploadFile = File(...),
    strength: float = Form(0.5),
    prompt: str = Form(...),
    negative_prompt: str = Form(DEFAULTS["negative_prompt"]),
    steps: int = Form(DEFAULTS["steps"]),
    guidance_scale: float = Form(DEFAULTS["cfg"]),
    seed: int | None = Form(None),
    num_images: int = Form(1),
    model: str | None = Form(None),
):
    if not 0 <= strength <= 1:
        raise HTTPException(status_code=400, detail="Strength must be between 0 and 1.")

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
    prepare_model(model)

    return run_flux_inpaint(
        initial_image=init_image,
        mask_image=mask,
        strength=strength,
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        model=model,
        num_images=num_images,
    )

## Inpainting related endpoints

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


@app.post("/api/cache/clear")
async def clear_model_cache():
    clear_all_pipelines()
    return {"status": "cleared"}
