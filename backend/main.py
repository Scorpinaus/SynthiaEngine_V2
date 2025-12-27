import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.sd15_pipeline import generate_images
from backend.config import DEFAULTS

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
        num_images = req.num_images,
    )

    return {
        "images": [f"/outputs/{name}" for name in filenames]
    }
