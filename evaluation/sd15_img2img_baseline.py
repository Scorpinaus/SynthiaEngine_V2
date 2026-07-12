"""Run a reproducible upstream Diffusers SD 1.5 image-to-image baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from PIL import Image, ImageDraw


DEFAULT_MODEL = r"D:\diffusion\diffusers\raemumix_v90"
DEFAULT_PROMPT = "a detailed fantasy castle at sunset, cinematic lighting"
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an SD 1.5 img2img baseline and metadata."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input-image", type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "sd15_img2img_baseline",
    )
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def validate_args(args: argparse.Namespace) -> None:
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    if not 0.0 < args.strength <= 1.0:
        raise ValueError("--strength must be within (0, 1]")
    if args.width < 64 or args.height < 64 or args.width % 8 or args.height % 8:
        raise ValueError("--width and --height must be at least 64 and divisible by 8")
    if args.input_image and not args.input_image.is_file():
        raise FileNotFoundError(f"Input image not found: {args.input_image}")


def make_fixture(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), "#efb366")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, height * 2 // 3, width, height), fill="#355c45")
    draw.polygon(
        ((width // 5, height * 2 // 3), (width // 2, height // 5), (width * 4 // 5, height * 2 // 3)),
        fill="#56697a",
    )
    draw.ellipse((width * 3 // 4, height // 10, width * 9 // 10, height // 4), fill="#fff1ad")
    return image


def main() -> int:
    args = parse_args()
    validate_args(args)

    import torch
    from diffusers import StableDiffusionImg2ImgPipeline

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    dtype = torch.float16 if device == "cuda" else torch.float32
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_image:
        with Image.open(args.input_image) as source:
            init_image = source.convert("RGB").resize((args.width, args.height))
        source_description = str(args.input_image.resolve())
    else:
        init_image = make_fixture(args.width, args.height)
        source_description = "built-in deterministic fixture"
    init_path = args.output_dir / "input.png"
    init_image.save(init_path)

    started = time.perf_counter()
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipeline = pipeline.to(device)
    if device == "cpu":
        pipeline.enable_attention_slicing()
    load_seconds = time.perf_counter() - started

    generator = torch.Generator(device=device).manual_seed(args.seed)
    generation_started = time.perf_counter()
    result = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=init_image,
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )
    generation_seconds = time.perf_counter() - generation_started
    if not result.images:
        raise RuntimeError("The pipeline returned no images")

    image_path = args.output_dir / "baseline.png"
    metadata_path = args.output_dir / "baseline.json"
    result.images[0].save(image_path)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "img2img",
        "model": args.model,
        "pipeline_class": type(pipeline).__name__,
        "scheduler_class": type(pipeline.scheduler).__name__,
        "input_source": source_description,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "strength": args.strength,
        "width": args.width,
        "height": args.height,
        "device": device,
        "dtype": str(dtype),
        "load_seconds": round(load_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "image_sha256": digest,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in ("torch", "diffusers", "transformers", "accelerate", "Pillow")},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Input:    {init_path.resolve()}")
    print(f"Image:    {image_path.resolve()}")
    print(f"Metadata: {metadata_path.resolve()}")
    print(f"SHA-256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
