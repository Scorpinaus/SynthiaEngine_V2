"""Run a reproducible upstream Diffusers SD 1.5 inpainting baseline."""

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
DEFAULT_PROMPT = "a detailed fantasy castle with a glowing crystal tower, cinematic lighting"
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an SD 1.5 inpainting baseline and metadata."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input-image", type=Path)
    parser.add_argument("--mask-image", type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "sd15_inpaint_baseline",
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
    for label, path in (("Input", args.input_image), ("Mask", args.mask_image)):
        if path and not path.is_file():
            raise FileNotFoundError(f"{label} image not found: {path}")


def make_fixtures(width: int, height: int) -> tuple[Image.Image, Image.Image]:
    image = Image.new("RGB", (width, height), "#efb366")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, height * 2 // 3, width, height), fill="#355c45")
    draw.polygon(
        ((width // 5, height * 2 // 3), (width // 2, height // 5), (width * 4 // 5, height * 2 // 3)),
        fill="#56697a",
    )
    mask = Image.new("L", (width, height), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle((width * 2 // 5, height // 4, width * 3 // 5, height * 2 // 3), fill=255)
    return image, mask


def main() -> int:
    args = parse_args()
    validate_args(args)

    import torch
    from diffusers import StableDiffusionInpaintPipeline

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    dtype = torch.float16 if device == "cuda" else torch.float32
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fixture_image, fixture_mask = make_fixtures(args.width, args.height)
    if args.input_image:
        with Image.open(args.input_image) as source:
            init_image = source.convert("RGB").resize((args.width, args.height))
        input_source = str(args.input_image.resolve())
    else:
        init_image = fixture_image
        input_source = "built-in deterministic fixture"
    if args.mask_image:
        with Image.open(args.mask_image) as source:
            mask_image = source.convert("L").resize((args.width, args.height))
        mask_source = str(args.mask_image.resolve())
    else:
        mask_image = fixture_mask
        mask_source = "built-in deterministic fixture"
    input_path = args.output_dir / "input.png"
    mask_path = args.output_dir / "mask.png"
    init_image.save(input_path)
    mask_image.save(mask_path)

    started = time.perf_counter()
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(args.model, torch_dtype=dtype)
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
        mask_image=mask_image,
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
        "operation": "inpaint",
        "model": args.model,
        "pipeline_class": type(pipeline).__name__,
        "scheduler_class": type(pipeline.scheduler).__name__,
        "input_source": input_source,
        "mask_source": mask_source,
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
    print(f"Input:    {input_path.resolve()}")
    print(f"Mask:     {mask_path.resolve()}")
    print(f"Image:    {image_path.resolve()}")
    print(f"Metadata: {metadata_path.resolve()}")
    print(f"SHA-256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
