"""Run a reproducible, upstream Diffusers SD 1.5 text-to-image baseline.

This script deliberately does not import SynthaEngine code. It is intended to
remain a control when application functions or Python packages are changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


DEFAULT_MODEL = r"D:\diffusion\diffusers\raemumix_v90"
DEFAULT_PROMPT = "a photo of an astronaut riding a horse on mars"
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an SD 1.5 baseline image and machine-readable metadata."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="auto selects CUDA when available, otherwise CPU",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "sd15_txt2img_baseline",
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
    if args.width < 64 or args.height < 64:
        raise ValueError("--width and --height must be at least 64")
    if args.width % 8 or args.height % 8:
        raise ValueError("--width and --height must be divisible by 8")


def main() -> int:
    args = parse_args()
    validate_args(args)

    # Keep heavyweight imports below argument parsing so `--help` stays useful
    # even while an environment is being repaired.
    import torch
    from diffusers import StableDiffusionPipeline

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")

    dtype = torch.float16 if device == "cuda" else torch.float32
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    pipeline = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipeline = pipeline.to(device)
    if device == "cpu":
        pipeline.enable_attention_slicing()
    load_seconds = time.perf_counter() - started

    generator = torch.Generator(device=device).manual_seed(args.seed)
    generation_started = time.perf_counter()
    result = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        generator=generator,
    )
    generation_seconds = time.perf_counter() - generation_started

    if not result.images:
        raise RuntimeError("The pipeline returned no images")
    image = result.images[0]
    image_path = args.output_dir / "baseline.png"
    metadata_path = args.output_dir / "baseline.json"
    image.save(image_path)

    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "pipeline_class": type(pipeline).__name__,
        "scheduler_class": type(pipeline.scheduler).__name__,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "width": args.width,
        "height": args.height,
        "device": device,
        "dtype": str(dtype),
        "load_seconds": round(load_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "image_sha256": image_sha256,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in ("torch", "diffusers", "transformers", "accelerate", "Pillow")
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Image:    {image_path.resolve()}")
    print(f"Metadata: {metadata_path.resolve()}")
    print(f"SHA-256:  {image_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
