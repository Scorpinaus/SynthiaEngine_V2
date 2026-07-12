"""Run a reproducible upstream Diffusers SDXL text-to-image baseline.

This control intentionally imports no SynthaEngine modules. It loads only local
model files so package or application changes can be evaluated without silently
downloading a different model revision.
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


DEFAULT_MODEL = Path(r"D:\diffusion\diffusers\stable-diffusion-xl-base-1-0")
DEFAULT_PROMPT = "a cinematic photograph of an astronaut riding a horse on Mars"
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an SDXL baseline image and machine-readable metadata."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="auto selects CUDA when available, otherwise CPU",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "sdxl_txt2img_baseline",
    )
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def validate_args(args: argparse.Namespace) -> None:
    if not (args.model / "model_index.json").is_file():
        raise FileNotFoundError(f"Diffusers model not found at: {args.model}")
    model_index = json.loads((args.model / "model_index.json").read_text(encoding="utf-8"))
    if model_index.get("_class_name") != "StableDiffusionXLPipeline":
        raise ValueError(
            f"Expected an SDXL model, found {model_index.get('_class_name')!r} at {args.model}"
        )
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    if args.width < 64 or args.height < 64:
        raise ValueError("--width and --height must be at least 64")
    if args.width % 8 or args.height % 8:
        raise ValueError("--width and --height must be divisible by 8")


def main() -> int:
    args = parse_args()
    validate_args(args)

    import torch
    from diffusers import StableDiffusionXLPipeline

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")

    dtype = torch.float16 if device == "cuda" else torch.float32
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        str(args.model), torch_dtype=dtype, local_files_only=True
    ).to(device)
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

    image_path = args.output_dir / "baseline.png"
    metadata_path = args.output_dir / "baseline.json"
    result.images[0].save(image_path)
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
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
