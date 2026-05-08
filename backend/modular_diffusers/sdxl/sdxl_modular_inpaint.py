"""Run a simple inpaint inference with Diffusers' built-in SDXL modular pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from diffusers import ModularPipeline

from backend.modular_diffusers.sdxl.adapters import (
    add_sdxl_adapter_arguments,
    apply_sdxl_modular_adapters_from_args,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "outputs" / "sdxl_modular_inpaint.png"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SDXL Modular Diffusers inpaint inference.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the input image.")
    parser.add_argument("--mask-image", type=Path, required=True, help="Path to the inpaint mask image.")
    parser.add_argument("--prompt", default="replace the masked area with a glowing portal", help="Positive prompt text.")
    parser.add_argument("--negative-prompt", default="blurry, distorted, low quality", help="Negative prompt text.")
    parser.add_argument("--strength", type=float, default=1.0, help="Inpaint denoising strength.")
    parser.add_argument("--steps", type=int, default=30, help="Number of denoising steps.")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="Classifier-free guidance scale.")
    parser.add_argument("--width", type=int, default=1024, help="Output image width.")
    parser.add_argument("--height", type=int, default=1024, help="Output image height.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument(
        "--padding-mask-crop",
        type=int,
        default=None,
        help="Optional padding around the detected mask crop before inpainting.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the generated image will be saved.",
    )
    parser.add_argument(
        "--model",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Diffusers model repo or local model path to load as a ModularPipeline.",
    )
    add_sdxl_adapter_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    generator = torch.Generator(device=device).manual_seed(args.seed)

    init_image = Image.open(args.image).convert("RGB")
    mask_image = Image.open(args.mask_image).convert("L")

    pipe = ModularPipeline.from_pretrained(args.model)
    pipe.load_components(torch_dtype=dtype)
    apply_sdxl_modular_adapters_from_args(pipe, args)
    pipe.to(device)

    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=init_image,
        mask_image=mask_image,
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        padding_mask_crop=args.padding_mask_crop,
        generator=generator,
        output="images",
    )[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)

    print(f"Saved image to: {args.output}")


if __name__ == "__main__":
    main()
