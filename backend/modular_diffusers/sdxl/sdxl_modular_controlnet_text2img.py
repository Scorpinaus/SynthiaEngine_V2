"""Run ControlNet text-to-image inference with Diffusers' SDXL modular pipeline."""

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
from backend.modular_diffusers.sdxl.controlnet import (
    add_sdxl_controlnet_arguments,
    load_sdxl_controlnet_component,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "outputs" / "sdxl_modular_controlnet_text2img.png"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SDXL Modular Diffusers ControlNet text-to-image inference.")
    parser.add_argument("--prompt", default="a cinematic portrait of an astronaut in a neon jungle", help="Positive prompt text.")
    parser.add_argument("--negative-prompt", default="blurry, distorted, low quality", help="Negative prompt text.")
    parser.add_argument("--steps", type=int, default=30, help="Number of denoising steps.")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="Classifier-free guidance scale.")
    parser.add_argument("--width", type=int, default=1024, help="Output image width.")
    parser.add_argument("--height", type=int, default=1024, help="Output image height.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
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
    add_sdxl_controlnet_arguments(parser)
    add_sdxl_adapter_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    generator = torch.Generator(device=device).manual_seed(args.seed)

    control_image = Image.open(args.control_image).convert("RGB")

    pipe = ModularPipeline.from_pretrained(args.model)
    pipe.load_components(torch_dtype=dtype)
    load_sdxl_controlnet_component(pipe, model=args.controlnet_model, torch_dtype=dtype)
    apply_sdxl_modular_adapters_from_args(pipe, args)
    pipe.to(device)

    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        control_image=control_image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        generator=generator,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        guess_mode=args.guess_mode,
        control_guidance_start=args.control_guidance_start,
        control_guidance_end=args.control_guidance_end,
        output="images",
    )[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)

    print(f"Saved image to: {args.output}")


if __name__ == "__main__":
    main()
