"""Run PixelDiT through Modular Diffusers with a ComponentsManager."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from diffusers import ComponentsManager

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.modular_diffusers.pixeldit import PixelDiTModularPipeline
from backend.modular_diffusers.pixeldit.encoders import DEFAULT_CHI_PROMPT


DEFAULT_MODEL_DIR = Path(r"D:\diffusion\diffusers\PixelDiT-Diffusers")
DEFAULT_PROMPT = (
    "a glass greenhouse at sunrise, lush plants, soft cinematic light, detailed architecture, "
    "natural colors"
)
DEFAULT_NEGATIVE_PROMPT = "low quality, blurry, distorted, oversaturated, artifacts"


def _dtype(value: str) -> torch.dtype:
    values = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return values[value]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"Expected one of {', '.join(values)}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=2.75)
    parser.add_argument("--flow-shift", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--collection", default="pixeldit")
    parser.add_argument("--transformer-dtype", type=_dtype, default=torch.bfloat16)
    parser.add_argument("--text-encoder-dtype", type=_dtype, default=torch.bfloat16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hf-modules-cache", type=Path, default=None)
    parser.add_argument("--use-chi-prompt", action="store_true")
    parser.add_argument("--chi-prompt-file", type=Path, default=None)
    parser.add_argument("--enable-auto-cpu-offload", action="store_true")
    parser.add_argument("--memory-reserve-margin", default="3GB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    output_dir = (args.output_dir or (model_dir / "test_outputs")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.hf_modules_cache is not None:
        hf_modules_cache = args.hf_modules_cache.resolve()
        hf_modules_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_MODULES_CACHE"] = str(hf_modules_cache)
    use_chi_prompt = bool(args.use_chi_prompt or args.chi_prompt_file is not None)
    chi_prompt = DEFAULT_CHI_PROMPT
    chi_prompt_source = "default"
    if args.chi_prompt_file is not None:
        chi_prompt_path = args.chi_prompt_file.resolve()
        chi_prompt = chi_prompt_path.read_text(encoding="utf-8")
        chi_prompt_source = str(chi_prompt_path)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    manager = ComponentsManager()
    pipe = PixelDiTModularPipeline.from_pretrained(
        model_dir,
        trust_remote_code=True,
        components_manager=manager,
        collection=args.collection,
    )

    pipe.load_components(names="tokenizer")
    pipe.load_components(names="text_encoder", torch_dtype=args.text_encoder_dtype)
    pipe.text_encoder.to("cpu")

    pipe.load_components(names="transformer", torch_dtype=args.transformer_dtype)
    pipe.transformer.to(device=device, dtype=args.transformer_dtype)

    if args.enable_auto_cpu_offload:
        manager.enable_auto_cpu_offload(device=device, memory_reserve_margin=args.memory_reserve_margin)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    started = time.perf_counter()
    images = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        use_chi_prompt=use_chi_prompt,
        chi_prompt=chi_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        flow_shift=args.flow_shift,
        generator=generator,
        output_type="pil",
        output="images",
    )
    elapsed_seconds = time.perf_counter() - started

    chi_suffix = "_chi" if use_chi_prompt else ""
    stem = f"pixeldit_components_manager{chi_suffix}_{args.width}x{args.height}_{args.steps}step_seed{args.seed}"
    image_path = output_dir / f"{stem}.png"
    report_path = output_dir / f"{stem}_report.json"
    images[0].save(image_path)

    report = {
        "model_dir": str(model_dir),
        "image_path": str(image_path),
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "use_chi_prompt": use_chi_prompt,
        "chi_prompt_source": chi_prompt_source if use_chi_prompt else None,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "flow_shift": args.flow_shift,
        "seed": args.seed,
        "device": str(device),
        "transformer_dtype": str(args.transformer_dtype),
        "text_encoder_dtype": str(args.text_encoder_dtype),
        "collection": args.collection,
        "components": manager.get_ids(
            names=["tokenizer", "text_encoder", "transformer"],
            collection=args.collection,
        ),
        "auto_cpu_offload": bool(args.enable_auto_cpu_offload),
        "elapsed_seconds": elapsed_seconds,
    }
    if device.type == "cuda":
        report["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        report["cuda_peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / 1024 / 1024

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved image: {image_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
