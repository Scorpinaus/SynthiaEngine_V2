from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_PROMPT = "a small workshop robot repairing a memory gauge, detailed but compact"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "modular_flux_tests"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    pipeline: str
    prompt: bool = True
    image: bool = False
    embeds: bool = False
    strength: bool = False


CASES: dict[str, CaseSpec] = {
    "flux-text2img": CaseSpec("flux-text2img", "flux"),
    "flux-img2img": CaseSpec("flux-img2img", "flux", image=True, strength=True),
    "flux-embeds2img": CaseSpec("flux-embeds2img", "flux", prompt=False, embeds=True),
    "flux-img2img-embeds": CaseSpec(
        "flux-img2img-embeds",
        "flux",
        prompt=False,
        image=True,
        embeds=True,
        strength=True,
    ),
    "kontext-text2img": CaseSpec("kontext-text2img", "kontext"),
    "kontext-image": CaseSpec("kontext-image", "kontext", image=True),
    "kontext-embeds2img": CaseSpec("kontext-embeds2img", "kontext", prompt=False, embeds=True),
    "kontext-image-embeds": CaseSpec(
        "kontext-image-embeds",
        "kontext",
        prompt=False,
        image=True,
        embeds=True,
    ),
}


CASE_ALIASES = {
    "text2img": {"flux": "flux-text2img", "kontext": "kontext-text2img"},
    "img2img": {"flux": "flux-img2img"},
    "image": {"kontext": "kontext-image"},
    "embeds2img": {"flux": "flux-embeds2img", "kontext": "kontext-embeds2img"},
    "img2img-embeds": {"flux": "flux-img2img-embeds"},
    "image-embeds": {"kontext": "kontext-image-embeds"},
}


PipelineLoader = Callable[[str, argparse.Namespace], tuple[Any, dict[str, Any]]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure local custom FluxModular pipeline wall time, CUDA memory, and process RSS."
    )
    parser.add_argument(
        "--case",
        default="flux-text2img",
        help=(
            "Case to run. Use a full case name, a short alias like text2img/img2img, "
            "or all."
        ),
    )
    parser.add_argument(
        "--pipeline",
        choices=("flux", "kontext", "all"),
        default="flux",
        help="Pipeline family for --case all or short aliases.",
    )
    parser.add_argument("--model", default="black-forest-labs/FLUX.1-dev", help="Flux model path or HF id.")
    parser.add_argument("--kontext-model", default=None, help="Optional Kontext model path or HF id.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to generate.")
    parser.add_argument("--prompt-2", default=None, help="Optional secondary T5 prompt.")
    parser.add_argument("--image", type=Path, default=None, help="Optional input image for image-conditioned cases.")
    parser.add_argument("--width", type=int, default=768, help="Output width.")
    parser.add_argument("--height", type=int, default=768, help="Output height.")
    parser.add_argument("--max-area", type=int, default=None, help="Kontext max area. Defaults to width * height.")
    parser.add_argument("--steps", type=int, default=8, help="Inference steps.")
    parser.add_argument("--guidance-scale", type=float, default=3.5, help="Flux guidance scale.")
    parser.add_argument("--strength", type=float, default=0.6, help="Flux img2img strength.")
    parser.add_argument("--seed", type=int, default=12345, help="Base seed. Each run increments it by one.")
    parser.add_argument("--num-images", type=int, default=1, help="Images per prompt.")
    parser.add_argument("--runs", type=int, default=1, help="Measured runs.")
    parser.add_argument("--warmup-runs", type=int, default=0, help="Warmups excluded from summary.")
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Component dtype.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="Execution device.")
    parser.add_argument(
        "--offload",
        choices=("auto", "none"),
        default="auto",
        help="Use custom low-memory offload helper or keep components on the selected device.",
    )
    parser.add_argument(
        "--memory-reserve-margin",
        default="3GB",
        help="Reserve margin for Modular Diffusers component-manager offload.",
    )
    parser.add_argument("--decode-chunk-size", type=int, default=1, help="VAE decode chunk size.")
    parser.add_argument(
        "--vae-decode-device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Optional VAE decode device.",
    )
    parser.add_argument("--max-sequence-length", type=int, default=None, help="Optional T5 max sequence length.")
    parser.add_argument("--output-type", choices=("pil", "latent", "pt", "np"), default="pil")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated images.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--revision", default=None, help="Optional model revision.")
    parser.add_argument("--variant", default=None, help="Optional model variant.")
    parser.add_argument("--token", default=None, help="Optional Hugging Face token.")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download model files.")
    parser.add_argument("--reload-per-case", action="store_true", help="Reload the pipeline for every case.")
    parser.add_argument("--rss-sample-interval", type=float, default=0.05, help="Peak RSS sampling interval.")
    parser.add_argument(
        "--low-memory-sequential-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate num-images sequentially instead of batching.",
    )
    parser.add_argument(
        "--low-memory-transformer-buffers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable reusable Flux transformer concat buffers.",
    )
    parser.add_argument(
        "--low-memory-transformer-attention-buffers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable reusable attention Q/K/V concat buffers.",
    )
    parser.add_argument(
        "--low-memory-transformer-single-block-buffers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable reusable FluxSingleTransformerBlock concat buffers.",
    )
    parser.add_argument(
        "--low-memory-eager-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Move completed heavy components to CPU between phases.",
    )
    parser.add_argument(
        "--low-memory-prune-intermediates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop denoise-only intermediates before decode.",
    )
    return parser.parse_args(argv)


def resolve_cases(args: argparse.Namespace) -> list[CaseSpec]:
    case_name = args.case.lower().replace("_", "-")
    if case_name == "all":
        return [case for case in CASES.values() if args.pipeline == "all" or case.pipeline == args.pipeline]
    if case_name in CASES:
        case = CASES[case_name]
        if args.pipeline != "all" and case.pipeline != args.pipeline:
            raise ValueError(f"Case '{case_name}' belongs to pipeline '{case.pipeline}', not '{args.pipeline}'.")
        return [case]
    if case_name in CASE_ALIASES:
        aliases = CASE_ALIASES[case_name]
        if args.pipeline == "all":
            if len(aliases) != 1:
                raise ValueError(f"Short case alias '{case_name}' is ambiguous with --pipeline all.")
            resolved = next(iter(aliases.values()))
        elif args.pipeline not in aliases:
            raise ValueError(f"Case alias '{case_name}' is not valid for pipeline '{args.pipeline}'.")
        else:
            resolved = aliases[args.pipeline]
        return [CASES[resolved]]
    valid = ", ".join(["all", *CASES.keys(), *CASE_ALIASES.keys()])
    raise ValueError(f"Unknown case '{args.case}'. Valid cases: {valid}")


def resolve_device(args: argparse.Namespace):
    import torch

    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(args.device)


def resolve_dtype(args: argparse.Namespace):
    import torch

    if args.torch_dtype == "auto":
        return None
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.torch_dtype]


def model_load_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for name in ("revision", "variant", "token"):
        value = getattr(args, name)
        if value is not None:
            kwargs[name] = value
    if args.cache_dir is not None:
        kwargs["cache_dir"] = str(args.cache_dir)
    if args.local_files_only:
        kwargs["local_files_only"] = True
    return kwargs


def create_synthetic_image(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), "#20242c")
    draw = ImageDraw.Draw(image)
    for y in range(height):
        shade = int(40 + 120 * (y / max(1, height - 1)))
        draw.line([(0, y), (width, y)], fill=(shade, 80, 120))
    margin = max(16, min(width, height) // 8)
    draw.rectangle((margin, margin, width - margin, height - margin), outline="#f4d35e", width=4)
    draw.ellipse((width // 3, height // 3, width * 2 // 3, height * 2 // 3), fill="#4ecdc4")
    return image


def load_input_image(args: argparse.Namespace) -> Image.Image:
    if args.image is None:
        return create_synthetic_image(args.width, args.height)
    with Image.open(args.image) as image:
        return image.convert("RGB")


def make_generator(seed: int):
    import torch

    return torch.Generator(device="cpu").manual_seed(int(seed))


def get_process_rss_mb() -> float | None:
    try:
        import psutil  # type: ignore[import-not-found]

        return psutil.Process().memory_info().rss / 1024**2
    except Exception:
        return None


class PeakRSSSampler:
    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = max(0.01, float(interval_seconds))
        self.peak_mb: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            import psutil  # type: ignore[import-not-found]

            self._process = psutil.Process()
        except Exception:
            self._process = None

    def __enter__(self):
        if self._process is None:
            return self
        self.peak_mb = self._process.memory_info().rss / 1024**2
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._process is not None:
            self.peak_mb = max(self.peak_mb or 0.0, self._process.memory_info().rss / 1024**2)

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            current = self._process.memory_info().rss / 1024**2
            self.peak_mb = max(self.peak_mb or 0.0, current)


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def reset_cuda_memory_stats() -> None:
    if not cuda_available():
        return
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def synchronize_cuda() -> None:
    if not cuda_available():
        return
    import torch

    torch.cuda.synchronize()


def get_cuda_memory_stats() -> dict[str, float | bool | None]:
    if not cuda_available():
        return {
            "cuda_available": False,
            "cuda_max_allocated_mb": None,
            "cuda_max_reserved_mb": None,
            "cuda_allocated_after_mb": None,
            "cuda_reserved_after_mb": None,
        }

    import torch

    return {
        "cuda_available": True,
        "cuda_max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "cuda_max_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
        "cuda_allocated_after_mb": torch.cuda.memory_allocated() / 1024**2,
        "cuda_reserved_after_mb": torch.cuda.memory_reserved() / 1024**2,
    }


def default_pipeline_loader(kind: str, args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    import torch

    from custom_pipelines.FluxModular import (
        FluxKontextModularPipeline,
        FluxModularPipeline,
        enable_low_memory_flux_modular,
    )

    pipeline_cls = FluxKontextModularPipeline if kind == "kontext" else FluxModularPipeline
    model = args.kontext_model if kind == "kontext" and args.kontext_model else args.model
    dtype = resolve_dtype(args)
    device = resolve_device(args)
    load_kwargs = model_load_kwargs(args)
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype

    reset_cuda_memory_stats()
    rss_before_mb = get_process_rss_mb()
    start = time.perf_counter()
    pipe = pipeline_cls.from_pretrained(model, **load_kwargs)
    pipe.load_components(**load_kwargs)
    if args.offload == "auto":
        offload_mode = enable_low_memory_flux_modular(
            pipe,
            device=device,
            memory_reserve_margin=args.memory_reserve_margin,
        )
    else:
        to_kwargs: dict[str, Any] = {"device": device}
        if dtype is not None:
            to_kwargs["dtype"] = dtype
        pipe.to(**to_kwargs)
        offload_mode = "none"
    synchronize_cuda()
    load_seconds = time.perf_counter() - start

    return pipe, {
        "pipeline": kind,
        "model": model,
        "torch_dtype": str(dtype or "component_default"),
        "device": str(device),
        "offload_mode": offload_mode,
        "load_seconds": load_seconds,
        "load_rss_before_mb": rss_before_mb,
        "load_rss_after_mb": get_process_rss_mb(),
        **get_cuda_memory_stats(),
    }


def precompute_prompt_embeds(pipe: Any, args: argparse.Namespace) -> tuple[Any, Any]:
    import torch

    from custom_pipelines.FluxModular.low_memory import LowMemoryFluxTextEncoderStep

    device = getattr(pipe, "_execution_device", None) or resolve_device(args)
    with torch.no_grad():
        return LowMemoryFluxTextEncoderStep.encode_prompt(
            pipe,
            prompt=args.prompt,
            prompt_2=args.prompt_2,
            device=device,
            max_sequence_length=args.max_sequence_length,
        )


def build_case_kwargs(
    args: argparse.Namespace,
    case: CaseSpec,
    pipe: Any,
    *,
    run_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prepare_start = time.perf_counter()
    kwargs: dict[str, Any] = {
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "num_images_per_prompt": args.num_images,
        "generator": make_generator(run_seed),
        "output_type": args.output_type,
        "decode_chunk_size": args.decode_chunk_size,
        "low_memory_sequential_images": args.low_memory_sequential_images,
        "low_memory_transformer_buffers": args.low_memory_transformer_buffers,
        "low_memory_transformer_attention_buffers": args.low_memory_transformer_attention_buffers,
        "low_memory_transformer_single_block_buffers": args.low_memory_transformer_single_block_buffers,
        "low_memory_eager_offload": args.low_memory_eager_offload,
        "low_memory_prune_intermediates": args.low_memory_prune_intermediates,
    }
    if args.prompt_2 is not None and case.prompt:
        kwargs["prompt_2"] = args.prompt_2
    if args.max_sequence_length is not None:
        kwargs["max_sequence_length"] = args.max_sequence_length
    if args.vae_decode_device != "auto":
        kwargs["vae_decode_device"] = args.vae_decode_device
    if args.max_area is not None:
        kwargs["max_area"] = args.max_area
    elif case.pipeline == "kontext" and case.image:
        kwargs["max_area"] = args.width * args.height
    if case.strength:
        kwargs["strength"] = args.strength
    if case.image:
        kwargs["image"] = load_input_image(args)
    if case.prompt:
        kwargs["prompt"] = args.prompt
    if case.embeds:
        embed_start = time.perf_counter()
        prompt_embeds, pooled_prompt_embeds = precompute_prompt_embeds(pipe, args)
        kwargs["prompt_embeds"] = prompt_embeds
        kwargs["pooled_prompt_embeds"] = pooled_prompt_embeds
        embed_seconds = time.perf_counter() - embed_start
    else:
        embed_seconds = None

    prepare_seconds = time.perf_counter() - prepare_start
    return kwargs, {"prepare_seconds": prepare_seconds, "embed_seconds": embed_seconds}


def extract_images(result: Any) -> Any:
    if hasattr(result, "get"):
        images = result.get("images")
        if images is not None:
            return images
    return getattr(result, "images", None)


def save_images(images: Any, output_dir: Path, case_name: str, run_label: str) -> list[str]:
    if images is None:
        return []
    if not isinstance(images, list):
        images = [images]
    saved_paths: list[str] = []
    case_dir = output_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        if not isinstance(image, Image.Image):
            continue
        path = case_dir / f"{run_label}_{index:02d}.png"
        image.save(path)
        saved_paths.append(str(path))
    return saved_paths


def run_single_case(
    args: argparse.Namespace,
    *,
    pipe: Any,
    case: CaseSpec,
    run_index: int,
    kind: str,
) -> dict[str, Any]:
    run_seed = int(args.seed) + run_index - 1
    try:
        kwargs, prepare_stats = build_case_kwargs(args, case, pipe, run_seed=run_seed)
    except Exception as exc:
        return {
            "case": case.name,
            "pipeline": case.pipeline,
            "kind": kind,
            "run_index": run_index,
            "seed": run_seed,
            "status": "error",
            "phase": "prepare",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    reset_cuda_memory_stats()
    rss_before_mb = get_process_rss_mb()
    start = time.perf_counter()
    with PeakRSSSampler(args.rss_sample_interval) as rss_sampler:
        try:
            result = pipe(**kwargs)
            synchronize_cuda()
            elapsed_seconds = time.perf_counter() - start
            images = extract_images(result)
            run_label = f"{kind}_{run_index:02d}"
            image_paths = save_images(images, args.output_dir, case.name, run_label)
            return {
                "case": case.name,
                "pipeline": case.pipeline,
                "kind": kind,
                "run_index": run_index,
                "seed": run_seed,
                "status": "success",
                "elapsed_seconds": elapsed_seconds,
                "rss_before_mb": rss_before_mb,
                "rss_after_mb": get_process_rss_mb(),
                "rss_peak_sampled_mb": rss_sampler.peak_mb,
                "image_paths": image_paths,
                **prepare_stats,
                **get_cuda_memory_stats(),
            }
        except Exception as exc:
            synchronize_cuda()
            return {
                "case": case.name,
                "pipeline": case.pipeline,
                "kind": kind,
                "run_index": run_index,
                "seed": run_seed,
                "status": "error",
                "phase": "inference",
                "elapsed_seconds": time.perf_counter() - start,
                "rss_before_mb": rss_before_mb,
                "rss_after_mb": get_process_rss_mb(),
                "rss_peak_sampled_mb": rss_sampler.peak_mb,
                "error_type": type(exc).__name__,
                "error": str(exc),
                **prepare_stats,
                **get_cuda_memory_stats(),
            }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [run for run in runs if run.get("kind") == "measured"]
    successes = [run for run in measured if run.get("status") == "success"]
    failures = [run for run in measured if run.get("status") != "success"]

    def values(name: str) -> list[float]:
        return [float(run[name]) for run in successes if run.get(name) is not None]

    elapsed = values("elapsed_seconds")
    cuda_allocated = values("cuda_max_allocated_mb")
    cuda_reserved = values("cuda_max_reserved_mb")
    rss_peak = values("rss_peak_sampled_mb")
    rss_after = values("rss_after_mb")

    return {
        "runs": len(measured),
        "successes": len(successes),
        "failures": len(failures),
        "avg_elapsed_seconds": sum(elapsed) / len(elapsed) if elapsed else None,
        "min_elapsed_seconds": min(elapsed) if elapsed else None,
        "max_elapsed_seconds": max(elapsed) if elapsed else None,
        "max_cuda_allocated_mb": max(cuda_allocated) if cuda_allocated else None,
        "max_cuda_reserved_mb": max(cuda_reserved) if cuda_reserved else None,
        "max_rss_peak_sampled_mb": max(rss_peak) if rss_peak else None,
        "max_rss_after_mb": max(rss_after) if rss_after else None,
    }


def run_measurement(
    args: argparse.Namespace,
    *,
    pipeline_loader: PipelineLoader | None = None,
) -> dict[str, Any]:
    loader = pipeline_loader or default_pipeline_loader
    cases = resolve_cases(args)
    loads: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    loaded: dict[str, tuple[Any, dict[str, Any]]] = {}

    for case in cases:
        if args.reload_per_case or case.pipeline not in loaded:
            pipe, load_stats = loader(case.pipeline, args)
            loaded[case.pipeline] = (pipe, load_stats)
            loads.append(load_stats)
        else:
            pipe, _ = loaded[case.pipeline]

        for index in range(args.warmup_runs):
            runs.append(run_single_case(args, pipe=pipe, case=case, run_index=index + 1, kind="warmup"))
        for index in range(args.runs):
            runs.append(run_single_case(args, pipe=pipe, case=case, run_index=index + 1, kind="measured"))

    result = {
        "tool": "measure_flux_modular",
        "cases": [case.name for case in cases],
        "settings": {
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "strength": args.strength,
            "num_images": args.num_images,
            "torch_dtype": args.torch_dtype,
            "device": args.device,
            "offload": args.offload,
            "low_memory_sequential_images": args.low_memory_sequential_images,
            "low_memory_transformer_buffers": args.low_memory_transformer_buffers,
            "decode_chunk_size": args.decode_chunk_size,
            "vae_decode_device": args.vae_decode_device,
        },
        "loads": loads,
        "summary": summarize_runs(runs),
        "runs": runs,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def print_result(result: dict[str, Any]) -> None:
    print(json.dumps(result["summary"], indent=2))
    for load in result["loads"]:
        print(
            f"load {load['pipeline']}: {load['load_seconds']:.2f}s, "
            f"rss_after={load.get('load_rss_after_mb')}, offload={load.get('offload_mode')}"
        )
    for run in result["runs"]:
        label = f"{run['case']} {run['kind']} #{run['run_index']}"
        status = run["status"]
        elapsed = run.get("elapsed_seconds")
        cuda_allocated = run.get("cuda_max_allocated_mb")
        cuda_reserved = run.get("cuda_max_reserved_mb")
        rss_peak = run.get("rss_peak_sampled_mb")
        print(
            f"{label}: {status}, elapsed={elapsed}, "
            f"cuda_allocated={cuda_allocated}, cuda_reserved={cuda_reserved}, rss_peak={rss_peak}"
        )
        if status != "success":
            print(f"  {run.get('phase')}: {run.get('error_type')}: {run.get('error')}")
        elif run.get("image_paths"):
            print(f"  images: {run['image_paths']}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_measurement(args)
    print_result(result)
    return 0 if result["summary"]["failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
