from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


GenerateFn = Callable[[dict[str, object]], dict[str, Any]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure ERNIE-Image text-to-image wall time, CUDA memory, and process RSS."
    )
    parser.add_argument(
        "--prompt",
        default="a serene mountain lake at sunrise, detailed but compact",
        help="Prompt to generate.",
    )
    parser.add_argument("--model", default=None, help="Optional registered ERNIE-Image model name.")
    parser.add_argument("--width", type=int, default=768, help="Output width.")
    parser.add_argument("--height", type=int, default=768, help="Output height.")
    parser.add_argument("--steps", type=int, default=8, help="Inference steps.")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Guidance scale.")
    parser.add_argument("--seed", type=int, default=12345, help="Base seed. Use 0 for random backend seed.")
    parser.add_argument("--num-images", type=int, default=1, help="Images per run. Keep at 1 for low memory tests.")
    parser.add_argument(
        "--memory-preset",
        choices=("sequential_offload", "model_offload"),
        default="sequential_offload",
        help="ERNIE-Image backend memory preset.",
    )
    parser.add_argument("--use-pe", action="store_true", help="Enable ERNIE prompt enhancement.")
    parser.add_argument("--load-pe", action="store_true", help="Load PE prompt enhancer components.")
    parser.add_argument(
        "--execution-mode",
        choices=("subprocess", "in_process"),
        default="subprocess",
        help="Run ERNIE in a child process or inside this process.",
    )
    parser.add_argument("--runs", type=int, default=1, help="Measured runs to execute.")
    parser.add_argument("--warmup-runs", type=int, default=0, help="Warmup runs excluded from the summary.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to write JSON results.")
    return parser.parse_args(argv)


def build_generation_params(args: argparse.Namespace) -> dict[str, object]:
    return {
        "prompt": args.prompt,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "model": args.model,
        "num_images": args.num_images,
        "use_pe": args.use_pe,
        "load_pe": args.load_pe,
        "memory_preset": args.memory_preset,
        "execution_mode": args.execution_mode,
    }


def get_process_rss_mb() -> float | None:
    try:
        import psutil  # type: ignore[import-not-found]

        return psutil.Process().memory_info().rss / 1024**2
    except Exception:
        return None


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def reset_cuda_memory_stats() -> None:
    if not _cuda_available():
        return
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def synchronize_cuda() -> None:
    if not _cuda_available():
        return
    import torch

    torch.cuda.synchronize()


def get_cuda_memory_stats() -> dict[str, float | bool | None]:
    if not _cuda_available():
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


def default_generate(params: dict[str, object]) -> dict[str, Any]:
    from backend.ernie_image.pipeline import generate_text2img

    return generate_text2img(params)


def run_single_measurement(
    params: dict[str, object],
    *,
    run_index: int,
    kind: str,
    generate_fn: GenerateFn,
) -> dict[str, Any]:
    reset_cuda_memory_stats()
    rss_before_mb = get_process_rss_mb()
    start = time.perf_counter()

    try:
        generated = generate_fn(params)
        synchronize_cuda()
        elapsed_seconds = time.perf_counter() - start
        cuda_stats = get_cuda_memory_stats()
        rss_after_mb = get_process_rss_mb()
        return {
            "run_index": run_index,
            "kind": kind,
            "status": "success",
            "elapsed_seconds": elapsed_seconds,
            "rss_before_mb": rss_before_mb,
            "rss_after_mb": rss_after_mb,
            "images": list(generated.get("images", [])),
            **cuda_stats,
        }
    except Exception as exc:
        synchronize_cuda()
        elapsed_seconds = time.perf_counter() - start
        cuda_stats = get_cuda_memory_stats()
        rss_after_mb = get_process_rss_mb()
        return {
            "run_index": run_index,
            "kind": kind,
            "status": "error",
            "elapsed_seconds": elapsed_seconds,
            "rss_before_mb": rss_before_mb,
            "rss_after_mb": rss_after_mb,
            "error_type": type(exc).__name__,
            "error": str(exc),
            **cuda_stats,
        }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [run for run in runs if run["kind"] == "measured"]
    successes = [run for run in measured if run["status"] == "success"]
    failures = [run for run in measured if run["status"] != "success"]
    elapsed = [float(run["elapsed_seconds"]) for run in successes]
    max_allocated = [
        float(run["cuda_max_allocated_mb"])
        for run in successes
        if run.get("cuda_max_allocated_mb") is not None
    ]
    max_reserved = [
        float(run["cuda_max_reserved_mb"])
        for run in successes
        if run.get("cuda_max_reserved_mb") is not None
    ]
    rss_after = [float(run["rss_after_mb"]) for run in successes if run.get("rss_after_mb") is not None]

    return {
        "runs": len(measured),
        "successes": len(successes),
        "failures": len(failures),
        "avg_elapsed_seconds": sum(elapsed) / len(elapsed) if elapsed else None,
        "max_cuda_allocated_mb": max(max_allocated) if max_allocated else None,
        "max_cuda_reserved_mb": max(max_reserved) if max_reserved else None,
        "max_rss_after_mb": max(rss_after) if rss_after else None,
    }


def run_measurement(args: argparse.Namespace, *, generate_fn: GenerateFn | None = None) -> dict[str, Any]:
    generate = generate_fn or default_generate
    params = build_generation_params(args)
    runs: list[dict[str, Any]] = []

    for index in range(args.warmup_runs):
        runs.append(
            run_single_measurement(
                params,
                run_index=index + 1,
                kind="warmup",
                generate_fn=generate,
            )
        )

    for index in range(args.runs):
        runs.append(
            run_single_measurement(
                params,
                run_index=index + 1,
                kind="measured",
                generate_fn=generate,
            )
        )

    result = {
        "tool": "measure_ernie_image_memory",
        "params": params,
        "summary": summarize_runs(runs),
        "runs": runs,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def print_result(result: dict[str, Any]) -> None:
    print(json.dumps(result["summary"], indent=2))
    for run in result["runs"]:
        label = f"{run['kind']} #{run['run_index']}"
        status = run["status"]
        elapsed = float(run["elapsed_seconds"])
        cuda_allocated = run.get("cuda_max_allocated_mb")
        cuda_reserved = run.get("cuda_max_reserved_mb")
        rss_after = run.get("rss_after_mb")
        print(
            f"{label}: {status}, {elapsed:.2f}s, "
            f"cuda_allocated={cuda_allocated}, cuda_reserved={cuda_reserved}, rss_after={rss_after}"
        )
        if status != "success":
            print(f"  {run.get('error_type')}: {run.get('error')}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_measurement(args)
    print_result(result)
    return 0 if result["summary"]["failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
