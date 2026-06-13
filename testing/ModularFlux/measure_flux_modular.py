from __future__ import annotations

import argparse
import gc
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
MODEL_COMPONENTS = ("text_encoder", "text_encoder_2", "transformer", "vae")
DIFFUSERS_MODEL_COMPONENTS = ("transformer", "vae")
PROMPT_COMPONENTS = ("tokenizer", "tokenizer_2", "text_encoder", "text_encoder_2")
PROMPT_CACHE_ATTR = "_modular_flux_prompt_cache"
PROMPT_CACHE_STATS_ATTR = "_modular_flux_prompt_cache_stats"
PLACEMENT_EVENTS_ATTR = "_fluxmodular_device_placement_events"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    pipeline: str
    prompt: bool = True
    image: bool = False
    mask: bool = False
    embeds: bool = False
    strength: bool = False


CASES: dict[str, CaseSpec] = {
    "flux-text2img": CaseSpec("flux-text2img", "flux"),
    "flux-img2img": CaseSpec("flux-img2img", "flux", image=True, strength=True),
    "flux-inpaint": CaseSpec("flux-inpaint", "flux", image=True, mask=True, strength=True),
    "flux-embeds2img": CaseSpec("flux-embeds2img", "flux", prompt=False, embeds=True),
    "flux-img2img-embeds": CaseSpec(
        "flux-img2img-embeds",
        "flux",
        prompt=False,
        image=True,
        embeds=True,
        strength=True,
    ),
    "flux-inpaint-embeds": CaseSpec(
        "flux-inpaint-embeds",
        "flux",
        prompt=False,
        image=True,
        mask=True,
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
    "inpaint": {"flux": "flux-inpaint"},
    "image": {"kontext": "kontext-image"},
    "embeds2img": {"flux": "flux-embeds2img", "kontext": "kontext-embeds2img"},
    "img2img-embeds": {"flux": "flux-img2img-embeds"},
    "inpaint-embeds": {"flux": "flux-inpaint-embeds"},
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
    parser.add_argument("--mask-image", type=Path, default=None, help="Optional mask image for inpaint cases.")
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
    parser.add_argument(
        "--load-strategy",
        choices=("eager", "phased"),
        default="phased",
        help="Load all components upfront or pre-encode prompts before loading generation components.",
    )
    parser.add_argument(
        "--prompt-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse precomputed prompt embeddings across compatible phased pipeline loads.",
    )
    parser.add_argument(
        "--prompt-cache-device",
        choices=("cpu", "device"),
        default="cpu",
        help="Store staged prompt embeddings on CPU or keep them on the encoding device.",
    )
    parser.add_argument(
        "--cuda-placement",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Stage active Flux components on CUDA, keep them on CPU, or choose automatically.",
    )
    parser.add_argument(
        "--vram-reserve-margin",
        default="3GB",
        help="CUDA memory margin reserved when auto-selecting staged component placement.",
    )
    parser.add_argument(
        "--transformer-stream-blocks",
        default="auto",
        help="Transformer blocks per streamed CUDA group when full transformer placement does not fit.",
    )
    parser.add_argument(
        "--low-cpu-mem-usage",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Forward low_cpu_mem_usage to model component loaders. Omit to use Diffusers defaults.",
    )
    parser.add_argument(
        "--offload-state-dict",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Forward offload_state_dict to model component loaders.",
    )
    parser.add_argument(
        "--offload-folder",
        type=Path,
        default=None,
        help="Folder for load-time disk offload when device maps use disk or offload_state_dict is enabled.",
    )
    parser.add_argument(
        "--disable-mmap",
        action="store_true",
        help="Disable safetensors mmap for Diffusers model components. May help HDD/network mounts.",
    )
    parser.add_argument(
        "--device-map",
        default=None,
        help="Optional model component device_map, such as cpu, cuda, auto, balanced, or sequential.",
    )
    parser.add_argument(
        "--max-memory",
        action="append",
        default=None,
        metavar="DEVICE=LIMIT",
        help="Repeatable max_memory entry, for example --max-memory 0=10GB --max-memory cpu=48GB.",
    )
    parser.add_argument(
        "--use-safetensors",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Forward use_safetensors to model component loaders. Omit to use Diffusers defaults.",
    )
    parser.add_argument(
        "--quantization",
        choices=("none", "bnb_8bit", "bnb_4bit"),
        default="none",
        help="Experimental low-system-RAM quantization for text_encoder_2 and transformer.",
    )
    parser.add_argument(
        "--bnb-4bit-use-double-quant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use nested/double quantization for --quantization bnb_4bit.",
    )
    parser.add_argument("--reload-per-case", action="store_true", help="Reload the pipeline for every case.")
    parser.add_argument("--rss-sample-interval", type=float, default=0.05, help="Peak RSS sampling interval.")
    parser.add_argument(
        "--system-ram-limit",
        default=None,
        help="Optional process RSS limit such as 16GB. Exceeding it marks the load/run as failed.",
    )
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


def build_quantization_config_map(args: argparse.Namespace) -> dict[str, Any] | None:
    quantization = str(args.quantization or "none")
    if quantization == "none":
        return None

    try:
        import torch
        from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
        from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
    except Exception as exc:  # pragma: no cover - depends on optional runtime packages
        raise RuntimeError(
            "--quantization requires diffusers, transformers, accelerate, and bitsandbytes support."
        ) from exc

    if quantization == "bnb_8bit":
        return {
            "text_encoder_2": TransformersBitsAndBytesConfig(load_in_8bit=True),
            "transformer": DiffusersBitsAndBytesConfig(load_in_8bit=True),
        }
    if quantization == "bnb_4bit":
        compute_dtype = resolve_dtype(args) or torch.bfloat16
        return {
            "text_encoder_2": TransformersBitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=bool(args.bnb_4bit_use_double_quant),
            ),
            "transformer": DiffusersBitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=bool(args.bnb_4bit_use_double_quant),
            ),
        }

    raise ValueError("quantization must be one of: none, bnb_8bit, bnb_4bit.")


def model_load_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for name in ("revision", "token"):
        value = getattr(args, name)
        if value is not None:
            kwargs[name] = value
    if args.cache_dir is not None:
        kwargs["cache_dir"] = str(args.cache_dir)
    if args.local_files_only:
        kwargs["local_files_only"] = True
    if args.variant is not None:
        kwargs["variant"] = _component_kwarg(args.variant, MODEL_COMPONENTS)
    if args.low_cpu_mem_usage is not None:
        kwargs["low_cpu_mem_usage"] = _component_kwarg(bool(args.low_cpu_mem_usage), MODEL_COMPONENTS)
    if args.offload_state_dict is not None:
        kwargs["offload_state_dict"] = _component_kwarg(bool(args.offload_state_dict), MODEL_COMPONENTS)
    if args.offload_folder is not None:
        kwargs["offload_folder"] = _component_kwarg(str(args.offload_folder), MODEL_COMPONENTS)
    if args.device_map is not None and args.device_map.lower() != "none":
        kwargs["device_map"] = _component_kwarg(args.device_map, MODEL_COMPONENTS)
    max_memory = parse_max_memory(args.max_memory)
    if max_memory is not None:
        kwargs["max_memory"] = _component_kwarg(max_memory, MODEL_COMPONENTS)
    if args.use_safetensors is not None:
        kwargs["use_safetensors"] = _component_kwarg(bool(args.use_safetensors), MODEL_COMPONENTS)
    if args.disable_mmap:
        kwargs["disable_mmap"] = _component_kwarg(True, DIFFUSERS_MODEL_COMPONENTS)
    quantization_config = build_quantization_config_map(args)
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    return kwargs


def model_for_kind(kind: str, args: argparse.Namespace) -> str:
    return args.kontext_model if kind == "kontext" and args.kontext_model else args.model


def model_config_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs = model_load_kwargs(args)
    for key in (
        "variant",
        "low_cpu_mem_usage",
        "offload_state_dict",
        "offload_folder",
        "device_map",
        "max_memory",
        "use_safetensors",
        "disable_mmap",
        "quantization_config",
    ):
        kwargs.pop(key, None)
    return kwargs


def _component_kwarg(value: Any, components: tuple[str, ...]) -> dict[str, Any]:
    return {component: value for component in components}


def parse_memory_bytes(value: str | int | float | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    text = str(value).strip().upper().replace(" ", "")
    if not text:
        return None
    units = (
        ("GIB", 1024**3),
        ("GB", 1000**3),
        ("MIB", 1024**2),
        ("MB", 1000**2),
        ("KIB", 1024),
        ("KB", 1000),
    )
    for suffix, scale in units:
        if text.endswith(suffix):
            return max(0, int(float(text[: -len(suffix)]) * scale))
    return max(0, int(float(text)))


def system_ram_limit_mb(args: argparse.Namespace) -> float | None:
    limit_bytes = parse_memory_bytes(args.system_ram_limit)
    if limit_bytes is None:
        return None
    return limit_bytes / 1024**2


def _profile_rss_peak(profile: dict[str, Any]) -> float | None:
    values = [
        value
        for value in (
            profile.get("rss_peak_sampled_mb"),
            profile.get("rss_after_mb"),
            profile.get("rss_before_mb"),
        )
        if value is not None
    ]
    return max(float(value) for value in values) if values else None


def apply_system_ram_limit(profile: dict[str, Any], args: argparse.Namespace, *, success_statuses: set[str]) -> dict[str, Any]:
    limit_mb = system_ram_limit_mb(args)
    if limit_mb is None:
        return profile

    peak_mb = _profile_rss_peak(profile)
    exceeded = peak_mb is not None and peak_mb > limit_mb
    profile["rss_limit_mb"] = limit_mb
    profile["rss_limit_exceeded"] = bool(exceeded)
    if exceeded and profile.get("status") in success_statuses:
        profile["status"] = "rss_limit_exceeded"
        profile["error_type"] = "RSSLimitExceeded"
        profile["error"] = f"Peak process RSS {peak_mb:.1f} MB exceeded limit {limit_mb:.1f} MB."
    return profile


def parse_max_memory(entries: list[str] | None) -> dict[int | str, str] | None:
    if not entries:
        return None
    parsed: dict[int | str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid --max-memory entry '{entry}'. Expected DEVICE=LIMIT, for example 0=10GB.")
        raw_device, limit = entry.split("=", 1)
        device = raw_device.strip()
        limit = limit.strip()
        if not device or not limit:
            raise ValueError(f"Invalid --max-memory entry '{entry}'. Device and limit are required.")
        parsed[int(device) if device.isdigit() else device] = limit
    return parsed


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


def create_synthetic_mask(width: int, height: int) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    margin = max(16, min(width, height) // 4)
    draw.rectangle((margin, margin, width - margin, height - margin), fill=255)
    return mask


def load_input_image(args: argparse.Namespace) -> Image.Image:
    if args.image is None:
        return create_synthetic_image(args.width, args.height)
    with Image.open(args.image) as image:
        return image.convert("RGB")


def load_mask_image(args: argparse.Namespace) -> Image.Image:
    if args.mask_image is None:
        return create_synthetic_mask(args.width, args.height)
    with Image.open(args.mask_image) as image:
        return image.convert("L")


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


def clear_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
    except Exception:
        return


def component_names_to_load(
    pipe: Any,
    *,
    include: tuple[str, ...] | None = None,
    exclude: tuple[str, ...] = (),
) -> list[str]:
    names: list[str] = []
    specs = getattr(pipe, "_component_specs", {})
    candidates = include if include is not None else tuple(specs.keys())
    for name in candidates:
        if name in exclude:
            continue
        spec = specs.get(name)
        if spec is None:
            continue
        if (
            getattr(spec, "default_creation_method", None) == "from_pretrained"
            and getattr(spec, "pretrained_model_name_or_path", None) is not None
            and getattr(pipe, name, None) is None
        ):
            names.append(name)
    return names


def component_type_name(pipe: Any, name: str) -> str | None:
    spec = getattr(pipe, "_component_specs", {}).get(name)
    type_hint = getattr(spec, "type_hint", None)
    if type_hint is None:
        return None
    return getattr(type_hint, "__name__", str(type_hint))


def load_kwargs_for_component(load_kwargs: dict[str, Any], name: str) -> dict[str, Any]:
    component_kwargs: dict[str, Any] = {}
    for key, value in load_kwargs.items():
        if not isinstance(value, dict):
            component_kwargs[key] = value
        elif name in value:
            component_kwargs[key] = value[name]
        elif "default" in value:
            component_kwargs[key] = value["default"]
    return component_kwargs


def load_component_with_profile(
    pipe: Any,
    name: str,
    args: argparse.Namespace,
    load_kwargs: dict[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    reset_cuda_memory_stats()
    clear_placement_events(pipe)
    rss_before_mb = get_process_rss_mb()
    start = time.perf_counter()
    with PeakRSSSampler(args.rss_sample_interval) as rss_sampler:
        try:
            pipe.load_components(name, **load_kwargs_for_component(load_kwargs, name))
            synchronize_cuda()
            elapsed_seconds = time.perf_counter() - start
            loaded = getattr(pipe, name, None) is not None
            status = "loaded" if loaded else "missing_after_load"
            profile = {
                "component": name,
                "component_type": component_type_name(pipe, name),
                "phase": phase,
                "status": status,
                "elapsed_seconds": elapsed_seconds,
                "rss_before_mb": rss_before_mb,
                "rss_after_mb": get_process_rss_mb(),
                "rss_peak_sampled_mb": rss_sampler.peak_mb,
                **get_cuda_memory_stats(),
            }
            return apply_system_ram_limit(profile, args, success_statuses={"loaded"})
        except Exception as exc:
            synchronize_cuda()
            profile = {
                "component": name,
                "component_type": component_type_name(pipe, name),
                "phase": phase,
                "status": "error",
                "elapsed_seconds": time.perf_counter() - start,
                "rss_before_mb": rss_before_mb,
                "rss_after_mb": get_process_rss_mb(),
                "rss_peak_sampled_mb": rss_sampler.peak_mb,
                "error_type": type(exc).__name__,
                "error": str(exc),
                **get_cuda_memory_stats(),
            }
            return apply_system_ram_limit(profile, args, success_statuses=set())


def load_components_with_profile(
    pipe: Any,
    args: argparse.Namespace,
    load_kwargs: dict[str, Any],
    *,
    names: list[str] | None = None,
    phase: str,
) -> list[dict[str, Any]]:
    profiles = []
    for name in names if names is not None else component_names_to_load(pipe):
        profiles.append(load_component_with_profile(pipe, name, args, load_kwargs, phase=phase))
    return profiles


def _max_profile_value(profiles: list[dict[str, Any]], name: str) -> float | None:
    values = [float(profile[name]) for profile in profiles if profile.get(name) is not None]
    return max(values) if values else None


def phase_profile_from_components(phase: str, profiles: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = [profile.get("status") for profile in profiles]
    failure = next(
        (
            profile
            for profile in profiles
            if profile.get("status") in {"error", "rss_limit_exceeded"}
        ),
        None,
    )
    if not profiles:
        status = "skipped"
    elif any(status == "error" for status in statuses):
        status = "error"
    elif any(status == "rss_limit_exceeded" for status in statuses):
        status = "rss_limit_exceeded"
    elif all(status == "loaded" for status in statuses):
        status = "success"
    else:
        status = "partial"
    summary = {
        "phase": phase,
        "status": status,
        "component_count": len(profiles),
        "components": [profile["component"] for profile in profiles],
        "elapsed_seconds": sum(float(profile["elapsed_seconds"]) for profile in profiles),
        "max_cuda_allocated_mb": _max_profile_value(profiles, "cuda_max_allocated_mb"),
        "max_cuda_reserved_mb": _max_profile_value(profiles, "cuda_max_reserved_mb"),
        "max_rss_peak_sampled_mb": _max_profile_value(profiles, "rss_peak_sampled_mb"),
        "rss_limit_exceeded": any(bool(profile.get("rss_limit_exceeded")) for profile in profiles),
        "rss_limit_mb": _max_profile_value(profiles, "rss_limit_mb"),
    }
    if failure is not None:
        summary["error_type"] = failure.get("error_type")
        summary["error"] = failure.get("error")
    return summary


def profile_callable(phase: str, args: argparse.Namespace, fn: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    reset_cuda_memory_stats()
    rss_before_mb = get_process_rss_mb()
    start = time.perf_counter()
    with PeakRSSSampler(args.rss_sample_interval) as rss_sampler:
        try:
            result = fn()
            synchronize_cuda()
            profile = {
                "phase": phase,
                "status": "success",
                "elapsed_seconds": time.perf_counter() - start,
                "rss_before_mb": rss_before_mb,
                "rss_after_mb": get_process_rss_mb(),
                "rss_peak_sampled_mb": rss_sampler.peak_mb,
                **get_cuda_memory_stats(),
            }
            return result, apply_system_ram_limit(profile, args, success_statuses={"success"})
        except Exception as exc:
            synchronize_cuda()
            profile = {
                "phase": phase,
                "status": "error",
                "elapsed_seconds": time.perf_counter() - start,
                "rss_before_mb": rss_before_mb,
                "rss_after_mb": get_process_rss_mb(),
                "rss_peak_sampled_mb": rss_sampler.peak_mb,
                "error_type": type(exc).__name__,
                "error": str(exc),
                **get_cuda_memory_stats(),
            }
            return None, apply_system_ram_limit(profile, args, success_statuses=set())


def prompt_cache_key(kind: str, model: str, args: argparse.Namespace, dtype: Any) -> tuple[Any, ...]:
    return (
        kind,
        str(model),
        args.revision,
        args.variant,
        args.prompt,
        args.prompt_2,
        args.max_sequence_length,
        str(dtype or "component_default"),
    )


def prompt_cache_store(args: argparse.Namespace) -> dict[tuple[Any, ...], dict[str, Any]]:
    store = getattr(args, PROMPT_CACHE_ATTR, None)
    if store is None:
        store = {}
        setattr(args, PROMPT_CACHE_ATTR, store)
    return store


def prompt_cache_stats(args: argparse.Namespace) -> dict[str, int]:
    stats = getattr(args, PROMPT_CACHE_STATS_ATTR, None)
    if stats is None:
        stats = {"hits": 0, "misses": 0, "stores": 0}
        setattr(args, PROMPT_CACHE_STATS_ATTR, stats)
    return stats


def prompt_cache_summary(args: argparse.Namespace) -> dict[str, Any]:
    stats = prompt_cache_stats(args)
    return {
        "prompt_cache_enabled": bool(args.prompt_cache),
        "prompt_cache_device": args.prompt_cache_device,
        "prompt_cache_hits": stats["hits"],
        "prompt_cache_misses": stats["misses"],
        "prompt_cache_stores": stats["stores"],
        "prompt_cache_entries": len(prompt_cache_store(args)),
    }


def prompt_cache_lookup(args: argparse.Namespace, cache_key: tuple[Any, ...]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    start = time.perf_counter()
    if not args.prompt_cache:
        return None, {
            "phase": "prompt_cache",
            "status": "disabled",
            "elapsed_seconds": time.perf_counter() - start,
            **prompt_cache_summary(args),
        }

    store = prompt_cache_store(args)
    stats = prompt_cache_stats(args)
    if cache_key in store:
        stats["hits"] += 1
        return store[cache_key], {
            "phase": "prompt_cache",
            "status": "hit",
            "elapsed_seconds": time.perf_counter() - start,
            **prompt_cache_summary(args),
        }

    stats["misses"] += 1
    return None, {
        "phase": "prompt_cache",
        "status": "miss",
        "elapsed_seconds": time.perf_counter() - start,
        **prompt_cache_summary(args),
    }


def _move_prompt_cache_value(value: Any, cache_device: str) -> Any:
    if cache_device != "cpu":
        return value
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    to = getattr(value, "to", None)
    if not callable(to):
        return value
    try:
        return value.to("cpu")
    except TypeError:
        return value.to(device="cpu")


def make_prompt_cache_entry(
    embeds: tuple[Any, Any],
    *,
    cache_key: tuple[Any, ...],
    cache_device: str,
) -> dict[str, Any]:
    return {
        "prompt_embeds": _move_prompt_cache_value(embeds[0], cache_device),
        "pooled_prompt_embeds": _move_prompt_cache_value(embeds[1], cache_device),
        "cache_key": cache_key,
        "cache_device": cache_device,
    }


def store_prompt_cache_entry(
    args: argparse.Namespace,
    cache_key: tuple[Any, ...],
    entry: dict[str, Any],
) -> None:
    if not args.prompt_cache:
        return
    prompt_cache_store(args)[cache_key] = entry
    prompt_cache_stats(args)["stores"] += 1


def skipped_phase_profile(phase: str, reason: str) -> dict[str, Any]:
    return {
        "phase": phase,
        "status": "skipped",
        "reason": reason,
        "elapsed_seconds": 0.0,
    }


def cache_prompt_embeds(pipe: Any, embeds_or_entry: tuple[Any, Any] | dict[str, Any]) -> None:
    if isinstance(embeds_or_entry, dict):
        entry = embeds_or_entry
    else:
        entry = {
            "prompt_embeds": embeds_or_entry[0],
            "pooled_prompt_embeds": embeds_or_entry[1],
        }
    setattr(pipe, "_modular_flux_cached_prompt_embeds", entry)


def cached_prompt_embeds(pipe: Any) -> dict[str, Any] | None:
    cached = getattr(pipe, "_modular_flux_cached_prompt_embeds", None)
    if isinstance(cached, dict) and "prompt_embeds" in cached and "pooled_prompt_embeds" in cached:
        return cached
    return None


def release_prompt_components(pipe: Any) -> list[str]:
    released = []
    for name in PROMPT_COMPONENTS:
        if getattr(pipe, name, None) is not None:
            setattr(pipe, name, None)
            released.append(name)
    clear_memory()
    return released


def placement_events(pipe: Any) -> list[dict[str, Any]]:
    events = getattr(pipe, PLACEMENT_EVENTS_ATTR, [])
    return list(events) if isinstance(events, list) else []


def clear_placement_events(pipe: Any) -> None:
    setattr(pipe, PLACEMENT_EVENTS_ATTR, [])


def default_pipeline_loader(kind: str, args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    import torch

    from custom_pipelines.FluxModular import (
        FluxKontextModularPipeline,
        FluxModularPipeline,
        enable_low_memory_flux_modular,
    )

    pipeline_cls = FluxKontextModularPipeline if kind == "kontext" else FluxModularPipeline
    model = model_for_kind(kind, args)
    dtype = resolve_dtype(args)
    device = resolve_device(args)
    config_kwargs = model_config_kwargs(args)
    load_kwargs = model_load_kwargs(args)
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype

    reset_cuda_memory_stats()
    rss_before_mb = get_process_rss_mb()
    start = time.perf_counter()
    constructor_start = time.perf_counter()
    pipe = pipeline_cls(pretrained_model_name_or_path=model, **config_kwargs)
    constructor_seconds = time.perf_counter() - constructor_start

    prompt_components_released: list[str] = []
    if args.load_strategy == "eager":
        component_loads = load_components_with_profile(pipe, args, load_kwargs, phase="eager_load")
        phase_loads = [phase_profile_from_components("eager_load", component_loads)]
    else:
        key = prompt_cache_key(kind, model, args, dtype)
        cache_entry, cache_profile = prompt_cache_lookup(args, key)
        if cache_entry is not None:
            cache_prompt_embeds(pipe, cache_entry)
            prompt_component_loads = []
            encode_profile = skipped_phase_profile("prompt_encode", "prompt_cache_hit")
            release_profile = skipped_phase_profile("prompt_release", "prompt_cache_hit")
        else:
            prompt_names = component_names_to_load(pipe, include=PROMPT_COMPONENTS)
            prompt_component_loads = load_components_with_profile(
                pipe,
                args,
                load_kwargs,
                names=prompt_names,
                phase="prompt_load",
            )
            embeds, encode_profile = profile_callable("prompt_encode", args, lambda: precompute_prompt_embeds(pipe, args))
            if encode_profile["status"] == "success" and embeds is not None:
                entry = make_prompt_cache_entry(embeds, cache_key=key, cache_device=args.prompt_cache_device)
                cache_prompt_embeds(pipe, entry)
                store_prompt_cache_entry(args, key, entry)
                del embeds
                clear_memory()
                released, release_profile = profile_callable(
                    "prompt_release",
                    args,
                    lambda: release_prompt_components(pipe),
                )
                prompt_components_released = released if isinstance(released, list) else []
            else:
                setattr(pipe, "_modular_flux_prompt_embed_error", encode_profile)
                release_profile = {
                    "phase": "prompt_release",
                    "status": "skipped",
                    "elapsed_seconds": 0.0,
                    "released_components": [],
                }

        generation_names = component_names_to_load(pipe, exclude=PROMPT_COMPONENTS)
        generation_component_loads = load_components_with_profile(
            pipe,
            args,
            load_kwargs,
            names=generation_names,
            phase="generation_load",
        )
        component_loads = prompt_component_loads + generation_component_loads
        phase_loads = [
            cache_profile,
            phase_profile_from_components("prompt_load", prompt_component_loads),
            encode_profile,
            {**release_profile, "released_components": prompt_components_released},
            phase_profile_from_components("generation_load", generation_component_loads),
        ]

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
    load_statuses = [phase.get("status") for phase in phase_loads]
    if any(status == "error" for status in load_statuses):
        load_status = "error"
    elif any(status == "rss_limit_exceeded" for status in load_statuses):
        load_status = "rss_limit_exceeded"
    else:
        load_status = "success"

    return pipe, {
        "pipeline": kind,
        "model": model,
        "status": load_status,
        "load_strategy": args.load_strategy,
        "torch_dtype": str(dtype or "component_default"),
        "device": str(device),
        "offload_mode": offload_mode,
        "load_seconds": load_seconds,
        "constructor_seconds": constructor_seconds,
        "component_load_seconds": sum(float(load["elapsed_seconds"]) for load in component_loads),
        "component_load_count": len(component_loads),
        "component_loads": component_loads,
        "phase_loads": phase_loads,
        "prompt_components_released": prompt_components_released,
        "cached_prompt_embeds": cached_prompt_embeds(pipe) is not None,
        "device_placement_events": placement_events(pipe),
        **prompt_cache_summary(args),
        "quantization": args.quantization,
        "system_ram_limit_mb": system_ram_limit_mb(args),
        "load_max_cuda_allocated_mb": _max_profile_value(component_loads, "cuda_max_allocated_mb"),
        "load_max_cuda_reserved_mb": _max_profile_value(component_loads, "cuda_max_reserved_mb"),
        "load_max_rss_peak_sampled_mb": _max_profile_value(component_loads, "rss_peak_sampled_mb"),
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
            low_memory_cuda_placement=args.cuda_placement,
            low_memory_vram_reserve_margin=args.vram_reserve_margin,
            low_memory_eager_offload=True,
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
        "low_memory_cuda_placement": args.cuda_placement,
        "low_memory_vram_reserve_margin": args.vram_reserve_margin,
        "low_memory_transformer_stream_blocks": args.transformer_stream_blocks,
    }
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
    if case.mask:
        kwargs["mask_image"] = load_mask_image(args)

    prompt_error = getattr(pipe, "_modular_flux_prompt_embed_error", None)
    if args.load_strategy == "phased" and prompt_error is not None:
        error_type = prompt_error.get("error_type", "Error")
        error = prompt_error.get("error", "prompt pre-encoding failed")
        raise RuntimeError(f"Phased prompt encode failed: {error_type}: {error}")

    cached = cached_prompt_embeds(pipe)
    use_cached_embeds = cached is not None and (case.prompt or case.embeds)
    if use_cached_embeds:
        kwargs["prompt_embeds"] = cached["prompt_embeds"]
        kwargs["pooled_prompt_embeds"] = cached["pooled_prompt_embeds"]
        embed_seconds = 0.0
    elif case.prompt:
        kwargs["prompt"] = args.prompt
        if args.prompt_2 is not None:
            kwargs["prompt_2"] = args.prompt_2
        embed_seconds = None
    elif case.embeds:
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
            profile = {
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
                "device_placement_events": placement_events(pipe),
                **prepare_stats,
                **get_cuda_memory_stats(),
            }
            return apply_system_ram_limit(profile, args, success_statuses={"success"})
        except Exception as exc:
            synchronize_cuda()
            profile = {
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
                "device_placement_events": placement_events(pipe),
                "error_type": type(exc).__name__,
                "error": str(exc),
                **prepare_stats,
                **get_cuda_memory_stats(),
            }
            return apply_system_ram_limit(profile, args, success_statuses=set())


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


def result_has_failures(result: dict[str, Any]) -> bool:
    if int(result.get("summary", {}).get("failures") or 0) > 0:
        return True
    for load in result.get("loads", []):
        if load.get("status") not in {None, "success"}:
            return True
    return False


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
            "load_strategy": args.load_strategy,
            "prompt_cache": args.prompt_cache,
            "prompt_cache_device": args.prompt_cache_device,
            "cuda_placement": args.cuda_placement,
            "vram_reserve_margin": args.vram_reserve_margin,
            "transformer_stream_blocks": args.transformer_stream_blocks,
            "low_memory_sequential_images": args.low_memory_sequential_images,
            "low_memory_transformer_buffers": args.low_memory_transformer_buffers,
            "decode_chunk_size": args.decode_chunk_size,
            "vae_decode_device": args.vae_decode_device,
            "low_cpu_mem_usage": args.low_cpu_mem_usage,
            "offload_state_dict": args.offload_state_dict,
            "offload_folder": str(args.offload_folder) if args.offload_folder is not None else None,
            "disable_mmap": args.disable_mmap,
            "device_map": args.device_map,
            "max_memory": parse_max_memory(args.max_memory),
            "use_safetensors": args.use_safetensors,
            "quantization": args.quantization,
            "bnb_4bit_use_double_quant": args.bnb_4bit_use_double_quant,
            "system_ram_limit_mb": system_ram_limit_mb(args),
        },
        "prompt_cache": prompt_cache_summary(args),
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
            f"status={load.get('status')}, "
            f"strategy={load.get('load_strategy')}, "
            f"rss_after={load.get('load_rss_after_mb')}, "
            f"peak_rss={load.get('load_max_rss_peak_sampled_mb')}, "
            f"peak_cuda_allocated={load.get('load_max_cuda_allocated_mb')}, "
            f"offload={load.get('offload_mode')}, "
            f"prompt_cache={load.get('prompt_cache_enabled')}"
        )
        print(
            "  prompt cache: "
            f"hits={load.get('prompt_cache_hits')}, "
            f"misses={load.get('prompt_cache_misses')}, "
            f"stores={load.get('prompt_cache_stores')}, "
            f"entries={load.get('prompt_cache_entries')}, "
            f"device={load.get('prompt_cache_device')}"
        )
        for event in load.get("device_placement_events", []):
            print(
                f"  placement {event.get('component')}: "
                f"mode={event.get('mode')}, device={event.get('device')}, "
                f"blocks_per_group={event.get('blocks_per_group')}"
            )
        for phase in load.get("phase_loads", []):
            print(
                f"  phase {phase['phase']}: {phase['status']}, "
                f"{phase.get('elapsed_seconds', 0.0):.2f}s, "
                f"components={phase.get('components', phase.get('released_components'))}, "
                f"rss_peak={phase.get('max_rss_peak_sampled_mb', phase.get('rss_peak_sampled_mb'))}, "
                f"cuda_peak_allocated={phase.get('max_cuda_allocated_mb', phase.get('cuda_max_allocated_mb'))}, "
                f"cuda_peak_reserved={phase.get('max_cuda_reserved_mb', phase.get('cuda_max_reserved_mb'))}"
            )
            if phase["status"] in {"error", "rss_limit_exceeded"}:
                print(f"    {phase.get('error_type')}: {phase.get('error')}")
        for component in load.get("component_loads", []):
            print(
                f"  component {component['component']} ({component.get('phase')}): {component['status']}, "
                f"{component['elapsed_seconds']:.2f}s, "
                f"rss_before={component.get('rss_before_mb')}, "
                f"rss_after={component.get('rss_after_mb')}, "
                f"rss_peak={component.get('rss_peak_sampled_mb')}, "
                f"cuda_peak_allocated={component.get('cuda_max_allocated_mb')}, "
                f"cuda_peak_reserved={component.get('cuda_max_reserved_mb')}"
            )
            if component["status"] in {"error", "rss_limit_exceeded"}:
                print(f"    {component.get('error_type')}: {component.get('error')}")
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
        for event in run.get("device_placement_events", []):
            print(
                f"  placement {event.get('component')}: "
                f"mode={event.get('mode')}, device={event.get('device')}, "
                f"blocks_per_group={event.get('blocks_per_group')}"
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_measurement(args)
    print_result(result)
    return 1 if result_has_failures(result) else 0


if __name__ == "__main__":
    raise SystemExit(main())
