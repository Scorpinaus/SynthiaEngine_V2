import logging
import threading
from typing import Literal

import torch
from diffusers import ErnieImagePipeline

from backend.config import OUTPUT_DIR
from backend.lora.utils import apply_lora_adapters_with_validation, write_lora_coverage_report
from backend.registries.model import ModelRegistryEntry, list_model_entries
from backend.utilities.logging import configure_logging
from backend.utilities.pipeline import (
    build_batch_output_relpath,
    build_png_metadata,
    cleanup_memory,
    get_batch_output_dir,
    make_batch_id,
    resolve_model_source,
)
from backend.utilities.subprocess_transport import (
    SubprocessTransport,
    normalize_image_result,
    run_subprocess,
)

_ERNIE_IMAGE_SUBPROCESS_SEMAPHORE = threading.Semaphore(1)

logger = logging.getLogger(__name__)
configure_logging()

_DEFAULT_MODEL_NAME = "ERNIE-Image-Turbo"
_DEFAULT_MODEL_LINK = "baidu/ERNIE-Image-Turbo"


def _default_model_entry() -> ModelRegistryEntry:
    return ModelRegistryEntry(
        name=_DEFAULT_MODEL_NAME,
        family="ernie-image",
        model_type="diffusers",
        location_type="hub",
        model_id=13,
        version="turbo",
        link=_DEFAULT_MODEL_LINK,
    )


def _get_ernie_model_entry(model_name: str | None) -> ModelRegistryEntry:
    entries = list_model_entries()
    if model_name:
        for entry in entries:
            if entry.name == model_name:
                if entry.family.lower() != "ernie-image":
                    raise ValueError(f"Model '{model_name}' is not an ERNIE-Image model.")
                return entry
        if model_name in {_DEFAULT_MODEL_NAME, _DEFAULT_MODEL_LINK, "ernie-image"}:
            return _default_model_entry()
        raise ValueError(f"Model '{model_name}' not found.")

    for entry in entries:
        if entry.family.lower() == "ernie-image":
            return entry
    return _default_model_entry()


def load_text2img_pipeline(
    model_name: str | None, *,
    memory_preset: Literal["model_offload", "sequential_offload"] = "sequential_offload",
    load_pe: bool = False,
) -> ErnieImagePipeline:
    entry = _get_ernie_model_entry(model_name)
    source = resolve_model_source(entry)
    logger.info("ERNIE-Image model source: %s", source)

    if entry.model_type != "diffusers":
        raise ValueError("ERNIE-Image currently supports only diffusers model folders or Hub repos.")

    load_kwargs: dict[str, object] = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    if not load_pe:
        load_kwargs.update(pe=None, pe_tokenizer=None)

    pipe = ErnieImagePipeline.from_pretrained(source, **load_kwargs)

    if memory_preset == "model_offload":
        pipe.enable_model_cpu_offload()
    elif memory_preset == "sequential_offload":
        pipe.enable_sequential_cpu_offload()
    else:
        raise ValueError(f"Unsupported ERNIE-Image memory_preset: {memory_preset}")

    if getattr(pipe, "vae", None) is not None:
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()

    cleanup_memory()
    return pipe


def run_text2img_subprocess(params: dict[str, object]) -> dict[str, list[str]]:
    result = run_subprocess(
        SubprocessTransport(
            family="ERNIE-Image",
            runner_module="backend.ernie_image.subprocess_runner",
            temp_prefix="ernie_image_",
            launch_gate=_ERNIE_IMAGE_SUBPROCESS_SEMAPHORE,
        ),
        "text2img",
        params,
    )
    return normalize_image_result(result, family="ERNIE-Image")


def generate_text2img(params: dict[str, object]) -> dict[str, list[str]]:
    execution_mode = params.get("execution_mode")
    if execution_mode not in (None, "", "subprocess"):
        raise ValueError("ERNIE-Image supports only subprocess execution.")
    return run_text2img_subprocess(params)


@torch.inference_mode()
def _generate_text2img_subprocess_child(params: dict[str, object]) -> dict[str, list[str]]:
    prompt = str(params.get("prompt") or "")
    negative_prompt = str(params.get("negative_prompt") or "")
    steps = int(params.get("steps", 8))
    guidance_scale = float(params.get("guidance_scale", 1.0))
    width = int(params.get("width", 768))
    height = int(params.get("height", 768))
    seed = params.get("seed")
    model = params.get("model")
    num_images = int(params.get("num_images", 1))
    use_pe = bool(params.get("use_pe", False))
    load_pe = bool(params.get("load_pe", False))
    memory_preset = str(params.get("memory_preset") or "sequential_offload")
    lora_adapters = params.get("lora_adapters")

    if use_pe and not load_pe:
        raise ValueError("use_pe=true requires load_pe=true")

    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)

    logger.info(
        "ERNIE-Image T2I Generate: model=%s, seed=%s, steps=%s, guidance_scale=%s, WidthxHeight=%sx%s, num_images=%s, use_pe=%s, load_pe=%s, memory_preset=%s",
        model,
        base_seed,
        steps,
        guidance_scale,
        width,
        height,
        num_images,
        use_pe,
        load_pe,
        memory_preset,
    )

    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)
    pipe = load_text2img_pipeline(
        str(model) if model else None,
        memory_preset=memory_preset,  # type: ignore[arg-type]
        load_pe=load_pe,
    )

    filenames: list[str] = []
    _adapter_names, lora_coverage = apply_lora_adapters_with_validation(
        pipe,
        lora_adapters,  # type: ignore[arg-type]
        expected_family="ernie-image",
        validate=False,
    )
    report_path = write_lora_coverage_report(batch_output_dir, batch_id, lora_coverage)
    if report_path is not None:
        logger.info("LoRA coverage report saved to %s", report_path)

    for i in range(num_images):
        current_seed = base_seed + i
        generator = torch.Generator(device="cpu").manual_seed(current_seed)

        call_kwargs: dict[str, object] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "generator": generator,
            "use_pe": use_pe,
        }
        image = pipe(**call_kwargs).images[0]

        filename = batch_output_dir / f"{batch_id}_{current_seed}.png"
        image_params = dict(params)
        image_params.update(
            {
                "mode": "txt2img",
                "pipeline": "ernie-image",
                "seed": current_seed,
                "batch_id": batch_id,
            }
        )
        pnginfo = build_png_metadata(image_params)
        image.save(filename, pnginfo=pnginfo)
        logger.info("Image %s saved to %s", i, filename.name)
        filenames.append(build_batch_output_relpath(batch_id, filename.name))

        del image
        cleanup_memory()

    return {"images": [f"/outputs/{name}" for name in filenames]}
