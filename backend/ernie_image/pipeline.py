import json
import logging
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Literal

import torch
from diffusers import ErnieImagePipeline

from backend.config import OUTPUT_DIR
from backend.registries.model import ModelRegistryEntry, list_model_entries
from backend.utilities.logging import configure_logging
from backend.utilities.pipeline import (
    build_batch_output_relpath,
    build_png_metadata,
    cleanup_memory,
    get_batch_output_dir,
    make_batch_id,
    release_pipeline,
    resolve_model_source,
)

GEN_LOCK = threading.Lock()
_REPO_ROOT = Path(__file__).resolve().parents[2]

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
    model_name: str | None,
    *,
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
    child_params = dict(params)
    child_params["execution_mode"] = "in_process"

    with tempfile.TemporaryDirectory(prefix="ernie_image_") as tmpdir:
        input_path = Path(tmpdir) / "input.json"
        output_path = Path(tmpdir) / "output.json"
        input_path.write_text(
            json.dumps(child_params, separators=(",", ": ")),
            encoding="utf-8",
        )

        cmd = [
            sys.executable,
            "-m",
            "backend.ernie_image.subprocess_runner",
            str(input_path),
            str(output_path),
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_REPO_ROOT))

        if not output_path.exists():
            detail = completed.stderr.strip() or completed.stdout.strip() or "No subprocess result was written."
            raise RuntimeError(f"ERNIE-Image subprocess failed: {detail}")

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if completed.returncode != 0 or not payload.get("ok"):
            detail = payload.get("error") or completed.stderr.strip() or "Unknown subprocess failure."
            error_type = payload.get("error_type")
            if error_type:
                detail = f"{error_type}: {detail}"
            raise RuntimeError(f"ERNIE-Image subprocess failed: {detail}")

        result = payload.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("images"), list):
            raise RuntimeError("ERNIE-Image subprocess returned an invalid result.")

        return {"images": [str(path) for path in result["images"]]}


def generate_text2img(params: dict[str, object]) -> dict[str, list[str]]:
    execution_mode = str(params.get("execution_mode") or "subprocess")
    if execution_mode == "subprocess":
        return run_text2img_subprocess(params)
    if execution_mode == "in_process":
        return generate_text2img_in_process(params)
    raise ValueError(f"Unsupported ERNIE-Image execution_mode: {execution_mode}")


@torch.inference_mode()
def generate_text2img_in_process(params: dict[str, object]) -> dict[str, list[str]]:
    prompt = str(params.get("prompt") or "")
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

    if use_pe and not load_pe:
        raise ValueError("use_pe=true requires load_pe=true")

    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)

    logger.info(
        "ERNIE-Image Generate: model=%s seed=%s steps=%s guidance_scale=%s size=%sx%s num_images=%s use_pe=%s load_pe=%s memory_preset=%s",
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
    try:
        with GEN_LOCK:
            for i in range(num_images):
                current_seed = base_seed + i
                generator = torch.Generator(device="cpu").manual_seed(current_seed)

                call_kwargs: dict[str, object] = {
                    "prompt": prompt,
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
    finally:
        release_pipeline(pipe, logger=logger)
        pipe = None

    return {"images": [f"/outputs/{name}" for name in filenames]}
