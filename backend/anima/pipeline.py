import json
import logging
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Literal

import torch
from diffusers import DiffusionPipeline

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
from backend.utilities.schedulers import create_scheduler

_ANIMA_SUBPROCESS_SEMAPHORE = threading.Semaphore(1)
_REPO_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)
configure_logging()

_DEFAULT_MODEL_NAME = "Anima-Preview-3"
_DEFAULT_MODEL_LINK = "CalamitousFelicitousness/Anima-Preview-3-sdnext-diffusers"


def _default_model_entry() -> ModelRegistryEntry:
    return ModelRegistryEntry(
        name=_DEFAULT_MODEL_NAME,
        family="anima",
        model_type="diffusers",
        location_type="hub",
        model_id=14,
        version="main",
        link=_DEFAULT_MODEL_LINK,
    )


def _get_anima_model_entry(model_name: str | None) -> ModelRegistryEntry:
    entries = list_model_entries()
    if model_name:
        for entry in entries:
            if entry.name == model_name:
                if entry.family.lower() != "anima":
                    raise ValueError(f"Model '{model_name}' is not an Anima model.")
                return entry
        if model_name in {_DEFAULT_MODEL_NAME, _DEFAULT_MODEL_LINK, "anima"}:
            return _default_model_entry()
        raise ValueError(f"Model '{model_name}' not found.")

    for entry in entries:
        if entry.family.lower() == "anima":
            return entry
    return _default_model_entry()


def load_text2img_pipeline(
    model_name: str | None,
    *,
    memory_preset: Literal["model_offload", "sequential_offload"] = "sequential_offload",
) -> DiffusionPipeline:
    entry = _get_anima_model_entry(model_name)
    source = resolve_model_source(entry)
    logger.info("Anima model source: %s", source)

    if entry.model_type != "diffusers":
        raise ValueError("Anima currently supports only diffusers model folders or Hub repos.")

    load_kwargs: dict[str, object] = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if entry.location_type == "hub" and entry.version not in {"", "hub", "local"}:
        load_kwargs["revision"] = entry.version

    pipe = DiffusionPipeline.from_pretrained(source, **load_kwargs)

    if memory_preset == "model_offload":
        pipe.enable_model_cpu_offload()
    elif memory_preset == "sequential_offload":
        pipe.enable_sequential_cpu_offload()
    else:
        raise ValueError(f"Unsupported Anima memory_preset: {memory_preset}")

    if getattr(pipe, "vae", None) is not None:
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()

    cleanup_memory()
    return pipe


def run_text2img_subprocess(params: dict[str, object]) -> dict[str, list[str]]:
    child_params = dict(params)

    with tempfile.TemporaryDirectory(prefix="anima_") as tmpdir:
        input_path = Path(tmpdir) / "input.json"
        output_path = Path(tmpdir) / "output.json"
        input_path.write_text(
            json.dumps(child_params, separators=(",", ": ")),
            encoding="utf-8",
        )

        cmd = [
            sys.executable,
            "-m",
            "backend.anima.subprocess_runner",
            str(input_path),
            str(output_path),
        ]
        with _ANIMA_SUBPROCESS_SEMAPHORE:
            completed = subprocess.run(cmd, cwd=str(_REPO_ROOT))

        if not output_path.exists():
            raise RuntimeError("Anima subprocess failed: No subprocess result was written.")

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if completed.returncode != 0 or not payload.get("ok"):
            detail = payload.get("error") or "Unknown subprocess failure."
            error_type = payload.get("error_type")
            if error_type:
                detail = f"{error_type}: {detail}"
            raise RuntimeError(f"Anima subprocess failed: {detail}")

        result = payload.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("images"), list):
            raise RuntimeError("Anima subprocess returned an invalid result.")

        return {"images": [str(path) for path in result["images"]]}


def generate_text2img(params: dict[str, object]) -> dict[str, list[str]]:
    execution_mode = params.get("execution_mode")
    if execution_mode not in (None, "", "subprocess"):
        raise ValueError("Anima supports only subprocess execution.")
    return run_text2img_subprocess(params)


@torch.inference_mode()
def _generate_text2img_subprocess_child(params: dict[str, object]) -> dict[str, list[str]]:
    prompt = str(params.get("prompt") or "")
    negative_prompt = str(params.get("negative_prompt") or "")
    steps = int(params.get("steps", 35))
    guidance_scale = float(params.get("guidance_scale", 4.5))
    width = int(params.get("width", 1024))
    height = int(params.get("height", 1024))
    seed = params.get("seed")
    model = params.get("model")
    num_images = int(params.get("num_images", 1))
    scheduler = str(params.get("scheduler") or "flowmatch_euler")
    memory_preset = str(params.get("memory_preset") or "sequential_offload")

    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)

    logger.info(
        "Anima T2I Generate: model=%s seed=%s steps=%s guidance_scale=%s size=%sx%s num_images=%s memory_preset=%s",
        model,
        base_seed,
        steps,
        guidance_scale,
        width,
        height,
        num_images,
        memory_preset,
    )

    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)
    pipe = None
    filenames: list[str] = []
    try:
        pipe = load_text2img_pipeline(
            str(model) if model else None,
            memory_preset=memory_preset,  # type: ignore[arg-type]
        )
        if scheduler:
            pipe.scheduler = create_scheduler(scheduler, pipe)

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
            }
            image = pipe(**call_kwargs).images[0]

            filename = batch_output_dir / f"{batch_id}_{current_seed}.png"
            image_params = dict(params)
            image_params.update(
                {
                    "mode": "txt2img",
                    "pipeline": "anima",
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
