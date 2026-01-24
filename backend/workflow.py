from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from pydantic import BaseModel, Field
import re

from backend.config import OUTPUT_DIR
from backend.controlnet_preprocessors import get_preprocessor
from backend.pipeline_utils import get_batch_output_dir, make_batch_id
from backend.sd15_pipeline import (
    generate_images,
    generate_images_controlnet,
    generate_images_img2img,
    generate_images_inpaint,
    run_sd15_hires_fix,
)


class WorkflowTask(BaseModel):
    id: str
    type: str
    inputs: dict[str, Any] = Field(default_factory=dict)


class WorkflowRequest(BaseModel):
    tasks: list[WorkflowTask]
    return_value: Any | None = Field(default=None, alias="return")


@dataclass(frozen=True)
class WorkflowContext:
    update_progress: Callable[[dict[str, Any]], None] | None = None


def _artifact_dir() -> Path:
    artifacts = OUTPUT_DIR / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    return artifacts


_ARTIFACT_ID_RE = re.compile(r"^[ap][0-9a-f]{32}$")


def _validate_artifact_id(value: str) -> str:
    artifact_id = value.strip()
    if not _ARTIFACT_ID_RE.match(artifact_id):
        raise ValueError("Invalid artifact_id")
    return artifact_id


def save_artifact_png(image: Image.Image, *, prefix: str = "a") -> dict[str, str]:
    artifact_id = f"{prefix}{uuid.uuid4().hex}"
    path = _artifact_dir() / f"{artifact_id}.png"
    image.save(path, format="PNG")
    rel = path.relative_to(OUTPUT_DIR).as_posix()
    return {"artifact_id": artifact_id, "path": rel, "url": f"/outputs/{rel}"}


def _load_image_from_outputs_url(url: str) -> Image.Image:
    if not url.startswith("/outputs/"):
        raise ValueError("Expected /outputs/ URL.")
    rel = url.removeprefix("/outputs/").lstrip("/")
    path = (OUTPUT_DIR / rel).resolve()
    if not str(path).startswith(str(OUTPUT_DIR.resolve())):
        raise ValueError("Invalid outputs path.")
    with Image.open(path) as img:
        return img.copy()

def _open_image_ref(value: Any) -> Image.Image:
    if isinstance(value, dict) and "artifact_id" in value:
        artifact_id = _validate_artifact_id(str(value["artifact_id"]))
        path = (_artifact_dir() / f"{artifact_id}.png").resolve()
        with Image.open(path) as img:
            return img.copy()
    if isinstance(value, str) and value.startswith("@artifact:"):
        artifact_id = _validate_artifact_id(value.removeprefix("@artifact:"))
        path = (_artifact_dir() / f"{artifact_id}.png").resolve()
        with Image.open(path) as img:
            return img.copy()
    if isinstance(value, str) and value.startswith("/outputs/"):
        return _load_image_from_outputs_url(value)
    raise ValueError("Unsupported image reference.")


def _resolve_refs(value: Any, task_results: dict[str, dict[str, Any]]) -> Any:
    if isinstance(value, str) and value.startswith("@"):
        token = value[1:]
        if token.startswith("artifact:"):
            return {"artifact_id": token.removeprefix("artifact:").strip()}
        if "." in token:
            task_id, key = token.split(".", 1)
            if task_id not in task_results:
                raise KeyError(f"Unknown task id: {task_id}")
            return task_results[task_id].get(key)
        raise ValueError(f"Invalid reference: {value}")

    if isinstance(value, list):
        return [_resolve_refs(item, task_results) for item in value]
    if isinstance(value, dict):
        return {k: _resolve_refs(v, task_results) for k, v in value.items()}
    return value


def _sd15_text2img(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    batch_id = str(inputs.get("batch_id") or make_batch_id())
    filenames = generate_images(
        prompt=str(inputs["prompt"]),
        negative_prompt=str(inputs.get("negative_prompt") or ""),
        steps=int(inputs.get("steps") or 20),
        cfg=float(inputs.get("cfg") or 7.5),
        width=int(inputs.get("width") or 512),
        height=int(inputs.get("height") or 512),
        seed=inputs.get("seed"),
        scheduler=str(inputs.get("scheduler") or "euler"),
        model=inputs.get("model"),
        num_images=int(inputs.get("num_images") or 1),
        clip_skip=int(inputs.get("clip_skip") or 1),
        lora_adapters=inputs.get("lora_adapters"),
        hires_enabled=bool(inputs.get("hires_enabled") or False),
        hires_scale=float(inputs.get("hires_scale") or 1.0),
        weighting_policy=str(inputs.get("weighting_policy") or "diffusers-like"),
        batch_id=batch_id,
        lora_scale=inputs.get("lora_scale"),
    )
    return {"batch_id": batch_id, "images": [f"/outputs/{name}" for name in filenames]}


def _remap_img2img_strength(strength: float, *, min_strength: float = 0.0, gamma: float = 0.5) -> float:
    clamped = max(0.0, min(1.0, strength))
    if min_strength <= 0.0:
        remapped = clamped**gamma
    else:
        normalized = max(0.0, min(1.0, (clamped - min_strength) / (1.0 - min_strength)))
        remapped = min_strength + (normalized**gamma) * (1.0 - min_strength)
    return max(0.0, min(1.0, remapped))


def _sd15_img2img(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    initial_image = _open_image_ref(inputs["initial_image"]).convert("RGB")
    width = int(inputs.get("width") or initial_image.width)
    height = int(inputs.get("height") or initial_image.height)
    initial_image = initial_image.resize((width, height))

    strength = float(inputs.get("strength") or 0.75)
    strength = _remap_img2img_strength(strength)
    batch_id = str(inputs.get("batch_id") or make_batch_id())

    filenames = generate_images_img2img(
        initial_image=initial_image,
        strength=strength,
        prompt=str(inputs["prompt"]),
        negative_prompt=str(inputs.get("negative_prompt") or ""),
        steps=int(inputs.get("steps") or 20),
        cfg=float(inputs.get("cfg") or 7.5),
        width=width,
        height=height,
        seed=inputs.get("seed"),
        scheduler=str(inputs.get("scheduler") or "euler"),
        model=inputs.get("model"),
        num_images=int(inputs.get("num_images") or 1),
        clip_skip=int(inputs.get("clip_skip") or 1),
        lora_adapters=inputs.get("lora_adapters"),
        batch_id=batch_id,
    )
    return {"batch_id": batch_id, "images": [f"/outputs/{name}" for name in filenames]}


def _sd15_inpaint(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    initial_image = _open_image_ref(inputs["initial_image"]).convert("RGB")
    mask_image = _open_image_ref(inputs["mask_image"]).convert("L")
    if mask_image.size != initial_image.size:
        mask_image = mask_image.resize(initial_image.size)

    strength = float(inputs.get("strength") or 0.5)
    strength = _remap_img2img_strength(strength)
    batch_id = str(inputs.get("batch_id") or make_batch_id())

    filenames = generate_images_inpaint(
        initial_image=initial_image,
        mask_image=mask_image,
        prompt=str(inputs["prompt"]),
        negative_prompt=str(inputs.get("negative_prompt") or ""),
        steps=int(inputs.get("steps") or 20),
        cfg=float(inputs.get("cfg") or 7.5),
        seed=inputs.get("seed"),
        scheduler=str(inputs.get("scheduler") or "euler"),
        model=inputs.get("model"),
        num_images=int(inputs.get("num_images") or 1),
        strength=strength,
        padding_mask_crop=int(inputs.get("padding_mask_crop") or 32),
        clip_skip=int(inputs.get("clip_skip") or 1),
        batch_id=batch_id,
    )
    return {"batch_id": batch_id, "images": [f"/outputs/{name}" for name in filenames]}


def _sd15_controlnet_text2img(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    control_image = _open_image_ref(inputs["control_image"]).convert("RGB")
    width = int(inputs.get("width") or 512)
    height = int(inputs.get("height") or 512)
    control_image = control_image.resize((width, height))
    batch_id = str(inputs.get("batch_id") or make_batch_id())

    filenames = generate_images_controlnet(
        prompt=str(inputs["prompt"]),
        negative_prompt=str(inputs.get("negative_prompt") or ""),
        steps=int(inputs.get("steps") or 20),
        cfg=float(inputs.get("cfg") or 7.5),
        width=width,
        height=height,
        seed=inputs.get("seed"),
        scheduler=str(inputs.get("scheduler") or "euler"),
        model=inputs.get("model"),
        num_images=int(inputs.get("num_images") or 1),
        clip_skip=int(inputs.get("clip_skip") or 1),
        controlnet_image=control_image,
        controlnet_model=str(inputs.get("controlnet_model") or "lllyasviel/sd-controlnet-canny"),
        batch_id=batch_id,
    )
    return {"batch_id": batch_id, "images": [f"/outputs/{name}" for name in filenames]}


def _controlnet_preprocess(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    source = _open_image_ref(inputs["image"]).convert("RGB")
    preprocessor_id = str(inputs["preprocessor_id"])
    preprocessor = get_preprocessor(preprocessor_id)

    params = inputs.get("params") or {}
    if isinstance(params, str):
        params = json.loads(params)
    if not isinstance(params, dict):
        raise ValueError("params must be an object")

    for key in ("low_threshold", "high_threshold"):
        if inputs.get(key) is not None:
            params[key] = inputs[key]

    processed = preprocessor.process(source, params)
    artifact = save_artifact_png(processed, prefix="p")
    return {"artifact": artifact}


def _sd15_hires_fix(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    images_in = inputs["images"]
    if not isinstance(images_in, list):
        raise ValueError("images must be a list")
    images = [_open_image_ref(item).convert("RGB") for item in images_in]

    batch_id = str(inputs.get("batch_id") or make_batch_id())
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    relpaths = run_sd15_hires_fix(
        images=images,
        prompt=str(inputs["prompt"]),
        negative_prompt=str(inputs.get("negative_prompt") or ""),
        steps=int(inputs.get("steps") or 20),
        cfg=float(inputs.get("cfg") or 7.5),
        seed=inputs.get("seed"),
        scheduler=str(inputs.get("scheduler") or "euler"),
        model=inputs.get("model"),
        clip_skip=int(inputs.get("clip_skip") or 1),
        hires_scale=float(inputs.get("hires_scale") or 1.0),
        hires_strength=float(inputs.get("hires_strength") or 0.35),
        lora_adapters=inputs.get("lora_adapters"),
        weighting_policy=str(inputs.get("weighting_policy") or "diffusers-like"),
        lora_scale=inputs.get("lora_scale"),
        output_dir=batch_output_dir,
        batch_id=batch_id,
    )
    return {"batch_id": batch_id, "images": [f"/outputs/{p}" for p in relpaths]}


TASK_REGISTRY: dict[str, Callable[[dict[str, Any], WorkflowContext], dict[str, Any]]] = {
    "sd15.text2img": _sd15_text2img,
    "sd15.img2img": _sd15_img2img,
    "sd15.inpaint": _sd15_inpaint,
    "sd15.controlnet.text2img": _sd15_controlnet_text2img,
    "sd15.hires_fix": _sd15_hires_fix,
    "controlnet.preprocess": _controlnet_preprocess,
}


def execute_workflow(payload: dict[str, Any], *, ctx: WorkflowContext | None = None) -> dict[str, Any]:
    wf = WorkflowRequest.model_validate(payload)
    context = ctx or WorkflowContext()

    task_results: dict[str, dict[str, Any]] = {}
    for idx, task in enumerate(wf.tasks):
        if task.id in task_results:
            raise ValueError(f"Duplicate task id: {task.id}")

        resolved_inputs = _resolve_refs(task.inputs, task_results)
        handler = TASK_REGISTRY.get(task.type)
        if handler is None:
            raise ValueError(f"Unsupported task type: {task.type}")

        if context.update_progress:
            context.update_progress(
                {
                    "current_task": task.id,
                    "current_task_index": idx,
                    "total_tasks": len(wf.tasks),
                    "phase": "running",
                }
            )

        result = handler(resolved_inputs, context)
        if not isinstance(result, dict):
            raise ValueError(f"Task {task.id} must return an object")
        task_results[task.id] = result

        if context.update_progress:
            context.update_progress(
                {
                    "current_task": task.id,
                    "current_task_index": idx,
                    "total_tasks": len(wf.tasks),
                    "phase": "completed_task",
                }
            )

    if wf.return_value is None:
        final_value: Any = task_results[wf.tasks[-1].id] if wf.tasks else {}
    else:
        final_value = _resolve_refs(wf.return_value, task_results)

    return {"outputs": final_value, "tasks": task_results}
