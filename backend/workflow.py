from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, TypeAlias

from PIL import Image
from pydantic import BaseModel, Field
import re
from pydantic_core import PydanticUndefined

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


TaskType = Literal[
    "sd15.text2img",
    "sd15.img2img",
    "sd15.inpaint",
    "sd15.controlnet.text2img",
    "sd15.hires_fix",
    "controlnet.preprocess",
    "sdxl.text2img",
    "sdxl.img2img",
    "sdxl.inpaint",
    "flux.text2img",
    "flux.img2img",
    "flux.inpaint",
    "qwen-image.text2img",
    "qwen-image.img2img",
    "qwen-image.inpaint",
    "z-image.text2img",
    "z-image.img2img",
]


class ArtifactRef(BaseModel):
    artifact_id: str = Field(
        ...,
        description="Artifact id returned by POST /api/artifacts.",
        pattern=r"^[ap][0-9a-f]{32}$",
        examples=["a0123456789abcdef0123456789abcdef"],
    )


ImageRef: TypeAlias = ArtifactRef | str


class ArtifactInfo(BaseModel):
    artifact_id: str
    url: str
    path: str


class Sd15Text2ImgInputs(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = 20
    cfg: float = 7.5
    width: int = 512
    height: int = 512
    seed: int | None = None
    scheduler: str = "euler"
    model: str | None = None
    num_images: int = 1
    clip_skip: int = 1
    lora_adapters: Any | None = None
    hires_enabled: bool = False
    hires_scale: float = 1.0
    weighting_policy: str = "diffusers-like"
    lora_scale: float | None = None
    batch_id: str | None = None


class Sd15Img2ImgInputs(BaseModel):
    initial_image: ImageRef = Field(
        ...,
        description='Image reference: {"artifact_id":"..."} OR "@artifact:..." OR "/outputs/...".',
    )
    prompt: str
    negative_prompt: str = ""
    strength: float = 0.75
    steps: int = 20
    cfg: float = 7.5
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    scheduler: str = "euler"
    model: str | None = None
    num_images: int = 1
    clip_skip: int = 1
    lora_adapters: Any | None = None
    batch_id: str | None = None


class Sd15InpaintInputs(BaseModel):
    initial_image: ImageRef = Field(
        ...,
        description='Image reference: {"artifact_id":"..."} OR "@artifact:..." OR "/outputs/...".',
    )
    mask_image: ImageRef = Field(
        ...,
        description='Mask reference: {"artifact_id":"..."} OR "@artifact:..." OR "/outputs/...".',
    )
    prompt: str
    negative_prompt: str = ""
    strength: float = 0.5
    steps: int = 20
    cfg: float = 7.5
    seed: int | None = None
    scheduler: str = "euler"
    model: str | None = None
    num_images: int = 1
    padding_mask_crop: int = 32
    clip_skip: int = 1
    batch_id: str | None = None


class Sd15ControlNetText2ImgInputs(BaseModel):
    control_image: ImageRef = Field(
        ...,
        description='Control image reference: {"artifact_id":"..."} OR "@artifact:..." OR "/outputs/...".',
    )
    prompt: str
    negative_prompt: str = ""
    steps: int = 20
    cfg: float = 7.5
    width: int = 512
    height: int = 512
    seed: int | None = None
    scheduler: str = "euler"
    model: str | None = None
    num_images: int = 1
    clip_skip: int = 1
    controlnet_model: str = "lllyasviel/sd-controlnet-canny"
    lora_adapters: Any | None = None
    batch_id: str | None = None


class ControlNetPreprocessInputs(BaseModel):
    image: ImageRef = Field(
        ...,
        description='Source image reference: {"artifact_id":"..."} OR "@artifact:..." OR "/outputs/...".',
    )
    preprocessor_id: str
    params: dict[str, Any] = Field(default_factory=dict)


class Sd15HiresFixInputs(BaseModel):
    images: list[ImageRef] = Field(
        ...,
        description='List of image references (usually from @t1.images).',
    )
    prompt: str
    negative_prompt: str = ""
    steps: int = 20
    cfg: float = 7.5
    seed: int | None = None
    scheduler: str = "euler"
    model: str | None = None
    clip_skip: int = 1
    hires_scale: float = 1.0
    hires_strength: float = 0.35
    lora_adapters: Any | None = None
    weighting_policy: str = "diffusers-like"
    lora_scale: float | None = None
    batch_id: str | None = None


class SdxlText2ImgInputs(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    steps: int = 20
    guidance_scale: float = 7.5
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    model: str | None = None
    num_images: int = 1
    clip_skip: int = 1
    scheduler: str = "euler"


class SdxlImg2ImgInputs(BaseModel):
    initial_image: ImageRef
    prompt: str
    negative_prompt: str = ""
    strength: float = 0.75
    steps: int = 20
    guidance_scale: float = 7.5
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    scheduler: str = "euler"
    model: str | None = None
    num_images: int = 1
    clip_skip: int = 1


class SdxlInpaintInputs(BaseModel):
    initial_image: ImageRef
    mask_image: ImageRef
    prompt: str
    negative_prompt: str = ""
    strength: float = 0.5
    steps: int = 20
    guidance_scale: float = 7.5
    seed: int | None = None
    scheduler: str = "euler"
    model: str | None = None
    num_images: int = 1
    padding_mask_crop: int = 32
    clip_skip: int = 1


class FluxText2ImgInputs(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    steps: int = 20
    guidance_scale: float = 0.0
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    model: str | None = None
    num_images: int = 1
    scheduler: str = "euler"


class FluxImg2ImgInputs(BaseModel):
    initial_image: ImageRef
    prompt: str
    negative_prompt: str = ""
    strength: float = 0.75
    steps: int = 20
    guidance_scale: float = 0.0
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    scheduler: str = "euler"
    model: str | None = None
    num_images: int = 1


class FluxInpaintInputs(BaseModel):
    initial_image: ImageRef
    mask_image: ImageRef
    prompt: str
    negative_prompt: str = ""
    strength: float = 0.5
    steps: int = 20
    guidance_scale: float = 0.0
    seed: int | None = None
    scheduler: str = "euler"
    model: str | None = None
    num_images: int = 1


class QwenImageText2ImgInputs(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    steps: int = 30
    true_cfg_scale: float = 4.0
    guidance_scale: float = 7.5
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    model: str | None = None
    num_images: int = 1
    scheduler: str = "euler"


class QwenImageImg2ImgInputs(BaseModel):
    initial_image: ImageRef
    prompt: str
    negative_prompt: str = ""
    strength: float = 0.75
    steps: int = 30
    true_cfg_scale: float = 4.0
    guidance_scale: float = 7.5
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    scheduler: str = "euler"
    model: str | None = None
    num_images: int = 1


class QwenImageInpaintInputs(BaseModel):
    initial_image: ImageRef
    mask_image: ImageRef
    prompt: str
    negative_prompt: str = ""
    strength: float = 0.5
    steps: int = 30
    true_cfg_scale: float = 4.0
    guidance_scale: float = 7.5
    seed: int | None = None
    scheduler: str = "euler"
    model: str | None = None
    num_images: int = 1


class ZImageText2ImgInputs(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    steps: int = 8
    guidance_scale: float = 0.0
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    model: str | None = None
    num_images: int = 1
    scheduler: str = "euler"


class ZImageImg2ImgInputs(BaseModel):
    initial_image: ImageRef
    prompt: str
    negative_prompt: str = ""
    strength: float = 0.75
    steps: int = 8
    guidance_scale: float = 0.0
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    scheduler: str = "euler"
    model: str | None = None
    num_images: int = 1


class ImagesOutput(BaseModel):
    """Standard output for image-generating tasks."""

    images: list[str] = Field(
        ...,
        description='List of output image URLs ("/outputs/...").',
    )


class ImagesWithBatchOutput(BaseModel):
    """Image-generating output that includes a batch id."""

    batch_id: str = Field(..., description="Batch identifier used to group outputs on disk.")
    images: list[str] = Field(
        ...,
        description='List of output image URLs ("/outputs/...").',
    )


class ControlNetPreprocessOutput(BaseModel):
    """Output of controlnet.preprocess (produces a new artifact)."""

    artifact: ArtifactInfo


TASK_INPUT_MODELS: dict[str, type[BaseModel]] = {
    "sd15.text2img": Sd15Text2ImgInputs,
    "sd15.img2img": Sd15Img2ImgInputs,
    "sd15.inpaint": Sd15InpaintInputs,
    "sd15.controlnet.text2img": Sd15ControlNetText2ImgInputs,
    "sd15.hires_fix": Sd15HiresFixInputs,
    "controlnet.preprocess": ControlNetPreprocessInputs,
    "sdxl.text2img": SdxlText2ImgInputs,
    "sdxl.img2img": SdxlImg2ImgInputs,
    "sdxl.inpaint": SdxlInpaintInputs,
    "flux.text2img": FluxText2ImgInputs,
    "flux.img2img": FluxImg2ImgInputs,
    "flux.inpaint": FluxInpaintInputs,
    "qwen-image.text2img": QwenImageText2ImgInputs,
    "qwen-image.img2img": QwenImageImg2ImgInputs,
    "qwen-image.inpaint": QwenImageInpaintInputs,
    "z-image.text2img": ZImageText2ImgInputs,
    "z-image.img2img": ZImageImg2ImgInputs,
}


TASK_OUTPUT_MODELS: dict[str, type[BaseModel]] = {
    "sd15.text2img": ImagesWithBatchOutput,
    "sd15.img2img": ImagesWithBatchOutput,
    "sd15.inpaint": ImagesWithBatchOutput,
    "sd15.controlnet.text2img": ImagesWithBatchOutput,
    "sd15.hires_fix": ImagesWithBatchOutput,
    "controlnet.preprocess": ControlNetPreprocessOutput,
    "sdxl.text2img": ImagesOutput,
    "sdxl.img2img": ImagesOutput,
    "sdxl.inpaint": ImagesOutput,
    "flux.text2img": ImagesOutput,
    "flux.img2img": ImagesOutput,
    "flux.inpaint": ImagesOutput,
    "qwen-image.text2img": ImagesOutput,
    "qwen-image.img2img": ImagesOutput,
    "qwen-image.inpaint": ImagesOutput,
    "z-image.text2img": ImagesOutput,
    "z-image.img2img": ImagesOutput,
}


_SCHEDULER_OPTIONS: list[str] = [
    "euler",
    "euler_a",
    "ddim",
    "dpm++2m",
    "dpm++2m_karras",
    "dpm++2m_sde",
    "dpm++2m_sde_karras",
    "dpm++_sde",
    "dpm++_sde_karras",
    "dpm2",
    "dpm2_karras",
    "dpm2_a",
    "dpm2_a_karras",
    "flowmatch_euler",
    "flowmatch_heun",
    "heun",
    "lms",
    "lms_karras",
    "deis",
    "unipc",
]


_WEIGHTING_POLICY_OPTIONS: list[str] = [
    "diffusers-like",
    "a1111-like",
    "comfyui-like",
]


def _infer_model_family(task_type: str) -> str | None:
    prefix = task_type.split(".", 1)[0]
    if prefix in {"sd15", "sdxl", "flux"}:
        return prefix
    if prefix == "qwen-image":
        return "qwen-image"
    if prefix == "z-image":
        return "z-image"
    return None


def _build_task_ui_hints(task_type: str, model_cls: type[BaseModel]) -> dict[str, Any]:
    # Minimal, stable contract for UIs/workflow builders. Everything here is optional
    # and should be treated as best-effort.
    family = _infer_model_family(task_type)

    title = task_type
    if task_type.endswith(".text2img"):
        title = f"{task_type} (Text to Image)"
    elif task_type.endswith(".img2img"):
        title = f"{task_type} (Image to Image)"
    elif task_type.endswith(".inpaint"):
        title = f"{task_type} (Inpaint)"
    elif task_type == "controlnet.preprocess":
        title = "controlnet.preprocess (Preprocessor)"
    elif task_type == "sd15.hires_fix":
        title = "sd15.hires_fix (Hires Fix)"

    inputs: dict[str, Any] = {}
    input_order: list[str] = []

    common_numeric: dict[str, dict[str, Any]] = {
        "steps": {"min": 1, "max": 200, "step": 1, "integer": True},
        "cfg": {"min": 0, "max": 30, "step": 0.1},
        "guidance_scale": {"min": 0, "max": 30, "step": 0.1},
        "true_cfg_scale": {"min": 0, "max": 30, "step": 0.1},
        "strength": {"min": 0, "max": 1, "step": 0.01},
        "width": {"min": 64, "max": 2048, "step": 8, "integer": True},
        "height": {"min": 64, "max": 2048, "step": 8, "integer": True},
        "num_images": {"min": 1, "max": 8, "step": 1, "integer": True},
        "clip_skip": {"min": 1, "max": 4, "step": 1, "integer": True},
        "padding_mask_crop": {"min": 0, "max": 128, "step": 1, "integer": True},
        "hires_scale": {"min": 1, "max": 4, "step": 0.05},
        "hires_strength": {"min": 0, "max": 1, "step": 0.01},
        "lora_scale": {"min": 0, "max": 2, "step": 0.05},
        "seed": {"min": 0, "max": 2**31 - 1, "step": 1, "integer": True},
    }

    for field_name, field_info in model_cls.model_fields.items():
        input_order.append(field_name)
        hint: dict[str, Any] = {"label": field_name.replace("_", " ").title()}

        if field_name in {"prompt", "negative_prompt"}:
            hint.update(
                widget="textarea",
                placeholder="",
                multiline=True,
            )
            if field_name == "prompt":
                hint["placeholder"] = "Describe what you want to generate..."
            else:
                hint["placeholder"] = "Describe what to avoid..."

        if field_name in {"initial_image", "mask_image", "control_image", "image"}:
            hint.update(
                widget="image_ref",
                accepts=["artifact", "outputs", "task_ref"],
                help="Upload via /api/artifacts, or reference a prior task output (e.g. @t1.images[0]).",
            )

        if field_name == "images":
            hint.update(
                widget="image_list_ref",
                accepts=["artifact", "outputs", "task_ref"],
            )

        if field_name == "scheduler":
            hint.update(widget="select", options=_SCHEDULER_OPTIONS)

        if field_name == "weighting_policy":
            hint.update(widget="select", options=_WEIGHTING_POLICY_OPTIONS)

        if field_name == "model":
            if family:
                hint.update(
                    widget="model_select",
                    source={"type": "models", "params": {"family": family}},
                )
            else:
                hint.update(widget="text")

        if field_name == "preprocessor_id":
            hint.update(
                widget="select",
                source={"type": "controlnet_preprocessors", "endpoint": "/api/controlnet/preprocessors"},
            )

        if field_name == "lora_adapters":
            hint.update(
                widget="json",
                advanced=True,
                help="List of LoRA adapter objects; UI may provide a dedicated editor.",
            )

        if field_name in common_numeric:
            hint.setdefault("widget", "number")
            hint.update(common_numeric[field_name])

        if field_info.annotation is bool or str(field_info.annotation) == "bool":
            hint.setdefault("widget", "checkbox")

        inputs[field_name] = hint

    return {
        "title": title,
        "task_type": task_type,
        "input_order": input_order,
        "inputs": inputs,
    }


def build_workflow_catalog() -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for task_type, model_cls in TASK_INPUT_MODELS.items():
        defaults: dict[str, Any] = {}
        for field_name, field_info in model_cls.model_fields.items():
            if field_info.default is not PydanticUndefined:
                defaults[field_name] = field_info.default
                continue
            if field_info.default_factory is not None:
                defaults[field_name] = field_info.default_factory()
        output_model = TASK_OUTPUT_MODELS.get(task_type)
        tasks[task_type] = {
            "input_schema": model_cls.model_json_schema(by_alias=True),
            "input_defaults": defaults,
            "output_schema": output_model.model_json_schema(by_alias=True) if output_model else None,
            "ui_hints": _build_task_ui_hints(task_type, model_cls),
        }
    return {"version": "v2", "tasks": tasks}


class WorkflowTask(BaseModel):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    type: TaskType
    inputs: dict[str, Any] = Field(default_factory=dict)


class WorkflowRequest(BaseModel):
    tasks: list[WorkflowTask] = Field(max_length=64)
    return_value: Any | None = Field(default=None, alias="return")


@dataclass(frozen=True)
class WorkflowContext:
    update_progress: Callable[[dict[str, Any]], None] | None = None
    should_cancel: Callable[[], bool] | None = None


class WorkflowCanceled(Exception):
    pass


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


def collect_artifact_ids(value: Any) -> set[str]:
    out: set[str] = set()

    def _walk(node: Any) -> None:
        if isinstance(node, str):
            if node.startswith("@artifact:"):
                try:
                    out.add(_validate_artifact_id(node.removeprefix("@artifact:")))
                except ValueError:
                    pass
            return

        if isinstance(node, dict):
            artifact_id = node.get("artifact_id")
            if isinstance(artifact_id, str):
                try:
                    out.add(_validate_artifact_id(artifact_id))
                except ValueError:
                    pass
            for v in node.values():
                _walk(v)
            return

        if isinstance(node, list):
            for item in node:
                _walk(item)
            return

    _walk(value)
    return out


def cleanup_artifacts(artifact_ids: set[str]) -> None:
    if not artifact_ids:
        return
    artifacts_dir = _artifact_dir().resolve()
    for artifact_id in artifact_ids:
        try:
            safe_id = _validate_artifact_id(artifact_id)
        except ValueError:
            continue
        path = (artifacts_dir / f"{safe_id}.png").resolve()
        if not str(path).startswith(str(artifacts_dir)):
            continue
        try:
            path.unlink(missing_ok=True)
        except Exception:
            # Best-effort cleanup; job success shouldn't depend on deletion.
            pass


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

def _sdxl_text2img(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    from backend.sdxl_pipeline import run_sdxl_text2img

    result = run_sdxl_text2img(dict(inputs))
    if not isinstance(result, dict):
        raise ValueError("sdxl.text2img must return an object")
    return result


def _sdxl_img2img(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    from backend.sdxl_pipeline import run_sdxl_img2img

    initial_image = _open_image_ref(inputs["initial_image"]).convert("RGB")
    width = int(inputs.get("width") or 1024)
    height = int(inputs.get("height") or 1024)
    initial_image = initial_image.resize((width, height))

    strength = float(inputs.get("strength") or 0.75)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    strength = _remap_img2img_strength(strength)

    result = run_sdxl_img2img(
        initial_image=initial_image,
        strength=strength,
        prompt=str(inputs["prompt"]),
        negative_prompt=str(inputs.get("negative_prompt") or ""),
        steps=int(inputs.get("steps") or 20),
        guidance_scale=float(inputs.get("guidance_scale") or inputs.get("cfg") or 7.5),
        width=width,
        height=height,
        seed=inputs.get("seed"),
        scheduler=str(inputs.get("scheduler") or "euler"),
        model=inputs.get("model"),
        num_images=int(inputs.get("num_images") or 1),
        clip_skip=int(inputs.get("clip_skip") or 1),
    )
    if not isinstance(result, dict):
        raise ValueError("sdxl.img2img must return an object")
    return result


def _sdxl_inpaint(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    from backend.sdxl_pipeline import run_sdxl_inpaint

    initial_image = _open_image_ref(inputs["initial_image"]).convert("RGB")
    mask_image = _open_image_ref(inputs["mask_image"]).convert("L")
    if mask_image.size != initial_image.size:
        mask_image = mask_image.resize(initial_image.size, resample=Image.NEAREST)

    strength = float(inputs.get("strength") or 0.5)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    strength = _remap_img2img_strength(strength)

    result = run_sdxl_inpaint(
        initial_image=initial_image,
        mask_image=mask_image,
        strength=strength,
        prompt=str(inputs["prompt"]),
        negative_prompt=str(inputs.get("negative_prompt") or ""),
        steps=int(inputs.get("steps") or 20),
        guidance_scale=float(inputs.get("guidance_scale") or inputs.get("cfg") or 7.5),
        seed=inputs.get("seed"),
        scheduler=str(inputs.get("scheduler") or "euler"),
        model=inputs.get("model"),
        num_images=int(inputs.get("num_images") or 1),
        padding_mask_crop=int(inputs.get("padding_mask_crop") or 32),
        clip_skip=int(inputs.get("clip_skip") or 1),
    )
    if not isinstance(result, dict):
        raise ValueError("sdxl.inpaint must return an object")
    return result


def _flux_text2img(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    from backend.flux_pipeline import run_flux_text2img

    result = run_flux_text2img(dict(inputs))
    if not isinstance(result, dict):
        raise ValueError("flux.text2img must return an object")
    return result


def _flux_img2img(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    from backend.flux_pipeline import run_flux_img2img

    initial_image = _open_image_ref(inputs["initial_image"]).convert("RGB")
    width = int(inputs.get("width") or 1024)
    height = int(inputs.get("height") or 1024)
    initial_image = initial_image.resize((width, height))

    strength = float(inputs.get("strength") or 0.75)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")

    result = run_flux_img2img(
        initial_image=initial_image,
        strength=strength,
        prompt=str(inputs["prompt"]),
        negative_prompt=str(inputs.get("negative_prompt") or ""),
        steps=int(inputs.get("steps") or 20),
        guidance_scale=float(inputs.get("guidance_scale") or 0.0),
        width=width,
        height=height,
        seed=inputs.get("seed"),
        scheduler=str(inputs.get("scheduler") or "euler"),
        model=inputs.get("model"),
        num_images=int(inputs.get("num_images") or 1),
    )
    if not isinstance(result, dict):
        raise ValueError("flux.img2img must return an object")
    return result


def _flux_inpaint(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    from backend.flux_pipeline import run_flux_inpaint

    initial_image = _open_image_ref(inputs["initial_image"]).convert("RGB")
    mask_image = _open_image_ref(inputs["mask_image"]).convert("L")
    if mask_image.size != initial_image.size:
        mask_image = mask_image.resize(initial_image.size, resample=Image.NEAREST)

    strength = float(inputs.get("strength") or 0.5)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")

    result = run_flux_inpaint(
        initial_image=initial_image,
        mask_image=mask_image,
        strength=strength,
        prompt=str(inputs["prompt"]),
        negative_prompt=str(inputs.get("negative_prompt") or ""),
        steps=int(inputs.get("steps") or 20),
        guidance_scale=float(inputs.get("guidance_scale") or 0.0),
        seed=inputs.get("seed"),
        scheduler=str(inputs.get("scheduler") or "euler"),
        model=inputs.get("model"),
        num_images=int(inputs.get("num_images") or 1),
    )
    if not isinstance(result, dict):
        raise ValueError("flux.inpaint must return an object")
    return result


def _qwen_image_text2img(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    from backend.qwen_image_pipeline import run_qwen_image_text2img

    result = run_qwen_image_text2img(dict(inputs))
    if not isinstance(result, dict):
        raise ValueError("qwen-image.text2img must return an object")
    return result


def _qwen_image_img2img(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    from backend.qwen_image_pipeline import run_qwen_image_img2img

    initial_image = _open_image_ref(inputs["initial_image"]).convert("RGB")
    width = int(inputs.get("width") or 1024)
    height = int(inputs.get("height") or 1024)
    initial_image = initial_image.resize((width, height))

    strength = float(inputs.get("strength") or 0.75)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    strength = _remap_img2img_strength(strength)

    result = run_qwen_image_img2img(
        initial_image=initial_image,
        strength=strength,
        prompt=str(inputs["prompt"]),
        negative_prompt=str(inputs.get("negative_prompt") or ""),
        steps=int(inputs.get("steps") or 30),
        true_cfg_scale=float(inputs.get("true_cfg_scale") or 4.0),
        guidance_scale=float(inputs.get("guidance_scale") or 7.5),
        width=width,
        height=height,
        seed=inputs.get("seed"),
        scheduler=str(inputs.get("scheduler") or "euler"),
        model=inputs.get("model"),
        num_images=int(inputs.get("num_images") or 1),
    )
    if not isinstance(result, dict):
        raise ValueError("qwen-image.img2img must return an object")
    return result


def _qwen_image_inpaint(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    from backend.qwen_image_pipeline import run_qwen_image_inpaint

    initial_image = _open_image_ref(inputs["initial_image"]).convert("RGB")
    mask_image = _open_image_ref(inputs["mask_image"]).convert("L")
    if mask_image.size != initial_image.size:
        mask_image = mask_image.resize(initial_image.size, resample=Image.NEAREST)

    strength = float(inputs.get("strength") or 0.5)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")

    result = run_qwen_image_inpaint(
        initial_image=initial_image,
        mask_image=mask_image,
        strength=strength,
        prompt=str(inputs["prompt"]),
        negative_prompt=str(inputs.get("negative_prompt") or ""),
        steps=int(inputs.get("steps") or 30),
        true_cfg_scale=float(inputs.get("true_cfg_scale") or 4.0),
        guidance_scale=float(inputs.get("guidance_scale") or 7.5),
        seed=inputs.get("seed"),
        scheduler=str(inputs.get("scheduler") or "euler"),
        model=inputs.get("model"),
        num_images=int(inputs.get("num_images") or 1),
    )
    if not isinstance(result, dict):
        raise ValueError("qwen-image.inpaint must return an object")
    return result


def _z_image_text2img(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    from backend.z_image_pipeline import run_z_image_text2img

    result = run_z_image_text2img(dict(inputs))
    if not isinstance(result, dict):
        raise ValueError("z-image.text2img must return an object")
    return result


def _z_image_img2img(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    from backend.z_image_pipeline import run_z_image_img2img

    initial_image = _open_image_ref(inputs["initial_image"]).convert("RGB")
    width = int(inputs.get("width") or 1024)
    height = int(inputs.get("height") or 1024)
    initial_image = initial_image.resize((width, height))

    strength = float(inputs.get("strength") or 0.75)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    strength = _remap_img2img_strength(strength)

    result = run_z_image_img2img(
        initial_image=initial_image,
        strength=strength,
        prompt=str(inputs["prompt"]),
        negative_prompt=str(inputs.get("negative_prompt") or ""),
        steps=int(inputs.get("steps") or 8),
        guidance_scale=float(inputs.get("guidance_scale") or 0.0),
        width=width,
        height=height,
        seed=inputs.get("seed"),
        scheduler=str(inputs.get("scheduler") or "euler"),
        model=inputs.get("model"),
        num_images=int(inputs.get("num_images") or 1),
    )
    if not isinstance(result, dict):
        raise ValueError("z-image.img2img must return an object")
    return result


TASK_REGISTRY: dict[str, Callable[[dict[str, Any], WorkflowContext], dict[str, Any]]] = {
    "sd15.text2img": _sd15_text2img,
    "sd15.img2img": _sd15_img2img,
    "sd15.inpaint": _sd15_inpaint,
    "sd15.controlnet.text2img": _sd15_controlnet_text2img,
    "sd15.hires_fix": _sd15_hires_fix,
    "controlnet.preprocess": _controlnet_preprocess,
    "sdxl.text2img": _sdxl_text2img,
    "sdxl.img2img": _sdxl_img2img,
    "sdxl.inpaint": _sdxl_inpaint,
    "flux.text2img": _flux_text2img,
    "flux.img2img": _flux_img2img,
    "flux.inpaint": _flux_inpaint,
    "qwen-image.text2img": _qwen_image_text2img,
    "qwen-image.img2img": _qwen_image_img2img,
    "qwen-image.inpaint": _qwen_image_inpaint,
    "z-image.text2img": _z_image_text2img,
    "z-image.img2img": _z_image_img2img,
}


def execute_workflow(payload: dict[str, Any], *, ctx: WorkflowContext | None = None) -> dict[str, Any]:
    wf = WorkflowRequest.model_validate(payload)
    context = ctx or WorkflowContext()

    task_results: dict[str, dict[str, Any]] = {}
    created_artifacts: set[str] = set()
    try:
        for idx, task in enumerate(wf.tasks):
            if context.should_cancel and context.should_cancel():
                raise WorkflowCanceled("Cancel requested")
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
            created_artifacts |= collect_artifact_ids(result)
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
    except Exception as exc:
        setattr(exc, "_workflow_created_artifacts", created_artifacts)
        raise

    if wf.return_value is None:
        final_value: Any = task_results[wf.tasks[-1].id] if wf.tasks else {}
    else:
        final_value = _resolve_refs(wf.return_value, task_results)

    return {"outputs": final_value, "tasks": task_results, "created_artifacts": sorted(created_artifacts)}
