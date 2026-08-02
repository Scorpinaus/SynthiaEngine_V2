from __future__ import annotations

from typing import Any

from backend.workflow.registry import TaskDefinition, TaskHandler, bind_task
from backend.workflow.schema_input import WanImage2VideoInputs, WanText2VideoInputs
from backend.workflow.schema_output import VideosWithBatchOutput


def task_definitions(handlers: dict[str, TaskHandler]) -> dict[str, TaskDefinition]:
    contracts = {
        "wan.text2video": WanText2VideoInputs,
        "wan.image2video": WanImage2VideoInputs,
    }
    return {
        name: bind_task(handlers, name, input_model, VideosWithBatchOutput)
        for name, input_model in contracts.items()
    }


def run_wan_text2video_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    batch_id = str(inputs.get("batch_id") or deps["make_batch_id"]())
    generation_params = {
        "prompt": str(inputs["prompt"]),
        "negative_prompt": str(inputs.get("negative_prompt") or ""),
        "steps": int(inputs.get("steps") or 30),
        "guidance_scale": float(inputs.get("guidance_scale") or 6.0),
        "width": int(inputs.get("width") or 832),
        "height": int(inputs.get("height") or 480),
        "seed": inputs.get("seed"),
        "model": str(inputs.get("model") or "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"),
        "num_frames": int(inputs.get("num_frames") or 49),
        "fps": int(inputs.get("fps") or 16),
        "num_videos": int(inputs.get("num_videos") or 1),
        "memory_preset": str(inputs.get("memory_preset") or "safe"),
        "quantization": str(inputs.get("quantization") or "none"),
        "reference_image": (
            deps["open_image_ref"](inputs["reference_image"])
            if inputs.get("reference_image") is not None
            else None
        ),
        "mask_image": (
            deps["open_image_ref"](inputs["mask_image"])
            if inputs.get("mask_image") is not None
            else None
        ),
        "conditioning_video": (
            deps["open_video_ref"](inputs["conditioning_video"])
            if inputs.get("conditioning_video") is not None
            else None
        ),
        "conditioning_scale": float(inputs.get("conditioning_scale") or 1.0),
        "batch_id": batch_id,
    }
    filenames = deps["generate_text2video"](generation_params)
    return {"batch_id": batch_id, "videos": [f"/outputs/{name}" for name in filenames]}


def run_wan_image2video_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    batch_id = str(inputs.get("batch_id") or deps["make_batch_id"]())
    generation_params = {
        "image": deps["open_image_ref"](inputs["image"]),
        "prompt": str(inputs["prompt"]),
        "negative_prompt": str(inputs.get("negative_prompt") or ""),
        "steps": int(inputs.get("steps") or 50),
        "guidance_scale": float(inputs.get("guidance_scale") or 5.0),
        "width": int(inputs.get("width") or 832),
        "height": int(inputs.get("height") or 480),
        "seed": inputs.get("seed"),
        "model": str(
            inputs.get("model")
            or r"D:\diffusion\diffusers\Wan2.1-I2V-14B-480P-Diffusers"
        ),
        "num_frames": int(inputs.get("num_frames") or 81),
        "fps": int(inputs.get("fps") or 16),
        "num_videos": int(inputs.get("num_videos") or 1),
        "memory_preset": str(inputs.get("memory_preset") or "offload"),
        "quantization": str(inputs.get("quantization") or "none"),
        "experimental_ack": bool(inputs.get("experimental_ack", True)),
        "batch_id": batch_id,
    }
    filenames = deps["generate_image2video"](generation_params)
    return {"batch_id": batch_id, "videos": [f"/outputs/{name}" for name in filenames]}
