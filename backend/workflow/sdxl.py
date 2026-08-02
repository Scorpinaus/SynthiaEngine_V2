"""Compatibility facade for SDXL workflow task adapters."""

from backend.workflow.sdxl_controlnet_task import run_sdxl_controlnet_text2img_task
from backend.workflow.sdxl_img2img_task import run_sdxl_img2img_task
from backend.workflow.sdxl_inpaint_task import run_sdxl_inpaint_task
from backend.workflow.sdxl_ip_adapter_task import run_sdxl_ip_adapter_encode_task
from backend.workflow.sdxl_shared import task_definitions
from backend.workflow.sdxl_text2img_task import run_sdxl_text2img_task

__all__ = [
    "run_sdxl_controlnet_text2img_task",
    "run_sdxl_img2img_task",
    "run_sdxl_inpaint_task",
    "run_sdxl_ip_adapter_encode_task",
    "run_sdxl_text2img_task",
    "task_definitions",
]
