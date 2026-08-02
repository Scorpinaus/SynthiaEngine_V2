"""Compatibility facade for SD1.5 workflow task adapters.

Each operation lives in a focused module so task validation and parameter
mapping can be changed and tested without navigating the entire model family.
Public task identifiers and handler names remain unchanged.
"""

from backend.workflow.sd15_animatediff_task import run_sd15_animatediff_text2video
from backend.workflow.sd15_controlnet_task import run_sd15_controlnet_text2img
from backend.workflow.sd15_hires_fix_task import run_sd15_hires_fix
from backend.workflow.sd15_img2img_task import run_sd15_img2img
from backend.workflow.sd15_inpaint_task import run_sd15_inpaint
from backend.workflow.sd15_ip_adapter_task import run_sd15_ip_adapter_encode_task
from backend.workflow.sd15_shared import task_definitions
from backend.workflow.sd15_text2img_task import run_sd15_text2img

__all__ = [
    "run_sd15_animatediff_text2video",
    "run_sd15_controlnet_text2img",
    "run_sd15_hires_fix",
    "run_sd15_img2img",
    "run_sd15_inpaint",
    "run_sd15_ip_adapter_encode_task",
    "run_sd15_text2img",
    "task_definitions",
]
