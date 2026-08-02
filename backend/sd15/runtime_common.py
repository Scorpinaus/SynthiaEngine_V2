"""
Stable Diffusion 1.5 (SD1.5) pipeline helpers.

This module is responsible for:
- Loading Diffusers pipelines for txt2img, img2img, inpaint, and ControlNet.
- Running inference (CUDA / fp16) and writing PNG outputs + embedded metadata.
- Optional LoRA adapter application and pipeline-layer logging/diagnostics.

The functions here are used by workflow tasks (e.g. `sd15.text2img`), so they
aim to be deterministic (seeded) and side-effectful only in well-defined ways
(writing files under `OUTPUT_DIR`).
"""

import torch
import logging
import math
import threading
import numpy as np
from contextlib import contextmanager
from pathlib import Path
from typing import cast
from PIL import ImageFilter, Image

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
)

from backend.config import OUTPUT_DIR
from backend.utilities.logging import configure_logging
from backend.registries.model import get_model_entry
from backend.utilities.resource_logging import resource_logger
from backend.adapters.ip_adapter import IpAdapterManager
from backend.adapters.ip_adapter_embeds import (
    load_ip_adapter_embeds_artifact,
    validate_ip_adapter_embeds_metadata,
)
# from testing.pipeline_stable_diffusion import(StableDiffusionPipeline)
from backend.utilities.pipeline import (
    build_fixed_step_timesteps,
    build_png_metadata,
    build_batch_output_relpath,
    get_batch_output_dir,
    make_batch_id,
    release_pipeline,
    resolve_model_source,
)
from backend.utilities.schedulers import create_scheduler
from backend.utilities.prompt import build_prompt_embeddings
from backend import config
from backend.utilities.pipeline_layer_logging import (
    append_layers_report,
    capture_runtime_used_layers,
    collect_pipeline_layers,
)
from backend.lora.utils import apply_lora_adapters_with_validation, write_lora_coverage_report
from backend.utilities.subprocess_transport import (
    SubprocessTransport,
    normalize_path_list,
    run_subprocess,
)

logger = logging.getLogger(__name__)
configure_logging()

_LCM_LORA_MODEL_ID = "latent-consistency/lcm-lora-sdv1-5"
_LCM_LORA_ADAPTER_NAME = "lcm_lora_sd15"
_LCM_DEFAULT_STEPS = 4
_LCM_DEFAULT_CFG = 0.0
_DEFAULT_IP_ADAPTER_MODEL = "h94/IP-Adapter"
_DEFAULT_IP_ADAPTER_SUBFOLDER = "models"
_DEFAULT_IP_ADAPTER_WEIGHT_NAME = "ip-adapter_sd15.bin"
_DEFAULT_IP_ADAPTER_SCALE = 0.6
_SD15_SUBPROCESS_SEMAPHORE = threading.Semaphore(1)

__all__ = [name for name in globals() if not name.startswith("__")]

