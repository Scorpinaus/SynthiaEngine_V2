import logging
import threading
from contextlib import contextmanager
from pathlib import Path

import torch
from PIL import Image
from diffusers import (
    ControlNetModel,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLInpaintPipeline,
)

from backend.config import OUTPUT_DIR
from backend.adapters.ip_adapter import IpAdapterManager
from backend.adapters.ip_adapter_embeds import (
    load_ip_adapter_embeds_artifact,
    validate_ip_adapter_embeds_metadata,
)
from backend.utilities.logging import configure_logging
from backend.lora.utils import apply_lora_adapters_with_validation, write_lora_coverage_report
from backend.registries.model import get_model_entry
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
from backend.utilities.subprocess_transport import (
    SubprocessTransport,
    normalize_image_result,
    run_subprocess,
)

logger = logging.getLogger(__name__)
configure_logging()

_DEFAULT_IP_ADAPTER_MODEL = "h94/IP-Adapter"
_DEFAULT_IP_ADAPTER_SUBFOLDER = "sdxl_models"
_DEFAULT_IP_ADAPTER_WEIGHT_NAME = "ip-adapter_sdxl.bin"
_DEFAULT_IP_ADAPTER_SCALE = 0.6
_SDXL_SUBPROCESS_SEMAPHORE = threading.Semaphore(1)

__all__ = [name for name in globals() if not name.startswith("__")]

