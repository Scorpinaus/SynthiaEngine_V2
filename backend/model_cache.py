import gc
import logging
import threading
import torch

from backend import flux_pipeline
from backend import sd15_pipeline
from backend import sdxl_pipeline
from backend import z_image_pipeline
from backend.model_registry import get_model_entry

logger = logging.getLogger(__name__)

_FAMILY_LOCK = threading.Lock()
_ACTIVE_FAMILY: str | None = None
_ACTIVE_MODEL: str | None = None


def _collect_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _clear_cache(cache: dict[str, object]) -> None:
    cached = list(cache.values())
    cache.clear()
    for pipe in cached:
        try:
            if hasattr(pipe, "to"):
                pipe_dtype = getattr(pipe, "dtype", None)
                if pipe_dtype != torch.float16:
                    pipe.to("cpu")
            del pipe
        except Exception:
            logger.debug("Failed to delete cached pipeline reference.", exc_info=True)


def clear_sd15_pipelines(collect_memory: bool = True) -> None:
    _clear_cache(sd15_pipeline.PIPELINE_CACHE)
    _clear_cache(sd15_pipeline.IMG2IMG_PIPELINE_CACHE)
    _clear_cache(sd15_pipeline.INPAINT_PIPELINE_CACHE)
    if collect_memory:
        _collect_memory()


def clear_sdxl_pipelines(collect_memory: bool = True) -> None:
    _clear_cache(sdxl_pipeline.PIPELINE_CACHE)
    _clear_cache(sdxl_pipeline.IMG2IMG_PIPELINE_CACHE)
    _clear_cache(sdxl_pipeline.INPAINT_PIPELINE_CACHE)
    if collect_memory:
        _collect_memory()


def clear_z_image_pipelines(collect_memory: bool = True) -> None:
    _clear_cache(z_image_pipeline.PIPELINE_CACHE)
    _clear_cache(z_image_pipeline.IMG2IMG_PIPELINE_CACHE)
    if collect_memory:
        _collect_memory()


def clear_flux_pipelines(collect_memory: bool = True) -> None:
    _clear_cache(flux_pipeline.PIPELINE_CACHE)
    _clear_cache(flux_pipeline.IMG2IMG_PIPELINE_CACHE)
    _clear_cache(flux_pipeline.INPAINT_PIPELINE_CACHE)
    if collect_memory:
        _collect_memory()


def clear_all_pipelines() -> None:
    clear_sd15_pipelines(collect_memory=False)
    clear_sdxl_pipelines(collect_memory=False)
    clear_z_image_pipelines(collect_memory=False)
    clear_flux_pipelines(collect_memory=False)
    _collect_memory()


def _normalize_family(family: str | None) -> str | None:
    if not family:
        return None
    lowered = family.strip().lower()
    if not lowered:
        return None
    if "sdxl" in lowered:
        return "sdxl"
    if "z-image" in lowered or "z image" in lowered or "zimage" in lowered or "turbo" in lowered:
        return "z-image-turbo"
    if "flux" in lowered:
        return "flux"
    if "sd1.5" in lowered or "sd15" in lowered or "sd 1.5" in lowered or "sd_1.5" in lowered:
        return "sd15"
    return lowered


def prepare_model_family(target_family: str | None) -> None:
    normalized = _normalize_family(target_family)
    if normalized is None:
        return

    with _FAMILY_LOCK:
        global _ACTIVE_FAMILY, _ACTIVE_MODEL
        if _ACTIVE_FAMILY == normalized:
            return

        if normalized == "sd15":
            clear_sdxl_pipelines()
            clear_z_image_pipelines()
            clear_flux_pipelines()
        elif normalized == "sdxl":
            clear_sd15_pipelines()
            clear_z_image_pipelines()
            clear_flux_pipelines()
        elif normalized == "z-image-turbo":
            clear_sd15_pipelines()
            clear_sdxl_pipelines()
            clear_flux_pipelines()
        elif normalized == "flux":
            clear_sd15_pipelines()
            clear_sdxl_pipelines()
            clear_z_image_pipelines()
        else:
            clear_all_pipelines()

        _ACTIVE_FAMILY = normalized
        _ACTIVE_MODEL = None


def prepare_model(model_name: str | None) -> None:
    entry = get_model_entry(model_name)
    normalized = _normalize_family(entry.family)
    if normalized is None:
        return

    with _FAMILY_LOCK:
        global _ACTIVE_FAMILY, _ACTIVE_MODEL
        if _ACTIVE_FAMILY != normalized:
            if normalized == "sd15":
                clear_sdxl_pipelines()
                clear_z_image_pipelines()
                clear_flux_pipelines()
            elif normalized == "sdxl":
                clear_sd15_pipelines()
                clear_z_image_pipelines()
                clear_flux_pipelines()
            elif normalized == "z-image-turbo":
                clear_sd15_pipelines()
                clear_sdxl_pipelines()
                clear_flux_pipelines()
            elif normalized == "flux":
                clear_sd15_pipelines()
                clear_sdxl_pipelines()
                clear_z_image_pipelines()
            else:
                clear_all_pipelines()

            _ACTIVE_FAMILY = normalized
            _ACTIVE_MODEL = entry.name
            return

        if _ACTIVE_MODEL == entry.name:
            return

        if normalized == "sd15":
            clear_sd15_pipelines()
        elif normalized == "sdxl":
            clear_sdxl_pipelines()
        elif normalized == "z-image-turbo":
            clear_z_image_pipelines()
        elif normalized == "flux":
            clear_flux_pipelines()
        else:
            clear_all_pipelines()

        _ACTIVE_MODEL = entry.name
