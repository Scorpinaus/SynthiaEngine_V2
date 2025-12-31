import gc
import logging
import threading

import torch

logger = logging.getLogger(__name__)

_FAMILY_LOCK = threading.Lock()
_ACTIVE_FAMILY: str | None = None


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
                pipe.to("cpu")
            del pipe
        except Exception:
            logger.debug("Failed to delete cached pipeline reference.", exc_info=True)


def clear_sd15_pipelines(collect_memory: bool = True) -> None:
    from backend import sd15_pipeline

    _clear_cache(sd15_pipeline.PIPELINE_CACHE)
    _clear_cache(sd15_pipeline.IMG2IMG_PIPELINE_CACHE)
    _clear_cache(sd15_pipeline.INPAINT_PIPELINE_CACHE)
    if collect_memory:
        _collect_memory()


def clear_sdxl_pipelines(collect_memory: bool = True) -> None:
    from backend import sdxl_pipeline

    _clear_cache(sdxl_pipeline.PIPELINE_CACHE)
    _clear_cache(sdxl_pipeline.IMG2IMG_PIPELINE_CACHE)
    _clear_cache(sdxl_pipeline.INPAINT_PIPELINE_CACHE)
    if collect_memory:
        _collect_memory()


def clear_z_image_pipelines(collect_memory: bool = True) -> None:
    from backend import z_image_pipeline

    _clear_cache(z_image_pipeline.PIPELINE_CACHE)
    _clear_cache(z_image_pipeline.IMG2IMG_PIPELINE_CACHE)
    if collect_memory:
        _collect_memory()


def clear_all_pipelines() -> None:
    clear_sd15_pipelines(collect_memory=False)
    clear_sdxl_pipelines(collect_memory=False)
    clear_z_image_pipelines(collect_memory=False)
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
    if "sd1.5" in lowered or "sd15" in lowered or "sd 1.5" in lowered or "sd_1.5" in lowered:
        return "sd15"
    return lowered


def prepare_model_family(target_family: str | None) -> None:
    normalized = _normalize_family(target_family)
    if normalized is None:
        return

    with _FAMILY_LOCK:
        global _ACTIVE_FAMILY
        if _ACTIVE_FAMILY == normalized:
            return

        if normalized == "sd15":
            clear_sdxl_pipelines()
            clear_z_image_pipelines()
        elif normalized == "sdxl":
            clear_sd15_pipelines()
            clear_z_image_pipelines()
        elif normalized == "z-image-turbo":
            clear_sd15_pipelines()
            clear_sdxl_pipelines()
        else:
            clear_all_pipelines()

        _ACTIVE_FAMILY = normalized
