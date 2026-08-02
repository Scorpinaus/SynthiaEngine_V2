"""Task-scoped LoRA cleanup policy for SDXL runtimes."""

from backend.sdxl.runtime_common import *

def _cleanup_lora_adapters(pipe, adapter_names: list[str]) -> None:
    if not adapter_names or not hasattr(pipe, "unload_lora_weights"):
        return
    try:
        pipe.unload_lora_weights()
    except Exception:
        logger.exception("Failed to unload LoRA weights cleanly.")

