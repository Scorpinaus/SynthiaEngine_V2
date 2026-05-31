from __future__ import annotations

from typing import Any

import torch


def enable_flux_vae_memory_savers(pipe: Any) -> None:
    """Enable VAE memory savers without depending on deprecated pipeline shims."""
    vae = getattr(pipe, "vae", None)
    if vae is None:
        return

    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()


def enable_low_memory_flux(
    pipe: Any,
    *,
    mode: str = "auto",
    use_stream: bool = True,
    exclude_vae_from_group_offload: bool = True,
) -> str:
    """Configure Flux for low-memory inference while avoiding quantization.

    Returns the offload mode that was applied. ``auto`` prefers group offload
    because it is a useful middle ground between model offload and sequential
    CPU offload on single-GPU systems with limited VRAM.
    """
    normalized_mode = (mode or "auto").strip().lower()
    if normalized_mode not in {"auto", "group", "model", "sequential", "cuda", "none"}:
        raise ValueError(
            "Flux low-memory mode must be one of: auto, group, model, sequential, cuda, none."
        )

    enable_flux_vae_memory_savers(pipe)

    if normalized_mode == "none":
        return "none"

    if normalized_mode == "cuda":
        pipe.to("cuda")
        return "cuda"

    if normalized_mode in {"auto", "group"} and torch.cuda.is_available() and hasattr(pipe, "enable_group_offload"):
        try:
            exclude_modules = ["vae"] if exclude_vae_from_group_offload else None
            pipe.enable_group_offload(
                onload_device=torch.device("cuda"),
                offload_device=torch.device("cpu"),
                offload_type="leaf_level",
                use_stream=use_stream,
                low_cpu_mem_usage=True,
                exclude_modules=exclude_modules,
            )
            return "group"
        except Exception:
            if normalized_mode == "group":
                raise

    if normalized_mode in {"auto", "model"} and torch.cuda.is_available() and hasattr(pipe, "enable_model_cpu_offload"):
        try:
            pipe.enable_model_cpu_offload()
            return "model"
        except Exception:
            if normalized_mode == "model":
                raise

    pipe.enable_sequential_cpu_offload()
    return "sequential"
