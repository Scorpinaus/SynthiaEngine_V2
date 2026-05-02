"""Shared Diffusers quantization helpers."""

from __future__ import annotations

from typing import Sequence


SUPPORTED_DIFFUSERS_QUANTIZATION = {"none", "bnb_8bit"}


def build_diffusers_pipeline_quantization_config(
    quantization: str,
    *,
    components_to_quantize: Sequence[str],
    task_type: str,
):
    """Build a Diffusers pipeline quantization config for supported modes."""
    if quantization == "none":
        return None
    if quantization != "bnb_8bit":
        raise ValueError(f"quantization must be 'none' or 'bnb_8bit' for {task_type}")

    from diffusers.quantizers import PipelineQuantizationConfig

    return PipelineQuantizationConfig(
        quant_backend="bitsandbytes_8bit",
        quant_kwargs={"load_in_8bit": True},
        components_to_quantize=list(components_to_quantize),
    )
