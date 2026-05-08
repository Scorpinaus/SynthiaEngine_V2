"""ControlNet helpers for Diffusers' SDXL ModularPipeline smoke scripts."""

from __future__ import annotations

import argparse

from diffusers import ControlNetModel


DEFAULT_CONTROLNET_MODEL = "diffusers/controlnet-canny-sdxl-1.0"


def add_sdxl_controlnet_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--control-image", required=True, help="Path to the ControlNet conditioning image.")
    parser.add_argument(
        "--controlnet-model",
        default=DEFAULT_CONTROLNET_MODEL,
        help="ControlNet model repo or local path.",
    )
    parser.add_argument(
        "--controlnet-conditioning-scale",
        type=float,
        default=1.0,
        help="ControlNet conditioning scale.",
    )
    parser.add_argument(
        "--control-guidance-start",
        type=float,
        default=0.0,
        help="Fraction of denoising where ControlNet starts.",
    )
    parser.add_argument(
        "--control-guidance-end",
        type=float,
        default=1.0,
        help="Fraction of denoising where ControlNet ends.",
    )
    parser.add_argument(
        "--guess-mode",
        action="store_true",
        help="Enable ControlNet guess mode.",
    )


def load_sdxl_controlnet_component(pipe: object, *, model: str, torch_dtype: object) -> object:
    controlnet = ControlNetModel.from_pretrained(model, torch_dtype=torch_dtype)
    pipe.update_components(controlnet=controlnet)
    return controlnet
