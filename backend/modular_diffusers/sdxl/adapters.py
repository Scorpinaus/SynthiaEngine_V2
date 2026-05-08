"""Experimental adapter helpers for Diffusers' SDXL ModularPipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from types import MethodType
from typing import Any

from diffusers.loaders import StableDiffusionXLLoraLoaderMixin, TextualInversionLoaderMixin


@dataclass(frozen=True)
class LoraSpec:
    path: str
    weight: float = 1.0
    adapter_name: str | None = None


@dataclass(frozen=True)
class TextualInversionSpec:
    path: str
    token: str | None = None
    encoder: str = "text_encoder"


def add_sdxl_adapter_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lora",
        action="append",
        default=[],
        help="LoRA repo, directory, or weight file to load. Can be provided multiple times.",
    )
    parser.add_argument(
        "--lora-weight",
        action="append",
        type=float,
        default=[],
        help="LoRA weight for each --lora. Omit to use 1.0 for every adapter.",
    )
    parser.add_argument(
        "--lora-name",
        action="append",
        default=[],
        help="Optional adapter name for each --lora.",
    )
    parser.add_argument(
        "--textual-inversion",
        action="append",
        default=[],
        help="Textual inversion embedding for tokenizer/text_encoder. Can be provided multiple times.",
    )
    parser.add_argument(
        "--textual-inversion-token",
        action="append",
        default=[],
        help="Optional token override for each --textual-inversion.",
    )
    parser.add_argument(
        "--textual-inversion-2",
        action="append",
        default=[],
        help="Textual inversion embedding for tokenizer_2/text_encoder_2. Can be provided multiple times.",
    )
    parser.add_argument(
        "--textual-inversion-2-token",
        action="append",
        default=[],
        help="Optional token override for each --textual-inversion-2.",
    )


def _align_optional_values(values: list[Any], count: int, *, field_name: str) -> list[Any]:
    if not values:
        return [None] * count
    if len(values) != count:
        raise ValueError(f"{field_name} count must match the number of adapter paths.")
    return values


def parse_lora_specs(args: argparse.Namespace) -> list[LoraSpec]:
    paths = list(getattr(args, "lora", []) or [])
    weights = list(getattr(args, "lora_weight", []) or [])
    names = list(getattr(args, "lora_name", []) or [])

    if weights and len(weights) != len(paths):
        raise ValueError("--lora-weight count must match --lora count.")
    if names and len(names) != len(paths):
        raise ValueError("--lora-name count must match --lora count.")

    if not weights:
        weights = [1.0] * len(paths)
    if not names:
        names = [None] * len(paths)

    return [
        LoraSpec(path=path, weight=float(weight), adapter_name=name)
        for path, weight, name in zip(paths, weights, names, strict=True)
    ]


def parse_textual_inversion_specs(args: argparse.Namespace) -> list[TextualInversionSpec]:
    primary_paths = list(getattr(args, "textual_inversion", []) or [])
    secondary_paths = list(getattr(args, "textual_inversion_2", []) or [])
    primary_tokens = _align_optional_values(
        list(getattr(args, "textual_inversion_token", []) or []),
        len(primary_paths),
        field_name="--textual-inversion-token",
    )
    secondary_tokens = _align_optional_values(
        list(getattr(args, "textual_inversion_2_token", []) or []),
        len(secondary_paths),
        field_name="--textual-inversion-2-token",
    )

    specs = [
        TextualInversionSpec(path=path, token=token, encoder="text_encoder")
        for path, token in zip(primary_paths, primary_tokens, strict=True)
    ]
    specs.extend(
        TextualInversionSpec(path=path, token=token, encoder="text_encoder_2")
        for path, token in zip(secondary_paths, secondary_tokens, strict=True)
    )
    return specs


def _bind_mixin_methods(pipe: object, mixin: type) -> None:
    for cls in reversed(mixin.mro()):
        if cls is object:
            continue
        for name, value in cls.__dict__.items():
            if name.startswith("__"):
                continue
            if isinstance(value, classmethod):
                setattr(pipe, name, value.__get__(mixin, mixin))
            elif isinstance(value, staticmethod):
                setattr(pipe, name, value.__get__(pipe, type(pipe)))
            elif callable(value):
                setattr(pipe, name, MethodType(value, pipe))


def enable_sdxl_modular_adapter_support(pipe: object) -> object:
    _bind_mixin_methods(pipe, StableDiffusionXLLoraLoaderMixin)
    _bind_mixin_methods(pipe, TextualInversionLoaderMixin)
    pipe._lora_loadable_modules = ["unet", "text_encoder", "text_encoder_2"]
    pipe.unet_name = "unet"
    pipe.text_encoder_name = "text_encoder"
    pipe.tokenizer_name = "tokenizer"
    pipe.lora_scale = 1.0
    if not hasattr(pipe, "hf_device_map"):
        pipe.hf_device_map = None
    return pipe


def load_sdxl_modular_loras(pipe: object, specs: list[LoraSpec]) -> list[str]:
    if not specs:
        return []

    enable_sdxl_modular_adapter_support(pipe)
    adapter_names: list[str] = []
    adapter_weights: list[float] = []
    for index, spec in enumerate(specs):
        adapter_name = spec.adapter_name or f"lora_{index + 1}"
        pipe.load_lora_weights(spec.path, adapter_name=adapter_name)
        adapter_names.append(adapter_name)
        adapter_weights.append(float(spec.weight))

    pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
    return adapter_names


def load_sdxl_modular_textual_inversions(
    pipe: object,
    specs: list[TextualInversionSpec],
) -> None:
    if not specs:
        return

    enable_sdxl_modular_adapter_support(pipe)
    for spec in specs:
        if spec.encoder == "text_encoder_2":
            pipe.load_textual_inversion(
                spec.path,
                token=spec.token,
                tokenizer=pipe.tokenizer_2,
                text_encoder=pipe.text_encoder_2,
            )
        else:
            pipe.load_textual_inversion(
                spec.path,
                token=spec.token,
                tokenizer=pipe.tokenizer,
                text_encoder=pipe.text_encoder,
            )


def apply_sdxl_modular_adapters_from_args(pipe: object, args: argparse.Namespace) -> None:
    load_sdxl_modular_loras(pipe, parse_lora_specs(args))
    load_sdxl_modular_textual_inversions(pipe, parse_textual_inversion_specs(args))
