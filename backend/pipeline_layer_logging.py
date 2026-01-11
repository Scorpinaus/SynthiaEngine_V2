from __future__ import annotations

import contextlib
import datetime as _dt
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import nn


logger = logging.getLogger(__name__)

_MAX_SUMMARY_DEPTH = 2
_MAX_SUMMARY_ITEMS = 4
_MAX_SUMMARY_CHARS = 500


@dataclass(frozen=True)
class LayerRow:
    component: str
    name: str
    type_name: str


def _sanitize_field(value: str) -> str:
    return value.replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")


def _truncate(value: str, max_chars: int = _MAX_SUMMARY_CHARS) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def _summarize_value(value: Any, *, depth: int = 0) -> str:
    if value is None:
        return "None"

    if isinstance(value, torch.Tensor):
        try:
            shape = list(value.shape)
        except Exception:
            shape = "NA"
        return f"Tensor(shape={shape}, dtype={value.dtype}, device={value.device})"

    if depth >= _MAX_SUMMARY_DEPTH:
        return f"{type(value).__name__}"

    if isinstance(value, (list, tuple)):
        items = ", ".join(
            _summarize_value(item, depth=depth + 1)
            for item in value[:_MAX_SUMMARY_ITEMS]
        )
        suffix = ", ..." if len(value) > _MAX_SUMMARY_ITEMS else ""
        return f"{type(value).__name__}(len={len(value)}, items=[{items}{suffix}])"

    if isinstance(value, dict):
        keys = list(value.keys())
        shown_keys = keys[:_MAX_SUMMARY_ITEMS]
        key_summaries = ", ".join(_summarize_value(k, depth=depth + 1) for k in shown_keys)
        suffix = ", ..." if len(keys) > _MAX_SUMMARY_ITEMS else ""
        return f"dict(len={len(keys)}, keys=[{key_summaries}{suffix}])"

    if isinstance(value, (str, int, float, bool)):
        return repr(value)

    return f"{type(value).__name__}"


def _is_leaf(module: nn.Module) -> bool:
    try:
        next(module.children())
    except StopIteration:
        return True
    return False


def iter_module_layers(
    component: str,
    module: nn.Module,
    *,
    leaf_only: bool = True,
) -> Iterable[tuple[str, nn.Module]]:
    for name, submodule in module.named_modules():
        if name:
            qualified_name = f"{component}.{name}"
        else:
            qualified_name = component
        if leaf_only and not _is_leaf(submodule):
            continue
        yield qualified_name, submodule


def collect_pipeline_layers(
    pipe: object,
    *,
    leaf_only: bool = True,
) -> list[LayerRow]:
    components = getattr(pipe, "components", None)
    if not isinstance(components, dict):
        return []

    rows: list[LayerRow] = []
    for component_name, component_obj in components.items():
        if isinstance(component_obj, nn.Module):
            for layer_name, layer in iter_module_layers(
                str(component_name),
                component_obj,
                leaf_only=leaf_only,
            ):
                rows.append(
                    LayerRow(
                        component=str(component_name),
                        name=layer_name,
                        type_name=layer.__class__.__name__,
                    )
                )
    return rows


def _default_runtime_components(pipe: object) -> dict[str, nn.Module]:
    components = getattr(pipe, "components", None)
    if not isinstance(components, dict):
        return {}

    preferred = (
        "text_encoder",
        "text_encoder_2",
        "unet",
        "vae",
        "controlnet",
        "transformer",
    )
    picked: dict[str, nn.Module] = {}
    for name in preferred:
        obj = components.get(name)
        if isinstance(obj, nn.Module):
            picked[name] = obj

    if picked:
        return picked

    return {k: v for k, v in components.items() if isinstance(v, nn.Module)}


@contextlib.contextmanager
def capture_runtime_used_layers(
    pipe: object,
    *,
    leaf_only: bool = True,
    components: Mapping[str, nn.Module] | None = None,
):
    component_modules = dict(components) if components is not None else _default_runtime_components(pipe)

    name_to_type: dict[str, str] = {}
    used: set[str] = set()
    name_to_input_summary: dict[str, str] = {}
    name_to_call_count: dict[str, int] = {}
    lock = threading.Lock()
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def record_layer(layer_name: str, args: tuple[Any, ...] | None = None, kwargs: dict[str, Any] | None = None):
        with lock:
            used.add(layer_name)
            name_to_call_count[layer_name] = name_to_call_count.get(layer_name, 0) + 1
            if layer_name not in name_to_input_summary and args is not None and kwargs is not None:
                args_summary = _summarize_value(list(args), depth=0)
                kwargs_summary = _summarize_value(dict(kwargs), depth=0)
                summary = f"args={args_summary} kwargs={kwargs_summary}"
                name_to_input_summary[layer_name] = _truncate(_sanitize_field(summary))

    for component, module in component_modules.items():
        for layer_name, submodule in iter_module_layers(component, module, leaf_only=leaf_only):
            name_to_type[layer_name] = submodule.__class__.__name__

            try:
                def pre_hook(_module, args, kwargs, layer_name=layer_name):
                    record_layer(layer_name, args=args, kwargs=kwargs)

                handles.append(submodule.register_forward_pre_hook(pre_hook, with_kwargs=True))
            except TypeError:
                handles.append(
                    submodule.register_forward_pre_hook(
                        lambda _m, _a, layer_name=layer_name: record_layer(layer_name),
                    )
                )

    try:
        yield used, name_to_type, name_to_input_summary, name_to_call_count
    finally:
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass


def append_layers_report(
    *,
    output_dir: Path,
    batch_id: str,
    label: str,
    pipeline_name: str,
    architecture_layers: Iterable[LayerRow] | None = None,
    runtime_used_layer_names: Iterable[str] | None = None,
    runtime_name_to_type: Mapping[str, str] | None = None,
    runtime_name_to_input_summary: Mapping[str, str] | None = None,
    runtime_name_to_call_count: Mapping[str, int] | None = None,
) -> Path:
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / f"{batch_id}_layers.txt"

    timestamp = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    arch = list(architecture_layers or [])
    runtime_used = sorted(set(runtime_used_layer_names or []))
    runtime_types = dict(runtime_name_to_type or {})
    runtime_inputs = dict(runtime_name_to_input_summary or {})
    runtime_calls = dict(runtime_name_to_call_count or {})

    with report_path.open("a", encoding="utf-8", newline="\n") as f:
        f.write("=== PIPELINE LAYER REPORT ===\n")
        f.write(f"timestamp_utc: {timestamp}\n")
        f.write(f"label: {label}\n")
        f.write(f"pipeline: {pipeline_name}\n")
        f.write(f"batch_id: {batch_id}\n")
        f.write("\n")

        if arch:
            f.write("[ARCHITECTURE]\n")
            f.write("component\tname\ttype\n")
            for row in arch:
                f.write(f"{row.component}\t{row.name}\t{row.type_name}\n")
            f.write(f"count: {len(arch)}\n")
            f.write("\n")

        if runtime_used:
            f.write("[RUNTIME_USED]\n")
            f.write("name\ttype\n")
            for layer_name in runtime_used:
                f.write(f"{layer_name}\t{runtime_types.get(layer_name, 'NA')}\n")
            f.write(f"count: {len(runtime_used)}\n")
            f.write("\n")

        if runtime_used:
            f.write("[RUNTIME_INPUTS]\n")
            f.write("name\tcalls\tinputs\n")
            for layer_name in runtime_used:
                calls = runtime_calls.get(layer_name, 0)
                inputs = runtime_inputs.get(layer_name, "NA")
                f.write(f"{layer_name}\t{calls}\t{inputs}\n")
            f.write(f"count: {len(runtime_used)}\n")
            f.write("\n")

    logger.info("Wrote pipeline layer report: %s", report_path)
    return report_path
