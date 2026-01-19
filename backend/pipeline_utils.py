from __future__ import annotations

import gc
import random
import time
from pathlib import Path

import torch
from PIL.PngImagePlugin import PngInfo

from backend.model_registry import ModelRegistryEntry


def resolve_model_source(entry: ModelRegistryEntry) -> str:
    if entry.location_type == "hub":
        return entry.link

    return str(Path(entry.link).expanduser())


def make_batch_id() -> str:
    return f"b{int(time.time())}_{random.randint(1000, 9999)}"


def build_png_metadata(metadata: dict[str, object]) -> PngInfo:
    info = PngInfo()
    for key, value in metadata.items():
        if value is None:
            continue
        info.add_text(key, str(value))
    return info


def get_batch_output_dir(output_dir: Path, batch_id: str) -> Path:
    batch_dir = output_dir / f"batch_{batch_id}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir


def build_batch_output_relpath(batch_id: str, filename: str) -> str:
    return (Path(f"batch_{batch_id}") / filename).as_posix()


def build_fixed_step_timesteps(
    scheduler,
    steps: int,
    strength: float,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    clamped_strength = max(0.0, min(1.0, strength))
    
    if device is not None:
        scheduler.set_timesteps(steps, device = device)
    else:
        scheduler.set_timesteps(steps)
        
    initial_timesteps = scheduler.timesteps
    if steps <= 1:
        return initial_timesteps
    
    max_index = len(initial_timesteps) - 1
    start_index = int(round((1.0 - clamped_strength) * max_index))
    start_index = max(0, min(max_index, start_index))
    
    index_positions = torch.linspace(
        start_index, max_index, steps, device = initial_timesteps.device,
    ).round().long()
    return initial_timesteps[index_positions]


def cleanup_memory(*, clear_cuda: bool = True) -> None:
    gc.collect()
    if clear_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
