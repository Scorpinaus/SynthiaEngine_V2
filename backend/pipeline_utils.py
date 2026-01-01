from __future__ import annotations

import torch

def build_fixed_step_timesteps(scheduler, steps: int, strength: float, device: torch.device | str | None = None, ) -> torch.Tensor:
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