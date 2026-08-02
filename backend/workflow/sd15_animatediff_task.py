"""SD1.5 AnimateDiff workflow task adapter."""

from backend.workflow.sd15_shared import *

def run_sd15_animatediff_text2video(
    inputs: dict[str, Any],
    deps: dict[str, Any],
) -> dict[str, Any]:
    _normalized_lora_adapters = deps["normalized_lora_adapters"]
    make_batch_id = deps["make_batch_id"]
    generate_videos_text2video = deps["generate_videos_text2video"]

    lora_adapters = _normalized_lora_adapters(inputs)
    batch_id = str(inputs.get("batch_id") or make_batch_id())
    generation_params = {
        "prompt": str(inputs["prompt"]),
        "negative_prompt": str(inputs.get("negative_prompt") or ""),
        "steps": int(inputs.get("steps") or 25),
        "cfg": float(inputs.get("cfg") or 7.5),
        "width": int(inputs.get("width") or 512),
        "height": int(inputs.get("height") or 512),
        "seed": inputs.get("seed"),
        "scheduler": str(inputs.get("scheduler") or "ddim"),
        "model": inputs.get("model"),
        "motion_adapter": str(
            inputs.get("motion_adapter")
            or "guoyww/animatediff-motion-adapter-v1-5-2"
        ),
        "num_frames": int(inputs.get("num_frames") or 16),
        "fps": int(inputs.get("fps") or 8),
        "num_videos": int(inputs.get("num_videos") or 1),
        "free_noise_enabled": inputs.get("free_noise_enabled", False),
        "free_noise_context_length": int(inputs.get("free_noise_context_length") or 16),
        "free_noise_context_stride": int(inputs.get("free_noise_context_stride") or 4),
        "free_init_enabled": inputs.get("free_init_enabled", False),
        "free_init_num_iters": int(inputs.get("free_init_num_iters") or 3),
        "free_init_use_fast_sampling": inputs.get("free_init_use_fast_sampling", False),
        "free_init_method": str(inputs.get("free_init_method") or "butterworth"),
        "free_init_order": int(inputs.get("free_init_order") or 4),
        "free_init_spatial_stop_frequency": float(
            inputs.get("free_init_spatial_stop_frequency", 0.25)
        ),
        "free_init_temporal_stop_frequency": float(
            inputs.get("free_init_temporal_stop_frequency", 0.25)
        ),
        "clip_skip": int(inputs.get("clip_skip") or 1),
        "lora_adapters": lora_adapters,
        "weighting_policy": str(inputs.get("weighting_policy") or "diffusers-like"),
        "batch_id": batch_id,
    }
    filenames = generate_videos_text2video(generation_params)
    return {"batch_id": batch_id, "videos": [f"/outputs/{name}" for name in filenames]}

