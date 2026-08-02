"""Compatibility constants backed by the centralized settings boundary."""

from backend.settings import load_settings


_SETTINGS = load_settings()
OUTPUT_DIR = _SETTINGS.paths.output_dir
DATABASE_DIR = _SETTINGS.paths.database_dir

DEFAULTS = {
    "steps": 20,
    "cfg": 7.5,
    "width": 512,
    "height": 512,
    "negative_prompt": "low quality, blurry, extra fingers",
    "controlnet_model": "lllyasviel/control_v11p_sd15_canny",
}

RESOURCE_LOGGING_ENABLED = True
RESOURCE_LOGGING_INTERVAL_S = 0.5
SUMMARY_PROFILER_INTERVAL_S = 1.0

# Writes `outputs/batch_{batch_id}/{batch_id}_layers.txt` during pipeline runs.
PIPELINE_LAYER_LOGGING_ENABLED = True
PIPELINE_LAYER_LOGGING_LEAF_ONLY = True

# Includes a per-layer summary of the first observed call inputs (args/kwargs).
PIPELINE_LAYER_LOGGING_CAPTURE_INPUTS = True
