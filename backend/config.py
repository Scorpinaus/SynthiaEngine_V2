from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DATABASE_DIR = Path("database")
DATABASE_DIR.mkdir(exist_ok=True)

DEFAULTS = {
    "steps": 20,
    "cfg": 7.5,
    "width": 512,
    "height": 512,
    "negative_prompt": "low quality, blurry, extra fingers",
    "controlnet_model": "lllyasviel/sd-controlnet-canny",
}

RESOURCE_LOGGING_ENABLED = True
RESOURCE_LOGGING_INTERVAL_S = 0.5

# Writes `outputs/batch_{batch_id}/{batch_id}_layers.txt` during pipeline runs.
PIPELINE_LAYER_LOGGING_ENABLED = True
PIPELINE_LAYER_LOGGING_LEAF_ONLY = True

# Includes a per-layer summary of the first observed call inputs (args/kwargs).
PIPELINE_LAYER_LOGGING_CAPTURE_INPUTS = True
