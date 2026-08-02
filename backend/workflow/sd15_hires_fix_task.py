"""SD1.5 Hi-Res Fix workflow task adapter."""

from backend.workflow.sd15_shared import *

def run_sd15_hires_fix(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    _open_image_ref = deps["open_image_ref"]
    make_batch_id = deps["make_batch_id"]
    get_batch_output_dir = deps["get_batch_output_dir"]
    OUTPUT_DIR = deps["output_dir"]
    _normalized_lora_adapters = deps["normalized_lora_adapters"]
    run_sd15_hires_fix = deps["run_sd15_hires_fix"]

    images_in = inputs["images"]
    if not isinstance(images_in, list):
        raise ValueError("images must be a list")
    images = [_open_image_ref(item).convert("RGB") for item in images_in]

    batch_id = str(inputs.get("batch_id") or make_batch_id())
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    lora_adapters = _normalized_lora_adapters(inputs)
    relpaths = run_sd15_hires_fix(
        images=images,
        prompt=str(inputs["prompt"]),
        negative_prompt=str(inputs.get("negative_prompt") or ""),
        steps=int(inputs.get("steps") or 20),
        cfg=float(inputs.get("cfg") or 7.5),
        seed=inputs.get("seed"),
        scheduler=str(inputs.get("scheduler") or "euler"),
        model=inputs.get("model"),
        clip_skip=int(inputs.get("clip_skip") or 1),
        hires_scale=float(inputs.get("hires_scale") or 1.0),
        hires_strength=float(inputs.get("hires_strength") or 0.35),
        lora_adapters=lora_adapters,
        weighting_policy=str(inputs.get("weighting_policy") or "diffusers-like"),
        output_dir=batch_output_dir,
        batch_id=batch_id,
    )
    return {"batch_id": batch_id, "images": [f"/outputs/{p}" for p in relpaths]}

