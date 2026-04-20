from __future__ import annotations

from typing import Any

from PIL import Image


def run_z_image_text2img_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    generate_text2img = deps["generate_text2img"]

    result = generate_text2img(dict(inputs))
    if not isinstance(result, dict):
        raise ValueError("z-image.text2img must return an object")
    return result


def run_z_image_img2img_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    _open_image_ref = deps["open_image_ref"]
    _remap_img2img_strength = deps["remap_img2img_strength"]
    generate_img2img = deps["generate_img2img"]

    initial_image = _open_image_ref(inputs["initial_image"]).convert("RGB")
    width = int(inputs.get("width") or 1024)
    height = int(inputs.get("height") or 1024)
    initial_image = initial_image.resize((width, height))

    strength = float(inputs.get("strength") or 0.75)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    strength = _remap_img2img_strength(strength)

    result = generate_img2img(
        {
            "initial_image": initial_image,
            "strength": strength,
            "prompt": str(inputs["prompt"]),
            "negative_prompt": str(inputs.get("negative_prompt") or ""),
            "steps": int(inputs.get("steps") or 8),
            "guidance_scale": float(inputs.get("guidance_scale") or 0.0),
            "width": width,
            "height": height,
            "seed": inputs.get("seed"),
            "scheduler": str(inputs.get("scheduler") or "euler"),
            "model": inputs.get("model"),
            "num_images": int(inputs.get("num_images") or 1),
            "lora_adapters": inputs.get("lora_adapters"),
        }
    )
    if not isinstance(result, dict):
        raise ValueError("z-image.img2img must return an object")
    return result


def run_z_image_inpaint_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    _open_image_ref = deps["open_image_ref"]
    generate_inpaint = deps["generate_inpaint"]

    initial_image = _open_image_ref(inputs["initial_image"]).convert("RGB")
    mask_image = _open_image_ref(inputs["mask_image"]).convert("L")
    if mask_image.size != initial_image.size:
        mask_image = mask_image.resize(initial_image.size, resample=Image.NEAREST)

    strength = float(inputs.get("strength") or 0.5)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")

    result = generate_inpaint(
        {
            "initial_image": initial_image,
            "mask_image": mask_image,
            "strength": strength,
            "prompt": str(inputs["prompt"]),
            "negative_prompt": str(inputs.get("negative_prompt") or ""),
            "steps": int(inputs.get("steps") or 8),
            "guidance_scale": float(inputs.get("guidance_scale") or 0.0),
            "seed": inputs.get("seed"),
            "scheduler": str(inputs.get("scheduler") or "euler"),
            "model": inputs.get("model"),
            "num_images": int(inputs.get("num_images") or 1),
            "lora_adapters": inputs.get("lora_adapters"),
        }
    )
    if not isinstance(result, dict):
        raise ValueError("z-image.inpaint must return an object")
    return result
