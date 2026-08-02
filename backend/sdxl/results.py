"""Result metadata and image saving helpers for SDXL."""

from backend.sdxl.runtime_common import *

def save_image(
    *,
    image: Image.Image,
    batch_output_dir: Path,
    batch_id: str,
    seed: int,
    metadata: dict[str, object],
) -> str:
    filename = batch_output_dir / f"{batch_id}_{seed}.png"
    pnginfo = build_png_metadata(metadata)
    image.save(filename, pnginfo=pnginfo)
    return build_batch_output_relpath(batch_id, filename.name)

def _metadata_without_runtime_images(params: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in params.items()
        if key not in {"ip_adapter_image", "ip_adapter_image_embeds_ref"}
    }

