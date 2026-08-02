"""SD1.5 Hi-Res Fix generation operation."""

from backend.sd15.runtime_common import *
from backend.sd15.adapters import _apply_lora_adapters, _cleanup_lora_adapters
from backend.sd15.loaders import load_img2img_pipeline
from backend.sd15.preparation import _upscale_image

@torch.inference_mode()
def run_sd15_hires_fix(
    *,
    images: list[Image.Image],
    prompt: str,
    negative_prompt: str,
    steps: int,
    cfg: float,
    seed: int | None,
    scheduler: str,
    model: str | None,
    clip_skip: int,
    hires_scale: float,
    hires_strength: float = 0.35,
    lora_adapters: list[object] | None = None,
    weighting_policy: str = "diffusers-like",
    output_dir: Path | None = None,
    batch_id: str | None = None,
) -> list[str]:
    """
    Apply SD1.5 hires-fix to each input image and write PNGs to disk.

    Args:
        images: Source images to upscale/refine.
        prompt: Positive prompt text.
        negative_prompt: Negative prompt text.
        steps: Number of denoising steps.
        cfg: Classifier-free guidance scale.
        seed: Optional base seed. ``None`` or ``0`` selects a random base seed.
        scheduler: Scheduler name.
        model: Optional model registry key.
        clip_skip: CLIP skip value.
        hires_scale: Upscale factor. Must be ``> 1.0``.
        hires_strength: Img2img strength for refinement.
        lora_adapters: Optional LoRA adapter specs.
        weighting_policy: Prompt-weighting policy for embedding construction.
        output_dir: Optional output root. Defaults to batch folder under ``OUTPUT_DIR``.
        batch_id: Optional batch identifier.

    Returns:
        List of output PNG paths relative to ``OUTPUT_DIR``.

    Raises:
        ValueError: If ``hires_scale <= 1.0``.
    """
    if hires_scale <= 1.0:
        raise ValueError("hires_scale must be > 1.0 for sd15.hires_fix")
    if not images:
        return []

    if seed is None or seed == 0:
        base_seed = torch.randint(0, 2**31, (1,)).item()
    else:
        base_seed = int(seed)

    if batch_id is None:
        batch_id = make_batch_id()
    batch_output_dir = output_dir or get_batch_output_dir(OUTPUT_DIR, batch_id)

    adapter_names: list[str] = []
    pipe = load_img2img_pipeline(model)
    try:
        pipe.scheduler = create_scheduler(scheduler, pipe)
        adapter_names = _apply_lora_adapters(pipe, lora_adapters, validate=False)
        prompt_embeds, negative_prompt_embeds, use_prompt_embeds = build_prompt_embeddings(
            pipe,
            prompt,
            negative_prompt,
            clip_skip=clip_skip,
            weighting_policy=weighting_policy,
        )

        relpaths: list[str] = []
        for idx, image in enumerate(images):
            # Offset the seed per image to make batch outputs deterministic and distinct.
            current_seed = base_seed + idx
            generator = torch.Generator(device="cuda").manual_seed(current_seed)

            upscaled = _upscale_image(image, hires_scale)
            out_image = pipe(
                prompt=None if use_prompt_embeds else prompt,
                negative_prompt=None if use_prompt_embeds else negative_prompt,
                image=upscaled,
                strength=hires_strength,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
                clip_skip=clip_skip,
                prompt_embeds=prompt_embeds if use_prompt_embeds else None,
                negative_prompt_embeds=negative_prompt_embeds if use_prompt_embeds else None,
            ).images[0]

            filename = batch_output_dir / f"{batch_id}_{current_seed}.png"
            # Store prompt/settings inside the PNG for later reproduction/debugging.
            pnginfo = build_png_metadata(
                {
                    "mode": "hires_fix",
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "steps": steps,
                    "cfg": cfg,
                    "seed": current_seed,
                    "scheduler": scheduler,
                    "model": model,
                    "clip_skip": clip_skip,
                    "hires_scale": hires_scale,
                    "hires_strength": hires_strength,
                    "batch_id": batch_id,
                }
            )
            out_image.save(filename, pnginfo=pnginfo)
            relpaths.append(build_batch_output_relpath(batch_id, filename.name))
    finally:
        _cleanup_lora_adapters(pipe, adapter_names)
        release_pipeline(pipe, logger=logger)

    return relpaths
