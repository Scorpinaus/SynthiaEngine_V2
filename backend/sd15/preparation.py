"""Prompt, image, ControlNet, and memory preparation helpers for SD1.5."""

from backend.sd15.runtime_common import *

def create_blur_mask(mask_image, blur_factor: int):
    """
    Return a blurred copy of `mask_image` with a bounded Gaussian blur radius.

    Args:
        mask_image: PIL image used as an inpaint mask.
        blur_factor: Requested blur radius. Values are clamped to ``[0, 128]``.

    Returns:
        The original image when blur is ``0``; otherwise a blurred copy.
    """
    blur_factor = max(0, min(blur_factor, 128))
    if blur_factor == 0:
        return mask_image
    return mask_image.filter(ImageFilter.GaussianBlur(radius=blur_factor))


def _build_sd15_prompt_call_kwargs(
    pipe,
    prompt: str,
    negative_prompt: str,
    *,
    clip_skip: int | None,
    weighting_policy: str = "diffusers-like",
) -> dict[str, object]:
    """Build mutually exclusive raw-prompt or precomputed-embedding kwargs."""
    prompt_embeds, negative_prompt_embeds, use_prompt_embeds = build_prompt_embeddings(
        pipe,
        prompt,
        negative_prompt,
        clip_skip=clip_skip,
        weighting_policy=weighting_policy,
    )
    if use_prompt_embeds:
        return {
            "prompt": None,
            "negative_prompt": None,
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            # Embeddings already include clip-skip. Keeping this unset also
            # avoids Diffusers 0.39's incompatible Transformers 5.x lookup.
            "clip_skip": None,
        }
    return {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "prompt_embeds": None,
        "negative_prompt_embeds": None,
        "clip_skip": None,
    }


def _resource_metadata(bound_args):
    """
    Build resource-logging metadata from a function's bound arguments.

    This keeps the logging payload small and consistent across generation calls.
    """
    return {
        "batch_id": bound_args.arguments.get("batch_id"),
        "model": bound_args.arguments.get("model"),
        "num_images": bound_args.arguments.get("num_images"),
    }


def _snap_dimension(value: int, multiple: int = 8) -> int:
    """Round a dimension up to the next multiple (SD models commonly prefer multiples of 8)."""
    if multiple <= 0:
        return value
    return max(multiple, int(math.ceil(value / multiple)) * multiple)


def _upscale_image(image: Image.Image, scale: float) -> Image.Image:
    """Upscale an image by `scale` using Lanczos, snapping size to SD-friendly dimensions."""
    if scale <= 1.0:
        return image
    target_width = _snap_dimension(int(round(image.width * scale)))
    target_height = _snap_dimension(int(round(image.height * scale)))
    return image.resize((target_width, target_height), resample=Image.LANCZOS)


def _resize_control_image_to_target(
    control_image: Image.Image | list[Image.Image],
    *,
    target_width: int,
    target_height: int,
) -> Image.Image | list[Image.Image]:
    """Resize ControlNet image(s) to exactly match the rendered output size."""

    def _resize_single(image: Image.Image, index: int | None = None) -> Image.Image:
        source_width, source_height = image.size
        if source_width == target_width and source_height == target_height:
            return image

        if source_width != target_width and source_height != target_height:
            resize_case = "resize_width_and_height"
        elif source_height != target_height:
            resize_case = "resize_height_only"
        else:
            resize_case = "resize_width_only"

        if index is None:
            logger.info(
                "Resizing ControlNet control_image (%s): %sx%s -> %sx%s",
                resize_case,
                source_width,
                source_height,
                target_width,
                target_height,
            )
        else:
            logger.info(
                "Resizing ControlNet control_image[%s] (%s): %sx%s -> %sx%s",
                index,
                resize_case,
                source_width,
                source_height,
                target_width,
                target_height,
            )
        return image.resize((target_width, target_height), resample=Image.LANCZOS)

    if isinstance(control_image, list):
        return [_resize_single(image, index=i) for i, image in enumerate(control_image)]
    return _resize_single(control_image)


def _make_inpaint_controlnet_condition(
    image: Image.Image,
    mask_image: Image.Image,
) -> torch.Tensor:
    """
    Build the special conditioning tensor expected by SD1.5 inpaint ControlNet.

    The ControlNet v1.1 inpaint checkpoint is conditioned on the original image
    with masked pixels set to -1.0, matching the Diffusers model-card example.
    """
    rgb_image = image.convert("RGB")
    mask = mask_image.convert("L").resize(rgb_image.size)
    image_array = np.array(rgb_image).astype(np.float32) / 255.0
    mask_array = np.array(mask).astype(np.float32) / 255.0
    image_array[mask_array > 0.5] = -1.0
    image_array = np.expand_dims(image_array, 0).transpose(0, 3, 1, 2)
    return torch.from_numpy(image_array)


def _enable_xformers_memory_efficient_attention_if_available(pipe) -> bool:
    """
    Enable xFormers attention when the optional dependency is installed.

    xFormers is a performance optimization, not a functional requirement. Some
    Windows/local installs do not include it, so generation should keep running
    with Diffusers' default attention path when it is unavailable.
    """
    if not hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        logger.debug(
            "Pipeline %s does not expose xFormers memory efficient attention.",
            pipe.__class__.__name__,
        )
        return False

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except (ImportError, ModuleNotFoundError) as exc:
        logger.warning(
            "xFormers memory efficient attention is unavailable; continuing without it. %s",
            exc,
        )
        return False

    logger.info("Enabled xFormers memory efficient attention.")
    return True

