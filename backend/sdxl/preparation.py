"""Prompt conditioning and latent rendering helpers for SDXL."""

from backend.sdxl.runtime_common import *

def _get_pipe_device(
    pipe: StableDiffusionXLPipeline | StableDiffusionXLImg2ImgPipeline | StableDiffusionXLInpaintPipeline | StableDiffusionXLControlNetPipeline | StableDiffusionXLControlNetImg2ImgPipeline | StableDiffusionXLControlNetInpaintPipeline,
) -> torch.device | str:
    return getattr(pipe, "_execution_device", None) or pipe.device


def _get_module_device(module, fallback: torch.device | str) -> torch.device | str:
    module_device = getattr(module, "device", None)
    if module_device is not None:
        return module_device
    try:
        return next(module.parameters()).device
    except (AttributeError, StopIteration):
        return fallback


def _enable_vae_memory_savers(pipe: object) -> None:
    vae = getattr(pipe, "vae", None)
    vae_enable_slicing = getattr(vae, "enable_slicing", None)
    if callable(vae_enable_slicing):
        vae_enable_slicing()

    vae_enable_tiling = getattr(vae, "enable_tiling", None)
    if callable(vae_enable_tiling):
        vae_enable_tiling()


class _LatentDecoder:
    def __init__(
        self,
        *,
        vae: object,
        image_processor: object | None,
        device: torch.device | str,
    ) -> None:
        self.vae = vae
        self.image_processor = image_processor
        self.device = device


def _build_latent_decoder(pipe: object) -> _LatentDecoder:
    vae = getattr(pipe, "vae", None)
    if vae is None:
        raise RuntimeError("SDXL pipeline does not have a VAE for latent decoding.")
    return _LatentDecoder(
        vae=vae,
        image_processor=getattr(pipe, "image_processor", None),
        device=_get_module_device(vae, _get_pipe_device(pipe)),
    )


@contextmanager
def _hide_image_encoder_while_using_ip_adapter_embeds(pipe, *, enabled: bool):
    if not enabled or pipe is None or not hasattr(pipe, "image_encoder"):
        yield
        return

    image_encoder = pipe.image_encoder
    pipe.image_encoder = None
    try:
        yield
    finally:
        pipe.image_encoder = image_encoder


def _decode_latents_to_pil(
    pipe: object,
    latents: torch.Tensor,
) -> Image.Image:
    if latents.ndim == 3:
        latents = latents.unsqueeze(0)

    latents = latents.to(device=_get_module_device(pipe.vae, _get_pipe_device(pipe)), dtype=pipe.vae.dtype)
    latents = latents / pipe.vae.config.scaling_factor

    image = pipe.vae.decode(latents, return_dict=False)[0]

    if hasattr(pipe, "image_processor") and hasattr(pipe.image_processor, "postprocess"):
        return pipe.image_processor.postprocess(image, output_type="pil")[0]

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    return Image.fromarray((image[0] * 255).round().astype("uint8"))


def render_text2img_latents(
    pipe: StableDiffusionXLPipeline,
    *,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    seed: int,
    clip_skip: int,
    ip_adapter_image_embeds: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    ip_adapter_kwargs = (
        {"ip_adapter_image_embeds": ip_adapter_image_embeds}
        if ip_adapter_image_embeds is not None
        else {}
    )
    with _hide_image_encoder_while_using_ip_adapter_embeds(
        pipe,
        enabled=ip_adapter_image_embeds is not None,
    ):
        device = _get_pipe_device(pipe)
        generator = torch.Generator(device=device).manual_seed(seed)
        return pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            clip_skip=clip_skip,
            output_type="latent",
            **ip_adapter_kwargs,
        ).images[0]


def render_img2img_latents(
    pipe: StableDiffusionXLImg2ImgPipeline,
    *,
    initial_image: Image.Image,
    strength: float,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    seed: int,
    clip_skip: int,
    ip_adapter_image: Image.Image | None = None,
) -> torch.Tensor:
    device = _get_pipe_device(pipe)
    generator = torch.Generator(device=device).manual_seed(seed)
    ip_adapter_kwargs = (
        {"ip_adapter_image": ip_adapter_image} if ip_adapter_image is not None else {}
    )
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=initial_image,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        clip_skip=clip_skip,
        output_type="latent",
        **ip_adapter_kwargs,
    ).images[0]


def render_inpaint_image(
    pipe: StableDiffusionXLInpaintPipeline,
    *,
    initial_image: Image.Image,
    mask_image: Image.Image,
    strength: float,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    seed: int,
    padding_mask_crop: int,
    clip_skip: int,
    ip_adapter_image: Image.Image | None = None,
) -> Image.Image:
    device = _get_pipe_device(pipe)
    generator = torch.Generator(device=device).manual_seed(seed)
    ip_adapter_kwargs = (
        {"ip_adapter_image": ip_adapter_image} if ip_adapter_image is not None else {}
    )
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=initial_image,
        mask_image=mask_image,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        padding_mask_crop=padding_mask_crop,
        clip_skip=clip_skip,
        **ip_adapter_kwargs,
    ).images[0]

