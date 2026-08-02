"""Diffusers pipeline factories for SDXL operations."""

from backend.sdxl.runtime_common import *
from backend.sdxl.preparation import _enable_vae_memory_savers

def load_text2img_pipeline(model_name: str | None) -> StableDiffusionXLPipeline:
    entry = get_model_entry(model_name)

    source = resolve_model_source(entry)
    logger.info("SDXL model source: %s", source)

    if entry.model_type == "diffusers":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        pipe = StableDiffusionXLPipeline.from_single_file(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    _enable_vae_memory_savers(pipe)
    pipe.to("cuda")
    return pipe


def load_controlnet_text2img_pipeline(
    model_name: str | None,
    controlnet_model: str | list[str],
) -> StableDiffusionXLControlNetPipeline:
    
    entry = get_model_entry(model_name)
    source = resolve_model_source(entry)
    logger.info("SDXL ControlNet model source: %s", source)

    controlnet: ControlNetModel | list[ControlNetModel]
    if isinstance(controlnet_model, list):
        controlnet = [
            ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)
            for model_id in controlnet_model
        ]
    else:
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model,
            torch_dtype=torch.float16,
        )

    if entry.model_type == "diffusers":
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        pipe = StableDiffusionXLControlNetPipeline.from_single_file(
            source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    pipe.to("cuda")
    return pipe


def load_img2img_pipeline(model_name: str | None) -> StableDiffusionXLImg2ImgPipeline:
    entry = get_model_entry(model_name)
    source = resolve_model_source(entry)
    logger.info("SDXL img2img model source: %s", source)

    if entry.model_type == "diffusers":
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    pipe.to("cuda")
    return pipe


def load_controlnet_img2img_pipeline(
    model_name: str | None,
    controlnet_model: str | list[str],
) -> StableDiffusionXLControlNetImg2ImgPipeline:
    entry = get_model_entry(model_name)
    source = resolve_model_source(entry)
    logger.info("SDXL ControlNet img2img model source: %s", source)

    controlnet: ControlNetModel | list[ControlNetModel]
    if isinstance(controlnet_model, list):
        controlnet = [
            ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)
            for model_id in controlnet_model
        ]
    else:
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model,
            torch_dtype=torch.float16,
        )

    if entry.model_type == "diffusers":
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
            source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    pipe.to("cuda")
    return pipe


def load_inpaint_pipeline(model_name: str | None) -> StableDiffusionXLInpaintPipeline:
    entry = get_model_entry(model_name)
    source = resolve_model_source(entry)
    logger.info("SDXL inpaint model source: %s", source)

    if entry.model_type == "diffusers":
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        pipe = StableDiffusionXLInpaintPipeline.from_single_file(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    pipe.to("cuda")
    return pipe


def load_controlnet_inpaint_pipeline(
    model_name: str | None,
    controlnet_model: str | list[str],
) -> StableDiffusionXLControlNetInpaintPipeline:
    entry = get_model_entry(model_name)
    source = resolve_model_source(entry)
    logger.info("SDXL ControlNet inpaint model source: %s", source)

    controlnet: ControlNetModel | list[ControlNetModel]
    if isinstance(controlnet_model, list):
        controlnet = [
            ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)
            for model_id in controlnet_model
        ]
    else:
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model,
            torch_dtype=torch.float16,
        )

    if entry.model_type == "diffusers":
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_single_file(
            source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    pipe.to("cuda")
    return pipe

