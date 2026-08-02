"""Diffusers pipeline factories for SD1.5 operations."""

from backend.sd15.runtime_common import *

def load_text2img_pipeline(model_name: str | None):
    """
    Load the base SD1.5 txt2img pipeline on CUDA fp16.

    ``model_name`` is resolved via the model registry and may point to a
    Diffusers directory model or a single-file checkpoint.

    Side effects:
        Moves the pipeline to GPU (``cuda``) and disables the safety checker.
    """
    entry = get_model_entry(model_name)

    source = resolve_model_source(entry)
    logger.info("URL: %s", source)
    
    if entry.model_type == "diffusers":
        pipe = StableDiffusionPipeline.from_pretrained(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,  # keep simple; can re-enable later
        )
    elif entry.model_type == "single-file":
        pipe = StableDiffusionPipeline.from_single_file(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    # Run on CUDA in fp16 for performance. Safety checker is disabled by design here.
    pipe.to("cuda")
    return pipe


def load_img2img_pipeline(model_name: str | None):
    """
    Load the SD1.5 img2img pipeline on CUDA fp16.

    Side effects:
        Moves the pipeline to GPU (``cuda``) and disables the safety checker.
    """
    entry = get_model_entry(model_name)

    source = resolve_model_source(entry)
    logger.info("URL: %s", source)
    if entry.model_type == "diffusers":
        img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        img2img_pipe = StableDiffusionImg2ImgPipeline.from_single_file(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    img2img_pipe.to("cuda")
    return img2img_pipe


def load_inpaint_pipeline(model_name: str | None):
    """
    Load the SD1.5 inpainting pipeline on CUDA fp16.

    Side effects:
        Moves the pipeline to GPU (``cuda``) and disables the safety checker.
    """
    entry = get_model_entry(model_name)

    source = resolve_model_source(entry)
    logger.info("URL: %s", source)
    if entry.model_type == "diffusers":
        inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        inpaint_pipe = StableDiffusionInpaintPipeline.from_single_file(
            source,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    inpaint_pipe.to("cuda")
    return inpaint_pipe


def load_controlnet_pipeline(model_name: str | None, controlnet_model: str | list[str]):
    """
    Load a ControlNet-enabled SD1.5 pipeline on CUDA fp16.

    Args:
        model_name: Optional base model registry key.
        controlnet_model: Diffusers ControlNet model id/path or list of ids/paths.

    Side effects:
        Loads both base and ControlNet weights and moves the pipeline to GPU.
    """
    entry = get_model_entry(model_name)

    source = resolve_model_source(entry)
    logger.info("Base model: %s", source)
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
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        pipe = StableDiffusionControlNetPipeline.from_single_file(
            source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    pipe.to("cuda")
    return pipe


def load_controlnet_img2img_pipeline(model_name: str | None, controlnet_model: str | list[str]):
    """
    Load a ControlNet-enabled SD1.5 img2img pipeline on CUDA fp16.

    Args:
        model_name: Optional base model registry key.
        controlnet_model: Diffusers ControlNet model id/path or list of ids/paths.

    Side effects:
        Loads both base and ControlNet weights and moves the pipeline to GPU.
    """
    entry = get_model_entry(model_name)

    source = resolve_model_source(entry)
    logger.info("Base model: %s", source)
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
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
            source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    pipe.to("cuda")
    return pipe


def load_controlnet_inpaint_pipeline(model_name: str | None, controlnet_model: str | list[str]):
    """
    Load a ControlNet-enabled SD1.5 inpaint pipeline on CUDA fp16.

    Args:
        model_name: Optional base model registry key.
        controlnet_model: Diffusers ControlNet model id/path or list of ids/paths.

    Side effects:
        Loads both base and ControlNet weights and moves the pipeline to GPU.
    """
    entry = get_model_entry(model_name)

    source = resolve_model_source(entry)
    logger.info("Base model: %s", source)
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
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    elif entry.model_type == "single-file":
        pipe = StableDiffusionControlNetInpaintPipeline.from_single_file(
            source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        raise ValueError(f"Unsupported model type: {entry.model_type}")

    pipe.to("cuda")
    return pipe

