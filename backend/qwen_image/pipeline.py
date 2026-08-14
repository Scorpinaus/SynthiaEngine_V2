from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version as package_version
import logging
import threading
from typing import TypeVar

import torch
from diffusers import QwenImageImg2ImgPipeline, QwenImageInpaintPipeline, QwenImagePipeline

from backend.config import OUTPUT_DIR
from backend.utilities.logging import configure_logging
from backend.registries.model import ModelRegistryEntry, list_model_entries
from backend.utilities.pipeline import (
    cleanup_memory,
    get_batch_output_dir,
    make_batch_id,
    release_pipeline,
    resolve_base_seed,
    resolve_model_source,
    save_generated_image,
)
from backend.utilities.schedulers import create_scheduler
from backend.utilities.subprocess_transport import (
    SubprocessTransport,
    normalize_image_result,
    run_subprocess,
)

_QWEN_IMAGE_SUBPROCESS_SEMAPHORE = threading.Semaphore(1)

_DEFAULT_MODEL_NAME = "Qwen-Image-2512-SDNQ-4bit-dynamic"
_DEFAULT_MODEL_LINK = r"D:\diffusion\diffusers\Qwen-Image-2512-SDNQ-4bit-dynamic"
_DEFAULT_MODEL_VERSION = "51bbb04c6c9664cc226f4403a9175aa2d0b29b9d"
_DEFAULT_MODEL_ALIASES = {
    _DEFAULT_MODEL_NAME,
    _DEFAULT_MODEL_LINK,
    "Disty0/Qwen-Image-2512-SDNQ-4bit-dynamic",
    "qwen-image",
}
_DEFAULT_NEGATIVE_PROMPT = (
    "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，"
    "过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
)
_DEFAULT_INFERENCE_STEPS = 50
_DEFAULT_TRUE_CFG_SCALE = 4.0
_DEFAULT_IMAGE_SIZE = 1328
_DEFAULT_STRENGTH = 0.6
_DEFAULT_SCHEDULER = "flowmatch_euler"
_NO_REVISION_VALUES = {"", "hub", "local"}
_SDNQ_REQUIRED_MESSAGE = (
    "Qwen-Image SDNQ requires the 'sdnq' package. Install the project requirements."
)
_SDNQ_LORA_UNSUPPORTED_MESSAGE = (
    "SynthiaEngine Qwen-Image SDNQ does not support LoRA adapters in the current "
    "compatibility profile."
)
_SDNQ_SCHEDULER_UNSUPPORTED_MESSAGE = (
    "Qwen-Image SDNQ supports only scheduler 'flowmatch_euler' in the current "
    "compatibility profile."
)
_SEQUENTIAL_OFFLOAD_METHOD = "sequential_cpu_offload"

_QwenPipelineT = TypeVar(
    "_QwenPipelineT",
    QwenImagePipeline,
    QwenImageImg2ImgPipeline,
    QwenImageInpaintPipeline,
)

logger = logging.getLogger(__name__)
configure_logging()

""" Methods involving loading of pipelines"""


def _default_model_entry() -> ModelRegistryEntry:
    return ModelRegistryEntry(
        name=_DEFAULT_MODEL_NAME,
        family="qwen-image",
        model_type="diffusers",
        location_type="local",
        model_id=15,
        version=_DEFAULT_MODEL_VERSION,
        link=_DEFAULT_MODEL_LINK,
    )


def _get_qwen_image_model_entry(model_name: str | None) -> ModelRegistryEntry:
    entries = list_model_entries()
    if model_name:
        requested_name = model_name.strip()
        for entry in entries:
            if entry.name == requested_name:
                if entry.family.lower() != "qwen-image":
                    raise ValueError(f"Model '{requested_name}' is not a Qwen-Image model.")
                return entry
        if requested_name in _DEFAULT_MODEL_ALIASES:
            return _default_model_entry()
        raise ValueError(f"Model '{requested_name}' not found.")

    for entry in entries:
        if entry.family.lower() == "qwen-image":
            return entry
    return _default_model_entry()


def _check_model_capabilities(entry: ModelRegistryEntry) -> None:
    if entry.family.strip().lower() != "qwen-image":
        raise ValueError(f"Model '{entry.name}' is not a Qwen-Image model.")
    if entry.model_type.strip().lower() != "diffusers":
        raise ValueError(
            "Qwen-Image SDNQ supports only Diffusers model folders or Hub repositories."
        )
    if entry.location_type.strip().lower() not in {"hub", "local"}:
        raise ValueError(
            "Qwen-Image SDNQ supports only local model folders or Hub repositories."
        )


def _resolve_model_revision(entry: ModelRegistryEntry) -> str | None:
    revision = entry.version.strip()
    if entry.location_type.strip().lower() != "hub":
        return None
    if revision.lower() in _NO_REVISION_VALUES:
        return None
    return revision


def _build_diffusers_load_arguments(entry: ModelRegistryEntry) -> dict[str, object]:
    load_arguments: dict[str, object] = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    revision = _resolve_model_revision(entry)
    if revision is not None:
        load_arguments["revision"] = revision
    return load_arguments


def _register_sdnq() -> str:
    try:
        import sdnq
        from sdnq import SDNQConfig
    except ImportError as exc:
        raise RuntimeError(_SDNQ_REQUIRED_MESSAGE) from exc

    # Importing this class registers SDNQ with Transformers and Diffusers.
    _ = SDNQConfig
    try:
        return package_version("sdnq")
    except PackageNotFoundError:
        return str(getattr(sdnq, "__version__", "unknown"))


def _config_value(config: object, name: str) -> object | None:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _component_quantization_config(pipe: object, component_name: str) -> object | None:
    component = getattr(pipe, component_name, None)
    component_config = getattr(component, "config", None)
    return _config_value(component_config, "quantization_config")


def _check_loaded_model_capabilities(pipe: object) -> None:
    missing_capabilities: list[str] = []
    for component_name in ("transformer", "text_encoder", "vae"):
        if getattr(pipe, component_name, None) is None:
            missing_capabilities.append(component_name)

    vae = getattr(pipe, "vae", None)
    if vae is not None:
        if not callable(getattr(vae, "enable_slicing", None)):
            missing_capabilities.append("vae.enable_slicing")
        if not callable(getattr(vae, "enable_tiling", None)):
            missing_capabilities.append("vae.enable_tiling")
    if not callable(getattr(pipe, "enable_sequential_cpu_offload", None)):
        missing_capabilities.append("enable_sequential_cpu_offload")

    if missing_capabilities:
        details = ", ".join(missing_capabilities)
        raise RuntimeError(
            f"Qwen-Image SDNQ pipeline is missing required capabilities: {details}."
        )


def _verify_sdnq_quantization(pipe: object) -> str:
    reported_methods: dict[str, object | None] = {}
    reported_versions: dict[str, object | None] = {}
    for component_name in ("transformer", "text_encoder"):
        quantization_config = _component_quantization_config(pipe, component_name)
        reported_methods[component_name] = _config_value(
            quantization_config,
            "quant_method",
        )
        reported_versions[component_name] = _config_value(
            quantization_config,
            "sdnq_version",
        )

    invalid_components = [
        component_name
        for component_name, quant_method in reported_methods.items()
        if str(getattr(quant_method, "value", quant_method)).strip().lower()
        != "sdnq"
    ]
    if invalid_components:
        reported = ", ".join(
            f"{component_name}={reported_methods[component_name]!r}"
            for component_name in reported_methods
        )
        raise RuntimeError(
            "Qwen-Image SDNQ quantization validation failed. Expected "
            f"quant_method='sdnq' for transformer and text_encoder; reported {reported}. "
            "Loading stopped to prevent a full BF16 fallback."
        )

    version_values = {
        str(value)
        for value in reported_versions.values()
        if value not in (None, "")
    }
    if len(version_values) == 1:
        return version_values.pop()
    if not version_values:
        return "unknown"
    return ", ".join(
        f"{name}={reported_versions[name]}" for name in reported_versions
    )


def _configure_pipeline_memory(pipe: object) -> str:
    if callable(getattr(pipe, "enable_attention_slicing", None)):
        pipe.enable_attention_slicing("max")

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.enable_sequential_cpu_offload()
    return _SEQUENTIAL_OFFLOAD_METHOD


def _load_qwen_image_pipeline(
    pipeline_class: type[_QwenPipelineT],
    model_name: str | None,
) -> _QwenPipelineT:
    entry = _get_qwen_image_model_entry(model_name)
    _check_model_capabilities(entry)
    source = resolve_model_source(entry)
    load_arguments = _build_diffusers_load_arguments(entry)
    revision = _resolve_model_revision(entry)

    sdnq_package_version = _register_sdnq()
    logger.info("Loading Qwen-Image SDNQ checkpoint: %s", source)

    pipe: _QwenPipelineT | None = None
    try:
        pipe = pipeline_class.from_pretrained(source, **load_arguments)
        _check_loaded_model_capabilities(pipe)
        checkpoint_sdnq_version = _verify_sdnq_quantization(pipe)
        memory_method = _configure_pipeline_memory(pipe)
    except Exception:
        release_pipeline(pipe, logger=logger)
        raise

    logger.info(
        "Qwen-Image SDNQ pipeline ready: checkpoint=%s revision=%s "
        "sdnq_package_version=%s sdnq_checkpoint_version=%s memory=%s",
        source,
        revision or entry.version or "default",
        sdnq_package_version,
        checkpoint_sdnq_version,
        memory_method,
    )
    return pipe


def load_text2img_pipeline(model_name: str | None) -> QwenImagePipeline:
    return _load_qwen_image_pipeline(QwenImagePipeline, model_name)


def load_img2img_pipeline(model_name: str | None) -> QwenImageImg2ImgPipeline:
    return _load_qwen_image_pipeline(QwenImageImg2ImgPipeline, model_name)


def load_inpaint_pipeline(model_name: str | None) -> QwenImageInpaintPipeline:
    return _load_qwen_image_pipeline(QwenImageInpaintPipeline, model_name)


""" Methods involving generation using Qwen_Image related pipelines """


def _has_requested_lora_adapters(value: object | None) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, bytes, list, tuple, set, dict)):
        return bool(value)
    return True


def _validate_feature_compatibility(params: Mapping[str, object]) -> str:
    scheduler = str(params.get("scheduler") or _DEFAULT_SCHEDULER).strip().lower()
    if scheduler != _DEFAULT_SCHEDULER:
        raise ValueError(_SDNQ_SCHEDULER_UNSUPPORTED_MESSAGE)
    if _has_requested_lora_adapters(params.get("lora_adapters")):
        raise ValueError(_SDNQ_LORA_UNSUPPORTED_MESSAGE)
    return scheduler


def _run_qwen_image_subprocess(operation: str, params: dict[str, object]) -> dict[str, list[str]]:
    _validate_feature_compatibility(params)
    result = run_subprocess(
        SubprocessTransport(
            family="Qwen-Image",
            runner_module="backend.qwen_image.subprocess_runner",
            temp_prefix="qwen_image_",
            launch_gate=_QWEN_IMAGE_SUBPROCESS_SEMAPHORE,
        ),
        operation,
        params,
    )
    return normalize_image_result(result, family="Qwen-Image")


def generate_text2img(params: dict[str, object]) -> dict[str, list[str]]:
    return _run_qwen_image_subprocess("text2img", params)


def generate_img2img(params: dict[str, object]) -> dict[str, list[str]]:
    return _run_qwen_image_subprocess("img2img", params)


def generate_inpaint(params: dict[str, object]) -> dict[str, list[str]]:
    return _run_qwen_image_subprocess("inpaint", params)


def _negative_prompt(params: dict[str, object]) -> str:
    value = params.get("negative_prompt")
    return _DEFAULT_NEGATIVE_PROMPT if value is None else str(value)


@torch.inference_mode()
def generate_text2img_in_process(params: dict[str, object]) -> dict[str, list[str]]:
    prompt = str(params.get("prompt") or "")
    negative_prompt = _negative_prompt(params)
    steps = int(params.get("steps", _DEFAULT_INFERENCE_STEPS))
    true_cfg_scale = float(params.get("true_cfg_scale", _DEFAULT_TRUE_CFG_SCALE))
    width = int(params.get("width", _DEFAULT_IMAGE_SIZE))
    height = int(params.get("height", _DEFAULT_IMAGE_SIZE))
    seed = params.get("seed")
    model = params.get("model")
    num_images = int(params.get("num_images", 1))
    scheduler = _validate_feature_compatibility(params)

    logger.info("seed=%s", seed)
    base_seed = resolve_base_seed(seed)

    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    logger.info(
        "Qwen-Image Generate: model=%s seed=%s steps=%s true_cfg_scale=%s "
        "size=%sx%s num_images=%s",
        model,
        base_seed,
        steps,
        true_cfg_scale,
        width,
        height,
        num_images,
    )

    filenames: list[str] = []
    pipe = None
    try:
        pipe = load_text2img_pipeline(model)
        pipe.scheduler = create_scheduler(scheduler, pipe)

        for i in range(num_images):
            current_seed = base_seed + i
            generator = torch.Generator(device="cpu").manual_seed(current_seed)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                call_kwargs: dict[str, object] = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": steps,
                    "true_cfg_scale": true_cfg_scale,
                    "width": width,
                    "height": height,
                    "generator": generator,
                }
                image = pipe(**call_kwargs).images[0]

            relpath = save_generated_image(
                image, batch_output_dir, batch_id, current_seed, params,
                mode="txt2img", pipeline="qwen-image",
            )
            logger.info("Image %s saved to %s", i, relpath)
            filenames.append(relpath)

            del image
            cleanup_memory()
    finally:
        release_pipeline(pipe, logger=logger)
        pipe = None

    return {"images": [f"/outputs/{name}" for name in filenames]}


@torch.inference_mode()
def generate_img2img_in_process(params: dict[str, object]) -> dict[str, list[str]]:
    initial_image = params.get("initial_image")
    if initial_image is None:
        raise ValueError("initial_image is required")
    strength = float(params.get("strength", _DEFAULT_STRENGTH))
    prompt = str(params.get("prompt") or "")
    negative_prompt = _negative_prompt(params)
    steps = int(params.get("steps", _DEFAULT_INFERENCE_STEPS))
    true_cfg_scale = float(params.get("true_cfg_scale", _DEFAULT_TRUE_CFG_SCALE))
    width = int(params.get("width", _DEFAULT_IMAGE_SIZE))
    height = int(params.get("height", _DEFAULT_IMAGE_SIZE))
    seed = params.get("seed")
    model = params.get("model")
    num_images = int(params.get("num_images", 1))
    scheduler = _validate_feature_compatibility(params)

    logger.info("seed=%s", seed)
    base_seed = resolve_base_seed(seed)

    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    logger.info(
        "Qwen-Image Img2Img: model=%s seed=%s steps=%s true_cfg_scale=%s "
        "size=%sx%s strength=%s num_images=%s",
        model,
        base_seed,
        steps,
        true_cfg_scale,
        width,
        height,
        strength,
        num_images,
    )

    filenames: list[str] = []
    pipe = None
    try:
        pipe = load_img2img_pipeline(model)
        pipe.scheduler = create_scheduler(scheduler, pipe)

        for i in range(num_images):
            current_seed = base_seed + i
            generator = torch.Generator(device="cpu").manual_seed(current_seed)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                call_kwargs: dict[str, object] = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "image": initial_image,
                    "strength": strength,
                    "num_inference_steps": steps,
                    "true_cfg_scale": true_cfg_scale,
                    "width": width,
                    "height": height,
                    "generator": generator,
                }
                image = pipe(**call_kwargs).images[0]

            image_width, image_height = initial_image.size
            relpath = save_generated_image(
                image, batch_output_dir, batch_id, current_seed, params,
                mode="img2img", pipeline="qwen-image",
                remove_params=("initial_image",),
                size=(image_width, image_height),
            )
            logger.info("Image %s saved to %s", i, relpath)
            filenames.append(relpath)

            del image
            cleanup_memory()
    finally:
        release_pipeline(pipe, logger=logger)
        pipe = None

    return {"images": [f"/outputs/{name}" for name in filenames]}


@torch.inference_mode()
def generate_inpaint_in_process(params: dict[str, object]) -> dict[str, list[str]]:
    initial_image = params.get("initial_image")
    if initial_image is None:
        raise ValueError("initial_image is required")
    mask_image = params.get("mask_image")
    if mask_image is None:
        raise ValueError("mask_image is required")
    strength = float(params.get("strength", _DEFAULT_STRENGTH))
    prompt = str(params.get("prompt") or "")
    negative_prompt = _negative_prompt(params)
    steps = int(params.get("steps", _DEFAULT_INFERENCE_STEPS))
    true_cfg_scale = float(params.get("true_cfg_scale", _DEFAULT_TRUE_CFG_SCALE))
    seed = params.get("seed")
    model = params.get("model")
    num_images = int(params.get("num_images", 1))
    scheduler = _validate_feature_compatibility(params)

    logger.info("seed=%s", seed)
    base_seed = resolve_base_seed(seed)

    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)

    width, height = initial_image.size
    logger.info(
        "Qwen-Image Inpaint: model=%s seed=%s steps=%s true_cfg_scale=%s "
        "size=%sx%s strength=%s num_images=%s",
        model,
        base_seed,
        steps,
        true_cfg_scale,
        width,
        height,
        strength,
        num_images,
    )

    filenames: list[str] = []
    pipe = None
    try:
        pipe = load_inpaint_pipeline(model)
        pipe.scheduler = create_scheduler(scheduler, pipe)

        for i in range(num_images):
            current_seed = base_seed + i
            generator = torch.Generator(device="cpu").manual_seed(current_seed)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                call_kwargs: dict[str, object] = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "image": initial_image,
                    "mask_image": mask_image,
                    "strength": strength,
                    "num_inference_steps": steps,
                    "true_cfg_scale": true_cfg_scale,
                    "generator": generator,
                }
                image = pipe(**call_kwargs).images[0]

            relpath = save_generated_image(
                image, batch_output_dir, batch_id, current_seed, params,
                mode="inpaint", pipeline="qwen-image",
                remove_params=("initial_image", "mask_image"),
                size=(width, height),
            )
            logger.info("Image %s saved to %s", i, relpath)
            filenames.append(relpath)

            del image
            cleanup_memory()
    finally:
        release_pipeline(pipe, logger=logger)
        pipe = None

    return {"images": [f"/outputs/{name}" for name in filenames]}
