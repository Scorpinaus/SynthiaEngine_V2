from collections.abc import Callable, Mapping
from importlib.metadata import PackageNotFoundError, version as package_version
import logging
import threading
from pathlib import Path
from typing import Any, TypeVar

import torch
from diffusers import QwenImageImg2ImgPipeline, QwenImageInpaintPipeline, QwenImagePipeline
from PIL import Image

from backend.config import OUTPUT_DIR
from backend.lora.utils import (
    apply_lora_adapters_with_validation,
    write_lora_coverage_report,
)
from backend.lora.registry import LoraRegistryEntry
from backend.qwen_image.lightning import (
    QwenImageLightningResolution,
    resolve_qwen_image_lightning_profile,
    select_qwen_image_scheduler,
)
from backend.utilities.logging import configure_logging
from backend.registries.model import ModelRegistryEntry, list_model_entries
from backend.utilities.pipeline import (
    build_batch_output_relpath,
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
    SubprocessCanceled,
    SubprocessRuntime,
    SubprocessTransport,
    normalize_image_result,
    pop_subprocess_runtime,
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
_DEFAULT_INPAINT_IMAGE_SIZE = 1024
_DEFAULT_STRENGTH = 0.6
_DEFAULT_SCHEDULER = "flowmatch_euler"
_PREVIEW_MAX_EDGE = 768
_NO_REVISION_VALUES = {"", "hub", "local"}
_SDNQ_REQUIRED_MESSAGE = (
    "Qwen-Image SDNQ requires the 'sdnq' package. Install the project requirements."
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


def _validate_feature_compatibility(params: Mapping[str, object]) -> str:
    scheduler = str(params.get("scheduler") or _DEFAULT_SCHEDULER).strip().lower()
    if scheduler != _DEFAULT_SCHEDULER:
        raise ValueError(_SDNQ_SCHEDULER_UNSUPPORTED_MESSAGE)
    return scheduler


def _qwen_lora_adapters(params: Mapping[str, object]) -> list[object] | None:
    value = params.get("lora_adapters")
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("Qwen-Image lora_adapters must be a list.")
    return value


def _apply_qwen_lora_adapters(
    pipe: Any,
    lora_adapters: list[object] | None,
    *,
    batch_output_dir: Path,
    batch_id: str,
    resolved_entries: Mapping[int, LoraRegistryEntry] | None = None,
    resolution: QwenImageLightningResolution | None = None,
) -> list[str]:
    if not lora_adapters:
        return []

    mixed_lightning_stack = (
        resolution is not None
        and resolution.lightning_profile is not None
        and len(resolution.adapters) == 2
    )
    if mixed_lightning_stack and not callable(getattr(pipe, "set_adapters", None)):
        raise RuntimeError(
            "Qwen Image Lightning companion stack requires callable pipeline.set_adapters support."
        )

    adapter_names, coverage = apply_lora_adapters_with_validation(
        pipe,
        lora_adapters,
        expected_family="qwen-image",
        validate=True,
        allowed_lora_types=("lora",),
        allowed_targets=("both",),
        coverage_components=("transformer",),
        resolved_entries=resolved_entries,
    )
    if resolution is not None:
        logger.info(
            "Qwen-Image LoRA resolved: adapter_ids=%s adapter_names=%s strengths=%s "
            "base_variant=%s task=%s lightning_steps=%s",
            [adapter.lora_id for adapter in resolution.adapters],
            adapter_names,
            [adapter.strength for adapter in resolution.adapters],
            resolution.model_variant,
            resolution.task,
            (
                resolution.lightning_profile.steps
                if resolution.lightning_profile is not None
                else None
            ),
        )
    if mixed_lightning_stack:
        get_active_adapters = getattr(pipe, "get_active_adapters", None)
        if callable(get_active_adapters):
            active_adapter_names = set(get_active_adapters())
            missing_adapter_names = [
                name for name in adapter_names if name not in active_adapter_names
            ]
            if missing_adapter_names:
                raise RuntimeError(
                    "Qwen Image Lightning companion stack activation is missing "
                    f"adapter names: {', '.join(missing_adapter_names)}."
                )
    report_path = write_lora_coverage_report(
        batch_output_dir,
        batch_id,
        coverage,
    )
    if report_path is not None:
        logger.info("Qwen-Image LoRA coverage report saved to %s", report_path)
    return adapter_names


def _cleanup_qwen_lora_adapters(pipe: Any | None, *, requested: bool) -> None:
    if pipe is None or not requested:
        return
    unload_lora_weights = getattr(pipe, "unload_lora_weights", None)
    if not callable(unload_lora_weights):
        logger.warning("Qwen-Image pipeline cannot unload requested LoRA adapters.")
        return
    try:
        unload_lora_weights()
    except Exception:
        logger.exception("Failed to unload Qwen-Image LoRA weights cleanly.")


def _run_qwen_image_subprocess(
    operation: str,
    params: dict[str, object],
    *,
    update_progress: Callable[[dict[str, Any]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, list[str]]:
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
        on_progress=update_progress,
        should_cancel=should_cancel,
    )
    return normalize_image_result(result, family="Qwen-Image")


def generate_text2img(
    params: dict[str, object],
    *,
    update_progress: Callable[[dict[str, Any]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, list[str]]:
    return _run_qwen_image_subprocess(
        "text2img",
        params,
        update_progress=update_progress,
        should_cancel=should_cancel,
    )


def generate_img2img(
    params: dict[str, object],
    *,
    update_progress: Callable[[dict[str, Any]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, list[str]]:
    return _run_qwen_image_subprocess(
        "img2img",
        params,
        update_progress=update_progress,
        should_cancel=should_cancel,
    )


def generate_inpaint(
    params: dict[str, object],
    *,
    update_progress: Callable[[dict[str, Any]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, list[str]]:
    return _run_qwen_image_subprocess(
        "inpaint",
        params,
        update_progress=update_progress,
        should_cancel=should_cancel,
    )


def _negative_prompt(params: dict[str, object]) -> str:
    value = params.get("negative_prompt")
    return _DEFAULT_NEGATIVE_PROMPT if value is None else str(value)


def _raise_if_cancelled(runtime: SubprocessRuntime) -> None:
    if runtime.should_cancel():
        raise SubprocessCanceled("Cancel requested")


def _preview_output(
    batch_id: str,
    batch_output_dir: Path,
) -> tuple[Path, str]:
    filename = f"{batch_id}_preview.png"
    return (
        batch_output_dir / filename,
        f"/outputs/{build_batch_output_relpath(batch_id, filename)}",
    )


def _remove_preview(preview_path: Path) -> None:
    for path in (preview_path, preview_path.with_suffix(".tmp")):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove Qwen-Image preview file: %s", path)


def _save_preview_image(image: Image.Image, preview_path: Path) -> None:
    preview = image.copy()
    preview.thumbnail(
        (_PREVIEW_MAX_EDGE, _PREVIEW_MAX_EDGE),
        Image.Resampling.LANCZOS,
    )
    temp_path = preview_path.with_suffix(".tmp")
    preview.save(temp_path, format="PNG", optimize=True)
    temp_path.replace(preview_path)


def _decode_preview_image(
    pipe: Any,
    packed_latents: Any,
    *,
    width: int,
    height: int,
) -> Image.Image:
    latents = pipe._unpack_latents(
        packed_latents.detach(),
        height,
        width,
        pipe.vae_scale_factor,
    )
    latents = latents.to(pipe.vae.dtype)
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
        1,
        pipe.vae.config.z_dim,
        1,
        1,
        1,
    ).to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean
    decoded = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")[0]


def _build_step_callback(
    runtime: SubprocessRuntime,
    *,
    requested_steps: int,
    image_index: int,
    total_images: int,
    width: int,
    height: int,
    preview_path: Path,
    preview_url: str,
    preview_transform: Callable[[Any, Image.Image], Image.Image] | None = None,
    live_preview: bool = True,
):
    def _on_step_end(pipe, step, _timestep, callback_kwargs):
        _raise_if_cancelled(runtime)
        total_steps = max(1, int(getattr(pipe, "num_timesteps", requested_steps)))
        current_step = min(int(step) + 1, total_steps)
        overall_total = total_steps * total_images
        overall_step = image_index * total_steps + current_step
        progress = {
            "phase": "denoising",
            "image_number": image_index + 1,
            "total_images": total_images,
            "step": current_step,
            "total_steps": total_steps,
            "percent": round(overall_step * 100 / overall_total, 1),
        }
        runtime.update_progress(progress)

        should_preview = (
            live_preview
            and runtime.progress_path is not None
            and current_step < total_steps
            and callback_kwargs.get("latents") is not None
        )
        if should_preview:
            try:
                preview = _decode_preview_image(
                    pipe,
                    callback_kwargs["latents"],
                    width=width,
                    height=height,
                )
                if preview_transform is not None:
                    preview = preview_transform(pipe, preview)
                _save_preview_image(preview, preview_path)
                runtime.update_progress({**progress, "preview_url": preview_url})
            except Exception:
                logger.exception(
                    "Qwen-Image preview decode failed at image=%s step=%s.",
                    image_index + 1,
                    current_step,
                )
        return callback_kwargs

    return _on_step_end


def _install_step_callback(
    call_kwargs: dict[str, object],
    runtime: SubprocessRuntime,
    **callback_options: Any,
) -> None:
    if runtime.progress_path is None and runtime.cancel_path is None:
        return
    preview_enabled = (
        bool(callback_options.get("live_preview", True))
        and runtime.progress_path is not None
    )
    callback_options["live_preview"] = preview_enabled
    call_kwargs["callback_on_step_end"] = _build_step_callback(
        runtime,
        **callback_options,
    )
    call_kwargs["callback_on_step_end_tensor_inputs"] = (
        ["latents"] if preview_enabled else []
    )


def _publish_image_complete(
    runtime: SubprocessRuntime,
    *,
    image_index: int,
    total_images: int,
    output_url: str,
) -> None:
    runtime.update_progress(
        {
            "phase": "image_completed",
            "image_number": image_index + 1,
            "total_images": total_images,
            "percent": round((image_index + 1) * 100 / total_images, 1),
            "preview_url": output_url,
        }
    )


def _publish_loading_model(runtime: SubprocessRuntime, total_images: int) -> None:
    runtime.update_progress(
        {
            "phase": "loading_model",
            "image_number": 1,
            "total_images": total_images,
            "percent": 0.0,
        }
    )


@torch.inference_mode()
def generate_text2img_in_process(params: dict[str, object]) -> dict[str, list[str]]:
    params = dict(params)
    runtime = pop_subprocess_runtime(params)
    prompt = str(params.get("prompt") or "")
    negative_prompt = _negative_prompt(params)
    steps = int(params.get("steps", _DEFAULT_INFERENCE_STEPS))
    true_cfg_scale = float(params.get("true_cfg_scale", _DEFAULT_TRUE_CFG_SCALE))
    width = int(params.get("width", _DEFAULT_IMAGE_SIZE))
    height = int(params.get("height", _DEFAULT_IMAGE_SIZE))
    seed = params.get("seed")
    model = params.get("model")
    num_images = int(params.get("num_images", 1))
    _validate_feature_compatibility(params)
    lora_adapters = _qwen_lora_adapters(params)
    model_entry = _get_qwen_image_model_entry(model)
    resolution = resolve_qwen_image_lightning_profile(
        lora_adapters,
        model_entry,
        "text2img",
        steps,
        true_cfg_scale,
    )
    resolved_entries = {adapter.lora_id: adapter.entry for adapter in resolution.adapters}
    lora_requested = bool(lora_adapters)

    logger.info("seed=%s", seed)
    base_seed = resolve_base_seed(seed)

    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)
    preview_path, preview_url = _preview_output(batch_id, batch_output_dir)

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
        _raise_if_cancelled(runtime)
        _publish_loading_model(runtime, num_images)
        pipe = load_text2img_pipeline(model)
        pipe.scheduler = select_qwen_image_scheduler(resolution, pipe)
        _apply_qwen_lora_adapters(
            pipe,
            lora_adapters,
            batch_output_dir=batch_output_dir,
            batch_id=batch_id,
            resolved_entries=resolved_entries,
            resolution=resolution,
        )

        for i in range(num_images):
            _raise_if_cancelled(runtime)
            current_seed = base_seed + i
            generator = torch.Generator(device="cpu").manual_seed(current_seed)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                call_kwargs: dict[str, object] = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": resolution.steps,
                    "true_cfg_scale": resolution.true_cfg_scale,
                    "width": width,
                    "height": height,
                    "generator": generator,
                }
                _install_step_callback(
                    call_kwargs,
                    runtime,
                    requested_steps=resolution.steps,
                    image_index=i,
                    total_images=num_images,
                    width=width,
                    height=height,
                    preview_path=preview_path,
                    preview_url=preview_url,
                    live_preview=bool(params.get("live_preview", True)),
                )
                image = pipe(**call_kwargs).images[0]

            _raise_if_cancelled(runtime)
            relpath = save_generated_image(
                image, batch_output_dir, batch_id, current_seed, params,
                mode="txt2img", pipeline="qwen-image",
            )
            logger.info("Image %s saved to %s", i, relpath)
            filenames.append(relpath)
            _publish_image_complete(
                runtime,
                image_index=i,
                total_images=num_images,
                output_url=f"/outputs/{relpath}",
            )

            del image
            cleanup_memory()
    finally:
        _remove_preview(preview_path)
        _cleanup_qwen_lora_adapters(pipe, requested=lora_requested)
        release_pipeline(pipe, logger=logger)
        pipe = None

    return {"images": [f"/outputs/{name}" for name in filenames]}


@torch.inference_mode()
def generate_img2img_in_process(params: dict[str, object]) -> dict[str, list[str]]:
    params = dict(params)
    runtime = pop_subprocess_runtime(params)
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
    lora_adapters = _qwen_lora_adapters(params)
    model_entry = _get_qwen_image_model_entry(model)
    resolution = resolve_qwen_image_lightning_profile(
        lora_adapters,
        model_entry,
        "img2img",
        steps,
        true_cfg_scale,
    )
    resolved_entries = {adapter.lora_id: adapter.entry for adapter in resolution.adapters}
    lora_requested = bool(lora_adapters)

    logger.info("seed=%s", seed)
    base_seed = resolve_base_seed(seed)

    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)
    preview_path, preview_url = _preview_output(batch_id, batch_output_dir)

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
        _raise_if_cancelled(runtime)
        _publish_loading_model(runtime, num_images)
        pipe = load_img2img_pipeline(model)
        if resolution.lightning_profile is not None:
            pipe.scheduler = select_qwen_image_scheduler(resolution, pipe)
        else:
            pipe.scheduler = create_scheduler(scheduler, pipe)
        _apply_qwen_lora_adapters(
            pipe,
            lora_adapters,
            batch_output_dir=batch_output_dir,
            batch_id=batch_id,
            resolved_entries=resolved_entries,
            resolution=resolution,
        )

        for i in range(num_images):
            _raise_if_cancelled(runtime)
            current_seed = base_seed + i
            generator = torch.Generator(device="cpu").manual_seed(current_seed)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                call_kwargs: dict[str, object] = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "image": initial_image,
                    "strength": strength,
                    "num_inference_steps": resolution.steps,
                    "true_cfg_scale": resolution.true_cfg_scale,
                    "width": width,
                    "height": height,
                    "generator": generator,
                }
                _install_step_callback(
                    call_kwargs,
                    runtime,
                    requested_steps=resolution.steps,
                    image_index=i,
                    total_images=num_images,
                    width=width,
                    height=height,
                    preview_path=preview_path,
                    preview_url=preview_url,
                    live_preview=bool(params.get("live_preview", True)),
                )
                image = pipe(**call_kwargs).images[0]

            _raise_if_cancelled(runtime)
            image_width, image_height = initial_image.size
            relpath = save_generated_image(
                image, batch_output_dir, batch_id, current_seed, params,
                mode="img2img", pipeline="qwen-image",
                remove_params=("initial_image",),
                size=(image_width, image_height),
            )
            logger.info("Image %s saved to %s", i, relpath)
            filenames.append(relpath)
            _publish_image_complete(
                runtime,
                image_index=i,
                total_images=num_images,
                output_url=f"/outputs/{relpath}",
            )

            del image
            cleanup_memory()
    finally:
        _remove_preview(preview_path)
        _cleanup_qwen_lora_adapters(pipe, requested=lora_requested)
        release_pipeline(pipe, logger=logger)
        pipe = None

    return {"images": [f"/outputs/{name}" for name in filenames]}


@torch.inference_mode()
def generate_inpaint_in_process(params: dict[str, object]) -> dict[str, list[str]]:
    params = dict(params)
    runtime = pop_subprocess_runtime(params)
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
    width = int(params.get("width") or _DEFAULT_INPAINT_IMAGE_SIZE)
    height = int(params.get("height") or _DEFAULT_INPAINT_IMAGE_SIZE)
    padding_mask_crop_value = params.get("padding_mask_crop")
    padding_mask_crop = (
        None if padding_mask_crop_value is None else int(padding_mask_crop_value)
    )
    seed = params.get("seed")
    model = params.get("model")
    num_images = int(params.get("num_images", 1))
    scheduler = _validate_feature_compatibility(params)
    lora_adapters = _qwen_lora_adapters(params)
    model_entry = _get_qwen_image_model_entry(model)
    resolution = resolve_qwen_image_lightning_profile(
        lora_adapters,
        model_entry,
        "inpaint",
        steps,
        true_cfg_scale,
    )
    resolved_entries = {adapter.lora_id: adapter.entry for adapter in resolution.adapters}
    lora_requested = bool(lora_adapters)

    logger.info("seed=%s", seed)
    base_seed = resolve_base_seed(seed)

    batch_id = make_batch_id()
    batch_output_dir = get_batch_output_dir(OUTPUT_DIR, batch_id)
    preview_path, preview_url = _preview_output(batch_id, batch_output_dir)

    logger.info(
        "Qwen-Image Inpaint: model=%s seed=%s steps=%s true_cfg_scale=%s "
        "size=%sx%s strength=%s num_images=%s padding_mask_crop=%s",
        model,
        base_seed,
        steps,
        true_cfg_scale,
        width,
        height,
        strength,
        num_images,
        padding_mask_crop,
    )

    filenames: list[str] = []
    pipe = None
    try:
        _raise_if_cancelled(runtime)
        _publish_loading_model(runtime, num_images)
        pipe = load_inpaint_pipeline(model)
        if resolution.lightning_profile is not None:
            pipe.scheduler = select_qwen_image_scheduler(resolution, pipe)
        else:
            pipe.scheduler = create_scheduler(scheduler, pipe)
        _apply_qwen_lora_adapters(
            pipe,
            lora_adapters,
            batch_output_dir=batch_output_dir,
            batch_id=batch_id,
            resolved_entries=resolved_entries,
            resolution=resolution,
        )

        for i in range(num_images):
            _raise_if_cancelled(runtime)
            current_seed = base_seed + i
            generator = torch.Generator(device="cpu").manual_seed(current_seed)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                call_kwargs: dict[str, object] = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "image": initial_image,
                    "mask_image": mask_image,
                    "strength": strength,
                    "width": width,
                    "height": height,
                    "num_inference_steps": resolution.steps,
                    "true_cfg_scale": resolution.true_cfg_scale,
                    "generator": generator,
                }
                if padding_mask_crop is not None:
                    call_kwargs["padding_mask_crop"] = padding_mask_crop
                preview_transform = None
                if padding_mask_crop is not None:
                    def _overlay_preview(current_pipe, preview):
                        crop = current_pipe.mask_processor.get_crop_region(
                            mask_image,
                            width,
                            height,
                            pad=padding_mask_crop,
                        )
                        return current_pipe.image_processor.apply_overlay(
                            mask_image,
                            initial_image,
                            preview,
                            crop,
                        )

                    preview_transform = _overlay_preview
                _install_step_callback(
                    call_kwargs,
                    runtime,
                    requested_steps=resolution.steps,
                    image_index=i,
                    total_images=num_images,
                    width=width,
                    height=height,
                    preview_path=preview_path,
                    preview_url=preview_url,
                    preview_transform=preview_transform,
                    live_preview=bool(params.get("live_preview", True)),
                )
                image = pipe(**call_kwargs).images[0]

            _raise_if_cancelled(runtime)
            relpath = save_generated_image(
                image, batch_output_dir, batch_id, current_seed, params,
                mode="inpaint", pipeline="qwen-image",
                remove_params=("initial_image", "mask_image"),
                size=image.size,
            )
            logger.info("Image %s saved to %s", i, relpath)
            filenames.append(relpath)
            _publish_image_complete(
                runtime,
                image_index=i,
                total_images=num_images,
                output_url=f"/outputs/{relpath}",
            )

            del image
            cleanup_memory()
    finally:
        _remove_preview(preview_path)
        _cleanup_qwen_lora_adapters(pipe, requested=lora_requested)
        release_pipeline(pipe, logger=logger)
        pipe = None

    return {"images": [f"/outputs/{name}" for name in filenames]}
