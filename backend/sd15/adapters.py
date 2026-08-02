"""Task-scoped LoRA and IP-Adapter policy for SD1.5 runtimes."""

from backend.sd15.runtime_common import *

def _apply_lora_adapters(
    pipe,
    lora_adapters: list[object] | None,
    *,
    validate: bool = False,
) -> list[str]:
    """
    Apply requested LoRA adapters to a pipeline.

    Returns:
        A list of adapter names actually loaded into the pipeline.
    """
    adapter_names, _ = apply_lora_adapters_with_validation(
        pipe,
        lora_adapters,
        expected_family="sd15",
        validate=validate,
    )
    return adapter_names


def _apply_lcm_lora(pipe) -> str:
    """Load the hard-coded SD1.5 LCM LoRA adapter."""
    logger.info("Loading SD1.5 LCM LoRA adapter: %s", _LCM_LORA_MODEL_ID)
    pipe.load_lora_weights(_LCM_LORA_MODEL_ID, adapter_name=_LCM_LORA_ADAPTER_NAME)
    return _LCM_LORA_ADAPTER_NAME


def _cleanup_lora_adapters(pipe, adapter_names: list[str]) -> None:
    """Best-effort cleanup for both pipeline-level and component-level LoRA adapters."""
    if not adapter_names:
        return
    logger.info("Cleaning up %s LoRA adapter(s): %s", len(adapter_names), adapter_names)

    if hasattr(pipe, "unload_lora_weights"):
        try:
            logger.debug("Attempting pipeline-level LoRA unload via unload_lora_weights().")
            pipe.unload_lora_weights()
            logger.debug("Pipeline-level LoRA unload completed.")
        except Exception:
            logger.exception("Failed to unload pipeline LoRA weights cleanly.")

    for component_name in ("unet", "text_encoder", "text_encoder_2", "transformer"):
        component = getattr(pipe, component_name, None)
        if component is None or not hasattr(component, "delete_adapters"):
            continue
        try:
            logger.debug("Attempting adapter deletion on component '%s'.", component_name)
            component.delete_adapters(adapter_names)
            logger.debug("Adapter deletion succeeded on component '%s'.", component_name)
        except Exception:
            logger.debug(
                "Skipping component LoRA adapter cleanup for %s; delete_adapters failed.",
                component_name,
                exc_info=True,
            )


def _metadata_without_runtime_images(params: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in params.items()
        if key not in {"ip_adapter_image", "ip_adapter_mask_image"}
    }


def _build_ip_adapter_kwargs(
    *,
    enabled: bool,
    image_embeds: list[torch.Tensor] | None,
    masks: list[torch.Tensor] | None,
) -> dict[str, object]:
    if not enabled:
        return {}

    kwargs: dict[str, object] = {"ip_adapter_image_embeds": image_embeds}
    if masks is not None:
        kwargs["cross_attention_kwargs"] = {"ip_adapter_masks": masks}
    return kwargs
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

