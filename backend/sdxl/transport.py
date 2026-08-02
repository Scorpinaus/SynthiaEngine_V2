"""One-shot subprocess entrypoints for SDXL image operations."""

from backend.sdxl.runtime_common import *

def _run_sdxl_subprocess(operation: str, params: dict[str, object]) -> dict[str, list[str]]:
    result = run_subprocess(
        SubprocessTransport(
            family="SDXL",
            runner_module="backend.sdxl.subprocess_runner",
            temp_prefix="sdxl_",
            launch_gate=_SDXL_SUBPROCESS_SEMAPHORE,
        ),
        operation,
        params,
    )
    return normalize_image_result(result, family="SDXL")


def generate_controlnet_text2img(params: dict[str, object]) -> dict[str, list[str]]:
    return _run_sdxl_subprocess("controlnet_text2img", params)


def generate_img2img_controlnet(params: dict[str, object]) -> dict[str, list[str]]:
    return _run_sdxl_subprocess("img2img_controlnet", params)


def generate_text2img(payload: dict[str, object]) -> dict[str, list[str]]:
    return _run_sdxl_subprocess("text2img", payload)


def generate_img2img(params: dict[str, object]) -> dict[str, list[str]]:
    return _run_sdxl_subprocess("img2img", params)


def generate_inpaint(params: dict[str, object]) -> dict[str, list[str]]:
    return _run_sdxl_subprocess("inpaint", params)


def generate_inpaint_controlnet(params: dict[str, object]) -> dict[str, list[str]]:
    return _run_sdxl_subprocess("inpaint_controlnet", params)

