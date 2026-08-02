"""One-shot subprocess entrypoints for SD1.5 image operations."""

from backend.sd15.runtime_common import *

def _run_sd15_subprocess(operation: str, params: dict[str, object]) -> list[str]:
    result = run_subprocess(
        SubprocessTransport(
            family="SD1.5",
            runner_module="backend.sd15.subprocess_runner",
            temp_prefix="sd15_",
            launch_gate=_SD15_SUBPROCESS_SEMAPHORE,
        ),
        operation,
        params,
    )
    return normalize_path_list(result, family="SD1.5")


def generate_images_controlnet(params: dict[str, object]) -> list[str]:
    return _run_sd15_subprocess("controlnet_text2img", params)


def generate_images(params: dict[str, object]) -> list[str]:
    return _run_sd15_subprocess("text2img", params)


def generate_images_img2img(params: dict[str, object]) -> list[str]:
    return _run_sd15_subprocess("img2img", params)


def generate_images_img2img_controlnet(params: dict[str, object]) -> list[str]:
    return _run_sd15_subprocess("img2img_controlnet", params)


def generate_images_inpaint(params: dict[str, object]) -> list[str]:
    return _run_sd15_subprocess("inpaint", params)


def generate_images_inpaint_controlnet(params: dict[str, object]) -> list[str]:
    return _run_sd15_subprocess("inpaint_controlnet", params)

