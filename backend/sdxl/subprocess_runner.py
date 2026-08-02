from __future__ import annotations

from typing import Callable

from backend.utilities.pipeline import cleanup_memory
from backend.utilities.subprocess_transport import run_subprocess_child
from backend.sdxl.pipeline import (
        generate_controlnet_text2img_in_process,
        generate_img2img_controlnet_in_process,
        generate_img2img_in_process,
        generate_inpaint_controlnet_in_process,
        generate_inpaint_in_process,
        generate_text2img_in_process,
    )

def _dispatch_table() -> dict[str, Callable[[dict[str, object]], dict[str, list[str]]]]:

    return {
        "text2img": generate_text2img_in_process,
        "controlnet_text2img": generate_controlnet_text2img_in_process,
        "img2img": generate_img2img_in_process,
        "img2img_controlnet": generate_img2img_controlnet_in_process,
        "inpaint": generate_inpaint_in_process,
        "inpaint_controlnet": generate_inpaint_controlnet_in_process,
    }


def main(argv: list[str] | None = None) -> int:
    return run_subprocess_child(
        family="SDXL",
        runner_module="backend.sdxl.subprocess_runner",
        dispatch=_dispatch_table(),
        cleanup=cleanup_memory,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
