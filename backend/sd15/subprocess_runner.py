from __future__ import annotations

from typing import Callable

from backend.utilities.pipeline import cleanup_memory
from backend.utilities.subprocess_transport import run_subprocess_child
from backend.sd15.animatediff_pipeline import generate_videos_text2video_in_process
from backend.sd15.pipeline import (
        generate_images_controlnet_in_process,
        generate_images_img2img_controlnet_in_process,
        generate_images_img2img_in_process,
        generate_images_in_process,
        generate_images_inpaint_controlnet_in_process,
        generate_images_inpaint_in_process,
    )

def _dispatch_table() -> dict[str, Callable[[dict[str, object]], list[str]]]:

    return {
        "text2img": generate_images_in_process,
        "controlnet_text2img": generate_images_controlnet_in_process,
        "img2img": generate_images_img2img_in_process,
        "img2img_controlnet": generate_images_img2img_controlnet_in_process,
        "inpaint": generate_images_inpaint_in_process,
        "inpaint_controlnet": generate_images_inpaint_controlnet_in_process,
        "animatediff_text2video": generate_videos_text2video_in_process,
    }


def main(argv: list[str] | None = None) -> int:
    return run_subprocess_child(
        family="SD1.5",
        runner_module="backend.sd15.subprocess_runner",
        dispatch=_dispatch_table(),
        cleanup=cleanup_memory,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
