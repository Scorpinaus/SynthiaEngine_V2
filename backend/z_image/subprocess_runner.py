from __future__ import annotations

from typing import Callable

from backend.utilities.pipeline import cleanup_memory
from backend.utilities.subprocess_transport import run_subprocess_child
from backend.z_image.pipeline import (
        generate_img2img_in_process,
        generate_inpaint_in_process,
        generate_text2img_in_process,
    )


def _dispatch_table() -> dict[str, Callable[[dict[str, object]], dict[str, list[str]]]]:

    return {
        "text2img": generate_text2img_in_process,
        "img2img": generate_img2img_in_process,
        "inpaint": generate_inpaint_in_process,
    }


def main(argv: list[str] | None = None) -> int:
    return run_subprocess_child(
        family="Z-Image",
        runner_module="backend.z_image.subprocess_runner",
        dispatch=_dispatch_table(),
        cleanup=cleanup_memory,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
