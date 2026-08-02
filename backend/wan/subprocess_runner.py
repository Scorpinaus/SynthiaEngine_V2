from __future__ import annotations

from typing import Callable

from backend.utilities.pipeline import cleanup_memory
from backend.utilities.subprocess_transport import run_subprocess_child
from backend.wan.pipeline import (
        generate_image2video_in_process,
        generate_text2video_in_process,
    )

def _dispatch_table() -> dict[str, Callable[[dict[str, object]], list[str]]]:

    return {
        "text2video": generate_text2video_in_process,
        "image2video": generate_image2video_in_process,
    }


def main(argv: list[str] | None = None) -> int:
    return run_subprocess_child(
        family="WAN",
        runner_module="backend.wan.subprocess_runner",
        dispatch=_dispatch_table(),
        cleanup=cleanup_memory,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
