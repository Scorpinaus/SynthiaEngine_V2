from __future__ import annotations

from backend.ernie_image.pipeline import _generate_text2img_subprocess_child
from backend.utilities.pipeline import cleanup_memory
from backend.utilities.subprocess_transport import run_subprocess_child


def main(argv: list[str] | None = None) -> int:
    return run_subprocess_child(
        family="ERNIE-Image",
        runner_module="backend.ernie_image.subprocess_runner",
        dispatch={"text2img": _generate_text2img_subprocess_child},
        cleanup=cleanup_memory,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
