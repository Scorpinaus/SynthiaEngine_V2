from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Callable

from backend.sd15.subprocess_io import deserialize_params_from_subprocess


def _dispatch_table() -> dict[str, Callable[[dict[str, object]], list[str]]]:
    from backend.sd15.pipeline import (
        generate_images_controlnet_in_process,
        generate_images_img2img_controlnet_in_process,
        generate_images_img2img_in_process,
        generate_images_in_process,
        generate_images_inpaint_controlnet_in_process,
        generate_images_inpaint_in_process,
    )

    return {
        "text2img": generate_images_in_process,
        "controlnet_text2img": generate_images_controlnet_in_process,
        "img2img": generate_images_img2img_in_process,
        "img2img_controlnet": generate_images_img2img_controlnet_in_process,
        "inpaint": generate_images_inpaint_in_process,
        "inpaint_controlnet": generate_images_inpaint_controlnet_in_process,
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("Usage: python -m backend.sd15.subprocess_runner <input-json> <output-json>", file=sys.stderr)
        return 2

    input_path = Path(args[0])
    output_path = Path(args[1])

    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Subprocess input JSON must be an object.")
        operation = payload.get("operation")
        params = payload.get("params")
        if not isinstance(operation, str):
            raise ValueError("Subprocess input operation must be a string.")
        if not isinstance(params, dict):
            raise ValueError("Subprocess input params must be an object.")

        dispatch = _dispatch_table()
        if operation not in dispatch:
            raise ValueError(f"Unsupported SD1.5 subprocess operation: {operation}")

        result = dispatch[operation](deserialize_params_from_subprocess(params))
        output_path.write_text(
            json.dumps({"ok": True, "result": result}),
            encoding="utf-8",
        )
        return 0
    except Exception as exc:
        result_payload = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        output_path.write_text(json.dumps(result_payload), encoding="utf-8")
        print(result_payload["traceback"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
