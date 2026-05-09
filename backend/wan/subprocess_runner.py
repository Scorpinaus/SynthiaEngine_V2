from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Callable

from backend.utilities.pipeline import cleanup_memory
from backend.wan.subprocess_io import deserialize_params_from_subprocess
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
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("Usage: python -m backend.wan.subprocess_runner <input-json> <output-json>", file=sys.stderr)
        return 2

    input_path = Path(args[0])
    output_path = Path(args[1])

    exit_code = 1
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
            raise ValueError(f"Unsupported WAN subprocess operation: {operation}")

        result = dispatch[operation](deserialize_params_from_subprocess(params))
        output_path.write_text(
            json.dumps({"ok": True, "result": result}),
            encoding="utf-8",
        )
        exit_code = 0
    except Exception as exc:
        result_payload = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        output_path.write_text(json.dumps(result_payload), encoding="utf-8")
        print(result_payload["traceback"], file=sys.stderr)
    finally:
        try:
            cleanup_memory()
        except Exception:
            print("WAN subprocess cleanup failed:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
