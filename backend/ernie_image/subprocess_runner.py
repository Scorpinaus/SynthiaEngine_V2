from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from backend.ernie_image.pipeline import _generate_text2img_subprocess_child
from backend.utilities.pipeline import cleanup_memory


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("Usage: python -m backend.ernie_image.subprocess_runner <input-json> <output-json>", file=sys.stderr)
        return 2

    input_path = Path(args[0])
    output_path = Path(args[1])

    exit_code = 1
    try:
        params = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(params, dict):
            raise ValueError("Subprocess input JSON must be an object.")
        result = _generate_text2img_subprocess_child(params)
        payload = {"ok": True, "result": result}
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        exit_code = 0
    except Exception as exc:
        payload = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        print(payload["traceback"], file=sys.stderr)
    finally:
        try:
            cleanup_memory()
        except Exception:
            print("ERNIE-Image subprocess cleanup failed:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
