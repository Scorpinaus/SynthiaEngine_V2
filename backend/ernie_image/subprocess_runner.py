from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from backend.ernie_image.pipeline import generate_text2img_in_process


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("Usage: python -m backend.ernie_image.subprocess_runner <input-json> <output-json>", file=sys.stderr)
        return 2

    input_path = Path(args[0])
    output_path = Path(args[1])

    try:
        params = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(params, dict):
            raise ValueError("Subprocess input JSON must be an object.")
        params["execution_mode"] = "in_process"
        result = generate_text2img_in_process(params)
        payload = {"ok": True, "result": result}
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return 0
    except Exception as exc:
        payload = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        print(payload["traceback"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
