from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import traceback
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol, TypedDict, TypeVar, cast

from PIL import Image

from backend.settings import REPOSITORY_ROOT

logger = logging.getLogger(__name__)

_VALUE_MARKER = "__syntha_subprocess_value__"


class SubprocessRequest(TypedDict):
    operation: str
    params: dict[str, Any]


class SubprocessSuccess(TypedDict):
    ok: Literal[True]
    result: Any


class SubprocessFailure(TypedDict):
    ok: Literal[False]
    error_type: str
    error: str
    traceback: str


class ProcessResult(Protocol):
    returncode: int


ProcessRunner = Callable[..., ProcessResult]


@dataclass(frozen=True)
class SubprocessTransport:
    family: str
    runner_module: str
    temp_prefix: str
    launch_gate: AbstractContextManager[object]


def serialize_params_for_subprocess(
    params: Mapping[str, object],
    temp_dir: Path,
) -> dict[str, Any]:
    """Serialize supported workflow values into the shared JSON protocol."""

    image_index = 0

    def _serialize(value: object, location: str) -> Any:
        nonlocal image_index
        if isinstance(value, Image.Image):
            image_path = temp_dir / f"image_{image_index}.png"
            image_index += 1
            value.save(image_path)
            return {
                _VALUE_MARKER: "image",
                "path": str(image_path),
                "mode": value.mode,
            }
        if isinstance(value, Path):
            return {_VALUE_MARKER: "path", "path": str(value)}
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, (list, tuple)):
            return [
                _serialize(item, f"{location}[{index}]")
                for index, item in enumerate(value)
            ]
        if isinstance(value, Mapping):
            return {
                str(key): _serialize(item, f"{location}.{key}")
                for key, item in value.items()
            }
        raise TypeError(
            f"Unsupported subprocess value at {location}: {type(value).__name__}."
        )

    return {
        str(key): _serialize(value, f"params.{key}")
        for key, value in params.items()
    }


def deserialize_params_from_subprocess(params: Mapping[str, Any]) -> dict[str, object]:
    """Rehydrate values written by :func:`serialize_params_for_subprocess`."""

    def _deserialize(value: Any) -> Any:
        if isinstance(value, dict):
            marker = value.get(_VALUE_MARKER)
            if marker == "image":
                image_path = value.get("path")
                if not isinstance(image_path, str):
                    raise ValueError("Serialized subprocess image is missing its path.")
                image = Image.open(image_path)
                image.load()
                mode = value.get("mode")
                if isinstance(mode, str) and image.mode != mode:
                    image = image.convert(mode)
                return image
            if marker == "path":
                path_value = value.get("path")
                if not isinstance(path_value, str):
                    raise ValueError("Serialized subprocess path is missing its value.")
                return Path(path_value)
            if marker is not None:
                raise ValueError(f"Unknown serialized subprocess value kind: {marker!r}.")
            return {str(key): _deserialize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_deserialize(item) for item in value]
        return value

    return {str(key): _deserialize(value) for key, value in params.items()}


def _serialize_result_for_subprocess(result: object, temp_dir: Path) -> Any:
    return serialize_params_for_subprocess({"result": result}, temp_dir)["result"]


def _deserialize_result_from_subprocess(result: Any) -> object:
    return deserialize_params_from_subprocess({"result": result})["result"]


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{description} contains malformed JSON at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object.")
    return cast(dict[str, Any], payload)


def _parse_request(path: Path) -> SubprocessRequest:
    payload = _read_json_object(path, description="Subprocess request")
    operation = payload.get("operation")
    params = payload.get("params")
    if not isinstance(operation, str) or not operation:
        raise ValueError("Subprocess request operation must be a non-empty string.")
    if not isinstance(params, dict):
        raise ValueError("Subprocess request params must be an object.")
    return {"operation": operation, "params": cast(dict[str, Any], params)}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, separators=(",", ": ")),
        encoding="utf-8",
    )


def run_subprocess(
    transport: SubprocessTransport,
    operation: str,
    params: Mapping[str, object],
    *,
    process_runner: ProcessRunner | None = None,
) -> object:
    """Run one family operation through the shared one-shot transport."""

    runner = process_runner or subprocess.run
    with tempfile.TemporaryDirectory(prefix=transport.temp_prefix) as tmpdir:
        temp_dir = Path(tmpdir)
        input_path = temp_dir / "input.json"
        output_path = temp_dir / "output.json"
        request: SubprocessRequest = {
            "operation": operation,
            "params": serialize_params_for_subprocess(params, temp_dir),
        }
        _write_json(input_path, request)

        command = [
            sys.executable,
            "-m",
            transport.runner_module,
            str(input_path),
            str(output_path),
        ]
        logger.info(
            "Starting %s subprocess operation=%s module=%s",
            transport.family,
            operation,
            transport.runner_module,
        )
        with transport.launch_gate:
            completed = runner(command, cwd=str(REPOSITORY_ROOT))

        if not output_path.exists():
            raise RuntimeError(
                f"{transport.family} subprocess failed: No subprocess result was written "
                f"(exit code {completed.returncode})."
            )

        try:
            payload = _read_json_object(
                output_path,
                description=f"{transport.family} subprocess result",
            )
        except ValueError as exc:
            raise RuntimeError(f"{transport.family} subprocess failed: {exc}") from exc

        ok = payload.get("ok")
        if ok is False:
            detail = payload.get("error")
            error_type = payload.get("error_type")
            if not isinstance(detail, str) or not detail:
                detail = "Unknown subprocess failure."
            if isinstance(error_type, str) and error_type:
                detail = f"{error_type}: {detail}"
            raise RuntimeError(f"{transport.family} subprocess failed: {detail}")
        if ok is not True or "result" not in payload:
            raise RuntimeError(
                f"{transport.family} subprocess failed: Invalid result envelope "
                f"(exit code {completed.returncode})."
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{transport.family} subprocess failed: Process exited with code "
                f"{completed.returncode} after reporting success."
            )

        logger.info(
            "Completed %s subprocess operation=%s exit_code=%s",
            transport.family,
            operation,
            completed.returncode,
        )
        return _deserialize_result_from_subprocess(payload["result"])


DispatchResult = TypeVar("DispatchResult")
Dispatch = Mapping[str, Callable[[dict[str, object]], DispatchResult]]


def run_subprocess_child(
    *,
    family: str,
    runner_module: str,
    dispatch: Dispatch[Any],
    cleanup: Callable[[], None],
    argv: list[str] | None = None,
) -> int:
    """Execute the shared child-side protocol around a family dispatch table."""

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print(
            f"Usage: python -m {runner_module} <input-json> <output-json>",
            file=sys.stderr,
        )
        return 2

    input_path = Path(args[0])
    output_path = Path(args[1])
    exit_code = 1
    try:
        request = _parse_request(input_path)
        operation = request["operation"]
        generate = dispatch.get(operation)
        if generate is None:
            raise ValueError(f"Unsupported {family} subprocess operation: {operation}")
        params = deserialize_params_from_subprocess(request["params"])
        result = generate(params)
        success: SubprocessSuccess = {
            "ok": True,
            "result": _serialize_result_for_subprocess(result, output_path.parent),
        }
        _write_json(output_path, success)
        exit_code = 0
    except Exception as exc:
        failure: SubprocessFailure = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(output_path, failure)
        print(failure["traceback"], file=sys.stderr)
    finally:
        try:
            cleanup()
        except Exception:
            print(f"{family} subprocess cleanup failed:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
    return exit_code


def normalize_path_list(result: object, *, family: str) -> list[str]:
    if not isinstance(result, list):
        raise RuntimeError(f"{family} subprocess returned an invalid result.")
    return [str(path) for path in result]


def normalize_image_result(
    result: object,
    *,
    family: str,
    include_runtime_profile: bool = False,
) -> dict[str, Any]:
    if not isinstance(result, dict) or not isinstance(result.get("images"), list):
        raise RuntimeError(f"{family} subprocess returned an invalid result.")
    normalized: dict[str, Any] = {
        "images": [str(path) for path in result["images"]],
    }
    runtime_profile = result.get("runtime_profile")
    if include_runtime_profile and isinstance(runtime_profile, dict):
        normalized["runtime_profile"] = runtime_profile
    return normalized
