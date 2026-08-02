from __future__ import annotations

import json
import subprocess
from contextlib import nullcontext
from pathlib import Path

import pytest
from PIL import Image

from backend.settings import REPOSITORY_ROOT
from backend.utilities.subprocess_transport import (
    SubprocessTransport,
    deserialize_params_from_subprocess,
    run_subprocess,
    run_subprocess_child,
    serialize_params_for_subprocess,
)


def _transport() -> SubprocessTransport:
    return SubprocessTransport(
        family="Test-Family",
        runner_module="backend.test_family.subprocess_runner",
        temp_prefix="test_family_",
        launch_gate=nullcontext(),
    )


def test_shared_serializer_round_trips_images_paths_primitives_and_profiles(tmp_path):
    image = Image.new("RGBA", (3, 2), (255, 0, 0, 128))
    params = {
        "image": image,
        "conditioning_path": Path("inputs") / "clip.mp4",
        "values": (None, True, 7, 2.5, "text"),
        "runtime_profile": {"duration_ms": 42.0, "cache_hit": False},
    }

    serialized = serialize_params_for_subprocess(params, tmp_path)
    restored = deserialize_params_from_subprocess(serialized)

    assert serialized["image"]["__syntha_subprocess_value__"] == "image"
    assert serialized["conditioning_path"]["__syntha_subprocess_value__"] == "path"
    assert isinstance(restored["image"], Image.Image)
    assert restored["image"].size == (3, 2)
    assert restored["image"].mode == "RGBA"
    assert restored["conditioning_path"] == Path("inputs") / "clip.mp4"
    assert restored["values"] == [None, True, 7, 2.5, "text"]
    assert restored["runtime_profile"] == {"duration_ms": 42.0, "cache_hit": False}


def test_parent_reports_missing_result_and_cleans_temporary_artifacts():
    captured_temp_dirs: list[Path] = []

    def fake_run(command, cwd):
        assert Path(cwd) == REPOSITORY_ROOT
        assert command[1:3] == ["-m", "backend.test_family.subprocess_runner"]
        captured_temp_dirs.append(Path(command[-2]).parent)
        return subprocess.CompletedProcess(command, 9)

    with pytest.raises(
        RuntimeError,
        match=r"Test-Family subprocess failed: No subprocess result was written \(exit code 9\)",
    ):
        run_subprocess(
            _transport(),
            "text2img",
            {"image": Image.new("RGB", (2, 2), "blue")},
            process_runner=fake_run,
        )

    assert captured_temp_dirs
    assert not captured_temp_dirs[0].exists()


@pytest.mark.parametrize(
    ("result_text", "returncode", "message"),
    [
        (
            "{not-json",
            1,
            "Test-Family subprocess result contains malformed JSON at line 1, column 2",
        ),
        (
            '{"ok": true}',
            0,
            "Test-Family subprocess failed: Invalid result envelope",
        ),
        (
            '{"ok": true, "result": []}',
            7,
            "Test-Family subprocess failed: Process exited with code 7 after reporting success",
        ),
    ],
)
def test_parent_reports_malformed_invalid_and_crashed_results(
    result_text,
    returncode,
    message,
):
    def fake_run(command, cwd):
        Path(command[-1]).write_text(result_text, encoding="utf-8")
        return subprocess.CompletedProcess(command, returncode)

    with pytest.raises(RuntimeError, match=message):
        run_subprocess(
            _transport(),
            "text2img",
            {"prompt": "test"},
            process_runner=fake_run,
        )


def test_child_writes_typed_error_and_always_cleans_up(tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(
        json.dumps({"operation": "text2img", "params": {"prompt": "test"}}),
        encoding="utf-8",
    )
    cleanup_calls: list[str] = []

    def fail(_params):
        raise ValueError("bad prompt")

    exit_code = run_subprocess_child(
        family="Test-Family",
        runner_module="backend.test_family.subprocess_runner",
        dispatch={"text2img": fail},
        cleanup=lambda: cleanup_calls.append("cleanup"),
        argv=[str(input_path), str(output_path)],
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error_type"] == "ValueError"
    assert payload["error"] == "bad prompt"
    assert "ValueError: bad prompt" in payload["traceback"]
    assert cleanup_calls == ["cleanup"]


def test_family_protocol_copies_are_removed():
    repository_root = Path(__file__).resolve().parents[1]
    families = ("sd15", "sdxl", "flux", "qwen_image", "z_image", "wan")

    for family in families:
        assert not (repository_root / "backend" / family / "subprocess_io.py").exists()

    for family in (*families, "ernie_image", "anima"):
        pipeline_source = (repository_root / "backend" / family / "pipeline.py").read_text(
            encoding="utf-8"
        )
        runner_source = (
            repository_root / "backend" / family / "subprocess_runner.py"
        ).read_text(encoding="utf-8")
        assert "TemporaryDirectory" not in pipeline_source
        assert "subprocess.run" not in pipeline_source
        assert "json.loads" not in runner_source
        assert "run_subprocess_child" in runner_source
