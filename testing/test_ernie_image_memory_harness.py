import importlib.util
import json
from pathlib import Path


def load_harness_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "measure_ernie_image_memory.py"
    spec = importlib.util.spec_from_file_location("measure_ernie_image_memory", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_args_defaults_are_low_memory_safe():
    harness = load_harness_module()

    args = harness.parse_args([])

    assert args.width == 768
    assert args.height == 768
    assert args.steps == 8
    assert args.guidance_scale == 1.0
    assert args.num_images == 1
    assert args.memory_preset == "sequential_offload"
    assert args.use_pe is False
    assert args.load_pe is False
    assert args.runs == 1


def test_build_generation_params_forwards_runtime_controls():
    harness = load_harness_module()
    args = harness.parse_args(
        [
            "--prompt",
            "a compact test prompt",
            "--width",
            "640",
            "--height",
            "512",
            "--steps",
            "6",
            "--guidance-scale",
            "0.8",
            "--seed",
            "123",
            "--num-images",
            "1",
            "--memory-preset",
            "model_offload",
            "--use-pe",
            "--load-pe",
            "--model",
            "ERNIE-Image-Turbo",
        ]
    )

    params = harness.build_generation_params(args)

    assert params == {
        "prompt": "a compact test prompt",
        "steps": 6,
        "guidance_scale": 0.8,
        "width": 640,
        "height": 512,
        "seed": 123,
        "model": "ERNIE-Image-Turbo",
        "num_images": 1,
        "use_pe": True,
        "load_pe": True,
        "memory_preset": "model_offload",
    }


def test_run_measurement_records_success_and_writes_json(tmp_path, monkeypatch):
    harness = load_harness_module()
    output_json = tmp_path / "measurement.json"
    args = harness.parse_args(["--runs", "2", "--output-json", str(output_json)])

    monkeypatch.setattr(harness, "get_process_rss_mb", lambda: 512.0)
    monkeypatch.setattr(
        harness,
        "get_cuda_memory_stats",
        lambda: {
            "cuda_available": False,
            "cuda_max_allocated_mb": None,
            "cuda_max_reserved_mb": None,
            "cuda_allocated_after_mb": None,
            "cuda_reserved_after_mb": None,
        },
    )

    calls = []

    def fake_generate(params):
        calls.append(dict(params))
        return {"images": ["/outputs/fake.png"]}

    result = harness.run_measurement(args, generate_fn=fake_generate)

    assert result["summary"]["runs"] == 2
    assert result["summary"]["successes"] == 2
    assert result["summary"]["failures"] == 0
    assert result["runs"][0]["status"] == "success"
    assert result["runs"][0]["images"] == ["/outputs/fake.png"]
    assert len(calls) == 2

    saved = json.loads(output_json.read_text(encoding="utf-8"))
    assert saved["summary"] == result["summary"]


def test_run_measurement_records_generator_failure(monkeypatch):
    harness = load_harness_module()
    args = harness.parse_args(["--runs", "1"])

    monkeypatch.setattr(harness, "get_process_rss_mb", lambda: None)
    monkeypatch.setattr(
        harness,
        "get_cuda_memory_stats",
        lambda: {
            "cuda_available": False,
            "cuda_max_allocated_mb": None,
            "cuda_max_reserved_mb": None,
            "cuda_allocated_after_mb": None,
            "cuda_reserved_after_mb": None,
        },
    )

    def failing_generate(params):
        raise RuntimeError("synthetic OOM")

    result = harness.run_measurement(args, generate_fn=failing_generate)

    assert result["summary"]["runs"] == 1
    assert result["summary"]["successes"] == 0
    assert result["summary"]["failures"] == 1
    assert result["runs"][0]["status"] == "error"
    assert result["runs"][0]["error_type"] == "RuntimeError"
    assert "synthetic OOM" in result["runs"][0]["error"]
