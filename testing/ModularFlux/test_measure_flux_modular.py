from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from PIL import Image

import measure_flux_modular as harness


class FakePipe:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {"images": [Image.new("RGB", (8, 8), "green")]}


def test_parse_args_defaults_are_low_memory_safe():
    args = harness.parse_args([])

    assert args.case == "flux-text2img"
    assert args.width == 768
    assert args.height == 768
    assert args.steps == 8
    assert args.num_images == 1
    assert args.offload == "auto"
    assert args.low_memory_sequential_images is True
    assert args.low_memory_transformer_buffers is True
    assert args.decode_chunk_size == 1


def test_resolve_cases_supports_pipeline_all():
    args = harness.parse_args(["--case", "all", "--pipeline", "all"])

    cases = harness.resolve_cases(args)

    assert [case.name for case in cases] == [
        "flux-text2img",
        "flux-img2img",
        "flux-embeds2img",
        "flux-img2img-embeds",
        "kontext-text2img",
        "kontext-image",
        "kontext-embeds2img",
        "kontext-image-embeds",
    ]


def test_resolve_cases_supports_short_aliases():
    flux_args = harness.parse_args(["--case", "img2img", "--pipeline", "flux"])
    kontext_args = harness.parse_args(["--case", "image", "--pipeline", "kontext"])

    assert [case.name for case in harness.resolve_cases(flux_args)] == ["flux-img2img"]
    assert [case.name for case in harness.resolve_cases(kontext_args)] == ["kontext-image"]


def test_build_case_kwargs_for_flux_img2img_uses_synthetic_image():
    args = harness.parse_args(["--case", "flux-img2img", "--pipeline", "flux", "--seed", "99"])
    case = harness.CASES["flux-img2img"]

    kwargs, stats = harness.build_case_kwargs(args, case, FakePipe(), run_seed=99)

    assert kwargs["prompt"] == args.prompt
    assert kwargs["strength"] == args.strength
    assert kwargs["low_memory_transformer_buffers"] is True
    assert kwargs["decode_chunk_size"] == 1
    assert isinstance(kwargs["image"], Image.Image)
    assert stats["prepare_seconds"] >= 0
    assert stats["embed_seconds"] is None


def test_default_pipeline_loader_uses_direct_local_constructor(monkeypatch):
    calls = {}

    class FakeFluxModularPipeline:
        def __init__(self, **kwargs):
            calls["constructor_kwargs"] = kwargs

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise AssertionError("from_pretrained should not be used for the local modular harness")

        def load_components(self, **kwargs):
            calls["load_kwargs"] = kwargs

    fake_module = types.ModuleType("custom_pipelines.FluxModular")
    fake_module.FluxModularPipeline = FakeFluxModularPipeline
    fake_module.FluxKontextModularPipeline = FakeFluxModularPipeline

    def fake_enable_low_memory_flux_modular(_pipe, **kwargs):
        calls["offload_kwargs"] = kwargs
        return "auto"

    fake_module.enable_low_memory_flux_modular = fake_enable_low_memory_flux_modular

    args = harness.parse_args(
        [
            "--model",
            r"D:\diffusion\diffusers\FLUX.1-dev",
            "--device",
            "cpu",
            "--torch-dtype",
            "float32",
            "--variant",
            "fp16",
            "--local-files-only",
        ]
    )
    monkeypatch.setitem(sys.modules, "custom_pipelines.FluxModular", fake_module)
    monkeypatch.setattr(harness, "reset_cuda_memory_stats", lambda: None)
    monkeypatch.setattr(harness, "synchronize_cuda", lambda: None)
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

    pipe, load_stats = harness.default_pipeline_loader("flux", args)

    assert isinstance(pipe, FakeFluxModularPipeline)
    assert calls["constructor_kwargs"]["pretrained_model_name_or_path"] == r"D:\diffusion\diffusers\FLUX.1-dev"
    assert calls["constructor_kwargs"]["local_files_only"] is True
    assert "variant" not in calls["constructor_kwargs"]
    assert calls["load_kwargs"]["variant"] == "fp16"
    assert calls["load_kwargs"]["local_files_only"] is True
    assert load_stats["offload_mode"] == "auto"


def test_run_measurement_records_success_and_writes_json(tmp_path, monkeypatch):
    output_json = tmp_path / "flux_modular.json"
    output_dir = tmp_path / "images"
    args = harness.parse_args(
        [
            "--case",
            "flux-text2img",
            "--runs",
            "2",
            "--output-json",
            str(output_json),
            "--output-dir",
            str(output_dir),
        ]
    )
    fake_pipe = FakePipe()

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
    monkeypatch.setattr(harness, "reset_cuda_memory_stats", lambda: None)
    monkeypatch.setattr(harness, "synchronize_cuda", lambda: None)

    def fake_loader(kind, _args):
        return fake_pipe, {"pipeline": kind, "load_seconds": 0.01, "offload_mode": "fake"}

    result = harness.run_measurement(args, pipeline_loader=fake_loader)

    assert result["summary"]["runs"] == 2
    assert result["summary"]["successes"] == 2
    assert result["summary"]["failures"] == 0
    assert len(fake_pipe.calls) == 2
    assert fake_pipe.calls[0]["low_memory_sequential_images"] is True
    assert fake_pipe.calls[0]["low_memory_transformer_buffers"] is True
    assert result["runs"][0]["image_paths"]

    saved = json.loads(output_json.read_text(encoding="utf-8"))
    assert saved["summary"] == result["summary"]
    assert Path(result["runs"][0]["image_paths"][0]).exists()


def test_run_measurement_records_inference_failure(monkeypatch, tmp_path):
    args = harness.parse_args(["--case", "flux-text2img", "--output-dir", str(tmp_path)])

    class FailingPipe:
        def __call__(self, **_kwargs):
            raise RuntimeError("synthetic OOM")

    monkeypatch.setattr(harness, "reset_cuda_memory_stats", lambda: None)
    monkeypatch.setattr(harness, "synchronize_cuda", lambda: None)

    def fake_loader(kind, _args):
        return FailingPipe(), {"pipeline": kind, "load_seconds": 0.01, "offload_mode": "fake"}

    result = harness.run_measurement(args, pipeline_loader=fake_loader)

    assert result["summary"]["runs"] == 1
    assert result["summary"]["successes"] == 0
    assert result["summary"]["failures"] == 1
    assert result["runs"][0]["status"] == "error"
    assert result["runs"][0]["phase"] == "inference"
    assert result["runs"][0]["error_type"] == "RuntimeError"
    assert "synthetic OOM" in result["runs"][0]["error"]
