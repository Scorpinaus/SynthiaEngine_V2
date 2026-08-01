from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from backend.flux.subprocess_io import serialize_params_for_subprocess


def _ensure_lightweight_runtime_modules():
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")

        def inference_mode():
            def _decorate(func):
                return func

            return _decorate

        torch.inference_mode = inference_mode
        torch.bfloat16 = object()
        torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
        sys.modules["torch"] = torch

    if "diffusers" not in sys.modules:
        diffusers = types.ModuleType("diffusers")
        diffusers.FluxPipeline = type("FluxPipeline", (), {})
        diffusers.FluxImg2ImgPipeline = type("FluxImg2ImgPipeline", (), {})
        diffusers.FluxInpaintPipeline = type("FluxInpaintPipeline", (), {})
        sys.modules["diffusers"] = diffusers

    custom_pipelines = sys.modules.setdefault(
        "custom_pipelines",
        types.ModuleType("custom_pipelines"),
    )
    flux_pkg = sys.modules.setdefault(
        "custom_pipelines.Flux",
        types.ModuleType("custom_pipelines.Flux"),
    )
    custom_flux_module = sys.modules.setdefault(
        "custom_pipelines.Flux.pipeline_flux",
        types.ModuleType("custom_pipelines.Flux.pipeline_flux"),
    )
    custom_flux_module.FluxPipeline = type("CustomFluxPipeline", (), {})
    custom_pipelines.Flux = flux_pkg
    flux_pkg.pipeline_flux = custom_flux_module


def _import_flux_pipeline():
    _ensure_lightweight_runtime_modules()
    return importlib.import_module("backend.flux.pipeline")


class FluxSubprocessTests(unittest.TestCase):
    def test_flux_bridge_uses_single_generation_gate(self):
        flux_pipeline = _import_flux_pipeline()
        events = []
        params = {"prompt": "test prompt"}

        class FakeSemaphore:
            def __enter__(self):
                events.append("enter")

            def __exit__(self, exc_type, exc, traceback):
                events.append("exit")

        def fake_run(cmd, cwd):
            events.append("run")
            output_path = Path(cmd[-1])
            output_path.write_text(
                '{"ok": true, "result": {"images": ["/outputs/fake.png"]}}',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with (
            patch.object(flux_pipeline, "_FLUX_SUBPROCESS_SEMAPHORE", FakeSemaphore()),
            patch.object(flux_pipeline.subprocess, "run", side_effect=fake_run),
        ):
            result = flux_pipeline.generate_text2img(params)

        self.assertEqual(result, {"images": ["/outputs/fake.png"]})
        self.assertEqual(events, ["enter", "run", "exit"])

    def test_flux_text2img_bridge_invokes_child_and_reads_result(self):
        flux_pipeline = _import_flux_pipeline()
        params = {"prompt": "test prompt", "seed": 123}

        def fake_run(cmd, cwd):
            input_path = Path(cmd[-2])
            output_path = Path(cmd[-1])
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["operation"], "text2img")
            self.assertEqual(payload["params"], params)
            output_path.write_text(
                '{"ok": true, "result": {"images": ["/outputs/fake.png"]}}',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch.object(flux_pipeline.subprocess, "run", side_effect=fake_run) as run_mock:
            result = flux_pipeline.generate_text2img(params)

        self.assertEqual(result, {"images": ["/outputs/fake.png"]})
        command = run_mock.call_args.args[0]
        self.assertIn("-m", command)
        self.assertIn("backend.flux.subprocess_runner", command)

    def test_flux_bridge_propagates_typed_child_failure(self):
        flux_pipeline = _import_flux_pipeline()

        def fake_run(cmd, cwd):
            output_path = Path(cmd[-1])
            output_path.write_text(
                '{"ok": false, "error_type": "ValueError", "error": "bad input"}',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

        with patch.object(flux_pipeline.subprocess, "run", side_effect=fake_run):
            with self.assertRaisesRegex(
                RuntimeError,
                "Flux subprocess failed: ValueError: bad input",
            ):
                flux_pipeline.generate_text2img({"prompt": "test prompt"})

    def test_flux_bridge_serializes_pil_images_for_child_process(self):
        flux_pipeline = _import_flux_pipeline()
        image = Image.new("RGB", (4, 3), "red")

        def fake_run(cmd, cwd):
            input_path = Path(cmd[-2])
            output_path = Path(cmd[-1])
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            marker = payload["params"]["initial_image"]
            image_path = Path(marker["__flux_subprocess_image__"])
            self.assertTrue(image_path.exists())
            with Image.open(image_path) as saved:
                self.assertEqual(saved.size, (4, 3))
                self.assertEqual(saved.mode, "RGB")
            output_path.write_text(
                '{"ok": true, "result": {"images": ["/outputs/fake.png"]}}',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch.object(flux_pipeline.subprocess, "run", side_effect=fake_run):
            result = flux_pipeline.generate_img2img({"initial_image": image})

        self.assertEqual(result, {"images": ["/outputs/fake.png"]})

    def test_flux_subprocess_runner_dispatches_with_rehydrated_images(self):
        _ensure_lightweight_runtime_modules()
        from backend.flux import subprocess_runner

        with tempfile.TemporaryDirectory(prefix="flux_test_") as tmpdir:
            tmp_path = Path(tmpdir)
            image = Image.new("RGB", (2, 2), "blue")
            input_path = tmp_path / "input.json"
            output_path = tmp_path / "output.json"
            input_payload = {
                "operation": "img2img",
                "params": serialize_params_for_subprocess({"initial_image": image}, tmp_path),
            }
            input_path.write_text(json.dumps(input_payload), encoding="utf-8")

            with patch("backend.flux.subprocess_runner._dispatch_table") as dispatch_mock:
                captured = {}

                def fake_generate(params):
                    captured.update(params)
                    return {"images": ["/outputs/fake.png"]}

                dispatch_mock.return_value = {"img2img": fake_generate}
                code = subprocess_runner.main([str(input_path), str(output_path)])
                payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertIsInstance(captured["initial_image"], Image.Image)
        self.assertEqual(captured["initial_image"].size, (2, 2))
        self.assertEqual(payload, {"ok": True, "result": {"images": ["/outputs/fake.png"]}})

    def test_flux_subprocess_runner_cleans_memory_after_success(self):
        _ensure_lightweight_runtime_modules()
        from backend.flux import subprocess_runner

        with patch("backend.flux.subprocess_runner._dispatch_table") as dispatch_mock:
            with patch("backend.flux.subprocess_runner.cleanup_memory") as cleanup:
                dispatch_mock.return_value = {
                    "text2img": lambda _params: {"images": ["/outputs/fake.png"]}
                }
                with tempfile.TemporaryDirectory(prefix="flux_runner_test_") as tmpdir:
                    input_path = Path(tmpdir) / "input.json"
                    output_path = Path(tmpdir) / "output.json"
                    input_path.write_text(
                        '{"operation": "text2img", "params": {"prompt": "test"}}',
                        encoding="utf-8",
                    )

                    code = subprocess_runner.main([str(input_path), str(output_path)])

        self.assertEqual(code, 0)
        cleanup.assert_called_once()

    def test_flux_subprocess_runner_cleans_memory_after_failure(self):
        _ensure_lightweight_runtime_modules()
        from backend.flux import subprocess_runner

        with patch("backend.flux.subprocess_runner._dispatch_table") as dispatch_mock:
            with patch("backend.flux.subprocess_runner.cleanup_memory") as cleanup:
                def fake_generate(_params):
                    raise RuntimeError("synthetic failure")

                dispatch_mock.return_value = {"text2img": fake_generate}
                with patch("sys.stderr"):
                    with tempfile.TemporaryDirectory(prefix="flux_runner_test_") as tmpdir:
                        input_path = Path(tmpdir) / "input.json"
                        output_path = Path(tmpdir) / "output.json"
                        input_path.write_text(
                            '{"operation": "text2img", "params": {"prompt": "test"}}',
                            encoding="utf-8",
                        )

                        code = subprocess_runner.main([str(input_path), str(output_path)])
                        payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(code, 1)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_type"], "RuntimeError")
        cleanup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
