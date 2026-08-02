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

from backend.utilities.subprocess_transport import serialize_params_for_subprocess


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
        diffusers.QwenImagePipeline = type("QwenImagePipeline", (), {})
        diffusers.QwenImageImg2ImgPipeline = type("QwenImageImg2ImgPipeline", (), {})
        diffusers.QwenImageInpaintPipeline = type("QwenImageInpaintPipeline", (), {})
        sys.modules["diffusers"] = diffusers


def _import_qwen_image_pipeline():
    _ensure_lightweight_runtime_modules()
    return importlib.import_module("backend.qwen_image.pipeline")


class QwenImageSubprocessTests(unittest.TestCase):
    def test_qwen_image_bridge_uses_single_generation_gate(self):
        qwen_image_pipeline = _import_qwen_image_pipeline()
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
            patch.object(qwen_image_pipeline, "_QWEN_IMAGE_SUBPROCESS_SEMAPHORE", FakeSemaphore()),
            patch("backend.utilities.subprocess_transport.subprocess.run", side_effect=fake_run),
        ):
            result = qwen_image_pipeline.generate_text2img(params)

        self.assertEqual(result, {"images": ["/outputs/fake.png"]})
        self.assertEqual(events, ["enter", "run", "exit"])

    def test_qwen_image_text2img_bridge_invokes_child_and_reads_result(self):
        qwen_image_pipeline = _import_qwen_image_pipeline()
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

        with patch(
            "backend.utilities.subprocess_transport.subprocess.run",
            side_effect=fake_run,
        ) as run_mock:
            result = qwen_image_pipeline.generate_text2img(params)

        self.assertEqual(result, {"images": ["/outputs/fake.png"]})
        command = run_mock.call_args.args[0]
        self.assertIn("-m", command)
        self.assertIn("backend.qwen_image.subprocess_runner", command)

    def test_qwen_image_bridge_serializes_pil_images_for_child_process(self):
        qwen_image_pipeline = _import_qwen_image_pipeline()
        image = Image.new("RGB", (4, 3), "red")

        def fake_run(cmd, cwd):
            input_path = Path(cmd[-2])
            output_path = Path(cmd[-1])
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            marker = payload["params"]["initial_image"]
            self.assertEqual(marker["__syntha_subprocess_value__"], "image")
            image_path = Path(marker["path"])
            self.assertTrue(image_path.exists())
            with Image.open(image_path) as saved:
                self.assertEqual(saved.size, (4, 3))
                self.assertEqual(saved.mode, "RGB")
            output_path.write_text(
                '{"ok": true, "result": {"images": ["/outputs/fake.png"]}}',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch(
            "backend.utilities.subprocess_transport.subprocess.run",
            side_effect=fake_run,
        ):
            result = qwen_image_pipeline.generate_img2img({"initial_image": image})

        self.assertEqual(result, {"images": ["/outputs/fake.png"]})

    def test_qwen_image_subprocess_runner_dispatches_with_rehydrated_images(self):
        _ensure_lightweight_runtime_modules()
        from backend.qwen_image import subprocess_runner

        with tempfile.TemporaryDirectory(prefix="qwen_image_test_") as tmpdir:
            tmp_path = Path(tmpdir)
            image = Image.new("RGB", (2, 2), "blue")
            input_path = tmp_path / "input.json"
            output_path = tmp_path / "output.json"
            input_payload = {
                "operation": "img2img",
                "params": serialize_params_for_subprocess({"initial_image": image}, tmp_path),
            }
            input_path.write_text(json.dumps(input_payload), encoding="utf-8")

            with patch("backend.qwen_image.subprocess_runner._dispatch_table") as dispatch_mock:
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

    def test_qwen_image_subprocess_runner_cleans_memory_after_success(self):
        _ensure_lightweight_runtime_modules()
        from backend.qwen_image import subprocess_runner

        with patch("backend.qwen_image.subprocess_runner._dispatch_table") as dispatch_mock:
            with patch("backend.qwen_image.subprocess_runner.cleanup_memory") as cleanup:
                dispatch_mock.return_value = {
                    "text2img": lambda _params: {"images": ["/outputs/fake.png"]}
                }
                with tempfile.TemporaryDirectory(prefix="qwen_image_runner_test_") as tmpdir:
                    input_path = Path(tmpdir) / "input.json"
                    output_path = Path(tmpdir) / "output.json"
                    input_path.write_text(
                        '{"operation": "text2img", "params": {"prompt": "test"}}',
                        encoding="utf-8",
                    )

                    code = subprocess_runner.main([str(input_path), str(output_path)])

        self.assertEqual(code, 0)
        cleanup.assert_called_once()

    def test_qwen_image_subprocess_runner_cleans_memory_after_failure(self):
        _ensure_lightweight_runtime_modules()
        from backend.qwen_image import subprocess_runner

        with patch("backend.qwen_image.subprocess_runner._dispatch_table") as dispatch_mock:
            with patch("backend.qwen_image.subprocess_runner.cleanup_memory") as cleanup:
                def fake_generate(_params):
                    raise RuntimeError("synthetic failure")

                dispatch_mock.return_value = {"text2img": fake_generate}
                with patch("sys.stderr"):
                    with tempfile.TemporaryDirectory(prefix="qwen_image_runner_test_") as tmpdir:
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
