import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from backend.sdxl import pipeline as sdxl_pipeline
from backend.sdxl import subprocess_runner
from backend.sdxl.subprocess_io import serialize_params_for_subprocess


class SdxlSubprocessTests(unittest.TestCase):
    def test_sdxl_bridge_uses_single_generation_gate(self):
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
                '{"ok": true, "result": {"images": ["/outputs/batch_b1/out.png"]}}',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with (
            patch("backend.sdxl.pipeline._SDXL_SUBPROCESS_SEMAPHORE", FakeSemaphore()),
            patch("backend.sdxl.pipeline.subprocess.run", side_effect=fake_run),
        ):
            result = sdxl_pipeline.generate_text2img(params)

        self.assertEqual(result, {"images": ["/outputs/batch_b1/out.png"]})
        self.assertEqual(events, ["enter", "run", "exit"])

    def test_sdxl_text2img_bridge_invokes_child_and_reads_result(self):
        params = {"prompt": "test prompt", "seed": 123}

        def fake_run(cmd, cwd):
            input_path = Path(cmd[-2])
            output_path = Path(cmd[-1])
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["operation"], "text2img")
            self.assertEqual(payload["params"], params)
            output_path.write_text(
                '{"ok": true, "result": {"images": ["/outputs/batch_b1/out.png"]}}',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("backend.sdxl.pipeline.subprocess.run", side_effect=fake_run) as run_mock:
            result = sdxl_pipeline.generate_text2img(params)

        self.assertEqual(result, {"images": ["/outputs/batch_b1/out.png"]})
        command = run_mock.call_args.args[0]
        self.assertIn("-m", command)
        self.assertIn("backend.sdxl.subprocess_runner", command)

    def test_sdxl_bridge_serializes_pil_images_for_child_process(self):
        image = Image.new("RGB", (4, 3), "red")

        def fake_run(cmd, cwd):
            input_path = Path(cmd[-2])
            output_path = Path(cmd[-1])
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            marker = payload["params"]["initial_image"]
            image_path = Path(marker["__sdxl_subprocess_image__"])
            self.assertTrue(image_path.exists())
            with Image.open(image_path) as saved:
                self.assertEqual(saved.size, (4, 3))
                self.assertEqual(saved.mode, "RGB")
            output_path.write_text(
                '{"ok": true, "result": {"images": ["/outputs/batch_b1/img2img.png"]}}',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("backend.sdxl.pipeline.subprocess.run", side_effect=fake_run):
            result = sdxl_pipeline.generate_img2img({"initial_image": image})

        self.assertEqual(result, {"images": ["/outputs/batch_b1/img2img.png"]})

    def test_sdxl_subprocess_runner_dispatches_with_rehydrated_images(self):
        with tempfile.TemporaryDirectory(prefix="sdxl_test_") as tmpdir:
            tmp_path = Path(tmpdir)
            image = Image.new("RGB", (2, 2), "blue")
            input_path = tmp_path / "input.json"
            output_path = tmp_path / "output.json"
            input_payload = {
                "operation": "img2img",
                "params": serialize_params_for_subprocess({"initial_image": image}, tmp_path),
            }
            input_path.write_text(json.dumps(input_payload), encoding="utf-8")

            with (
                patch("backend.sdxl.subprocess_runner.cleanup_memory"),
                patch("backend.sdxl.subprocess_runner._dispatch_table") as dispatch_mock,
            ):
                captured = {}

                def fake_generate(params):
                    captured.update(params)
                    return {"images": ["/outputs/batch_b2/out.png"]}

                dispatch_mock.return_value = {"img2img": fake_generate}
                code = subprocess_runner.main([str(input_path), str(output_path)])

            self.assertEqual(code, 0)
            self.assertIsInstance(captured["initial_image"], Image.Image)
            self.assertEqual(captured["initial_image"].size, (2, 2))
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload, {"ok": True, "result": {"images": ["/outputs/batch_b2/out.png"]}})


if __name__ == "__main__":
    unittest.main()
