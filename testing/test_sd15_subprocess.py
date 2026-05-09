import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from backend.sd15 import pipeline as sd15_pipeline
from backend.sd15 import subprocess_runner
from backend.sd15.subprocess_io import serialize_params_for_subprocess


class Sd15SubprocessTests(unittest.TestCase):
    def test_sd15_bridge_uses_single_generation_gate(self):
        events = []
        params = {"prompt": "test prompt"}

        class FakeSemaphore:
            def __enter__(self):
                events.append("enter")

            def __exit__(self, exc_type, exc, traceback):
                events.append("exit")

        def fake_run(cmd, capture_output, text, cwd):
            events.append("run")
            output_path = Path(cmd[-1])
            output_path.write_text(
                '{"ok": true, "result": ["batch_b1/out.png"]}',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with (
            patch("backend.sd15.pipeline._SD15_SUBPROCESS_SEMAPHORE", FakeSemaphore()),
            patch("backend.sd15.pipeline.subprocess.run", side_effect=fake_run),
        ):
            result = sd15_pipeline.generate_images(params)

        self.assertEqual(result, ["batch_b1/out.png"])
        self.assertEqual(events, ["enter", "run", "exit"])

    def test_sd15_text2img_bridge_invokes_child_and_reads_result(self):
        params = {"prompt": "test prompt", "seed": 123}

        def fake_run(cmd, capture_output, text, cwd):
            input_path = Path(cmd[-2])
            output_path = Path(cmd[-1])
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["operation"], "text2img")
            self.assertEqual(payload["params"], params)
            output_path.write_text(
                '{"ok": true, "result": ["batch_b1/out.png"]}',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("backend.sd15.pipeline.subprocess.run", side_effect=fake_run) as run_mock:
            result = sd15_pipeline.generate_images(params)

        self.assertEqual(result, ["batch_b1/out.png"])
        command = run_mock.call_args.args[0]
        self.assertIn("-m", command)
        self.assertIn("backend.sd15.subprocess_runner", command)

    def test_sd15_bridge_serializes_pil_images_for_child_process(self):
        image = Image.new("RGB", (4, 3), "red")

        def fake_run(cmd, capture_output, text, cwd):
            input_path = Path(cmd[-2])
            output_path = Path(cmd[-1])
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            marker = payload["params"]["initial_image"]
            image_path = Path(marker["__sd15_subprocess_image__"])
            self.assertTrue(image_path.exists())
            with Image.open(image_path) as saved:
                self.assertEqual(saved.size, (4, 3))
                self.assertEqual(saved.mode, "RGB")
            output_path.write_text(
                '{"ok": true, "result": ["batch_b1/img2img.png"]}',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("backend.sd15.pipeline.subprocess.run", side_effect=fake_run):
            result = sd15_pipeline.generate_images_img2img({"initial_image": image})

        self.assertEqual(result, ["batch_b1/img2img.png"])

    def test_sd15_subprocess_runner_dispatches_with_rehydrated_images(self):
        with tempfile.TemporaryDirectory(prefix="sd15_test_") as tmpdir:
            tmp_path = Path(tmpdir)
            image = Image.new("RGB", (2, 2), "blue")
            input_path = tmp_path / "input.json"
            output_path = tmp_path / "output.json"
            input_payload = {
                "operation": "img2img",
                "params": serialize_params_for_subprocess({"initial_image": image}, tmp_path),
            }
            input_path.write_text(json.dumps(input_payload), encoding="utf-8")

            with patch("backend.sd15.subprocess_runner._dispatch_table") as dispatch_mock:
                captured = {}

                def fake_generate(params):
                    captured.update(params)
                    return ["batch_b2/out.png"]

                dispatch_mock.return_value = {"img2img": fake_generate}
                code = subprocess_runner.main([str(input_path), str(output_path)])

            self.assertEqual(code, 0)
            self.assertIsInstance(captured["initial_image"], Image.Image)
            self.assertEqual(captured["initial_image"].size, (2, 2))
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload, {"ok": True, "result": ["batch_b2/out.png"]})


if __name__ == "__main__":
    unittest.main()
