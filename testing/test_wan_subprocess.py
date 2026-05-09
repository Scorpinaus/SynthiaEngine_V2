import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from backend.wan import pipeline as wan_pipeline
from backend.wan import subprocess_runner
from backend.wan.subprocess_io import serialize_params_for_subprocess


class WanSubprocessTests(unittest.TestCase):
    def test_wan_bridge_uses_single_generation_gate(self):
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
                '{"ok": true, "result": ["batch_b1/out.mp4"]}',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with (
            patch("backend.wan.pipeline._WAN_SUBPROCESS_SEMAPHORE", FakeSemaphore()),
            patch("backend.wan.pipeline.subprocess.run", side_effect=fake_run),
        ):
            result = wan_pipeline.generate_text2video(params)

        self.assertEqual(result, ["batch_b1/out.mp4"])
        self.assertEqual(events, ["enter", "run", "exit"])

    def test_wan_text2video_bridge_invokes_child_and_reads_result(self):
        params = {"prompt": "test prompt", "seed": 123}

        def fake_run(cmd, cwd):
            input_path = Path(cmd[-2])
            output_path = Path(cmd[-1])
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["operation"], "text2video")
            self.assertEqual(payload["params"], params)
            output_path.write_text(
                '{"ok": true, "result": ["batch_b1/out.mp4"]}',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("backend.wan.pipeline.subprocess.run", side_effect=fake_run) as run_mock:
            result = wan_pipeline.generate_text2video(params)

        self.assertEqual(result, ["batch_b1/out.mp4"])
        command = run_mock.call_args.args[0]
        self.assertIn("-m", command)
        self.assertIn("backend.wan.subprocess_runner", command)

    def test_wan_bridge_serializes_pil_images_for_child_process(self):
        image = Image.new("RGB", (4, 3), "red")

        def fake_run(cmd, cwd):
            input_path = Path(cmd[-2])
            output_path = Path(cmd[-1])
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            marker = payload["params"]["image"]
            image_path = Path(marker["__wan_subprocess_image__"])
            self.assertTrue(image_path.exists())
            with Image.open(image_path) as saved:
                self.assertEqual(saved.size, (4, 3))
                self.assertEqual(saved.mode, "RGB")
            output_path.write_text(
                '{"ok": true, "result": ["batch_b1/i2v.mp4"]}',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("backend.wan.pipeline.subprocess.run", side_effect=fake_run):
            result = wan_pipeline.generate_image2video({"prompt": "test", "image": image})

        self.assertEqual(result, ["batch_b1/i2v.mp4"])

    def test_wan_bridge_serializes_conditioning_video_path_for_child_process(self):
        video_path = Path("inputs") / "conditioning.mp4"

        def fake_run(cmd, cwd):
            input_path = Path(cmd[-2])
            output_path = Path(cmd[-1])
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            marker = payload["params"]["conditioning_video"]
            self.assertEqual(marker["__wan_subprocess_path__"], str(video_path))
            output_path.write_text(
                '{"ok": true, "result": ["batch_b1/vace.mp4"]}',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("backend.wan.pipeline.subprocess.run", side_effect=fake_run):
            result = wan_pipeline.generate_text2video(
                {"prompt": "test", "conditioning_video": video_path}
            )

        self.assertEqual(result, ["batch_b1/vace.mp4"])

    def test_wan_subprocess_runner_dispatches_with_rehydrated_values(self):
        with tempfile.TemporaryDirectory(prefix="wan_test_") as tmpdir:
            tmp_path = Path(tmpdir)
            image = Image.new("RGB", (2, 2), "blue")
            video_path = Path("conditioning.mp4")
            input_path = tmp_path / "input.json"
            output_path = tmp_path / "output.json"
            input_payload = {
                "operation": "text2video",
                "params": serialize_params_for_subprocess(
                    {
                        "prompt": "test",
                        "reference_image": image,
                        "conditioning_video": video_path,
                    },
                    tmp_path,
                ),
            }
            input_path.write_text(json.dumps(input_payload), encoding="utf-8")

            with patch("backend.wan.subprocess_runner._dispatch_table") as dispatch_mock:
                captured = {}

                def fake_generate(params):
                    captured.update(params)
                    return ["batch_b2/out.mp4"]

                dispatch_mock.return_value = {"text2video": fake_generate}
                code = subprocess_runner.main([str(input_path), str(output_path)])

            self.assertEqual(code, 0)
            self.assertIsInstance(captured["reference_image"], Image.Image)
            self.assertEqual(captured["reference_image"].size, (2, 2))
            self.assertEqual(captured["conditioning_video"], video_path)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload, {"ok": True, "result": ["batch_b2/out.mp4"]})

    def test_wan_subprocess_runner_cleans_memory_after_success(self):
        with tempfile.TemporaryDirectory(prefix="wan_test_") as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "input.json"
            output_path = tmp_path / "output.json"
            input_path.write_text(
                json.dumps({"operation": "text2video", "params": {"prompt": "test"}}),
                encoding="utf-8",
            )

            with patch("backend.wan.subprocess_runner._dispatch_table") as dispatch_mock:
                with patch("backend.wan.subprocess_runner.cleanup_memory") as cleanup:
                    dispatch_mock.return_value = {"text2video": lambda _params: ["batch/out.mp4"]}
                    code = subprocess_runner.main([str(input_path), str(output_path)])

            self.assertEqual(code, 0)
            cleanup.assert_called_once()

    def test_wan_subprocess_runner_cleans_memory_after_failure(self):
        with tempfile.TemporaryDirectory(prefix="wan_test_") as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "input.json"
            output_path = tmp_path / "output.json"
            input_path.write_text(
                json.dumps({"operation": "text2video", "params": {"prompt": "test"}}),
                encoding="utf-8",
            )

            with patch("backend.wan.subprocess_runner._dispatch_table") as dispatch_mock:
                with patch("backend.wan.subprocess_runner.cleanup_memory") as cleanup:
                    dispatch_mock.return_value = {
                        "text2video": lambda _params: (_ for _ in ()).throw(RuntimeError("boom"))
                    }
                    code = subprocess_runner.main([str(input_path), str(output_path)])
                    payload = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(code, 1)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["error_type"], "RuntimeError")
            cleanup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
