import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.sd15 import animatediff_pipeline
from backend.sd15 import subprocess_runner


class Sd15AnimateDiffSubprocessTests(unittest.TestCase):
    def test_animatediff_bridge_uses_sd15_subprocess_operation(self):
        params = {"prompt": "test prompt", "seed": 123}

        with patch(
            "backend.sd15.animatediff_pipeline._run_sd15_subprocess",
            return_value=["batch_b1/out.mp4"],
        ) as run_subprocess:
            result = animatediff_pipeline.generate_videos_text2video(params)

        self.assertEqual(result, ["batch_b1/out.mp4"])
        run_subprocess.assert_called_once_with("animatediff_text2video", params)

    def test_sd15_runner_dispatches_animatediff_operation(self):
        with tempfile.TemporaryDirectory(prefix="sd15_animatediff_test_") as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "input.json"
            output_path = tmp_path / "output.json"
            input_path.write_text(
                json.dumps(
                    {
                        "operation": "animatediff_text2video",
                        "params": {"prompt": "test prompt"},
                    }
                ),
                encoding="utf-8",
            )

            with patch("backend.sd15.subprocess_runner._dispatch_table") as dispatch_mock:
                captured = {}

                def fake_generate(params):
                    captured.update(params)
                    return ["batch_b2/out.mp4"]

                dispatch_mock.return_value = {"animatediff_text2video": fake_generate}
                code = subprocess_runner.main([str(input_path), str(output_path)])

            self.assertEqual(code, 0)
            self.assertEqual(captured, {"prompt": "test prompt"})
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload, {"ok": True, "result": ["batch_b2/out.mp4"]})

    def test_sd15_runner_cleans_memory_after_success(self):
        with tempfile.TemporaryDirectory(prefix="sd15_animatediff_test_") as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "input.json"
            output_path = tmp_path / "output.json"
            input_path.write_text(
                json.dumps(
                    {
                        "operation": "animatediff_text2video",
                        "params": {"prompt": "test prompt"},
                    }
                ),
                encoding="utf-8",
            )

            with patch("backend.sd15.subprocess_runner._dispatch_table") as dispatch_mock:
                with patch("backend.sd15.subprocess_runner.cleanup_memory") as cleanup:
                    dispatch_mock.return_value = {
                        "animatediff_text2video": lambda _params: ["batch/out.mp4"]
                    }
                    code = subprocess_runner.main([str(input_path), str(output_path)])

            self.assertEqual(code, 0)
            cleanup.assert_called_once()

    def test_sd15_runner_cleans_memory_after_failure(self):
        with tempfile.TemporaryDirectory(prefix="sd15_animatediff_test_") as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "input.json"
            output_path = tmp_path / "output.json"
            input_path.write_text(
                json.dumps(
                    {
                        "operation": "animatediff_text2video",
                        "params": {"prompt": "test prompt"},
                    }
                ),
                encoding="utf-8",
            )

            with patch("backend.sd15.subprocess_runner._dispatch_table") as dispatch_mock:
                with patch("backend.sd15.subprocess_runner.cleanup_memory") as cleanup:
                    dispatch_mock.return_value = {
                        "animatediff_text2video": lambda _params: (
                            _ for _ in ()
                        ).throw(RuntimeError("boom"))
                    }
                    code = subprocess_runner.main([str(input_path), str(output_path)])
                    payload = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(code, 1)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["error_type"], "RuntimeError")
            cleanup.assert_called_once()

    def test_animatediff_bridge_invokes_child_runner_and_reads_result(self):
        params = {"prompt": "test prompt", "seed": 123}

        def fake_run(cmd, cwd):
            input_path = Path(cmd[-2])
            output_path = Path(cmd[-1])
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["operation"], "animatediff_text2video")
            self.assertEqual(payload["params"], params)
            output_path.write_text(
                '{"ok": true, "result": ["batch_b1/out.mp4"]}',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("backend.sd15.pipeline.subprocess.run", side_effect=fake_run) as run_mock:
            result = animatediff_pipeline.generate_videos_text2video(params)

        self.assertEqual(result, ["batch_b1/out.mp4"])
        command = run_mock.call_args.args[0]
        self.assertIn("-m", command)
        self.assertIn("backend.sd15.subprocess_runner", command)


if __name__ == "__main__":
    unittest.main()
