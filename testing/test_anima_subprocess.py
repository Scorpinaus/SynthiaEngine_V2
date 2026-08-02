import importlib
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


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

        class _Scheduler:
            @classmethod
            def from_config(cls, *_args, **_kwargs):
                return cls()

        class _DiffusionPipeline:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

        for name in (
            "DDIMScheduler",
            "DEISMultistepScheduler",
            "DPMSolverMultistepScheduler",
            "DPMSolverSinglestepScheduler",
            "EulerAncestralDiscreteScheduler",
            "EulerDiscreteScheduler",
            "FlowMatchEulerDiscreteScheduler",
            "FlowMatchHeunDiscreteScheduler",
            "HeunDiscreteScheduler",
            "KDPM2AncestralDiscreteScheduler",
            "KDPM2DiscreteScheduler",
            "LCMScheduler",
            "LMSDiscreteScheduler",
            "UniPCMultistepScheduler",
        ):
            setattr(diffusers, name, _Scheduler)
        diffusers.DiffusionPipeline = _DiffusionPipeline
        sys.modules["diffusers"] = diffusers


def _import_anima_pipeline():
    _ensure_lightweight_runtime_modules()
    return importlib.import_module("backend.anima.pipeline")


class AnimaSubprocessTests(unittest.TestCase):
    def test_anima_bridge_uses_single_generation_gate(self):
        anima_pipeline = _import_anima_pipeline()
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
            patch.object(anima_pipeline, "_ANIMA_SUBPROCESS_SEMAPHORE", FakeSemaphore()),
            patch("backend.utilities.subprocess_transport.subprocess.run", side_effect=fake_run),
        ):
            result = anima_pipeline.generate_text2img(params)

        self.assertEqual(result, {"images": ["/outputs/fake.png"]})
        self.assertEqual(events, ["enter", "run", "exit"])

    def test_anima_text2img_bridge_invokes_child_and_reads_result(self):
        anima_pipeline = _import_anima_pipeline()
        params = {"prompt": "test prompt", "seed": 123}

        def fake_run(cmd, cwd):
            input_path = Path(cmd[-2])
            output_path = Path(cmd[-1])
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            self.assertEqual(payload, {"operation": "text2img", "params": params})
            output_path.write_text(
                '{"ok": true, "result": {"images": ["/outputs/fake.png"]}}',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch(
            "backend.utilities.subprocess_transport.subprocess.run",
            side_effect=fake_run,
        ) as run_mock:
            result = anima_pipeline.generate_text2img(params)

        self.assertEqual(result, {"images": ["/outputs/fake.png"]})
        command = run_mock.call_args.args[0]
        self.assertIn("-m", command)
        self.assertIn("backend.anima.subprocess_runner", command)

    def test_anima_subprocess_runner_cleans_memory_after_success(self):
        _ensure_lightweight_runtime_modules()
        from backend.anima import subprocess_runner

        with patch("backend.anima.subprocess_runner._generate_text2img_subprocess_child") as generate:
            with patch("backend.anima.subprocess_runner.cleanup_memory") as cleanup:
                generate.return_value = {"images": ["/outputs/fake.png"]}
                with tempfile.TemporaryDirectory(prefix="anima_runner_test_") as tmpdir:
                    input_path = Path(tmpdir) / "input.json"
                    output_path = Path(tmpdir) / "output.json"
                    input_path.write_text(
                        '{"operation": "text2img", "params": {"prompt": "test"}}',
                        encoding="utf-8",
                    )

                    code = subprocess_runner.main([str(input_path), str(output_path)])

        self.assertEqual(code, 0)
        cleanup.assert_called_once()

    def test_anima_loader_uses_local_custom_pipeline(self):
        anima_pipeline = _import_anima_pipeline()
        captured = {}

        class FakePipeline:
            scheduler = types.SimpleNamespace(config={})
            vae = None

            @classmethod
            def from_pretrained(cls, source, **kwargs):
                captured["source"] = source
                captured.update(kwargs)
                return cls()

            def enable_sequential_cpu_offload(self):
                captured["offload"] = "sequential"

        with (
            patch.object(anima_pipeline, "_get_anima_pipeline_class", return_value=FakePipeline),
            patch.object(anima_pipeline, "list_model_entries", return_value=[]),
            patch.object(anima_pipeline, "cleanup_memory"),
        ):
            pipe = anima_pipeline.load_text2img_pipeline(None)

        self.assertIsInstance(pipe, FakePipeline)
        self.assertEqual(captured["source"], "CalamitousFelicitousness/Anima-Preview-3-sdnext-diffusers")
        self.assertTrue(captured["trust_remote_code"])
        self.assertEqual(captured["revision"], "main")
        self.assertEqual(captured["offload"], "sequential")


if __name__ == "__main__":
    unittest.main()
