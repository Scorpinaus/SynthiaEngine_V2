import unittest
import json
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.ernie_image import subprocess_runner
from backend.ernie_image.pipeline import (
    _generate_text2img_subprocess_child,
    generate_text2img,
    load_text2img_pipeline,
    run_text2img_subprocess,
)
from backend.registries.model import ModelRegistryEntry
from backend.workflow import (
    ErnieImageText2ImgInputs,
    build_workflow_catalog,
)
from backend.workflow.assembly import _ernie_image_text2img


class ErnieImageWorkflowTests(unittest.TestCase):
    def test_ernie_image_text2img_defaults_are_safe_for_12gb_windows(self):
        inputs = ErnieImageText2ImgInputs(prompt="test")

        self.assertEqual(inputs.steps, 8)
        self.assertEqual(inputs.guidance_scale, 1.0)
        self.assertEqual(inputs.width, 768)
        self.assertEqual(inputs.height, 768)
        self.assertEqual(inputs.negative_prompt, "")
        self.assertEqual(inputs.num_images, 1)
        self.assertFalse(inputs.use_pe)
        self.assertFalse(inputs.load_pe)
        self.assertEqual(inputs.memory_preset, "sequential_offload")
        self.assertIsNone(inputs.lora_adapters)

    def test_ernie_image_text2img_accepts_lora_adapters(self):
        inputs = ErnieImageText2ImgInputs(
            prompt="test",
            lora_adapters=[{"lora_id": 101, "strength": 0.8}],
        )
        self.assertEqual(inputs.lora_adapters, [{"lora_id": 101, "strength": 0.8}])

    def test_ernie_image_text2img_forwards_runtime_controls(self):
        captured = {}

        def _fake_generate_text2img(payload):
            captured.update(payload)
            return {"images": ["/outputs/batch/out.png"]}

        fake_module = SimpleNamespace(generate_text2img=_fake_generate_text2img)
        with patch.dict("sys.modules", {"backend.ernie_image.pipeline": fake_module}):
            result = _ernie_image_text2img(
                {
                    "prompt": "test prompt",
                    "negative_prompt": "avoid blur",
                    "steps": 8,
                    "guidance_scale": 1.0,
                    "width": 768,
                    "height": 768,
                    "seed": 123,
                    "model": "ERNIE-Image-Turbo",
                    "num_images": 1,
                    "use_pe": False,
                    "load_pe": False,
                    "memory_preset": "sequential_offload",
                },
                _ctx=None,
            )

        self.assertEqual(result["images"], ["/outputs/batch/out.png"])
        self.assertEqual(captured["prompt"], "test prompt")
        self.assertEqual(captured["negative_prompt"], "avoid blur")
        self.assertEqual(captured["steps"], 8)
        self.assertEqual(captured["guidance_scale"], 1.0)
        self.assertEqual(captured["width"], 768)
        self.assertEqual(captured["height"], 768)
        self.assertEqual(captured["seed"], 123)
        self.assertEqual(captured["model"], "ERNIE-Image-Turbo")
        self.assertEqual(captured["num_images"], 1)
        self.assertFalse(captured["use_pe"])
        self.assertFalse(captured["load_pe"])
        self.assertEqual(captured["memory_preset"], "sequential_offload")

    def test_ernie_image_text2img_forwards_lora_adapters(self):
        captured = {}

        def _fake_generate_text2img(payload):
            captured.update(payload)
            return {"images": ["/outputs/batch/out.png"]}

        fake_module = SimpleNamespace(generate_text2img=_fake_generate_text2img)
        with patch.dict("sys.modules", {"backend.ernie_image.pipeline": fake_module}):
            result = _ernie_image_text2img(
                {
                    "prompt": "test prompt",
                    "lora_adapters": [{"lora_id": 101, "strength": 0.8}],
                },
                _ctx=None,
            )

        self.assertEqual(result["images"], ["/outputs/batch/out.png"])
        self.assertIn("lora_adapters", captured)
        self.assertEqual(captured["lora_adapters"], [{"lora_id": 101, "strength": 0.8}])

    def test_ernie_image_rejects_use_pe_without_loading_pe(self):
        with self.assertRaisesRegex(ValueError, "use_pe=true requires load_pe=true"):
            ErnieImageText2ImgInputs(prompt="test", use_pe=True, load_pe=False)

    def test_ernie_image_task_is_exposed_in_catalog(self):
        catalog = build_workflow_catalog()

        self.assertIn("ernie-image.text2img", catalog["tasks"])
        self.assertIn("ernie-image", catalog["capabilities"])
        self.assertIn(
            "ernie-image.text2img",
            catalog["capabilities"]["ernie-image"]["task_types"],
        )
        self.assertTrue(catalog["capabilities"]["ernie-image"]["features"]["text2img"])
        self.assertFalse(catalog["capabilities"]["ernie-image"]["features"]["img2img"])
        self.assertTrue(catalog["capabilities"]["ernie-image"]["features"]["lora_adapters"])

        defaults = catalog["tasks"]["ernie-image.text2img"]["input_defaults"]
        self.assertEqual(defaults["steps"], 8)
        self.assertEqual(defaults["guidance_scale"], 1.0)
        self.assertEqual(defaults["width"], 768)
        self.assertEqual(defaults["height"], 768)
        self.assertEqual(defaults["negative_prompt"], "")
        self.assertFalse(defaults["use_pe"])
        self.assertFalse(defaults["load_pe"])
        self.assertEqual(defaults["memory_preset"], "sequential_offload")
        self.assertIsNone(defaults["lora_adapters"])
        self.assertNotIn("execution_mode", defaults)

    def test_ernie_image_subprocess_bridge_invokes_child_and_reads_result(self):
        params = {"prompt": "test"}

        def fake_run(cmd, cwd):
            input_path = Path(cmd[-2])
            output_path = Path(cmd[-1])
            self.assertEqual(
                __import__("json").loads(input_path.read_text(encoding="utf-8")),
                {"operation": "text2img", "params": {"prompt": "test"}},
            )
            output_path.write_text(
                '{"ok": true, "result": {"images": ["/outputs/fake.png"]}}',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch(
            "backend.utilities.subprocess_transport.subprocess.run",
            side_effect=fake_run,
        ) as run_mock:
            result = run_text2img_subprocess(params)

        self.assertEqual(result, {"images": ["/outputs/fake.png"]})
        command = run_mock.call_args.args[0]
        self.assertIn("-m", command)
        self.assertIn("backend.ernie_image.subprocess_runner", command)

    def test_ernie_image_subprocess_child_forwards_negative_prompt(self):
        captured = {}

        class FakeImage:
            def save(self, filename, pnginfo=None):
                captured["saved_to"] = filename
                captured["pnginfo"] = pnginfo

        class FakePipe:
            def __call__(self, **kwargs):
                captured["call_kwargs"] = kwargs
                return SimpleNamespace(images=[FakeImage()])

        with tempfile.TemporaryDirectory(prefix="ernie_output_test_") as tmpdir:
            with (
                patch("backend.ernie_image.pipeline.load_text2img_pipeline", return_value=FakePipe()),
                patch("backend.ernie_image.pipeline.make_batch_id", return_value="batch_test"),
                patch("backend.ernie_image.pipeline.get_batch_output_dir", return_value=Path(tmpdir)),
                patch("backend.ernie_image.pipeline.build_png_metadata", return_value=None),
                patch(
                    "backend.ernie_image.pipeline.build_batch_output_relpath",
                    side_effect=lambda batch, name: f"{batch}/{name}",
                ),
                patch("backend.ernie_image.pipeline.cleanup_memory"),
            ):
                result = _generate_text2img_subprocess_child(
                    {
                        "prompt": "test prompt",
                        "negative_prompt": "avoid blur",
                        "steps": 8,
                        "guidance_scale": 2.0,
                        "width": 768,
                        "height": 768,
                        "seed": 123,
                        "num_images": 1,
                        "use_pe": False,
                        "load_pe": False,
                    }
                )

        self.assertEqual(captured["call_kwargs"]["prompt"], "test prompt")
        self.assertEqual(captured["call_kwargs"]["negative_prompt"], "avoid blur")
        self.assertEqual(result["images"], ["/outputs/batch_test/batch_test_123.png"])

    def test_ernie_image_rejects_direct_in_process_execution_mode(self):
        with self.assertRaisesRegex(ValueError, "supports only subprocess execution"):
            generate_text2img({"prompt": "test", "execution_mode": "in_process"})

    def test_ernie_image_subprocess_bridge_uses_single_generation_gate(self):
        events = []
        params = {"prompt": "test"}

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
            patch("backend.ernie_image.pipeline._ERNIE_IMAGE_SUBPROCESS_SEMAPHORE", FakeSemaphore()),
            patch("backend.utilities.subprocess_transport.subprocess.run", side_effect=fake_run),
        ):
            result = run_text2img_subprocess(params)

        self.assertEqual(result, {"images": ["/outputs/fake.png"]})
        self.assertEqual(events, ["enter", "run", "exit"])

    def test_ernie_image_subprocess_runner_cleans_memory_after_success(self):
        with patch("backend.ernie_image.subprocess_runner._generate_text2img_subprocess_child") as generate:
            with patch("backend.ernie_image.subprocess_runner.cleanup_memory") as cleanup:
                generate.return_value = {"images": ["/outputs/fake.png"]}
                with patch("sys.stderr"):
                    with tempfile.TemporaryDirectory(prefix="ernie_runner_test_") as tmpdir:
                        input_path = Path(tmpdir) / "input.json"
                        output_path = Path(tmpdir) / "output.json"
                        input_path.write_text(
                            '{"operation": "text2img", "params": {"prompt": "test"}}',
                            encoding="utf-8",
                        )

                        code = subprocess_runner.main([str(input_path), str(output_path)])

        self.assertEqual(code, 0)
        cleanup.assert_called_once()

    def test_ernie_image_subprocess_runner_cleans_memory_after_failure(self):
        with patch("backend.ernie_image.subprocess_runner._generate_text2img_subprocess_child") as generate:
            with patch("backend.ernie_image.subprocess_runner.cleanup_memory") as cleanup:
                generate.side_effect = RuntimeError("synthetic failure")
                with patch("sys.stderr"):
                    with tempfile.TemporaryDirectory(prefix="ernie_runner_test_") as tmpdir:
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

    def test_ernie_image_pipeline_skips_pe_components_when_load_pe_false(self):
        captured = {}
        entry = ModelRegistryEntry(
            name="ERNIE-Image-Turbo",
            family="ernie-image",
            model_type="diffusers",
            location_type="local",
            model_id=13,
            version="turbo",
            link=r"D:\diffusion\diffusers\Ernie-Image-Turbo",
        )
        fake_pipe = SimpleNamespace(
            enable_sequential_cpu_offload=lambda: None,
            vae=None,
        )

        def fake_from_pretrained(source, **kwargs):
            captured["source"] = source
            captured.update(kwargs)
            return fake_pipe

        with (
            patch("backend.ernie_image.pipeline._get_ernie_model_entry", return_value=entry),
            patch("backend.ernie_image.pipeline.ErnieImagePipeline.from_pretrained", side_effect=fake_from_pretrained),
            patch("backend.ernie_image.pipeline.cleanup_memory"),
        ):
            result = load_text2img_pipeline(
                "ERNIE-Image-Turbo",
                memory_preset="sequential_offload",
                load_pe=False,
            )

        self.assertIs(result, fake_pipe)
        self.assertEqual(captured["source"], r"D:\diffusion\diffusers\Ernie-Image-Turbo")
        self.assertIsNone(captured["pe"])
        self.assertIsNone(captured["pe_tokenizer"])

    def test_ernie_image_pipeline_keeps_pe_components_when_load_pe_true(self):
        captured = {}
        entry = ModelRegistryEntry(
            name="ERNIE-Image-Turbo",
            family="ernie-image",
            model_type="diffusers",
            location_type="local",
            model_id=13,
            version="turbo",
            link=r"D:\diffusion\diffusers\Ernie-Image-Turbo",
        )
        fake_pipe = SimpleNamespace(
            enable_sequential_cpu_offload=lambda: None,
            vae=None,
        )

        def fake_from_pretrained(source, **kwargs):
            captured.update(kwargs)
            return fake_pipe

        with (
            patch("backend.ernie_image.pipeline._get_ernie_model_entry", return_value=entry),
            patch("backend.ernie_image.pipeline.ErnieImagePipeline.from_pretrained", side_effect=fake_from_pretrained),
            patch("backend.ernie_image.pipeline.cleanup_memory"),
        ):
            load_text2img_pipeline(
                "ERNIE-Image-Turbo",
                memory_preset="sequential_offload",
                load_pe=True,
            )

        self.assertNotIn("pe", captured)
        self.assertNotIn("pe_tokenizer", captured)


if __name__ == "__main__":
    unittest.main()
