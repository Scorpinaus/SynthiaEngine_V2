import unittest
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.ernie_image.pipeline import load_text2img_pipeline, run_text2img_subprocess
from backend.registries.model import ModelRegistryEntry
from backend.workflow import (
    ErnieImageText2ImgInputs,
    _ernie_image_text2img,
    build_workflow_catalog,
)


class ErnieImageWorkflowTests(unittest.TestCase):
    def test_ernie_image_text2img_defaults_are_safe_for_12gb_windows(self):
        inputs = ErnieImageText2ImgInputs(prompt="test")

        self.assertEqual(inputs.steps, 8)
        self.assertEqual(inputs.guidance_scale, 1.0)
        self.assertEqual(inputs.width, 768)
        self.assertEqual(inputs.height, 768)
        self.assertEqual(inputs.num_images, 1)
        self.assertFalse(inputs.use_pe)
        self.assertFalse(inputs.load_pe)
        self.assertEqual(inputs.memory_preset, "sequential_offload")
        self.assertEqual(inputs.execution_mode, "subprocess")

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
                    "execution_mode": "subprocess",
                },
                _ctx=None,
            )

        self.assertEqual(result["images"], ["/outputs/batch/out.png"])
        self.assertEqual(captured["prompt"], "test prompt")
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
        self.assertEqual(captured["execution_mode"], "subprocess")

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

        defaults = catalog["tasks"]["ernie-image.text2img"]["input_defaults"]
        self.assertEqual(defaults["steps"], 8)
        self.assertEqual(defaults["guidance_scale"], 1.0)
        self.assertEqual(defaults["width"], 768)
        self.assertEqual(defaults["height"], 768)
        self.assertFalse(defaults["use_pe"])
        self.assertFalse(defaults["load_pe"])
        self.assertEqual(defaults["memory_preset"], "sequential_offload")
        self.assertEqual(defaults["execution_mode"], "subprocess")

    def test_ernie_image_subprocess_bridge_invokes_child_and_reads_result(self):
        params = {"prompt": "test", "execution_mode": "subprocess"}

        def fake_run(cmd, capture_output, text, cwd):
            input_path = Path(cmd[-2])
            output_path = Path(cmd[-1])
            self.assertEqual(
                __import__("json").loads(input_path.read_text(encoding="utf-8")),
                {"prompt": "test", "execution_mode": "in_process"},
            )
            output_path.write_text(
                '{"ok": true, "result": {"images": ["/outputs/fake.png"]}}',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("backend.ernie_image.pipeline.subprocess.run", side_effect=fake_run) as run_mock:
            result = run_text2img_subprocess(params)

        self.assertEqual(result, {"images": ["/outputs/fake.png"]})
        command = run_mock.call_args.args[0]
        self.assertIn("-m", command)
        self.assertIn("backend.ernie_image.subprocess_runner", command)

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
