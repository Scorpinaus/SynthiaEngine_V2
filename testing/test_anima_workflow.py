import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.workflow import (
    AnimaText2ImgInputs,
    build_workflow_catalog,
)
from backend.workflow.assembly import _anima_text2img


class AnimaWorkflowTests(unittest.TestCase):
    def test_anima_text2img_defaults_match_contract(self):
        inputs = AnimaText2ImgInputs(prompt="test")

        self.assertEqual(inputs.steps, 35)
        self.assertEqual(inputs.guidance_scale, 4.5)
        self.assertEqual(inputs.width, 1024)
        self.assertEqual(inputs.height, 1024)
        self.assertEqual(inputs.scheduler, "flowmatch_euler")
        self.assertEqual(inputs.memory_preset, "sequential_offload")

    def test_anima_text2img_forwards_inputs_to_runtime(self):
        captured = {}

        def _fake_generate_text2img(payload):
            captured.update(payload)
            return {"images": ["/outputs/batch/out.png"]}

        fake_module = SimpleNamespace(generate_text2img=_fake_generate_text2img)
        with patch.dict("sys.modules", {"backend.anima.pipeline": fake_module}):
            result = _anima_text2img(
                {
                    "prompt": "test prompt",
                    "steps": 30,
                    "guidance_scale": 4.0,
                    "memory_preset": "model_offload",
                },
                _ctx=None,
            )

        self.assertEqual(result["images"], ["/outputs/batch/out.png"])
        self.assertEqual(captured["prompt"], "test prompt")
        self.assertEqual(captured["steps"], 30)
        self.assertEqual(captured["guidance_scale"], 4.0)
        self.assertEqual(captured["memory_preset"], "model_offload")

    def test_catalog_exposes_anima_text2img(self):
        catalog = build_workflow_catalog()

        self.assertIn("anima.text2img", catalog["tasks"])
        self.assertIn("anima", catalog["capabilities"])
        self.assertTrue(catalog["capabilities"]["anima"]["features"]["text2img"])
        self.assertFalse(catalog["capabilities"]["anima"]["features"]["img2img"])


if __name__ == "__main__":
    unittest.main()
