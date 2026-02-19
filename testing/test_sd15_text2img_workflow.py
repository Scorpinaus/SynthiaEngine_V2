import unittest
from unittest.mock import patch

from backend.workflow import _sd15_text2img


class Sd15Text2ImgWorkflowPlumbingTests(unittest.TestCase):
    def test_sd15_text2img_passes_expected_generation_params(self):
        captured = {}

        def _fake_generate_images(params):
            captured.update(params)
            return ["batch/out.png"]

        with patch("backend.workflow.make_batch_id", return_value="batch123"):
            with patch("backend.workflow.generate_images", side_effect=_fake_generate_images):
                result = _sd15_text2img(
                    {
                        "prompt": "test prompt",
                        "negative_prompt": "bad",
                        "steps": 25,
                        "cfg": 8.0,
                        "width": 768,
                        "height": 512,
                        "seed": 123,
                        "scheduler": "euler",
                        "model": "stable-diffusion-v1-5",
                        "num_images": 2,
                        "clip_skip": 2,
                        "lora_adapters": [{"lora_id": 101, "strength": 0.75}],
                        "hires_enabled": True,
                        "hires_scale": 1.5,
                        "weighting_policy": "a1111-like",
                        "lora_scale": 0.9,
                    },
                    _ctx=None,
                )

        self.assertEqual(result["batch_id"], "batch123")
        self.assertEqual(result["images"], ["/outputs/batch/out.png"])
        self.assertEqual(captured["prompt"], "test prompt")
        self.assertEqual(captured["negative_prompt"], "bad")
        self.assertEqual(captured["steps"], 25)
        self.assertEqual(captured["cfg"], 8.0)
        self.assertEqual(captured["width"], 768)
        self.assertEqual(captured["height"], 512)
        self.assertEqual(captured["seed"], 123)
        self.assertEqual(captured["scheduler"], "euler")
        self.assertEqual(captured["model"], "stable-diffusion-v1-5")
        self.assertEqual(captured["num_images"], 2)
        self.assertEqual(captured["clip_skip"], 2)
        self.assertEqual(captured["lora_adapters"], [{"lora_id": 101, "strength": 0.75}])
        self.assertTrue(captured["hires_enabled"])
        self.assertEqual(captured["hires_scale"], 1.5)
        self.assertEqual(captured["weighting_policy"], "a1111-like")
        self.assertEqual(captured["batch_id"], "batch123")
        self.assertEqual(captured["lora_scale"], 0.9)

    def test_sd15_text2img_accepts_contract_lora_and_hires_fields(self):
        captured = {}

        def _fake_generate_images(params):
            captured.update(params)
            return ["batch/out.png"]

        with patch("backend.workflow.make_batch_id", return_value="batch123"):
            with patch("backend.workflow.generate_images", side_effect=_fake_generate_images):
                result = _sd15_text2img(
                    {
                        "prompt": "test prompt",
                        "Lora": {
                            "loraStatus": True,
                            "adapters": [{"lora_id": 303, "strength": 0.6}],
                        },
                        "hires": {
                            "hiresEnabled": True,
                            "hires_scale": 1.8,
                        },
                    },
                    _ctx=None,
                )

        self.assertEqual(result["batch_id"], "batch123")
        self.assertEqual(captured["lora_adapters"], [{"lora_id": 303, "strength": 0.6}])
        self.assertTrue(captured["hires_enabled"])
        self.assertEqual(captured["hires_scale"], 1.8)

    def test_sd15_text2img_prefers_unified_lora_contract(self):
        captured = {}

        def _fake_generate_images(params):
            captured.update(params)
            return ["batch/out.png"]

        with patch("backend.workflow.make_batch_id", return_value="batch123"):
            with patch("backend.workflow.generate_images", side_effect=_fake_generate_images):
                _sd15_text2img(
                    {
                        "prompt": "test prompt",
                        "lora": {
                            "lora_enabled": True,
                            "lora_adapters": [{"lora_id": 404, "strength": 0.55}],
                        },
                        "lora_adapters": [{"lora_id": 999, "strength": 0.2}],
                        "Lora": {
                            "loraStatus": True,
                            "adapters": [{"lora_id": 998, "strength": 0.1}],
                        },
                    },
                    _ctx=None,
                )

        self.assertEqual(captured["lora_adapters"], [{"lora_id": 404, "strength": 0.55}])

    def test_sd15_text2img_disables_lora_when_unified_flag_is_false(self):
        captured = {}

        def _fake_generate_images(params):
            captured.update(params)
            return ["batch/out.png"]

        with patch("backend.workflow.make_batch_id", return_value="batch123"):
            with patch("backend.workflow.generate_images", side_effect=_fake_generate_images):
                _sd15_text2img(
                    {
                        "prompt": "test prompt",
                        "lora": {
                            "lora_enabled": False,
                            "lora_adapters": [{"lora_id": 505, "strength": 0.8}],
                        },
                        "lora_adapters": [{"lora_id": 999, "strength": 0.2}],
                        "Lora": {
                            "loraStatus": True,
                            "adapters": [{"lora_id": 998, "strength": 0.1}],
                        },
                    },
                    _ctx=None,
                )

        self.assertEqual(captured["lora_adapters"], [])


if __name__ == "__main__":
    unittest.main()
