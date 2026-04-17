import unittest
from unittest.mock import patch

from PIL import Image
from pydantic import ValidationError

from backend.workflow import Sd15Text2ImgInputs, _sd15_text2img


class Sd15Text2ImgInputValidationTests(unittest.TestCase):
    def test_ip_adapter_scale_out_of_range_rejected(self):
        with self.assertRaises(ValidationError):
            Sd15Text2ImgInputs(
                prompt="test",
                ip_adapter={
                    "enabled": True,
                    "image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                    "scale": 1.5,
                },
            )


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
                        "lora": {
                            "lora_enabled": True,
                            "lora_adapters": [{"lora_id": 101, "strength": 0.75}],
                        },
                        "hires_enabled": True,
                        "hires_scale": 1.5,
                        "weighting_policy": "a1111-like",
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

    def test_sd15_text2img_accepts_unified_lora_and_hires_fields(self):
        captured = {}

        def _fake_generate_images(params):
            captured.update(params)
            return ["batch/out.png"]

        with patch("backend.workflow.make_batch_id", return_value="batch123"):
            with patch("backend.workflow.generate_images", side_effect=_fake_generate_images):
                result = _sd15_text2img(
                    {
                        "prompt": "test prompt",
                        "lora": {
                            "lora_enabled": True,
                            "lora_adapters": [{"lora_id": 303, "strength": 0.6}],
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
                    },
                    _ctx=None,
                )

        self.assertEqual(captured["lora_adapters"], [])

    def test_sd15_text2img_lcm_mode_uses_lcm_scheduler_defaults_and_adapter(self):
        captured = {}

        def _fake_generate_images(params):
            captured.update(params)
            return ["batch/out.png"]

        with patch("backend.workflow.make_batch_id", return_value="batch123"):
            with patch("backend.workflow.generate_images", side_effect=_fake_generate_images):
                _sd15_text2img(
                    {
                        "prompt": "test prompt",
                        "lcm": {"enabled": True},
                    },
                    _ctx=None,
                )

        self.assertTrue(captured["lcm_enabled"])
        self.assertEqual(captured["scheduler"], "lcm")
        self.assertEqual(captured["steps"], 4)
        self.assertEqual(captured["cfg"], 0.0)
        self.assertEqual(captured["lcm_lora_model"], "latent-consistency/lcm-lora-sdv1-5")
        self.assertIsNone(captured["lora_adapters"])

    def test_sd15_text2img_lcm_scheduler_implies_lcm_mode(self):
        captured = {}

        def _fake_generate_images(params):
            captured.update(params)
            return ["batch/out.png"]

        with patch("backend.workflow.make_batch_id", return_value="batch123"):
            with patch("backend.workflow.generate_images", side_effect=_fake_generate_images):
                _sd15_text2img(
                    {
                        "prompt": "test prompt",
                        "scheduler": "lcm",
                        "steps": 8,
                        "cfg": 1.5,
                    },
                    _ctx=None,
                )

        self.assertTrue(captured["lcm_enabled"])
        self.assertEqual(captured["scheduler"], "lcm")
        self.assertEqual(captured["steps"], 8)
        self.assertEqual(captured["cfg"], 1.5)

    def test_sd15_text2img_lcm_mode_forwards_user_loras(self):
        captured = {}

        def _fake_generate_images(params):
            captured.update(params)
            return ["batch/out.png"]

        with patch("backend.workflow.make_batch_id", return_value="batch123"):
            with patch("backend.workflow.generate_images", side_effect=_fake_generate_images):
                _sd15_text2img(
                    {
                        "prompt": "test prompt",
                        "lcm": {"enabled": True},
                        "lora": {
                            "lora_enabled": True,
                            "lora_adapters": [{"lora_id": 505, "strength": 0.8}],
                        },
                    },
                    _ctx=None,
                )

        self.assertTrue(captured["lcm_enabled"])
        self.assertEqual(captured["scheduler"], "lcm")
        self.assertEqual(captured["lora_adapters"], [{"lora_id": 505, "strength": 0.8}])

    def test_sd15_text2img_forwards_ip_adapter_settings(self):
        captured = {}
        reference_image = Image.new("RGBA", (32, 32))

        def _fake_generate_images(params):
            captured.update(params)
            return ["batch/out.png"]

        with patch("backend.workflow._open_image_ref", return_value=reference_image):
            with patch("backend.workflow.make_batch_id", return_value="batch123"):
                with patch("backend.workflow.generate_images", side_effect=_fake_generate_images):
                    result = _sd15_text2img(
                        {
                            "prompt": "test prompt",
                            "ip_adapter": {
                                "enabled": True,
                                "image": {
                                    "artifact_id": "a0123456789abcdef0123456789abcdef"
                                },
                                "scale": 0.55,
                                "model": "h94/IP-Adapter",
                                "subfolder": "models",
                                "weight_name": "ip-adapter_sd15.bin",
                            },
                        },
                        _ctx=None,
                    )

        self.assertEqual(result["batch_id"], "batch123")
        self.assertEqual(result["images"], ["/outputs/batch/out.png"])
        self.assertEqual(captured["ip_adapter_image"].mode, "RGB")
        self.assertEqual(captured["ip_adapter_scale"], 0.55)
        self.assertEqual(captured["ip_adapter_model"], "h94/IP-Adapter")
        self.assertEqual(captured["ip_adapter_subfolder"], "models")
        self.assertEqual(captured["ip_adapter_weight_name"], "ip-adapter_sd15.bin")

    def test_sd15_text2img_requires_ip_adapter_image_when_enabled(self):
        with self.assertRaisesRegex(
            ValueError, "ip_adapter.image is required when IP-Adapter is enabled"
        ):
            _sd15_text2img(
                {
                    "prompt": "test prompt",
                    "ip_adapter": {"enabled": True},
                },
                _ctx=None,
            )

    def test_sd15_text2img_rejects_ip_adapter_with_lcm(self):
        with self.assertRaisesRegex(
            ValueError, "sd15.text2img IP-Adapter cannot be combined with LCM mode"
        ):
            _sd15_text2img(
                {
                    "prompt": "test prompt",
                    "lcm": {"enabled": True},
                    "ip_adapter": {
                        "enabled": True,
                        "image": {
                            "artifact_id": "a0123456789abcdef0123456789abcdef"
                        },
                    },
                },
                _ctx=None,
            )

    def test_sd15_text2img_lcm_mode_validates_steps_and_cfg(self):
        with self.assertRaisesRegex(ValueError, r"steps within \[1, 8\]"):
            _sd15_text2img(
                {
                    "prompt": "test prompt",
                    "lcm": {"enabled": True},
                    "steps": 12,
                },
                _ctx=None,
            )

        with self.assertRaisesRegex(ValueError, r"cfg within \[0, 2\]"):
            _sd15_text2img(
                {
                    "prompt": "test prompt",
                    "lcm": {"enabled": True},
                    "cfg": 7.5,
                },
                _ctx=None,
            )

    def test_sd15_text2img_rejects_top_level_lora_adapters(self):
        with self.assertRaisesRegex(
            ValueError, "Top-level SD1.5 `lora_adapters` is no longer supported"
        ):
            _sd15_text2img(
                {
                    "prompt": "test prompt",
                    "lora_adapters": [{"lora_id": 999, "strength": 0.2}],
                },
                _ctx=None,
            )

    def test_sd15_text2img_rejects_legacy_lora_wrapper(self):
        with self.assertRaisesRegex(
            ValueError, "Legacy SD1.5 LoRA field `Lora` is no longer supported"
        ):
            _sd15_text2img(
                {
                    "prompt": "test prompt",
                    "Lora": {
                        "loraStatus": True,
                        "adapters": [{"lora_id": 505, "strength": 0.8}],
                    },
                },
                _ctx=None,
            )


if __name__ == "__main__":
    unittest.main()
