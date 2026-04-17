import unittest
from unittest.mock import patch

from PIL import Image
from pydantic import ValidationError

from backend.workflow import Sd15InpaintInputs, _sd15_inpaint


class Sd15InpaintControlNetInputValidationTests(unittest.TestCase):
    def test_conditioning_scale_out_of_range_rejected(self):
        with self.assertRaises(ValidationError):
            Sd15InpaintInputs(
                initial_image={"artifact_id": "a0123456789abcdef0123456789abcdef"},
                mask_image={"artifact_id": "a0123456789abcdef0123456789abcdef"},
                prompt="test",
                control_image={"artifact_id": "a0123456789abcdef0123456789abcdef"},
                controlnet_conditioning_scale=2.5,
            )

    def test_guidance_end_out_of_range_rejected(self):
        with self.assertRaises(ValidationError):
            Sd15InpaintInputs(
                initial_image={"artifact_id": "a0123456789abcdef0123456789abcdef"},
                mask_image={"artifact_id": "a0123456789abcdef0123456789abcdef"},
                prompt="test",
                control_image={"artifact_id": "a0123456789abcdef0123456789abcdef"},
                control_guidance_end=1.5,
            )

    def test_ip_adapter_scale_out_of_range_rejected(self):
        with self.assertRaises(ValidationError):
            Sd15InpaintInputs(
                initial_image={"artifact_id": "a0123456789abcdef0123456789abcdef"},
                mask_image={"artifact_id": "a0123456789abcdef0123456789abcdef"},
                prompt="test",
                ip_adapter={
                    "enabled": True,
                    "image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                    "scale": 1.5,
                },
            )


class Sd15InpaintControlNetWorkflowPlumbingTests(unittest.TestCase):
    def test_non_controlnet_path_forwards_lora_adapters(self):
        captured = {}
        lora_adapters = [{"lora_id": 101, "strength": 0.8}]

        def _fake_generate_images_inpaint(params):
            captured.update(params)
            return ["batch/out.png"]

        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with patch("backend.workflow.make_batch_id", return_value="batch123"):
                with patch(
                    "backend.workflow.generate_images_inpaint",
                    side_effect=_fake_generate_images_inpaint,
                ):
                    result = _sd15_inpaint(
                        {
                            "initial_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
                            "mask_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
                            "prompt": "test prompt",
                            "lora": {
                                "lora_enabled": True,
                                "lora_adapters": lora_adapters,
                            },
                        },
                        _ctx=None,
                    )

        self.assertEqual(result["batch_id"], "batch123")
        self.assertEqual(result["images"], ["/outputs/batch/out.png"])
        self.assertIn("lora_adapters", captured)
        self.assertEqual(captured["lora_adapters"], lora_adapters)
        self.assertEqual(captured["batch_id"], "batch123")

    def test_non_controlnet_path_forwards_ip_adapter_settings(self):
        captured = {}
        reference_image = Image.new("RGBA", (32, 32))

        def _fake_generate_images_inpaint(params):
            captured.update(params)
            return ["batch/out.png"]

        with patch("backend.workflow._open_image_ref", return_value=reference_image):
            with patch("backend.workflow.make_batch_id", return_value="batch123"):
                with patch(
                    "backend.workflow.generate_images_inpaint",
                    side_effect=_fake_generate_images_inpaint,
                ):
                    result = _sd15_inpaint(
                        {
                            "initial_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
                            "mask_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
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

    def test_ip_adapter_requires_image_when_enabled(self):
        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with self.assertRaisesRegex(
                ValueError, "ip_adapter.image is required when IP-Adapter is enabled"
            ):
                _sd15_inpaint(
                    {
                        "initial_image": {
                            "artifact_id": "a0123456789abcdef0123456789abcdef"
                        },
                        "mask_image": {
                            "artifact_id": "a0123456789abcdef0123456789abcdef"
                        },
                        "prompt": "test prompt",
                        "ip_adapter": {"enabled": True},
                    },
                    _ctx=None,
                )

    def test_ip_adapter_rejects_lcm_combination(self):
        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with self.assertRaisesRegex(
                ValueError, "sd15.inpaint IP-Adapter cannot be combined with LCM mode"
            ):
                _sd15_inpaint(
                    {
                        "initial_image": {
                            "artifact_id": "a0123456789abcdef0123456789abcdef"
                        },
                        "mask_image": {
                            "artifact_id": "a0123456789abcdef0123456789abcdef"
                        },
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

    def test_ip_adapter_rejects_controlnet_combination(self):
        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with self.assertRaisesRegex(
                ValueError, "sd15.inpaint IP-Adapter cannot be combined with ControlNet"
            ):
                _sd15_inpaint(
                    {
                        "initial_image": {
                            "artifact_id": "a0123456789abcdef0123456789abcdef"
                        },
                        "mask_image": {
                            "artifact_id": "a0123456789abcdef0123456789abcdef"
                        },
                        "control_image": {
                            "artifact_id": "a0123456789abcdef0123456789abcdef"
                        },
                        "prompt": "test prompt",
                        "ip_adapter": {
                            "enabled": True,
                            "image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
                        },
                    },
                    _ctx=None,
                )

    def test_non_controlnet_path_forwards_unified_lora_contract(self):
        captured = {}

        def _fake_generate_images_inpaint(params):
            captured.update(params)
            return ["batch/out.png"]

        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with patch("backend.workflow.make_batch_id", return_value="batch123"):
                with patch(
                    "backend.workflow.generate_images_inpaint",
                    side_effect=_fake_generate_images_inpaint,
                ):
                    _sd15_inpaint(
                        {
                            "initial_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
                            "mask_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
                            "prompt": "test prompt",
                            "lora": {
                                "lora_enabled": True,
                                "lora_adapters": [{"lora_id": 121, "strength": 0.65}],
                            },
                        },
                        _ctx=None,
                    )

        self.assertEqual(captured["lora_adapters"], [{"lora_id": 121, "strength": 0.65}])

    def test_non_controlnet_path_disables_lora_when_unified_flag_is_false(self):
        captured = {}

        def _fake_generate_images_inpaint(params):
            captured.update(params)
            return ["batch/out.png"]

        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with patch("backend.workflow.make_batch_id", return_value="batch123"):
                with patch(
                    "backend.workflow.generate_images_inpaint",
                    side_effect=_fake_generate_images_inpaint,
                ):
                    _sd15_inpaint(
                        {
                            "initial_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
                            "mask_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
                            "prompt": "test prompt",
                            "lora": {
                                "lora_enabled": False,
                                "lora_adapters": [{"lora_id": 121, "strength": 0.65}],
                            },
                        },
                        _ctx=None,
                    )

        self.assertEqual(captured["lora_adapters"], [])

    def test_non_controlnet_lcm_mode_uses_lcm_scheduler_defaults_and_adapter(self):
        captured = {}

        def _fake_generate_images_inpaint(params):
            captured.update(params)
            return ["batch/out.png"]

        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with patch("backend.workflow.make_batch_id", return_value="batch123"):
                with patch(
                    "backend.workflow.generate_images_inpaint",
                    side_effect=_fake_generate_images_inpaint,
                ):
                    _sd15_inpaint(
                        {
                            "initial_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
                            "mask_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
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

    def test_non_controlnet_lcm_scheduler_implies_lcm_mode(self):
        captured = {}

        def _fake_generate_images_inpaint(params):
            captured.update(params)
            return ["batch/out.png"]

        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with patch("backend.workflow.make_batch_id", return_value="batch123"):
                with patch(
                    "backend.workflow.generate_images_inpaint",
                    side_effect=_fake_generate_images_inpaint,
                ):
                    _sd15_inpaint(
                        {
                            "initial_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
                            "mask_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
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

    def test_non_controlnet_lcm_mode_forwards_user_loras(self):
        captured = {}

        def _fake_generate_images_inpaint(params):
            captured.update(params)
            return ["batch/out.png"]

        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with patch("backend.workflow.make_batch_id", return_value="batch123"):
                with patch(
                    "backend.workflow.generate_images_inpaint",
                    side_effect=_fake_generate_images_inpaint,
                ):
                    _sd15_inpaint(
                        {
                            "initial_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
                            "mask_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
                            "prompt": "test prompt",
                            "lcm": {"enabled": True},
                            "lora": {
                                "lora_enabled": True,
                                "lora_adapters": [{"lora_id": 121, "strength": 0.65}],
                            },
                        },
                        _ctx=None,
                    )

        self.assertTrue(captured["lcm_enabled"])
        self.assertEqual(captured["scheduler"], "lcm")
        self.assertEqual(captured["lora_adapters"], [{"lora_id": 121, "strength": 0.65}])

    def test_non_controlnet_lcm_mode_validates_steps_and_cfg(self):
        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with self.assertRaisesRegex(ValueError, r"steps within \[1, 8\]"):
                _sd15_inpaint(
                    {
                        "initial_image": {
                            "artifact_id": "a0123456789abcdef0123456789abcdef"
                        },
                        "mask_image": {
                            "artifact_id": "a0123456789abcdef0123456789abcdef"
                        },
                        "prompt": "test prompt",
                        "lcm": {"enabled": True},
                        "steps": 12,
                    },
                    _ctx=None,
                )

            with self.assertRaisesRegex(ValueError, r"cfg within \[0, 2\]"):
                _sd15_inpaint(
                    {
                        "initial_image": {
                            "artifact_id": "a0123456789abcdef0123456789abcdef"
                        },
                        "mask_image": {
                            "artifact_id": "a0123456789abcdef0123456789abcdef"
                        },
                        "prompt": "test prompt",
                        "lcm": {"enabled": True},
                        "cfg": 7.5,
                    },
                    _ctx=None,
                )

    def test_lcm_mode_rejects_controlnet_combination(self):
        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with self.assertRaisesRegex(
                ValueError, "sd15.inpaint LCM mode cannot be combined with ControlNet"
            ):
                _sd15_inpaint(
                    {
                        "initial_image": {
                            "artifact_id": "a0123456789abcdef0123456789abcdef"
                        },
                        "mask_image": {
                            "artifact_id": "a0123456789abcdef0123456789abcdef"
                        },
                        "prompt": "test prompt",
                        "lcm": {"enabled": True},
                        "control_image": {
                            "artifact_id": "a0123456789abcdef0123456789abcdef"
                        },
                    },
                    _ctx=None,
                )

    def test_non_controlnet_path_rejects_top_level_lora_adapters(self):
        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with self.assertRaisesRegex(
                ValueError, "Top-level SD1.5 `lora_adapters` is no longer supported"
            ):
                _sd15_inpaint(
                    {
                        "initial_image": {
                            "artifact_id": "a0123456789abcdef0123456789abcdef"
                        },
                        "mask_image": {
                            "artifact_id": "a0123456789abcdef0123456789abcdef"
                        },
                        "prompt": "test prompt",
                        "lora_adapters": [{"lora_id": 999, "strength": 0.1}],
                    },
                    _ctx=None,
                )

    def test_non_controlnet_path_rejects_legacy_lora_wrapper(self):
        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with self.assertRaisesRegex(
                ValueError, "Legacy SD1.5 LoRA field `Lora` is no longer supported"
            ):
                _sd15_inpaint(
                    {
                        "initial_image": {
                            "artifact_id": "a0123456789abcdef0123456789abcdef"
                        },
                        "mask_image": {
                            "artifact_id": "a0123456789abcdef0123456789abcdef"
                        },
                        "prompt": "test prompt",
                        "Lora": {
                            "loraStatus": True,
                            "adapters": [{"lora_id": 121, "strength": 0.65}],
                        },
                    },
                    _ctx=None,
                )

    def test_non_controlnet_path_preserves_zero_padding_mask_crop(self):
        captured = {}

        def _fake_generate_images_inpaint(params):
            captured.update(params)
            return ["batch/out.png"]

        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with patch("backend.workflow.make_batch_id", return_value="batch123"):
                with patch(
                    "backend.workflow.generate_images_inpaint",
                    side_effect=_fake_generate_images_inpaint,
                ):
                    _sd15_inpaint(
                        {
                            "initial_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
                            "mask_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
                            "prompt": "test prompt",
                            "padding_mask_crop": 0,
                        },
                        _ctx=None,
                    )

        self.assertEqual(captured["padding_mask_crop"], 0)

    def test_controlnet_path_passes_expected_pipeline_kwargs(self):
        captured = {}

        def _fake_generate_images_inpaint_controlnet(params):
            captured.update(params)
            return ["batch/out.png"]

        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with patch("backend.workflow.make_batch_id", return_value="batch123"):
                with patch(
                    "backend.workflow.generate_images_inpaint_controlnet",
                    side_effect=_fake_generate_images_inpaint_controlnet,
                ):
                    result = _sd15_inpaint(
                        {
                            "initial_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
                            "mask_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
                            "control_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
                            "prompt": "test prompt",
                            "negative_prompt": "",
                            "strength": 0.5,
                            "steps": 20,
                            "cfg": 7.5,
                            "seed": 123,
                            "scheduler": "euler",
                            "model": "stable-diffusion-v1-5",
                            "num_images": 1,
                            "padding_mask_crop": 32,
                            "clip_skip": 1,
                            "controlnet_model": "lllyasviel/sd-controlnet-canny",
                            "controlnet_conditioning_scale": 1.25,
                            "controlnet_guess_mode": True,
                            "control_guidance_start": 0.1,
                            "control_guidance_end": 0.9,
                        },
                        _ctx=None,
                    )

        self.assertEqual(result["batch_id"], "batch123")
        self.assertEqual(result["images"], ["/outputs/batch/out.png"])
        self.assertIn("control_image", captured)
        self.assertEqual(captured["controlnet_conditioning_scale"], 1.25)
        self.assertTrue(captured["controlnet_guess_mode"])
        self.assertEqual(captured["control_guidance_start"], 0.1)
        self.assertEqual(captured["control_guidance_end"], 0.9)

    def test_controlnet_path_forwards_lora_adapters(self):
        captured = {}
        lora_adapters = [{"lora_id": 201, "strength": 0.65}]

        def _fake_generate_images_inpaint_controlnet(params):
            captured.update(params)
            return ["batch/out.png"]

        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with patch("backend.workflow.make_batch_id", return_value="batch123"):
                with patch(
                    "backend.workflow.generate_images_inpaint_controlnet",
                    side_effect=_fake_generate_images_inpaint_controlnet,
                ):
                    result = _sd15_inpaint(
                        {
                            "initial_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
                            "mask_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
                            "control_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
                            "prompt": "test prompt",
                            "controlnet_model": "lllyasviel/sd-controlnet-canny",
                            "lora": {
                                "lora_enabled": True,
                                "lora_adapters": lora_adapters,
                            },
                        },
                        _ctx=None,
                    )

        self.assertEqual(result["batch_id"], "batch123")
        self.assertEqual(result["images"], ["/outputs/batch/out.png"])
        self.assertIn("lora_adapters", captured)
        self.assertEqual(captured["lora_adapters"], lora_adapters)

    def test_controlnet_path_preserves_zero_padding_mask_crop(self):
        captured = {}

        def _fake_generate_images_inpaint_controlnet(params):
            captured.update(params)
            return ["batch/out.png"]

        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with patch("backend.workflow.make_batch_id", return_value="batch123"):
                with patch(
                    "backend.workflow.generate_images_inpaint_controlnet",
                    side_effect=_fake_generate_images_inpaint_controlnet,
                ):
                    _sd15_inpaint(
                        {
                            "initial_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
                            "mask_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
                            "control_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
                            "prompt": "test prompt",
                            "padding_mask_crop": 0,
                        },
                        _ctx=None,
                    )

        self.assertEqual(captured["padding_mask_crop"], 0)

    def test_controlnet_missing_control_image_rejected(self):
        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with self.assertRaisesRegex(
                ValueError, "control_image is required when using ControlNet in sd15.inpaint"
            ):
                _sd15_inpaint(
                    {
                        "initial_image": {
                            "artifact_id": "a0123456789abcdef0123456789abcdef"
                        },
                        "mask_image": {
                            "artifact_id": "a0123456789abcdef0123456789abcdef"
                        },
                        "prompt": "test prompt",
                        "controlnet_model": "lllyasviel/sd-controlnet-canny",
                    },
                    _ctx=None,
                )

    def test_warn_mode_returns_warning_on_mismatch(self):
        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with patch("backend.workflow.make_batch_id", return_value="batch123"):
                with patch(
                    "backend.workflow.generate_images_inpaint_controlnet",
                    return_value=["batch/out.png"],
                ):
                    result = _sd15_inpaint(
                        {
                            "initial_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
                            "mask_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
                            "control_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
                            "prompt": "test prompt",
                            "controlnet_model": "lllyasviel/control_v11p_sd15_openpose",
                            "controlnet_preprocessor_id": "canny",
                            "controlnet_compat_mode": "warn",
                        },
                        _ctx=None,
                    )
        self.assertIn("warnings", result)
        self.assertGreaterEqual(len(result["warnings"]), 1)


if __name__ == "__main__":
    unittest.main()
