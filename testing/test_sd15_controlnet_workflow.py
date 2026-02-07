import unittest
from unittest.mock import patch

from PIL import Image
from pydantic import ValidationError

from backend.workflow import Sd15ControlNetText2ImgInputs, _sd15_controlnet_text2img


class Sd15ControlNetInputValidationTests(unittest.TestCase):
    def test_conditioning_scale_out_of_range_rejected(self):
        with self.assertRaises(ValidationError):
            Sd15ControlNetText2ImgInputs(
                control_image={"artifact_id": "a0123456789abcdef0123456789abcdef"},
                prompt="test",
                controlnet_conditioning_scale=2.5,
            )

    def test_guidance_end_out_of_range_rejected(self):
        with self.assertRaises(ValidationError):
            Sd15ControlNetText2ImgInputs(
                control_image={"artifact_id": "a0123456789abcdef0123456789abcdef"},
                prompt="test",
                control_guidance_end=1.5,
            )

    def test_default_controlnet_model_uses_sd15_v11(self):
        inputs = Sd15ControlNetText2ImgInputs(
            control_image={"artifact_id": "a0123456789abcdef0123456789abcdef"},
            prompt="test",
        )
        self.assertEqual(inputs.controlnet_model, "lllyasviel/control_v11p_sd15_canny")


class Sd15ControlNetWorkflowPlumbingTests(unittest.TestCase):
    def test_controlnet_task_passes_expected_pipeline_kwargs(self):
        captured = {}

        def _fake_generate_images_controlnet(**kwargs):
            captured.update(kwargs)
            return ["batch/out.png"]

        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with patch("backend.workflow.make_batch_id", return_value="batch123"):
                with patch(
                    "backend.workflow.generate_images_controlnet",
                    side_effect=_fake_generate_images_controlnet,
                ):
                    result = _sd15_controlnet_text2img(
                        {
                            "control_image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
                            "prompt": "test prompt",
                            "negative_prompt": "",
                            "steps": 20,
                            "cfg": 7.5,
                            "width": 512,
                            "height": 512,
                            "seed": 123,
                            "scheduler": "euler",
                            "model": "stable-diffusion-v1-5",
                            "num_images": 1,
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

    def test_control_guidance_start_must_be_lte_end(self):
        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with self.assertRaisesRegex(
                ValueError, "control_guidance_start must be <= control_guidance_end"
            ):
                _sd15_controlnet_text2img(
                    {
                        "control_image": {
                            "artifact_id": "a0123456789abcdef0123456789abcdef"
                        },
                        "prompt": "test prompt",
                        "control_guidance_start": 0.8,
                        "control_guidance_end": 0.2,
                    },
                    _ctx=None,
                )

    def test_warn_mode_returns_warning_on_mismatch(self):
        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with patch("backend.workflow.make_batch_id", return_value="batch123"):
                with patch(
                    "backend.workflow.generate_images_controlnet",
                    return_value=["batch/out.png"],
                ):
                    result = _sd15_controlnet_text2img(
                        {
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

    def test_error_mode_rejects_mismatch(self):
        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with self.assertRaisesRegex(
                ValueError, "ControlNet model/preprocessor pairing mismatch"
            ):
                _sd15_controlnet_text2img(
                    {
                        "control_image": {
                            "artifact_id": "a0123456789abcdef0123456789abcdef"
                        },
                        "prompt": "test prompt",
                        "controlnet_model": "lllyasviel/control_v11p_sd15_openpose",
                        "controlnet_preprocessor_id": "canny",
                        "controlnet_compat_mode": "error",
                    },
                    _ctx=None,
                )


if __name__ == "__main__":
    unittest.main()
