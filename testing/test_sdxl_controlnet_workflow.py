import unittest
from unittest.mock import patch

from PIL import Image
from pydantic import ValidationError

from backend.workflow import (
    SdxlControlNetText2ImgInputs,
    SdxlImg2ImgInputs,
    SdxlInpaintInputs,
    SdxlText2ImgInputs,
    _sdxl_controlnet_text2img,
    _sdxl_img2img,
    _sdxl_inpaint,
    _sdxl_text2img,
)


class SdxlControlNetInputValidationTests(unittest.TestCase):
    def test_sdxl_text2img_accepts_lora_adapters(self):
        inputs = SdxlText2ImgInputs(
            prompt="test",
            lora_adapters=[{"lora_id": 101, "strength": 0.8}],
        )
        self.assertEqual(inputs.lora_adapters, [{"lora_id": 101, "strength": 0.8}])

    def test_sdxl_text2img_ip_adapter_scale_out_of_range_rejected(self):
        with self.assertRaises(ValidationError):
            SdxlText2ImgInputs(
                prompt="test",
                ip_adapter={
                    "enabled": True,
                    "image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                    "scale": 1.5,
                },
            )

    def test_sdxl_img2img_ip_adapter_scale_out_of_range_rejected(self):
        with self.assertRaises(ValidationError):
            SdxlImg2ImgInputs(
                initial_image={"artifact_id": "a0123456789abcdef0123456789abcdef"},
                prompt="test",
                ip_adapter={
                    "enabled": True,
                    "image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                    "scale": 1.5,
                },
            )

    def test_conditioning_scale_out_of_range_rejected(self):
        with self.assertRaises(ValidationError):
            SdxlControlNetText2ImgInputs(
                control_image={"artifact_id": "a0123456789abcdef0123456789abcdef"},
                prompt="test",
                controlnet_conditioning_scale=2.5,
            )

    def test_default_controlnet_model_uses_sdxl_default(self):
        inputs = SdxlControlNetText2ImgInputs(
            control_image={"artifact_id": "a0123456789abcdef0123456789abcdef"},
            prompt="test",
        )
        self.assertEqual(inputs.controlnet_model, "diffusers/controlnet-canny-sdxl-1.0")

    def test_img2img_conditioning_scale_out_of_range_rejected(self):
        with self.assertRaises(ValidationError):
            SdxlImg2ImgInputs(
                initial_image={"artifact_id": "a0123456789abcdef0123456789abcdef"},
                prompt="test",
                control_image={"artifact_id": "a0123456789abcdef0123456789abcdef"},
                controlnet_conditioning_scale=2.5,
            )

    def test_sdxl_img2img_accepts_lora_adapters(self):
        inputs = SdxlImg2ImgInputs(
            initial_image={"artifact_id": "a0123456789abcdef0123456789abcdef"},
            prompt="test",
            lora_adapters=[{"lora_id": 202, "strength": 0.7}],
        )
        self.assertEqual(inputs.lora_adapters, [{"lora_id": 202, "strength": 0.7}])

    def test_sdxl_img2img_accepts_ip_adapter(self):
        inputs = SdxlImg2ImgInputs(
            initial_image={"artifact_id": "a0123456789abcdef0123456789abcdef"},
            prompt="test",
            ip_adapter={
                "enabled": True,
                "image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                "scale": 0.55,
            },
        )
        self.assertIsNotNone(inputs.ip_adapter)
        self.assertEqual(inputs.ip_adapter.scale, 0.55)

    def test_inpaint_conditioning_scale_out_of_range_rejected(self):
        with self.assertRaises(ValidationError):
            SdxlInpaintInputs(
                initial_image={"artifact_id": "a0123456789abcdef0123456789abcdef"},
                mask_image={"artifact_id": "a0123456789abcdef0123456789abcdef"},
                prompt="test",
                control_image={"artifact_id": "a0123456789abcdef0123456789abcdef"},
                controlnet_conditioning_scale=2.5,
            )

    def test_sdxl_inpaint_accepts_lora_adapters(self):
        inputs = SdxlInpaintInputs(
            initial_image={"artifact_id": "a0123456789abcdef0123456789abcdef"},
            mask_image={"artifact_id": "a0123456789abcdef0123456789abcdef"},
            prompt="test",
            lora_adapters=[{"lora_id": 303, "strength": 0.6}],
        )
        self.assertEqual(inputs.lora_adapters, [{"lora_id": 303, "strength": 0.6}])


class SdxlControlNetWorkflowPlumbingTests(unittest.TestCase):
    def test_sdxl_text2img_forwards_lora_adapters(self):
        captured = {}

        def _fake_generate_text2img(payload):
            captured.update(payload)
            return {"images": ["/outputs/batch/out.png"]}

        with patch("backend.sdxl_pipeline.generate_text2img", side_effect=_fake_generate_text2img):
            result = _sdxl_text2img(
                {
                    "prompt": "test prompt",
                    "lora_adapters": [{"lora_id": 101, "strength": 0.8}],
                },
                _ctx=None,
            )

        self.assertEqual(result["images"], ["/outputs/batch/out.png"])
        self.assertIn("lora_adapters", captured)
        self.assertEqual(captured["lora_adapters"], [{"lora_id": 101, "strength": 0.8}])

    def test_sdxl_text2img_defaults_lora_adapters_when_missing(self):
        captured = {}

        def _fake_generate_text2img(payload):
            captured.update(payload)
            return {"images": ["/outputs/batch/out.png"]}

        with patch("backend.sdxl_pipeline.generate_text2img", side_effect=_fake_generate_text2img):
            result = _sdxl_text2img(
                {
                    "prompt": "test prompt",
                },
                _ctx=None,
            )

        self.assertEqual(result["images"], ["/outputs/batch/out.png"])
        self.assertIn("lora_adapters", captured)
        self.assertIsNone(captured["lora_adapters"])

    def test_sdxl_text2img_forwards_ip_adapter_settings(self):
        captured = {}
        reference_image = Image.new("RGBA", (32, 32))

        def _fake_generate_text2img(payload):
            captured.update(payload)
            return {"images": ["/outputs/batch/out.png"]}

        with patch("backend.workflow._open_image_ref", return_value=reference_image):
            with patch("backend.sdxl_pipeline.generate_text2img", side_effect=_fake_generate_text2img):
                result = _sdxl_text2img(
                    {
                        "prompt": "test prompt",
                        "ip_adapter": {
                            "enabled": True,
                            "image": {
                                "artifact_id": "a0123456789abcdef0123456789abcdef"
                            },
                            "scale": 0.55,
                            "model": "h94/IP-Adapter",
                            "subfolder": "sdxl_models",
                            "weight_name": "ip-adapter_sdxl.bin",
                        },
                    },
                    _ctx=None,
                )

        self.assertEqual(result["images"], ["/outputs/batch/out.png"])
        self.assertEqual(captured["ip_adapter_image"].mode, "RGB")
        self.assertEqual(captured["ip_adapter_scale"], 0.55)
        self.assertEqual(captured["ip_adapter_model"], "h94/IP-Adapter")
        self.assertEqual(captured["ip_adapter_subfolder"], "sdxl_models")
        self.assertEqual(captured["ip_adapter_weight_name"], "ip-adapter_sdxl.bin")

    def test_sdxl_text2img_requires_ip_adapter_image_when_enabled(self):
        with self.assertRaisesRegex(
            ValueError, "ip_adapter.image is required when IP-Adapter is enabled"
        ):
            _sdxl_text2img(
                {
                    "prompt": "test prompt",
                    "ip_adapter": {"enabled": True},
                },
                _ctx=None,
            )

    def test_sdxl_img2img_forwards_ip_adapter_settings(self):
        captured = {}
        reference_image = Image.new("RGBA", (32, 32))

        def _fake_generate_img2img(payload):
            captured.update(payload)
            return {"images": ["/outputs/batch/out.png"]}

        with patch("backend.workflow._open_image_ref", return_value=reference_image):
            with patch("backend.sdxl_pipeline.generate_img2img", side_effect=_fake_generate_img2img):
                result = _sdxl_img2img(
                    {
                        "initial_image": {
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
                            "subfolder": "sdxl_models",
                            "weight_name": "ip-adapter_sdxl.bin",
                        },
                    },
                    _ctx=None,
                )

        self.assertEqual(result["images"], ["/outputs/batch/out.png"])
        self.assertEqual(captured["ip_adapter_image"].mode, "RGB")
        self.assertEqual(captured["ip_adapter_scale"], 0.55)
        self.assertEqual(captured["ip_adapter_model"], "h94/IP-Adapter")
        self.assertEqual(captured["ip_adapter_subfolder"], "sdxl_models")
        self.assertEqual(captured["ip_adapter_weight_name"], "ip-adapter_sdxl.bin")

    def test_sdxl_img2img_requires_ip_adapter_image_when_enabled(self):
        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with self.assertRaisesRegex(
                ValueError, "ip_adapter.image is required when IP-Adapter is enabled"
            ):
                _sdxl_img2img(
                    {
                        "initial_image": {
                            "artifact_id": "a0123456789abcdef0123456789abcdef"
                        },
                        "prompt": "test prompt",
                        "ip_adapter": {"enabled": True},
                    },
                    _ctx=None,
                )

    def test_sdxl_img2img_rejects_ip_adapter_controlnet_combination(self):
        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with self.assertRaisesRegex(
                ValueError, "sdxl.img2img IP-Adapter cannot be combined with ControlNet"
            ):
                _sdxl_img2img(
                    {
                        "initial_image": {
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

    def test_controlnet_task_passes_expected_pipeline_params(self):
        captured = {}

        def _fake_generate_controlnet_text2img(params):
            captured.update(params)
            return {"images": ["/outputs/batch/out.png"]}

        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with patch(
                "backend.sdxl_pipeline.generate_controlnet_text2img",
                side_effect=_fake_generate_controlnet_text2img,
            ):
                result = _sdxl_controlnet_text2img(
                    {
                        "control_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "prompt": "test prompt",
                        "negative_prompt": "",
                        "steps": 20,
                        "guidance_scale": 7.5,
                        "width": 1024,
                        "height": 1024,
                        "seed": 123,
                        "scheduler": "euler",
                        "model": "stable-diffusion-xl-base-1.0",
                        "num_images": 1,
                        "clip_skip": 1,
                        "controlnet_model": "diffusers/controlnet-canny-sdxl-1.0",
                        "controlnet_conditioning_scale": 1.25,
                        "controlnet_guess_mode": True,
                        "control_guidance_start": 0.1,
                        "control_guidance_end": 0.9,
                    },
                    _ctx=None,
                )

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
                _sdxl_controlnet_text2img(
                    {
                        "control_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "prompt": "test prompt",
                        "control_guidance_start": 0.8,
                        "control_guidance_end": 0.2,
                    },
                    _ctx=None,
                )

    def test_multi_controlnet_passes_list_params_and_warns_for_perf(self):
        captured = {}

        def _fake_generate_controlnet_text2img(params):
            captured.update(params)
            return {"images": ["/outputs/batch/out.png"]}

        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with patch(
                "backend.sdxl_pipeline.generate_controlnet_text2img",
                side_effect=_fake_generate_controlnet_text2img,
            ):
                result = _sdxl_controlnet_text2img(
                    {
                        "control_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "prompt": "test prompt",
                        "controlnet_models": [
                            "diffusers/controlnet-canny-sdxl-1.0",
                            "diffusers/controlnet-depth-sdxl-1.0",
                        ],
                        "controlnet_preprocessor_ids": ["canny", "midas-depth"],
                        "controlnet_conditioning_scales": [1.0, 0.7],
                    },
                    _ctx=None,
                )

        self.assertIsInstance(captured["controlnet_model"], list)
        self.assertEqual(len(captured["controlnet_model"]), 2)
        self.assertIsInstance(captured["control_image"], list)
        self.assertEqual(len(captured["control_image"]), 2)
        self.assertEqual(captured["controlnet_conditioning_scale"], [1.0, 0.7])
        self.assertIn("warnings", result)
        self.assertTrue(any("VRAM use" in warning for warning in result["warnings"]))

    def test_multi_controlnet_scale_list_must_align(self):
        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with self.assertRaisesRegex(
                ValueError, "controlnet_conditioning_scales length must match"
            ):
                _sdxl_controlnet_text2img(
                    {
                        "control_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "prompt": "test prompt",
                        "controlnet_models": [
                            "diffusers/controlnet-canny-sdxl-1.0",
                            "diffusers/controlnet-depth-sdxl-1.0",
                        ],
                        "controlnet_conditioning_scales": [1.0],
                    },
                    _ctx=None,
                )

    def test_multi_controlnet_model_count_guardrail(self):
        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with self.assertRaisesRegex(
                ValueError, "At most 2 ControlNet models are supported"
            ):
                _sdxl_controlnet_text2img(
                    {
                        "control_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "prompt": "test prompt",
                        "controlnet_models": [
                            "diffusers/controlnet-canny-sdxl-1.0",
                            "diffusers/controlnet-depth-sdxl-1.0",
                            "diffusers/controlnet-openpose-sdxl-1.0",
                        ],
                    },
                    _ctx=None,
                )

    def test_multi_controlnet_scale_range_validation(self):
        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with self.assertRaisesRegex(
                ValueError, "controlnet conditioning scales must be within \\[0, 2\\]"
            ):
                _sdxl_controlnet_text2img(
                    {
                        "control_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "prompt": "test prompt",
                        "controlnet_models": [
                            "diffusers/controlnet-canny-sdxl-1.0",
                            "diffusers/controlnet-depth-sdxl-1.0",
                        ],
                        "controlnet_conditioning_scales": [1.0, 2.5],
                    },
                    _ctx=None,
                )

    def test_img2img_controlnet_path_passes_expected_pipeline_kwargs(self):
        captured = {}

        def _fake_generate_img2img_controlnet(params):
            captured.update(params)
            return {"images": ["/outputs/batch/out.png"]}

        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with patch(
                "backend.sdxl_pipeline.generate_img2img_controlnet",
                side_effect=_fake_generate_img2img_controlnet,
            ):
                result = _sdxl_img2img(
                    {
                        "initial_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "control_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "prompt": "test prompt",
                        "negative_prompt": "",
                        "strength": 0.75,
                        "steps": 20,
                        "guidance_scale": 7.5,
                        "width": 1024,
                        "height": 1024,
                        "seed": 123,
                        "scheduler": "euler",
                        "model": "stable-diffusion-xl-base-1.0",
                        "num_images": 1,
                        "clip_skip": 1,
                        "controlnet_model": "diffusers/controlnet-canny-sdxl-1.0",
                        "controlnet_conditioning_scale": 1.25,
                        "controlnet_guess_mode": True,
                        "control_guidance_start": 0.1,
                        "control_guidance_end": 0.9,
                    },
                    _ctx=None,
                )

        self.assertEqual(result["images"], ["/outputs/batch/out.png"])
        self.assertIn("control_image", captured)
        self.assertEqual(captured["controlnet_conditioning_scale"], 1.25)
        self.assertTrue(captured["controlnet_guess_mode"])
        self.assertEqual(captured["control_guidance_start"], 0.1)
        self.assertEqual(captured["control_guidance_end"], 0.9)

    def test_img2img_controlnet_path_forwards_lora_adapters(self):
        captured = {}

        def _fake_generate_img2img_controlnet(params):
            captured.update(params)
            return {"images": ["/outputs/batch/out.png"]}

        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with patch(
                "backend.sdxl_pipeline.generate_img2img_controlnet",
                side_effect=_fake_generate_img2img_controlnet,
            ):
                _sdxl_img2img(
                    {
                        "initial_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "control_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "prompt": "test prompt",
                        "lora_adapters": [{"lora_id": 101, "strength": 0.8}],
                    },
                    _ctx=None,
                )

        self.assertIn("lora_adapters", captured)
        self.assertEqual(captured["lora_adapters"], [{"lora_id": 101, "strength": 0.8}])

    def test_img2img_controlnet_missing_control_image_rejected(self):
        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with self.assertRaisesRegex(
                ValueError, "control_image is required when using ControlNet in sdxl.img2img"
            ):
                _sdxl_img2img(
                    {
                        "initial_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "prompt": "test prompt",
                        "controlnet_model": "diffusers/controlnet-canny-sdxl-1.0",
                    },
                    _ctx=None,
                )

    def test_img2img_warn_mode_returns_warning_on_unknown_preprocessor(self):
        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with patch(
                "backend.sdxl_pipeline.generate_img2img_controlnet",
                return_value={"images": ["/outputs/batch/out.png"]},
            ):
                result = _sdxl_img2img(
                    {
                        "initial_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "control_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "prompt": "test prompt",
                        "controlnet_model": "diffusers/controlnet-canny-sdxl-1.0",
                        "controlnet_preprocessor_id": "unknown-id",
                        "controlnet_compat_mode": "warn",
                    },
                    _ctx=None,
                )
        self.assertIn("warnings", result)
        self.assertGreaterEqual(len(result["warnings"]), 1)

    def test_img2img_without_controlnet_uses_default_pipeline(self):
        captured = {}

        def _fake_generate_img2img(params):
            captured.update(params)
            return {"images": ["/outputs/batch/plain.png"]}

        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with patch(
                "backend.sdxl_pipeline.generate_img2img",
                side_effect=_fake_generate_img2img,
            ) as default_runner:
                with patch("backend.sdxl_pipeline.generate_img2img_controlnet") as controlnet_runner:
                    result = _sdxl_img2img(
                        {
                            "initial_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                            "prompt": "test prompt",
                            "lora_adapters": [{"lora_id": 101, "strength": 0.8}],
                        },
                        _ctx=None,
                    )

        self.assertEqual(result["images"], ["/outputs/batch/plain.png"])
        default_runner.assert_called_once()
        controlnet_runner.assert_not_called()
        self.assertIn("lora_adapters", captured)
        self.assertEqual(captured["lora_adapters"], [{"lora_id": 101, "strength": 0.8}])

    def test_inpaint_controlnet_path_passes_expected_pipeline_kwargs(self):
        captured = {}

        def _fake_generate_inpaint_controlnet(params):
            captured.update(params)
            return {"images": ["/outputs/batch/out.png"]}

        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with patch(
                "backend.sdxl_pipeline.generate_inpaint_controlnet",
                side_effect=_fake_generate_inpaint_controlnet,
            ):
                result = _sdxl_inpaint(
                    {
                        "initial_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "mask_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "control_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "prompt": "test prompt",
                        "negative_prompt": "",
                        "strength": 0.5,
                        "steps": 20,
                        "guidance_scale": 7.5,
                        "seed": 123,
                        "scheduler": "euler",
                        "model": "stable-diffusion-xl-base-1.0",
                        "num_images": 1,
                        "padding_mask_crop": 32,
                        "clip_skip": 1,
                        "controlnet_model": "diffusers/controlnet-canny-sdxl-1.0",
                        "controlnet_conditioning_scale": 1.25,
                        "controlnet_guess_mode": True,
                        "control_guidance_start": 0.1,
                        "control_guidance_end": 0.9,
                    },
                    _ctx=None,
                )

        self.assertEqual(result["images"], ["/outputs/batch/out.png"])
        self.assertIn("control_image", captured)
        self.assertEqual(captured["controlnet_conditioning_scale"], 1.25)
        self.assertTrue(captured["controlnet_guess_mode"])
        self.assertEqual(captured["control_guidance_start"], 0.1)
        self.assertEqual(captured["control_guidance_end"], 0.9)

    def test_inpaint_controlnet_path_forwards_lora_adapters(self):
        captured = {}

        def _fake_generate_inpaint_controlnet(params):
            captured.update(params)
            return {"images": ["/outputs/batch/out.png"]}

        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with patch(
                "backend.sdxl_pipeline.generate_inpaint_controlnet",
                side_effect=_fake_generate_inpaint_controlnet,
            ):
                _sdxl_inpaint(
                    {
                        "initial_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "mask_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "control_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "prompt": "test prompt",
                        "lora_adapters": [{"lora_id": 101, "strength": 0.8}],
                    },
                    _ctx=None,
                )

        self.assertIn("lora_adapters", captured)
        self.assertEqual(captured["lora_adapters"], [{"lora_id": 101, "strength": 0.8}])

    def test_inpaint_controlnet_missing_control_image_rejected(self):
        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with self.assertRaisesRegex(
                ValueError, "control_image is required when using ControlNet in sdxl.inpaint"
            ):
                _sdxl_inpaint(
                    {
                        "initial_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "mask_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "prompt": "test prompt",
                        "controlnet_model": "diffusers/controlnet-canny-sdxl-1.0",
                    },
                    _ctx=None,
                )

    def test_inpaint_controlnet_mismatched_mask_dimensions_rejected(self):
        opened = [
            Image.new("RGB", (64, 64)),
            Image.new("L", (32, 32)),
            Image.new("RGB", (64, 64)),
        ]
        with patch("backend.workflow._open_image_ref", side_effect=opened):
            with self.assertRaisesRegex(
                ValueError,
                "mask_image dimensions must match initial_image dimensions when using ControlNet in sdxl.inpaint",
            ):
                _sdxl_inpaint(
                    {
                        "initial_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "mask_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "control_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "prompt": "test prompt",
                    },
                    _ctx=None,
                )

    def test_inpaint_controlnet_mismatched_control_dimensions_rejected(self):
        opened = [
            Image.new("RGB", (64, 64)),
            Image.new("L", (64, 64)),
            Image.new("RGB", (32, 32)),
        ]
        with patch("backend.workflow._open_image_ref", side_effect=opened):
            with self.assertRaisesRegex(
                ValueError,
                "control_image dimensions must match initial_image dimensions in sdxl.inpaint",
            ):
                _sdxl_inpaint(
                    {
                        "initial_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "mask_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "control_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "prompt": "test prompt",
                    },
                    _ctx=None,
                )

    def test_inpaint_warn_mode_returns_warning_on_unknown_preprocessor(self):
        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with patch(
                "backend.sdxl_pipeline.generate_inpaint_controlnet",
                return_value={"images": ["/outputs/batch/out.png"]},
            ):
                result = _sdxl_inpaint(
                    {
                        "initial_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "mask_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "control_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "prompt": "test prompt",
                        "controlnet_model": "diffusers/controlnet-canny-sdxl-1.0",
                        "controlnet_preprocessor_id": "unknown-id",
                        "controlnet_compat_mode": "warn",
                    },
                    _ctx=None,
                )
        self.assertIn("warnings", result)
        self.assertGreaterEqual(len(result["warnings"]), 1)

    def test_inpaint_without_controlnet_uses_default_pipeline(self):
        captured = {}

        def _fake_generate_inpaint(params):
            captured.update(params)
            return {"images": ["/outputs/batch/plain.png"]}

        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with patch(
                "backend.sdxl_pipeline.generate_inpaint",
                side_effect=_fake_generate_inpaint,
            ) as default_runner:
                with patch("backend.sdxl_pipeline.generate_inpaint_controlnet") as controlnet_runner:
                    result = _sdxl_inpaint(
                        {
                            "initial_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                            "mask_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                            "prompt": "test prompt",
                            "lora_adapters": [{"lora_id": 101, "strength": 0.8}],
                        },
                        _ctx=None,
                    )

        self.assertEqual(result["images"], ["/outputs/batch/plain.png"])
        default_runner.assert_called_once()
        controlnet_runner.assert_not_called()
        self.assertIn("lora_adapters", captured)
        self.assertEqual(captured["lora_adapters"], [{"lora_id": 101, "strength": 0.8}])

    def test_inpaint_without_controlnet_preserves_zero_padding_mask_crop(self):
        captured = {}

        def _fake_generate_inpaint(params):
            captured.update(params)
            return {"images": ["/outputs/batch/plain.png"]}

        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with patch("backend.sdxl_pipeline.generate_inpaint", side_effect=_fake_generate_inpaint):
                _sdxl_inpaint(
                    {
                        "initial_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "mask_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "prompt": "test prompt",
                        "padding_mask_crop": 0,
                    },
                    _ctx=None,
                )

        self.assertEqual(captured["padding_mask_crop"], 0)

    def test_inpaint_controlnet_preserves_zero_padding_mask_crop(self):
        captured = {}

        def _fake_generate_inpaint_controlnet(params):
            captured.update(params)
            return {"images": ["/outputs/batch/out.png"]}

        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
            with patch(
                "backend.sdxl_pipeline.generate_inpaint_controlnet",
                side_effect=_fake_generate_inpaint_controlnet,
            ):
                _sdxl_inpaint(
                    {
                        "initial_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "mask_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "control_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                        "prompt": "test prompt",
                        "padding_mask_crop": 0,
                    },
                    _ctx=None,
                )

        self.assertEqual(captured["padding_mask_crop"], 0)


if __name__ == "__main__":
    unittest.main()
