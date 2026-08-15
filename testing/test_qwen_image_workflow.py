import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from backend.workflow import (
    QwenImageInpaintInputs,
    QwenImageImg2ImgInputs,
    QwenImageText2ImgInputs,
)
from backend.workflow.assembly import (
    _qwen_image_inpaint,
    _qwen_image_img2img,
    _qwen_image_text2img,
)
from backend.workflow.types import WorkflowContext


class QwenImageWorkflowTests(unittest.TestCase):
    def test_qwen_image_text2img_accepts_lora_adapters(self):
        inputs = QwenImageText2ImgInputs(
            prompt="test",
            lora_adapters=[{"lora_id": 101}],
        )

        self.assertEqual(
            inputs.lora_adapters[0].model_dump(),
            {"lora_id": 101, "strength": 0.8, "target": "both"},
        )

    def test_qwen_image_text2img_accepts_multiple_lora_adapters(self):
        inputs = QwenImageText2ImgInputs(
            prompt="test",
            lora_adapters=[
                {"lora_id": 101, "strength": 0.65},
                {"lora_id": 102, "strength": 0.35, "target": "both"},
            ],
        )

        self.assertEqual(
            [adapter.model_dump() for adapter in inputs.lora_adapters],
            [
                {"lora_id": 101, "strength": 0.65, "target": "both"},
                {"lora_id": 102, "strength": 0.35, "target": "both"},
            ],
        )

    def test_qwen_image_text2img_accepts_empty_lora_adapters(self):
        self.assertIsNone(QwenImageText2ImgInputs(prompt="test").lora_adapters)
        self.assertEqual(
            QwenImageText2ImgInputs(prompt="test", lora_adapters=[]).lora_adapters,
            [],
        )

    def test_qwen_image_text2img_rejects_component_lora_options(self):
        invalid_adapters = (
            {"lora_id": 101, "target": "unet"},
            {"lora_id": 101, "target": "text_encoder"},
            {"lora_id": 101, "unet_strength": 0.8},
            {"lora_id": 101, "text_encoder_strength": 0.8},
            {"lora_id": 101, "unet_scales": {"down": 0.8}},
            {"lora_id": 101, "text_encoder_scales": {"text_encoder": 0.8}},
        )

        for adapter in invalid_adapters:
            with self.subTest(adapter=adapter), self.assertRaises(ValueError):
                QwenImageText2ImgInputs(prompt="test", lora_adapters=[adapter])

    def test_qwen_image_text2img_rejects_lora_strength_outside_range(self):
        for strength in (-0.01, 1.01):
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                QwenImageText2ImgInputs(
                    prompt="test",
                    lora_adapters=[{"lora_id": 101, "strength": strength}],
                )

    def test_qwen_image_text2img_forwards_lora_adapters(self):
        captured = {}

        def _fake_generate_text2img(payload):
            captured.update(payload)
            return {"images": ["/outputs/batch/out.png"]}

        fake_module = SimpleNamespace(generate_text2img=_fake_generate_text2img)
        with patch.dict("sys.modules", {"backend.qwen_image.pipeline": fake_module}):
            result = _qwen_image_text2img(
                {
                    "prompt": "test prompt",
                    "lora_adapters": [{"lora_id": 101, "strength": 0.8}],
                },
                _ctx=None,
            )

        self.assertEqual(result["images"], ["/outputs/batch/out.png"])
        self.assertIn("lora_adapters", captured)
        self.assertEqual(captured["lora_adapters"], [{"lora_id": 101, "strength": 0.8}])

    def test_qwen_image_text2img_binds_workflow_runtime_callbacks(self):
        captured = {}
        update_progress = lambda _patch: None
        should_cancel = lambda: False

        def _fake_generate_text2img(payload, **runtime):
            captured.update(runtime)
            return {"images": ["/outputs/batch/out.png"]}

        fake_module = SimpleNamespace(generate_text2img=_fake_generate_text2img)
        with patch.dict("sys.modules", {"backend.qwen_image.pipeline": fake_module}):
            _qwen_image_text2img(
                {"prompt": "test prompt"},
                WorkflowContext(
                    update_progress=update_progress,
                    should_cancel=should_cancel,
                ),
            )

        self.assertIs(captured["update_progress"], update_progress)
        self.assertIs(captured["should_cancel"], should_cancel)

    def test_qwen_image_text2img_accepts_legacy_lora_contract_enabled(self):
        captured = {}

        def _fake_generate_text2img(payload):
            captured.update(payload)
            return {"images": ["/outputs/batch/out.png"]}

        fake_module = SimpleNamespace(generate_text2img=_fake_generate_text2img)
        with patch.dict("sys.modules", {"backend.qwen_image.pipeline": fake_module}):
            result = _qwen_image_text2img(
                {
                    "prompt": "test prompt",
                    "Lora": {"enabled": True, "adapters": [{"lora_id": 102, "strength": 0.6}]},
                },
                _ctx=None,
            )

        self.assertEqual(result["images"], ["/outputs/batch/out.png"])
        self.assertEqual(captured.get("lora_adapters"), [{"lora_id": 102, "strength": 0.6}])

    def test_qwen_image_text2img_honors_legacy_lora_contract_disabled(self):
        captured = {}

        def _fake_generate_text2img(payload):
            captured.update(payload)
            return {"images": ["/outputs/batch/out.png"]}

        fake_module = SimpleNamespace(generate_text2img=_fake_generate_text2img)
        with patch.dict("sys.modules", {"backend.qwen_image.pipeline": fake_module}):
            result = _qwen_image_text2img(
                {
                    "prompt": "test prompt",
                    "Lora": {"enabled": False, "adapters": [{"lora_id": 103, "strength": 1.0}]},
                },
                _ctx=None,
            )

        self.assertEqual(result["images"], ["/outputs/batch/out.png"])
        self.assertEqual(captured.get("lora_adapters"), [])

    def test_qwen_image_img2img_accepts_lora_adapters(self):
        inputs = QwenImageImg2ImgInputs(
            initial_image="@artifact:abc123",
            prompt="test",
            lora_adapters=[{"lora_id": 101, "strength": 0.7, "target": "both"}],
        )

        self.assertEqual(inputs.lora_adapters[0].strength, 0.7)

    def test_qwen_image_img2img_forwards_lora_adapters(self):
        captured = {}

        def _fake_generate_img2img(payload):
            captured.update(payload)
            return {"images": ["/outputs/batch/out.png"]}

        fake_module = SimpleNamespace(generate_img2img=_fake_generate_img2img)
        with patch.dict("sys.modules", {"backend.qwen_image.pipeline": fake_module}):
            with patch("backend.workflow.assembly._open_image_ref", return_value=Image.new("RGB", (64, 64))):
                result = _qwen_image_img2img(
                    {
                        "initial_image": "@artifact:abc123",
                        "prompt": "test prompt",
                        "lora_adapters": [{"lora_id": 101, "strength": 0.8}],
                        "live_preview": False,
                    },
                    _ctx=None,
                )

        self.assertEqual(result["images"], ["/outputs/batch/out.png"])
        self.assertIn("lora_adapters", captured)
        self.assertEqual(captured["lora_adapters"], [{"lora_id": 101, "strength": 0.8}])
        self.assertIs(captured["live_preview"], False)

    def test_qwen_image_inpaint_accepts_lora_adapters(self):
        inputs = QwenImageInpaintInputs(
            initial_image="@artifact:abc123",
            mask_image="@artifact:def456",
            prompt="test",
            lora_adapters=[{"lora_id": 101, "strength": 0.6}],
        )

        self.assertEqual(inputs.lora_adapters[0].strength, 0.6)

    def test_qwen_image_inpaint_forwards_size_and_mask_crop(self):
        captured = {}
        initial_image = Image.new("RGB", (64, 48))
        mask_image = Image.new("L", (64, 48))

        def _fake_generate_inpaint(payload):
            captured.update(payload)
            return {"images": ["/outputs/batch/out.png"]}

        def _open_image_ref(reference):
            return mask_image if reference == "@artifact:def456" else initial_image

        fake_module = SimpleNamespace(generate_inpaint=_fake_generate_inpaint)
        with (
            patch.dict("sys.modules", {"backend.qwen_image.pipeline": fake_module}),
            patch(
                "backend.workflow.assembly._open_image_ref",
                side_effect=_open_image_ref,
            ),
        ):
            result = _qwen_image_inpaint(
                {
                    "initial_image": "@artifact:abc123",
                    "mask_image": "@artifact:def456",
                    "prompt": "test prompt",
                    "width": 768,
                    "height": 1024,
                    "padding_mask_crop": 0,
                    "live_preview": False,
                },
                _ctx=None,
            )

        self.assertEqual(result["images"], ["/outputs/batch/out.png"])
        self.assertEqual(captured["width"], 768)
        self.assertEqual(captured["height"], 1024)
        self.assertEqual(captured["padding_mask_crop"], 0)
        self.assertIs(captured["live_preview"], False)
        self.assertEqual(captured["initial_image"].size, (64, 48))
        self.assertEqual(captured["mask_image"].size, (64, 48))

    def test_qwen_image_inpaint_forwards_lora_adapters(self):
        captured = {}

        def _fake_generate_inpaint(payload):
            captured.update(payload)
            return {"images": ["/outputs/batch/out.png"]}

        fake_module = SimpleNamespace(generate_inpaint=_fake_generate_inpaint)
        with patch.dict("sys.modules", {"backend.qwen_image.pipeline": fake_module}):
            with patch("backend.workflow.assembly._open_image_ref", return_value=Image.new("RGB", (64, 64))):
                result = _qwen_image_inpaint(
                    {
                        "initial_image": "@artifact:abc123",
                        "mask_image": "@artifact:def456",
                        "prompt": "test prompt",
                        "lora_adapters": [{"lora_id": 101, "strength": 0.8}],
                    },
                    _ctx=None,
                )

        self.assertEqual(result["images"], ["/outputs/batch/out.png"])
        self.assertIn("lora_adapters", captured)
        self.assertEqual(captured["lora_adapters"], [{"lora_id": 101, "strength": 0.8}])


if __name__ == "__main__":
    unittest.main()
