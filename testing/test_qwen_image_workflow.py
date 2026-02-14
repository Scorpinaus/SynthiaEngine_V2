import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from backend.workflow import (
    QwenImageInpaintInputs,
    QwenImageImg2ImgInputs,
    QwenImageText2ImgInputs,
    _qwen_image_inpaint,
    _qwen_image_img2img,
    _qwen_image_text2img,
)


class QwenImageWorkflowTests(unittest.TestCase):
    def test_qwen_image_text2img_accepts_lora_adapters(self):
        inputs = QwenImageText2ImgInputs(
            prompt="test",
            lora_adapters=[{"lora_id": 101, "strength": 0.8}],
        )
        self.assertEqual(inputs.lora_adapters, [{"lora_id": 101, "strength": 0.8}])

    def test_qwen_image_text2img_forwards_lora_adapters(self):
        captured = {}

        def _fake_run_qwen_image_text2img(payload):
            captured.update(payload)
            return {"images": ["/outputs/batch/out.png"]}

        fake_module = SimpleNamespace(run_qwen_image_text2img=_fake_run_qwen_image_text2img)
        with patch.dict("sys.modules", {"backend.qwen_image_pipeline": fake_module}):
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

    def test_qwen_image_img2img_accepts_lora_adapters(self):
        inputs = QwenImageImg2ImgInputs(
            initial_image="@artifact:abc123",
            prompt="test",
            lora_adapters=[{"lora_id": 101, "strength": 0.8}],
        )
        self.assertEqual(inputs.lora_adapters, [{"lora_id": 101, "strength": 0.8}])

    def test_qwen_image_img2img_forwards_lora_adapters(self):
        captured = {}

        def _fake_run_qwen_image_img2img(payload):
            captured.update(payload)
            return {"images": ["/outputs/batch/out.png"]}

        fake_module = SimpleNamespace(run_qwen_image_img2img=_fake_run_qwen_image_img2img)
        with patch.dict("sys.modules", {"backend.qwen_image_pipeline": fake_module}):
            with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
                result = _qwen_image_img2img(
                    {
                        "initial_image": "@artifact:abc123",
                        "prompt": "test prompt",
                        "lora_adapters": [{"lora_id": 101, "strength": 0.8}],
                    },
                    _ctx=None,
                )

        self.assertEqual(result["images"], ["/outputs/batch/out.png"])
        self.assertIn("lora_adapters", captured)
        self.assertEqual(captured["lora_adapters"], [{"lora_id": 101, "strength": 0.8}])

    def test_qwen_image_inpaint_accepts_lora_adapters(self):
        inputs = QwenImageInpaintInputs(
            initial_image="@artifact:abc123",
            mask_image="@artifact:def456",
            prompt="test",
            lora_adapters=[{"lora_id": 101, "strength": 0.8}],
        )
        self.assertEqual(inputs.lora_adapters, [{"lora_id": 101, "strength": 0.8}])

    def test_qwen_image_inpaint_forwards_lora_adapters(self):
        captured = {}

        def _fake_run_qwen_image_inpaint(payload):
            captured.update(payload)
            return {"images": ["/outputs/batch/out.png"]}

        fake_module = SimpleNamespace(run_qwen_image_inpaint=_fake_run_qwen_image_inpaint)
        with patch.dict("sys.modules", {"backend.qwen_image_pipeline": fake_module}):
            with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))):
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
