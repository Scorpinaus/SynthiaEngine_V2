import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from backend.workflow import (
    ZImageInpaintInputs,
    ZImageImg2ImgInputs,
    ZImageText2ImgInputs,
    _z_image_inpaint,
    _z_image_img2img,
    _z_image_text2img,
)


class ZImageWorkflowTests(unittest.TestCase):
    def test_z_image_text2img_accepts_lora_adapters(self):
        inputs = ZImageText2ImgInputs(
            prompt="test",
            lora_adapters=[{"lora_id": 101, "strength": 0.8}],
        )
        self.assertEqual(inputs.lora_adapters, [{"lora_id": 101, "strength": 0.8}])

    def test_z_image_text2img_forwards_lora_adapters(self):
        captured = {}

        def _fake_run_z_image_text2img(payload):
            captured.update(payload)
            return {"images": ["/outputs/batch/out.png"]}

        fake_module = SimpleNamespace(run_z_image_text2img=_fake_run_z_image_text2img)
        with patch.dict("sys.modules", {"backend.z_image_pipeline": fake_module}):
            result = _z_image_text2img(
                {
                    "prompt": "test prompt",
                    "lora_adapters": [{"lora_id": 101, "strength": 0.8}],
                },
                _ctx=None,
            )

        self.assertEqual(result["images"], ["/outputs/batch/out.png"])
        self.assertIn("lora_adapters", captured)
        self.assertEqual(captured["lora_adapters"], [{"lora_id": 101, "strength": 0.8}])

    def test_z_image_img2img_accepts_lora_adapters(self):
        inputs = ZImageImg2ImgInputs(
            initial_image="@artifact:a0123456789abcdef0123456789abcdef",
            prompt="test",
            lora_adapters=[{"lora_id": 101, "strength": 0.7}],
        )
        self.assertEqual(inputs.lora_adapters, [{"lora_id": 101, "strength": 0.7}])

    def test_z_image_img2img_forwards_lora_adapters(self):
        captured = {}

        def _fake_run_z_image_img2img(payload):
            captured.update(payload)
            return {"images": ["/outputs/batch/out.png"]}

        fake_module = SimpleNamespace(run_z_image_img2img=_fake_run_z_image_img2img)
        with patch.dict("sys.modules", {"backend.z_image_pipeline": fake_module}), patch(
            "backend.workflow._open_image_ref"
        ) as mock_open_image:
            fake_image = SimpleNamespace(
                convert=lambda _mode: SimpleNamespace(
                    resize=lambda _size: SimpleNamespace(),
                )
            )
            mock_open_image.return_value = fake_image
            result = _z_image_img2img(
                {
                    "initial_image": "@artifact:a0123456789abcdef0123456789abcdef",
                    "prompt": "test prompt",
                    "lora_adapters": [{"lora_id": 101, "strength": 0.7}],
                },
                _ctx=None,
            )

        self.assertEqual(result["images"], ["/outputs/batch/out.png"])
        self.assertIn("lora_adapters", captured)
        self.assertEqual(captured["lora_adapters"], [{"lora_id": 101, "strength": 0.7}])

    def test_z_image_inpaint_accepts_lora_adapters(self):
        inputs = ZImageInpaintInputs(
            initial_image="@artifact:a0123456789abcdef0123456789abcdef",
            mask_image="@artifact:p0123456789abcdef0123456789abcdef",
            prompt="test",
            lora_adapters=[{"lora_id": 101, "strength": 0.7}],
        )
        self.assertEqual(inputs.lora_adapters, [{"lora_id": 101, "strength": 0.7}])

    def test_z_image_inpaint_forwards_lora_adapters(self):
        captured = {}

        def _fake_run_z_image_inpaint(**kwargs):
            captured.update(kwargs)
            return {"images": ["/outputs/batch/out.png"]}

        fake_module = SimpleNamespace(run_z_image_inpaint=_fake_run_z_image_inpaint)
        with patch.dict("sys.modules", {"backend.z_image_pipeline": fake_module}), patch(
            "backend.workflow._open_image_ref",
            return_value=Image.new("RGB", (64, 64)),
        ):
            result = _z_image_inpaint(
                {
                    "initial_image": "@artifact:a0123456789abcdef0123456789abcdef",
                    "mask_image": "@artifact:p0123456789abcdef0123456789abcdef",
                    "prompt": "test prompt",
                    "lora_adapters": [{"lora_id": 101, "strength": 0.7}],
                },
                _ctx=None,
            )

        self.assertEqual(result["images"], ["/outputs/batch/out.png"])
        self.assertIn("lora_adapters", captured)
        self.assertEqual(captured["lora_adapters"], [{"lora_id": 101, "strength": 0.7}])


if __name__ == "__main__":
    unittest.main()
