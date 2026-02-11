import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from backend.workflow import FluxImg2ImgInputs, FluxText2ImgInputs, _flux_img2img, _flux_text2img


class FluxWorkflowTests(unittest.TestCase):
    def test_flux_text2img_accepts_lora_adapters(self):
        inputs = FluxText2ImgInputs(
            prompt="test",
            lora_adapters=[{"lora_id": 101, "strength": 0.8}],
        )
        self.assertEqual(inputs.lora_adapters, [{"lora_id": 101, "strength": 0.8}])

    def test_flux_text2img_forwards_lora_adapters(self):
        captured = {}

        def _fake_run_flux_text2img(payload):
            captured.update(payload)
            return {"images": ["/outputs/batch/out.png"]}

        fake_module = SimpleNamespace(run_flux_text2img=_fake_run_flux_text2img)
        with patch.dict("sys.modules", {"backend.flux_pipeline": fake_module}):
            result = _flux_text2img(
                {
                    "prompt": "test prompt",
                    "lora_adapters": [{"lora_id": 101, "strength": 0.8}],
                },
                _ctx=None,
            )

        self.assertEqual(result["images"], ["/outputs/batch/out.png"])
        self.assertIn("lora_adapters", captured)
        self.assertEqual(captured["lora_adapters"], [{"lora_id": 101, "strength": 0.8}])

    def test_flux_img2img_accepts_lora_adapters(self):
        inputs = FluxImg2ImgInputs(
            initial_image="@artifact:a0123456789abcdef0123456789abcdef",
            prompt="test",
            lora_adapters=[{"lora_id": 201, "strength": 0.6}],
        )
        self.assertEqual(inputs.lora_adapters, [{"lora_id": 201, "strength": 0.6}])

    def test_flux_img2img_forwards_lora_adapters(self):
        captured = {}

        def _fake_run_flux_img2img(**kwargs):
            captured.update(kwargs)
            return {"images": ["/outputs/batch/out.png"]}

        fake_module = SimpleNamespace(run_flux_img2img=_fake_run_flux_img2img)
        with (
            patch.dict("sys.modules", {"backend.flux_pipeline": fake_module}),
            patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (64, 64))),
        ):
            result = _flux_img2img(
                {
                    "initial_image": "@artifact:a0123456789abcdef0123456789abcdef",
                    "prompt": "test prompt",
                    "lora_adapters": [{"lora_id": 201, "strength": 0.6}],
                    "width": 1024,
                    "height": 1024,
                },
                _ctx=None,
            )

        self.assertEqual(result["images"], ["/outputs/batch/out.png"])
        self.assertIn("lora_adapters", captured)
        self.assertEqual(captured["lora_adapters"], [{"lora_id": 201, "strength": 0.6}])


if __name__ == "__main__":
    unittest.main()
