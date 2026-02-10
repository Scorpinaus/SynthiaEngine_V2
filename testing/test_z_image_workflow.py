import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.workflow import ZImageText2ImgInputs, _z_image_text2img


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


if __name__ == "__main__":
    unittest.main()
