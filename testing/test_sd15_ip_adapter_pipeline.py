import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from backend import sd15_pipeline


class _FakeGenerator:
    def __init__(self, device=None):
        self.device = device
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        return self


class _FakePipe:
    def __init__(self):
        self.scheduler = None
        self.loaded_ip_adapter = None
        self.ip_adapter_scale = None
        self.call_kwargs = []
        self.unloaded_ip_adapter = False

    def to(self, device):
        self.device = device
        return self

    def load_ip_adapter(self, model, *, subfolder, weight_name):
        self.loaded_ip_adapter = {
            "model": model,
            "subfolder": subfolder,
            "weight_name": weight_name,
        }

    def set_ip_adapter_scale(self, scale):
        self.ip_adapter_scale = scale

    def unload_ip_adapter(self):
        self.unloaded_ip_adapter = True

    def __call__(self, **kwargs):
        self.call_kwargs.append(kwargs)
        return type("Result", (), {"images": [Image.new("RGB", (8, 8))]})()


class Sd15IpAdapterPipelineTests(unittest.TestCase):
    def test_generate_images_loads_and_passes_ip_adapter_image(self):
        fake_pipe = _FakePipe()
        reference_image = Image.new("RGB", (16, 16))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.sd15_pipeline.load_text2img_pipeline", return_value=fake_pipe):
                with patch("backend.sd15_pipeline.create_scheduler", return_value="scheduler"):
                    with patch(
                        "backend.sd15_pipeline.build_prompt_embeddings",
                        return_value=(None, None, False),
                    ):
                        with patch("backend.sd15_pipeline.torch.Generator", _FakeGenerator):
                            with patch(
                                "backend.sd15_pipeline.get_batch_output_dir",
                                return_value=Path(tmpdir),
                            ):
                                filenames = sd15_pipeline.generate_images(
                                    {
                                        "prompt": "test prompt",
                                        "negative_prompt": "",
                                        "steps": 2,
                                        "cfg": 7.5,
                                        "width": 64,
                                        "height": 64,
                                        "seed": 123,
                                        "scheduler": "euler",
                                        "model": "stable-diffusion-v1-5",
                                        "num_images": 1,
                                        "clip_skip": 1,
                                        "ip_adapter_image": reference_image,
                                        "ip_adapter_scale": 0.45,
                                        "ip_adapter_model": "h94/IP-Adapter",
                                        "ip_adapter_subfolder": "models",
                                        "ip_adapter_weight_name": "ip-adapter_sd15.bin",
                                        "batch_id": "batch123",
                                    }
                                )

        self.assertEqual(filenames, ["batch_batch123/batch123_123.png"])
        self.assertEqual(
            fake_pipe.loaded_ip_adapter,
            {
                "model": "h94/IP-Adapter",
                "subfolder": "models",
                "weight_name": "ip-adapter_sd15.bin",
            },
        )
        self.assertEqual(fake_pipe.ip_adapter_scale, 0.45)
        self.assertIs(fake_pipe.call_kwargs[0]["ip_adapter_image"], reference_image)
        self.assertTrue(fake_pipe.unloaded_ip_adapter)

    def test_generate_images_img2img_loads_and_passes_ip_adapter_image(self):
        fake_pipe = _FakePipe()
        reference_image = Image.new("RGB", (16, 16))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.sd15_pipeline.load_img2img_pipeline", return_value=fake_pipe):
                with patch("backend.sd15_pipeline.create_scheduler", return_value="scheduler"):
                    with patch("backend.sd15_pipeline.torch.Generator", _FakeGenerator):
                        with patch(
                            "backend.sd15_pipeline.get_batch_output_dir",
                            return_value=Path(tmpdir),
                        ):
                            filenames = sd15_pipeline.generate_images_img2img(
                                {
                                    "initial_image": Image.new("RGB", (32, 32)),
                                    "prompt": "test prompt",
                                    "negative_prompt": "",
                                    "steps": 2,
                                    "cfg": 7.5,
                                    "width": 32,
                                    "height": 32,
                                    "seed": 123,
                                    "scheduler": "euler",
                                    "model": "stable-diffusion-v1-5",
                                    "num_images": 1,
                                    "clip_skip": 1,
                                    "ip_adapter_image": reference_image,
                                    "ip_adapter_scale": 0.45,
                                    "ip_adapter_model": "h94/IP-Adapter",
                                    "ip_adapter_subfolder": "models",
                                    "ip_adapter_weight_name": "ip-adapter_sd15.bin",
                                    "batch_id": "batch123",
                                }
                            )

        self.assertEqual(filenames, ["batch_batch123/batch123_123.png"])
        self.assertEqual(
            fake_pipe.loaded_ip_adapter,
            {
                "model": "h94/IP-Adapter",
                "subfolder": "models",
                "weight_name": "ip-adapter_sd15.bin",
            },
        )
        self.assertEqual(fake_pipe.ip_adapter_scale, 0.45)
        self.assertIs(fake_pipe.call_kwargs[0]["ip_adapter_image"], reference_image)
        self.assertTrue(fake_pipe.unloaded_ip_adapter)

    def test_generate_images_inpaint_loads_and_passes_ip_adapter_image(self):
        fake_pipe = _FakePipe()
        reference_image = Image.new("RGB", (16, 16))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.sd15_pipeline.load_inpaint_pipeline", return_value=fake_pipe):
                with patch("backend.sd15_pipeline.create_scheduler", return_value="scheduler"):
                    with patch("backend.sd15_pipeline.torch.Generator", _FakeGenerator):
                        with patch("backend.sd15_pipeline._apply_lora_adapters", return_value=[]):
                            with patch(
                                "backend.sd15_pipeline.get_batch_output_dir",
                                return_value=Path(tmpdir),
                            ):
                                filenames = sd15_pipeline.generate_images_inpaint(
                                    {
                                        "initial_image": Image.new("RGB", (32, 32)),
                                        "mask_image": Image.new("L", (32, 32)),
                                        "prompt": "test prompt",
                                        "negative_prompt": "",
                                        "steps": 2,
                                        "cfg": 7.5,
                                        "seed": 123,
                                        "scheduler": "euler",
                                        "model": "stable-diffusion-v1-5",
                                        "num_images": 1,
                                        "strength": 0.5,
                                        "padding_mask_crop": 32,
                                        "clip_skip": 1,
                                        "ip_adapter_image": reference_image,
                                        "ip_adapter_scale": 0.45,
                                        "ip_adapter_model": "h94/IP-Adapter",
                                        "ip_adapter_subfolder": "models",
                                        "ip_adapter_weight_name": "ip-adapter_sd15.bin",
                                        "batch_id": "batch123",
                                    }
                                )

        self.assertEqual(filenames, ["batch_batch123/batch123_123.png"])
        self.assertEqual(
            fake_pipe.loaded_ip_adapter,
            {
                "model": "h94/IP-Adapter",
                "subfolder": "models",
                "weight_name": "ip-adapter_sd15.bin",
            },
        )
        self.assertEqual(fake_pipe.ip_adapter_scale, 0.45)
        self.assertIs(fake_pipe.call_kwargs[0]["ip_adapter_image"], reference_image)
        self.assertTrue(fake_pipe.unloaded_ip_adapter)


if __name__ == "__main__":
    unittest.main()
