import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from backend import sdxl_pipeline


class _FakeLatents:
    def detach(self):
        return self

    def cpu(self):
        return self


class _FakePipe:
    def __init__(self):
        self.scheduler = None
        self.loaded_ip_adapter = None
        self.ip_adapter_scale = None
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


class SdxlIpAdapterPipelineTests(unittest.TestCase):
    def test_generate_text2img_loads_and_passes_ip_adapter_image(self):
        fake_pipe = _FakePipe()
        reference_image = Image.new("RGB", (16, 16))
        captured_render_kwargs = {}

        def _fake_render_text2img_latents(pipe, **kwargs):
            captured_render_kwargs.update(kwargs)
            return _FakeLatents()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.sdxl_pipeline.load_text2img_pipeline", return_value=fake_pipe):
                with patch("backend.sdxl_pipeline.create_scheduler", return_value="scheduler"):
                    with patch(
                        "backend.sdxl_pipeline.apply_lora_adapters_with_validation",
                        return_value=([], {}),
                    ):
                        with patch("backend.sdxl_pipeline.write_lora_coverage_report", return_value=None):
                            with patch(
                                "backend.sdxl_pipeline.render_text2img_latents",
                                side_effect=_fake_render_text2img_latents,
                            ):
                                with patch(
                                    "backend.sdxl_pipeline._decode_latents_to_pil",
                                    return_value=Image.new("RGB", (8, 8)),
                                ):
                                    with patch(
                                        "backend.sdxl_pipeline.make_batch_id",
                                        return_value="batch123",
                                    ):
                                        with patch(
                                            "backend.sdxl_pipeline.get_batch_output_dir",
                                            return_value=Path(tmpdir),
                                        ):
                                            result = sdxl_pipeline.generate_text2img(
                                                {
                                                    "prompt": "test prompt",
                                                    "negative_prompt": "",
                                                    "steps": 2,
                                                    "guidance_scale": 7.5,
                                                    "width": 64,
                                                    "height": 64,
                                                    "seed": 123,
                                                    "scheduler": "euler",
                                                    "model": "stable-diffusion-xl-base-1.0",
                                                    "num_images": 1,
                                                    "clip_skip": 1,
                                                    "ip_adapter_image": reference_image,
                                                    "ip_adapter_scale": 0.45,
                                                    "ip_adapter_model": "h94/IP-Adapter",
                                                    "ip_adapter_subfolder": "sdxl_models",
                                                    "ip_adapter_weight_name": "ip-adapter_sdxl.bin",
                                                }
                                            )

        self.assertEqual(
            result,
            {"images": ["/outputs/batch_batch123/batch123_123.png"]},
        )
        self.assertEqual(
            fake_pipe.loaded_ip_adapter,
            {
                "model": "h94/IP-Adapter",
                "subfolder": "sdxl_models",
                "weight_name": "ip-adapter_sdxl.bin",
            },
        )
        self.assertEqual(fake_pipe.ip_adapter_scale, 0.45)
        self.assertIs(captured_render_kwargs["ip_adapter_image"], reference_image)
        self.assertTrue(fake_pipe.unloaded_ip_adapter)

    def test_generate_img2img_loads_and_passes_ip_adapter_image(self):
        fake_pipe = _FakePipe()
        initial_image = Image.new("RGB", (16, 16))
        reference_image = Image.new("RGB", (16, 16))
        captured_render_kwargs = {}

        def _fake_render_img2img_latents(pipe, **kwargs):
            captured_render_kwargs.update(kwargs)
            return _FakeLatents()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.sdxl_pipeline.load_img2img_pipeline", return_value=fake_pipe):
                with patch("backend.sdxl_pipeline.create_scheduler", return_value="scheduler"):
                    with patch(
                        "backend.sdxl_pipeline.apply_lora_adapters_with_validation",
                        return_value=([], {}),
                    ):
                        with patch("backend.sdxl_pipeline.write_lora_coverage_report", return_value=None):
                            with patch(
                                "backend.sdxl_pipeline.render_img2img_latents",
                                side_effect=_fake_render_img2img_latents,
                            ):
                                with patch(
                                    "backend.sdxl_pipeline._decode_latents_to_pil",
                                    return_value=Image.new("RGB", (8, 8)),
                                ):
                                    with patch(
                                        "backend.sdxl_pipeline.make_batch_id",
                                        return_value="batch123",
                                    ):
                                        with patch(
                                            "backend.sdxl_pipeline.get_batch_output_dir",
                                            return_value=Path(tmpdir),
                                        ):
                                            result = sdxl_pipeline.generate_img2img(
                                                {
                                                    "initial_image": initial_image,
                                                    "strength": 0.65,
                                                    "prompt": "test prompt",
                                                    "negative_prompt": "",
                                                    "steps": 2,
                                                    "guidance_scale": 7.5,
                                                    "width": 64,
                                                    "height": 64,
                                                    "seed": 123,
                                                    "scheduler": "euler",
                                                    "model": "stable-diffusion-xl-base-1.0",
                                                    "num_images": 1,
                                                    "clip_skip": 1,
                                                    "lora_adapters": [],
                                                    "ip_adapter_image": reference_image,
                                                    "ip_adapter_scale": 0.45,
                                                    "ip_adapter_model": "h94/IP-Adapter",
                                                    "ip_adapter_subfolder": "sdxl_models",
                                                    "ip_adapter_weight_name": "ip-adapter_sdxl.bin",
                                                }
                                            )

        self.assertEqual(
            result,
            {"images": ["/outputs/batch_batch123/batch123_123.png"]},
        )
        self.assertEqual(
            fake_pipe.loaded_ip_adapter,
            {
                "model": "h94/IP-Adapter",
                "subfolder": "sdxl_models",
                "weight_name": "ip-adapter_sdxl.bin",
            },
        )
        self.assertEqual(fake_pipe.ip_adapter_scale, 0.45)
        self.assertIs(captured_render_kwargs["ip_adapter_image"], reference_image)
        self.assertTrue(fake_pipe.unloaded_ip_adapter)

    def test_generate_inpaint_loads_and_passes_ip_adapter_image(self):
        fake_pipe = _FakePipe()
        initial_image = Image.new("RGB", (16, 16))
        mask_image = Image.new("L", (16, 16))
        reference_image = Image.new("RGB", (16, 16))
        captured_render_kwargs = {}

        def _fake_render_inpaint_image(pipe, **kwargs):
            captured_render_kwargs.update(kwargs)
            return Image.new("RGB", (8, 8))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.sdxl_pipeline.load_inpaint_pipeline", return_value=fake_pipe):
                with patch("backend.sdxl_pipeline.create_scheduler", return_value="scheduler"):
                    with patch(
                        "backend.sdxl_pipeline.apply_lora_adapters_with_validation",
                        return_value=([], {}),
                    ):
                        with patch("backend.sdxl_pipeline.write_lora_coverage_report", return_value=None):
                            with patch(
                                "backend.sdxl_pipeline.render_inpaint_image",
                                side_effect=_fake_render_inpaint_image,
                            ):
                                with patch(
                                    "backend.sdxl_pipeline.make_batch_id",
                                    return_value="batch123",
                                ):
                                    with patch(
                                        "backend.sdxl_pipeline.get_batch_output_dir",
                                        return_value=Path(tmpdir),
                                    ):
                                        result = sdxl_pipeline.generate_inpaint(
                                            {
                                                "initial_image": initial_image,
                                                "mask_image": mask_image,
                                                "strength": 0.65,
                                                "prompt": "test prompt",
                                                "negative_prompt": "",
                                                "steps": 2,
                                                "guidance_scale": 7.5,
                                                "seed": 123,
                                                "scheduler": "euler",
                                                "model": "stable-diffusion-xl-base-1.0",
                                                "num_images": 1,
                                                "padding_mask_crop": 32,
                                                "clip_skip": 1,
                                                "lora_adapters": [],
                                                "ip_adapter_image": reference_image,
                                                "ip_adapter_scale": 0.45,
                                                "ip_adapter_model": "h94/IP-Adapter",
                                                "ip_adapter_subfolder": "sdxl_models",
                                                "ip_adapter_weight_name": "ip-adapter_sdxl.bin",
                                            }
                                        )

        self.assertEqual(
            result,
            {"images": ["/outputs/batch_batch123/batch123_123.png"]},
        )
        self.assertEqual(
            fake_pipe.loaded_ip_adapter,
            {
                "model": "h94/IP-Adapter",
                "subfolder": "sdxl_models",
                "weight_name": "ip-adapter_sdxl.bin",
            },
        )
        self.assertEqual(fake_pipe.ip_adapter_scale, 0.45)
        self.assertIs(captured_render_kwargs["ip_adapter_image"], reference_image)
        self.assertTrue(fake_pipe.unloaded_ip_adapter)


if __name__ == "__main__":
    unittest.main()
