import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from backend.sd15 import ip_adapter_pipeline as sd15_ip_adapter_pipeline
from backend.sd15 import pipeline as sd15_pipeline


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
        self.load_ip_adapter_kwargs = None
        self.ip_adapter_scale = None
        self.call_kwargs = []
        self.image_encoder = object()
        self.image_encoder_during_call = None
        self.unloaded_ip_adapter = False
        self.prepare_ip_adapter_image_embeds_kwargs = None
        self.prepared_ip_adapter_image_embeds = ["prepared-ip-adapter-embeds"]

    def to(self, device):
        self.device = device
        return self

    def load_ip_adapter(self, model, *, subfolder, weight_name, **kwargs):
        self.loaded_ip_adapter = {
            "model": model,
            "subfolder": subfolder,
            "weight_name": weight_name,
        }
        self.load_ip_adapter_kwargs = kwargs

    def set_ip_adapter_scale(self, scale):
        self.ip_adapter_scale = scale

    def unload_ip_adapter(self):
        self.unloaded_ip_adapter = True

    def prepare_ip_adapter_image_embeds(self, **kwargs):
        self.prepare_ip_adapter_image_embeds_kwargs = kwargs
        return self.prepared_ip_adapter_image_embeds

    def __call__(self, **kwargs):
        self.call_kwargs.append(kwargs)
        self.image_encoder_during_call = self.image_encoder
        return type("Result", (), {"images": [Image.new("RGB", (8, 8))]})()


class _FakePixelValues:
    def __init__(self):
        self.to_kwargs = None

    def to(self, **kwargs):
        self.to_kwargs = kwargs
        return self


class _FakeCLIPImageProcessor:
    pixel_values = None

    def __init__(self, **_kwargs):
        pass

    def __call__(self, image, *, return_tensors):
        type(self).pixel_values = _FakePixelValues()
        return type("ProcessedImage", (), {"pixel_values": type(self).pixel_values})()


class _FakeImageEncoder:
    def __init__(self):
        self.config = type("Config", (), {"image_size": 224})()
        self.to_devices = []

    def to(self, device):
        self.to_devices.append(device)
        return self

    def __call__(self, pixel_values):
        return type(
            "ImageEncoderOutput",
            (),
            {"image_embeds": torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float16)},
        )()


class Sd15IpAdapterPipelineTests(unittest.TestCase):
    def test_generate_ip_adapter_image_embeds_uses_minimal_encoder(self):
        image = Image.new("RGB", (16, 16))
        fake_encoder = _FakeImageEncoder()
        saved = {}

        def _fake_save_embeds(embeds, *, family, metadata):
            saved["embeds"] = embeds
            saved["family"] = family
            saved["metadata"] = metadata
            return {
                "artifact_id": "e123",
                "path": "artifacts/e123.pt",
                "url": "/outputs/artifacts/e123.pt",
            }

        with patch("backend.sd15.ip_adapter_pipeline.torch.cuda.is_available", return_value=True):
            with patch("backend.sd15.ip_adapter_pipeline.hf_hub_download", return_value="adapter.bin") as download:
                with patch(
                    "backend.sd15.ip_adapter_pipeline.load_state_dict",
                    return_value={
                        "image_proj": {"proj.weight": torch.zeros(3072, 1024)},
                        "ip_adapter": {},
                    },
                ):
                    with patch(
                        "backend.sd15.ip_adapter_pipeline.CLIPVisionModelWithProjection.from_pretrained",
                        return_value=fake_encoder,
                    ) as from_pretrained:
                        with patch(
                            "backend.sd15.ip_adapter_pipeline.CLIPImageProcessor",
                            _FakeCLIPImageProcessor,
                        ):
                            with patch(
                                "backend.sd15.ip_adapter_pipeline.save_ip_adapter_embeds_artifact",
                                side_effect=_fake_save_embeds,
                            ):
                                with patch("backend.sd15.ip_adapter_pipeline.cleanup_memory") as cleanup:
                                    result = sd15_ip_adapter_pipeline.generate_ip_adapter_image_embeds(
                                        {
                                            "image": image,
                                            "model": "stable-diffusion-v1-5",
                                            "guidance_scale": 7.5,
                                            "ip_adapter_model": "h94/IP-Adapter",
                                            "ip_adapter_subfolder": "models",
                                            "ip_adapter_weight_name": "ip-adapter_sd15.bin",
                                            "ip_adapter_scale": 0.45,
                                        }
                                    )

        self.assertEqual(
            result,
            {
                "image_embeds": {
                    "artifact_id": "e123",
                    "path": "artifacts/e123.pt",
                    "url": "/outputs/artifacts/e123.pt",
                }
            },
        )
        download.assert_called_once_with(
            repo_id="h94/IP-Adapter",
            filename="ip-adapter_sd15.bin",
            subfolder="models",
        )
        from_pretrained.assert_called_once_with(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
        )
        self.assertEqual(fake_encoder.to_devices, ["cuda", "cpu"])
        self.assertEqual(saved["family"], "SD15")
        self.assertEqual(saved["metadata"]["base_model"], "stable-diffusion-v1-5")
        self.assertTrue(saved["metadata"]["do_classifier_free_guidance"])
        self.assertEqual(saved["metadata"]["adapters"][0]["scale"], 0.45)
        self.assertEqual(len(saved["embeds"]), 1)
        self.assertEqual(saved["embeds"][0].shape, (2, 1, 3))
        cleanup.assert_called_once()

    def test_generate_ip_adapter_image_embeds_rejects_non_default_adapter(self):
        with self.assertRaisesRegex(ValueError, "default SD1.5 base IP-Adapter"):
            sd15_ip_adapter_pipeline.generate_ip_adapter_image_embeds(
                {
                    "image": Image.new("RGB", (16, 16)),
                    "ip_adapter_model": "custom/model",
                    "ip_adapter_subfolder": "models",
                    "ip_adapter_weight_name": "ip-adapter_sd15.bin",
                }
            )

    def test_generate_images_prepares_and_passes_ip_adapter_image_embeds(self):
        fake_pipe = _FakePipe()
        reference_image = Image.new("RGB", (16, 16))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.sd15.pipeline.load_text2img_pipeline", return_value=fake_pipe):
                with patch("backend.sd15.pipeline.create_scheduler", return_value="scheduler"):
                    with patch(
                        "backend.sd15.pipeline.build_prompt_embeddings",
                        return_value=(None, None, False),
                    ):
                        with patch("backend.sd15.pipeline.torch.Generator", _FakeGenerator):
                            with patch(
                                "backend.sd15.pipeline.get_batch_output_dir",
                                return_value=Path(tmpdir),
                            ):
                                filenames = sd15_pipeline.generate_images_in_process(
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
        self.assertEqual(
            fake_pipe.prepare_ip_adapter_image_embeds_kwargs,
            {
                "ip_adapter_image": reference_image,
                "ip_adapter_image_embeds": None,
                "device": "cuda",
                "num_images_per_prompt": 1,
                "do_classifier_free_guidance": True,
            },
        )
        self.assertIs(
            fake_pipe.call_kwargs[0]["ip_adapter_image_embeds"],
            fake_pipe.prepared_ip_adapter_image_embeds,
        )
        self.assertNotIn("ip_adapter_image", fake_pipe.call_kwargs[0])
        self.assertTrue(fake_pipe.unloaded_ip_adapter)

    def test_generate_images_loads_precomputed_ip_adapter_embeds_without_image_encoder(self):
        fake_pipe = _FakePipe()
        image_encoder = fake_pipe.image_encoder
        embeds = ["loaded-ip-adapter-embeds"]
        embeds_payload = {"family": "SD15", "metadata": {}, "embeds": embeds}

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.sd15.pipeline.load_text2img_pipeline", return_value=fake_pipe):
                with patch("backend.sd15.pipeline.create_scheduler", return_value="scheduler"):
                    with patch(
                        "backend.sd15.pipeline.build_prompt_embeddings",
                        return_value=(None, None, False),
                    ):
                        with patch("backend.sd15.pipeline.torch.Generator", _FakeGenerator):
                            with patch(
                                "backend.sd15.pipeline.get_batch_output_dir",
                                return_value=Path(tmpdir),
                            ):
                                with patch(
                                    "backend.sd15.pipeline.load_ip_adapter_embeds_artifact",
                                    return_value=embeds_payload,
                                ) as load_embeds:
                                    with patch(
                                        "backend.sd15.pipeline.validate_ip_adapter_embeds_metadata"
                                    ) as validate_embeds:
                                        sd15_pipeline.generate_images_in_process(
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
                                                "ip_adapter_image_embeds_ref": {
                                                    "artifact_id": "e0123456789abcdef0123456789abcdef"
                                                },
                                                "ip_adapter_scale": 0.45,
                                                "ip_adapter_model": "h94/IP-Adapter",
                                                "ip_adapter_subfolder": "models",
                                                "ip_adapter_weight_name": "ip-adapter_sd15.bin",
                                                "batch_id": "batch123",
                                            }
                                        )

        load_embeds.assert_called_once_with(
            {"artifact_id": "e0123456789abcdef0123456789abcdef"}
        )
        validate_embeds.assert_called_once_with(
            embeds_payload,
            expected_model="h94/IP-Adapter",
            expected_subfolder="models",
            expected_weight_name="ip-adapter_sd15.bin",
            do_classifier_free_guidance=True,
            expected_family="SD15",
        )
        self.assertEqual(fake_pipe.load_ip_adapter_kwargs, {"image_encoder_folder": None})
        self.assertIs(fake_pipe.call_kwargs[0]["ip_adapter_image_embeds"], embeds)
        self.assertIsNone(fake_pipe.prepare_ip_adapter_image_embeds_kwargs)
        self.assertIsNone(fake_pipe.image_encoder_during_call)
        self.assertIs(fake_pipe.image_encoder, image_encoder)
        self.assertTrue(fake_pipe.unloaded_ip_adapter)

    def test_generate_images_img2img_prepares_and_passes_ip_adapter_image_embeds(self):
        fake_pipe = _FakePipe()
        reference_image = Image.new("RGB", (16, 16))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.sd15.pipeline.load_img2img_pipeline", return_value=fake_pipe):
                with patch("backend.sd15.pipeline.create_scheduler", return_value="scheduler"):
                    with patch("backend.sd15.pipeline.torch.Generator", _FakeGenerator):
                        with patch(
                            "backend.sd15.pipeline.get_batch_output_dir",
                            return_value=Path(tmpdir),
                        ):
                            filenames = sd15_pipeline.generate_images_img2img_in_process(
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
        self.assertEqual(
            fake_pipe.prepare_ip_adapter_image_embeds_kwargs,
            {
                "ip_adapter_image": reference_image,
                "ip_adapter_image_embeds": None,
                "device": "cuda",
                "num_images_per_prompt": 1,
                "do_classifier_free_guidance": True,
            },
        )
        self.assertIs(
            fake_pipe.call_kwargs[0]["ip_adapter_image_embeds"],
            fake_pipe.prepared_ip_adapter_image_embeds,
        )
        self.assertNotIn("ip_adapter_image", fake_pipe.call_kwargs[0])
        self.assertTrue(fake_pipe.unloaded_ip_adapter)

    def test_generate_images_img2img_loads_precomputed_ip_adapter_embeds(self):
        fake_pipe = _FakePipe()
        embeds = ["loaded-ip-adapter-embeds"]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.sd15.pipeline.load_img2img_pipeline", return_value=fake_pipe):
                with patch("backend.sd15.pipeline.create_scheduler", return_value="scheduler"):
                    with patch("backend.sd15.pipeline.torch.Generator", _FakeGenerator):
                        with patch(
                            "backend.sd15.pipeline.get_batch_output_dir",
                            return_value=Path(tmpdir),
                        ):
                            with patch(
                                "backend.sd15.pipeline.load_ip_adapter_embeds_artifact",
                                return_value={"family": "SD15", "metadata": {}, "embeds": embeds},
                            ):
                                with patch(
                                    "backend.sd15.pipeline.validate_ip_adapter_embeds_metadata"
                                ) as validate_embeds:
                                    sd15_pipeline.generate_images_img2img_in_process(
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
                                            "ip_adapter_image_embeds_ref": "@artifact:e0123456789abcdef0123456789abcdef",
                                            "ip_adapter_scale": 0.45,
                                            "batch_id": "batch123",
                                        }
                                    )

        validate_embeds.assert_called_once()
        self.assertEqual(fake_pipe.load_ip_adapter_kwargs, {"image_encoder_folder": None})
        self.assertIs(fake_pipe.call_kwargs[0]["ip_adapter_image_embeds"], embeds)
        self.assertIsNone(fake_pipe.prepare_ip_adapter_image_embeds_kwargs)
        self.assertIsNone(fake_pipe.image_encoder_during_call)
        self.assertTrue(fake_pipe.unloaded_ip_adapter)

    def test_generate_images_inpaint_prepares_and_passes_ip_adapter_image_embeds(self):
        fake_pipe = _FakePipe()
        reference_image = Image.new("RGB", (16, 16))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.sd15.pipeline.load_inpaint_pipeline", return_value=fake_pipe):
                with patch("backend.sd15.pipeline.create_scheduler", return_value="scheduler"):
                    with patch("backend.sd15.pipeline.torch.Generator", _FakeGenerator):
                        with patch("backend.sd15.pipeline._apply_lora_adapters", return_value=[]):
                            with patch(
                                "backend.sd15.pipeline.get_batch_output_dir",
                                return_value=Path(tmpdir),
                            ):
                                filenames = sd15_pipeline.generate_images_inpaint_in_process(
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
        self.assertEqual(
            fake_pipe.prepare_ip_adapter_image_embeds_kwargs,
            {
                "ip_adapter_image": reference_image,
                "ip_adapter_image_embeds": None,
                "device": "cuda",
                "num_images_per_prompt": 1,
                "do_classifier_free_guidance": True,
            },
        )
        self.assertIs(
            fake_pipe.call_kwargs[0]["ip_adapter_image_embeds"],
            fake_pipe.prepared_ip_adapter_image_embeds,
        )
        self.assertNotIn("ip_adapter_image", fake_pipe.call_kwargs[0])
        self.assertTrue(fake_pipe.unloaded_ip_adapter)

    def test_generate_images_inpaint_loads_precomputed_ip_adapter_embeds(self):
        fake_pipe = _FakePipe()
        embeds = ["loaded-ip-adapter-embeds"]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.sd15.pipeline.load_inpaint_pipeline", return_value=fake_pipe):
                with patch("backend.sd15.pipeline.create_scheduler", return_value="scheduler"):
                    with patch("backend.sd15.pipeline.torch.Generator", _FakeGenerator):
                        with patch("backend.sd15.pipeline._apply_lora_adapters", return_value=[]):
                            with patch(
                                "backend.sd15.pipeline.get_batch_output_dir",
                                return_value=Path(tmpdir),
                            ):
                                with patch(
                                    "backend.sd15.pipeline.load_ip_adapter_embeds_artifact",
                                    return_value={"family": "SD15", "metadata": {}, "embeds": embeds},
                                ):
                                    with patch(
                                        "backend.sd15.pipeline.validate_ip_adapter_embeds_metadata"
                                    ) as validate_embeds:
                                        sd15_pipeline.generate_images_inpaint_in_process(
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
                                                "ip_adapter_image_embeds_ref": "@artifact:e0123456789abcdef0123456789abcdef",
                                                "ip_adapter_scale": 0.45,
                                                "batch_id": "batch123",
                                            }
                                        )

        validate_embeds.assert_called_once()
        self.assertEqual(fake_pipe.load_ip_adapter_kwargs, {"image_encoder_folder": None})
        self.assertIs(fake_pipe.call_kwargs[0]["ip_adapter_image_embeds"], embeds)
        self.assertIsNone(fake_pipe.prepare_ip_adapter_image_embeds_kwargs)
        self.assertIsNone(fake_pipe.image_encoder_during_call)
        self.assertTrue(fake_pipe.unloaded_ip_adapter)

    def test_generate_images_passes_ip_adapter_masks(self):
        fake_pipe = _FakePipe()
        reference_image = Image.new("RGB", (16, 16))
        mask_image = Image.new("L", (64, 64), color="white")
        prepared_masks = ["prepared-ip-adapter-masks"]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.sd15.pipeline.load_text2img_pipeline", return_value=fake_pipe):
                with patch("backend.sd15.pipeline.create_scheduler", return_value="scheduler"):
                    with patch(
                        "backend.sd15.pipeline.build_prompt_embeddings",
                        return_value=(None, None, False),
                    ):
                        with patch("backend.sd15.pipeline.torch.Generator", _FakeGenerator):
                            with patch(
                                "backend.sd15.pipeline.get_batch_output_dir",
                                return_value=Path(tmpdir),
                            ):
                                with patch.object(
                                    sd15_pipeline.IpAdapterManager,
                                    "prepare_masks",
                                    return_value=prepared_masks,
                                ) as prepare_masks:
                                    sd15_pipeline.generate_images_in_process(
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
                                            "ip_adapter_mask_image": mask_image,
                                            "ip_adapter_scale": 0.45,
                                            "batch_id": "batch123",
                                        }
                                    )

        prepare_masks.assert_called_once_with(mask_image, height=64, width=64)
        self.assertEqual(
            fake_pipe.call_kwargs[0]["cross_attention_kwargs"],
            {"ip_adapter_masks": prepared_masks},
        )

    def test_generate_images_img2img_passes_ip_adapter_masks(self):
        fake_pipe = _FakePipe()
        reference_image = Image.new("RGB", (16, 16))
        mask_image = Image.new("L", (32, 32), color="white")
        prepared_masks = ["prepared-ip-adapter-masks"]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.sd15.pipeline.load_img2img_pipeline", return_value=fake_pipe):
                with patch("backend.sd15.pipeline.create_scheduler", return_value="scheduler"):
                    with patch("backend.sd15.pipeline.torch.Generator", _FakeGenerator):
                        with patch(
                            "backend.sd15.pipeline.get_batch_output_dir",
                            return_value=Path(tmpdir),
                        ):
                            with patch.object(
                                sd15_pipeline.IpAdapterManager,
                                "prepare_masks",
                                return_value=prepared_masks,
                            ) as prepare_masks:
                                sd15_pipeline.generate_images_img2img_in_process(
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
                                        "ip_adapter_mask_image": mask_image,
                                        "ip_adapter_scale": 0.45,
                                        "batch_id": "batch123",
                                    }
                                )

        prepare_masks.assert_called_once_with(mask_image, height=32, width=32)
        self.assertEqual(
            fake_pipe.call_kwargs[0]["cross_attention_kwargs"],
            {"ip_adapter_masks": prepared_masks},
        )

    def test_generate_images_inpaint_passes_ip_adapter_masks(self):
        fake_pipe = _FakePipe()
        reference_image = Image.new("RGB", (16, 16))
        mask_image = Image.new("L", (32, 32), color="white")
        prepared_masks = ["prepared-ip-adapter-masks"]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.sd15.pipeline.load_inpaint_pipeline", return_value=fake_pipe):
                with patch("backend.sd15.pipeline.create_scheduler", return_value="scheduler"):
                    with patch("backend.sd15.pipeline.torch.Generator", _FakeGenerator):
                        with patch("backend.sd15.pipeline._apply_lora_adapters", return_value=[]):
                            with patch(
                                "backend.sd15.pipeline.get_batch_output_dir",
                                return_value=Path(tmpdir),
                            ):
                                with patch.object(
                                    sd15_pipeline.IpAdapterManager,
                                    "prepare_masks",
                                    return_value=prepared_masks,
                                ) as prepare_masks:
                                    sd15_pipeline.generate_images_inpaint_in_process(
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
                                            "ip_adapter_mask_image": mask_image,
                                            "ip_adapter_scale": 0.45,
                                            "batch_id": "batch123",
                                        }
                                    )

        prepare_masks.assert_called_once_with(mask_image, height=32, width=32)
        self.assertEqual(
            fake_pipe.call_kwargs[0]["cross_attention_kwargs"],
            {"ip_adapter_masks": prepared_masks},
        )


if __name__ == "__main__":
    unittest.main()
