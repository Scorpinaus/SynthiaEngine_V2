import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from backend.sdxl import ip_adapter_pipeline as sdxl_ip_adapter_pipeline
from backend.sdxl import pipeline as sdxl_pipeline


class _FakeLatents:
    def __init__(self):
        self.ndim = 4
        self.to_kwargs = None

    def detach(self):
        return self

    def cpu(self):
        return self

    def to(self, **kwargs):
        self.to_kwargs = kwargs
        return self

    def __truediv__(self, _value):
        return self


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
        self.unloaded_ip_adapter = False
        self.prepare_ip_adapter_image_embeds_kwargs = None
        self.prepared_ip_adapter_image_embeds = ["prepared-ip-adapter-embeds"]
        self.device = "cuda"
        self.image_encoder = object()
        self.call_kwargs = None
        self.image_encoder_during_call = None
        self.vae = _FakeVae()
        self.image_processor = object()

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
        self.call_kwargs = kwargs
        self.image_encoder_during_call = self.image_encoder
        return type("Result", (), {"images": [_FakeLatents()]})()


class _FakeDeviceAwarePipe(_FakePipe):
    @property
    def _execution_device(self):
        return "cpu" if self.image_encoder is not None else "cuda"


class _FakeLoadableText2ImgPipe:
    def __init__(self):
        self.vae = type("FakeVae", (), {})()
        self.vae.slicing_enabled = False
        self.vae.tiling_enabled = False
        self.device = None

        def enable_slicing():
            self.vae.slicing_enabled = True

        def enable_tiling():
            self.vae.tiling_enabled = True

        self.vae.enable_slicing = enable_slicing
        self.vae.enable_tiling = enable_tiling

    def to(self, device):
        self.device = device
        return self


class _FakePixelValues:
    def __init__(self):
        self.to_kwargs = None

    def to(self, **kwargs):
        self.to_kwargs = kwargs
        return self


class _FakeCLIPImageProcessor:
    init_kwargs = None
    pixel_values = None

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs

    def __call__(self, image, *, return_tensors):
        type(self).pixel_values = _FakePixelValues()
        return type(
            "ProcessedImage",
            (),
            {"pixel_values": type(self).pixel_values},
        )()


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


class _FakeVae:
    dtype = "float16"

    def __init__(self):
        self.device = "cuda"
        self.config = type("Config", (), {"scaling_factor": 0.18215})()
        self.decoded_latents = None

    def decode(self, latents, return_dict=False):
        self.decoded_latents = latents
        return ["decoded-image"]


class SdxlIpAdapterPipelineTests(unittest.TestCase):
    def test_load_text2img_pipeline_enables_vae_memory_savers(self):
        fake_pipe = _FakeLoadableText2ImgPipe()
        fake_entry = type(
            "ModelEntry",
            (),
            {"model_type": "diffusers", "location_type": "hub", "link": "model/repo"},
        )()

        with patch("backend.sdxl.pipeline.get_model_entry", return_value=fake_entry):
            with patch(
                "backend.sdxl.pipeline.StableDiffusionXLPipeline.from_pretrained",
                return_value=fake_pipe,
            ):
                pipe = sdxl_pipeline.load_text2img_pipeline("stable-diffusion-xl-base-1.0")

        self.assertIs(pipe, fake_pipe)
        self.assertTrue(fake_pipe.vae.slicing_enabled)
        self.assertTrue(fake_pipe.vae.tiling_enabled)
        self.assertEqual(fake_pipe.device, "cuda")

    def test_generate_ip_adapter_image_embeds_uses_minimal_encoder(self):
        image = Image.new("RGB", (16, 16))
        fake_encoder = _FakeImageEncoder()
        saved = {}

        def _fake_save_embeds(embeds, *, metadata):
            saved["embeds"] = embeds
            saved["metadata"] = metadata
            return {
                "artifact_id": "e123",
                "path": "artifacts/e123.pt",
                "url": "/outputs/artifacts/e123.pt",
            }

        with patch("backend.sdxl.ip_adapter_pipeline.torch.cuda.is_available", return_value=True):
            with patch("backend.sdxl.ip_adapter_pipeline.hf_hub_download", return_value="adapter.bin") as download:
                with patch(
                    "backend.sdxl.ip_adapter_pipeline.load_state_dict",
                    return_value={
                        "image_proj": {"proj.weight": torch.zeros(8192, 1280)},
                        "ip_adapter": {},
                    },
                ):
                    with patch(
                        "backend.sdxl.ip_adapter_pipeline.CLIPVisionModelWithProjection.from_pretrained",
                        return_value=fake_encoder,
                    ) as from_pretrained:
                        with patch(
                            "backend.sdxl.ip_adapter_pipeline.CLIPImageProcessor",
                            _FakeCLIPImageProcessor,
                        ):
                            with patch(
                                "backend.sdxl.ip_adapter_pipeline.save_ip_adapter_embeds_artifact",
                                side_effect=_fake_save_embeds,
                            ):
                                with patch("backend.sdxl.ip_adapter_pipeline.cleanup_memory") as cleanup:
                                    with patch(
                                        "backend.sdxl.ip_adapter_pipeline.load_text2img_pipeline",
                                        create=True,
                                    ) as load_text2img_pipeline:
                                        result = sdxl_ip_adapter_pipeline.generate_ip_adapter_image_embeds(
                                            {
                                                "image": image,
                                                "model": "stable-diffusion-xl-base-1.0",
                                                "guidance_scale": 7.5,
                                                "ip_adapter_model": "h94/IP-Adapter",
                                                "ip_adapter_subfolder": "sdxl_models",
                                                "ip_adapter_weight_name": "ip-adapter_sdxl.bin",
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
            filename="ip-adapter_sdxl.bin",
            subfolder="sdxl_models",
        )
        from_pretrained.assert_called_once_with(
            "h94/IP-Adapter",
            subfolder="sdxl_models/image_encoder",
            torch_dtype=torch.float16,
        )
        load_text2img_pipeline.assert_not_called()
        cleanup.assert_called_once()
        self.assertEqual(fake_encoder.to_devices, ["cuda", "cpu"])
        self.assertEqual(
            _FakeCLIPImageProcessor.init_kwargs,
            {"size": 224, "crop_size": 224},
        )
        self.assertEqual(
            _FakeCLIPImageProcessor.pixel_values.to_kwargs,
            {"device": "cuda", "dtype": torch.float16},
        )
        self.assertEqual(len(saved["embeds"]), 1)
        self.assertEqual(tuple(saved["embeds"][0].shape), (2, 1, 3))
        self.assertTrue(
            torch.equal(
                saved["embeds"][0][0],
                torch.zeros(1, 3, dtype=torch.float16),
            )
        )
        self.assertTrue(
            torch.equal(
                saved["embeds"][0][1],
                torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float16),
            )
        )
        self.assertEqual(
            saved["metadata"],
            {
                "base_model": "stable-diffusion-xl-base-1.0",
                "adapters": [
                    {
                        "model": "h94/IP-Adapter",
                        "subfolder": "sdxl_models",
                        "weight_name": "ip-adapter_sdxl.bin",
                        "scale": 0.45,
                    }
                ],
                "do_classifier_free_guidance": True,
                "num_images_per_prompt": 1,
            },
        )

    def test_generate_ip_adapter_image_embeds_rejects_non_default_adapter(self):
        with self.assertRaisesRegex(
            ValueError,
            "Only the default SDXL base IP-Adapter is supported by the minimal encoder",
        ):
            sdxl_ip_adapter_pipeline.generate_ip_adapter_image_embeds(
                {
                    "image": Image.new("RGB", (16, 16)),
                    "ip_adapter_model": "custom/model",
                    "ip_adapter_subfolder": "sdxl_models",
                    "ip_adapter_weight_name": "ip-adapter_sdxl.bin",
                }
            )

    def test_generate_ip_adapter_image_embeds_requires_cuda(self):
        with patch("backend.sdxl.ip_adapter_pipeline.torch.cuda.is_available", return_value=False):
            with patch("backend.sdxl.ip_adapter_pipeline.hf_hub_download") as download:
                with self.assertRaisesRegex(
                    ValueError,
                    "CUDA is required for SDXL IP-Adapter minimal encode",
                ):
                    sdxl_ip_adapter_pipeline.generate_ip_adapter_image_embeds(
                        {"image": Image.new("RGB", (16, 16))}
                    )

        download.assert_not_called()

    def test_generate_ip_adapter_image_embeds_rejects_unsupported_projection(self):
        with patch("backend.sdxl.ip_adapter_pipeline.torch.cuda.is_available", return_value=True):
            with patch("backend.sdxl.ip_adapter_pipeline.hf_hub_download", return_value="adapter.bin"):
                with patch(
                    "backend.sdxl.ip_adapter_pipeline.load_state_dict",
                    return_value={"image_proj": {"latents": torch.zeros(1, 4, 8)}, "ip_adapter": {}},
                ):
                    with patch(
                        "backend.sdxl.ip_adapter_pipeline.CLIPVisionModelWithProjection.from_pretrained"
                    ) as from_pretrained:
                        with patch("backend.sdxl.ip_adapter_pipeline.cleanup_memory"):
                            with self.assertRaisesRegex(
                                ValueError,
                                "Only the base SDXL IP-Adapter is supported by the minimal encoder",
                            ):
                                sdxl_ip_adapter_pipeline.generate_ip_adapter_image_embeds(
                                    {"image": Image.new("RGB", (16, 16))}
                                )

        from_pretrained.assert_not_called()

    def test_render_text2img_latents_hides_image_encoder_when_using_ip_adapter_embeds(self):
        fake_pipe = _FakeDeviceAwarePipe()
        image_encoder = fake_pipe.image_encoder
        ip_adapter_image_embeds = ["prepared-ip-adapter-embeds"]

        with patch("backend.sdxl.pipeline.torch.Generator", _FakeGenerator):
            latents = sdxl_pipeline.render_text2img_latents(
                fake_pipe,
                prompt="test prompt",
                negative_prompt="",
                steps=2,
                guidance_scale=7.5,
                width=64,
                height=64,
                seed=123,
                clip_skip=1,
                ip_adapter_image_embeds=ip_adapter_image_embeds,
            )

        self.assertIsInstance(latents, _FakeLatents)
        self.assertIsNone(fake_pipe.image_encoder_during_call)
        self.assertIs(fake_pipe.image_encoder, image_encoder)
        self.assertEqual(fake_pipe.call_kwargs["generator"].device, "cuda")
        self.assertIs(fake_pipe.call_kwargs["ip_adapter_image_embeds"], ip_adapter_image_embeds)
        self.assertNotIn("ip_adapter_image", fake_pipe.call_kwargs)

    def test_decode_latents_uses_vae_device_instead_of_pipeline_device(self):
        fake_pipe = _FakePipe()
        fake_pipe.vae = _FakeVae()
        fake_pipe.image_processor = type(
            "ImageProcessor",
            (),
            {"postprocess": lambda self, image, output_type: ["decoded-pil"]},
        )()
        latents = _FakeLatents()

        image = sdxl_pipeline._decode_latents_to_pil(fake_pipe, latents)

        self.assertEqual(image, "decoded-pil")
        self.assertEqual(latents.to_kwargs, {"device": "cuda", "dtype": "float16"})
        self.assertIs(fake_pipe.vae.decoded_latents, latents)

    def test_generate_text2img_releases_pipeline_before_latent_decode(self):
        fake_pipe = _FakePipe()
        events = []

        def _fake_render_text2img_latents(_pipe, **_kwargs):
            events.append("render")
            return _FakeLatents()

        def _fake_release_pipeline(pipe):
            self.assertIs(pipe, fake_pipe)
            events.append("release")

        def _fake_decode_latents_to_pil(decoder, _latents):
            events.append("decode")
            self.assertIsNot(decoder, fake_pipe)
            self.assertIs(decoder.vae, fake_pipe.vae)
            self.assertLess(events.index("release"), events.index("decode"))
            return Image.new("RGB", (8, 8))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.sdxl.pipeline.load_text2img_pipeline", return_value=fake_pipe):
                with patch("backend.sdxl.pipeline.create_scheduler", return_value="scheduler"):
                    with patch(
                        "backend.sdxl.pipeline.apply_lora_adapters_with_validation",
                        return_value=([], {}),
                    ):
                        with patch("backend.sdxl.pipeline.write_lora_coverage_report", return_value=None):
                            with patch(
                                "backend.sdxl.pipeline.render_text2img_latents",
                                side_effect=_fake_render_text2img_latents,
                            ):
                                with patch(
                                    "backend.sdxl.pipeline._release_pipeline",
                                    side_effect=_fake_release_pipeline,
                                ):
                                    with patch(
                                        "backend.sdxl.pipeline._decode_latents_to_pil",
                                        side_effect=_fake_decode_latents_to_pil,
                                    ):
                                        with patch(
                                            "backend.sdxl.pipeline.make_batch_id",
                                            return_value="batch123",
                                        ):
                                            with patch(
                                                "backend.sdxl.pipeline.get_batch_output_dir",
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
                                                    }
                                                )

        self.assertEqual(
            result,
            {"images": ["/outputs/batch_batch123/batch123_123.png"]},
        )
        self.assertEqual(events, ["render", "release", "decode"])

    def test_generate_text2img_prepares_and_passes_ip_adapter_image_embeds(self):
        fake_pipe = _FakePipe()
        reference_image = Image.new("RGB", (16, 16))
        captured_render_kwargs = {}

        def _fake_render_text2img_latents(pipe, **kwargs):
            captured_render_kwargs.update(kwargs)
            return _FakeLatents()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.sdxl.pipeline.load_text2img_pipeline", return_value=fake_pipe):
                with patch("backend.sdxl.pipeline.create_scheduler", return_value="scheduler"):
                    with patch(
                        "backend.sdxl.pipeline.apply_lora_adapters_with_validation",
                        return_value=([], {}),
                    ):
                        with patch("backend.sdxl.pipeline.write_lora_coverage_report", return_value=None):
                            with patch(
                                "backend.sdxl.pipeline.render_text2img_latents",
                                side_effect=_fake_render_text2img_latents,
                            ):
                                with patch(
                                    "backend.sdxl.pipeline._decode_latents_to_pil",
                                    return_value=Image.new("RGB", (8, 8)),
                                ):
                                    with patch(
                                        "backend.sdxl.pipeline.make_batch_id",
                                        return_value="batch123",
                                    ):
                                        with patch(
                                            "backend.sdxl.pipeline.get_batch_output_dir",
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
            captured_render_kwargs["ip_adapter_image_embeds"],
            fake_pipe.prepared_ip_adapter_image_embeds,
        )
        self.assertNotIn("ip_adapter_image", captured_render_kwargs)
        self.assertTrue(fake_pipe.unloaded_ip_adapter)

    def test_generate_text2img_loads_precomputed_ip_adapter_embeds_without_image_encoder(self):
        fake_pipe = _FakePipe()
        captured_render_kwargs = {}
        embeds = ["loaded-ip-adapter-embeds"]
        embeds_payload = {"embeds": embeds, "metadata": {"adapters": []}}

        def _fake_render_text2img_latents(pipe, **kwargs):
            captured_render_kwargs.update(kwargs)
            return _FakeLatents()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.sdxl.pipeline.load_text2img_pipeline", return_value=fake_pipe):
                with patch("backend.sdxl.pipeline.create_scheduler", return_value="scheduler"):
                    with patch(
                        "backend.sdxl.pipeline.load_ip_adapter_embeds_artifact",
                        return_value=embeds_payload,
                    ) as load_embeds:
                        with patch("backend.sdxl.pipeline.validate_ip_adapter_embeds_metadata") as validate_embeds:
                            with patch(
                                "backend.sdxl.pipeline.apply_lora_adapters_with_validation",
                                return_value=([], {}),
                            ):
                                with patch("backend.sdxl.pipeline.write_lora_coverage_report", return_value=None):
                                    with patch(
                                        "backend.sdxl.pipeline.render_text2img_latents",
                                        side_effect=_fake_render_text2img_latents,
                                    ):
                                        with patch(
                                            "backend.sdxl.pipeline._decode_latents_to_pil",
                                            return_value=Image.new("RGB", (8, 8)),
                                        ):
                                            with patch(
                                                "backend.sdxl.pipeline.make_batch_id",
                                                return_value="batch123",
                                            ):
                                                with patch(
                                                    "backend.sdxl.pipeline.get_batch_output_dir",
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
                                                            "ip_adapter_image_embeds_ref": {
                                                                "artifact_id": "e0123456789abcdef0123456789abcdef"
                                                            },
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
        load_embeds.assert_called_once_with(
            {"artifact_id": "e0123456789abcdef0123456789abcdef"}
        )
        validate_embeds.assert_called_once_with(
            embeds_payload,
            expected_model="h94/IP-Adapter",
            expected_subfolder="sdxl_models",
            expected_weight_name="ip-adapter_sdxl.bin",
            do_classifier_free_guidance=True,
        )
        self.assertEqual(fake_pipe.load_ip_adapter_kwargs, {"image_encoder_folder": None})
        self.assertIs(captured_render_kwargs["ip_adapter_image_embeds"], embeds)
        self.assertIsNone(fake_pipe.prepare_ip_adapter_image_embeds_kwargs)
        self.assertTrue(fake_pipe.unloaded_ip_adapter)

    def test_generate_img2img_releases_pipeline_before_latent_decode(self):
        fake_pipe = _FakePipe()
        initial_image = Image.new("RGB", (16, 16))
        events = []

        def _fake_render_img2img_latents(_pipe, **_kwargs):
            events.append("render")
            return _FakeLatents()

        def _fake_release_pipeline(pipe):
            self.assertIs(pipe, fake_pipe)
            events.append("release")

        def _fake_decode_latents_to_pil(decoder, _latents):
            events.append("decode")
            self.assertIsNot(decoder, fake_pipe)
            self.assertIs(decoder.vae, fake_pipe.vae)
            self.assertLess(events.index("release"), events.index("decode"))
            return Image.new("RGB", (8, 8))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.sdxl.pipeline.load_img2img_pipeline", return_value=fake_pipe):
                with patch("backend.sdxl.pipeline.create_scheduler", return_value="scheduler"):
                    with patch(
                        "backend.sdxl.pipeline.apply_lora_adapters_with_validation",
                        return_value=([], {}),
                    ):
                        with patch("backend.sdxl.pipeline.write_lora_coverage_report", return_value=None):
                            with patch(
                                "backend.sdxl.pipeline.render_img2img_latents",
                                side_effect=_fake_render_img2img_latents,
                            ):
                                with patch(
                                    "backend.sdxl.pipeline._release_pipeline",
                                    side_effect=_fake_release_pipeline,
                                ):
                                    with patch(
                                        "backend.sdxl.pipeline._decode_latents_to_pil",
                                        side_effect=_fake_decode_latents_to_pil,
                                    ):
                                        with patch(
                                            "backend.sdxl.pipeline.make_batch_id",
                                            return_value="batch123",
                                        ):
                                            with patch(
                                                "backend.sdxl.pipeline.get_batch_output_dir",
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
                                                    }
                                                )

        self.assertEqual(
            result,
            {"images": ["/outputs/batch_batch123/batch123_123.png"]},
        )
        self.assertEqual(events, ["render", "release", "decode"])

    def test_generate_img2img_loads_and_passes_ip_adapter_image(self):
        fake_pipe = _FakePipe()
        initial_image = Image.new("RGB", (16, 16))
        reference_image = Image.new("RGB", (16, 16))
        captured_render_kwargs = {}

        def _fake_render_img2img_latents(pipe, **kwargs):
            captured_render_kwargs.update(kwargs)
            return _FakeLatents()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.sdxl.pipeline.load_img2img_pipeline", return_value=fake_pipe):
                with patch("backend.sdxl.pipeline.create_scheduler", return_value="scheduler"):
                    with patch(
                        "backend.sdxl.pipeline.apply_lora_adapters_with_validation",
                        return_value=([], {}),
                    ):
                        with patch("backend.sdxl.pipeline.write_lora_coverage_report", return_value=None):
                            with patch(
                                "backend.sdxl.pipeline.render_img2img_latents",
                                side_effect=_fake_render_img2img_latents,
                            ):
                                with patch(
                                    "backend.sdxl.pipeline._decode_latents_to_pil",
                                    return_value=Image.new("RGB", (8, 8)),
                                ):
                                    with patch(
                                        "backend.sdxl.pipeline.make_batch_id",
                                        return_value="batch123",
                                    ):
                                        with patch(
                                            "backend.sdxl.pipeline.get_batch_output_dir",
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
            with patch("backend.sdxl.pipeline.load_inpaint_pipeline", return_value=fake_pipe):
                with patch("backend.sdxl.pipeline.create_scheduler", return_value="scheduler"):
                    with patch(
                        "backend.sdxl.pipeline.apply_lora_adapters_with_validation",
                        return_value=([], {}),
                    ):
                        with patch("backend.sdxl.pipeline.write_lora_coverage_report", return_value=None):
                            with patch(
                                "backend.sdxl.pipeline.render_inpaint_image",
                                side_effect=_fake_render_inpaint_image,
                            ):
                                with patch(
                                    "backend.sdxl.pipeline.make_batch_id",
                                    return_value="batch123",
                                ):
                                    with patch(
                                        "backend.sdxl.pipeline.get_batch_output_dir",
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
