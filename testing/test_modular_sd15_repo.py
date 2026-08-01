from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from diffusers import ModularPipeline
from diffusers.guiders import ClassifierFreeGuidance
from diffusers.modular_pipelines import PipelineState
from diffusers.schedulers import DDIMScheduler


SD15_MODULAR_REPO = REPO_ROOT / "backend" / "modular_diffusers" / "sd15"


def test_sd15_modular_repo_loads_custom_blocks():
    pipe = ModularPipeline.from_pretrained(str(SD15_MODULAR_REPO), trust_remote_code=True)

    assert pipe.blocks.__class__.__name__ == "SD15AutoBlocks"
    assert set(pipe.components.keys()) == {
        "tokenizer",
        "text_encoder",
        "unet",
        "vae",
        "scheduler",
        "image_processor",
        "mask_processor",
        "guider",
    }
    assert pipe.image_processor.__class__.__name__ == "VaeImageProcessor"
    assert pipe.mask_processor.__class__.__name__ == "VaeImageProcessor"
    assert pipe.guider.__class__.__name__ == "ClassifierFreeGuidance"
    assert pipe.blocks.get_execution_blocks().__class__.__name__ == "SD15Text2ImgBlocks"
    assert pipe.blocks.get_execution_blocks(image=True).__class__.__name__ == "SD15Img2ImgBlocks"
    assert pipe.blocks.get_execution_blocks(image=True, mask_image=True).__class__.__name__ == "SD15InpaintBlocks"


def test_sd15_modular_repo_exposes_sdxl_style_modules_and_pipeline_defaults():
    from backend.modular_diffusers.sd15 import SD15AutoBlocks, SD15ModularPipeline
    from backend.modular_diffusers.sd15.before_denoise import SD15Text2ImgLatentsStep
    from backend.modular_diffusers.sd15.decoders import SD15DecodeStep
    from backend.modular_diffusers.sd15.denoise import SD15DenoiseStep
    from backend.modular_diffusers.sd15.encoders import SD15PromptEncodingStep
    from backend.modular_diffusers.sd15.modular_blocks_sd15 import SD15Text2ImgBlocks

    pipe = SD15ModularPipeline(blocks=SD15AutoBlocks())

    assert pipe.default_blocks_name == "SD15AutoBlocks"
    assert pipe.default_sample_size == 64
    assert pipe.vae_scale_factor == 8
    assert pipe.default_height == 512
    assert pipe.default_width == 512
    assert pipe.num_channels_unet == 4
    assert pipe.num_channels_latents == 4
    assert SD15Text2ImgBlocks.block_classes == [
        pytest.importorskip("backend.modular_diffusers.sd15.encoders").SD15InputValidationStep,
        SD15PromptEncodingStep,
        SD15Text2ImgLatentsStep,
        SD15DenoiseStep,
        SD15DecodeStep,
    ]


def test_sd15_modular_repo_exposes_input_schema_with_pipeline_image_inputs():
    from diffusers.image_processor import PipelineImageInput

    from backend.modular_diffusers.sd15.modular_pipeline import SD15_INPUTS_SCHEMA

    assert SD15_INPUTS_SCHEMA["height"].default is None
    assert SD15_INPUTS_SCHEMA["width"].default is None
    assert SD15_INPUTS_SCHEMA["image"].type_hint is PipelineImageInput
    assert SD15_INPUTS_SCHEMA["mask_image"].type_hint is PipelineImageInput
    assert SD15_INPUTS_SCHEMA["padding_mask_crop"].default is None
    assert SD15_INPUTS_SCHEMA["output_type"].default == "pil"


class _FakeLatentDist:
    def __init__(self, latents):
        self._latents = latents

    def sample(self, generator=None):
        return self._latents


class _FakeVae:
    dtype = torch.float32

    def __init__(self):
        self.config = SimpleNamespace(
            block_out_channels=[1, 2, 3, 4],
            scaling_factor=1.0,
            force_upcast=False,
            latent_channels=4,
        )

    def encode(self, image):
        batch_size = image.shape[0]
        latents = torch.ones((batch_size, 4, image.shape[-2] // 8, image.shape[-1] // 8), dtype=image.dtype)
        return SimpleNamespace(latent_dist=_FakeLatentDist(latents))

    def to(self, *args, **kwargs):
        return self


class _FakeScheduler:
    order = 1
    init_noise_sigma = 1.0

    def set_timesteps(self, num_inference_steps, device=None, **kwargs):
        self.timesteps = torch.arange(num_inference_steps, 0, -1, device=device)

    def add_noise(self, latents, noise, timestep):
        return latents + noise

    def scale_model_input(self, latents, timestep):
        return latents

    def step(self, noise_pred, timestep, latents, return_dict=False, **kwargs):
        return (torch.full_like(latents, 2.0),)


class _RecordingUnet:
    def __init__(self, in_channels):
        self.config = SimpleNamespace(in_channels=in_channels)
        self.seen_channels = []

    def __call__(self, latent_model_input, timestep, encoder_hidden_states=None, return_dict=False):
        self.seen_channels.append(latent_model_input.shape[1])
        return (torch.zeros_like(latent_model_input[:, :4]),)


class _RecordingImageProcessor:
    def __init__(self):
        self.preprocess_calls = []
        self.overlay_calls = []

    def preprocess(self, image, height=None, width=None, crops_coords=None, resize_mode="default"):
        self.preprocess_calls.append(
            {
                "image": image,
                "height": height,
                "width": width,
                "crops_coords": crops_coords,
                "resize_mode": resize_mode,
            }
        )
        return torch.ones((1, 3, height, width), dtype=torch.float32)

    def postprocess(self, image, output_type="pil", **kwargs):
        return [Image.new("RGB", (16, 16), color=(1, 2, 3))]

    def apply_overlay(self, mask_image, original_image, image, crops_coords):
        self.overlay_calls.append((mask_image, original_image, image, crops_coords))
        return Image.new("RGB", original_image.size, color=(9, 8, 7))


class _RecordingMaskProcessor:
    def __init__(self, crops_coords=(4, 8, 60, 56)):
        self.crops_coords = crops_coords
        self.crop_region_calls = []
        self.preprocess_calls = []

    def get_crop_region(self, mask_image, width, height, pad=0):
        self.crop_region_calls.append(
            {
                "mask_image": mask_image,
                "width": width,
                "height": height,
                "pad": pad,
            }
        )
        return self.crops_coords

    def preprocess(self, mask_image, height=None, width=None, crops_coords=None, resize_mode="default"):
        self.preprocess_calls.append(
            {
                "mask_image": mask_image,
                "height": height,
                "width": width,
                "crops_coords": crops_coords,
                "resize_mode": resize_mode,
            }
        )
        return torch.ones((1, 1, height, width), dtype=torch.float32)


def test_sd15_inpaint_latents_prepare_masked_image_latents_for_9_channel_unet():
    from backend.modular_diffusers.sd15.before_denoise import SD15InpaintLatentsStep

    components = SimpleNamespace(
        _execution_device=torch.device("cpu"),
        unet=_RecordingUnet(in_channels=9),
        vae=_FakeVae(),
        scheduler=_FakeScheduler(),
        image_processor=None,
        mask_processor=None,
    )
    state = PipelineState(
        values={
            "prompt_embeds": torch.zeros((1, 4, 8), dtype=torch.float32),
            "batch_size": 1,
            "num_images_per_prompt": 1,
            "height": 64,
            "width": 64,
            "num_inference_steps": 4,
            "eta": 0.0,
            "generator": torch.Generator(device="cpu").manual_seed(0),
            "latents": None,
            "timesteps": None,
            "sigmas": None,
            "image": Image.new("RGB", (64, 64), color=(120, 160, 200)),
            "mask_image": Image.new("L", (64, 64), color=255),
            "strength": 1.0,
        }
    )

    _, state = SD15InpaintLatentsStep()(components, state)

    assert state.latents.shape == (1, 4, 8, 8)
    assert state.mask.shape == (1, 1, 8, 8)
    assert state.masked_image_latents.shape == (1, 4, 8, 8)
    assert state.image_latents is None
    assert state.latent_noise is None


def test_sd15_inpaint_padding_mask_crop_threads_crop_preprocessing():
    from backend.modular_diffusers.sd15.before_denoise import SD15InpaintLatentsStep

    image_processor = _RecordingImageProcessor()
    mask_processor = _RecordingMaskProcessor(crops_coords=(8, 8, 56, 56))
    components = SimpleNamespace(
        _execution_device=torch.device("cpu"),
        unet=_RecordingUnet(in_channels=9),
        vae=_FakeVae(),
        scheduler=_FakeScheduler(),
        image_processor=image_processor,
        mask_processor=mask_processor,
    )
    image = Image.new("RGB", (64, 64), color=(120, 160, 200))
    mask_image = Image.new("L", (64, 64), color=0)
    mask_image.paste(255, (24, 24, 40, 40))
    state = PipelineState(
        values={
            "prompt_embeds": torch.zeros((1, 4, 8), dtype=torch.float32),
            "batch_size": 1,
            "num_images_per_prompt": 1,
            "height": 64,
            "width": 64,
            "num_inference_steps": 4,
            "eta": 0.0,
            "generator": torch.Generator(device="cpu").manual_seed(0),
            "latents": None,
            "timesteps": None,
            "sigmas": None,
            "image": image,
            "mask_image": mask_image,
            "strength": 1.0,
            "padding_mask_crop": 12,
        }
    )

    _, state = SD15InpaintLatentsStep()(components, state)

    assert mask_processor.crop_region_calls == [
        {"mask_image": mask_image, "width": 64, "height": 64, "pad": 12}
    ]
    assert state.crops_coords == (8, 8, 56, 56)
    assert state.resize_mode == "fill"
    assert state.original_image is image
    assert state.original_mask_image is mask_image
    assert {call["crops_coords"] for call in image_processor.preprocess_calls} == {(8, 8, 56, 56)}
    assert {call["resize_mode"] for call in image_processor.preprocess_calls} == {"fill"}
    assert {call["crops_coords"] for call in mask_processor.preprocess_calls} == {(8, 8, 56, 56)}
    assert {call["resize_mode"] for call in mask_processor.preprocess_calls} == {"fill"}


def test_sd15_denoise_concatenates_9_channel_inpaint_inputs_without_mask_blend():
    from backend.modular_diffusers.sd15.denoise import SD15DenoiseStep

    unet = _RecordingUnet(in_channels=9)
    components = SimpleNamespace(
        unet=unet,
        scheduler=_FakeScheduler(),
        guider=ClassifierFreeGuidance(guidance_scale=1.0),
    )
    state = PipelineState(
        values={
            "prompt_embeds": torch.zeros((1, 4, 8), dtype=torch.float32),
            "negative_prompt_embeds": None,
            "guidance_scale": 1.0,
            "latents": torch.zeros((1, 4, 8, 8), dtype=torch.float32),
            "timesteps": torch.tensor([1]),
            "num_inference_steps": 1,
            "eta": 0.0,
            "generator": None,
            "image_latents": torch.full((1, 4, 8, 8), 9.0),
            "latent_noise": torch.full((1, 4, 8, 8), 9.0),
            "mask": torch.ones((1, 1, 8, 8), dtype=torch.float32),
            "masked_image_latents": torch.ones((1, 4, 8, 8), dtype=torch.float32),
        }
    )

    _, state = SD15DenoiseStep()(components, state)

    assert unet.seen_channels == [9]
    assert torch.equal(state.latents, torch.full((1, 4, 8, 8), 2.0))


def test_sd15_decode_applies_padding_mask_crop_overlay():
    from backend.modular_diffusers.sd15.decoders import SD15DecodeStep

    image_processor = _RecordingImageProcessor()
    components = SimpleNamespace(
        vae=SimpleNamespace(
            config=SimpleNamespace(scaling_factor=1.0, force_upcast=False),
            decode=lambda latents, return_dict=False: (torch.zeros((1, 3, 16, 16)),),
        ),
        image_processor=image_processor,
    )
    original_image = Image.new("RGB", (64, 64), color=(120, 160, 200))
    mask_image = Image.new("L", (64, 64), color=255)
    state = PipelineState(
        values={
            "latents": torch.zeros((1, 4, 2, 2), dtype=torch.float32),
            "output_type": "pil",
            "crops_coords": (8, 8, 56, 56),
            "original_image": original_image,
            "original_mask_image": mask_image,
        }
    )

    _, state = SD15DecodeStep()(components, state)

    assert len(image_processor.overlay_calls) == 1
    assert image_processor.overlay_calls[0][0] is mask_image
    assert image_processor.overlay_calls[0][1] is original_image
    assert image_processor.overlay_calls[0][3] == (8, 8, 56, 56)
    assert state.images[0].size == original_image.size


def test_sd15_modular_repo_rejects_padding_mask_crop_without_pil_images():
    pipe = ModularPipeline.from_pretrained(str(SD15_MODULAR_REPO), trust_remote_code=True)

    with pytest.raises(ValueError, match="padding_mask_crop"):
        pipe(
            prompt="crop validation",
            image=torch.zeros((1, 3, 64, 64)),
            mask_image=Image.new("L", (64, 64), color=255),
            padding_mask_crop=16,
            height=64,
            width=64,
        )


def test_sd15_modular_inpaint_script_parser_defaults_to_inpaint_model():
    from backend.modular_diffusers.sd15 import sd15_modular_inpaint

    parser = sd15_modular_inpaint.build_parser()
    args = parser.parse_args(["--image", "input.png", "--mask-image", "mask.png"])

    assert args.image == Path("input.png")
    assert args.mask_image == Path("mask.png")
    assert args.strength == 1.0
    assert args.padding_mask_crop is None
    assert args.model == "stable-diffusion-v1-5/stable-diffusion-inpainting"
    assert args.output.name == "sd15_modular_inpaint.png"


def test_sd15_modular_inpaint_script_main_calls_modular_pipeline(tmp_path, monkeypatch):
    from backend.modular_diffusers.sd15 import sd15_modular_inpaint

    input_path = tmp_path / "input.png"
    mask_path = tmp_path / "mask.png"
    output_path = tmp_path / "out.png"
    Image.new("RGB", (16, 16), color=(120, 160, 200)).save(input_path)
    Image.new("L", (16, 16), color=255).save(mask_path)

    class FakePipeline:
        def __init__(self):
            self.load_kwargs = None
            self.device = None
            self.call_kwargs = None

        def load_components(self, **kwargs):
            self.load_kwargs = kwargs

        def to(self, device):
            self.device = device

        def __call__(self, **kwargs):
            self.call_kwargs = kwargs
            return [Image.new("RGB", (16, 16), color=(1, 2, 3))]

    fake_pipe = FakePipeline()
    monkeypatch.setattr(sd15_modular_inpaint.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sd15_modular_inpaint.ModularPipeline,
        "from_pretrained",
        lambda repo_path, trust_remote_code: fake_pipe,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sd15_modular_inpaint.py",
            "--image",
            str(input_path),
            "--mask-image",
            str(mask_path),
            "--prompt",
            "paint a bright portal",
            "--negative-prompt",
            "blurry",
            "--strength",
            "0.8",
            "--steps",
            "5",
            "--guidance-scale",
            "6.5",
            "--width",
            "16",
            "--height",
            "16",
            "--seed",
            "42",
            "--padding-mask-crop",
            "16",
            "--output",
            str(output_path),
        ],
    )

    sd15_modular_inpaint.main()

    assert output_path.exists()
    assert fake_pipe.load_kwargs["torch_dtype"] == torch.float32
    assert fake_pipe.load_kwargs["pretrained_model_name_or_path"] == {
        "default": "stable-diffusion-v1-5/stable-diffusion-inpainting"
    }
    assert fake_pipe.device == "cpu"
    assert fake_pipe.call_kwargs["prompt"] == "paint a bright portal"
    assert fake_pipe.call_kwargs["negative_prompt"] == "blurry"
    assert fake_pipe.call_kwargs["image"].mode == "RGB"
    assert fake_pipe.call_kwargs["mask_image"].mode == "L"
    assert fake_pipe.call_kwargs["strength"] == 0.8
    assert fake_pipe.call_kwargs["num_inference_steps"] == 5
    assert fake_pipe.call_kwargs["guidance_scale"] == 6.5
    assert fake_pipe.call_kwargs["width"] == 16
    assert fake_pipe.call_kwargs["height"] == 16
    assert fake_pipe.call_kwargs["padding_mask_crop"] == 16
    assert fake_pipe.call_kwargs["output"] == "images"


@pytest.mark.integration
def test_sd15_modular_repo_guidance_scale_one_uses_single_condition_path():
    pipe = ModularPipeline.from_pretrained(str(SD15_MODULAR_REPO), trust_remote_code=True)
    pipe.load_components(
        torch_dtype=torch.float32,
        pretrained_model_name_or_path={"default": "hf-internal-testing/tiny-stable-diffusion-pipe"},
    )
    pipe.update_components(scheduler=DDIMScheduler.from_config(pipe.scheduler.config))
    pipe.to("cpu")

    latents = pipe(
        prompt="a small test image of a lighthouse",
        negative_prompt="blurry",
        height=64,
        width=64,
        num_inference_steps=10,
        guidance_scale=1.0,
        generator=torch.Generator(device="cpu").manual_seed(0),
        output="latents",
    )

    assert latents.ndim == 4
    assert pipe.guider.num_conditions == 1


@pytest.mark.integration
def test_sd15_modular_repo_guided_denoise_is_deterministic_for_same_seed():
    pipe = ModularPipeline.from_pretrained(str(SD15_MODULAR_REPO), trust_remote_code=True)
    pipe.load_components(
        torch_dtype=torch.float32,
        pretrained_model_name_or_path={"default": "hf-internal-testing/tiny-stable-diffusion-pipe"},
    )
    pipe.update_components(scheduler=DDIMScheduler.from_config(pipe.scheduler.config))
    pipe.to("cpu")

    kwargs = dict(
        prompt="a small test image of a lighthouse",
        negative_prompt="blurry",
        height=64,
        width=64,
        num_inference_steps=10,
        guidance_scale=7.5,
        output="latents",
    )

    latents_a = pipe(generator=torch.Generator(device="cpu").manual_seed(123), **kwargs)
    latents_b = pipe(generator=torch.Generator(device="cpu").manual_seed(123), **kwargs)

    assert torch.allclose(latents_a, latents_b)
    assert pipe.guider.num_conditions == 2


@pytest.mark.integration
def test_sd15_modular_repo_runs_tiny_smoke_inference():
    pipe = ModularPipeline.from_pretrained(str(SD15_MODULAR_REPO), trust_remote_code=True)
    pipe.load_components(
        torch_dtype=torch.float32,
        pretrained_model_name_or_path={"default": "hf-internal-testing/tiny-stable-diffusion-pipe"},
    )
    pipe.to("cpu")

    outputs = pipe(
        prompt="a small test image of a lighthouse",
        negative_prompt="blurry",
        height=64,
        width=64,
        num_inference_steps=10,
        guidance_scale=7.5,
        generator=torch.Generator(device="cpu").manual_seed(0),
        output=["images", "latents"],
    )

    expected_latent_size = 64 // (2 ** (len(pipe.vae.config.block_out_channels) - 1))

    assert len(outputs["images"]) == 1
    assert outputs["images"][0].size == (64, 64)
    assert tuple(outputs["latents"].shape) == (1, 4, expected_latent_size, expected_latent_size)


@pytest.mark.integration
def test_sd15_modular_repo_accepts_precomputed_prompt_embeds():
    pipe = ModularPipeline.from_pretrained(str(SD15_MODULAR_REPO), trust_remote_code=True)
    pipe.load_components(
        torch_dtype=torch.float32,
        pretrained_model_name_or_path={"default": "hf-internal-testing/tiny-stable-diffusion-pipe"},
    )
    pipe.to("cpu")

    prompt = "a small test image of a lighthouse"
    negative_prompt = "blurry"
    text_inputs = pipe.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    negative_inputs = pipe.tokenizer(
        [negative_prompt],
        padding="max_length",
        max_length=text_inputs.input_ids.shape[-1],
        truncation=True,
        return_tensors="pt",
    )
    prompt_embeds = pipe.text_encoder(text_inputs.input_ids, return_dict=False)[0]
    negative_prompt_embeds = pipe.text_encoder(negative_inputs.input_ids, return_dict=False)[0]

    outputs = pipe(
        prompt=None,
        negative_prompt=None,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        height=64,
        width=64,
        num_inference_steps=10,
        guidance_scale=7.5,
        generator=torch.Generator(device="cpu").manual_seed(0),
        output=["images", "latents"],
    )

    assert len(outputs["images"]) == 1
    assert outputs["images"][0].size == (64, 64)
    assert outputs["latents"].ndim == 4


@pytest.mark.integration
def test_sd15_modular_repo_supports_latent_output():
    pipe = ModularPipeline.from_pretrained(str(SD15_MODULAR_REPO), trust_remote_code=True)
    pipe.load_components(
        torch_dtype=torch.float32,
        pretrained_model_name_or_path={"default": "hf-internal-testing/tiny-stable-diffusion-pipe"},
    )
    pipe.to("cpu")

    latents = pipe(
        prompt="a small test image of a lighthouse",
        negative_prompt="blurry",
        height=64,
        width=64,
        num_inference_steps=10,
        guidance_scale=7.5,
        generator=torch.Generator(device="cpu").manual_seed(0),
        output_type="latent",
        output="images",
    )

    assert isinstance(latents, torch.Tensor)
    assert latents.ndim == 4


@pytest.mark.integration
def test_sd15_modular_repo_supports_numpy_output():
    pipe = ModularPipeline.from_pretrained(str(SD15_MODULAR_REPO), trust_remote_code=True)
    pipe.load_components(
        torch_dtype=torch.float32,
        pretrained_model_name_or_path={"default": "hf-internal-testing/tiny-stable-diffusion-pipe"},
    )
    pipe.to("cpu")

    images = pipe(
        prompt="a small test image of a lighthouse",
        negative_prompt="blurry",
        height=64,
        width=64,
        num_inference_steps=10,
        guidance_scale=7.5,
        generator=torch.Generator(device="cpu").manual_seed(0),
        output_type="np",
        output="images",
    )

    assert isinstance(images, np.ndarray)
    assert images.shape[0] == 1
    assert images.shape[1:3] == (64, 64)


@pytest.mark.integration
def test_sd15_modular_repo_allows_scheduler_replacement():
    pipe = ModularPipeline.from_pretrained(str(SD15_MODULAR_REPO), trust_remote_code=True)
    pipe.load_components(
        torch_dtype=torch.float32,
        pretrained_model_name_or_path={"default": "hf-internal-testing/tiny-stable-diffusion-pipe"},
    )
    pipe.update_components(scheduler=DDIMScheduler.from_config(pipe.scheduler.config))
    pipe.to("cpu")

    outputs = pipe(
        prompt="a small test image of a lighthouse",
        negative_prompt="blurry",
        height=64,
        width=64,
        num_inference_steps=2,
        guidance_scale=7.5,
        generator=torch.Generator(device="cpu").manual_seed(0),
        output=["images", "latents"],
    )

    assert len(outputs["images"]) == 1
    assert outputs["images"][0].size == (64, 64)


def test_sd15_modular_repo_rejects_mismatched_negative_prompt_batch():
    pipe = ModularPipeline.from_pretrained(str(SD15_MODULAR_REPO), trust_remote_code=True)

    with pytest.raises(ValueError, match="same batch size"):
        pipe(
            prompt=["first", "second"],
            negative_prompt=["only one"],
            height=64,
            width=64,
        )


def test_sd15_modular_repo_rejects_invalid_latent_shape():
    pipe = ModularPipeline.from_pretrained(str(SD15_MODULAR_REPO), trust_remote_code=True)

    with pytest.raises(ValueError, match="latents"):
        pipe(
            prompt="shape validation",
            height=64,
            width=64,
            latents=torch.zeros((1, 4, 7, 7)),
        )


@pytest.mark.integration
def test_sd15_modular_repo_runs_tiny_img2img_smoke_inference():
    pipe = ModularPipeline.from_pretrained(str(SD15_MODULAR_REPO), trust_remote_code=True)
    pipe.load_components(
        torch_dtype=torch.float32,
        pretrained_model_name_or_path={"default": "hf-internal-testing/tiny-stable-diffusion-pipe"},
    )
    pipe.to("cpu")

    image = Image.new("RGB", (64, 64), color=(120, 160, 200))
    outputs = pipe(
        prompt="turn this into a painterly lighthouse scene",
        negative_prompt="blurry",
        image=image,
        height=64,
        width=64,
        strength=0.75,
        num_inference_steps=10,
        guidance_scale=7.5,
        generator=torch.Generator(device="cpu").manual_seed(0),
        output=["images", "latents"],
    )

    assert len(outputs["images"]) == 1
    assert outputs["images"][0].size == (64, 64)
    assert outputs["latents"].ndim == 4


@pytest.mark.integration
def test_sd15_modular_repo_img2img_is_deterministic_for_same_seed():
    pipe = ModularPipeline.from_pretrained(str(SD15_MODULAR_REPO), trust_remote_code=True)
    pipe.load_components(
        torch_dtype=torch.float32,
        pretrained_model_name_or_path={"default": "hf-internal-testing/tiny-stable-diffusion-pipe"},
    )
    pipe.to("cpu")

    image = Image.new("RGB", (64, 64), color=(80, 90, 100))
    kwargs = dict(
        prompt="make this image dreamy",
        negative_prompt="blurry",
        image=image,
        height=64,
        width=64,
        strength=0.6,
        num_inference_steps=10,
        guidance_scale=7.5,
        output="latents",
    )

    latents_a = pipe(
        generator=torch.Generator(device="cpu").manual_seed(1234),
        **kwargs,
    )
    latents_b = pipe(
        generator=torch.Generator(device="cpu").manual_seed(1234),
        **kwargs,
    )

    assert torch.allclose(latents_a, latents_b)


def test_sd15_modular_repo_rejects_invalid_img2img_strength_low():
    pipe = ModularPipeline.from_pretrained(str(SD15_MODULAR_REPO), trust_remote_code=True)

    with pytest.raises(ValueError, match="strength"):
        pipe(
            prompt="invalid strength",
            image=Image.new("RGB", (64, 64), color=(0, 0, 0)),
            strength=0.0,
            height=64,
            width=64,
        )


def test_sd15_modular_repo_rejects_invalid_img2img_strength_high():
    pipe = ModularPipeline.from_pretrained(str(SD15_MODULAR_REPO), trust_remote_code=True)

    with pytest.raises(ValueError, match="strength"):
        pipe(
            prompt="invalid strength",
            image=Image.new("RGB", (64, 64), color=(0, 0, 0)),
            strength=1.1,
            height=64,
            width=64,
        )


@pytest.mark.integration
def test_sd15_modular_repo_runs_tiny_inpaint_smoke_inference():
    pipe = ModularPipeline.from_pretrained(str(SD15_MODULAR_REPO), trust_remote_code=True)
    pipe.load_components(
        torch_dtype=torch.float32,
        pretrained_model_name_or_path={"default": "hf-internal-testing/tiny-stable-diffusion-pipe"},
    )
    pipe.to("cpu")

    image = Image.new("RGB", (64, 64), color=(120, 160, 200))
    mask_image = Image.new("L", (64, 64), color=0)
    mask_image.paste(255, (16, 16, 48, 48))

    outputs = pipe(
        prompt="replace the center with a glowing lighthouse",
        negative_prompt="blurry",
        image=image,
        mask_image=mask_image,
        height=64,
        width=64,
        strength=0.75,
        num_inference_steps=10,
        guidance_scale=7.5,
        generator=torch.Generator(device="cpu").manual_seed(0),
        output=["images", "latents"],
    )

    assert len(outputs["images"]) == 1
    assert outputs["images"][0].size == (64, 64)
    assert outputs["latents"].ndim == 4


@pytest.mark.integration
def test_sd15_modular_repo_inpaint_is_deterministic_for_same_seed():
    pipe = ModularPipeline.from_pretrained(str(SD15_MODULAR_REPO), trust_remote_code=True)
    pipe.load_components(
        torch_dtype=torch.float32,
        pretrained_model_name_or_path={"default": "hf-internal-testing/tiny-stable-diffusion-pipe"},
    )
    pipe.to("cpu")

    image = Image.new("RGB", (64, 64), color=(80, 90, 100))
    mask_image = Image.new("L", (64, 64), color=0)
    mask_image.paste(255, (20, 20, 44, 44))
    kwargs = dict(
        prompt="paint a bright portal in the center",
        negative_prompt="blurry",
        image=image,
        mask_image=mask_image,
        height=64,
        width=64,
        strength=0.6,
        num_inference_steps=10,
        guidance_scale=7.5,
        output="latents",
    )

    latents_a = pipe(generator=torch.Generator(device="cpu").manual_seed(5678), **kwargs)
    latents_b = pipe(generator=torch.Generator(device="cpu").manual_seed(5678), **kwargs)

    assert torch.allclose(latents_a, latents_b)


def test_sd15_modular_repo_rejects_mask_without_image():
    pipe = ModularPipeline.from_pretrained(str(SD15_MODULAR_REPO), trust_remote_code=True)

    with pytest.raises(ValueError, match="mask_image"):
        pipe(
            prompt="missing base image",
            mask_image=Image.new("L", (64, 64), color=255),
            height=64,
            width=64,
        )


def test_sd15_modular_repo_rejects_mismatched_mask_size():
    pipe = ModularPipeline.from_pretrained(str(SD15_MODULAR_REPO), trust_remote_code=True)

    with pytest.raises(ValueError, match="same size"):
        pipe(
            prompt="bad mask size",
            image=Image.new("RGB", (64, 64), color=(0, 0, 0)),
            mask_image=Image.new("L", (32, 32), color=255),
            height=64,
            width=64,
        )
