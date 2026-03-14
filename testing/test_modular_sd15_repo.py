from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch


diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from diffusers import ModularPipeline
from diffusers.schedulers import DDIMScheduler


REPO_ROOT = Path(__file__).resolve().parents[1]
SD15_MODULAR_REPO = REPO_ROOT / "backend" / "modular_diffusers" / "sd15"


def test_sd15_modular_repo_loads_custom_blocks():
    pipe = ModularPipeline.from_pretrained(str(SD15_MODULAR_REPO), trust_remote_code=True)

    assert pipe.blocks.__class__.__name__ == "SD15Text2ImgBlocks"
    assert set(pipe.components.keys()) == {"tokenizer", "text_encoder", "unet", "vae", "scheduler"}


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
