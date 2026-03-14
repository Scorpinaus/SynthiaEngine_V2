from pathlib import Path

import pytest
import torch


diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from diffusers import ModularPipeline


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
