from pathlib import Path

import PIL.Image
import torch

from backend.modular_diffusers.pixeldit import PixelDiTModularPipeline, PixelDiTText2ImgBlocks
from backend.modular_diffusers.pixeldit.encoders import apply_chi_prompt, select_chi_token_window
from backend.modular_diffusers.pixeldit.pixeldit_transformer import PixelDiTTransformer2DModel
from backend.modular_diffusers.pixeldit.sampling import flow_dpm_sample


def _tiny_transformer() -> PixelDiTTransformer2DModel:
    return PixelDiTTransformer2DModel(
        in_channels=3,
        patch_size=4,
        num_groups=4,
        hidden_size=16,
        pixel_hidden_size=4,
        pixel_attn_hidden_size=16,
        pixel_num_groups=4,
        patch_depth=1,
        pixel_depth=1,
        num_text_blocks=1,
        txt_embed_dim=8,
        txt_max_length=4,
        repa_encoder_index=-1,
        image_size=8,
        flow_shift=1.0,
        default_steps=1,
    )


class _FakeTokenBatch:
    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids
        self.attention_mask = torch.ones_like(input_ids)

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self


class _FakeTokenizer:
    def __init__(self, chi_token_count: int = 3):
        self.chi_token_count = chi_token_count
        self.padding_side = "left"
        self.calls = []

    def encode(self, text: str):
        return list(range(self.chi_token_count))

    def __call__(self, texts, *, max_length, padding, truncation, return_tensors):
        self.calls.append({"texts": list(texts), "max_length": max_length})
        input_ids = torch.arange(max_length).view(1, max_length).repeat(len(texts), 1)
        return _FakeTokenBatch(input_ids)


class _FakeTextEncoder(torch.nn.Module):
    def __init__(self, embed_dim: int = 8):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.embed_dim = embed_dim

    def forward(self, input_ids, attention_mask):
        embeds = input_ids.float().unsqueeze(-1).repeat(1, 1, self.embed_dim)
        return (embeds,)


def test_pixeldit_transformer_forward_with_tiny_config():
    model = _tiny_transformer()
    sample = torch.randn(1, 3, 8, 8)
    timestep = torch.ones(1)
    embeds = torch.randn(1, 4, 8)

    output = model(sample, timestep, embeds)

    assert output["sample"].shape == sample.shape


def test_pixeldit_flow_dpm_sampler_with_tiny_config():
    model = _tiny_transformer()
    latents = torch.randn(1, 3, 8, 8)
    embeds = torch.randn(1, 4, 8)
    negative_embeds = torch.zeros_like(embeds)

    output = flow_dpm_sample(
        model,
        latents,
        embeds,
        negative_embeds,
        attention_mask=None,
        negative_attention_mask=None,
        num_inference_steps=2,
        guidance_scale=2.0,
        flow_shift=4.0,
        interval_guidance=(0.0, 1.0),
    )

    assert output.shape == latents.shape
    assert torch.isfinite(output).all()


def test_pixeldit_modular_pipeline_runs_with_prompt_embeds():
    pipe = PixelDiTModularPipeline(blocks=PixelDiTText2ImgBlocks())
    pipe.update_components(transformer=_tiny_transformer())
    generator = torch.Generator(device="cpu").manual_seed(123)
    prompt_embeds = torch.randn(1, 4, 8, generator=generator)
    negative_prompt_embeds = torch.zeros_like(prompt_embeds)

    images = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        height=8,
        width=8,
        num_inference_steps=1,
        guidance_scale=1.0,
        generator=torch.Generator(device="cpu").manual_seed(456),
        output="images",
    )

    assert len(images) == 1
    assert isinstance(images[0], PIL.Image.Image)
    assert images[0].size == (8, 8)


def test_pixeldit_modular_pipeline_loads_tiny_transformer_component(tmp_path: Path):
    repo = tmp_path / "pixeldit_repo"
    transformer_dir = repo / "transformer"
    transformer_dir.mkdir(parents=True)
    model = _tiny_transformer()
    model.save_pretrained(transformer_dir)

    loaded = PixelDiTTransformer2DModel.from_pretrained(transformer_dir)

    pipe = PixelDiTModularPipeline(blocks=PixelDiTText2ImgBlocks())
    pipe.update_components(transformer=loaded)
    prompt_embeds = torch.randn(1, 4, 8)

    images = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=torch.zeros_like(prompt_embeds),
        height=8,
        width=8,
        num_inference_steps=1,
        guidance_scale=1.0,
        output="images",
    )

    assert len(images) == 1


def test_pixeldit_chi_prompt_helpers_select_official_token_window():
    prompts = apply_chi_prompt("a glass greenhouse", "CHI: ")
    assert prompts == ["CHI: a glass greenhouse"]

    values = torch.arange(10).view(1, 10)
    selected = select_chi_token_window(values, model_max_length=4)

    assert selected.tolist() == [[0, 7, 8, 9]]


def test_pixeldit_modular_pipeline_applies_chi_prompt_to_positive_prompt_only():
    tokenizer = _FakeTokenizer(chi_token_count=3)
    pipe = PixelDiTModularPipeline(blocks=PixelDiTText2ImgBlocks())
    pipe.update_components(
        transformer=_tiny_transformer(),
        tokenizer=tokenizer,
        text_encoder=_FakeTextEncoder(embed_dim=8),
    )

    images = pipe(
        prompt="greenhouse",
        negative_prompt="bad",
        use_chi_prompt=True,
        chi_prompt="CHI: ",
        height=8,
        width=8,
        num_inference_steps=1,
        guidance_scale=1.0,
        generator=torch.Generator(device="cpu").manual_seed(456),
        output="images",
    )

    assert len(images) == 1
    assert tokenizer.padding_side == "right"
    assert tokenizer.calls[0] == {"texts": ["CHI: greenhouse"], "max_length": 5}
    assert tokenizer.calls[1] == {"texts": ["bad"], "max_length": 4}
