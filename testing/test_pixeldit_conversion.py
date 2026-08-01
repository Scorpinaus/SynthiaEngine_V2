import argparse
import importlib
import json
from pathlib import Path
import sys

import torch
from diffusers import ComponentsManager
from safetensors.torch import load_file, save_file

from backend.modular_diffusers.pixeldit import PixelDiTModularPipeline
from backend.modular_diffusers.pixeldit.pixeldit_transformer import PixelDiTTransformer2DModel
from tools.convert_pixeldit_to_diffusers import convert


def _tiny_config() -> dict:
    return {
        "architectures": ["PixDiT_T2I"],
        "model_type": "pixeldit",
        "in_channels": 3,
        "patch_size": 4,
        "num_groups": 4,
        "hidden_size": 16,
        "pixel_hidden_size": 4,
        "pixel_attn_hidden_size": 16,
        "pixel_num_groups": 4,
        "patch_depth": 1,
        "pixel_depth": 1,
        "num_text_blocks": 1,
        "txt_embed_dim": 8,
        "txt_max_length": 4,
        "use_text_rope": True,
        "text_rope_theta": 10000.0,
        "repa_encoder_index": -1,
        "use_pixel_abs_pos": True,
        "image_size": 8,
        "text_encoder": "gemma-2-2b-it",
        "scheduler": {"type": "flow_matching", "flow_shift": 4.0},
        "sampling": {
            "algorithm": "flow_dpm-solver",
            "default_steps": 2,
            "default_cfg_scale": 2.75,
            "default_negative_prompt": "low quality",
        },
    }


def _args(tmp_path: Path, input_dir: Path, output_dir: Path, **overrides):
    values = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "checkpoint": None,
        "elm_safetensors": None,
        "text_encoder_source": None,
        "tokenizer_source": None,
        "torch_dtype": "bfloat16",
        "overwrite": False,
        "dry_run": False,
        "allow_missing_text_components": True,
        "write_hashes": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_pixeldit_converter_writes_diffusers_component_repo(tmp_path: Path, monkeypatch):
    input_dir = tmp_path / "PixelDiT"
    input_dir.mkdir()
    (input_dir / "config.json").write_text(json.dumps(_tiny_config()), encoding="utf-8")

    model = PixelDiTTransformer2DModel(
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
        flow_shift=4.0,
        default_steps=2,
        default_negative_prompt="low quality",
    )
    torch.save({"state_dict": model.state_dict()}, input_dir / "pixeldit_t2i_v1.pth")

    output_dir = tmp_path / "PixelDiT-Diffusers"
    manifest = convert(_args(tmp_path, input_dir, output_dir))

    assert manifest["state_dict_tensor_count"] == len(model.state_dict())
    assert (output_dir / "config.json").exists()
    assert (output_dir / "modular_model_index.json").exists()
    assert (output_dir / "transformer" / "config.json").exists()
    weights_path = output_dir / "transformer" / "diffusion_pytorch_model.safetensors"
    assert weights_path.exists()
    assert "core.y_pos_embedding" in load_file(weights_path)
    assert (output_dir / "text_encoder" / "README.md").exists()
    assert (output_dir / "tokenizer" / "README.md").exists()

    loaded = PixelDiTTransformer2DModel.from_pretrained(output_dir / "transformer")
    assert loaded.config.patch_size == 4
    assert loaded.config.txt_embed_dim == 8

    module_cache = tmp_path / "hf_modules_cache"
    monkeypatch.setenv("HF_MODULES_CACHE", str(module_cache))
    import diffusers.utils.dynamic_modules_utils as dynamic_modules_utils

    monkeypatch.setattr(dynamic_modules_utils, "HF_MODULES_CACHE", str(module_cache))
    # Diffusers maps every local custom repository to the same dynamic package.
    # Put this test's cache first and remove the package loaded from any earlier
    # local-repository test before switching to this generated repo.
    monkeypatch.syspath_prepend(str(module_cache))
    for module_name in tuple(sys.modules):
        if module_name == "diffusers_modules" or module_name.startswith("diffusers_modules."):
            monkeypatch.delitem(sys.modules, module_name)
    importlib.invalidate_caches()

    components_manager = ComponentsManager()
    pipe = PixelDiTModularPipeline.from_pretrained(
        output_dir,
        trust_remote_code=True,
        components_manager=components_manager,
        collection="pixeldit_test",
    )
    pipe.load_components(names="transformer")
    assert isinstance(pipe.transformer, PixelDiTTransformer2DModel)
    assert components_manager.get_one(name="transformer", collection="pixeldit_test") is pipe.transformer


def test_pixeldit_converter_requires_text_components_for_complete_repo(tmp_path: Path):
    input_dir = tmp_path / "PixelDiT"
    input_dir.mkdir()
    (input_dir / "config.json").write_text(json.dumps(_tiny_config()), encoding="utf-8")
    torch.save({"state_dict": {}}, input_dir / "pixeldit_t2i_v1.pth")

    output_dir = tmp_path / "PixelDiT-Diffusers"
    args = _args(tmp_path, input_dir, output_dir, allow_missing_text_components=False)
    try:
        convert(args)
    except ValueError as exc:
        assert "text_encoder/tokenizer sources are required" in str(exc)
    else:
        raise AssertionError("Expected converter to require local text components.")


def test_pixeldit_converter_copies_text_components_into_output(tmp_path: Path):
    input_dir = tmp_path / "PixelDiT"
    input_dir.mkdir()
    (input_dir / "config.json").write_text(json.dumps(_tiny_config()), encoding="utf-8")
    torch.save({"state_dict": {}}, input_dir / "pixeldit_t2i_v1.pth")
    text_encoder_source = tmp_path / "gemma_text_encoder"
    tokenizer_source = tmp_path / "gemma_tokenizer"
    text_encoder_source.mkdir()
    tokenizer_source.mkdir()
    (text_encoder_source / "config.json").write_text('{"model_type":"gemma2"}', encoding="utf-8")
    (tokenizer_source / "tokenizer.json").write_text("{}", encoding="utf-8")

    output_dir = tmp_path / "PixelDiT-Diffusers"
    convert(
        _args(
            tmp_path,
            input_dir,
            output_dir,
            text_encoder_source=str(text_encoder_source),
            tokenizer_source=str(tokenizer_source),
            allow_missing_text_components=False,
        )
    )

    assert (output_dir / "text_encoder" / "config.json").exists()
    assert (output_dir / "tokenizer" / "tokenizer.json").exists()
    index = json.loads((output_dir / "modular_model_index.json").read_text(encoding="utf-8"))
    assert Path(index["text_encoder"][2]["pretrained_model_name_or_path"]).name == "text_encoder"
    assert Path(index["tokenizer"][2]["pretrained_model_name_or_path"]).name == "tokenizer"
    assert index["text_encoder"][2]["subfolder"] == ""
    assert index["tokenizer"][2]["subfolder"] == ""


def test_pixeldit_converter_unpacks_elm_safetensors_bundle(tmp_path: Path):
    input_dir = tmp_path / "PixelDiT"
    input_dir.mkdir()
    (input_dir / "config.json").write_text(json.dumps(_tiny_config()), encoding="utf-8")
    torch.save({"state_dict": {}}, input_dir / "pixeldit_t2i_v1.pth")
    save_file(
        {
            "model.embed_tokens.weight": torch.zeros(2, 3, dtype=torch.bfloat16),
            "spiece_model": torch.tensor([1, 2, 3, 4], dtype=torch.uint8),
        },
        input_dir / "gemma_2_2b_it_elm_bf16.safetensors",
    )

    output_dir = tmp_path / "PixelDiT-Diffusers"
    manifest = convert(_args(tmp_path, input_dir, output_dir, allow_missing_text_components=False))

    text_weights = load_file(output_dir / "text_encoder" / "model.safetensors")
    assert "model.embed_tokens.weight" in text_weights
    assert "spiece_model" not in text_weights
    assert (output_dir / "tokenizer" / "tokenizer.model").read_bytes() == b"\x01\x02\x03\x04"
    tokenizer_config = json.loads((output_dir / "tokenizer" / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert tokenizer_config["tokenizer_class"] == "GemmaTokenizer"
    assert tokenizer_config["padding_side"] == "right"
    assert manifest["text_components_complete"] is True
    assert manifest["elm_safetensors"].endswith("gemma_2_2b_it_elm_bf16.safetensors")


def test_pixeldit_converter_dry_run_does_not_write(tmp_path: Path):
    input_dir = tmp_path / "PixelDiT"
    input_dir.mkdir()
    (input_dir / "config.json").write_text(json.dumps(_tiny_config()), encoding="utf-8")
    output_dir = tmp_path / "PixelDiT-Diffusers"

    summary = convert(_args(tmp_path, input_dir, output_dir, dry_run=True, allow_missing_text_components=False))

    assert summary["dry_run"] is True
    assert summary["checkpoint_exists"] is False
    assert not output_dir.exists()
