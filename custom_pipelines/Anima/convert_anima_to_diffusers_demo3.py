"""
Convert Anima checkpoints to the local SynthaEngine Anima Diffusers layout.

This is a separate adaptation of the upstream Anima converter. It intentionally
targets ``custom_pipelines/Anima/anima_pipeline.py`` and the component layout
used by CalamitousFelicitousness/Anima-Preview-3-sdnext-diffusers:

    text_encoder/
    tokenizer/
    t5_tokenizer/
    llm_adapter/
    transformer/
    vae/
    scheduler/
    model_index.json
    pipeline.py

Example:
```powershell
.venv\\Scripts\\python.exe custom_pipelines\\Anima\\convert_anima_to_diffusers_demo3.py `
  --transformer_ckpt_path "D:\\diffusion\\checkpoints\\Anima\\diffusion_models\\anima-preview.safetensors" `
  --text_encoder_ckpt_path "D:\\diffusion\\checkpoints\\Anima\\text_encoders\\qwen_3_06b_base.safetensors" `
  --vae_ckpt_path "D:\\diffusion\\checkpoints\\Anima\\vae\\qwen_image_vae.safetensors" `
  --qwen_tokenizer_path "D:\\diffusion\\models\\Qwen3-0.6B-Base" `
  --t5_tokenizer_path "D:\\diffusion\\models\\t5-large" `
  --output_path "D:\\diffusion\\diffusers\\Anima-Preview-3-demo3" `
  --llm_adapter_modeling_path "path\\to\\modeling_llm_adapter.py" `
  --save_pipeline
```

Required helper files/classes that are not part of stock Diffusers 0.38.0:
- ``convert_cosmos_to_diffusers.py`` providing ``convert_transformer``.
- ``modeling_llm_adapter.py`` providing ``AnimaLLMAdapter``. Download it from
  the converted Anima repo's ``llm_adapter`` folder and pass
  ``--llm_adapter_modeling_path``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import shutil
import sys
from typing import Any

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen3Config, Qwen3Model, T5TokenizerFast

from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
from custom_pipelines.Anima.anima_pipeline import AnimaTextToImagePipeline


DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

ANIMA_MODEL_INDEX = {
    "_class_name": "AnimaTextToImagePipeline",
    "_diffusers_version": "0.38.0",
    "text_encoder": ["transformers", "Qwen3Model"],
    "tokenizer": ["transformers", "PreTrainedTokenizerFast"],
    "t5_tokenizer": ["transformers", "T5TokenizerFast"],
    "llm_adapter": ["modeling_llm_adapter", "AnimaLLMAdapter"],
    "transformer": ["diffusers", "CosmosTransformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLWan"],
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
}


def import_from_file(module_name: str, path: str | pathlib.Path):
    module_path = pathlib.Path(path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_name!r} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_convert_transformer():
    try:
        from convert_cosmos_to_diffusers import convert_transformer

        return convert_transformer
    except ImportError as exc:
        raise ImportError(
            "Missing convert_cosmos_to_diffusers.py. Put the Cosmos conversion helper "
            "on PYTHONPATH or beside this script before running conversion."
        ) from exc


def load_llm_adapter_class(modeling_path: str | None):
    if not modeling_path:
        raise ValueError(
            "--llm_adapter_modeling_path is required. Download modeling_llm_adapter.py "
            "from the converted Anima Diffusers repo's llm_adapter folder."
        )
    module = import_from_file("modeling_llm_adapter", modeling_path)
    if not hasattr(module, "AnimaLLMAdapter"):
        raise ImportError(f"{modeling_path} does not define AnimaLLMAdapter.")
    return module.AnimaLLMAdapter


def rename_residual_key(key: str) -> str:
    replacements = {
        ".residual.0.": ".norm1.",
        ".residual.2.": ".conv1.",
        ".residual.3.": ".norm2.",
        ".residual.6.": ".conv2.",
        ".shortcut.": ".conv_shortcut.",
    }
    for old, new in replacements.items():
        key = key.replace(old, new)
    return key


def rename_mid_key(key: str) -> str:
    replacements = {
        ".middle.0.": ".mid_block.resnets.0.",
        ".middle.1.": ".mid_block.attentions.0.",
        ".middle.2.": ".mid_block.resnets.1.",
    }
    for old, new in replacements.items():
        key = key.replace(old, new)
    return rename_residual_key(key)


def rename_decoder_upsample_key(key: str) -> str:
    prefix = "decoder.upsamples."
    suffix = key.removeprefix(prefix)
    index_str, rest = suffix.split(".", 1)
    index = int(index_str)

    if index in (3, 7, 11):
        block_index = (index - 3) // 4
        new_key = f"decoder.up_blocks.{block_index}.upsamplers.0.{rest}"
    else:
        block_index = index // 4
        resnet_index = index % 4
        new_key = f"decoder.up_blocks.{block_index}.resnets.{resnet_index}.{rest}"

    return rename_residual_key(new_key)


def convert_wan_vae_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("conv1."):
            new_key = key.replace("conv1.", "quant_conv.", 1)
        elif key.startswith("conv2."):
            new_key = key.replace("conv2.", "post_quant_conv.", 1)
        elif key.startswith("encoder.conv1."):
            new_key = key.replace("encoder.conv1.", "encoder.conv_in.", 1)
        elif key.startswith("decoder.conv1."):
            new_key = key.replace("decoder.conv1.", "decoder.conv_in.", 1)
        elif key.startswith("encoder.downsamples."):
            new_key = rename_residual_key(key.replace("encoder.downsamples.", "encoder.down_blocks.", 1))
        elif key.startswith("decoder.upsamples."):
            new_key = rename_decoder_upsample_key(key)
        elif key.startswith("encoder.middle.") or key.startswith("decoder.middle."):
            new_key = rename_mid_key(key)
        elif key.startswith("encoder.head.0."):
            new_key = key.replace("encoder.head.0.", "encoder.norm_out.", 1)
        elif key.startswith("encoder.head.2."):
            new_key = key.replace("encoder.head.2.", "encoder.conv_out.", 1)
        elif key.startswith("decoder.head.0."):
            new_key = key.replace("decoder.head.0.", "decoder.norm_out.", 1)
        elif key.startswith("decoder.head.2."):
            new_key = key.replace("decoder.head.2.", "decoder.conv_out.", 1)
        else:
            new_key = rename_residual_key(key)

        if new_key in converted_state_dict:
            raise ValueError(f"Duplicate converted VAE key: {new_key}")
        converted_state_dict[new_key] = value
    return converted_state_dict


def report_key_mismatch(component: str, missing_keys: set[str], unexpected_keys: set[str]) -> None:
    if missing_keys:
        print(f"ERROR: missing {component} keys ({len(missing_keys)}):", file=sys.stderr)
        for key in sorted(missing_keys):
            print(key, file=sys.stderr)
    if unexpected_keys:
        print(f"ERROR: unexpected {component} keys ({len(unexpected_keys)}):", file=sys.stderr)
        for key in sorted(unexpected_keys):
            print(key, file=sys.stderr)


def convert_wan_vae(state_dict: dict[str, torch.Tensor]) -> AutoencoderKLWan:
    converted_state_dict = convert_wan_vae_state_dict(state_dict)
    with init_empty_weights():
        vae = AutoencoderKLWan()

    expected_keys = set(vae.state_dict().keys())
    converted_keys = set(converted_state_dict.keys())
    missing_keys = expected_keys - converted_keys
    unexpected_keys = converted_keys - expected_keys
    if missing_keys or unexpected_keys:
        report_key_mismatch("VAE", missing_keys, unexpected_keys)
        sys.exit(1)

    vae.load_state_dict(converted_state_dict, strict=True, assign=True)
    return vae


def infer_llm_adapter_config(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    model_dim = state_dict["blocks.0.self_attn.q_proj.weight"].shape[0]
    source_dim = state_dict["blocks.0.cross_attn.k_proj.weight"].shape[1]
    target_vocab_size, target_dim = state_dict["embed.weight"].shape
    attention_head_dim = state_dict["blocks.0.self_attn.q_norm.weight"].shape[0]
    num_layers = 1 + max(int(key.split(".")[1]) for key in state_dict if key.startswith("blocks."))
    return {
        "source_dim": source_dim,
        "target_dim": target_dim,
        "model_dim": model_dim,
        "num_layers": num_layers,
        "num_heads": model_dim // attention_head_dim,
        "mlp_ratio": 4.0,
        "vocab_size": target_vocab_size,
        "use_self_attn": True,
    }


def convert_llm_adapter(state_dict: dict[str, torch.Tensor], adapter_cls):
    config = infer_llm_adapter_config(state_dict)
    with init_empty_weights():
        llm_adapter = adapter_cls(**config)

    expected_keys = set(llm_adapter.state_dict().keys())
    converted_keys = set(state_dict.keys())
    missing_keys = expected_keys - converted_keys
    unexpected_keys = converted_keys - expected_keys
    if missing_keys or unexpected_keys:
        report_key_mismatch("LLM adapter", missing_keys, unexpected_keys)
        sys.exit(1)

    llm_adapter.load_state_dict(state_dict, strict=True, assign=True)
    return llm_adapter


def infer_qwen3_config(state_dict: dict[str, torch.Tensor]) -> Qwen3Config:
    vocab_size, hidden_size = state_dict["embed_tokens.weight"].shape
    intermediate_size = state_dict["layers.0.mlp.gate_proj.weight"].shape[0]
    num_hidden_layers = 1 + max(int(key.split(".")[1]) for key in state_dict if key.startswith("layers."))
    head_dim = state_dict["layers.0.self_attn.q_norm.weight"].shape[0]
    num_attention_heads = state_dict["layers.0.self_attn.q_proj.weight"].shape[0] // head_dim
    num_key_value_heads = state_dict["layers.0.self_attn.k_proj.weight"].shape[0] // head_dim
    return Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        head_dim=head_dim,
        attention_bias=False,
        tie_word_embeddings=False,
    )


def convert_text_encoder(state_dict: dict[str, torch.Tensor]) -> Qwen3Model:
    state_dict = {key.removeprefix("model."): value for key, value in state_dict.items()}
    config = infer_qwen3_config(state_dict)
    with init_empty_weights():
        text_encoder = Qwen3Model(config)

    expected_keys = set(text_encoder.state_dict().keys())
    converted_keys = set(state_dict.keys())
    missing_keys = expected_keys - converted_keys
    unexpected_keys = converted_keys - expected_keys
    if missing_keys or unexpected_keys:
        report_key_mismatch("Qwen3", missing_keys, unexpected_keys)
        sys.exit(1)

    text_encoder.load_state_dict(state_dict, strict=True, assign=True)
    return text_encoder


def split_anima_transformer_checkpoint(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    transformer_state_dict = {}
    llm_adapter_state_dict = {}
    adapter_prefix = "net.llm_adapter."

    for key, value in state_dict.items():
        if key.startswith(adapter_prefix):
            llm_adapter_state_dict[key.removeprefix(adapter_prefix)] = value
        else:
            transformer_state_dict[key] = value

    return transformer_state_dict, llm_adapter_state_dict


def copy_local_pipeline_files(output_path: pathlib.Path, modeling_path: str | pathlib.Path) -> None:
    script_dir = pathlib.Path(__file__).resolve().parent
    shutil.copy2(script_dir / "anima_pipeline.py", output_path / "pipeline.py")
    shutil.copy2(modeling_path, output_path / "llm_adapter" / "modeling_llm_adapter.py")
    (output_path / "model_index.json").write_text(
        json.dumps(ANIMA_MODEL_INDEX, indent=2) + "\n",
        encoding="utf-8",
    )


def save_pipeline(args, transformer, llm_adapter, text_encoder, vae) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_tokenizer_path)
    t5_tokenizer = T5TokenizerFast.from_pretrained(args.t5_tokenizer_path)
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)

    pipe = AnimaTextToImagePipeline(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        t5_tokenizer=t5_tokenizer,
        llm_adapter=llm_adapter,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
    )
    pipe.save_pretrained(args.output_path, safe_serialization=True, max_shard_size=args.max_shard_size)
    copy_local_pipeline_files(pathlib.Path(args.output_path), args.llm_adapter_modeling_path)


def save_components(args, transformer, llm_adapter, text_encoder, vae) -> None:
    output_path = pathlib.Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    transformer.save_pretrained(output_path / "transformer", safe_serialization=True, max_shard_size=args.max_shard_size)
    llm_adapter.save_pretrained(output_path / "llm_adapter", safe_serialization=True, max_shard_size=args.max_shard_size)
    text_encoder.save_pretrained(output_path / "text_encoder", safe_serialization=True, max_shard_size=args.max_shard_size)
    vae.save_pretrained(output_path / "vae", safe_serialization=True, max_shard_size=args.max_shard_size)
    copy_local_pipeline_files(output_path, args.llm_adapter_modeling_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer_ckpt_path", type=str, required=True, help="Path to Anima DiT safetensors")
    parser.add_argument("--text_encoder_ckpt_path", type=str, required=True, help="Path to Qwen3 text encoder")
    parser.add_argument("--vae_ckpt_path", type=str, required=True, help="Path to Qwen-Image VAE safetensors")
    parser.add_argument("--qwen_tokenizer_path", type=str, default=None)
    parser.add_argument("--t5_tokenizer_path", type=str, default=None)
    parser.add_argument("--llm_adapter_modeling_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--save_pipeline", action="store_true")
    parser.add_argument("--dtype", default="bf16", choices=list(DTYPE_MAPPING.keys()))
    parser.add_argument("--max_shard_size", default="5GB")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    dtype = DTYPE_MAPPING[args.dtype]
    convert_transformer = load_convert_transformer()
    llm_adapter_cls = load_llm_adapter_class(args.llm_adapter_modeling_path)

    raw_transformer_state_dict = load_file(args.transformer_ckpt_path, device="cpu")
    transformer_state_dict, llm_adapter_state_dict = split_anima_transformer_checkpoint(raw_transformer_state_dict)
    transformer = convert_transformer(
        "Cosmos-2.0-Diffusion-2B-Text2Image",
        state_dict=transformer_state_dict,
        weights_only=True,
    ).to(dtype=dtype)
    llm_adapter = convert_llm_adapter(llm_adapter_state_dict, llm_adapter_cls).to(dtype=dtype)

    text_encoder_state_dict = load_file(args.text_encoder_ckpt_path, device="cpu")
    text_encoder = convert_text_encoder(text_encoder_state_dict).to(dtype=dtype)

    vae_state_dict = load_file(args.vae_ckpt_path, device="cpu")
    vae = convert_wan_vae(vae_state_dict).to(dtype=dtype)

    if args.save_pipeline:
        if args.qwen_tokenizer_path is None or args.t5_tokenizer_path is None:
            raise ValueError("`--qwen_tokenizer_path` and `--t5_tokenizer_path` are required with `--save_pipeline`.")
        save_pipeline(args, transformer, llm_adapter, text_encoder, vae)
    else:
        save_components(args, transformer, llm_adapter, text_encoder, vae)


if __name__ == "__main__":
    main()
