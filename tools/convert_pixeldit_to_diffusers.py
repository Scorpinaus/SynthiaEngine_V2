"""Convert NVIDIA PixelDiT T2I weights into a local Modular Diffusers repo."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
PIXELDIT_MODULE_DIR = REPO_ROOT / "backend" / "modular_diffusers" / "pixeldit"
DEFAULT_INPUT_DIR = Path(r"D:\diffusion\diffusers\PixelDiT")
DEFAULT_OUTPUT_DIR = Path(r"D:\diffusion\diffusers\PixelDiT-Diffusers")
DEFAULT_CHECKPOINT_NAME = "pixeldit_t2i_v1.pth"
DEFAULT_ELM_SAFETENSORS_NAME = "gemma_2_2b_it_elm_bf16.safetensors"
CODE_FILES = [
    "__init__.py",
    "block.py",
    "config.json",
    "denoise.py",
    "encoders.py",
    "modular_blocks_pixeldit.py",
    "modular_config.json",
    "modular_pipeline.py",
    "pixeldit_transformer.py",
    "sampling.py",
]


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copytree_contents(src: Path, dest: Path, *, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Component source does not exist: {src}")
    if dest.exists() and overwrite:
        shutil.rmtree(dest)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"Destination component folder already exists and is not empty: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dest / src.name)
        return
    for item in src.iterdir():
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _ensure_empty_or_missing(dest: Path, *, overwrite: bool) -> None:
    if dest.exists() and overwrite:
        shutil.rmtree(dest)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"Destination component folder already exists and is not empty: {dest}")
    dest.mkdir(parents=True, exist_ok=True)


def _load_pixeldit_config(input_dir: Path) -> dict[str, Any]:
    config_path = input_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"PixelDiT config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _transformer_config_from_source(source_config: dict[str, Any], *, torch_dtype: str) -> dict[str, Any]:
    scheduler = source_config.get("scheduler") or {}
    sampling = source_config.get("sampling") or {}
    return {
        "_class_name": "PixelDiTTransformer2DModel",
        "architectures": ["PixelDiTTransformer2DModel"],
        "model_type": "pixeldit",
        "in_channels": int(source_config.get("in_channels", 3)),
        "patch_size": int(source_config.get("patch_size", 16)),
        "num_groups": int(source_config.get("num_groups", 24)),
        "hidden_size": int(source_config.get("hidden_size", 1536)),
        "pixel_hidden_size": int(source_config.get("pixel_hidden_size", 16)),
        "pixel_attn_hidden_size": int(source_config.get("pixel_attn_hidden_size", 1152)),
        "pixel_num_groups": int(source_config.get("pixel_num_groups", 16)),
        "patch_depth": int(source_config.get("patch_depth", 14)),
        "pixel_depth": int(source_config.get("pixel_depth", 2)),
        "num_text_blocks": int(source_config.get("num_text_blocks", 4)),
        "txt_embed_dim": int(source_config.get("txt_embed_dim", 2304)),
        "txt_max_length": int(source_config.get("txt_max_length", 300)),
        "use_text_rope": bool(source_config.get("use_text_rope", True)),
        "text_rope_theta": float(source_config.get("text_rope_theta", 10000.0)),
        "repa_encoder_index": int(source_config.get("repa_encoder_index", 6)),
        "use_pixel_abs_pos": bool(source_config.get("use_pixel_abs_pos", True)),
        "image_size": int(source_config.get("image_size", 1024)),
        "text_encoder": str(source_config.get("text_encoder", "gemma-2-2b-it")),
        "flow_shift": float(scheduler.get("flow_shift", 4.0)),
        "default_steps": int(sampling.get("default_steps", 50)),
        "default_cfg_scale": float(sampling.get("default_cfg_scale", 2.75)),
        "default_negative_prompt": str(
            sampling.get(
                "default_negative_prompt",
                "low quality, worst quality, over-saturated, blurry, deformed, watermark",
            )
        ),
        "torch_dtype": torch_dtype,
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _write_placeholder_component(dest: Path, component_name: str, source_hint: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "README.md").write_text(
        (
            f"# Missing PixelDiT {component_name}\n\n"
            "The original PixelDiT checkpoint folder does not contain this component.\n"
            f"Populate this folder from a local Transformers-compatible `{source_hint}` model before text prompt inference.\n"
        ),
        encoding="utf-8",
    )


def _looks_like_pixeldit_elm_bundle(path: Path) -> bool:
    if not path.is_file() or path.suffix != ".safetensors":
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
        return "spiece_model" in keys and "model.embed_tokens.weight" in keys
    except Exception:
        return False


def _gemma2_2b_config(*, torch_dtype: str) -> dict[str, Any]:
    return {
        "architectures": ["Gemma2ForCausalLM"],
        "model_type": "gemma2",
        "vocab_size": 256000,
        "hidden_size": 2304,
        "intermediate_size": 9216,
        "num_hidden_layers": 26,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "hidden_activation": "gelu_pytorch_tanh",
        "max_position_embeddings": 8192,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-06,
        "use_cache": True,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "bos_token_id": 2,
        "tie_word_embeddings": True,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "attention_bias": False,
        "attention_dropout": 0.0,
        "query_pre_attn_scalar": 256,
        "sliding_window": 4096,
        "layer_types": ["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(26)],
        "final_logit_softcapping": 30.0,
        "attn_logit_softcapping": 50.0,
        "_name_or_path": "Efficient-Large-Model/gemma-2-2b-it",
        "dtype": torch_dtype,
        "torch_dtype": torch_dtype,
    }


def _gemma_tokenizer_config() -> dict[str, Any]:
    return {
        "add_bos_token": True,
        "add_eos_token": False,
        "bos_token": "<bos>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<eos>",
        "legacy": False,
        "model_max_length": 8192,
        "pad_token": "<pad>",
        "padding_side": "right",
        "tokenizer_class": "GemmaTokenizer",
        "unk_token": "<unk>",
    }


def _write_gemma_components_from_elm_bundle(src: Path, output_dir: Path, *, torch_dtype: str, overwrite: bool) -> None:
    text_encoder_dir = output_dir / "text_encoder"
    tokenizer_dir = output_dir / "tokenizer"
    _ensure_empty_or_missing(text_encoder_dir, overwrite=overwrite)
    _ensure_empty_or_missing(tokenizer_dir, overwrite=overwrite)

    text_tensors: dict[str, torch.Tensor] = {}
    with safe_open(src, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if "spiece_model" not in keys:
            raise ValueError(f"PixelDiT ELM bundle does not contain `spiece_model`: {src}")
        tokenizer_bytes = handle.get_tensor("spiece_model").cpu().numpy().tobytes()
        for key in keys:
            if key == "spiece_model":
                continue
            text_tensors[key] = handle.get_tensor(key).detach().cpu()

    save_file(
        text_tensors,
        text_encoder_dir / "model.safetensors",
        metadata={
            "format": "pt",
            "source_checkpoint": src.name,
            "model_type": "gemma2",
        },
    )
    _write_json(text_encoder_dir / "config.json", _gemma2_2b_config(torch_dtype=torch_dtype))

    (tokenizer_dir / "tokenizer.model").write_bytes(tokenizer_bytes)
    _write_json(tokenizer_dir / "tokenizer_config.json", _gemma_tokenizer_config())
    _write_json(
        tokenizer_dir / "special_tokens_map.json",
        {
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "unk_token": "<unk>",
        },
    )


def _modular_model_index(output_dir: Path, include_text_components: bool) -> dict[str, Any]:
    index: dict[str, Any] = {
        "_class_name": "ModularPipeline",
        "_blocks_class_name": "PixelDiTText2ImgBlocks",
        "transformer": [
            "backend.modular_diffusers.pixeldit.pixeldit_transformer",
            "PixelDiTTransformer2DModel",
            {
                "type_hint": [
                    "backend.modular_diffusers.pixeldit.pixeldit_transformer",
                    "PixelDiTTransformer2DModel",
                ],
                "pretrained_model_name_or_path": str(output_dir / "transformer"),
                "subfolder": "",
                "variant": None,
                "revision": None,
            },
        ],
    }
    if include_text_components:
        index["tokenizer"] = [
            "transformers",
            "AutoTokenizer",
            {
                "type_hint": ["transformers", "AutoTokenizer"],
                "pretrained_model_name_or_path": str(output_dir / "tokenizer"),
                "subfolder": "",
                "variant": None,
                "revision": None,
            },
        ]
        index["text_encoder"] = [
            "transformers",
            "AutoModelForCausalLM",
            {
                "type_hint": ["transformers", "AutoModelForCausalLM"],
                "pretrained_model_name_or_path": str(output_dir / "text_encoder"),
                "subfolder": "",
                "variant": None,
                "revision": None,
            },
        ]
    return index


def convert(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else input_dir / DEFAULT_CHECKPOINT_NAME
    source_config = _load_pixeldit_config(input_dir)

    elm_safetensors = Path(args.elm_safetensors) if getattr(args, "elm_safetensors", None) else None
    if elm_safetensors is None:
        candidate_elm_safetensors = input_dir / DEFAULT_ELM_SAFETENSORS_NAME
        if candidate_elm_safetensors.exists():
            elm_safetensors = candidate_elm_safetensors

    text_encoder_source = Path(args.text_encoder_source) if args.text_encoder_source else None
    tokenizer_source = Path(args.tokenizer_source) if args.tokenizer_source else text_encoder_source
    if text_encoder_source is not None and tokenizer_source == text_encoder_source and _looks_like_pixeldit_elm_bundle(text_encoder_source):
        elm_safetensors = text_encoder_source
        text_encoder_source = None
        tokenizer_source = None

    include_text_components = elm_safetensors is not None or (text_encoder_source is not None and tokenizer_source is not None)
    missing_text_components = not include_text_components
    if missing_text_components and not args.allow_missing_text_components and not args.dry_run:
        raise ValueError(
            "PixelDiT text_encoder/tokenizer sources are required for a complete converted repo. "
            "Pass --text-encoder-source and --tokenizer-source, or use --allow-missing-text-components "
            "for scaffold-only output."
        )

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_exists": checkpoint_path.exists(),
        "text_encoder_source": str(text_encoder_source) if text_encoder_source else None,
        "tokenizer_source": str(tokenizer_source) if tokenizer_source else None,
        "elm_safetensors": str(elm_safetensors) if elm_safetensors else None,
        "will_include_text_components": include_text_components,
        "torch_dtype": args.torch_dtype,
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        return summary

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"PixelDiT checkpoint not found: {checkpoint_path}")
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in CODE_FILES:
        shutil.copy2(PIXELDIT_MODULE_DIR / filename, output_dir / filename)
    readme_src = PIXELDIT_MODULE_DIR / "README.md"
    if readme_src.exists():
        shutil.copy2(readme_src, output_dir / "README.md")

    transformer_dir = output_dir / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    transformer_config = _transformer_config_from_source(source_config, torch_dtype=args.torch_dtype)
    _write_json(transformer_dir / "config.json", transformer_config)

    payload = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    state_dict = payload.get("state_dict") if isinstance(payload, dict) and "state_dict" in payload else payload
    if not isinstance(state_dict, dict):
        raise ValueError("PixelDiT checkpoint must be a state_dict or an object containing `state_dict`.")
    tensors = {str(key): value.detach().cpu() for key, value in state_dict.items() if torch.is_tensor(value)}
    if len(tensors) != len(state_dict):
        raise ValueError("PixelDiT state_dict contains non-tensor values.")
    save_file(
        tensors,
        transformer_dir / "diffusion_pytorch_model.safetensors",
        metadata={
            "format": "pt",
            "source_checkpoint": checkpoint_path.name,
            "model_type": "pixeldit",
        },
    )

    if elm_safetensors is not None:
        _write_gemma_components_from_elm_bundle(
            elm_safetensors,
            output_dir,
            torch_dtype=args.torch_dtype,
            overwrite=args.overwrite,
        )
    elif include_text_components:
        _copytree_contents(text_encoder_source, output_dir / "text_encoder", overwrite=args.overwrite)
        _copytree_contents(tokenizer_source, output_dir / "tokenizer", overwrite=args.overwrite)
    else:
        _write_placeholder_component(output_dir / "text_encoder", "text encoder", "gemma-2-2b-it")
        _write_placeholder_component(output_dir / "tokenizer", "tokenizer", "gemma-2-2b-it")

    _write_json(output_dir / "config.json", json.loads((PIXELDIT_MODULE_DIR / "config.json").read_text(encoding="utf-8")))
    _write_json(output_dir / "modular_config.json", json.loads((PIXELDIT_MODULE_DIR / "modular_config.json").read_text(encoding="utf-8")))
    _write_json(output_dir / "modular_model_index.json", _modular_model_index(output_dir, include_text_components))

    manifest = {
        **summary,
        "source_config": source_config,
        "state_dict_tensor_count": len(tensors),
        "state_dict_parameter_count": int(sum(tensor.numel() for tensor in tensors.values())),
        "checkpoint_sha256": _sha256(checkpoint_path) if args.write_hashes else None,
        "transformer_safetensors": "transformer/diffusion_pytorch_model.safetensors",
        "text_components_complete": include_text_components,
    }
    _write_json(output_dir / "conversion_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Folder containing PixelDiT config and pth.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Converted Diffusers repo output folder.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path. Defaults to input-dir/pixeldit_t2i_v1.pth.")
    parser.add_argument(
        "--elm-safetensors",
        default=None,
        help=(
            "PixelDiT Gemma ELM safetensors bundle to unpack. Defaults to "
            "input-dir/gemma_2_2b_it_elm_bf16.safetensors when present."
        ),
    )
    parser.add_argument(
        "--text-encoder-source",
        default=None,
        help="Local Transformers text encoder folder to copy, or a PixelDiT Gemma ELM safetensors bundle to unpack.",
    )
    parser.add_argument("--tokenizer-source", default=None, help="Local tokenizer folder to copy. Defaults to text encoder source.")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--overwrite", action="store_true", help="Replace output folder if it exists.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned conversion without writing.")
    parser.add_argument(
        "--allow-missing-text-components",
        action="store_true",
        help="Create placeholder text_encoder/tokenizer folders instead of requiring local sources.",
    )
    parser.add_argument("--write-hashes", action="store_true", help="Compute SHA256 of the source checkpoint.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = convert(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
