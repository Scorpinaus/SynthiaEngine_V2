from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from safetensors import safe_open

SUPPORTED_EXTS = {".safetensors", ".pt", ".ckpt", ".ckpr", ".model"}


@dataclass(frozen=True)
class ModelArchitectureAnalysis:
    architecture: str | None
    confidence: str
    metadata_available: bool
    metadata: dict[str, str] = field(default_factory=dict)
    metadata_keys: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


def load_rows_safetensors(
    path: Path,
) -> Tuple[List[Tuple[str, str, str]], str, dict[str, str] | None]:
    """
    Returns: (rows, loader_name)
      rows = [(key, shape_str, dtype_str), ...]
    """
    rows: List[Tuple[str, str, str]] = []
    metadata: dict[str, str] | None = None
    with safe_open(str(path), framework="pt", device="cpu") as f:
        raw_metadata = f.metadata()
        if raw_metadata:
            metadata = dict(raw_metadata)
        for k in f.keys():
            t = f.get_tensor(k)
            shape_str = str(list(t.shape))
            dtype_str = str(t.dtype)
            rows.append((k, shape_str, dtype_str))
    return rows, "safetensors", metadata


def _extract_state_dict(obj) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        # Some SD checkpoints are already state_dict-like
        if any(hasattr(v, "shape") and hasattr(v, "dtype") for v in obj.values()):
            return obj
    raise ValueError(f"Unsupported checkpoint structure: {type(obj)}")


def load_rows_torch(path: Path) -> Tuple[List[Tuple[str, str, str]], str, None]:
    """
    Loads via torch.load (pickle risk!). Use only on trusted files.
    Returns: (rows, loader_name)
    """
    obj = torch.load(str(path), map_location="cpu")
    sd = _extract_state_dict(obj)

    rows: List[Tuple[str, str, str]] = []
    for k, v in sd.items():
        if hasattr(v, "shape") and hasattr(v, "dtype"):
            shape_str = str(list(v.shape))
            dtype_str = str(v.dtype)
        else:
            # Rare: non-tensor values inside dict
            shape_str = "NA"
            dtype_str = str(type(v))
        rows.append((k, shape_str, dtype_str))

    return rows, "torch.load", None


def load_param_rows(path: Path) -> Tuple[List[Tuple[str, str, str]], str, dict[str, str] | None]:
    # Try safetensors first (some files may be safetensors with odd extension)
    try:
        return load_rows_safetensors(path)
    except Exception:
        return load_rows_torch(path)


def _metadata_architecture(metadata: dict[str, str] | None) -> tuple[str | None, list[str]]:
    if not metadata:
        return None, []

    candidates = {
        key: value
        for key, value in metadata.items()
        if key
        in {
            "ss_base_model_version",
            "modelspec.architecture",
            "modelspec.sai_model_spec",
            "base_model",
            "trained_on",
        }
    }
    search_text = " ".join(str(value).lower() for value in candidates.values())
    checks = [
        ("sdxl", ("sdxl", "sd_xl", "stable-diffusion-xl", "sd xl")),
        ("sd15", ("sd_v1", "sd1", "sd 1", "stable-diffusion-v1", "v1-5", "1.5")),
        ("sd2", ("sd_v2", "sd2", "sd 2", "stable-diffusion-2", "768-v")),
        ("flux", ("flux",)),
        ("qwen-image", ("qwen",)),
        ("z-image", ("z-image", "z image")),
        ("ernie-image", ("ernie",)),
    ]
    for architecture, needles in checks:
        if any(needle in search_text for needle in needles):
            evidence = [f"{key}: {value}" for key, value in candidates.items()]
            return architecture, evidence
    return None, [f"{key}: {value}" for key, value in candidates.items()]


def infer_model_architecture(
    rows: Iterable[Tuple[str, str, str]],
    metadata: dict[str, str] | None,
) -> ModelArchitectureAnalysis:
    metadata_architecture, metadata_evidence = _metadata_architecture(metadata)
    if metadata_architecture:
        return ModelArchitectureAnalysis(
            architecture=metadata_architecture,
            confidence="high",
            metadata_available=bool(metadata),
            metadata=dict(sorted(metadata.items())) if metadata else {},
            metadata_keys=sorted(metadata.keys()) if metadata else [],
            evidence=metadata_evidence[:5],
        )

    keys = [key.lower() for key, _shape, _dtype in rows]
    evidence: list[str] = []

    if any("text_encoder_2" in key or "lora_te2" in key for key in keys):
        evidence.append("Found SDXL second text encoder LoRA keys.")
        return ModelArchitectureAnalysis(
            architecture="sdxl",
            confidence="medium",
            metadata_available=bool(metadata),
            metadata=dict(sorted(metadata.items())) if metadata else {},
            metadata_keys=sorted(metadata.keys()) if metadata else [],
            evidence=evidence,
        )

    if any("conditioner.embedders.1" in key for key in keys):
        evidence.append("Found SDXL conditioner second-embedder keys.")
        return ModelArchitectureAnalysis(
            architecture="sdxl",
            confidence="medium",
            metadata_available=bool(metadata),
            metadata=dict(sorted(metadata.items())) if metadata else {},
            metadata_keys=sorted(metadata.keys()) if metadata else [],
            evidence=evidence,
        )

    if any(key.startswith(("transformer.", "lora_transformer_", "diffusion_model.")) for key in keys):
        evidence.append("Found transformer-style LoRA keys.")
        return ModelArchitectureAnalysis(
            architecture="flux",
            confidence="low",
            metadata_available=bool(metadata),
            metadata=dict(sorted(metadata.items())) if metadata else {},
            metadata_keys=sorted(metadata.keys()) if metadata else [],
            evidence=evidence,
        )

    has_sd_unet = any("lora_unet" in key or ".unet." in key for key in keys)
    has_text_encoder = any("lora_te" in key or "text_encoder" in key for key in keys)
    if has_sd_unet and has_text_encoder:
        evidence.append("Found SD-style UNet and text encoder LoRA keys without SDXL second text encoder keys.")
        return ModelArchitectureAnalysis(
            architecture="sd15",
            confidence="low",
            metadata_available=bool(metadata),
            metadata=dict(sorted(metadata.items())) if metadata else {},
            metadata_keys=sorted(metadata.keys()) if metadata else [],
            evidence=evidence,
        )

    if metadata_evidence:
        evidence.extend(metadata_evidence[:5])
    if not metadata:
        evidence.append("Safetensors metadata is not present or not available.")
    evidence.append("No architecture-specific tensor key pattern matched.")
    return ModelArchitectureAnalysis(
        architecture=None,
        confidence="unknown",
        metadata_available=bool(metadata),
        metadata=dict(sorted(metadata.items())) if metadata else {},
        metadata_keys=sorted(metadata.keys()) if metadata else [],
        evidence=evidence,
    )


def analyze_model_file(
    path: Path,
    limit: int | None = None,
) -> Tuple[List[Tuple[str, str, str]], str, int, ModelArchitectureAnalysis]:
    rows, loader, metadata = load_param_rows(path)
    rows = sorted(rows, key=lambda r: r[0])
    total = len(rows)
    architecture = infer_model_architecture(rows, metadata)
    if limit and limit > 0:
        rows = rows[:limit]
    return rows, loader, total, architecture


def iter_model_files(path: Path, recursive: bool) -> List[Path]:
    if path.is_file():
        return [path]

    pattern = "**/*" if recursive else "*"
    files = [
        p
        for p in path.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]
    return sorted(files)


def write_tsv(
    model_path: Path,
    rows: Iterable[Tuple[str, str, str]],
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_path.stem}.params.tsv"
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("key\tshape\tdtype\n")
        for key, shape, dtype in sorted(rows, key=lambda r: r[0]):
            # ensure no tabs/newlines in key (shouldn't happen, but safe)
            key = key.replace("\t", " ").replace("\n", " ")
            f.write(f"{key}\t{shape}\t{dtype}\n")
    return out_path
