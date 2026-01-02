from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from safetensors import safe_open

SUPPORTED_EXTS = {".safetensors", ".pt", ".ckpt", ".ckpr", ".model"}


def load_rows_safetensors(path: Path) -> Tuple[List[Tuple[str, str, str]], str]:
    """
    Returns: (rows, loader_name)
      rows = [(key, shape_str, dtype_str), ...]
    """
    rows: List[Tuple[str, str, str]] = []
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            shape_str = str(list(t.shape))
            dtype_str = str(t.dtype)
            rows.append((k, shape_str, dtype_str))
    return rows, "safetensors"


def _extract_state_dict(obj) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        # Some SD checkpoints are already state_dict-like
        if any(hasattr(v, "shape") and hasattr(v, "dtype") for v in obj.values()):
            return obj
    raise ValueError(f"Unsupported checkpoint structure: {type(obj)}")


def load_rows_torch(path: Path) -> Tuple[List[Tuple[str, str, str]], str]:
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

    return rows, "torch.load"


def load_param_rows(path: Path) -> Tuple[List[Tuple[str, str, str]], str]:
    # Try safetensors first (some files may be safetensors with odd extension)
    try:
        return load_rows_safetensors(path)
    except Exception:
        return load_rows_torch(path)


def analyze_model_file(
    path: Path,
    limit: int | None = None,
) -> Tuple[List[Tuple[str, str, str]], str, int]:
    rows, loader = load_param_rows(path)
    rows = sorted(rows, key=lambda r: r[0])
    total = len(rows)
    if limit and limit > 0:
        rows = rows[:limit]
    return rows, loader, total


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
