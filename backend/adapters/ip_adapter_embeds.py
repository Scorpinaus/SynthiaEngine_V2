from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import torch

from backend.artifacts import artifact_path_for_id, validate_artifact_id
from backend.config import OUTPUT_DIR

IP_ADAPTER_EMBEDS_FORMAT = "synthengine.sdxl.ip_adapter_image_embeds.v1"


def save_ip_adapter_embeds_artifact(
    embeds: list[torch.Tensor],
    *,
    metadata: dict[str, Any],
    family: str = "SDXL",
) -> dict[str, str]:
    artifact_id = f"e{uuid.uuid4().hex}"
    path = artifact_path_for_id(artifact_id, output_dir=OUTPUT_DIR)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": IP_ADAPTER_EMBEDS_FORMAT,
        "family": family,
        "embeds": [embed.detach().cpu() for embed in embeds],
        "metadata": metadata,
    }
    torch.save(payload, path)
    rel = path.relative_to(OUTPUT_DIR.resolve()).as_posix()
    return {"artifact_id": artifact_id, "path": rel, "url": f"/outputs/{rel}"}


def _artifact_id_from_ref(ref: Any) -> str:
    if isinstance(ref, dict) and "artifact_id" in ref:
        artifact_id = str(ref["artifact_id"])
    elif isinstance(ref, str) and ref.startswith("@artifact:"):
        artifact_id = ref.removeprefix("@artifact:").strip()
    elif isinstance(ref, str) and ref.startswith("/outputs/"):
        name = Path(ref.removeprefix("/outputs/")).name
        artifact_id = name.removesuffix(".pt")
    elif isinstance(ref, str):
        artifact_id = ref.strip()
    else:
        raise ValueError("Unsupported IP-Adapter embeds reference.")

    artifact_id = validate_artifact_id(artifact_id)
    if not artifact_id.startswith("e"):
        raise ValueError("IP-Adapter embeds reference must use an embed artifact_id.")
    return artifact_id


def load_ip_adapter_embeds_artifact(ref: Any) -> dict[str, Any]:
    path = artifact_path_for_id(_artifact_id_from_ref(ref), output_dir=OUTPUT_DIR)
    if not path.exists():
        raise ValueError("IP-Adapter embeds artifact was not found.")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("IP-Adapter embeds artifact must contain an object.")
    if payload.get("format") != IP_ADAPTER_EMBEDS_FORMAT:
        raise ValueError("Unsupported IP-Adapter embeds artifact format.")
    if payload.get("family") not in {"SD15", "SDXL"}:
        raise ValueError("IP-Adapter embeds artifact family must be SD15 or SDXL.")
    embeds = payload.get("embeds")
    if not isinstance(embeds, list) or not embeds:
        raise ValueError("IP-Adapter embeds artifact must contain a non-empty embeds list.")
    if not all(isinstance(embed, torch.Tensor) for embed in embeds):
        raise ValueError("IP-Adapter embeds artifact embeds must be tensors.")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("IP-Adapter embeds artifact metadata must be an object.")
    return payload


def validate_ip_adapter_embeds_metadata(
    payload: dict[str, Any],
    *,
    expected_model: str,
    expected_subfolder: str,
    expected_weight_name: str,
    do_classifier_free_guidance: bool,
    expected_family: str = "SDXL",
) -> None:
    actual_family = str(payload.get("family") or "")
    if actual_family != expected_family:
        raise ValueError(
            f"IP-Adapter embeds artifact family must be {expected_family}."
        )

    metadata = payload["metadata"]
    adapters = metadata.get("adapters")
    if not isinstance(adapters, list) or len(adapters) != 1:
        raise ValueError("Exactly one SDXL IP-Adapter embeds adapter is supported.")

    adapter = adapters[0]
    if not isinstance(adapter, dict):
        raise ValueError("IP-Adapter embeds adapter metadata must be an object.")

    checks = {
        "model": expected_model,
        "subfolder": expected_subfolder,
        "weight_name": expected_weight_name,
    }
    for key, expected in checks.items():
        actual = str(adapter.get(key) or "")
        if actual != expected:
            raise ValueError(
                f"IP-Adapter embeds metadata mismatch for {key}: expected {expected!r}, got {actual!r}."
            )

    actual_cfg = bool(metadata.get("do_classifier_free_guidance"))
    if actual_cfg != do_classifier_free_guidance:
        raise ValueError(
            "IP-Adapter embeds classifier-free guidance setting does not match render guidance scale."
        )

    if len(payload["embeds"]) != len(adapters):
        raise ValueError("IP-Adapter embeds count must match adapter metadata count.")
