from pathlib import Path

import torch
from safetensors.torch import save_file

from backend.utilities.model_analysis import analyze_model_file, infer_model_architecture


def test_analyze_model_file_detects_sdxl_from_safetensors_metadata(tmp_path: Path):
    model_path = tmp_path / "style.safetensors"
    save_file(
        {
            "lora_unet_down_blocks_0_attentions_0_to_q.lora_down.weight": torch.zeros(4, 320),
            "lora_te2_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": torch.zeros(4, 1280),
        },
        str(model_path),
        metadata={"ss_base_model_version": "sdxl_base_v1-0"},
    )

    rows, loader, total, architecture = analyze_model_file(model_path)

    assert loader == "safetensors"
    assert total == 2
    assert len(rows) == 2
    assert architecture.architecture == "sdxl"
    assert architecture.confidence == "high"
    assert architecture.metadata_available is True
    assert architecture.metadata_keys == ["ss_base_model_version"]
    assert architecture.evidence == ["ss_base_model_version: sdxl_base_v1-0"]


def test_infer_model_architecture_reports_missing_metadata_when_unknown():
    architecture = infer_model_architecture(
        rows=[("misc.weight", "[1]", "torch.float32")],
        metadata=None,
    )

    assert architecture.architecture is None
    assert architecture.confidence == "unknown"
    assert architecture.metadata_available is False
    assert "Safetensors metadata is not present or not available." in architecture.evidence


def test_infer_model_architecture_uses_key_heuristics_without_metadata():
    architecture = infer_model_architecture(
        rows=[
            ("lora_unet_down_blocks_0_attentions_0_to_q.lora_down.weight", "[4, 320]", "torch.float32"),
            ("lora_te2_text_model_encoder_layers_0_mlp_fc1.lora_down.weight", "[4, 1280]", "torch.float32"),
        ],
        metadata=None,
    )

    assert architecture.architecture == "sdxl"
    assert architecture.confidence == "medium"
    assert architecture.metadata_available is False
