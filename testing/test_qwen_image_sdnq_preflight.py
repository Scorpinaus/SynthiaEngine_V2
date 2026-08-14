from __future__ import annotations

from importlib.metadata import version as package_version
import json
from pathlib import Path
from typing import Any

import pytest

from backend.qwen_image import pipeline as qwen_image_pipeline


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "backend" / "registries" / "model_registry.json"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _qwen_registry_entry() -> dict[str, Any]:
    entries = _read_json(REGISTRY_PATH)
    return next(
        entry
        for entry in entries
        if entry.get("name") == qwen_image_pipeline._DEFAULT_MODEL_NAME
    )


def _local_checkpoint_root() -> Path:
    checkpoint_root = Path(_qwen_registry_entry()["link"])
    if not checkpoint_root.is_dir():
        pytest.skip(f"Local Qwen-Image SDNQ checkpoint is not present: {checkpoint_root}")
    return checkpoint_root


def test_sdnq_dependency_files_and_installed_distribution_match():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    constraints = (ROOT / "constraints.txt").read_text(encoding="utf-8").splitlines()

    assert "sdnq>=0.2.2" in requirements
    assert "sdnq==0.2.2" in constraints
    assert package_version("sdnq") == "0.2.2"
    assert qwen_image_pipeline._register_sdnq() == "0.2.2"


@pytest.mark.integration
def test_local_qwen_image_checkpoint_has_required_sdnq_metadata_and_shards():
    checkpoint_root = _local_checkpoint_root()
    entry = _qwen_registry_entry()

    assert entry["family"] == "qwen-image"
    assert entry["model_type"] == "diffusers"
    assert entry["location_type"] == "local"
    assert entry["version"] == qwen_image_pipeline._DEFAULT_MODEL_VERSION

    model_index = _read_json(checkpoint_root / "model_index.json")
    assert model_index["_class_name"] == "QwenImagePipeline"
    assert model_index["scheduler"] == [
        "diffusers",
        "FlowMatchEulerDiscreteScheduler",
    ]
    assert model_index["transformer"] == [
        "diffusers",
        "QwenImageTransformer2DModel",
    ]
    assert model_index["text_encoder"] == [
        "transformers",
        "Qwen2_5_VLForConditionalGeneration",
    ]

    scheduler_config = _read_json(checkpoint_root / "scheduler" / "scheduler_config.json")
    assert scheduler_config["_class_name"] == "FlowMatchEulerDiscreteScheduler"

    checkpoint_versions: set[str] = set()
    component_indexes = {
        "transformer": "diffusion_pytorch_model.safetensors.index.json",
        "text_encoder": "model.safetensors.index.json",
    }
    for component_name, index_name in component_indexes.items():
        component_root = checkpoint_root / component_name
        component_config = _read_json(component_root / "config.json")
        quantization_config = component_config["quantization_config"]

        assert quantization_config["quant_method"] == "sdnq"
        assert quantization_config["use_dynamic_quantization"] is True
        assert quantization_config["weights_dtype"] == "int4"
        checkpoint_versions.add(quantization_config["sdnq_version"])

        weight_index = _read_json(component_root / index_name)
        shard_names = set(weight_index["weight_map"].values())
        assert shard_names
        for shard_name in shard_names:
            shard_path = component_root / shard_name
            assert shard_path.is_file(), f"Missing {component_name} shard: {shard_name}"
            assert shard_path.stat().st_size > 0

    assert checkpoint_versions == {"0.1.4"}
