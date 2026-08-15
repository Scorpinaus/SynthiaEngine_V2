import json

from fastapi.testclient import TestClient

import backend.lora.registry as lora_registry
from backend.main import app


def _reset_lora_registry_paths(tmp_path):
    if hasattr(lora_registry, "_ENGINE"):
        lora_registry._ENGINE.dispose()

    lora_registry.REGISTRY_DB_PATH = tmp_path / "lora_registry.sqlite3"
    lora_registry.REGISTRY_DB_URL = f"sqlite:///{lora_registry.REGISTRY_DB_PATH.as_posix()}"
    lora_registry.REGISTRY_JSON_PATH = tmp_path / "lora_registry.json"
    lora_registry.REGISTRY_JSON_PATH.write_text(json.dumps([]), encoding="utf-8")

    lora_registry._ENGINE = lora_registry.create_engine(
        lora_registry.REGISTRY_DB_URL,
        future=True,
        pool_pre_ping=True,
        connect_args={"check_same_thread": False},
    )
    lora_registry._SessionLocal = lora_registry.sessionmaker(
        bind=lora_registry._ENGINE,
        autoflush=False,
        autocommit=False,
        future=True,
    )
    lora_registry.init_lora_registry_db()
    lora_registry._migrate_json_if_needed()
    lora_registry.LORA_REGISTRY = lora_registry.load_lora_registry()


def test_lora_list_and_create_contract_compatible(tmp_path):
    _reset_lora_registry_paths(tmp_path)
    client = TestClient(app)

    payload = {
        "lora_id": 101,
        "lora_model_family": "sd15",
        "lora_type": "lora",
        "lora_location": "local",
        "file_path": "C:/loras/a.safetensors",
        "name": "A",
    }
    create_response = client.post("/lora-models", json=payload)
    assert create_response.status_code == 200
    expected_response = {
        **payload,
        "prompt_presets": [],
        "runtime_profile": None,
        "weight_name": None,
        "subfolder": None,
        "revision": None,
    }
    assert create_response.json() == expected_response

    list_response = client.get("/lora-models")
    assert list_response.status_code == 200
    assert list_response.json() == [expected_response]

    filtered_response = client.get("/lora-models?family=sd15")
    assert filtered_response.status_code == 200
    assert filtered_response.json() == [expected_response]


def test_lora_detail_endpoint(tmp_path):
    _reset_lora_registry_paths(tmp_path)
    client = TestClient(app)
    client.post(
        "/lora-models",
        json={
            "lora_id": 102,
            "lora_model_family": "sdxl",
            "lora_type": "lycoris",
            "lora_location": "local",
            "file_path": "C:/loras/b.safetensors",
            "name": "B",
        },
    )

    found = client.get("/lora-models/102")
    assert found.status_code == 200
    assert found.json()["lora_id"] == 102

    missing = client.get("/lora-models/9999")
    assert missing.status_code == 404
    assert missing.json()["detail"] == "LoRA with id 9999 not found."


def test_lora_patch_endpoint(tmp_path):
    _reset_lora_registry_paths(tmp_path)
    client = TestClient(app)
    client.post(
        "/lora-models",
        json={
            "lora_id": 103,
            "lora_model_family": "sd15",
            "lora_type": "lora",
            "lora_location": "local",
            "file_path": "C:/loras/c.safetensors",
            "name": "C",
        },
    )

    updated = client.patch(
        "/lora-models/103",
        json={"file_path": "C:/loras/c-updated.safetensors", "name": None},
    )
    assert updated.status_code == 200
    assert updated.json()["lora_id"] == 103
    assert updated.json()["file_path"] == "C:/loras/c-updated.safetensors"
    assert updated.json()["name"] is None

    empty_payload = client.patch("/lora-models/103", json={})
    assert empty_payload.status_code == 400
    assert empty_payload.json()["detail"] == "At least one editable field must be provided."

    not_editable = client.patch("/lora-models/103", json={"lora_id": 200})
    assert not_editable.status_code == 422

    missing = client.patch("/lora-models/9999", json={"name": "X"})
    assert missing.status_code == 404
    assert missing.json()["detail"] == "LoRA with id 9999 not found."


def test_lora_delete_endpoint(tmp_path):
    _reset_lora_registry_paths(tmp_path)
    client = TestClient(app)
    client.post(
        "/lora-models",
        json={
            "lora_id": 104,
            "lora_model_family": "sd15",
            "lora_type": "lora",
            "lora_location": "local",
            "file_path": "C:/loras/d.safetensors",
            "name": "D",
        },
    )

    deleted = client.delete("/lora-models/104")
    assert deleted.status_code == 204
    assert deleted.text == ""

    after_delete = client.get("/lora-models/104")
    assert after_delete.status_code == 404

    missing = client.delete("/lora-models/104")
    assert missing.status_code == 404
    assert missing.json()["detail"] == "LoRA with id 104 not found."


def test_lora_create_rejects_name_with_dot(tmp_path):
    _reset_lora_registry_paths(tmp_path)
    client = TestClient(app)
    response = client.post(
        "/lora-models",
        json={
            "lora_id": 105,
            "lora_model_family": "sdxl",
            "lora_type": "lora",
            "lora_location": "local",
            "file_path": "C:/loras/e.safetensors",
            "name": "Pinpin Art Style V3.0",
        },
    )

    assert response.status_code == 422
    errors = response.json().get("detail", [])
    assert any("LoRA name cannot contain '.'" in str(item.get("msg", "")) for item in errors)


def test_lora_patch_rejects_name_with_dot(tmp_path):
    _reset_lora_registry_paths(tmp_path)
    client = TestClient(app)
    client.post(
        "/lora-models",
        json={
            "lora_id": 106,
            "lora_model_family": "sdxl",
            "lora_type": "lora",
            "lora_location": "local",
            "file_path": "C:/loras/f.safetensors",
            "name": "Pinpin Art Style V3",
        },
    )
    response = client.patch("/lora-models/106", json={"name": "Pinpin Art Style V3.0"})

    assert response.status_code == 422
    errors = response.json().get("detail", [])
    assert any("LoRA name cannot contain '.'" in str(item.get("msg", "")) for item in errors)


def test_lightning_metadata_create_get_and_patch_contract(tmp_path):
    _reset_lora_registry_paths(tmp_path)
    client = TestClient(app)
    profile = {
        "kind": "qwen_image_lightning",
        "base_variant": "qwen-image-2512",
        "steps": 4,
        "true_cfg_scale": 1.0,
        "scheduler_profile": "qwen_image_lightning_shift3",
        "adapter_strength": 1.0,
        "supported_tasks": ["text2img", "img2img", "inpaint"],
    }
    payload = {
        "lora_id": 107,
        "lora_model_family": "qwen-image",
        "lora_type": "lora",
        "lora_location": "hub",
        "file_path": "lightx2v/Qwen-Image-2512-Lightning",
        "runtime_profile": profile,
        "weight_name": "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
        "subfolder": "weights",
        "revision": "main",
    }
    created = client.post("/lora-models", json=payload)
    assert created.status_code == 200
    assert created.json()["runtime_profile"] == profile
    assert created.json()["weight_name"] == payload["weight_name"]

    found = client.get("/lora-models/107")
    assert found.status_code == 200
    assert found.json()["subfolder"] == "weights"

    updated_profile = {**profile, "steps": 8}
    updated = client.patch(
        "/lora-models/107",
        json={
            "runtime_profile": updated_profile,
            "weight_name": "Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors",
            "revision": "v1.0",
        },
    )
    assert updated.status_code == 200
    assert updated.json()["runtime_profile"] == updated_profile
    assert updated.json()["revision"] == "v1.0"


def test_lightning_hub_entries_require_weight_name_and_fixed_profile_values(tmp_path):
    _reset_lora_registry_paths(tmp_path)
    client = TestClient(app)
    profile = {
        "kind": "qwen_image_lightning",
        "base_variant": "qwen-image-2512",
        "steps": 4,
        "true_cfg_scale": 1.0,
        "scheduler_profile": "qwen_image_lightning_shift3",
        "adapter_strength": 1.0,
        "supported_tasks": ["text2img", "img2img", "inpaint"],
    }
    payload = {
        "lora_id": 108,
        "lora_model_family": "qwen-image",
        "lora_type": "lora",
        "lora_location": "hub",
        "file_path": "lightx2v/Qwen-Image-2512-Lightning",
        "runtime_profile": profile,
    }
    missing_weight_name = client.post("/lora-models", json=payload)
    assert missing_weight_name.status_code == 400
    assert "Hub Qwen Image Lightning entries require weight_name." in missing_weight_name.json()["detail"]

    invalid_profile = {**profile, "true_cfg_scale": 2.0}
    invalid_value = client.post(
        "/lora-models",
        json={**payload, "lora_location": "local", "runtime_profile": invalid_profile},
    )
    assert invalid_value.status_code == 422


def test_lightning_api_normalizes_legacy_text2img_supported_tasks(tmp_path):
    _reset_lora_registry_paths(tmp_path)
    client = TestClient(app)
    response = client.post(
        "/lora-models",
        json={
            "lora_id": 109,
            "lora_model_family": "qwen-image",
            "lora_type": "lora",
            "lora_location": "local",
            "file_path": "C:/loras/lightning.safetensors",
            "runtime_profile": {
                "kind": "qwen_image_lightning",
                "base_variant": "qwen-image-2512",
                "steps": 4,
                "true_cfg_scale": 1.0,
                "scheduler_profile": "qwen_image_lightning_shift3",
                "adapter_strength": 1.0,
                "supported_tasks": ["text2img"],
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["runtime_profile"]["supported_tasks"] == [
        "text2img", "img2img", "inpaint"
    ]
