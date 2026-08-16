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


def test_create_and_list_and_filter_loras(tmp_path):
    _reset_lora_registry_paths(tmp_path)
    client = TestClient(app)

    payload_a = {
        "lora_id": 201,
        "lora_model_family": "sd15",
        "lora_type": "lora",
        "lora_location": "local",
        "file_path": "C:/loras/a.safetensors",
        "name": "A",
    }
    payload_b = {
        "lora_id": 202,
        "lora_model_family": "sdxl",
        "lora_type": "lycoris",
        "lora_location": "hub",
        "file_path": "org/repo-lora",
        "name": "B",
    }

    created_a = client.post("/lora-models", json=payload_a)
    created_b = client.post("/lora-models", json=payload_b)
    assert created_a.status_code == 200
    assert created_b.status_code == 200

    listed = client.get("/lora-models")
    assert listed.status_code == 200
    assert len(listed.json()) == 2

    filtered = client.get("/lora-models?family=sd15")
    assert filtered.status_code == 200
    assert filtered.json() == [
        {
            **payload_a,
            "prompt_presets": [],
            "runtime_profile": None,
            "compatibility": None,
            "weight_name": None,
            "subfolder": None,
            "revision": None,
        }
    ]


def test_lora_prompt_presets_round_trip_and_update(tmp_path):
    _reset_lora_registry_paths(tmp_path)
    client = TestClient(app)
    payload = {
        "lora_id": 205,
        "lora_model_family": "sd15",
        "lora_type": "lora",
        "lora_location": "local",
        "file_path": "C:/loras/preset.safetensors",
        "name": "Preset LoRA",
        "prompt_presets": [
            {"name": "Soft watercolor", "words": ["soft watercolor", "paper texture"]},
            {"name": "Ink detail", "words": ["fine ink lines"]},
        ],
    }

    created = client.post("/lora-models", json=payload)
    assert created.status_code == 200
    assert created.json()["prompt_presets"] == payload["prompt_presets"]

    updated = client.patch(
        "/lora-models/205",
        json={"prompt_presets": [{"name": "Portrait", "words": ["portrait lighting", "sharp eyes"]}]},
    )
    assert updated.status_code == 200
    assert updated.json()["prompt_presets"] == [
        {"name": "Portrait", "words": ["portrait lighting", "sharp eyes"]}
    ]

    found = client.get("/lora-models/205")
    assert found.status_code == 200
    assert found.json()["prompt_presets"] == updated.json()["prompt_presets"]


def test_lora_prompt_preset_requires_words(tmp_path):
    _reset_lora_registry_paths(tmp_path)
    client = TestClient(app)
    payload = {
        "lora_id": 206,
        "lora_model_family": "sd15",
        "lora_type": "lora",
        "lora_location": "local",
        "file_path": "C:/loras/invalid.safetensors",
        "name": "Invalid Preset LoRA",
        "prompt_presets": [{"name": "Empty", "words": []}],
    }

    created = client.post("/lora-models", json=payload)
    assert created.status_code == 422


def test_duplicate_create_returns_domain_error(tmp_path):
    _reset_lora_registry_paths(tmp_path)
    client = TestClient(app)
    payload = {
        "lora_id": 203,
        "lora_model_family": "sd15",
        "lora_type": "lora",
        "lora_location": "local",
        "file_path": "C:/loras/c.safetensors",
        "name": "C",
    }

    first = client.post("/lora-models", json=payload)
    duplicate = client.post("/lora-models", json=payload)
    assert first.status_code == 200
    assert duplicate.status_code == 400
    assert duplicate.json()["detail"] == "LoRA with id 203 already exists."


def test_get_update_delete_and_not_found_cases(tmp_path):
    _reset_lora_registry_paths(tmp_path)
    client = TestClient(app)
    payload = {
        "lora_id": 204,
        "lora_model_family": "sd15",
        "lora_type": "lora",
        "lora_location": "local",
        "file_path": "C:/loras/d.safetensors",
        "name": "D",
    }
    client.post("/lora-models", json=payload)

    found = client.get("/lora-models/204")
    assert found.status_code == 200
    assert found.json()["lora_id"] == 204

    updated = client.patch(
        "/lora-models/204",
        json={"name": "D-Updated", "file_path": "C:/loras/d-updated.safetensors"},
    )
    assert updated.status_code == 200
    assert updated.json()["name"] == "D-Updated"
    assert updated.json()["file_path"] == "C:/loras/d-updated.safetensors"

    deleted = client.delete("/lora-models/204")
    assert deleted.status_code == 204

    missing_get = client.get("/lora-models/204")
    missing_patch = client.patch("/lora-models/204", json={"name": "X"})
    missing_delete = client.delete("/lora-models/204")
    assert missing_get.status_code == 404
    assert missing_patch.status_code == 404
    assert missing_delete.status_code == 404
    assert missing_get.json()["detail"] == "LoRA with id 204 not found."
    assert missing_patch.json()["detail"] == "LoRA with id 204 not found."
    assert missing_delete.json()["detail"] == "LoRA with id 204 not found."
