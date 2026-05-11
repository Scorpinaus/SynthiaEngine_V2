import json

from fastapi.testclient import TestClient

import backend.registries.model as model_registry
from backend.main import app


def _reset_model_registry_paths(tmp_path):
    if hasattr(model_registry, "_ENGINE"):
        model_registry._ENGINE.dispose()

    model_registry.REGISTRY_DB_PATH = tmp_path / "model_registry.sqlite3"
    model_registry.REGISTRY_DB_URL = f"sqlite:///{model_registry.REGISTRY_DB_PATH.as_posix()}"
    model_registry.REGISTRY_JSON_PATH = tmp_path / "model_registry.json"
    model_registry.REGISTRY_JSON_PATH.write_text(json.dumps([]), encoding="utf-8")

    model_registry._ENGINE = model_registry.create_engine(
        model_registry.REGISTRY_DB_URL,
        future=True,
        pool_pre_ping=True,
        connect_args={"check_same_thread": False},
    )
    model_registry._SessionLocal = model_registry.sessionmaker(
        bind=model_registry._ENGINE,
        autoflush=False,
        autocommit=False,
        future=True,
    )
    model_registry.init_model_registry_db()
    model_registry._migrate_json_if_needed()


def test_model_create_list_get_patch_delete_and_not_found(tmp_path):
    _reset_model_registry_paths(tmp_path)
    client = TestClient(app)

    payload = {
        "name": "Base A",
        "family": "sd15",
        "model_type": "diffusers",
        "location_type": "local",
        "model_id": 11,
        "version": "v1",
        "link": "C:/models/base_a",
    }
    created = client.post("/models", json=payload)
    assert created.status_code == 201
    assert created.json() == payload

    listed = client.get("/models")
    assert listed.status_code == 200
    assert len(listed.json()) == 1

    found = client.get("/models/Base%20A")
    assert found.status_code == 200
    assert found.json()["name"] == "Base A"

    updated = client.patch(
        "/models/Base%20A",
        json={"version": "v2", "link": "C:/models/base_a_v2", "model_id": 12},
    )
    assert updated.status_code == 200
    assert updated.json()["version"] == "v2"
    assert updated.json()["model_id"] == 12

    deleted = client.delete("/models/Base%20A")
    assert deleted.status_code == 204

    missing_get = client.get("/models/Base%20A")
    missing_patch = client.patch("/models/Base%20A", json={"version": "v3"})
    missing_delete = client.delete("/models/Base%20A")
    assert missing_get.status_code == 404
    assert missing_patch.status_code == 404
    assert missing_delete.status_code == 404
    assert missing_get.json()["detail"] == "Model 'Base A' not found."


def test_model_patch_validation_and_duplicate_create(tmp_path):
    _reset_model_registry_paths(tmp_path)
    client = TestClient(app)

    payload = {
        "name": "Base B",
        "family": "sdxl",
        "model_type": "single_file",
        "location_type": "hub",
        "model_id": 21,
        "version": "v1",
        "link": "org/base_b",
    }
    created = client.post("/models", json=payload)
    duplicate = client.post("/models", json=payload)
    assert created.status_code == 201
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == "Model name already exists."

    empty_patch = client.patch("/models/Base%20B", json={})
    assert empty_patch.status_code == 400
    assert empty_patch.json()["detail"] == "At least one editable field must be provided."

    invalid_field_patch = client.patch("/models/Base%20B", json={"name": "Changed"})
    assert invalid_field_patch.status_code == 422


def test_local_path_select_returns_picker_path(monkeypatch):
    client = TestClient(app)

    def fake_picker(selection_type):
        assert selection_type == "folder"
        return r"D:\diffusion\models\base"

    monkeypatch.setattr("backend.main._open_local_path_dialog", fake_picker)

    selected = client.post("/api/local-path/select", json={"selection_type": "folder"})

    assert selected.status_code == 200
    assert selected.json() == {"path": r"D:\diffusion\models\base"}


def test_local_path_select_rejects_unknown_selection_type():
    client = TestClient(app)

    selected = client.post("/api/local-path/select", json={"selection_type": "drive"})

    assert selected.status_code == 422
