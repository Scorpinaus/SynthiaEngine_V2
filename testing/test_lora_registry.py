import json

import pytest

import backend.lora_registry as lora_registry


def _reset_lora_registry_paths(tmp_path, json_payload=None):
    if hasattr(lora_registry, "_ENGINE"):
        lora_registry._ENGINE.dispose()

    lora_registry.REGISTRY_DB_PATH = tmp_path / "lora_registry.sqlite3"
    lora_registry.REGISTRY_DB_URL = f"sqlite:///{lora_registry.REGISTRY_DB_PATH.as_posix()}"
    lora_registry.REGISTRY_JSON_PATH = tmp_path / "lora_registry.json"

    if json_payload is None:
        lora_registry.REGISTRY_JSON_PATH.unlink(missing_ok=True)
    else:
        lora_registry.REGISTRY_JSON_PATH.write_text(json.dumps(json_payload), encoding="utf-8")

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


def test_lora_registry_persists_and_reads_from_sqlite(tmp_path):
    _reset_lora_registry_paths(tmp_path, json_payload=None)
    assert lora_registry.REGISTRY_DB_PATH.exists()

    created = lora_registry.add_lora(
        lora_registry.LoraRegistryEntry(
            lora_id=100,
            lora_model_family="sd15",
            lora_type="lora",
            lora_location="local",
            file_path="C:/loras/a.safetensors",
            name="A",
        )
    )
    assert created.lora_id == 100

    rows = lora_registry.load_lora_registry()
    assert len(rows) == 1
    assert rows[0].lora_id == 100
    assert rows[0].name == "A"
    assert lora_registry.get_lora_entry(100).file_path == "C:/loras/a.safetensors"


def test_json_migrates_once_and_skips_invalid_rows(tmp_path, caplog):
    json_payload = [
        {
            "lora_id": 1,
            "lora_model_family": "sd15",
            "lora_type": "lora",
            "lora_location": "local",
            "file_path": "C:/loras/one.safetensors",
            "name": "One",
        },
        {"lora_id": 2, "lora_type": "lora"},
        {
            "lora_id": 1,
            "lora_model_family": "sd15",
            "lora_type": "lora",
            "lora_location": "local",
            "file_path": "C:/loras/dupe.safetensors",
            "name": "Dupe",
        },
        "not-a-dict",
        {
            "lora_id": 3,
            "lora_model_family": "sdxl",
            "lora_type": "lycoris",
            "lora_location": "local",
            "file_path": "C:/loras/three.safetensors",
            "name": "Three",
        },
    ]

    with caplog.at_level("WARNING"):
        _reset_lora_registry_paths(tmp_path, json_payload=json_payload)

    migrated = lora_registry.load_lora_registry()
    assert [item.lora_id for item in migrated] == [1, 3]
    assert "Skipping invalid LoRA registry entry at index 1" in caplog.text
    assert "Skipping invalid LoRA registry entry at index 2" in caplog.text
    assert "Skipping invalid LoRA registry entry at index 3" in caplog.text

    lora_registry.REGISTRY_JSON_PATH.write_text(
        json.dumps(
            [
                {
                    "lora_id": 99,
                    "lora_model_family": "sd15",
                    "lora_type": "lora",
                    "lora_location": "local",
                    "file_path": "C:/loras/new.safetensors",
                    "name": "New",
                }
            ]
        ),
        encoding="utf-8",
    )
    lora_registry._migrate_json_if_needed()
    migrated_again = lora_registry.load_lora_registry()
    assert [item.lora_id for item in migrated_again] == [1, 3]


def test_domain_errors_are_stable(tmp_path):
    _reset_lora_registry_paths(tmp_path, json_payload=None)

    lora_registry.add_lora(
        lora_registry.LoraRegistryEntry(
            lora_id=10,
            lora_model_family="sd15",
            lora_type="lora",
            lora_location="local",
            file_path="C:/loras/base.safetensors",
            name="Base",
        )
    )

    with pytest.raises(ValueError, match=r"^LoRA with id 10 already exists\.$"):
        lora_registry.add_lora(
            lora_registry.LoraRegistryEntry(
                lora_id=10,
                lora_model_family="sd15",
                lora_type="lora",
                lora_location="local",
                file_path="C:/loras/dup.safetensors",
                name="Dup",
            )
        )

    with pytest.raises(ValueError, match=r"^LoRA with id 404 not found\.$"):
        lora_registry.get_lora_entry(404)
