import json

import pytest

import backend.lora.registry as lora_registry


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


def test_additive_migration_keeps_legacy_rows_without_runtime_metadata(tmp_path):
    _reset_lora_registry_paths(tmp_path, json_payload=None)
    lora_registry._ENGINE.dispose()
    lora_registry.REGISTRY_DB_PATH.unlink()
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
    with lora_registry._ENGINE.begin() as connection:
        connection.execute(
            lora_registry.text(
                "CREATE TABLE lora_registry ("
                "id INTEGER PRIMARY KEY, lora_id INTEGER NOT NULL UNIQUE, "
                "lora_model_family VARCHAR(64) NOT NULL, lora_type VARCHAR(64) NOT NULL, "
                "lora_location VARCHAR(64) NOT NULL, file_path TEXT NOT NULL, "
                "name VARCHAR(256), prompt_presets_json TEXT NOT NULL DEFAULT '[]'"
                ")"
            )
        )
        connection.execute(
            lora_registry.text(
                "INSERT INTO lora_registry "
                "(lora_id, lora_model_family, lora_type, lora_location, file_path, name, prompt_presets_json) "
                "VALUES (301, 'qwen-image', 'lora', 'local', 'C:/loras/legacy.safetensors', 'Legacy', '[]')"
            )
        )

    lora_registry.init_lora_registry_db()
    column_names = {
        column["name"]
        for column in lora_registry.inspect(lora_registry._ENGINE).get_columns("lora_registry")
    }
    assert {
        "runtime_profile_json",
        "compatibility_json",
        "weight_name",
        "subfolder",
        "revision",
    } <= column_names
    legacy_entry = lora_registry.get_lora_entry(301)
    assert legacy_entry.runtime_profile is None
    assert legacy_entry.compatibility is None
    assert legacy_entry.weight_name is None
    assert legacy_entry.subfolder is None
    assert legacy_entry.revision is None


def test_stored_legacy_lightning_profile_normalizes_supported_tasks_on_read(tmp_path):
    _reset_lora_registry_paths(tmp_path, json_payload=None)
    legacy_profile = {
        "kind": "qwen_image_lightning",
        "base_variant": "qwen-image-2512",
        "steps": 4,
        "true_cfg_scale": 1.0,
        "scheduler_profile": "qwen_image_lightning_shift3",
        "adapter_strength": 1.0,
        "supported_tasks": ["text2img"],
    }
    with lora_registry._SessionLocal.begin() as session:
        session.add(
            lora_registry.LoraRegistryRow(
                lora_id=304,
                lora_model_family="qwen-image",
                lora_type="lora",
                lora_location="local",
                file_path="C:/loras/legacy-lightning.safetensors",
                prompt_presets_json="[]",
                runtime_profile_json=json.dumps(legacy_profile),
            )
        )

    entry = lora_registry.get_lora_entry(304)

    assert entry.lora_type == "lora"
    assert entry.runtime_profile is not None
    assert entry.runtime_profile.kind == "qwen_image_lightning"
    assert entry.runtime_profile.supported_tasks == ["text2img", "img2img", "inpaint"]


def test_lightning_runtime_profile_round_trips_and_validates_entry_constraints(tmp_path):
    _reset_lora_registry_paths(tmp_path, json_payload=None)
    profile = {
        "kind": "qwen_image_lightning",
        "base_variant": "qwen-image-2512",
        "steps": 4,
        "true_cfg_scale": 1.0,
        "scheduler_profile": "qwen_image_lightning_shift3",
        "adapter_strength": 1.0,
        "supported_tasks": ["text2img", "img2img", "inpaint"],
    }
    created = lora_registry.add_lora(
        lora_registry.LoraRegistryEntry(
            lora_id=302,
            lora_model_family="qwen-image",
            lora_type="lora",
            lora_location="hub",
            file_path="lightx2v/Qwen-Image-2512-Lightning",
            weight_name="Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
            subfolder="weights",
            revision="main",
            runtime_profile=profile,
        )
    )
    assert created.runtime_profile is not None
    assert created.runtime_profile.model_dump() == profile
    assert lora_registry.get_lora_entry(302).model_dump() == created.model_dump()

    legacy_profile = lora_registry.LoraRuntimeProfile.model_validate(
        {**profile, "supported_tasks": ["text2img"]}
    )
    assert legacy_profile.supported_tasks == ["text2img", "img2img", "inpaint"]

    invalid_entry = {
        "lora_id": 303,
        "lora_model_family": "sdxl",
        "lora_type": "lora",
        "lora_location": "local",
        "file_path": "C:/loras/invalid.safetensors",
        "runtime_profile": profile,
    }
    with pytest.raises(ValueError, match="lora_model_family 'qwen-image'"):
        lora_registry.LoraRegistryEntry.model_validate(invalid_entry)

    invalid_entry["lora_model_family"] = "qwen-image"
    invalid_entry["lora_type"] = "lycoris"
    with pytest.raises(ValueError, match="lora_type 'lora'"):
        lora_registry.LoraRegistryEntry.model_validate(invalid_entry)

    invalid_entry["lora_type"] = "lora"
    invalid_entry["lora_location"] = "hub"
    with pytest.raises(ValueError, match="require weight_name"):
        lora_registry.LoraRegistryEntry.model_validate(invalid_entry)

    with pytest.raises(ValueError, match="steps"):
        lora_registry.LoraRuntimeProfile.model_validate({**profile, "steps": 6})
    with pytest.raises(ValueError, match="true_cfg_scale must be 1.0"):
        lora_registry.LoraRuntimeProfile.model_validate({**profile, "true_cfg_scale": 2.0})
    with pytest.raises(ValueError, match="supported_tasks"):
        lora_registry.LoraRuntimeProfile.model_validate({**profile, "supported_tasks": ["img2img"]})


def test_qwen_image_compatibility_round_trips_and_validates_entry_constraints(tmp_path):
    _reset_lora_registry_paths(tmp_path, json_payload=None)
    compatibility = {
        "base_variants": ["qwen-image-2512"],
        "runtime_profile_kinds": ["qwen_image_lightning"],
        "supported_tasks": ["text2img", "img2img", "inpaint"],
    }
    created = lora_registry.add_lora(
        lora_registry.LoraRegistryEntry(
            lora_id=305,
            lora_model_family="qwen-image",
            lora_type="lora",
            lora_location="local",
            file_path="C:/loras/companion.safetensors",
            compatibility=compatibility,
        )
    )
    assert created.compatibility is not None
    assert created.compatibility.model_dump() == compatibility
    assert lora_registry.get_lora_entry(305).model_dump() == created.model_dump()

    cleared = lora_registry.update_lora_entry(305, {"compatibility": None})
    assert cleared.compatibility is None
    restored = lora_registry.update_lora_entry(305, {"compatibility": compatibility})
    assert restored.compatibility is not None
    assert restored.compatibility.model_dump() == compatibility

    for field_name in compatibility:
        with pytest.raises(ValueError, match=f"{field_name} cannot be empty"):
            lora_registry.LoraCompatibility.model_validate({**compatibility, field_name: []})
        with pytest.raises(ValueError, match=f"{field_name} cannot contain duplicates"):
            lora_registry.LoraCompatibility.model_validate(
                {**compatibility, field_name: [compatibility[field_name][0], compatibility[field_name][0]]}
            )

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        lora_registry.LoraCompatibility.model_validate({**compatibility, "unknown": True})

    invalid_entry = {
        "lora_id": 306,
        "lora_model_family": "sdxl",
        "lora_type": "lora",
        "lora_location": "local",
        "file_path": "C:/loras/invalid.safetensors",
        "compatibility": compatibility,
    }
    with pytest.raises(ValueError, match="lora_model_family 'qwen-image'"):
        lora_registry.LoraRegistryEntry.model_validate(invalid_entry)

    invalid_entry["lora_model_family"] = "qwen-image"
    invalid_entry["lora_type"] = "lycoris"
    with pytest.raises(ValueError, match="lora_type 'lora'"):
        lora_registry.LoraRegistryEntry.model_validate(invalid_entry)

    invalid_entry["lora_type"] = "lora"
    invalid_entry["runtime_profile"] = {
        "kind": "qwen_image_lightning",
        "base_variant": "qwen-image-2512",
        "steps": 4,
        "true_cfg_scale": 1.0,
        "scheduler_profile": "qwen_image_lightning_shift3",
        "adapter_strength": 1.0,
        "supported_tasks": ["text2img", "img2img", "inpaint"],
    }
    with pytest.raises(ValueError, match="not allowed on Lightning runtime-profile entries"):
        lora_registry.LoraRegistryEntry.model_validate(invalid_entry)
