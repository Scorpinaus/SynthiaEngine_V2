from fastapi.testclient import TestClient

import backend.registries.preset as preset_registry
from backend.main import app


def _reset_preset_registry_paths(tmp_path):
    if hasattr(preset_registry, "_ENGINE"):
        preset_registry._ENGINE.dispose()

    preset_registry.REGISTRY_DB_PATH = tmp_path / "preset_registry.sqlite3"
    preset_registry.REGISTRY_DB_URL = f"sqlite:///{preset_registry.REGISTRY_DB_PATH.as_posix()}"
    preset_registry._ENGINE = preset_registry.create_engine(
        preset_registry.REGISTRY_DB_URL,
        future=True,
        pool_pre_ping=True,
        connect_args={"check_same_thread": False},
    )
    preset_registry._SessionLocal = preset_registry.sessionmaker(
        bind=preset_registry._ENGINE,
        autoflush=False,
        autocommit=False,
        future=True,
    )
    preset_registry.init_preset_registry_db()


def test_create_list_get_patch_delete_preset(tmp_path):
    _reset_preset_registry_paths(tmp_path)
    client = TestClient(app)

    create_payload = {
        "name": "sd15 baseline",
        "family": "sd15",
        "task_type": "sd15.text2img",
        "settings": {
            "prompt": "a product shot",
            "negative_prompt": "low quality",
            "steps": 30,
            "cfg": 7.0,
            "scheduler": "euler",
            "seed": 1234,
            "width": 640,
            "height": 640,
            "num_images": 2,
            "clip_skip": 1,
            "weighting_policy": "diffusers-like",
            "hires_enabled": True,
            "hires_scale": 1.5,
            "controlnet_enabled": False,
            "controlnet_conditioning_scale": 1.0,
            "control_guidance_start": 0.0,
            "control_guidance_end": 1.0,
            "controlnet_guess_mode": False,
            "controlnet_compat_mode": "warn",
            "model": "stable-diffusion-v1-5",
            "lora_adapters": [{"lora_id": 101, "strength": 0.8}],
        },
    }

    created = client.post("/api/presets", json=create_payload)
    assert created.status_code == 201
    created_json = created.json()
    assert created_json["preset_id"] > 0
    assert created_json["name"] == "sd15 baseline"
    assert created_json["family"] == "sd15"
    assert created_json["task_type"] == "sd15.text2img"
    assert created_json["settings"]["steps"] == 30

    preset_id = created_json["preset_id"]

    listed = client.get("/api/presets")
    assert listed.status_code == 200
    assert len(listed.json()) == 1

    filtered = client.get("/api/presets?family=sd15&task_type=sd15.text2img")
    assert filtered.status_code == 200
    assert len(filtered.json()) == 1
    assert filtered.json()[0]["preset_id"] == preset_id

    found = client.get(f"/api/presets/{preset_id}")
    assert found.status_code == 200
    assert found.json()["name"] == "sd15 baseline"

    patched = client.patch(
        f"/api/presets/{preset_id}",
        json={
            "name": "sd15 baseline v2",
            "settings": {
                "prompt": "a product shot",
                "negative_prompt": "low quality",
                "steps": 25,
                "cfg": 6.5,
                "scheduler": "euler",
                "seed": 5678,
                "width": 640,
                "height": 640,
                "num_images": 1,
                "clip_skip": 1,
                "weighting_policy": "diffusers-like",
                "hires_enabled": False,
                "hires_scale": 1.2,
                "controlnet_enabled": True,
                "controlnet_conditioning_scale": 0.9,
                "control_guidance_start": 0.05,
                "control_guidance_end": 0.9,
                "controlnet_guess_mode": False,
                "controlnet_compat_mode": "warn",
                "model": "stable-diffusion-v1-5",
                "lora_adapters": [{"lora_id": 101, "strength": 0.7}],
            },
        },
    )
    assert patched.status_code == 200
    assert patched.json()["name"] == "sd15 baseline v2"
    assert patched.json()["settings"]["steps"] == 25
    assert patched.json()["settings"]["controlnet_enabled"] is True

    deleted = client.delete(f"/api/presets/{preset_id}")
    assert deleted.status_code == 204

    missing_get = client.get(f"/api/presets/{preset_id}")
    missing_patch = client.patch(f"/api/presets/{preset_id}", json={"name": "x"})
    missing_delete = client.delete(f"/api/presets/{preset_id}")
    assert missing_get.status_code == 404
    assert missing_patch.status_code == 404
    assert missing_delete.status_code == 404
    assert missing_get.json()["detail"] == f"Preset with id {preset_id} not found."


def test_preset_validation_errors(tmp_path):
    _reset_preset_registry_paths(tmp_path)
    client = TestClient(app)

    unsupported_task = client.post(
        "/api/presets",
        json={
            "name": "bad task",
            "family": "sd15",
            "task_type": "sd15.unknown",
            "settings": {"steps": 20},
        },
    )
    assert unsupported_task.status_code == 400
    assert unsupported_task.json()["detail"] == "Unsupported task_type 'sd15.unknown'."

    mismatched_family = client.post(
        "/api/presets",
        json={
            "name": "bad family",
            "family": "sdxl",
            "task_type": "sd15.text2img",
            "settings": {"steps": 20},
        },
    )
    assert mismatched_family.status_code == 400
    assert mismatched_family.json()["detail"] == "Field 'family' must match task_type family 'sd15'."

    unknown_settings_field = client.post(
        "/api/presets",
        json={
            "name": "bad field",
            "family": "sd15",
            "task_type": "sd15.text2img",
            "settings": {"steps": 20, "unknown_field": 1},
        },
    )
    assert unknown_settings_field.status_code == 400
    assert (
        unknown_settings_field.json()["detail"]
        == "Unknown preset setting fields for task_type 'sd15.text2img': unknown_field."
    )

    invalid_filter = client.get("/api/presets?task_type=sd15.unknown")
    assert invalid_filter.status_code == 400
    assert invalid_filter.json()["detail"] == "Unsupported task_type 'sd15.unknown'."


def test_sd15_controlnet_preset_accepts_new_contract_fields(tmp_path):
    _reset_preset_registry_paths(tmp_path)
    client = TestClient(app)

    created = client.post(
        "/api/presets",
        json={
            "name": "sd15 contract fields",
            "family": "sd15",
            "task_type": "sd15.controlnet.text2img",
            "settings": {
                "prompt": "portrait",
                "controlNetEnabled": True,
                "effectiveItems": [
                    {
                        "control_image": "@artifact:a0123456789abcdef0123456789abcdef",
                        "model_id": "lllyasviel/control_v11p_sd15_canny",
                        "conditioning_scale": 0.9,
                        "preprocessor_id": "canny",
                    }
                ],
                "Lora": {
                    "loraStatus": True,
                    "adapters": [{"lora_id": 101, "strength": 0.7}],
                },
                "hires": {
                    "hiresEnabled": True,
                    "hires_scale": 1.4,
                },
            },
        },
    )

    assert created.status_code == 201
    body = created.json()
    assert body["task_type"] == "sd15.controlnet.text2img"
    assert body["settings"]["controlNetEnabled"] is True
    assert body["settings"]["effectiveItems"][0]["model_id"] == "lllyasviel/control_v11p_sd15_canny"
    assert body["settings"]["Lora"]["loraStatus"] is True
    assert body["settings"]["hires"]["hiresEnabled"] is True


def test_sdxl_preset_allows_frontend_extra_fields(tmp_path):
    _reset_preset_registry_paths(tmp_path)
    client = TestClient(app)

    created = client.post(
        "/api/presets",
        json={
            "name": "sdxl baseline",
            "family": "sdxl",
            "task_type": "sdxl.text2img",
            "settings": {
                "prompt": "studio portrait",
                "negative_prompt": "blurry",
                "steps": 30,
                "cfg": 6.5,
                "guidance_scale": 6.5,
                "scheduler": "euler",
                "seed": 42,
                "width": 1024,
                "height": 1024,
                "num_images": 1,
                "clip_skip": 1,
                "hires_enabled": True,
                "hires_scale": 1.5,
                "controlnet_enabled": False,
                "controlnet_conditioning_scale": 1.0,
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0,
                "controlnet_guess_mode": False,
                "controlnet_compat_mode": "warn",
                "model": "stable-diffusion-xl-base-1.0",
                "lora_adapters": [{"lora_id": 101, "strength": 0.8}],
            },
        },
    )
    assert created.status_code == 201
    body = created.json()
    assert body["family"] == "sdxl"
    assert body["task_type"] == "sdxl.text2img"
    assert body["settings"]["cfg"] == 6.5
    assert body["settings"]["hires_enabled"] is True
