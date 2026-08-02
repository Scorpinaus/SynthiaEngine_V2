"""Focused ARC-02 coverage for settings, assembly, and extracted API domains."""

from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image
from starlette.requests import Request

import backend.api.local_paths as local_paths
from backend.main import app, create_app
from backend.settings import REPOSITORY_ROOT, load_settings


def _settings_for(tmp_path, **environment):
    values = {
        "SYNTHA_OUTPUT_DIR": str(tmp_path / "outputs"),
        "SYNTHA_DATABASE_DIR": str(tmp_path / "database"),
        "SYNTHA_API_START_WORKER": "0",
        **environment,
    }
    return load_settings(values, repository_root=tmp_path)


def _png_bytes(size: tuple[int, int] = (2, 2)) -> bytes:
    payload = BytesIO()
    Image.new("RGB", size, color=(12, 34, 56)).save(payload, format="PNG")
    return payload.getvalue()


def test_settings_resolve_repository_paths_without_filesystem_side_effects(tmp_path):
    repository_root = tmp_path / "checkout"
    settings = load_settings(
        {
            "SYNTHA_OUTPUT_DIR": "var/generated",
            "SYNTHA_DATABASE_DIR": "var/state",
        },
        repository_root=repository_root,
    )

    assert settings.paths.repository_root == repository_root.resolve()
    assert settings.paths.output_dir == (repository_root / "var/generated").resolve()
    assert settings.paths.database_dir == (repository_root / "var/state").resolve()
    assert not repository_root.exists()


def test_settings_parse_process_controls_and_keep_model_tuning_outside_boundary(tmp_path):
    settings = _settings_for(
        tmp_path,
        SYNTHA_CORS_ORIGINS="https://one.example, https://two.example ",
        SYNTHA_MAX_UPLOAD_BYTES="123",
        SYNTHA_MAX_IMAGE_PIXELS="456",
        SYNTHA_API_START_WORKER="false",
        SYNTHA_ALLOW_REMOTE_PATH_PICKER="yes",
        SYNTHA_LOG_ROLE="test-api",
        SYNTHA_PIPELINE_CACHE_MAX_ENTRIES="2",
        SYNTHA_PIPELINE_CACHE_MAX_MB="4096",
        SYNTHA_WORKER_VRAM_MB="8192",
        SYNTHA_FLUX_OFFLOAD="cpu",
    )

    assert settings.api.cors_origins == (
        "https://one.example",
        "https://two.example",
    )
    assert settings.api.max_artifact_upload_bytes == 123
    assert settings.api.max_artifact_image_pixels == 456
    assert settings.api.start_embedded_worker is False
    assert settings.api.allow_remote_path_picker is True
    assert settings.logging.role == "test-api"
    assert settings.pipeline_cache.max_entries == 2
    assert settings.pipeline_cache.max_cost_mb == 4096
    assert settings.worker.vram_mb == 8192
    assert not hasattr(settings, "flux_offload")


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("SYNTHA_MAX_UPLOAD_BYTES", "invalid", "must be an integer"),
        ("SYNTHA_MAX_IMAGE_PIXELS", "0", "must be positive"),
        ("SYNTHA_WORKER_VRAM_MB", "-1", "must be non-negative"),
    ],
)
def test_invalid_settings_fail_with_actionable_errors(tmp_path, name, value, message):
    with pytest.raises(ValueError, match=message):
        _settings_for(tmp_path, **{name: value})


def test_application_factory_preserves_startup_contract_and_configures_cors(tmp_path):
    settings = _settings_for(
        tmp_path,
        SYNTHA_CORS_ORIGINS="https://ui.example",
    )
    application = create_app(settings)

    assert app.title == "SynthiaEngine API"
    assert application.state.settings is settings
    with TestClient(application) as client:
        health = client.get("/health")
        preflight = client.options(
            "/health",
            headers={
                "Origin": "https://ui.example",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "https://ui.example"
    assert (settings.paths.database_dir / "jobs.sqlite3").exists()


def test_artifact_upload_uses_factory_limits_and_output_path(tmp_path):
    settings = _settings_for(
        tmp_path,
        SYNTHA_MAX_UPLOAD_BYTES="1024",
        SYNTHA_MAX_IMAGE_PIXELS="4",
    )
    client = TestClient(create_app(settings))

    created = client.post(
        "/api/artifacts",
        files={"file": ("image.png", _png_bytes(), "image/png")},
    )
    too_large = client.post(
        "/api/artifacts",
        files={"file": ("large.png", b"x" * 1025, "image/png")},
    )

    assert created.status_code == 201
    assert set(created.json()) == {"artifact_id", "path", "url"}
    assert (settings.paths.output_dir / created.json()["path"]).is_file()
    assert too_large.status_code == 413
    assert too_large.json() == {
        "detail": "Artifact exceeds the configured upload size limit."
    }


def test_artifact_pixel_limit_and_extracted_utility_errors_are_preserved(tmp_path):
    settings = _settings_for(
        tmp_path,
        SYNTHA_MAX_UPLOAD_BYTES="4096",
        SYNTHA_MAX_IMAGE_PIXELS="3",
    )
    client = TestClient(create_app(settings))

    too_many_pixels = client.post(
        "/api/artifacts",
        files={"file": ("image.png", _png_bytes(), "image/png")},
    )
    invalid_mask = client.post(
        "/create-blur-mask",
        files={"mask_image": ("mask.png", b"not-an-image", "image/png")},
    )
    invalid_model = client.post(
        "/api/tools/analyze-model",
        files={"file": ("model.txt", b"not-a-model", "text/plain")},
    )

    assert too_many_pixels.status_code == 413
    assert too_many_pixels.json() == {
        "detail": "Image exceeds the configured pixel limit."
    }
    assert invalid_mask.status_code == 400
    assert invalid_mask.json() == {"detail": "Invalid mask image file."}
    assert invalid_model.status_code == 400
    assert invalid_model.json() == {"detail": "Unsupported file extension: .txt."}


def test_local_path_selection_rejects_remote_clients_unless_explicitly_enabled(
    tmp_path,
    monkeypatch,
):
    settings = _settings_for(tmp_path)
    application = create_app(settings)
    request = Request(
        {
            "type": "http",
            "app": application,
            "client": ("203.0.113.10", 4321),
            "headers": [],
        }
    )
    payload = local_paths.LocalPathSelectRequest(selection_type="folder")

    with pytest.raises(HTTPException) as exc_info:
        local_paths.select_local_path(payload, request)
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == (
        "Local path selection is restricted to loopback clients."
    )

    allowed = replace(
        settings,
        api=replace(settings.api, allow_remote_path_picker=True),
    )
    application.state.settings = allowed
    monkeypatch.setattr(
        local_paths,
        "open_local_path_dialog",
        lambda selection_type: f"selected-{selection_type}",
    )

    response = local_paths.select_local_path(payload, request)
    assert response.model_dump() == {"path": "selected-folder"}


def test_default_repository_root_is_not_process_working_directory():
    assert REPOSITORY_ROOT == Path(__file__).resolve().parents[1]
    assert load_settings({}).paths.output_dir == REPOSITORY_ROOT / "outputs"
