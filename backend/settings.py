"""Typed process settings and repository-relative path resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CORS_ORIGINS = (
    "http://127.0.0.1:4173",
    "http://localhost:4173",
)


@dataclass(frozen=True)
class RuntimePaths:
    repository_root: Path
    output_dir: Path
    database_dir: Path


@dataclass(frozen=True)
class ApiSettings:
    cors_origins: tuple[str, ...]
    max_artifact_upload_bytes: int
    max_artifact_image_pixels: int
    start_embedded_worker: bool
    allow_remote_path_picker: bool


@dataclass(frozen=True)
class LoggingSettings:
    role: str


@dataclass(frozen=True)
class PipelineCacheSettings:
    max_entries: int
    max_cost_mb: int


@dataclass(frozen=True)
class WorkerSettings:
    vram_mb: int


@dataclass(frozen=True)
class AppSettings:
    paths: RuntimePaths
    api: ApiSettings
    logging: LoggingSettings
    pipeline_cache: PipelineCacheSettings
    worker: WorkerSettings


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _parse_non_negative_int(
    environment: Mapping[str, str],
    name: str,
    *,
    default: int,
    positive: bool = False,
) -> int:
    raw_value = environment.get(name)
    try:
        value = default if raw_value is None else int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    minimum = 1 if positive else 0
    if value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}.")
    return value


def _resolve_path(root: Path, configured: str | None, default_name: str) -> Path:
    path = Path(configured).expanduser() if configured else root / default_name
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def load_settings(
    environment: Mapping[str, str] | None = None,
    *,
    repository_root: Path | None = None,
    default_log_role: str = "app",
) -> AppSettings:
    """Parse environment values without mutating the process or filesystem."""
    env = os.environ if environment is None else environment
    root = (repository_root or REPOSITORY_ROOT).expanduser().resolve()

    configured_origins = env.get("SYNTHA_CORS_ORIGINS", "")
    cors_origins = tuple(
        origin.strip() for origin in configured_origins.split(",") if origin.strip()
    ) or DEFAULT_CORS_ORIGINS

    return AppSettings(
        paths=RuntimePaths(
            repository_root=root,
            output_dir=_resolve_path(root, env.get("SYNTHA_OUTPUT_DIR"), "outputs"),
            database_dir=_resolve_path(root, env.get("SYNTHA_DATABASE_DIR"), "database"),
        ),
        api=ApiSettings(
            cors_origins=cors_origins,
            max_artifact_upload_bytes=_parse_non_negative_int(
                env,
                "SYNTHA_MAX_UPLOAD_BYTES",
                default=100 * 1024 * 1024,
                positive=True,
            ),
            max_artifact_image_pixels=_parse_non_negative_int(
                env,
                "SYNTHA_MAX_IMAGE_PIXELS",
                default=64 * 1024 * 1024,
                positive=True,
            ),
            start_embedded_worker=_parse_bool(
                env.get("SYNTHA_API_START_WORKER"),
                default=True,
            ),
            allow_remote_path_picker=_parse_bool(
                env.get("SYNTHA_ALLOW_REMOTE_PATH_PICKER"),
                default=False,
            ),
        ),
        logging=LoggingSettings(
            role=env.get("SYNTHA_LOG_ROLE", default_log_role).strip() or default_log_role
        ),
        pipeline_cache=PipelineCacheSettings(
            max_entries=_parse_non_negative_int(
                env,
                "SYNTHA_PIPELINE_CACHE_MAX_ENTRIES",
                default=0,
            ),
            max_cost_mb=_parse_non_negative_int(
                env,
                "SYNTHA_PIPELINE_CACHE_MAX_MB",
                default=0,
            ),
        ),
        worker=WorkerSettings(
            vram_mb=_parse_non_negative_int(
                env,
                "SYNTHA_WORKER_VRAM_MB",
                default=0,
            )
        ),
    )


def ensure_runtime_directories(settings: AppSettings) -> None:
    """Create writable runtime directories during explicit application startup."""
    settings.paths.output_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.database_dir.mkdir(parents=True, exist_ok=True)
