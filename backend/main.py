"""SynthiaEngine FastAPI application composition root."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.jobs.queue import JobQueueConfig, create_job_queue
from backend.settings import AppSettings, ensure_runtime_directories, load_settings
from backend.utilities.logging import configure_logging


logger = logging.getLogger(__name__)


def _job_db_url(settings: AppSettings) -> str:
    path = settings.paths.database_dir / "jobs.sqlite3"
    return f"sqlite:///{path.as_posix()}"


def _startup_job_queue(application: FastAPI) -> None:
    """Initialize queue state as a directly testable lifecycle unit."""
    settings: AppSettings = application.state.settings
    queue_config = JobQueueConfig(
        db_url=_job_db_url(settings),
        worker_vram_mb=settings.worker.vram_mb,
    )
    engine, sessionmaker, worker = create_job_queue(queue_config)
    application.state.job_engine = engine
    application.state.job_sessionmaker = sessionmaker
    application.state.job_worker = worker
    application.state.job_worker_started = False

    if settings.api.start_embedded_worker:
        worker.start()
        application.state.job_worker_started = True
        logger.info("Embedded API job worker started.")
    else:
        logger.info("Embedded API job worker disabled; external render worker expected.")


def _shutdown_job_queue(application: FastAPI) -> None:
    worker = getattr(application.state, "job_worker", None)
    if worker is not None:
        worker.stop()
    engine = getattr(application.state, "job_engine", None)
    if engine is not None:
        engine.dispose()


@asynccontextmanager
async def _app_lifespan(application: FastAPI):
    """Own runtime directories, queue state, and renderer cleanup."""
    settings: AppSettings = application.state.settings
    ensure_runtime_directories(settings)
    _startup_job_queue(application)
    try:
        yield
    finally:
        _shutdown_job_queue(application)


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Assemble a SynthiaEngine API instance with explicit process settings."""
    resolved_settings = settings or load_settings(default_log_role="api")
    configure_logging(role=resolved_settings.logging.role)
    ensure_runtime_directories(resolved_settings)

    # Import routers after runtime paths exist because the legacy registry
    # compatibility modules still initialize their SQLite stores on import.
    from backend.api.artifacts import router as artifacts_router
    from backend.api.controlnet import router as controlnet_router
    from backend.api.history import router as history_router
    from backend.api.jobs import router as jobs_router
    from backend.api.local_paths import router as local_paths_router
    from backend.api.loras import router as loras_router
    from backend.api.masks import router as masks_router
    from backend.api.model_analysis import router as model_analysis_router
    from backend.api.models import router as models_router
    from backend.api.presets import router as presets_router
    from backend.api.workflow import router as workflow_router

    application = FastAPI(title="SynthiaEngine API", lifespan=_app_lifespan)
    application.state.settings = resolved_settings
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.api.cors_origins),
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Idempotency-Key"],
    )
    application.mount(
        "/outputs",
        StaticFiles(directory=resolved_settings.paths.output_dir, check_dir=False),
        name="outputs",
    )

    application.include_router(jobs_router)
    application.include_router(workflow_router)
    application.include_router(history_router)
    application.include_router(presets_router)
    application.include_router(models_router)
    application.include_router(loras_router)
    application.include_router(artifacts_router)
    application.include_router(local_paths_router)
    application.include_router(controlnet_router)
    application.include_router(model_analysis_router)
    application.include_router(masks_router)

    @application.get("/health")
    async def health_check():
        return {"status": "ok"}

    return application


app = create_app()
