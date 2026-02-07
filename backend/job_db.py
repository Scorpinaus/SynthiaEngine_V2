from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from backend.config import DATABASE_DIR, OUTPUT_DIR

DEFAULT_JOBS_DB_PATH = DATABASE_DIR / "jobs.sqlite3"
LEGACY_JOBS_DB_PATH = OUTPUT_DIR / "jobs.sqlite3"
DEFAULT_JOB_DB_URL = f"sqlite:///{DEFAULT_JOBS_DB_PATH.as_posix()}"


@dataclass(frozen=True)
class JobDbConfig:
    url: str = DEFAULT_JOB_DB_URL


def create_job_engine(config: JobDbConfig) -> Engine:
    connect_args: dict[str, object] = {}
    if config.url.startswith("sqlite:"):
        connect_args["check_same_thread"] = False
        db_path = config.url.removeprefix("sqlite:///")
        if db_path and not db_path.startswith(":memory:"):
            sqlite_path = Path(db_path).expanduser().resolve()
            default_path = DEFAULT_JOBS_DB_PATH.expanduser().resolve()
            if sqlite_path == default_path and not sqlite_path.exists():
                legacy_path = LEGACY_JOBS_DB_PATH.expanduser().resolve()
                if legacy_path.exists():
                    try:
                        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
                        legacy_path.replace(sqlite_path)
                    except Exception:
                        pass

            sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    return create_engine(
        config.url,
        future=True,
        pool_pre_ping=True,
        connect_args=connect_args,
    )


def create_sessionmaker(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
