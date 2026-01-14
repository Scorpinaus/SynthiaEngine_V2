from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker


@dataclass(frozen=True)
class JobDbConfig:
    url: str = "sqlite:///outputs/jobs.sqlite3"


def create_job_engine(config: JobDbConfig) -> Engine:
    connect_args: dict[str, object] = {}
    if config.url.startswith("sqlite:"):
        connect_args["check_same_thread"] = False
        db_path = config.url.removeprefix("sqlite:///")
        if db_path and not db_path.startswith(":memory:"):
            Path(db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    return create_engine(
        config.url,
        future=True,
        pool_pre_ping=True,
        connect_args=connect_args,
    )


def create_sessionmaker(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

