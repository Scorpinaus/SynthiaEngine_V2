from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel
from sqlalchemy import Integer, String, Text, create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from backend.config import OUTPUT_DIR

logger = logging.getLogger(__name__)


class ModelRegistryEntry(BaseModel):
    name: str
    family: str
    model_type: str
    location_type: str
    model_id: int
    version: str
    link: str


class Base(DeclarativeBase):
    pass


class ModelRegistryRow(Base):
    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False, unique=True, index=True)
    family: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_type: Mapped[str] = mapped_column(String(64), nullable=False)
    location_type: Mapped[str] = mapped_column(String(64), nullable=False)
    model_id: Mapped[int] = mapped_column(Integer, nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    link: Mapped[str] = mapped_column(Text, nullable=False)


REGISTRY_JSON_PATH = Path(__file__).with_name("model_registry.json")
REGISTRY_DB_PATH = OUTPUT_DIR / "model_registry.sqlite3"
REGISTRY_DB_URL = f"sqlite:///{REGISTRY_DB_PATH.as_posix()}"

_ENGINE = create_engine(
    REGISTRY_DB_URL,
    future=True,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False},
)
_SessionLocal = sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False, future=True)


def _row_to_entry(row: ModelRegistryRow) -> ModelRegistryEntry:
    return ModelRegistryEntry(
        name=row.name,
        family=row.family,
        model_type=row.model_type,
        location_type=row.location_type,
        model_id=row.model_id,
        version=row.version,
        link=row.link,
    )


def init_model_registry_db() -> None:
    Base.metadata.create_all(_ENGINE)


def _db_has_rows() -> bool:
    with _SessionLocal() as session:
        existing = session.execute(select(ModelRegistryRow.id).limit(1)).scalar_one_or_none()
        return existing is not None


def _migrate_json_if_needed() -> None:
    if not REGISTRY_JSON_PATH.exists():
        return

    init_model_registry_db()
    if _db_has_rows():
        return

    try:
        raw_data = json.loads(REGISTRY_JSON_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read model registry JSON: %s", exc)
        return

    if not isinstance(raw_data, list):
        logger.warning("Model registry JSON must be a list; skipping migration.")
        return

    with _SessionLocal() as session:
        for entry in raw_data:
            if not isinstance(entry, dict):
                continue
            try:
                model_entry = ModelRegistryEntry(**entry)
            except Exception as exc:
                logger.warning("Skipping invalid model registry entry: %s", exc)
                continue
            session.add(
                ModelRegistryRow(
                    name=model_entry.name,
                    family=model_entry.family,
                    model_type=model_entry.model_type,
                    location_type=model_entry.location_type,
                    model_id=model_entry.model_id,
                    version=model_entry.version,
                    link=model_entry.link,
                )
            )
        session.commit()


init_model_registry_db()
_migrate_json_if_needed()


def list_model_entries() -> list[ModelRegistryEntry]:
    with _SessionLocal() as session:
        rows = (
            session.execute(
                select(ModelRegistryRow).order_by(
                    ModelRegistryRow.model_id.asc(),
                    ModelRegistryRow.id.asc(),
                )
            )
            .scalars()
            .all()
        )
    return [_row_to_entry(row) for row in rows]


def create_model_entry(entry: ModelRegistryEntry) -> ModelRegistryEntry:
    init_model_registry_db()
    with _SessionLocal() as session:
        row = ModelRegistryRow(
            name=entry.name,
            family=entry.family,
            model_type=entry.model_type,
            location_type=entry.location_type,
            model_id=entry.model_id,
            version=entry.version,
            link=entry.link,
        )
        session.add(row)
        try:
            session.commit()
        except IntegrityError as exc:
            session.rollback()
            raise ValueError("Model name already exists.") from exc
        session.refresh(row)
    return _row_to_entry(row)


def get_model_entry(model_name: str | None) -> ModelRegistryEntry:
    init_model_registry_db()
    with _SessionLocal() as session:
        if model_name:
            row = (
                session.execute(
                    select(ModelRegistryRow).where(ModelRegistryRow.name == model_name).limit(1)
                )
                .scalars()
                .first()
            )
            if row is not None:
                return _row_to_entry(row)

        row = (
            session.execute(
                select(ModelRegistryRow).order_by(
                    ModelRegistryRow.model_id.asc(),
                    ModelRegistryRow.id.asc(),
                )
            )
            .scalars()
            .first()
        )
        if row is None:
            raise ValueError("Model registry is empty.")
        return _row_to_entry(row)


def get_model_family(model_name: str | None) -> str | None:
    if model_name:
        with _SessionLocal() as session:
            row = (
                session.execute(
                    select(ModelRegistryRow.family).where(ModelRegistryRow.name == model_name).limit(1)
                )
                .scalars()
                .first()
            )
            if row is not None:
                return row

        lowered = model_name.lower()
        if re.search(r"flux", lowered):
            return "flux"
        if re.search(r"sdxl", lowered):
            return "sdxl"
        if re.search(r"qwen[-_\\s]?image", lowered):
            return "qwen-image"
        if re.search(r"z[-_\\s]?image|turbo", lowered):
            return "z-image-turbo"
        if re.search(r"sd[\\s_-]*1\\.?5|sd15", lowered):
            return "sd15"

    entries = list_model_entries()
    if entries:
        return entries[0].family

    return None
