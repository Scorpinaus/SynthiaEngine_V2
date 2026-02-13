from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from backend.config import DATABASE_DIR
from backend.workflow import TASK_INPUT_MODELS


class PresetRegistryCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    family: str = Field(min_length=1, max_length=64)
    task_type: str = Field(min_length=1, max_length=128)
    settings: dict[str, Any] = Field(default_factory=dict)


class PresetRegistryEntry(PresetRegistryCreate):
    preset_id: int


class Base(DeclarativeBase):
    pass


class PresetRegistryRow(Base):
    __tablename__ = "preset_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    family: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    task_type: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    settings_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")


REGISTRY_DB_PATH = DATABASE_DIR / "preset_registry.sqlite3"
REGISTRY_DB_URL = f"sqlite:///{REGISTRY_DB_PATH.as_posix()}"

_ENGINE = create_engine(
    REGISTRY_DB_URL,
    future=True,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False},
)
_SessionLocal = sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False, future=True)

_EXTRA_FIELDS_BY_TASK: dict[str, set[str]] = {
    "sd15.text2img": {
        "hires_enabled",
        "hires_scale",
        "controlnet_enabled",
        "controlnet_conditioning_scale",
        "control_guidance_start",
        "control_guidance_end",
        "controlnet_guess_mode",
        "controlnet_compat_mode",
    },
    "sd15.img2img": {"controlnet_enabled"},
    "sd15.inpaint": {"controlnet_enabled"},
}


def _canonical_family_for_task_type(task_type: str) -> str | None:
    if task_type.startswith("qwen-image."):
        return "qwen-image"
    if task_type.startswith("z-image."):
        return "z-image"
    if "." not in task_type:
        return None
    prefix = task_type.split(".", 1)[0]
    if prefix in {"sd15", "sdxl", "flux"}:
        return prefix
    return None


def _normalize_name(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("Field 'name' cannot be empty.")
    return normalized


def _normalize_family(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError("Field 'family' cannot be empty.")
    return normalized


def _normalize_task_type(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("Field 'task_type' cannot be empty.")
    if normalized not in TASK_INPUT_MODELS:
        raise ValueError(f"Unsupported task_type '{normalized}'.")
    return normalized


def _normalize_settings(task_type: str, settings: Any) -> dict[str, Any]:
    if settings is None:
        return {}
    if not isinstance(settings, dict):
        raise ValueError("Field 'settings' must be an object.")

    allowed = set(TASK_INPUT_MODELS[task_type].model_fields.keys())
    allowed.update(_EXTRA_FIELDS_BY_TASK.get(task_type, set()))
    unknown = sorted(key for key in settings.keys() if key not in allowed)
    if unknown:
        raise ValueError(
            f"Unknown preset setting fields for task_type '{task_type}': {', '.join(unknown)}."
        )

    try:
        json.dumps(settings)
    except TypeError as exc:
        raise ValueError("Field 'settings' must be JSON-serializable.") from exc

    return settings


def _validate_task_family_alignment(*, family: str, task_type: str) -> None:
    expected = _canonical_family_for_task_type(task_type)
    if expected and family != expected:
        raise ValueError(f"Field 'family' must match task_type family '{expected}'.")


def _row_to_entry(row: PresetRegistryRow) -> PresetRegistryEntry:
    settings: dict[str, Any]
    try:
        parsed = json.loads(row.settings_json)
        settings = parsed if isinstance(parsed, dict) else {}
    except Exception:
        settings = {}

    return PresetRegistryEntry(
        preset_id=row.id,
        name=row.name,
        family=row.family,
        task_type=row.task_type,
        settings=settings,
    )


def init_preset_registry_db() -> None:
    Base.metadata.create_all(_ENGINE)


def list_preset_entries(*, family: str | None = None, task_type: str | None = None) -> list[PresetRegistryEntry]:
    init_preset_registry_db()
    with _SessionLocal() as session:
        query = select(PresetRegistryRow).order_by(PresetRegistryRow.id.asc())
        if family is not None:
            family_value = family.strip()
            if family_value:
                query = query.where(PresetRegistryRow.family == _normalize_family(family_value))
        if task_type is not None:
            task_value = task_type.strip()
            if task_value:
                query = query.where(PresetRegistryRow.task_type == _normalize_task_type(task_value))
        rows = session.execute(query).scalars().all()
    return [_row_to_entry(row) for row in rows]


def create_preset_entry(entry: PresetRegistryCreate) -> PresetRegistryEntry:
    init_preset_registry_db()

    name = _normalize_name(entry.name)
    family = _normalize_family(entry.family)
    task_type = _normalize_task_type(entry.task_type)
    settings = _normalize_settings(task_type, entry.settings)
    _validate_task_family_alignment(family=family, task_type=task_type)

    with _SessionLocal() as session:
        row = PresetRegistryRow(
            name=name,
            family=family,
            task_type=task_type,
            settings_json=json.dumps(settings, ensure_ascii=True),
        )
        session.add(row)
        session.commit()
        session.refresh(row)
    return _row_to_entry(row)


def get_preset_entry(preset_id: int) -> PresetRegistryEntry:
    init_preset_registry_db()
    with _SessionLocal() as session:
        row = (
            session.execute(select(PresetRegistryRow).where(PresetRegistryRow.id == preset_id).limit(1))
            .scalars()
            .first()
        )
        if row is None:
            raise ValueError(f"Preset with id {preset_id} not found.")
    return _row_to_entry(row)


def update_preset_entry(preset_id: int, updates: dict[str, object]) -> PresetRegistryEntry:
    if not updates:
        raise ValueError("At least one editable field must be provided.")

    editable_fields = {"name", "family", "task_type", "settings"}
    unknown = sorted(field for field in updates.keys() if field not in editable_fields)
    if unknown:
        raise ValueError(f"Unknown editable fields: {', '.join(unknown)}.")

    init_preset_registry_db()
    with _SessionLocal() as session:
        row = (
            session.execute(select(PresetRegistryRow).where(PresetRegistryRow.id == preset_id).limit(1))
            .scalars()
            .first()
        )
        if row is None:
            raise ValueError(f"Preset with id {preset_id} not found.")

        name = _normalize_name(str(updates["name"])) if "name" in updates else row.name
        family = _normalize_family(str(updates["family"])) if "family" in updates else row.family
        task_type = (
            _normalize_task_type(str(updates["task_type"])) if "task_type" in updates else row.task_type
        )
        if "settings" in updates:
            settings = _normalize_settings(task_type, updates["settings"])
        else:
            settings = _row_to_entry(row).settings
            settings = _normalize_settings(task_type, settings)
        _validate_task_family_alignment(family=family, task_type=task_type)

        row.name = name
        row.family = family
        row.task_type = task_type
        row.settings_json = json.dumps(settings, ensure_ascii=True)
        session.commit()
        session.refresh(row)
    return _row_to_entry(row)


def delete_preset_entry(preset_id: int) -> None:
    init_preset_registry_db()
    with _SessionLocal() as session:
        row = (
            session.execute(select(PresetRegistryRow).where(PresetRegistryRow.id == preset_id).limit(1))
            .scalars()
            .first()
        )
        if row is None:
            raise ValueError(f"Preset with id {preset_id} not found.")
        session.delete(row)
        session.commit()


init_preset_registry_db()
