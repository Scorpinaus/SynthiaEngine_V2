from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import Integer, String, Text, create_engine, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from backend.config import DATABASE_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)


class LoraPromptPreset(BaseModel):
    name: str
    words: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("LoRA prompt preset name cannot be empty.")
        return normalized

    @field_validator("words")
    @classmethod
    def validate_words(cls, value: list[str]) -> list[str]:
        normalized = [word.strip() for word in value if isinstance(word, str) and word.strip()]
        if not normalized:
            raise ValueError("LoRA prompt preset words cannot be empty.")
        return normalized


class LoraRuntimeProfile(BaseModel):
    """Fixed runtime settings for a LoRA with specialized inference behavior."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["qwen_image_lightning"]
    base_variant: Literal["qwen-image-2512"]
    steps: Literal[4, 8]
    true_cfg_scale: float
    scheduler_profile: Literal["qwen_image_lightning_shift3"]
    adapter_strength: float
    supported_tasks: list[Literal["text2img", "img2img", "inpaint"]]

    @field_validator("true_cfg_scale", "adapter_strength", mode="before")
    @classmethod
    def validate_fixed_number(cls, value: Any, info) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Qwen Image Lightning {info.field_name} must be 1.0.")
        if float(value) != 1.0:
            raise ValueError(f"Qwen Image Lightning {info.field_name} must be 1.0.")
        return float(value)

    @field_validator("supported_tasks", mode="before")
    @classmethod
    def validate_supported_tasks(cls, value: Any) -> list[str]:
        legacy_tasks = ["text2img"]
        supported_tasks = ["text2img", "img2img", "inpaint"]
        if value == legacy_tasks:
            return supported_tasks
        if value != supported_tasks:
            raise ValueError(
                "Qwen Image Lightning supported_tasks must be "
                "['text2img', 'img2img', 'inpaint']."
            )
        return supported_tasks


class LoraCompatibility(BaseModel):
    """Declared support for a standard LoRA in a specialized runtime profile."""

    model_config = ConfigDict(extra="forbid")

    base_variants: list[Literal["qwen-image-2512"]]
    runtime_profile_kinds: list[Literal["qwen_image_lightning"]]
    supported_tasks: list[Literal["text2img", "img2img", "inpaint"]]

    @field_validator("base_variants", "runtime_profile_kinds", "supported_tasks")
    @classmethod
    def validate_unique_nonempty_list(cls, value: list[str], info) -> list[str]:
        if not value:
            raise ValueError(f"LoRA compatibility {info.field_name} cannot be empty.")
        if len(value) != len(set(value)):
            raise ValueError(f"LoRA compatibility {info.field_name} cannot contain duplicates.")
        return value


class LoraRegistryEntry(BaseModel):
    lora_id: int
    lora_model_family: str
    lora_type: str
    lora_location: str
    file_path: str
    name: str | None = None
    prompt_presets: list[LoraPromptPreset] = Field(default_factory=list)
    runtime_profile: LoraRuntimeProfile | None = None
    compatibility: LoraCompatibility | None = None
    weight_name: str | None = None
    subfolder: str | None = None
    revision: str | None = None

    @model_validator(mode="after")
    def validate_runtime_profile_and_compatibility(self) -> LoraRegistryEntry:
        if self.runtime_profile is not None:
            if self.lora_model_family != "qwen-image":
                raise ValueError("Qwen Image Lightning requires lora_model_family 'qwen-image'.")
            if self.lora_type != "lora":
                raise ValueError("Qwen Image Lightning requires lora_type 'lora'.")
            if self.lora_location == "hub" and not (self.weight_name or "").strip():
                raise ValueError("Hub Qwen Image Lightning entries require weight_name.")

        if self.compatibility is not None:
            if self.lora_model_family != "qwen-image":
                raise ValueError("LoRA compatibility requires lora_model_family 'qwen-image'.")
            if self.lora_type != "lora":
                raise ValueError("LoRA compatibility requires lora_type 'lora'.")
            if self.runtime_profile is not None:
                raise ValueError("LoRA compatibility is not allowed on Lightning runtime-profile entries.")
        return self


class Base(DeclarativeBase):
    pass


class LoraRegistryRow(Base):
    __tablename__ = "lora_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lora_id: Mapped[int] = mapped_column(Integer, nullable=False, unique=True, index=True)
    lora_model_family: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    lora_type: Mapped[str] = mapped_column(String(64), nullable=False)
    lora_location: Mapped[str] = mapped_column(String(64), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    prompt_presets_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    runtime_profile_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    compatibility_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    weight_name: Mapped[str | None] = mapped_column(String(512), nullable=True)
    subfolder: Mapped[str | None] = mapped_column(String(512), nullable=True)
    revision: Mapped[str | None] = mapped_column(String(512), nullable=True)


REGISTRY_JSON_PATH = Path(__file__).with_name("lora_registry.json")
REGISTRY_DB_PATH = DATABASE_DIR / "lora_registry.sqlite3"
LEGACY_REGISTRY_DB_PATH = OUTPUT_DIR / "lora_registry.sqlite3"
if not REGISTRY_DB_PATH.exists() and LEGACY_REGISTRY_DB_PATH.exists():
    try:
        REGISTRY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        LEGACY_REGISTRY_DB_PATH.replace(REGISTRY_DB_PATH)
    except Exception:
        pass
REGISTRY_DB_URL = f"sqlite:///{REGISTRY_DB_PATH.as_posix()}"

_ENGINE = create_engine(
    REGISTRY_DB_URL,
    future=True,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False},
)
_SessionLocal = sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False, future=True)


def _normalize_prompt_presets(value: Any) -> list[LoraPromptPreset]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Field 'prompt_presets' must be a list.")
    return [LoraPromptPreset.model_validate(preset) for preset in value]


def _prompt_presets_to_json(value: Any) -> str:
    presets = _normalize_prompt_presets(value)
    return json.dumps([preset.model_dump() for preset in presets], ensure_ascii=True)


def _prompt_presets_from_json(value: str | None) -> list[LoraPromptPreset]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
        return _normalize_prompt_presets(parsed)
    except Exception:
        return []


def _runtime_profile_to_json(value: LoraRuntimeProfile | dict[str, Any] | None) -> str | None:
    if value is None:
        return None
    return json.dumps(LoraRuntimeProfile.model_validate(value).model_dump(), ensure_ascii=True)


def _runtime_profile_from_json(value: str | None) -> LoraRuntimeProfile | None:
    if not value:
        return None
    try:
        return LoraRuntimeProfile.model_validate(json.loads(value))
    except Exception:
        return None


def _compatibility_to_json(value: LoraCompatibility | dict[str, Any] | None) -> str | None:
    if value is None:
        return None
    return json.dumps(LoraCompatibility.model_validate(value).model_dump(), ensure_ascii=True)


def _compatibility_from_json(value: str | None) -> LoraCompatibility | None:
    if not value:
        return None
    try:
        return LoraCompatibility.model_validate(json.loads(value))
    except Exception:
        return None


def _row_to_entry(row: LoraRegistryRow) -> LoraRegistryEntry:
    return LoraRegistryEntry(
        lora_id=row.lora_id,
        lora_model_family=row.lora_model_family,
        lora_type=row.lora_type,
        lora_location=row.lora_location,
        file_path=row.file_path,
        name=row.name,
        prompt_presets=_prompt_presets_from_json(row.prompt_presets_json),
        runtime_profile=_runtime_profile_from_json(row.runtime_profile_json),
        compatibility=_compatibility_from_json(row.compatibility_json),
        weight_name=row.weight_name,
        subfolder=row.subfolder,
        revision=row.revision,
    )


def init_lora_registry_db() -> None:
    Base.metadata.create_all(_ENGINE)
    existing_columns = {column["name"] for column in inspect(_ENGINE).get_columns(LoraRegistryRow.__tablename__)}
    with _ENGINE.begin() as connection:
        if "prompt_presets_json" not in existing_columns:
            connection.execute(
                text("ALTER TABLE lora_registry ADD COLUMN prompt_presets_json TEXT NOT NULL DEFAULT '[]'")
            )
        for column_name, column_sql in (
            ("runtime_profile_json", "TEXT"),
            ("compatibility_json", "TEXT"),
            ("weight_name", "VARCHAR(512)"),
            ("subfolder", "VARCHAR(512)"),
            ("revision", "VARCHAR(512)"),
        ):
            if column_name not in existing_columns:
                connection.execute(text(f"ALTER TABLE lora_registry ADD COLUMN {column_name} {column_sql}"))


def _db_has_rows() -> bool:
    with _SessionLocal() as session:
        existing = session.execute(select(LoraRegistryRow.id).limit(1)).scalar_one_or_none()
        return existing is not None


def _migrate_json_if_needed() -> None:
    if not REGISTRY_JSON_PATH.exists():
        return

    init_lora_registry_db()
    if _db_has_rows():
        return

    try:
        raw_data = json.loads(REGISTRY_JSON_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read LoRA registry JSON: %s", exc)
        return

    if not isinstance(raw_data, list):
        logger.warning("LoRA registry JSON must be a list; skipping migration.")
        return

    with _SessionLocal() as session:
        for index, entry in enumerate(raw_data):
            if not isinstance(entry, dict):
                logger.warning("Skipping invalid LoRA registry entry at index %s: not an object.", index)
                continue
            try:
                lora_entry = LoraRegistryEntry(**entry)
            except Exception as exc:
                logger.warning("Skipping invalid LoRA registry entry at index %s: %s", index, exc)
                continue

            try:
                with session.begin_nested():
                    session.add(
                        LoraRegistryRow(
                            lora_id=lora_entry.lora_id,
                            lora_model_family=lora_entry.lora_model_family,
                            lora_type=lora_entry.lora_type,
                            lora_location=lora_entry.lora_location,
                            file_path=lora_entry.file_path,
                            name=lora_entry.name,
                            prompt_presets_json=_prompt_presets_to_json(lora_entry.prompt_presets),
                            runtime_profile_json=_runtime_profile_to_json(lora_entry.runtime_profile),
                            compatibility_json=_compatibility_to_json(lora_entry.compatibility),
                            weight_name=lora_entry.weight_name,
                            subfolder=lora_entry.subfolder,
                            revision=lora_entry.revision,
                        )
                    )
            except IntegrityError as exc:
                logger.warning(
                    "Skipping invalid LoRA registry entry at index %s: duplicate lora_id %s.",
                    index,
                    entry.get("lora_id"),
                )
                continue
        session.commit()


def load_lora_registry() -> list[LoraRegistryEntry]:
    init_lora_registry_db()
    with _SessionLocal() as session:
        rows = (
            session.execute(select(LoraRegistryRow).order_by(LoraRegistryRow.lora_id.asc(), LoraRegistryRow.id.asc()))
            .scalars()
            .all()
        )
    return [_row_to_entry(row) for row in rows]


def list_lora_entries() -> list[LoraRegistryEntry]:
    return load_lora_registry()


def save_lora_registry(entries: list[LoraRegistryEntry]) -> None:
    init_lora_registry_db()
    with _SessionLocal() as session:
        session.query(LoraRegistryRow).delete()
        for entry in entries:
            session.add(
                LoraRegistryRow(
                    lora_id=entry.lora_id,
                    lora_model_family=entry.lora_model_family,
                    lora_type=entry.lora_type,
                    lora_location=entry.lora_location,
                    file_path=entry.file_path,
                    name=entry.name,
                    prompt_presets_json=_prompt_presets_to_json(entry.prompt_presets),
                    runtime_profile_json=_runtime_profile_to_json(entry.runtime_profile),
                    compatibility_json=_compatibility_to_json(entry.compatibility),
                    weight_name=entry.weight_name,
                    subfolder=entry.subfolder,
                    revision=entry.revision,
                )
            )
        try:
            session.commit()
        except IntegrityError as exc:
            session.rollback()
            raise ValueError("LoRA registry contains duplicate ids.") from exc
    LORA_REGISTRY[:] = entries


def add_lora(entry: LoraRegistryEntry) -> LoraRegistryEntry:
    init_lora_registry_db()
    with _SessionLocal() as session:
        row = LoraRegistryRow(
            lora_id=entry.lora_id,
            lora_model_family=entry.lora_model_family,
            lora_type=entry.lora_type,
            lora_location=entry.lora_location,
            file_path=entry.file_path,
            name=entry.name,
            prompt_presets_json=_prompt_presets_to_json(entry.prompt_presets),
            runtime_profile_json=_runtime_profile_to_json(entry.runtime_profile),
            compatibility_json=_compatibility_to_json(entry.compatibility),
            weight_name=entry.weight_name,
            subfolder=entry.subfolder,
            revision=entry.revision,
        )
        session.add(row)
        try:
            session.commit()
        except IntegrityError as exc:
            session.rollback()
            raise ValueError(f"LoRA with id {entry.lora_id} already exists.") from exc
        session.refresh(row)
    created = _row_to_entry(row)
    LORA_REGISTRY.append(created)
    return created


def update_lora_entry(lora_id: int, updates: dict[str, object]) -> LoraRegistryEntry:
    if not updates:
        raise ValueError("At least one editable field must be provided.")

    editable_fields = {
        "lora_model_family",
        "lora_type",
        "lora_location",
        "file_path",
        "name",
        "prompt_presets",
        "runtime_profile",
        "compatibility",
        "weight_name",
        "subfolder",
        "revision",
    }
    unknown_fields = sorted(field for field in updates.keys() if field not in editable_fields)
    if unknown_fields:
        raise ValueError(f"Unknown editable fields: {', '.join(unknown_fields)}.")

    for field_name in ("lora_model_family", "lora_type", "lora_location", "file_path"):
        if field_name in updates and updates[field_name] is None:
            raise ValueError(f"Field '{field_name}' cannot be null.")

    init_lora_registry_db()
    with _SessionLocal() as session:
        row = (
            session.execute(select(LoraRegistryRow).where(LoraRegistryRow.lora_id == lora_id).limit(1))
            .scalars()
            .first()
        )
        if row is None:
            raise ValueError(f"LoRA with id {lora_id} not found.")

        candidate_data = _row_to_entry(row).model_dump()
        candidate_data.update(updates)
        candidate = LoraRegistryEntry.model_validate(candidate_data)
        validated_data = candidate.model_dump()

        for key in updates:
            if key == "prompt_presets":
                row.prompt_presets_json = _prompt_presets_to_json(candidate.prompt_presets)
            elif key == "runtime_profile":
                row.runtime_profile_json = _runtime_profile_to_json(candidate.runtime_profile)
            elif key == "compatibility":
                row.compatibility_json = _compatibility_to_json(candidate.compatibility)
            else:
                setattr(row, key, validated_data[key])
        session.commit()
        session.refresh(row)

    updated = _row_to_entry(row)
    for index, existing in enumerate(LORA_REGISTRY):
        if existing.lora_id == lora_id:
            LORA_REGISTRY[index] = updated
            break
    else:
        LORA_REGISTRY.append(updated)
    return updated


def get_lora_entry(lora_id: int) -> LoraRegistryEntry:
    init_lora_registry_db()
    with _SessionLocal() as session:
        row = (
            session.execute(select(LoraRegistryRow).where(LoraRegistryRow.lora_id == lora_id).limit(1))
            .scalars()
            .first()
        )
        if row is None:
            raise ValueError(f"LoRA with id {lora_id} not found.")
        return _row_to_entry(row)


def delete_lora_entry(lora_id: int) -> None:
    init_lora_registry_db()
    with _SessionLocal() as session:
        row = (
            session.execute(select(LoraRegistryRow).where(LoraRegistryRow.lora_id == lora_id).limit(1))
            .scalars()
            .first()
        )
        if row is None:
            raise ValueError(f"LoRA with id {lora_id} not found.")
        session.delete(row)
        session.commit()

    LORA_REGISTRY[:] = [entry for entry in LORA_REGISTRY if entry.lora_id != lora_id]


init_lora_registry_db()
_migrate_json_if_needed()
LORA_REGISTRY: list[LoraRegistryEntry] = load_lora_registry()
