"""Characterization tests for public contracts and backend dependencies."""

from __future__ import annotations

import ast
from pathlib import Path

from backend.main import app
from backend.workflow import (
    TASK_DEFINITIONS,
    WorkflowRequest,
    WorkflowTask,
    build_workflow_catalog,
)


ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = ROOT / "backend"

EXPECTED_PUBLIC_ROUTES = {
    "/api/artifacts": frozenset({"post"}),
    "/api/controlnet/preprocess": frozenset({"post"}),
    "/api/controlnet/preprocessor-models": frozenset({"get"}),
    "/api/controlnet/preprocessors": frozenset({"get"}),
    "/api/jobs": frozenset({"get", "post"}),
    "/api/jobs/{job_id}": frozenset({"get"}),
    "/api/jobs/{job_id}/cancel": frozenset({"post"}),
    "/api/jobs/{job_id}/events": frozenset({"get"}),
    "/api/jobs/{job_id}/tasks": frozenset({"get"}),
    "/api/local-path/select": frozenset({"post"}),
    "/api/presets": frozenset({"get", "post"}),
    "/api/presets/{preset_id}": frozenset({"delete", "get", "patch"}),
    "/api/tools/analyze-model": frozenset({"post"}),
    "/api/workflow/catalog": frozenset({"get"}),
    "/api/workflow/schema": frozenset({"get"}),
    "/api/workflow/task-types": frozenset({"get"}),
    "/create-blur-mask": frozenset({"post"}),
    "/health": frozenset({"get"}),
    "/history": frozenset({"get"}),
    "/lora-models": frozenset({"get", "post"}),
    "/lora-models/{lora_id}": frozenset({"delete", "get", "patch"}),
    "/models": frozenset({"get", "post"}),
    "/models/{model_name}": frozenset({"delete", "get", "patch"}),
}

EXPECTED_TASK_TYPES = {
    "anima.text2img",
    "controlnet.preprocess",
    "ernie-image.text2img",
    "flux.img2img",
    "flux.inpaint",
    "flux.text2img",
    "qwen-image.img2img",
    "qwen-image.inpaint",
    "qwen-image.text2img",
    "sd15.animatediff.text2video",
    "sd15.controlnet.text2img",
    "sd15.hires_fix",
    "sd15.img2img",
    "sd15.inpaint",
    "sd15.ip_adapter.encode",
    "sd15.text2img",
    "sdxl.controlnet.text2img",
    "sdxl.img2img",
    "sdxl.inpaint",
    "sdxl.ip_adapter.encode",
    "sdxl.text2img",
    "wan.image2video",
    "wan.text2video",
    "z-image.img2img",
    "z-image.inpaint",
    "z-image.text2img",
}

RUNTIME_AREAS = {
    "anima",
    "ernie_image",
    "flux",
    "modular_diffusers",
    "qwen_image",
    "sd15",
    "sdxl",
    "wan",
    "z_image",
}

# Imports not listed here remain allowed. These rules prevent dependencies from
# pointing toward HTTP/job orchestration or sideways into another model family.
# Current narrow exceptions are documented in docs/ARCHITECTURE.md.
FORBIDDEN_IMPORTS_BY_AREA = {
    "api": {"main", *RUNTIME_AREAS},
    "jobs": {"main", "api", "adapters", "registries", "lora", *RUNTIME_AREAS},
    "workflow": {"main", "api", "jobs", "registries", "lora"},
    "adapters": {"main", "api", "jobs", "registries", "lora", *RUNTIME_AREAS},
    "registries": {"main", "api", "jobs", "adapters", "lora", *RUNTIME_AREAS},
    "lora": {"main", "api", "jobs", "workflow", "adapters", "registries", *RUNTIME_AREAS},
    "utilities": {"main", "api", "jobs", "workflow", "adapters", "lora", *RUNTIME_AREAS},
    **{
        area: {"main", "api", "jobs", "workflow", *(RUNTIME_AREAS - {area})}
        for area in RUNTIME_AREAS
    },
}


def _backend_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.lineno, node.module))
    return imports


def _backend_area(import_name: str) -> str | None:
    parts = import_name.split(".")
    if len(parts) >= 2 and parts[0] == "backend":
        return parts[1]
    return None


def test_public_http_route_and_method_surface_is_stable():
    http_methods = {"delete", "get", "patch", "post", "put"}
    actual = {
        path: frozenset(operation for operation in operations if operation in http_methods)
        for path, operations in app.openapi()["paths"].items()
    }

    assert actual == EXPECTED_PUBLIC_ROUTES


def test_workflow_task_and_schema_surface_is_stable_and_registry_derived():
    assert set(TASK_DEFINITIONS) == EXPECTED_TASK_TYPES

    request_schema = WorkflowRequest.model_json_schema(by_alias=True)
    task_schema = WorkflowTask.model_json_schema(by_alias=True)
    assert request_schema["required"] == ["tasks"]
    assert set(request_schema["properties"]) == {"tasks", "return"}
    assert request_schema["properties"]["tasks"]["maxItems"] == 64
    assert task_schema["required"] == ["id", "type"]
    assert set(task_schema["properties"]) == {"id", "type", "inputs"}
    assert task_schema["properties"]["id"]["pattern"] == r"^[A-Za-z0-9_-]+$"

    catalog = build_workflow_catalog()
    assert catalog["version"] == "v2"
    assert set(catalog["tasks"]) == EXPECTED_TASK_TYPES
    for task_type, definition in TASK_DEFINITIONS.items():
        task_contract = catalog["tasks"][task_type]
        assert task_contract["input_schema"] == definition.input_model.model_json_schema(
            by_alias=True
        )
        assert task_contract["output_schema"] == definition.output_model.model_json_schema(
            by_alias=True
        )
        assert task_contract["ui_hints"]["task_type"] == task_type
        assert set(task_contract["input_defaults"]) <= set(definition.input_model.model_fields)


def test_backend_static_imports_follow_layer_boundaries():
    violations: list[str] = []
    for owner, forbidden_areas in FORBIDDEN_IMPORTS_BY_AREA.items():
        for path in sorted((BACKEND_ROOT / owner).rglob("*.py")):
            for line_number, import_name in _backend_imports(path):
                imported_area = _backend_area(import_name)
                if imported_area in forbidden_areas:
                    relative_path = path.relative_to(ROOT).as_posix()
                    violations.append(
                        f"{relative_path}:{line_number}: {owner} must not import "
                        f"{imported_area} ({import_name})"
                    )

    assert violations == []


def test_backend_main_is_the_only_fastapi_composition_root():
    app_constructors: list[str] = []
    for path in sorted(BACKEND_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "FastAPI"
            for node in ast.walk(tree)
        ):
            app_constructors.append(path.relative_to(ROOT).as_posix())

    assert app_constructors == ["backend/main.py"]
