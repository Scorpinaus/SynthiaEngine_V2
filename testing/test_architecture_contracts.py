"""Characterization tests for public contracts and backend dependencies."""

from __future__ import annotations

import ast
from pathlib import Path
from types import ModuleType

from fastapi.routing import APIRoute

import backend.workflow as workflow_package
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


def test_workflow_package_has_explicit_compatibility_exports():
    package_source = (BACKEND_ROOT / "workflow" / "__init__.py").read_text(encoding="utf-8")

    assert isinstance(workflow_package, ModuleType)
    assert "__getattr__" not in vars(workflow_package)
    assert "_WorkflowModule" not in package_source
    assert "sys.modules" not in package_source
    assert "_sd15_text2img" not in vars(workflow_package)
    assert workflow_package.execute_workflow.__module__ == "backend.workflow.engine"


def test_workflow_engine_is_orchestration_only():
    engine_path = BACKEND_ROOT / "workflow" / "engine.py"
    tree = ast.parse(engine_path.read_text(encoding="utf-8"), filename=str(engine_path))
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    backend_imports = {
        import_name
        for _, import_name in _backend_imports(engine_path)
        if import_name.startswith("backend.")
    }

    assert functions == {"_execution_order", "execute_workflow"}
    assert backend_imports == {
        "backend.workflow.assembly",
        "backend.workflow.types",
        "backend.workflow.utility",
    }


def test_workflow_assembly_has_no_central_handler_map():
    assembly_source = (BACKEND_ROOT / "workflow" / "assembly.py").read_text(encoding="utf-8")

    assert "_TASK_HANDLERS" not in assembly_source
    assert "merge_task_definitions(" in assembly_source


def test_backend_workflow_callers_import_owning_modules():
    package_imports: list[str] = []
    for path in sorted(BACKEND_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            isinstance(node, ast.ImportFrom) and node.module == "backend.workflow"
            for node in ast.walk(tree)
        ):
            package_imports.append(path.relative_to(ROOT).as_posix())

    assert package_imports == []


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


def test_http_domains_are_owned_by_focused_api_modules():
    expected_owners = {
        "/api/artifacts": "backend.api.artifacts",
        "/api/controlnet/preprocess": "backend.api.controlnet",
        "/api/controlnet/preprocessor-models": "backend.api.controlnet",
        "/api/controlnet/preprocessors": "backend.api.controlnet",
        "/api/local-path/select": "backend.api.local_paths",
        "/api/tools/analyze-model": "backend.api.model_analysis",
        "/create-blur-mask": "backend.api.masks",
        "/history": "backend.api.history",
    }
    included_routes = []
    for route in app.routes:
        if isinstance(route, APIRoute):
            included_routes.append(route)
            continue
        original_router = getattr(route, "original_router", None)
        if original_router is not None:
            included_routes.extend(original_router.routes)
    actual_owners = {
        route.path: route.endpoint.__module__
        for route in included_routes
        if isinstance(route, APIRoute) and route.path in expected_owners
    }

    assert actual_owners == expected_owners


def test_composition_root_has_no_scattered_environment_parsing_or_path_creation():
    main_source = (BACKEND_ROOT / "main.py").read_text(encoding="utf-8")
    config_source = (BACKEND_ROOT / "config.py").read_text(encoding="utf-8")

    assert "os.getenv" not in main_source
    assert "os.environ" not in main_source
    assert ".mkdir(" not in config_source
    assert "def create_app(" in main_source
