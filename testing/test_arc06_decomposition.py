"""Architecture contracts for the ARC-06 SD1.5/SDXL decomposition."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from backend.sd15 import pipeline as sd15_pipeline
from backend.sdxl import pipeline as sdxl_pipeline


ROOT = Path(__file__).resolve().parents[1]


def _top_level_functions(path: str) -> set[str]:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _parameter_contract(function) -> list[tuple[str, inspect._ParameterKind]]:
    return [(parameter.name, parameter.kind) for parameter in inspect.signature(function).parameters.values()]


def _assert_release_finally_covers_loaded_pipeline(path: str, function_names: set[str]) -> None:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert function_names <= set(functions)

    for function_name in function_names:
        function = functions[function_name]
        release_tries = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Try)
            and any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "release_pipeline"
                for statement in node.finalbody
                for call in ast.walk(statement)
            )
        ]
        assert release_tries, f"{path}:{function_name} must release its pipeline in finally"

        load_assignments = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "pipe" for target in node.targets)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id.startswith("load_")
        ]
        assert load_assignments, f"{path}:{function_name} must have an explicit pipeline load"
        lifecycle_try = release_tries[0]
        for assignment in load_assignments:
            loaded_inside_try = any(assignment is node for node in ast.walk(lifecycle_try))
            try_starts_immediately_after_load = lifecycle_try.lineno <= assignment.end_lineno + 1
            assert loaded_inside_try or try_starts_immediately_after_load


def test_sd15_runtime_facade_is_thin_and_operations_have_explicit_owners():
    assert len((ROOT / "backend/sd15/pipeline.py").read_text(encoding="utf-8").splitlines()) < 300
    assert {
        "generate_images_controlnet_in_process",
        "generate_images_in_process",
    } <= _top_level_functions("backend/sd15/text2img.py")
    assert {
        "generate_images_img2img_in_process",
        "generate_images_img2img_controlnet_in_process",
    } <= _top_level_functions("backend/sd15/img2img.py")
    assert {
        "generate_images_inpaint_in_process",
        "generate_images_inpaint_controlnet_in_process",
    } <= _top_level_functions("backend/sd15/inpaint.py")
    assert {"run_sd15_hires_fix"} <= _top_level_functions("backend/sd15/hires_fix.py")


def test_sdxl_runtime_facade_is_thin_and_operations_have_explicit_owners():
    assert len((ROOT / "backend/sdxl/pipeline.py").read_text(encoding="utf-8").splitlines()) < 200
    assert {"generate_text2img_in_process"} <= _top_level_functions("backend/sdxl/text2img.py")
    assert {"generate_img2img_in_process"} <= _top_level_functions("backend/sdxl/img2img.py")
    assert {"generate_inpaint_in_process"} <= _top_level_functions("backend/sdxl/inpaint.py")
    assert {
        "generate_controlnet_text2img_in_process",
        "generate_img2img_controlnet_in_process",
        "generate_inpaint_controlnet_in_process",
    } <= _top_level_functions("backend/sdxl/controlnet.py")


def test_public_generation_parameter_contracts_are_preserved():
    from backend.sd15 import hires_fix as sd15_hires_fix
    from backend.sd15 import img2img as sd15_img2img
    from backend.sd15 import inpaint as sd15_inpaint
    from backend.sd15 import text2img as sd15_text2img
    from backend.sdxl import controlnet as sdxl_controlnet
    from backend.sdxl import img2img as sdxl_img2img
    from backend.sdxl import inpaint as sdxl_inpaint
    from backend.sdxl import text2img as sdxl_text2img

    pairs = (
        (sd15_pipeline.generate_images_in_process, sd15_text2img.generate_images_in_process),
        (sd15_pipeline.generate_images_controlnet_in_process, sd15_text2img.generate_images_controlnet_in_process),
        (sd15_pipeline.generate_images_img2img_in_process, sd15_img2img.generate_images_img2img_in_process),
        (sd15_pipeline.generate_images_inpaint_in_process, sd15_inpaint.generate_images_inpaint_in_process),
        (sd15_pipeline.run_sd15_hires_fix, sd15_hires_fix.run_sd15_hires_fix),
        (sdxl_pipeline.generate_text2img_in_process, sdxl_text2img.generate_text2img_in_process),
        (sdxl_pipeline.generate_img2img_in_process, sdxl_img2img.generate_img2img_in_process),
        (sdxl_pipeline.generate_inpaint_in_process, sdxl_inpaint.generate_inpaint_in_process),
        (sdxl_pipeline.generate_controlnet_text2img_in_process, sdxl_controlnet.generate_controlnet_text2img_in_process),
    )
    for facade, implementation in pairs:
        assert _parameter_contract(facade) == _parameter_contract(implementation)


def test_generation_pipeline_loads_are_guarded_by_release_finally():
    _assert_release_finally_covers_loaded_pipeline(
        "backend/sd15/text2img.py",
        {"generate_images_controlnet_in_process", "generate_images_in_process"},
    )
    _assert_release_finally_covers_loaded_pipeline(
        "backend/sd15/img2img.py",
        {"generate_images_img2img_in_process", "generate_images_img2img_controlnet_in_process"},
    )
    _assert_release_finally_covers_loaded_pipeline(
        "backend/sd15/inpaint.py",
        {"generate_images_inpaint_in_process", "generate_images_inpaint_controlnet_in_process"},
    )
    _assert_release_finally_covers_loaded_pipeline("backend/sd15/hires_fix.py", {"run_sd15_hires_fix"})
    _assert_release_finally_covers_loaded_pipeline("backend/sdxl/text2img.py", {"generate_text2img_in_process"})
    _assert_release_finally_covers_loaded_pipeline("backend/sdxl/img2img.py", {"generate_img2img_in_process"})
    _assert_release_finally_covers_loaded_pipeline("backend/sdxl/inpaint.py", {"generate_inpaint_in_process"})
    _assert_release_finally_covers_loaded_pipeline(
        "backend/sdxl/controlnet.py",
        {
            "generate_controlnet_text2img_in_process",
            "generate_img2img_controlnet_in_process",
            "generate_inpaint_controlnet_in_process",
        },
    )


def test_workflow_facades_only_compose_operation_adapters():
    assert len((ROOT / "backend/workflow/sd15.py").read_text(encoding="utf-8").splitlines()) < 40
    assert len((ROOT / "backend/workflow/sdxl.py").read_text(encoding="utf-8").splitlines()) < 30
    assert {"run_sd15_img2img"} <= _top_level_functions("backend/workflow/sd15_img2img_task.py")
    assert {"run_sd15_inpaint"} <= _top_level_functions("backend/workflow/sd15_inpaint_task.py")
    assert {"run_sd15_controlnet_text2img"} <= _top_level_functions("backend/workflow/sd15_controlnet_task.py")
    assert {"run_sdxl_img2img_task"} <= _top_level_functions("backend/workflow/sdxl_img2img_task.py")
    assert {"run_sdxl_inpaint_task"} <= _top_level_functions("backend/workflow/sdxl_inpaint_task.py")
    assert {"run_sdxl_controlnet_text2img_task"} <= _top_level_functions("backend/workflow/sdxl_controlnet_task.py")
