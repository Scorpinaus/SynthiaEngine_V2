from __future__ import annotations

from importlib import import_module
import sys
import types
from typing import Any

_ENGINE_MODULE = "backend.workflow.engine"


def _load_engine() -> types.ModuleType:
    module = import_module(_ENGINE_MODULE)
    for name, value in vars(module).items():
        if not (name.startswith("__") and name.endswith("__")):
            globals().setdefault(name, value)
    return module


def __getattr__(name: str) -> Any:
    module = _load_engine()
    try:
        value = getattr(module, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    module = _load_engine()
    return sorted(set(globals()) | set(dir(module)))


class _WorkflowModule(types.ModuleType):
    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        module = sys.modules.get(_ENGINE_MODULE)
        if module is not None and hasattr(module, name):
            setattr(module, name, value)


sys.modules[__name__].__class__ = _WorkflowModule
