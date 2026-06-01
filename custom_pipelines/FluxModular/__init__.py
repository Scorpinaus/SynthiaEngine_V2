from typing import TYPE_CHECKING

from diffusers.utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_torch_available,
    is_transformers_available,
)


_dummy_objects = {}
_import_structure = {}

try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from diffusers.utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["low_memory"] = [
        "LowMemoryFluxAutoBlocks",
        "LowMemoryFluxKontextAutoBlocks",
        "enable_low_memory_flux_modular",
    ]
    _import_structure["modular_blocks_flux"] = ["FluxAutoBlocks"]
    _import_structure["modular_blocks_flux_kontext"] = ["FluxKontextAutoBlocks"]
    _import_structure["modular_pipeline"] = ["FluxKontextModularPipeline", "FluxModularPipeline"]
    _import_structure["transformer_memory"] = [
        "clear_low_memory_flux_transformer_buffers",
        "disable_low_memory_flux_transformer_buffers",
        "enable_low_memory_flux_transformer_buffers",
    ]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from diffusers.utils.dummy_torch_and_transformers_objects import *  # noqa F403
    else:
        from .low_memory import (
            LowMemoryFluxAutoBlocks,
            LowMemoryFluxKontextAutoBlocks,
            enable_low_memory_flux_modular,
        )
        from .modular_blocks_flux import FluxAutoBlocks
        from .modular_blocks_flux_kontext import FluxKontextAutoBlocks
        from .modular_pipeline import FluxKontextModularPipeline, FluxModularPipeline
        from .transformer_memory import (
            clear_low_memory_flux_transformer_buffers,
            disable_low_memory_flux_transformer_buffers,
            enable_low_memory_flux_transformer_buffers,
        )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
