# Custom pipeline audit

Audit date: 12 July 2026

## Result

The folder now contains only code with a current runtime, conversion, or manual
benchmark purpose.

| Area | Why it remains |
|---|---|
| `Anima/` | The Anima runtime imports `anima_pipeline.py`. The other files are checkpoint conversion and documented experiment helpers. |
| `Flux/` | The production Flux runtime imports `memory.py`, `pipeline_flux.py`, `pipeline_flux_img2img.py`, and `pipeline_flux_inpaint.py`. |
| `FluxModular/` | Used by the low-memory Flux benchmark, tests, and `docs/FLUX_12GB_VRAM.md`. |

## Removed

- Six unused root-level Stable Diffusion pipeline experiments.
- Ten unused Flux ControlNet, Kontext, Fill, Control, and Redux pipeline copies.
- Four unused local Z-Image pipeline files. Production Z-Image uses the installed
  Diffusers pipelines directly.

Searches found no imports of the removed modules from backend, frontend, tests,
tools, or documentation. Remaining custom pipelines compile, and focused Flux,
Z-Image, Anima, and FluxModular tests pass.

## Upstream comparison

The 42 files present before cleanup were compared by SHA-256 with same-named
Python files in the installed Diffusers package. None were byte-identical.
Therefore no remaining file should be replaced with an upstream import based on
filename alone. Future upgrades should compare behavior or patches, not assume
that similar names mean identical implementations.

## Measurement

Custom pipeline physical lines decreased from 26,635 to 11,193. These lines are
reported separately from maintained application LOC.
