# Pipeline Lifecycle and Memory Cleanup Policy

This document defines how generation pipelines should be loaded, used, released,
and cleaned up in SynthaEngine.

It is intended for maintainers who edit runtime generation code in:

- `backend/sd15/` runtime modules (with `pipeline.py` as the compatibility facade)
- `backend/sdxl/` runtime modules (with `pipeline.py` as the compatibility facade)
- `backend/flux/pipeline.py`
- `backend/qwen_image/pipeline.py`
- `backend/z_image/pipeline.py`

## Goals

- Keep local GPU memory predictable between jobs.
- Avoid leaking Diffusers offload hooks, adapter state, or large tensors.
- Keep behavior simple for the current local, serialized renderer model.
- Make future pipeline caching explicit instead of accidental.

## Current Policy

Pipelines are **job-scoped by default**.

That means a generation function should:

1. Load the required pipeline for the current task.
2. Apply scheduler, LoRA, IP-Adapter, ControlNet, and other runtime options.
3. Generate all images or videos for that task.
4. Save outputs and metadata.
5. Release adapters, hooks, pipeline references, and memory in a `finally` block.

Do not keep hidden module-level pipeline instances unless a future change adds an
explicit cache with ownership, eviction, and cleanup rules.

### Shared subprocess transport

`backend.utilities.subprocess_transport` owns the one-shot parent/child
protocol for every subprocess-backed family. The parent writes a typed request
envelope containing `operation` and `params`; the child writes either a success
envelope containing `result` or an error envelope containing the exception
type, message, and traceback. The shared serializer handles nested primitives,
PIL images, local paths, and runtime-profile dictionaries.

The transport also owns temporary artifacts, command construction, repository
working-directory selection, launch logging, exit-code and envelope validation,
and cleanup of the temporary directory. Family pipeline modules select the
runner, operation, launch gate, and public result shape. Family
`subprocess_runner.py` modules retain only their operation dispatch table,
generation functions, and final `cleanup_memory()` callback.

SD1.5 image renders run in a one-shot subprocess by default. The API worker
serializes the task parameters, launches `backend.sd15.subprocess_runner`, and
the child process loads Diffusers, runs inference, writes outputs, and exits.
This keeps the public workflow contract unchanged while letting process exit be
the final cleanup boundary for CUDA, VRAM, and large Python heap allocations.
SD1.5 subprocess launches are serialized with a default concurrency limit of
one per API worker process to avoid overlapping model loads and VRAM spikes.
SDXL image renders use the same one-shot subprocess pattern for text-to-image,
img2img, inpaint, and ControlNet variants. The public generation functions keep
their existing response shapes while child processes own Diffusers pipeline load,
inference, output writes, cleanup, and process exit.
ERNIE-Image text-to-image renders follow the same one-shot subprocess pattern
by default, with a serialized parent-side launch gate and child-side cleanup
before process exit.
Qwen-Image image renders use the same one-shot subprocess pattern for
text-to-image, img2img, and inpaint. The parent process serializes workflow
parameters, including PIL image inputs for image-guided tasks, and the child
process owns Diffusers pipeline load, inference, output writes, cleanup, and
process exit.
Flux image renders use the same one-shot subprocess pattern for text-to-image,
img2img, and inpaint. The parent process serializes workflow parameters,
including PIL image inputs for image-guided tasks, and the child process owns
Diffusers pipeline load, inference, output writes, cleanup, and process exit.
Z-Image image renders use the same one-shot subprocess pattern for text-to-image,
img2img, and inpaint. The parent process serializes workflow parameters,
including PIL image inputs for image-guided tasks, and the child process owns
Diffusers pipeline load, inference, output writes, cleanup, and process exit.
WAN video renders use the same one-shot subprocess pattern for text-to-video,
VACE, and image-to-video. The parent process serializes workflow parameters,
including PIL image inputs and local conditioning video paths, and the child
process owns Diffusers pipeline load, inference, video export, cleanup, and
process exit.

## Required Cleanup Order

Every generation function that creates a Diffusers pipeline should use this
cleanup order in `finally`:

1. Clean feature-specific runtime state.
   - Unload LoRA adapters.
   - Clean IP-Adapter state.
   - Clear temporary adapter names or references.
2. Release Diffusers hooks.
   - Call `maybe_free_model_hooks()` when present.
   - Call `remove_all_hooks()` when present.
3. Drop strong references.
   - Set `pipe = None`.
   - Delete large local tensors or images when they are no longer needed.
4. Run memory cleanup.
   - Call `cleanup_memory()` from `backend.utilities.pipeline`.

Cleanup must be best-effort. A cleanup failure should be logged, but it should
not hide the original generation error.

## Shared Cleanup Helper

The preferred long-term shape is a single helper in `backend.utilities.pipeline`
or a nearby runtime utility, for example:

```python
def release_pipeline(pipe: object | None) -> None:
    if pipe is None:
        return
    if hasattr(pipe, "maybe_free_model_hooks"):
        pipe.maybe_free_model_hooks()
    if hasattr(pipe, "remove_all_hooks"):
        pipe.remove_all_hooks()
    cleanup_memory()
```

Family modules may wrap that helper when they need family-specific cleanup, but
the core hook release and cache cleanup should not be duplicated indefinitely.

## Memory Saver Policy

Use Diffusers memory-saving features deliberately:

- SD1.5 and SDXL pipelines that run directly on CUDA should clean up after every
  task unless an explicit cache is introduced.
- Large transformer families such as Flux, Qwen-Image, and Z-Image should prefer
  Diffusers offload features when supported.
- VAE slicing and VAE tiling are acceptable defaults for large image pipelines
  when supported by the pipeline or VAE.
- If a pipeline uses `enable_sequential_cpu_offload()` or another offload mode,
  hook release is required at the end of the task.

## Adapter Policy

Adapters are task-scoped.

- LoRA adapters must be unloaded before releasing the pipeline.
- IP-Adapter state must be cleaned before releasing the pipeline.
- A generation task must not leave adapter weights active for the next task.
- If adapter cleanup is not supported by a specific pipeline class, log that
  limitation and still release hooks and memory.

## Error Handling

Generation functions must release runtime resources even when:

- image generation fails,
- validation raises after pipeline load,
- saving an output fails,
- cancellation interrupts the workflow at a task boundary.

Use `try`/`finally` around pipeline usage. Avoid cleanup only on the success path.

## Current Family Status

- SD1.5: subprocess-backed for image renders. Operation modules enter a
  release-guarded `try/finally` immediately after a successful pipeline load;
  process exit provides the final memory boundary.
- SDXL: subprocess-backed for image renders. Operation modules own task-scoped
  cleanup in `finally`, and process exit provides the final memory boundary.
- Flux: subprocess-backed for text2img, img2img, and inpaint. The child process
  still runs task-scoped pipeline release, runs final memory cleanup in the
  subprocess runner, and then exits.
- Qwen-Image: subprocess-backed for text2img, img2img, and inpaint. The child
  process still runs task-scoped pipeline release, runs final memory cleanup in
  the subprocess runner, and then exits.
- Z-Image: subprocess-backed for text2img, img2img, and inpaint. The child
  process still runs task-scoped cleanup, and process exit provides the final
  memory boundary.
- ERNIE-Image: subprocess-backed for text-to-image renders by default. The child
  process releases its pipeline in a task-scoped `finally`, runs final memory
  cleanup in the subprocess runner, and then exits.
- WAN: subprocess-backed for text2video, VACE, and image2video. The child
  process still runs task-scoped pipeline release, runs final memory cleanup in
  the subprocess runner, and then exits.
- Anima: subprocess-backed for text-to-image renders by default. The child runs
  task-scoped pipeline release and final memory cleanup before process exit.

## Explicit Pipeline Cache

Pipeline caching is opt-in. The default remains isolated, one-shot subprocess
execution. Set both `SYNTHA_PIPELINE_CACHE_MAX_ENTRIES` and
`SYNTHA_PIPELINE_CACHE_MAX_MB` to positive values to enable the worker-local
cache. Flux is the first integrated family; cached Flux tasks execute inside the
persistent render worker.

The cache defines:

- key: family, operation, model, and device/offload strategy;
- ownership: the renderer process owns cached pipelines;
- eviction: LRU under both entry and estimated-memory budgets;
- adapter isolation: task LoRAs are unloaded in `finally` before reuse;
- failure isolation: generation or adapter-cleanup failures invalidate and release the cached instance;
- cleanup: eviction and renderer shutdown call the standard pipeline release helper;
- observability: cache hit/miss/eviction state is included in `result.profile.pipeline_caches`.

`SYNTHA_FLUX_PIPELINE_ESTIMATED_MB` controls the estimated cost assigned to
each Flux entry (default 12000 MB). Values larger than the cache budget are
loaded for the current task and released without caching.

## Maintenance Checklist

When editing a pipeline generation function:

- Is the pipeline created inside the function or explicitly acquired from a
  documented cache?
- Is all adapter state unloaded in `finally`?
- Are Diffusers hooks released in `finally`?
- Is `cleanup_memory()` called after releasing the pipeline?
- Are large intermediate tensors moved to CPU or deleted when appropriate?
- Does the change preserve workflow output shapes documented in
  `docs/WORKFLOW_API.md`?
- Are focused tests updated when behavior changes?
