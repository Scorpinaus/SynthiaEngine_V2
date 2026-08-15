# SynthaEngine Architecture (Current)

This document describes how SynthaEngine is currently structured in this repository.

## 1. System Overview

SynthaEngine is a local image-generation system with:

- A FastAPI backend (`backend/main.py`) on `http://127.0.0.1:8000`
- A static frontend served from `frontend/` (default `http://127.0.0.1:4173`)
- A workflow-first execution model (`kind: "workflow"`) for all generation jobs

Runtime startup is defined in `run_app.bat`:

- Starts the API process with its embedded renderer disabled
- Starts a separate renderer process with `python -m backend.jobs.render_worker`
- Starts `python -m http.server` for `frontend/`
- Opens `sd15/text2img.html` after the services start

## 2. High-Level Component Diagram

```text
Browser (frontend/*.html + *.js)
  -> FastAPI (backend/main.py)
     -> API Routers (backend/api/)
        -> Job Store (backend/jobs/store.py)

Renderer (backend/jobs/render_worker.py)
  -> Job Subsystem Composition (backend/jobs/queue.py)
     -> Job Store (backend/jobs/store.py)
     -> Worker Polling (backend/jobs/worker.py)
     -> Injected Job Execution (backend/jobs/execution.py)
        -> Workflow Assembly + Engine (backend/workflow/)
           -> Family Task Adapters (backend/workflow/<family>.py)
              -> Family Runtimes (backend/<family>/)
                 -> Shared Adapters, Registries, and Utilities

Data/Artifacts:
- SQLite DBs in database/
- Generated files and uploaded artifacts in outputs/
```

### 2.1 Dependency Direction and Contract Guards

Backend dependencies should point from orchestration toward implementation:

```text
backend/main.py
  -> backend/settings.py
  -> backend/api/
     -> backend/jobs/ and workflow contract views

backend/jobs/render_worker.py
  -> backend/jobs/queue.py
     -> backend/jobs/store.py
     -> backend/jobs/worker.py
     -> backend/jobs/execution.py
        -> backend/workflow/
           -> family runtimes
              -> adapters, LoRA/registries, and utilities
```

The detailed rules are intentionally conservative:

- `backend/main.py` is the only FastAPI application composition root.
- API modules may call jobs, workflow, registries, LoRA, adapters, and shared
  utilities, but not concrete model-family runtimes.
- Jobs may call workflow and shared utilities, but not API modules or concrete
  model-family runtimes.
- Workflow may dispatch to adapters and family runtimes, but it must not depend
  on API or job orchestration.
- Model-family runtimes may use adapters, LoRA, registries, and utilities. They
  must not import API, jobs, workflow, or another model family.
- Supporting adapters, registries, LoRA, and utilities must not depend on API,
  jobs, or concrete model-family runtimes.

Two narrow support-layer exceptions are still explicit:

- `backend/registries/preset.py` reads the assembled workflow input models.
- `backend/utilities/pipeline.py` uses the model registry contract.

These edges are deferred. The static rules do not give a general exemption to
the full layer.

`testing/test_architecture_contracts.py` enforces the static import boundaries,
the single composition root, API-router ownership, the exact public
route/method set, the public task identifier set, the workflow envelope, and
catalog derivation. Related contracts remain in focused tests:

- Job/task transitions and lease recovery: `testing/test_job_task_persistence.py`
  and `testing/test_job_worker_leases.py`.
- Workflow execution and artifact cleanup on success/failure:
  `testing/test_job_api.py`.
- Subprocess result/error propagation and child cleanup:
  `testing/test_subprocess_transport.py` plus the family subprocess suites.
- Pipeline hook and memory release: `testing/test_pipeline_lifecycle.py` plus
  family pipeline tests.
- SD1.5 and SDXL runtime ownership: `testing/test_arc06_decomposition.py`.
- Shared page composition: `testing/test_frontend_arc07.py`.
- Stable style entrypoint and layer ownership: `testing/test_frontend_styles.py`.

## 3. Backend Architecture

### 3.1 API Composition and Routers

`backend/main.py` is the application composition root. Its `create_app()`
factory owns settings selection, logging setup, middleware, static mounts,
router inclusion, health, and lifespan/queue wiring while preserving the
`backend.main:app` startup target. Endpoint behavior lives under `backend/api/`:

- `artifacts.py`: bounded image/video uploads, backed by `backend/artifacts.py`
- `local_paths.py`: loopback-protected native path selection
- `controlnet.py`: preprocessor catalog and preprocessing
- `model_analysis.py`: temporary-file model inspection
- `masks.py`: inpainting mask utilities
- `jobs.py`, `workflow.py`, `history.py`, `models.py`, `loras.py`, and
  `presets.py`: their named HTTP domains

`backend/settings.py` parses process configuration into frozen typed settings.
It resolves the repository root, outputs/database paths, CORS origins, upload
limits, embedded worker and path-picker policy, logging role, worker capacity,
and shared pipeline-cache budgets. Model-family tuning remains in the owning
runtime module. `backend/config.py` retains compatibility constants without
creating directories when merely imported; directory creation is explicit in
application startup and persistence services.

Major route groups:

- Health:
  - `GET /health`
- Workflow jobs:
  - `POST /api/jobs`
  - `GET /api/jobs/{job_id}`
  - `GET /api/jobs/{job_id}/tasks`
  - `GET /api/jobs`
  - `POST /api/jobs/{job_id}/cancel`
  - `GET /api/jobs/{job_id}/events` (SSE)
- Workflow metadata:
  - `GET /api/workflow/task-types`
  - `GET /api/workflow/schema`
  - `GET /api/workflow/catalog`
- Artifacts:
  - `POST /api/artifacts`
  - Static serving via the configured output directory at `/outputs`
- Registries:
  - Models: `/models` and `/models/{model_name}`
  - LoRAs: `/lora-models` and `/lora-models/{lora_id}`
  - Presets: `/api/presets` and `/api/presets/{preset_id}`
- ControlNet preprocessors:
  - `GET /api/controlnet/preprocessors`
  - `GET /api/controlnet/preprocessor-models`
  - `POST /api/controlnet/preprocess`
- Utility/history:
  - `POST /api/tools/analyze-model`
    - Returns layer rows plus best-effort architecture detection for LoRA/model files, including whether safetensors metadata was present.
  - `GET /history`
  - `POST /create-blur-mask`

### 3.2 Job Queue and Worker

Core modules:

- `backend/jobs/contracts.py`: typed claim, persistence, execution, cancellation,
  and cleanup boundaries
- `backend/jobs/store.py`: SQLite job/task persistence, leases, recovery, and
  state transitions
- `backend/jobs/execution.py`: workflow execution, progress/profile reporting,
  cancellation checks, and artifact cleanup
- `backend/jobs/worker.py`: polling, heartbeat, and terminal-state orchestration
- `backend/jobs/queue.py`: compatibility exports and subsystem composition
- `backend/jobs/models.py`
- `backend/jobs/db.py`
- `backend/jobs/render_worker.py`

Design details:

- Jobs are persisted in SQLite (`database/jobs.sqlite3` by default)
- SQLite uses WAL mode and a busy timeout; idle workers exponentially back off polling up to the configured maximum
- Declared workflow steps and their status, timing, inputs, outputs, and errors are persisted in `job_tasks`
- Worker can run embedded in FastAPI or as the separate renderer process used by `run_app.bat`
- Queue uses status transitions (`queued -> running -> succeeded|failed|canceled`)
- `cancel_requested` supports best-effort cancellation for running jobs
- Workflow cancellation is enforced at task boundaries through `WorkflowContext.should_cancel`
- Idempotency is supported via `Idempotency-Key` header or body `idempotency_key`
- SSE endpoint polls job state and emits updates until terminal status
- Each claimed job has a worker-owned renewable lease and heartbeat; only expired leases are recovered after a renderer crash
- Terminal updates require the same worker ownership as the active lease
- Jobs carry derived resource requirements; workers may filter claims by configured VRAM capacity
- Persistence and lease tests load only the store; execution and polling tests
  use fake stores/executors without SQLite or workflow runtimes
- Artifact cleanup runs once after execution. If rendering and cleanup both
  fail, the rendering error remains the terminal failure and the cleanup error
  is logged; cleanup-only failures fail the job
- Execution is effectively serialized:
  - DB uniqueness on one `running` row
  - `EXECUTION_LOCK` around execution path

### 3.3 Workflow Engine and Task Dispatch

Core modules:

- `backend/workflow/__init__.py`
- `backend/workflow/assembly.py`
- `backend/workflow/engine.py`
- `backend/workflow/registry.py`
- `backend/workflow/types.py`
- `backend/workflow/utility.py`
- `backend/workflow/schema_input.py`
- `backend/workflow/schema_output.py`
- `backend/workflow/catalog.py`

Ownership:

- `schema_input.py`, `schema_output.py`, and `types.py` own the Pydantic workflow contracts.
- Family facades such as `sd15.py` and `sdxl.py` compose operation-specific task
  adapters; other family modules such as `flux.py` own authoritative task
  definitions and family-specific input normalization directly.
- `assembly.py` binds concrete runtime dependencies, merges family registrations with duplicate detection, and derives the compatibility registry/schema views.
- `engine.py` validates workflow envelopes, preflights references and cycles, derives stable execution order, resolves references, dispatches authoritative definitions, publishes progress, and aggregates results.
- `catalog.py` derives schema, defaults, UI hints, and capabilities from the assembled task definitions; it is not a second registry.
- `utility.py` owns artifact/reference helpers; the job layer invokes cleanup after workflow completion.
- `__init__.py` contains only explicit public compatibility exports. Internal callers import the module that owns each symbol.

Current task families in `TASK_REGISTRY`:

- SD1.5: `sd15.ip_adapter.encode`, `sd15.text2img`, `sd15.animatediff.text2video`, `sd15.img2img`, `sd15.inpaint`, `sd15.controlnet.text2img`, `sd15.hires_fix`
- SDXL: `sdxl.ip_adapter.encode`, `sdxl.text2img`, `sdxl.controlnet.text2img`, `sdxl.img2img`, `sdxl.inpaint`
- WAN: `wan.text2video`, `wan.image2video`
- Flux: `flux.text2img`, `flux.img2img`, `flux.inpaint`
- Qwen-Image: `qwen-image.text2img`, `qwen-image.img2img`, `qwen-image.inpaint`
- Z-Image: `z-image.text2img`, `z-image.img2img`, `z-image.inpaint`
- ERNIE-Image: `ernie-image.text2img`
- Anima: `anima.text2img`
- Utility: `controlnet.preprocess`

The assembled task definitions are the source of truth for supported task
identifiers, runtime dispatch, and the generated workflow catalog.
`docs/WORKFLOW_API.md` documents the public contract and feature-combination
notes in more detail.

### 3.3.1 Feature Surface by Model Family

This matrix summarizes the current generation surface exposed by workflow
tasks. The same information is available to clients at
`GET /api/workflow/catalog` under `capabilities`.

| Family | text2img | text2video | img2img | inpaint | ControlNet | Hi-Res Fix | LoRA | IP-Adapter | true CFG |
|---|---|---|---|---|---|---|---|---|---|
| `sd15` | yes | yes | yes | yes | yes | yes | yes | yes | no |
| `sdxl` | yes | no | yes | yes | yes | no | yes | yes | no |
| `wan` | no | yes | no | no | no | no | no | no | no |
| `flux` | yes | no | yes | yes | no | no | yes | no | no |
| `qwen-image` | yes | no | yes | yes | no | no | yes | no | yes |
| `z-image` | yes | no | yes | yes | no | no | yes | no | no |
| `ernie-image` | yes | no | no | no | no | no | yes | no | no |
| `anima` | yes | no | no | no | no | no | no | no | no |

Important family-specific boundaries:

- SD1.5 has the widest surface: AnimateDiff text-to-video, ControlNet and multi-ControlNet, Hi-Res Fix, LoRA, IP-Adapter, and LCM mode.
- SDXL supports ControlNet, multi-ControlNet, LoRA, and IP-Adapter across the image tasks, but IP-Adapter and ControlNet combinations are intentionally rejected for img2img/inpaint.
- Flux and Z-Image expose the core text2img/img2img/inpaint surface with LoRA and scheduler selection.
- Qwen-Image exposes text2img, img2img, inpaint, `true_cfg_scale`, and
  transformer-only LoRA. The current SDNQ profile fixes the scheduler to Flow
  Match Euler.
- ERNIE-Image and Anima start with text-to-image only; Anima loads SynthaEngine's local community Diffusers pipeline while keeping `trust_remote_code=True` for custom model components.

### 3.4 Pipeline Runtime Layer

Primary pipeline modules:

- `backend/sd15/pipeline.py`
- `backend/sd15/animatediff_pipeline.py`
- `backend/sd15/ip_adapter_pipeline.py`
- `backend/sdxl/pipeline.py`
- `backend/sdxl/ip_adapter_pipeline.py`
- `backend/flux/pipeline.py`
- `backend/qwen_image/pipeline.py`
- `backend/z_image/pipeline.py`
- `backend/ernie_image/pipeline.py`
- `backend/anima/pipeline.py`

The SD1.5 and SDXL `pipeline.py` modules are compatibility facades. Their
runtime ownership is intentionally split as follows:

- `runtime_common.py`: shared imports, constants, and runtime dependencies
- `loaders.py`: Diffusers pipeline factories
- `preparation.py`: prompt/image/latent preparation and render-call helpers
- `adapters.py`: task-scoped LoRA/IP-Adapter policy and cleanup
- `transport.py`: stable one-shot subprocess entrypoints
- `text2img.py`, `img2img.py`, and `inpaint.py`: operation-specific generation
- SD1.5 `hires_fix.py`: Hi-Res Fix generation
- SDXL `controlnet.py` and `results.py`: ControlNet operations and result saving

Stable parent-process entrypoints continue to accept one parameter dictionary:
SD1.5 `generate_images*` returns relative output paths, while SDXL
`generate_text2img`, `generate_img2img`, `generate_inpaint`, and their
ControlNet variants return the existing image-result object. Their matching
`*_in_process` entrypoints retain the child-process call signatures. SD1.5
`run_sd15_hires_fix` retains its keyword-only operation contract. Architecture
tests compare the facade parameter surface with the owning implementation.

Common behavior across families:

- Resolve selected model from model registry
- Build and configure Diffusers pipeline(s)
- Run seeded generation loops
- Apply optional LoRA adapters
- Save PNG outputs under `outputs/batch_<batch_id>/...`
- Embed generation metadata in PNG text chunks

Family-specific runtime adapters in workflow layer:

- `backend/workflow/sd15.py` plus `sd15_*_task.py` operation adapters
- `backend/workflow/sdxl.py` plus `sdxl_*_task.py` operation adapters
- `backend/workflow/flux.py`
- `backend/workflow/qwen_image.py`
- `backend/workflow/z_image.py`
- `backend/workflow/ernie_image.py`
- `backend/workflow/anima.py`

The facades preserve public imports and task identifiers; the focused operation
modules normalize inputs and call the concrete pipeline functions.

### 3.5 Shared Adapter Infrastructure

Core modules:

- `backend/adapters/controlnet_preprocessors.py`
- `backend/adapters/controlnet_preprocessor_registry.py`
- `backend/adapters/controlnet_preprocessor_registry.json`
- `backend/adapters/ip_adapter.py`
- `backend/adapters/ip_adapter_embeds.py`

Responsibilities:

- Register and run ControlNet preprocessors used by API and workflow tasks
- Manage shared Diffusers IP-Adapter lifecycle helpers
- Save and validate temporary IP-Adapter image-embedding artifacts


### 3.6 Persistence and Data Stores

- Job queue DB: `database/jobs.sqlite3`
- Model registry DB: `database/model_registry.sqlite3`
- LoRA registry DB: `database/lora_registry.sqlite3`
- Preset registry DB: `database/preset_registry.sqlite3`
- Generated outputs: `outputs/`
- Ephemeral uploaded artifacts: `outputs/artifacts/`

Migration pattern used in registries:

- DB-first persistence
- Optional one-time JSON bootstrap from legacy JSON files where present

### 3.7 Model and Preset Registries

Core modules:

- `backend/registries/model.py`
- `backend/registries/model_registry.json`
- `backend/registries/preset.py`

Responsibilities:

- Persist and validate base model entries for `/models`
- Bootstrap base models from the JSON sidecar when the SQLite registry is empty
- Persist and validate saved generation presets for `/api/presets`

### 3.8 LoRA Registry and Runtime Helpers

Core modules:

- `backend/lora/registry.py`
- `backend/lora/lora_registry.json`
- `backend/lora/utils.py`

Responsibilities:

- Persist and validate LoRA registry entries for `/lora-models`
- Bootstrap from the JSON sidecar when the SQLite registry is empty
- Apply selected LoRA adapters and write coverage reports during generation
- For Qwen-Image, require registry family `qwen-image`, type `lora`, and the
  whole transformer target. The runtime loads and activates all selected
  adapters once for a request and writes transformer coverage.
- The Qwen-Image runtime unloads requested adapters in `finally`. It then calls
  `release_pipeline`, including after load failure, generation failure,
  cancellation, or unload failure.

### 3.9 Shared Utilities

Core modules:

- `backend/utilities/logging.py`
- `backend/utilities/model_analysis.py`
- `backend/utilities/pipeline.py`
- `backend/utilities/pipeline_layer_logging.py`
- `backend/utilities/prompt.py`
- `backend/utilities/resource_logging.py`
- `backend/utilities/schedulers.py`

Responsibilities:

- Configure process logging
- Analyze model files for registry/tool endpoints
- Provide shared pipeline output, model-source, and cleanup helpers
- Parse prompt weighting and build weighted prompt embeddings
- Delegate ordinary short, unweighted prompts to Diffusers; use custom embeddings for clip-skip, prompt weighting, long prompts, or explicit LoRA text-encoder scaling
- Resolve CLIP final LayerNorm across both flattened and nested Transformers model layouts before using clip-skip hidden states
- Create Diffusers scheduler instances from user-facing scheduler ids
- Capture pipeline layer and runtime resource diagnostics

## 4. Frontend Architecture

Frontend is plain HTML + JS (no build step) served by static HTTP server.

### 4.1 Shared Frontend Runtime Modules

- `frontend/api_config.js`
  - Defines `API_BASE` (default `http://127.0.0.1:8000`)
- `frontend/workflow_client.js`
  - Upload artifacts
  - Submit workflow jobs
  - Subscribe to SSE job events
  - Input parsing helpers
- `frontend/workflow_catalog.js`
  - Fetch/caches `/api/workflow/catalog`
  - Applies backend defaults into form controls
- `frontend/generation_page.js`
  - Owns shared form, model, preset, job, SSE, and image-result mechanics
- `frontend/sd15/generation_controller.js`,
  `frontend/sdxl/generation_controller.js`
  - Compose family inputs, task names, and supported feature combinations
- `frontend/components/header.js`, `frontend/components/nav_bar.js`
  - Shared navigation/header shell

### 4.2 Page Pattern

Generation pages use small entrypoint scripts. Each entrypoint declares its
task name, input fields, defaults, and feature controllers. The shared page and
family controllers do the repeated work:

1. Load model options (`GET /models?family=<family>`)
2. Load defaults from the workflow catalog
3. Read form inputs and build `payload.tasks` in a family controller
4. Upload any required source images through `POST /api/artifacts`
5. Submit `POST /api/jobs` with `kind: "workflow"`
6. Watch `GET /api/jobs/{id}/events` via `EventSource`
7. Render output image URLs from `job.result.outputs`

### 4.3 Reusable Feature Controllers and UI Panels

- `frontend/components/adapter_controller.js`
  - Coordinates the combined adapter status and summary
- `frontend/components/controlnet_controller.js`
  - Adds SD1.5 or SDXL ControlNet inputs to a task
- `frontend/components/ip_adapter_controller.js`
  - Adds direct or encoded IP-Adapter inputs
- `frontend/components/inpaint_editor.js`
  - Owns the base image and saved mask contract
- `frontend/components/animatediff_controller.js`
  - Builds the AnimateDiff video task and renders video results

- `frontend/components/controlnet_panel.js`, `frontend/components/controlnet_preprocessor.js`
  - ControlNet item management + preprocessor integration
- `frontend/components/lora_panel.js`
  - LoRA picker/weights mapped to workflow `lora_adapters`; Qwen-Image uses one
    transformer strength control and no UNet/text-encoder target control
- `frontend/components/preset_panel.js`
  - Save/apply presets through `/api/presets`
- `frontend/components/jobs_queue.js`
  - Poll recent jobs and trigger cancellation
- `frontend/components/gallery.js`, `frontend/components/video_gallery.js`
  - Shared image/video viewers used by workflow pages
- `others/history.js`
  - Read `/history` and render outputs grouped by `batch_id`

### 4.4 Registry and Utility Pages

- Base models: `models/base/registry.html`, `models/base/add.html`, `models/base/edit.html`
- LoRAs: `models/lora/model_page.html`, `models/lora/add.html`, `models/lora/edit.html`
- Tools: `others/tools_analysis.html`

These pages call model/LoRA registry endpoints and tool endpoints directly.

### 4.5 Style Layers

All pages load `frontend/style.css`. This stable entrypoint imports the files in
`frontend/styles/` in this order: tokens, base, layout, components, generation,
registry/tools, and responsive rules. The responsive layer stays last. HTML
pages must not load a layer directly. See `frontend/styles/README.md` for the
rule ownership map.

## 5. End-to-End Request Flow (Current)

Example: text-to-image from UI

1. Browser page builds workflow payload
2. Browser sends `POST /api/jobs` (`kind: "workflow"`)
3. API stores job in DB as `queued`
4. Worker claims job and marks `running`
5. Worker calls `execute_workflow(...)`
6. Workflow task handler validates inputs and runs family pipeline
7. Pipeline saves PNG files into `outputs/batch_<batch_id>/...`
8. Workflow returns `outputs` and `tasks` payload
9. Worker writes final job result and marks `succeeded`
10. Browser receives terminal SSE event and renders returned image URLs

If artifact inputs are used:

- Browser uploads images first to `POST /api/artifacts`
- Workflow references those artifact IDs
- Worker runs cleanup of referenced/created artifacts after execution

## 6. Key Design Characteristics

- Workflow-first API surface for all generation families
- Thin API handlers, core behavior in backend modules
- Schema-driven task contracts exposed to frontend/tooling
- Queue-backed execution with idempotency and cancellation support
- Local-file output model with metadata-rich PNG artifacts
- Static frontend with shared page mechanics, feature controllers, and small
  family entrypoints
- Stable CSS entrypoint with explicit style layers and no build step
- Localhost-only default startup, origin-restricted CORS, bounded artifact uploads, and loopback-only native path selection

## 7. Pipeline Lifecycle and Memory Policy

Generation pipelines are job-scoped by default: a task loads the pipeline it
needs, applies runtime options, generates outputs, and releases adapters, hooks,
pipeline references, and memory in a `finally` block.

The current serialized execution model protects local GPU memory, but it does
not replace explicit cleanup. Runtime changes should follow
`docs/PIPELINE_LIFECYCLE.md` before adding new pipeline loading, adapter, or
offload behavior.
