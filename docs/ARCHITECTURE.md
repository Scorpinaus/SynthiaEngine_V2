# SynthaEngine Architecture (Current)

This document describes how SynthaEngine is currently structured in this repository.

## 1. System Overview

SynthaEngine is a local image-generation system with:

- A FastAPI backend (`backend/main.py`) on `http://127.0.0.1:8000`
- A static frontend served from `frontend/` (default `http://127.0.0.1:4173`)
- A workflow-first execution model (`kind: "workflow"`) for all generation jobs

Runtime startup is defined in `run_app.bat`:

- Starts Uvicorn for `backend.main:app`
- Starts `python -m http.server` for `frontend/`
- Opens `sd15/text2img.html`

## 2. High-Level Component Diagram

```text
Browser (frontend/*.html + *.js)
  -> FastAPI (backend/main.py)
     -> Job Queue + Worker (backend/jobs/)
        -> Workflow Engine (backend/workflow/)
           -> Family Task Runtimes (backend/workflow/sd15.py, backend/workflow/sdxl.py, backend/workflow/flux.py, backend/workflow/qwen_image.py, backend/workflow/z_image.py, backend/workflow/ernie_image.py, backend/workflow/anima.py, plus WAN handlers)
              -> Diffusers Pipelines (backend/sd15/pipeline.py, backend/sdxl/pipeline.py, backend/flux/pipeline.py, backend/qwen_image/pipeline.py, backend/z_image/pipeline.py, backend/ernie_image/pipeline.py, backend/anima/pipeline.py, backend/wan/pipeline.py)
                 -> Shared Adapters (backend/adapters/)

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
     -> backend/jobs/ and backend/workflow/
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

Three current narrow support-layer edges remain visible until their owning
refactor tasks: adapters use workflow artifact/reference helpers, the preset
registry uses workflow input schemas, and utilities use model registry lookup.
They are allowed explicitly rather than being hidden as general exemptions.

`testing/test_architecture_contracts.py` enforces the static import boundaries,
the single composition root, the exact public route/method set, the public task
identifier set, the workflow envelope, and catalog derivation. Related runtime
contracts remain owned by focused tests:

- Job/task transitions and lease recovery: `testing/test_job_task_persistence.py`
  and `testing/test_job_worker_leases.py`.
- Workflow execution and artifact cleanup on success/failure:
  `testing/test_job_api.py`.
- Subprocess result/error propagation and child cleanup:
  `testing/test_flux_subprocess.py` plus the family subprocess suites.
- Pipeline hook and memory release: `testing/test_pipeline_lifecycle.py` plus
  family pipeline tests.

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

- `backend/jobs/queue.py`
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
- Jobs carry derived resource requirements; workers may filter claims by configured VRAM capacity
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
- Family modules such as `sd15.py`, `sdxl.py`, and `flux.py` own authoritative task definitions and family-specific input normalization.
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
- Flux, Qwen-Image, and Z-Image expose the core text2img/img2img/inpaint surface with LoRA and scheduler selection.
- Qwen-Image additionally exposes `true_cfg_scale`.
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

Common behavior across families:

- Resolve selected model from model registry
- Build and configure Diffusers pipeline(s)
- Run seeded generation loops
- Apply optional LoRA adapters
- Save PNG outputs under `outputs/batch_<batch_id>/...`
- Embed generation metadata in PNG text chunks

Family-specific runtime adapters in workflow layer:

- `backend/workflow/sd15.py`
- `backend/workflow/sdxl.py`
- `backend/workflow/flux.py`
- `backend/workflow/qwen_image.py`
- `backend/workflow/z_image.py`
- `backend/workflow/ernie_image.py`
- `backend/workflow/anima.py`

These modules normalize inputs and call the concrete pipeline functions.

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
- `frontend/components/header.js`, `frontend/components/nav_bar.js`
  - Shared navigation/header shell

### 4.2 Page Pattern

Generation pages (for example `frontend/sd15/text2img.html` + `frontend/sd15/text2img.js`) follow a common pattern:

1. Load model options (`GET /models?family=<family>`)
2. Optionally load defaults from workflow catalog
3. Read form inputs and build `payload.tasks`
4. Upload any required source images through `POST /api/artifacts`
5. Submit `POST /api/jobs` with `kind: "workflow"`
6. Watch `GET /api/jobs/{id}/events` via `EventSource`
7. Render output image URLs from `job.result.outputs`

### 4.3 Reusable UI Panels

- `frontend/components/controlnet_panel.js`, `frontend/components/controlnet_preprocessor.js`
  - ControlNet item management + preprocessor integration
- `frontend/components/lora_panel.js`
  - LoRA picker/weights mapped to workflow `lora_adapters`
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
- Static frontend with shared JS clients and per-workflow page scripts
- Localhost-only default startup, origin-restricted CORS, bounded artifact uploads, and loopback-only native path selection

## 7. Pipeline Lifecycle and Memory Policy

Generation pipelines are job-scoped by default: a task loads the pipeline it
needs, applies runtime options, generates outputs, and releases adapters, hooks,
pipeline references, and memory in a `finally` block.

The current serialized execution model protects local GPU memory, but it does
not replace explicit cleanup. Runtime changes should follow
`docs/PIPELINE_LIFECYCLE.md` before adding new pipeline loading, adapter, or
offload behavior.
