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
- Opens `sd15.html`

## 2. High-Level Component Diagram

```text
Browser (frontend/*.html + *.js)
  -> FastAPI (backend/main.py)
     -> Job Queue + Worker (backend/job_queue.py)
        -> Workflow Engine (backend/workflow.py)
           -> Family Task Runtimes (workflow_sd15.py, workflow_sdxl.py, workflow_flux.py, workflow_qwen_image.py, workflow_z_image.py)
              -> Diffusers Pipelines (sd15_pipeline.py, sdxl_pipeline.py, flux_pipeline.py, qwen_image_pipeline.py, z_image_pipeline.py)

Data/Artifacts:
- SQLite DBs in database/
- Generated files and uploaded artifacts in outputs/
```

## 3. Backend Architecture

### 3.1 API Layer (`backend/main.py`)

`backend/main.py` owns HTTP concerns and keeps core behavior in backend modules.

Major route groups:

- Health:
  - `GET /health`
- Workflow jobs:
  - `POST /api/jobs`
  - `GET /api/jobs/{job_id}`
  - `GET /api/jobs`
  - `POST /api/jobs/{job_id}/cancel`
  - `GET /api/jobs/{job_id}/events` (SSE)
- Workflow metadata:
  - `GET /api/workflow/task-types`
  - `GET /api/workflow/schema`
  - `GET /api/workflow/catalog`
- Artifacts:
  - `POST /api/artifacts`
  - Static serving via `app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")`
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
  - `GET /history`
  - `POST /create-blur-mask`

### 3.2 Job Queue and Worker

Core modules:

- `backend/job_queue.py`
- `backend/job_models.py`
- `backend/job_db.py`

Design details:

- Jobs are persisted in SQLite (`database/jobs.sqlite3` by default)
- Worker starts on FastAPI startup and stops on shutdown
- Queue uses status transitions (`queued -> running -> succeeded|failed|canceled`)
- `cancel_requested` supports best-effort cancellation for running jobs
- Workflow cancellation is enforced at task boundaries through `WorkflowContext.should_cancel`
- Idempotency is supported via `Idempotency-Key` header or body `idempotency_key`
- SSE endpoint polls job state and emits updates until terminal status
- Execution is effectively serialized:
  - DB uniqueness on one `running` row
  - `EXECUTION_LOCK` around execution path

### 3.3 Workflow Engine and Task Dispatch

Core modules:

- `backend/workflow.py`
- `backend/workflow_types.py`
- `backend/workflow_utility.py`
- `backend/workflow_schema_input.py`
- `backend/workflow_schema_output.py`
- `backend/workflow_catalog.py`

Responsibilities:

- Validate workflow payloads with Pydantic models (`WorkflowRequest`, typed task input models)
- Resolve runtime references (`@taskId.field`, `@artifact:...`) using `_resolve_refs`
- Dispatch tasks via `TASK_REGISTRY`
- Aggregate per-task outputs into final `outputs` result
- Track created artifact IDs and clean up ephemeral artifacts at job completion
- Expose schema/catalog for frontend/tooling generation support

Current task families in `TASK_REGISTRY`:

- SD1.5: `sd15.text2img`, `sd15.img2img`, `sd15.inpaint`, `sd15.controlnet.text2img`, `sd15.hires_fix`
- SDXL: `sdxl.text2img`, `sdxl.controlnet.text2img`, `sdxl.img2img`, `sdxl.inpaint`
- Flux: `flux.text2img`, `flux.img2img`, `flux.inpaint`
- Qwen-Image: `qwen-image.text2img`, `qwen-image.img2img`, `qwen-image.inpaint`
- Z-Image: `z-image.text2img`, `z-image.img2img`, `z-image.inpaint`
- Utility: `controlnet.preprocess`

### 3.4 Pipeline Runtime Layer

Primary pipeline modules:

- `backend/sd15_pipeline.py`
- `backend/sdxl_pipeline.py`
- `backend/flux_pipeline.py`
- `backend/qwen_image_pipeline.py`
- `backend/z_image_pipeline.py`

Common behavior across families:

- Resolve selected model from model registry
- Build and configure Diffusers pipeline(s)
- Run seeded generation loops
- Apply optional LoRA adapters
- Save PNG outputs under `outputs/batch_<batch_id>/...`
- Embed generation metadata in PNG text chunks

Family-specific runtime adapters in workflow layer:

- `backend/workflow_sd15.py`
- `backend/workflow_sdxl.py`
- `backend/workflow_flux.py`
- `backend/workflow_qwen_image.py`
- `backend/workflow_z_image.py`

These modules normalize inputs and call the concrete pipeline functions.

### 3.5 Persistence and Data Stores

- Job queue DB: `database/jobs.sqlite3`
- Model registry DB: `database/model_registry.sqlite3`
- LoRA registry DB: `database/lora_registry.sqlite3`
- Preset registry DB: `database/preset_registry.sqlite3`
- Generated outputs: `outputs/`
- Ephemeral uploaded artifacts: `outputs/artifacts/`

Migration pattern used in registries:

- DB-first persistence
- Optional one-time JSON bootstrap from legacy JSON files where present

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
- `frontend/header.js`, `frontend/nav_bar.js`
  - Shared navigation/header shell

### 4.2 Page Pattern

Generation pages (for example `frontend/sd15.html` + `frontend/sd15.js`) follow a common pattern:

1. Load model options (`GET /models?family=<family>`)
2. Optionally load defaults from workflow catalog
3. Read form inputs and build `payload.tasks`
4. Upload any required source images through `POST /api/artifacts`
5. Submit `POST /api/jobs` with `kind: "workflow"`
6. Watch `GET /api/jobs/{id}/events` via `EventSource`
7. Render output image URLs from `job.result.outputs`

### 4.3 Reusable UI Panels

- `controlnet_panel.js`, `controlnet_preprocessor.js`
  - ControlNet item management + preprocessor integration
- `lora_panel.js`
  - LoRA picker/weights mapped to workflow `lora_adapters`
- `preset_panel.js`
  - Save/apply presets through `/api/presets`
- `jobs_queue.js`
  - Poll recent jobs and trigger cancellation
- `history.js`
  - Read `/history` and render outputs grouped by `batch_id`

### 4.4 Registry and Utility Pages

- Base models: `models.html`, `model_base_add.html`, `model_base_edit.html`
- LoRAs: `lora_models.html`, `lora_add.html`, `lora_edit.html`
- Tools: `tools_analysis.html`

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

