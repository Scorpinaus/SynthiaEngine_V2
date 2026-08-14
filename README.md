# SynthaEngine

SynthaEngine is a local **image generation server + lightweight web UI** built around a single, consistent concept: **submit a workflow job** and poll/stream results. Under the hood it uses Hugging Face **Diffusers/Transformers** pipelines and exposes them via a **FastAPI** backend.

It's designed to make it easy to:
- Run multiple model families behind one API surface (SD1.5, SDXL, Flux, Qwen-Image, Z-Image).
- Chain steps together (e.g., preprocess -> generate) using workflow tasks and runtime references.
- Manage long-running work with a job queue, status endpoints, and SSE events.

## What's in this repo

- `backend/`: FastAPI app (`backend/main.py`) + workflow/task package (`backend/workflow/`) + model/LoRA registries.
- `frontend/`: static HTML/JS pages for common workflows (SD1.5 / SDXL / Flux / Qwen-Image / Z-Image).
- `docs/WORKFLOW_API.md`: the current "workflow-only" API contract (v2).
- `outputs/`: generated images + uploaded artifacts (ephemeral).
- `database/`: sqlite databases (default jobs DB at `database/jobs.sqlite3`).

## Core concepts

### 1) Workflow jobs (single entrypoint)

All generation is submitted as a job with `kind: "workflow"`:
- `POST /api/jobs` submits work
- `GET /api/jobs/{job_id}` polls status/results
- `GET /api/jobs/{job_id}/events` streams status via SSE

Work is expressed as an ordered list of `tasks`:

```json
{
  "kind": "workflow",
  "payload": {
    "tasks": [
      { "id": "t1", "type": "sd15.text2img", "inputs": { "prompt": "..." } }
    ],
    "return": "@t1.images"
  }
}
```

### 2) Artifacts for image inputs

If a task needs an image input (img2img/inpaint), upload it first:
- `POST /api/artifacts` (multipart form field: `file`)
- Use the returned `artifact_id` in task inputs

Artifacts are **ephemeral** and are cleaned up when the workflow finishes.

### 3) Task types

Current task types are documented in `docs/WORKFLOW_API.md` and implemented in `backend/workflow/`:
- SD1.5: `sd15.ip_adapter.encode`, `sd15.text2img`, `sd15.animatediff.text2video`, `sd15.img2img`, `sd15.inpaint`, `sd15.controlnet.text2img`, `sd15.hires_fix`
- SDXL: `sdxl.ip_adapter.encode`, `sdxl.text2img`, `sdxl.controlnet.text2img`, `sdxl.img2img`, `sdxl.inpaint`
- WAN: `wan.text2video`, `wan.image2video`
- Flux: `flux.text2img`, `flux.img2img`, `flux.inpaint`
- Qwen-Image: `qwen-image.text2img`, `qwen-image.img2img`, `qwen-image.inpaint`
- Z-Image: `z-image.text2img`, `z-image.img2img`, `z-image.inpaint`
- ERNIE-Image: `ernie-image.text2img`
- Anima: `anima.text2img`
- Utility: `controlnet.preprocess`

### 4) Model family capabilities

The machine-readable source for this matrix is `GET /api/workflow/catalog`.

| Family | text2img | text2video | img2img | inpaint | ControlNet | Hi-Res Fix | LoRA | IP-Adapter | true CFG |
|---|---|---|---|---|---|---|---|---|---|
| `sd15` | yes | yes | yes | yes | yes | yes | yes | yes | no |
| `sdxl` | yes | no | yes | yes | yes | no | yes | yes | no |
| `wan` | no | yes | no | no | no | no | no | no | no |
| `flux` | yes | no | yes | yes | no | no | yes | no | no |
| `qwen-image` | yes | no | yes | yes | no | no | no | no | yes |
| `z-image` | yes | no | yes | yes | no | no | yes | no | no |
| `ernie-image` | yes | no | no | no | no | no | yes | no | no |
| `anima` | yes | no | no | no | no | no | no | no | no |

## Quickstart (Windows)

### Prereqs

- Python **3.10+**
- A PyTorch build for your CPU or CUDA platform
- The Python packages in `requirements.txt`

### Run

1) Create a virtualenv at `.venv` and install runtime dependencies from
   `requirements.txt`. Contributors can use `requirements-dev.txt`. Both use
   the tested direct-dependency pins in `constraints.txt`; install the
   platform-appropriate PyTorch/CUDA build separately.

```bat
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

2) Start the app:
```bat
run_app.bat
```

This command launches three processes:

- Backend API: `http://127.0.0.1:8000`
- Renderer worker: a separate command window for generation/job logs
- Frontend: `http://127.0.0.1:4173` (opens `sd15/text2img.html` by default)

The API and renderer are split so request/access logs stay separate from
generation warnings and errors. The batch file disables the API's embedded
renderer before it starts the separate renderer.

You can start one process at a time for diagnosis:

```bat
run_app.bat api
run_app.bat render
run_app.bat frontend
```

To start the same processes without the batch-file modes, run the API with the
embedded renderer disabled:

```bat
set SYNTHA_LOG_ROLE=api
set SYNTHA_API_START_WORKER=0
.venv\Scripts\python.exe -m uvicorn backend.main:app --workers 1 --host 127.0.0.1 --port 8000
```

Then start the renderer in another terminal:

```bat
set SYNTHA_LOG_ROLE=render
.venv\Scripts\python.exe -m backend.jobs.render_worker
```

Start the frontend in a third terminal:

```bat
.venv\Scripts\python.exe -m http.server 4173 --directory frontend
```

Use only one renderer process. Stop all three processes when you close the app.

Process configuration is parsed centrally by `backend/settings.py`. Defaults
are repository-relative and can be overridden before startup:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SYNTHA_OUTPUT_DIR` | `outputs` | Generated and uploaded artifact directory |
| `SYNTHA_DATABASE_DIR` | `database` | SQLite state directory |
| `SYNTHA_CORS_ORIGINS` | local frontend origins | Comma-separated allowed browser origins |
| `SYNTHA_MAX_UPLOAD_BYTES` | `104857600` | Artifact upload byte limit |
| `SYNTHA_MAX_IMAGE_PIXELS` | `67108864` | Decoded artifact image pixel limit |
| `SYNTHA_API_START_WORKER` | `1` | Enable the embedded API renderer |
| `SYNTHA_ALLOW_REMOTE_PATH_PICKER` | `0` | Allow non-loopback clients to open the host picker |
| `SYNTHA_LOG_ROLE` | process default | Role included in log records |
| `SYNTHA_WORKER_VRAM_MB` | `0` | Worker capacity filter (`0` disables filtering) |
| `SYNTHA_PIPELINE_CACHE_MAX_ENTRIES` | `0` | Shared pipeline-cache entry budget |
| `SYNTHA_PIPELINE_CACHE_MAX_MB` | `0` | Shared pipeline-cache memory budget |

## API docs

- Architecture overview: `docs/ARCHITECTURE.md`
- Full contract: `docs/WORKFLOW_API.md`
- Pipeline lifecycle and memory cleanup policy: `docs/PIPELINE_LIFECYCLE.md`
- Helpful discovery endpoints:
  - `GET /api/workflow/task-types`
  - `GET /api/workflow/schema`
  - `GET /api/workflow/catalog`

## Runtime maintenance

Before changing generation runtime code, read `docs/PIPELINE_LIFECYCLE.md`.
Pipeline cleanup is task-scoped by default, and adapter/hook/memory cleanup must
stay in `finally` paths so failed renders do not leave GPU state behind.

## Current limitations

Use the workflow catalog and `docs/WORKFLOW_API.md` as the current source of truth for supported task fields and feature combinations. Some advanced combinations are intentionally limited, such as IP-Adapter with ControlNet, and SD1.5 LCM with ControlNet/IP-Adapter. Runtime changes should also follow `docs/PIPELINE_LIFECYCLE.md` so failed renders do not leave GPU state behind.

## License

No license file is included yet. If you plan to publish this repo, add a `LICENSE` file to clarify usage and contribution terms.
