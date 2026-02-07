# SynthaEngine

SynthaEngine is a local **image generation server + lightweight web UI** built around a single, consistent concept: **submit a workflow job** and poll/stream results. Under the hood it uses Hugging Face **Diffusers/Transformers** pipelines and exposes them via a **FastAPI** backend.

It's designed to make it easy to:
- Run multiple model families behind one API surface (SD1.5, SDXL, Flux, Qwen-Image, Z-Image).
- Chain steps together (e.g., preprocess -> generate) using workflow tasks and runtime references.
- Manage long-running work with a job queue, status endpoints, and SSE events.

## What's in this repo

- `backend/`: FastAPI app (`backend/main.py`) + workflow/task system (`backend/workflow.py`) + model/LoRA registries.
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

Current task types are documented in `docs/WORKFLOW_API.md` and implemented in `backend/workflow.py`:
- SD1.5: `sd15.text2img`, `sd15.img2img`, `sd15.inpaint`, `sd15.controlnet.text2img`, `sd15.hires_fix`
- SDXL: `sdxl.text2img`, `sdxl.img2img`, `sdxl.inpaint`
- Flux: `flux.text2img`, `flux.img2img`, `flux.inpaint`
- Qwen-Image: `qwen-image.text2img`, `qwen-image.img2img`, `qwen-image.inpaint`
- Z-Image: `z-image.text2img`, `z-image.img2img`
- Utility: `controlnet.preprocess`

## Quickstart (Windows)

### Prereqs

- Python **3.10+**
- A working PyTorch install for your platform (CPU or CUDA), plus the Python deps in `requirements.txt`

### Run

1) Create a virtualenv at `.venv` and install deps.

2) Start the app:
```bat
run_app.bat
```

This launches:
- Backend API: `http://127.0.0.1:8000`
- Frontend: `http://127.0.0.1:4173` (opens `sd15.html` by default)

Note: `requirements.txt` currently references a local editable dependency `-e ./controlnet_aux`. If you don't have that folder in this repo checkout, `pip install -r requirements.txt` will fail; either add it (if you use it) or remove/replace that line.

## API docs

- Full contract: `docs/WORKFLOW_API.md`
- Helpful discovery endpoints:
  - `GET /api/workflow/task-types`
  - `GET /api/workflow/schema`
  - `GET /api/workflow/catalog`

## Known issues / next steps

See `next_steps.txt` for the current roadmap and known limitations (e.g., pipeline lifecycle/cleanup and OOM scenarios under some workloads).

## License

No license file is included yet. If you plan to publish this repo, add a `LICENSE` file to clarify usage and contribution terms.
