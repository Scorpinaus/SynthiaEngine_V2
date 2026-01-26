# Workflow-Only API Contract (v2)

This project uses a **single workflow job API** for all generation (SD1.5, SDXL, Flux, Qwen-Image, Z-Image). Image inputs are uploaded as **artifacts** first, then referenced by `artifact_id` in workflow task inputs.

## Endpoints

### Upload an artifact (image input)

`POST /api/artifacts` (multipart/form-data)

- Form field: `file` (image)
- Returns: `artifact_id` + `url` + `path`
- Artifact lifecycle: **ephemeral** -- artifacts are deleted automatically when the workflow finishes (success/fail/canceled).

Response (201):
```json
{
  "artifact_id": "a0123456789abcdef0123456789abcdef",
  "url": "/outputs/artifacts/a0123456789abcdef0123456789abcdef.png",
  "path": "artifacts/a0123456789abcdef0123456789abcdef.png"
}
```

### Submit a workflow job (the only generation entrypoint)

`POST /api/jobs`

Only supported `kind`:
- `"workflow"`

Headers:
- Optional: `Idempotency-Key: <string>` (recommended)

Body:
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

Idempotency:
- If `Idempotency-Key` (or `idempotency_key` field) is provided:
  - Same key + same request -> returns the existing job (HTTP 200).
- Same key + different request -> HTTP 409.

### Fetch job status / results

`GET /api/jobs/{job_id}`

### List jobs

`GET /api/jobs?limit=50` (clamped 1..500)

### Cancel a job

`POST /api/jobs/{job_id}/cancel`

Cancellation semantics:
- If queued: job transitions to `canceled` immediately.
- If running: server sets `cancel_requested=true`; workflow stops at **task boundaries**.

### Stream job status (SSE)

`GET /api/jobs/{job_id}/events` -> `text/event-stream`

- Emits JSON job snapshots when `status` or `updated_at` changes.
- Stops when job is terminal: `succeeded`, `failed`, `canceled`.

### Discover supported tasks (optional helper for UI/builders)

`GET /api/workflow/task-types`

Response:
```json
{ "task_types": ["sd15.text2img", "sd15.img2img", "..."] }
```

### Fetch workflow JSON schema (optional helper for tooling)

`GET /api/workflow/schema`

Response:
```json
{
  "workflow_request_schema": { /* JSON Schema */ },
  "workflow_task_schema": { /* JSON Schema */ }
}
```

### Fetch workflow task catalog (recommended for builders)

`GET /api/workflow/catalog`

Response:
```json
{
  "version": "v2",
  "tasks": {
    "sd15.text2img": {
      "input_schema": { /* JSON Schema for inputs */ },
      "input_defaults": { /* defaults for optional fields */ },
      "output_schema": { /* JSON Schema for task outputs */ },
      "ui_hints": { /* optional UI metadata */ }
    }
  }
}
```

Notes:
- `ui_hints` is best-effort metadata for workflow builders (labels, widgets, suggested min/max, option lists, etc.).
- `output_schema` describes the per-task result object stored under `result.tasks[taskId]`.

## Job object

Job `status` values:
- `queued` | `running` | `succeeded` | `failed` | `canceled`

Job response shape (subset):
```json
{
  "id": "f2c1...",
  "idempotency_key": "client-action-123",
  "cancel_requested": false,
  "kind": "workflow",
  "status": "running",
  "payload": { "tasks": [/*...*/], "return": "@t1.images" },
  "result": {
    "progress": {
      "current_task": "t1",
      "current_task_index": 0,
      "total_tasks": 2,
      "phase": "running"
    },
    "outputs": { /* resolved return value */ },
    "tasks": { /* task_id -> task result object */ }
  },
  "error": null,
  "created_at": "2026-01-25T12:34:56.789+00:00",
  "updated_at": "2026-01-25T12:35:01.234+00:00"
}
```

Notes:
- `result.outputs` is the resolved final output.
- `result.tasks` is the per-task result map (useful for debugging / UI).
- `result.progress` is best-effort; it may be absent for completed jobs created before progress reporting existed.

## Workflow payload schema

### WorkflowRequest

```json
{
  "tasks": [ /* WorkflowTask[] */ ],
  "return": /* optional; defaults to last task result */
}
```

If `"return"` is omitted, `outputs` becomes the **last task's result object** (or `{}` if no tasks).

### WorkflowTask

```json
{
  "id": "t1",
  "type": "sd15.img2img",
  "inputs": { /* task-specific */ }
}
```

Rules:
- `id` must be unique within the workflow.
- `id` must match `^[A-Za-z0-9_-]+$` (max 64 chars). Don't use `.` because `@taskId.key` uses `.` as a separator.
- Tasks run strictly in order.

## Reference syntax in inputs / return

References are resolved at runtime:

- Prior task field: `@<taskId>.<key>`
  - Example: `@t1.images` (use `t1` output images)
- Artifact reference (uploaded via `/api/artifacts`):
  - String form: `@artifact:<artifact_id>`
  - Object form: `{ "artifact_id": "<artifact_id>" }`
- Output file reference:
  - String form: `"/outputs/<relative-path>.png"`

Resolution behavior:
- References can appear anywhere inside `inputs` objects/arrays.
- Unknown task ids -> error.

## Supported task types (current)

- SD1.5: `sd15.text2img`, `sd15.img2img`, `sd15.inpaint`, `sd15.controlnet.text2img`, `sd15.hires_fix`
- SDXL: `sdxl.text2img`, `sdxl.img2img`, `sdxl.inpaint`
- Flux: `flux.text2img`, `flux.img2img`, `flux.inpaint`
- Qwen-Image: `qwen-image.text2img`, `qwen-image.img2img`, `qwen-image.inpaint`
- Z-Image: `z-image.text2img`, `z-image.img2img`
- ControlNet utility: `controlnet.preprocess`

Task inputs/outputs are task-specific. As a convention, image-generating tasks return:
- `images`: list of `"/outputs/..."` URLs

## Example: img2img workflow (artifact input)

1) Upload image:
```bash
curl -F "file=@input.png" http://localhost:8000/api/artifacts
```

2) Submit job:
```json
{
  "kind": "workflow",
  "payload": {
    "tasks": [
      {
        "id": "t1",
        "type": "sd15.img2img",
        "inputs": {
          "initial_image": { "artifact_id": "a0123456789abcdef0123456789abcdef" },
          "prompt": "a product photo, studio lighting",
          "strength": 0.6,
          "steps": 30,
          "cfg": 7.0
        }
      }
    ],
    "return": "@t1.images"
  }
}
```
