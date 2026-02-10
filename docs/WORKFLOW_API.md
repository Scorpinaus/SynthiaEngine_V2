# Workflow-Only API Contract (v2)

This project uses a **single workflow job API** for all generation (SD1.5, SDXL, Flux, Qwen-Image, Z-Image). Image inputs are uploaded as **artifacts** first, then referenced by `artifact_id` in workflow task inputs.

Registry persistence note:
- `/lora-models` entries are persisted in `database/lora_registry.sqlite3`.
- If that SQLite registry is empty and `backend/lora_registry.json` exists, the backend performs a one-time import on startup and skips invalid rows with warning logs.
- API payload shape for LoRA entries is unchanged.

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
  "capabilities": {
    "sd15": {
      "label": "SD 1.5",
      "aliases": ["sd1.5"],
      "task_types": ["sd15.text2img", "sd15.img2img", "sd15.inpaint", "sd15.controlnet.text2img", "sd15.hires_fix"],
      "features": {
        "text2img": true,
        "img2img": true,
        "inpaint": true,
        "controlnet": true,
        "multi_controlnet": true,
        "hires_fix": true,
        "lora_adapters": true,
        "scheduler": true,
        "true_cfg_scale": false
      }
    }
  },
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
- `capabilities` is a model-family matrix for builder UIs. Current families include `sd15`, `sdxl`, `flux`, `qwen-image`, and `z-image`.
- `ui_hints` is best-effort metadata for workflow builders (labels, widgets, suggested min/max, option lists, etc.).
- `output_schema` describes the per-task result object stored under `result.tasks[taskId]`.

### LoRA registry endpoints

`GET /lora-models`
- Lists registered LoRA entries.
- Optional query `family` filters by exact `lora_model_family` (case-insensitive).
- Response `200`: `LoraRegistryEntry[]`

`LoraRegistryEntry` shape:
```json
{
  "lora_id": 101,
  "lora_model_family": "sd15",
  "lora_type": "lora",
  "lora_location": "local",
  "file_path": "C:/loras/example.safetensors",
  "name": "Example"
}
```

`POST /lora-models`
- Creates a LoRA entry.
- Request/response shape is unchanged:
```json
{
  "lora_id": 101,
  "lora_model_family": "sd15",
  "lora_type": "lora",
  "lora_location": "local",
  "file_path": "C:/loras/example.safetensors",
  "name": "Example"
}
```
- Response `200`: created `LoraRegistryEntry`
- Error `400`: validation/domain error in `{"detail": "<message>"}`.
- Duplicate id error is deterministic: `LoRA with id <lora_id> already exists.`

`GET /lora-models/{lora_id}`
- Response `200`: `LoraRegistryEntry`
- Error `404`: missing id in `{"detail": "LoRA with id <lora_id> not found."}`

`PATCH /lora-models/{lora_id}`
- Updates editable fields only: `lora_model_family`, `lora_type`, `lora_location`, `file_path`, `name`.
- `lora_id` is not editable.
- Request shape:
```json
{
  "lora_model_family": "sdxl",
  "lora_type": "lycoris",
  "lora_location": "local",
  "file_path": "C:/loras/example_v2.safetensors",
  "name": "Example v2"
}
```
- Response `200`: updated `LoraRegistryEntry`
- Error `400`: explicit validation/domain error in `{"detail": "<message>"}`.
- Error `404`: missing id in `{"detail": "LoRA with id <lora_id> not found."}`
- Error `422`: schema validation failure (for example attempting to patch non-editable `lora_id`).

`DELETE /lora-models/{lora_id}`
- Deletes one LoRA entry.
- Returns `204` on success.
- Error `404`: missing id in `{"detail": "LoRA with id <lora_id> not found."}`

Compatibility guarantees:
- Existing `GET /lora-models` and `POST /lora-models` consumers are backward-compatible.
- Existing list/create payload fields and response field names are unchanged.
- Existing list/create status codes remain unchanged (`200` success, `400` domain/validation error for create).

Frontend note (registry pages):
- `frontend/models.html` now serves base model listing with edit/delete actions.
- `frontend/model_base_add.html` serves base model create flow.
- `frontend/model_base_edit.html` serves base model edit flow via `name` query parameter.
- `frontend/lora_models.html` provides LoRA list/search/filter plus edit/delete actions.
- `frontend/lora_add.html` provides LoRA create flow.
- `frontend/lora_edit.html` provides LoRA edit flow via `lora_id` query parameter.

### Base model registry endpoints

`GET /models`
- Lists registered base model entries.
- Optional query `family` applies case-insensitive family matching.
- Response `200`: `ModelRegistryEntry[]`

`POST /models`
- Creates a base model entry.
- Response `201`: created `ModelRegistryEntry`
- Error `409`: duplicate name in `{"detail": "Model name already exists."}`

`GET /models/{model_name}`
- Fetches one base model entry by exact `name`.
- Response `200`: `ModelRegistryEntry`
- Error `404`: missing name in `{"detail": "Model '<model_name>' not found."}`

`PATCH /models/{model_name}`
- Updates editable fields: `family`, `model_type`, `location_type`, `model_id`, `version`, `link`.
- Response `200`: updated `ModelRegistryEntry`
- Error `400`: explicit validation/domain error in `{"detail": "<message>"}`.
- Error `404`: missing name in `{"detail": "Model '<model_name>' not found."}`
- Error `422`: schema validation failure (for example non-editable fields).

`DELETE /models/{model_name}`
- Deletes one base model entry by exact `name`.
- Returns `204` on success.
- Error `404`: missing name in `{"detail": "Model '<model_name>' not found."}`

### List ControlNet preprocessors (for SD1.5 ControlNet setup)

`GET /api/controlnet/preprocessors`

Returns available preprocessors plus typed parameter schema and SD1.5 model compatibility hints.

Response item shape:
```json
{
  "id": "canny",
  "name": "Canny",
  "description": "Detects edges...",
  "defaults": {
    "low_threshold": 100,
    "high_threshold": 200
  },
  "param_schema": {
    "low_threshold": {
      "type": "int",
      "description": "Lower Canny threshold.",
      "minimum": 0,
      "maximum": 255
    },
    "high_threshold": {
      "type": "int",
      "description": "Upper Canny threshold.",
      "minimum": 0,
      "maximum": 255
    }
  },
  "recommended_sd15_control_models": ["lllyasviel/control_v11p_sd15_canny"],
  "legacy_aliases": ["lllyasviel/sd-controlnet-canny"]
}
```

### Run a ControlNet preprocessor

`POST /api/controlnet/preprocess` (multipart/form-data)

Form fields:
- `image`: uploaded image file (required)
- `preprocessor_id`: preprocessor id from `GET /api/controlnet/preprocessors` (required)
- `params`: JSON object string of preprocessor params (optional)
- `low_threshold` / `high_threshold`: convenience overrides for canny-compatible flows (optional)

Validation behavior:
- `params` must decode to a JSON object.
- Unknown param keys are rejected.
- Param values are type-coerced/validated against `param_schema` bounds.
- Returns `400` with an actionable message for invalid params.

Frontend note (SD1.5 page):
- `frontend/controlnet_panel.html` is loaded by `frontend/controlnet_panel.js`.
- `frontend/controlnet_preprocessor.html` is loaded by `frontend/controlnet_preprocessor.js`.
- `frontend/sd15.js` consumes shared ControlNet state via `window.ControlNetPanel.getState()`.
- `frontend/sd15_img2img.js` also consumes shared ControlNet state via `window.ControlNetPanel.getState()`.
- `frontend/sd15_inpainting.js` also consumes shared ControlNet state via `window.ControlNetPanel.getState()`.
- `frontend/controlnet_panel.html` groups ControlNet runtime knobs (`controlnet_conditioning_scale`, `controlnet_guess_mode`, `control_guidance_start`, `control_guidance_end`).
- The preprocessor modal layout uses a two-column split (`settings` + `preview`) and caps preview height to viewport.
- `frontend/controlnet_preprocessor.js` applies a runtime layout fallback so stale cached modal markup is upgraded in-place.
- ControlNet HTML fragments are fetched with `cache: "no-store"` to avoid stale modal/panel assets.
- `frontend/controlnet_preprocessor.html` also carries inline layout styles as a last-resort cache-resistant fallback.
- The preprocessor modal collapses to one column only on narrow screens (`<=700px`).
- `frontend/sdxl.js` also consumes shared ControlNet state via `window.ControlNetPanel.getState()` for `sdxl.controlnet.text2img`.
- `frontend/sdxl_img2img.js` also consumes shared ControlNet state via `window.ControlNetPanel.getState()` for `sdxl.img2img` optional ControlNet usage.
- `frontend/sdxl_inpaint.js` also consumes shared ControlNet state via `window.ControlNetPanel.getState()` for `sdxl.inpaint` optional ControlNet usage.

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
- SDXL: `sdxl.text2img`, `sdxl.controlnet.text2img`, `sdxl.img2img`, `sdxl.inpaint`
- Flux: `flux.text2img`, `flux.img2img`, `flux.inpaint`
- Qwen-Image: `qwen-image.text2img`, `qwen-image.img2img`, `qwen-image.inpaint`
- Z-Image: `z-image.text2img`, `z-image.img2img`
- ControlNet utility: `controlnet.preprocess`

## Model capabilities matrix

This same matrix is available in machine-readable form at `GET /api/workflow/catalog` under `capabilities`.

| Family | text2img | img2img | inpaint | controlnet | hires_fix | lora_adapters | true_cfg_scale |
|---|---|---|---|---|---|---|---|
| `sd15` | yes | yes | yes | yes | yes | yes | no |
| `sdxl` | yes | yes | yes | yes | no | yes | no |
| `flux` | yes | yes | yes | no | no | no | no |
| `qwen-image` | yes | yes | yes | no | no | no | yes |
| `z-image` (`zimage`) | yes | yes | no | no | no | no | no |

Task inputs/outputs are task-specific. As a convention, image-generating tasks return:
- `images`: list of `"/outputs/..."` URLs

`controlnet.preprocess` input notes:
- `image`: image reference
- `preprocessor_id`: string id
- `params`: object only (not JSON string in workflow payload)

`sd15.controlnet.text2img` extra input notes:
- `controlnet_conditioning_scale`: float in `[0, 2]` (default `1.0`)
- `controlnet_conditioning_scales`: optional list form for multi-ControlNet; length must match model/image list length
- `controlnet_guess_mode`: boolean (default `false`)
- `control_guidance_start`: float in `[0, 1]` (default `0.0`)
- `control_guidance_end`: float in `[0, 1]` (default `1.0`)
- `control_guidance_start` must be `<= control_guidance_end`
- `controlnet_model`: defaults to `lllyasviel/control_v11p_sd15_canny` (SD1.5 v1.1 family)
- `controlnet_models`: optional list form for multi-ControlNet (backward-compatible with `controlnet_model`)
- `control_images`: optional list form for multi-ControlNet (backward-compatible with `control_image`)
- `controlnet_preprocessor_id`: optional preprocessor id used for compatibility checks
- `controlnet_preprocessor_ids`: optional list form for multi-ControlNet compatibility checks
- `controlnet_compat_mode`: `"warn"` (default), `"error"`, or `"off"`
  - `warn`: continue generation and add a warning in task result when pairing is mismatched
  - `error`: fail task when pairing is mismatched
  - `off`: skip compatibility check
- Guardrail: up to `2` ControlNet models per task; more than `1` emits a VRAM/perf warning.
- List alignment: when list forms are provided, list lengths must align with the resolved ControlNet count.

`sd15.controlnet.text2img` output notes:
- May include `warnings: string[]` (compatibility mismatch warnings and/or VRAM/perf warnings).

`sd15.img2img` optional ControlNet input notes:
- Existing `sd15.img2img` payloads remain valid without any ControlNet fields.
- To enable ControlNet, provide `control_image` (single) or `control_image` + `control_images` (multi).
- `controlnet_model` defaults to `lllyasviel/control_v11p_sd15_canny`.
- `controlnet_models`, `controlnet_conditioning_scales`, `controlnet_preprocessor_ids` are optional list forms and must align to resolved ControlNet count.
- Runtime controls mirror text2img: `controlnet_conditioning_scale`, `controlnet_guess_mode`, `control_guidance_start`, `control_guidance_end`, `controlnet_compat_mode`.
- `control_guidance_start` must be `<= control_guidance_end`.
- Guardrail: up to `2` ControlNet models per task; more than `1` emits a VRAM/perf warning.

`sd15.img2img` optional ControlNet output notes:
- May include `warnings: string[]` when compatibility/perf warnings are produced.

`sd15.img2img` LoRA input notes:
- `lora_adapters` is optional. When omitted or empty, img2img runs without LoRA adapters.
- `lora_adapters` entries are resolved through the LoRA registry (`/lora-models`) by `lora_id`.
- Each adapter may provide `strength` (default `1.0`) and optional per-component overrides (`unet_strength`, `text_encoder_strength`).
- Family validation is enforced: only LoRAs registered with `lora_model_family: "sd15"` are accepted for `sd15.img2img`.
- Invalid adapter references (for example missing `lora_id`, unknown id, or incompatible family) fail the task with a validation/runtime error.

`sd15.inpaint` optional ControlNet input notes:
- Existing `sd15.inpaint` payloads remain valid without any ControlNet fields.
- To enable ControlNet, provide `control_image` (single) or `control_image` + `control_images` (multi).
- `controlnet_model` defaults to `lllyasviel/control_v11p_sd15_canny`.
- `controlnet_models`, `controlnet_conditioning_scales`, `controlnet_preprocessor_ids` are optional list forms and must align to resolved ControlNet count.
- Runtime controls mirror text2img/img2img: `controlnet_conditioning_scale`, `controlnet_guess_mode`, `control_guidance_start`, `control_guidance_end`, `controlnet_compat_mode`.
- `control_guidance_start` must be `<= control_guidance_end`.
- Guardrail: up to `2` ControlNet models per task; more than `1` emits a VRAM/perf warning.

`sd15.inpaint` optional ControlNet output notes:
- May include `warnings: string[]` when compatibility/perf warnings are produced.

`sd15.inpaint` LoRA input notes:
- `lora_adapters` is optional. When omitted or empty, inpaint runs without LoRA adapters.
- `lora_adapters` entries are resolved through the LoRA registry (`/lora-models`) by `lora_id`.
- Each adapter may provide `strength` (default `1.0`) and optional per-component overrides (`unet_strength`, `text_encoder_strength`).
- Family validation is enforced: only LoRAs registered with `lora_model_family: "sd15"` are accepted for `sd15.inpaint`.
- Invalid adapter references (for example missing `lora_id`, unknown id, or incompatible family) fail the task with a validation/runtime error.

`sdxl.text2img` LoRA input notes:
- `lora_adapters` is optional. When omitted or empty, text2img runs without LoRA adapters.
- `lora_adapters` entries are resolved through the LoRA registry (`/lora-models`) by `lora_id`.
- Each adapter may provide `strength` (default `1.0`) and optional per-component overrides (`unet_strength`, `text_encoder_strength`).
- Family validation is enforced: only LoRAs registered with `lora_model_family: "sdxl"` are accepted for `sdxl.text2img`.
- Invalid adapter references (for example missing `lora_id`, unknown id, or incompatible family) fail the task with a validation/runtime error.

`sdxl.controlnet.text2img` extra input notes:
- `controlnet_conditioning_scale`: float in `[0, 2]` (default `1.0`)
- `controlnet_conditioning_scales`: optional list form for multi-ControlNet; length must match model/image list length
- `controlnet_guess_mode`: boolean (default `false`)
- `control_guidance_start`: float in `[0, 1]` (default `0.0`)
- `control_guidance_end`: float in `[0, 1]` (default `1.0`)
- `control_guidance_start` must be `<= control_guidance_end`
- `controlnet_model`: defaults to `diffusers/controlnet-canny-sdxl-1.0`
- `controlnet_models`: optional list form for multi-ControlNet (backward-compatible with `controlnet_model`)
- `control_images`: optional list form for multi-ControlNet (backward-compatible with `control_image`)
- `controlnet_preprocessor_id`: optional preprocessor id for compatibility checks
- `controlnet_preprocessor_ids`: optional list form for multi-ControlNet compatibility checks
- `controlnet_compat_mode`: `"warn"` (default), `"error"`, or `"off"`
  - `warn`: continue generation and add a warning in task result when preprocessor id is unknown
  - `error`: fail task when preprocessor id is unknown
  - `off`: skip compatibility checks
- Guardrail: up to `2` ControlNet models per task; more than `1` emits a VRAM/perf warning.
- List alignment: when list forms are provided, list lengths must align with the resolved ControlNet count.

`sdxl.controlnet.text2img` output notes:
- May include `warnings: string[]` (for compatibility/perf warnings).

`sdxl.img2img` optional ControlNet input notes:
- Existing `sdxl.img2img` payloads remain valid without any ControlNet fields.
- To enable ControlNet, provide `control_image` (single) or `control_image` + `control_images` (multi).
- `controlnet_model` defaults to `diffusers/controlnet-canny-sdxl-1.0`.
- `controlnet_models`, `controlnet_conditioning_scales`, `controlnet_preprocessor_ids` are optional list forms and must align to resolved ControlNet count.
- Runtime controls mirror text2img: `controlnet_conditioning_scale`, `controlnet_guess_mode`, `control_guidance_start`, `control_guidance_end`, `controlnet_compat_mode`.
- `control_guidance_start` must be `<= control_guidance_end`.
- Guardrail: up to `2` ControlNet models per task; more than `1` emits a VRAM/perf warning.

`sdxl.img2img` optional ControlNet output notes:
- May include `warnings: string[]` when compatibility/perf warnings are produced.

`sdxl.inpaint` optional ControlNet input notes:
- Existing `sdxl.inpaint` payloads remain valid without any ControlNet fields.
- To enable ControlNet, provide `control_image` (single) or `control_image` + `control_images` (multi).
- `controlnet_model` defaults to `diffusers/controlnet-canny-sdxl-1.0`.
- `controlnet_models`, `controlnet_conditioning_scales`, `controlnet_preprocessor_ids` are optional list forms and must align to resolved ControlNet count.
- Runtime controls mirror text2img/img2img: `controlnet_conditioning_scale`, `controlnet_guess_mode`, `control_guidance_start`, `control_guidance_end`, `controlnet_compat_mode`.
- `control_guidance_start` must be `<= control_guidance_end`.
- `mask_image` dimensions must match `initial_image` dimensions when using ControlNet.
- `control_image`/`control_images[*]` dimensions must match `initial_image` dimensions.
- Guardrail: up to `2` ControlNet models per task; more than `1` emits a VRAM/perf warning.

`sdxl.inpaint` optional ControlNet output notes:
- May include `warnings: string[]` when compatibility/perf warnings are produced.

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
