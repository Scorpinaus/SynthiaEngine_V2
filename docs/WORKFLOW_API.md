# Workflow-Only API Contract (v2)

This project uses a **single workflow job API** for all generation (SD1.5, SDXL, WAN, Flux, Qwen-Image, Z-Image). Image and video inputs are uploaded as **artifacts** first, then referenced by `artifact_id` in workflow task inputs.

Registry persistence note:
- `/lora-models` entries are persisted in `database/lora_registry.sqlite3`.
- If that SQLite registry is empty and `backend/lora/lora_registry.json` exists, the backend performs a one-time import on startup and skips invalid rows with warning logs.
- API payload shape for LoRA entries is unchanged.
- `/api/presets` entries are persisted in `database/preset_registry.sqlite3`.

## Endpoints

### Upload an artifact (image/video input)

`POST /api/artifacts` (multipart/form-data)

- Form field: `file` (image or video)
- Returns: `artifact_id` + `url` + `path`
- Artifact lifecycle: **ephemeral** -- artifacts are deleted automatically when the workflow finishes (success/fail/canceled).
- Image artifact ids use the `a...` prefix and are stored as PNG. Video artifact ids use the `v...` prefix and preserve supported extensions (`.mp4`, `.webm`, `.mov`, `.gif`).

Response (201):
```json
{
  "artifact_id": "a0123456789abcdef0123456789abcdef",
  "url": "/outputs/artifacts/a0123456789abcdef0123456789abcdef.png",
  "path": "artifacts/a0123456789abcdef0123456789abcdef.png"
}
```

### List render history

`GET /history`

- Returns generated media records from `outputs/`.
- Includes PNG image records and video records such as MP4/WebM/MOV.
- Existing fields are preserved for compatibility. New consumers can use `media_type` to choose image or video rendering.
- PNG records include embedded text metadata when available.
- Video records load adjacent JSON sidecars named `video_<batch_id>.mp4.json` when present.
- Video records still infer `metadata.batch_id` from `outputs/batch_<batch_id>/...` paths when no sidecar is available.

Response (200):
```json
[
  {
    "filename": "batch_abc123/abc123_42.mp4",
    "url": "/outputs/batch_abc123/abc123_42.mp4",
    "timestamp": 1710000000.0,
    "created_at": "2024-03-09T16:00:00+00:00",
    "media_type": "video",
    "metadata": {
      "prompt": "waves rolling under moonlight",
      "negative_prompt": "jitter",
      "steps": 25,
      "cfg": 7.5,
      "seed": 42,
      "batch_id": "abc123"
    }
  }
]
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

- Job timestamp fields (`created_at`, `updated_at`, `started_at`, `finished_at`) are returned as ISO-8601 strings with an explicit timezone offset.
- Current job APIs normalize timestamps to UTC (`+00:00`), which lets clients safely convert them to the viewer's system timezone.

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
      "task_types": ["sd15.text2img", "sd15.animatediff.text2video", "sd15.img2img", "sd15.inpaint", "sd15.controlnet.text2img", "sd15.hires_fix"],
      "features": {
        "text2img": true,
        "text2video": true,
        "img2img": true,
        "inpaint": true,
        "controlnet": true,
        "multi_controlnet": true,
        "hires_fix": true,
        "lora_adapters": true,
        "ip_adapter": true,
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
- `capabilities` is a model-family matrix for builder UIs. Current families include `sd15`, `sdxl`, `wan`, `flux`, `qwen-image`, and `z-image`.
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
- Validation: `name` cannot contain `.` (dot). Requests with dot in `name` return `422`.

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
- Validation: when provided, `name` cannot contain `.` (dot). Requests with dot in `name` return `422`.
- Error `422`: schema validation failure (for example attempting to patch non-editable `lora_id`).

`DELETE /lora-models/{lora_id}`
- Deletes one LoRA entry.
- Returns `204` on success.
- Error `404`: missing id in `{"detail": "LoRA with id <lora_id> not found."}`

Compatibility guarantees:
- Existing `GET /lora-models` and `POST /lora-models` consumers are backward-compatible.
- Existing list/create payload fields and response field names are unchanged.
- Existing list/create status codes remain unchanged (`200` success, `400` domain/validation error for create).

### Preset registry endpoints

`GET /api/presets`
- Lists saved prompt/generation presets.
- Optional query `family` filters by exact family (case-insensitive).
- Optional query `task_type` filters by exact workflow task type.
- Response `200`: `PresetRegistryEntry[]`

`PresetRegistryEntry` shape:
```json
{
  "preset_id": 1,
  "name": "SD15 Product Baseline",
  "family": "sd15",
  "task_type": "sd15.text2img",
  "settings": {
    "prompt": "product photo, studio lighting",
    "negative_prompt": "blurry, low quality",
    "steps": 30,
    "cfg": 7.0,
    "scheduler": "euler",
    "seed": 1234,
    "width": 640,
    "height": 640,
    "num_images": 1,
    "clip_skip": 1,
    "weighting_policy": "diffusers-like",
    "hires_enabled": true,
    "hires_scale": 1.5,
    "controlnet_enabled": false,
    "lora_adapters": [{ "lora_id": 101, "strength": 0.8 }]
  }
}
```

`POST /api/presets`
- Creates a new preset.
- Request shape:
```json
{
  "name": "SD15 Product Baseline",
  "family": "sd15",
  "task_type": "sd15.text2img",
  "settings": { "prompt": "..." }
}
```
- Response `201`: created `PresetRegistryEntry`
- Error `400`: explicit validation/domain error in `{"detail": "<message>"}`.

`GET /api/presets/{preset_id}`
- Response `200`: `PresetRegistryEntry`
- Error `404`: missing id in `{"detail": "Preset with id <preset_id> not found."}`

`PATCH /api/presets/{preset_id}`
- Updates editable fields: `name`, `family`, `task_type`, `settings`.
- Request shape:
```json
{
  "name": "SD15 Product Baseline v2",
  "settings": {
    "prompt": "updated prompt",
    "steps": 24
  }
}
```
- Response `200`: updated `PresetRegistryEntry`
- Error `400`: explicit validation/domain error in `{"detail": "<message>"}`.
- Error `404`: missing id in `{"detail": "Preset with id <preset_id> not found."}`
- Error `422`: schema validation failure (for example unknown patch fields).

`DELETE /api/presets/{preset_id}`
- Deletes one preset.
- Returns `204` on success.
- Error `404`: missing id in `{"detail": "Preset with id <preset_id> not found."}`

Frontend note (registry pages):
- `frontend/models/base/registry.html` now serves base model listing with edit/delete actions.
- `frontend/models/base/add.html` serves base model create flow.
- `frontend/models/base/edit.html` serves base model edit flow via `name` query parameter.
- `frontend/models/lora/model_page.html` provides LoRA list/search/filter plus edit/delete actions.
- `frontend/models/lora/add.html` provides LoRA create flow.
- `frontend/models/lora/edit.html` provides LoRA edit flow via `lora_id` query parameter.

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

Implemented SD1.5-oriented preprocessor ids:
- Edges/soft edges/scribble: `canny`, `hed`, `softedge-hed`, `softedge-hedsafe`, `scribble-hed`, `pidinet`, `softedge-pidinet`, `softedge-pidsafe`, `scribble-pidinet`
- Depth/normal: `midas-depth`, `depth-zoe`, `depth-leres`, `depth-leres-plus`, `normal-midas`, `normal-bae`
- Pose/lines/structure: `openpose`, `dwpose`, `mlsd`, `lineart`, `lineart-anime`, `lineart-standard`, `teed`, `anyline`, `shuffle`
- Face/segmentation: `mediapipe-face`, `sam-mobile`, `sam`
- Instruct-pix2pix source condition: `ip2p-source` passes the uploaded source image through unchanged for `lllyasviel/control_v11e_sd15_ip2p`.
- Derived inpaint condition: `inpaint-condition` is a compatibility id for `lllyasviel/control_v11p_sd15_inpaint`; the backend derives its condition from `initial_image` + `mask_image` instead of running `/api/controlnet/preprocess`.

Some heavier processors are exposed as optional entries. The catalog includes `available`, `unavailable_reason`, and `install_hint` so clients can disable processors that require extra local dependencies. `dwpose` requires `easy-dwpose`, `mediapipe-face` requires `mediapipe`, `sam` downloads a large Segment Anything checkpoint, `sam-mobile` uses MobileSAM, and `teed`/`anyline` use their upstream checkpoint repos.

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
  "available": true,
  "unavailable_reason": null,
  "install_hint": null,
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
- Returns `503` when an optional heavy preprocessor is unavailable because a runtime dependency is missing.
- `ip2p-source` is a no-op pass-through processor for instruction-based image editing. Use the source image as `image`, then pair the resulting ControlNet image with `controlnet_model: "lllyasviel/control_v11e_sd15_ip2p"` and an edit-style prompt such as `"make it on fire"`.

Frontend note (SD1.5 page):
- `frontend/components/controlnet_panel.html` is loaded by `frontend/components/controlnet_panel.js`.
- `frontend/components/controlnet_preprocessor.html` is loaded by `frontend/components/controlnet_preprocessor.js`.
- The preprocessor modal renders parameter controls from each entry's `param_schema`; new backend preprocessors do not require hardcoded frontend parameter fields.
- `frontend/sd15/text2img.html`, `frontend/sd15/img2img.html`, and `frontend/sd15/inpainting.html` group ControlNet preprocessors, LoRA adapters, and IP-Adapter controls behind one adapter modal. The overview tab shows available and active adapter counts without changing workflow payload shape.
- `frontend/sd15/text2img.js` consumes shared ControlNet state via `window.ControlNetPanel.getState()`.
- `frontend/sd15/text2img.js` uploads the optional SD1.5 IP-Adapter reference image, creates a `sd15.ip_adapter.encode` task, and sends the resulting `image_embeds` into `sd15.text2img.inputs.ip_adapter.image_embeds`. It uploads the optional IP-Adapter mask as `sd15.text2img.inputs.ip_adapter.mask_image`.
- `frontend/sd15/img2img.js` also consumes shared ControlNet state via `window.ControlNetPanel.getState()`.
- `frontend/sd15/img2img.js` uploads the optional SD1.5 IP-Adapter reference image, creates a `sd15.ip_adapter.encode` task, and sends the resulting `image_embeds` into `sd15.img2img.inputs.ip_adapter.image_embeds`. It uploads the optional IP-Adapter mask as `sd15.img2img.inputs.ip_adapter.mask_image`.
- `frontend/sd15/inpainting.js` also consumes shared ControlNet state via `window.ControlNetPanel.getState()`.
- `frontend/sd15/inpainting.js` adds an SD1.5 inpaint ControlNet condition toggle that sends `lllyasviel/control_v11p_sd15_inpaint` + `inpaint-condition` without uploading a separate ControlNet preprocessor image.
- `frontend/sd15/inpainting.js` uploads the optional SD1.5 IP-Adapter reference image, creates a `sd15.ip_adapter.encode` task, and sends the resulting `image_embeds` into `sd15.inpaint.inputs.ip_adapter.image_embeds`. It uploads the optional IP-Adapter mask as `sd15.inpaint.inputs.ip_adapter.mask_image`.
- `frontend/components/ip_adapter_panel.js` supports either uploading an IP-Adapter mask image or creating one in a lightweight canvas editor. White = apply image prompt; black = suppress image prompt.
- `frontend/components/controlnet_panel.html` groups ControlNet runtime knobs (`controlnet_conditioning_scale`, `controlnet_guess_mode`, `control_guidance_start`, `control_guidance_end`).
- The preprocessor modal layout uses a two-column split (`settings` + `preview`) and caps preview height to viewport.
- `frontend/components/controlnet_preprocessor.js` applies a runtime layout fallback so stale cached modal markup is upgraded in-place.
- ControlNet HTML fragments are fetched with `cache: "no-store"` to avoid stale modal/panel assets.
- `frontend/components/controlnet_preprocessor.html` also carries inline layout styles as a last-resort cache-resistant fallback.
- The preprocessor modal collapses to one column only on narrow screens (`<=700px`).
- `frontend/sd15/animatediff.html` serves SD1.5 AnimateDiff text-to-video generation and renders `videos` outputs in `frontend/components/video_gallery.js`.
- `frontend/wan/text2video.html` serves WAN text-to-video generation and renders `videos` outputs in `frontend/components/video_gallery.js`.
- `frontend/sdxl/text2img.js` also consumes shared ControlNet state via `window.ControlNetPanel.getState()` for `sdxl.controlnet.text2img`.
- `frontend/sdxl/text2img.js` uploads the optional SDXL IP-Adapter reference image through `/api/artifacts` and sends it as `sdxl.text2img.inputs.ip_adapter.image`.
- `frontend/sdxl/img2img.js` also consumes shared ControlNet state via `window.ControlNetPanel.getState()` for `sdxl.img2img` optional ControlNet usage.
- `frontend/sdxl/img2img.js` uploads the optional SDXL IP-Adapter reference image through `/api/artifacts` and sends it as `sdxl.img2img.inputs.ip_adapter.image`.
- `frontend/sdxl/inpaint.js` also consumes shared ControlNet state via `window.ControlNetPanel.getState()` for `sdxl.inpaint` optional ControlNet usage.
- `frontend/sdxl/inpaint.js` also consumes shared LoRA state via `window.LoraPanel.getSelectedAdapters()` for `sdxl.inpaint`.
- `frontend/sdxl/inpaint.js` uploads the optional SDXL IP-Adapter reference image through `/api/artifacts` and sends it as `sdxl.inpaint.inputs.ip_adapter.image`.

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
- Embed artifact reference (created by `sd15.ip_adapter.encode` or `sdxl.ip_adapter.encode`):
  - Object form: `{ "artifact_id": "e..." }`
  - These are ephemeral `.pt` files under `outputs/artifacts/` and are deleted with workflow artifacts after the job finishes.
- Output file reference:
  - String form: `"/outputs/<relative-path>.png"`

Resolution behavior:
- References can appear anywhere inside `inputs` objects/arrays.
- Unknown task ids -> error.

## Supported task types (current)

- SD1.5: `sd15.ip_adapter.encode`, `sd15.text2img`, `sd15.animatediff.text2video`, `sd15.img2img`, `sd15.inpaint`, `sd15.controlnet.text2img`, `sd15.hires_fix`
- SDXL: `sdxl.ip_adapter.encode`, `sdxl.text2img`, `sdxl.controlnet.text2img`, `sdxl.img2img`, `sdxl.inpaint`
- WAN: `wan.text2video`
- Flux: `flux.text2img`, `flux.img2img`, `flux.inpaint`
- Qwen-Image: `qwen-image.text2img`, `qwen-image.img2img`, `qwen-image.inpaint`
- Z-Image: `z-image.text2img`, `z-image.img2img`, `z-image.inpaint`
- ControlNet utility: `controlnet.preprocess`

## Model capabilities matrix

This same matrix is available in machine-readable form at `GET /api/workflow/catalog` under `capabilities`.

| Family | text2img | text2video | img2img | inpaint | controlnet | hires_fix | lora_adapters | ip_adapter | true_cfg_scale |
|---|---|---|---|---|---|---|---|---|---|
| `sd15` | yes | yes | yes | yes | yes | yes | yes | yes | no |
| `sdxl` | yes | no | yes | yes | yes | no | yes | yes | no |
| `wan` (`wan2.1`, `wan2.2`) | no | yes | no | no | no | no | no | no | no |
| `flux` | yes | no | yes | yes | no | no | yes | no | no |
| `qwen-image` | yes | no | yes | yes | no | no | yes | no | yes |
| `z-image` (`zimage`) | yes | no | yes | yes | no | no | yes | no | no |

Task inputs/outputs are task-specific. As a convention, image-generating tasks return:
- `images`: list of `"/outputs/..."` URLs

Video-generating tasks return:
- `videos`: list of `"/outputs/..."` URLs

LoRA adapter targeting:
- For tasks that accept `lora_adapters`, each adapter can optionally include `target` with one of:
  - `"both"` (default): load for both UNet and text encoder (existing behavior)
  - `"unet"`: load and apply only on UNet
  - `"text_encoder"`: load and apply only on text encoder
- Existing payloads without `target` remain valid and behave as before.

`sd15.text2img` contract-extension input notes:
- Existing flat input fields remain valid and are still recommended for backward compatibility.
- `controlNetEnabled`: optional boolean UI state flag.
- `scheduler`: supports the shared scheduler ids plus `"lcm"` for SD1.5 LCM text-to-image mode.
- `lcm`: optional object `{ "enabled": boolean }`.
  - When `lcm.enabled` is `true`, or when `scheduler` is `"lcm"`, backend uses `LCMScheduler` and loads the hard-coded SD1.5 LCM LoRA `latent-consistency/lcm-lora-sdv1-5`.
  - LCM mode defaults to `steps: 4` and `cfg: 0.0` when those fields are omitted.
  - LCM mode requires `steps` within `[1, 8]` and `cfg` within `[0, 2]`.
  - LCM mode may be combined with user-selected SD1.5 LoRA adapters; the hard-coded LCM LoRA is loaded first and selected adapters are added to the active adapter stack.
  - LCM mode is available for SD1.5 text-to-image, image-to-image, and inpainting. For `sd15.text2img`, do not combine it with ControlNet, Hi-Res Fix, or AnimateDiff.
- `lora`: optional object `{ "lora_enabled": boolean, "lora_adapters": [...] }`.
  - Canonical and only supported SD1.5 LoRA contract.
  - When `lora.lora_enabled` is `false`, SD1.5 tasks run without LoRA adapters.
  - When omitted, or when `lora.lora_adapters` is omitted/empty, SD1.5 tasks run without LoRA adapters.
- Deprecated SD1.5 LoRA inputs are rejected with a validation/runtime error:
  - top-level `lora_adapters`
  - legacy `Lora` object
- `hires`: optional object `{ "hiresEnabled": boolean, "hires_scale": number }`.
  - When `hires_enabled` / `hires_scale` are omitted, backend can derive values from `hires`.
- `ip_adapter`: optional SD1.5 text-to-image image prompt object.
  - Shape: `{ "enabled": boolean, "image": ImageRef, "image_embeds": EmbedRef, "mask_image": ImageRef, "scale": number, "model": string, "subfolder": string, "weight_name": string }`.
  - Minimal supported default adapter is `model: "h94/IP-Adapter"`, `subfolder: "models"`, `weight_name: "ip-adapter_sd15.bin"`.
  - Exactly one of `image` or `image_embeds` is required when `enabled` is `true`.
  - `image` accepts references matching other workflow image inputs (`{"artifact_id":"..."}`, `"@artifact:..."`, or `"/outputs/..."`).
  - `image_embeds` accepts an embed artifact produced by `sd15.ip_adapter.encode`. When `image_embeds` is provided, render loads the IP-Adapter with `image_encoder_folder: null` and uses the precomputed embeds instead of the reference image.
  - `mask_image` is optional; accepted references match other workflow image inputs. White pixels apply the IP-Adapter image prompt and black pixels suppress it. The backend preprocesses it with Diffusers `IPAdapterMaskProcessor` at the final output size.
  - `scale` defaults to `0.6` and must be within `[0, 1]`; Diffusers uses this to control image-prompt influence.
  - Initial support is one SD1.5 base IP-Adapter for `sd15.text2img`, `sd15.img2img`, and `sd15.inpaint`. FaceID, Plus variants, multiple adapters, ControlNet combinations, LCM combinations, and Hi-Res Fix combinations are outside the initial contract.

`sd15.ip_adapter.encode` input notes:
- Creates a temporary SD1.5 IP-Adapter image-embeds artifact for later use by SD1.5 render tasks.
- The encoder loads only the IP-Adapter weights and CLIP vision image encoder; it does not load the SD1.5 UNet, VAE, text encoder, tokenizer, or scheduler.
- Minimal encoder v1 supports only the default base SD1.5 adapter: `ip_adapter_model: "h94/IP-Adapter"`, `ip_adapter_subfolder: "models"`, `ip_adapter_weight_name: "ip-adapter_sd15.bin"`.
- Plus variants, FaceID variants, multiple IP-Adapters, and custom projection formats that require hidden states are rejected by the minimal encoder.
- CUDA is required for the minimal encode step.
- Inputs:
  - `image`: reference image (`ImageRef`).
  - `model`: SD1.5 base model registry name saved in embed metadata for render-side compatibility checks.
  - `guidance_scale`: render guidance scale; this determines whether classifier-free guidance embeds are created and must match the render step's guidance mode.
  - `ip_adapter_model`, `ip_adapter_subfolder`, `ip_adapter_weight_name`, `ip_adapter_scale`: adapter settings. Defaults mirror SD1.5 render tasks.
- Output:
  - `{ "image_embeds": { "artifact_id": "e...", "path": "artifacts/e....pt", "url": "/outputs/artifacts/e....pt" } }`
- Embed artifacts are ephemeral and cleaned up after the workflow finishes.

`sd15.animatediff.text2video` input notes:
- `prompt` / `negative_prompt`: prompt text.
- `steps`: inference steps (default `25`).
- `cfg`: guidance scale (default `7.5`).
- `width` / `height`: output frame size (default `512x512`).
- `seed`: optional seed; `null` or `0` selects a random base seed.
- `scheduler`: defaults to `ddim`; the backend applies AnimateDiff-friendly DDIM settings (`clip_sample=false`, `timestep_spacing=linspace`, `beta_schedule=linear`, `steps_offset=1`).
- `model`: SD1.5 base model registry name.
- `motion_adapter`: MotionAdapter hub id, local directory, or local single-file adapter path. Default: `guoyww/animatediff-motion-adapter-v1-5-2`.
- `num_frames`: number of generated frames per video (default `16`). The default `guoyww/animatediff-motion-adapter-v1-5-2` adapter has a 32-frame temporal context limit for normal AnimateDiff generation.
- `fps`: MP4 export frame rate (default `8`).
- `num_videos`: number of videos to generate (default `1`).
- `free_noise_enabled`: optional boolean (default `false`). Enable FreeNoise for longer videos where `num_frames` exceeds the motion adapter temporal context limit.
- `free_noise_context_length`: FreeNoise temporal window length (default `16`, minimum `1`). Keep this value within the motion adapter temporal context limit.
- `free_noise_context_stride`: FreeNoise temporal window stride (default `4`, minimum `1`). Must be `<= free_noise_context_length`.
- FreeNoise mode uses Diffusers raw prompt encoding. Custom prompt-weight embedding expansion is disabled in FreeNoise mode because Diffusers FreeNoise does not support direct `prompt_embeds`.
- `free_init_enabled`: optional boolean (default `false`). Enable Diffusers FreeInit for improved temporal consistency and video quality at additional inference cost.
- `free_init_num_iters`: FreeInit noise re-initialization iterations (default `3`, minimum `1`). Higher values increase sampling work.
- `free_init_use_fast_sampling`: optional boolean (default `false`). Enables Diffusers coarse-to-fine FreeInit sampling for better speed at a possible quality tradeoff.
- `free_init_method`: FreeInit low-pass filter method (default `butterworth`). Must be one of `butterworth`, `ideal`, or `gaussian`.
- `free_init_order`: FreeInit filter order for `butterworth` mode (default `4`, minimum `1`).
- `free_init_spatial_stop_frequency`: FreeInit spatial stop frequency (default `0.25`, range `[0, 1]`).
- `free_init_temporal_stop_frequency`: FreeInit temporal stop frequency (default `0.25`, range `[0, 1]`).
- `clip_skip`: CLIP skip value (default `1`).
- `weighting_policy`: prompt-weighting parser policy.
- `lora`: optional SD1.5 unified LoRA contract `{ "lora_enabled": boolean, "lora_adapters": [...] }`.
- `batch_id`: optional batch identifier.

`sd15.animatediff.text2video` output notes:
- Returns `{ "batch_id": "...", "videos": ["/outputs/...mp4"] }`.
- Writes batch video metadata to `outputs/batch_<batch_id>/video_<batch_id>.mp4.json`; `/history` uses this sidecar to show prompt and generation settings for SD1.5 AnimateDiff videos.

`wan.text2video` input notes:
- `prompt` / `negative_prompt`: prompt text.
- `model`: defaults to the local Diffusers folder `D:\diffusion\diffusers\Wan2.1-T2V-1.3B-Diffusers`.
- `model`: set `D:\diffusion\diffusers\Wan2.1-VACE-1.3B-diffusers` for the local VACE controllable generation model.
- After both local folders are present, the Hugging Face cache copies for `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` and `Wan-AI/Wan2.1-VACE-1.3B-diffusers` are not required by SynthaEngine defaults.
- `width` / `height`: supported values are `832x480` and `512x512`.
- `num_frames`: must be one of `33`, `49`, or `81`.
- `steps`: inference steps (default `30`).
- `guidance_scale`: guidance scale (default `6.0`).
- `fps`: MP4 export frame rate (default `16`).
- `seed`: optional seed; `null` or `0` selects a random base seed.
- `num_videos`: fixed to `1`; WAN uses the existing single workflow generation queue.
- `memory_preset`: fixed to `"safe"`; backend loads the VAE in float32, uses bfloat16 pipeline weights, sets 480P `flow_shift=3.0`, and enables Diffusers model CPU offload.
- VACE additions:
  - `conditioning_video`: single video artifact/output reference. Required for the VACE conditioning path.
  - `mask_image`: single image artifact/output reference. Required when `conditioning_video` is provided. Black pixels condition/preserve the source video region; white pixels mark regions to generate.
  - `reference_image`: single image artifact/output reference for subject/composition conditioning. Required for the VACE conditioning path.
  - `conditioning_scale`: VACE conditioning scale in `[0, 2]` (default `1.0`).
- `batch_id`: optional batch identifier.

`wan.text2video` output notes:
- Returns `{ "batch_id": "...", "videos": ["/outputs/...mp4"] }`.
- Writes batch video metadata to `outputs/batch_<batch_id>/video_<batch_id>.mp4.json`; `/history` uses this sidecar to show prompt and WAN generation settings.
- WAN 2.2 planning note: `WanPipeline` supports WAN 2.2 text-to-video models, but official Wan-AI VACE Diffusers coverage is centered on Wan2.1 VACE. Community Wan2.2 VACE/Fun checkpoints may be evaluated later behind the same single-video constraints.

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
- `control_guidance_starts`: optional list form for per-ControlNet guidance start timing; length must match model/image list length
- `control_guidance_ends`: optional list form for per-ControlNet guidance end timing; length must match model/image list length
- each per-index guidance start/end pair must satisfy `control_guidance_starts[i] <= control_guidance_ends[i]`
- `controlnet_model`: defaults to `lllyasviel/control_v11p_sd15_canny` (SD1.5 v1.1 family)
- `controlnet_models`: optional list form for multi-ControlNet (backward-compatible with `controlnet_model`)
- `control_images`: optional list form for multi-ControlNet (backward-compatible with `control_image`)
- `controlnet_preprocessor_id`: optional preprocessor id used for compatibility checks
- `controlnet_preprocessor_ids`: optional list form for multi-ControlNet compatibility checks
- `controlNetEnabled`: optional boolean UI state flag.
- `effectiveItems`: optional list contract form for ControlNet items:
  - item shape: `{ "control_image": <ImageRef>, "model_id": string?, "conditioning_scale": number?, "guidance_start": number?, "guidance_end": number?, "preprocessor_id": string? }`
  - when provided, backend can derive flat fields (`control_image(s)`, `controlnet_model(s)`, `controlnet_conditioning_scale(s)`, `control_guidance_start(s)`, `control_guidance_end(s)`, `controlnet_preprocessor_id(s)`) if those are omitted.
- `lora`: optional object `{ "lora_enabled": boolean, "lora_adapters": [...] }`.
- `hires`: optional object `{ "hiresEnabled": boolean, "hires_scale": number }`.
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
- `lora` is an optional SD1.5 unified object `{ "lora_enabled": boolean, "lora_adapters": [...] }`.
  - If `lora_enabled` is `false`, img2img runs without LoRA adapters.
  - When omitted, or when `lora.lora_adapters` is omitted/empty, img2img runs without LoRA adapters.
- Deprecated SD1.5 LoRA inputs are rejected: top-level `lora_adapters` and legacy `Lora`.
- `lora.lora_adapters` entries are resolved through the LoRA registry (`/lora-models`) by `lora_id`.
- Each adapter may provide `strength` (default `1.0`), optional per-component overrides (`unet_strength`, `text_encoder_strength`), and optional fine-grained scales (`unet_scales`, `text_encoder_scales`).
- `unet_scales` accepts Diffusers-style UNet LoRA scales (number or nested object) and is forwarded to `set_adapters(..., adapter_weights=...)` for per-layer control.
- `text_encoder_scales` is an object mapping text-encoder module-name substrings to scale values; unmatched text-encoder LoRA layers fall back to `text_encoder_strength` when provided, otherwise `strength`.
- Family validation is enforced: only LoRAs registered with `lora_model_family: "sd15"` are accepted for `sd15.img2img`.
- Invalid adapter references (for example missing `lora_id`, unknown id, or incompatible family) fail the task with a validation/runtime error.

`sd15.img2img` LCM input notes:
- `lcm`: optional object `{ "enabled": boolean }`.
- `scheduler: "lcm"` also enables LCM mode.
- When enabled, backend uses `LCMScheduler` and loads the hard-coded SD1.5 LCM LoRA `latent-consistency/lcm-lora-sdv1-5`.
- LCM mode defaults to `steps: 4` and `cfg: 0.0` when those fields are omitted.
- LCM mode requires `steps` within `[1, 8]` and `cfg` within `[0, 2]`.
- LCM mode may be combined with user-selected SD1.5 LoRA adapters; the hard-coded LCM LoRA is loaded first and selected adapters are added to the active adapter stack.
- Minimal initial support is non-ControlNet img2img only. Do not combine `sd15.img2img` LCM mode with ControlNet fields.

`sd15.img2img` IP-Adapter input notes:
- `ip_adapter`: optional SD1.5 image-to-image image prompt object.
- Shape mirrors `sd15.text2img`: `{ "enabled": boolean, "image": ImageRef, "image_embeds": EmbedRef, "mask_image": ImageRef, "scale": number, "model": string, "subfolder": string, "weight_name": string }`.
- Minimal supported default adapter is `model: "h94/IP-Adapter"`, `subfolder: "models"`, `weight_name: "ip-adapter_sd15.bin"`.
- Exactly one of `image` or `image_embeds` is required when `enabled` is `true`.
- `image_embeds` accepts an embed artifact produced by `sd15.ip_adapter.encode`; render loads the IP-Adapter without the image encoder when embeds are provided.
- `mask_image` is optional and uses the same white-applies/black-suppresses convention as `sd15.text2img`.
- `scale` defaults to `0.6` and must be within `[0, 1]`.
- Minimal initial support is non-ControlNet, non-LCM `sd15.img2img` only. User-selected SD1.5 LoRA adapters may still be combined with IP-Adapter.

`sd15.inpaint` optional ControlNet input notes:
- Existing `sd15.inpaint` payloads remain valid without any ControlNet fields.
- To enable ControlNet, provide `control_image` (single) or `control_image` + `control_images` (multi).
- Exception: for `controlnet_model: "lllyasviel/control_v11p_sd15_inpaint"` with `controlnet_preprocessor_id: "inpaint-condition"`, `control_image` may be omitted. The backend builds the Diffusers inpaint ControlNet condition from `initial_image` and `mask_image` by marking masked pixels in the conditioning image.
- `controlnet_model` defaults to `lllyasviel/control_v11p_sd15_canny`.
- `controlnet_models`, `controlnet_conditioning_scales`, `controlnet_preprocessor_ids` are optional list forms and must align to resolved ControlNet count.
- Runtime controls mirror text2img/img2img: `controlnet_conditioning_scale`, `controlnet_guess_mode`, `control_guidance_start`, `control_guidance_end`, `controlnet_compat_mode`.
- `control_guidance_start` must be `<= control_guidance_end`.
- Guardrail: up to `2` ControlNet models per task; more than `1` emits a VRAM/perf warning.

`sd15.inpaint` optional ControlNet output notes:
- May include `warnings: string[]` when compatibility/perf warnings are produced.

`sd15.inpaint` LoRA input notes:
- `lora` is an optional SD1.5 unified object `{ "lora_enabled": boolean, "lora_adapters": [...] }`.
  - If `lora_enabled` is `false`, inpaint runs without LoRA adapters.
  - When omitted, or when `lora.lora_adapters` is omitted/empty, inpaint runs without LoRA adapters.
- Deprecated SD1.5 LoRA inputs are rejected: top-level `lora_adapters` and legacy `Lora`.
- `lora.lora_adapters` entries are resolved through the LoRA registry (`/lora-models`) by `lora_id`.
- Each adapter may provide `strength` (default `1.0`), optional per-component overrides (`unet_strength`, `text_encoder_strength`), and optional fine-grained scales (`unet_scales`, `text_encoder_scales`).
- Family validation is enforced: only LoRAs registered with `lora_model_family: "sd15"` are accepted for `sd15.inpaint`.
- Invalid adapter references (for example missing `lora_id`, unknown id, or incompatible family) fail the task with a validation/runtime error.

`sd15.inpaint` LCM input notes:
- `lcm`: optional object `{ "enabled": boolean }`.
- `scheduler: "lcm"` also enables LCM mode.
- When enabled, backend uses `LCMScheduler` and loads the hard-coded SD1.5 LCM LoRA `latent-consistency/lcm-lora-sdv1-5`.
- LCM mode defaults to `steps: 4` and `cfg: 0.0` when those fields are omitted.
- LCM mode requires `steps` within `[1, 8]` and `cfg` within `[0, 2]`.
- LCM mode may be combined with user-selected SD1.5 LoRA adapters; the hard-coded LCM LoRA is loaded first and selected adapters are added to the active adapter stack.
- Minimal initial support is non-ControlNet inpaint only. Do not combine `sd15.inpaint` LCM mode with ControlNet fields.

`sd15.inpaint` IP-Adapter input notes:
- `ip_adapter`: optional SD1.5 inpainting image prompt object.
- Shape mirrors `sd15.text2img`: `{ "enabled": boolean, "image": ImageRef, "image_embeds": EmbedRef, "mask_image": ImageRef, "scale": number, "model": string, "subfolder": string, "weight_name": string }`.
- Minimal supported default adapter is `model: "h94/IP-Adapter"`, `subfolder: "models"`, `weight_name: "ip-adapter_sd15.bin"`.
- Exactly one of `image` or `image_embeds` is required when `enabled` is `true`.
- `image_embeds` accepts an embed artifact produced by `sd15.ip_adapter.encode`; render loads the IP-Adapter without the image encoder when embeds are provided.
- `mask_image` is optional and separate from the inpaint `mask_image`; the inpaint mask controls repainting while `ip_adapter.mask_image` controls IP-Adapter influence.
- `scale` defaults to `0.6` and must be within `[0, 1]`.
- Minimal initial support is non-ControlNet, non-LCM `sd15.inpaint` only. User-selected SD1.5 LoRA adapters may still be combined with IP-Adapter.

`sdxl.text2img` LoRA input notes:
- `lora_adapters` is optional. When omitted or empty, text2img runs without LoRA adapters.
- `lora_adapters` entries are resolved through the LoRA registry (`/lora-models`) by `lora_id`.
- Each adapter may provide `strength` (default `1.0`), optional per-component overrides (`unet_strength`, `text_encoder_strength`), and optional fine-grained scales (`unet_scales`, `text_encoder_scales`).
- Family validation is enforced: only LoRAs registered with `lora_model_family: "sdxl"` are accepted for `sdxl.text2img`.
- Invalid adapter references (for example missing `lora_id`, unknown id, or incompatible family) fail the task with a validation/runtime error.

`sdxl.text2img` IP-Adapter input notes:
- `ip_adapter`: optional SDXL text-to-image image prompt object.
- Shape: `{ "enabled": boolean, "image": ImageRef, "image_embeds": EmbedRef, "scale": number, "model": string, "subfolder": string, "weight_name": string }`.
- Minimal supported default adapter is `model: "h94/IP-Adapter"`, `subfolder: "sdxl_models"`, `weight_name: "ip-adapter_sdxl.bin"`.
- Exactly one of `image` or `image_embeds` is required when `enabled` is `true`.
- `image` accepts references matching other workflow image inputs (`{"artifact_id":"..."}`, `"@artifact:..."`, or `"/outputs/..."`).
- `image_embeds` accepts an embed artifact produced by `sdxl.ip_adapter.encode`. When `image_embeds` is provided, render loads the IP-Adapter with `image_encoder_folder: null` and uses the precomputed embeds instead of the reference image.
- `scale` defaults to `0.6` and must be within `[0, 1]`; Diffusers uses this to control image-prompt influence.
- Initial support is one SDXL base IP-Adapter for `sdxl.text2img`, `sdxl.img2img`, and `sdxl.inpaint`. FaceID, Plus variants, multiple adapters, and ControlNet combinations are outside the initial contract.

`sdxl.ip_adapter.encode` input notes:
- Creates a temporary SDXL IP-Adapter image-embeds artifact for later use by `sdxl.text2img`.
- The encoder loads only the IP-Adapter weights and CLIP vision image encoder; it does not load the SDXL UNet, VAE, text encoders, tokenizers, or scheduler.
- Minimal encoder v1 supports only the default base SDXL adapter: `ip_adapter_model: "h94/IP-Adapter"`, `ip_adapter_subfolder: "sdxl_models"`, `ip_adapter_weight_name: "ip-adapter_sdxl.bin"`.
- Plus variants, FaceID variants, multiple IP-Adapters, and custom projection formats that require hidden states are rejected by the minimal encoder.
- CUDA is required for the minimal encode step.
- Inputs:
  - `image`: reference image (`ImageRef`).
  - `model`: SDXL base model registry name saved in embed metadata for render-side compatibility checks.
  - `guidance_scale`: render guidance scale; this determines whether classifier-free guidance embeds are created and must match the render step's guidance mode.
  - `ip_adapter_model`, `ip_adapter_subfolder`, `ip_adapter_weight_name`, `ip_adapter_scale`: adapter settings. Defaults mirror `sdxl.text2img`.
- Output:
  - `{ "image_embeds": { "artifact_id": "e...", "path": "artifacts/e....pt", "url": "/outputs/artifacts/e....pt" } }`
- Embed artifacts are ephemeral and cleaned up after the workflow finishes.

Example two-step SDXL IP-Adapter workflow:

```json
{
  "tasks": [
    {
      "id": "ip_embeds",
      "type": "sdxl.ip_adapter.encode",
      "inputs": {
        "image": { "artifact_id": "a0123456789abcdef0123456789abcdef" },
        "model": "stable-diffusion-xl-base-1.0",
        "guidance_scale": 7.5
      }
    },
    {
      "id": "image_render",
      "type": "sdxl.text2img",
      "inputs": {
        "prompt": "portrait photo, cinematic light",
        "negative_prompt": "",
        "model": "stable-diffusion-xl-base-1.0",
        "guidance_scale": 7.5,
        "ip_adapter": {
          "enabled": true,
          "image_embeds": "@ip_embeds.image_embeds",
          "scale": 0.6,
          "model": "h94/IP-Adapter",
          "subfolder": "sdxl_models",
          "weight_name": "ip-adapter_sdxl.bin"
        }
      }
    }
  ],
  "return": "@image_render.images"
}
```

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

`sdxl.img2img` LoRA input notes:
- `lora_adapters` is optional. When omitted or empty, img2img runs without LoRA adapters.
- `lora_adapters` entries are resolved through the LoRA registry (`/lora-models`) by `lora_id`.
- Each adapter may provide `strength` (default `1.0`), optional per-component overrides (`unet_strength`, `text_encoder_strength`), and optional fine-grained scales (`unet_scales`, `text_encoder_scales`).
- Family validation is enforced: only LoRAs registered with `lora_model_family: "sdxl"` are accepted for `sdxl.img2img`.
- Invalid adapter references (for example missing `lora_id`, unknown id, or incompatible family) fail the task with a validation/runtime error.

`sdxl.img2img` IP-Adapter input notes:
- `ip_adapter`: optional SDXL image-to-image image prompt object.
- Shape mirrors `sdxl.text2img`: `{ "enabled": boolean, "image": ImageRef, "scale": number, "model": string, "subfolder": string, "weight_name": string }`.
- Minimal supported default adapter is `model: "h94/IP-Adapter"`, `subfolder: "sdxl_models"`, `weight_name: "ip-adapter_sdxl.bin"`.
- `image` is required when `enabled` is `true`; accepted references match other workflow image inputs (`{"artifact_id":"..."}`, `"@artifact:..."`, or `"/outputs/..."`).
- `scale` defaults to `0.6` and must be within `[0, 1]`.
- Minimal initial support is non-ControlNet `sdxl.img2img` only. User-selected SDXL LoRA adapters may still be combined with IP-Adapter.

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

`sdxl.inpaint` LoRA input notes:
- `lora_adapters` is optional. When omitted or empty, inpaint runs without LoRA adapters.
- `lora_adapters` entries are resolved through the LoRA registry (`/lora-models`) by `lora_id`.
- Each adapter may provide `strength` (default `1.0`), optional per-component overrides (`unet_strength`, `text_encoder_strength`), and optional fine-grained scales (`unet_scales`, `text_encoder_scales`).
- Family validation is enforced: only LoRAs registered with `lora_model_family: "sdxl"` are accepted for `sdxl.inpaint`.
- Invalid adapter references (for example missing `lora_id`, unknown id, or incompatible family) fail the task with a validation/runtime error.

`sdxl.inpaint` IP-Adapter input notes:
- `ip_adapter`: optional SDXL inpainting image prompt object.
- Shape mirrors `sdxl.text2img`: `{ "enabled": boolean, "image": ImageRef, "scale": number, "model": string, "subfolder": string, "weight_name": string }`.
- Minimal supported default adapter is `model: "h94/IP-Adapter"`, `subfolder: "sdxl_models"`, `weight_name: "ip-adapter_sdxl.bin"`.
- `image` is required when `enabled` is `true`; accepted references match other workflow image inputs (`{"artifact_id":"..."}`, `"@artifact:..."`, or `"/outputs/..."`).
- `scale` defaults to `0.6` and must be within `[0, 1]`.
- Minimal initial support is non-ControlNet `sdxl.inpaint` only. User-selected SDXL LoRA adapters may still be combined with IP-Adapter.

`flux.text2img` LoRA input notes:
- `lora_adapters` is optional. When omitted or empty, text2img runs without LoRA adapters.
- `lora_adapters` entries are resolved through the LoRA registry (`/lora-models`) by `lora_id`.
- Each adapter may provide `strength` (default `1.0`), optional per-component overrides (`unet_strength`, `text_encoder_strength`), and optional fine-grained scales (`unet_scales`, `text_encoder_scales`).
- Family validation is enforced: only LoRAs registered with `lora_model_family: "flux"` are accepted for `flux.text2img`.
- Invalid adapter references (for example missing `lora_id`, unknown id, or incompatible family) fail the task with a validation/runtime error.

`flux.img2img` LoRA input notes:
- `lora_adapters` is optional. When omitted or empty, img2img runs without LoRA adapters.
- `lora_adapters` entries are resolved through the LoRA registry (`/lora-models`) by `lora_id`.
- Each adapter may provide `strength` (default `1.0`), optional per-component overrides (`unet_strength`, `text_encoder_strength`), and optional fine-grained scales (`unet_scales`, `text_encoder_scales`).
- Family validation is enforced: only LoRAs registered with `lora_model_family: "flux"` are accepted for `flux.img2img`.
- Invalid adapter references (for example missing `lora_id`, unknown id, or incompatible family) fail the task with a validation/runtime error.

`flux.inpaint` LoRA input notes:
- `lora_adapters` is optional. When omitted or empty, inpaint runs without LoRA adapters.
- `lora_adapters` entries are resolved through the LoRA registry (`/lora-models`) by `lora_id`.
- Each adapter may provide `strength` (default `1.0`), optional per-component overrides (`unet_strength`, `text_encoder_strength`), and optional fine-grained scales (`unet_scales`, `text_encoder_scales`).
- Family validation is enforced: only LoRAs registered with `lora_model_family: "flux"` are accepted for `flux.inpaint`.
- Invalid adapter references (for example missing `lora_id`, unknown id, or incompatible family) fail the task with a validation/runtime error.
- Runtime selects a Fill-compatible Flux inpaint backend automatically when the selected Flux model metadata indicates a Fill variant.

`qwen-image.text2img` LoRA input notes:
- `lora_adapters` is optional. When omitted or empty, text2img runs without LoRA adapters.
- Legacy fallback `Lora` object is accepted as `{ "enabled": boolean, "adapters": [...] }` (also supports legacy `loraStatus`).
  - When `Lora.enabled`/`Lora.loraStatus` is `false`, backend treats LoRA as disabled.
  - If present and enabled, `Lora.adapters` is used when top-level `lora_adapters` is absent.
- `lora_adapters` entries are resolved through the LoRA registry (`/lora-models`) by `lora_id`.
- Each adapter may provide `strength` (default `1.0`).
- Family validation is enforced: only LoRAs registered with `lora_model_family: "qwen-image"` are accepted for `qwen-image.text2img`.
- Invalid adapter references (for example missing `lora_id`, unknown id, or incompatible family) fail the task with a validation/runtime error.

`qwen-image.img2img` LoRA input notes:
- `lora_adapters` is optional. When omitted or empty, img2img runs without LoRA adapters.
- Legacy fallback `Lora` object is accepted as `{ "enabled": boolean, "adapters": [...] }` (also supports legacy `loraStatus`).
  - When `Lora.enabled`/`Lora.loraStatus` is `false`, backend treats LoRA as disabled.
  - If present and enabled, `Lora.adapters` is used when top-level `lora_adapters` is absent.
- `lora_adapters` entries are resolved through the LoRA registry (`/lora-models`) by `lora_id`.
- Each adapter may provide `strength` (default `1.0`).
- Family validation is enforced: only LoRAs registered with `lora_model_family: "qwen-image"` are accepted for `qwen-image.img2img`.
- Invalid adapter references (for example missing `lora_id`, unknown id, or incompatible family) fail the task with a validation/runtime error.

`qwen-image.inpaint` LoRA input notes:
- `lora_adapters` is optional. When omitted or empty, inpaint runs without LoRA adapters.
- Legacy fallback `Lora` object is accepted as `{ "enabled": boolean, "adapters": [...] }` (also supports legacy `loraStatus`).
  - When `Lora.enabled`/`Lora.loraStatus` is `false`, backend treats LoRA as disabled.
  - If present and enabled, `Lora.adapters` is used when top-level `lora_adapters` is absent.
- `lora_adapters` entries are resolved through the LoRA registry (`/lora-models`) by `lora_id`.
- Each adapter may provide `strength` (default `1.0`).
- Family validation is enforced: only LoRAs registered with `lora_model_family: "qwen-image"` are accepted for `qwen-image.inpaint`.
- Invalid adapter references (for example missing `lora_id`, unknown id, or incompatible family) fail the task with a validation/runtime error.

`z-image.text2img` LoRA input notes:
- `lora_adapters` is optional. When omitted or empty, text2img runs without LoRA adapters.
- `lora_adapters` entries are resolved through the LoRA registry (`/lora-models`) by `lora_id`.
- Each adapter may provide `strength` (default `1.0`).
- Family validation is enforced: only LoRAs registered with `lora_model_family: "z-image"` are accepted for `z-image.text2img`.
- Invalid adapter references (for example missing `lora_id`, unknown id, or incompatible family) fail the task with a validation/runtime error.

`z-image.img2img` LoRA input notes:
- `lora_adapters` is optional. When omitted or empty, img2img runs without LoRA adapters.
- `lora_adapters` entries are resolved through the LoRA registry (`/lora-models`) by `lora_id`.
- Each adapter may provide `strength` (default `1.0`).
- Family validation is enforced: only LoRAs registered with `lora_model_family: "z-image"` are accepted for `z-image.img2img`.
- Invalid adapter references (for example missing `lora_id`, unknown id, or incompatible family) fail the task with a validation/runtime error.

`z-image.inpaint` LoRA input notes:
- `lora_adapters` is optional. When omitted or empty, inpaint runs without LoRA adapters.
- `lora_adapters` entries are resolved through the LoRA registry (`/lora-models`) by `lora_id`.
- Each adapter may provide `strength` (default `1.0`).
- Family validation is enforced: only LoRAs registered with `lora_model_family: "z-image"` are accepted for `z-image.inpaint`.
- Invalid adapter references (for example missing `lora_id`, unknown id, or incompatible family) fail the task with a validation/runtime error.

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
