# Nucleus-MoE Diffusers Implementation Guide for SynthaEngine

Date: 2026-05-30

Selected model: Nucleus-MoE / Nucleus-Image  
Primary Hub repository: `NucleusAI/Nucleus-Image`  
Older/alias name: `NucleusAI/NucleusMoE-Image` redirects to `NucleusAI/Nucleus-Image`

This guide explains what Nucleus-MoE is, how practical it is on a PC with
64 GB system RAM and an RTX 3060 with 12 GB VRAM, and how to implement it in
SynthaEngine later. No application code has been changed by this guide.

## 1. Executive Summary

Nucleus-MoE, now published on Hugging Face as `NucleusAI/Nucleus-Image`, is a
text-to-image diffusion model built around a sparse Mixture-of-Experts diffusion
transformer. The model has 17B total parameters, but only about 2B parameters
are active for a forward pass. That makes compute more efficient than a dense
17B image model, but it does not make the full model small.

The practical answer for your PC:

- Your installed environment already has `diffusers 0.38.0`, `transformers
  5.8.0`, `torch 2.10.0+cu128`, CUDA available, and BF16 support on the RTX
  3060.
- The local import check passed for `NucleusMoEImagePipeline`,
  `NucleusMoEImageTransformer2DModel`, and `TextKVCacheConfig`.
- The official BF16 model is not a comfortable local fit for RTX 3060 12 GB.
  The Hub file tree reports about 51.7 GB total model files, including a
  33.8 GB transformer and a 17.5 GB text encoder.
- 64 GB system RAM helps, but it is still tight once Windows, Python, model
  loading, CPU offload buffers, and Hugging Face cache overhead are included.
- Recommended local path: treat Nucleus-MoE as experimental, run a standalone
  smoke test first, use one-shot subprocesses, one image per job, conservative
  resolution, and CPU offload. Expect slow generation.
- Recommended production path on this hardware: use a hosted provider such as
  fal for Nucleus jobs, or wait for an official smaller/quantized variant that
  is proven under 12 GB VRAM.

Do not implement it as a quiet drop-in replacement for Qwen-Image or Flux.
Add it as a separate experimental family with explicit memory warnings.

## 2. What Nucleus-MoE Is

Nucleus-MoE is the model family behind `NucleusAI/Nucleus-Image`. The Diffusers
documentation describes `NucleusMoEImagePipeline` as a text-to-image pipeline
using:

- a single-stream diffusion transformer,
- Mixture-of-Experts feed-forward layers,
- cross-attention to a Qwen3-VL text encoder,
- an `AutoencoderKLQwenImage` VAE,
- a `FlowMatchEulerDiscreteScheduler`.

The model card describes Nucleus-Image as a 17B sparse MoE diffusion transformer
with about 2B active parameters per forward pass. Its architecture details are
important for memory planning:

| Item | Value |
| --- | --- |
| Total parameters | 17B |
| Active parameters | About 2B |
| Architecture | Sparse MoE diffusion transformer |
| Layers | 32 |
| Hidden dimension | 2048 |
| Attention heads | 16 query heads, 4 KV heads |
| Experts per MoE layer | 64 routed experts plus 1 shared expert |
| Text encoder | Qwen3-VL-8B-Instruct |
| Image tokenizer | Qwen-Image VAE, 16 channels |
| Scheduler | FlowMatch Euler discrete scheduler |
| License | Apache 2.0 |

The core idea:

- Dense model: all parameters in each layer tend to participate.
- Sparse MoE model: a router selects a subset of experts, so active compute is
  much lower than the total parameter count.
- Important caveat: inactive experts still exist in the checkpoint and must be
  stored, loaded, offloaded, or streamed. Sparse compute does not eliminate
  weight memory.

## 3. Current Diffusers Support

Nucleus-MoE is now a first-class Diffusers pipeline.

Verified locally in this repo virtual environment:

```powershell
.venv\Scripts\python.exe -c "import diffusers; print(diffusers.__version__); print(hasattr(diffusers, 'NucleusMoEImagePipeline')); print(hasattr(diffusers, 'NucleusMoEImageTransformer2DModel'))"
```

Result:

```text
0.38.0
True
True
```

The pipeline call signature in the local environment includes:

- `prompt`
- `negative_prompt`
- `guidance_scale`
- `height`
- `width`
- `num_inference_steps`
- `num_images_per_prompt`
- `max_sequence_length`
- `generator`
- `output_type`
- `callback_on_step_end`

The official model card quick start uses:

```python
import torch
from diffusers import DiffusionPipeline
from diffusers import TextKVCacheConfig

model_name = "NucleusAI/Nucleus-Image"
pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.transformer.enable_cache(TextKVCacheConfig())
```

For your 12 GB GPU, do not use `pipe.to("cuda")` as the implementation default.
That approach assumes far more VRAM than the RTX 3060 has.

## 4. Hardware Fit on 64 GB RAM + RTX 3060 12 GB

Your machine is capable enough to experiment, but not enough for a smooth
official-BF16 local Nucleus workflow.

Observed local environment:

```text
GPU: NVIDIA GeForce RTX 3060
VRAM: 12,884,377,600 bytes reported by PyTorch
CUDA available: yes
BF16 supported: yes
diffusers: 0.38.0
torch: 2.10.0+cu128
transformers: 5.8.0
accelerate: 1.13.0
```

Hub file sizes:

| Component | Approximate size |
| --- | ---: |
| Full repository | 51.7 GB |
| Transformer | 33.8 GB |
| Text encoder | 17.5 GB |
| VAE | 254 MB |
| Processor | 11.5 MB |
| Scheduler | 482 bytes |

### Viability Matrix

| Mode | Expected on RTX 3060 12 GB | Recommendation |
| --- | --- | --- |
| Official BF16 with `pipe.to("cuda")` | Not viable | Do not use. Weights exceed VRAM before activations. |
| Official BF16 with `enable_model_cpu_offload()` | May load, likely very slow, RAM tight | Smoke test only. Use one image and low settings. |
| Official BF16 with `enable_sequential_cpu_offload()` | More likely to reduce VRAM, very slow | Safest local experiment, not a good default UX. |
| Pipeline `device_map="cuda"` | Not viable | Avoid on 12 GB. |
| Pipeline `device_map="balanced"` | Not useful on one GPU | Only helps with multiple GPUs. |
| Generic bitsandbytes or torchao quantization | Uncertain | Nucleus MoE experts may not be fully covered by generic linear-only quantizers. Validate before relying on it. |
| Community FP8 variant | Still above 12 GB at 1024x1024 based on its card | Interesting, but not a first implementation target. |
| Hosted fal provider | Practical | Best path if you want reliable Nucleus output now. |

### Why 2B Active Parameters Still Does Not Fit

The model may activate only about 2B parameters per forward pass, but the full
set of experts and the Qwen3-VL text encoder still need to live somewhere. On a
single RTX 3060, that "somewhere" becomes CPU RAM, disk cache, or a remote GPU.

The key implementation rule:

> Treat Nucleus-MoE as a large offloaded pipeline, not as a compact 2B model.

## 5. Recommended First Local Smoke Test

Before changing SynthaEngine code, create a temporary standalone script later
and run it outside the app. The purpose is to answer one question: can this PC
load and complete a single small Nucleus image without crashing?

Suggested future file:

- `tools/smoke_nucleus_moe.py`

Suggested first settings:

- width: `512`
- height: `512`
- steps: `8`
- guidance_scale: `4.0`
- num_images_per_prompt: `1`
- output_type: `pil`
- memory mode: `sequential_offload`
- Text KV cache: on for one prompt

Smoke-test sketch:

```python
import torch
from diffusers import DiffusionPipeline, TextKVCacheConfig

model_id = "NucleusAI/Nucleus-Image"

pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)

if getattr(pipe, "transformer", None) is not None and hasattr(pipe.transformer, "enable_cache"):
    pipe.transformer.enable_cache(TextKVCacheConfig())

if getattr(pipe, "vae", None) is not None:
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

pipe.enable_sequential_cpu_offload()

generator = torch.Generator(device="cpu").manual_seed(42)
image = pipe(
    prompt="a clean product photo of a matte black ceramic mug on a walnut desk",
    width=512,
    height=512,
    num_inference_steps=8,
    guidance_scale=4.0,
    generator=generator,
).images[0]
image.save("nucleus_moe_smoke.png")
```

Run later with:

```powershell
.venv\Scripts\python.exe tools\smoke_nucleus_moe.py
```

If this fails with out-of-memory or process termination, do not wire the model
into the UI as a local pipeline. Use a hosted provider path or postpone until a
proven low-VRAM model variant exists.

## 6. Implementation Pattern for New Diffusers Pipelines

Use this general checklist whenever a newly released Diffusers pipeline is added
to SynthaEngine.

1. Identify the official source.
   - Prefer Hugging Face model cards, Diffusers docs, and Diffusers source.
   - Confirm the repository has a `model_index.json`.
   - Confirm the pipeline class is first-class Diffusers or requires a custom
     pipeline.

2. Verify local imports.
   - Use the repo venv only: `.venv\Scripts\python.exe`.
   - Check `diffusers.__version__`.
   - Import the pipeline and any model classes.
   - Inspect the pipeline `__call__` signature.

3. Classify the task.
   - Text-to-image, image-to-image, inpaint, text-to-video, image-to-video,
     text-to-text, utility task, or adapter task.
   - Do not reuse existing task identifiers unless behavior is the same.

4. Estimate memory before implementation.
   - Check Hub file sizes.
   - Identify text encoder size, transformer size, VAE size, and whether the
     model is sharded.
   - Compare model weight size with VRAM and system RAM.
   - Decide whether local inference is first-class, experimental, or hosted-only.

5. Create a standalone smoke test.
   - No API, no frontend, no workflow integration.
   - One prompt, one output, conservative settings.
   - Record memory mode and runtime behavior.

6. Design the workflow task contract.
   - Add a new family if behavior is distinct.
   - Keep inputs minimal and stable.
   - Add optional fields with safe defaults.
   - Preserve output shapes used by similar task types.

7. Implement runtime in a family package.
   - Use subprocess boundaries for large models.
   - Release hooks and memory in `finally`.
   - Keep model loading, generation, and output writing isolated.

8. Wire the workflow adapter.
   - Add a thin `backend/workflow/<family>.py`.
   - Register input and output schemas.
   - Register the handler in `backend/workflow/engine.py`.
   - Add catalog family metadata.

9. Align docs and tests.
   - Update `docs/WORKFLOW_API.md`.
   - Update `docs/PIPELINE_LIFECYCLE.md`.
   - Add mocked workflow tests.
   - Add catalog capability tests.
   - Avoid downloading real large models in automated tests.

10. Add frontend only after backend behavior is proven.
    - Use static HTML/JS.
    - Prefer `/api/workflow/catalog` defaults.
    - Keep controls aligned with schema.

## 7. SynthaEngine Implementation Plan for Nucleus-MoE

This section is the concrete future implementation path. It is intentionally
designed to fit the current application patterns.

### Step 1: Add a New Family Name

Recommended family key:

```text
nucleus-moe
```

Recommended aliases:

```text
nucleus-image
nucleusmoe
nucleus
```

Recommended initial task type:

```text
nucleus-moe.text2img
```

Why not use `qwen-image.text2img`?

- Nucleus uses Qwen components, but it is its own MoE transformer family.
- Memory profile and defaults are different.
- The public task identifier should not hide that it is a different model.

### Step 2: Add a Model Registry Fallback

Create a Nucleus-specific model resolver later. Do not use the generic
`get_model_entry(model_name)` fallback blindly, because if there is no Nucleus
entry it may return the first registered model from another family.

Recommended fallback:

```text
name: Nucleus-Image
family: nucleus-moe
model_type: diffusers
location_type: hub
link: NucleusAI/Nucleus-Image
```

Implementation note:

- If `model` is supplied, resolve exact registry entry and validate
  `family == "nucleus-moe"`.
- If `model` is omitted, first look for the lowest `model_id` in the
  `nucleus-moe` family.
- If none exists, fall back to `NucleusAI/Nucleus-Image`.

### Step 3: Add Workflow Input Schema

Future file:

- `backend/workflow/schema_input.py`

Suggested schema:

```python
class NucleusMoeText2ImgInputs(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    steps: int = Field(default=8, ge=1, le=50)
    guidance_scale: float = Field(default=4.0, ge=0.0, le=30.0)
    width: int = Field(default=512, ge=256, le=1344)
    height: int = Field(default=512, ge=256, le=1344)
    seed: int | None = None
    model: str | None = None
    num_images: int = Field(default=1, ge=1, le=1)
    max_sequence_length: int | None = Field(default=None, ge=64, le=1024)
    memory_preset: Literal["sequential_offload", "model_offload"] = "sequential_offload"
    enable_text_kv_cache: bool = True
    experimental_ack: bool = True
```

Recommended defaults for your PC:

```json
{
  "steps": 8,
  "guidance_scale": 4.0,
  "width": 512,
  "height": 512,
  "num_images": 1,
  "memory_preset": "sequential_offload",
  "enable_text_kv_cache": true,
  "experimental_ack": true
}
```

Add stricter validation:

- Require `experimental_ack == true`.
- Limit `num_images` to `1`.
- Consider allowing only known aspect ratios once local viability is proven.

### Step 4: Use Existing Image Output Shape

Future file:

- `backend/workflow/schema_output.py`

Nucleus is text-to-image, so reuse:

```python
ImagesOutput
```

Expected task result:

```json
{
  "images": ["/outputs/<batch>/<file>.png"]
}
```

If you want batch metadata later, use `ImagesWithBatchOutput`, but the current
large-model families mostly use `ImagesOutput` for image lists.

### Step 5: Add a Thin Workflow Adapter

Future file:

- `backend/workflow/nucleus_moe.py`

Pattern:

```python
from __future__ import annotations

from typing import Any


def run_nucleus_moe_text2img_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    generate_text2img = deps["generate_text2img"]
    result = generate_text2img(dict(inputs))
    if not isinstance(result, dict):
        raise ValueError("nucleus-moe.text2img must return an object")
    if not isinstance(result.get("images"), list):
        raise ValueError("nucleus-moe.text2img must return images")
    return result
```

Keep it boring. The workflow adapter should validate shape and delegate runtime
work to `backend/nucleus_moe/pipeline.py`.

### Step 6: Add a Runtime Package

Future files:

- `backend/nucleus_moe/__init__.py`
- `backend/nucleus_moe/pipeline.py`
- `backend/nucleus_moe/subprocess_runner.py`

Optional if image tasks are added later:

- `backend/nucleus_moe/subprocess_io.py`

Why subprocess-backed:

- The model is too large to trust in the long-running API process.
- This repo already uses one-shot subprocesses for SD1.5, SDXL, Flux,
  Qwen-Image, Z-Image, ERNIE-Image, and WAN.
- Process exit is the strongest cleanup boundary on Windows.

Runtime responsibilities:

1. Resolve model source.
2. Load `DiffusionPipeline.from_pretrained(..., torch_dtype=torch.bfloat16)`.
3. Enable `TextKVCacheConfig` when requested and available.
4. Enable VAE slicing and tiling when available.
5. Apply selected offload mode before generation.
6. Generate one image at a time.
7. Save PNG output and metadata.
8. Release hooks and memory in `finally`.

Pseudo-code sketch:

```python
@torch.inference_mode()
def generate_text2img_in_process(params: dict[str, object]) -> dict[str, list[str]]:
    prompt = str(params.get("prompt") or "")
    negative_prompt = str(params.get("negative_prompt") or "").strip()
    steps = int(params.get("steps") or 8)
    guidance_scale = float(params.get("guidance_scale") or 4.0)
    width = int(params.get("width") or 512)
    height = int(params.get("height") or 512)
    memory_preset = str(params.get("memory_preset") or "sequential_offload")

    pipe = None
    try:
        pipe = DiffusionPipeline.from_pretrained(
            resolve_nucleus_moe_source(params.get("model")),
            torch_dtype=torch.bfloat16,
        )

        if bool(params.get("enable_text_kv_cache", True)):
            if getattr(pipe, "transformer", None) is not None and hasattr(pipe.transformer, "enable_cache"):
                pipe.transformer.enable_cache(TextKVCacheConfig())

        if getattr(pipe, "vae", None) is not None:
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()

        if memory_preset == "model_offload":
            pipe.enable_model_cpu_offload()
        else:
            pipe.enable_sequential_cpu_offload()

        call_kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": torch.Generator(device="cpu").manual_seed(seed),
        }
        if negative_prompt:
            call_kwargs["negative_prompt"] = negative_prompt

        image = pipe(**call_kwargs).images[0]
        save_image_and_return_output(image)
    finally:
        release_pipeline(pipe, logger=logger)
```

Important caution:

- Do not call `.to("cuda")` after selecting offload mode.
- For sequential offload, do not move the pipeline to CUDA first.
- Do not batch multiple images on 12 GB VRAM.
- Keep `generator` on CPU unless a smoke test proves CUDA generator is required.

### Step 7: Register in the Workflow Engine

Future file:

- `backend/workflow/engine.py`

Add imports:

```python
from backend.workflow.schema_input import NucleusMoeText2ImgInputs
from backend.workflow.nucleus_moe import run_nucleus_moe_text2img_task as _run_nucleus_moe_text2img
```

Add to `TASK_INPUT_MODELS`:

```python
"nucleus-moe.text2img": NucleusMoeText2ImgInputs,
```

Add to `TASK_OUTPUT_MODELS`:

```python
"nucleus-moe.text2img": ImagesOutput,
```

Add runtime deps:

```python
def _nucleus_moe_runtime_deps() -> dict[str, Any]:
    nucleus_moe_pipeline_module = importlib.import_module("backend.nucleus_moe.pipeline")
    return {
        "generate_text2img": nucleus_moe_pipeline_module.generate_text2img,
    }
```

Add handler:

```python
def _nucleus_moe_text2img(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    return _run_nucleus_moe_text2img(inputs, _nucleus_moe_runtime_deps())
```

Add to `TASK_REGISTRY`:

```python
"nucleus-moe.text2img": _nucleus_moe_text2img,
```

### Step 8: Update Catalog Capabilities

Future file:

- `backend/workflow/catalog.py`

Add metadata:

```python
"nucleus-moe": {"label": "Nucleus-MoE", "aliases": ["nucleus-image", "nucleusmoe", "nucleus"]},
```

Update `_infer_model_family`:

```python
if prefix == "nucleus-moe":
    return "nucleus-moe"
```

Expected capability result:

```json
{
  "nucleus-moe": {
    "label": "Nucleus-MoE",
    "aliases": ["nucleus-image", "nucleusmoe", "nucleus"],
    "task_types": ["nucleus-moe.text2img"],
    "features": {
      "text2img": true,
      "scheduler": false,
      "lora_adapters": false,
      "ip_adapter": false
    }
  }
}
```

Initial contract should not expose LoRA, ControlNet, img2img, inpaint, or
scheduler swaps. Add those only after the base text-to-image path is proven.

### Step 9: Update API and Lifecycle Docs

Future files:

- `docs/WORKFLOW_API.md`
- `docs/PIPELINE_LIFECYCLE.md`

Add Nucleus to:

- project family list,
- supported task types,
- capability matrix,
- task input notes,
- example workflow,
- subprocess-backed family list.

Example workflow:

```json
{
  "kind": "workflow",
  "payload": {
    "tasks": [
      {
        "id": "t1",
        "type": "nucleus-moe.text2img",
        "inputs": {
          "prompt": "a quiet alpine lake at sunrise, soft pink and gold sky",
          "width": 512,
          "height": 512,
          "steps": 8,
          "guidance_scale": 4.0,
          "memory_preset": "sequential_offload",
          "experimental_ack": true
        }
      }
    ],
    "return": "@t1.images"
  }
}
```

Docs should state:

- local Nucleus-MoE is experimental on 12 GB VRAM,
- first-run download is very large,
- one image per job is the only supported initial contract,
- sequential CPU offload is safest and slowest,
- hosted inference is recommended for reliable use.

### Step 10: Add Focused Tests

Do not download Nucleus in automated tests.

Add tests that mock runtime dependencies:

- `nucleus-moe.text2img` appears in task types.
- Catalog exposes `nucleus-moe`.
- Minimal schema validates.
- Invalid output from adapter raises a clear error.
- Workflow engine dispatch works with mocked `generate_text2img`.
- Docs contract test includes the task type if the docs test enforces lists.

Likely files:

- `testing/test_workflow_catalog_capabilities.py`
- new `testing/test_nucleus_moe_workflow.py`
- optional `testing/test_nucleus_moe_subprocess.py` with no real model load

Required validation for a real implementation:

```powershell
.venv\Scripts\python.exe -m compileall backend
.venv\Scripts\python.exe -m pytest testing/test_workflow_catalog_capabilities.py -q
.venv\Scripts\python.exe -m pytest testing/test_*workflow*.py -q
```

### Step 11: Add Frontend Only After Backend Success

Future files:

- `frontend/nucleus_moe/text2img.html`
- `frontend/nucleus_moe/text2img.js`

Frontend principles:

- Static HTML/JS only.
- Submit `kind: "workflow"` jobs.
- Poll and stream job state using existing workflow helpers.
- Render output with the existing gallery component if compatible.
- Make experimental status visible.
- Use compact controls:
  - prompt textarea,
  - negative prompt textarea,
  - aspect ratio or width/height controls,
  - steps numeric input,
  - guidance scale input,
  - memory preset select,
  - seed input.

Recommended initial UI guardrail:

- Disable generation until `experimental_ack` is checked.
- Cap the UI to 512x512 and 8 steps unless the user opens advanced settings.

## 8. Optional Hosted Provider Path

If the goal is to use Nucleus reliably now on this PC, a hosted provider is the
most practical path. The Hugging Face model page lists fal as an inference
provider, and the Nucleus website also presents fal-backed generation.

Hosted implementation shape:

- Add `provider: "local" | "fal"` to the future task input only if you are
  comfortable adding non-local execution.
- Keep local and hosted code paths explicit.
- Store API keys outside the repo.
- Return the same `ImagesOutput` shape by downloading the provider result into
  `outputs/` before returning the workflow result.

This is a behavior and deployment policy change, so it should be a separate
approved implementation plan.

## 9. Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Official BF16 model is about 51.7 GB on Hub | Long download and high disk/RAM pressure | Require a smoke test and document disk needs. |
| RTX 3060 has only 12 GB VRAM | OOM if loaded directly | Never default to `pipe.to("cuda")`; use offload only. |
| 64 GB system RAM is tight | Process termination or heavy paging | One-shot subprocess, low resolution, one image, no batching. |
| Sequential offload is very slow | Poor UX | Mark local Nucleus as experimental; prefer hosted path for regular use. |
| Generic quantization may miss MoE expert tensors | False confidence in memory savings | Validate real memory before exposing quantized mode. |
| Text KV cache state may be prompt-shape sensitive | Incorrect reuse across prompts | Create a fresh subprocess/pipeline per task. |
| Public API task name becomes permanent | Compatibility burden | Choose `nucleus-moe.text2img` carefully and keep initial surface small. |
| Model output has no safety checker | Public-facing risk | Add deployment policy before exposing beyond local use. |

## 10. Definition of Done for Future Implementation

A Nucleus-MoE implementation should be considered done only when:

- A standalone smoke test completes on the target PC or the feature is explicitly
  marked hosted-only.
- `nucleus-moe.text2img` appears in `/api/workflow/task-types`.
- `/api/workflow/catalog` includes `nucleus-moe` capabilities and schema.
- A minimal workflow job returns `{ "images": ["/outputs/..."] }`.
- Runtime uses a one-shot subprocess by default.
- Runtime cleanup follows `docs/PIPELINE_LIFECYCLE.md`.
- Docs include the task contract and hardware warning.
- Tests cover schema, catalog, and mocked workflow dispatch without downloading
  the model.
- Backend compiles with `.venv\Scripts\python.exe -m compileall backend`.

## 11. Bottom Line

Nucleus-MoE is real Diffusers support, and this repo can import the pipeline
today. The model is impressive because sparse MoE reduces active compute, but it
is still a very large model stack. On an RTX 3060 12 GB machine, local execution
should be treated as experimental and probably slow.

The safest SynthaEngine path is:

1. Run a standalone local smoke test.
2. If it works, add `nucleus-moe.text2img` as an experimental subprocess-backed
   workflow family with 512x512, 8-step defaults.
3. If it does not work, implement a hosted provider path or wait for an
   official low-VRAM variant.

## Sources

- Hugging Face model card: https://huggingface.co/NucleusAI/Nucleus-Image
- Hugging Face file tree: https://huggingface.co/NucleusAI/Nucleus-Image/tree/main
- Diffusers NucleusMoE pipeline docs: https://huggingface.co/docs/diffusers/main/api/pipelines/nucleusmoe_image
- Diffusers pipeline loading docs: https://huggingface.co/docs/diffusers/using-diffusers/loading
- Diffusers memory reduction docs: https://huggingface.co/docs/diffusers/main/en/optimization/memory
- Diffusers quantization overview: https://huggingface.co/docs/diffusers/main/quantization/overview
- Diffusers torchao quantization docs: https://huggingface.co/docs/diffusers/main/quantization/torchao
- Nucleus-Image arXiv paper: https://arxiv.org/abs/2604.12163
- Community FP8 reference, not recommended as first target for 12 GB VRAM: https://huggingface.co/D-Squarius-Green-Jr/Nucleus-Image-FP8
