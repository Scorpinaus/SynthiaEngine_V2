# ERNIE-Image Diffusers Implementation Guide for SynthaEngine

Date: 2026-05-30

Selected model architecture: ERNIE-Image  
Primary Hub repositories: `baidu/ERNIE-Image`, `baidu/ERNIE-Image-Turbo`  
Primary Diffusers pipeline: `ErnieImagePipeline`

This guide explains what ERNIE-Image is, how practical it is on a local PC with
64 GB system RAM and an RTX 3060 with 12 GB VRAM, and how to integrate or
complete the integration in SynthaEngine. No existing application files were
changed by this research note.

## 1. Executive Summary

ERNIE-Image is Baidu's open text-to-image model family in Diffusers. It is an
8B-parameter, single-stream diffusion transformer model with a multilingual
prompt stack and optional Prompt Enhancer. Diffusers exposes it through
`ErnieImagePipeline`.

The practical answer for this PC:

- Yes, it is feasible as a constrained local workflow if you use
  `baidu/ERNIE-Image-Turbo`, one image per job, conservative resolution,
  BF16, CPU offload, VAE tiling/slicing, Prompt Enhancer disabled by default,
  and a short-lived subprocess so Windows can reclaim RAM after each render.
- No, it is not a comfortable "load everything and batch at full quality" fit
  for 12 GB VRAM. The model card recommends consumer GPUs with 24 GB VRAM, and
  the full BF16 model footprint is much larger than the RTX 3060 VRAM budget.
- Quantization is possible to investigate, but it should be treated as an
  experimental path for this repo. The current environment has
  `bitsandbytes 0.49.2` and Diffusers quantization classes, but it does not
  have `torchao` or `gguf` installed. The existing ERNIE-Image contract does
  not expose quantization yet.
- Hosted execution is a good fallback. Use Hugging Face Inference Providers
  through fal for a managed API path, or use GPU rental services such as
  RunPod, Lambda Cloud, Vast.ai, or a dedicated Hugging Face Inference Endpoint
  with 24 GB or larger GPUs.

Important current-repo finding:

- SynthaEngine already contains ERNIE-Image support in
  `backend/ernie_image/`, `backend/workflow/ernie_image.py`,
  `frontend/ernie_image/`, `docs/WORKFLOW_API.md`, and focused tests.
- Treat this document as both a step-by-step implementation guide and a gap
  checklist for future Diffusers ERNIE-Image upgrades.

Official references:

- Diffusers ERNIE-Image docs:
  https://huggingface.co/docs/diffusers/main/api/pipelines/ernie_image
- Diffusers pipeline overview:
  https://huggingface.co/docs/diffusers/main/api/pipelines/overview
- ERNIE-Image Hub model:
  https://huggingface.co/baidu/ERNIE-Image
- ERNIE-Image-Turbo Hub model:
  https://huggingface.co/baidu/ERNIE-Image-Turbo
- Diffusers quantization overview:
  https://huggingface.co/docs/diffusers/main/api/quantization
- Diffusers bitsandbytes quantization:
  https://huggingface.co/docs/diffusers/main/quantization/bitsandbytes

## 2. What ERNIE-Image Is

ERNIE-Image is a text-to-image diffusion model family. In Diffusers, the
pipeline accepts text prompts and produces images by denoising latent image
representations through a diffusion transformer, then decoding the final latent
with a VAE.

The model family is intended for high-resolution, prompt-following image
generation. The official Hub card describes ERNIE-Image as a strong
text-to-image foundation model with multilingual understanding, prompt
enhancement, and an open Apache 2.0 license.

Key architecture points:

| Area | ERNIE-Image detail |
| --- | --- |
| Main task | Text to image |
| Main Diffusers class | `ErnieImagePipeline` |
| Core denoiser | `ErnieImageTransformer2DModel` |
| Transformer type | Single-stream diffusion transformer |
| Parameter scale | About 8B parameters |
| Text encoder | Mistral-3 style text encoder in Diffusers |
| Optional prompt enhancer | Ministral-3 style causal LM Prompt Enhancer |
| VAE | `AutoencoderKLFlux2` |
| Scheduler | `FlowMatchEulerDiscreteScheduler` |
| Default precision | BF16 in official examples |
| License | Apache 2.0 |

The mental model:

1. User submits a text prompt.
2. Optional Prompt Enhancer rewrites or enriches the prompt.
3. Text encoder turns prompt text into embeddings.
4. Diffusion transformer denoises image latents over a fixed number of steps.
5. VAE decodes latents into a final PIL image.
6. SynthaEngine stores the image under `/outputs/...` and returns image paths.

## 3. Modalities And Sub-Pipelines

The official Diffusers ERNIE-Image integration is text-to-image only.

| Modality or sub-pipeline | Available in official Diffusers ERNIE-Image pipeline? | Notes |
| --- | --- | --- |
| Text to image | Yes | `ErnieImagePipeline` |
| Image to image | No | No official `ErnieImageImg2ImgPipeline` currently exposed |
| Inpaint | No | No official ERNIE-Image inpainting pipeline currently exposed |
| ControlNet | No | No official ERNIE-Image ControlNet pipeline currently exposed |
| IP-Adapter | No | Not part of the official pipeline contract |
| Text to video | No | ERNIE-Image is an image model, not a video pipeline |
| Prompt enhancement | Yes | Optional `use_pe`; requires PE components to be loaded |
| LoRA adapters | Technically possible | Diffusers pipelines generally support LoRA loading when target modules match; SynthaEngine already exposes ERNIE-Image LoRA adapter selection |
| Quantized loading | Experimental | Diffusers supports quantization generally; ERNIE-specific local reliability must be tested |

Released checkpoints to handle:

| Checkpoint | Hub ID | Intended local default | Typical settings |
| --- | --- | --- | --- |
| ERNIE-Image-Turbo | `baidu/ERNIE-Image-Turbo` | Yes | 8 steps, `guidance_scale=1.0`, PE off for first local smoke tests |
| ERNIE-Image | `baidu/ERNIE-Image` | Optional/heavier | 50 steps, stronger CFG, slower and less practical on 12 GB VRAM |

Recommended SynthaEngine policy:

- Default to `ERNIE-Image-Turbo`.
- Keep the base `ERNIE-Image` checkpoint selectable through the model registry,
  but mark it heavier.
- Keep ERNIE as a separate family, not a replacement for Qwen-Image, Z-Image,
  Flux, SDXL, or SD 1.5.

## 4. Feasibility On 64 GB RAM And RTX 3060 12 GB

### Current Environment Snapshot

A read-only environment check in this repo's virtual environment reported:

```text
torch=2.10.0+cu128
diffusers=0.38.0
transformers=5.8.0
accelerate=1.13.0
bitsandbytes=0.49.2
torchao=NOT_AVAILABLE
gguf=NOT_AVAILABLE
cuda_available=True
cuda_device=NVIDIA GeForce RTX 3060
cuda_capability=(8, 6)
bf16_supported=True
total_vram_gb=12.00
diffusers_imports=OK ErnieImagePipeline PipelineQuantizationConfig TorchAoConfig BitsAndBytesConfig
```

This is a good starting point for ERNIE-Image-Turbo because:

- CUDA is visible.
- BF16 is supported.
- Diffusers `ErnieImagePipeline` imports successfully.
- `bitsandbytes` is installed for quantization experiments.
- The app already uses subprocess-backed large model execution.

The main constraint is VRAM. A 12 GB RTX 3060 cannot comfortably host all
components of an 8B BF16 image model and its runtime activations. 64 GB system
RAM helps with CPU offload, but it does not make local generation fast.

### Yes Path: Feasible With Constraints

Use this profile for the first real local smoke test:

```json
{
  "model": "ERNIE-Image-Turbo",
  "width": 768,
  "height": 768,
  "steps": 8,
  "guidance_scale": 1.0,
  "num_images": 1,
  "use_pe": false,
  "load_pe": false,
  "memory_preset": "sequential_offload"
}
```

Why this profile is plausible:

- Turbo uses far fewer inference steps than the base model.
- `guidance_scale=1.0` avoids classifier-free guidance duplication.
- Disabling PE avoids loading the extra prompt-enhancer language model.
- Sequential CPU offload reduces VRAM pressure at the cost of speed.
- VAE tiling and slicing reduce decode spikes.
- A subprocess lets the OS fully reclaim memory after the job exits.

Expected tradeoffs:

- Generation may be slow.
- First run may spend time loading model files.
- Large prompts, PE, high resolution, and model-offload mode can trigger OOM.
- Windows pagefile pressure may become noticeable.

### No Path: Not Comfortable Locally

Avoid presenting these as reliable on RTX 3060 12 GB:

- Full BF16 base `baidu/ERNIE-Image` at 1024 px or larger.
- Prompt Enhancer loaded and enabled by default.
- Multiple images per prompt in one job.
- Keeping the full pipeline hot in the web server process.
- `model_offload` as the default memory preset.
- Experimental quantization exposed as a user-facing production feature before
  it has a measured pass in this repo.

For quality-focused ERNIE work, prefer 24 GB VRAM minimum. For PE-heavy,
parallel, or production workloads, prefer 48 GB to 80 GB GPUs.

## 5. Quantization Options

Quantization can reduce memory by storing selected model components in lower
precision, such as 8-bit or 4-bit weights. Diffusers now exposes general
quantization helpers, including `PipelineQuantizationConfig`, `BitsAndBytesConfig`,
`TorchAoConfig`, and GGUF-oriented loading paths.

### Current Local Status

| Option | Current status in this repo | Recommendation |
| --- | --- | --- |
| BF16 | Available | Keep as default precision |
| bitsandbytes 8-bit | Installed | Good first experiment |
| bitsandbytes 4-bit | Installed | Possible, but validate image quality and component support |
| torchao int8/int4 | Not installed | Add only after approval and isolated testing |
| torchao fp8 | Not recommended here | RTX 3060 compute capability 8.6 is below the usual practical FP8 target |
| GGUF | Not installed | Treat as separate research path |
| Pre-quantized Hub variants | Possible | Useful if a trusted ERNIE-Image quantized repo exists and loads cleanly |

### Yes, Quantization Is Possible To Explore

Example research-only loading idea:

```python
import torch
from diffusers import BitsAndBytesConfig, ErnieImagePipeline, PipelineQuantizationConfig

quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
    },
    # Start with the largest module. Add other components only after import tests.
    components_to_quantize=["transformer"],
)

pipe = ErnieImagePipeline.from_pretrained(
    "baidu/ERNIE-Image-Turbo",
    torch_dtype=torch.bfloat16,
    quantization_config=quant_config,
)
pipe.enable_model_cpu_offload()
```

If direct pipeline quantization is unreliable for this model, test component
loading instead:

```python
import torch
from diffusers import BitsAndBytesConfig, ErnieImagePipeline
from diffusers.models import ErnieImageTransformer2DModel

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

transformer = ErnieImageTransformer2DModel.from_pretrained(
    "baidu/ERNIE-Image-Turbo",
    subfolder="transformer",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)

pipe = ErnieImagePipeline.from_pretrained(
    "baidu/ERNIE-Image-Turbo",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
    pe=None,
    pe_tokenizer=None,
)
pipe.enable_model_cpu_offload()
```

### No, Do Not Ship Quantization By Default Yet

Reasons to keep quantization behind an experimental flag:

- The current app contract says quantization is outside initial ERNIE-Image
  support.
- `torchao` and `gguf` are not installed in the repo environment.
- Quantized component support can vary by model class and Diffusers version.
- Quantization may reduce image quality or break LoRA adapter compatibility.
- A broken quantization path should not block the normal BF16/offload path.

Recommended implementation policy:

- Add no public `quantization` field until a local smoke test passes.
- When added, use a strict enum: `"none"`, `"bnb_8bit"`, `"bnb_4bit"`.
- Keep `"none"` as default.
- Emit the selected quantization mode in PNG metadata.
- Add a memory harness run for each mode before enabling it in the UI.

## 6. Cloud And Virtual Hosting Options

Use cloud execution when the local yes path is too slow or unstable.

| Service path | Good for | Suggested GPU target |
| --- | --- | --- |
| Hugging Face Inference Providers | Quick managed API path, especially if fal is available for the model | Provider-managed |
| fal | Managed image generation API and queues | Provider-managed |
| Hugging Face Inference Endpoints | Dedicated endpoint with predictable app integration | 24 GB minimum, 48 GB plus preferred |
| RunPod | On-demand GPU pod with custom environment | RTX 4090 24 GB, A40/A6000 48 GB, A100/H100 80 GB |
| Lambda Cloud | Stable rented GPU instances | A10/A100/H100/L40S class depending availability |
| Vast.ai | Lowest-cost marketplace experiments | RTX 4090 24 GB or larger |

Recommended cloud decision tree:

1. Need fastest integration with minimal ops: try Hugging Face Inference
   Providers or fal first.
2. Need to run your own SynthaEngine worker: use RunPod, Lambda, or Vast.ai.
3. Need production reliability: use a dedicated endpoint or a reserved GPU
   instance with health checks and persistent model cache.
4. Need only occasional testing: rent an RTX 4090 24 GB pod, run the memory
   harness, then shut it down.

Remote-provider workflow idea:

```json
{
  "provider": "fal",
  "model": "baidu/ERNIE-Image-Turbo",
  "prompt": "a detailed product photo of a brushed steel desk lamp",
  "width": 1024,
  "height": 1024,
  "steps": 8,
  "guidance_scale": 1.0,
  "seed": 12345
}
```

Keep local and hosted execution as separate runtime modes:

- `execution_mode="subprocess"` for local Diffusers.
- `execution_mode="remote"` for hosted providers.
- Never silently fall back from local to paid remote execution.
- Store remote provider name and request ID in job metadata.

## 7. Complete SynthaEngine Implementation Plan

This section is written as a full implementation plan. Because this repository
already has an ERNIE-Image integration, each step includes a current-state note
and the remaining checklist.

### Step 1: Verify Dependency Baseline

Required for local Diffusers execution:

```text
diffusers >= 0.38.0
transformers >= 5.0.0
accelerate installed
torch with CUDA
safetensors installed
Pillow installed
```

Verification command:

```powershell
.venv\Scripts\python.exe -c "from diffusers import ErnieImagePipeline; print('ok')"
```

Current repo status:

- `ErnieImagePipeline` imports successfully.
- `diffusers 0.38.0`, `transformers 5.8.0`, and `torch 2.10.0+cu128` are
  installed in the virtual environment.

Implementation note:

- If this is moved to another machine, update `requirements.txt` only after
  confirming the local venv does not already satisfy the pipeline.

### Step 2: Register ERNIE-Image Models

Model registry entries should use:

```json
{
  "name": "ERNIE-Image-Turbo",
  "family": "ernie-image",
  "model_type": "diffusers",
  "location_type": "hub",
  "model_id": 13,
  "version": "turbo",
  "link": "baidu/ERNIE-Image-Turbo"
}
```

Optional base-model entry:

```json
{
  "name": "ERNIE-Image",
  "family": "ernie-image",
  "model_type": "diffusers",
  "location_type": "hub",
  "model_id": 14,
  "version": "base",
  "link": "baidu/ERNIE-Image"
}
```

Current repo status:

- `backend/registries/model_registry.json` already contains an
  `ERNIE-Image-Turbo` entry pointing to a local Diffusers folder.

Checklist:

- Keep `family` exactly `"ernie-image"`.
- Accept local and Hub Diffusers sources.
- Reject non-Diffusers ERNIE entries until another format is explicitly
  supported.
- Prefer local model paths for repeated use to avoid download delays.

### Step 3: Define Workflow Task Contract

Public task identifier:

```text
ernie-image.text2img
```

Recommended input contract:

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `prompt` | string | `""` | Main text prompt |
| `negative_prompt` | string | `""` | Effective only when CFG is active, usually `guidance_scale > 1.0` |
| `steps` | int | `8` | Turbo-safe default; base model may need 50 |
| `guidance_scale` | float | `1.0` | Turbo default; base model may use stronger CFG |
| `width` | int | `768` | Safe local default; official examples often use 1024 |
| `height` | int | `768` | Safe local default |
| `seed` | int or null | null | Null or 0 can mean random |
| `model` | string or null | null | Registry model name |
| `num_images` | int | `1` | Keep fixed to 1 on 12 GB VRAM |
| `use_pe` | bool | false | Enables Prompt Enhancer at inference |
| `load_pe` | bool | false | Loads PE components; required for `use_pe=true` |
| `memory_preset` | enum | `"sequential_offload"` | `"model_offload"` can be faster but riskier |
| `lora_adapters` | list or null | null | Optional adapter stack |

Current repo status:

- `ErnieImageText2ImgInputs` already exists with these safe defaults.
- `use_pe=true` is rejected unless `load_pe=true`.
- The workflow catalog exposes `ernie-image.text2img`.

Compatibility rule:

- Add optional fields only. Do not rename `ernie-image.text2img`, existing
  fields, or output shape.

### Step 4: Map Diffusers Pipeline Options

Diffusers `ErnieImagePipeline.__call__` options to consider:

| Diffusers option | SynthaEngine field | Expose now? | Notes |
| --- | --- | --- | --- |
| `prompt` | `prompt` | Yes | Required for text2img |
| `negative_prompt` | `negative_prompt` | Yes | Useful with CFG |
| `height` | `height` | Yes | Validate bounds and multiples |
| `width` | `width` | Yes | Validate bounds and multiples |
| `num_inference_steps` | `steps` | Yes | Use Turbo default 8 |
| `guidance_scale` | `guidance_scale` | Yes | Turbo default 1.0 |
| `num_images_per_prompt` | `num_images` | Limited | Keep max 1 locally |
| `generator` | `seed` | Yes | CPU generator is fine |
| `latents` | none | No | Advanced reproducibility/debug path |
| `prompt_embeds` | none | No | Not needed for first app integration |
| `negative_prompt_embeds` | none | No | Not needed for first app integration |
| `output_type` | fixed `"pil"` | No | App expects saved images |
| `return_dict` | fixed true | No | Keep normal Diffusers return object |
| `callback_on_step_end` | progress hook | Later | Useful for richer job events |
| `callback_on_step_end_tensor_inputs` | progress hook | Later | Only if callback is implemented |
| `use_pe` | `use_pe` | Yes | Requires loaded PE components |

Load-time options to control:

| Load option | App field or policy | Notes |
| --- | --- | --- |
| Model source | `model` | Resolve through model registry |
| `torch_dtype` | fixed BF16 | Good default for RTX 3060 with BF16 support |
| PE components | `load_pe` | Pass `pe=None`, `pe_tokenizer=None` when disabled |
| CPU offload | `memory_preset` | Sequential offload safest |
| VAE slicing | fixed enabled | Reduces memory |
| VAE tiling | fixed enabled | Reduces memory for larger images |
| Quantization | future `quantization` | Keep experimental |

### Step 5: Implement Runtime Loader

Recommended local runtime pattern:

```python
import torch
from diffusers import ErnieImagePipeline

pipe = ErnieImagePipeline.from_pretrained(
    model_source,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    pe=None,
    pe_tokenizer=None,
)
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
```

Current repo status:

- `backend/ernie_image/pipeline.py` already follows this pattern.
- It defaults to `baidu/ERNIE-Image-Turbo` if no registry model is found.
- It can skip PE loading by passing `pe=None` and `pe_tokenizer=None`.

Checklist:

- Keep the loader in `backend/ernie_image/pipeline.py`.
- Keep route logic thin.
- Keep `memory_preset` validation explicit.
- Log model source, seed, dimensions, steps, CFG, PE flags, and memory preset.
- Store generation parameters in PNG metadata.

### Step 6: Use A Short-Lived Subprocess

Do not load ERNIE-Image directly inside the long-running FastAPI process.

Recommended control flow:

```text
workflow engine
  -> backend.workflow.ernie_image task adapter
  -> backend.ernie_image.pipeline.generate_text2img
  -> subprocess runner
  -> load ErnieImagePipeline
  -> generate one image
  -> save /outputs/... png
  -> write JSON result
  -> process exits
```

Current repo status:

- `backend/ernie_image/subprocess_runner.py` exists.
- `generate_text2img` rejects direct in-process execution.
- A semaphore gates ERNIE subprocess generation to one active job.

Checklist:

- Keep one ERNIE subprocess active at a time on 12 GB VRAM.
- Bubble child-process errors back into the workflow job.
- Always call memory cleanup in the child process `finally` block.
- Avoid persistent pipeline caching until a larger GPU profile is available.

### Step 7: Add Workflow Engine Wiring

Engine wiring should map:

```python
TASK_INPUT_MODELS["ernie-image.text2img"] = ErnieImageText2ImgInputs
TASK_OUTPUT_MODELS["ernie-image.text2img"] = ImagesOutput
TASK_HANDLERS["ernie-image.text2img"] = _ernie_image_text2img
```

Task adapter behavior:

```python
def run_ernie_image_text2img_task(inputs, deps):
    payload = dict(inputs)
    result = deps["generate_text2img"](payload)
    if not isinstance(result, dict):
        raise ValueError("ernie-image.text2img must return an object")
    return result
```

Current repo status:

- This wiring already exists.

Checklist:

- Preserve output shape: `{"images": ["/outputs/...png"]}`.
- Keep workflow return syntax compatible: `"return": "@t1.images"`.

### Step 8: Build Frontend Controls

The ERNIE-Image page should expose:

- Prompt textarea.
- Negative prompt textarea.
- Model select filtered by `family=ernie-image`.
- Width and height numeric inputs.
- Steps numeric input.
- Guidance scale numeric input.
- Seed input.
- Number of images fixed or capped at 1.
- Prompt Enhancer toggles: `load_pe`, `use_pe`.
- Memory preset select: `sequential_offload`, `model_offload`.
- LoRA adapter modal filtered by `ernie-image`.
- Generate button submitting `kind: "workflow"`.

Workflow payload:

```json
{
  "kind": "workflow",
  "payload": {
    "tasks": [
      {
        "id": "t1",
        "type": "ernie-image.text2img",
        "inputs": {
          "prompt": "a quiet sunlit library with glass walls",
          "negative_prompt": "blurry, distorted text",
          "steps": 8,
          "guidance_scale": 1.0,
          "width": 768,
          "height": 768,
          "memory_preset": "sequential_offload",
          "use_pe": false,
          "load_pe": false
        }
      }
    ],
    "return": "@t1.images"
  }
}
```

Current repo status:

- `frontend/ernie_image/text2img.html` and `text2img.js` already exist.
- The frontend uses the shared workflow client, preset panel, model registry,
  gallery, and LoRA panel.

Checklist:

- Keep UI defaults aligned with workflow catalog defaults.
- Hide ControlNet and IP-Adapter sections for ERNIE.
- Surface memory warnings near PE and high-resolution controls.
- Do not add unsupported img2img/inpaint/ControlNet buttons until backend
  support exists.

### Step 9: Document API And Lifecycle

Docs to update when behavior changes:

| File | Required content |
| --- | --- |
| `docs/WORKFLOW_API.md` | Task identifier, input fields, defaults, example workflow |
| `docs/PIPELINE_LIFECYCLE.md` | Subprocess lifecycle, memory cleanup policy |
| `docs/ARCHITECTURE.md` | Family task/runtime map, capability table |
| Frontend docs if any | UI payload behavior |

Current repo status:

- These docs already mention ERNIE-Image.

Checklist:

- If adding quantization later, update docs and frontend payload together.
- If adding remote execution later, document billing/credential behavior.
- If adding base ERNIE defaults, document when to use Turbo vs base.

### Step 10: Add Tests

Focused test coverage:

```powershell
.venv\Scripts\python.exe -m pytest testing/test_ernie_image_workflow.py -q
.venv\Scripts\python.exe -m pytest testing/test_ernie_image_pipeline.py -q
.venv\Scripts\python.exe -m pytest testing/test_frontend_ernie_image_scripts.py -q
.venv\Scripts\python.exe -m pytest testing/test_workflow_catalog_capabilities.py -q
.venv\Scripts\python.exe -m compileall backend
```

Test categories:

- Schema defaults are low-memory safe.
- `use_pe=true` requires `load_pe=true`.
- Workflow task forwards fields correctly.
- Catalog exposes `ernie-image.text2img`.
- Subprocess bridge invokes the child runner and reads JSON result.
- Subprocess runner cleans up after success and failure.
- Loader skips PE components when `load_pe=false`.
- Loader keeps PE components when `load_pe=true`.
- LoRA adapters are passed through and family-validated.
- Frontend submits `ernie-image.text2img` workflow payloads.

Current repo status:

- These focused test files already exist.

### Step 11: Add Memory Harness Runs

Use the existing harness before declaring a configuration safe:

```powershell
.venv\Scripts\python.exe tools\measure_ernie_image_memory.py `
  --model ERNIE-Image-Turbo `
  --width 768 `
  --height 768 `
  --steps 8 `
  --guidance-scale 1.0 `
  --memory-preset sequential_offload `
  --runs 1 `
  --output-json outputs\ernie_image_memory_768.json
```

Escalation sequence:

1. 512 x 512, 4 to 8 steps, PE off.
2. 768 x 768, 8 steps, PE off.
3. 1024 x 1024, 8 steps, PE off.
4. 768 x 768, PE loaded but `use_pe=false`.
5. 768 x 768, PE loaded and `use_pe=true`.
6. Optional quantization mode test after quantization is implemented.

Accept or reject each profile based on:

- Success/failure.
- Peak CUDA allocated memory.
- Peak CUDA reserved memory.
- Process RSS.
- Wall-clock time.
- Output image sanity.

### Step 12: Optional Future Flags

Do not expose all of these at once. Add only after a focused test confirms the
behavior.

| Future field | Type | Default | Purpose | Risk |
| --- | --- | --- | --- | --- |
| `quantization` | enum | `"none"` | `none`, `bnb_8bit`, `bnb_4bit` | Quality and load failures |
| `execution_mode` | enum | `"subprocess"` | Local vs remote provider | Billing and credentials |
| `provider` | enum | null | `fal`, `hf_endpoint`, etc. | External API dependency |
| `output_type` | enum | `"pil"` | Could allow latent/debug output | Breaks output assumptions |
| `callback_progress` | bool | false | Rich step events | More event plumbing |
| `pe_temperature` | float | 0.6 | Prompt Enhancer generation | Requires PE internals |
| `pe_top_p` | float | 0.95 | Prompt Enhancer generation | Requires PE internals |
| `base_profile` | enum | `"turbo"` | Turbo vs base defaults | Default complexity |

Recommended policy:

- Keep user-facing flags small.
- Put experimental controls behind advanced UI.
- Record every non-default flag in PNG metadata and job logs.

## 8. API Compatibility Notes

Backward-compatible:

- Adding ERNIE-Image as a new task family.
- Adding optional inputs with safe defaults.
- Adding model registry entries.
- Adding frontend controls that submit existing fields.
- Adding docs and tests.

Potentially breaking:

- Renaming `ernie-image.text2img`.
- Renaming `prompt`, `negative_prompt`, `steps`, `guidance_scale`, `width`,
  `height`, `seed`, `model`, `num_images`, `use_pe`, `load_pe`,
  `memory_preset`, or `lora_adapters`.
- Changing output from `{"images": [...]}` to another shape.
- Silently switching from local execution to paid hosted execution.
- Enabling PE by default on 12 GB VRAM.
- Enabling quantization by default before validation.

Recommended stable contract:

```json
{
  "type": "ernie-image.text2img",
  "inputs": {
    "prompt": "text prompt",
    "negative_prompt": "",
    "steps": 8,
    "guidance_scale": 1.0,
    "width": 768,
    "height": 768,
    "seed": 123,
    "model": "ERNIE-Image-Turbo",
    "num_images": 1,
    "use_pe": false,
    "load_pe": false,
    "memory_preset": "sequential_offload"
  }
}
```

## 9. Recommended Implementation Order

If starting from a repo without the existing ERNIE files, implement in this
order:

1. Add dependency/import smoke test.
2. Add model registry family and default model entry.
3. Add `ErnieImageText2ImgInputs` and catalog exposure.
4. Add `backend/workflow/ernie_image.py` task adapter.
5. Add `backend/ernie_image/pipeline.py` loader and subprocess generation.
6. Add `backend/ernie_image/subprocess_runner.py`.
7. Wire task handler into the workflow engine.
8. Add workflow API docs and lifecycle docs.
9. Add focused backend/workflow tests.
10. Add frontend page and payload tests.
11. Run memory harness on the target PC.
12. Only then consider PE-on defaults, higher resolutions, quantization, or
    remote provider support.

For this repository, the immediate practical next steps are:

1. Run the focused ERNIE tests after any future ERNIE edits.
2. Run the memory harness against the local `D:\diffusion\diffusers\Ernie-Image-Turbo`
   model path if the weights are present.
3. Record measured safe profiles in docs before expanding defaults.
4. Keep quantization and remote-provider execution as explicit follow-up work.

## 10. Final Recommendation

Use `ERNIE-Image-Turbo` as the default local ERNIE model in SynthaEngine. Keep
the current conservative defaults:

- `steps=8`
- `guidance_scale=1.0`
- `width=768`
- `height=768`
- `num_images=1`
- `use_pe=false`
- `load_pe=false`
- `memory_preset="sequential_offload"`
- subprocess-only local execution

This is the best balance for a 64 GB RAM, RTX 3060 12 GB VRAM Windows machine.
The base ERNIE-Image checkpoint, Prompt Enhancer, high-resolution generation,
and quantization should remain opt-in until measured in this repo.

