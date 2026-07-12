# LLaDA2 Diffusers Implementation Guide for SynthaEngine

Date: 2026-05-30

This guide explains what LLaDA2 is, how practical it is on a local PC with 64 GB
system RAM and an RTX 3060 with 12 GB VRAM, and how to add it to this
SynthaEngine application when you are ready to implement it. No application code
has been changed by this research note.

## 1. Executive Summary

LLaDA2 is a text-to-text discrete diffusion language model family. It is not an
image-generation model. In Diffusers it appears as `LLaDA2Pipeline`, a pipeline
that generates text by starting from masked tokens and refining blocks of tokens
over repeated steps.

Your current local environment is already close to ready:

- Installed: `diffusers 0.38.0`, `transformers 5.8.0`, `torch 2.10.0+cu128`.
- Hardware visible to PyTorch: `NVIDIA GeForce RTX 3060`.
- CUDA is available, the GPU reports compute capability `(8, 6)`, and PyTorch
  reports BF16 support.
- `bitsandbytes` is installed, which matters because the full BF16 model is
  larger than 12 GB VRAM.
- `from diffusers import LLaDA2Pipeline, BlockRefinementScheduler` works in the
  repo virtual environment.

The practical answer:

- `inclusionAI/LLaDA2.1-mini` is runnable experimentally on this PC if you use
  `device_map="auto"` with CPU offload and/or 8-bit or 4-bit quantization.
- It is not a clean "fits fully in RTX 3060 VRAM" model. The model card lists
  16B parameters, and the repository stores roughly 32.5 GB of safetensors
  shards. Full BF16 weights alone are about 32 GB before runtime overhead, so
  12 GB VRAM requires CPU RAM participation or quantization.
- `inclusionAI/LLaDA2.1-flash` is not recommended for this machine. It is a
  100B/103B-class model and is outside a comfortable local consumer-GPU path.

Recommended first implementation target:

- Add an experimental workflow task named `llada2.text2text`.
- Support only `inclusionAI/LLaDA2.1-mini` at first.
- Use a short-lived subprocess, matching large model families already in this
  repo.
- Default to a safe speed profile: `gen_length=256`, `block_length=32`,
  `num_inference_steps=32`, `threshold=0.5`, `editing_threshold=None`,
  `max_post_steps=16`, `temperature=0.0`, and `quantization="8bit"` or
  `quantization="4bit"` as an explicit user option.

## 2. What LLaDA2 Is

LLaDA2 is a diffusion language model family. Unlike a conventional
autoregressive language model that generates one token after another from left
to right, LLaDA2 starts with a masked output sequence and progressively fills
and optionally edits tokens block by block.

The official Diffusers documentation describes LLaDA2 as a family of discrete
diffusion language models that generate text through block-wise iterative
refinement. The Diffusers pipeline keeps a template sequence filled with mask
tokens, refines active blocks, samples candidate tokens from model logits, and
commits tokens based on confidence.

Key mental model:

- Input: prompt or chat messages.
- Output: generated text, not images.
- Core model: loaded with Transformers `AutoModelForCausalLM` and
  `trust_remote_code=True`.
- Diffusers wrapper: `LLaDA2Pipeline`.
- Scheduler: `BlockRefinementScheduler`.
- Generation knobs: `gen_length`, `block_length`, `num_inference_steps`,
  `threshold`, `editing_threshold`, `max_post_steps`, `temperature`,
  `top_p`, `top_k`.

Official references:

- Diffusers LLaDA2 pipeline docs:
  https://huggingface.co/docs/diffusers/api/pipelines/llada2
- Diffusers pipeline overview listing LLaDA2 as `text2text`:
  https://huggingface.co/docs/diffusers/api/pipelines/overview
- Diffusers 0.38.0 release notes, where LLaDA2 was added as a new pipeline:
  https://github.com/huggingface/diffusers/releases/tag/v0.38.0
- LLaDA2.1-mini model card:
  https://huggingface.co/inclusionAI/LLaDA2.1-mini
- LLaDA2.1 paper:
  https://arxiv.org/abs/2602.08676

## 3. Current Diffusers Support

As of the checked documentation and this repo environment:

- Diffusers `0.38.0` includes `LLaDA2Pipeline`.
- Diffusers `0.38.0` includes `BlockRefinementScheduler`.
- The official example loads `inclusionAI/LLaDA2.1-mini` with
  `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True,
  dtype=torch.bfloat16, device_map="auto")`, creates
  `BlockRefinementScheduler()`, then constructs `LLaDA2Pipeline`.
- The pipeline accepts either `prompt`, `messages`, or pre-tokenized
  `input_ids`.
- The pipeline output has `sequences` and optional decoded `texts`.

The local import check passed with:

```powershell
.venv\Scripts\python.exe -c "from diffusers import LLaDA2Pipeline, BlockRefinementScheduler; import inspect; print(inspect.signature(LLaDA2Pipeline.__call__))"
```

This means the first implementation does not need a community pipeline or a
vendored pipeline copy. Treat it as a first-class Diffusers pipeline.

## 4. Model Variants and Hardware Fit

### LLaDA2.1-mini

The `inclusionAI/LLaDA2.1-mini` model card lists:

- Type: Mixture-of-Experts diffusion language model.
- Total non-embedding parameters: 16B.
- Layers: 20.
- Attention heads: 16.
- Context length: 32,768 tokens.
- Vocabulary size: 157,184.
- License: Apache 2.0.

The file listing contains 8 safetensors shards totaling roughly 32.5 GB:

- one shard around 5.74 GB
- seven shards around 3.83 GB each

This is consistent with a 16B BF16 model footprint before activation and runtime
overhead. A 12 GB VRAM GPU cannot hold it fully in BF16.

Practical rating on this PC:

| Mode | Expected viability | Notes |
| --- | --- | --- |
| Full BF16 on GPU | Not viable | Weights alone exceed 12 GB VRAM. |
| BF16 with `device_map="auto"` CPU offload | Viable but slow | 64 GB RAM should help, but CPU/GPU transfers can dominate. |
| 8-bit bitsandbytes | Best first local path | Smaller VRAM pressure, likely acceptable quality, still may use CPU offload. |
| 4-bit bitsandbytes | Most memory-efficient path | Better fit for 12 GB VRAM, but quality/performance should be validated. |
| SGLang/vLLM service | Interesting later | Better for serving, but adds a second runtime stack and Windows friction. |

Recommended first local smoke test:

- Use `LLaDA2.1-mini`.
- Start with `gen_length=128` or `256`.
- Use speed mode: `threshold=0.5`, `editing_threshold=None`.
- Keep batch size at 1.
- Expect first-run download and model load to be the slow part.

### LLaDA2.1-flash

The `inclusionAI/LLaDA2.1-flash` model card lists a 100B/103B-class model. Do not
target this variant for the RTX 3060 12 GB implementation. It belongs in a
server, multi-GPU, or hosted inference plan.

## 5. Why LLaDA2 Is Different for SynthaEngine

SynthaEngine is currently workflow-first and generation-family oriented. The
primary job API accepts `kind: "workflow"` jobs, executes tasks from
`backend/workflow/`, and returns per-task outputs.

Existing generation families are image or video oriented:

- `sd15.*`
- `sdxl.*`
- `flux.*`
- `qwen-image.*`
- `z-image.*`
- `ernie-image.*`
- `anima.*`
- `wan.*`

LLaDA2 should not be forced into the image output contract. It should be a new
text output family:

- Family key: `llada2`
- Initial task type: `llada2.text2text`
- Output shape: text-oriented, for example `{ "texts": ["..."] }`

This is API-visible, so implementation must update code, docs, schema, catalog,
and tests together.

## 6. Implementation Plan for This App

This section is the concrete step-by-step path to add LLaDA2 later. It is
written to fit the current codebase patterns and the repository rules.

### Step 1: Add dependency policy

Current `requirements.txt` already has the core pieces:

- `diffusers>=0.38.0`
- `transformers>=5.8.0`
- `accelerate>=1.2.1`
- `safetensors>=0.4.5`

For a realistic 12 GB VRAM implementation, add or confirm:

```txt
bitsandbytes>=0.49.2
```

Rationale:

- Transformers supports bitsandbytes quantization through
  `BitsAndBytesConfig`.
- Current bitsandbytes docs list CUDA and Windows support for modern NVIDIA
  hardware.
- The local environment already has `bitsandbytes 0.49.2`.

### Step 2: Add a model registry fallback

Add a new family in the base-model registry flow:

- Family: `llada2`
- Default model name: `LLaDA2.1-mini`
- Default Hub id: `inclusionAI/LLaDA2.1-mini`
- Model type: `transformers-diffusers` or plain `diffusers`, depending on how
  strictly you want to classify it.

Implementation location:

- Similar lookup pattern to `backend/ernie_image/pipeline.py`.
- Create a helper in the new LLaDA2 runtime module:
  `_get_llada2_model_entry(model_name: str | None)`.

The fallback entry should resolve to the Hub repo when no registry entry is
present. That keeps the first smoke test simple.

### Step 3: Add workflow input schema

Edit:

- `backend/workflow/schema_input.py`

Add:

```python
class LLaDA2Text2TextInputs(BaseModel):
    prompt: str = ""
    model: str | None = None
    gen_length: int = Field(default=256, ge=1, le=4096)
    block_length: int = Field(default=32, ge=1, le=256)
    num_inference_steps: int = Field(default=32, ge=1, le=256)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    editing_threshold: float | None = None
    max_post_steps: int = Field(default=16, ge=0, le=128)
    temperature: float = Field(default=0.0, ge=0.0, le=5.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1, le=1000)
    sampling_method: str = "multinomial"
    seed: int | None = None
    use_chat_template: bool = True
    add_generation_prompt: bool = True
    quantization: Literal["none", "8bit", "4bit"] = "8bit"
    cpu_offload: bool = True
    trust_remote_code: bool = True
```

Notes:

- `trust_remote_code=True` is required by the official examples and model card,
  so make it visible in docs.
- Keep `gen_length` conservative. Long outputs can be expensive because the
  model repeatedly refines blocks.
- `messages` can be added later. Start with `prompt` to keep the initial
  contract small.

### Step 4: Add workflow output schema

Edit:

- `backend/workflow/schema_output.py`

Add:

```python
class TextsOutput(BaseModel):
    texts: list[str]
```

Optionally include metadata later:

```python
class TextsWithMetadataOutput(BaseModel):
    texts: list[str]
    model: str | None = None
    seed: int | None = None
    warnings: list[str] = Field(default_factory=list)
```

Recommendation for first implementation:

- Use `{ "texts": [...] }` only.
- Add metadata only if frontend/history support needs it.

### Step 5: Add the workflow adapter

Create:

- `backend/workflow/llada2.py`

Pattern:

```python
from __future__ import annotations

from typing import Any


def run_llada2_text2text_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    generate_text2text = deps["generate_text2text"]
    result = generate_text2text(dict(inputs))
    if not isinstance(result, dict):
        raise ValueError("llada2.text2text must return an object")
    if not isinstance(result.get("texts"), list):
        raise ValueError("llada2.text2text must return texts")
    return result
```

This mirrors the thin family adapter style used by ERNIE-Image and Anima.

### Step 6: Add the runtime package

Create:

- `backend/llada2/__init__.py`
- `backend/llada2/pipeline.py`
- `backend/llada2/subprocess_runner.py`

Why subprocess:

- `docs/PIPELINE_LIFECYCLE.md` says pipelines are job-scoped by default.
- Large families in this repo use one-shot subprocesses so Windows can reclaim
  GPU and system memory after generation.
- LLaDA2 is large enough that a subprocess boundary is the safest first design.

Runtime responsibilities:

- Resolve model from registry or fallback Hub id.
- Load tokenizer with `trust_remote_code=True`.
- Load model with safe memory mode:
  - 8-bit or 4-bit bitsandbytes by default.
  - `device_map="auto"`.
  - `dtype=torch.bfloat16` for BF16 capable hardware.
- Build `BlockRefinementScheduler`.
- Build `LLaDA2Pipeline`.
- Run text generation.
- Return `{ "texts": [text] }`.
- In `finally`, release pipeline/model/tokenizer refs and call
  `cleanup_memory()`.

Pseudo-code sketch:

```python
import torch
from diffusers import BlockRefinementScheduler, LLaDA2Pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _load_quant_config(mode: str):
    if mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if mode == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    return None


def _generate_text2text_subprocess_child(params: dict[str, object]) -> dict[str, list[str]]:
    source = resolve_llada2_source(params.get("model"))
    quantization = str(params.get("quantization") or "8bit")
    quant_config = _load_quant_config(quantization)

    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "dtype": torch.bfloat16,
    }
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config

    model = AutoModelForCausalLM.from_pretrained(source, **load_kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    scheduler = BlockRefinementScheduler()
    pipe = LLaDA2Pipeline(model=model, tokenizer=tokenizer, scheduler=scheduler)

    output = pipe(
        prompt=str(params.get("prompt") or ""),
        gen_length=int(params.get("gen_length") or 256),
        block_length=int(params.get("block_length") or 32),
        num_inference_steps=int(params.get("num_inference_steps") or 32),
        threshold=float(params.get("threshold") or 0.5),
        editing_threshold=params.get("editing_threshold"),
        max_post_steps=int(params.get("max_post_steps") or 16),
        temperature=float(params.get("temperature") or 0.0),
        top_p=params.get("top_p"),
        top_k=params.get("top_k"),
        sampling_method=str(params.get("sampling_method") or "multinomial"),
        use_chat_template=bool(params.get("use_chat_template", True)),
        add_generation_prompt=bool(params.get("add_generation_prompt", True)),
    )
    return {"texts": [str(text) for text in (output.texts or [])]}
```

Implementation caution:

- Verify whether installed Transformers expects `dtype=` or `torch_dtype=`.
  The current local environment uses Transformers 5.8.0; the official Diffusers
  example uses `dtype=torch.bfloat16`, while older examples often use
  `torch_dtype`.
- Do not call `.to("cuda")` after loading with bitsandbytes and `device_map`.
  Let Accelerate manage placement.
- Keep `num_images`, artifact handling, PNG metadata, and output folders out of
  the LLaDA2 runtime. This is text generation, not media generation.

### Step 7: Register the task in the workflow engine

Edit:

- `backend/workflow/engine.py`

Add imports:

```python
from backend.workflow.schema_input import LLaDA2Text2TextInputs
from backend.workflow.schema_output import TextsOutput
from backend.workflow.llada2 import run_llada2_text2text_task as _run_llada2_text2text
```

Add to `TASK_INPUT_MODELS`:

```python
"llada2.text2text": LLaDA2Text2TextInputs,
```

Add to `TASK_OUTPUT_MODELS`:

```python
"llada2.text2text": TextsOutput,
```

Add runtime deps and handler:

```python
def _llada2_runtime_deps() -> dict[str, Any]:
    llada2_pipeline_module = importlib.import_module("backend.llada2.pipeline")
    return {
        "generate_text2text": llada2_pipeline_module.generate_text2text,
    }


def _llada2_text2text(inputs: dict[str, Any], _ctx: WorkflowContext) -> dict[str, Any]:
    return _run_llada2_text2text(inputs, _llada2_runtime_deps())
```

Add to `TASK_REGISTRY`:

```python
"llada2.text2text": _llada2_text2text,
```

### Step 8: Update catalog capabilities

Edit:

- `backend/workflow/catalog.py`

Add family metadata:

```python
"llada2": {"label": "LLaDA2", "aliases": ["llada"]},
```

Update `_infer_model_family` so `llada2.text2text` maps to `llada2`.

Expected `/api/workflow/catalog` behavior:

```json
{
  "capabilities": {
    "llada2": {
      "label": "LLaDA2",
      "aliases": ["llada"],
      "task_types": ["llada2.text2text"],
      "features": {
        "text2text": true
      }
    }
  }
}
```

If the generic feature inference does not produce `text2text`, add a small
catalog rule for text tasks.

### Step 9: Update API docs

Edit:

- `docs/WORKFLOW_API.md`
- `docs/PIPELINE_LIFECYCLE.md`

In `WORKFLOW_API.md`, add:

- `llada2` to the list of current families.
- Input contract for `llada2.text2text`.
- Output contract `{ "texts": ["..."] }`.
- A workflow example.

Example:

```json
{
  "kind": "workflow",
  "payload": {
    "tasks": [
      {
        "id": "answer",
        "type": "llada2.text2text",
        "inputs": {
          "prompt": "Write a two sentence explanation of block diffusion.",
          "gen_length": 128,
          "threshold": 0.5,
          "editing_threshold": null,
          "quantization": "8bit"
        }
      }
    ],
    "return": "@answer.texts"
  }
}
```

In `PIPELINE_LIFECYCLE.md`, add LLaDA2 to the subprocess-backed family list and
state that LLaDA2 text renders are job-scoped and subprocess-backed by default.

### Step 10: Add tests

Add focused tests before running a real model:

- Catalog exposes `llada2.text2text`.
- Schema accepts minimal input.
- Task output shape is `texts`.
- Workflow engine can run the task with a mocked `generate_text2text`.
- Invalid output from the runtime adapter raises a clear error.

Likely test files:

- `testing/test_workflow_catalog_capabilities.py`
- A new or existing workflow test for task dispatch.
- A small unit test for `backend.workflow.llada2.run_llada2_text2text_task`.

Do not download the 32 GB model in normal tests. Mock the runtime dependency.

### Step 11: Add optional frontend only after backend works

The app can support LLaDA2 without a dedicated frontend page because workflows
can be posted directly to `/api/jobs`.

If adding UI later:

- Create `frontend/llada2/text2text.html`.
- Create `frontend/llada2/text2text.js`.
- Use `frontend/workflow_client.js`.
- Submit a workflow job with task type `llada2.text2text`.
- Render text output rather than gallery media.

Keep this out of the first backend implementation unless the UI is explicitly
requested.

## 7. Local Smoke Test Script Before Full App Integration

Before wiring the workflow task, run a one-off script outside the app. This is
the fastest way to learn whether your exact Windows, CUDA, and bitsandbytes
stack can load the model.

Suggested file name if you create it later:

- `tools/smoke_llada2.py`

Sketch:

```python
import torch
from diffusers import BlockRefinementScheduler, LLaDA2Pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "inclusionAI/LLaDA2.1-mini"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config,
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
scheduler = BlockRefinementScheduler()
pipe = LLaDA2Pipeline(model=model, scheduler=scheduler, tokenizer=tokenizer)

out = pipe(
    prompt="Write a short note about why local AI tools need memory planning.",
    gen_length=128,
    block_length=32,
    num_inference_steps=32,
    threshold=0.5,
    editing_threshold=None,
    max_post_steps=16,
    temperature=0.0,
)

print(out.texts[0])
```

Run:

```powershell
.venv\Scripts\python.exe tools\smoke_llada2.py
```

If 8-bit fails:

1. Try 4-bit NF4.
2. Try `device_map="auto"` with no quantization and expect heavy CPU RAM use.
3. Reduce `gen_length` to `64`.
4. Confirm `bitsandbytes` loads CUDA support.
5. Consider WSL2 if native Windows quantized loading is unstable.

## 8. Recommended Defaults for RTX 3060 12 GB

For first integration:

```json
{
  "gen_length": 256,
  "block_length": 32,
  "num_inference_steps": 32,
  "threshold": 0.5,
  "editing_threshold": null,
  "max_post_steps": 16,
  "temperature": 0.0,
  "top_p": null,
  "top_k": null,
  "quantization": "8bit",
  "cpu_offload": true
}
```

Why these defaults:

- The official Diffusers docs recommend `block_length=32`,
  `temperature=0.0`, and `num_inference_steps=32`.
- The model card recommends `threshold=0.5` plus disabled editing for speed
  mode, and `threshold=0.7` with `editing_threshold=0.5` for quality mode.
- On 12 GB VRAM, speed mode is the better first target because post-mask editing
  means additional refinement work.

Expose quality mode later:

```json
{
  "threshold": 0.7,
  "editing_threshold": 0.5,
  "max_post_steps": 16
}
```

## 9. Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Model download is roughly 32 GB | Long first run, disk pressure | Document expected download and cache location. |
| Full BF16 does not fit VRAM | OOM | Default to 8-bit or 4-bit plus `device_map="auto"`. |
| `trust_remote_code=True` required | Security review needed | Only allow approved model ids by default. Document the trust boundary. |
| Transformers `dtype` vs `torch_dtype` changes | Load errors across versions | Test against local Transformers 5.8.0 and pin examples in docs. |
| Text output differs from media output | API/frontend assumptions | Add explicit `TextsOutput` and docs; do not reuse `ImagesOutput`. |
| Windows bitsandbytes edge cases | Load failures | Keep smoke test separate; consider WSL2 fallback. |
| Long generation length is slow | Poor UX | Conservative defaults and clear `gen_length` bounds. |
| Large model memory not released | Subsequent jobs fail | Use a one-shot subprocess and `cleanup_memory()`. |

## 10. Definition of Done for a Future Implementation

A proper code implementation should be considered done only when:

- `llada2.text2text` appears in `/api/workflow/task-types`.
- `/api/workflow/catalog` includes `llada2` capabilities and schema.
- A minimal workflow job can return `{ "texts": ["..."] }`.
- Docs include the task contract and example workflow.
- `docs/PIPELINE_LIFECYCLE.md` covers LLaDA2 cleanup.
- Focused unit tests pass without downloading the real model.
- Backend compiles:

```powershell
.venv\Scripts\python.exe -m compileall backend
```

- Relevant workflow/catalog tests pass:

```powershell
.venv\Scripts\python.exe -m pytest testing/test_workflow_catalog_capabilities.py -q
.venv\Scripts\python.exe -m pytest testing/test_*workflow*.py -q
```

Optional but strongly recommended:

- One manual local smoke test with `inclusionAI/LLaDA2.1-mini`,
  `gen_length=128`, and `quantization="8bit"` or `"4bit"`.

## 11. Bottom Line

LLaDA2 is now a real Diffusers pipeline, and this repo's installed Diffusers
version can import it today. The model family is text-to-text, so the right
SynthaEngine integration is a new text workflow family, not another image
pipeline.

On this PC, target `LLaDA2.1-mini` only. Treat it as experimental, use a
subprocess, default to quantization/offload, keep output length conservative,
and validate with a standalone smoke test before wiring the workflow task.
