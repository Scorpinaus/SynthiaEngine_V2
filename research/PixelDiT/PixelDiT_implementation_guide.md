# PixelDiT Implementation Guide for SynthaEngine

Date: 2026-06-03

Research target: PixelDiT, Pixel Diffusion Transformers for Image Generation  
Project page: https://pixeldit.github.io/  
Code: https://github.com/NVlabs/PixelDiT  
Main Hugging Face models:
- https://huggingface.co/nvidia/PixelDiT-ImageNet
- https://huggingface.co/nvidia/PixelDiT-1300M-1024px

This guide explains what PixelDiT is, whether Hugging Face Diffusers already
implements it, how realistic it is on a PC with 64 GB system RAM and an RTX
3060 12 GB GPU, and how to add it to SynthaEngine as a workflow-first model
family. This is a research and implementation plan only. No application files
were changed by this guide.

## 1. Executive Summary

PixelDiT is a VAE-free pixel-space diffusion transformer from NVIDIA and the
University of Rochester. Unlike Stable Diffusion, SDXL, Flux, and most modern
latent diffusion systems, PixelDiT denoises directly in RGB pixel space instead
of using a VAE to compress images into latent tensors.

Current implementation status:

| Question | Answer |
| --- | --- |
| Is there official PixelDiT code? | Yes. NVIDIA released `NVlabs/PixelDiT` with `c2i`, `t2i`, and shared `pixdit_core` code. |
| Are there Hugging Face model weights? | Yes. NVIDIA hosts ImageNet class-to-image checkpoints and a 1.3B text-to-image checkpoint. |
| Is PixelDiT implemented as an official Hugging Face Diffusers pipeline? | No. Current Diffusers docs and source search do not show a PixelDiT pipeline or `PixelDiT*Pipeline` class. |
| Does NVIDIA PixelDiT depend on Diffusers? | Yes, its requirements include `diffusers==0.30.0`, but the actual PixelDiT model and sampling code are custom NVIDIA code, not a Diffusers `DiffusionPipeline`. |
| Should SynthaEngine wait for official Diffusers support? | Not if you want to experiment now. The practical path is a custom `pixeldit` runtime wrapper with subprocess isolation, then later replace it with an official Diffusers pipeline if one appears. |

Practical local answer for the requested PC:

| Option | Verdict | What it means |
| --- | --- | --- |
| Yes, local proof-of-concept | Feasible to attempt | Try ImageNet class-to-image at 256x256 or 512x512 first, then T2I at 512x512, batch size 1, 25-50 steps, subprocess execution, aggressive cleanup, and no training. |
| No, comfortable local production | Not a good fit | Do not promise 1024x1024 T2I, batching, 50K evaluation, or training on an RTX 3060 12 GB. Use cloud GPUs for those. |

Recommended SynthaEngine integration path:

1. Add a new `pixeldit` model family, not a Flux/SDXL variant.
2. Start with `pixeldit.text2img` using NVIDIA's T2I checkpoint.
3. Optionally add `pixeldit.class2img` for ImageNet class-conditioned testing.
4. Run PixelDiT in a short-lived subprocess, mirroring existing heavy model
   families such as ERNIE-Image, Anima, Z-Image, and WAN.
5. Use conservative defaults for 12 GB VRAM.
6. Keep quantization as experimental until PixelDiT is ported to Diffusers or
   manually adapted for `torchao`, `quanto`, or `bitsandbytes`.

## 2. What PixelDiT Is

PixelDiT is a single-stage, end-to-end diffusion transformer that generates
images directly in pixel space. It removes the VAE stage used by latent
diffusion models.

The motivation is straightforward:

- A VAE can blur high-frequency detail such as small text, texture, and fine
  edges.
- A separately trained VAE creates a two-stage training and inference system.
- Pixel-space denoising avoids VAE reconstruction artifacts, which helps
  detail preservation and image editing consistency.

The cost is also straightforward:

- Pixel-space tensors are much larger than compressed latents.
- Direct pixel modeling can be expensive unless the architecture reduces
  attention cost.
- High-resolution inference has heavier activation pressure than latent-space
  models of similar parameter count.

## 3. Architecture

PixelDiT uses a dual-level transformer architecture.

| Component | Role |
| --- | --- |
| Patch-level pathway | Processes coarse patch tokens and handles global layout, object semantics, and prompt alignment. |
| Pixel-level pathway | Processes dense pixel tokens and refines local texture and detail. |
| Pixel-wise AdaLN | Conditions pixel updates with semantic context from the patch pathway. |
| Pixel token compaction | Compresses dense pixel tokens for attention, then expands them back, reducing attention cost without using a VAE. |
| Rectified Flow objective | PixelDiT uses flow matching / velocity prediction in pixel space. |
| FlowDPM solver | Official inference uses `FlowDPMSolver` style sampling. |

For text-to-image, PixelDiT extends the patch-level pathway with MM-DiT-style
text-image fusion. Text embeddings are produced by a frozen Gemma-2 text
encoder. The text stream conditions patch-level semantic tokens; the
pixel-level pathway remains focused on dense refinement.

### Mental Model

```text
prompt
  -> Gemma-2 text encoder
  -> patch-level MM-DiT semantic pathway
  -> semantic tokens + timestep
  -> pixel-wise AdaLN conditioning
  -> pixel-level transformer refinement
  -> RGB pixel output
```

For class-to-image, the text encoder is not used. The model is conditioned on an
ImageNet class label instead.

## 4. Modalities And Sub-Pipelines

PixelDiT has two released practical sub-pipelines and one research editing
direction.

| Modality / sub-pipeline | Official availability | SynthaEngine recommendation |
| --- | --- | --- |
| ImageNet class-to-image | Available in `c2i/` with 256x256 and 512x512 checkpoints | Add as optional `pixeldit.class2img` after T2I smoke tests, or use first for low-complexity local validation. |
| Text-to-image | Available in `t2i/` with `PixelDiT-T2I`, 1.3B params, 512x512 and 1024x1024 training stages | Add as initial public workflow task `pixeldit.text2img`, but default to conservative local settings. |
| Train-free image editing via FlowEdit | Shown in project/paper results | Treat as research-only until there is stable code and a clear input/output contract. |
| Image-to-image | No standalone official PixelDiT img2img pipeline | Do not expose initially. |
| Inpainting | No standalone official PixelDiT inpaint pipeline | Do not expose initially. |
| ControlNet | No official PixelDiT ControlNet pipeline | Do not expose initially. |
| IP-Adapter | No official PixelDiT IP-Adapter pipeline | Do not expose initially. |
| LoRA adapters | Not documented as an inference feature | Do not expose initially. Consider only after module names and training support are understood. |
| Text-to-video / image-to-video | Not available | Out of scope. |
| Training | Available in official code, but very compute-heavy | Document only. Do not support from SynthaEngine UI initially. |

## 5. Released Checkpoints

| Model | Hub repo | Task | Size / params | Notes |
| --- | --- | --- | --- | --- |
| PixelDiT-ImageNet | `nvidia/PixelDiT-ImageNet` | ImageNet class-to-image | PixelDiT-XL, 797M params | Checkpoints for 256x256 epochs 80/160/320 and 512x512 epoch 850. |
| PixelDiT-1300M-1024px | `nvidia/PixelDiT-1300M-1024px` | Text-to-image | 1.3B params, checkpoint file around 5.25 GB | Uses Gemma-2-2B-IT text encoder and supports multi-aspect 1024px generation. |

Important license note: the NVIDIA PixelDiT model cards and repository use
NSCLv1. The model cards state the work and derivative works are for
non-commercial research or evaluation only. Confirm this is acceptable before
shipping PixelDiT in any commercial or public service.

## 6. Hugging Face Diffusers Support Check

As of 2026-06-03:

- The official Diffusers pipeline overview does not list PixelDiT.
- Search results for official Diffusers docs and the Hugging Face Diffusers
  GitHub repository do not expose a PixelDiT pipeline page or class.
- The local SynthaEngine virtual environment has Diffusers `0.38.0`, and
  `dir(diffusers)` does not expose a `PixelDiT` or `PixDiT` symbol.
- NVIDIA's PixelDiT `requirements.txt` includes `diffusers==0.30.0`, but
  PixelDiT runtime code is custom, with folders such as `pixdit_core`, `c2i`,
  and `t2i`.

Conclusion: PixelDiT is not currently implemented as a first-class Hugging Face
Diffusers pipeline. SynthaEngine should integrate it as a custom runtime. If
Diffusers support appears later, replace the custom runtime wrapper with the
official `DiffusionPipeline` class and keep the public `pixeldit.text2img`
workflow contract stable.

## 7. Local Feasibility On 64 GB RAM + RTX 3060 12 GB VRAM

### Yes Path: Local Experimentation

This path is realistic if the target is "make it run for research."

Use these defaults first:

| Setting | Local default |
| --- | --- |
| Task | `pixeldit.class2img` at 256x256 or `pixeldit.text2img` at 512x512 |
| Batch size | 1 |
| Number of images | 1 |
| Steps | 25 for fast smoke tests, 50 for quality smoke tests |
| Precision | Try BF16 because the official configs use it; fall back to FP16 if the Windows/PyTorch/CUDA stack has BF16 issues |
| Execution | Subprocess only |
| Text encoder | Load only in the subprocess; consider CPU/offload or encode-then-release for T2I |
| Prompt file | Generate a one-line temporary prompt file per workflow job |
| Output | Copy or save final PNGs into SynthaEngine `outputs/batch_<batch_id>/` |

Why this may work:

- The T2I checkpoint is around 5.25 GB on Hugging Face, and system RAM is 64 GB.
- The RTX 3060 has 12 GB VRAM, which can handle many single-image research
  tasks if activations are controlled.
- PixelDiT removes VAE memory, though this does not make it cheap because it
  denoises in pixel space.
- SynthaEngine already has a subprocess pattern that prevents heavy models
  from staying resident after a job.

### No Path: Local Production Or Training

This path is not realistic on the requested PC.

Avoid promising:

- 1024x1024 T2I as the default local experience.
- Batch size greater than 1.
- 50K ImageNet evaluation.
- Training or fine-tuning.
- Multi-GPU official evaluation settings.
- Runtime quantization as a stable default before dedicated tests.

Why this is risky:

- PixelDiT-T2I includes a 1.3B image generator plus a Gemma-2-2B-IT text
  encoder path.
- Pixel-space activations grow directly with output resolution.
- NVIDIA's evaluation and training commands are written for multi-GPU use.
- The official code currently loads model components directly to CUDA, so a
  SynthaEngine low-VRAM wrapper must be more careful than the reference script.

### Recommended Local Default Contract

```json
{
  "task_type": "pixeldit.text2img",
  "defaults": {
    "width": 512,
    "height": 512,
    "steps": 25,
    "cfg_scale": 2.75,
    "num_images": 1,
    "seed": null,
    "negative_prompt": "low quality, worst quality, over-saturated, blurry, deformed, watermark",
    "sampling_algo": "flow_dpm-solver",
    "interval_guidance": [0.0, 1.0],
    "memory_preset": "subprocess_low_vram",
    "precision": "bf16",
    "experimental_ack": true
  }
}
```

## 8. Quantization Options

There is no official PixelDiT quantized checkpoint and no official Diffusers
PixelDiT pipeline-level quantization path at the time of this guide.

| Option | Possible? | Recommended now? | Notes |
| --- | --- | --- | --- |
| BF16 / FP16 mixed precision | Yes | Yes | Official configs use BF16. FP16 may be more predictable on some Windows RTX 3060 setups. |
| CPU offload / staged load | Yes, with custom wrapper work | Yes | Encode prompt, release text encoder, load generator, run sample. This is more important than quantization at first. |
| `bitsandbytes` text encoder quantization | Maybe | Experimental | Could quantize Gemma-2 text encoder with Transformers if the installed Windows bitsandbytes stack supports it. Requires code changes. |
| `bitsandbytes` PixelDiT model quantization | Maybe | Not initially | PixelDiT is not a Diffusers model class. Manual replacement of Linear layers may work but needs careful quality and device tests. |
| Diffusers `PipelineQuantizationConfig` | No direct path | No | Requires a Diffusers pipeline and named components such as `transformer` and `text_encoder`. |
| Diffusers GGUF loading | No direct path | No | Current Diffusers GGUF docs support loading model classes from single files, not PixelDiT pipeline loading. |
| `torchao` / `quanto` / FP8 layerwise casting | Maybe | Future | Needs a native PixelDiT model wrapper with module-level skip rules for norms/modulation layers. |
| TensorRT / ONNX | Theoretically | Future | Pixel-space dynamic sizes and custom sampler make this a later optimization. |

Quantization implementation rule for SynthaEngine:

1. Add `quantization: "none"` first.
2. Make all non-none values experimental and reject unsupported combinations
   with clear errors.
3. Test image quality and exact VRAM deltas before exposing a frontend toggle.
4. Never silently quantize; include quantization state in PNG metadata.

Proposed enum:

```python
PixelDiTQuantization = Literal[
    "none",
    "text_encoder_bnb_8bit",
    "text_encoder_bnb_4bit",
    "model_torchao_int8_experimental",
    "model_quanto_int8_experimental"
]
```

## 9. Cloud And Virtual Hosting Options

Use cloud if the target is 1024x1024 T2I, reliable user-facing generation,
evaluation, or any training.

| Provider | Good for | Current notes |
| --- | --- | --- |
| RunPod Pods | Fast experiments, custom Docker, RTX 4090/A6000/A100/H100 | Official page lists custom containers and on-demand GPUs. As of current page, examples include RTX 4090 24 GB, RTX 6000 Ada 48 GB, A100 80 GB, H100 80 GB, and H200 141 GB. |
| Lambda Cloud | Cleaner managed GPU VM experience | Official instances page lists A6000 48 GB, A100 40 GB, H100 80 GB, B200, and multi-GPU options. |
| Vast.ai | Lowest-cost marketplace experiments | Pricing is marketplace-driven and varies by host. Good for interruptible batch jobs; check reliability and storage terms. |
| Hugging Face Inference Endpoints | Managed private endpoint with custom container | Good once a custom PixelDiT container works. The PixelDiT model cards currently say the models are not deployed by an Inference Provider, so use a custom endpoint/container rather than expecting a hosted provider call. |
| AWS / GCP / Azure | Enterprise controls and quotas | More expensive, but useful when compliance, private networking, or team billing matters. |

Recommended GPU tiers:

| Workload | Minimum comfortable GPU |
| --- | --- |
| 256x256 class-to-image smoke tests | RTX 3060 12 GB may work with custom low-VRAM wrapper |
| 512x512 T2I experimentation | 24 GB GPU preferred; 12 GB is experimental |
| 1024x1024 T2I | 48 GB or 80 GB GPU recommended |
| Official 50K eval | Multi-GPU, preferably A100/H100 class |
| Training | Multi-GPU 80 GB class, not local RTX 3060 |

## 10. SynthaEngine Implementation Strategy

SynthaEngine is workflow-first. PixelDiT should be implemented as a new model
family and new workflow task, preserving all existing public task identifiers.

### Recommended Initial Public Surface

Add:

- `pixeldit.text2img`

Optional second task:

- `pixeldit.class2img`

Do not add initially:

- `pixeldit.img2img`
- `pixeldit.inpaint`
- `pixeldit.controlnet.*`
- `pixeldit.ip_adapter.*`
- `pixeldit.video.*`
- `pixeldit.lora.*`

Those can be future extensions only if official or stable custom code exists.

### Files To Add In A Future Implementation

```text
backend/pixeldit/__init__.py
backend/pixeldit/pipeline.py
backend/pixeldit/subprocess_runner.py
backend/pixeldit/subprocess_io.py
backend/workflow/pixeldit.py
frontend/pixeldit/text2img.html
frontend/pixeldit/text2img.js
testing/test_pixeldit_workflow.py
testing/test_pixeldit_subprocess.py
testing/test_frontend_pixeldit_scripts.py
```

Files to update:

```text
backend/workflow/types.py
backend/workflow/schema_input.py
backend/workflow/schema_output.py
backend/workflow/engine.py
backend/workflow/catalog.py
docs/WORKFLOW_API.md
docs/PIPELINE_LIFECYCLE.md
frontend/workflow_catalog.js
README.md, if the model family is user-facing
```

### Model Registry

Add model family metadata:

```python
"pixeldit": {"label": "PixelDiT", "aliases": ["pixdit", "pixel-dit"]}
```

Recommended model registry entries:

```json
[
  {
    "name": "PixelDiT-T2I-v1",
    "family": "pixeldit",
    "model_type": "pixeldit-t2i",
    "location_type": "hub",
    "model_id": 20,
    "version": "main",
    "link": "nvidia/PixelDiT-1300M-1024px"
  },
  {
    "name": "PixelDiT-ImageNet-XL-256",
    "family": "pixeldit",
    "model_type": "pixeldit-c2i",
    "location_type": "hub",
    "model_id": 21,
    "version": "main",
    "link": "nvidia/PixelDiT-ImageNet"
  }
]
```

Because the Hugging Face model repos store `.pth` and `.ckpt` checkpoint files,
not Diffusers folders, `resolve_model_source()` may need a PixelDiT-specific
resolver that can resolve:

- Hub repo plus known filename.
- Local checkpoint file.
- Local PixelDiT repository path.
- Local config YAML path.

## 11. Workflow Input Contract

### `pixeldit.text2img`

Recommended Pydantic input model:

```python
class PixelDiTText2ImgInputs(BaseModel):
    prompt: str = ""
    negative_prompt: str = "low quality, worst quality, over-saturated, blurry, deformed, watermark"
    steps: int = Field(default=25, ge=1, le=100)
    cfg_scale: float = Field(default=2.75, ge=0.0, le=30.0)
    width: int = Field(default=512, ge=256, le=1536)
    height: int = Field(default=512, ge=256, le=1536)
    seed: int | None = None
    model: str | None = None
    model_path: str | None = None
    config: str = "t2i/configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml"
    num_images: int = Field(default=1, ge=1, le=1)
    sampling_algo: Literal["flow_dpm-solver"] = "flow_dpm-solver"
    interval_guidance: tuple[float, float] = (0.0, 1.0)
    flow_shift: float | None = None
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    memory_preset: Literal["subprocess_low_vram", "subprocess_standard"] = "subprocess_low_vram"
    quantization: Literal["none", "text_encoder_bnb_8bit", "text_encoder_bnb_4bit"] = "none"
    text_encoder_device: Literal["auto", "cuda", "cpu"] = "auto"
    keep_intermediate_dir: bool = False
    experimental_ack: bool = True
    batch_id: str | None = None
```

Suggested validation:

- `num_images` must be 1 for local 12 GB defaults.
- `sampling_algo` must be `flow_dpm-solver` until another sampler is actually
  implemented.
- `width` and `height` should match supported PixelDiT aspect buckets or be
  mapped to the nearest official bucket.
- If `width > 512 or height > 512` on local profile, require
  `experimental_ack=true`.
- Reject quantization values that are not implemented on the current platform.

### Example Workflow Payload

```json
{
  "kind": "workflow",
  "payload": {
    "tasks": [
      {
        "id": "t1",
        "type": "pixeldit.text2img",
        "inputs": {
          "prompt": "a quiet glass greenhouse at sunrise, detailed plants, soft mist",
          "negative_prompt": "low quality, blurry, watermark",
          "width": 512,
          "height": 512,
          "steps": 25,
          "cfg_scale": 2.75,
          "seed": 2026,
          "memory_preset": "subprocess_low_vram",
          "experimental_ack": true
        }
      }
    ],
    "return": "@t1.images"
  }
}
```

### `pixeldit.class2img`

Recommended Pydantic input model:

```python
class PixelDiTClass2ImgInputs(BaseModel):
    class_id: int = Field(ge=0, le=999)
    class_label: str | None = None
    checkpoint: Literal[
        "imagenet256_pixeldit_xl_epoch80.ckpt",
        "imagenet256_pixeldit_xl_epoch160.ckpt",
        "imagenet256_pixeldit_xl_epoch320.ckpt",
        "imagenet512_pixeldit_xl.ckpt"
    ] = "imagenet256_pixeldit_xl_epoch320.ckpt"
    resolution: Literal[256, 512] = 256
    steps: int = Field(default=100, ge=1, le=200)
    cfg_scale: float = Field(default=2.75, ge=0.0, le=30.0)
    time_shift: float = 1.0
    guidance_interval_min: float = Field(default=0.1, ge=0.0, le=1.0)
    guidance_interval_max: float = Field(default=0.9, ge=0.0, le=1.0)
    seed: int | None = None
    num_images: int = Field(default=1, ge=1, le=4)
    memory_preset: Literal["subprocess_low_vram", "subprocess_standard"] = "subprocess_low_vram"
    experimental_ack: bool = True
    batch_id: str | None = None
```

## 12. Official PixelDiT Flags And Options To Preserve

These are the user-changeable knobs exposed by NVIDIA's code or config files.
SynthaEngine does not need to expose all of them in the first UI, but the
backend plan should know where they map.

### T2I Inference Flags

| Official option | Meaning | SynthaEngine field |
| --- | --- | --- |
| `--config` | T2I config YAML | `config` |
| `--model_path` | `.pth` checkpoint | `model_path` or registry model |
| `--txt_file` | Prompt text file | Backend temp file, not public field |
| `--json_file` | Prompt JSON file | Not initial public field |
| `--custom_height` | Output height | `height` |
| `--custom_width` | Output width | `width` |
| `--cfg_scale` | Classifier-free guidance | `cfg_scale` |
| `--step` | Sampling steps | `steps` |
| `--seed` | Seed | `seed` |
| `--negative_prompt` | Negative CFG prompt | `negative_prompt` |
| `--work_dir` | Output root | Backend batch output dir |
| `--sample_nums` | Max prompts to process | Internal; use `num_images` cautiously |
| `--bs` | Batch size | Keep fixed at 1 locally |
| `--sampling_algo` | Sampler | Fixed `flow_dpm-solver` |
| `--interval_guidance` | CFG interval | `interval_guidance` |
| `--custom_image_size` | Square output override | Use `width`/`height` instead |
| `--start_index`, `--end_index` | Prompt range | Not initial public field |
| `--tar_and_del` | Archive output dir | Not initial public field |
| `--if_save_dirname` | Metrics helper | Not initial public field |
| `--ablation_key`, `--ablation_selections` | Research ablations | Not public |

### T2I Config Knobs

| Config area | Important options |
| --- | --- |
| Data | `image_size`, `multi_scale`, `aspect_ratio_type`, `caption_proportion`, `clip_thr`, `load_text_feat` |
| Model | `mixed_precision`, `fp32_attention`, `patch_size`, `hidden_size`, `patch_depth`, `pixel_depth`, `pixel_hidden_size`, `pixel_attn_hidden_size`, `num_text_blocks`, `txt_embed_dim`, `txt_max_length`, `use_text_rope` |
| Text encoder | `text_encoder_name`, `model_max_length`, `chi_prompt`, `y_norm`, `y_norm_scale_factor` |
| Scheduler | `predict_flow_v`, `noise_schedule`, `flow_shift`, `weighting_scheme`, `vis_sampler` |
| Training | `train_batch_size`, `num_epochs`, `gradient_accumulation_steps`, `grad_checkpointing`, `gradient_clip`, `repa_loss_weight`, `optimizer`, `lr`, `save_model_steps` |

### C2I Evaluation Flags

| Official option | Meaning | SynthaEngine field |
| --- | --- | --- |
| `-c configs/pix256_xl.yaml` | Model/eval config | `config` or `resolution` |
| `--ckpt_path` | ImageNet checkpoint | `checkpoint` |
| `--model.diffusion_sampler.init_args.num_steps` | Steps | `steps` |
| `--model.diffusion_sampler.init_args.guidance` | CFG scale | `cfg_scale` |
| `--model.diffusion_sampler.init_args.timeshift` | Flow time shift | `time_shift` |
| `--model.diffusion_sampler.init_args.guidance_interval_min` | Guidance interval start | `guidance_interval_min` |
| `--model.diffusion_sampler.init_args.guidance_interval_max` | Guidance interval end | `guidance_interval_max` |
| `--seed_everything` | Global seed | `seed` |
| `--per_run_seed` | Random per-run seed behavior | Internal |

### C2I Training Script Flags

| Official flag | Meaning |
| --- | --- |
| `--num-nodes` | Distributed training node count |
| `--num-gpus` | GPUs per node |
| `--master-addr` | DDP master address |
| `--master-port` | DDP master port |
| `--node-rank` | DDP node rank |
| `--config` | C2I config YAML |
| `--ckpt-path` | Resume checkpoint |

Training flags should be documented, not exposed in the main SynthaEngine image
generation UI.

## 13. Runtime Wrapper Design

Use a subprocess runtime for PixelDiT.

```text
workflow task
  -> backend/workflow/pixeldit.py
  -> backend/pixeldit/pipeline.py
  -> backend/pixeldit/subprocess_runner.py
  -> temporary JSON input
  -> PixelDiT import or CLI call
  -> SynthaEngine output PNGs
  -> JSON result
```

Why subprocess:

- PixelDiT is heavy and custom.
- Windows often does not return all GPU/system memory cleanly after large
  PyTorch pipelines in a long-lived process.
- SynthaEngine already uses this pattern for heavy model families.
- It keeps dependency and cleanup failures localized.

### Wrapper Pseudocode

```python
def generate_text2img(params: dict[str, object]) -> dict[str, list[str]]:
    return run_pixeldit_subprocess("text2img", params)


def run_pixeldit_subprocess(operation: str, params: dict[str, object]) -> dict[str, list[str]]:
    with tempfile.TemporaryDirectory(prefix="pixeldit_") as tmpdir:
        input_path = Path(tmpdir) / "input.json"
        output_path = Path(tmpdir) / "output.json"
        input_path.write_text(json.dumps({"operation": operation, "params": params}))
        completed = subprocess.run(
            [sys.executable, "-m", "backend.pixeldit.subprocess_runner", str(input_path), str(output_path)],
            cwd=str(REPO_ROOT),
        )
        payload = json.loads(output_path.read_text())
        if completed.returncode != 0 or not payload.get("ok"):
            raise RuntimeError(f"PixelDiT subprocess failed: {payload.get('error')}")
        return payload["result"]
```

### Output Handling

PixelDiT's official script writes into `work_dir/vis/...`. SynthaEngine should
avoid exposing that tree directly. The wrapper should either:

1. Save directly to `outputs/batch_<batch_id>/`, or
2. Let PixelDiT write to a temporary dir, then copy final images into
   `outputs/batch_<batch_id>/`.

Recommended output shape:

```json
{
  "images": [
    "/outputs/batch_abcd1234/abcd1234_2026.png"
  ]
}
```

For consistency with other families, include PNG metadata:

```json
{
  "mode": "txt2img",
  "pipeline": "pixeldit",
  "prompt": "...",
  "negative_prompt": "...",
  "steps": 25,
  "cfg_scale": 2.75,
  "width": 512,
  "height": 512,
  "seed": 2026,
  "model": "PixelDiT-T2I-v1",
  "sampling_algo": "flow_dpm-solver",
  "batch_id": "abcd1234"
}
```

## 14. Dependency Plan

NVIDIA's reference requirements include:

```text
lightning==2.5.0.post0
omegaconf==2.3.0
torch==2.5.0
diffusers==0.30.0
transformers==5.1.0
jsonargparse[signatures]==4.27.7
torchvision
timm
accelerate
gradio
wandb
h5py
webdataset
pyrallis
termcolor
```

SynthaEngine currently has its own dependency stack. Do not blindly downgrade
the repo's existing Diffusers install to `0.30.0`.

Recommended dependency approach:

1. Create an isolated PixelDiT smoke-test environment first.
2. Prove NVIDIA's reference T2I and C2I scripts run outside SynthaEngine.
3. Import only the required PixelDiT modules into SynthaEngine.
4. If dependency conflicts appear, keep PixelDiT in an isolated subprocess
   environment and call that interpreter from `backend/pixeldit/pipeline.py`.

Possible subprocess interpreter setting:

```python
PIXELDIT_PYTHON = os.getenv("SYNTHA_PIXELDIT_PYTHON") or sys.executable
```

This lets SynthaEngine use:

```powershell
$env:SYNTHA_PIXELDIT_PYTHON="C:\Users\Admin\DiffusersProject\PixelDiT\.venv\Scripts\python.exe"
```

## 15. Frontend Plan

Add a dedicated PixelDiT text-to-image page instead of reusing Flux or SDXL UI.

Recommended controls:

| Control | Widget | Default |
| --- | --- | --- |
| Prompt | Textarea | empty |
| Negative prompt | Textarea | official negative prompt from model card |
| Model | Model select for family `pixeldit` | first registry entry |
| Width | Number/select | 512 |
| Height | Number/select | 512 |
| Steps | Number | 25 |
| CFG scale | Slider/number | 2.75 |
| Seed | Number/randomize | random |
| Interval guidance min/max | Advanced number controls | 0.0 / 1.0 |
| Precision | Select | `bf16` |
| Memory preset | Select | `subprocess_low_vram` |
| Quantization | Select | `none`, disabled until implemented |
| Experimental acknowledgement | Checkbox | true |

For 12 GB VRAM, UI should show 512x512 as the default. 1024x1024 can be present
as an advanced option with a warning state, but the backend should enforce safe
limits.

## 16. Workflow Catalog And Capability Metadata

Add PixelDiT to the catalog metadata:

```python
"_MODEL_FAMILY_METADATA": {
    "pixeldit": {"label": "PixelDiT", "aliases": ["pixdit", "pixel-dit"]}
}
```

Expected capability matrix:

```json
{
  "pixeldit": {
    "label": "PixelDiT",
    "aliases": ["pixdit", "pixel-dit"],
    "task_types": ["pixeldit.text2img"],
    "features": {
      "text2img": true,
      "text2video": false,
      "img2img": false,
      "inpaint": false,
      "controlnet": false,
      "multi_controlnet": false,
      "hires_fix": false,
      "lora_adapters": false,
      "ip_adapter": false,
      "scheduler": false,
      "true_cfg_scale": false
    }
  }
}
```

PixelDiT uses `cfg_scale` rather than the existing `guidance_scale` naming in
some families. The UI can label it "CFG scale" and the workflow contract can
keep `cfg_scale` to match NVIDIA's flags.

## 17. Testing Plan

Add focused tests before trying full GPU renders.

| Test | Purpose |
| --- | --- |
| `testing/test_pixeldit_workflow.py` | Validates task schema, registry dispatch, output shape, unsupported values, and mocked runtime calls. |
| `testing/test_pixeldit_subprocess.py` | Validates subprocess JSON IO, error propagation, path serialization, and output URL shape. |
| `testing/test_workflow_catalog_capabilities.py` update | Confirms `pixeldit` family appears with correct task flags. |
| `testing/test_docs_model_contract.py` update | Confirms docs mention `pixeldit.text2img` if docs contract tests require it. |
| `testing/test_frontend_pixeldit_scripts.py` | Confirms page script builds valid workflow payloads. |

Required validation commands after implementation:

```powershell
.venv\Scripts\python.exe -m compileall backend
.venv\Scripts\python.exe -m pytest testing/test_pixeldit_workflow.py -q
.venv\Scripts\python.exe -m pytest testing/test_pixeldit_subprocess.py -q
.venv\Scripts\python.exe -m pytest testing/test_workflow_catalog_capabilities.py -q
```

GPU smoke test:

```json
{
  "kind": "workflow",
  "payload": {
    "tasks": [
      {
        "id": "smoke",
        "type": "pixeldit.text2img",
        "inputs": {
          "prompt": "a small red teapot on a wooden table",
          "width": 512,
          "height": 512,
          "steps": 4,
          "cfg_scale": 2.75,
          "seed": 123,
          "experimental_ack": true
        }
      }
    ],
    "return": "@smoke.images"
  }
}
```

Success criteria:

- Job reaches `succeeded`.
- One image is saved under `/outputs/batch_<id>/`.
- PNG metadata includes PixelDiT parameters.
- Process exits and VRAM returns close to baseline.
- A second job can run without restarting the API server.

## 18. Step-By-Step Implementation Plan

### Phase 0: License And Repository Prep

1. Confirm NSCLv1 non-commercial research/evaluation constraints.
2. Decide whether PixelDiT code is vendored, submoduled, or referenced by an
   external path.
3. Add configuration variables:

```powershell
set SYNTHA_PIXELDIT_REPO=C:\Users\Admin\DiffusersProject\PixelDiT
set SYNTHA_PIXELDIT_PYTHON=C:\Users\Admin\DiffusersProject\PixelDiT\.venv\Scripts\python.exe
```

### Phase 1: Standalone PixelDiT Smoke Test

1. Clone NVIDIA PixelDiT outside SynthaEngine or under an explicit third-party
   folder.
2. Create a PixelDiT-specific venv or Docker environment.
3. Download `pixeldit_t2i_v1.pth`.
4. Run the official T2I command with one prompt at 512x512 and low steps.
5. Record peak VRAM and system RAM.

Official-style command:

```bash
cd t2i/
python inference.py \
  --config configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml \
  --model_path pixeldit_t2i_v1.pth \
  --txt_file prompts.txt \
  --custom_height 512 --custom_width 512 \
  --cfg_scale 2.75 --seed 2026 \
  --step 25 \
  --negative_prompt "low quality, worst quality, over-saturated, blurry, deformed, watermark" \
  --work_dir "."
```

### Phase 2: Backend Runtime

1. Add `backend/pixeldit/subprocess_io.py`.
2. Add `backend/pixeldit/subprocess_runner.py`.
3. Add `backend/pixeldit/pipeline.py`.
4. Implement `generate_text2img(params)`.
5. Start with CLI delegation to NVIDIA `t2i/inference.py` for minimal risk.
6. Later replace CLI delegation with direct imports for better output control.
7. Save outputs to SynthaEngine batch directories.
8. Add PNG metadata.
9. Enforce one subprocess at a time with a semaphore.

### Phase 3: Workflow Integration

1. Add `PixelDiTText2ImgInputs`.
2. Add task literal `pixeldit.text2img`.
3. Add `backend/workflow/pixeldit.py`.
4. Register task input/output models in `engine.py`.
5. Register task handler in `TASK_REGISTRY`.
6. Add catalog family metadata.
7. Ensure output schema uses `ImagesOutput` or `ImagesWithBatchOutput`.

### Phase 4: Frontend

1. Add `frontend/pixeldit/text2img.html`.
2. Add `frontend/pixeldit/text2img.js`.
3. Reuse existing workflow client and gallery components.
4. Use model registry filtering by `family=pixeldit`.
5. Keep advanced fields collapsed by default.
6. Disable unsupported features with no fake toggles.

### Phase 5: Docs

1. Update `docs/WORKFLOW_API.md`.
2. Update `docs/PIPELINE_LIFECYCLE.md` to mention subprocess cleanup.
3. Add examples for `pixeldit.text2img`.
4. Add compatibility notes:
   - Not a Diffusers pipeline yet.
   - NSCLv1 non-commercial license.
   - 12 GB VRAM is experimental.

### Phase 6: Tests And Validation

1. Run compileall.
2. Run focused workflow tests.
3. Run mocked subprocess tests.
4. Run catalog/capability tests.
5. Run frontend script tests.
6. Run one manual GPU smoke test if PixelDiT dependencies and checkpoint are
   installed.

## 19. Risk Register

| Risk | Mitigation |
| --- | --- |
| Dependency conflict with existing Diffusers stack | Use isolated PixelDiT subprocess interpreter first. |
| 12 GB VRAM OOM | Default 512x512, batch 1, subprocess cleanup, reject unsafe settings unless acknowledged. |
| BF16 problems on RTX 3060 / Windows | Add `precision` fallback and test FP16. |
| Output directory mismatch from NVIDIA script | Copy outputs into SynthaEngine batch directories or implement direct save wrapper. |
| No official Diffusers pipeline | Use custom runtime now; design workflow contract so a future Diffusers backend can replace it. |
| License constraints | Mark family as research/evaluation; do not use commercially without legal review. |
| Quantization quality loss | Keep quantization disabled by default and metadata-visible. |
| FlowEdit ambiguity | Do not expose editing task until code and contract are validated. |

## 20. Recommended First PR Scope

Keep the first implementation intentionally small:

- `pixeldit.text2img` only.
- One model checkpoint: `nvidia/PixelDiT-1300M-1024px`.
- 512x512 default.
- Batch size 1.
- `flow_dpm-solver` only.
- `quantization: "none"` only.
- No LoRA, ControlNet, IP-Adapter, img2img, inpaint, or training.
- Subprocess execution only.

This creates a stable base without pretending PixelDiT is already a normal
Diffusers pipeline.

## 21. Sources

- PixelDiT project page: https://pixeldit.github.io/
- NVIDIA PixelDiT GitHub repository: https://github.com/NVlabs/PixelDiT
- PixelDiT ImageNet model card: https://huggingface.co/nvidia/PixelDiT-ImageNet
- PixelDiT T2I model card: https://huggingface.co/nvidia/PixelDiT-1300M-1024px
- PixelDiT paper page: https://arxiv.org/abs/2511.20645
- Diffusers pipeline overview: https://huggingface.co/docs/diffusers/main/en/api/pipelines/overview
- Diffusers memory optimization docs: https://huggingface.co/docs/diffusers/main/en/optimization/memory
- Diffusers quantization overview: https://huggingface.co/docs/diffusers/main/en/quantization/overview
- Diffusers bitsandbytes docs: https://huggingface.co/docs/diffusers/main/en/quantization/bitsandbytes
- Diffusers GGUF docs: https://huggingface.co/docs/diffusers/main/en/quantization/gguf
- RunPod GPU cloud page: https://www.runpod.io/product/cloud-gpus
- Lambda Cloud instances page: https://lambda.ai/instances
- Vast.ai pricing docs: https://docs.vast.ai/guides/instances/pricing
- Hugging Face Inference Endpoints pricing: https://huggingface.co/docs/inference-endpoints/en/pricing
