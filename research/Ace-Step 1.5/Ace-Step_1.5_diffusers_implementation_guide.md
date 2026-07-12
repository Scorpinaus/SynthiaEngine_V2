# Ace-Step 1.5 Diffusers Implementation Guide for SynthaEngine

Date: 2026-05-30

Selected model architecture: Ace-Step 1.5  
Primary Diffusers pipeline: `diffusers.AceStepPipeline`  
Primary Hub repository: `ACE-Step/Ace-Step1.5`  
Initial recommended SynthaEngine family: `ace-step`  
Initial recommended workflow task: `ace-step.text2music`

This guide explains what Ace-Step 1.5 is, what modalities and sub-pipelines are
available, how feasible it is on a PC with 64 GB system RAM and an RTX 3060
with 12 GB VRAM, and how it should be implemented in SynthaEngine later. No
application code has been changed by this guide.

## 1. Executive Summary

Ace-Step 1.5 is an open-source music generation model family. It generates
variable-length stereo audio at 48 kHz from text prompts and optional structured
lyrics. The full upstream system combines a language-model planner with a
Diffusion Transformer synthesizer. Diffusers exposes the generation/editing
side through `AceStepPipeline`, which wraps the DiT synthesizer, audio VAE,
text encoder, condition encoder, and scheduler.

The practical answer for your PC:

- Your repo virtual environment already has `diffusers 0.38.0`.
- `AceStepPipeline`, `AceStepTransformer1DModel`, and `AutoencoderOobleck`
  import successfully.
- CUDA and BF16 are available on the RTX 3060 12 GB.
- `bitsandbytes` is installed.
- `torchao`, `optimum-quanto`, `flash_attn`, and `soundfile` are not installed.
- `scipy` and `torchaudio` are installed, so a future smoke test can write WAV
  files without immediately adding `soundfile`.

Recommended local path:

1. Start with `ACE-Step/Ace-Step1.5` turbo.
2. Use short 10-30 second smoke tests.
3. Run one audio per job.
4. Prefer model CPU offload first, then direct CUDA only if measured stable.
5. Keep base/SFT and XL variants behind explicit advanced controls.

Recommended implementation path:

1. Add audio artifact/output support first.
2. Add `ace-step.text2music` only.
3. Use a one-shot subprocess runtime.
4. Add editing tasks after text-to-music is proven.

Do not implement Ace-Step as an image/video task. It is an audio/music family
and requires new audio workflow/output support in SynthaEngine.

## 2. What This Is

Ace-Step 1.5 is a music foundation model created by the ACE-Step team. The
Diffusers documentation describes it as an open-source text-to-music model that
generates commercial-grade stereo music with lyrics from text prompts.

The model's main ideas:

- Text and lyrics describe the music.
- A Qwen3-based text encoder embeds the prompt and lyrics.
- An `AceStepConditionEncoder` combines text, lyric, and timbre conditioning.
- An `AutoencoderOobleck` VAE compresses/decompresses 48 kHz stereo waveforms
  into 25 Hz stereo latents.
- An `AceStepTransformer1DModel` diffusion transformer denoises audio latents.
- A `FlowMatchEulerDiscreteScheduler` drives the flow-matching sampling process.

The full upstream system also includes language-model planning: it can turn a
simple user request into a richer song blueprint with metadata, lyrics, and
captioning. The Diffusers pipeline is the inference pipeline for the DiT/audio
synthesis side. That distinction matters for SynthaEngine: first integration
should call `AceStepPipeline` directly and expose prompt/lyrics/music metadata
controls, not attempt to recreate the full upstream planner UI.

### Architecture Snapshot

| Component | Role |
| --- | --- |
| `AceStepPipeline` | Diffusers pipeline for ACE-Step music generation and audio editing tasks |
| `AutoencoderOobleck` | Audio VAE; waveform to/from low-rate stereo latents |
| `AceStepTransformer1DModel` | DiT denoiser operating in latent audio space |
| Qwen3-based text encoder | Encodes prompts and lyrics |
| `AceStepConditionEncoder` | Combines text, lyric, and timbre conditioning |
| `FlowMatchEulerDiscreteScheduler` | Flow-matching scheduler |
| Optional audio tokenizer/detokenizer | Semantic audio code support for cover-like workflows |

### Modalities

Ace-Step 1.5 is an audio/music pipeline.

Inputs:

- text prompt,
- optional structured lyrics,
- optional music metadata such as BPM, key, and time signature,
- optional source audio for audio-to-audio/editing tasks,
- optional reference audio for timbre/style transfer,
- optional semantic audio codes,
- optional track names/classes for stem-style tasks.

Outputs:

- generated audio tensor from Diffusers,
- recommended persisted format for SynthaEngine: WAV under `outputs/batch_*`.

It is not a text-to-image, image-to-image, inpaint, or video pipeline.

## 3. Modalities and Sub-Pipelines Available

Diffusers lists these supported `task_type` values for `AceStepPipeline`:

| Task type | What it does | Required/important inputs | Initial SynthaEngine status |
| --- | --- | --- | --- |
| `text2music` | Generate music from text prompts and optional lyrics | `prompt`, `lyrics`, `audio_duration`, optional BPM/key/time signature | Implement first |
| `cover` | Generate audio from source audio or semantic codes with timbre transfer from reference audio | `reference_audio`, optional `audio_codes`, prompt/lyrics | Later |
| `repaint` | Regenerate a section of existing audio while keeping the rest | `src_audio`, `repainting_start`, `repainting_end` | Later |
| `extract` | Extract a specific track such as vocals or drums | `src_audio`, `track_name` | Later |
| `lego` | Generate a specific track based on audio context | `src_audio`, `track_name`, repaint region | Later |
| `complete` | Complete input audio with additional track classes | `src_audio`, `complete_track_classes` | Later |

Initial implementation should only expose `ace-step.text2music`. The remaining
tasks require audio upload/reference support, track-specific UI, source-audio
normalization, and stronger validation around audio duration and sample rate.

## 4. Model Variants

Diffusers documents three Ace-Step 1.5 DiT checkpoints with shared architecture
and different guidance behavior:

| Variant | Hub repo | CFG behavior | Default steps | Default `guidance_scale` | Default `shift` | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| turbo | `ACE-Step/Ace-Step1.5` | guidance-distilled, CFG off | 8 | ignored | 3.0 | Best first local target |
| base | `ACE-Step/acestep-v15-base` | CFG/APG on | 8 documented; often 30-60 for quality | 7.0 | 3.0 | More tunable, slower |
| sft | `ACE-Step/acestep-v15-sft` | CFG/APG on | 8 documented; often 30-60 for quality | 7.0 | 3.0 | Higher quality, less broad than base |

The model card for base/SFT reports a 2B parameter model and BF16 weights. The
base and SFT file trees show roughly 4.79 GB for the main model file. The main
turbo repository file tree reports roughly 10.1 GB total files because it also
contains additional model components.

### Advanced XL Turbo Note

There is also a newer official Diffusers-format XL Turbo checkpoint:

```text
ACE-Step/acestep-v15-xl-turbo-diffusers
```

Its model card describes it as a 5B-parameter flow-matching DiT packaged in
standard Diffusers layout with `model_index.json` plus component folders. It is
not the recommended first target for this PC. The non-Diffusers XL Turbo card
lists these practical VRAM expectations:

- 12 GB: possible only with CPU offload plus INT8 quantization,
- 16 GB: possible with CPU offload,
- 20 GB or more: more comfortable without offload,
- 24 GB: better for full quality with the larger LM path.

For SynthaEngine, XL should be an advanced option after the 2B turbo path works.

## 5. Current Local Environment Check

Verified in this repo virtual environment:

```powershell
.venv\Scripts\python.exe -c "import diffusers, torch; print(diffusers.__version__); print(hasattr(diffusers, 'AceStepPipeline')); print(torch.cuda.is_available()); print(torch.cuda.is_bf16_supported())"
```

Observed:

```text
diffusers 0.38.0
has AceStepPipeline True
has AceStepTransformer1DModel True
has AutoencoderOobleck True
torch 2.10.0+cu128
cuda True
bf16 True
gpu NVIDIA GeForce RTX 3060
vram 12884377600
```

Dependency availability:

| Dependency | Local status | Implementation meaning |
| --- | --- | --- |
| `bitsandbytes` | installed | Existing `bnb_8bit` helper can be evaluated |
| `scipy` | installed | Can write WAV with `scipy.io.wavfile.write` |
| `torchaudio` | installed | Can load/resample audio if needed |
| `soundfile` | not installed | Official examples use it, but app cannot assume it yet |
| `flash_attn` | not installed | Do not expose FlashAttention backends by default |
| `torchao` | not installed | Do not expose torchao quantization by default |
| `optimum-quanto` | not installed | Do not expose Quanto quantization by default |

Local call signature:

```python
(
    prompt=None,
    lyrics="",
    audio_duration=60.0,
    vocal_language="en",
    num_inference_steps=8,
    guidance_scale=7.0,
    shift=3.0,
    generator=None,
    latents=None,
    output_type="pt",
    return_dict=True,
    callback=None,
    callback_steps=1,
    callback_on_step_end=None,
    callback_on_step_end_tensor_inputs=("latents",),
    instruction=None,
    max_text_length=256,
    max_lyric_length=2048,
    bpm=None,
    keyscale=None,
    timesignature=None,
    task_type="text2music",
    track_name=None,
    complete_track_classes=None,
    src_audio=None,
    reference_audio=None,
    audio_codes=None,
    repainting_start=None,
    repainting_end=None,
    audio_cover_strength=1.0,
    cfg_interval_start=0.0,
    cfg_interval_end=1.0,
    timesteps=None,
)
```

## 6. Hardware Feasibility on 64 GB RAM + RTX 3060 12 GB

Your PC is a reasonable local target for the 2B Ace-Step variants, especially
turbo, but the implementation should still be conservative. Audio duration
matters: a short 10-30 second generation is very different from a 10-minute
song. The VAE decode stage and latent length scale with duration.

### Yes / No Matrix

| Option | Feasible? | Recommendation |
| --- | --- | --- |
| 2B turbo, 10-30 sec, BF16, one job at a time | Yes | First target |
| 2B turbo with `enable_model_cpu_offload()` | Yes | Recommended default for app stability |
| 2B turbo with direct `.to("cuda")` | Probably yes, but measure | Good for a standalone smoke test, not first app default |
| 2B base/SFT, 30-60 steps | Yes, slower | Advanced quality mode after turbo works |
| Long 5-10 minute audio | Not by default | Add only after duration stress tests |
| Batching multiple songs | No by default | Submit multiple serialized jobs instead |
| Cover/repaint/extract/lego/complete all at launch | No | Add after audio artifact support is solid |
| XL Turbo Diffusers on 12 GB | Yes with caution | Require CPU offload plus INT8 quantization experiment |
| XL Turbo direct CUDA without offload | No | Needs more VRAM |
| GGUF quantized Ace-Step through Diffusers | No | GGUF belongs to separate C++ runner work |
| Hosted GPU with 24 GB+ VRAM | Yes | Good fallback for reliable or long-form jobs |

### Recommended Local Defaults

For `ace-step.text2music`:

```json
{
  "model": "ACE-Step1.5 Turbo",
  "variant": "turbo",
  "audio_duration": 30.0,
  "num_inference_steps": 8,
  "guidance_scale": 1.0,
  "shift": 3.0,
  "num_audios": 1,
  "vocal_language": "en",
  "max_text_length": 256,
  "max_lyric_length": 2048,
  "memory_preset": "model_offload",
  "quantization": "none",
  "output_format": "wav",
  "sample_rate": 48000
}
```

Why these defaults:

- Turbo ignores CFG guidance above 1.0 because guidance is distilled into the
  checkpoint.
- 8 steps is the documented turbo default.
- `shift=3.0` is the documented default sampling recipe.
- 30 seconds is useful but small enough for a first UX.
- One audio per job keeps memory and runtime predictable.
- Model offload is less drastic than sequential offload and should fit the 2B
  path more comfortably on a 12 GB GPU.

## 7. Quantization Options

### Existing SynthaEngine Option: `bnb_8bit`

The repo already has `backend/quantization.py` with a Diffusers pipeline
quantization helper for:

```text
none
bnb_8bit
```

It builds a `PipelineQuantizationConfig` using the bitsandbytes 8-bit backend.
For Ace-Step, the first quantized experiment should be:

```python
from backend.quantization import build_diffusers_pipeline_quantization_config

quantization_config = build_diffusers_pipeline_quantization_config(
    "bnb_8bit",
    components_to_quantize=["transformer", "text_encoder"],
    task_type="ace-step.text2music",
)
```

Then pass it into `AceStepPipeline.from_pretrained(...)` if the installed
Diffusers version accepts it for this pipeline:

```python
from diffusers import AceStepPipeline
import torch

pipe = AceStepPipeline.from_pretrained(
    "ACE-Step/Ace-Step1.5",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
)
```

Do not expose this as the default until a real smoke test confirms both load and
audio quality.

### Diffusers bitsandbytes 8-bit and 4-bit

Diffusers documents bitsandbytes 8-bit and 4-bit quantization for models that
support Accelerate loading and contain `torch.nn.Linear` layers. Ace-Step's
transformer and text encoder are plausible targets, but this must be validated
locally. Quantization can reduce weight memory, but it does not remove runtime
activation memory, audio latent memory, or VAE decode memory.

Recommended product stance:

| Quantization | Local status | Expose in initial UI? | Notes |
| --- | --- | --- | --- |
| `none` BF16 | Available | Yes | First stable path |
| `bnb_8bit` | bitsandbytes installed | Advanced only after smoke test | Best first quantized experiment |
| `bnb_4bit` | possible in Diffusers | No | Requires new helper branch and quality tests |
| torchao | not installed | No | Possible later |
| Quanto | not installed | No | Possible later |
| GGUF | external ecosystem | No | Requires a separate C++ runner, not Diffusers |

### GGUF and AceStep.cpp

There are community GGUF quantizations for Ace-Step, but those are not
Diffusers pipeline checkpoints. Treat them as a separate backend family:

- different model loader,
- different runtime dependency,
- different output path,
- separate smoke tests,
- separate license/model provenance checks.

Do not mix GGUF into the first Diffusers implementation.

## 8. Cloud and Virtual Hosting Options

The official ACE-Step model cards checked for this guide do not show an active
Hugging Face Inference Provider deployment. That means cloud use is best framed
as "rent a GPU and run the app/runtime" rather than "call a managed model API"
unless a provider adds ACE-Step support later.

| Provider | Good option | Why use it | Notes |
| --- | --- | --- | --- |
| Hugging Face Spaces GPU | L4 24 GB, A10G 24 GB, L40S 48 GB, A100 80 GB | Easiest demo hosting near Hub models | Paid GPUs bill while running unless slept/paused |
| RunPod | RTX 3090/4090 24 GB, A40/L40S 48 GB, A100/H100 80 GB | Flexible pods/serverless and custom containers | Good for repeat experiments and private app runtime |
| Lambda GPU Cloud | A10 24 GB, A6000 48 GB, A100/H100 classes when available | Managed VM experience with ML images | Good for longer jobs and development |
| Local RTX 3060 | 2B turbo/base/SFT | No cloud cost, private | Best for smoke tests and short jobs |

Recommended cloud sizing:

- 24 GB VRAM: comfortable for 2B models and some XL quantized/offloaded tests.
- 48 GB VRAM: better for XL and longer audio.
- 80 GB VRAM: best for no-drama validation, benchmarking, and long-form stress
  tests.

## 9. Step-by-Step Method for New Diffusers Pipelines

Use this checklist for any newly released Diffusers pipeline, including
Ace-Step:

1. Identify the official source.
   - Prefer Hugging Face Diffusers docs and model cards.
   - Confirm model IDs, pipeline class, task type, and license.

2. Verify local support.
   - Use `.venv\Scripts\python.exe`.
   - Import the pipeline class.
   - Inspect `__call__`.
   - Check optional runtime dependencies.

3. Classify the modality.
   - Audio, image, video, text, or utility.
   - Do not force a new modality into an old output shape.

4. Estimate memory.
   - Check parameter size and Hub file sizes.
   - Decide direct CUDA, model offload, sequential offload, quantized, or cloud.

5. Write a standalone smoke test.
   - One prompt.
   - One output.
   - Conservative duration/resolution/frames.
   - No API or frontend.

6. Design the workflow task contract.
   - Choose a stable family and task identifier.
   - Add safe defaults.
   - Keep nonessential controls advanced.

7. Implement runtime behind a subprocess.
   - Load pipeline in the child process.
   - Generate output.
   - Write artifact/media files.
   - Release hooks and memory in `finally`.

8. Update catalog, docs, frontend, and tests together.
   - Keep the workflow API contract accurate.
   - Mock heavy model calls in tests.
   - Never download large models in automated tests.

## 10. Standalone Smoke Test

Future file:

```text
tools/smoke_ace_step_text2music.py
```

Use this before touching app integration:

```python
from pathlib import Path

import numpy as np
import torch
from diffusers import AceStepPipeline
from scipy.io import wavfile


model_id = "ACE-Step/Ace-Step1.5"
output_path = Path("ace_step_smoke.wav")

pipe = AceStepPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)

if torch.cuda.is_available():
    # Safer app default for 12 GB VRAM. For direct CUDA comparison, test
    # pipe.to("cuda") separately in a throwaway script.
    pipe.enable_model_cpu_offload()

if getattr(pipe, "vae", None) is not None and hasattr(pipe.vae, "enable_tiling"):
    pipe.vae.enable_tiling()

generator = torch.Generator(device="cpu").manual_seed(42)
output = pipe(
    prompt="An upbeat synthwave track with driving drums and warm analog bass",
    lyrics="[verse]\nNeon lights are calling\n[chorus]\nWe ride the wave tonight",
    audio_duration=20.0,
    vocal_language="en",
    num_inference_steps=8,
    guidance_scale=1.0,
    shift=3.0,
    generator=generator,
)

audio = output.audios[0].detach().cpu().float()
if audio.ndim == 2:
    audio_np = audio.T.numpy()
else:
    audio_np = audio.numpy()
audio_np = np.clip(audio_np, -1.0, 1.0)
wavfile.write(output_path, int(pipe.sample_rate), (audio_np * 32767).astype(np.int16))
print(f"Saved {output_path}")
```

Run later with:

```powershell
.venv\Scripts\python.exe tools\smoke_ace_step_text2music.py
```

Pass criteria:

- model loads,
- one WAV is written,
- generation does not OOM,
- output duration is roughly expected,
- process exits cleanly and VRAM is released.

## 11. SynthaEngine Implementation Plan

This is the future implementation path. It is not performed by this guide.

### Step 1: Add Audio as a First-Class Workflow Media Type

Current workflow docs describe image and video artifacts. Ace-Step needs audio.

Recommended additions:

- audio upload support in `/api/artifacts`,
- new artifact id prefix, for example `u...` for audio,
- accepted extensions: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`,
- normalize source audio to stereo 48 kHz tensors in runtime code,
- update artifact cleanup to include audio artifacts.

Suggested schema shape:

```python
class AudioArtifactRef(BaseModel):
    artifact_id: str = Field(
        ...,
        description="Audio artifact id returned by POST /api/artifacts.",
        pattern=r"^u[0-9a-f]{32}$",
    )


AudioRef: TypeAlias = AudioArtifactRef | str
```

### Step 2: Add Audio Output Shape

Future file:

```text
backend/workflow/schema_output.py
```

Suggested output model:

```python
class AudiosWithBatchOutput(BaseModel):
    batch_id: str = Field(..., description="Batch identifier used to group outputs on disk.")
    audios: list[str] = Field(
        ...,
        description='List of output audio URLs ("/outputs/...").',
    )
```

Expected workflow result:

```json
{
  "batch_id": "b1780000000_1234",
  "audios": ["/outputs/batch_b1780000000_1234/b1780000000_1234_42.wav"]
}
```

### Step 3: Add Input Schemas

Future file:

```text
backend/workflow/schema_input.py
```

Initial schema:

```python
class AceStepText2MusicInputs(BaseModel):
    prompt: str = ""
    lyrics: str = ""
    audio_duration: float = Field(default=30.0, ge=10.0, le=120.0)
    vocal_language: str = "en"
    steps: int = Field(default=8, ge=1, le=100)
    guidance_scale: float = Field(default=1.0, ge=0.0, le=30.0)
    shift: float = Field(default=3.0, ge=1.0, le=3.0)
    seed: int | None = None
    model: str | None = None
    variant: Literal["turbo", "base", "sft", "xl_turbo"] = "turbo"
    num_audios: int = Field(default=1, ge=1, le=1)
    bpm: int | None = Field(default=None, ge=40, le=240)
    keyscale: str | None = None
    timesignature: str | None = None
    instruction: str | None = None
    max_text_length: int = Field(default=256, ge=32, le=512)
    max_lyric_length: int = Field(default=2048, ge=0, le=4096)
    cfg_interval_start: float = Field(default=0.0, ge=0.0, le=1.0)
    cfg_interval_end: float = Field(default=1.0, ge=0.0, le=1.0)
    output_format: Literal["wav"] = "wav"
    memory_preset: Literal["direct_cuda", "model_offload", "sequential_offload"] = "model_offload"
    quantization: Literal["none", "bnb_8bit"] = "none"
    experimental_ack: bool = True
```

Later editing schemas:

```python
class AceStepCoverInputs(AceStepText2MusicInputs):
    reference_audio: AudioRef | None = None
    audio_codes: str | list[str] | None = None
    audio_cover_strength: float = Field(default=1.0, ge=0.0, le=1.0)


class AceStepRepaintInputs(AceStepText2MusicInputs):
    src_audio: AudioRef
    repainting_start: float = Field(default=0.0, ge=0.0)
    repainting_end: float | None = None


class AceStepTrackTaskInputs(AceStepText2MusicInputs):
    src_audio: AudioRef
    track_name: str | None = None
    complete_track_classes: list[str] | None = None
```

Validation rules:

- `experimental_ack` must be true for all Ace-Step tasks at launch.
- `num_audios` must remain 1.
- `cfg_interval_start <= cfg_interval_end`.
- Turbo should normalize `guidance_scale` to `1.0` or warn that CFG is ignored.
- Audio editing tasks must require audio artifact support and 48 kHz stereo
  preprocessing.
- Long durations above 120 seconds should remain disabled until stress-tested.

### Step 4: Add Family Registry Defaults

Recommended model registry fallback:

```text
name: ACE-Step1.5 Turbo
family: ace-step
model_type: diffusers
location_type: hub
link: ACE-Step/Ace-Step1.5
version: turbo
```

Additional optional entries:

```text
ACE-Step1.5 Base -> ACE-Step/acestep-v15-base
ACE-Step1.5 SFT -> ACE-Step/acestep-v15-sft
ACE-Step1.5 XL Turbo Diffusers -> ACE-Step/acestep-v15-xl-turbo-diffusers
```

Resolver behavior:

- If `model` is supplied, require an exact registry match with
  `family == "ace-step"`.
- If no model is supplied, use the first registered `ace-step` model.
- If none exists, fall back to `ACE-Step/Ace-Step1.5`.
- Do not silently use another family.

### Step 5: Add a Runtime Package

Future files:

```text
backend/ace_step/__init__.py
backend/ace_step/pipeline.py
backend/ace_step/subprocess_runner.py
backend/ace_step/subprocess_io.py
```

Runtime responsibilities:

1. Resolve the model source.
2. Build optional quantization config.
3. Load `AceStepPipeline.from_pretrained(..., torch_dtype=torch.bfloat16)`.
4. Apply memory preset.
5. Enable VAE tiling for longer audio if supported.
6. Convert audio inputs to stereo 48 kHz tensors for editing tasks.
7. Generate one audio item.
8. Save WAV output and JSON sidecar metadata.
9. Release hooks and memory in `finally`.

Pseudo-code:

```python
@torch.inference_mode()
def generate_text2music_in_process(params: dict[str, object]) -> dict[str, object]:
    pipe = None
    try:
        pipe = load_ace_step_pipeline(
            model=params.get("model"),
            variant=str(params.get("variant") or "turbo"),
            memory_preset=str(params.get("memory_preset") or "model_offload"),
            quantization=str(params.get("quantization") or "none"),
        )

        if getattr(pipe, "vae", None) is not None and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()

        seed = resolve_seed(params.get("seed"))
        output = pipe(
            prompt=str(params.get("prompt") or ""),
            lyrics=str(params.get("lyrics") or ""),
            audio_duration=float(params.get("audio_duration") or 30.0),
            vocal_language=str(params.get("vocal_language") or "en"),
            num_inference_steps=int(params.get("steps") or 8),
            guidance_scale=float(params.get("guidance_scale") or 1.0),
            shift=float(params.get("shift") or 3.0),
            generator=torch.Generator(device="cpu").manual_seed(seed),
            instruction=params.get("instruction"),
            max_text_length=int(params.get("max_text_length") or 256),
            max_lyric_length=int(params.get("max_lyric_length") or 2048),
            bpm=params.get("bpm"),
            keyscale=params.get("keyscale"),
            timesignature=params.get("timesignature"),
            task_type="text2music",
            cfg_interval_start=float(params.get("cfg_interval_start") or 0.0),
            cfg_interval_end=float(params.get("cfg_interval_end") or 1.0),
        )
        return save_audio_output(output.audios[0], sample_rate=pipe.sample_rate, params=params, seed=seed)
    finally:
        release_pipeline(pipe, logger=logger)
```

Memory preset behavior:

```python
if memory_preset == "direct_cuda":
    pipe.to("cuda")
elif memory_preset == "model_offload":
    pipe.enable_model_cpu_offload()
elif memory_preset == "sequential_offload":
    pipe.enable_sequential_cpu_offload()
else:
    raise ValueError(f"Unsupported Ace-Step memory_preset: {memory_preset}")
```

Important rule:

- Do not call `.to("cuda")` before `enable_sequential_cpu_offload()`.
- Use subprocess execution only in the public generation path.
- Keep direct CUDA as a smoke-test/advanced option, not the default.

### Step 6: Add Workflow Adapter

Future file:

```text
backend/workflow/ace_step.py
```

Suggested adapter:

```python
from __future__ import annotations

from typing import Any


def run_ace_step_text2music_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    generate_text2music = deps["generate_text2music"]
    result = generate_text2music(dict(inputs))
    if not isinstance(result, dict):
        raise ValueError("ace-step.text2music must return an object")
    if not isinstance(result.get("audios"), list):
        raise ValueError("ace-step.text2music must return audios")
    return result
```

### Step 7: Register Workflow Task and Catalog

Future file:

```text
backend/workflow/engine.py
```

Add:

```python
"ace-step.text2music": AceStepText2MusicInputs
"ace-step.text2music": AudiosWithBatchOutput
"ace-step.text2music": _ace_step_text2music
```

Future file:

```text
backend/workflow/catalog.py
```

Add family metadata:

```python
"ace-step": {"label": "Ace-Step", "aliases": ["acestep", "ace-step-1.5"]}
```

Catalog features should gain audio flags:

```json
{
  "ace-step": {
    "label": "Ace-Step",
    "aliases": ["acestep", "ace-step-1.5"],
    "task_types": ["ace-step.text2music"],
    "features": {
      "text2music": true,
      "audio2audio": false,
      "track_editing": false,
      "lora_adapters": false,
      "scheduler": false
    }
  }
}
```

Do not pretend this is `text2video` or `text2img`. Add audio-specific catalog
features when implementing.

### Step 8: Add History and Frontend Audio Playback

Current history supports image and video records. Ace-Step needs audio records.

Future changes:

- `_history_media_type()` recognizes `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`.
- History returns `media_type: "audio"`.
- Write sidecar metadata such as `audio_<batch_id>.wav.json`.
- Add frontend audio preview in history.
- Add reusable `frontend/components/audio_gallery.js`.
- Add `frontend/ace_step/text2music.html` and `frontend/ace_step/text2music.js`.
- Add nav entry under a new audio/music section.

Frontend controls:

- prompt textarea,
- lyrics textarea,
- duration input/slider,
- language select/text input,
- model/variant select,
- steps input,
- guidance scale input,
- shift select or numeric input,
- seed input,
- BPM input,
- key input,
- time signature input,
- memory preset select,
- quantization select,
- advanced instruction/max length/CFG interval controls.

### Step 9: Docs to Update Later

Future files:

- `docs/WORKFLOW_API.md`
- `docs/PIPELINE_LIFECYCLE.md`
- possibly `README.md`

Workflow example:

```json
{
  "kind": "workflow",
  "payload": {
    "tasks": [
      {
        "id": "t1",
        "type": "ace-step.text2music",
        "inputs": {
          "prompt": "upbeat synthwave track with driving drums and warm analog bass",
          "lyrics": "[verse]\nNeon lights are calling\n[chorus]\nWe ride the wave tonight",
          "audio_duration": 30.0,
          "vocal_language": "en",
          "steps": 8,
          "guidance_scale": 1.0,
          "shift": 3.0,
          "memory_preset": "model_offload",
          "quantization": "none",
          "experimental_ack": true
        }
      }
    ],
    "return": "@t1.audios"
  }
}
```

### Step 10: Tests to Add Later

Do not download Ace-Step in automated tests.

Focused tests:

- `AceStepText2MusicInputs` defaults are safe for RTX 3060 12 GB.
- `experimental_ack=false` is rejected.
- `num_audios > 1` is rejected.
- `cfg_interval_start > cfg_interval_end` is rejected.
- `ace-step.text2music` appears in task type discovery.
- Catalog exposes `ace-step` and `text2music`.
- Workflow adapter rejects non-object and missing `audios`.
- Subprocess bridge writes/reads JSON and serializes one job at a time.
- Runtime save helper writes WAV sidecar metadata without loading a real model.
- History recognizes audio media.
- Frontend payload submits `kind: "workflow"` with `ace-step.text2music`.

Validation commands for a future implementation:

```powershell
.venv\Scripts\python.exe -m compileall backend
.venv\Scripts\python.exe -m pytest testing/test_workflow_catalog_capabilities.py -q
.venv\Scripts\python.exe -m pytest testing/test_*workflow*.py -q
.venv\Scripts\python.exe -m pytest testing/test_ace_step*.py -q
```

## 12. Complete Options and Flags Map

Use this map when designing schemas and frontend controls.

| Pipeline option | Type | Default | Expose initially? | Notes |
| --- | --- | --- | --- | --- |
| `prompt` | string/list | `None` | Yes | Music description |
| `lyrics` | string/list | `""` | Yes | Supports `[verse]`, `[chorus]`, etc. |
| `audio_duration` | float | `60.0` | Yes | Use 30.0 app default; cap initially |
| `vocal_language` | string/list | `"en"` | Yes | Should match lyrics |
| `num_inference_steps` | int | `8` | Yes as `steps` | Turbo designed for 8 |
| `guidance_scale` | float | `7.0` | Yes | Ignored by turbo when >1 |
| `shift` | float | `3.0` | Yes advanced | 1.0, 2.0, 3.0 useful |
| `generator` | torch generator | `None` | Indirect via `seed` | CPU generator recommended |
| `latents` | tensor | `None` | No | Internal/repro advanced |
| `output_type` | string | `"pt"` | No | Keep `"pt"` and write WAV yourself |
| `return_dict` | bool | `True` | No | Internal |
| `callback` | callable | `None` | No | Could support progress later |
| `callback_steps` | int | `1` | No | Progress implementation detail |
| `callback_on_step_end` | callable | `None` | No | Progress/cancel later |
| `callback_on_step_end_tensor_inputs` | list | `("latents",)` | No | Internal |
| `instruction` | string | `None` | Advanced | Auto-generated by task type if omitted |
| `max_text_length` | int | `256` | Advanced | Prompt token cap |
| `max_lyric_length` | int | `2048` | Advanced | Lyric token cap |
| `bpm` | int | `None` | Yes | Optional metadata/control |
| `keyscale` | string | `None` | Yes | Example: `C major` |
| `timesignature` | string | `None` | Yes | Example: `4` |
| `task_type` | string | `"text2music"` | No direct free text | Map to separate workflow tasks |
| `track_name` | string | `None` | Later | `extract`/`lego` |
| `complete_track_classes` | list | `None` | Later | `complete` |
| `src_audio` | tensor | `None` | Later via `AudioRef` | Required for edit tasks |
| `reference_audio` | tensor | `None` | Later via `AudioRef` | Timbre/reference conditioning |
| `audio_codes` | string/list | `None` | Later | Semantic code path |
| `repainting_start` | float | `None` | Later | Repaint/lego |
| `repainting_end` | float | `None` | Later | Repaint/lego |
| `audio_cover_strength` | float | `1.0` | Later | Cover blending |
| `cfg_interval_start` | float | `0.0` | Advanced | CFG window |
| `cfg_interval_end` | float | `1.0` | Advanced | CFG window |
| `timesteps` | list | `None` | No | Custom sampling schedule |

SynthaEngine-only controls:

| App option | Values | Default | Notes |
| --- | --- | --- | --- |
| `variant` | `turbo`, `base`, `sft`, `xl_turbo` | `turbo` | Maps model defaults/warnings |
| `model` | registry name | `None` | Fallback to Hub turbo |
| `num_audios` | fixed `1` | `1` | No batching initially |
| `memory_preset` | `direct_cuda`, `model_offload`, `sequential_offload` | `model_offload` | Default for 12 GB stability |
| `quantization` | `none`, `bnb_8bit` | `none` | Advanced after validation |
| `output_format` | `wav` | `wav` | Keep simple |
| `experimental_ack` | bool | `true` | Required initially |

## 13. Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Audio is not currently a first-class workflow media type | Ace-Step cannot fit existing image/video output shapes | Add audio artifacts, audio outputs, history support first |
| `soundfile` is not installed | Official sample code will not run unchanged | Use `scipy.io.wavfile.write` or add `soundfile` explicitly later |
| Long audio increases memory and runtime | OOM or bad UX | Cap duration to 120 sec initially; default 30 sec |
| Direct CUDA may be unstable with other app memory | Job failure on RTX 3060 | Use subprocess and model offload default |
| Turbo ignores CFG tuning | Confusing UI | Warn/disable guidance for turbo above 1.0 |
| Editing tasks require audio preprocessing | Bad outputs or shape errors | Delay cover/repaint/extract/lego/complete |
| Quantization quality unknown | Artifacts or load failures | Keep quantization advanced and test output quality |
| XL variants may exceed local comfort | OOM/slow jobs | Require explicit variant and quantization/offload |
| Managed provider support may change | Cloud guidance can go stale | Treat providers as GPU hosting options, not fixed model APIs |

## 14. Definition of Done for Future Implementation

A real Ace-Step implementation should be considered done only when:

- Audio artifacts can be uploaded and cleaned up.
- `ace-step.text2music` appears in `/api/workflow/task-types`.
- `/api/workflow/catalog` exposes the `ace-step` family and audio capability.
- A workflow job returns `audios` URLs.
- WAV files are written under `outputs/batch_*`.
- History can show and play audio records.
- Runtime is subprocess-backed and serialized.
- Runtime cleanup follows `docs/PIPELINE_LIFECYCLE.md`.
- Docs include the audio workflow contract.
- Tests cover schemas, catalog, adapter dispatch, subprocess bridge, and history.
- No automated test downloads the real Ace-Step model.

## 15. Bottom Line

Ace-Step 1.5 is a strong fit for a future SynthaEngine audio/music workflow, but
it should not be squeezed into the current image/video abstractions. Your RTX
3060 12 GB machine should be able to run the 2B turbo path for short local
generations, especially with model CPU offload and one-shot subprocess cleanup.

The safest path is:

1. Run the standalone 20-30 second turbo smoke test.
2. Add audio media support to the workflow system.
3. Implement `ace-step.text2music` only.
4. Add base/SFT quality modes.
5. Add editing tasks and XL variants after local behavior is measured.

## Sources

- Diffusers ACE-Step 1.5 docs: https://huggingface.co/docs/diffusers/api/pipelines/ace_step
- ACE-Step/Ace-Step1.5 model card: https://huggingface.co/ACE-Step/Ace-Step1.5
- ACE-Step base model card: https://huggingface.co/ACE-Step/acestep-v15-base
- ACE-Step SFT model card: https://huggingface.co/ACE-Step/acestep-v15-sft
- ACE-Step XL Turbo Diffusers model card: https://huggingface.co/ACE-Step/acestep-v15-xl-turbo-diffusers
- ACE-Step XL Turbo model card: https://huggingface.co/ACE-Step/acestep-v15-xl-turbo
- Diffusers memory optimization docs: https://huggingface.co/docs/diffusers/main/en/optimization/memory
- Diffusers bitsandbytes quantization docs: https://huggingface.co/docs/diffusers/v0.38.0/quantization/bitsandbytes
- Diffusers torchao quantization docs: https://huggingface.co/docs/diffusers/main/quantization/torchao
- Diffusers Quanto quantization docs: https://huggingface.co/docs/diffusers/quantization/quanto
- Hugging Face GPU Spaces docs: https://huggingface.co/docs/hub/spaces-gpus
- RunPod GPU types docs: https://docs.runpod.io/references/gpu-types
- Lambda GPU Cloud: https://lambda.ai/service/gpu-cloud
- ACE-Step 1.5 arXiv paper: https://arxiv.org/abs/2602.00744
