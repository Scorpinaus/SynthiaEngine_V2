# LongCat-AudioDiT Diffusers Implementation Guide for SynthaEngine

Date: 2026-05-30

Selected model architecture: LongCat-AudioDiT  
Primary Diffusers-format Hub repository: `ruixiangma/LongCat-AudioDiT-1B-Diffusers`  
Reference Hub repositories: `meituan-longcat/LongCat-AudioDiT-1B`, `meituan-longcat/LongCat-AudioDiT-3.5B`  
Primary Diffusers pipeline: `LongCatAudioDiTPipeline`

This guide explains what LongCat-AudioDiT is, how practical it is on a local PC
with 64 GB system RAM and an RTX 3060 with 12 GB VRAM, and how to implement it
in SynthaEngine later. No existing application files were changed by this
research note.

## 1. Executive Summary

LongCat-AudioDiT is a Meituan LongCat text-to-audio / text-to-speech diffusion
model family. The current Diffusers integration exposes a standard
`LongCatAudioDiTPipeline` for text-conditioned mono waveform generation.

The practical answer for this PC:

- Yes, the 1B Diffusers-format checkpoint is a plausible local fit for short
  audio clips on an RTX 3060 12 GB if you use FP16 or BF16, one output per job,
  short durations such as 5 to 10 seconds, subprocess execution, and CPU
  offload when needed.
- No, the broader LongCat-AudioDiT reference feature set is not fully available
  through Diffusers yet. The reference repository includes TTS, voice cloning,
  batch inference, and APG-style guidance, but the official Diffusers pipeline
  currently exposes text-to-audio generation only.
- No, the 3.5B reference checkpoint should not be treated as a comfortable
  local target for this PC. It is much larger, is not the primary
  Diffusers-format checkpoint, and should be considered a cloud or dedicated
  high-VRAM path unless an official Diffusers-format and measured quantized
  path is available.
- Quantization is possible to investigate with Diffusers pipeline
  quantization, especially bitsandbytes 8-bit or 4-bit on the `transformer`
  and `text_encoder` components. Treat this as experimental until a real
  local smoke test proves audio quality, memory use, and runtime stability.
- SynthaEngine does not currently have a first-class audio output contract.
  Implementation should add LongCat as a new audio family instead of forcing
  audio outputs into the existing image/video-only conventions.

Important local environment facts gathered from this repo virtual environment:

```text
torch=2.10.0+cu128
diffusers=0.38.0
transformers=5.8.0
accelerate=1.13.0
bitsandbytes=0.49.2
torchao=NOT_AVAILABLE
gguf=NOT_AVAILABLE
soundfile=NOT_AVAILABLE
scipy=AVAILABLE
torchaudio=AVAILABLE
cuda_available=True
cuda_device=NVIDIA GeForce RTX 3060
cuda_capability=(8, 6)
bf16_supported=True
total_vram_gb=12.00
longcat_imports=OK LongCatAudioDiTPipeline PipelineQuantizationConfig BitsAndBytesConfig
```

Official references:

- Diffusers LongCat-AudioDiT docs:
  https://huggingface.co/docs/diffusers/main/api/pipelines/longcat_audio_dit
- Diffusers pipeline overview:
  https://huggingface.co/docs/diffusers/api/pipelines/overview
- Diffusers 0.38.0 release notes:
  https://github.com/huggingface/diffusers/releases
- Diffusers quantization overview:
  https://huggingface.co/docs/diffusers/main/quantization/overview
- Diffusers bitsandbytes quantization:
  https://huggingface.co/docs/diffusers/main/quantization/bitsandbytes
- Diffusers memory optimization:
  https://huggingface.co/docs/diffusers/main/en/optimization/memory
- LongCat-AudioDiT 1B reference model:
  https://huggingface.co/meituan-longcat/LongCat-AudioDiT-1B
- LongCat-AudioDiT 1B Diffusers-format model:
  https://huggingface.co/ruixiangma/LongCat-AudioDiT-1B-Diffusers
- Hugging Face Inference Endpoints:
  https://huggingface.co/docs/huggingface_hub/en/guides/inference_endpoints
- RunPod:
  https://www.runpod.io/
- Vast.ai:
  https://vast.ai/
- Lambda Cloud pricing:
  https://lambda.ai/pricing

## 2. What LongCat-AudioDiT Is

LongCat-AudioDiT is an audio diffusion architecture for generating speech or
audio waveforms from text. The reference model card frames the system as a
simplified TTS pipeline built around two main ideas:

1. A waveform VAE compresses raw waveform audio into a latent waveform space.
2. A diffusion transformer denoises those audio latents under text
   conditioning.

In Diffusers, this becomes a normal `DiffusionPipeline` subclass:

| Area | LongCat-AudioDiT detail |
| --- | --- |
| Main task in Diffusers | Text to audio |
| Main pipeline class | `LongCatAudioDiTPipeline` |
| Denoiser | `LongCatAudioDiTTransformer` |
| Audio autoencoder | `LongCatAudioDiTVae` |
| Text encoder | `UMT5EncoderModel` |
| Tokenizer | `T5TokenizerFast` / T5 tokenizer family |
| Scheduler | `FlowMatchEulerDiscreteScheduler` |
| Default sample rate | 24 kHz from the VAE config |
| Output channels | Mono |
| Official Diffusers-format checkpoint | `ruixiangma/LongCat-AudioDiT-1B-Diffusers` |
| Reference checkpoints | `meituan-longcat/LongCat-AudioDiT-1B`, `meituan-longcat/LongCat-AudioDiT-3.5B` |

The mental model:

1. User provides a text prompt.
2. The tokenizer and UMT5 encoder produce text embeddings.
3. The pipeline chooses a latent audio duration from `audio_duration_s`, or
   estimates one from the text when duration is omitted.
4. The transformer denoises audio latents for `num_inference_steps`.
5. Classifier-free guidance is applied when `guidance_scale > 1.0`.
6. The VAE decodes final latents to a mono waveform.
7. SynthaEngine should save the waveform as `.wav` and return an audio URL.

The current Diffusers docs show the canonical usage:

```python
import soundfile as sf
import torch
from diffusers import LongCatAudioDiTPipeline

pipeline = LongCatAudioDiTPipeline.from_pretrained(
    "ruixiangma/LongCat-AudioDiT-1B-Diffusers",
    torch_dtype=torch.float16,
)
pipeline = pipeline.to("cuda")

prompt = "A calm ocean wave ambience with soft wind in the background."
audio = pipeline(
    prompt,
    audio_duration_s=5.0,
    num_inference_steps=16,
    guidance_scale=4.0,
    generator=torch.Generator("cuda").manual_seed(42),
).audios[0, 0]

sf.write("longcat.wav", audio, pipeline.sample_rate)
```

Because this repo does not currently have `soundfile` installed, use
`scipy.io.wavfile.write` or `torchaudio.save` in SynthaEngine unless you choose
to add `soundfile` as a dependency later.

## 3. Modalities And Sub-Pipelines

The official Diffusers integration is narrower than the LongCat reference
repository.

| Modality or sub-pipeline | Available in official Diffusers pipeline? | Notes |
| --- | --- | --- |
| Text to audio | Yes | `LongCatAudioDiTPipeline` |
| Text to speech | Yes, through text-to-audio | The pipeline returns waveform audio; it does not expose a separate TTS class. |
| Negative prompt / CFG | Yes | `negative_prompt` and `guidance_scale` are in the pipeline call signature. |
| Duration control | Yes | `audio_duration_s` is the direct user-facing control. |
| Seeded generation | Yes | Pass a `torch.Generator`. |
| Latent input | Yes, advanced | `latents` can be passed, but should stay internal/debug-only in SynthaEngine. |
| `np`, `pt`, or latent output | Yes | `output_type` supports `"np"`, `"pt"`, and `"latent"`. User-facing app should save WAV. |
| Voice cloning with prompt audio | No, not in Diffusers pipeline | Present in the reference repo scripts, not in `LongCatAudioDiTPipeline`. |
| Prompt text + prompt audio conditioning | No, not in Diffusers pipeline | Requires reference implementation work if needed. |
| APG guidance | No, not in Diffusers pipeline | Reference repo mentions APG; Diffusers wrapper uses `guidance_scale`. |
| Batch inference file format | No, not in Diffusers pipeline | App can still submit one workflow job per prompt. |
| Text to music / long-form stereo music | No | For that family, compare Stable Audio or ACE-Step instead. |
| Image/video generation | No | LongCat-AudioDiT is audio-only. |
| LoRA adapters | Not documented for this pipeline | Do not expose LoRA until a compatible adapter target and load path are tested. |

Recommended SynthaEngine policy:

- Add LongCat-AudioDiT as a separate `longcat-audiodit` family.
- Add one task first: `longcat-audiodit.text2audio`.
- Keep voice cloning and prompt-audio conditioning out of the first Diffusers
  implementation because those are not official `LongCatAudioDiTPipeline`
  options.
- Keep audio output support generic enough that future audio families such as
  Stable Audio or ACE-Step can reuse it.

## 4. Checkpoint And Size Notes

Current Hub metadata check:

| Repository | Role | Approximate file size | Notes |
| --- | --- | ---: | --- |
| `ruixiangma/LongCat-AudioDiT-1B-Diffusers` | Primary Diffusers checkpoint | 5.31 GB | Has `model_index.json` and pipeline components. |
| `meituan-longcat/LongCat-AudioDiT-1B` | Reference 1B checkpoint | 5.29 GB | Single reference-format safetensors plus scripts. |
| `meituan-longcat/LongCat-AudioDiT-3.5B` | Reference 3.5B checkpoint | 14.28 GB | Higher quality reference target, much heavier. |

The Diffusers-format 1B repository contains:

```text
model_index.json
text_encoder/config.json
text_encoder/model.safetensors              ~1.13 GB
tokenizer/*
transformer/config.json
transformer/diffusion_pytorch_model.safetensors ~3.93 GB
vae/config.json
vae/diffusion_pytorch_model.safetensors     ~0.62 GB
```

The pipeline component map is:

```json
{
  "_class_name": "LongCatAudioDiTPipeline",
  "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
  "text_encoder": ["transformers", "UMT5EncoderModel"],
  "tokenizer": ["transformers", "T5TokenizerFast"],
  "transformer": ["diffusers", "LongCatAudioDiTTransformer"],
  "vae": ["diffusers", "LongCatAudioDiTVae"]
}
```

## 5. Feasibility On 64 GB RAM And RTX 3060 12 GB

### Yes Path: Feasible Locally With Guardrails

Start with this local profile:

```json
{
  "model": "LongCat-AudioDiT-1B-Diffusers",
  "prompt": "A calm ocean wave ambience with soft wind in the background.",
  "negative_prompt": "",
  "audio_duration_s": 5.0,
  "steps": 16,
  "guidance_scale": 4.0,
  "seed": 42,
  "num_audios": 1,
  "precision": "fp16",
  "memory_preset": "cuda",
  "quantization": "none"
}
```

Why this should be realistic:

- The official Diffusers example uses `torch.float16` and CUDA.
- The 1B Diffusers-format model is around 5.31 GB on disk.
- The model generates mono 24 kHz waveform audio, not images or video frames.
- A 5 second clip is only about 120,000 output samples before encoding and
  decoding overhead.
- The installed environment already imports `LongCatAudioDiTPipeline`.
- The RTX 3060 has 12 GB VRAM and local CUDA is visible.

Recommended first local validation:

```powershell
$code = @'
import numpy as np
import torch
from scipy.io import wavfile
from diffusers import LongCatAudioDiTPipeline

pipe = LongCatAudioDiTPipeline.from_pretrained(
    "ruixiangma/LongCat-AudioDiT-1B-Diffusers",
    torch_dtype=torch.float16,
)
pipe.to("cuda")

audio = pipe(
    "A calm ocean wave ambience with soft wind in the background.",
    audio_duration_s=5.0,
    num_inference_steps=16,
    guidance_scale=4.0,
    generator=torch.Generator("cuda").manual_seed(42),
).audios[0, 0]

audio = np.asarray(audio, dtype=np.float32)
audio = np.clip(audio, -1.0, 1.0)
wavfile.write("outputs/longcat_smoke.wav", pipe.sample_rate, (audio * 32767).astype(np.int16))
'@
$code | .venv\Scripts\python.exe -
```

If direct CUDA OOMs, try:

```python
pipe = LongCatAudioDiTPipeline.from_pretrained(
    "ruixiangma/LongCat-AudioDiT-1B-Diffusers",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
pipe.enable_model_cpu_offload()
```

If that still OOMs, try sequential offload:

```python
pipe = LongCatAudioDiTPipeline.from_pretrained(
    "ruixiangma/LongCat-AudioDiT-1B-Diffusers",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
pipe.enable_sequential_cpu_offload()
```

Important Diffusers offload rule: when using
`enable_sequential_cpu_offload()`, do not call `.to("cuda")` first.

### No Path: Not Comfortable Or Not In Scope Locally

Avoid promising these as reliable on the current PC:

- The 3.5B reference checkpoint in a production web-server process.
- Voice cloning through the Diffusers pipeline, because the Diffusers class
  does not expose prompt audio conditioning.
- Long clips near the pipeline's internal 30 second cap without testing.
- Multiple outputs per job while the UI and queue are still single-worker.
- Keeping the audio pipeline hot inside the FastAPI worker.
- Quantized production mode without a measured quality and memory pass.
- Adding audio to `/history` by pretending it is video.

Recommended local limits for first implementation:

| Setting | Conservative default | Initial upper bound |
| --- | ---: | ---: |
| `audio_duration_s` | 5.0 | 30.0 |
| `steps` | 16 | 50 |
| `guidance_scale` | 4.0 | 12.0 |
| `num_audios` | 1 | 1 |
| `output_format` | `wav` | `wav` |
| `precision` | `fp16` | `fp16`, `bf16` |
| `memory_preset` | `cuda` first, fallback to `model_offload` | `cuda`, `model_offload`, `sequential_offload` |
| `quantization` | `none` | `none`, `bnb_8bit`, `bnb_4bit` after validation |

## 6. Quantization Options

Diffusers supports pipeline-level quantization with `PipelineQuantizationConfig`.
The docs list supported backends such as `bitsandbytes_4bit`,
`bitsandbytes_8bit`, `gguf`, `quanto`, and `torchao`.

Local status:

| Quantization backend | Installed locally? | LongCat recommendation |
| --- | --- | --- |
| bitsandbytes | Yes, `0.49.2` | Best first experiment. |
| torchao | No | Do not expose until installed and tested. |
| gguf | No | Not relevant until a compatible Diffusers/GGUF artifact exists. |
| MLX community quantizations | Not for this PC | Useful for Apple Silicon, not RTX 3060/Windows. |

Recommended first quantization experiment:

```python
import torch
from diffusers import LongCatAudioDiTPipeline
from diffusers.quantizers import PipelineQuantizationConfig

quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_8bit",
    quant_kwargs={"load_in_8bit": True},
    components_to_quantize=["transformer", "text_encoder"],
)

pipe = LongCatAudioDiTPipeline.from_pretrained(
    "ruixiangma/LongCat-AudioDiT-1B-Diffusers",
    torch_dtype=torch.float16,
    quantization_config=quant_config,
)
pipe.enable_model_cpu_offload()
```

Recommended 4-bit experiment:

```python
import torch
from diffusers import LongCatAudioDiTPipeline
from diffusers.quantizers import PipelineQuantizationConfig

quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.float16,
    },
    components_to_quantize=["transformer", "text_encoder"],
)

pipe = LongCatAudioDiTPipeline.from_pretrained(
    "ruixiangma/LongCat-AudioDiT-1B-Diffusers",
    torch_dtype=torch.float16,
    quantization_config=quant_config,
)
pipe.enable_model_cpu_offload()
```

Advanced component-specific mapping can use `diffusers.BitsAndBytesConfig` for
the `transformer` and `transformers.BitsAndBytesConfig` for the UMT5
`text_encoder`, because the text encoder is a Transformers component:

```python
import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import LongCatAudioDiTPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

quant_config = PipelineQuantizationConfig(
    quant_mapping={
        "transformer": DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        ),
        "text_encoder": TransformersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        ),
    }
)

pipe = LongCatAudioDiTPipeline.from_pretrained(
    "ruixiangma/LongCat-AudioDiT-1B-Diffusers",
    torch_dtype=torch.float16,
    quantization_config=quant_config,
)
```

Implementation caution:

- Quantize `transformer` first; add `text_encoder` only if needed.
- Do not quantize the VAE initially. It is convolution-heavy and smaller than
  the transformer.
- Quantization can affect audio quality. Compare the same seed and prompt
  against FP16 before exposing it as a stable option.
- Existing `backend/quantization.py` currently supports `none` and `bnb_8bit`.
  A future LongCat implementation should either reuse and expand that helper or
  add a family-local helper with tests.

## 7. Cloud Or Virtual Hosting Options

Use these when local generation is too slow, when you want 3.5B/reference
features, or when you need multiple simultaneous users.

| Option | Good fit | Suggested GPU target | Notes |
| --- | --- | --- | --- |
| Hugging Face Inference Endpoints | Managed production API | 24 GB+ for 1B comfort; 40 GB+ for heavier experiments | Dedicated and autoscaling. Use a custom handler if the model is not catalog-ready. |
| RunPod Pods | Manual experimentation | RTX 4090 24 GB, A40 48 GB, A100 40/80 GB | Good for interactive smoke tests and dependency debugging. |
| RunPod Serverless | Queue-backed API | 24 GB+ | Good match for SynthaEngine's job queue style once a Docker worker is stable. |
| Vast.ai GPU Cloud | Lowest-cost rentals | RTX 4090 24 GB, L40/L40S 48 GB, A40 48 GB, A100 40/80 GB | Marketplace reliability varies; pin a known-good image and storage strategy. |
| Lambda Cloud | More standard GPU VM | A100 40/80 GB or H100 | More predictable than marketplace rentals, usually higher cost. |
| Modal | Python-first serverless GPU | A10G/A100/H100 depending on duration | Good for a thin remote LongCat worker function. |
| Replicate custom model | Public or private hosted model API | A40/A100/H100 class | Useful when you want hosted inference without running a full VM. |

Cloud "yes" option:

- Implement a remote provider mode behind the same workflow task.
- Keep the public task shape stable.
- Return the same `{ "batch_id": "...", "audios": ["/outputs/...wav"] }`
  shape by downloading the generated WAV back into `outputs/`.

Cloud "no" option:

- Do not make cloud mandatory for the first version. The 1B Diffusers checkpoint
  is small enough to justify a local implementation path.
- Do not store provider-specific response shapes in workflow outputs.
- Do not add API keys to model registry rows; keep secrets in environment
  variables or a future secret store.

## 8. Complete SynthaEngine Implementation Plan

This section is a step-by-step plan for fitting LongCat-AudioDiT into the
existing app without breaking the workflow-only API.

### Step 1: Define The Public Feature Boundary

Add a new model family:

```text
longcat-audiodit
```

Add one workflow task first:

```text
longcat-audiodit.text2audio
```

Do not add voice cloning in the first Diffusers implementation. The official
Diffusers pipeline does not expose `prompt_audio`, `prompt_text`, or
`guidance_method="apg"`.

### Step 2: Add Audio Output Contract

Current SynthaEngine output conventions are image/video oriented. Add audio as a
first-class media type.

Recommended output model:

```python
class AudiosWithBatchOutput(BaseModel):
    batch_id: str = Field(..., description="Batch identifier used to group outputs on disk.")
    audios: list[str] = Field(
        ...,
        description='List of output audio URLs ("/outputs/...").',
    )
```

Recommended workflow return:

```json
{
  "batch_id": "b1780100000_1234",
  "audios": [
    "/outputs/batch_b1780100000_1234/b1780100000_1234_42.wav"
  ]
}
```

Recommended metadata sidecar:

```text
outputs/batch_<batch_id>/audio_<batch_id>.wav.json
```

Sidecar shape:

```json
{
  "mode": "longcat-audiodit.text2audio",
  "pipeline": "longcat-audiodit",
  "prompt": "A calm ocean wave ambience with soft wind in the background.",
  "negative_prompt": "",
  "audio_duration_s": 5.0,
  "steps": 16,
  "guidance_scale": 4.0,
  "seed": 42,
  "model": "LongCat-AudioDiT-1B-Diffusers",
  "sample_rate": 24000,
  "channels": 1,
  "precision": "fp16",
  "memory_preset": "cuda",
  "quantization": "none",
  "batch_id": "b1780100000_1234",
  "audios": [
    {
      "filename": "b1780100000_1234_42.wav",
      "path": "batch_b1780100000_1234/b1780100000_1234_42.wav",
      "seed": 42,
      "duration_s": 5.0
    }
  ]
}
```

### Step 3: Extend History And Static Media Handling

Current history supports image extensions and video extensions. Add audio:

```python
HISTORY_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3"}
```

Then update media type inference:

```python
def _history_media_type(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in HISTORY_IMAGE_EXTENSIONS:
        return "image"
    if suffix in HISTORY_VIDEO_EXTENSIONS:
        return "video"
    if suffix in HISTORY_AUDIO_EXTENSIONS:
        return "audio"
    return None
```

Recommended first version should save WAV only. FLAC and MP3 can be recognized
later if encoders are added.

### Step 4: Add Input Schema

Recommended Pydantic model:

```python
class LongCatAudioDiTText2AudioInputs(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    audio_duration_s: float = Field(default=5.0, ge=0.5, le=30.0)
    steps: int = Field(default=16, ge=1, le=100)
    guidance_scale: float = Field(default=4.0, ge=0.0, le=30.0)
    seed: int | None = None
    model: str | None = None
    num_audios: int = Field(default=1, ge=1, le=1)
    precision: Literal["fp16", "bf16"] = "fp16"
    memory_preset: Literal["cuda", "model_offload", "sequential_offload"] = "cuda"
    quantization: Literal["none", "bnb_8bit", "bnb_4bit"] = "none"
    output_format: Literal["wav"] = "wav"
    mono_to_stereo: bool = False
    execution_mode: Literal["subprocess"] = "subprocess"
    batch_id: str | None = None
```

Why these defaults:

- `audio_duration_s=5.0` matches the official example and is a gentle local
  smoke-test duration.
- `steps=16` matches current Diffusers docs usage.
- `guidance_scale=4.0` matches current Diffusers docs usage.
- `num_audios=1` preserves SynthaEngine's serialized local renderer model.
- `execution_mode="subprocess"` follows the repo's memory lifecycle policy.

### Step 5: Add Runtime Module

Add a new package:

```text
backend/longcat_audiodit/
  __init__.py
  pipeline.py
  subprocess_runner.py
```

Runtime responsibilities:

1. Resolve model registry entries for family `longcat-audiodit`.
2. Load `LongCatAudioDiTPipeline` from a Hub repo or local Diffusers folder.
3. Apply precision, offload, and quantization options.
4. Generate one WAV per requested output.
5. Write audio metadata sidecar.
6. Release hooks and CUDA memory in a `finally` block.
7. Return relative output paths to the workflow adapter.

Recommended default registry fallback:

```python
_DEFAULT_MODEL_NAME = "LongCat-AudioDiT-1B-Diffusers"
_DEFAULT_MODEL_LINK = "ruixiangma/LongCat-AudioDiT-1B-Diffusers"
```

### Step 6: Implement Pipeline Loading

Recommended loader sketch:

```python
def load_text2audio_pipeline(
    model_name: str | None,
    *,
    precision: Literal["fp16", "bf16"] = "fp16",
    memory_preset: Literal["cuda", "model_offload", "sequential_offload"] = "cuda",
    quantization: Literal["none", "bnb_8bit", "bnb_4bit"] = "none",
) -> LongCatAudioDiTPipeline:
    entry = _get_longcat_model_entry(model_name)
    source = resolve_model_source(entry)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    quantization_config = build_longcat_quantization_config(quantization)

    kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config

    pipe = LongCatAudioDiTPipeline.from_pretrained(source, **kwargs)

    if memory_preset == "cuda":
        pipe.to("cuda")
    elif memory_preset == "model_offload":
        pipe.enable_model_cpu_offload()
    elif memory_preset == "sequential_offload":
        pipe.enable_sequential_cpu_offload()
    else:
        raise ValueError(f"Unsupported LongCat memory_preset: {memory_preset}")

    cleanup_memory()
    return pipe
```

### Step 7: Save WAV Outputs

Use SciPy because it is already available locally:

```python
import numpy as np
from scipy.io import wavfile

def save_wav(path: Path, audio: object, sample_rate: int) -> None:
    waveform = np.asarray(audio, dtype=np.float32)
    waveform = np.squeeze(waveform)
    waveform = np.clip(waveform, -1.0, 1.0)
    pcm16 = (waveform * 32767.0).astype(np.int16)
    wavfile.write(path, sample_rate, pcm16)
```

If `mono_to_stereo` is exposed:

```python
if mono_to_stereo and waveform.ndim == 1:
    waveform = np.stack([waveform, waveform], axis=1)
```

### Step 8: Add Subprocess Runner

Follow the ERNIE/WAN pattern:

```text
python -m backend.longcat_audiodit.subprocess_runner <input-json> <output-json>
```

The parent should:

- write params to a temp JSON file,
- call the child process under a semaphore,
- read output JSON,
- raise a useful `RuntimeError` on failure.

The child should:

- parse JSON,
- call `_generate_text2audio_subprocess_child`,
- write `{"ok": true, "result": {"audios": [...]}}`,
- always call `cleanup_memory()` in `finally`.

### Step 9: Add Workflow Adapter

Add:

```text
backend/workflow/longcat_audiodit.py
```

Adapter sketch:

```python
def run_longcat_audiodit_text2audio_task(inputs: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    generate_text2audio = deps["generate_text2audio"]
    result = generate_text2audio(dict(inputs))
    if not isinstance(result, dict):
        raise ValueError("longcat-audiodit.text2audio must return an object")
    return result
```

Then update the workflow engine:

- import `LongCatAudioDiTText2AudioInputs`
- import `AudiosWithBatchOutput`
- register `TASK_INPUT_MODELS["longcat-audiodit.text2audio"]`
- register `TASK_OUTPUT_MODELS["longcat-audiodit.text2audio"]`
- add `_longcat_audiodit_runtime_deps`
- add `_longcat_audiodit_text2audio`
- add `TASK_REGISTRY["longcat-audiodit.text2audio"]`

### Step 10: Extend Workflow Catalog Capabilities

Add family metadata:

```python
"longcat-audiodit": {"label": "LongCat-AudioDiT", "aliases": ["longcat", "audiodit"]}
```

Extend `_infer_model_family` for the `longcat-audiodit` prefix.

Add a new capability flag:

```python
"text2audio": False
```

Set it when task type ends with `.text2audio`.

Update UI title inference:

```python
elif task_type.endswith(".text2audio"):
    title = f"{task_type} (Text to Audio)"
```

Add numeric hints:

```python
"audio_duration_s": {"min": 0.5, "max": 30, "step": 0.5},
"num_audios": {"min": 1, "max": 1, "step": 1, "integer": True},
```

Add select hints:

```python
if field_name == "precision":
    hint.update(widget="select", options=["fp16", "bf16"])
if field_name == "quantization":
    hint.update(widget="select", options=["none", "bnb_8bit", "bnb_4bit"])
if field_name == "output_format":
    hint.update(widget="select", options=["wav"])
```

### Step 11: Add Frontend Page

Add:

```text
frontend/longcat_audiodit/text2audio.html
frontend/longcat_audiodit/text2audio.js
frontend/components/audio_gallery.js
```

Workflow payload:

```javascript
const workflowPayload = {
    tasks: [
        {
            id: "t1",
            type: "longcat-audiodit.text2audio",
            inputs,
        },
    ],
    return: "@t1.audios",
};
```

Audio viewer should use:

```html
<audio id="viewer-audio" controls></audio>
```

Keep the controls plain:

- model select,
- prompt,
- negative prompt,
- duration,
- steps,
- guidance scale,
- seed,
- precision,
- memory preset,
- quantization,
- mono-to-stereo checkbox,
- generate button,
- audio output gallery.

### Step 12: Update Public Docs

Update these source-of-truth docs when code is actually implemented:

- `docs/WORKFLOW_API.md`
- `docs/PIPELINE_LIFECYCLE.md`
- `docs/ARCHITECTURE.md`

Document:

- new `audio` media type in `/history`,
- new `longcat-audiodit.text2audio` task,
- input defaults,
- output shape,
- WAV sidecar metadata,
- subprocess lifecycle,
- local feasibility notes.

### Step 13: Add Tests

Minimum focused tests:

```text
testing/test_longcat_audiodit_workflow.py
testing/test_longcat_audiodit_pipeline.py
testing/test_frontend_longcat_audiodit_scripts.py
testing/test_history_api.py
testing/test_workflow_catalog_capabilities.py
```

Test coverage checklist:

- schema defaults are conservative,
- invalid duration and `num_audios` are rejected,
- workflow task is in catalog,
- capability matrix exposes `text2audio`,
- workflow handler forwards all runtime controls,
- subprocess bridge serializes and reads result,
- subprocess runner cleans memory after success and failure,
- fake pipeline output is saved as WAV,
- history recognizes WAV as `media_type="audio"`,
- frontend submits `longcat-audiodit.text2audio`,
- frontend returns `@t1.audios`,
- audio gallery uses an `<audio controls>` element.

### Step 14: Validation Commands

After implementation:

```powershell
.venv\Scripts\python.exe -m compileall backend
.venv\Scripts\python.exe -m pytest testing\test_longcat_audiodit_workflow.py testing\test_longcat_audiodit_pipeline.py
.venv\Scripts\python.exe -m pytest testing\test_workflow_catalog_capabilities.py testing\test_history_api.py
.venv\Scripts\python.exe -m pytest testing\test_frontend_longcat_audiodit_scripts.py
```

Optional real smoke test:

```powershell
.venv\Scripts\python.exe -m backend.longcat_audiodit.subprocess_runner input.json output.json
```

With `input.json`:

```json
{
  "prompt": "A calm ocean wave ambience with soft wind in the background.",
  "negative_prompt": "",
  "audio_duration_s": 5.0,
  "steps": 16,
  "guidance_scale": 4.0,
  "seed": 42,
  "model": "LongCat-AudioDiT-1B-Diffusers",
  "num_audios": 1,
  "precision": "fp16",
  "memory_preset": "cuda",
  "quantization": "none",
  "output_format": "wav"
}
```

## 9. Feature And Flag Inventory

### Diffusers Pipeline Call Options

| Diffusers option | User-facing? | SynthaEngine name | Recommendation |
| --- | --- | --- | --- |
| `prompt` | Yes | `prompt` | Required or default empty string with frontend validation. |
| `negative_prompt` | Yes | `negative_prompt` | Optional. |
| `audio_duration_s` | Yes | `audio_duration_s` | Default 5.0, clamp 0.5 to 30.0. |
| `latents` | No | internal only | Keep for testing/debugging, not public UI. |
| `num_inference_steps` | Yes | `steps` | Map `steps` to Diffusers call. |
| `guidance_scale` | Yes | `guidance_scale` | Default 4.0. |
| `generator` | Yes via seed | `seed` | Build generator from seed. |
| `output_type` | Mostly no | internal `output_type` | Always save WAV in production path. |
| `return_dict` | No | internal | Keep default `True`. |
| `callback_on_step_end` | No | future progress hook | Not needed until step-level progress is added. |
| `callback_on_step_end_tensor_inputs` | No | internal | Keep default. |

### SynthaEngine Runtime Options

| Option | Values | Default | Notes |
| --- | --- | --- | --- |
| `model` | registry name or null | default 1B Diffusers repo | Family must be `longcat-audiodit`. |
| `precision` | `fp16`, `bf16` | `fp16` | Official docs use FP16; BF16 is supported locally but should be tested. |
| `memory_preset` | `cuda`, `model_offload`, `sequential_offload` | `cuda` | Fall back to offload on OOM. |
| `quantization` | `none`, `bnb_8bit`, `bnb_4bit` | `none` | Experimental until measured. |
| `num_audios` | integer | `1` | Keep fixed to 1 initially. |
| `output_format` | `wav` | `wav` | Add other formats later only with encoders and tests. |
| `mono_to_stereo` | boolean | `false` | Optional convenience duplication, not true stereo generation. |
| `execution_mode` | `subprocess` | `subprocess` | Preserve memory cleanup policy. |
| `batch_id` | string or null | generated | Keep for workflow chaining/debugging. |

## 10. Recommended Implementation Order

1. Add audio history support and `AudiosWithBatchOutput`.
2. Add `backend/longcat_audiodit` subprocess runtime with fake-pipeline tests.
3. Add workflow schema, adapter, registry, and catalog support.
4. Add docs contract updates.
5. Add frontend audio page and audio gallery.
6. Run focused tests and backend compile.
7. Run one real local smoke test at 5 seconds, FP16, no quantization.
8. If local direct CUDA fails, test `model_offload`.
9. If memory is still tight, test `sequential_offload`.
10. Only after that, test `bnb_8bit` and `bnb_4bit`.

## 11. API Compatibility Notes

- This should be additive only.
- Do not rename existing task identifiers.
- Do not change existing image/video output shapes.
- Add `media_type="audio"` to history as a new value; existing consumers that
  check only image/video should continue to ignore unknown media types.
- Return audio paths under a new `audios` key, not `images` or `videos`.
- Keep `kind: "workflow"` as the only generation entrypoint.
- Store generated WAV files under `outputs/batch_<batch_id>/...` to match the
  existing batch organization.

## 12. Final Recommendation

Implement LongCat-AudioDiT locally first as a small, carefully bounded
text-to-audio feature:

```text
longcat-audiodit.text2audio
```

Use:

```text
ruixiangma/LongCat-AudioDiT-1B-Diffusers
fp16
5 second default duration
16 default steps
guidance_scale 4.0
one WAV per job
subprocess execution
```

Treat quantization, BF16, longer clips, cloud providers, and the 3.5B reference
model as follow-up capability tracks after the first WAV-generating workflow is
stable.
