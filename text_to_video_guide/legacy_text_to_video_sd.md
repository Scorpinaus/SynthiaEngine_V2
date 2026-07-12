# Legacy ModelScope Text-to-Video SD Implementation Guide

Last checked: 2026-06-18 against the Hugging Face Diffusers `main`,
`v0.36.0`, and `v0.33.1` Text-to-video API pages, the linked Diffusers source,
and the Hugging Face model cards for the documented Zeroscope checkpoints.

Research target:
https://huggingface.co/docs/diffusers/main/en/api/pipelines/text_to_video

Primary Diffusers classes on the page:

- `TextToVideoSDPipeline`
- `VideoToVideoSDPipeline`
- `TextToVideoSDPipelineOutput`

Important status: this is a legacy/deprecated pipeline family. Current
Diffusers docs say it can still be used, but it is no longer tested and new
changes are not accepted. The current source wraps both pipeline classes with
`DeprecatedPipelineMixin` and marks `_last_supported_version = "0.33.1"`.
Treat `diffusers==0.33.1` as the safest compatibility target for production
or regression testing. Newer docs may require installing Diffusers from source,
and newer releases may keep only compatibility shims.

## 1. What This Pipeline Family Is

The ModelScope Text-to-Video SD pipeline family is an older Stable
Diffusion-style text-to-video implementation. It evolved from text-to-image
Stable Diffusion by replacing the 2D denoiser with a 3D conditional UNet that
denoises a latent video tensor over frame, height, and width dimensions.

The official Diffusers page summarizes ModelScopeT2V as a model with three
major parts: a VQGAN/VAE-style image autoencoder, a text encoder, and a
spatio-temporal denoising UNet. The docs also state that the 1.7B-parameter
model dedicates roughly 0.5B parameters to temporal capabilities.

For new work, prefer modern video pipelines such as CogVideoX, HunyuanVideo,
LTX-Video, Mochi, Wan, or Stable Video Diffusion where they fit. Use this guide
when you specifically need compatibility with:

- `damo-vilab/text-to-video-ms-1.7b`
- `cerspense/zeroscope_v2_576w`
- `cerspense/zeroscope_v2_XL`
- existing ModelScope/Zeroscope workflows, seeds, dimensions, or outputs
- old applications that already call `TextToVideoSDPipeline` or
  `VideoToVideoSDPipeline`

## 2. Pipeline Selection

| Class | Main task | Typical checkpoint | Best implementation use |
| --- | --- | --- | --- |
| `TextToVideoSDPipeline` | Prompt to video | `damo-vilab/text-to-video-ms-1.7b`, `cerspense/zeroscope_v2_576w` | Generate a short latent video directly from text. |
| `VideoToVideoSDPipeline` | Prompt-guided video-to-video | `cerspense/zeroscope_v2_XL` | Upscale or restyle an existing frame sequence by adding noise and denoising with the same or refined prompt. |
| `TextToVideoSDPipelineOutput` | Shared output object | Returned by both classes | Access generated frames through `.frames`. |

The Zeroscope workflow usually runs in two stages:

1. Generate a lower-resolution 16:9 video with `cerspense/zeroscope_v2_576w`
   through `TextToVideoSDPipeline`.
2. Resize those frames to `1024x576` and refine them with
   `cerspense/zeroscope_v2_XL` through `VideoToVideoSDPipeline`.

## 3. Version And Installation Strategy

Because this family is deprecated, pinning matters more than usual.

Recommended legacy baseline:

```powershell
.venv\Scripts\python.exe -m pip install "diffusers==0.33.1" transformers accelerate torch imageio imageio-ffmpeg pillow
```

When following the current `main` docs exactly, install from source:

```powershell
.venv\Scripts\python.exe -m pip install -U git+https://github.com/huggingface/diffusers
.venv\Scripts\python.exe -m pip install -U transformers accelerate torch imageio imageio-ffmpeg pillow
```

Notes:

- Use `torch_dtype=torch.float16` on CUDA for the documented examples.
- The `variant="fp16"` argument is documented for
  `damo-vilab/text-to-video-ms-1.7b`.
- `enable_model_cpu_offload()` requires `accelerate`.
- `export_to_video` requires video-writing dependencies such as `imageio` and
  `imageio-ffmpeg`.
- Current stable docs may show the page but still warn that unsupported
  pipeline issues should be solved by reinstalling the last supported
  Diffusers version.

## 4. Checkpoints

| Checkpoint | Pipeline | Native or recommended size | Notes |
| --- | --- | --- | --- |
| `damo-vilab/text-to-video-ms-1.7b` | `TextToVideoSDPipeline` | Default 16 frames, about 2 seconds at 8 fps in the docs | Original ModelScope text-to-video checkpoint. Use `variant="fp16"` when loading the documented fp16 weights. |
| `cerspense/zeroscope_v2_576w` | `TextToVideoSDPipeline` | `576x320`, often 24 frames | Watermark-free ModelScope-based model for lower-resolution 16:9 composition. The model card says it was trained at 24 frames and `576x320`. |
| `cerspense/zeroscope_v2_XL` | `VideoToVideoSDPipeline` | `1024x576`, often 24-36 frames | Watermark-free ModelScope-based video-to-video model designed to upscale/refine clips from `zeroscope_v2_576w`. |

The Diffusers docs describe Zeroscope as trained on specific sizes such as
`576x320` and `1024x576`. The Zeroscope model cards also warn that lower
resolutions or too few frames can produce suboptimal output.

## 5. Shared Components

Both pipeline constructors take the same component set:

- `vae`: `AutoencoderKL` used to encode and decode frames between pixel and
  latent space. The video latents are reshaped frame-by-frame for VAE decode.
- `text_encoder`: `CLIPTextModel`, usually based on
  `openai/clip-vit-large-patch14`.
- `tokenizer`: `CLIPTokenizer`.
- `unet`: `UNet3DConditionModel`, the spatio-temporal denoiser.
- `scheduler`: a Diffusers scheduler compatible with the denoising loop.
  The docs mention DDIM, LMS, and PNDM; examples also swap in
  `DPMSolverMultistepScheduler`.

Both classes inherit general `DiffusionPipeline` behavior and Stable
Diffusion-style loading helpers:

- `from_pretrained(...)`
- `.to("cuda")` or `enable_model_cpu_offload()`
- `load_textual_inversion(...)`
- `load_lora_weights(...)`
- `save_lora_weights(...)`

Implementation detail worth keeping in mind: this is not the same architecture
as AnimateDiff, Stable Video Diffusion, or transformer-based modern video
pipelines. The UNet is `UNet3DConditionModel`, and latents carry a frame axis.

## 6. `TextToVideoSDPipeline`

Source:
https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth.py

Use this pipeline for prompt-only text-to-video generation.

### Minimal ModelScope Example

```python
import torch
from diffusers import TextToVideoSDPipeline
from diffusers.utils import export_to_video

pipe = TextToVideoSDPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe = pipe.to("cuda")

prompt = "Spiderman is surfing"
output = pipe(prompt)
frames = output.frames[0]
video_path = export_to_video(frames, output_video_path="modelscope_spiderman.mp4", fps=8)
```

### Memory-Friendly ModelScope Example

The official docs show a 64-frame example with CPU offloading and VAE slicing,
noting about 7 GB of GPU memory with PyTorch 2.0 and fp16.

```python
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

frames = pipe(
    "Darth Vader surfing a wave",
    num_frames=64,
).frames[0]
export_to_video(frames, output_video_path="modelscope_64_frames.mp4", fps=8)
```

### Faster Scheduler Example

The docs use `DPMSolverMultistepScheduler` the same way it is commonly used
for Stable Diffusion.

```python
import torch
from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

frames = pipe(
    "Spiderman is surfing",
    num_inference_steps=25,
).frames[0]
export_to_video(frames, output_video_path="modelscope_dpm.mp4", fps=8)
```

### Key Parameters

| Parameter | Default | Implementation notes |
| --- | --- | --- |
| `prompt` | `None` | String or list of strings. Required unless passing `prompt_embeds`. |
| `height` / `width` | `unet.config.sample_size * vae_scale_factor` | Must be divisible by 8. For Zeroscope 576w, use `height=320`, `width=576`. |
| `num_frames` | `16` | The docs describe 16 frames as about 2 seconds at 8 fps. More frames increase memory roughly linearly. |
| `num_inference_steps` | `50` | More steps usually improve quality and temporal stability but slow generation. DPM examples often use 25-40. |
| `guidance_scale` | Signature shows `9.0`; parameter docs describe CFG generally | CFG is active when `guidance_scale > 1`. Higher values follow prompt more strongly but can degrade image quality. |
| `negative_prompt` | `None` | Ignored when CFG is disabled. Do not pass with `negative_prompt_embeds`. |
| `eta` | `0.0` | Only affects DDIM schedulers. Ignored by most other schedulers. |
| `generator` | `None` | Pass `torch.Generator(...).manual_seed(...)` for reproducible noise. |
| `latents` | `None` | Optional noisy latent tensor. Shape is `(batch, channels, frames, latent_height, latent_width)` in source; docs describe the video latent shape conceptually. |
| `prompt_embeds` / `negative_prompt_embeds` | `None` | Use for precomputed CLIP embeddings or custom prompt weighting. Shapes must match when both are passed. |
| `output_type` | `"np"` | Use `"np"` for NumPy arrays, `"pt"`/`"latent"` when supported by source paths, or pipeline-specific postprocess choices. |
| `return_dict` | `True` | If false, returns a tuple whose first element is the frame output. |
| `callback` / `callback_steps` | `None` / `1` | Legacy callback receives `step`, `timestep`, and `latents`. |
| `cross_attention_kwargs` | `None` | Passed to attention processors; can carry LoRA scale in Stable Diffusion-style pipelines. |
| `clip_skip` | `None` | Uses an earlier CLIP hidden layer when set. |

## 7. `VideoToVideoSDPipeline`

Source:
https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth_img2img.py

Use this pipeline when you already have a video frame sequence and want a
prompt-guided denoise/refinement pass. In practice, it is most often used for
Zeroscope upscaling: resize frames from the 576w model, then denoise with the
XL model.

### Zeroscope Two-Stage Upscaling Example

```python
import torch
from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image

prompt = "Darth Vader surfing a wave"

pipe = DiffusionPipeline.from_pretrained(
    "cerspense/zeroscope_v2_576w",
    torch_dtype=torch.float16,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
pipe.enable_vae_slicing()

low_res_frames = pipe(
    prompt,
    height=320,
    width=576,
    num_frames=24,
    num_inference_steps=40,
).frames[0]
export_to_video(low_res_frames, output_video_path="zeroscope_576.mp4", fps=8)

pipe = DiffusionPipeline.from_pretrained(
    "cerspense/zeroscope_v2_XL",
    torch_dtype=torch.float16,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
pipe.enable_vae_slicing()

video = [Image.fromarray(frame).resize((1024, 576)) for frame in low_res_frames]

upscaled_frames = pipe(
    prompt,
    video=video,
    strength=0.6,
    num_inference_steps=40,
).frames[0]
export_to_video(upscaled_frames, output_video_path="zeroscope_1024.mp4", fps=8)
```

Some older examples on the `zeroscope_v2_XL` model card use
`revision="refs/pr/15"`. The current Diffusers docs example no longer includes
that revision. Keep it only if a pinned legacy environment or cached model
layout requires it.

### Key Parameters

| Parameter | Default | Implementation notes |
| --- | --- | --- |
| `prompt` | `None` | Required unless using `prompt_embeds`. Use the same prompt as the low-res generation for upscaling consistency. |
| `video` | `None` | List of NumPy frames/PIL-compatible frames or a tensor. Source also accepts latent-like tensors with 4 channels and skips VAE encoding. |
| `strength` | Signature shows `0.6`; parameter docs describe the 0-1 range | Controls how much noise is added to the input. `0` keeps the source nearly intact; `1` effectively ignores it. Zeroscope docs/model cards recommend roughly `0.6` in Diffusers and `0.66-0.85` in some external vid2vid workflows. |
| `num_inference_steps` | `50` | The effective number of denoising steps depends on `strength`. Higher strength starts earlier in the schedule. |
| `guidance_scale` | Signature shows `15.0`; parameter docs describe CFG generally | Video-to-video examples often use stronger guidance than text-to-video. Lower it if frames become overcooked or unstable. |
| `negative_prompt` | `None` | Same rules as text-to-video. |
| `eta` | `0.0` | DDIM-only. |
| `generator` | `None` | Use for reproducibility; list length must match batch size if passing a generator list. |
| `latents` | `None` | Optional precomputed latents/noise inputs. |
| `prompt_embeds` / `negative_prompt_embeds` | `None` | Same CLIP embedding behavior as the text-to-video pipeline. |
| `output_type` | `"np"` | Output is usually converted to frames for `export_to_video`. |
| `return_dict` | `True` | If false, first tuple element is the frame output. |
| `callback` / `callback_steps` | `None` / `1` | Legacy step callback. |
| `cross_attention_kwargs` | `None` | Passed through to attention processors. |
| `clip_skip` | `None` | Optional CLIP layer skip. |

## 8. Outputs And Frame Handling

Both pipelines return `TextToVideoSDPipelineOutput` by default. Its `frames`
field may be a Torch tensor, a NumPy array, or a nested list of PIL images,
depending on `output_type` and the installed Diffusers version. The official
output docs describe the tensor/array shape as:

```text
(batch_size, num_frames, channels, height, width)
```

Most user-facing examples access:

```python
frames = pipe(...).frames[0]
```

Then export:

```python
from diffusers.utils import export_to_video

export_to_video(frames, output_video_path="output.mp4", fps=8)
```

Gotcha: some older model-card snippets omit `[0]` and pass `.frames` directly.
The Diffusers API docs consistently show `.frames[0]` for a single generated
video.

## 9. Memory And Performance Options

Recommended options for local servers:

- Load fp16 weights on CUDA with `torch_dtype=torch.float16`.
- Use `enable_model_cpu_offload()` instead of `.to("cuda")` when VRAM is tight.
- Use `enable_vae_slicing()` or `pipe.vae.enable_slicing()` to lower VAE decode
  memory.
- For Zeroscope, call
  `pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)` to reduce UNet
  feed-forward memory. The Zeroscope XL model card notes this can slow
  generation significantly, so disable it when VRAM allows.
- Keep `height`, `width`, and `num_frames` close to checkpoint training sizes.
- Use `DPMSolverMultistepScheduler` for fewer denoising steps when quality is
  acceptable.
- Reuse loaded components when possible if a service runs both low-res and XL
  stages, but test carefully because the low-res and XL checkpoints are
  different repositories.

Memory scaling rules of thumb:

- More frames cost more UNet memory and more VAE decode work.
- Larger spatial dimensions grow latent area and attention/feed-forward cost.
- Video-to-video costs extra memory because the input video must be
  preprocessed and encoded before denoising.
- CPU offload reduces peak VRAM at the cost of PCIe/device transfer latency.

## 10. Gotchas

- Deprecated means unsupported. If a newer Diffusers version breaks this
  pipeline, pin or reinstall `diffusers==0.33.1`.
- Current docs for `main` require source install. Stable version pages may be
  available, but the pipeline is still legacy.
- `height` and `width` must be divisible by 8.
- CLIP prompts can be truncated at the tokenizer max length. The source warns
  when text is truncated.
- Do not pass both `prompt` and `prompt_embeds`, or both `negative_prompt` and
  `negative_prompt_embeds`.
- If passing a list of generators, its length must match the effective batch
  size.
- Zeroscope is size-sensitive. Use `576x320` for `zeroscope_v2_576w` and
  `1024x576` for `zeroscope_v2_XL` unless you have tested alternatives.
- Zeroscope model cards warn that too few frames or lower resolutions can be
  suboptimal. Prefer at least 24 frames for that family.
- `strength=1.0` in video-to-video mostly discards the input video; use lower
  values for upscaling/refinement.
- `eta` only matters for DDIM. It is ignored by schedulers that do not accept
  an `eta` argument.
- Use the same prompt for low-res generation and XL video-to-video upscaling
  unless the desired behavior is deliberate restyling.
- The pipeline is research/legacy code and does not include newer video-model
  conveniences such as transformer quantization recipes, temporal tiling, or
  modern callback APIs.

## 11. Implementation Checklist For SynthaEngine-Style Workflows

If exposing this family behind a workflow task, keep the public surface small
and legacy-labeled:

- `model_id`: one of the known legacy checkpoints.
- `mode`: `text_to_video` or `video_to_video`.
- `prompt` and optional `negative_prompt`.
- `height`, `width`, `num_frames`, `fps`.
- `num_inference_steps`, `guidance_scale`, `seed`.
- `strength` only for `video_to_video`.
- `memory`: booleans for CPU offload, VAE slicing, and UNet forward chunking.
- `scheduler`: default checkpoint scheduler or `dpm_solver_multistep`.
- `output_path` and `output_type`, with MP4 export as the normal final
  artifact.

Validation should include:

- A 16-frame ModelScope smoke test if the checkpoint is available.
- A 24-frame `576x320` Zeroscope low-res smoke test.
- A short XL video-to-video pass with resized frames and `strength=0.6`.
- Shape checks for `.frames[0]`.
- A fallback error message that tells users to pin `diffusers==0.33.1` when a
  deprecated-pipeline import or runtime failure occurs.

## 12. Official Source Links

- Current Diffusers Text-to-video docs:
  https://huggingface.co/docs/diffusers/main/en/api/pipelines/text_to_video
- Diffusers `v0.36.0` docs source with the deprecation warning:
  https://raw.githubusercontent.com/huggingface/diffusers/v0.36.0/docs/source/en/api/pipelines/text_to_video.md
- Diffusers `v0.33.1` Text-to-video API docs:
  https://huggingface.co/docs/diffusers/v0.33.1/api/pipelines/text_to_video
- Diffusers `v0.33.1` docs source:
  https://raw.githubusercontent.com/huggingface/diffusers/v0.33.1/docs/source/en/api/pipelines/text_to_video.md
- `TextToVideoSDPipeline` source:
  https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth.py
- `VideoToVideoSDPipeline` source:
  https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth_img2img.py
- `TextToVideoSDPipelineOutput` source:
  https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/text_to_video_synthesis/pipeline_output.py
- ModelScope checkpoint org:
  https://huggingface.co/damo-vilab
- `damo-vilab/text-to-video-ms-1.7b` checkpoint:
  https://huggingface.co/damo-vilab/text-to-video-ms-1.7b
- Zeroscope checkpoint org:
  https://huggingface.co/cerspense
- `cerspense/zeroscope_v2_576w` model card:
  https://huggingface.co/cerspense/zeroscope_v2_576w
- `cerspense/zeroscope_v2_XL` model card:
  https://huggingface.co/cerspense/zeroscope_v2_XL
- ModelScope Text-to-Video paper:
  https://huggingface.co/papers/2308.06571
