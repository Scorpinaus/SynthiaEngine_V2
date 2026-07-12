# Latte Diffusers Implementation Guide

Last checked: 2026-06-17 against the Hugging Face Diffusers Latte API page,
the linked Diffusers source, and the `maxin-cn/Latte-1` model card.

Latte is a latent diffusion transformer for video generation. Diffusers exposes
the text-to-video integration through `LattePipeline`, with the denoising model
implemented by `LatteTransformer3DModel` and the output returned as
`LattePipelineOutput.frames`.

Official entry points:

- Pipeline docs: <https://huggingface.co/docs/diffusers/api/pipelines/latte>
- Pipeline docs source: <https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/latte.md>
- Pipeline source: <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/latte/pipeline_latte.py>
- Transformer docs: <https://huggingface.co/docs/diffusers/api/models/latte_transformer3d>
- Transformer source: <https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/latte_transformer_3d.py>
- Primary Diffusers checkpoint: <https://huggingface.co/maxin-cn/Latte-1>
- Latte paper: <https://huggingface.co/papers/2401.03048>
- Original project: <https://github.com/Vchitect/Latte>

## 1. Executive Summary

Use `LattePipeline` for prompt-only text-to-video generation with the
Diffusers-format `maxin-cn/Latte-1` checkpoint. The official docs show 16-frame,
512x512 generation in float16 on CUDA and export the returned frame list with
`export_to_gif`.

Practical integration answer:

- Load `maxin-cn/Latte-1` with `LattePipeline.from_pretrained(...)`.
- Prefer `torch.float16` on CUDA and batch size 1 unless the host has very large
  VRAM.
- Start with the documented defaults: `video_length=16`, `height=512`,
  `width=512`, `num_inference_steps=50`, `guidance_scale=7.5`,
  `enable_temporal_attentions=True`, and `decode_chunk_size=14`.
- Keep `negative_prompt=""` unless a product specifically validates a different
  negative prompt strategy. The Diffusers docs call out the empty string as the
  expected negative prompt for Latte.
- Output is `output.frames`. With the default `output_type="pil"`,
  `output.frames[0]` is the first generated video as a list of PIL frames.
- For lower memory use, call `enable_model_cpu_offload()` and/or lower
  `decode_chunk_size`. For lower latency on a fixed CUDA target, compile the
  transformer and VAE decode path as shown in the official docs.

## 2. What Latte Is

Latte models videos in latent space with transformer blocks. In the Diffusers
implementation, the model receives latent video tensors shaped like
`(batch, channels, frames, latent_height, latent_width)`, denoises them with a
spatial transformer path and an optional temporal transformer path, and decodes
the final latents back into video frames through an `AutoencoderKL`.

The official Latte paper covers broader video-generation research and benchmarks
on FaceForensics, SkyTimelapse, UCF101, and Taichi-HD. The Diffusers pipeline
page is narrower: it documents the text-to-video version and the
`maxin-cn/Latte-1` Diffusers checkpoint.

## 3. Pipeline and Components

`LattePipeline` has this component layout:

| Component | Diffusers / Transformers class | Role |
| --- | --- | --- |
| `tokenizer` | `T5Tokenizer` | Tokenizes the prompt and negative prompt. |
| `text_encoder` | `T5EncoderModel` | Frozen T5 text encoder. Diffusers notes Latte uses T5, specifically the `t5-v1_1-xxl` variant. |
| `transformer` | `LatteTransformer3DModel` | Text-conditioned 3D transformer that denoises video latents. |
| `vae` | `AutoencoderKL` | Decodes denoised video latents to frames. The source also supports latent output without decoding. |
| `scheduler` | `KarrasDiffusionSchedulers` compatible, checkpoint uses `DDIMScheduler` | Drives the denoising timesteps. |
| `video_processor` | `VideoProcessor` | Converts decoded tensors to PIL, NumPy, or PyTorch frame batches. |

The `maxin-cn/Latte-1` `model_index.json` declares `LattePipeline`,
`DDIMScheduler`, `T5EncoderModel`, `T5Tokenizer`, `LatteTransformer3DModel`, and
`AutoencoderKL`, so it can be loaded directly with `LattePipeline` or the generic
`DiffusionPipeline`.

The pipeline source defines:

- `_optional_components = ["tokenizer", "text_encoder"]`, which mainly matters
  if a caller provides precomputed prompt embeddings.
- `model_cpu_offload_seq = "text_encoder->transformer->vae"`, which is the order
  used by Diffusers model CPU offload.
- `_callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]`.

## 4. Checkpoints and Model IDs

The primary documented Diffusers checkpoint is:

| Model ID | Use | Notes |
| --- | --- | --- |
| `maxin-cn/Latte-1` | Text-to-video with `LattePipeline` | Apache-2.0 Hub repo, Diffusers and Safetensors tags, model card says it contains text-to-video pretrained weights. |

The official Diffusers docs also link to the contributor namespace
`hf.co/maxin-cn` for original weights. The older `maxin-cn/Latte` Hub repo
contains original research checkpoints such as benchmark-specific `.pt` files,
but it is not the main Diffusers-format checkpoint for `LattePipeline`. For a
new Diffusers integration, treat `maxin-cn/Latte-1` as the default model ID.

The `maxin-cn/Latte-1` repo is large. Its file tree includes a multi-gigabyte
transformer safetensors file, T5 text encoder/tokenizer files, VAE files, a
scheduler folder, and an additional original `.pt` file. Use normal Hub caching
and plan for slow first downloads.

## 5. Installation and Runtime Setup

Install a recent Diffusers stack with PyTorch, Transformers, Accelerate, and a
video writer package:

```powershell
.venv\Scripts\python.exe -m pip install -U diffusers transformers accelerate safetensors imageio imageio-ffmpeg
```

Optional packages:

- `beautifulsoup4` and `ftfy` enable the full `clean_caption=True` prompt
  cleanup path. Without them, Diffusers warns and falls back to raw prompt text.
- `bitsandbytes` enables the official 8-bit quantization example.

Use CUDA when possible. The official examples use `torch.float16`; the model card
snippet also shows `torch.bfloat16` through the generic `DiffusionPipeline`, but
the pipeline docs consistently demonstrate float16 for `LattePipeline`.

## 6. Minimal Text-to-Video Example

```python
import torch
from diffusers import LattePipeline
from diffusers.utils import export_to_gif, export_to_video

model_id = "maxin-cn/Latte-1"

pipe = LattePipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
)

# Good default for local apps that need to reduce steady GPU residency.
pipe.enable_model_cpu_offload()

generator = torch.Generator(device="cuda").manual_seed(1234)

output = pipe(
    prompt="A small cactus with a happy face in the Sahara desert, cinematic light",
    negative_prompt="",
    video_length=16,
    height=512,
    width=512,
    num_inference_steps=50,
    guidance_scale=7.5,
    generator=generator,
)

frames = output.frames[0]
export_to_gif(frames, "latte.gif")
export_to_video(frames, "latte.mp4", fps=8)
```

The docs describe the default 16 frames as an 8 fps clip, so exporting at
`fps=8` preserves the documented timing. Increase the export fps only if the UI
intentionally wants a shorter, faster playback.

## 7. Direct CUDA and Compile Example

The official docs recommend `torch.compile` to reduce inference latency. The
documented flow is:

```python
import torch
from diffusers import LattePipeline

pipe = LattePipeline.from_pretrained(
    "maxin-cn/Latte-1",
    torch_dtype=torch.float16,
).to("cuda")

pipe.transformer.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)

pipe.transformer = torch.compile(pipe.transformer)
pipe.vae.decode = torch.compile(pipe.vae.decode)

video = pipe(
    prompt="A dog wearing sunglasses floating in space, surreal, nebulae in background",
).frames[0]
```

Compile notes:

- Compile has a warmup cost. It is most useful for a process that will serve
  multiple Latte jobs with the same shapes.
- Shape changes can trigger new compilation. Keep `height`, `width`, and
  `video_length` constrained in production if compile latency matters.
- Channels-last is applied to the transformer and VAE before compilation in the
  official example.
- The official benchmark cited by Diffusers reports a modest speedup on an 80GB
  A100: about 16.246 seconds without compile and 14.573 seconds with compile.
  Treat that as a reference point, not a local SLA.

## 8. Key `LattePipeline.__call__` Parameters

Signature documented by Diffusers:

```python
pipe(
    prompt=None,
    negative_prompt="",
    num_inference_steps=50,
    timesteps=None,
    guidance_scale=7.5,
    num_images_per_prompt=1,
    video_length=16,
    height=512,
    width=512,
    eta=0.0,
    generator=None,
    latents=None,
    prompt_embeds=None,
    negative_prompt_embeds=None,
    output_type="pil",
    return_dict=True,
    callback_on_step_end=None,
    callback_on_step_end_tensor_inputs=["latents"],
    clean_caption=True,
    mask_feature=True,
    enable_temporal_attentions=True,
    decode_chunk_size=14,
)
```

| Parameter | Implementation guidance |
| --- | --- |
| `prompt` | String or list of strings. Required unless `prompt_embeds` is supplied. The source tokenizes to a maximum length of 120 tokens and warns if text is truncated. |
| `negative_prompt` | Ignored when `guidance_scale <= 1`. For Latte, Diffusers specifically says this should be the empty string. |
| `num_inference_steps` | Denoising step count. More steps usually improve quality at the cost of latency. The source default is 50. Some docs text still says defaults of 100/7.0 in parameter prose, but the actual signature is the source of truth. |
| `timesteps` | Optional custom descending timestep list. If supplied, it overrides `num_inference_steps` through the scheduler's `set_timesteps`. |
| `guidance_scale` | Classifier-free guidance strength. Guidance is enabled when `guidance_scale > 1`; `guidance_scale=1` disables CFG. Higher values track the prompt more strongly but can reduce video quality. |
| `num_images_per_prompt` | Number of videos generated per prompt. Each extra video multiplies memory and runtime pressure. |
| `video_length` | Number of frames to generate. Default is 16, documented as 16 frames at 8 fps. This also controls latent tensor shape and final frame reshaping. |
| `height`, `width` | Pixel dimensions. The source validates both are divisible by 8. Defaults are 512x512 in the call signature. |
| `eta` | DDIM eta value. Only used by schedulers whose `step` method accepts `eta`; ignored by other schedulers. |
| `generator` | A `torch.Generator` or list of generators for deterministic noise sampling. If a list is supplied, its length must match the effective batch size. |
| `latents` | Pre-generated noisy video latents. Useful for rerunning prompt variations against the same initial noise. Expected shape follows `(batch, channels, frames, height / vae_scale_factor, width / vae_scale_factor)`. |
| `prompt_embeds`, `negative_prompt_embeds` | Precomputed text embeddings. Use these for prompt weighting or to avoid loading the text encoder in specialized flows. If both positive and negative embeds are supplied, shapes must match. |
| `output_type` | Docs list `"pil"` and NumPy output. Source also accepts `"latent"` to return latents directly, and maps deprecated `"latents"` to `"latent"`. `VideoProcessor` supports `"pil"`, `"np"`, and `"pt"` after decode. |
| `return_dict` | If `True`, returns `LattePipelineOutput`. If `False`, returns a tuple whose first element is the frames or latents payload. |
| `callback_on_step_end` | Called at each denoising step. It receives `latents` by default and may also receive `prompt_embeds` and `negative_prompt_embeds` if requested. |
| `clean_caption` | When `True`, applies PixArt/DeepFloyd-style cleanup before encoding. Requires `beautifulsoup4` and `ftfy`; otherwise Diffusers disables cleanup with warnings. |
| `mask_feature` | Masks text embeddings using the tokenizer attention mask. Keep enabled unless you are deliberately testing raw embedding behavior. |
| `enable_temporal_attentions` | Enables temporal transformer blocks inside `LatteTransformer3DModel`. This is on by default and should stay on for video. |
| `decode_chunk_size` | Number of frames decoded by the VAE at a time. Larger chunks can improve temporal consistency but use more memory. Lower it to reduce VAE decode OOM risk. If `None`, the source decodes all `video_length` frames at once. |

## 9. `video_length` and Temporal Attention

`video_length` is the number of generated frames, not seconds. It is used in
three important places:

1. Latent allocation: `prepare_latents(...)` builds a tensor shaped like
   `(batch, latent_channels, video_length, height / scale, width / scale)`.
2. Transformer processing: `LatteTransformer3DModel.forward(...)` receives the
   frame dimension as `num_frame`.
3. Decode reshaping: `decode_latents(...)` flattens frames for VAE decoding and
   then reshapes the decoded frames back to `(batch, channels, video_length,
   height, width)`.

`enable_temporal_attentions=True` lets each transformer layer run an additional
temporal block after the spatial block. The transformer source reshapes hidden
states from frame-major spatial tokens into token-major frame sequences, adds a
temporal positional embedding on the first temporal block when `num_frame > 1`,
runs temporal self-attention, and reshapes the result back.

Use cases:

- Keep `enable_temporal_attentions=True` for normal text-to-video generation.
- Set `video_length=1` only for image-like smoke tests or specialized
  text-to-image behavior. With one frame, there is no meaningful temporal motion
  to model.
- Disabling temporal attention can reduce work, but it weakens the main reason
  to use Latte as a video model. Treat it as an experimental/debug option.

The transformer config also has a `video_length` value, used when constructing
the model's temporal positional embedding. The pipeline call can still pass a
runtime `video_length`; validate non-default lengths visually because temporal
position assumptions and memory use both change.

## 10. Output Functionality

Default output:

```python
result = pipe(prompt)
frames = result.frames[0]
```

With `return_dict=True`, `result` is `LattePipelineOutput` and `result.frames`
contains the batch of generated videos. With default `output_type="pil"`,
`result.frames[0]` is a list of PIL frames. With `output_type="np"`, the output
is a NumPy batch. With `output_type="pt"`, supported by `VideoProcessor`, the
output is a PyTorch tensor batch after decoding. With `output_type="latent"`,
the pipeline returns the final latent tensor without VAE decoding.

Common save paths:

```python
from diffusers.utils import export_to_gif, export_to_video

frames = pipe(prompt).frames[0]
export_to_gif(frames, "latte.gif")
export_to_video(frames, "latte.mp4", fps=8)
```

For web APIs, prefer saving a video or returning a manifest that points to the
generated file. Returning raw frame lists over JSON is usually too large.

For preview UIs, it is convenient to keep both:

- A GIF for quick inline previews.
- An MP4/WebM for realistic playback controls and lower file size.

## 11. Latents, Embeddings, and Callbacks

Latte supports the standard Diffusers advanced controls:

- `generator` makes random latent sampling reproducible.
- `latents` lets callers reuse or mutate the initial noise.
- `prompt_embeds` and `negative_prompt_embeds` bypass tokenization and text
  encoding.
- `callback_on_step_end` can inspect or replace tensors during denoising.

Callback example:

```python
def log_latent_shape(pipe, step, timestep, callback_kwargs):
    latents = callback_kwargs["latents"]
    print(step, timestep, tuple(latents.shape))
    return callback_kwargs

output = pipe(
    "A paper boat crossing a reflective studio floor",
    callback_on_step_end=log_latent_shape,
    callback_on_step_end_tensor_inputs=["latents"],
)
```

The callback contract is powerful but easy to misuse. In an application API,
avoid exposing arbitrary callbacks; expose safer controls such as seed,
step count, dimensions, frame count, and output format.

## 12. Quantization

The official docs show 8-bit bitsandbytes loading for both the T5 text encoder
and `LatteTransformer3DModel`:

```python
import torch
from diffusers import (
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
    LattePipeline,
    LatteTransformer3DModel,
)
from diffusers.utils import export_to_gif
from transformers import BitsAndBytesConfig, T5EncoderModel

model_id = "maxin-cn/Latte-1"

text_quant_config = BitsAndBytesConfig(load_in_8bit=True)
text_encoder_8bit = T5EncoderModel.from_pretrained(
    model_id,
    subfolder="text_encoder",
    quantization_config=text_quant_config,
    torch_dtype=torch.float16,
)

transformer_quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer_8bit = LatteTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=transformer_quant_config,
    torch_dtype=torch.float16,
)

pipe = LattePipeline.from_pretrained(
    model_id,
    text_encoder=text_encoder_8bit,
    transformer=transformer_8bit,
    torch_dtype=torch.float16,
    device_map="balanced",
)

frames = pipe("A small cactus with a happy face in the Sahara desert.").frames[0]
export_to_gif(frames, "latte.gif")
```

Quantization notes:

- It reduces model memory, but quality and speed can vary by GPU and backend.
- The docs use the Transformers bitsandbytes config for the T5 text encoder and
  the Diffusers bitsandbytes config for the Latte transformer.
- Validate video quality after quantization. Video artifacts can be more visible
  than single-image artifacts because they flicker across frames.

## 13. Performance and Memory Notes

Recommended first-pass production defaults:

| Setting | Suggested default | Reason |
| --- | --- | --- |
| dtype | `torch.float16` on CUDA | Matches official `LattePipeline` docs. |
| batch size | 1 | Video latents and T5 XXL embeddings are memory-heavy. |
| `video_length` | 16 | Documented default and known checkpoint behavior. |
| size | 512x512 | Documented default. |
| `decode_chunk_size` | 14 or lower under memory pressure | Trades temporal decode consistency for lower memory. |
| offload | `enable_model_cpu_offload()` for shared local servers | Reduces steady GPU residency. |
| compile | Optional for resident CUDA workers | Helps repeated same-shape inference, adds warmup cost. |

Memory pressure points:

- T5 text encoding is large.
- Transformer denoising scales with batch, frames, resolution, and guidance.
- Classifier-free guidance doubles transformer inputs internally when
  `guidance_scale > 1`.
- VAE decode can OOM near the end of generation; lower `decode_chunk_size`
  before lowering denoising quality.

For a local image-generation server, a safe execution model is a short-lived or
pooled worker that loads Latte on demand, produces an artifact, frees model
hooks, and returns file paths plus metadata. Keeping Latte resident is faster
for repeated jobs but increases the chance of starving other pipelines.

## 14. Gotchas

- `height` and `width` must be divisible by 8. The pipeline raises a
  `ValueError` otherwise.
- Prompt tokenization uses a maximum length of 120. Longer prompts are
  truncated after tokenization.
- The docs parameter prose has stale defaults for a few fields, such as
  `num_inference_steps` and `guidance_scale`. The call signature and source show
  the effective defaults: 50 and 7.5.
- `negative_prompt` should usually stay `""` for Latte, per the official docs.
- `output_type="latents"` is deprecated in the source and converted to
  `output_type="latent"`.
- The return tuple for `return_dict=False` contains the generated video payload
  as its first element, even though some generated docs prose says "images".
- `clean_caption=True` silently depends on optional packages. Install
  `beautifulsoup4` and `ftfy` for consistent prompt cleanup, or explicitly set
  `clean_caption=False`.
- `enable_temporal_attentions=False` is available, but it is not a normal
  quality setting for text-to-video output.
- Non-default `video_length` values should be tested visually. Longer clips
  raise memory use and may drift from the temporal assumptions in the checkpoint.
- The original `maxin-cn/Latte` Hub repo contains pickle `.pt` files. For normal
  Diffusers loading, prefer the safetensors-based `maxin-cn/Latte-1` checkpoint.

## 15. Implementation Checklist

For adding Latte to a workflow runner:

1. Register a text-to-video task that maps prompt, seed, steps, guidance,
   dimensions, frame count, fps, and output format to `LattePipeline.__call__`.
2. Validate `height % 8 == 0`, `width % 8 == 0`, `video_length >= 1`, and
   `num_images_per_prompt == 1` unless batching is explicitly supported.
3. Default `negative_prompt` to `""`.
4. Use `torch.Generator(device="cuda").manual_seed(seed)` when running on CUDA.
5. Save `output.frames[0]` with `export_to_video(..., fps=8)` or the user
   selected fps.
6. Record metadata: model ID, prompt, seed, dimensions, `video_length`, fps,
   steps, guidance scale, scheduler class, dtype, output type, and Diffusers
   version.
7. Apply `enable_model_cpu_offload()` or process-level cleanup for local server
   stability.
8. Keep compile and quantization behind explicit runtime options until they are
   benchmarked on the deployment GPU.

## 16. Source Links

- Diffusers Latte API page:
  <https://huggingface.co/docs/diffusers/api/pipelines/latte>
- Diffusers Latte docs source:
  <https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/latte.md>
- `LattePipeline` source:
  <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/latte/pipeline_latte.py>
- `LatteTransformer3DModel` API page:
  <https://huggingface.co/docs/diffusers/api/models/latte_transformer3d>
- `LatteTransformer3DModel` source:
  <https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/latte_transformer_3d.py>
- `maxin-cn/Latte-1` model card and files:
  <https://huggingface.co/maxin-cn/Latte-1>
- `maxin-cn/Latte-1` model index:
  <https://huggingface.co/maxin-cn/Latte-1/raw/main/model_index.json>
- Original Latte project:
  <https://github.com/Vchitect/Latte>
- Latte paper:
  <https://huggingface.co/papers/2401.03048>
