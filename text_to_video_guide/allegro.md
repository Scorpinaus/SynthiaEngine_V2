# Allegro Diffusers Implementation Guide

This guide is based on the current Hugging Face Diffusers Allegro API page and
the linked Diffusers source as checked on 2026-06-17. The rendered docs page
currently links its source to Diffusers `v0.38.0`; model/checkpoint notes below
also reference the current `rhymes-ai/Allegro` model repository on Hugging Face.

Official sources:

- Diffusers Allegro API: https://huggingface.co/docs/diffusers/api/pipelines/allegro
- Diffusers Allegro pipeline source: https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/allegro/pipeline_allegro.py
- Diffusers Allegro output source: https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/allegro/pipeline_output.py
- Diffusers Allegro package exports: https://github.com/huggingface/diffusers/tree/v0.38.0/src/diffusers/pipelines/allegro
- Allegro checkpoint: https://huggingface.co/rhymes-ai/Allegro
- `AutoencoderKLAllegro` docs: https://huggingface.co/docs/diffusers/api/models/autoencoderkl_allegro
- `AllegroTransformer3DModel` docs: https://huggingface.co/docs/diffusers/api/models/allegro_transformer3d

## Pipeline Family

The Diffusers Allegro pipeline family contains one public pipeline class and
one pipeline output class in `v0.38.0`.

| Class | Role |
| --- | --- |
| `diffusers.AllegroPipeline` | Text-to-video generation from text prompts or precomputed T5 embeddings. |
| `diffusers.pipelines.allegro.pipeline_output.AllegroPipelineOutput` | Return container with a `frames` field for generated videos or latents. |

The `src/diffusers/pipelines/allegro` package contains `pipeline_allegro.py`,
`pipeline_output.py`, and `__init__.py`. In `v0.38.0`, `__init__.py` exports
only `AllegroPipeline` when Torch and Transformers are available. There is no
Diffusers Allegro image-to-video, text-image-to-video, inpainting, ControlNet,
or editing pipeline class in this family. RhymesAI's upstream repository
mentions Allegro-TI2V variants, but those are not exposed as separate Diffusers
pipeline classes in the official Allegro API page/source covered here.

## What Allegro Generates

Allegro is a text-to-video diffusion transformer pipeline. The official
checkpoint `rhymes-ai/Allegro` is tagged as text-to-video, Diffusers,
Safetensors, English, and `AllegroPipeline`, with Apache-2.0 licensing. The
model card describes the main checkpoint as generating 88 frames at
720 x 1280, roughly 6 seconds at 15 FPS. It lists a 175M parameter VideoVAE and
a 2.8B parameter VideoDiT model, with BF16 CPU-offloaded inference around
9.3 GB GPU memory and non-offloaded inference around 27.5 GB.

The Diffusers pipeline defaults are derived from component configs:

- `num_frames`: `transformer.config.sample_frames * vae_scale_factor_temporal`
  -> 22 * 4 = 88 for `rhymes-ai/Allegro`.
- `height`: `transformer.config.sample_height * vae_scale_factor_spatial`
  -> 90 * 8 = 720.
- `width`: `transformer.config.sample_width * vae_scale_factor_spatial`
  -> 160 * 8 = 1280.

## Required Components

`AllegroPipeline.__init__` requires every component below. The pipeline has no
optional component list in `v0.38.0`.

| Component | Class | Purpose |
| --- | --- | --- |
| `tokenizer` | `transformers.T5Tokenizer` | Tokenizes prompts for the T5 encoder. |
| `text_encoder` | `transformers.T5EncoderModel` | Frozen T5 encoder producing prompt embeddings. The docs note the PixArt-style T5 v1.1 XXL setup. |
| `vae` | `diffusers.AutoencoderKLAllegro` | 3D video VAE that decodes latent videos to frames and can encode videos to latents. |
| `transformer` | `diffusers.AllegroTransformer3DModel` | Text-conditioned 3D diffusion transformer that denoises video latents. |
| `scheduler` | `diffusers.KarrasDiffusionSchedulers` compatible scheduler | Drives the denoising timesteps. The current checkpoint config uses `EulerAncestralDiscreteScheduler`. |

The current `rhymes-ai/Allegro` `model_index.json` maps:

- `scheduler`: `diffusers.EulerAncestralDiscreteScheduler`
- `text_encoder`: `transformers.T5EncoderModel`
- `tokenizer`: `transformers.T5Tokenizer`
- `transformer`: `diffusers.AllegroTransformer3DModel`
- `vae`: `diffusers.AutoencoderKLAllegro`

The current checkpoint component configs are native Diffusers classes. Older
repository history included custom module names, so prefer the current `main`
revision or a known-good revision when reproducing results.

## Installation Notes

Use current Diffusers, Transformers, Accelerate, and the optional packages
needed for tokenization, caption cleaning, and video writing.

```bash
pip install -U diffusers transformers accelerate sentencepiece imageio imageio-ffmpeg
pip install -U beautifulsoup4 ftfy
```

For quantized examples, install a compatible `bitsandbytes` build for your
platform.

```bash
pip install -U bitsandbytes
```

## Basic Usage

The official docs load the Allegro VAE in `float32` and the rest of the
pipeline in `bfloat16`. This matches the model card guidance that the VAE is
best in FP32/TF32 while the DiT/T5 stack supports BF16.

```python
import torch
from diffusers import AllegroPipeline, AutoencoderKLAllegro
from diffusers.utils import export_to_video

model_id = "rhymes-ai/Allegro"

vae = AutoencoderKLAllegro.from_pretrained(
    model_id,
    subfolder="vae",
    torch_dtype=torch.float32,
)

pipe = AllegroPipeline.from_pretrained(
    model_id,
    vae=vae,
    torch_dtype=torch.bfloat16,
).to("cuda")

# Current source keeps pipe.enable_vae_tiling(), but deprecates it for v0.40.0.
# Prefer the VAE method directly.
pipe.vae.enable_tiling()

prompt = (
    "A seaside harbor with bright sunlight and sparkling seawater, with many "
    "boats in the water. From an aerial view, the boats vary in size and color, "
    "some moving and some stationary."
)

result = pipe(
    prompt,
    negative_prompt="low quality, blurry, watermark, text",
    guidance_scale=7.5,
    num_inference_steps=100,
    max_sequence_length=512,
)

video_frames = result.frames[0]
export_to_video(video_frames, "allegro_output.mp4", fps=15)
```

The generic "Use this model" snippet on the model card may show `.images[0]`.
For `AllegroPipeline`, use `.frames[0]`.

## `AllegroPipeline.__call__`

Current stable signature:

```python
pipe(
    prompt: str | list[str] = None,
    negative_prompt: str | list[str] = "",
    num_inference_steps: int = 100,
    timesteps: list[int] | None = None,
    guidance_scale: float = 7.5,
    num_frames: int | None = None,
    height: int | None = None,
    width: int | None = None,
    num_videos_per_prompt: int = 1,
    eta: float = 0.0,
    generator: torch.Generator | list[torch.Generator] | None = None,
    latents: torch.Tensor | None = None,
    prompt_embeds: torch.Tensor | None = None,
    prompt_attention_mask: torch.Tensor | None = None,
    negative_prompt_embeds: torch.Tensor | None = None,
    negative_prompt_attention_mask: torch.Tensor | None = None,
    output_type: str | None = "pil",
    return_dict: bool = True,
    callback_on_step_end=None,
    callback_on_step_end_tensor_inputs: list[str] = ["latents"],
    clean_caption: bool = True,
    max_sequence_length: int = 512,
)
```

Important parameters:

| Parameter | Use |
| --- | --- |
| `prompt` | Text prompt or list of prompts. Required unless `prompt_embeds` is provided. |
| `negative_prompt` | Negative text prompt or list. Used only when classifier-free guidance is active. |
| `num_inference_steps` | Denoising step count. More steps generally increase quality and runtime. Default is 100. |
| `timesteps` | Optional custom timestep list. Must be descending, and the active scheduler must support custom timesteps. Do not pass with a conflicting `num_inference_steps` schedule. |
| `guidance_scale` | Classifier-free guidance scale. Guidance is enabled when `guidance_scale > 1`. Higher values follow text more strongly but may reduce visual quality. |
| `num_frames` | Generated frame count. Defaults to 88 for the main checkpoint. Must be positive. |
| `height`, `width` | Output resolution. Defaults to 720 x 1280 for the main checkpoint. Both must be divisible by 8. |
| `num_videos_per_prompt` | Present in the signature, but current `v0.38.0` source resets it to `1` inside `__call__`. Loop manually if you need multiple samples per prompt. |
| `eta` | DDIM eta. Only used by schedulers whose `step()` accepts `eta`; ignored by Euler A and other schedulers that do not. |
| `generator` | `torch.Generator` or a list of generators for deterministic sampling. If a list is used, its length must match the effective batch size. |
| `latents` | Optional pre-sampled noisy latents. Use to reuse noise across prompt changes or to resume/customize sampling. |
| `prompt_embeds`, `prompt_attention_mask` | Precomputed positive T5 embeddings and mask. If embeddings are passed, the attention mask is required. |
| `negative_prompt_embeds`, `negative_prompt_attention_mask` | Precomputed negative embeddings and mask. Must match positive embedding and mask shapes when both are passed. |
| `output_type` | Rendered docs list PIL/NumPy output. The source also supports `"latent"` to return final latents without VAE decode. |
| `return_dict` | When `True`, returns `AllegroPipelineOutput`; when `False`, returns `(video,)`. |
| `callback_on_step_end` | Optional modern Diffusers callback called after each denoising step. |
| `callback_on_step_end_tensor_inputs` | Tensor names passed to the callback. Allowed values are `latents`, `prompt_embeds`, and `negative_prompt_embeds`. |
| `clean_caption` | Cleans captions before T5 encoding. Requires `beautifulsoup4` and `ftfy`; without them, the pipeline falls back to raw prompt text. |
| `max_sequence_length` | Maximum T5 token length. Default is 512. Longer prompts are truncated with a warning. |

The rendered docs prose still mentions older `callback` and `callback_steps`
fields, but the actual `v0.38.0` call signature uses
`callback_on_step_end` and `callback_on_step_end_tensor_inputs`.

## Internal Generation Flow

`AllegroPipeline.__call__` performs the following high-level steps:

1. Derive default `num_frames`, `height`, and `width` from the transformer and
   VAE configs.
2. Validate inputs:
   - `num_frames` must be positive.
   - `height` and `width` must be divisible by 8.
   - pass either `prompt` or `prompt_embeds`, not both.
   - pass matching attention masks when passing embeddings directly.
   - callback tensor input names must be from the pipeline's supported list.
3. Encode the prompt with T5, optionally with negative prompt embeddings for
   classifier-free guidance.
4. Build a timestep schedule with the scheduler.
5. Prepare video latents with shape:

```text
(
    batch_size,
    transformer.config.in_channels,
    compressed_frames,
    height // vae_scale_factor_spatial,
    width // vae_scale_factor_spatial,
)
```

For the main checkpoint, `transformer.config.in_channels` is 4 and
`vae_scale_factor_spatial` is 8. Temporal compression is 4. Even frame counts
use `ceil(num_frames / 4)` latent frames; odd counts use
`ceil((num_frames - 1) / 4) + 1`.

6. Prepare 3D rotary positional embeddings from the latent grid.
7. Run the denoising loop:
   - duplicate latents and embeddings for classifier-free guidance when
     `guidance_scale > 1`.
   - call `AllegroTransformer3DModel`.
   - apply guidance.
   - call `scheduler.step(...)`.
   - invoke `callback_on_step_end` if supplied.
8. Decode latents with `AutoencoderKLAllegro` unless `output_type == "latent"`.
9. Crop decoded video to the requested frame count and resolution.
10. Postprocess frames and call `maybe_free_model_hooks()` for offloaded models.

## Prompt Encoding

`encode_prompt` can be called directly when you want to cache text embeddings or
apply custom embedding logic.

```python
prompt_embeds, prompt_mask, negative_embeds, negative_mask = pipe.encode_prompt(
    prompt="a cinematic tracking shot of a glass greenhouse at sunrise",
    negative_prompt="low quality, watermark, text",
    do_classifier_free_guidance=True,
    num_videos_per_prompt=1,
    device=pipe.device,
    clean_caption=True,
    max_sequence_length=512,
)

video = pipe(
    prompt_embeds=prompt_embeds,
    prompt_attention_mask=prompt_mask,
    negative_prompt_embeds=negative_embeds,
    negative_prompt_attention_mask=negative_mask,
).frames[0]
```

When passing embeddings into `__call__`, always pass the corresponding attention
masks. Positive and negative embedding shapes must match if both are supplied.

## Output Structure

`AllegroPipelineOutput` is a dataclass with one field:

```python
frames: torch.Tensor | np.ndarray | list[list[PIL.Image.Image]]
```

For normal decoded output, `frames` is a batch-indexed collection of video
frames. The common case is:

```python
result = pipe(prompt)
first_video_frames = result.frames[0]
```

If `return_dict=False`, the pipeline returns a tuple whose first element is the
same video object:

```python
(video_batch,) = pipe(prompt, return_dict=False)
first_video_frames = video_batch[0]
```

If `output_type="latent"`, `frames` contains the final latent tensor rather than
decoded frames. This is useful for custom decode/postprocessing experiments, but
it is not directly viewable as a video.

## Memory and Performance Options

### VAE tiling and slicing

The API page lists `enable_vae_tiling`, `disable_vae_tiling`,
`enable_vae_slicing`, and `disable_vae_slicing` on the pipeline. In `v0.38.0`,
these wrappers still exist but emit deprecation warnings saying they will be
removed in Diffusers `0.40.0`. Prefer calling methods on the VAE directly:

```python
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
```

Tiling is the most important VAE memory saver for 720p video decode. Slicing can
also reduce memory pressure for larger batches.

### CPU offload

The pipeline source declares this offload sequence:

```text
text_encoder -> transformer -> vae
```

Use only one offload strategy at a time.

```python
pipe = AllegroPipeline.from_pretrained(
    "rhymes-ai/Allegro",
    vae=vae,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
```

For the lowest GPU memory footprint, the model card recommends sequential CPU
offload. It is slower because modules are moved more aggressively between CPU
and GPU.

```python
pipe = AllegroPipeline.from_pretrained(
    "rhymes-ai/Allegro",
    vae=vae,
    torch_dtype=torch.bfloat16,
)
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_tiling()
```

When using an offload helper, do not immediately call `pipe.to("cuda")` after
enabling offload, because that can undo the placement strategy.

### Device maps

Diffusers can place pipeline components with a `device_map`. The official
quantization example uses `device_map="balanced"`.

```python
pipe = AllegroPipeline.from_pretrained(
    "rhymes-ai/Allegro",
    vae=vae,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
)
```

### Quantization

The Allegro docs include a bitsandbytes example that quantizes both the T5 text
encoder and the Allegro transformer to 8-bit. The VAE is left unquantized.

```python
import torch
from diffusers import (
    AllegroPipeline,
    AllegroTransformer3DModel,
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
)
from diffusers.utils import export_to_video
from transformers import BitsAndBytesConfig, T5EncoderModel

model_id = "rhymes-ai/Allegro"

t5_quant_config = BitsAndBytesConfig(load_in_8bit=True)
text_encoder_8bit = T5EncoderModel.from_pretrained(
    model_id,
    subfolder="text_encoder",
    quantization_config=t5_quant_config,
    torch_dtype=torch.float16,
)

transformer_quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer_8bit = AllegroTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=transformer_quant_config,
    torch_dtype=torch.float16,
)

pipe = AllegroPipeline.from_pretrained(
    model_id,
    text_encoder=text_encoder_8bit,
    transformer=transformer_8bit,
    torch_dtype=torch.float16,
    device_map="balanced",
)
pipe.vae.enable_tiling()

prompt = "A sunlit harbor filmed from above, boats crossing sparkling blue water."
frames = pipe(prompt, guidance_scale=7.5, max_sequence_length=512).frames[0]
export_to_video(frames, "harbor.mp4", fps=15)
```

Quantization can reduce memory significantly, but video quality and speed can
vary by backend and GPU.

### Step count, frame count, and resolution

The main runtime drivers are:

- `num_inference_steps`: lower values are faster; 100 is the documented default.
- `height` and `width`: must be divisible by 8; memory scales with spatial area.
- `num_frames`: memory and runtime scale with the number of generated frames.
- `guidance_scale`: affects quality and prompt adherence, not just speed.
- VAE decode: often the part that benefits most from tiling at 720p.

For quick smoke tests, reduce steps, frame count, or resolution:

```python
frames = pipe(
    "a quiet mountain road at dawn",
    num_inference_steps=20,
    num_frames=24,
    height=360,
    width=640,
    guidance_scale=6.0,
).frames[0]
```

Lower-resolution or fewer-frame checkpoints from RhymesAI may exist upstream,
but this guide covers the official Diffusers API family and the
`rhymes-ai/Allegro` Diffusers checkpoint.

## Scheduler Notes

The pipeline accepts a `KarrasDiffusionSchedulers` compatible scheduler. The
current checkpoint config uses `EulerAncestralDiscreteScheduler` with:

- `beta_start`: 0.0001
- `beta_end`: 0.02
- `beta_schedule`: `linear`
- `num_train_timesteps`: 1000
- `prediction_type`: `epsilon`
- `timestep_spacing`: `linspace`

You can swap to another compatible scheduler from the existing scheduler config:

```python
from diffusers import DDIMScheduler

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
frames = pipe(
    "a misty forest path with shafts of light",
    num_inference_steps=80,
    eta=0.0,
).frames[0]
```

Only schedulers whose `set_timesteps` accepts custom timesteps can use the
`timesteps` parameter, and only schedulers whose `step` accepts `eta` use the
`eta` parameter.

## Callbacks and Interruption

`callback_on_step_end` receives `(pipeline, step_index, timestep, callback_kwargs)`.
It may return updated tensors. The pipeline reads back `latents`,
`prompt_embeds`, and `negative_prompt_embeds` from the callback output when
present.

```python
def stop_after_10_steps(pipe, step_index, timestep, callback_kwargs):
    if step_index == 10:
        pipe._interrupt = True
    return callback_kwargs

frames = pipe(
    "a macro shot of frost forming on a leaf",
    callback_on_step_end=stop_after_10_steps,
    callback_on_step_end_tensor_inputs=["latents"],
).frames[0]
```

The public property `pipe.interrupt` exposes the current interrupt flag, but the
source uses the internal `_interrupt` field during the denoising loop. Treat
interrupt usage as advanced behavior.

## Reproducibility

Use a CUDA generator for deterministic noise on CUDA.

```python
generator = torch.Generator(device="cuda").manual_seed(42)

frames = pipe(
    "a polished chrome train crossing a snowy bridge",
    generator=generator,
    guidance_scale=7.5,
).frames[0]
```

To compare prompt changes against identical initial noise, pass `latents`
directly. Ensure the latent shape matches the requested batch, frame count, and
resolution.

## Gotchas

- There is only one Diffusers Allegro pipeline class in `v0.38.0`:
  `AllegroPipeline`. Do not expect an official Allegro image-to-video pipeline
  class from this API page.
- `num_videos_per_prompt` is in the signature, but the current source resets it
  to `1` inside `__call__`. Use an outer loop for multiple samples per prompt.
- `height` and `width` must be divisible by 8. Defaults are 720 x 1280 for the
  main checkpoint.
- `num_frames` must be positive. The default is 88 for the main checkpoint.
- Long prompts are tokenized to `max_sequence_length` and truncated with a
  warning. The documented default is 512.
- `clean_caption=True` needs `beautifulsoup4` and `ftfy`. Without them, the
  pipeline warns and uses less aggressive preprocessing.
- If you pass `prompt_embeds`, also pass `prompt_attention_mask`. If you pass
  `negative_prompt_embeds`, also pass `negative_prompt_attention_mask`.
- Positive and negative embedding tensors and masks must have matching shapes
  when passed directly.
- If `generator` is a list, its length must equal the effective batch size.
- `eta` is ignored unless the active scheduler supports it.
- The rendered docs prose still references older `callback`/`callback_steps`
  wording, but the actual source uses `callback_on_step_end`.
- `pipe.enable_vae_tiling()` and related pipeline-level VAE helpers are present
  but deprecated for removal in Diffusers `0.40.0`; call `pipe.vae.enable_tiling()`
  and `pipe.vae.enable_slicing()` instead.
- The generic model-card library snippet can show `.images[0]`; Allegro returns
  `.frames`.
- `output_type="latent"` skips VAE decode and returns compressed latent tensors,
  not displayable video frames.
- Full 88-frame 720p generation is large. Start with offload, VAE tiling, BF16,
  lower test settings, or quantization before running long production jobs.

## Quick Implementation Checklist

1. Load `AutoencoderKLAllegro` from `subfolder="vae"` in `torch.float32`.
2. Load `AllegroPipeline` from `rhymes-ai/Allegro` in `torch.bfloat16` or a
   quantized setup.
3. Enable VAE tiling with `pipe.vae.enable_tiling()`.
4. Use `.to("cuda")`, `device_map="balanced"`, `enable_model_cpu_offload()`, or
   `enable_sequential_cpu_offload()` according to available memory.
5. Generate with `guidance_scale=7.5`, `num_inference_steps=100`, and
   `max_sequence_length=512` for the documented baseline.
6. Read generated video frames from `result.frames[0]`.
7. Save with `diffusers.utils.export_to_video(..., fps=15)` or another video
   writer.

