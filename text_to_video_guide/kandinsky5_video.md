# Kandinsky 5.0 Video Diffusers Implementation Guide

Last checked: 2026-06-17 against the Hugging Face Diffusers
`kandinsky5_video` API page, the linked Diffusers `main` source, and current
Kandinsky Lab Hub model cards.

Kandinsky 5.0 Video is a Diffusers video family with two public pipeline
classes on the API page:

- `Kandinsky5T2VPipeline` for text-to-video.
- `Kandinsky5I2VPipeline` for image-to-video.

The Lite video line is the 2B parameter family. The Pro line is the 19B
parameter family for higher quality HD video and image-to-video variants. Both
families use latent diffusion with Flow Matching, a 3D DiT transformer,
Qwen2.5-VL plus CLIP text conditioning, HunyuanVideo 3D VAE latents, and NABLA
sparse attention support.

## Source Links

- Diffusers API page:
  <https://huggingface.co/docs/diffusers/api/pipelines/kandinsky5_video>
- Diffusers docs source:
  <https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/kandinsky5_video.md>
- Text-to-video pipeline source:
  <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/kandinsky5/pipeline_kandinsky.py>
- Image-to-video pipeline source:
  <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/kandinsky5/pipeline_kandinsky_i2v.py>
- Kandinsky output classes:
  <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/kandinsky5/pipeline_output.py>
- Kandinsky Lab Hub organization:
  <https://huggingface.co/kandinskylab/models>
- Original Kandinsky 5 repository:
  <https://github.com/kandinskylab/Kandinsky-5>

## Pipeline Selection

| Need | Pipeline | Typical checkpoint family | Notes |
| --- | --- | --- | --- |
| Text prompt to video | `Kandinsky5T2VPipeline` | T2V Lite or T2V Pro Diffusers checkpoints | Use this for the eight documented T2V Lite variants and the T2V Pro Diffusers variants. |
| Image plus prompt to video | `Kandinsky5I2VPipeline` | I2V Pro or I2V Lite Diffusers checkpoints | Pass `image=...`; the first latent frame is encoded from the input image and is not denoised in the main loop. |

Important documentation gotcha: the top-level "Basic Image-to-Video
Generation" block on the API page currently imports `Kandinsky5T2VPipeline` and
does not pass `image` to the call. The class autodoc and the I2V model cards use
`Kandinsky5I2VPipeline`, which matches the source and should be treated as the
implementation path for image-to-video.

## Checkpoint Matrix

The Diffusers API page lists the following drop-in Diffusers checkpoints.

### Pro Diffusers Checkpoints On The API Page

| Checkpoint | Pipeline | Duration | Type | Best use |
| --- | --- | ---: | --- | --- |
| `kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers` | `Kandinsky5T2VPipeline` | 5s | SFT | Highest quality Pro text-to-video in the documented Diffusers example. |
| `kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s-Diffusers` | `Kandinsky5I2VPipeline` | 5s | SFT | Highest quality documented Pro image-to-video. |

The Kandinsky Lab Hub also currently exposes Pro Diffusers distilled variants:

| Checkpoint | Pipeline | Duration | Type | Recommended settings |
| --- | --- | ---: | --- | --- |
| `kandinskylab/Kandinsky-5.0-T2V-Pro-distilled-5s-Diffusers` | `Kandinsky5T2VPipeline` | 5s | diffusion distilled | `num_inference_steps=16`, `guidance_scale=1.0`, Pro offload/compile settings. |
| `kandinskylab/Kandinsky-5.0-I2V-Pro-distilled-5s-Diffusers` | `Kandinsky5I2VPipeline` | 5s | diffusion distilled | `num_inference_steps=16`, `guidance_scale=1.0`, Pro offload/compile settings. |

The original Kandinsky 5 model zoo also lists Pro SFT and pretrain 5s/10s
repos without the `-Diffusers` suffix. Treat those as original-layout or
model-zoo checkpoints unless their model repo has a valid Diffusers
`model_index.json` for these classes. For a direct Diffusers integration,
prefer the `-Diffusers` checkpoints.

### Lite T2V Diffusers Checkpoints On The API Page

| Checkpoint | Duration | Type | Best use |
| --- | ---: | --- | --- |
| `kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers` | 5s | SFT | Highest quality Lite 5s T2V. |
| `kandinskylab/Kandinsky-5.0-T2V-Lite-sft-10s-Diffusers` | 10s | SFT | Highest quality Lite 10s T2V. |
| `kandinskylab/Kandinsky-5.0-T2V-Lite-nocfg-5s-Diffusers` | 5s | no-CFG distilled | 2x faster inference, run with `guidance_scale=1.0`. |
| `kandinskylab/Kandinsky-5.0-T2V-Lite-nocfg-10s-Diffusers` | 10s | no-CFG distilled | 2x faster 10s inference, run with `guidance_scale=1.0`. |
| `kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-5s-Diffusers` | 5s | diffusion distilled | 16 denoising steps, about 6x faster, run with `guidance_scale=1.0`. |
| `kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-10s-Diffusers` | 10s | diffusion distilled | 16 denoising steps for 10s clips, run with `guidance_scale=1.0`. |
| `kandinskylab/Kandinsky-5.0-T2V-Lite-pretrain-5s-Diffusers` | 5s | pretrain | Research and fine-tuning baseline. |
| `kandinskylab/Kandinsky-5.0-T2V-Lite-pretrain-10s-Diffusers` | 10s | pretrain | Longer research and fine-tuning baseline. |

The Hub also currently lists `kandinskylab/Kandinsky-5.0-I2V-Lite-5s-Diffusers`
for `Kandinsky5I2VPipeline`. It is not in the API page's "Available Models"
table, but the model card identifies it as a Diffusers I2V Lite checkpoint.

## Duration, Frames, And Resolution

The examples use 24 fps:

- 5 second outputs use `num_frames=121`.
- 10 second outputs use `num_frames=241`.

The extra frame is intentional. The source requires `num_frames % temporal_scale
== 1`; with the HunyuanVideo VAE temporal compression ratio normally equal to
4, valid frame counts are `4k + 1`. If the value does not match, the source logs
a warning and adjusts the frame count to `num_frames // 4 * 4 + 1`.

Resolution must be divisible by 16. Common documented settings are:

- Lite T2V: `height=512`, `width=768`.
- Pro T2V: `height=768`, `width=1024`.
- I2V examples: `height=512`, `width=768`, or square/HD settings on Pro model
  cards. Keep the input image resize aligned to the same `(width, height)` you
  pass to the pipeline.

The source picks an internal scale factor of `(1, 2, 2)` when both sides are in
the 480 to 854 range, otherwise `(1, 3.16, 3.16)`. That scale factor is passed
to the transformer, so changing resolution is not just a memory decision; it can
affect the model's denoising behavior.

## Shared Components

Both pipeline classes register the same main Diffusers components:

| Component | Type | Purpose |
| --- | --- | --- |
| `transformer` | `Kandinsky5Transformer3DModel` | Conditional 3D transformer that predicts velocity/noise over video latents. |
| `vae` | `AutoencoderKLHunyuanVideo` | Encodes and decodes HunyuanVideo video latents. |
| `text_encoder` | `Qwen2_5_VLForConditionalGeneration` | Frozen Qwen2.5-VL encoder for long prompt hidden states. |
| `tokenizer` | `Qwen2VLProcessor` / `AutoProcessor` | Tokenizes Qwen text input. |
| `text_encoder_2` | `CLIPTextModel` | Frozen CLIP text model, specifically CLIP ViT-L/14 in the docs. |
| `tokenizer_2` | `CLIPTokenizer` | Tokenizes the CLIP branch. |
| `scheduler` | `FlowMatchEulerDiscreteScheduler` | Flow Matching denoising scheduler. |

Both classes inherit `DiffusionPipeline` and `KandinskyLoraLoaderMixin`, so they
get standard pipeline loading, saving, device placement, offload hooks, and
compatible LoRA loading behavior. The model CPU offload sequence is:

```text
text_encoder -> text_encoder_2 -> transformer -> vae
```

Prompt encoding is dual-path. The Qwen branch wraps the user prompt in a video
prompt-engineering template, reserves 129 tokens for that template, and then
applies `max_sequence_length` to the user portion. The CLIP branch tokenizes to
77 tokens and returns pooled CLIP embeddings. The public `encode_prompt` method
returns:

- Qwen hidden states.
- CLIP pooled projections.
- Qwen cumulative sequence lengths (`cu_seqlens`) for variable-length attention.

## `Kandinsky5T2VPipeline`

Use this class for text-to-video checkpoints.

### Minimal Lite SFT Example

```python
import torch
from diffusers import Kandinsky5T2VPipeline
from diffusers.utils import export_to_video

model_id = "kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers"

pipe = Kandinsky5T2VPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

prompt = "A cat and a dog baking a cake together in a kitchen."
negative_prompt = (
    "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst "
    "quality, low quality, ugly, deformed, walking backwards"
)

frames = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=512,
    width=768,
    num_frames=121,
    num_inference_steps=50,
    guidance_scale=5.0,
    generator=torch.Generator(device="cuda").manual_seed(7),
).frames[0]

export_to_video(frames, "kandinsky5_lite_t2v.mp4", fps=24, quality=9)
```

### Pro Example

Pro checkpoints are much larger. The Diffusers docs and model cards warn to use
CPU offload for Pro inference, and the examples also set Flex attention and
compile the transformer.

```python
import torch
from diffusers import Kandinsky5T2VPipeline
from diffusers.utils import export_to_video

pipe = Kandinsky5T2VPipeline.from_pretrained(
    "kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
pipe.transformer.set_attention_backend("flex")
pipe.enable_model_cpu_offload()
pipe.transformer.compile(mode="max-autotune-no-cudagraphs", dynamic=True)

frames = pipe(
    prompt="A cat and a dog baking a cake together in a kitchen.",
    negative_prompt="Static, 2D cartoon, cartoon, low quality, ugly, deformed",
    height=768,
    width=1024,
    num_frames=121,
    num_inference_steps=50,
    guidance_scale=5.0,
).frames[0]

export_to_video(frames, "kandinsky5_pro_t2v.mp4", fps=24, quality=9)
```

### 10 Second Lite Example

The API page warns that all 10 second models should use Flex attention and
`max-autotune-no-cudagraphs` compilation.

```python
import torch
from diffusers import Kandinsky5T2VPipeline
from diffusers.utils import export_to_video

pipe = Kandinsky5T2VPipeline.from_pretrained(
    "kandinskylab/Kandinsky-5.0-T2V-Lite-sft-10s-Diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
pipe.transformer.set_attention_backend("flex")
pipe.transformer.compile(mode="max-autotune-no-cudagraphs", dynamic=True)

frames = pipe(
    prompt="A wide shot of a night market after rain, cinematic camera motion.",
    negative_prompt="Static, low quality, distorted motion, ugly, deformed",
    height=512,
    width=768,
    num_frames=241,
    num_inference_steps=50,
    guidance_scale=5.0,
).frames[0]

export_to_video(frames, "kandinsky5_lite_10s.mp4", fps=24, quality=9)
```

### No-CFG And Diffusion-Distilled Example

For `nocfg` and `distilled16steps` Lite checkpoints, and for the currently
listed Pro distilled Diffusers checkpoints, run without classifier-free
guidance:

```python
import torch
from diffusers import Kandinsky5T2VPipeline
from diffusers.utils import export_to_video

pipe = Kandinsky5T2VPipeline.from_pretrained(
    "kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-5s-Diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

frames = pipe(
    prompt="A beautiful sunset over mountains, slow aerial camera motion.",
    num_frames=121,
    num_inference_steps=16,
    guidance_scale=1.0,
).frames[0]

export_to_video(frames, "kandinsky5_lite_distilled.mp4", fps=24, quality=9)
```

When `guidance_scale <= 1.0`, the negative prompt path is skipped. Passing a
negative prompt is harmless but not useful for no-CFG generation.

### T2V Parameters

| Parameter | Default in source | Implementation notes |
| --- | --- | --- |
| `prompt` | `None` | String or list. Required unless all positive embedding tensors are supplied. |
| `negative_prompt` | `None` | Used only when `guidance_scale > 1.0`. If omitted under CFG, source inserts a default negative prompt. |
| `height`, `width` | `512`, `768` | Must be divisible by 16. |
| `num_frames` | `121` | Use `121` for 5s and `241` for 10s at 24 fps. Source adjusts to `4k + 1` when needed. |
| `num_inference_steps` | `50` | Use `16` for distilled16steps checkpoints. |
| `guidance_scale` | `5.0` | Set to `1.0` for no-CFG and distilled checkpoints. |
| `num_videos_per_prompt` | `1` | Keep as `1` for T2V unless validating current source behavior; see gotchas. |
| `generator` | `None` | Pass a CUDA generator for deterministic GPU runs. |
| `latents` | `None` | Precomputed latents bypass random latent creation. Must match latent shape. |
| `prompt_embeds_qwen`, `prompt_embeds_clip`, `prompt_cu_seqlens` | `None` | If any one is supplied, all three positive embedding inputs must be supplied. |
| `negative_prompt_embeds_qwen`, `negative_prompt_embeds_clip`, `negative_prompt_cu_seqlens` | `None` | Same all-or-none rule for negative embeddings. |
| `output_type` | `"pil"` | Use `"latent"` to skip VAE decode and return latent tensors. Other values are passed to `VideoProcessor.postprocess_video`. |
| `return_dict` | `True` | Returns `KandinskyPipelineOutput`; `False` returns a one-element tuple. |
| `callback_on_step_end` | `None` | Can update supported tensors at each denoising step. |
| `callback_on_step_end_tensor_inputs` | `["latents"]` | Allowed names are `latents`, positive/negative Qwen embeddings, and positive/negative CLIP embeddings. |
| `max_sequence_length` | `512` | Must not exceed `1024` in source validation. |

## `Kandinsky5I2VPipeline`

Use this class for image-to-video checkpoints. It is structurally similar to
the T2V class, but `prepare_latents` encodes the input image with the VAE,
places it in the first latent frame, and the denoising loop updates only frames
after the first one.

### Pro I2V Example

```python
import torch
from diffusers import Kandinsky5I2VPipeline
from diffusers.utils import export_to_video, load_image

height = 512
width = 768
image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/"
    "resolve/main/diffusers/astronaut.jpg"
).resize((width, height))

pipe = Kandinsky5I2VPipeline.from_pretrained(
    "kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s-Diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
pipe.transformer.set_attention_backend("flex")
pipe.enable_model_cpu_offload()
pipe.transformer.compile(mode="max-autotune-no-cudagraphs", dynamic=True)

frames = pipe(
    image=image,
    prompt="An astronaut floating in space with Earth in the background.",
    negative_prompt="Static, 2D cartoon, cartoon, low quality, ugly, deformed",
    height=height,
    width=width,
    num_frames=121,
    num_inference_steps=50,
    guidance_scale=5.0,
).frames[0]

export_to_video(frames, "kandinsky5_pro_i2v.mp4", fps=24, quality=9)
```

### Lite I2V Example

```python
import torch
from diffusers import Kandinsky5I2VPipeline
from diffusers.utils import export_to_video, load_image

height = 512
width = 768
image = load_image("input.png").resize((width, height))

pipe = Kandinsky5I2VPipeline.from_pretrained(
    "kandinskylab/Kandinsky-5.0-I2V-Lite-5s-Diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

frames = pipe(
    image=image,
    prompt="The subject turns toward camera while soft daylight moves across the scene.",
    negative_prompt="Static, low quality, warped anatomy, deformed motion",
    height=height,
    width=width,
    num_frames=121,
    num_inference_steps=50,
    guidance_scale=5.0,
).frames[0]

export_to_video(frames, "kandinsky5_lite_i2v.mp4", fps=24, quality=9)
```

### I2V Parameters

`Kandinsky5I2VPipeline.__call__` has the same core parameters as the T2V class
plus a required `image` input.

| Parameter | Default in source | Implementation notes |
| --- | --- | --- |
| `image` | required | Accepts a PIL image, NumPy array, torch tensor, or list. It is preprocessed to the requested `height` and `width`. |
| `prompt` | `None` | Required unless positive Qwen, CLIP, and Qwen `cu_seqlens` embeddings are supplied. |
| `negative_prompt` | `None` | Used only with CFG. Default negative prompt is inserted when CFG is enabled and no negative prompt is provided. |
| `height`, `width` | `512`, `768` | Must be divisible by 16. Match manual image resizing to these values. |
| `num_frames` | `121` | Same `4k + 1` rule as T2V. |
| `num_inference_steps` | `50` | Use `16` for distilled checkpoints. |
| `guidance_scale` | `5.0` | Set to `1.0` for no-CFG/distilled variants. |
| `num_videos_per_prompt` | `1` | I2V source forwards this value into prompt encoding. |
| `latents` | `None` | If supplied, source returns them directly after dtype/device transfer and does not encode `image` into the first frame. |
| `output_type` | `"pil"` | `"latent"` returns latents after denoising and I2V first-frame normalization. |
| `max_sequence_length` | `512` | Source validation rejects values greater than `1024`. |

After denoising, I2V source calls `normalize_first_frame` to reduce mesh-like
artifacts around the conditioned first frames. That helper normalizes the first
four latent frames against following reference frames when more than one latent
frame is present.

## Outputs

Both video classes return `KandinskyPipelineOutput(frames=video)` when
`return_dict=True`.

The output class documents `frames` as either:

- a nested list of `batch_size` videos, each containing `num_frames` PIL images;
- a NumPy array; or
- a torch tensor with video shape.

In the common `output_type="pil"` path, examples use:

```python
frames = pipe(...).frames[0]
export_to_video(frames, "output.mp4", fps=24, quality=9)
```

With `return_dict=False`, the call returns `(video,)`. With
`output_type="latent"`, the VAE decode and postprocessing are skipped and the
returned value is the latent tensor.

## Memory And Performance

- Use `torch_dtype=torch.bfloat16`, which is what the official examples use.
- For all Pro Diffusers checkpoints, call `enable_model_cpu_offload()`. The Pro
  model cards and API page explicitly warn that Pro models should use CPU
  offload for single-GPU inference.
- For Pro examples, the official cards also set
  `pipe.transformer.set_attention_backend("flex")` and compile the transformer
  with `mode="max-autotune-no-cudagraphs", dynamic=True`.
- For 10 second models, the API page warns to use Flex attention and the same
  compile mode.
- For no-CFG and diffusion-distilled checkpoints, use `guidance_scale=1.0`.
  Diffusion-distilled checkpoints are intended for `num_inference_steps=16`.
- Reduce `height`, `width`, or `num_frames` first when memory is tight. Moving
  from 10s (`241` frames) to 5s (`121` frames) is a large memory and runtime
  reduction.
- `output_type="latent"` is useful for debugging scheduler or conditioning
  issues because it skips VAE decoding, but those latents are not directly
  viewable videos.

## Source-Level Gotchas

- The API page's T2V parameter prose says `num_frames` defaults to 25, but the
  source signature and examples use `121`. Trust the source signature for
  current behavior.
- The API page's top I2V usage block appears to contain a class/call typo. Use
  `Kandinsky5I2VPipeline` and pass `image=...` for image-to-video.
- `height` and `width` must be divisible by 16 or `check_inputs` raises
  `ValueError`.
- `max_sequence_length` must be `1024` or less. The Qwen prompt template also
  consumes 129 tokens before the user prompt.
- If you pass precomputed positive embeddings, pass all three:
  `prompt_embeds_qwen`, `prompt_embeds_clip`, and `prompt_cu_seqlens`. The
  negative branch has the same all-or-none rule.
- With CFG enabled, a list `negative_prompt` must have the same length as the
  prompt list.
- The T2V source currently does not forward `num_videos_per_prompt` into
  `encode_prompt` inside `__call__`, while I2V does. Keep
  `num_videos_per_prompt=1` for T2V unless you have validated the installed
  Diffusers version or are supplying correctly repeated embeddings yourself.
- If a generator list is supplied, its length must match the effective batch
  size (`batch_size * num_videos_per_prompt`).
- In I2V, supplying custom `latents` skips image encoding in `prepare_latents`.
  Only do this when the latents already include the intended image conditioning.
- `Kandinsky5I2VPipeline` denoises `latents[:, 1:]`, leaving the first latent
  frame anchored to the input image. This is the key behavioral difference from
  T2V.
- For checkpoints discovered on the Hub but not listed on the Diffusers API
  page, verify they include a Diffusers `model_index.json` and the expected
  pipeline class before wiring them into production code.

## Implementation Checklist

1. Pick `Kandinsky5T2VPipeline` or `Kandinsky5I2VPipeline` based on whether the
   request has an input image.
2. Pick a checkpoint by quality, duration, and speed:
   SFT for quality, no-CFG for faster CFG-free inference, distilled16steps for
   16-step low latency, pretrain for research/fine-tuning.
3. Set `num_frames=121` for 5s or `num_frames=241` for 10s at 24 fps.
4. Keep `height` and `width` divisible by 16 and aligned with input image
   resizing for I2V.
5. Use `torch.bfloat16`; add Pro offload and Flex/compile settings for Pro and
   10s runs.
6. Use `guidance_scale=1.0` for no-CFG and distilled checkpoints.
7. Export `pipe(...).frames[0]` with `export_to_video(..., fps=24, quality=9)`.
