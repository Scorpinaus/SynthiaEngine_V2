# SkyReels-V2 Diffusers Implementation Guide

Last checked: 2026-06-18 against the Hugging Face Diffusers SkyReels-V2 API
page, the linked Diffusers source files, and the official Skywork model cards.

Research target:
https://huggingface.co/docs/diffusers/api/pipelines/skyreels_v2

Primary Diffusers classes on the page:

- `SkyReelsV2DiffusionForcingPipeline`
- `SkyReelsV2DiffusionForcingImageToVideoPipeline`
- `SkyReelsV2DiffusionForcingVideoToVideoPipeline`
- `SkyReelsV2Pipeline`
- `SkyReelsV2ImageToVideoPipeline`
- `SkyReelsV2PipelineOutput`

SkyReels-V2 is Skywork AI's video generation family for short clips and
long-form generation. The Diffusers page exposes two related surfaces:
Diffusion Forcing pipelines for autoregressive long video work, and regular
SkyReels-V2 pipelines for direct text-to-video or image-to-video generation.
For a workflow server, the important split is simple: use the non-DF pipelines
for normal single-window generation, and use the DF pipelines when you need
long clips, video extension, synchronous/asynchronous autoregressive settings,
or first/last frame control with the DF checkpoints.

## 1. Pipeline Selection

| Class | Task | Use when |
| --- | --- | --- |
| `SkyReelsV2DiffusionForcingPipeline` | Text-to-video with Diffusion Forcing | You want text-only generation from a DF checkpoint, especially long-form generation with `base_num_frames`, `overlap_history`, `ar_step`, and `causal_block_size`. |
| `SkyReelsV2DiffusionForcingImageToVideoPipeline` | Image-to-video with Diffusion Forcing | You want to condition on a first frame, or on both a first frame and `last_image`, while retaining the DF long-window controls. |
| `SkyReelsV2DiffusionForcingVideoToVideoPipeline` | Video-to-video / video extension with Diffusion Forcing | You want to extend an existing video. The docs note that output length is the input video frames plus the requested generated frames. |
| `SkyReelsV2Pipeline` | Text-to-video without Diffusion Forcing | You want a regular T2V run with a T2V checkpoint and do not need long-window DF controls. |
| `SkyReelsV2ImageToVideoPipeline` | Image-to-video without Diffusion Forcing | You want a regular I2V run from one image, optionally with a last-frame condition in versions that expose `last_image`. |
| `SkyReelsV2PipelineOutput` | Shared output dataclass | All SkyReels-V2 pipelines return this when `return_dict=True`; read generated video data from `.frames`. |

All five generation pipelines share the same core model family:

- `AutoTokenizer`
- `T5EncoderModel` or `UMT5EncoderModel`, using the `google/umt5-xxl`
  tokenizer/text-encoder family documented by Diffusers
- `SkyReelsV2Transformer3DModel`
- `AutoencoderKLWan`
- `UniPCMultistepScheduler`

The source sets the model CPU offload order to
`text_encoder -> transformer -> vae` and uses `VideoProcessor` with the Wan VAE
scale factors.

## 2. Official Entry Points

- Pipeline docs: <https://huggingface.co/docs/diffusers/api/pipelines/skyreels_v2>
- Docs source: <https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/skyreels_v2.md>
- DF text-to-video source: <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing.py>
- DF image-to-video source: <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_i2v.py>
- DF video-to-video source: <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_v2v.py>
- Regular text-to-video source: <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2.py>
- Regular image-to-video source: <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_i2v.py>
- Output source: <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/skyreels_v2/pipeline_output.py>
- Original Skywork repository: <https://github.com/SkyworkAI/SkyReels-V2>
- Skywork SkyReels-V2 collection: <https://huggingface.co/collections/Skywork/skyreels-v2-6801b1b93df627d441d0d0d9>

## 3. Checkpoints And Model IDs

The Diffusers API page lists these supported Diffusers-format model IDs:

| Model ID | Pipeline family | Practical use |
| --- | --- | --- |
| `Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers` | DF | Lower-memory DF text/image long-video work at the documented 540P shape. Good first local target. |
| `Skywork/SkyReels-V2-DF-14B-540P-Diffusers` | DF | Higher-quality DF long-video generation at 540P. Much heavier than 1.3B. |
| `Skywork/SkyReels-V2-DF-14B-720P-Diffusers` | DF | Higher-resolution DF generation. Use when the machine can tolerate very high VRAM demand. |
| `Skywork/SkyReels-V2-T2V-14B-540P-Diffusers` | Regular T2V | Prompt-only short-window text-to-video at 540P. |
| `Skywork/SkyReels-V2-T2V-14B-720P-Diffusers` | Regular T2V | Prompt-only short-window text-to-video at 720P. |
| `Skywork/SkyReels-V2-I2V-1.3B-540P-Diffusers` | Regular I2V | Lower-memory image-to-video. Useful for local validation. |
| `Skywork/SkyReels-V2-I2V-14B-540P-Diffusers` | Regular I2V | Higher-quality image-to-video at 540P. |
| `Skywork/SkyReels-V2-I2V-14B-720P-Diffusers` | Regular I2V | Higher-resolution image-to-video. |

The original Skywork model card gives the practical base shapes:

| Resolution family | Recommended generation shape |
| --- | --- |
| 540P | `height=544`, `width=960`, `num_frames=97` |
| 720P | `height=720`, `width=1280`, `num_frames=121` |

For DF long-video runs, keep `base_num_frames` tied to the model family:
`97` for 540P examples and `121` for 720P examples. `num_frames` can be larger
than `base_num_frames`, but then `overlap_history` is required.

Do not mix the checkpoint families casually. DF checkpoints are meant for the
DF classes. T2V checkpoints are meant for `SkyReelsV2Pipeline`. I2V checkpoints
are meant for `SkyReelsV2ImageToVideoPipeline`.

## 4. Installation

Use a recent Diffusers build that includes the SkyReels-V2 classes.

```powershell
.venv\Scripts\python.exe -m pip install -U diffusers transformers accelerate torch torchvision ftfy imageio imageio-ffmpeg
```

For the newest SkyReels-V2 integration before a package release:

```powershell
.venv\Scripts\python.exe -m pip install -U git+https://github.com/huggingface/diffusers
```

`ftfy` is worth installing because the source prompt-cleaning helpers use it.
The official Skywork quickstart also calls it out.

## 5. Common Loading Pattern

The Diffusers examples load the VAE in `torch.float32` and the rest of the
pipeline in `torch.bfloat16`.

```python
import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler

model_id = "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers"

vae = AutoencoderKLWan.from_pretrained(
    model_id,
    subfolder="vae",
    torch_dtype=torch.float32,
)
```

Then pass `vae=vae` into the selected pipeline's `from_pretrained(...)`.

Scheduler shift matters:

| Use | Diffusers page setting |
| --- | --- |
| T2V | `flow_shift=8.0` |
| I2V / first-last-frame / V2V | `flow_shift=5.0` |

```python
pipeline.scheduler = UniPCMultistepScheduler.from_config(
    pipeline.scheduler.config,
    flow_shift=8.0,  # use 5.0 for I2V/V2V examples
)
```

Use one device strategy:

- Fast path when VRAM is available: `pipeline.to("cuda")`.
- Memory path: `pipeline.enable_model_cpu_offload()` instead of moving the
  whole pipeline to CUDA up front.

The original model card recommends offload for the large models. The Diffusers
source advertises the offload sequence `text_encoder -> transformer -> vae`,
which is the correct ordering for `enable_model_cpu_offload()`.

## 6. Diffusion Forcing Controls

The DF pipelines add long-video and asynchronous generation parameters that do
not exist on the regular pipelines.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `base_num_frames` | `97` | The per-window frame count, effectively the sliding context size for long-video generation. Larger values cost more VRAM. |
| `overlap_history` | `None` | Number of frames to reuse between windows for smooth long-video transitions. Required by source validation when `num_frames > base_num_frames`. Docs and model cards use `17` for long video. |
| `addnoise_condition` | `0` | Adds noise to clean conditioning frames. Skywork recommends `20` for long-video smoothing and warns that too much can cause inconsistency. |
| `ar_step` | `0` | `0` is synchronous mode. Values greater than `0` enable asynchronous autoregressive Diffusion Forcing. Docs use `5`. |
| `causal_block_size` | `None` in pipeline calls | Number of latent frames per causal block. Use `5` with `ar_step=5` in the documented async mode. |
| `fps` | `24` | FPS conditioning passed into the transformer. Match this with `export_to_video(..., fps=24)` unless you intentionally separate model conditioning from export playback. |

The API page's visual explanation uses `vae_scale_factor_temporal=4`:

```text
num_latent_frames = (num_frames - 1) // 4 + 1
```

For `num_frames=97`, this gives `25` latent frames. With
`causal_block_size=5`, the async run has `5` blocks. In asynchronous mode,
later blocks lag behind earlier blocks by `ar_step`, creating the staggered
Diffusion Forcing schedule. This takes more total denoising iterations than
synchronous mode, but the original notes say it may improve prompt following
and visual consistency.

The source raises a `ValueError` if `ar_step` is too small for the current
number of latent blocks. The original model-card note is also strict that, for
async generation, the latent frame count processed in each iteration must be
divisible by `causal_block_size`. Treat arbitrary `num_frames` values as risky;
prefer the documented long-video counts until you have a tested helper that
checks the latent math.

## 7. Text-To-Video With `SkyReelsV2DiffusionForcingPipeline`

Use this for DF text-only generation. This is the best entry point for long
clips and "infinite length" style generation.

```python
import torch
from diffusers import (
    AutoencoderKLWan,
    SkyReelsV2DiffusionForcingPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils import export_to_video

model_id = "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers"

vae = AutoencoderKLWan.from_pretrained(
    model_id,
    subfolder="vae",
    torch_dtype=torch.float32,
)
pipe = SkyReelsV2DiffusionForcingPipeline.from_pretrained(
    model_id,
    vae=vae,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config,
    flow_shift=8.0,
)

prompt = (
    "A cat and a dog baking a cake together in a cozy kitchen. The cat "
    "carefully measures flour while the dog stirs batter with a wooden spoon. "
    "Sunlight streams through the window, cinematic, detailed, warm colors."
)

frames = pipe(
    prompt=prompt,
    negative_prompt="",
    height=544,
    width=960,
    num_frames=97,
    base_num_frames=97,
    num_inference_steps=30,
    guidance_scale=6.0,
    ar_step=5,
    causal_block_size=5,
    overlap_history=None,
    addnoise_condition=20,
    fps=24,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

export_to_video(frames, "skyreels_v2_df_t2v.mp4", fps=24, quality=8)
```

For synchronous short generation, use `ar_step=0` and omit
`causal_block_size`. For long generation, set `num_frames` greater than
`base_num_frames` and provide `overlap_history=17`.

## 8. Long-Form DF Settings

The original model card gives these useful long-generation patterns:

| Goal | Settings |
| --- | --- |
| Short 540P clip | `num_frames=97`, `base_num_frames=97`, `overlap_history=None` |
| Short 720P clip | `num_frames=121`, `base_num_frames=121`, `overlap_history=None` |
| Longer synchronous 540P clip | `ar_step=0`, `base_num_frames=97`, `num_frames=257`, `overlap_history=17`, `addnoise_condition=20` |
| Longer asynchronous 540P clip | `ar_step=5`, `causal_block_size=5`, `base_num_frames=97`, `num_frames=737`, `overlap_history=17`, `addnoise_condition=20` |
| Very long 540P clip | The model card mentions `num_frames=1457` for roughly 60 seconds, but also says these frame counts are training-aligned rather than exact duration math. |

Template:

```python
frames = pipe(
    prompt=prompt,
    height=544,
    width=960,
    num_frames=257,
    base_num_frames=97,
    overlap_history=17,
    addnoise_condition=20,
    ar_step=0,
    num_inference_steps=30,
    guidance_scale=6.0,
    fps=24,
).frames[0]
```

For asynchronous long form:

```python
frames = pipe(
    prompt=prompt,
    height=544,
    width=960,
    num_frames=737,
    base_num_frames=97,
    overlap_history=17,
    addnoise_condition=20,
    ar_step=5,
    causal_block_size=5,
    num_inference_steps=30,
    guidance_scale=6.0,
    fps=24,
).frames[0]
```

The tradeoff is clear: asynchronous mode is slower because it takes more
denoising iterations, but the official notes say it may improve instruction
following and visual consistency.

## 9. First/Last Frame With `SkyReelsV2DiffusionForcingImageToVideoPipeline`

Use the DF I2V pipeline when you want a first frame and an optional ending
frame while still using DF settings. The docs demonstrate this as
First-Last-Frame-to-Video.

```python
import numpy as np
import torch
import torchvision.transforms.functional as TF
from diffusers import (
    AutoencoderKLWan,
    SkyReelsV2DiffusionForcingImageToVideoPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils import export_to_video, load_image

model_id = "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers"

vae = AutoencoderKLWan.from_pretrained(
    model_id,
    subfolder="vae",
    torch_dtype=torch.float32,
)
pipe = SkyReelsV2DiffusionForcingImageToVideoPipeline.from_pretrained(
    model_id,
    vae=vae,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config,
    flow_shift=5.0,
)

first_frame = load_image("path/to/first_frame.png")
last_frame = load_image("path/to/last_frame.png")

def aspect_ratio_resize(image, pipeline, max_area=720 * 1280):
    aspect_ratio = image.height / image.width
    mod_value = pipeline.vae_scale_factor_spatial * pipeline.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    return image.resize((width, height)), height, width

def center_crop_resize(image, height, width):
    resize_ratio = max(width / image.width, height / image.height)
    resized_width = round(image.width * resize_ratio)
    resized_height = round(image.height * resize_ratio)
    image = image.resize((resized_width, resized_height))
    image = TF.center_crop(image, [height, width])
    return image

first_frame, height, width = aspect_ratio_resize(first_frame, pipe)
if last_frame.size != first_frame.size:
    last_frame = center_crop_resize(last_frame, height, width)

prompt = (
    "CG animation style, a small blue bird takes off from the ground, "
    "flapping its wings. The camera follows the bird upward through a bright "
    "blue sky with soft clouds."
)

frames = pipe(
    image=first_frame,
    last_image=last_frame,
    prompt=prompt,
    height=height,
    width=width,
    guidance_scale=5.0,
    num_frames=97,
    base_num_frames=97,
    num_inference_steps=30,
    fps=24,
).frames[0]

export_to_video(frames, "skyreels_v2_df_flf2v.mp4", fps=24, quality=8)
```

Implementation notes:

- `image` is the starting visual condition.
- `last_image` is the ending visual condition. The rendered signature displays
  `last_image` as a tensor, while the official example passes an image object.
  In app code, preprocess it to the same dimensions as `image` before calling
  the pipeline.
- `image_embeds` can be supplied when you already have encoded image
  conditioning and want to skip image encoding.
- Use `flow_shift=5.0` for this Diffusers I2V/FLF path.

## 10. Image-To-Video With `SkyReelsV2DiffusionForcingImageToVideoPipeline`

For a normal first-frame-to-video DF run, pass only `image`.

```python
import torch
from diffusers import (
    AutoencoderKLWan,
    SkyReelsV2DiffusionForcingImageToVideoPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils import export_to_video, load_image

model_id = "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers"

vae = AutoencoderKLWan.from_pretrained(
    model_id,
    subfolder="vae",
    torch_dtype=torch.float32,
)
pipe = SkyReelsV2DiffusionForcingImageToVideoPipeline.from_pretrained(
    model_id,
    vae=vae,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config,
    flow_shift=5.0,
)

image = load_image("path/to/start.png")
prompt = (
    "A person in a red raincoat walks along a reflective city street at night, "
    "neon signs glowing in puddles, smooth camera movement."
)

frames = pipe(
    image=image,
    prompt=prompt,
    height=544,
    width=960,
    num_frames=97,
    base_num_frames=97,
    num_inference_steps=30,
    guidance_scale=5.0,
    fps=24,
).frames[0]

export_to_video(frames, "skyreels_v2_df_i2v.mp4", fps=24, quality=8)
```

For long I2V, use the same long-form parameters as DF T2V:
`num_frames > base_num_frames`, `overlap_history=17`,
`addnoise_condition=20`, and optionally `ar_step=5` plus
`causal_block_size=5`.

## 11. Video-To-Video With `SkyReelsV2DiffusionForcingVideoToVideoPipeline`

Use this pipeline to extend an existing video. The docs example states that
the total output frames are the number of frames in the input video plus the
requested generated frame count.

```python
import torch
from diffusers import (
    AutoencoderKLWan,
    SkyReelsV2DiffusionForcingVideoToVideoPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils import export_to_video, load_video

model_id = "Skywork/SkyReels-V2-DF-14B-720P-Diffusers"

vae = AutoencoderKLWan.from_pretrained(
    model_id,
    subfolder="vae",
    torch_dtype=torch.float32,
)
pipe = SkyReelsV2DiffusionForcingVideoToVideoPipeline.from_pretrained(
    model_id,
    vae=vae,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config,
    flow_shift=5.0,
)

video = load_video("input_video.mp4")
prompt = (
    "CG animation style, a small blue bird keeps flying through a bright sky. "
    "The camera follows from a close low angle with smooth cinematic motion."
)

frames = pipe(
    video=video,
    prompt=prompt,
    height=720,
    width=1280,
    guidance_scale=5.0,
    overlap_history=17,
    num_inference_steps=30,
    num_frames=257,
    base_num_frames=121,
    fps=24,
).frames[0]

export_to_video(frames, "skyreels_v2_df_v2v.mp4", fps=24, quality=8)
```

V2V-specific notes:

- `video` is a list of input frames, usually from `diffusers.utils.load_video`.
- The default `num_frames` for this pipeline is `120`, unlike the T2V/I2V
  defaults of `97`.
- The same prompt rules apply: pass either `prompt` or `prompt_embeds`, and
  pass either `negative_prompt` or `negative_prompt_embeds`.
- Match `height` and `width` to the model family and to your input video
  preprocessing. If the input dimensions differ, resize/crop before calling.

## 12. Text-To-Video With `SkyReelsV2Pipeline`

Use the regular T2V pipeline when you do not need DF long-window behavior.
The API signature is simpler:

```python
pipe(
    prompt=None,
    negative_prompt=None,
    height=544,
    width=960,
    num_frames=97,
    num_inference_steps=50,
    guidance_scale=6.0,
    num_videos_per_prompt=1,
    generator=None,
    latents=None,
    prompt_embeds=None,
    negative_prompt_embeds=None,
    output_type="np",
    return_dict=True,
    attention_kwargs=None,
    callback_on_step_end=None,
    callback_on_step_end_tensor_inputs=["latents"],
    max_sequence_length=512,
)
```

Example:

```python
import torch
from diffusers import AutoencoderKLWan, SkyReelsV2Pipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video

model_id = "Skywork/SkyReels-V2-T2V-14B-540P-Diffusers"

vae = AutoencoderKLWan.from_pretrained(
    model_id,
    subfolder="vae",
    torch_dtype=torch.float32,
)
pipe = SkyReelsV2Pipeline.from_pretrained(
    model_id,
    vae=vae,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config,
    flow_shift=8.0,
)

prompt = (
    "A serene lake surrounded by towering mountains, a few swans gliding "
    "across the water, sunlight dancing on the surface, cinematic realism."
)

frames = pipe(
    prompt=prompt,
    height=544,
    width=960,
    num_frames=97,
    num_inference_steps=50,
    guidance_scale=6.0,
).frames[0]

export_to_video(frames, "skyreels_v2_t2v.mp4", fps=24, quality=8)
```

The regular `SkyReelsV2Pipeline` signature on the current API page does not
include `fps`; playback FPS is chosen when exporting:

```python
frames = pipe(
    prompt=prompt,
    height=544,
    width=960,
    num_frames=97,
    num_inference_steps=50,
    guidance_scale=6.0,
).frames[0]
```

The regular pipeline is a better fit for short T2V because it has fewer moving
parts. It does not expose `base_num_frames`, `overlap_history`,
`addnoise_condition`, `ar_step`, `causal_block_size`, or `fps`.

## 13. Image-To-Video With `SkyReelsV2ImageToVideoPipeline`

Use the regular I2V pipeline with the I2V checkpoints.

```python
import torch
from diffusers import (
    AutoencoderKLWan,
    SkyReelsV2ImageToVideoPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils import export_to_video, load_image

model_id = "Skywork/SkyReels-V2-I2V-1.3B-540P-Diffusers"

vae = AutoencoderKLWan.from_pretrained(
    model_id,
    subfolder="vae",
    torch_dtype=torch.float32,
)
pipe = SkyReelsV2ImageToVideoPipeline.from_pretrained(
    model_id,
    vae=vae,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config,
    flow_shift=5.0,
)

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/guitar-man.png"
)
prompt = "A man with short gray hair plays a red electric guitar."

frames = pipe(
    image=image,
    prompt=prompt,
    height=544,
    width=960,
    num_frames=97,
    num_inference_steps=50,
    guidance_scale=5.0,
).frames[0]

export_to_video(frames, "skyreels_v2_i2v.mp4", fps=24, quality=8)
```

The regular I2V call accepts:

- `image`: the visual starting condition.
- `image_embeds`: optional precomputed image embeddings.
- `last_image`: optional ending-frame condition in the current API signature.
- `guidance_scale=5.0` by default.

The rendered docs say `return_dict=False` returns a tuple, and
`return_dict=True` returns `SkyReelsV2PipelineOutput`.

## 14. Key Parameters By Pipeline

| Parameter | DF T2V | DF I2V | DF V2V | Regular T2V | Regular I2V |
| --- | --- | --- | --- | --- | --- |
| `prompt` | Yes | Optional if using embeds | Optional if using embeds | Optional if using embeds | Optional if using embeds |
| `image` | No | Required unless `image_embeds` | No | No | Required unless `image_embeds` |
| `video` | No | No | Required | No | No |
| `last_image` | No | Yes | No | No | Yes |
| `height` / `width` | Default `544` / `960` | Default `544` / `960` | Default `544` / `960` | Default `544` / `960` | Default `544` / `960` |
| `num_frames` | Default `97` | Default `97` | Default `120` | Default `97` | Default `97` |
| `guidance_scale` | Default `6.0` | Default `5.0` | Default `6.0` | Default `6.0` | Default `5.0` |
| `base_num_frames` | Yes, default `97` | Yes, default `97` | Yes, default `97` | No | No |
| `overlap_history` | Yes | Yes | Yes | No | No |
| `addnoise_condition` | Yes | Yes | Yes | No | No |
| `ar_step` | Yes | Yes | Yes | No | No |
| `causal_block_size` | Yes | Yes | Yes | No | No |
| `fps` | Yes, default `24` | Yes, default `24` | Yes, default `24` | No | No |
| `max_sequence_length` | Default `512` | Default `512` | Default `512` | Default `512` | Default `512` |

Common source-level validation:

- `height` and `width` must be divisible by `16`.
- Pass either `prompt` or `prompt_embeds`, not both.
- Pass either `negative_prompt` or `negative_prompt_embeds`, not both.
- If using `negative_prompt`, its type and batch size must match `prompt`.
- `callback_on_step_end_tensor_inputs` must be one of the pipeline's allowed
  callback tensors. The source class allows `latents`, `prompt_embeds`, and
  `negative_prompt_embeds`; the call default is `["latents"]`.
- I2V pipelines require either `image` or `image_embeds`, not both.
- DF long-video calls require `overlap_history` when
  `num_frames > base_num_frames`.

## 15. Memory And Performance

SkyReels-V2 is not a lightweight video model. The official Skywork notes give
these approximate 540P peak VRAM numbers:

| Scenario | Approximate peak VRAM from model card |
| --- | --- |
| 1.3B 540P generation | `14.7GB` |
| 14B 540P regular T2V/I2V generation | `43.4GB` |
| 14B 540P DF generation | `51.2GB` |
| Prompt enhancer | `64GB+` |

Practical recommendations:

- Start with a 1.3B 540P checkpoint before wiring 14B or 720P models.
- Load the VAE in `torch.float32` and the pipeline in `torch.bfloat16`, as the
  official Diffusers examples do.
- Use `enable_model_cpu_offload()` when peak VRAM is tight. It is slower but
  matches the source offload order.
- Lower `base_num_frames` to reduce peak VRAM for long videos. The original
  notes mention values like `77` or `57`, with a possible quality reduction.
- Lower `num_frames`, resolution, and `num_inference_steps` for smoke tests.
- Avoid `num_videos_per_prompt > 1` until a single generation fits reliably.
- Use `output_type="latent"` only for internal chaining. For normal export,
  keep `output_type="np"` or use a PIL-compatible output.
- Keep `export_to_video(..., fps=24)` aligned with the pipeline `fps` value on
  DF pipelines unless there is a deliberate playback reason to differ.

The original repository also documents xDiT USP for multi-GPU inference in its
own scripts. That is not the same as a one-line Diffusers pipeline option, so
treat it as a separate distributed inference path rather than a default local
integration path.

## 16. Outputs

All SkyReels-V2 pipelines return `SkyReelsV2PipelineOutput` by default:

```python
output = pipe(...)
frames = output.frames[0]
```

`SkyReelsV2PipelineOutput.frames` may be:

- a nested list of PIL image sequences with shape-like structure
  `batch_size x num_frames`;
- a NumPy array;
- a Torch tensor shaped like
  `(batch_size, num_frames, channels, height, width)`.

The API page documents `output_type="np"` as the default. The source also has a
latent return branch when `output_type == "latent"`, in which case `.frames`
contains latents rather than decoded video frames.

Export examples:

```python
from diffusers.utils import export_to_video

export_to_video(output.frames[0], "video.mp4", fps=24, quality=8)
```

If `return_dict=False`, the pipelines return a tuple and the frames are the
first element:

```python
frames = pipe(prompt=prompt, return_dict=False)[0]
```

## 17. Gotchas

- Some Hugging Face model-card auto snippets use generic `DiffusionPipeline`
  examples and may refer to `.images[0]`. The SkyReels-V2 API examples and
  output class use `.frames[0]` for video output.
- The API docs inherit some copied wording such as "image generation" in
  parameter descriptions. These are video pipelines.
- `height` and `width` must be divisible by `16`; the documented dimensions
  already satisfy this.
- Long-video DF generation needs `overlap_history` when
  `num_frames > base_num_frames`; the source raises if it is missing.
- Async DF is slower than sync DF because the staggered schedule takes more
  total iterations.
- When using async DF, do not invent frame counts casually. The original notes
  warn that causal block divisibility matters in latent space.
- `addnoise_condition=20` is the recommended long-video smoothing value in the
  original notes. Values above `50` are discouraged there.
- Keep first and last frames at matching dimensions for first/last-frame runs.
- `guidance_scale > 1.0` enables classifier-free guidance. T2V examples use
  `6.0`; I2V and V2V examples use `5.0`.
- `max_sequence_length` defaults to `512` in pipeline calls. If prompts are
  long or generated by a prompt enhancer, watch for truncation or saturation.
- The model card license is `skywork-license`; check deployment and
  redistribution constraints before exposing the models in a product.

## 18. Implementation Checklist

For a local workflow backend:

1. Expose the pipeline class as an explicit model mode: `df_t2v`, `df_i2v`,
   `df_v2v`, `t2v`, or `i2v`.
2. Validate model ID compatibility with the selected mode before loading.
3. Normalize dimensions to a documented shape or to multiples of `16`.
4. Load VAE as `float32`; load the rest as `bfloat16`.
5. Set `UniPCMultistepScheduler.from_config(..., flow_shift=8.0)` for T2V and
   `flow_shift=5.0` for I2V/V2V.
6. For DF long-video runs, require `overlap_history` when
   `num_frames > base_num_frames`.
7. For async DF, require both `ar_step > 0` and `causal_block_size`, then check
   the latent-frame divisibility rules before dispatch.
8. Return `.frames` metadata consistently, including `fps`, `height`, `width`,
   `num_frames`, `base_num_frames`, and the selected model ID.
9. Stream progress through `callback_on_step_end` using `latents` if the local
   job system supports per-step events.
10. Export with `diffusers.utils.export_to_video` or a server-native video
    encoder, keeping output FPS aligned with the generation settings.

## 19. Source Links

- Hugging Face Diffusers API page:
  <https://huggingface.co/docs/diffusers/api/pipelines/skyreels_v2>
- Diffusers docs source:
  <https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/skyreels_v2.md>
- DF T2V source:
  <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing.py>
- DF I2V source:
  <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_i2v.py>
- DF V2V source:
  <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_v2v.py>
- Regular T2V source:
  <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2.py>
- Regular I2V source:
  <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_i2v.py>
- Output dataclass source:
  <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/skyreels_v2/pipeline_output.py>
- Skywork 1.3B DF model card:
  <https://huggingface.co/Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers>
- Skywork 14B DF 720P model card:
  <https://huggingface.co/Skywork/SkyReels-V2-DF-14B-720P-Diffusers>
- Skywork 14B T2V 720P model card:
  <https://huggingface.co/Skywork/SkyReels-V2-T2V-14B-720P-Diffusers>
- Skywork 14B I2V 720P model card:
  <https://huggingface.co/Skywork/SkyReels-V2-I2V-14B-720P-Diffusers>
- Original Skywork repository:
  <https://github.com/SkyworkAI/SkyReels-V2>
