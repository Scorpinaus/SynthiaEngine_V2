# Wan Diffusers Implementation Guide

Last checked: 2026-06-18 against the Hugging Face Diffusers Wan API page,
the linked Diffusers docs source, the linked `v0.38.0` pipeline source files,
and the official Wan model cards linked from the docs.

Research target:
https://huggingface.co/docs/diffusers/api/pipelines/wan

Primary Diffusers classes:

| Class | Role |
| --- | --- |
| `WanPipeline` | Text-to-video pipeline for Wan T2V checkpoints. Also covers Wan 2.2 two-denoiser checkpoints through `transformer_2`, `boundary_ratio`, and `guidance_scale_2`. |
| `WanImageToVideoPipeline` | Image-to-video pipeline. It supports single starting-frame conditioning and first-last-frame generation with `last_image` when the checkpoint supports it. |
| `WanVACEPipeline` | Controllable any-to-video pipeline for VACE tasks such as control-to-video, image/video-to-video, inpainting, outpainting, subject-to-video, and composition workflows. |
| `WanVideoToVideoPipeline` | Video-to-video/editing pipeline that starts from an input video and uses `strength` to control how far generation drifts from it. |
| `WanAnimatePipeline` | Wan-Animate pipeline for character animation and character replacement from a character image plus preprocessed pose and face videos. |
| `WanPipelineOutput` | Shared output container. Read generated clips from `.frames`. |

Related components and helpers:

| Class or helper | Role |
| --- | --- |
| `AutoencoderKLWan` | Wan video VAE. Use `torch.float32` for decode quality when memory allows. Supports `from_single_file()` in the Wan docs. |
| `WanTransformer3DModel` | Base Wan diffusion transformer for T2V, I2V, and V2V pipelines. Supports `from_single_file()` in the Wan docs. |
| `WanVACETransformer3DModel` | VACE control transformer used by `WanVACEPipeline`. |
| `WanAnimateTransformer3DModel` | Motion/control transformer used by `WanAnimatePipeline`. |
| `WanAnimateImageProcessor` | Wan-Animate image/mask preprocessing helper used internally by the pipeline. |
| `encode_prompt()` | Shared public helper on the pipelines for precomputing UMT5 prompt and negative prompt embeddings. |
| `pad_video_frames()` | Wan-Animate helper that pads frame lists with a reflect-like strategy. |
| `load_lora_weights()` | LoRA loader inherited through `WanLoraLoaderMixin` on the Wan pipelines. |

This is a docs-only implementation reference. It does not change SynthaEngine
runtime behavior.

## 1. Executive Summary

Wan is the Wan Team's open video foundation model family. The Diffusers Wan
page covers prompt-only text-to-video, image-to-video, first-last-frame
image-to-video, controllable any-to-video through VACE, video-to-video editing,
and character animation/replacement through Wan-Animate.

Practical integration answer:

| Need | Start with |
| --- | --- |
| Text-to-video | `WanPipeline` with `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` for a smaller smoke test or `Wan-AI/Wan2.1-T2V-14B-Diffusers` / `Wan-AI/Wan2.2-T2V-A14B-Diffusers` for heavier quality runs. |
| Image-to-video | `WanImageToVideoPipeline` with `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`, `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers`, or `Wan-AI/Wan2.2-I2V-A14B-Diffusers`. |
| First-last-frame I2V | `WanImageToVideoPipeline` with `Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers`, passing `image` and `last_image`. |
| Controllable generation | `WanVACEPipeline` with `Wan-AI/Wan2.1-VACE-1.3B-diffusers` or `Wan-AI/Wan2.1-VACE-14B-diffusers`. |
| Video-to-video | `WanVideoToVideoPipeline`; the docs load a T2V checkpoint and pass an input `video` plus `strength`. |
| Character animation/replacement | `WanAnimatePipeline` with `Wan-AI/Wan2.2-Animate-14B-Diffusers`; inputs must be preprocessed pose and face videos, not raw reference videos. |

Recommended defaults for first integration:

- Use `torch_dtype=torch.bfloat16` for the pipeline and transformer/text
  encoder, but load or upcast `AutoencoderKLWan` to `torch.float32` for decode
  quality when possible.
- Start at `height=480`, `width=832`, `num_frames=81`, `num_inference_steps=30`
  to `50`, and `guidance_scale=5.0` for T2V/I2V/VACE.
- Use `num_frames = 4 * k + 1` values such as `81`. The docs call out this
  frame rule, and the source rounds frame counts to the nearest compatible
  value when needed.
- Set the scheduler with `UniPCMultistepScheduler.from_config(...,
  flow_shift=...)` when following the docs examples. The Wan notes recommend
  lower shift values, about `2.0` to `5.0`, for lower-resolution videos and
  higher shift values, about `7.0` to `12.0`, for higher-resolution outputs.
- Read the result from `.frames[0]`, then call
  `diffusers.utils.export_to_video(frames, "output.mp4", fps=16)` for most Wan
  2.1 examples or `fps=30` for Wan-Animate examples.

## 2. Official Entry Points

- Pipeline docs: <https://huggingface.co/docs/diffusers/api/pipelines/wan>
- Docs source: <https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/wan.md>
- Wan T2V source: <https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/wan/pipeline_wan.py>
- Wan I2V source: <https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/wan/pipeline_wan_i2v.py>
- Wan VACE source: <https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/wan/pipeline_wan_vace.py>
- Wan V2V source: <https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/wan/pipeline_wan_video2video.py>
- Wan-Animate source: <https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/wan/pipeline_wan_animate.py>
- Wan output source: <https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/wan/pipeline_output.py>
- Wan image processor source: <https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/wan/image_processor.py>
- Diffusers quantization overview: <https://huggingface.co/docs/diffusers/quantization/overview>
- Diffusers quantization API: <https://huggingface.co/docs/diffusers/api/quantization>
- Wan-AI organization: <https://huggingface.co/Wan-AI>
- Original Wan 2.1 code: <https://github.com/Wan-Video/Wan2.1>
- Original Wan 2.2 code: <https://github.com/Wan-Video/Wan2.2>

## 3. Checkpoints And Model IDs

The Diffusers Wan page currently lists these supported models:

| Model ID | Pipeline | Task | Notes |
| --- | --- | --- | --- |
| `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | `WanPipeline` | Text-to-video | Smaller Wan 2.1 model. The model card says the original 1.3B model targets consumer GPUs and recommends 480p for stability. Good first local smoke test. |
| `Wan-AI/Wan2.1-T2V-14B-Diffusers` | `WanPipeline` | Text-to-video | Larger Wan 2.1 T2V model. The docs memory example says the 14B text-to-video setup can run around 13 GB VRAM with grouped offloading. |
| `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` | `WanImageToVideoPipeline` | Image-to-video | 480p image-to-video checkpoint. |
| `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` | `WanImageToVideoPipeline` | Image-to-video | 720p image-to-video checkpoint. |
| `Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers` | `WanImageToVideoPipeline` | First-last-frame to video | Use `image` for the first frame and `last_image` for the last frame. Note the lowercase `diffusers` suffix in the docs example. |
| `Wan-AI/Wan2.1-VACE-1.3B-diffusers` | `WanVACEPipeline` | Controllable any-to-video | Smaller VACE checkpoint. Good first VACE integration target. |
| `Wan-AI/Wan2.1-VACE-14B-diffusers` | `WanVACEPipeline` | Controllable any-to-video | Larger VACE checkpoint. |
| `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | `WanPipeline` | Text-to-video | Wan 2.2 A14B T2V model. The model card describes Wan 2.2 as a MoE upgrade and the T2V A14B checkpoint as supporting 5-second clips at 480p and 720p. |
| `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | `WanImageToVideoPipeline` | Image-to-video | Wan 2.2 A14B I2V model for 480p and 720p image-to-video. |
| `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | `WanPipeline` or auto-selected Diffusers pipeline | Hybrid text/image-to-video | The model card describes this as a 5B hybrid TI2V model using Wan 2.2 VAE compression `16x16x4`, supporting text-to-video and image-to-video at 720p and 24 fps. Test the auto-selected pipeline class before hard-coding a class. |
| `Wan-AI/Wan2.2-Animate-14B-Diffusers` | `WanAnimatePipeline` | Character animation/replacement | Wan-Animate checkpoint. The model card's README is sparse, so follow the Diffusers API page and source examples. |

Do not rely on generic Hub "Use this model" snippets for video output shape.
Several Hub pages show `DiffusionPipeline` examples with `.images[0]`; the Wan
pipeline docs and source return `WanPipelineOutput(frames=...)`, so integration
code should read `.frames[0]`.

## 4. Installation

Use a recent Diffusers release that includes the Wan classes. The API page
links to `v0.38.0` source, and newer Wan 2.2 behavior may require current
Diffusers if your installed package predates the linked docs.

```powershell
.venv\Scripts\python.exe -m pip install -U diffusers transformers accelerate torch ftfy imageio imageio-ffmpeg
```

For quantization and memory experiments:

```powershell
.venv\Scripts\python.exe -m pip install -U bitsandbytes
```

For control-video preprocessing in VACE workflows, the Wan docs recommend the
`huggingface/controlnet_aux` library for deriving control videos such as depth,
pose, sketch, flow, grayscale, scribble, layout, and bounding-box signals.

For Wan-Animate, Diffusers expects preprocessed `pose_video` and `face_video`
inputs. The docs point to the original Wan-Animate repository for those
preprocessing scripts and say Diffusers integration of preprocessing is planned
for a future release.

## 5. Shared Components And Runtime Shape

Most Wan pipelines share the same high-level structure:

| Component | Typical class | Purpose |
| --- | --- | --- |
| `tokenizer` | `AutoTokenizer` / T5 tokenizer | Tokenizes prompts for UMT5. The docs refer to the `google/umt5-xxl` family. |
| `text_encoder` | `UMT5EncoderModel` | Encodes prompt and negative prompt text. |
| `vae` | `AutoencoderKLWan` | Encodes and decodes video frames to and from latent videos. |
| `transformer` | `WanTransformer3DModel`, `WanVACETransformer3DModel`, or `WanAnimateTransformer3DModel` | Denoises latent video tensors. |
| `transformer_2` | Optional second transformer for `WanPipeline`, `WanImageToVideoPipeline`, and `WanVACEPipeline` | Wan 2.2 low-noise-stage denoiser. Used with `boundary_ratio`. |
| `scheduler` | `FlowMatchEulerDiscreteScheduler` in signatures; examples often replace with `UniPCMultistepScheduler` | Controls timesteps and denoising updates. |
| `image_encoder` | `CLIPVisionModel` for I2V and Animate | Encodes conditioning images. |
| `image_processor` | `CLIPImageProcessor` for image-conditioned pipelines | Preprocesses images before CLIP encoding. |
| `video_processor` | Diffusers `VideoProcessor` internally | Converts decoded frame tensors to the requested `output_type`. |

Source-level behavior to account for:

- Wan pipelines inherit `DiffusionPipeline`, so they support common methods such
  as `from_pretrained()`, `.to(...)`, `.enable_model_cpu_offload()`,
  `.save_pretrained()`, callback hooks, and `return_dict=False`.
- The pipeline classes also inherit `WanLoraLoaderMixin`, so
  `load_lora_weights()` is available on the documented Wan pipelines.
- Prompt strings are cleaned before UMT5 encoding. The docs examples install
  `ftfy`; install it to match the official path.
- `max_sequence_length` defaults to `512` in pipeline calls, while helper
  docstrings show `226` in `encode_prompt()`. Keep it explicit if prompt length
  is important.
- `height` and `width` should be compatible with both the VAE spatial scale and
  the transformer's patch size. The docs examples compute a `mod_value` as
  `pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]` when
  resizing images.
- The source checks divisibility by `16` and then adjusts dimensions down to
  the exact patch multiple if needed. Avoid relying on silent adjustment in a
  server; precompute valid dimensions and log them.
- Video frame counts should follow `4 * k + 1`. Common examples use `81`.
- `negative_prompt` is ignored when classifier-free guidance is disabled
  (`guidance_scale <= 1`).
- Callback tensor inputs are limited to pipeline-declared names such as
  `latents`, `prompt_embeds`, and `negative_prompt_embeds`.

## 6. Output Handling

All documented Wan pipelines return `WanPipelineOutput` by default.

`WanPipelineOutput.frames` can be:

- a nested list shaped like `[batch_size][num_frames]` containing PIL image
  frames,
- a NumPy array,
- or a Torch tensor with shape `(batch_size, num_frames, channels, height,
  width)`.

Most docs examples use the default `output_type="np"` and then read the first
clip with:

```python
frames = pipe(...).frames[0]
```

Export with:

```python
from diffusers.utils import export_to_video

export_to_video(frames, "wan_output.mp4", fps=16)
```

For Wan-Animate examples, use `fps=30` unless the source video or product
requirements specify otherwise.

If `return_dict=False`, Diffusers returns a tuple. For a server integration,
prefer the default output class because `.frames` is explicit and survives API
extension better than tuple indexing.

## 7. `WanPipeline` For Text-To-Video

Use `WanPipeline` when the job input is prompt-only text-to-video.

Constructor signature documented by Diffusers:

```python
WanPipeline(
    tokenizer,
    text_encoder,
    vae,
    scheduler,
    transformer=None,
    transformer_2=None,
    boundary_ratio=None,
    expand_timesteps=False,
)
```

Important call parameters:

| Parameter | Notes |
| --- | --- |
| `prompt` | String or list of strings. Pass `prompt_embeds` instead if the server precomputes UMT5 embeddings. |
| `negative_prompt` | Used only when `guidance_scale > 1`. Must match prompt type and batch size when batched. |
| `height`, `width` | Defaults are `480x832`. Use dimensions divisible by the VAE/patch multiple. |
| `num_frames` | Default is `81`. Use `4 * k + 1` values. |
| `num_inference_steps` | Default is `50`. Lower to `20` to `30` for smoke tests, increase for quality. |
| `guidance_scale` | Default is `5.0`. Higher values follow the prompt more strongly but can reduce image quality. |
| `guidance_scale_2` | Guidance for `transformer_2` in two-stage Wan 2.2 denoising. If omitted and `boundary_ratio` is set, it follows `guidance_scale`. |
| `num_videos_per_prompt` | Usually `1` for server jobs. Larger values multiply memory. |
| `generator` | Use `torch.Generator(device="cuda").manual_seed(seed)` for deterministic jobs. |
| `latents` | Optional pregenerated latents for reproducible prompt sweeps. Shape must match latent video dimensions. |
| `output_type` | Defaults to `"np"`. Use a supported Diffusers output type and export accordingly. |
| `callback_on_step_end` | Useful for cancellation/progress in a long-running local server. |

Minimal text-to-video example:

```python
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

vae = AutoencoderKLWan.from_pretrained(
    model_id,
    subfolder="vae",
    torch_dtype=torch.float32,
)

pipe = WanPipeline.from_pretrained(
    model_id,
    vae=vae,
    torch_dtype=torch.bfloat16,
)

pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config,
    flow_shift=3.0,
)
pipe.to("cuda")

prompt = (
    "A slow cinematic tracking shot through a rainy neon market at night, "
    "with reflections on the pavement and gentle handheld camera motion."
)
negative_prompt = "blur, low quality, static, warped faces, subtitles"

frames = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    num_frames=81,
    num_inference_steps=30,
    guidance_scale=5.0,
).frames[0]

export_to_video(frames, "wan_t2v.mp4", fps=16)
```

Wan 2.2 two-denoiser note:

```python
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    torch_dtype=torch.bfloat16,
    boundary_ratio=0.9,
)

frames = pipe(
    prompt=prompt,
    guidance_scale=4.5,
    guidance_scale_2=3.5,
).frames[0]
```

Use `boundary_ratio` only when the checkpoint has both `transformer` and
`transformer_2`. The docs describe `transformer` as the high-noise denoiser and
`transformer_2` as the low-noise denoiser; `boundary_ratio` selects the switch
timestep as a ratio of training timesteps.

## 8. `WanImageToVideoPipeline` For I2V And First-Last-Frame I2V

Use `WanImageToVideoPipeline` when generation is conditioned by an image.

Constructor signature documented by Diffusers:

```python
WanImageToVideoPipeline(
    tokenizer,
    text_encoder,
    vae,
    scheduler,
    image_processor=None,
    image_encoder=None,
    transformer=None,
    transformer_2=None,
    boundary_ratio=None,
    expand_timesteps=False,
)
```

Additional call parameters beyond T2V:

| Parameter | Notes |
| --- | --- |
| `image` | Required unless `image_embeds` are provided. Accepts PIL images, NumPy arrays, Torch tensors, or lists. |
| `last_image` | Optional last-frame conditioning for first-last-frame checkpoints. Resize/crop it to match `image`. |
| `image_embeds` | Optional precomputed CLIP image embeddings. |
| `height`, `width` | Should match the resized conditioning image. |
| `guidance_scale_2`, `transformer_2`, `boundary_ratio` | Same two-denoiser concept as `WanPipeline` for supported Wan 2.2 checkpoints. |

Standard image-to-video example:

```python
import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel

model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

image_encoder = CLIPVisionModel.from_pretrained(
    model_id,
    subfolder="image_encoder",
    torch_dtype=torch.float32,
)
vae = AutoencoderKLWan.from_pretrained(
    model_id,
    subfolder="vae",
    torch_dtype=torch.float32,
)

pipe = WanImageToVideoPipeline.from_pretrained(
    model_id,
    image_encoder=image_encoder,
    vae=vae,
    torch_dtype=torch.bfloat16,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config,
    flow_shift=3.0,
)
pipe.to("cuda")

image = load_image("input.png")
max_area = 480 * 832
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
image = image.resize((width, height))

frames = pipe(
    image=image,
    prompt="A cinematic shot where the subject begins moving naturally.",
    negative_prompt="blur, low quality, unstable motion, subtitles",
    height=height,
    width=width,
    num_frames=81,
    num_inference_steps=30,
    guidance_scale=5.0,
).frames[0]

export_to_video(frames, "wan_i2v.mp4", fps=16)
```

First-last-frame example:

```python
import numpy as np
import torch
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel

model_id = "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"

image_encoder = CLIPVisionModel.from_pretrained(
    model_id,
    subfolder="image_encoder",
    torch_dtype=torch.float32,
)
vae = AutoencoderKLWan.from_pretrained(
    model_id,
    subfolder="vae",
    torch_dtype=torch.float32,
)
pipe = WanImageToVideoPipeline.from_pretrained(
    model_id,
    image_encoder=image_encoder,
    vae=vae,
    torch_dtype=torch.bfloat16,
).to("cuda")

first_frame = load_image("first.png")
last_frame = load_image("last.png")

max_area = 720 * 1280
aspect_ratio = first_frame.height / first_frame.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
first_frame = first_frame.resize((width, height))

if last_frame.size != first_frame.size:
    resize_ratio = max(width / last_frame.width, height / last_frame.height)
    resized = last_frame.resize(
        (round(last_frame.width * resize_ratio), round(last_frame.height * resize_ratio))
    )
    last_frame = TF.center_crop(resized, [height, width])

frames = pipe(
    image=first_frame,
    last_image=last_frame,
    prompt="A smooth transition between the two frames with natural camera motion.",
    height=height,
    width=width,
    guidance_scale=5.5,
).frames[0]

export_to_video(frames, "wan_flf2v.mp4", fps=16)
```

I2V gotchas:

- Load `CLIPVisionModel` in `torch.float32` when following the docs.
- Keep first and last frames the same dimensions.
- Preserve aspect ratio, then snap dimensions down to the VAE/patch multiple.
- `last_image` is not a generic interpolation feature for every I2V checkpoint;
  use it with first-last-frame models that were trained for it.
- For Wan 2.2 I2V, check whether the checkpoint has a second denoiser and pass
  `boundary_ratio` only when the model index provides one.

## 9. `WanVACEPipeline` For Controllable Any-To-Video

Use `WanVACEPipeline` for controllable generation. The docs describe VACE as
supporting:

- control-to-video from signals such as depth, pose, sketch, flow, grayscale,
  scribble, layout, and boundary boxes,
- image/video-to-video from a first frame, last frame, starting clip, ending
  clip, or random clips,
- inpainting and outpainting,
- subject-to-video with faces, objects, or characters,
- composition workflows such as reference-anything, animate-anything,
  swap-anything, expand-anything, and move-anything.

Constructor signature documented by Diffusers:

```python
WanVACEPipeline(
    tokenizer,
    text_encoder,
    vae,
    scheduler,
    transformer=None,
    transformer_2=None,
    boundary_ratio=None,
)
```

Additional call parameters:

| Parameter | Notes |
| --- | --- |
| `video` | Optional conditioning video. The docs say VACE currently supports generating one video at a time. |
| `mask` | Optional mask video. Black regions are conditioning/preserved regions; white regions are generated. |
| `reference_images` | Optional one or more images for subject/composition conditioning, such as a new character reference for inpainting. |
| `conditioning_scale` | Float, list, or tensor controlling the control stream. If a list/tensor is used, its length should match `len(transformer.config.vace_layers)`. |
| `prompt`, `negative_prompt`, `height`, `width`, `num_frames`, `guidance_scale` | Same general meaning as the T2V pipeline. |

The VACE mask convention is the single most important integration detail:

| Mask color | Meaning |
| --- | --- |
| Black | Condition on this region/frame. The model should preserve/use it rather than generate new content there. |
| White | Generate this region/frame. |

First-last-frame VACE example:

```python
import PIL.Image
import torch
from diffusers import AutoencoderKLWan, WanVACEPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image

def make_flf_conditioning(first, last, height, width, num_frames):
    first = first.resize((width, height))
    last = last.resize((width, height))

    hold = PIL.Image.new("RGB", (width, height), (128, 128, 128))
    video = [first, *([hold] * (num_frames - 2)), last]

    black = PIL.Image.new("L", (width, height), 0)
    white = PIL.Image.new("L", (width, height), 255)
    mask = [black, *([white] * (num_frames - 2)), black]
    return video, mask

model_id = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
vae = AutoencoderKLWan.from_pretrained(
    model_id,
    subfolder="vae",
    torch_dtype=torch.float32,
)
pipe = WanVACEPipeline.from_pretrained(
    model_id,
    vae=vae,
    torch_dtype=torch.bfloat16,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config,
    flow_shift=3.0,
)
pipe.to("cuda")

height = 512
width = 512
num_frames = 81
first = load_image("first.png")
last = load_image("last.png")
video, mask = make_flf_conditioning(first, last, height, width, num_frames)

frames = pipe(
    video=video,
    mask=mask,
    prompt="A small bird launches into flight between the two keyframes.",
    negative_prompt="blur, low quality, static, subtitles",
    height=height,
    width=width,
    num_frames=num_frames,
    num_inference_steps=30,
    guidance_scale=5.0,
    conditioning_scale=1.0,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

export_to_video(frames, "wan_vace.mp4", fps=16)
```

VACE integration notes:

- Prepare control videos before calling the pipeline. For pose/depth/etc.,
  generate the control signal into a frame list with the same length and
  dimensions expected by the output.
- Keep `video`, `mask`, and output dimensions aligned.
- Prefer explicit `height`, `width`, and `num_frames`; do not infer them from
  arbitrary user uploads without snapping to supported multiples.
- When using `reference_images`, treat them as conditioning identity or subject
  hints, not as output frames.
- If using per-layer `conditioning_scale`, validate length against
  `transformer.config.vace_layers` at load time.

## 10. `WanVideoToVideoPipeline` For Video Editing

Use `WanVideoToVideoPipeline` when there is a source video and the prompt
should transform it rather than create a clip from scratch.

Constructor signature documented by Diffusers:

```python
WanVideoToVideoPipeline(
    tokenizer,
    text_encoder,
    transformer,
    vae,
    scheduler,
)
```

Important call parameters:

| Parameter | Notes |
| --- | --- |
| `video` | Input video as a frame list. Use `diffusers.utils.load_video()` or decode frames yourself. |
| `prompt` | Edit/generation instruction. |
| `negative_prompt` | Same guidance rules as T2V. |
| `height`, `width` | Output dimensions. Resize/crop the source video frames consistently. |
| `num_inference_steps` | Default is `50`. |
| `timesteps` | Optional custom scheduler timesteps. |
| `guidance_scale` | Default is `5.0`. |
| `strength` | Default is `0.8`. Higher values allow more change from the input video. Lower values preserve more source content. |

Example:

```python
import torch
from diffusers import AutoencoderKLWan, WanVideoToVideoPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_video

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(
    model_id,
    subfolder="vae",
    torch_dtype=torch.float32,
)
pipe = WanVideoToVideoPipeline.from_pretrained(
    model_id,
    vae=vae,
    torch_dtype=torch.bfloat16,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config,
    flow_shift=3.0,
)
pipe.to("cuda")

video = load_video("input.mp4")

frames = pipe(
    video=video,
    prompt="A robot hiking across a mountain ridge at sunset.",
    negative_prompt="blur, low quality, distorted anatomy, subtitles",
    height=480,
    width=720,
    guidance_scale=5.0,
    strength=0.7,
    num_inference_steps=30,
).frames[0]

export_to_video(frames, "wan_v2v.mp4", fps=16)
```

V2V gotchas:

- `strength` is the main edit-preservation knob. Start around `0.6` to `0.8`.
- Make frame count and dimensions compatible before inference.
- Use a small clip for the first pass; source-video length multiplies VAE and
  transformer memory cost.
- The docs example uses a T2V checkpoint for V2V. Pin and test the model ID
  rather than assuming every Wan checkpoint supports every video-edit behavior.

## 11. `WanAnimatePipeline` For Character Animation And Replacement

Use `WanAnimatePipeline` for Wan-Animate character workflows.

The docs describe two modes:

| Mode | Purpose | Required inputs |
| --- | --- | --- |
| `"animate"` | Animate the character image using motion and expression from the reference controls. | `image`, `pose_video`, `face_video`, `prompt`. |
| `"replace"` | Replace a character in a background video while preserving the scene. | `image`, `pose_video`, `face_video`, `background_video`, `mask_video`, `prompt`. |

Constructor signature documented by Diffusers:

```python
WanAnimatePipeline(
    tokenizer,
    text_encoder,
    vae,
    scheduler,
    image_processor,
    image_encoder,
    transformer,
)
```

Wan-Animate call parameters:

| Parameter | Notes |
| --- | --- |
| `image` | Character image. Required unless `image_embeds` are provided. |
| `pose_video` | Preprocessed pose/keypoint video as a list of PIL frames. Required. |
| `face_video` | Preprocessed facial feature video as a list of PIL frames. Required. |
| `background_video` | Required only for `mode="replace"`. |
| `mask_video` | Required only for `mode="replace"`. Black preserves background; white generates replacement content. |
| `prompt`, `negative_prompt` | Text guidance. CFG is off by default because `guidance_scale=1.0`. |
| `height`, `width` | Defaults are `720x1280`. Resize the character image and control videos consistently. |
| `segment_frame_length` | Default is `77`; generated in segments until reaching the pose-video length. The docs say values should generally follow `4 * k + 1`. |
| `prev_segment_conditioning_frames` | Recommended values are `1` or `5`; `5` improves temporal consistency but uses more memory. |
| `motion_encode_batch_size` | Lower it to trade speed for lower memory when encoding the face video. |
| `guidance_scale` | Default is `1.0`, so CFG is disabled. Higher values enable prompt/negative-prompt influence, mainly targeting text and face conditioning. |

Animation mode:

```python
import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanAnimatePipeline
from diffusers.utils import export_to_video, load_image, load_video

model_id = "Wan-AI/Wan2.2-Animate-14B-Diffusers"

vae = AutoencoderKLWan.from_pretrained(
    model_id,
    subfolder="vae",
    torch_dtype=torch.float32,
)
pipe = WanAnimatePipeline.from_pretrained(
    model_id,
    vae=vae,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

image = load_image("character.png")
pose_video = load_video("pose_preprocessed.mp4")
face_video = load_video("face_preprocessed.mp4")

max_area = 720 * 1280
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
image = image.resize((width, height))

frames = pipe(
    image=image,
    pose_video=pose_video,
    face_video=face_video,
    prompt="A person dancing in a studio with cinematic lighting.",
    negative_prompt="blur, low quality, distorted face, static",
    height=height,
    width=width,
    segment_frame_length=77,
    prev_segment_conditioning_frames=1,
    guidance_scale=1.0,
    num_inference_steps=20,
    mode="animate",
).frames[0]

export_to_video(frames, "wan_animate.mp4", fps=30)
```

Replacement mode:

```python
background_video = load_video("background.mp4")
mask_video = load_video("mask.mp4")

frames = pipe(
    image=image,
    pose_video=pose_video,
    face_video=face_video,
    background_video=background_video,
    mask_video=mask_video,
    prompt="The replacement character matches the scene lighting.",
    height=height,
    width=width,
    segment_frame_length=77,
    guidance_scale=1.0,
    num_inference_steps=20,
    mode="replace",
).frames[0]

export_to_video(frames, "wan_replace.mp4", fps=30)
```

Wan-Animate gotchas:

- `pose_video` and `face_video` must be preprocessed control videos. Raw camera
  footage is not a valid substitute.
- `pose_video`, `face_video`, `background_video`, and `mask_video` should be
  lists of PIL frames. Decode paths or arrays before calling the pipeline.
- `mode` must be either `"animate"` or `"replace"`.
- In replacement mode, provide both `background_video` and `mask_video`.
- `prev_segment_conditioning_frames` is constrained to `1` or `5` in the
  source validation.
- If the source clip length is not compatible with segment generation, use
  `pad_video_frames()` or pre-trim/pad the frame list before inference.

## 12. Schedulers, `flow_shift`, And Wan 2.2 Two-Stage Denoising

The Wan docs are slightly easy to misread because the constructor signatures
show `FlowMatchEulerDiscreteScheduler`, while the examples often replace the
scheduler with `UniPCMultistepScheduler.from_config(...)` and a `flow_shift`.
For implementation, follow the checkpoint docs and pin the scheduler choice in
configuration instead of relying on defaults.

Typical scheduler replacement:

```python
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config,
    flow_shift=3.0,
)
```

Recommended shift heuristics from the Wan notes:

| Output target | `flow_shift` starting point |
| --- | --- |
| Lower resolution, such as 480p | `2.0` to `5.0` |
| Higher resolution, such as 720p | `7.0` to `12.0` |

Examples in the docs use:

- `flow_shift=3.0` for 480p examples,
- `flow_shift=5.0` in a 720p T2V example and a LoRA example.

For Wan 2.2 models with two denoisers:

- `transformer` handles high-noise timesteps.
- `transformer_2` handles low-noise timesteps.
- `boundary_ratio` selects the switch point.
- `guidance_scale_2` controls low-noise-stage guidance; if omitted while
  `boundary_ratio` is active, it falls back to `guidance_scale`.
- The Wan notes say LoRAs load into the first denoiser by default and require
  `load_into_transformer_2=True` to load into the second denoiser.

## 13. LoRA

The Wan docs say Wan 2.1 supports LoRAs with `load_lora_weights()`, and the
pipeline API page also lists `load_lora_weights()` for Wan-Animate. The
pipelines inherit `WanLoraLoaderMixin`, so use the standard Diffusers adapter
flow:

```python
pipe.load_lora_weights(
    "benjamin-paine/steamboat-willie-1.3b",
    adapter_name="steamboat-willie",
)
pipe.set_adapters("steamboat-willie")
```

For adapter strength:

```python
pipe.set_adapters("steamboat-willie", adapter_weights=0.8)
```

Wan 2.2 caveat:

```python
pipe.load_lora_weights(
    "path/or/repo",
    adapter_name="speedup-or-style",
    load_into_transformer_2=True,
)
```

Only pass `load_into_transformer_2=True` when the pipeline actually has
`transformer_2`. For Wan 2.2, test outputs with the LoRA loaded into one or
both denoisers because speedup/style LoRAs may be trained for a specific stage.

The Wan notes also mention LightX2V LoRAs for speeding up Wan 2.1 and Wan 2.2.
Treat those as checkpoint-specific accelerators: pin the LoRA repo, scheduler
settings, step count, and target Wan base model together.

## 14. Memory, Performance, And Quantization

Wan checkpoints are large video DiTs. A server should make memory strategy an
explicit runtime option rather than an incidental snippet detail.

Official Wan page memory facts:

- The Wan 2.1 T2V 1.3B model is advertised by the model card as requiring
  about 8.19 GB VRAM in the original project context.
- The Diffusers Wan T2V memory example says the Wan 2.1 14B T2V model can run
  around 13 GB VRAM with dtype splitting and grouped offloading.
- The Wan page recommends the general Diffusers "Reduce memory usage" guide for
  memory-saving techniques.

Recommended loading strategy:

```python
import torch
from diffusers import AutoModel, WanPipeline
from diffusers.hooks.group_offloading import apply_group_offloading
from transformers import UMT5EncoderModel

model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"

text_encoder = UMT5EncoderModel.from_pretrained(
    model_id,
    subfolder="text_encoder",
    torch_dtype=torch.bfloat16,
)
vae = AutoModel.from_pretrained(
    model_id,
    subfolder="vae",
    torch_dtype=torch.float32,
)
transformer = AutoModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)

apply_group_offloading(
    text_encoder,
    onload_device=torch.device("cuda"),
    offload_device=torch.device("cpu"),
    offload_type="block_level",
    num_blocks_per_group=4,
)
transformer.enable_group_offload(
    onload_device=torch.device("cuda"),
    offload_device=torch.device("cpu"),
    offload_type="leaf_level",
    use_stream=True,
)

pipe = WanPipeline.from_pretrained(
    model_id,
    text_encoder=text_encoder,
    transformer=transformer,
    vae=vae,
    torch_dtype=torch.bfloat16,
).to("cuda")
```

Other memory knobs:

| Technique | Use |
| --- | --- |
| `pipe.enable_model_cpu_offload()` | Easiest general-purpose offload path. Slower than keeping everything resident. |
| Group offloading | Better fit for video models because compute can overlap transfer more effectively. The Wan docs use it for the 14B T2V example. |
| VAE `torch.float32` | Better decode quality, higher memory. The Wan notes explicitly recommend `AutoencoderKLWan` in float32 for better decoding. |
| Lower resolution | Reduces latent spatial size and attention cost. Snap dimensions to VAE/patch multiples. |
| Lower `num_frames` | Reduces temporal latent size. Keep `4 * k + 1`. |
| Lower `num_inference_steps` | Reduces latency. Test quality floor for each model. |
| `motion_encode_batch_size` | Wan-Animate-specific memory/speed tradeoff for face-video motion encoding. |
| `prev_segment_conditioning_frames=1` | Wan-Animate lower-memory option. Use `5` only when temporal consistency needs it. |

Pipeline-level quantization:

The Wan page imports `PipelineQuantizationConfig` in the memory example, and the
Diffusers quantization guide documents this pattern for on-the-fly pipeline
quantization:

```python
import torch
from diffusers import WanPipeline
from diffusers.quantizers import PipelineQuantizationConfig

quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
    },
    components_to_quantize="transformer",
)

pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
```

Quantization notes:

- Quantize the most expensive component first, usually `transformer`.
- Quantizing `text_encoder` can save memory but may alter prompt adherence;
  validate before enabling by default.
- Keep the VAE unquantized, and prefer float32 decode if output quality matters.
- `bitsandbytes_4bit`, `bitsandbytes_8bit`, `torchao`, `quanto`, and `gguf`
  are documented Diffusers quantization backends, but backend support varies by
  platform and model component.
- Combine quantization with offloading carefully. Measure both peak VRAM and
  latency; lower VRAM can cost throughput.

## 15. Single-File And Repackaged Weights

The Wan notes say `WanTransformer3DModel` and `AutoencoderKLWan` support
`from_single_file()`. This is useful for ComfyUI-style or repackaged assets.

Example shape:

```python
import torch
from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel

vae = AutoencoderKLWan.from_single_file(
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/vae/wan_2.1_vae.safetensors"
)
transformer = WanTransformer3DModel.from_single_file(
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors",
    torch_dtype=torch.bfloat16,
)

pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    vae=vae,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
```

Single-file gotchas:

- Keep the base Diffusers repo ID in `from_pretrained()` so tokenizer,
  scheduler, model index, and config files are still sourced from a known
  layout.
- Match the single-file transformer to the correct Wan family and task.
- Test Wan 2.2 single-file loads carefully because Wan 2.2 has different
  two-denoiser behavior from Wan 2.1.

## 16. Server Integration Checklist

For a local workflow server, model loading and validation should be explicit:

| Check | Recommendation |
| --- | --- |
| Pipeline selection | Map task to pipeline class and model ID; do not auto-guess from user prompt alone. |
| Model revision | Pin model ID and revision in runtime config for repeatability. |
| Dimensions | Compute `height` and `width` from target area and snap to VAE/patch multiples. |
| Frames | Validate `num_frames = 4 * k + 1`; default to `81`. |
| Dtypes | Use bfloat16 for transformer/text encoder and float32 VAE unless memory forces otherwise. |
| Scheduler | Store `flow_shift` per model/resolution profile. |
| Offload | Choose one offload policy per profile: resident, model CPU offload, or group offload. |
| Quantization | Make it opt-in and model-profile-specific. |
| Output | Always read `.frames`; never `.images`. |
| Export | Use `export_to_video()` with an explicit fps. |
| Cleanup | Release pipeline references or move models off device after jobs if the server is not keeping them warm. |

Validation for a first implementation:

1. Load the smallest intended checkpoint.
2. Run `num_frames=17`, `height=256`, `width=448` only if the checkpoint and
   patch multiples allow it; otherwise use the documented 480p dimensions.
3. Confirm `.frames[0]` is non-empty and frame dimensions match the exported
   video.
4. Repeat with the production resolution and frame count.
5. Record peak VRAM, latency, scheduler, dtype, offload, quantization, and
   model revision in logs.

## 17. Gotchas And Compatibility Notes

- The Wan docs page is on current Diffusers docs and links to `v0.38.0` source
  for the API signatures. If a local environment has an older Diffusers version,
  verify the classes exist before wiring runtime support.
- Generic Hub snippets may show `.images[0]`; Wan pipelines return `.frames`.
- The docs constructor signatures and scheduler prose are not perfectly
  uniform: signatures mention `FlowMatchEulerDiscreteScheduler`, while examples
  often use `UniPCMultistepScheduler` with `flow_shift`. Follow the model card
  and pipeline example for the checkpoint you pin.
- The docs mention "fps or `k` should be calculated by `4 * k + 1`." In
  practice this is the frame-count rule for common Wan examples, not the export
  fps value itself.
- `AutoencoderKLWan` in `torch.float32` improves decoding quality but costs
  memory.
- VACE and Animate masks use black for preserve/condition and white for
  generate.
- VACE currently supports one generated video at a time according to the
  parameter docs.
- Wan-Animate requires preprocessed pose and face videos. Raw reference video
  is insufficient.
- `guidance_scale=1.0` disables CFG; Wan-Animate uses this as the default.
- Wan 2.2 LoRA loading has a second-denoiser wrinkle. Use
  `load_into_transformer_2=True` only when the pipeline includes
  `transformer_2`.
- Height and width may be rounded down by source code to satisfy patchification.
  A server should compute valid dimensions upfront and surface the final values.
- Long prompts can be truncated by `max_sequence_length`. Keep prompt extension
  and truncation behavior explicit if prompt quality matters.

## 18. Minimal Decision Matrix

| User input shape | Recommended pipeline | Required payload fields |
| --- | --- | --- |
| Prompt only | `WanPipeline` | `prompt`, optional `negative_prompt`, dimensions, frames, seed. |
| Prompt plus start image | `WanImageToVideoPipeline` | `prompt`, `image`, dimensions, frames, seed. |
| Prompt plus first and last images | `WanImageToVideoPipeline` with FLF checkpoint | `prompt`, `image`, `last_image`, matched dimensions. |
| Prompt plus control video/mask/reference images | `WanVACEPipeline` | `prompt`, `video` and/or `mask` and/or `reference_images`, `conditioning_scale`. |
| Prompt plus source video edit | `WanVideoToVideoPipeline` | `prompt`, `video`, `strength`, dimensions. |
| Character image plus motion controls | `WanAnimatePipeline` | `image`, preprocessed `pose_video`, preprocessed `face_video`, `mode`. |
| Character replacement | `WanAnimatePipeline` | Animation inputs plus `background_video`, `mask_video`, `mode="replace"`. |

