# Text2Video-Zero Diffusers Implementation Guide

Last checked: 2026-06-18 against the Hugging Face Diffusers `main`
Text2Video-Zero API page, the `v0.33.1` stable page, and the current
Diffusers GitHub source.

Research target:
https://huggingface.co/docs/diffusers/main/en/api/pipelines/text_to_video_zero

Important version note: the current `main` documentation marks
Text2Video-Zero as deprecated. The pipeline can still be used, but Diffusers
no longer tests it or accepts changes for it. The current source places the
implementation under `src/diffusers/pipelines/deprecated/text_to_video_synthesis`
and both main pipeline classes declare `_last_supported_version = "0.33.1"`.
For production or reproducible local integration, pin Diffusers to `0.33.1`
unless you have tested the current `main` source build. The latest stable docs
page for `v0.38.0` no longer contains this pipeline and redirects readers back
to `main`.

Official entry points:

- Main docs: <https://huggingface.co/docs/diffusers/main/en/api/pipelines/text_to_video_zero>
- Last supported stable docs: <https://huggingface.co/docs/diffusers/v0.33.1/api/pipelines/text_to_video_zero>
- Current base source: <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deprecated/text_to_video_synthesis/pipeline_text_to_video_zero.py>
- Current SDXL source: <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deprecated/text_to_video_synthesis/pipeline_text_to_video_zero_sdxl.py>
- Last supported base source: <https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_zero.py>
- Last supported SDXL source: <https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_zero_sdxl.py>

## 1. What Text2Video-Zero Is

Text2Video-Zero is a zero-shot video method that turns an existing
text-to-image diffusion model, usually Stable Diffusion, into a short video
generator without video training, video-specific finetuning, or temporal
modules. Diffusers implements the paper's two key changes:

- It generates a first-frame latent trajectory, copies that latent state to
  later frames, and warps those later-frame latents with a simple translation
  motion field.
- It replaces frame-level self-attention with cross-frame attention so each
  generated frame attends to the first frame for identity and appearance
  consistency.

The official docs present three user-facing task families:

- Text-to-video from a prompt.
- Text-to-video with pose or edge guidance by combining Text2Video-Zero's
  attention processor with ControlNet.
- Video Instruct-Pix2Pix, where an existing video is edited frame-by-frame
  while the custom attention processor preserves temporal consistency.

This is useful when you want quick video experiments from existing Stable
Diffusion, SDXL, DreamBooth, or ControlNet assets. It is not a modern
high-fidelity video foundation model like CogVideoX, HunyuanVideo, Wan, LTX, or
AnimateDiff. Expect short clips, low FPS examples, simple camera/object motion,
and more manual tuning.

## 2. Pipeline And Variant Selection

| Surface | Official class or pattern | Best use | Notes |
| --- | --- | --- | --- |
| Prompt-only SD 1.x text-to-video | `TextToVideoZeroPipeline` | Short videos from Stable Diffusion 1.x-compatible models | Returns `TextToVideoPipelineOutput`; current source is deprecated and last-supported at `0.33.1`. |
| Prompt-only SDXL text-to-video | `TextToVideoZeroSDXLPipeline` | Short videos from `stabilityai/stable-diffusion-xl-base-1.0` or compatible SDXL weights | Uses SDXL dual text encoders, size conditioning, `guidance_rescale`, and optional watermarking. |
| Pose-controlled SD 1.x video | `StableDiffusionControlNetPipeline` plus `CrossFrameAttnProcessor` | Generate frames that follow OpenPose skeleton images | There is no separate public `TextToVideoZeroControlNetPipeline` class in the current docs/source. |
| Edge-controlled SD 1.x video | `StableDiffusionControlNetPipeline` plus `CrossFrameAttnProcessor` | Generate frames that follow Canny/edge maps | Same pattern as pose control with a Canny ControlNet. |
| SDXL ControlNet video | `StableDiffusionXLControlNetPipeline` plus `CrossFrameAttnProcessor` | Pose or edge guidance with SDXL base/control checkpoints | The docs' SDXL ControlNet snippet imports the XL class but appears to instantiate the non-XL class; use the XL pipeline for SDXL checkpoints. |
| Video Instruct-Pix2Pix | `StableDiffusionInstructPix2PixPipeline` plus `CrossFrameAttnProcessor` | Instruction-guided video editing | Set the processor batch size to `3`, matching InstructPix2Pix's guidance layout. |
| DreamBooth specialization | Custom DreamBooth model loaded into the text-to-video or ControlNet pattern | Personalized subject/style videos | Use the DreamBooth trigger tokens and keep the ControlNet/base family compatible. |

## 3. Installation And Version Strategy

Prefer the last supported Diffusers version for this pipeline:

```powershell
.venv\Scripts\python.exe -m pip install "diffusers==0.33.1" transformers accelerate safetensors imageio imageio-ffmpeg
```

For ControlNet preprocessing examples, add the appropriate preprocessors:

```powershell
.venv\Scripts\python.exe -m pip install controlnet-aux opencv-python pillow
```

If you intentionally test the current `main` implementation, install from the
repository and expect the pipeline to live in deprecated source paths:

```powershell
.venv\Scripts\python.exe -m pip install -U git+https://github.com/huggingface/diffusers
```

Use `torch.float16` on CUDA for the official examples. CPU inference is
possible but generally impractical for video because each frame goes through a
full diffusion denoising path.

## 4. Core Algorithm In Diffusers

The source implementation is helpful for integration because it explains why
some parameters are more sensitive than normal text-to-image parameters.

The base `TextToVideoZeroPipeline.__call__` flow is:

1. Validate `video_length`, `frame_ids`, callback settings, height, and width.
2. Temporarily replace the UNet attention processors with
   `CrossFrameAttnProcessor2_0(batch_size=2)` on PyTorch 2.0+ or
   `CrossFrameAttnProcessor(batch_size=2)` otherwise.
3. Encode prompt and negative prompt with Stable Diffusion text encoding.
4. Prepare a single initial latent for the first frame.
5. Denoise the first-frame latent backward to timestep `t1`.
6. Continue denoising that first-frame latent to timestep `t0`.
7. Repeat the first-frame `t0` latent for `video_length - 1` later frames.
8. Create a translation motion field from `motion_field_strength_x`,
   `motion_field_strength_y`, and `frame_ids[1:]`.
9. Warp the repeated later-frame latents with the motion field.
10. Add forward-process noise from `t0` to `t1` for the later frames.
11. Concatenate the first-frame `t1` latent with the later-frame `t1` latents.
12. Repeat prompt embeddings across frames and denoise all frames from `t1`
    to zero using cross-frame attention.
13. Decode latents, optionally run the safety checker, restore the original
    attention processors, and return the output.

That structure means `t0`, `t1`, motion strength, scheduler choice, and
`frame_ids` are more central than in modern video pipelines. They control the
amount of latent propagation and the apparent direction/speed of motion.

## 5. `TextToVideoZeroPipeline`

Use `TextToVideoZeroPipeline` for prompt-only, Stable Diffusion 1.x-style
text-to-video. It accepts the usual Stable Diffusion components:

- `vae`: `AutoencoderKL`.
- `text_encoder`: `CLIPTextModel`.
- `tokenizer`: `CLIPTokenizer`.
- `unet`: `UNet2DConditionModel`.
- `scheduler`: a compatible scheduler. The source mentions DDIM, LMS, and
  PNDM-style schedulers, and the implementation accesses scheduler `alphas`
  in `forward_loop`.
- `safety_checker` and `feature_extractor`: Stable Diffusion safety checker
  components. `requires_safety_checker=True` by default.

Minimal example:

```python
import imageio
import torch
from diffusers import TextToVideoZeroPipeline

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

pipe = TextToVideoZeroPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
).to("cuda")

output = pipe(
    prompt="A panda is playing guitar in Times Square, cinematic",
    negative_prompt="blurry, low quality, distorted",
    video_length=8,
    num_inference_steps=50,
    guidance_scale=7.5,
    motion_field_strength_x=12,
    motion_field_strength_y=12,
    t0=44,
    t1=47,
    output_type="np",
    generator=torch.Generator(device="cuda").manual_seed(1234),
)

frames = [(frame * 255).round().clip(0, 255).astype("uint8") for frame in output.images]
imageio.mimsave("text2video_zero.mp4", frames, fps=4)
```

### Important Call Parameters

| Parameter | Default in current API/source | Implementation guidance |
| --- | --- | --- |
| `prompt` | Required unless prompt embeds are supplied internally | Single prompt or list of prompts. Keep `num_videos_per_prompt=1`; the base source asserts this. |
| `video_length` | `8` | Number of frames returned. Keep clips short unless chunking. The source asserts `video_length > 0`. |
| `height`, `width` | UNet sample size times VAE scale factor | Must be divisible by 8. SD 1.5 defaults are typically 512x512. |
| `num_inference_steps` | `50` | More steps improve quality but increase cost linearly with the denoising loops. |
| `guidance_scale` | `7.5` | Classifier-free guidance is active when `guidance_scale > 1`. Cross-frame processor batch size assumes the normal CFG duplicated batch. |
| `negative_prompt` | `None` | Recommended for reducing blur, distortion, flicker, and artifacts. |
| `num_videos_per_prompt` | `1` | Documented as a parameter but base source asserts it must be `1`. Run multiple calls for multiple videos. |
| `eta` | `0.0` | Only used by DDIM-compatible schedulers that accept `eta`; ignored by others. |
| `generator` | `None` | Use a CUDA generator with a fixed seed for reproducibility. |
| `latents` | `None` | Shape should match one image latent, usually `(1, 4, height / 8, width / 8)`, before the pipeline expands frames. |
| `motion_field_strength_x` | `12` | Translation strength in x direction. Larger values increase motion but can smear or break identity. |
| `motion_field_strength_y` | `12` | Translation strength in y direction. Set one axis to `0` for simpler horizontal or vertical motion. |
| `t0` | `44` in API/source | Must be in `[0, num_inference_steps - 1]`. Controls the latent point where later frames are copied and motion-warped. |
| `t1` | `47` in API/source | Must be greater than `t0` and below `num_inference_steps`. Controls how much forward noise is added after motion warping. |
| `frame_ids` | `range(video_length)` | Frame indexes used when computing the motion field; important for chunked long videos. |
| `output_type` | Signature shows `"tensor"`; parameter docs describe `"np"` or `"latent"` | Pass explicitly. Use `"np"` for normal video writing, `"latent"` to keep latent output for downstream processing. |
| `return_dict` | `True` | If `False`, returns a tuple instead of `TextToVideoPipelineOutput`. |

The docs prose says the default `t0`/`t1` values are `45`/`48`, but the
rendered API signature and current source use `44`/`47`. Prefer the source/API
signature and pass the values explicitly in workflow payloads.

### Motion Field Tuning

The source creates a fixed 512x512 translation flow and fills it as:

- x flow: `motion_field_strength_x * frame_id`
- y flow: `motion_field_strength_y * frame_id`

That flow is resized to the latent shape and applied with grid sampling. In
practice:

- Use `motion_field_strength_x=12, motion_field_strength_y=12` as the
  documented baseline.
- Use one axis at `0` to reduce diagonal drift, for example
  `motion_field_strength_x=16, motion_field_strength_y=0`.
- Use smaller values, such as `4` to `8`, for portraits or subjects that
  should stay close to the first-frame composition.
- Use larger values only when the prompt and composition can tolerate strong
  camera/object translation.
- Negative values should reverse the translation direction, but test them in
  your target version because the official docs only document positive
  defaults.

### `t0` And `t1` Tuning

`t0` and `t1` are denoising-step indexes, not scheduler timesteps. The source
uses them as negative indexes into `self.scheduler.timesteps`, so they need to
be valid relative to `num_inference_steps`.

Recommended integration rules:

- Keep `0 <= t0 < t1 <= num_inference_steps - 1`.
- Keep the documented/default gap small, usually `t1 - t0` around `3`.
- Lower both values if the motion warp is too weak or the frames are almost
  static.
- Raise both values if identity or background consistency collapses.
- Do not expose unbounded values in a public workflow schema; validate them
  before calling the pipeline.

### Longer Videos With `frame_ids`

The official docs generate longer clips by processing overlapping chunks. The
first frame is repeated into each chunk for cross-frame attention, and
`frame_ids` tells the motion field the global frame indexes.

```python
import imageio
import numpy as np
import torch
from diffusers import TextToVideoZeroPipeline

pipe = TextToVideoZeroPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

prompt = "A panda is playing guitar in Times Square, cinematic"
seed = 0
video_length = 24
chunk_size = 8
chunk_starts = np.arange(0, video_length, chunk_size - 1)
generator = torch.Generator(device="cuda")

chunks = []
for index, start in enumerate(chunk_starts):
    end = video_length if index == len(chunk_starts) - 1 else chunk_starts[index + 1]
    frame_ids = [0] + list(range(start, end))
    generator.manual_seed(seed)
    output = pipe(
        prompt=prompt,
        video_length=len(frame_ids),
        frame_ids=frame_ids,
        generator=generator,
        output_type="np",
    )
    chunks.append(output.images[1:])

frames = np.concatenate(chunks)
frames = [(frame * 255).round().clip(0, 255).astype("uint8") for frame in frames]
imageio.mimsave("text2video_zero_long.mp4", frames, fps=4)
```

Chunking gives longer clips but does not make the model a long-video model.
Motion can drift, and the first-frame anchor may become visually stale.

## 6. `TextToVideoZeroSDXLPipeline`

Use `TextToVideoZeroSDXLPipeline` for prompt-only SDXL text-to-video. It keeps
the same Text2Video-Zero motion/cross-frame idea but uses SDXL components:

- `vae`: `AutoencoderKL`.
- `text_encoder`: first CLIP text encoder.
- `text_encoder_2`: second CLIP text encoder with projection.
- `tokenizer` and `tokenizer_2`.
- `unet`: SDXL `UNet2DConditionModel`.
- `scheduler`: compatible image scheduler.
- Optional `image_encoder`, `feature_extractor`, and `add_watermarker`.
- `force_zeros_for_empty_prompt=True` by default.

Minimal SDXL example:

```python
import imageio
import numpy as np
import torch
from diffusers import TextToVideoZeroSDXLPipeline

pipe = TextToVideoZeroSDXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")

output = pipe(
    prompt="A glass sculpture blooms into a city skyline, cinematic, detailed",
    negative_prompt="blurry, low quality, warped, flicker",
    video_length=8,
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=7.5,
    guidance_rescale=0.0,
    motion_field_strength_x=10,
    motion_field_strength_y=4,
    t0=44,
    t1=47,
    output_type="np",
    generator=torch.Generator(device="cuda").manual_seed(2024),
)

frames = output.images
if isinstance(frames, np.ndarray):
    frames = [(frame * 255).round().clip(0, 255).astype("uint8") for frame in frames]
imageio.mimsave("text2video_zero_sdxl.mp4", frames, fps=4)
```

### SDXL-Specific Parameters

`TextToVideoZeroSDXLPipeline.__call__` adds SDXL parameters on top of the base
pipeline:

- `prompt_2` and `negative_prompt_2` for the second text encoder. If omitted,
  the first prompt values are reused.
- `prompt_embeds`, `negative_prompt_embeds`, `pooled_prompt_embeds`, and
  `negative_pooled_prompt_embeds` for precomputed SDXL conditioning.
- `denoising_end` for partial denoising in mixture-of-denoisers workflows.
  This is inherited from SDXL image pipelines and is rarely needed for a
  simple Text2Video-Zero integration.
- `cross_attention_kwargs` forwarded to the attention processor.
- `guidance_rescale` for SDXL guidance overexposure mitigation.
- `original_size`, `crops_coords_top_left`, and `target_size` for SDXL
  micro-conditioning.

The current SDXL source defines a separate `TextToVideoSDXLPipelineOutput`
dataclass with an `images` field. The docs page's final output section focuses
on the base `TextToVideoPipelineOutput`, so code should handle both by reading
`output.images` rather than checking the concrete class name.

### SDXL Resolution And Latent Shapes

SDXL examples use 1024x1024 by default. For a 1024x1024 SDXL ControlNet or
manual-latent workflow, the latent spatial shape is usually 128x128 because
the VAE scale factor is 8. For 768x768, use 96x96 latents; for 1024x576, use
128x72 latents.

Keep `height` and `width` divisible by 8. SDXL memory use is much higher than
SD 1.5, and Text2Video-Zero denoises multiple frames, so memory offload,
smaller frame sizes, or fewer frames may be necessary.

## 7. ControlNet Pose And Edge Guidance

The current official docs/source do not expose a public
`TextToVideoZeroControlNetPipeline` class. Instead, the official pattern is:

1. Prepare a list of per-frame conditioning images, such as OpenPose skeletons
   or Canny edge maps.
2. Load a normal ControlNet pipeline.
3. Import `CrossFrameAttnProcessor` from the Text2Video-Zero implementation.
4. Set the processor on both `pipe.unet` and `pipe.controlnet`.
5. Use identical prompts for all frames.
6. Use fixed per-frame latents created by repeating a single latent tensor
   across the number of conditioning frames.

### Import Compatibility

The import path differs between the last supported release and current
deprecated source layout. Use a small fallback when writing reusable code:

```python
try:
    from diffusers.pipelines.deprecated.text_to_video_synthesis.pipeline_text_to_video_zero import (
        CrossFrameAttnProcessor,
    )
except ImportError:
    from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import (
        CrossFrameAttnProcessor,
    )
```

If you are strictly pinned to Diffusers `0.33.1`, the second import path is the
one documented by the stable page.

### Pose Control With SD 1.5

```python
import imageio
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

try:
    from diffusers.pipelines.deprecated.text_to_video_synthesis.pipeline_text_to_video_zero import (
        CrossFrameAttnProcessor,
    )
except ImportError:
    from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import (
        CrossFrameAttnProcessor,
    )

# `pose_images` should be a list of PIL images, one control image per frame.
# The official docs load demo pose frames from the PAIR/Text2Video-Zero Space.
frame_count = len(pose_images)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose",
    torch_dtype=torch.float16,
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to("cuda")

pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))

latents = torch.randn(
    (1, 4, 64, 64),
    device="cuda",
    dtype=torch.float16,
).repeat(frame_count, 1, 1, 1)

prompt = "A dancer wearing a silver jacket on a stage, cinematic"
frames = pipe(
    prompt=[prompt] * frame_count,
    image=pose_images,
    latents=latents,
).images

imageio.mimsave("pose_control.mp4", frames, fps=4)
```

`batch_size=2` in the processor is not the number of frames. It is the
effective non-frame batch dimension created by classifier-free guidance for a
single prompt. If you change the guidance pattern or run unusual batching,
the processor batch size must match that layout.

### SDXL ControlNet Pose

For SDXL ControlNet, use an SDXL ControlNet checkpoint and an SDXL ControlNet
pipeline. The docs mention SDXL support because the same attention processor
works with SDXL. The published snippet imports
`StableDiffusionXLControlNetPipeline` but appears to instantiate the non-XL
`StableDiffusionControlNetPipeline`; for implementation, instantiate the XL
pipeline when using SDXL weights.

```python
import imageio
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

try:
    from diffusers.pipelines.deprecated.text_to_video_synthesis.pipeline_text_to_video_zero import (
        CrossFrameAttnProcessor,
    )
except ImportError:
    from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import (
        CrossFrameAttnProcessor,
    )

frame_count = len(pose_images)

controlnet = ControlNetModel.from_pretrained(
    "thibaud/controlnet-openpose-sdxl-1.0",
    torch_dtype=torch.float16,
)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")

pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))

latents = torch.randn(
    (1, 4, 128, 128),
    device="cuda",
    dtype=torch.float16,
).repeat(frame_count, 1, 1, 1)

prompt = "A dancer wearing a silver jacket on a stage, cinematic, detailed"
frames = pipe(
    prompt=[prompt] * frame_count,
    image=pose_images,
    latents=latents,
).images

imageio.mimsave("pose_control_sdxl.mp4", frames, fps=4)
```

### Edge Control

Edge control follows the same pattern as pose control, but the conditioning
images are Canny/edge maps and the ControlNet model is a Canny model such as
`lllyasviel/sd-controlnet-canny`.

```python
import imageio
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

try:
    from diffusers.pipelines.deprecated.text_to_video_synthesis.pipeline_text_to_video_zero import (
        CrossFrameAttnProcessor,
    )
except ImportError:
    from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import (
        CrossFrameAttnProcessor,
    )

frame_count = len(canny_edges)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16,
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to("cuda")

pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))

latents = torch.randn(
    (1, 4, 64, 64),
    device="cuda",
    dtype=torch.float16,
).repeat(frame_count, 1, 1, 1)

prompt = "A red sports car driving through neon rain, cinematic"
frames = pipe(
    prompt=[prompt] * frame_count,
    image=canny_edges,
    latents=latents,
).images

imageio.mimsave("edge_control.mp4", frames, fps=4)
```

Implementation notes for control workflows:

- Ensure `len(prompt_list) == len(conditioning_images) == latents.shape[0]`.
- Resize control images to the same generation resolution.
- Use a single repeated latent to keep frame identity consistent.
- Match ControlNet family to base model family: SD 1.5 ControlNet with SD 1.5
  base, SDXL ControlNet with SDXL base.
- For actual video pose extraction, use the regular Diffusers ControlNet
  preprocessing guidance or `controlnet_aux` tools; Text2Video-Zero itself only
  consumes the resulting per-frame control images.

## 8. Video Instruct-Pix2Pix

The docs implement instruction-guided video editing by running
`StableDiffusionInstructPix2PixPipeline` over a list of input video frames while
using Text2Video-Zero's cross-frame attention processor.

```python
import imageio
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline

try:
    from diffusers.pipelines.deprecated.text_to_video_synthesis.pipeline_text_to_video_zero import (
        CrossFrameAttnProcessor,
    )
except ImportError:
    from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import (
        CrossFrameAttnProcessor,
    )

# `video_frames` should be a list of PIL images.
frame_count = len(video_frames)

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix",
    torch_dtype=torch.float16,
).to("cuda")

pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=3))

instruction = "make it look like a Van Gogh Starry Night painting"
frames = pipe(
    prompt=[instruction] * frame_count,
    image=video_frames,
).images

imageio.mimsave("edited_video.mp4", frames, fps=4)
```

Use `batch_size=3` because InstructPix2Pix uses a three-part guidance batch
rather than the normal two-part classifier-free guidance batch. If this value
is wrong, attention grouping can mix frames incorrectly and temporal
consistency degrades.

This variant is editing, not generation from scratch. It inherits
InstructPix2Pix controls such as image guidance and edit strength from that
pipeline, while cross-frame attention supplies a first-frame appearance anchor.

## 9. DreamBooth Specialization

The official docs state that text-to-video, pose control, and edge control can
run with custom DreamBooth models. In practice, DreamBooth is just the base
image model loaded by the Text2Video-Zero or ControlNet pipeline.

Prompt-only DreamBooth pattern:

```python
import imageio
import torch
from diffusers import TextToVideoZeroPipeline

pipe = TextToVideoZeroPipeline.from_pretrained(
    "your-org/your-dreambooth-sd15-model",
    torch_dtype=torch.float16,
).to("cuda")

output = pipe(
    prompt="sks person riding a bicycle through a sunny city, cinematic",
    negative_prompt="blurry, distorted, low quality",
    video_length=8,
    output_type="np",
)

frames = [(frame * 255).round().clip(0, 255).astype("uint8") for frame in output.images]
imageio.mimsave("dreambooth_text2video.mp4", frames, fps=4)
```

ControlNet plus DreamBooth pattern:

```python
import imageio
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

try:
    from diffusers.pipelines.deprecated.text_to_video_synthesis.pipeline_text_to_video_zero import (
        CrossFrameAttnProcessor,
    )
except ImportError:
    from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import (
        CrossFrameAttnProcessor,
    )

frame_count = len(canny_edges)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16,
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "PAIR/text2video-zero-controlnet-canny-avatar",
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to("cuda")

pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))

latents = torch.randn(
    (1, 4, 64, 64),
    device="cuda",
    dtype=torch.float16,
).repeat(frame_count, 1, 1, 1)

prompt = "oil painting of a beautiful girl, avatar style"
frames = pipe(
    prompt=[prompt] * frame_count,
    image=canny_edges,
    latents=latents,
).images

imageio.mimsave("dreambooth_edge_control.mp4", frames, fps=4)
```

DreamBooth gotchas:

- Use the model's trigger tokens exactly as trained.
- Keep subject motion modest. Text2Video-Zero preserves first-frame identity
  but can distort personalized subjects under aggressive motion fields.
- Match DreamBooth base family to the pipeline and ControlNet family.
- If the DreamBooth checkpoint includes its own VAE or scheduler choices, load
  them the same way you would for image generation before adding video logic.

## 10. Output Handling

### Base Output

The docs define:

```text
TextToVideoPipelineOutput(
    images: list[PIL.Image.Image] | numpy.ndarray,
    nsfw_content_detected: list[bool] | None,
)
```

For `TextToVideoZeroPipeline`, `output.images` is the generated frame sequence
when `return_dict=True`. If `return_dict=False`, the source returns
`(image, has_nsfw_concept)`.

The official text-to-video example treats frames as floating NumPy arrays in
`[0, 1]` and converts to `uint8` before `imageio.mimsave`. ControlNet and
InstructPix2Pix examples use PIL images directly. A robust saver can normalize
both:

```python
import numpy as np

def frames_to_uint8(frames):
    if isinstance(frames, np.ndarray):
        iterable = frames
    else:
        iterable = list(frames)

    converted = []
    for frame in iterable:
        array = np.asarray(frame)
        if np.issubdtype(array.dtype, np.floating):
            array = (array * 255).round().clip(0, 255).astype("uint8")
        elif array.dtype != np.uint8:
            array = array.clip(0, 255).astype("uint8")
        converted.append(array)
    return converted
```

### SDXL Output

The current SDXL source defines `TextToVideoSDXLPipelineOutput`, not the base
`TextToVideoPipelineOutput`, and it contains an `images` field. The public docs
page still groups the final output discussion under `TextToVideoPipelineOutput`.
For application code, treat the concrete output class as version-dependent and
read `output.images`.

### Video Encoding

The docs use `imageio.mimsave("video.mp4", frames, fps=4)`. For a workflow
server, expose FPS separately from frame generation. Text2Video-Zero examples
generate only 8 frames at 4 FPS, so increasing FPS without increasing frames
just shortens playback.

## 11. Scheduler, Memory, And Performance

Scheduler notes:

- `eta` only matters for DDIM-style schedulers that accept it.
- The source `forward_loop` accesses `self.scheduler.alphas[t0:t1]`, so not
  every modern scheduler is safe despite the generic scheduler type annotation.
- If changing the scheduler, validate a small 2-frame or 4-frame generation
  before exposing it as a runtime option.
- Keep `num_inference_steps` high enough for `t0` and `t1` to be valid.

Memory notes:

- SD 1.5 at 512x512 and 8 frames is the most practical baseline.
- SDXL at 1024x1024 and 8 frames can be much heavier; reduce resolution or
  frame count if CUDA memory is tight.
- `enable_model_cpu_offload()` may help if Accelerate is installed, but test
  it because this deprecated pipeline has a custom denoising flow.
- Attention slicing or VAE slicing may help in some versions, but the official
  Text2Video-Zero examples do not focus on those optimizations.

Performance notes:

- The method performs multiple denoising phases, so it is slower than one
  image generation multiplied by frame count would suggest.
- ControlNet and InstructPix2Pix variants run normal image pipelines over frame
  batches, with extra memory from control/image conditioning.
- It is usually better to generate fewer frames at low FPS than to stretch the
  method into long clips.

## 12. API Integration Notes

Recommended workflow schema fields:

- `model_id`: Stable Diffusion, SDXL, or DreamBooth model ID.
- `variant`: optional, usually `"fp16"` for SDXL.
- `task`: one of `text`, `sdxl_text`, `pose_control`, `edge_control`,
  `video_instruct_pix2pix`, or `dreambooth_control`.
- `prompt` and optional `negative_prompt`.
- `video_length`, default `8`.
- `height` and `width`, default from the pipeline family.
- `num_inference_steps`, default `50`.
- `guidance_scale`, default `7.5`.
- `motion_field_strength_x` and `motion_field_strength_y`, default `12`.
- `t0` and `t1`, default `44` and `47`.
- `fps`, default `4`, used only during export.
- `seed`.
- `controlnet_model_id` for pose/edge tasks.
- `conditioning_frames` for pose/edge tasks.
- `input_video_frames` for InstructPix2Pix.
- `output_type`, default `"np"` for text-to-video tasks.

Validation rules:

- Reject `video_length <= 0`.
- Require `height % 8 == 0` and `width % 8 == 0`.
- Require `0 <= t0 < t1 <= num_inference_steps - 1`.
- Keep `num_videos_per_prompt=1` for the base pipeline.
- Require `len(frame_ids) == video_length` if `frame_ids` is supplied.
- For ControlNet and InstructPix2Pix, require one conditioning/input image per
  output frame.
- Validate latent shapes if accepting caller-supplied latents.
- Pin or record the Diffusers version in job metadata because this pipeline is
  deprecated and version-sensitive.

Return contract guidance:

- Store frames or encoded video path as the primary artifact.
- Include `nsfw_content_detected` when present from the base pipeline.
- Include generation metadata: model ID, seed, video length, FPS,
  motion field strengths, `t0`, `t1`, scheduler class, and Diffusers version.
- Do not promise audio; Text2Video-Zero only generates silent visual frames.

## 13. Common Gotchas

- Deprecated pipeline: current docs say it is no longer tested. Pin
  `diffusers==0.33.1` for reliability.
- Latest stable docs gap: the `v0.38.0` page for Text2Video-Zero does not
  exist, while the `main` page still exists and requires installing Diffusers
  from source.
- Import path drift: `CrossFrameAttnProcessor` is under
  `diffusers.pipelines.text_to_video_synthesis...` in supported releases and
  under `diffusers.pipelines.deprecated.text_to_video_synthesis...` in current
  source.
- No dedicated ControlNet pipeline class: implement pose/edge video with
  `StableDiffusionControlNetPipeline` or `StableDiffusionXLControlNetPipeline`
  plus the custom attention processor.
- SDXL docs typo: the docs import `StableDiffusionXLControlNetPipeline` but
  show `StableDiffusionControlNetPipeline.from_pretrained(...)` in the SDXL
  ControlNet snippet. Use the XL class for SDXL.
- Default mismatch: usage prose says `t0=45`, `t1=48`, while the API signature
  and source use `t0=44`, `t1=47`. Pass explicit values.
- Output type mismatch: the rendered signatures show `output_type="tensor"`,
  while parameter text and examples use NumPy frames. Pass `output_type="np"`
  or `output_type="latent"` explicitly.
- `num_videos_per_prompt`: documented as an argument, but the base source
  asserts it is exactly `1`.
- Processor batch size: use `2` for normal CFG ControlNet usage and `3` for
  InstructPix2Pix. It is not frame count.
- Latent shape: repeated fixed latents in ControlNet examples must match the
  generation resolution and family, such as `(1, 4, 64, 64)` for 512x512 SD
  and `(1, 4, 128, 128)` for 1024x1024 SDXL.
- Motion limits: strong motion fields can create tearing, drift, and identity
  loss because the method uses simple translation flow rather than learned
  video dynamics.
- Safety checker: the base pipeline includes Stable Diffusion safety checker
  behavior; if disabled, treat that as an explicit deployment decision.

## 14. Quick Implementation Checklist

1. Decide whether this deprecated pipeline is acceptable for the workflow; if
   yes, pin Diffusers `0.33.1`.
2. Start with `TextToVideoZeroPipeline`, SD 1.5, 512x512, 8 frames, 4 FPS,
   `motion_field_strength_x=12`, `motion_field_strength_y=12`,
   `t0=44`, `t1=47`.
3. Add explicit validation for frame count, dimensions, `t0`/`t1`, and
   `num_videos_per_prompt`.
4. Normalize `output.images` to `uint8` before video encoding.
5. Add SDXL only after SD 1.5 works, and handle `TextToVideoSDXLPipelineOutput`
   by reading `.images`.
6. For pose/edge control, do not look for `TextToVideoZeroControlNetPipeline`;
   inject `CrossFrameAttnProcessor` into the appropriate ControlNet pipeline.
7. For InstructPix2Pix, set `CrossFrameAttnProcessor(batch_size=3)`.
8. Record Diffusers version and source path in debug logs because the pipeline
   is deprecated and source layout differs by version.

## 15. Source Links

- Hugging Face Diffusers main Text2Video-Zero docs:
  <https://huggingface.co/docs/diffusers/main/en/api/pipelines/text_to_video_zero>
- Hugging Face Diffusers `v0.33.1` Text2Video-Zero docs:
  <https://huggingface.co/docs/diffusers/v0.33.1/api/pipelines/text_to_video_zero>
- Hugging Face Diffusers `v0.38.0` missing-page notice:
  <https://huggingface.co/docs/diffusers/v0.38.0/api/pipelines/text_to_video_zero>
- Current deprecated base pipeline source:
  <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deprecated/text_to_video_synthesis/pipeline_text_to_video_zero.py>
- Current deprecated SDXL pipeline source:
  <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deprecated/text_to_video_synthesis/pipeline_text_to_video_zero_sdxl.py>
- Last supported base pipeline source:
  <https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_zero.py>
- Last supported SDXL pipeline source:
  <https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_zero_sdxl.py>
- Text2Video-Zero paper page linked by the docs:
  <https://huggingface.co/papers/2303.13439>
- Original project page linked by the docs:
  <https://text2video-zero.github.io/>
- Original codebase linked by the docs:
  <https://github.com/Picsart-AI-Research/Text2Video-Zero>
