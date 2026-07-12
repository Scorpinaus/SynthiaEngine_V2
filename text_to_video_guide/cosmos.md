# Cosmos Diffusers Implementation Guide

Last checked: 2026-06-17 against the Hugging Face Diffusers Cosmos API page and
the linked `v0.38.0` Diffusers source.

Cosmos is NVIDIA's world-generation family in Diffusers. In practice, "world"
usually means video, but the family also includes image generation, first-frame
or input-clip conditioning, and ControlNet-style world-to-world transfer. For a
new text-to-video integration, start with `Cosmos2_5_PredictBasePipeline` unless
you specifically need an older Cosmos 1.0 checkpoint or a control-video transfer
model.

## Pipeline selection

| Pipeline | Main task | Documented checkpoint(s) | Modes |
| --- | --- | --- | --- |
| `Cosmos2_5_PredictBasePipeline` | Latest general text/image/video-to-world pipeline | `nvidia/Cosmos-Predict2.5-2B` with `revision="diffusers/base/post-trained"` | Text2World, Image2World, Video2World, and `num_frames=1` image-like output |
| `Cosmos2_5_TransferPipeline` | Control-video guided world-to-world transfer | `nvidia/Cosmos-Transfer2.5-2B`; pipeline `revision="diffusers/general"` plus ControlNet revisions such as `diffusers/controlnet/general/edge` | Control video(s) plus text prompt; supports edge, depth, segmentation, and blur control variants |
| `CosmosTextToWorldPipeline` | Cosmos 1.0 text-to-world | `nvidia/Cosmos-1.0-Diffusion-7B-Text2World` | Text2World only |
| `CosmosVideoToWorldPipeline` | Cosmos 1.0 image/video-to-world | `nvidia/Cosmos-1.0-Diffusion-7B-Video2World` | Image2World or Video2World |
| `Cosmos2TextToImagePipeline` | Cosmos Predict2 text-to-image | `nvidia/Cosmos-Predict2-2B-Text2Image`, `nvidia/Cosmos-Predict2-14B-Text2Image` | Text2Image |
| `Cosmos2VideoToWorldPipeline` | Cosmos Predict2 image/video-to-world | `nvidia/Cosmos-Predict2-2B-Video2World`, `nvidia/Cosmos-Predict2-14B-Video2World` | Image2World or Video2World |

The Cosmos 1.0 model cards also describe 14B Text2World and Video2World
siblings, but the Diffusers Cosmos API examples use the 7B checkpoints for
those two classes. The Predict2 classes explicitly document both 2B and 14B
checkpoints in the API examples.

## Setup

Install the normal Diffusers stack and the Cosmos guardrail package. The source
constructs a `CosmosSafetyChecker` by default; if `cosmos_guardrail` is missing,
pipeline construction raises an import error.

```bash
pip install -U diffusers transformers accelerate torch torchvision
pip install cosmos_guardrail
```

Most NVIDIA Cosmos model repos are gated. Accept the model terms on Hugging Face
and authenticate before loading:

```bash
huggingface-cli login
```

Use `torch_dtype=torch.bfloat16` for the documented examples. NVIDIA's model
cards say BF16 is the tested precision, and Cosmos is optimized for NVIDIA
GPU-accelerated systems.

## Text-to-world with Cosmos 2.5

`Cosmos2_5_PredictBasePipeline` is the most useful text-to-video entry point on
the Cosmos page. It uses Qwen2.5-VL as its text encoder, `AutoencoderKLWan` as
the video VAE, and `UniPCMultistepScheduler`. Defaults are 1280x704, 93 frames,
36 denoising steps, and guidance scale 7.0. The docs export the 93-frame output
at 16 fps.

```python
import torch
from diffusers import Cosmos2_5_PredictBasePipeline
from diffusers.utils import export_to_video

model_id = "nvidia/Cosmos-Predict2.5-2B"

pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
    model_id,
    revision="diffusers/base/post-trained",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

prompt = (
    "A warehouse robot rolls between tall shelves of labeled containers. "
    "Its head light sweeps across the aisle while a second robot turns in the "
    "background, creating a realistic logistics scene with steady camera motion."
)
negative_prompt = (
    "low quality, blurry, flickering, distorted motion, unstable camera, "
    "oversaturated colors, artifacts"
)

frames = pipe(
    image=None,
    video=None,
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=93,
    generator=torch.Generator(device="cuda").manual_seed(1),
).frames[0]

export_to_video(frames, "cosmos25_text2world.mp4", fps=16)
```

Conditioning modes:

- Text2World: pass `image=None`, `video=None`, and a prompt.
- Image2World: pass one `image` and `video=None`; this conditions the clip on a
  first frame.
- Video2World: pass `video` and `image=None`; this predicts a world clip from an
  input clip.
- Image-like output: set `num_frames=1` and use `frames[0][0]`.

Important parameters:

- `height`, `width`: default 704 and 1280. Source validation requires both to be
  divisible by 16.
- `num_frames`: default 93. Use 93 for the documented world-video mode.
- `num_inference_steps`: default 36 in the signature. More steps are slower and
  usually higher quality.
- `guidance_scale`: default 7.0. Classifier-free guidance is enabled when this
  is greater than 1.0.
- `max_sequence_length`: default 512. Prompts longer than this are truncated.
- `num_latent_conditional_frames`: default 2 for Video2World. The source derives
  the required input pixel frames as `4 * (num_latent_conditional_frames - 1) + 1`,
  so the default consumes 5 conditioning frames. Set it to 1 for single-frame
  behavior.
- `conditional_frame_timestep`: default 0.1. This controls the sigma/timestep
  used for conditional latents.
- `prompt_embeds` and `negative_prompt_embeds`: advanced path for reusing or
  modifying text embeddings. Do not pass both raw prompts and embeddings.

Gotchas:

- `image` and `video` are mutually exclusive.
- The source only supports batch size 1 for image/video conditioning in the 2.5
  base pipeline.
- If you pass a video that has fewer frames than required by
  `num_latent_conditional_frames`, the source raises a validation error.

## Cosmos 1.0 text-to-world

`CosmosTextToWorldPipeline` is the original Cosmos Predict1 text-only video
pipeline. It uses T5, specifically `t5-11b`, with `AutoencoderKLCosmos` and an
Euler-style scheduler. Defaults are 1280x704, 121 frames, 36 steps, guidance
scale 7.0, and 30 fps.

```python
import torch
from diffusers import CosmosTextToWorldPipeline
from diffusers.utils import export_to_video

pipe = CosmosTextToWorldPipeline.from_pretrained(
    "nvidia/Cosmos-1.0-Diffusion-7B-Text2World",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

frames = pipe(
    prompt=(
        "A sleek humanoid robot stands in a warehouse aisle while stacked boxes "
        "and pallets recede into the background. The camera remains static."
    ),
    num_frames=121,
    fps=30,
).frames[0]

export_to_video(frames, "cosmos10_text2world.mp4", fps=30)
```

Use this pipeline when you need the Cosmos 1.0 Text2World checkpoint rather than
the newer 2.5 base model. It does not accept image or video conditioning inputs.

## Image/video-to-world with Cosmos 1.0

`CosmosVideoToWorldPipeline` uses the Cosmos 1.0 Video2World checkpoint for
first-frame or input-clip conditioning. The docs show both modes with
`nvidia/Cosmos-1.0-Diffusion-7B-Video2World`.

```python
import torch
from diffusers import CosmosVideoToWorldPipeline
from diffusers.utils import export_to_video, load_image

pipe = CosmosVideoToWorldPipeline.from_pretrained(
    "nvidia/Cosmos-1.0-Diffusion-7B-Video2World",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

image = load_image("input_first_frame.png")
frames = pipe(image=image, prompt="A vehicle drives down a bright highway.").frames[0]
export_to_video(frames, "cosmos10_image2world.mp4", fps=30)
```

Key parameters beyond the common prompt, size, step, guidance, generator, and
embedding options:

- `image`: optional conditioning image.
- `video`: optional conditioning video. Source validation requires exactly one
  of `image` or `video`.
- `num_frames`: default 121.
- `input_frames_guidance`: default `False`; documented in the signature as a
  conditioning control.
- `augment_sigma`: default 0.001. NVIDIA's model card explains that augment
  noise is added to conditional latent frames to bridge the train/inference gap.
- `fps`: default 30.

The docs example compiles the transformer for video conditioning:

```python
pipe.transformer = torch.compile(pipe.transformer)
```

Compilation adds first-run overhead, but it can improve repeated inference on
the same process and shape.

## Cosmos Predict2 image and video pipelines

### `Cosmos2TextToImagePipeline`

This class produces a single image with `CosmosImagePipelineOutput.images`.
Documented checkpoints are:

- `nvidia/Cosmos-Predict2-2B-Text2Image`
- `nvidia/Cosmos-Predict2-14B-Text2Image`

Defaults are 1360x768, 35 denoising steps, guidance scale 7.0, and one image
per prompt.

```python
import torch
from diffusers import Cosmos2TextToImagePipeline

pipe = Cosmos2TextToImagePipeline.from_pretrained(
    "nvidia/Cosmos-Predict2-2B-Text2Image",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

image = pipe(
    prompt="A yellow scrub brush cleans a plate under bright kitchen lights.",
    negative_prompt="low quality, blurry, distorted",
    generator=torch.Generator(device="cuda").manual_seed(1),
).images[0]

image.save("cosmos2_text2image.png")
```

### `Cosmos2VideoToWorldPipeline`

This class generates a video from an image or video conditioning input with a
text prompt. Documented checkpoints are:

- `nvidia/Cosmos-Predict2-2B-Video2World`
- `nvidia/Cosmos-Predict2-14B-Video2World`

Defaults are 1280x704, 93 frames, 35 steps, guidance scale 7.0, and 16 fps.

```python
import torch
from diffusers import Cosmos2VideoToWorldPipeline
from diffusers.utils import export_to_video, load_image

pipe = Cosmos2VideoToWorldPipeline.from_pretrained(
    "nvidia/Cosmos-Predict2-2B-Video2World",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

image = load_image("first_frame.png")
frames = pipe(
    image=image,
    prompt="A scrub brush moves across a plate as soap bubbles spread.",
    negative_prompt="low quality, blurry, flickering",
    generator=torch.Generator(device="cuda").manual_seed(1),
).frames[0]

export_to_video(frames, "cosmos2_video2world.mp4", fps=16)
```

The distinctive parameter is `sigma_conditioning`, default 0.0001. The docs say
it scales conditioning latents and should stay close to zero.

Although the class name says Video2World, the documented example uses image
conditioning. The source takes `image` first if it is provided, otherwise it
preprocesses `video`; use one conditioning source per call.

## Transfer2.5 control-video generation

`Cosmos2_5_TransferPipeline` is the Cosmos Transfer2.5 pipeline for controlled
world-to-world generation. It pairs the base transfer pipeline with a
`CosmosControlNetModel` loaded through `AutoModel`.

The docs show this pattern for edge control:

```python
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import AutoModel, Cosmos2_5_TransferPipeline
from diffusers.utils import export_to_video, load_video

model_id = "nvidia/Cosmos-Transfer2.5-2B"

controlnet = AutoModel.from_pretrained(
    model_id,
    revision="diffusers/controlnet/general/edge",
    torch_dtype=torch.bfloat16,
)

pipe = Cosmos2_5_TransferPipeline.from_pretrained(
    model_id,
    controlnet=controlnet,
    revision="diffusers/general",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

input_video = load_video("robot_input.mp4")
num_frames = 93

edge_maps = [
    cv2.Canny(cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR), 100, 200)
    for frame in input_video[:num_frames]
]
edge_maps = np.stack(edge_maps)[None]
controls = torch.from_numpy(edge_maps).expand(3, -1, -1, -1)
controls = [Image.fromarray(x.numpy()) for x in controls.permute(1, 2, 3, 0)]

frames = pipe(
    controls=controls,
    controls_conditioning_scale=1.0,
    prompt="Two robotic arms manipulate fabric on a cushion in a lab.",
    negative_prompt="low quality, shaky, flickering, artifacts",
    num_frames=num_frames,
).frames[0]

export_to_video(frames, "cosmos_transfer_edge.mp4", fps=30)
```

Conditioning notes:

- The Diffusers docs list control variants as edge, depth, segmentation, and
  blur. Use the matching ControlNet revision, for example
  `diffusers/controlnet/general/edge`.
- `controls` can be one control input or a list. For multiple controls, pass a
  matching list of `controls_conditioning_scale` values when you need different
  strengths.
- The docs state controls are assumed to be preprocessed. For edge control, the
  example computes Canny maps before calling the pipeline. Do not assume the
  Diffusers pipeline will extract depth, segmentation, edge, or blur maps for
  you.
- `width=None` is allowed. The source infers width from the first control frame
  and `height`, then validates that height and width are positive multiples of
  16. For reproducibility, pass an explicit width.

Transfer-specific parameters:

- `num_frames`: if omitted, output length follows the number of control frames.
- `num_frames_per_chunk`: default 93. Longer controls use auto-regressive
  sliding-window inference.
- `num_ar_conditional_frames`: default 1. Used between chunks unless
  `num_ar_latent_conditional_frames` is set.
- `num_ar_latent_conditional_frames`: optional latent-frame alternative to
  `num_ar_conditional_frames`.
- `conditional_frame_timestep`: default 0.1 and must be in `[0, 1]`.
- `guidance_scale`: default 3.0 for Transfer2.5, lower than the Predict
  pipelines' default 7.0.

## Outputs

All world/video pipelines return `CosmosPipelineOutput` when `return_dict=True`.
Use `output.frames`.

- With `output_type="pil"`, `frames` is a nested list shaped like
  `[batch][num_frames]`, where each item is a PIL image.
- With array/tensor output modes, the documented output shape is
  `(batch_size, num_frames, channels, height, width)` for tensors/arrays.
- `export_to_video(output.frames[0], "file.mp4", fps=...)` is the normal save
  path.

`Cosmos2TextToImagePipeline` returns `CosmosImagePipelineOutput` with
`output.images`, either a list of PIL images or a NumPy array shaped like
`(batch_size, height, width, num_channels)`.

The API docs say `return_dict=False` returns a tuple whose second element is an
NSFW boolean list. The linked `v0.38.0` source returns a one-element tuple such
as `(video,)` or `(image,)` in the concrete Cosmos pipelines. Treat
`return_dict=True` as the stable path unless you have checked the exact
installed Diffusers version.

## Safety checker behavior

Safety is not a cosmetic option in the current Diffusers Cosmos source:

- The pipelines mark `safety_checker` as an optional component for test/loading
  plumbing, but constructors create `CosmosSafetyChecker()` when no checker is
  supplied.
- If `cosmos_guardrail` is not installed, constructing that default checker
  raises an import error telling you to install `cosmos_guardrail`.
- The `__call__` methods raise a `ValueError` if `self.safety_checker is None`,
  with a message that disabling the checker violates the NVIDIA Open Model
  License Agreement.
- Prompts are checked with `check_text_safety`; unsafe text raises `ValueError`.
- World/video outputs are checked with `check_video_safety` after VAE decoding.
  The text-to-image pipeline decodes a one-frame video, runs the same video
  safety check, and then extracts the image frame.
- `output_type="latent"` skips image/video decoding and therefore skips the
  post-decode output safety pass. Use it only for internal pipeline chaining,
  not as a final user-facing media result.

NVIDIA model cards also warn that bypassing or reducing safety guardrails can
terminate rights under the NVIDIA Open Model License. Keep the guardrail in
place in production integrations.

## Memory and performance

- Start with the 2B checkpoints. The 14B checkpoints are much larger and should
  be treated as high-VRAM deployments.
- Use BF16: `torch_dtype=torch.bfloat16`. NVIDIA's Cosmos model cards state BF16
  is the tested precision.
- Prefer NVIDIA GPUs. The model cards list NVIDIA Ampere, Hopper, and Blackwell
  support and note that the models are optimized for NVIDIA GPU acceleration.
- The Cosmos-Predict2.5-2B model card reports 32.54 GB GPU VRAM for 720p
  16-fps Video2World and multi-minute inference times depending on GPU. Plan for
  large memory and latency even with the 2B model.
- Use `pipe.enable_model_cpu_offload()` if VRAM is tight. The source declares
  `model_cpu_offload_seq = "text_encoder->transformer->vae"` for Predict
  pipelines and includes `controlnet` in the Transfer2.5 sequence.
- Reduce `height`, `width`, `num_frames`, or `num_inference_steps` first when
  you need faster iterations. Keep spatial dimensions divisible by 16.
- Compile only after you have stable shapes: the docs show
  `pipe.transformer = torch.compile(pipe.transformer)` for Cosmos 1.0
  Video2World. This can help repeated runs but has first-run overhead.
- Reuse pipeline components when loading related pipelines in the same process;
  the Cosmos docs link to the Diffusers component reuse guide for this reason.

## Implementation gotchas

- The public docs currently mix generated autodoc text with source behavior.
  Check the exact installed Diffusers version for return tuple shapes, safety
  errors, and default values.
- Default negative prompts differ by pipeline. The 2.5 source defines a long
  default negative prompt, but explicit negative prompts make jobs easier to
  reproduce.
- Cosmos prompts should be concrete scene descriptions: camera, subject,
  environment, and motion. The Predict2.5 model card recommends staying under
  300 words for the input string.
- `height` and `width` must be divisible by 16 in all checked pipeline source.
- For Transfer2.5, precompute controls and keep their frame count aligned with
  `num_frames`. When generating more than `num_frames_per_chunk`, tune the
  auto-regressive conditioning frame parameters carefully.
- For Cosmos2.5 Video2World conditioning, remember the default
  `num_latent_conditional_frames=2` means 5 input frames, not 2 pixel frames.
- Keep `sigma_conditioning` near the default in `Cosmos2VideoToWorldPipeline`
  unless you are deliberately experimenting with conditioning strength.
- Use `output.frames[0]` for video and `output.images[0]` for images. The extra
  batch dimension is easy to forget.

## Official sources

- Diffusers Cosmos API page:
  https://huggingface.co/docs/diffusers/api/pipelines/cosmos
- Diffusers Cosmos docs source:
  https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/cosmos.md
- `Cosmos2_5_PredictBasePipeline` source:
  https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/cosmos/pipeline_cosmos2_5_predict.py
- `Cosmos2_5_TransferPipeline` source:
  https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/cosmos/pipeline_cosmos2_5_transfer.py
- `CosmosTextToWorldPipeline` source:
  https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/cosmos/pipeline_cosmos_text2world.py
- `CosmosVideoToWorldPipeline` source:
  https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/cosmos/pipeline_cosmos_video2world.py
- `Cosmos2TextToImagePipeline` source:
  https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/cosmos/pipeline_cosmos2_text2image.py
- `Cosmos2VideoToWorldPipeline` source:
  https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/cosmos/pipeline_cosmos2_video2world.py
- Cosmos output classes source:
  https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/cosmos/pipeline_output.py
- NVIDIA Cosmos-Predict2.5-2B model card:
  https://huggingface.co/nvidia/Cosmos-Predict2.5-2B
- NVIDIA Cosmos-Transfer2.5-2B model card:
  https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B
