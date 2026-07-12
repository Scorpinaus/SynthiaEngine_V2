# LTX-Video Diffusers Implementation Guide

Last checked: 2026-06-17 against the Hugging Face Diffusers LTX-Video API
page, the linked `v0.38.0` Diffusers source, and official Lightricks model
cards.

Research target:
https://huggingface.co/docs/diffusers/api/pipelines/ltx_video

Primary Diffusers classes on the page:

- `LTXPipeline`
- `LTXImageToVideoPipeline`
- `LTXConditionPipeline`
- `LTXI2VLongMultiPromptPipeline`
- `LTXLatentUpsamplePipeline`
- `LTXVideoCondition`
- `LTXPipelineOutput`

LTX-Video is Lightricks' video diffusion transformer family for fast text and
image conditioned video generation. The Diffusers page highlights the
Video-VAE as the key implementation detail: it has a high pixel-to-latent
compression ratio of `1:192`, and the decoder participates in the final
latent-to-pixel conversion and last denoising behavior. For a local workflow
server, the important design choice is which pipeline surface to expose:
simple text-to-video, first-frame image-to-video, general multi-condition
video generation, long-window I2V, or latent upscaling/refinement.

## 1. Pipeline Selection

| Class | Main task | Best use in an implementation |
| --- | --- | --- |
| `LTXPipeline` | Text-to-video | Use for pure prompt-based generation when no image or video conditioning is needed. |
| `LTXImageToVideoPipeline` | Image-to-video | Use for one input image, usually as a first-frame visual anchor. |
| `LTXConditionPipeline` | Text/image/video-to-video | Use for most production integrations because it accepts explicit `LTXVideoCondition` objects, can do text-only generation, image conditions, video segment conditions, and video editing/refinement with `latents` plus `denoise_strength`. |
| `LTXI2VLongMultiPromptPipeline` | Long-duration I2V with temporal windows | Use for long clips, per-window/multi-prompt scheduling, ComfyUI parity, and lower peak VRAM through temporal windowing. |
| `LTXLatentUpsamplePipeline` | Latent spatial upscaling | Use with `LTXConditionPipeline` when following the Lightricks multi-scale workflow: generate lower-resolution latents, upsample latent height/width by 2x, then denoise/refine with the base pipeline. |
| `LTXVideoCondition` | Conditioning helper | Use to place an image or video sequence at a target `frame_index` with a per-condition `strength`. |
| `LTXPipelineOutput` | Shared output dataclass | All LTX pipelines return `.frames` when `return_dict=True`. |

The underlying components are consistent across the generation pipelines:
`LTXVideoTransformer3DModel`, `AutoencoderKLLTXVideo`, a T5 text encoder
(`google/t5-v1_1-xxl`), a T5 tokenizer, and usually
`FlowMatchEulerDiscreteScheduler`. The long multi-prompt pipeline also
documents compatibility with `LTXEulerAncestralRFScheduler`.

## 2. Checkpoints And Model IDs

The Diffusers docs say the original checkpoints live under the
`Lightricks` organization. In practice, model choice matters because the
0.9.x family has different guidance, timestep, upscaling, and conditioning
recommendations.

| Model or asset | Documented use | Notes |
| --- | --- | --- |
| `Lightricks/LTX-Video` | General LTX-Video Diffusers loading; docs examples use `LTXPipeline` and `LTXImageToVideoPipeline` | The main model card is tagged image-to-video and includes a broad model/workflow table for 0.9.6, 0.9.8, FP8, distilled, and upscaler assets. |
| `Lightricks/LTX-Video-0.9.5` | `LTXConditionPipeline`, LoRA example, text/image/video conditioning | A good documented target when you need the condition pipeline and stable Diffusers examples. |
| `Lightricks/LTX-Video-0.9.7-dev` | 13B development model with spatial latent upscaler workflow | Diffusers notes say 0.9.7 includes a spatial latent upscaler and 13B transformer. |
| `Lightricks/LTX-Video-0.9.7-distilled` | Faster distilled workflow | Set `guidance_scale=1.0`; use roughly 4-10 steps and the custom timesteps documented by Diffusers. |
| `Lightricks/LTX-Video-0.9.8-13B-distilled` | 13B distilled, long video capable | Diffusers notes say it is similar to 0.9.7 distilled, supports very long videos, and recommends `tone_map_compression_ratio=0.6` during latent upsampling. |
| `Lightricks/ltxv-spatial-upscaler-0.9.7` | Latent upsampler paired with 0.9.7 examples | Loaded with `LTXLatentUpsamplePipeline.from_pretrained(..., vae=pipe.vae)`. |
| `Lightricks/ltxv-spatial-upscaler-0.9.8` | Latent upsampler paired with the main model card's 0.9.8 examples | The main model card uses this in its 0.9.8 Diffusers examples. |
| `a-r-r-o-w/LTX-0.9.8-Latent-Upsampler` | 0.9.8 latent upsampler model in Diffusers API example | The docs include a TODO saying this checkpoint should be updated once available in the LTX organization. |
| `Lightricks/LTX-Video-Cakeify-LoRA` | LoRA example with `LTXConditionPipeline` | Trigger word in the docs example is `CAKEIFY`. |
| `city96/LTX-Video-gguf` | GGUF single-file loading example | Use `AutoModel.from_single_file(..., quantization_config=GGUFQuantizationConfig(...))` for the transformer. |

The main Lightricks model card also lists named original workflow/checkpoint
families such as `ltxv-13b-0.9.8-dev`, `ltxv-13b-0.9.8-distilled`,
`ltxv-2b-0.9.8-distilled`, and FP8 variants. Those names appear as original
workflow/config assets, while the Diffusers API examples mostly use
Diffusers-compatible model repositories or explicit single-file loading.

## 3. Installation

Use a recent Diffusers build that includes the LTX classes. The official model
cards often recommend installing from GitHub for the newest examples.

```powershell
.venv\Scripts\python.exe -m pip install -U diffusers transformers accelerate torch imageio imageio-ffmpeg
```

For newest LTX examples before a release:

```powershell
.venv\Scripts\python.exe -m pip install -U git+https://github.com/huggingface/diffusers
```

Recommended dtype from the Diffusers notes is `torch.bfloat16` for the
transformer, VAE, and text encoder. The notes say the VAE and text encoder can
also be `torch.float32` or `torch.float16`.

## 4. Text-To-Video With `LTXPipeline`

Use `LTXPipeline` when the request is prompt-only. The API signature defaults
to `height=512`, `width=704`, `num_frames=161`, `frame_rate=25`,
`num_inference_steps=50`, `guidance_scale=3`, and `output_type="pil"`, but the
docs also say LTX often works best with explicit sizes such as `height=480` and
`width=848`.

```python
import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

pipe = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

prompt = (
    "A small research rover drives across a red desert at sunrise. Dust trails "
    "behind its wheels, a low camera follows from the side, and distant cliffs "
    "catch warm orange light."
)
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

frames = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=768,
    height=512,
    num_frames=161,
    num_inference_steps=50,
    guidance_scale=3.0,
    decode_timestep=0.03,
    decode_noise_scale=0.025,
    generator=torch.Generator("cuda").manual_seed(0),
).frames[0]

export_to_video(frames, "ltx_t2v.mp4", fps=24)
```

Important `LTXPipeline` parameters:

- `prompt` or `prompt_embeds`: exactly one text source is required.
- `negative_prompt` or `negative_prompt_embeds`: optional negative guidance.
- `height`, `width`: choose explicit values; make them divisible by 32.
- `num_frames`: decoded frame count. Prefer `8 * n + 1` frame counts, such as
  `121`, `161`, or `257`, to match LTX temporal compression expectations.
- `frame_rate`: passed to the model for temporal coordinates; export FPS is
  still controlled by `export_to_video(..., fps=...)`.
- `num_inference_steps` or `timesteps`: use custom descending timesteps for
  distilled checkpoints when the docs specify them.
- `guidance_scale`: CFG is enabled above 1. Distilled models often require
  `1.0`.
- `guidance_rescale`: optional CFG rescale to reduce overexposure.
- `decode_timestep` and `decode_noise_scale`: VAE decode-time controls for
  timestep-aware VAE variants.
- `output_type`: usually `"pil"` for export, but `"np"` or `"pt"` can be
  useful for server-side postprocessing.
- `callback_on_step_end`: use for progress telemetry if the workflow engine
  streams job events.

## 5. Image-To-Video With `LTXImageToVideoPipeline`

Use `LTXImageToVideoPipeline` when the user supplies one image and expects the
clip to grow from that visual anchor. The class has nearly the same parameters
as `LTXPipeline`, with an additional required `image`.

```python
import torch
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

pipe = LTXImageToVideoPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

image = load_image("input.png")
prompt = (
    "A person in a yellow raincoat stands on a city street at night. Reflections "
    "ripple across wet pavement as the camera slowly dollies forward."
)
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

frames = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=704,
    height=480,
    num_frames=161,
    num_inference_steps=50,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]

export_to_video(frames, "ltx_i2v.mp4", fps=24)
```

Use this class for the simple one-image path. If you need multiple images,
video segments, arbitrary frame placement, per-condition strength, or
video-to-video editing, use `LTXConditionPipeline` instead.

## 6. Conditions With `LTXConditionPipeline`

`LTXConditionPipeline` is the broadest and usually most integration-friendly
pipeline on the page. It supports:

- text-only generation by omitting `conditions`, `image`, and `video`
- image-to-video using one or more `LTXVideoCondition(image=..., frame_index=...)`
- video-to-video or clip continuation with `LTXVideoCondition(video=..., frame_index=...)`
- multiple condition items at different frame indices
- latent editing/refinement with `latents` and `denoise_strength`

`LTXVideoCondition` is a small dataclass:

```python
LTXVideoCondition(
    image=None,       # PIL image, optional
    video=None,       # list of PIL frames, optional
    frame_index=0,    # target frame for the image or first video frame
    strength=1.0,     # conditioning strength
)
```

The `conditions` list is mutually exclusive with the shorthand `image`,
`video`, `frame_index`, and `strength` inputs. If you provide `conditions`,
do not also pass `image` or `video`.

```python
import torch
from diffusers import LTXConditionPipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_image, load_video

pipe = LTXConditionPipeline.from_pretrained(
    "Lightricks/LTX-Video-0.9.5",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
pipe.vae.enable_tiling()

start_image = load_image("first_frame.png")
reference_clip = load_video("reference_motion.mp4")[:33]

conditions = [
    LTXVideoCondition(image=start_image, frame_index=0, strength=1.0),
    LTXVideoCondition(video=reference_clip, frame_index=80, strength=0.75),
]

prompt = (
    "A coastal road trip begins at sunrise and transitions into a snowy mountain "
    "pass. The camera movement is smooth and cinematic, with realistic motion."
)
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

frames = pipe(
    conditions=conditions,
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=768,
    height=512,
    num_frames=161,
    num_inference_steps=40,
    guidance_scale=3.0,
    image_cond_noise_scale=0.025,
    decode_timestep=0.05,
    decode_noise_scale=0.025,
    generator=torch.Generator("cuda").manual_seed(0),
).frames[0]

export_to_video(frames, "ltx_conditioned.mp4", fps=24)
```

Key condition parameters:

- `conditions`: one `LTXVideoCondition` or a list of them.
- `image` / `video`: shorthand inputs used only when `conditions` is not
  provided.
- `frame_index`: where the image or first video frame is placed in the output.
- `strength`: condition strength. Use lower values when the prompt should
  diverge more from the condition.
- `image_cond_noise_scale`: adds noise to hard-conditioning latents to improve
  motion continuity, especially for single-frame conditions.
- `denoise_strength`: edit/refinement strength when passing existing `latents`;
  higher values move farther away from the input latents.
- `max_sequence_length`: defaults to `256` for `LTXConditionPipeline`, unlike
  `128` on `LTXPipeline` and `LTXImageToVideoPipeline`.

The condition pipeline includes `trim_conditioning_sequence(start_frame,
sequence_num_frames, target_num_frames)`, which trims video conditions to fit
inside the target frame count and aligns them to the VAE temporal compression.

## 7. Video-To-Video And Latent Refinement

For video-to-video, encode or condition on a short input clip and use
`denoise_strength` to control how strongly the generated result departs from
the source. The official Lightricks examples commonly use a lower-resolution
latent pass, then upsample and refine.

```python
import torch
from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_video

pipe = LTXConditionPipeline.from_pretrained(
    "Lightricks/LTX-Video-0.9.7-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")
pipe.vae.enable_tiling()

pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
    "Lightricks/ltxv-spatial-upscaler-0.9.7",
    vae=pipe.vae,
    torch_dtype=torch.bfloat16,
).to("cuda")

source = load_video("input_clip.mp4")[:21]
condition = LTXVideoCondition(video=source, frame_index=0, strength=1.0)

prompt = "A winding mountain road covered in snow, filmed from a smooth aerial camera."
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

low_h, low_w = 512, 768
num_frames = 161

latents = pipe(
    conditions=[condition],
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=low_w,
    height=low_h,
    num_frames=num_frames,
    num_inference_steps=30,
    decode_timestep=0.05,
    decode_noise_scale=0.025,
    image_cond_noise_scale=0.0,
    guidance_scale=5.0,
    guidance_rescale=0.7,
    output_type="latent",
    generator=torch.Generator("cuda").manual_seed(0),
).frames

upscaled_latents = pipe_upsample(
    latents=latents,
    output_type="latent",
).frames

frames = pipe(
    conditions=[condition],
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=low_w * 2,
    height=low_h * 2,
    num_frames=num_frames,
    latents=upscaled_latents,
    denoise_strength=0.4,
    num_inference_steps=10,
    decode_timestep=0.05,
    decode_noise_scale=0.025,
    image_cond_noise_scale=0.0,
    guidance_scale=5.0,
    guidance_rescale=0.7,
    output_type="pil",
    generator=torch.Generator("cuda").manual_seed(0),
).frames[0]

export_to_video(frames, "ltx_v2v_refined.mp4", fps=24)
```

The official Diffusers page has a small variable-name typo in one 0.9.7
upscaler example: it creates `pipeline_upsample` but later calls
`pipe_upsample`. Keep the variable name consistent in real code.

## 8. Long I2V With `LTXI2VLongMultiPromptPipeline`

`LTXI2VLongMultiPromptPipeline` is documented as a long-duration I2V pipeline
with ComfyUI parity. Its distinguishing features are temporal sliding-window
sampling, multi-prompt segmentation per window, first-frame hard conditioning,
and VRAM control through temporal windowing plus tiled VAE decoding.

The call signature is different from the base pipelines. It defaults to
`guidance_scale=1.0`, `num_inference_steps=8`, `output_type="latent"`,
`temporal_tile_size=80`, `temporal_overlap=24`, `decode_horizontal_tiles=4`,
`decode_vertical_tiles=4`, and `decode_overlap=3`.

```python
import torch
from diffusers import LTXEulerAncestralRFScheduler, LTXI2VLongMultiPromptPipeline

pipe = LTXI2VLongMultiPromptPipeline.from_pretrained(
    "Lightricks/LTX-Video-0.9.8-13B-distilled",
    torch_dtype=torch.bfloat16,
)
pipe.scheduler = LTXEulerAncestralRFScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

out = pipe(
    prompt="a rover crosses a desert | the rover enters a glass research dome",
    negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
    num_frames=241,
    height=512,
    width=704,
    temporal_tile_size=80,
    temporal_overlap=24,
    guidance_scale=1.0,
    output_type="pil",
    return_dict=True,
)

frames = out.frames[0]
```

Prompt scheduling options:

- A single prompt containing `|` is split into per-window prompt parts.
- `prompt_segments` can override prompts per window with dictionaries like
  `{"start_window": 0, "end_window": 2, "text": "..."}`.
- `cond_image` and `cond_strength` are the first-frame conditioning path for
  this long pipeline.
- `seed` has special behavior: the docs say global latents are seeded once,
  while each window can use `seed + w_start`.

Long-pipeline output handling:

```python
latent_out = pipe(
    prompt="a slow push through a futuristic library",
    output_type="latent",
    return_dict=True,
).frames

frames = pipe.vae_decode_tiled(
    latent_out,
    output_type="pil",
    horizontal_tiles=4,
    vertical_tiles=4,
    overlap=3,
    decode_timestep=0.05,
    decode_noise_scale=0.025,
)[0]
```

This lets a workflow return latents from the sampling stage, then decode later
or on a different memory profile.

## 9. Latent Upscaling With `LTXLatentUpsamplePipeline`

`LTXLatentUpsamplePipeline` accepts either decoded `video` input or `latents`.
The LTX docs use it mainly with latent output from `LTXConditionPipeline`.
The documented upsampler increases latent height and width by 2x.

Important call parameters:

- `video`: input video frames if not passing `latents`.
- `latents`: latent tensor from another LTX pipeline.
- `height`, `width`: used when preparing latents from video input.
- `decode_timestep`, `decode_noise_scale`: VAE decode controls.
- `adain_factor`: applies Adaptive Instance Normalization against reference
  latents; useful for style/statistics consistency.
- `tone_map_compression_ratio`: 0.9.8 distilled examples recommend `0.6`.
- `output_type`: use `"latent"` when feeding the result back into
  `LTXConditionPipeline` for the denoise/refine stage.

The class also exposes VAE slicing and tiling helpers:
`enable_vae_slicing()`, `disable_vae_slicing()`, `enable_vae_tiling()`, and
`disable_vae_tiling()`.

## 10. Decoding, Tiling, And Noise Controls

LTX has several similarly named controls. Keep them separate in the workflow
schema and UI.

| Parameter | Where it appears | Purpose |
| --- | --- | --- |
| `decode_timestep` | All documented LTX generation/upsample calls | Decode-time timestep for timestep-aware VAE variants. |
| `decode_noise_scale` | All documented LTX generation/upsample calls | Interpolation/noise mix during decode at `decode_timestep`. |
| `image_cond_noise_scale` | `LTXConditionPipeline` | Adds timestep-dependent noise to hard-conditioning latents for motion continuity. |
| `decode_horizontal_tiles` | `LTXI2VLongMultiPromptPipeline` | Number of horizontal VAE decode tiles. |
| `decode_vertical_tiles` | `LTXI2VLongMultiPromptPipeline` | Number of vertical VAE decode tiles. |
| `decode_overlap` | `LTXI2VLongMultiPromptPipeline` | Latent-pixel overlap between decode tiles. |
| `pipe.vae.enable_tiling()` | VAE helper | Enables tiled VAE encode/decode for memory reduction. |
| `vae_decode_tiled(...)` | Long pipeline helper | Explicit tiled decoder with feathered tile blending, last-frame fix, optional `auto_denormalize`, and `compute_dtype=torch.float32` by default to reduce blur/color shifts. |

Diffusers notes recommend the following checkpoint-specific settings:

- Use `torch.bfloat16` for transformer, VAE, and text encoder when possible.
- For guidance-distilled variants, set `guidance_scale=1.0`.
- For non-distilled variants, use a higher value such as `guidance_scale=5.0`
  when quality is poor.
- For timestep-aware VAE variants, especially LTX-Video 0.9.1 and above, set
  `decode_timestep=0.05` and tune condition/decode noise around `0.025`.
- For 0.9.5 and above multiple image/video conditioning, use visually similar
  conditioning media when interpolating; very different inputs can cause abrupt
  transitions.

## 11. Distilled Timesteps

The Diffusers page documents custom timesteps for the 0.9.7 distilled model.
Use them instead of just lowering `num_inference_steps`.

Base pass before upscaling:

```python
timesteps = [1000, 993, 987, 981, 975, 909, 725, 0.03]
```

Upscale/refinement pass:

```python
timesteps = [1000, 909, 725, 421, 0]
```

For 0.9.7 distilled:

- set `guidance_scale=1.0`
- use about 4-10 steps
- use the custom timesteps above for best documented results

For 0.9.8 13B distilled:

- the docs say it is similar to 0.9.7 distilled
- it supports very long videos
- use `tone_map_compression_ratio=0.6` in the latent upsampler
- the docs example uses the same custom timestep style

## 12. Memory, Performance, And Quantization

The Diffusers page includes a memory-focused example that says the LTX model
shown requires about 10 GB of VRAM. It combines:

- BF16 components
- FP8 layerwise weight casting on the transformer
- group offloading for the transformer, text encoder, and VAE
- VAE tiling for larger outputs

Memory-oriented pattern:

```python
import torch
from diffusers import AutoModel, LTXPipeline
from diffusers.hooks import apply_group_offloading

transformer = AutoModel.from_pretrained(
    "Lightricks/LTX-Video",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)
transformer.enable_layerwise_casting(
    storage_dtype=torch.float8_e4m3fn,
    compute_dtype=torch.bfloat16,
)

pipe = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)

onload_device = torch.device("cuda")
offload_device = torch.device("cpu")

pipe.transformer.enable_group_offload(
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="leaf_level",
    use_stream=True,
)
apply_group_offloading(
    pipe.text_encoder,
    onload_device=onload_device,
    offload_type="block_level",
    num_blocks_per_group=2,
)
apply_group_offloading(pipe.vae, onload_device=onload_device, offload_type="leaf_level")
```

Other practical options:

- `pipe.enable_model_cpu_offload()` is useful for a simpler low-memory mode.
  The LTX pipelines define CPU offload order around text encoder, transformer,
  and VAE.
- `pipe.vae.enable_tiling()` should be enabled by default for larger
  resolutions and video-to-video workflows.
- Long clips should use `LTXI2VLongMultiPromptPipeline` temporal windowing
  instead of trying to denoise every frame as one huge sequence.
- Distilled checkpoints are the main speed path. 0.9.7/0.9.8 distilled
  examples use low step counts and `guidance_scale=1.0`.
- FP8 original LTX variants are listed on the main Lightricks model card.
- GGUF transformer loading is documented with `AutoModel.from_single_file` and
  `GGUFQuantizationConfig`.

GGUF single-file pattern:

```python
import torch
from diffusers import AutoModel, GGUFQuantizationConfig, LTXPipeline

transformer = AutoModel.from_single_file(
    "https://huggingface.co/city96/LTX-Video-gguf/blob/main/ltx-video-2b-v0.9-Q3_K_S.gguf",
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)

pipe = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
```

## 13. LoRA

The Diffusers page documents LoRA support through `load_lora_weights()`. The
example uses `LTXConditionPipeline` with `Lightricks/LTX-Video-Cakeify-LoRA`.

```python
import torch
from diffusers import LTXConditionPipeline
from diffusers.utils import export_to_video, load_image

pipe = LTXConditionPipeline.from_pretrained(
    "Lightricks/LTX-Video-0.9.5",
    torch_dtype=torch.bfloat16,
).to("cuda")

pipe.load_lora_weights("Lightricks/LTX-Video-Cakeify-LoRA", adapter_name="cakeify")
pipe.set_adapters("cakeify")

image = load_image("input.png")
frames = pipe(
    prompt="CAKEIFY a person using a knife to cut a cake shaped like a toy",
    image=image,
    width=576,
    height=576,
    num_frames=161,
    decode_timestep=0.03,
    decode_noise_scale=0.025,
    num_inference_steps=50,
).frames[0]

export_to_video(frames, "ltx_lora.mp4", fps=26)
```

## 14. Output Handling

All documented LTX pipelines return `LTXPipelineOutput` when
`return_dict=True`.

```python
result = pipe(prompt="a quiet forest path", return_dict=True)
frames = result.frames
```

`LTXPipelineOutput.frames` can be:

- `list[list[PIL.Image.Image]]`, usually indexed as `.frames[0]`
- `np.ndarray`
- `torch.Tensor` with shape `(batch_size, num_frames, channels, height, width)`
- normalized latent tensor when `output_type="latent"`

When `return_dict=False`, the pipeline returns a tuple and the first element is
the generated frames/images. For UI and API stability, prefer
`return_dict=True` internally and normalize the response in one place.

Export examples:

```python
from diffusers.utils import export_to_video

export_to_video(result.frames[0], "output.mp4", fps=24)
```

If a workflow returns latents for deferred decode, store metadata with the
latent artifact:

- source model id
- pipeline class
- `height`, `width`, `num_frames`
- VAE spatial and temporal compression ratios
- decode parameters
- random seed or generator seed

## 15. Gotchas

- `height` and `width` should be divisible by 32. The condition pipeline source
  validates this directly.
- The Lightricks model card says LTX works on frame counts divisible by 8 plus
  1. Good defaults are `121`, `161`, and `257`.
- The same model card says LTX works best under `720 x 1280` and below `257`
  frames. Long videos should use temporal windowing.
- Prompts should be English and detailed. Prompt-following is sensitive to
  prompting style.
- `LTXConditionPipeline` can run text-only, so it can be a single unified task
  backend if you do not need the smaller `LTXPipeline` surface.
- `conditions` cannot be mixed with shorthand `image`/`video` inputs.
- `decode_noise_scale` and `image_cond_noise_scale` are different controls.
  Keep both visible in implementation code, even if the UI hides one at first.
- Distilled models are guidance-distilled. A normal `guidance_scale=3` or `5`
  can hurt distilled outputs; use `1.0` when the docs say so.
- The long multi-prompt pipeline defaults to `output_type="latent"`, unlike
  the simpler pipelines that default to `"pil"`.
- The Diffusers long-pipeline example omits the `Lightricks/` namespace in
  one `from_pretrained("LTX-Video-0.9.8-13B-distilled")` snippet. Use the full
  Hub id `Lightricks/LTX-Video-0.9.8-13B-distilled` for reproducible loading.
- For multiple conditioning images/videos, Diffusers notes recommend using
  similar media. Very different conditions can cause abrupt visual transitions.
- Single-file and GGUF loading replace individual components, usually the
  transformer. Keep the base repo id available for the rest of the pipeline.

## 16. Implementation Checklist

For a local workflow task, expose these fields first:

- `model_id`
- `pipeline_class`
- `prompt`
- `negative_prompt`
- `width`
- `height`
- `num_frames`
- `frame_rate`
- `num_inference_steps`
- `timesteps`
- `guidance_scale`
- `guidance_rescale`
- `seed`
- `output_type`
- `decode_timestep`
- `decode_noise_scale`
- `image_cond_noise_scale`
- `conditions`: list of `{kind, path, frame_index, strength}`
- `denoise_strength`
- `temporal_tile_size`
- `temporal_overlap`
- `decode_horizontal_tiles`
- `decode_vertical_tiles`
- `decode_overlap`
- `adain_factor`
- `tone_map_compression_ratio`
- `enable_vae_tiling`
- `enable_model_cpu_offload`
- `enable_group_offload`
- `enable_layerwise_casting`

Normalize dimensions before dispatch:

```python
def normalize_ltx_size(height: int, width: int) -> tuple[int, int]:
    return height - height % 32, width - width % 32


def normalize_ltx_frames(num_frames: int) -> int:
    return max(9, ((num_frames - 1) // 8) * 8 + 1)
```

For 0.9.7/0.9.8 distilled presets, override:

```python
guidance_scale = 1.0
decode_timestep = 0.05
decode_noise_scale = 0.025
```

and use the documented custom timestep arrays when doing the two-stage
upscale/refine flow.

## 17. Source Links

- Hugging Face Diffusers LTX-Video API:
  https://huggingface.co/docs/diffusers/api/pipelines/ltx_video
- `LTXPipeline` source:
  https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/ltx/pipeline_ltx.py
- `LTXImageToVideoPipeline` source:
  https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/ltx/pipeline_ltx_image2video.py
- `LTXConditionPipeline` and `LTXVideoCondition` source:
  https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/ltx/pipeline_ltx_condition.py
- `LTXI2VLongMultiPromptPipeline` source:
  https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/ltx/pipeline_ltx_i2v_long_multi_prompt.py
- `LTXLatentUpsamplePipeline` source:
  https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/ltx/pipeline_ltx_latent_upsample.py
- `LTXPipelineOutput` source:
  https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/ltx/pipeline_output.py
- Main Lightricks LTX-Video model card:
  https://huggingface.co/Lightricks/LTX-Video
- LTX-Video 0.9.5 model card:
  https://huggingface.co/Lightricks/LTX-Video-0.9.5
- LTX-Video 0.9.7 distilled model card:
  https://huggingface.co/Lightricks/LTX-Video-0.9.7-distilled
- LTX-Video 0.9.8 13B distilled model card:
  https://huggingface.co/Lightricks/LTX-Video-0.9.8-13B-distilled
- LTX spatial upscaler 0.9.7 model card:
  https://huggingface.co/Lightricks/ltxv-spatial-upscaler-0.9.7
