# LTX-2 Diffusers Implementation Guide

Last checked: 2026-06-17 against the Hugging Face Diffusers LTX-2 API page,
the linked `main`/`v0.38.0` Diffusers source, and official Hugging Face model
cards for the documented checkpoints.

LTX-2 is a DiT-based video-and-audio generation family from Lightricks. The
Diffusers integration is unusual for a video pipeline because the main
generation path denoises video latents and audio latents together, then decodes
the video through the LTX-2 VAE and the audio through an audio VAE plus vocoder.
The official page documents text-to-video, image-to-video, arbitrary
image/video condition insertion, latent upsampling for two-stage generation,
multimodal guidance, prompt enhancement, and the shared `LTX2PipelineOutput`.

Official entry points:

- Pipeline docs: <https://huggingface.co/docs/diffusers/api/pipelines/ltx2>
- Pipeline docs source: <https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/ltx2.md>
- Package source: <https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/ltx2>
- Text-to-video source: <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ltx2/pipeline_ltx2.py>
- Image-to-video source: <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ltx2/pipeline_ltx2_image2video.py>
- Condition pipeline source: <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ltx2/pipeline_ltx2_condition.py>
- Latent upsample source: <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ltx2/pipeline_ltx2_latent_upsample.py>
- Output class source: <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ltx2/pipeline_output.py>
- LTX-2 paper: <https://huggingface.co/papers/2601.03233>
- Original LTX-2 codebase: <https://github.com/Lightricks/LTX-2>

## 1. Executive Summary

Use `LTX2Pipeline` for prompt-only text-to-video with synchronized audio. Use
`LTX2ImageToVideoPipeline` when the first frame should come from an input image.
Use `LTX2ConditionPipeline` when conditions must be inserted at arbitrary latent
indices, such as first-frame plus last-frame generation or mixed image/video
conditions. Use `LTX2LatentUpsamplePipeline` between Stage 1 and Stage 2 when
following the documented two-stage recipe.

Practical integration answer:

- Load a Diffusers-format checkpoint with `torch_dtype=torch.bfloat16`.
- For the official two-stage recipe, run Stage 1 with `output_type="latent"`,
  upsample the video latents with `LTX2LatentUpsamplePipeline`, then pass the
  upsampled video latents and the original Stage 1 `audio_latents` into Stage 2.
- Use the documented default shape family: `width=768`, `height=512`,
  `num_frames=121`, and `frame_rate=24.0`. Several source docstrings say best
  results are at 848x480, but the API examples consistently use 768x512.
- Use `num_inference_steps=40` and `guidance_scale=4.0` for the non-distilled
  LTX-2 examples. Use `DISTILLED_SIGMA_VALUES` with 8 steps for the distilled
  checkpoint and `STAGE_2_DISTILLED_SIGMA_VALUES` with 3 steps for Stage 2.
- For LTX-2.3 examples, use multimodal guidance:
  `guidance_scale=3.0`, `stg_scale=1.0`, `modality_scale=3.0`,
  `guidance_rescale=0.7`, `audio_guidance_scale=7.0`,
  `audio_stg_scale=1.0`, `audio_modality_scale=3.0`,
  `audio_guidance_rescale=0.7`, `spatio_temporal_guidance_blocks=[28]`, and
  `use_cross_timestep=True`.
- Enable `pipe.vae.enable_tiling()` before decoding larger videos. The official
  docs call this "usually necessary" to avoid out-of-memory errors during VAE
  decode.
- Save the output with `diffusers.pipelines.ltx2.export_utils.encode_video`,
  passing both `video[0]` and `audio[0].float().cpu()` plus
  `audio_sample_rate=pipe.vocoder.config.output_sampling_rate`.

## 2. Pipeline Selection

| Class | Main use | Extra inputs | Output |
| --- | --- | --- | --- |
| `LTX2Pipeline` | Prompt-only text-to-video plus audio | Optional `latents`, `audio_latents`, prompt enhancement fields | `LTX2PipelineOutput(frames, audio)` or `(video, audio)` |
| `LTX2ImageToVideoPipeline` | First-frame image-to-video plus audio | `image`, optional `latents`, `audio_latents`, prompt enhancement fields | `LTX2PipelineOutput(frames, audio)` or `(video, audio)` |
| `LTX2ConditionPipeline` | Image/video conditions at arbitrary latent indices | `conditions` containing `LTX2VideoCondition` instances | `LTX2PipelineOutput(frames, audio)` or `(video, audio)` |
| `LTX2LatentUpsamplePipeline` | Stage-1 video latent or video upsampling before Stage 2 | `latents` or `video`; `LTX2LatentUpsamplerModel` | Video only, returned as frames or latents |
| `LTX2PipelineOutput` | Shared return object for LTX-2 generation pipelines | `frames`, `audio` | Dataclass-like `BaseOutput` |

The LTX-2 package source also exports `LTX2InContextPipeline` and
`LTX2HDRPipeline`. They are not autodoc sections on the official LTX-2 API page
checked for this guide, but the package source and LTX-2.3 model card document
them as IC-LoRA and HDR IC-LoRA variants. Treat them as source/model-card
features rather than the baseline API-page surface.

## 3. Installation and Runtime Assumptions

The LTX-2 docs are on the Diffusers `main` documentation page. In practice, use
a recent Diffusers build that contains `diffusers.pipelines.ltx2`.

```shell
pip install -U diffusers transformers accelerate safetensors imageio
```

For the newest classes or model-card examples, the model cards sometimes show
installing Diffusers from GitHub:

```shell
pip install git+https://github.com/huggingface/diffusers
```

Use CUDA and BF16 for the documented examples:

```python
import torch
from diffusers import LTX2Pipeline

pipe = LTX2Pipeline.from_pretrained(
    "Lightricks/LTX-2",
    torch_dtype=torch.bfloat16,
)
pipe.enable_sequential_cpu_offload(device="cuda")
pipe.vae.enable_tiling()
```

The model repos can be large. Authenticate with Hugging Face if a model is gated
or if your environment needs a token for downloads:

```shell
huggingface-cli login
```

## 4. Model IDs and Checkpoints

Documented or official IDs to know:

| Model ID | Where it appears | Use |
| --- | --- | --- |
| `Lightricks/LTX-2` | LTX-2 API page | Main full LTX-2 checkpoint, latent upsampler subfolder, and Stage 2 distilled LoRA weights |
| `rootonchair/LTX-2-19b-distilled` | LTX-2 API page | Fast distilled two-stage example with `DISTILLED_SIGMA_VALUES` |
| `dg845/LTX-2.3-Diffusers` | Rendered LTX-2 docs and model card lineage | LTX-2.3 Diffusers checkpoint used in multimodal guidance and prompt-enhancement examples on the rendered page |
| `diffusers/LTX-2.3-Diffusers` | Current raw docs source and model card | Duplicate or maintained Diffusers-format LTX-2.3 checkpoint; the raw docs source currently uses this ID |
| `diffusers/LTX-2.3-Distilled-Diffusers` | Official model card examples/source-only HDR example | Distilled LTX-2.3 Diffusers-format checkpoint |
| `Lightricks/LTX-2.3` and `Lightricks/LTX-2.3-fp8` | `diffusers/LTX-2.3-Diffusers` model card | Original Lightricks release and FP8 variant |
| `Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In` | `LTX2InContextPipeline` source example | IC-LoRA camera-control adapter |
| `Lightricks/LTX-2.3-22b-IC-LoRA-HDR` | `LTX2HDRPipeline` source example | HDR IC-LoRA adapter |

The docs page and raw docs source were slightly out of sync at check time for
the LTX-2.3 model ID: the rendered page showed `dg845/LTX-2.3-Diffusers`, while
the raw docs source used `diffusers/LTX-2.3-Diffusers`. Check the model card you
intend to deploy and pin the ID in configuration.

## 5. Shared Components and Mental Model

The main LTX-2 pipelines contain these conceptual components:

| Component | Role |
| --- | --- |
| `tokenizer` and `text_encoder` | Encode prompts. The prompt enhancement path uses a Gemma 3 style processor/text encoder path when available. |
| `connectors` | Project text embeddings into video and audio conditioning streams. |
| `transformer` | Joint denoiser for video latents and audio latents. |
| `scheduler` | Flow-match scheduler for video denoising. Some pipelines duplicate or provide an audio scheduler for audio latents. |
| `vae` | LTX-2 video VAE. Decodes video latents to frames and supports optional timestep-conditioned decode. |
| `audio_vae` | Decodes audio latents to mel spectrograms. |
| `vocoder` | Converts generated mel spectrograms to waveform audio. |
| `video_processor` | Preprocesses and postprocesses videos/images. |
| `latent_upsampler` | Used only by `LTX2LatentUpsamplePipeline` to upscale Stage 1 video latents. |

The generation loop prepares text embeddings, creates video and audio latent
positions, denoises both modalities together, applies optional CFG, STG, and
modality isolation guidance, then decodes video and audio separately. For
`output_type="latent"`, the pipeline skips VAE/vocoder decode and returns
denormalized video latents plus unpacked audio latents.

## 6. Text-to-Video with `LTX2Pipeline`

`LTX2Pipeline` is the prompt-only text-to-video entry point. Its important call
arguments are:

| Parameter | Default | Notes |
| --- | --- | --- |
| `prompt`, `negative_prompt` | `None` | Provide text, or precomputed prompt embeddings. The default negative prompt constant is in `diffusers.pipelines.ltx2.utils.DEFAULT_NEGATIVE_PROMPT`. |
| `height`, `width` | `512`, `768` | Must be divisible by the VAE spatial compression ratio, effectively 32 in source checks. |
| `num_frames` | `121` | Best used as `8 * k + 1` for the LTX-2 temporal compression pattern. |
| `frame_rate` | `24.0` | Used for output FPS and temporal positional encoding. |
| `num_inference_steps` | `40` | Standard non-distilled default. Distilled examples use explicit sigma schedules. |
| `sigmas`, `timesteps` | `None` | Custom scheduler schedule. Do not pass both. |
| `guidance_scale` | `4.0` | Video CFG scale. CFG is active when greater than 1. |
| `stg_scale` | `0.0` | Video spatio-temporal guidance. Requires `spatio_temporal_guidance_blocks` when enabled. |
| `modality_scale` | `1.0` | Video modality isolation guidance. `1.0` disables it. |
| `audio_guidance_scale` | `None` | Falls back to `guidance_scale`. LTX-2.3 examples recommend higher audio CFG, such as 7.0. |
| `audio_stg_scale` | `None` | Falls back to `stg_scale`. |
| `audio_modality_scale` | `None` | Falls back to `modality_scale`. |
| `audio_guidance_rescale` | `None` | Falls back to `guidance_rescale`. |
| `spatio_temporal_guidance_blocks` | `None` | Official notes: `[29]` for LTX-2.0, `[28]` for LTX-2.3. |
| `latents`, `audio_latents` | `None` | Use for Stage 2, reruns, or continuation from prior latent outputs. |
| `decode_timestep`, `decode_noise_scale` | `0.0`, `None` | Used only when the VAE config has timestep conditioning. |
| `use_cross_timestep` | `False` | Use `True` for newer LTX-2.3 behavior. |
| `system_prompt` | `None` | Enables prompt enhancement when supplied. |
| `output_type` | `"pil"` | Use `"np"` for `encode_video` examples or `"latent"` for two-stage workflows. |
| `return_dict` | `True` | `False` returns `(video, audio)`. |

Minimal non-distilled text-to-video:

```python
import torch
from diffusers import LTX2Pipeline
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT

device = "cuda"
frame_rate = 24.0

pipe = LTX2Pipeline.from_pretrained(
    "Lightricks/LTX-2",
    torch_dtype=torch.bfloat16,
)
pipe.enable_sequential_cpu_offload(device=device)
pipe.vae.enable_tiling()

video, audio = pipe(
    prompt="A quiet city street after rain, reflections shimmering as cars pass.",
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    width=768,
    height=512,
    num_frames=121,
    frame_rate=frame_rate,
    num_inference_steps=40,
    guidance_scale=4.0,
    output_type="np",
    return_dict=False,
)

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_t2v.mp4",
)
```

For LTX-2.3, use the multimodal guidance defaults shown in the docs:

```python
video, audio = pipe(
    prompt=prompt,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    width=768,
    height=512,
    num_frames=121,
    frame_rate=24.0,
    num_inference_steps=30,
    guidance_scale=3.0,
    stg_scale=1.0,
    modality_scale=3.0,
    guidance_rescale=0.7,
    audio_guidance_scale=7.0,
    audio_stg_scale=1.0,
    audio_modality_scale=3.0,
    audio_guidance_rescale=0.7,
    spatio_temporal_guidance_blocks=[28],
    use_cross_timestep=True,
    output_type="np",
    return_dict=False,
)
```

## 7. Image-to-Video with `LTX2ImageToVideoPipeline`

`LTX2ImageToVideoPipeline` adds an `image` argument and otherwise mirrors the
text-to-video pipeline. The source encodes the input image through the VAE,
repeats the resulting latent over the latent time dimension, keeps the first
latent frame clean with a conditioning mask, and denoises the remaining latent
frames. The returned audio is still generated by the model; the image is not an
audio condition.

Important differences from `LTX2Pipeline`:

- `image` can be a PIL image, list of images, or tensor accepted by
  `PipelineImageInput`.
- The first latent frame is preserved as the clean image condition.
- During the denoising update, the source steps only the latent frames after the
  first frame, then concatenates the clean first-frame latent back in.
- Prompt enhancement is supported with `system_prompt`, and the utilities file
  includes `I2V_DEFAULT_SYSTEM_PROMPT` for image-to-video enhancement.

Basic image-to-video:

```python
import torch
from diffusers import LTX2ImageToVideoPipeline
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
from diffusers.utils import load_image

device = "cuda"
frame_rate = 24.0

pipe = LTX2ImageToVideoPipeline.from_pretrained(
    "diffusers/LTX-2.3-Diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.enable_sequential_cpu_offload(device=device)
pipe.vae.enable_tiling()

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg")

video, audio = pipe(
    image=image,
    prompt="The astronaut slowly turns toward the camera as lunar dust drifts around the suit.",
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    width=768,
    height=512,
    num_frames=121,
    frame_rate=frame_rate,
    num_inference_steps=30,
    guidance_scale=3.0,
    stg_scale=1.0,
    modality_scale=3.0,
    guidance_rescale=0.7,
    audio_guidance_scale=7.0,
    audio_stg_scale=1.0,
    audio_modality_scale=3.0,
    audio_guidance_rescale=0.7,
    spatio_temporal_guidance_blocks=[28],
    use_cross_timestep=True,
    output_type="np",
    return_dict=False,
)

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_i2v.mp4",
)
```

## 8. Arbitrary Conditions with `LTX2ConditionPipeline`

`LTX2ConditionPipeline` accepts `conditions`, a single `LTX2VideoCondition` or a
list of them. This is the page-documented way to do first-frame/last-frame
generation, insert image conditions at arbitrary points, or mix image and video
conditions.

`LTX2VideoCondition` fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `frames` | PIL image, list of PIL images, NumPy array, or torch tensor | Condition image or video frames. The source converts them to `(1, C, F, H, W)` video tensors. |
| `index` | `int` | Latent-frame index where the condition starts. Negative indices are supported, so `-1` targets the last latent index. |
| `strength` | `float`, default `1.0` | Conditioning strength. `1.0` keeps the condition fully clean; values between 0 and 1 blend condition strength with denoising. |

Condition preprocessing details from the source:

- Frames are converted to RGB, resized so the longer side fills the target
  resolution, center-cropped, then mapped from `[0, 255]` to `[-1, 1]`.
- `index` is interpreted in latent-frame space, not pixel-frame space.
- Negative indices are normalized against the latent-frame count.
- Conditions starting beyond the latent-frame range are skipped with a warning.
- Condition videos are trimmed to fit the requested output and to follow the
  temporal compression pattern, effectively `k * 8 + 1` frames.
- A condition at latent index `0` replaces first-frame tokens directly.
- A condition at a nonzero latent index is appended as keyframe tokens with
  extra positional coordinates; after denoising, those appended keyframe tokens
  are removed from the final latent before decode.
- The official docs warn that image conditions correspond to the 8 data-space
  frames associated with the selected latent frame, so an inserted image
  condition may appear static over that region.

First-frame plus last-frame example:

```python
import torch
from diffusers import LTX2ConditionPipeline
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT, DISTILLED_SIGMA_VALUES
from diffusers.utils import load_image

device = "cuda"
frame_rate = 24.0
generator = torch.Generator(device).manual_seed(42)

pipe = LTX2ConditionPipeline.from_pretrained(
    "rootonchair/LTX-2-19b-distilled",
    torch_dtype=torch.bfloat16,
)
pipe.enable_sequential_cpu_offload(device=device)
pipe.vae.enable_tiling()

first_image = load_image("first_frame.png")
last_image = load_image("last_frame.png")

conditions = [
    LTX2VideoCondition(frames=first_image, index=0, strength=1.0),
    LTX2VideoCondition(frames=last_image, index=-1, strength=1.0),
]

video, audio = pipe(
    conditions=conditions,
    prompt="A small bird takes off from the ground and flies into a bright sky.",
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    width=768,
    height=512,
    num_frames=121,
    frame_rate=frame_rate,
    num_inference_steps=8,
    sigmas=DISTILLED_SIGMA_VALUES,
    guidance_scale=1.0,
    generator=generator,
    output_type="np",
    return_dict=False,
)

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_condition.mp4",
)
```

Mixed video and image conditions are also documented. Create one
`LTX2VideoCondition` from a loaded video at `index=0`, then another from an image
at a later latent index such as `index=8`.

There is no separate external-audio condition argument on the documented
`LTX2ConditionPipeline` call. Audio is generated and guided alongside video, and
you can pass `audio_latents` for Stage 2 or latent continuation, but the API page
does not document a path for conditioning on an input audio clip.

## 9. Two-Stage Generation

The official docs recommend two-stage generation for production quality.

Stage 1:

- Run a base pipeline at the target Stage 1 resolution.
- Return latents with `output_type="latent"` and `return_dict=False`.
- Capture both `video_latent` and `audio_latent`.

Latent upsample:

- Load `LTX2LatentUpsamplerModel` from the checkpoint's `latent_upsampler`
  subfolder.
- Construct `LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=...)`.
- Pass Stage 1 `video_latent` with `output_type="latent"`.
- Keep `audio_latent` unchanged for Stage 2.

Stage 2:

- Feed `latents=upscaled_video_latent` and `audio_latents=audio_latent` back into
  `LTX2Pipeline` or `LTX2ConditionPipeline`.
- Use `STAGE_2_DISTILLED_SIGMA_VALUES`.
- Use `noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0]` when following the
  non-distilled Stage 2 LoRA recipe.
- Use lower guidance, commonly `guidance_scale=1.0`.
- Enable VAE tiling before decode.

Skeleton:

```python
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2 import LTX2Pipeline, LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES

pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload(device="cuda")

video_latent, audio_latent = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=768,
    height=512,
    num_frames=121,
    frame_rate=24.0,
    num_inference_steps=40,
    guidance_scale=4.0,
    output_type="latent",
    return_dict=False,
)

latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
    "Lightricks/LTX-2",
    subfolder="latent_upsampler",
    torch_dtype=torch.bfloat16,
)
upsample_pipe = LTX2LatentUpsamplePipeline(
    vae=pipe.vae,
    latent_upsampler=latent_upsampler,
)
upsample_pipe.enable_model_cpu_offload(device="cuda")

upscaled_video_latent = upsample_pipe(
    latents=video_latent,
    output_type="latent",
    return_dict=False,
)[0]

pipe.load_lora_weights(
    "Lightricks/LTX-2",
    adapter_name="stage_2_distilled",
    weight_name="ltx-2-19b-distilled-lora-384.safetensors",
)
pipe.set_adapters("stage_2_distilled", 1.0)
pipe.vae.enable_tiling()
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
    pipe.scheduler.config,
    use_dynamic_shifting=False,
    shift_terminal=None,
)

video, audio = pipe(
    latents=upscaled_video_latent,
    audio_latents=audio_latent,
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=3,
    noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
    sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
    guidance_scale=1.0,
    output_type="np",
    return_dict=False,
)
```

For the distilled checkpoint path, use `rootonchair/LTX-2-19b-distilled`,
`DISTILLED_SIGMA_VALUES` for Stage 1, 8 Stage 1 steps, and the same
`STAGE_2_DISTILLED_SIGMA_VALUES` for Stage 2. The distilled example does not
load the Stage 2 LoRA separately because the distilled checkpoint path is already
the fast recipe shown in the docs.

## 10. `LTX2LatentUpsamplePipeline`

`LTX2LatentUpsamplePipeline` is a utility pipeline, not a prompt-conditioned
generator. It accepts either a decoded input `video` or video `latents`.

Important parameters:

| Parameter | Default | Notes |
| --- | --- | --- |
| `video` | `None` | List-like video frames to encode and upsample. Source notes batched video input is not tested/supported. |
| `latents` | `None` | Either packed `(batch, seq_len, hidden_dim)` or unpacked `(batch, latent_channels, latent_frames, latent_height, latent_width)`. |
| `latents_normalized` | `False` | Set `True` only if supplied latents are normalized by VAE latent mean/std. |
| `height`, `width` | `512`, `768` | Input video dimensions, not final decoded dimensions. Must be divisible by 32. |
| `spatial_patch_size`, `temporal_patch_size` | `1`, `1` | Needed when unpacking packed latent sequences. |
| `adain_factor` | `0.0` | Optional AdaIN blending between upsampled and original latents. Source describes range `[-10, 10]`, but the call only applies it when greater than 0. |
| `tone_map_compression_ratio` | `0.0` | Optional latent tone mapping in `[0, 1]`. |
| `output_type` | `"pil"` | Use `"latent"` between generation stages. |

Gotchas:

- Exactly one of `video` or `latents` must be provided.
- If `video` is provided and its length is not `k * 8 + 1`, the source truncates
  it to that form and logs a warning.
- The upsampler returns video only. Audio latents are not upsampled here; carry
  Stage 1 `audio_latent` into the Stage 2 generation call.
- The source currently returns `LTXPipelineOutput(frames=video)` for
  `return_dict=True`, while the LTX-2 docs page lists only `LTX2PipelineOutput`
  as the LTX-2 output class. Use `return_dict=False` in two-stage glue code if
  you want to avoid depending on that detail.

## 11. Prompt Enhancement

The official docs say LTX-2.X models are sensitive to prompt style and point to
the Lightricks prompting guide. Diffusers exposes prompt enhancement in
`LTX2Pipeline` and `LTX2ImageToVideoPipeline` through `system_prompt`.

Key pieces:

- `system_prompt` enables enhancement. If it is `None`, the original prompt is
  used directly.
- `prompt_max_new_tokens` defaults to `512`.
- `prompt_enhancement_kwargs` are passed to `text_encoder.generate`; source
  defaults include sampling with temperature `0.7`.
- `prompt_enhancement_seed` defaults to `10`.
- The optional `processor` component must exist. The docs show adding
  `Gemma3Processor.from_pretrained("google/gemma-3-12b-it-qat-q4_0-unquantized")`
  if `pipe.processor` is missing.
- The utilities file provides `T2V_DEFAULT_SYSTEM_PROMPT` and
  `I2V_DEFAULT_SYSTEM_PROMPT`. Both explicitly ask the enhancer to include
  visible action and audio descriptions, and to include exact quoted speech only
  when speech is requested.

Example:

```python
from transformers import Gemma3Processor
from diffusers.pipelines.ltx2.utils import T2V_DEFAULT_SYSTEM_PROMPT

if getattr(pipe, "processor", None) is None:
    pipe.processor = Gemma3Processor.from_pretrained(
        "google/gemma-3-12b-it-qat-q4_0-unquantized"
    )

video, audio = pipe(
    prompt=prompt,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    system_prompt=T2V_DEFAULT_SYSTEM_PROMPT,
    prompt_max_new_tokens=512,
    output_type="np",
    return_dict=False,
)
```

`LTX2ConditionPipeline` does not expose `system_prompt` or prompt enhancement
arguments in its documented/source signature checked for this guide. Pre-enhance
the prompt externally if you need enhanced prompts with arbitrary condition
insertion.

## 12. Multimodal Guidance

The LTX-2.X pipelines combine up to three guidance terms:

| Guidance | Video parameter | Audio parameter | Disabled value | Notes |
| --- | --- | --- | --- | --- |
| CFG | `guidance_scale` | `audio_guidance_scale` | `1.0` or lower for CFG effect | Audio can use a higher value than video. |
| Spatio-Temporal Guidance | `stg_scale` | `audio_stg_scale` | `0.0` | Requires `spatio_temporal_guidance_blocks`. Adds an extra denoiser pass. |
| Modality isolation guidance | `modality_scale` | `audio_modality_scale` | `1.0` | Disables cross-modality attention in the perturbed pass, then moves away from it. Adds an extra denoiser pass. |
| Guidance rescale | `guidance_rescale` | `audio_guidance_rescale` | `0.0` | Helps reduce overexposure when guidance is high. |

The official LTX-2.3 settings use video CFG 3.0 and audio CFG 7.0, STG 1.0 for
both, modality scale 3.0 for both, rescale 0.7 for both, block `[28]`, and
`use_cross_timestep=True`.

Source gotcha: the pipelines set audio values with Python `or`, such as
`audio_guidance_scale = audio_guidance_scale or guidance_scale`. Passing
`audio_guidance_scale=0.0` will therefore fall back to the video value instead
of preserving zero. Use documented defaults or positive explicit values.

## 13. Outputs and Encoding

`LTX2PipelineOutput` has:

- `frames`: video outputs. Depending on `output_type`, this may be a nested list
  of PIL frame sequences, a NumPy array, a torch tensor, or denormalized video
  latents.
- `audio`: audio outputs. The source output docstring still says `TODO`, but the
  decode path returns generated waveform audio from the vocoder for non-latent
  output, or unpacked audio latents for `output_type="latent"`.

When `return_dict=False`, the page examples consistently unpack:

```python
video, audio = pipe(..., output_type="np", return_dict=False)
```

Save video plus generated audio:

```python
from diffusers.pipelines.ltx2.export_utils import encode_video

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="sample.mp4",
)
```

For latent output:

- `video` is denormalized video latents shaped like
  `(batch, latent_channels, latent_frames, latent_height, latent_width)`.
- `audio` is unpacked audio latents, not vocoder waveform audio.
- Pass both into Stage 2 as `latents=...` and `audio_latents=...`.

## 14. Memory, Performance, and Quantization

Officially documented memory/performance practices:

- Use `torch_dtype=torch.bfloat16`.
- Use `enable_sequential_cpu_offload(device="cuda")` when memory is tight.
- Use `enable_model_cpu_offload(device="cuda")` for components such as the
  latent upsampler when the example calls for it.
- Enable VAE tiling before decode with `pipe.vae.enable_tiling()`.
- Use the distilled checkpoint and `DISTILLED_SIGMA_VALUES` for the fastest
  documented two-stage path.
- Keep `num_videos_per_prompt=1` unless you have enough VRAM for batching.
- Use `output_type="latent"` for Stage 1 to skip expensive video/audio decode
  until Stage 2.

Official checkpoint-level quantization references:

- Lightricks publishes FP8 and FP4 variants for some LTX-2 releases, and the
  LTX-2.3 model card links `Lightricks/LTX-2.3-fp8`.
- The LTX-2 API page checked for this guide does not show a Diffusers-native
  `bitsandbytes` or `torchao` quantization snippet for LTX-2. Prefer documented
  pre-quantized checkpoints or the generic Diffusers quantization docs if you
  need to go beyond BF16/offload/tiling.

## 15. Source-Only IC-LoRA and HDR Variants

The package `__init__.py` exports two additional pipelines that are not API-page
autodoc sections on the LTX-2 page checked for this guide.

`LTX2InContextPipeline`:

- Source: <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ltx2/pipeline_ltx2_ic_lora.py>
- Condition type: `LTX2ReferenceCondition(frames, strength=1.0)`.
- Main extra arguments: `reference_conditions`, `conditions`,
  `reference_downscale_factor`, `conditioning_attention_strength`, and
  `conditioning_attention_mask`.
- The reference video is encoded into extra latent tokens and concatenated to
  the noisy latent sequence. IC-LoRA adapters can use those tokens for style,
  structure, depth, pose, or camera-control conditioning.
- The source example loads
  `Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In` on top of
  `diffusers/LTX-2.3-Diffusers`.

`LTX2HDRPipeline`:

- Source: <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ltx2/pipeline_ltx2_hdr_lora.py>
- Condition type: `LTX2HDRReferenceCondition(frames, strength=1.0)`.
- Main extra arguments: `reference_conditions`, `reference_downscale_factor`,
  `connector_video_embeds`, and `connector_audio_embeds`.
- The source says the pipeline is video-only and returns `(frames, None)` when
  `return_dict=False`.
- `output_type="pt"` returns a linear HDR torch tensor shaped
  `(batch_size, num_frames, height, width, channels)`.
- The source example uses `diffusers/LTX-2.3-Distilled-Diffusers` plus
  `Lightricks/LTX-2.3-22b-IC-LoRA-HDR` and
  `encode_hdr_tensor_to_mp4`.

Because these are not on the page's autodoc list, keep them behind feature flags
or separate integration paths until your target Diffusers version is pinned.

## 16. Implementation Gotchas

- Pin the Diffusers version or commit. LTX-2 is on active `main` docs, and the
  rendered docs/model IDs may lag raw source.
- Keep `height` and `width` divisible by 32.
- Prefer `num_frames = 8 * k + 1`; the latent upsampler truncates input video to
  that form when necessary.
- Do not confuse condition `index` with a pixel frame number. It is a latent
  frame index.
- Use `LTX2ConditionPipeline` for last-frame or mid-video conditions. The I2V
  pipeline only keeps the first latent frame clean.
- Carry audio latents around explicitly in two-stage workflows. The latent
  upsampler only upscales video latents.
- For packed latents, pass the matching `height`, `width`, and `num_frames`;
  source can infer dimensions from 5D unpacked latents but not from 3D packed
  latents.
- Enable VAE tiling before Stage 2 decode, especially after latent upsampling.
- STG requires `spatio_temporal_guidance_blocks`; use `[29]` for LTX-2.0 and
  `[28]` for LTX-2.3 according to source docstrings.
- `LTX2ConditionPipeline` supports `audio_latents` but does not document
  external audio-file conditioning.
- `negative_prompt_embeds` docstrings contain a copied PixArt-Sigma note. For
  LTX-2, use the LTX-2 negative prompt constants/examples instead.
- If `output_type="latent"`, the returned `audio` is audio latents, not decoded
  waveform audio.
- For prompt enhancement, loading the Gemma 3 processor/text path can add a
  large memory cost. If the server already has its own prompt rewriting system,
  consider pre-enhancing prompts outside the LTX-2 pipeline.

## 17. Integration Checklist

1. Pick the pipeline:
   `LTX2Pipeline`, `LTX2ImageToVideoPipeline`, or `LTX2ConditionPipeline`.
2. Pick the checkpoint:
   `Lightricks/LTX-2`, `rootonchair/LTX-2-19b-distilled`, or the pinned LTX-2.3
   Diffusers ID.
3. Load with BF16, CPU offload, and VAE tiling.
4. Normalize request shapes to `width`, `height`, `num_frames`, and `frame_rate`.
5. Use LTX-2 negative prompt defaults unless the product supplies its own tested
   negative prompt.
6. Use `output_type="latent"` for Stage 1 when doing two-stage generation.
7. Upsample only video latents with `LTX2LatentUpsamplePipeline`.
8. Pass Stage 1 audio latents into Stage 2 as `audio_latents`.
9. For conditions, convert user keyframes to `LTX2VideoCondition` and document
   that `index` is latent-space.
10. Save with `encode_video`, using the pipeline vocoder's output sample rate.

## 18. Source Notes

The official docs page autodocs:

- `LTX2Pipeline`
- `LTX2ImageToVideoPipeline`
- `LTX2ConditionPipeline`
- `LTX2LatentUpsamplePipeline`
- `pipelines.ltx2.pipeline_output.LTX2PipelineOutput`

The package source additionally exports:

- `LTX2VideoCondition`
- `LTX2LatentUpsamplerModel`
- `LTX2TextConnectors`
- `LTX2Vocoder`, `LTX2VocoderWithBWE`
- `LTX2InContextPipeline`, `LTX2ReferenceCondition`
- `LTX2HDRPipeline`, `LTX2HDRReferenceCondition`

For a stable product integration, treat the API page as the public contract and
pin any source-only classes to a tested Diffusers commit or release.
