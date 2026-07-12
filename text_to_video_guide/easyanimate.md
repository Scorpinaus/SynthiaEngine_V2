# EasyAnimate Diffusers Implementation Guide

Date: 2026-06-17

Research target: EasyAnimate in Hugging Face Diffusers

Primary Diffusers page:
https://huggingface.co/docs/diffusers/api/pipelines/easyanimate

Primary Diffusers source:
https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/easyanimate

This is a docs-only implementation guide. It does not change SynthaEngine
runtime behavior.

## 1. Executive Summary

EasyAnimate is Alibaba PAI's transformer-based image and video generation
family. In Diffusers, the public EasyAnimate docs page focuses on
`EasyAnimatePipeline` for text-to-video generation and
`EasyAnimatePipelineOutput`. The official source package also exports
`EasyAnimateInpaintPipeline` and `EasyAnimateControlPipeline`, which are the
classes that map to the image/video-to-video and control-to-video checkpoint
modes described by the EasyAnimate checkpoint table and source examples.

Practical answer for SynthaEngine:

| Question | Answer |
| --- | --- |
| Main supported task | Text-to-video with `EasyAnimatePipeline`. |
| Image-to-video / video-to-video | Use `EasyAnimateInpaintPipeline` with the `InP` checkpoint and video/mask conditioning tensors. |
| Control-to-video | Use `EasyAnimateControlPipeline` with the `Control` or `Control-Camera` checkpoint, but validate the exact installed Diffusers revision because current main-source control code has source-level gotchas noted below. |
| Recommended checkpoint dtype | The Diffusers docs table recommends `torch.float16` for the official V5.1 12B checkpoints. |
| Recommended video range | EasyAnimateV5.1 supports 1 to 49 frames; the docs recommend 49 frames and exporting at 8 FPS. |
| Recommended resolution range | The docs say V5.1 T2V and I2V work from 256 to 1024 pixels in width and height; source validation requires dimensions divisible by 16. |
| Local RTX 3060 12 GB fit | Not comfortable for 12B full-quality inference. Treat as cloud/high-VRAM first, or use offload/quantization and smaller/resolution-limited smoke tests. |

## 2. Official Checkpoint Modes

The Diffusers EasyAnimate page lists official V5.1 checkpoints and recommended
inference dtype. The Alibaba PAI model cards also list 7B and 12B V5.1 variants,
but the Diffusers docs page highlights the 12B family.

| Mode | Pipeline class | Official checkpoint(s) from Diffusers docs | Dtype |
| --- | --- | --- | --- |
| Text-to-video | `EasyAnimatePipeline` | `alibaba-pai/EasyAnimateV5.1-12b-zh` | `torch.float16` |
| Video-to-video | `EasyAnimatePipeline` or `EasyAnimateInpaintPipeline`, depending on workflow | `alibaba-pai/EasyAnimateV5.1-12b-zh`, `alibaba-pai/EasyAnimateV5.1-12b-zh-InP` | `torch.float16` |
| Image-to-video | `EasyAnimateInpaintPipeline` | `alibaba-pai/EasyAnimateV5.1-12b-zh-InP` | `torch.float16` |
| Control-to-video | `EasyAnimateControlPipeline` | `alibaba-pai/EasyAnimateV5.1-12b-zh-Control` | `torch.float16` |
| Camera control-to-video | `EasyAnimateControlPipeline` | `alibaba-pai/EasyAnimateV5.1-12b-zh-Control-Camera` | `torch.float16` |

Implementation notes:

- The Diffusers docs page text says there are two official checkpoints for
  text-to-video and video-to-video: the base checkpoint and the `InP`
  checkpoint.
- The docs page then lists the `InP` checkpoint for image-to-video and
  video-to-video.
- The docs page lists two control checkpoints: generic control and camera
  control.
- Source examples use some `-diffusers` model IDs, such as
  `alibaba-pai/EasyAnimateV5.1-7b-zh-diffusers` and
  `alibaba-pai/EasyAnimateV5.1-12b-zh-InP-diffusers`. Before wiring this into a
  product setting, verify whether the target Hub repo is the native original
  layout, a Diffusers-converted layout, or requires a specific branch/revision.

## 3. Architecture And Components

The Diffusers EasyAnimate pipelines are `DiffusionPipeline` subclasses with a
video VAE, a 3D diffusion transformer, a text encoder/tokenizer pair, and a flow
matching scheduler.

| Component | Diffusers class or type | Role |
| --- | --- | --- |
| VAE | `AutoencoderKLMagvit` | Encodes and decodes video frames to and from latent video tensors. |
| Text encoder | `Qwen2VLForConditionalGeneration` or `BertModel` | Encodes prompts. V5.1 uses Qwen2-VL according to the docs/source. |
| Tokenizer | `Qwen2Tokenizer` or `BertTokenizer` | Tokenizes prompts. The source applies the Qwen chat template before tokenization. |
| Denoiser | `EasyAnimateTransformer3DModel` | Transformer-based video denoiser. |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | Default denoising scheduler used with EasyAnimate. |
| Video processor | `VideoProcessor` | Postprocesses decoded video tensors into PIL/NumPy-style outputs. |
| Image/mask processor | `VaeImageProcessor` | Used by inpaint and control pipelines for conditioning inputs. |

The base source computes latent dimensions from the VAE compression ratios:

```text
latent frames = (num_frames - 1) // vae_temporal_compression_ratio + 1
latent height = height // vae_spatial_compression_ratio
latent width  = width // vae_spatial_compression_ratio
```

For V5.1, the source defaults to spatial compression ratio 8 and temporal
compression ratio 4 if the VAE is not already attached.

## 4. Pipeline Classes

### EasyAnimatePipeline

`EasyAnimatePipeline` is the class rendered on the public Diffusers docs page.
It is the cleanest starting point for SynthaEngine text-to-video support.

Constructor components:

```python
EasyAnimatePipeline(
    vae: AutoencoderKLMagvit,
    text_encoder: Qwen2VLForConditionalGeneration | BertModel,
    tokenizer: Qwen2Tokenizer | BertTokenizer,
    transformer: EasyAnimateTransformer3DModel,
    scheduler: FlowMatchEulerDiscreteScheduler,
)
```

Important class properties and behavior:

- `model_cpu_offload_seq = "text_encoder->transformer->vae"`, so Diffusers
  CPU offload hooks can move the large modules in a sensible order.
- `_callback_tensor_inputs` supports `latents`, `prompt_embeds`, and
  `negative_prompt_embeds`.
- `guidance_scale > 1` enables classifier-free guidance.
- `guidance_rescale` applies the same rescale-noise helper used by Stable
  Diffusion pipelines to reduce overexposure at high guidance values.
- For `FlowMatchEulerDiscreteScheduler`, the source calls
  `retrieve_timesteps(..., mu=1)`.

Primary `__call__` parameters:

| Parameter | Default | Notes |
| --- | --- | --- |
| `prompt` | `None` | String or list of strings. Required unless `prompt_embeds` is supplied. |
| `num_frames` | `49` | Video length in frames. V5.1 supports 1 to 49 and is documented as working best at 49. |
| `height`, `width` | `512`, `512` | Rounded down to multiples of 16 before validation. Docs recommend 256 to 1024 for V5.1 T2V/I2V. |
| `num_inference_steps` | `50` | More steps usually improve quality but slow inference. Docs quantization example uses 30. |
| `guidance_scale` | `5.0` | Higher values follow prompts more strongly but may reduce quality. |
| `negative_prompt` | `None` | Text to suppress; ignored when CFG is disabled. |
| `num_images_per_prompt` | `1` | Batch multiplier. Keep at 1 for video memory. |
| `eta` | `0.0` | Only used by schedulers that accept `eta`; FlowMatch ignores it. |
| `generator` | `None` | Use a `torch.Generator` for reproducible seeds. |
| `latents` | `None` | Optional precomputed latent tensor with the expected video latent shape. |
| `prompt_embeds` | `None` | Can replace prompt strings. Requires `prompt_attention_mask`. |
| `negative_prompt_embeds` | `None` | Requires `negative_prompt_attention_mask` when supplied. |
| `timesteps` | `None` | Optional custom scheduler timesteps. |
| `output_type` | `"pil"` | Use `"latent"` to skip VAE decode and postprocessing. |
| `return_dict` | `True` | Returns `EasyAnimatePipelineOutput` when true. |
| `callback_on_step_end` | `None` | Hook called after each denoising step. |
| `callback_on_step_end_tensor_inputs` | `["latents"]` | Must be one of the class callback tensor inputs. |
| `guidance_rescale` | `0.0` | Optional CFG rescale. |

Input validation:

- `height` and `width` must be divisible by 16 after rounding.
- Do not pass both `prompt` and `prompt_embeds`.
- Do not pass both `negative_prompt` and `negative_prompt_embeds`.
- If passing prompt embeddings directly, pass the matching attention mask.
- If passing prompt and negative prompt embeddings directly, their shapes must
  match.
- If passing a list of generators, the list length must match effective batch
  size.

### EasyAnimateInpaintPipeline

`EasyAnimateInpaintPipeline` is exported by the official EasyAnimate source but
is not expanded as a separate class on the public docs page. Its source example
uses the `InP` checkpoint and the helper
`get_image_to_video_latent(...)`, which makes it the relevant class for
image-to-video and masked video-to-video style workflows.

Constructor components match the base pipeline:

```python
EasyAnimateInpaintPipeline(
    vae: AutoencoderKLMagvit,
    text_encoder: Qwen2VLForConditionalGeneration | BertModel,
    tokenizer: Qwen2Tokenizer | BertTokenizer,
    transformer: EasyAnimateTransformer3DModel,
    scheduler: FlowMatchEulerDiscreteScheduler,
)
```

Additional processors and helpers:

- `image_processor` preprocesses input video frames.
- `mask_processor` is configured with grayscale conversion and binarization.
- `get_image_to_video_latent(validation_image_start, validation_image_end,
  num_frames, sample_size)` builds an input video tensor and mask tensor from
  optional start/end images.
- `prepare_mask_latents(...)` encodes the mask and masked video with the VAE.
- `prepare_latents(...)` can initialize from pure noise or from an encoded input
  video plus scheduler noise, depending on `strength`.
- `get_timesteps(num_inference_steps, strength, device)` follows img2img-style
  timestep slicing.

Primary `__call__` parameters beyond the base pipeline:

| Parameter | Default | Notes |
| --- | --- | --- |
| `video` | `None` | Conditioning video tensor, typically shaped `(batch, channels, frames, height, width)`. |
| `mask_video` | `None` | Mask tensor for frames/regions to regenerate. Helper creates 0 for fixed frames and 255 for generated frames. |
| `masked_video_latents` | `None` | Optional precomputed masked video latents. |
| `strength` | `1.0` | Controls how much denoising starts from noise versus input video. `1.0` means pure noise initialization. |
| `noise_aug_strength` | `0.0563` | Noise augmentation strength for conditioning video latents when enabled by transformer config. |

For image-to-video, `get_image_to_video_latent(...)` works like this:

- With a start image, it repeats the start image across `num_frames`, then masks
  all frames after the supplied start frame(s) for generation.
- With an end image, it fixes the last frame(s) by setting the mask to 0 there.
- With no start image, it creates an all-zero video and an all-255 mask.
- Input image types can be PIL images, NumPy arrays, or torch tensors.
- `sample_size` is `(height, width)`.

### EasyAnimateControlPipeline

`EasyAnimateControlPipeline` is also exported by official source but not
expanded on the public docs page. It is the class that matches the
`Control` and `Control-Camera` checkpoint modes.

Primary `__call__` additions:

| Parameter | Default | Notes |
| --- | --- | --- |
| `control_video` | `None` | Generic control video conditioning, such as Canny, depth, pose, MLSD, or trajectory-style control described by the model card. |
| `control_camera_video` | `None` | Camera-control conditioning tensor. The source resizes it to latent shape and scales it by 6. |
| `ref_image` | `None` | Reference image/video tensor used as first-frame conditioning. |

Control processing behavior from source:

- If `control_camera_video` is supplied, it is resized to latent shape with
  first-frame-aware mask resizing and multiplied by 6.
- Else if `control_video` is supplied, the source preprocesses frame tensors,
  encodes them with the VAE, and uses those latents as control conditioning.
- If no control tensor is supplied, the source uses zero control latents.
- If `ref_image` is supplied, its encoded latent is inserted into the first
  latent frame and concatenated with the control latents.
- The transformer call receives `control_latents=control_latents`.

Source-level gotchas for the control pipeline:

- The current main-source `EasyAnimateControlPipeline.__call__` passes
  `text_encoder_index=0` into `encode_prompt`, while the copied
  `encode_prompt` signature shown in the same source does not include that
  parameter. Validate your installed Diffusers version before relying on this
  path.
- The current main-source control pipeline calls `self.decode_latents(latents)`
  for non-latent outputs, but the same source file does not define
  `decode_latents`. Verify this is fixed in the target release, or keep control
  integration behind a smoke test.

## 5. Outputs

All EasyAnimate pipelines return `EasyAnimatePipelineOutput` when
`return_dict=True`.

```python
output = pipe(...)
frames = output.frames
```

`frames` may be:

- `list[list[PIL.Image.Image]]` with outer length `batch_size` and inner length
  `num_frames`.
- A NumPy array.
- A torch tensor shaped `(batch_size, num_frames, channels, height, width)`.

The common Diffusers usage pattern is:

```python
video_frames = pipe(...).frames[0]
export_to_video(video_frames, "output.mp4", fps=8)
```

For SynthaEngine, preserve the whole nested batch structure internally, but
store or return one exported MP4 per generated batch item.

## 6. Text-To-Video Example

```python
import torch
from diffusers import EasyAnimatePipeline
from diffusers.utils import export_to_video

model_id = "alibaba-pai/EasyAnimateV5.1-12b-zh"

pipe = EasyAnimatePipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
).to("cuda")

prompt = "A cinematic shot of a red train crossing snowy mountains at sunrise."
negative_prompt = "bad detail, distorted, low quality"

result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=49,
    height=512,
    width=512,
    num_inference_steps=30,
    guidance_scale=5.0,
)

export_to_video(result.frames[0], "easyanimate_t2v.mp4", fps=8)
```

SynthaEngine defaults should be more conservative than the full docs example on
12 GB VRAM:

```text
num_frames: 25 for local smoke tests, 49 for quality/cloud
height/width: 384x672 or 512x512 first, then larger only after VRAM tests
num_inference_steps: 20-30 for smoke tests, 30-50 for quality
num_images_per_prompt: 1
fps: 8
```

## 7. Image-To-Video Example

```python
import torch
from diffusers import EasyAnimateInpaintPipeline
from diffusers.pipelines.easyanimate.pipeline_easyanimate_inpaint import (
    get_image_to_video_latent,
)
from diffusers.utils import export_to_video, load_image

model_id = "alibaba-pai/EasyAnimateV5.1-12b-zh-InP"

pipe = EasyAnimateInpaintPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
).to("cuda")

start_image = load_image("input_start.png")
end_image = None
sample_size = (448, 576)
num_frames = 49

video, mask_video = get_image_to_video_latent(
    [start_image],
    end_image,
    num_frames,
    sample_size,
)

result = pipe(
    prompt="A spacecraft lifts off from a desert launch pad, cinematic.",
    negative_prompt="distorted, bad anatomy, low quality, text",
    num_frames=num_frames,
    height=sample_size[0],
    width=sample_size[1],
    video=video,
    mask_video=mask_video,
    num_inference_steps=30,
    guidance_scale=5.0,
)

export_to_video(result.frames[0], "easyanimate_i2v.mp4", fps=8)
```

For image-to-video UI design:

- Accept start image and optional end image.
- Convert to RGB before preprocessing.
- Expose `num_frames`, `height`, `width`, `steps`, `guidance_scale`, `seed`,
  and `negative_prompt`.
- Hide raw `mask_video` unless adding an advanced video editing mode.

## 8. Video-To-Video And Inpainting Pattern

The `InP` pipeline can be used for video-to-video by passing a conditioning
`video` tensor and a `mask_video`. The source expects channel-first video
tensors in the shape:

```text
(batch, channels, frames, height, width)
```

The pipeline internally permutes frames into image batches for preprocessing,
then reshapes back to video layout.

Suggested SynthaEngine contract:

| Input | Recommendation |
| --- | --- |
| `init_video` | Decode uploaded MP4/WebM into RGB frames, resize/crop to target dimensions, normalize to torch tensor. |
| `mask_video` | Start with full-frame regeneration masks, then add optional per-frame/per-pixel masks later. |
| `strength` | Expose as "motion/style strength"; default `0.75` for v2v and `1.0` for pure generation from fixed first frame. |
| `noise_aug_strength` | Advanced parameter; keep default `0.0563` first. |

Gotchas:

- `num_frames` must match the tensor frame count expected by the pipeline.
- Source docstring says `num_frames` is "seconds" in one place, but signature
  and docs page treat it as frame count. Use frame count.
- Use 8 FPS for export to match EasyAnimateV5.1 training/docs.
- Keep dimensions divisible by 16 after resizing.

## 9. Control-To-Video Pattern

Use `EasyAnimateControlPipeline` for the control checkpoints, but gate it behind
installed-version validation because of the source-level gotchas above.

Generic control:

```python
import torch
from diffusers import EasyAnimateControlPipeline
from diffusers.utils import export_to_video

pipe = EasyAnimateControlPipeline.from_pretrained(
    "alibaba-pai/EasyAnimateV5.1-12b-zh-Control",
    torch_dtype=torch.float16,
).to("cuda")

# control_video should be a torch tensor shaped:
# (batch, channels, frames, height, width)
result = pipe(
    prompt="A dancer moving through a neon-lit studio.",
    negative_prompt="low quality, distorted",
    control_video=control_video,
    num_frames=49,
    height=512,
    width=512,
    num_inference_steps=30,
)

export_to_video(result.frames[0], "easyanimate_control.mp4", fps=8)
```

Camera control:

```python
pipe = EasyAnimateControlPipeline.from_pretrained(
    "alibaba-pai/EasyAnimateV5.1-12b-zh-Control-Camera",
    torch_dtype=torch.float16,
).to("cuda")

result = pipe(
    prompt="A castle courtyard revealed by a slow upward camera move.",
    control_camera_video=control_camera_video,
    ref_image=first_frame_reference,
    num_frames=49,
    height=512,
    width=512,
)
```

Control preprocessing should be explicit in SynthaEngine:

- Canny/depth/pose/MLSD control videos should be generated at the same target
  frame count and resolution as the output request.
- Camera control tensors should be generated by the same representation expected
  by the EasyAnimate control-camera checkpoint. Do not infer arbitrary camera
  JSON support from the Diffusers pipeline signature; the source accepts a
  tensor.
- Reference images should be converted into a one-frame video tensor if the
  pipeline version expects `(batch, channels, frames, height, width)`.

## 10. Memory, Performance, And Quantization

EasyAnimateV5.1 12B is a large video model. Treat it as heavier than image-only
Diffusers pipelines because memory grows with:

```text
frames * height * width * transformer channels * denoising steps
```

Use these rules for local integration:

- Start with `num_frames=25`, `height=384`, `width=672`, batch size 1, and
  `num_inference_steps=20`.
- Move to `num_frames=49` only after a successful smoke test.
- Prefer `torch.float16` for the official 12B checkpoints because that is what
  the Diffusers docs table recommends.
- If using GPUs with strong BF16 support, benchmark BF16 separately; the public
  EasyAnimate model card and some source examples mention BF16 in other
  contexts, while the Diffusers page table recommends FP16.
- Use `enable_model_cpu_offload()` or `device_map="balanced"` for constrained
  cards.
- Keep `num_images_per_prompt=1`.
- Export MP4 immediately and free pipeline/model references after each job if
  using a subprocess runtime.

The Diffusers docs include a bitsandbytes quantization example for the
transformer:

```python
import torch
from diffusers import (
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
    EasyAnimatePipeline,
    EasyAnimateTransformer3DModel,
)

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)

transformer_8bit = EasyAnimateTransformer3DModel.from_pretrained(
    "alibaba-pai/EasyAnimateV5.1-12b-zh",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

pipe = EasyAnimatePipeline.from_pretrained(
    "alibaba-pai/EasyAnimateV5.1-12b-zh",
    transformer=transformer_8bit,
    torch_dtype=torch.float16,
    device_map="balanced",
)
```

Quantization guidance:

- Make quantization opt-in at first.
- Record the quantization backend in job metadata.
- Expect possible video quality changes; the Diffusers docs explicitly warn
  quantization impact varies by video model.
- Smoke test prompt-only T2V before enabling I2V or control modes under
  quantization.
- Do not combine multiple experimental memory techniques in the first user
  workflow. Validate offload, then validate 8-bit transformer loading, then
  validate any float8 path separately.

## 11. SynthaEngine Integration Plan

No repo code was changed by this guide, but a future implementation should use a
workflow-first shape:

### Workflow task IDs

Suggested public tasks:

| Task ID | Pipeline | Checkpoint family |
| --- | --- | --- |
| `easyanimate.text_to_video` | `EasyAnimatePipeline` | Base `EasyAnimateV5.1-12b-zh` |
| `easyanimate.image_to_video` | `EasyAnimateInpaintPipeline` | `EasyAnimateV5.1-12b-zh-InP` |
| `easyanimate.video_to_video` | `EasyAnimateInpaintPipeline` | `EasyAnimateV5.1-12b-zh-InP` |
| `easyanimate.control_to_video` | `EasyAnimateControlPipeline` | `EasyAnimateV5.1-12b-zh-Control` |
| `easyanimate.camera_control_to_video` | `EasyAnimateControlPipeline` | `EasyAnimateV5.1-12b-zh-Control-Camera` |

Keep these as additive task IDs. Do not rename or overload existing video task
IDs.

### Common request fields

```json
{
  "kind": "workflow",
  "task": "easyanimate.text_to_video",
  "prompt": "A cinematic aerial shot of a coastal city at sunrise.",
  "negative_prompt": "low quality, distorted, text",
  "model": "alibaba-pai/EasyAnimateV5.1-12b-zh",
  "num_frames": 25,
  "height": 384,
  "width": 672,
  "fps": 8,
  "steps": 20,
  "guidance_scale": 5.0,
  "seed": 12345,
  "dtype": "float16",
  "memory_mode": "model_cpu_offload"
}
```

### Mode-specific request fields

| Mode | Additional fields |
| --- | --- |
| Image-to-video | `start_image`, optional `end_image`, `strength`, `noise_aug_strength`. |
| Video-to-video | `init_video`, optional `mask_video`, `strength`, `noise_aug_strength`. |
| Control-to-video | `control_video`, `control_type`, optional `ref_image`. |
| Camera control-to-video | `control_camera_video`, optional `ref_image`. |

### Runtime recommendations

- Run EasyAnimate in a subprocess or isolated worker because the model is large
  and video jobs are long-lived.
- Reuse the workflow polling and event-stream pattern already used by
  SynthaEngine.
- Emit progress events per denoising step through `callback_on_step_end` where
  practical.
- Save both video and basic metadata: model ID, pipeline class, dtype,
  dimensions, frame count, FPS, steps, guidance scale, seed, memory mode, and
  quantization mode.
- Return MP4 paths in the workflow artifact list; optionally include extracted
  preview frames.
- Treat control pipeline integration as experimental until a real smoke test
  proves the installed Diffusers source path is usable.

## 12. Validation Checklist

Before exposing any EasyAnimate workflow publicly:

1. Import test the target pipeline class from the installed Diffusers build.
2. Load the intended checkpoint with `local_files_only=True` after first
   download to avoid network variance during jobs.
3. Run a 1-frame or very small frame-count latent-output smoke test if the
   pipeline supports `output_type="latent"`.
4. Run a 25-frame 384x672 MP4 export test.
5. Run a 49-frame 512x512 MP4 export test on the target deployment GPU.
6. Verify `height` and `width` are rounded or rejected consistently before
   calling Diffusers.
7. Verify deterministic seed behavior with `torch.Generator`.
8. Verify cancellation and cleanup release VRAM.
9. Verify generated output is a nested frame list or tensor matching
   `EasyAnimatePipelineOutput.frames`.
10. Run separate smoke tests for base, `InP`, `Control`, and `Control-Camera`
    checkpoints.

## 13. Gotchas

- The public Diffusers docs page highlights `EasyAnimatePipeline`, while the
  source package exports additional EasyAnimate inpaint and control pipelines.
  Do not assume the docs page has rendered every exported class.
- Dimensions must be divisible by 16. The source rounds down first, then
  validates.
- Video tensors are channel-first: `(batch, channels, frames, height, width)`.
  Many decoders produce `(frames, height, width, channels)`, so transpose
  carefully.
- `num_frames` means frame count. Export at 8 FPS for V5.1 to match official
  recommendations.
- Prompt embeddings require attention masks. This matters if SynthaEngine later
  caches text embeddings.
- The source uses Qwen chat-template formatting for prompts with Qwen2-VL.
  Avoid external prompt templating that fights the tokenizer template.
- `guidance_scale=1` disables classifier-free guidance. Negative prompts only
  matter when CFG is active.
- `output_type="latent"` skips video decoding and can be useful for smoke tests,
  but user-facing workflows need decoded frames exported to MP4.
- `EasyAnimateControlPipeline` should be version-checked before use because the
  current main source shows mismatched prompt-encoding/decode behavior.
- Model cards and docs mention both 7B and 12B families, and some source
  examples use `-diffusers` repository names. Normalize model IDs in config and
  test the exact IDs users can select.

## 14. Source Links

- Diffusers EasyAnimate docs:
  https://huggingface.co/docs/diffusers/api/pipelines/easyanimate
- Diffusers EasyAnimate docs source:
  https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/easyanimate.md
- `EasyAnimatePipeline` source:
  https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/easyanimate/pipeline_easyanimate.py
- `EasyAnimateInpaintPipeline` source:
  https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/easyanimate/pipeline_easyanimate_inpaint.py
- `EasyAnimateControlPipeline` source:
  https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/easyanimate/pipeline_easyanimate_control.py
- `EasyAnimatePipelineOutput` source:
  https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/easyanimate/pipeline_output.py
- EasyAnimate V5.1 base checkpoint:
  https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh
- EasyAnimate V5.1 InP checkpoint:
  https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh-InP
- EasyAnimate V5.1 control checkpoint:
  https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh-Control
- EasyAnimate V5.1 camera-control checkpoint:
  https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh-Control-Camera
