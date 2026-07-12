# CogVideoX implementation guide

This guide summarizes the current Hugging Face Diffusers CogVideoX API for
local video generation. It is based on the CogVideoX pipeline docs and source
on Diffusers `main` / v0.38.0-era docs, plus the current Hugging Face model
collection and model cards linked from those docs.

## What CogVideoX is

CogVideoX is a diffusion-transformer video model family for text-conditioned
video generation. In Diffusers, the family shares a T5 text encoder, a
`CogVideoXTransformer3DModel`, an `AutoencoderKLCogVideoX` video VAE, and a
scheduler. The public pipeline classes are:

| Class | Task | Main extra input |
| --- | --- | --- |
| `CogVideoXPipeline` | Text-to-video | `prompt` |
| `CogVideoXImageToVideoPipeline` | Image-to-video | `image` plus `prompt` |
| `CogVideoXVideoToVideoPipeline` | Video-to-video/editing | `video`, `prompt`, `strength` |
| `CogVideoXFunControlPipeline` | Controlled text-to-video with CogVideoX-Fun | `control_video` or `control_video_latents` |
| `CogVideoXPipelineOutput` | Shared output container | `frames` |

All four pipelines inherit Diffusers pipeline behavior such as
`from_pretrained`, `.to(...)`, CPU offload hooks, saving/loading, callbacks,
and `return_dict=False`. All four pipeline classes also inherit
`CogVideoXLoraLoaderMixin`, so they expose the CogVideoX LoRA loader methods.

## Current checkpoints and variants

Use the checkpoint variant that matches the pipeline task.

| Variant | Typical repository | Pipeline |
| --- | --- | --- |
| CogVideoX 2B text-to-video | `THUDM/CogVideoX-2b` or current `zai-org/CogVideoX-2b` | `CogVideoXPipeline` |
| CogVideoX 5B text-to-video | `THUDM/CogVideoX-5b` or current `zai-org/CogVideoX-5b` | `CogVideoXPipeline` |
| CogVideoX 5B image-to-video | `THUDM/CogVideoX-5b-I2V` or current `zai-org/CogVideoX-5b-I2V` | `CogVideoXImageToVideoPipeline` |
| CogVideoX 1.5 5B text-to-video | `THUDM/CogVideoX1.5-5B` or current `zai-org/CogVideoX1.5-5B` | `CogVideoXPipeline` |
| CogVideoX 1.5 5B image-to-video | `THUDM/CogVideoX1.5-5B-I2V` or current `zai-org/CogVideoX1.5-5B-I2V` | `CogVideoXImageToVideoPipeline` |
| CogVideoX 1.5 SAT | `zai-org/CogVideoX1.5-5B-SAT` | SAT/CogVideo tooling, not the normal Diffusers pipeline unless converted |
| CogVideoX-Fun control | `alibaba-pai/CogVideoX-Fun-V1.1-5b-Pose` and related Fun control repos | `CogVideoXFunControlPipeline` |

The Diffusers docs examples still use `THUDM/...` names. The official
CogVideo collection currently redirects to `zai-org/...` model pages. In a
production integration, prefer the repository name shown by the model card you
pin, and record the exact revision.

## Installation

Use a recent Diffusers release that includes the CogVideoX classes, or install
from source if a target model card asks for it.

```bash
pip install -U diffusers transformers accelerate sentencepiece imageio-ffmpeg
```

Optional optimization packages:

```bash
pip install -U torchao
```

For repeatable deployments, pin `diffusers`, `transformers`, `accelerate`,
`torch`, `torchao`, the model repository, and the model revision together.

## Text-to-video with `CogVideoXPipeline`

Start with text-to-video when the job is generated only from a prompt. The
pipeline loads the tokenizer, T5 encoder, transformer, VAE, and scheduler from
the model repo.

```python
import torch
from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler
from diffusers.utils import export_to_video

model_id = "zai-org/CogVideoX1.5-5B"

pipe = CogVideoXPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)

pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

prompt = (
    "A slow cinematic tracking shot through a glass greenhouse after rain, "
    "with warm lights reflected in puddles and leaves moving in a light breeze."
)
negative_prompt = "blur, flicker, distorted hands, unstable camera, low detail"

generator = torch.Generator(device="cuda").manual_seed(1234)

result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=768,
    width=1360,
    num_frames=81,
    num_inference_steps=50,
    guidance_scale=6.0,
    use_dynamic_cfg=False,
    generator=generator,
)

frames = result.frames[0]
export_to_video(frames, "cogvideox_t2v.mp4", fps=16)
```

### Text-to-video parameter guide

| Parameter | Notes |
| --- | --- |
| `prompt` | String or list of strings. Use detailed English prompts for the public model cards unless a downstream checkpoint says otherwise. |
| `negative_prompt` | Used only when `guidance_scale > 1`. It must match the prompt batch type and batch size. |
| `height`, `width` | Source validation requires divisibility by 8. Model cards and API notes recommend stricter checkpoint-specific sizes, often multiples of 16. |
| `num_frames` | Source default is the checkpoint config's `sample_frames`. Current CogVideoX 1.5 recommendations are usually `81` or `161`; older examples often use `49`. |
| `num_inference_steps` | Default is `50`. More steps usually improve quality but cost latency. |
| `timesteps` | Optional custom descending timestep list for schedulers that support it. Do not pass with incompatible scheduler settings. |
| `guidance_scale` | Default is `6`. Values above `1` enable classifier-free guidance; too high can over-constrain and reduce motion quality. |
| `use_dynamic_cfg` | Dynamically changes guidance through the denoising loop. It is often useful for I2V and can be tested for T2V style adherence. |
| `generator` | Use a seeded `torch.Generator` for reproducibility. If passing a list, its length must match effective batch size. |
| `latents` | Optional precomputed noise latents for advanced prompt sweeps or server-side retry workflows. |
| `prompt_embeds`, `negative_prompt_embeds` | Optional precomputed T5 embeddings. Do not pass both raw prompts and matching embeds. |
| `output_type` | `"pil"` by default. Source also supports `"np"` and `"latent"` behavior through the video processor and decode branch. |
| `callback_on_step_end` | Receives selected tensors from `callback_on_step_end_tensor_inputs`. Allowed tensor names are `latents`, `prompt_embeds`, and `negative_prompt_embeds`. |
| `max_sequence_length` | Default is `226`; it should match `transformer.config.max_text_seq_length`. Overlong prompts are truncated by the tokenizer. |

Current source sets `num_videos_per_prompt = 1` inside the pipeline call even
though the signature exposes the argument. Treat batching as "one video per
prompt" unless you have verified your installed Diffusers version changed this.

### Frame and resolution recommendations

Use the model card and config as the final authority for a pinned checkpoint.
The current API notes recommend:

- Text-to-video checkpoints work best at `1360x768`.
- Image-to-video checkpoints support multiple widths in the `768` to `1360`
  range, with dimensions divisible by 16.
- Current T2V and I2V checkpoints are commonly run at `81` or `161` frames and
  exported at `16` fps.

Older CogVideoX 2B/5B examples and model cards use `720x480`, `49` frames, and
`8` fps. Do not mix these assumptions blindly with CogVideoX 1.5.

The API page currently contains an apparent I2V height typo: it says the height
must be `758` while also saying height and width must be divisible by 16. The
linked model cards and practical examples point to `768`-based dimensions, so
prefer multiples of 16 and validate against the target checkpoint.

### CogVideoX 1.5 temporal padding

CogVideoX 1.5 transformers may define `patch_size_t`. For text-to-video, the
current source pads latent frames when the generated latent frame count is not
divisible by `patch_size_t`, then discards the padding before decoding. This is
why recommended frame counts should be taken from the checkpoint docs instead
of invented casually.

## Schedulers

The standard CogVideoX pipelines accept CogVideoX-specific schedulers:

- `CogVideoXDDIMScheduler`
- `CogVideoXDPMScheduler`

Load a scheduler from the pipeline config when swapping algorithms:

```python
from diffusers import CogVideoXDPMScheduler

pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)
```

Use `CogVideoXDDIMScheduler` when you need DDIM-style behavior and `eta`.
Use `CogVideoXDPMScheduler` when you want the CogVideoX DPM-Solver++ path.
The video-to-video docs example explicitly swaps to `CogVideoXDPMScheduler`.

`CogVideoXFunControlPipeline` accepts `KarrasDiffusionSchedulers` in source and
the docs example uses `DDIMScheduler.from_config(pipe.scheduler.config)`.
Keep the scheduler family compatible with the checkpoint config.

## LoRA support

CogVideoX supports LoRA loading through `load_lora_weights`. Load LoRAs that
match the base architecture, model generation, and task. A CogVideoX 1.5 LoRA
is not automatically safe on a 1.0 checkpoint, and an I2V/Fun LoRA may target
different modules than a T2V LoRA.

```python
pipe.load_lora_weights(
    "finetrainers/CogVideoX-1.5-crush-smol-v0",
    adapter_name="crush",
)
pipe.set_adapters("crush", 0.8)
```

Practical LoRA order:

1. Load the base pipeline.
2. Load and set LoRA adapters.
3. Apply offload or quantization hooks.
4. Compile only after adapters and dtype choices are final.

Layerwise casting docs warn that fp8 storage with PEFT layers can be less
well-tested, so validate LoRA quality after adding fp8 or aggressive
quantization.

## Image-to-video with `CogVideoXImageToVideoPipeline`

Use this pipeline when a still image defines the first-frame identity, style, or
composition.

```python
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

model_id = "zai-org/CogVideoX1.5-5B-I2V"

pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

image = load_image("input.png")
prompt = "The subject turns slightly toward the camera while soft window light moves across the scene."

frames = pipe(
    image=image,
    prompt=prompt,
    height=768,
    width=1360,
    num_frames=81,
    num_inference_steps=50,
    guidance_scale=6.0,
    use_dynamic_cfg=True,
    generator=torch.Generator(device="cuda").manual_seed(22),
).frames[0]

export_to_video(frames, "cogvideox_i2v.mp4", fps=16)
```

Important details:

- `image` may be a PIL image, a list of PIL images, or a tensor.
- The pipeline encodes the image through the VAE and concatenates image
  conditioning latents with denoising latents. Use an I2V checkpoint so channel
  counts match.
- Current source default for `num_frames` is `49`, but current 1.5 model cards
  commonly recommend `81` or `161`.
- Source validation requires `height` and `width` divisible by 8; the I2V docs
  and model cards recommend dimensions divisible by 16.
- CogVideoX 1.5 I2V source pads generated latent frames when required by
  temporal patching and removes the padding before decode.

## Video-to-video with `CogVideoXVideoToVideoPipeline`

Use this pipeline to restyle, edit, or prompt-shift an existing clip.

```python
import torch
from diffusers import CogVideoXDPMScheduler, CogVideoXVideoToVideoPipeline
from diffusers.utils import export_to_video, load_video

model_id = "zai-org/CogVideoX-5b"

pipe = CogVideoXVideoToVideoPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

input_video = load_video("source.mp4")
prompt = "The same camera motion, transformed into a crisp watercolor animation with gentle paper texture."

frames = pipe(
    video=input_video,
    prompt=prompt,
    height=480,
    width=720,
    strength=0.65,
    guidance_scale=6.0,
    num_inference_steps=50,
    generator=torch.Generator(device="cuda").manual_seed(7),
).frames[0]

export_to_video(frames, "cogvideox_v2v.mp4", fps=8)
```

`strength` controls how far the generated video moves away from the input
video. Source validation requires `0.0 <= strength <= 1.0`. Internally,
`strength` shortens the denoising schedule: lower values preserve more of the
source, while higher values give the prompt more room to rewrite motion and
appearance.

For checkpoints with `patch_size_t`, video-to-video does not auto-pad the input
clip. The latent frame count must already be divisible by `patch_size_t`, or
the pipeline raises an error. If you hit this on CogVideoX 1.5, trim or sample
the input to a compatible frame count before calling the pipeline.

## Controlled generation with `CogVideoXFunControlPipeline`

CogVideoX-Fun control uses a control video as structure guidance. The pipeline
does not compute pose, depth, Canny, or other control maps for you; prepare the
control frames with the preprocessing stack expected by the checkpoint.

```python
import torch
from diffusers import CogVideoXFunControlPipeline, DDIMScheduler
from diffusers.utils import export_to_video, load_video

model_id = "alibaba-pai/CogVideoX-Fun-V1.1-5b-Pose"

pipe = CogVideoXFunControlPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

control_video = load_video("pose_control.mp4")
prompt = "A futuristic dancer follows the same pose sequence on a reflective stage with blue rim lighting."

frames = pipe(
    prompt=prompt,
    control_video=control_video,
    height=768,
    width=768,
    num_inference_steps=50,
    guidance_scale=6.0,
    generator=torch.Generator(device="cuda").manual_seed(404),
).frames[0]

export_to_video(frames, "cogvideox_fun_control.mp4", fps=8)
```

Control-specific details:

- Pass either `control_video` or `control_video_latents`, not both.
- A flat list of PIL frames is wrapped as a single control video batch by the
  source code.
- `num_frames` is inferred from the control input.
- The Fun control source uses the offload sequence
  `text_encoder->vae->transformer->vae`, because the VAE is used before and
  after denoising.
- Control latents are concatenated with denoising latents along the channel
  dimension during the transformer call.
- Like video-to-video, CogVideoX 1.5-style temporal patching requires the
  control video latent frame count to be compatible with `patch_size_t`.

The `alibaba-pai` CogVideoX-Fun model cards include broader Fun variants such
as InP and Control checkpoints. In Diffusers, use `CogVideoXFunControlPipeline`
for the documented control pipeline and choose the checkpoint for the control
signal you actually prepared.

## Memory and performance options

CogVideoX is expensive. Combine the least invasive optimizations first, then
add quantization only if memory demands it.

### Offload and VAE memory savings

```python
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
```

The CogVideoX API notes report approximate 5B memory tradeoffs:

| Method | Enabled | Disabled |
| --- | --- | --- |
| `enable_model_cpu_offload` | about 19 GB | about 33 GB |
| `enable_sequential_cpu_offload` | below 4 GB | about 33 GB, but very slow |
| `vae.enable_tiling` | about 11 GB with model CPU offload | not listed |

Use sequential CPU offload only when fitting the model is more important than
latency. For multi-GPU inference, model cards warn that sequential CPU offload
should be disabled.

### Group offloading

Diffusers memory docs support group offloading at model or module level. This
can be useful when standard model CPU offload is still too large.

```python
import torch
from diffusers.hooks.group_offloading import apply_group_offloading

onload_device = torch.device("cuda")
offload_device = torch.device("cpu")

pipe.transformer.enable_group_offload(
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="leaf_level",
    use_stream=True,
)
apply_group_offloading(pipe.text_encoder, onload_device=onload_device, offload_type="block_level", num_blocks_per_group=2)
apply_group_offloading(pipe.vae, onload_device=onload_device, offload_type="leaf_level")
```

If using streamed group offload with VAE tiling, run a tiny warmup forward when
practical and validate for device mismatch errors.

### Layerwise fp8 casting

Layerwise casting stores weights in fp8 and upcasts for compute. It can reduce
model memory, but check GPU and PyTorch support.

```python
import torch
from diffusers import CogVideoXPipeline, CogVideoXTransformer3DModel

transformer = CogVideoXTransformer3DModel.from_pretrained(
    "zai-org/CogVideoX-5b",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)
transformer.enable_layerwise_casting(
    storage_dtype=torch.float8_e4m3fn,
    compute_dtype=torch.bfloat16,
)

pipe = CogVideoXPipeline.from_pretrained(
    "zai-org/CogVideoX-5b",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
```

### TorchAO int8 quantization

The CogVideoX docs show `PipelineQuantizationConfig` with TorchAO int8 weight
only quantization for the transformer. The same docs state that a quantized
CogVideoX 5B setup requires roughly 16 GB of VRAM in that example.

```python
import torch
from diffusers import AutoModel, CogVideoXPipeline, TorchAoConfig
from diffusers.quantizers import PipelineQuantizationConfig
from torchao.quantization import Int8WeightOnlyConfig

quant_config = PipelineQuantizationConfig(
    quant_mapping={"transformer": TorchAoConfig(Int8WeightOnlyConfig())}
)

transformer = AutoModel.from_pretrained(
    "zai-org/CogVideoX-5b",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)

pipe = CogVideoXPipeline.from_pretrained(
    "zai-org/CogVideoX-5b",
    transformer=transformer,
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
```

INT8 often lowers VRAM but may reduce speed. FP8 and layerwise casting require
more careful hardware and dependency validation.

### `torch.compile`

The CogVideoX docs include a `torch.compile` speed example for the 2B model and
note the first compiled run is slow. Compile only after model dtype, LoRA, and
offload decisions are stable.

```python
pipe.transformer.to(memory_format=torch.channels_last)
pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
```

Compilation may not be worthwhile for short-lived worker processes because the
compile cost is paid up front.

## Outputs and server integration

The shared output class is:

```python
CogVideoXPipelineOutput(frames=...)
```

`frames` may be:

- `list[list[PIL.Image.Image]]` with shape `batch_size x num_frames`
- a NumPy array
- a Torch tensor shaped like `(batch_size, num_frames, channels, height, width)`

With the default `return_dict=True`, read `result.frames[0]`. With
`return_dict=False`, the first tuple element is the video object. To skip VAE
decode in an advanced pipeline, pass `output_type="latent"` and handle latent
decoding yourself.

Use `diffusers.utils.export_to_video(frames, path, fps=...)` for quick MP4
export. In a server, keep generation and encoding failure modes separate: the
pipeline can succeed while video encoding fails because of codec, path, or
ffmpeg availability.

## Common gotchas

- Match pipeline and checkpoint type. T2V, I2V, V2V, and Fun control use
  different conditioning assumptions and channel counts.
- Keep `height` and `width` divisible by 8 at minimum, and usually by 16 for
  recommended CogVideoX resolutions.
- Use frame counts recommended by the specific checkpoint. CogVideoX 1.5 has
  temporal patch constraints that matter more for V2V/control than T2V/I2V.
- Do not pass both `prompt` and `prompt_embeds`, or both `negative_prompt` and
  `negative_prompt_embeds`.
- Do not pass both `control_video` and `control_video_latents`.
- `negative_prompt` is ignored when `guidance_scale <= 1`.
- The current pipeline implementations reset `num_videos_per_prompt` to `1`.
- `max_sequence_length` must stay consistent with the transformer config, and
  long prompts are truncated.
- Load LoRAs before compiling. Re-test if combining LoRA with fp8 layerwise
  casting or other PEFT-adjacent memory tricks.
- Use the SAT checkpoint variants with the SAT/CogVideo tooling unless the
  model card explicitly provides a Diffusers-format workflow.

## Source links

- Diffusers CogVideoX API docs: <https://huggingface.co/docs/diffusers/api/pipelines/cogvideox>
- Diffusers CogVideoX docs source: <https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/cogvideox.md>
- Text-to-video source: <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_cogvideox.py>
- Image-to-video source: <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_image2video.py>
- Video-to-video source: <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_video2video.py>
- Fun control source: <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_fun_control.py>
- Output class source: <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_output.py>
- CogVideo model collection: <https://huggingface.co/collections/zai-org/cogvideo>
- Diffusers memory guide: <https://huggingface.co/docs/diffusers/optimization/memory>
- CogVideoX DDIM scheduler: <https://huggingface.co/docs/diffusers/api/schedulers/ddim_cogvideox>
- CogVideoX DPM scheduler: <https://huggingface.co/docs/diffusers/api/schedulers/multistep_dpm_solver_cogvideox>
