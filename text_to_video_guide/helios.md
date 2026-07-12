# Helios Diffusers Implementation Guide

Date: 2026-06-17

Research target: Helios video generation in Hugging Face Diffusers  
Primary Diffusers page: https://huggingface.co/docs/diffusers/api/pipelines/helios  
Primary Diffusers classes: `HeliosPipeline`, `HeliosPyramidPipeline`, `HeliosPipelineOutput`

This guide is a docs-only implementation reference for adding or evaluating
Helios support. It covers the official Diffusers Helios pipelines, the Base,
Mid, and Distilled checkpoints documented upstream, key call parameters, memory
and speed options, outputs, examples, and gotchas. No existing application files
are changed by this guide.

## 1. Executive Summary

Helios is a 14B autoregressive video diffusion model family for text-to-video
(T2V), image-to-video (I2V), and video-to-video (V2V). Diffusers exposes the
family through two pipeline classes:

| Class | Best documented use | Scheduler | Typical checkpoint |
| --- | --- | --- | --- |
| `HeliosPipeline` | Standard single-scale Helios generation | `HeliosScheduler` | `BestWishYsh/Helios-Base` |
| `HeliosPyramidPipeline` | Pyramid / multi-scale generation, including Mid and Distilled examples | `HeliosScheduler` or `HeliosDMDScheduler` | `BestWishYsh/Helios-Mid`, `BestWishYsh/Helios-Distilled` |

The official Diffusers page documents these supported checkpoints:

| Checkpoint | Upstream description | Prediction / guidance | Recommended pipeline from examples |
| --- | --- | --- | --- |
| `BestWishYsh/Helios-Base` | Best quality | v-prediction, standard CFG, custom `HeliosScheduler` | `HeliosPipeline` |
| `BestWishYsh/Helios-Mid` | Intermediate weight | v-prediction, CFG-Zero*, custom `HeliosScheduler` | `HeliosPyramidPipeline` |
| `BestWishYsh/Helios-Distilled` | Best efficiency | x0-prediction, custom `HeliosDMDScheduler` | `HeliosPyramidPipeline` |

Practical integration answer:

- Use `HeliosPipeline` first for the Base checkpoint and expose T2V, I2V, and
  V2V with conservative defaults: 640x384, 99 or 132 frames, BF16 transformer,
  FP32 VAE, batch size 1, and explicit output FPS.
- Use `HeliosPyramidPipeline` for Mid and Distilled. Mid uses larger per-stage
  step counts such as `[20, 20, 20]`; Distilled uses very small step counts such
  as `[2, 2, 2]`, `guidance_scale=1.0`, and usually
  `is_amplify_first_chunk=True`.
- Treat Helios as a heavyweight video runtime. Even though upstream documents a
  group-offload example around roughly 6 GB VRAM, the model still needs large
  CPU memory, slow downloads, and careful cleanup. A short-lived subprocess is a
  safer local-server integration pattern than keeping the model resident.
- Do not rely on VAE tiling or slicing for memory savings. Diffusers documents
  `AutoencoderKLWan` as not supporting those generic VAE memory features.

## 2. What Helios Is

Helios is documented by Diffusers as a real-time long-video generation model
with a unified representation for T2V, I2V, and V2V. It generates video
autoregressively in chunks, using compressed history context so later chunks can
continue earlier motion.

For implementation purposes, the important shape of the system is:

1. A UMT5 text stack turns prompt text into embeddings.
2. An `AutoencoderKLWan` VAE encodes conditioning images/videos and decodes
   generated latent chunks back into frames.
3. `HeliosTransformer3DModel` denoises video latent chunks while attending to
   prompt embeddings and latent history.
4. `HeliosScheduler` or `HeliosDMDScheduler` drives denoising.
5. `HeliosPipelineOutput.frames` carries the generated video frames.

The upstream Diffusers page says Helios supports minute-scale generation and
reports real-time H100 performance. The Hub model card reports a slightly
different H100 FPS number than the Diffusers page, so local product docs should
avoid promising a fixed FPS and should benchmark on the actual machine.

## 3. Installation Notes

Helios support is documented on the Diffusers `v0.38.0` docs page. If the local
environment has an older Diffusers release, install a recent release or install
from source.

```powershell
.venv\Scripts\python.exe -m pip install -U diffusers transformers accelerate imageio imageio-ffmpeg
```

If the installed release does not expose `HeliosPipeline` and
`HeliosPyramidPipeline`, use the Diffusers source install recommended by the
upstream model card:

```powershell
.venv\Scripts\python.exe -m pip install git+https://github.com/huggingface/diffusers.git
```

Optional performance packages depend on the target hardware and should be
feature-gated:

```powershell
.venv\Scripts\python.exe -m pip install -U bitsandbytes
```

## 4. Checkpoints And Components

### 4.1 Checkpoint IDs

Use the exact IDs from the Diffusers documentation and Hub pages:

- `BestWishYsh/Helios-Base`
- `BestWishYsh/Helios-Mid`
- `BestWishYsh/Helios-Distilled`

The model repositories contain the pipeline components as subfolders. The
examples load the VAE explicitly and pass it into the pipeline:

```python
import torch
from diffusers import AutoModel

model_id = "BestWishYsh/Helios-Base"
vae = AutoModel.from_pretrained(
    model_id,
    subfolder="vae",
    torch_dtype=torch.float32,
)
```

Keep the VAE in FP32 unless you have verified that another dtype is safe. The
official examples use a FP32 VAE with a BF16 pipeline.

### 4.2 Text Stack

The pipeline constructor documents:

- `tokenizer`: `AutoTokenizer`, described as the T5 tokenizer for
  `google/umt5-xxl`.
- `text_encoder`: `UMT5EncoderModel`, described as the T5 text encoder for
  `google/umt5-xxl`.

The pipeline call accepts either `prompt` text or precomputed `prompt_embeds`.
Do not pass both. For classifier-free guidance, pass either `negative_prompt`
or `negative_prompt_embeds`, but not both.

### 4.3 Transformer

`HeliosTransformer3DModel` is the 14B video transformer used by the pipelines.
The model docs show loading it directly from each checkpoint's `transformer`
subfolder:

```python
import torch
from diffusers import HeliosTransformer3DModel

transformer = HeliosTransformer3DModel.from_pretrained(
    "BestWishYsh/Helios-Base",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)
```

Key documented defaults include:

| Transformer field | Default |
| --- | --- |
| `patch_size` | `(1, 2, 2)` |
| `num_attention_heads` | `40` |
| `attention_head_dim` | `128` |
| `in_channels` / `out_channels` | `16` / `16` |
| `text_dim` | `4096` in the signature |
| `ffn_dim` | `13824` |
| `num_layers` | `40` |
| `qk_norm` | `rms_norm_across_heads` |
| `guidance_cross_attn` | `True` |
| `zero_history_timestep` | `True` |

Most integrations should not instantiate the transformer manually at first.
Load the whole pipeline with `from_pretrained()` so the checkpoint config selects
the matching transformer and scheduler behavior.

### 4.4 VAE

Helios uses `AutoencoderKLWan` as the VAE. The documented VAE defaults include
`z_dim=16`, `scale_factor_temporal=4`, and `scale_factor_spatial=8`.

These scale factors matter:

- Default `num_latent_frames_per_chunk=9`.
- Generated frame chunk size is `(9 - 1) * 4 + 1 = 33` frames.
- Spatial latent size for 640x384 is 80x48.

The generic Diffusers memory guide says `AutoencoderKLWan` does not support VAE
slicing or VAE tiling. Avoid exposing `enable_vae_slicing()` or
`enable_vae_tiling()` as Helios optimization toggles unless Diffusers changes
that support status and the behavior is verified.

### 4.5 Schedulers

Use the scheduler shipped with each checkpoint unless there is a tested reason
to override it.

| Scheduler | Documented use |
| --- | --- |
| `HeliosScheduler` | Pyramidal flow-matching sampling; used by Base and Mid docs. |
| `HeliosDMDScheduler` | DMD-style pyramidal flow-matching sampling; used by Distilled docs. |

`HeliosScheduler` exposes UniPC-style internals such as `solver_order`,
`solver_type`, `lower_order_final`, and `time_shift_type`. `HeliosDMDScheduler`
has a smaller signature and defaults `time_shift_type` to `linear`. In practice,
the scheduler config from the checkpoint should be treated as part of the model.

## 5. Pipeline Class: `HeliosPipeline`

`HeliosPipeline` is the standard pipeline class documented for T2V, I2V, and
V2V generation. It inherits from `DiffusionPipeline` and
`HeliosLoraLoaderMixin`.

Constructor components:

```text
tokenizer: AutoTokenizer
text_encoder: UMT5EncoderModel
vae: AutoencoderKLWan
scheduler: HeliosScheduler
transformer: HeliosTransformer3DModel
```

Important class properties and behavior from source:

- CPU/model offload order is `text_encoder -> transformer -> vae`.
- Callback tensor inputs are `latents`, `prompt_embeds`, and
  `negative_prompt_embeds`.
- `transformer` is optional in the component registry, but it is required for
  normal generation.
- Guidance is active only when `guidance_scale > 1.0`.

### 5.1 Standard Text-To-Video Example

```python
import torch
from diffusers import AutoModel, HeliosPipeline
from diffusers.utils import export_to_video

model_id = "BestWishYsh/Helios-Base"

vae = AutoModel.from_pretrained(
    model_id,
    subfolder="vae",
    torch_dtype=torch.float32,
)

pipe = HeliosPipeline.from_pretrained(
    model_id,
    vae=vae,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

frames = pipe(
    prompt="A quiet coastal train passes cliffs at sunset, cinematic motion.",
    negative_prompt="blur, low quality, static, text, watermark",
    height=384,
    width=640,
    num_frames=99,
    num_inference_steps=50,
    guidance_scale=5.0,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]

export_to_video(frames, "helios_base_t2v.mp4", fps=24)
```

### 5.2 Image-To-Video Example

```python
import torch
from diffusers import AutoModel, HeliosPipeline
from diffusers.utils import export_to_video, load_image

model_id = "BestWishYsh/Helios-Base"
vae = AutoModel.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = HeliosPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16).to("cuda")

image = load_image("input.png").resize((640, 384))

frames = pipe(
    prompt="The scene comes alive with gentle camera motion and natural wind.",
    negative_prompt="blur, low quality, text, watermark",
    image=image,
    num_frames=99,
    num_inference_steps=50,
    guidance_scale=5.0,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]

export_to_video(frames, "helios_base_i2v.mp4", fps=24)
```

### 5.3 Video-To-Video Example

```python
import torch
from diffusers import AutoModel, HeliosPipeline
from diffusers.utils import export_to_video, load_video

model_id = "BestWishYsh/Helios-Base"
vae = AutoModel.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = HeliosPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16).to("cuda")

video = load_video("source.mp4")

frames = pipe(
    prompt="Keep the action and camera path, but render it as polished cinematic footage.",
    negative_prompt="blur, low quality, text, watermark",
    video=video,
    num_frames=99,
    num_inference_steps=50,
    guidance_scale=5.0,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]

export_to_video(frames, "helios_base_v2v.mp4", fps=24)
```

### 5.4 `HeliosPipeline.__call__` Parameters

| Parameter | Default | Implementation notes |
| --- | --- | --- |
| `prompt` | `None` | String or list of strings. Required unless `prompt_embeds` is supplied. |
| `negative_prompt` | `None` | Used only when classifier-free guidance runs. Guidance requires `guidance_scale > 1.0`. |
| `height` / `width` | `384` / `640` | Must both be divisible by 16. |
| `num_frames` | `132` | Requested output frame count. Internally generated in chunks; use multiples of 33 for predictable output. |
| `num_inference_steps` | `50` | Standard pipeline denoising steps. Higher is slower and may improve quality. |
| `sigmas` | `None` | Optional custom sigma schedule. Source falls back to a linear schedule from 0.999 to 0.0. |
| `guidance_scale` | `5.0` | CFG strength. `1.0` disables CFG in source. |
| `num_videos_per_prompt` | `1` | Duplicates text embeddings per prompt. Keep at 1 for memory. |
| `generator` | `None` | `torch.Generator` or list of generators for deterministic output. |
| `latents` | `None` | Optional pre-generated latent noise. Shape must match the requested latent geometry. |
| `prompt_embeds` | `None` | Precomputed text embeddings. Mutually exclusive with `prompt`. |
| `negative_prompt_embeds` | `None` | Precomputed negative embeddings. Mutually exclusive with `negative_prompt`. |
| `output_type` | `"np"` | Docs mention NumPy/PIL; source also accepts `"latent"` to return latents. |
| `return_dict` | `True` | `True` returns `HeliosPipelineOutput`; source returns a one-element tuple when `False`. |
| `attention_kwargs` | `None` | Passed to the attention processor. |
| `callback_on_step_end` | `None` | Callback after denoising steps. |
| `callback_on_step_end_tensor_inputs` | `["latents"]` | Must be among `latents`, `prompt_embeds`, `negative_prompt_embeds`. |
| `max_sequence_length` | `512` | Pipeline call max token length. Direct `encode_prompt()` defaults to 226. |
| `image` | `None` | I2V conditioning image. Do not pass with `video`. |
| `image_latents` | `None` | Precomputed image latent. Advanced use. |
| `fake_image_latents` | `None` | Extra I2V history latent used internally. Advanced use. |
| `add_noise_to_image_latents` | `True` | Adds random noise to I2V conditioning latents. |
| `image_noise_sigma_min` / `image_noise_sigma_max` | `0.111` / `0.135` | Noise range for image conditioning. |
| `video` | `None` | V2V conditioning video. Do not pass with `image`. |
| `video_latents` | `None` | Precomputed video latents. Advanced use. |
| `add_noise_to_video_latents` | `True` | Adds random noise to V2V conditioning latents. |
| `video_noise_sigma_min` / `video_noise_sigma_max` | `0.111` / `0.135` | Noise range for video conditioning. |
| `history_sizes` | `[16, 2, 1]` | Latent history windows. Source sorts descending before use. |
| `num_latent_frames_per_chunk` | `9` | With Wan temporal scale 4, this yields 33 video frames per chunk. |
| `keep_first_frame` | `True` | Keeps a first-frame prefix in latent history. |
| `is_skip_first_chunk` | `False` | Can help when the first generated chunk is static in I2V/V2V workflows. |

## 6. Pipeline Class: `HeliosPyramidPipeline`

`HeliosPyramidPipeline` is the documented pyramid / multi-scale Helios class.
It supports the same T2V, I2V, and V2V input modes as `HeliosPipeline`, but it
adds stage-wise denoising controls for pyramid sampling, CFG-Zero*, and
Distilled DMD behavior.

Constructor components:

```text
tokenizer: AutoTokenizer
text_encoder: UMT5EncoderModel
vae: AutoencoderKLWan
scheduler: HeliosScheduler | HeliosDMDScheduler
transformer: HeliosTransformer3DModel
is_cfg_zero_star: bool = False
is_distilled: bool = False
```

Important source behavior:

- `pyramid_num_inference_steps_list` controls the number of stages and the
  denoising steps per stage.
- Intermediate stages downsample and upsample latent spatial resolution.
- If `is_cfg_zero_star` is true, the source applies CFG-Zero* logic controlled
  by `use_zero_init` and `zero_steps`.
- If `is_distilled` is true, the source passes DMD-specific tensors into the
  scheduler. With `is_amplify_first_chunk=True`, the first chunk uses double
  the total listed stage steps.

### 6.1 Helios-Mid Pyramid Example

```python
import torch
from diffusers import AutoModel, HeliosPyramidPipeline
from diffusers.utils import export_to_video

model_id = "BestWishYsh/Helios-Mid"
vae = AutoModel.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)

pipe = HeliosPyramidPipeline.from_pretrained(
    model_id,
    vae=vae,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

frames = pipe(
    prompt="A neon-lit street market at night, steady forward camera motion.",
    negative_prompt="blur, low quality, static, text, watermark",
    num_frames=99,
    pyramid_num_inference_steps_list=[20, 20, 20],
    guidance_scale=5.0,
    use_zero_init=True,
    zero_steps=1,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]

export_to_video(frames, "helios_mid_t2v.mp4", fps=24)
```

### 6.2 Helios-Distilled Pyramid Example

```python
import torch
from diffusers import AutoModel, HeliosPyramidPipeline
from diffusers.utils import export_to_video

model_id = "BestWishYsh/Helios-Distilled"
vae = AutoModel.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)

pipe = HeliosPyramidPipeline.from_pretrained(
    model_id,
    vae=vae,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

frames = pipe(
    prompt="A tropical fish glides through coral reef water, smooth motion.",
    negative_prompt="blur, low quality, static, text, watermark",
    num_frames=240,
    pyramid_num_inference_steps_list=[2, 2, 2],
    guidance_scale=1.0,
    is_amplify_first_chunk=True,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]

export_to_video(frames, "helios_distilled_t2v.mp4", fps=24)
```

At `guidance_scale=1.0`, the source does not run classifier-free guidance, so
the negative prompt is effectively inert. The official examples still pass a
negative prompt for consistency across checkpoint examples.

### 6.3 Pyramid I2V And V2V

The same `image=` and `video=` call paths are documented for Mid and Distilled.
Only the stage-step parameters change:

```python
from diffusers.utils import load_image, load_video

image_frames = pipe(
    prompt="Animate the still image with natural parallax and wind.",
    image=load_image("input.png").resize((640, 384)),
    num_frames=99,
    pyramid_num_inference_steps_list=[20, 20, 20],
    guidance_scale=5.0,
).frames[0]

video_frames = pipe(
    prompt="Preserve the motion while changing the scene style.",
    video=load_video("source.mp4"),
    num_frames=99,
    pyramid_num_inference_steps_list=[20, 20, 20],
    guidance_scale=5.0,
).frames[0]
```

For Distilled, substitute:

```python
pyramid_num_inference_steps_list=[2, 2, 2]
guidance_scale=1.0
is_amplify_first_chunk=True
```

### 6.4 `HeliosPyramidPipeline.__call__` Parameters

`HeliosPyramidPipeline` accepts almost all `HeliosPipeline.__call__` parameters
except `num_inference_steps`. It adds these pyramid-specific parameters:

| Parameter | Default | Implementation notes |
| --- | --- | --- |
| `pyramid_num_inference_steps_list` | `[10, 10, 10]` | Per-stage denoising steps. The list length is the stage count. |
| `use_zero_init` | `True` | Used by CFG-Zero* logic when checkpoint config enables it. |
| `zero_steps` | `1` | Number of early stage-zero denoising steps zeroed by CFG-Zero* logic. |
| `is_amplify_first_chunk` | `False` | Distilled/DMD option. For distilled first chunk, source doubles total listed steps. |

Use the official example defaults as presets:

| Preset | Pipeline | Recommended call options |
| --- | --- | --- |
| Base quality | `HeliosPipeline` | `num_inference_steps=50`, `guidance_scale=5.0`, `num_frames=99` or `132` |
| Mid pyramid | `HeliosPyramidPipeline` | `pyramid_num_inference_steps_list=[20, 20, 20]`, `guidance_scale=5.0`, `use_zero_init=True`, `zero_steps=1` |
| Distilled speed | `HeliosPyramidPipeline` | `pyramid_num_inference_steps_list=[2, 2, 2]`, `guidance_scale=1.0`, `is_amplify_first_chunk=True`, `num_frames=240` |

## 7. Output Handling

Both pipelines return `HeliosPipelineOutput` when `return_dict=True`.

```text
HeliosPipelineOutput(frames=...)
```

`frames` is documented as one of:

- nested PIL image lists,
- a NumPy array,
- a Torch tensor.

The documented tensor/array shape is batch-major video output. The examples use
`pipeline(...).frames[0]` and pass that first video to `export_to_video()`.

```python
from diffusers.utils import export_to_video

result = pipe(prompt="...", num_frames=99)
frames = result.frames[0]
export_to_video(frames, "output.mp4", fps=24)
```

Source-level notes:

- `return_dict=False` returns `(video,)` in current source, not an NSFW tuple.
  There is no Helios safety-checker output documented.
- `output_type="latent"` returns latent history instead of decoded frames.
  This is useful for debugging or staged processing, but downstream code must
  not pass it to `export_to_video()`.

## 8. Memory, Performance, And Quantization

### 8.1 Baseline Dtypes

The official examples use:

- VAE: `torch.float32`
- Pipeline / transformer: `torch.bfloat16`
- Device: CUDA

```python
vae = AutoModel.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = HeliosPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
```

On non-BF16 GPUs, test carefully. Falling back to FP16 may work for some
components but is not the documented default.

### 8.2 Group Offloading

The Diffusers Helios page shows group offloading as the primary low-VRAM path
and states that the example is around 6 GB VRAM. Do not call `pipe.to("cuda")`
for this path; group offloading moves layers as needed.

```python
import torch
from diffusers import AutoModel, HeliosPipeline

model_id = "BestWishYsh/Helios-Base"
vae = AutoModel.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = HeliosPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)

pipe.enable_group_offload(
    onload_device=torch.device("cuda"),
    offload_device=torch.device("cpu"),
    offload_type="leaf_level",
    use_stream=True,
    record_stream=True,
)
```

Diffusers documents these group-offload tradeoffs:

- `block_level` offloads groups of layers and is controlled by
  `num_blocks_per_group`.
- `leaf_level` offloads individual layers and can be faster with CUDA streams.
- `use_stream=True` overlaps transfer and compute, but can significantly
  increase CPU memory requirements.
- `record_stream=True` can improve speed at a modest memory cost.
- `offload_to_disk_path` can be used if system RAM is the bottleneck, but disk
  offload will be slower.

### 8.3 Model CPU Offload

Because `HeliosPipeline` defines a model offload order, generic Diffusers
offloading can be exposed as an alternative:

```python
pipe = HeliosPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
```

Model CPU offload generally saves less VRAM than group offload but has lower
transfer overhead than sequential CPU offload. Sequential offload is usually a
last resort because it can be extremely slow.

### 8.4 Device Maps

For multi-GPU or CPU/GPU placement experiments, Diffusers supports `device_map`
and `max_memory`.

```python
pipe = HeliosPipeline.from_pretrained(
    model_id,
    vae=vae,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
)
print(pipe.hf_device_map)
```

Do not mix `device_map` with `.to("cuda")` or CPU offload APIs without first
calling `reset_device_map()`, as documented in the Diffusers memory guide.

### 8.5 Quantization

The generic Diffusers speed/memory guide says video generation often benefits
from combining quantization with group offloading because video models are
compute-bound. Helios-specific quantization is not shown on the Helios page, so
treat this as experimental and verify output quality.

Example adapted to Helios component names:

```python
import torch
from diffusers import AutoModel, HeliosPipeline
from diffusers.quantizers import PipelineQuantizationConfig

torch._dynamo.config.capture_dynamic_output_shape_ops = True

quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
    },
    components_to_quantize=["transformer", "text_encoder"],
)

model_id = "BestWishYsh/Helios-Base"
vae = AutoModel.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = HeliosPipeline.from_pretrained(
    model_id,
    vae=vae,
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
)
pipe.enable_group_offload(
    onload_device=torch.device("cuda"),
    offload_device=torch.device("cpu"),
    offload_type="leaf_level",
    use_stream=True,
    record_stream=True,
)
```

Implementation cautions:

- Quantize the transformer first if you need a narrower experiment.
- Quantizing the text encoder can save memory but may affect prompt adherence.
- Combining quantization, `torch.compile`, and offloading can trigger graph
  breaks or recompilation. Add feature flags and fallbacks.
- Do not quantize the FP32 VAE in the first implementation pass.

### 8.6 Compile And Attention Backends

Generic Diffusers optimization docs mention `torch.compile`, memory-efficient
attention backends, and `channels_last`. These are not Helios-specific
recommendations in the Helios page, so they should be opt-in experiments.

Potential experiments:

```python
pipe.transformer.compile(mode="max-autotune", fullgraph=True)
```

```python
from diffusers import set_attention_backend

set_attention_backend("flash")
```

Measure:

- wall-clock time,
- peak CUDA memory,
- CPU memory during group offload,
- first-run compile latency,
- repeat-run latency,
- output quality and determinism.

## 9. Integration Guidance For SynthaEngine

### 9.1 Suggested Runtime Shape

Helios should be integrated as a video model family with three preset variants:

| Runtime preset | Checkpoint | Pipeline class | Default task support |
| --- | --- | --- | --- |
| `helios-base` | `BestWishYsh/Helios-Base` | `HeliosPipeline` | T2V, I2V, V2V |
| `helios-mid` | `BestWishYsh/Helios-Mid` | `HeliosPyramidPipeline` | T2V, I2V, V2V |
| `helios-distilled` | `BestWishYsh/Helios-Distilled` | `HeliosPyramidPipeline` | T2V, I2V, V2V |

Recommended user-facing fields:

| Field | Type | Suggested default |
| --- | --- | --- |
| `prompt` | string | required |
| `negative_prompt` | string | reusable default negative prompt |
| `variant` | enum | `helios-distilled` for speed, `helios-base` for quality |
| `mode` | enum | `text_to_video`, `image_to_video`, `video_to_video` |
| `width` / `height` | int | `640` / `384` |
| `num_frames` | int | `99`, `132`, or `240` depending preset |
| `fps` | int | `24` |
| `seed` | int | optional |
| `guidance_scale` | float | `5.0` for Base/Mid, `1.0` for Distilled |
| `num_inference_steps` | int | Base only; default `50` |
| `pyramid_num_inference_steps_list` | list[int] | Mid `[20,20,20]`, Distilled `[2,2,2]` |
| `optimization` | enum | `cuda`, `group_offload`, `model_cpu_offload`, `quantized_group_offload` |

### 9.2 Conservative Local Defaults

For an unknown local GPU, start with Distilled:

```json
{
  "variant": "helios-distilled",
  "width": 640,
  "height": 384,
  "num_frames": 99,
  "fps": 24,
  "pyramid_num_inference_steps_list": [2, 2, 2],
  "guidance_scale": 1.0,
  "is_amplify_first_chunk": true,
  "optimization": "group_offload"
}
```

For quality testing on a large GPU:

```json
{
  "variant": "helios-base",
  "width": 640,
  "height": 384,
  "num_frames": 99,
  "fps": 24,
  "num_inference_steps": 50,
  "guidance_scale": 5.0,
  "optimization": "cuda"
}
```

### 9.3 Validation Checklist

For any runtime implementation, validate:

1. Import succeeds for `HeliosPipeline`, `HeliosPyramidPipeline`, and
   `AutoModel`.
2. Model download succeeds for each selected checkpoint.
3. T2V generates a playable MP4 at 640x384.
4. I2V accepts a PIL image resized to 640x384.
5. V2V accepts an input video with at least 33 frames by default.
6. `num_frames` behavior is documented in the API response, especially when a
   non-multiple of 33 is rounded up.
7. `guidance_scale=1.0` behavior is explained for Distilled.
8. Group offload does not call `.to("cuda")` afterward.
9. Cancellation and cleanup release CUDA memory and CPU offload hooks.
10. Output metadata records actual frames, requested frames, FPS, seed,
    checkpoint ID, pipeline class, and optimization mode.

## 10. Gotchas

- `height` and `width` must be divisible by 16. The source raises an error
  otherwise.
- `image` and `video` are mutually exclusive. Use one conditioning mode per
  call.
- With the default VAE temporal scale and chunk setting, the generation chunk is
  33 frames. Use `num_frames` values like 99, 132, 240, 720, or 1449 if you want
  predictable chunk counts.
- The source calculates the number of chunks using a ceiling operation, so a
  non-multiple frame count can produce more frames than requested.
- V2V input must contain at least
  `(num_latent_frames_per_chunk - 1) * vae_scale_factor_temporal + 1` frames.
  With defaults, that is 33 frames.
- `guidance_scale=1.0` disables classifier-free guidance in source. Negative
  prompts are not expected to affect Distilled outputs at that setting.
- Direct `encode_prompt()` defaults `max_sequence_length` to 226, while
  pipeline calls default to 512.
- The official examples use FP32 VAE. Start there before changing dtype.
- `AutoencoderKLWan` does not support generic Diffusers VAE slicing or tiling.
- I2V and V2V may be weaker than T2V because upstream notes the training is
  based on T2V. If the first chunks are static, try `is_skip_first_chunk=True`
  or increase the image/video noise sigma range.
- `Helios-Mid` is described upstream as an intermediate distillation checkpoint
  and may not match Base quality.
- The docs page and model card report different H100 FPS claims. Benchmark
  locally and report measured performance.
- Quantization is not documented as a Helios-specific recipe. Keep it behind an
  experimental flag.

## 11. Source Links

Official Diffusers docs:

- Helios pipeline docs:
  https://huggingface.co/docs/diffusers/api/pipelines/helios
- `HeliosTransformer3DModel` docs:
  https://huggingface.co/docs/diffusers/v0.38.0/en/api/models/helios_transformer3d
- `HeliosScheduler` docs:
  https://huggingface.co/docs/diffusers/v0.38.0/en/api/schedulers/helios
- `HeliosDMDScheduler` docs:
  https://huggingface.co/docs/diffusers/v0.38.0/en/api/schedulers/helios_dmd
- `AutoencoderKLWan` docs:
  https://huggingface.co/docs/diffusers/v0.38.0/en/api/models/autoencoder_kl_wan
- Diffusers memory optimization docs:
  https://huggingface.co/docs/diffusers/optimization/memory
- Diffusers quantization/compile/offload docs:
  https://huggingface.co/docs/diffusers/optimization/speed-memory-optims

Official Diffusers source:

- `pipeline_helios.py`:
  https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/helios/pipeline_helios.py
- `pipeline_helios_pyramid.py`:
  https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/helios/pipeline_helios_pyramid.py
- `pipeline_output.py`:
  https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/helios/pipeline_output.py
- Helios docs source:
  https://github.com/huggingface/diffusers/blob/v0.38.0/docs/source/en/api/pipelines/helios.md

Model pages:

- `BestWishYsh/Helios-Base`:
  https://huggingface.co/BestWishYsh/Helios-Base
- `BestWishYsh/Helios-Mid`:
  https://huggingface.co/BestWishYsh/Helios-Mid
- `BestWishYsh/Helios-Distilled`:
  https://huggingface.co/BestWishYsh/Helios-Distilled
