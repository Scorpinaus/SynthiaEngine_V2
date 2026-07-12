# HunyuanVideo-1.5 Diffusers Implementation Guide

Last checked: 2026-06-17 against the Hugging Face Diffusers
HunyuanVideo-1.5 API page, the linked Diffusers source on GitHub, and the
official `hunyuanvideo-community` Diffusers model cards.

Research target:
https://huggingface.co/docs/diffusers/api/pipelines/hunyuan_video15

Primary Diffusers classes:

- `HunyuanVideo15Pipeline`
- `HunyuanVideo15ImageToVideoPipeline`
- `HunyuanVideo15PipelineOutput`

HunyuanVideo-1.5 is Tencent's newer HunyuanVideo family in Diffusers. The docs
describe it as an 8.3B-parameter video model for both text-to-video (T2V) and
image-to-video (I2V), with an advanced DiT architecture, selective and sliding
tile attention (SSTA), glyph-aware text encoding, and support for multiple
durations and resolutions. For a local workflow integration, treat the two
Diffusers pipeline classes as separate entry points that share the same text
stack, VAE, scheduler, transformer family, guider behavior, and output type.

## 1. Executive Summary

Use `HunyuanVideo15Pipeline` for pure text-to-video generation and
`HunyuanVideo15ImageToVideoPipeline` when an input image should condition the
first frame and visual direction of the clip.

| Class | Task | Main extra inputs | Documented components |
| --- | --- | --- | --- |
| `HunyuanVideo15Pipeline` | Text-to-video | `prompt`, optional `height` and `width` | `Qwen2.5-VL` text encoder, ByT5 glyph encoder, HunyuanVideo1.5 transformer, HunyuanVideo1.5 VAE, FlowMatch Euler scheduler, classifier-free guider |
| `HunyuanVideo15ImageToVideoPipeline` | Image-to-video | `image`, `prompt` | Same text/transformer/VAE/scheduler/guider stack plus `SiglipVisionModel` and `SiglipImageProcessor` |

Recommended practical defaults:

- Start with `torch_dtype=torch.bfloat16`, `num_frames=121`, `num_inference_steps=50`, `num_videos_per_prompt=1`, and `output_type="np"`.
- Export with `diffusers.utils.export_to_video(..., fps=24)` for the model-card examples. The docs' short memory example exports 61 frames at 15 fps.
- Enable memory aids immediately for local server use: `pipe.enable_model_cpu_offload()` and `pipe.vae.enable_tiling()`.
- Use a padding-efficient attention backend. The official page recommends `_flash_3_hub` or `_flash_3_varlen_hub` on H100/H800, `flash_hub` or `flash_varlen_hub` on A100/A800/RTX 4090, and `sage_hub` on other GPUs.
- Do not pass `guidance_scale` to `pipe(...)`. The pipeline uses a `ClassifierFreeGuidance` guider object; update it with `pipe.guider = pipe.guider.new(guidance_scale=...)`.

## 2. Checkpoints and Model IDs

The Diffusers API page says the original HunyuanVideo checkpoints are under the
`tencent` organization, but its examples use `hunyuanvideo-community` because
those weights are stored in a Diffusers-compatible layout.

The official `hunyuanvideo-community/HunyuanVideo-1.5` collection currently
lists these Diffusers model IDs:

| Model ID | Task | Notes |
| --- | --- | --- |
| `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v` | T2V | General 480p text-to-video checkpoint |
| `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v` | T2V | General 720p text-to-video checkpoint |
| `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled` | T2V | 480p text-to-video distilled checkpoint |
| `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v` | I2V | General 480p image-to-video checkpoint |
| `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v` | I2V | General 720p image-to-video checkpoint |
| `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_distilled` | I2V | 480p image-to-video distilled checkpoint |
| `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled` | I2V | 720p image-to-video distilled checkpoint |
| `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled` | I2V | 480p image-to-video step-distilled checkpoint; model-card examples use 12 steps |

Some rendered Diffusers autodoc examples show model IDs without the
`Diffusers` infix, such as
`hunyuanvideo-community/HunyuanVideo-1.5-480p_t2v`. Prefer the fully qualified
collection IDs above for reproducible integration work.

## 3. Installation

Use a recent Diffusers build that includes the HunyuanVideo1.5 classes. The
docs page links to `v0.38.0` source, and the `main` docs/source also contains
the HunyuanVideo1.5 page.

```powershell
.venv\Scripts\python.exe -m pip install -U diffusers transformers accelerate torch imageio imageio-ffmpeg
```

For the recommended Hub attention kernels:

```powershell
.venv\Scripts\python.exe -m pip install -U kernels
```

Use Hugging Face authentication if the selected model requires accepting terms:

```powershell
huggingface-cli login
```

## 4. Components

Both pipeline classes are normal `DiffusionPipeline` subclasses and can be
loaded with `from_pretrained`, saved, moved to devices, and offloaded through
standard Diffusers APIs.

### Shared components

| Component | Type in docs | Purpose |
| --- | --- | --- |
| `transformer` | `HunyuanVideo15Transformer3DModel` | Conditional MMDiT denoiser for video latents |
| `scheduler` | `FlowMatchEulerDiscreteScheduler` | Flow-matching denoising schedule |
| `vae` | `AutoencoderKLHunyuanVideo15` | 3D VAE for encoding and decoding videos to and from latent space |
| `text_encoder` | `Qwen2.5-VL-7B-Instruct` / `Qwen2_5_VLTextModel` | Main text encoder. Source wraps prompts in a video-description chat template before encoding. |
| `tokenizer` | `Qwen2Tokenizer` | Tokenizer for the Qwen2.5-VL text path |
| `text_encoder_2` | `T5EncoderModel` | Second text encoder used for glyph-aware conditioning |
| `tokenizer_2` | `ByT5Tokenizer` | Tokenizer for glyph text extracted from prompts |
| `guider` | `ClassifierFreeGuidance` | Runtime guidance object; replaces a direct `guidance_scale` call argument |

### Image-to-video-only components

| Component | Type in docs | Purpose |
| --- | --- | --- |
| `image_encoder` | `SiglipVisionModel` | Encodes the input image for visual conditioning |
| `feature_extractor` | `SiglipImageProcessor` | Preprocesses the input image for the SigLIP image encoder |

The linked I2V source also prepares image latents through the VAE, repeats the
image condition across the batch, keeps only frame 0 populated, and creates a
mask whose first latent frame is active. In practical terms, the image behaves
like the first-frame condition while the prompt describes the desired motion,
style, and scene evolution.

## 5. Text Encoding and Glyph-Aware Prompts

HunyuanVideo1.5 has a two-part text-conditioning path:

1. The main prompt path uses Qwen2.5-VL text encoding. The source formats the
   prompt as a chat conversation with a system message asking the model to
   describe video content, objects, motion, background, light, style,
   atmosphere, camera angles, camera movement, and transitions.
2. The glyph-aware path uses `ByT5Tokenizer` and `T5EncoderModel`. The public
   `encode_prompt` docs call these `prompt_embeds_2` and
   `prompt_embeds_mask_2`, and describe them as glyph text embeddings and masks
   from ByT5.

The linked source extracts glyph text from quoted substrings in the prompt
using a pattern for straight or curly double quotes. If no quoted text is
present, the glyph embedding path receives zeros. For prompts that require
legible text in the video, put the target text in quotes:

```python
prompt = (
    'A girl holds a sheet of paper with the words "Hello, world!" written '
    'clearly in black ink. The camera slowly pushes in while the paper stays '
    'centered and readable.'
)
```

Prompt-embedding integrations need to keep both text paths together:

- If passing `prompt_embeds`, also pass `prompt_embeds_mask`.
- If passing `prompt_embeds_2`, also pass `prompt_embeds_mask_2`.
- Negative embedding variants have the same mask requirement.
- Do not pass both `prompt` and `prompt_embeds` in the same call.
- If bypassing text strings entirely, provide embeddings for both the main and
  second text encoder paths.

## 6. Text-to-Video

Use `HunyuanVideo15Pipeline` with a T2V checkpoint.

```python
import torch
from diffusers import HunyuanVideo15Pipeline
from diffusers.utils import export_to_video

model_id = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"

pipe = HunyuanVideo15Pipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

prompt = (
    "A quiet city tram glides through rain-soaked streets at night. Neon signs "
    "reflect on the pavement, passengers are visible through fogged windows, "
    "and the camera tracks beside the tram with smooth cinematic motion."
)

generator = torch.Generator(device="cuda").manual_seed(1234)
video = pipe(
    prompt=prompt,
    generator=generator,
    num_frames=121,
    num_inference_steps=50,
).frames[0]

export_to_video(video, "hunyuan15_t2v.mp4", fps=24)
```

The docs also show a memory-oriented short run:

```python
video = pipe(
    prompt="A fluffy teddy bear sits on a bed of soft pillows surrounded by children's toys.",
    num_frames=61,
    num_inference_steps=30,
).frames[0]
export_to_video(video, "output.mp4", fps=15)
```

### T2V call parameters

| Parameter | Default | Implementation notes |
| --- | --- | --- |
| `prompt` | `None` | Required unless `prompt_embeds` are supplied. Accepts `str` or `list[str]`. |
| `negative_prompt` | `None` | Used when the guider has more than one condition. Can be replaced with negative embeddings. |
| `height`, `width` | `None` | T2V-only public arguments. Provide both or neither. If both are omitted, source calculates a default 16:9 size from the transformer target size. |
| `num_frames` | `121` | Number of frames to generate. Examples use 61 and 121. |
| `num_inference_steps` | `50` | More steps usually improve quality but slow inference. Step-distilled I2V examples use fewer steps, but T2V cards still show 50. |
| `sigmas` | `None` | Optional custom sigma schedule for schedulers whose `set_timesteps` accepts sigmas. |
| `num_videos_per_prompt` | `1` | Multiplies batch size and memory. Keep at 1 for local video generation. |
| `generator` | `None` | Use `torch.Generator(device="cuda").manual_seed(seed)` for reproducibility. |
| `latents` | `None` | Optional pre-sampled video latents for controlled reruns. |
| `prompt_embeds`, `prompt_embeds_mask` | `None` | Main text encoder embeddings and mask. |
| `negative_prompt_embeds`, `negative_prompt_embeds_mask` | `None` | Negative embeddings and mask. |
| `prompt_embeds_2`, `prompt_embeds_mask_2` | `None` | Second/glyph text encoder embeddings and mask. |
| `negative_prompt_embeds_2`, `negative_prompt_embeds_mask_2` | `None` | Negative second/glyph embeddings and mask. |
| `output_type` | `"np"` | Choose `"np"`, `"pt"`, or `"latent"`. Use `"latent"` to skip VAE decode/postprocess. |
| `return_dict` | `True` | If false, returns a tuple whose first element is the video output. |
| `attention_kwargs` | `None` | Passed to the attention processor. Useful only for advanced attention integrations. |

### T2V resolution and duration

The public API exposes `height` and `width` for T2V. Use the checkpoint family
as the first constraint:

- 480p checkpoints: target 480p-class output.
- 720p checkpoints: target 720p-class output.

Prefer dimensions compatible with the VAE spatial compression ratio. The
HunyuanVideo1.5 VAE docs list spatial compression as 16 and temporal
compression as 4, and the linked pipeline source prepares latents as
`height // 16`, `width // 16`, and `(num_frames - 1) // 4 + 1`. In practice,
use dimensions divisible by 16 and favor frame counts like `4n + 1`, such as
61 or 121, because they map cleanly to the documented temporal compression.

At export time, duration is controlled by `num_frames / fps`:

| Frames | FPS | Approx duration |
| --- | --- | --- |
| 61 | 15 | 4.07 seconds |
| 121 | 24 | 5.04 seconds |

Diffusers returns frames; it does not bake an FPS into the tensor. Pick the FPS
in `export_to_video`.

## 7. Image-to-Video

Use `HunyuanVideo15ImageToVideoPipeline` with an I2V checkpoint and a
`PIL.Image.Image` input.

```python
import torch
from diffusers import HunyuanVideo15ImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

model_id = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v"

pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

image = load_image(
    "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG"
)
prompt = (
    "Summer beach vacation style, a white cat wearing sunglasses sits on a "
    "surfboard. The cat relaxes in warm sunlight while the camera holds a "
    "close-up view and the seaside background moves gently."
)

generator = torch.Generator(device="cuda").manual_seed(1)
video = pipe(
    image=image,
    prompt=prompt,
    generator=generator,
    num_frames=121,
    num_inference_steps=50,
).frames[0]

export_to_video(video, "hunyuan15_i2v.mp4", fps=24)
```

For the step-distilled 480p I2V checkpoint, the official model card uses 12
inference steps:

```python
model_id = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled"

pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

video = pipe(
    image=image,
    prompt=prompt,
    generator=torch.Generator(device="cuda").manual_seed(1),
    num_frames=121,
    num_inference_steps=12,
).frames[0]
export_to_video(video, "hunyuan15_i2v_step_distilled.mp4", fps=24)
```

### I2V call parameters

| Parameter | Default | Implementation notes |
| --- | --- | --- |
| `image` | required | Must be a `PIL.Image.Image`. Use `diffusers.utils.load_image` for URLs/paths. |
| `prompt` | `None` | Required unless prompt embeddings are supplied. Prompt still matters for motion and scene evolution. |
| `negative_prompt` | `None` | Same role as in T2V. |
| `num_frames` | `121` | Number of frames to generate. |
| `num_inference_steps` | `50` | General I2V cards use 50; step-distilled card uses 12. |
| `sigmas` | `None` | Optional custom sigma schedule. |
| `num_videos_per_prompt` | `1` | Keep at 1 unless memory has been validated. |
| `generator` | `None` | Use for deterministic outputs. |
| `latents` | `None` | Optional pre-sampled noisy latents. |
| `prompt_embeds`, `prompt_embeds_mask` | `None` | Main text encoder embeddings and mask. |
| `negative_prompt_embeds`, `negative_prompt_embeds_mask` | `None` | Negative main text embeddings and mask. |
| `prompt_embeds_2`, `prompt_embeds_mask_2` | `None` | Glyph-aware second text encoder embeddings and mask. |
| `negative_prompt_embeds_2`, `negative_prompt_embeds_mask_2` | `None` | Negative second text encoder embeddings and mask. |
| `output_type` | `"np"` | Choose `"np"`, `"pt"`, or `"latent"`. |
| `return_dict` | `True` | If false, returns a tuple. |
| `attention_kwargs` | `None` | Passed to attention processors. |

Unlike T2V, the I2V public call signature does not expose `height` and `width`.
Pick the 480p or 720p checkpoint that matches the intended output class, and
feed a sensible RGB image. The pipeline's image processor converts the image to
RGB, and the linked source derives conditioning latents from the image through
the HunyuanVideo1.5 VAE.

## 8. Guidance

The official Notes section is explicit: `HunyuanVideo15Pipeline` uses a guider
and does not accept `guidance_scale` as a runtime argument. The I2V signature
also omits `guidance_scale` and includes the same `ClassifierFreeGuidance`
component.

Inspect the default guider:

```python
print(pipe.guider)
```

The docs show a default `ClassifierFreeGuidance` configuration with
`guidance_scale=6.0`, `guidance_rescale=0.0`, `start=0.0`, and `stop=1.0`.
Update it by replacing the guider:

```python
pipe.guider = pipe.guider.new(guidance_scale=5.0)
```

For distilled checkpoints, check the model card and benchmark. Tencent's
original model card lists CFG-distilled variants with CFG scale 1 in its
non-Diffusers optimal configuration table, while the Diffusers collection model
cards generally keep the same Diffusers call shape and do not expose a runtime
`guidance_scale` parameter.

## 9. Memory and Performance

HunyuanVideo1.5 is smaller than the older 13B HunyuanVideo model, but it is
still a heavyweight video pipeline. Plan around large downloads, high CPU RAM
pressure, and long GPU execution.

### Minimum practical memory pattern

```python
pipe = HunyuanVideo15Pipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
```

Use the same pattern for I2V.

`enable_model_cpu_offload()` is the documented HunyuanVideo1.5 example. The
linked source declares these offload sequences:

- T2V: `text_encoder -> transformer -> vae`
- I2V: `image_encoder -> text_encoder -> transformer -> vae`

`pipe.vae.enable_tiling()` is recommended by the HunyuanVideo1.5 page and the
`AutoencoderKLHunyuanVideo15` docs, which explicitly say to enable tiling to
avoid OOM. Tiling splits VAE encode/decode work into tiles so larger videos can
fit in memory.

### Attention backends

HunyuanVideo1.5 uses attention masks with variable-length sequences. The docs
recommend an attention backend that handles padding efficiently:

| GPU family | Recommended backends |
| --- | --- |
| H100/H800 | `_flash_3_hub`, `_flash_3_varlen_hub` |
| A100/A800/RTX 4090 | `flash_hub`, `flash_varlen_hub` |
| Other GPUs | `sage_hub` |

Persistent model setting:

```python
pipe.transformer.set_attention_backend("flash_hub")
```

Temporary context-manager setting, matching the model-card examples:

```python
from diffusers import attention_backend

with attention_backend("_flash_3_hub"):
    video = pipe(prompt=prompt, num_frames=121, num_inference_steps=50).frames[0]
```

Install `kernels` for the Hub-backed kernels:

```powershell
.venv\Scripts\python.exe -m pip install -U kernels
```

### Quantization

The HunyuanVideo1.5 pipeline page does not currently provide a
Hunyuan-specific quantization recipe. If local memory requires quantization,
use the general Diffusers quantization APIs and validate the exact component
mapping with the installed `diffusers` and `transformers` versions.

The relevant component split is:

- Diffusers components: `transformer`, `vae`
- Transformers components: `text_encoder`, `text_encoder_2`, and for I2V,
  `image_encoder`

The Diffusers quantization docs distinguish Diffusers
`BitsAndBytesConfig`/`QuantoConfig` from Transformers `BitsAndBytesConfig`.
That matters here because the HunyuanVideo1.5 pipeline mixes Diffusers models
and Transformers models.

Example pattern to adapt and test:

```python
import torch
from diffusers import HunyuanVideo15Pipeline
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

quant_config = PipelineQuantizationConfig(
    quant_mapping={
        "transformer": DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        "text_encoder": TransformersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        "text_encoder_2": TransformersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    }
)

pipe = HunyuanVideo15Pipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
    torch_dtype=torch.bfloat16,
    quantization_config=quant_config,
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
```

For I2V, add an `image_encoder` mapping if quantizing the SigLIP vision model.
Do not assume a quantized component can be freely moved with `.to(...)`; test
the exact quantization backend and offload combination before exposing it in a
server.

### Step-distilled I2V

The official 480p I2V step-distilled model card demonstrates
`num_inference_steps=12`. Tencent's original model card says the step-distilled
480p I2V model is intended for 8 or 12 steps, with 12 recommended, and reports
large RTX 4090 speedups in the original code path. Diffusers integration should
surface this as a checkpoint-specific default, not as a global HunyuanVideo1.5
default.

## 10. Outputs

Both pipelines return `HunyuanVideo15PipelineOutput` by default.

```python
output = pipe(prompt=prompt)
frames = output.frames
first_video = frames[0]
```

The output class has one field:

| Field | Type | Notes |
| --- | --- | --- |
| `frames` | `torch.Tensor`, `np.ndarray`, or `list[list[PIL.Image.Image]]` | The docs describe a nested batch structure of length `batch_size`, with each sub-list containing `num_frames` denoised frames, or an array/tensor shaped `(batch_size, num_frames, channels, height, width)`. |

`output_type` controls the representation:

- `"np"`: NumPy output, convenient for `export_to_video`.
- `"pt"`: Torch tensor output.
- `"latent"`: return latents and skip VAE decode/postprocess. Useful for
  advanced pipelines or debugging memory pressure.

If `return_dict=False`, the call returns a tuple whose first element is the
video output.

Export:

```python
from diffusers.utils import export_to_video

export_to_video(output.frames[0], "output.mp4", fps=24)
```

## 11. Backend Integration Checklist

For a local workflow server, expose HunyuanVideo1.5 as two explicit task
modes:

- `hunyuan_video15_t2v`: `HunyuanVideo15Pipeline`
- `hunyuan_video15_i2v`: `HunyuanVideo15ImageToVideoPipeline`

Recommended user-facing fields:

| Field | T2V | I2V | Notes |
| --- | --- | --- | --- |
| `model_id` | yes | yes | Restrict to known collection IDs or allow advanced override. |
| `prompt` | yes | yes | Use detailed motion/camera prompts. |
| `negative_prompt` | optional | optional | Keep empty by default unless product has standard negatives. |
| `image` | no | required | Must become `PIL.Image.Image` before pipeline call. |
| `height`, `width` | optional | no | T2V only. Require both or neither. |
| `num_frames` | optional | optional | Default 121. |
| `fps` | export only | export only | Default 24 for normal examples, 15 for short memory examples. |
| `num_inference_steps` | optional | optional | Default 50, except step-distilled I2V default 12. |
| `seed` | optional | optional | Create a CUDA generator when running on CUDA. |
| `attention_backend` | optional | optional | Validate installed kernels and GPU support. |
| `guidance_scale` | advanced | advanced | Do not pass to `pipe(...)`; apply through `pipe.guider.new(...)`. |
| `output_type` | optional | optional | `"np"` for user videos, `"latent"` for internal workflows. |

Runtime recommendations:

- Load one pipeline per checkpoint/task type, not both modes in a single
  object. The I2V class has extra SigLIP components.
- Keep `num_videos_per_prompt=1` unless batch memory has been measured.
- Always call `pipe.vae.enable_tiling()` for production-size video.
- Prefer `enable_model_cpu_offload()` over keeping every component resident on
  GPU unless the machine has ample VRAM.
- Free or evict pipelines aggressively in a multi-model local server; video
  models leave little room for other jobs.
- Record the exact model ID, `num_frames`, `fps`, dimensions, steps, seed,
  attention backend, dtype, and guider config with each job for reproducibility.

## 12. Gotchas

- `guidance_scale` is not a call parameter. Use
  `pipe.guider = pipe.guider.new(guidance_scale=...)`.
- T2V accepts `height` and `width`, but you must provide both or neither.
- I2V requires a `PIL.Image.Image`; raw bytes, paths, or arrays should be
  converted with `load_image` or PIL before calling the pipeline.
- If you pass embeddings, pass the matching masks. This applies to the main
  text encoder and the ByT5 glyph-aware text encoder.
- Quoted text in the prompt is meaningful for glyph conditioning. If you want
  legible text, quote the text exactly.
- The docs/model cards use both `torch.float16` and `torch.bfloat16` in
  examples. The main page and community cards favor BF16; use BF16 first on
  capable NVIDIA hardware.
- Model-card quick snippets using generic `DiffusionPipeline` may show
  image-style access like `.images[0]`. For HunyuanVideo1.5 pipeline-specific
  use, read `.frames[0]`.
- Attention backend names are hardware-sensitive. `_flash_3_hub` is for
  Hopper-class GPUs; use `flash_hub`/`flash_varlen_hub` or `sage_hub` as
  appropriate.
- `AutoencoderKLHunyuanVideo15` tiling is important. The VAE docs explicitly
  show `vae.enable_tiling()` to avoid OOM.
- Quantization is not HunyuanVideo1.5-specific in the docs. Treat it as an
  advanced integration path and test the component/backends combination before
  making it a default.

## 13. Source Links

- Diffusers HunyuanVideo-1.5 API page:
  https://huggingface.co/docs/diffusers/api/pipelines/hunyuan_video15
- Diffusers HunyuanVideo-1.5 docs source:
  https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/hunyuan_video15.md
- Text-to-video pipeline source:
  https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5.py
- Image-to-video pipeline source:
  https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5_image2video.py
- Pipeline output source:
  https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/hunyuan_video1_5/pipeline_output.py
- `HunyuanVideo15Transformer3DModel` docs source:
  https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/models/hunyuan_video15_transformer_3d.md
- `AutoencoderKLHunyuanVideo15` API page:
  https://huggingface.co/docs/diffusers/api/models/autoencoder_kl_hunyuan_video15
- HunyuanVideo1.5 Diffusers collection:
  https://huggingface.co/collections/hunyuanvideo-community/hunyuanvideo-15
- 480p T2V model card:
  https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v
- 720p T2V model card:
  https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v
- 480p I2V model card:
  https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v
- 720p I2V model card:
  https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v
- 480p I2V step-distilled model card:
  https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled
- Tencent original HunyuanVideo-1.5 model card:
  https://huggingface.co/tencent/HunyuanVideo-1.5
- Diffusers attention backends guide:
  https://huggingface.co/docs/diffusers/optimization/attention_backends
- Diffusers memory optimization guide:
  https://huggingface.co/docs/diffusers/optimization/memory
- Diffusers quantization guide:
  https://huggingface.co/docs/diffusers/quantization/overview
