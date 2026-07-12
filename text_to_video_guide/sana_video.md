# Sana Video Diffusers Implementation Guide

Last checked: 2026-06-18 against the Hugging Face Diffusers Sana Video API
page, the linked `v0.38.0` Diffusers source, the Sana Video transformer API
page, and official `Efficient-Large-Model` model cards.

Research target:
https://huggingface.co/docs/diffusers/api/pipelines/sana_video

Primary Diffusers classes on the page:

- `SanaVideoPipeline`
- `SanaImageToVideoPipeline`
- `SanaVideoPipelineOutput`

Core component class linked from the page:

- `SanaVideoTransformer3DModel`

SANA-Video is NVIDIA and MIT HAN Lab's efficient video generation model based
on a block linear diffusion transformer. The Diffusers page describes it as a
small video diffusion model designed for high-resolution and long video
generation, with the paper abstract highlighting two main implementation ideas:
linear attention for large video token counts and a constant-memory KV cache
for block linear attention.

## 1. Executive Summary

Use `SanaVideoPipeline` for prompt-only text-to-video generation and
`SanaImageToVideoPipeline` when a first-frame image should condition the video.
Both pipelines use the same Sana Video transformer and Gemma text stack, but
their schedulers and latent preparation differ.

Practical integration answer:

- Start with `Efficient-Large-Model/SANA-Video_2B_480p_diffusers`, the only
  model listed in the Diffusers API page's "Available models" table.
- Use `torch.bfloat16` for the transformer. The docs explicitly warn that the
  text encoder and VAE should stay in `torch.bfloat16` or `torch.float32`.
- Use the documented defaults first: `height=480`, `width=832`, `frames=81`,
  `num_inference_steps=50`, `guidance_scale=6`, and export at `fps=16`.
- Append the documented motion prompt string, such as
  `" motion score: 30."`, to the text prompt. Motion is not a separate pipeline
  argument.
- Keep `use_resolution_binning=True` for non-square outputs unless you have
  verified the exact target resolution. The source maps requested sizes to
  built-in 480p or 720p aspect-ratio bins before decoding and then resizes back.
- Enable CPU offload or quantization for local workstation use, and enable VAE
  tiling if the VAE decode path hits CUDA memory limits.
- Read generated video from `.frames[0]` and write it with
  `diffusers.utils.export_to_video(...)`.

## 2. Official Entry Points

- Pipeline docs: <https://huggingface.co/docs/diffusers/api/pipelines/sana_video>
- Docs source: <https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/sana_video.md>
- Text-to-video source: <https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/sana_video/pipeline_sana_video.py>
- Image-to-video source: <https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/sana_video/pipeline_sana_video_i2v.py>
- Output source: <https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/sana_video/pipeline_output.py>
- Transformer docs: <https://huggingface.co/docs/diffusers/api/models/sana_video_transformer3d>
- Original Sana repository: <https://github.com/NVlabs/Sana>
- Official model namespace: <https://huggingface.co/Efficient-Large-Model>

## 3. Pipeline Selection

| Class | Main task | Use it when |
| --- | --- | --- |
| `SanaVideoPipeline` | Text-to-video | The request contains only a prompt and optional negative prompt, seed, frame count, and resolution. |
| `SanaImageToVideoPipeline` | Image/text-to-video | The request includes a first-frame image that should anchor the generated video. |
| `SanaVideoPipelineOutput` | Shared output dataclass | You use `return_dict=True` and need the generated frames from `.frames`. |

`SanaVideoPipeline` uses `DPMSolverMultistepScheduler` in the constructor.
`SanaImageToVideoPipeline` uses `FlowMatchEulerDiscreteScheduler`. Both inherit
from `DiffusionPipeline` and `SanaLoraLoaderMixin`, so standard Diffusers
pipeline loading, offloading, saving, and LoRA helper behavior apply.

## 4. Checkpoints And Model IDs

| Model or asset | Documented use | Notes |
| --- | --- | --- |
| `Efficient-Large-Model/SANA-Video_2B_480p_diffusers` | Main Diffusers checkpoint for the API page | The docs list this as the available model with recommended transformer dtype `torch.bfloat16`. The model card says it is a 2B BF16 model for 480p, 81-frame, roughly 5-second videos with multi-scale height and width. Use this first for both `SanaVideoPipeline` and `SanaImageToVideoPipeline`. |
| `Efficient-Large-Model/SANA-Video_2B_720p_diffusers` | Official Diffusers 720p model card | Not listed in the API page's available-model table, but hosted by the same official namespace. The model card includes a `SanaVideoPipeline` stage-1 example at `height=704`, `width=1280`, `frames=81`, and `output_type="latent"` for an LTX2 refiner workflow. |
| `Efficient-Large-Model/SANA-Video_2B_480p` | Original 480p checkpoint | Official model card points to the original Sana guidance rather than the Diffusers pipeline page. Prefer the `_diffusers` repository for `from_pretrained(...)`. |
| `Efficient-Large-Model/SANA-Video_2B_480p_LongLive_diffusers` | Official long-video Diffusers checkpoint | The model card uses `LongSanaVideoPipeline`, `frames=161`, custom timesteps, and says the model targets 5s-60s / 81-961 frame 480p videos. `LongSanaVideoPipeline` is not one of the classes on the Sana Video API page covered by this guide. |

The Diffusers page links to an official Sana Video collection for more weights.
For a compatibility-focused implementation, expose the API-page model first and
add the 720p or long-video checkpoints only behind explicit model selection.

## 5. Installation

Use a Diffusers version that includes the Sana Video classes. The docs page is
rendered for `v0.38.0` and `main`.

```powershell
.venv\Scripts\python.exe -m pip install -U diffusers transformers accelerate torch imageio imageio-ffmpeg
```

For the documented bitsandbytes quantization path:

```powershell
.venv\Scripts\python.exe -m pip install -U bitsandbytes
```

For latest source behavior before a packaged release catches up:

```powershell
.venv\Scripts\python.exe -m pip install -U git+https://github.com/huggingface/diffusers
```

## 6. Shared Components

| Component | Class | Role |
| --- | --- | --- |
| `tokenizer` | `GemmaTokenizer` or `GemmaTokenizerFast` | Tokenizes prompts for the Gemma text encoder. |
| `text_encoder` | `Gemma2PreTrainedModel` | Produces text embeddings and attention masks for prompts. |
| `vae` | `AutoencoderKLWan`, `AutoencoderDC`, or `AutoencoderKLLTX2Video` | Encodes and decodes video frames to and from latent tensors. |
| `transformer` | `SanaVideoTransformer3DModel` | Denoises video latents conditioned on prompt embeddings. |
| `scheduler` | `DPMSolverMultistepScheduler` for T2V, `FlowMatchEulerDiscreteScheduler` for I2V | Converts predicted noise into the next latent sample during denoising. |

Source-level details that matter:

- Both pipelines use model CPU offload order
  `text_encoder -> transformer -> vae`.
- Callback tensor inputs are limited to `latents`, `prompt_embeds`, and
  `negative_prompt_embeds`.
- The VAE scale factors are read from the loaded VAE. For
  `AutoencoderKLLTX2Video`, the source uses
  `temporal_compression_ratio` and `spatial_compression_ratio`; for
  `AutoencoderDC` and `AutoencoderKLWan`, it uses
  `scale_factor_temporal` and `scale_factor_spatial`.
- The linked transformer source shows `SanaVideoTransformer3DModel` defaults such as
  `in_channels=16`, `out_channels=16`, `num_layers=20`,
  `num_attention_heads=20`, `attention_head_dim=112`,
  `caption_channels=2304`, `sample_size=30`, and
  `patch_size=(1, 2, 2)`.

## 7. Text-To-Video With `SanaVideoPipeline`

This follows the official Diffusers example but keeps the prompt short enough
to use as an integration smoke test.

```python
import torch
from diffusers import SanaVideoPipeline
from diffusers.utils import export_to_video

model_id = "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"

pipe = SanaVideoPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
pipe.text_encoder.to(torch.bfloat16)
pipe.vae.to(torch.float32)
pipe.to("cuda")

prompt = (
    "A cat and a dog bake a cake together in a cozy kitchen. The cat carefully "
    "measures flour while the dog stirs batter with a wooden spoon. Sunlight "
    "streams through the window and dust motes drift through the warm light."
)
negative_prompt = (
    "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, "
    "sudden disappearance, jump cuts, jerky movements, rapid shot changes, "
    "frames out of sync, inconsistent character shapes, temporal artifacts, "
    "jitter, and ghosting effects, creating a disorienting visual experience."
)

motion_score = 30
prompt = prompt + f" motion score: {motion_score}."

frames = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    frames=81,
    guidance_scale=6,
    num_inference_steps=50,
    generator=torch.Generator(device="cuda").manual_seed(0),
).frames[0]

export_to_video(frames, "sana_video.mp4", fps=16)
```

Important call behavior:

- `prompt` may be a string or list of strings. If omitted, provide
  `prompt_embeds` and `prompt_attention_mask`.
- Classifier-free guidance is enabled when `guidance_scale > 1.0`.
- If guidance is enabled and `negative_prompt_embeds` is not provided, the
  pipeline encodes `negative_prompt`.
- `timesteps` and `sigmas` are mutually exclusive custom scheduler controls.
  Custom `timesteps` must be descending.
- `latents` can be supplied to reuse or externally control the initial noise.
  Otherwise the pipeline samples latents with `randn_tensor`.
- The source denoises latents in `torch.float32` and feeds transformer inputs
  cast to the transformer's dtype.

## 8. Image/Text-To-Video With `SanaImageToVideoPipeline`

Use this pipeline when the first frame is known. The source encodes the input
image through the VAE, writes that latent into the first latent frame, and masks
that first temporal patch during denoising so the remaining frames are
generated around the image condition.

```python
import torch
from diffusers import SanaImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

model_id = "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"

pipe = SanaImageToVideoPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
pipe.text_encoder.to(torch.bfloat16)
pipe.vae.to(torch.float32)
pipe.to("cuda")

image = load_image(
    "https://raw.githubusercontent.com/NVlabs/Sana/refs/heads/main/asset/samples/i2v-1.png"
)

prompt = (
    "A woman stands against a warm sunset backdrop, her long wavy hair moving "
    "gently in the breeze. The camera remains steady in a medium close-up while "
    "soft rolling hills and scattered clouds blur in the background."
)
negative_prompt = (
    "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, "
    "sudden disappearance, jump cuts, jerky movements, rapid shot changes, "
    "frames out of sync, inconsistent character shapes, temporal artifacts, "
    "jitter, and ghosting effects, creating a disorienting visual experience."
)

motion_score = 30
prompt = prompt + f" motion score: {motion_score}."

frames = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    frames=81,
    guidance_scale=6,
    num_inference_steps=50,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

export_to_video(frames, "sana_i2v.mp4", fps=16)
```

Implementation details specific to I2V:

- The public docs type `image` as `PipelineImageInput` and describe it as the
  image used to condition the first frame.
- The `v0.38.0` source-level `check_inputs` is stricter than the rendered type:
  it accepts a `torch.Tensor` or a single `PIL.Image.Image` before preprocessing.
  For maximum compatibility, pass a single PIL image or tensor rather than a
  list or NumPy array.
- The pipeline preprocesses the image to the requested `height` and `width`
  before VAE encoding.
- The source sets `conditioning_mask[:, :, 0] = 1.0`, multiplies timesteps by
  `(1 - conditioning_mask)`, updates only `latents[:, :, 1:]`, and concatenates
  the preserved first latent frame back after each scheduler step.

## 9. Prompt Enhancement And Motion Prompting

The Sana Video API page documents a `complex_human_instruction` parameter for
both pipelines and links it to the Sana app configuration. In the source, this
is not a separate LLM call. The list of instruction strings is joined and
prepended to the user's prompt before Gemma tokenization. The default
instruction asks for an "Enhanced prompt" and gives examples such as expanding
"A cat sleeping" into a more detailed visual and temporal description.

Use cases:

- Leave the default `complex_human_instruction` in place if you want the prompt
  encoder to see the documented enhancement instruction.
- Pass `complex_human_instruction=[]` or `None` if your application already
  performs prompt expansion and you want the model to encode only the user's
  prompt.
- Keep `max_sequence_length=300` unless you have verified truncation behavior
  with your prompt templates.

The docs examples also append a motion prompt:

```python
motion_score = 30
prompt = prompt + f" motion score: {motion_score}."
```

This is plain text appended to the prompt. It is not parsed as a separate
argument. A value around `30` is the documented example value; tune it as prompt
text and validate visually for the amount of motion your workflow expects.

The official examples use a long negative prompt focused on temporal artifacts:
motion blur, jump cuts, jerky motion, out-of-sync frames, inconsistent shapes,
jitter, and ghosting. Reuse that as a baseline negative prompt for early tests.

## 10. Key Parameters

| Parameter | Applies to | Meaning and implementation notes |
| --- | --- | --- |
| `image` | `SanaImageToVideoPipeline` only | First-frame conditioning image. Use a PIL image or tensor for best compatibility with `v0.38.0`. |
| `prompt` | Both | Text prompt or list of prompts. Mutually exclusive with `prompt_embeds`. |
| `negative_prompt` | Both | Negative prompt used when classifier-free guidance is active and custom negative embeddings are not supplied. |
| `num_inference_steps` | Both | Denoising step count. Default is `50`; higher usually costs more time and may improve quality. |
| `timesteps` | Both | Optional custom scheduler timesteps. Mutually exclusive with `sigmas`; docs say timesteps must be descending. |
| `sigmas` | Both | Optional custom sigma schedule for schedulers that support it. Mutually exclusive with `timesteps`. |
| `guidance_scale` | Both | Classifier-free guidance strength. Default signature value is `6.0`; guidance is active only above `1.0`. |
| `num_videos_per_prompt` | Both | Number of videos generated per prompt. Generator lists must match the effective batch size. |
| `height`, `width` | Both | Output size in pixels. Defaults are `480` and `832`; source validation requires dimensions divisible by `32` after binning. |
| `frames` | Both | Number of output frames. Default is `81`. Use `frames`, not `num_frames`. |
| `eta` | Both | DDIM eta parameter. The docs note it is ignored by non-DDIM schedulers. |
| `generator` | Both | A `torch.Generator` or list of generators for deterministic sampling. |
| `latents` | Both | Optional pre-generated latent tensor. Shape must match the effective batch, latent channels, latent frames, and latent spatial size. |
| `prompt_embeds`, `prompt_attention_mask` | Both | Custom prompt embeddings and masks. If you pass embeddings, you must also pass the attention mask. |
| `negative_prompt_embeds`, `negative_prompt_attention_mask` | Both | Custom negative embeddings and masks. Source validation requires matching shapes with prompt embeddings and masks. |
| `output_type` | Both | Default is `"pil"`. Source also supports `"latent"` to skip VAE decoding and return latents. |
| `return_dict` | Both | Default `True` returns `SanaVideoPipelineOutput`; `False` returns a tuple whose first element is the generated videos. |
| `clean_caption` | Both | Signature default is `False`; set explicitly if you want caption cleaning. Cleaning requires `beautifulsoup4` and `ftfy`. |
| `use_resolution_binning` | Both | Default `True`. Maps target size to the nearest supported aspect-ratio bin before generation, then resizes/crops back. |
| `attention_kwargs` | Both | Passed to the active attention processor. |
| `callback_on_step_end` | Both | Called after each denoising step with selected tensors. |
| `callback_on_step_end_tensor_inputs` | Both | Only `latents`, `prompt_embeds`, and `negative_prompt_embeds` are accepted. |
| `max_sequence_length` | Both | Prompt token sequence length. Default is `300`. |
| `complex_human_instruction` | Both | Optional prompt-enhancement instruction list prepended before prompt tokenization. |

## 11. Resolution And Frame Settings

The official examples use:

| Setting | Value |
| --- | --- |
| Height | `480` |
| Width | `832` |
| Frames | `81` |
| Export FPS | `16` |
| Approximate duration | About 5 seconds |

The source computes latent frame count as:

```python
num_latent_frames = (frames - 1) // vae_scale_factor_temporal + 1
```

Latent spatial size is:

```python
latent_height = height // vae_scale_factor_spatial
latent_width = width // vae_scale_factor_spatial
```

When `use_resolution_binning=True`, the source selects bins based on
`transformer.config.sample_size`:

- `sample_size == 30`: use the 480p bin table.
- `sample_size == 22`: use the 720p bin table.
- Any other sample size raises `ValueError("Invalid sample size")`.

480p bins from the source:

| Aspect ratio key | Binned height x width |
| --- | --- |
| `0.5` | `448 x 896` |
| `0.57` | `480 x 832` |
| `0.68` | `528 x 768` |
| `0.78` | `560 x 720` |
| `1.0` | `624 x 624` |
| `1.13` | `672 x 592` |
| `1.29` | `720 x 560` |
| `1.46` | `768 x 528` |
| `1.67` | `816 x 496` |
| `1.75` | `832 x 480` |
| `2.0` | `896 x 448` |

720p bins from the source:

| Aspect ratio key | Binned height x width |
| --- | --- |
| `0.5` | `672 x 1344` |
| `0.57` | `704 x 1280` |
| `0.68` | `800 x 1152` |
| `0.78` | `832 x 1088` |
| `1.0` | `960 x 960` |
| `1.13` | `1024 x 896` |
| `1.29` | `1088 x 832` |
| `1.46` | `1152 x 800` |
| `1.67` | `1248 x 736` |
| `1.75` | `1280 x 704` |
| `2.0` | `1344 x 672` |

For a server integration, validate requested dimensions before enqueueing the
job. If you keep resolution binning enabled, store both requested and effective
generation dimensions in logs because a request for an arbitrary size may be
generated at the nearest bin and resized back after decoding.

## 12. Outputs

Both pipelines return `SanaVideoPipelineOutput` when `return_dict=True`:

```python
result = pipe(...)
frames = result.frames
video = frames[0]
```

The output dataclass documents `.frames` as a `torch.Tensor`, `np.ndarray`, or
nested list of PIL images. For PIL output, the shape is conceptually
`batch_size` videos, each with `num_frames` decoded frames. For tensor/array
output, the documented shape is `(batch_size, num_frames, channels, height,
width)`.

If `return_dict=False`, the pipeline returns a tuple and the first element is
the generated videos.

If `output_type="latent"`, the source skips VAE decoding and returns the latent
tensor in `.frames`. This is useful when chaining Sana Video latents into a
second-stage refiner or avoiding VAE decode memory during tests.

The docs examples export with:

```python
from diffusers.utils import export_to_video

export_to_video(frames, "sana_video.mp4", fps=16)
```

## 13. Memory, Performance, And Quantization

Recommended standard loading:

```python
pipe = SanaVideoPipeline.from_pretrained(
    "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.text_encoder.to(torch.bfloat16)
pipe.vae.to(torch.float32)
pipe.to("cuda")
```

For lower peak VRAM, prefer model CPU offload:

```python
pipe.enable_model_cpu_offload()
```

For VAE decode memory issues, the source catches accelerator OOM errors and
warns to enable VAE tiling:

```python
pipe.vae.enable_tiling(
    tile_sample_min_width=512,
    tile_sample_min_height=512,
)
```

The docs include an 8-bit bitsandbytes example that quantizes both the Gemma
text encoder and `SanaVideoTransformer3DModel`, then creates the pipeline with
`device_map="balanced"`:

```python
import torch
from diffusers import (
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
    SanaVideoPipeline,
    SanaVideoTransformer3DModel,
)
from transformers import AutoModel, BitsAndBytesConfig

model_id = "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"

text_encoder_8bit = AutoModel.from_pretrained(
    model_id,
    subfolder="text_encoder",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    torch_dtype=torch.float16,
)

transformer_8bit = SanaVideoTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=DiffusersBitsAndBytesConfig(load_in_8bit=True),
    torch_dtype=torch.float16,
)

pipe = SanaVideoPipeline.from_pretrained(
    model_id,
    text_encoder=text_encoder_8bit,
    transformer=transformer_8bit,
    torch_dtype=torch.float16,
    device_map="balanced",
)
```

Quantization notes:

- Quantization reduces memory, but the docs warn it can affect video quality.
- The API page's dtype note is important for normal, non-quantized loading:
  do not freely cast the text encoder or VAE to unsupported low precision.
- The SANA-Video paper and docs mention RTX 5090 NVFP4 deployment, but the
  Diffusers API page demonstrates bitsandbytes 8-bit loading, not an NVFP4
  loading recipe.
- `output_type="latent"` can reduce memory if your workflow does not need to
  decode immediately.

## 14. Implementation Gotchas

- Use `frames`, not `num_frames`. The quantization snippet on the rendered docs
  page passes `num_frames=81`, but the `SanaVideoPipeline.__call__` signature
  is `frames=81`.
- The docs parameter prose says `guidance_scale` defaults to `4.5`, but the
  signatures and examples use `6.0`. Treat `6.0` as the Sana Video docs example
  baseline.
- The docs prose for `output_type` says to choose between mp4 or `np.array`,
  but the source default is `"pil"` and the examples export PIL frames with
  `export_to_video`. Do not assume the pipeline writes an mp4 directly.
- The docs prose for `clean_caption` says default `True`, while the signatures
  and source default to `False`. Set it explicitly if caption cleaning matters.
- The transformer API prose says `sample_size` defaults to `32`, while the
  linked source constructor default is `30`; the pipeline resolution binning
  logic relies on the loaded transformer's config value.
- The rendered docs include copy-pasted PixArt references in negative prompt
  embedding descriptions. For Sana Video, the source defaults
  `negative_prompt=""` and says Sana negative embeddings should correspond to
  the empty string when precomputed.
- If passing custom prompt embeddings, also pass `prompt_attention_mask`. If
  passing custom negative embeddings, also pass `negative_prompt_attention_mask`.
- Source validation does not allow `prompt` together with
  `negative_prompt_embeds`; if you provide custom negative embeddings, provide
  prompt embeddings too.
- Height and width must be divisible by `32` after any resolution binning.
- `use_resolution_binning=True` can change the actual generation size before
  resizing back. This is useful, but it can surprise tests that inspect internal
  latent shapes.
- The I2V docs type accepts broad `PipelineImageInput`, but `v0.38.0` source
  validation is narrower. Use a single PIL image or tensor until you verify the
  installed Diffusers version accepts lists or NumPy arrays.
- The source supports only callback tensor names in
  `["latents", "prompt_embeds", "negative_prompt_embeds"]`.
- If the transformer has an unknown `sample_size`, resolution binning raises
  `ValueError("Invalid sample size")`.
- The long-video checkpoint uses `LongSanaVideoPipeline`, which is outside the
  two pipeline classes on this API page.

## 15. Server Integration Checklist

Recommended request fields for a local workflow server:

| Field | Suggested default | Validation |
| --- | --- | --- |
| `pipeline` | `"sana_video"` or `"sana_image_to_video"` | Map to `SanaVideoPipeline` or `SanaImageToVideoPipeline`. |
| `model_id` | `Efficient-Large-Model/SANA-Video_2B_480p_diffusers` | Start with the API-page model. Gate other official checkpoints explicitly. |
| `prompt` | Required | String or list. Apply your own prompt expansion before calling if disabling `complex_human_instruction`. |
| `negative_prompt` | Official temporal-artifact negative prompt | Optional string. |
| `motion_score` | `30` | Append as prompt text: `f" motion score: {motion_score}."`. |
| `image` | Required for I2V only | Convert to PIL image or tensor before pipeline call. |
| `height`, `width` | `480`, `832` | Keep divisible by 32 or allow resolution binning. |
| `frames` | `81` | Validate positive integer; use smaller values for smoke tests only after verifying model behavior. |
| `fps` | `16` | Used by `export_to_video`, not by the pipeline call. |
| `num_inference_steps` | `50` | Expose lower values for previews and documented value for quality. |
| `guidance_scale` | `6.0` | Guidance active above `1.0`. |
| `seed` | Optional | Build `torch.Generator(device="cuda").manual_seed(seed)`. |
| `output_type` | `"pil"` | Use `"latent"` only for chaining/refiner workflows. |
| `use_resolution_binning` | `True` | Log effective dimensions when enabled. |

Minimal cache policy:

- Cache one loaded pipeline per `(pipeline_class, model_id, dtype_policy)`.
- Keep text encoder and VAE dtype handling explicit in the cache key.
- For mixed T2V/I2V serving, avoid sharing scheduler assumptions between the
  two pipeline classes because the constructors document different schedulers.
- If you switch from `.to("cuda")` to `enable_model_cpu_offload()`, recreate or
  carefully reinitialize the pipeline rather than mixing device placement modes.

## 16. Source Compatibility Notes

The official docs and source are mostly aligned, but the guide intentionally
tracks source behavior where it affects implementation:

- `SanaVideoPipeline.__call__` and `SanaImageToVideoPipeline.__call__` both use
  `frames`, not `num_frames`.
- Source default `clean_caption=False` wins over the prose default.
- Source output handling supports `output_type="latent"` and otherwise
  delegates to `VideoProcessor.postprocess_video`.
- The I2V source preserves the first frame by replacing the first latent frame
  with encoded image latents and only denoising the later frames.
- The API page's "Available models" table lists only the 480p Diffusers
  checkpoint, even though official model cards exist for other Sana Video
  checkpoints.
