# HunyuanVideo Diffusers Implementation Guide

Last checked: 2026-06-17 against the Hugging Face Diffusers HunyuanVideo API
page, the linked Diffusers docs source, and the linked `v0.38.0` source files.

Research target:
https://huggingface.co/docs/diffusers/api/pipelines/hunyuan_video

Primary Diffusers classes:

| Class | Role |
| --- | --- |
| `HunyuanVideoPipeline` | Text-to-video pipeline for HunyuanVideo. This is the main class to integrate first. |
| `HunyuanVideoPipelineOutput` | Output container with generated video frames. |

This guide is a docs-only implementation reference for adding or evaluating
HunyuanVideo support. It does not change SynthaEngine runtime behavior and it
does not cover the separate HunyuanVideo1.5 page except to note that
HunyuanVideo1.5 is a different pipeline family.

## 1. Executive Summary

HunyuanVideo is Tencent's 13B parameter video diffusion transformer family. The
Diffusers page currently documents one pipeline, `HunyuanVideoPipeline`, for
text-to-video generation. It uses a dual text stack, a large 3D transformer, a
FlowMatch Euler scheduler, and a HunyuanVideo-specific 3D causal VAE.

Practical integration answer:

| Question | Answer |
| --- | --- |
| Main task | Text-to-video. |
| Main pipeline | `HunyuanVideoPipeline`. |
| Diffusers-compatible checkpoint in examples | `hunyuanvideo-community/HunyuanVideo`. |
| Original checkpoint organization | Tencent checkpoints are under `https://huggingface.co/tencent`; the official docs use the community repo because it is laid out for Diffusers loading. |
| Recommended example size | The official source example uses `height=320`, `width=512`, `num_frames=61`, `num_inference_steps=30`, exported at 15 fps. |
| Default size | `height=720`, `width=1280`, `num_frames=129`, `num_inference_steps=50`. |
| Frame rule | Docs recommend `num_frames = 4 * k + 1`. |
| Memory strategy | Quantize the transformer to bitsandbytes int4, enable model CPU offload, and enable VAE tiling. The docs say the quantized example requires about 14 GB VRAM. |
| Speed strategy | Use the same quantized/offloaded setup, then compile the transformer with `torch.compile`; first run is slow, later calls are faster. |
| LoRA | Supported with `pipeline.load_lora_weights()` and `pipeline.set_adapters()`. |

For a local server, treat this as a heavyweight runtime. Start with a small
smoke-test resolution, one video per prompt, 61 frames, 30 steps, CPU offload,
VAE tiling, and explicit cleanup after each job unless the product intentionally
keeps the model resident.

## 2. What HunyuanVideo Is

The Diffusers page describes HunyuanVideo as a 13B diffusion transformer model
designed to compete with closed-source video foundation models. The documented
architecture has three implementation-relevant traits:

1. The transformer uses a "dual-stream to single-stream" design. Video and text
   tokens are processed separately first, then concatenated so the transformer
   can fuse multimodal information.
2. The text path uses a pretrained multimodal large language model as the main
   encoder. Diffusers exposes this as a Llama-based encoder, with a second CLIP
   text encoder for pooled projections.
3. The VAE is a 3D causal variational autoencoder. It compresses spatially and
   temporally, then decodes latent videos back to frame sequences.

The Diffusers integration is not a generic image pipeline with a video wrapper.
It prepares 5D latent tensors, denoises them with
`HunyuanVideoTransformer3DModel`, decodes with `AutoencoderKLHunyuanVideo`, and
returns video frames.

## 3. Checkpoints And Model IDs

Use the Diffusers-compatible checkpoint for normal Diffusers loading:

```text
hunyuanvideo-community/HunyuanVideo
```

The official Diffusers page says the original HunyuanVideo checkpoints are
under Tencent's Hugging Face organization, but the page examples use
`hunyuanvideo-community/HunyuanVideo` because its weights are stored in a layout
compatible with Diffusers.

Useful IDs and links:

| ID or organization | Use |
| --- | --- |
| `hunyuanvideo-community/HunyuanVideo` | Main Diffusers-compatible pipeline checkpoint used by the docs and model card examples. |
| `tencent/HunyuanVideo` | Original Tencent model repo and upstream model card. Do not assume direct `HunyuanVideoPipeline.from_pretrained()` loading from this layout without testing. |
| `hunyuanvideo-community` | Organization hosting Diffusers-layout community weights. |
| `lucataco/hunyuan-steamboat-willie-10` | LoRA example used by the official Diffusers docs. |

The community model card has a generic "Use this model" block that looks like
an image pipeline snippet, but the correct Diffusers example on that card uses
`HunyuanVideoPipeline` and reads `.frames[0]`. Follow the pipeline docs and the
"Using Diffusers" block, not generic Hub auto-snippets.

## 4. Installation Notes

Use a recent Diffusers release with the HunyuanVideo pipeline. The docs page is
available on the current Diffusers docs and links to `v0.38.0` source.

```powershell
.venv\Scripts\python.exe -m pip install -U diffusers transformers accelerate torch
.venv\Scripts\python.exe -m pip install -U bitsandbytes imageio imageio-ffmpeg
```

`bitsandbytes` is needed for the official int4 quantization example.
`imageio` and `imageio-ffmpeg` are useful when exporting videos with
`diffusers.utils.export_to_video()`.

## 5. Pipeline And Components

`HunyuanVideoPipeline` is a `DiffusionPipeline` subclass and mixes in
`HunyuanVideoLoraLoaderMixin`, which is why `load_lora_weights()` is supported.

Constructor components documented by Diffusers:

| Component | Type | Purpose |
| --- | --- | --- |
| `text_encoder` | `LlamaModel` | Main Llama-based MLLM text encoder. Docs link it to `xtuner/llava-llama-3-8b-v1_1-transformers`. |
| `tokenizer` | `LlamaTokenizerFast` in source, documented as Llama tokenizer | Tokenizer for the main text encoder. |
| `text_encoder_2` | `CLIPTextModel` | Second text encoder, specifically CLIP `clip-vit-large-patch14`, used for pooled prompt projections. |
| `tokenizer_2` | `CLIPTokenizer` | Tokenizer for the CLIP text encoder. |
| `transformer` | `HunyuanVideoTransformer3DModel` | Conditional video transformer that denoises encoded latent videos. |
| `vae` | `AutoencoderKLHunyuanVideo` | 3D VAE that decodes latent videos to frames and can encode videos for lower-level workflows. |
| `scheduler` | `FlowMatchEulerDiscreteScheduler` | Flow matching scheduler used for denoising. |
| `video_processor` | `VideoProcessor` created by the pipeline | Postprocesses decoded video tensors to PIL/NumPy-style frame outputs. |

Source-level runtime details worth knowing:

- Model CPU offload order is `text_encoder -> text_encoder_2 -> transformer -> vae`.
- Callback tensor inputs are limited to `latents` and `prompt_embeds`.
- The VAE spatial compression ratio is 8 and temporal compression ratio is 4.
- The normal latent shape is based on transformer input channels and VAE
  compression:

```text
(batch, transformer.config.in_channels,
 (num_frames - 1) // 4 + 1,
 height // 8,
 width // 8)
```

### HunyuanVideoTransformer3DModel

The transformer component is documented separately in the Diffusers model API.
Important defaults from the docs:

| Parameter | Default | Notes |
| --- | --- | --- |
| `in_channels`, `out_channels` | `16`, `16` | Latent channel dimensions for the denoiser. |
| `num_attention_heads` | `24` | Multi-head attention heads. |
| `attention_head_dim` | `128` | Per-head channel dimension. |
| `num_layers` | `20` | Dual-stream block count. |
| `num_single_layers` | `40` | Single-stream block count after fusion. |
| `num_refiner_layers` | `2` | Refiner block count. |
| `patch_size`, `patch_size_t` | `2`, `1` | Spatial and temporal patch sizes. |
| `qk_norm` | `rms_norm` | Query/key normalization type. |
| `guidance_embeds` | `True` | Enables embedded guidance inputs. |
| `text_embed_dim` | `4096` | Main text embedding dimension. |
| `pooled_projection_dim` | `768` | CLIP pooled projection dimension. |
| `rope_theta`, `rope_axes_dim` | `256.0`, `(16, 56, 56)` | Rotary embedding settings. |
| `image_condition_type` | `None` | Component supports `latent_concat` and `token_replace`, but `HunyuanVideoPipeline` is documented as text-to-video. |

For the documented pipeline, do not expose image-conditioning controls unless a
separate Diffusers pipeline or custom workflow is added and validated.

### AutoencoderKLHunyuanVideo

The VAE component is also documented separately. Important defaults:

| Parameter | Default | Notes |
| --- | --- | --- |
| `in_channels`, `out_channels` | `3`, `3` | RGB video frames in and out. |
| `latent_channels` | `16` | Latent channel count. |
| `scaling_factor` | `0.476986` | Used by the pipeline before VAE decode. |
| `spatial_compression_ratio` | `8` | Height and width are compressed by 8 in latents. |
| `temporal_compression_ratio` | `4` | Frames are compressed by 4, hence the `4 * k + 1` guidance. |
| `mid_block_add_attention` | `True` | Attention in the VAE mid block. |

Useful VAE methods:

- `vae.enable_tiling(...)`: split decoding and encoding into overlapping
  spatial and temporal tiles to save memory.
- `vae.disable_tiling()`: return to non-tiled VAE work.
- `vae.tiled_encode(x)`: encode a video batch with tiled encoding.
- `vae.tiled_decode(z, return_dict=True)`: decode latents with tiled decoding.
- `vae.forward(sample, sample_posterior=False, return_dict=True, generator=None)`:
  encode then decode a sample.

The pipeline still exposes `enable_vae_tiling()`, `disable_vae_tiling()`,
`enable_vae_slicing()`, and `disable_vae_slicing()` through inherited-style
helpers, and those methods are listed on the pipeline docs page. The `v0.38.0`
source marks those pipeline-level helpers as deprecated for removal in `0.40.0`
and tells users to call `pipe.vae.enable_tiling()` or
`pipe.vae.enable_slicing()` directly. The official examples already use
`pipeline.vae.enable_tiling()`.

## 6. Prompt Encoding

HunyuanVideo uses two prompt encoders:

1. The Llama text encoder produces sequence embeddings and an attention mask.
2. The CLIP text encoder produces pooled prompt embeddings.

The public `__call__` arguments split this as:

- `prompt`: prompt for the main Llama text encoder.
- `prompt_2`: prompt intended for `tokenizer_2` and `text_encoder_2`; docs say
  it defaults to `prompt`.
- `negative_prompt`: negative prompt for the main encoder, used only with true
  classifier-free guidance.
- `negative_prompt_2`: negative prompt intended for the second text encoder,
  defaulting to `negative_prompt`.
- `prompt_embeds`, `pooled_prompt_embeds`, `prompt_attention_mask`: precomputed
  positive embeddings.
- `negative_prompt_embeds`, `negative_pooled_prompt_embeds`,
  `negative_prompt_attention_mask`: precomputed negative embeddings.

The default prompt template wraps the user's prompt in a Llama chat-style
system/user template. The system instruction asks the encoder to describe:

- main content and theme;
- object color, shape, size, texture, quantity, text, and spatial relations;
- actions, events, temporal relationships, and motion changes;
- background environment, lighting, style, and atmosphere;
- camera angles, movement, and transitions.

The default template has `crop_start=95`, so the pipeline crops away the system
prompt prefix embeddings after encoding. If a custom `prompt_template` omits
`crop_start`, the source computes it from the template and subtracts two tokens
for the end token and `{}` placeholder. A custom template must be a dictionary
with a `template` key.

`max_sequence_length` defaults to 256 for the Llama path. The CLIP path uses a
fixed max length of 77 and warns if CLIP truncates the input.

Source gotcha: in the linked `v0.38.0` source, `encode_prompt()` sets
`prompt_2 = prompt` when `prompt_2` is missing, but the CLIP helper call passes
`prompt` rather than `prompt_2`. If your integration needs a separate CLIP
prompt, verify the installed Diffusers source before exposing `prompt_2` as a
meaningful independent control.

## 7. Key `__call__` Parameters

The documented signature is:

```python
pipeline(
    prompt=None,
    prompt_2=None,
    negative_prompt=None,
    negative_prompt_2=None,
    height=720,
    width=1280,
    num_frames=129,
    num_inference_steps=50,
    sigmas=None,
    true_cfg_scale=1.0,
    guidance_scale=6.0,
    num_videos_per_prompt=1,
    generator=None,
    latents=None,
    prompt_embeds=None,
    pooled_prompt_embeds=None,
    prompt_attention_mask=None,
    negative_prompt_embeds=None,
    negative_pooled_prompt_embeds=None,
    negative_prompt_attention_mask=None,
    output_type="pil",
    return_dict=True,
    attention_kwargs=None,
    callback_on_step_end=None,
    callback_on_step_end_tensor_inputs=["latents"],
    prompt_template=DEFAULT_PROMPT_TEMPLATE,
    max_sequence_length=256,
)
```

Parameter guidance:

| Parameter | Integration guidance |
| --- | --- |
| `prompt` | Required unless precomputed `prompt_embeds` are provided. Accepts `str` or `list[str]`. |
| `prompt_2` | Documented as the second-encoder prompt. Verify source behavior if relying on it separately from `prompt`. |
| `negative_prompt`, `negative_prompt_2` | Only affect generation when `true_cfg_scale > 1` and a negative prompt or negative embeddings are supplied. |
| `height`, `width` | Defaults are 720x1280. Source validation requires both divisible by 16. Use 320x512 for a first smoke test. |
| `num_frames` | Defaults to 129. Docs recommend `4 * k + 1`; example uses 61. |
| `num_inference_steps` | Defaults to 50. Example uses 30. More steps may improve quality but cost time. |
| `sigmas` | Optional custom sigma schedule for schedulers whose `set_timesteps()` supports `sigmas`. If omitted, source builds a linear schedule from 1.0 to 0.0 with `num_inference_steps` entries. |
| `true_cfg_scale` | Default 1.0. True CFG activates only when greater than 1 and negative conditioning exists. |
| `guidance_scale` | Default 6.0. Embedded guidance scale; higher values usually increase prompt adherence but can reduce visual quality. Source passes `guidance_scale * 1000` to the transformer. |
| `num_videos_per_prompt` | Defaults to 1. Increases batch and memory linearly enough that server defaults should keep it at 1. |
| `generator` | `torch.Generator` or list of generators for deterministic sampling. If passing a list, its length must match the effective batch size. |
| `latents` | Optional pre-generated latents. Use for controlled variation, advanced workflows, or latent output reuse. |
| `output_type` | Docs list PIL or NumPy-style outputs. Source also bypasses VAE decode when set to `"latent"`. |
| `return_dict` | Defaults to `True`, returning `HunyuanVideoPipelineOutput`. |
| `attention_kwargs` | Forwarded to the attention processor. Leave unset unless you have a tested processor-specific use. |
| `callback_on_step_end` | Called after each denoising step. Can receive only tensors listed in `callback_on_step_end_tensor_inputs`. |
| `callback_on_step_end_tensor_inputs` | Defaults to `["latents"]`; source allows only `latents` and `prompt_embeds`. |
| `prompt_template` | Defaults to the HunyuanVideo Llama system/user template. Custom templates must include `template`. |
| `max_sequence_length` | Defaults to 256 for the main text encoder. |

The autodoc prose currently mentions `clip_skip`, but the linked source
signature does not include a `clip_skip` argument. Do not expose `clip_skip`
unless the installed Diffusers version actually accepts it.

## 8. Resolution, Frames, And Scheduler Shift

Safe starting values from official examples:

```text
height=320
width=512
num_frames=61
num_inference_steps=30
fps=15 for export
```

Production-quality defaults from the signature are much heavier:

```text
height=720
width=1280
num_frames=129
num_inference_steps=50
```

Frame count:

- Use `num_frames = 4 * k + 1`, for example 61 or 129.
- This aligns with the VAE temporal compression ratio of 4.
- Avoid arbitrary frame counts unless you test temporal quality and boundary
  behavior.

Resolution:

- `height` and `width` must be divisible by 16.
- The VAE compresses spatial dimensions by 8, while the transformer uses
  patches, so multiples of 16 are the pipeline's validation boundary.
- Start at 320x512 or another small 16-divisible size for smoke tests.

Scheduler shift:

- The docs recommend lower `shift` values, around 2.0 to 5.0, for lower
  resolution videos.
- They recommend higher `shift` values, around 7.0 to 12.0, for higher
  resolution outputs.
- Treat `shift` as a scheduler configuration knob, not a `__call__` argument.
  A typical pattern is to recreate the scheduler from config with a new shift
  and then assign it to `pipe.scheduler`, but validate this against the exact
  Diffusers version in use.

## 9. Baseline Text-To-Video Example

This follows the official example shape and keeps the transformer in BF16 while
the rest of the pipeline loads in FP16.

```python
import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

model_id = "hunyuanvideo-community/HunyuanVideo"

transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)

pipe = HunyuanVideoPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.float16,
)
pipe.vae.enable_tiling()
pipe.to("cuda")

frames = pipe(
    prompt="A cat walks on the grass, realistic",
    height=320,
    width=512,
    num_frames=61,
    num_inference_steps=30,
).frames[0]

export_to_video(frames, "output.mp4", fps=15)
```

Use this as the first correctness test when a GPU has enough memory. For a
server integration, prefer the quantized/offloaded example below unless the
target hardware is known to handle the full model comfortably.

## 10. Memory-Optimized Example

The official memory example quantizes the transformer to bitsandbytes int4,
enables model CPU offload, and enables VAE tiling. The docs say this quantized
HunyuanVideo model requires about 14 GB VRAM.

```python
import torch
from diffusers import HunyuanVideoPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.utils import export_to_video

pipeline_quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
    },
    components_to_quantize="transformer",
)

pipe = HunyuanVideoPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
)

pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

frames = pipe(
    prompt="A fluffy teddy bear sits on a bed of soft pillows surrounded by children's toys.",
    num_frames=61,
    num_inference_steps=30,
).frames[0]

export_to_video(frames, "output.mp4", fps=15)
```

Implementation notes:

- Quantize the `transformer`; it is the main memory target.
- Keep `num_videos_per_prompt=1` for local serving.
- Keep `pipe.vae.enable_tiling()` on for larger frame sizes.
- Offload is slower than keeping everything on GPU, but it is the documented
  way to make the example fit a smaller VRAM budget.

## 11. Speed-Oriented Example

The official speed tab uses the same int4 quantized setup, then compiles the
transformer. Compilation is slow on the first call and faster on later calls.

```python
import torch
from diffusers import HunyuanVideoPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.utils import export_to_video

pipeline_quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
    },
    components_to_quantize="transformer",
)

pipe = HunyuanVideoPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
)

pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

pipe.transformer.to(memory_format=torch.channels_last)
pipe.transformer = torch.compile(
    pipe.transformer,
    mode="max-autotune",
    fullgraph=True,
)

frames = pipe(
    prompt="A fluffy teddy bear sits on a bed of soft pillows surrounded by children's toys.",
    num_frames=61,
    num_inference_steps=30,
).frames[0]

export_to_video(frames, "output.mp4", fps=15)
```

For an API server, compile only if the process is long-lived enough to amortize
the first-run compilation cost. Short one-shot workers usually benefit more
from quantization and offload than from compile.

## 12. LoRA Example

The HunyuanVideo page explicitly says LoRA is supported through
`load_lora_weights()`. The official example loads the Steamboat Willie LoRA and
then uses a trigger phrase in the prompt.

```python
import torch
from diffusers import HunyuanVideoPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.utils import export_to_video

pipeline_quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
    },
    components_to_quantize="transformer",
)

pipe = HunyuanVideoPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
)

pipe.load_lora_weights(
    "https://huggingface.co/lucataco/hunyuan-steamboat-willie-10",
    adapter_name="steamboat-willie",
)
pipe.set_adapters("steamboat-willie", 0.9)

pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

prompt = (
    "In the style of SWR. A black and white animated scene featuring a fluffy "
    "teddy bear sits on a bed of soft pillows surrounded by children's toys."
)

frames = pipe(
    prompt=prompt,
    num_frames=61,
    num_inference_steps=30,
).frames[0]

export_to_video(frames, "output.mp4", fps=15)
```

Expose LoRA controls conservatively:

- adapter source or local path;
- adapter name;
- adapter weight;
- optional trigger phrase guidance in UI or workflow metadata.

Do not silently bake a trigger phrase into user prompts. Keep it visible because
LoRA prompt triggers are adapter-specific.

## 13. Outputs

With `return_dict=True`, the pipeline returns:

```python
HunyuanVideoPipelineOutput(frames=video)
```

`frames` may be:

- a nested Python list of length `batch_size`, where each item is a list of
  `num_frames` PIL images;
- a NumPy array;
- a Torch tensor shaped like
  `(batch_size, num_frames, channels, height, width)`.

Normal usage reads the first generated video:

```python
result = pipe(prompt=prompt, num_frames=61, num_inference_steps=30)
frames = result.frames[0]
export_to_video(frames, "output.mp4", fps=15)
```

`output_type="latent"` returns latents instead of decoded frames according to
the source. Use this only for advanced workflows that know how to consume or
decode HunyuanVideo latents.

Tuple gotcha: the generated docs prose says `return_dict=False` returns a tuple
with generated images and NSFW booleans. The linked `v0.38.0` source returns
only `(video,)`, and `HunyuanVideoPipelineOutput` has only `frames`. Prefer
`return_dict=True` in product code.

## 14. Validation And Error Handling

Source-level validation to mirror in a workflow API:

- Reject `height` or `width` values not divisible by 16 before starting a job.
- Require either `prompt` or `prompt_embeds`, but not both.
- Reject `prompt_2` together with `prompt_embeds`.
- Require `prompt` and `prompt_2`, if provided, to be strings or string lists.
- Require `prompt_template` to be a dictionary with a `template` key.
- Require generator-list length to match effective batch size when using a list
  of `torch.Generator` instances.
- Limit callback tensor input names to `latents` and `prompt_embeds`.

Runtime failure modes to surface clearly:

- Out of memory during transformer load or denoising. Suggest int4
  quantization, CPU offload, smaller resolution, fewer frames, and VAE tiling.
- Out of memory during VAE decode. Suggest `pipe.vae.enable_tiling()`, smaller
  resolution, or `output_type="latent"` for advanced debugging.
- Slow first run with `torch.compile`. Tell users compile has a warmup cost.
- Long prompt truncation in CLIP. The pipeline logs which text was removed when
  the CLIP 77-token limit truncates input.
- Model layout mismatch. Use `hunyuanvideo-community/HunyuanVideo` unless a
  target repo is verified as Diffusers-compatible.

## 15. Recommended Workflow Defaults

Suggested SynthaEngine-style defaults for a first implementation:

| Setting | Default |
| --- | --- |
| `model_id` | `hunyuanvideo-community/HunyuanVideo` |
| `height` | `320` |
| `width` | `512` |
| `num_frames` | `61` |
| `num_inference_steps` | `30` |
| `guidance_scale` | `6.0` |
| `true_cfg_scale` | `1.0` |
| `num_videos_per_prompt` | `1` |
| `torch_dtype` | transformer BF16, VAE/text encoders FP16 where manually loaded; or BF16 for the quantized docs example |
| memory mode | int4 transformer quantization, `enable_model_cpu_offload()`, `vae.enable_tiling()` |
| export fps | `15` |

Recommended user-facing advanced controls:

- resolution, constrained to multiples of 16;
- frame count, constrained or hinted as `4 * k + 1`;
- denoising steps;
- embedded `guidance_scale`;
- optional `true_cfg_scale` and negative prompt;
- scheduler shift preset for low, medium, and high resolution;
- seed;
- LoRA adapter path/name/weight;
- output format and fps.

Avoid exposing by default:

- arbitrary `attention_kwargs`;
- arbitrary `prompt_template`;
- `prompt_embeds` and `latents`;
- `callback_on_step_end`;
- `clip_skip`, unless the installed version supports it.

## 16. Gotchas

- This guide is for `HunyuanVideoPipeline`, not
  `HunyuanVideo15Pipeline`. HunyuanVideo1.5 has different components and a
  separate docs page.
- The official docs examples use `hunyuanvideo-community/HunyuanVideo`, not the
  original Tencent repo, because of Diffusers weight layout compatibility.
- The docs recommend `num_frames = 4 * k + 1`; do not default to arbitrary
  values in a UI.
- `height` and `width` must be divisible by 16.
- Negative prompts are ignored unless `true_cfg_scale > 1` and negative
  conditioning is present.
- `guidance_scale` and `true_cfg_scale` are different controls. The first is
  embedded guidance; the second is true classifier-free guidance.
- `prompt_2` is documented, but the linked `v0.38.0` source appears to pass
  `prompt` into the CLIP helper. Test before depending on a separate second
  prompt.
- The docs prose mentions `clip_skip`, but the linked source signature does not
  accept it.
- Pipeline-level VAE tiling/slicing helpers are documented, but source
  deprecates them in favor of calling methods on `pipe.vae`.
- `return_dict=False` docs prose appears generic. The source returns a
  single-item tuple `(video,)`.
- The memory example still needs about 14 GB VRAM according to the docs, so
  "quantized" does not mean small.
- `torch.compile` may make later calls faster but adds a large first-call
  compile cost and can be awkward for short-lived job workers.

## 17. Source Links

- HunyuanVideo pipeline docs:
  https://huggingface.co/docs/diffusers/api/pipelines/hunyuan_video
- HunyuanVideo docs source:
  https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/hunyuan_video.md
- `HunyuanVideoPipeline` source:
  https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video.py
- `HunyuanVideoPipelineOutput` source:
  https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/hunyuan_video/pipeline_output.py
- `HunyuanVideoTransformer3DModel` docs:
  https://huggingface.co/docs/diffusers/api/models/hunyuan_video_transformer_3d
- `AutoencoderKLHunyuanVideo` docs:
  https://huggingface.co/docs/diffusers/api/models/autoencoder_kl_hunyuan_video
- Diffusers memory guide:
  https://huggingface.co/docs/diffusers/optimization/memory
- Diffusers quantization overview:
  https://huggingface.co/docs/diffusers/quantization/overview
- Diffusers LoRA inference guide:
  https://huggingface.co/docs/diffusers/tutorials/using_peft_for_inference
- Diffusers-compatible community model:
  https://huggingface.co/hunyuanvideo-community/HunyuanVideo
- Original Tencent model:
  https://huggingface.co/tencent/HunyuanVideo
