# Mochi Diffusers Implementation Guide

Last checked: 2026-06-17 against the Hugging Face Diffusers Mochi API page,
the linked `v0.38.0` Diffusers source, and the official Genmo Mochi model card.

Research target:
https://huggingface.co/docs/diffusers/api/pipelines/mochi

Primary Diffusers classes:

- `MochiPipeline`
- `MochiTransformer3DModel`
- `AutoencoderKLMochi`
- `MochiPipelineOutput`

Mochi 1 Preview is Genmo's text-to-video model focused on prompt adherence and
motion quality. The Diffusers integration exposes it through `MochiPipeline`.
The model is large: the docs describe a 10B-parameter asymmetric diffusion
transformer, a single T5-XXL prompt encoder, and a Mochi VAE that compresses
video by `8x8` spatially and `6x` temporally into a 12-channel latent space.
Treat Mochi as a high-VRAM, 480p-oriented pipeline rather than a lightweight
interactive model.

## 1. Executive Summary

Use `MochiPipeline` for prompt-only text-to-video generation with the
Diffusers-format checkpoint `genmo/mochi-1-preview`.

Practical integration answer:

- Start with `variant="bf16"` and `torch_dtype=torch.bfloat16` unless you have
  enough VRAM for the full-precision path.
- Enable model CPU offload and VAE tiling for local workstation use.
- Generate at the documented default resolution, `height=480` and `width=848`,
  before experimenting with other sizes.
- Use `num_frames=85` for a common short clip at 30 fps, or lower the frame
  count aggressively while validating memory behavior.
- Use `num_inference_steps=28` and `guidance_scale=3.5` for faster tests; use
  higher step counts, such as `50` or `64`, when chasing quality.
- Read output from `.frames[0]` and export the returned frame sequence with
  `diffusers.utils.export_to_video(..., fps=30)`.

The official docs say the full-precision example needs at least 42 GB VRAM.
The `bf16` variant example lowers that to roughly 22 GB VRAM with a slight
quality drop. The original Genmo-style precision recipe can need much more
memory, especially when decoding long clips in full precision.

## 2. Official Entry Points

- Pipeline docs: <https://huggingface.co/docs/diffusers/api/pipelines/mochi>
- Docs source: <https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/mochi.md>
- Pipeline source: <https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/mochi/pipeline_mochi.py>
- Output source: <https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/mochi/pipeline_output.py>
- Primary model card: <https://huggingface.co/genmo/mochi-1-preview>
- Original Genmo implementation: <https://github.com/genmoai/models>

## 3. Checkpoints And Model IDs

| Model or asset | Use | Notes |
| --- | --- | --- |
| `genmo/mochi-1-preview` | Main Diffusers checkpoint | Load with `MochiPipeline.from_pretrained(...)`. The docs describe this as a research preview of the model weights. |
| `genmo/mochi-1-preview`, `variant="bf16"` | Lower-memory Diffusers loading | Official example uses `torch_dtype=torch.bfloat16` and says this path needs about 22 GB VRAM with a slight quality drop. |
| `genmo/mochi-1-preview`, full precision | Highest-quality documented Diffusers path | Official example says it needs at least 42 GB VRAM and wraps inference in CUDA bfloat16 autocast. |
| `Comfy-Org/mochi_preview_repackaged` single-file transformer assets | Original/repackaged transformer loading | The Mochi docs mention single-file loading for the transformer and call out that FP8 scaled Mochi single-file checkpoints are not currently supported by Diffusers. |

The model card tags the checkpoint as text-to-video, Diffusers, Safetensors,
MochiPipeline, video, English, and Apache-2.0 licensed. Its model-card examples
also note the initial preview is optimized for 480p photorealistic video and
may show distortions in edge cases with extreme motion.

## 4. Installation

Use a recent Diffusers build that includes `MochiPipeline`.

```powershell
.venv\Scripts\python.exe -m pip install -U diffusers transformers accelerate torch imageio imageio-ffmpeg
```

For bitsandbytes quantization examples:

```powershell
.venv\Scripts\python.exe -m pip install -U bitsandbytes
```

For newest behavior before a released package catches up:

```powershell
.venv\Scripts\python.exe -m pip install -U git+https://github.com/huggingface/diffusers
```

## 5. Pipeline And Components

`MochiPipeline` is a `DiffusionPipeline` subclass with Mochi LoRA loader support
through `Mochi1LoraLoaderMixin`. Its constructor components are:

| Component | Class | Role |
| --- | --- | --- |
| `transformer` | `MochiTransformer3DModel` | Conditional 3D transformer that denoises video latents. |
| `scheduler` | `FlowMatchEulerDiscreteScheduler` | Denoising scheduler used with the transformer. |
| `vae` | `AutoencoderKLMochi` | Encodes/decodes video between pixel frames and latent video tensors. |
| `text_encoder` | `transformers.T5EncoderModel` | Prompt encoder; the docs specify the `google/t5-v1_1-xxl` variant. |
| `tokenizer` | `transformers.T5TokenizerFast` | Tokenizes prompts for T5. |
| `force_zeros_for_empty_prompt` | `bool` | Optional behavior matching the original implementation's empty-prompt handling. Default is `False`. |

Source-level details that matter for implementation:

- The pipeline offload order is `text_encoder -> transformer -> vae`.
- Default dimensions are `480x848`.
- Height and width must be divisible by 8.
- The VAE spatial scale factor is 8.
- The VAE temporal scale factor is 6.
- Latent frame count is computed as `(num_frames - 1) // 6 + 1`.
- Callback tensor inputs are limited to `latents`, `prompt_embeds`, and
  `negative_prompt_embeds`.
- The rendered API page includes a stray tokenizer reference to
  `CLIPTokenizer`; the constructor and source use `T5TokenizerFast`.

## 6. Minimal Text-To-Video Example

This is the recommended starting point for a local integration because it uses
the lower-precision variant and memory-saving hooks.

```python
import torch
from diffusers import MochiPipeline
from diffusers.utils import export_to_video

model_id = "genmo/mochi-1-preview"

pipe = MochiPipeline.from_pretrained(
    model_id,
    variant="bf16",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

# The Diffusers docs use pipe.enable_vae_tiling(). In newer source this pipeline
# helper is deprecated in favor of calling the VAE method directly.
if hasattr(pipe.vae, "enable_tiling"):
    pipe.vae.enable_tiling()
else:
    pipe.enable_vae_tiling()

prompt = (
    "A cinematic close-up of a glass hummingbird hovering above a silver "
    "flower, rain droplets suspended in slow motion, photorealistic, soft "
    "studio lighting, shallow depth of field."
)

generator = torch.Generator(device="cuda").manual_seed(1234)

frames = pipe(
    prompt=prompt,
    negative_prompt="",
    height=480,
    width=848,
    num_frames=85,
    num_inference_steps=28,
    guidance_scale=3.5,
    generator=generator,
    max_sequence_length=256,
).frames[0]

export_to_video(frames, "mochi.mp4", fps=30)
```

Notes:

- If `enable_model_cpu_offload()` is used, do not also call `pipe.to("cuda")`
  as the normal loading path. Let Accelerate move modules as needed.
- Use `torch.Generator(device="cuda")` when the denoising tensors live on CUDA.
- If the host cannot fit 85 frames, reduce `num_frames` first, then reduce
  resolution or step count.

## 7. Full-Precision Example

The official page presents the full-precision checkpoint as the highest-quality
path and says it needs at least 42 GB VRAM.

```python
import torch
from diffusers import MochiPipeline
from diffusers.utils import export_to_video

pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview")
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

prompt = (
    "Close-up of a chameleon's eye, with its scaly skin changing color. "
    "Ultra high resolution 4k."
)

with torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
    frames = pipe(prompt, num_frames=85).frames[0]

export_to_video(frames, "mochi.mp4", fps=30)
```

In production code, prefer the `pipe.vae.enable_tiling()` form when your
installed Diffusers version supports it.

## 8. Key `__call__` Parameters

Current `v0.38.0` source signature:

```python
pipe(
    prompt=None,
    negative_prompt=None,
    height=None,
    width=None,
    num_frames=19,
    num_inference_steps=64,
    timesteps=None,
    guidance_scale=4.5,
    num_videos_per_prompt=1,
    generator=None,
    latents=None,
    prompt_embeds=None,
    prompt_attention_mask=None,
    negative_prompt_embeds=None,
    negative_prompt_attention_mask=None,
    output_type="pil",
    return_dict=True,
    attention_kwargs=None,
    callback_on_step_end=None,
    callback_on_step_end_tensor_inputs=["latents"],
    max_sequence_length=256,
)
```

| Parameter | Practical guidance |
| --- | --- |
| `prompt` | String or list of strings. Required unless you pass `prompt_embeds`. Long prompts are tokenized by T5 and truncated at `max_sequence_length`. |
| `negative_prompt` | String or list matching prompt batch size. Defaults to an empty string when classifier-free guidance is active. Ignored when `guidance_scale <= 1`. |
| `height`, `width` | Defaults are `480` and `848`. Both must be divisible by 8. Start with the default 480p landscape size. |
| `num_frames` | Defaults to `19`; official examples commonly use `85` and `163`. Memory and decode cost grow quickly with frame count. |
| `num_inference_steps` | Source default is `64`. Docs examples use `28`, `50`, and `64`; more steps are slower and usually higher quality. |
| `guidance_scale` | Defaults to `4.5`. Official quick examples use `3.5`; Genmo-parity examples use `4.5`. Values above 1 enable classifier-free guidance. |
| `timesteps` | Optional custom scheduler timesteps in descending order, if the scheduler supports them. Test carefully because Mochi also builds a Genmo-style sigma schedule internally. |
| `num_videos_per_prompt` | Multiplies batch output and memory. Keep at `1` for server use unless batching is explicitly supported. |
| `generator` | Use for reproducibility. A list of generators must match the effective batch size. |
| `latents` | Optional pre-generated noisy latents. Use for controlled reruns, prompt sweeps, or external latent workflows. |
| `prompt_embeds`, `negative_prompt_embeds` | Optional precomputed embeddings. If passed, their attention masks are also required. Positive and negative embed shapes must match. |
| `output_type` | Defaults to `"pil"`. Source also supports `"latent"` to skip VAE decoding and return denoised latents. `VideoProcessor` handles tensor-to-PIL/NumPy conversion for decoded outputs. |
| `return_dict` | `True` returns `MochiPipelineOutput`; `False` returns a one-element tuple containing the video/latent output. |
| `attention_kwargs` | Passed through to the attention processor. Use only when integrating attention processors that require it. |
| `callback_on_step_end` | Called after each denoising step. May receive tensors listed in `callback_on_step_end_tensor_inputs`. |
| `max_sequence_length` | Defaults to `256`; longer prompt text is truncated. |

One small docs mismatch: the rendered API text says `num_inference_steps`
defaults to 50 in one parameter description, but the shown signature and
linked source use `64`.

## 9. Frame And Resolution Guidance

Mochi's documented sweet spot is 480p, especially `480x848`.

Use these starting presets:

| Goal | Height | Width | Frames | Steps | FPS export | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Smoke test | 480 | 848 | 19 | 28 | 30 | Fastest useful validation of loading and output plumbing. |
| Short clip | 480 | 848 | 85 | 28-50 | 30 | Matches the common Diffusers examples and gives about 2.8 seconds at 30 fps. |
| Longer clip | 480 | 848 | 163 | 64 | 30 | Used in the Genmo-parity example; much heavier, especially during decode. |

Implementation details:

- Spatial latent size is `height / 8` by `width / 8`.
- Temporal latent size is `(num_frames - 1) // 6 + 1`.
- Increasing `num_frames` can increase memory in steps because of the temporal
  compression formula.
- `export_to_video` defaults to macroblock alignment behavior. The documented
  `480x848` size is divisible by 16, so it exports cleanly with common codecs.
- The model card describes the preview as initially generating 480p video and
  optimized for photorealistic styles. Animated or highly stylized prompts may
  be less reliable.

## 10. Outputs And Export

With `return_dict=True`, `MochiPipeline` returns `MochiPipelineOutput`.

```python
output = pipe(prompt, output_type="pil")
videos = output.frames
first_video = videos[0]
export_to_video(first_video, "output.mp4", fps=30)
```

`MochiPipelineOutput.frames` can be:

- A nested list shaped like `batch_size x num_frames` containing PIL frames.
- A NumPy array.
- A Torch tensor.

For most app integrations, keep the default `output_type="pil"` and export the
first generated video as `output.frames[0]`. If `num_videos_per_prompt > 1` or
you pass a list of prompts, iterate over `output.frames`.

```python
for i, frame_sequence in enumerate(output.frames):
    export_to_video(frame_sequence, f"mochi_{i:02d}.mp4", fps=30)
```

For latent workflows:

```python
latents = pipe(
    prompt=prompt,
    height=480,
    width=848,
    num_frames=85,
    output_type="latent",
).frames
```

When `output_type="latent"`, the pipeline skips VAE decoding and returns the
latent tensor in the `frames` field. At `480x848` and `85` requested frames, the
latent spatial size is `60x106` and the temporal latent length is `15`.

When `return_dict=False`:

```python
(frames,) = pipe(prompt, return_dict=False)
```

## 11. Memory, Performance, And Offloading

Mochi is memory-heavy because it combines a large T5-XXL encoder, a 10B video
transformer, and VAE video decode.

Documented options:

| Technique | How | Trade-off |
| --- | --- | --- |
| BF16 checkpoint variant | `MochiPipeline.from_pretrained(..., variant="bf16", torch_dtype=torch.bfloat16)` | About 22 GB VRAM in the docs, with a slight quality drop. |
| Full checkpoint with autocast | Load without `variant`, run generation under CUDA bfloat16 autocast | Higher-quality documented path, at least 42 GB VRAM. |
| Model CPU offload | `pipe.enable_model_cpu_offload()` | Reduces peak VRAM by moving components through CPU/offload hooks. Slower than keeping all modules resident. |
| VAE tiling | `pipe.enable_vae_tiling()` or `pipe.vae.enable_tiling()` | Splits VAE work into tiles to save a large amount of memory during decode/encode. |
| VAE slicing | `pipe.enable_vae_slicing()` or `pipe.vae.enable_slicing()` | Slices VAE decoding. Useful for memory, usually slower. |
| Fewer frames | Lower `num_frames` | Largest quality-preserving lever for fitting small GPUs. Shorter clip. |
| Lower steps | Lower `num_inference_steps` | Faster and lower runtime memory pressure, but may reduce quality or motion coherence. |
| Multi-GPU transformer split | Load `MochiTransformer3DModel.from_pretrained(..., device_map="auto", max_memory={...})` and pass it into the pipeline | Helps distribute the large transformer. Requires Accelerate and multiple GPUs. |

The `v0.38.0` source deprecates pipeline-level
`enable_vae_tiling()`/`enable_vae_slicing()` wrappers for future removal and
points users to `pipe.vae.enable_tiling()` and `pipe.vae.enable_slicing()`.
The rendered Mochi docs still show the pipeline-level helpers, so support both
forms if you need compatibility across Diffusers versions.

## 12. Bitsandbytes 8-Bit Quantization

The official Mochi page demonstrates 8-bit bitsandbytes loading for both the
T5 text encoder and the Mochi transformer. Use the Transformers
`BitsAndBytesConfig` for `T5EncoderModel` and the Diffusers
`BitsAndBytesConfig` for `MochiTransformer3DModel`.

```python
import torch
from diffusers import (
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
    MochiPipeline,
    MochiTransformer3DModel,
)
from diffusers.utils import export_to_video
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import T5EncoderModel

model_id = "genmo/mochi-1-preview"

text_encoder = T5EncoderModel.from_pretrained(
    model_id,
    subfolder="text_encoder",
    quantization_config=TransformersBitsAndBytesConfig(load_in_8bit=True),
    torch_dtype=torch.float16,
)

transformer = MochiTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=DiffusersBitsAndBytesConfig(load_in_8bit=True),
    torch_dtype=torch.float16,
)

pipe = MochiPipeline.from_pretrained(
    model_id,
    text_encoder=text_encoder,
    transformer=transformer,
    torch_dtype=torch.float16,
    device_map="balanced",
)

frames = pipe(
    "Close-up of a cat's eye, with a galaxy reflected in the iris. "
    "Ultra high resolution 4k.",
    num_inference_steps=28,
    guidance_scale=3.5,
).frames[0]

export_to_video(frames, "cat.mp4", fps=30)
```

Quantization reduces weight memory, but it can affect video quality. Validate
motion, prompt adherence, and temporal consistency before making a quantized
path the default.

## 13. Multiple GPUs

The Mochi docs show splitting the transformer across two 24 GB GPUs with
`device_map="auto"` and `max_memory`.

```python
import torch
from diffusers import MochiPipeline, MochiTransformer3DModel
from diffusers.utils import export_to_video

model_id = "genmo/mochi-1-preview"

transformer = MochiTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    device_map="auto",
    max_memory={0: "24GB", 1: "24GB"},
)

pipe = MochiPipeline.from_pretrained(model_id, transformer=transformer)
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

with torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False):
    frames = pipe(
        prompt=(
            "Close-up of a chameleon's eye, with its scaly skin changing color. "
            "Ultra high resolution 4k."
        ),
        negative_prompt="",
        height=480,
        width=848,
        num_frames=85,
        num_inference_steps=50,
        guidance_scale=4.5,
        num_videos_per_prompt=1,
        generator=torch.Generator(device="cuda").manual_seed(0),
        max_sequence_length=256,
        output_type="pil",
    ).frames[0]

export_to_video(frames, "output.mp4", fps=30)
```

For a service, expose this as an advanced deployment preset rather than a
per-request option. Device maps should be fixed at startup and validated once.

## 14. Original Genmo-Style Precision Recipe

The Diffusers docs explain that the original Genmo implementation uses
different precision by stage:

- Text encoder: `torch.float32`
- VAE: `torch.float32`
- DiT/transformer: `torch.bfloat16`
- PyTorch attention kernel: `EFFICIENT_ATTENTION`

The docs also warn that Diffusers pipelines do not generally support setting
different dtypes for different stages through one simple `from_pretrained`
argument. Their parity example works around this by encoding prompts outside
the autocast context, running denoising with bfloat16 autocast and efficient
attention, returning `output_type="latent"`, and manually decoding latents.

Important warning: when `force_zeros_for_empty_prompt=True`, avoid wrapping
the T5 prompt encoding step in autocast. The docs say zeroing empty prompts
inside full-pipeline autocast can lead to numerical overflows with the T5 text
encoder. Encode prompts first in full precision, then run denoising.

The same parity example says full-precision latent decoding is very memory
intensive and may need at least 70 GB VRAM for 163 frames. To fit smaller
systems, reduce `num_frames` or decode in `torch.bfloat16`.

## 15. Prompt Encoding And Embedding Reuse

Use `encode_prompt` when you need to:

- Cache prompt embeddings across multiple runs.
- Keep T5 prompt encoding outside an autocast block.
- Manually edit prompt embeddings.
- Share positive and negative embeddings across reproducibility tests.

```python
with torch.no_grad():
    (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    ) = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt="",
        max_sequence_length=256,
    )

frames = pipe(
    prompt_embeds=prompt_embeds,
    prompt_attention_mask=prompt_attention_mask,
    negative_prompt_embeds=negative_prompt_embeds,
    negative_prompt_attention_mask=negative_prompt_attention_mask,
    height=480,
    width=848,
    num_frames=85,
    num_inference_steps=50,
    guidance_scale=4.5,
).frames[0]
```

If you pass `prompt_embeds`, you must also pass `prompt_attention_mask`. If you
pass `negative_prompt_embeds`, you must also pass
`negative_prompt_attention_mask`. Positive and negative embeddings and masks
must have matching shapes.

## 16. Integration Gotchas

- Mochi is text-to-video only in Diffusers. There is no separate image-to-video
  `MochiPipeline` on the official API page.
- The model is a research preview. Avoid treating checkpoint behavior, quality,
  and optimal settings as frozen product contracts.
- Height and width must be divisible by 8. Validate this before queueing a job.
- The default prompt token budget is 256 T5 tokens. Long prompts are truncated;
  surface truncation warnings in logs if prompt fidelity matters.
- Default decoded output is PIL frame lists. Do not look for `.images`; use
  `.frames`.
- `num_videos_per_prompt` multiplies output count and memory.
- `guidance_scale <= 1` disables classifier-free guidance and ignores negative
  prompt inputs.
- `force_zeros_for_empty_prompt=True` is a parity switch, not a casual default.
  Use it only if you understand the autocast warning.
- The docs' single-file section says FP8 scaled Mochi single-file checkpoints
  are not supported.
- The official model card says Mochi is optimized for photorealistic styles and
  may struggle with animated content.
- Extreme motion can produce warping or distortions, according to the model
  card.
- `export_to_video` accepts PIL or NumPy frame sequences and lets you control
  `fps`, `quality`, `bitrate`, and `macro_block_size`. Use explicit `fps=30`
  for the official Mochi examples.

## 17. Suggested Server Defaults

For a local workflow server, start conservatively:

```json
{
  "model_id": "genmo/mochi-1-preview",
  "variant": "bf16",
  "torch_dtype": "bfloat16",
  "height": 480,
  "width": 848,
  "num_frames": 85,
  "num_inference_steps": 28,
  "guidance_scale": 3.5,
  "num_videos_per_prompt": 1,
  "max_sequence_length": 256,
  "output_type": "pil",
  "export_fps": 30,
  "enable_model_cpu_offload": true,
  "enable_vae_tiling": true
}
```

Validation checklist:

- Reject dimensions not divisible by 8.
- Limit frame count by deployment profile.
- Log actual model ID, dtype, variant, frame count, size, step count, guidance,
  seed, and output path.
- Smoke test with 19 frames after startup before allowing long jobs.
- Confirm output export opens in a standard video player.
- Capture quantized and non-quantized quality samples separately.

## 18. Source Notes

The guide above is based on these official references:

- Diffusers Mochi API page:
  <https://huggingface.co/docs/diffusers/api/pipelines/mochi>
- Diffusers Mochi docs source:
  <https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/mochi.md>
- Diffusers `MochiPipeline` source:
  <https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/mochi/pipeline_mochi.py>
- Diffusers `MochiPipelineOutput` source:
  <https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/mochi/pipeline_output.py>
- Genmo Mochi model card:
  <https://huggingface.co/genmo/mochi-1-preview>
- Diffusers utilities docs for `export_to_video`:
  <https://huggingface.co/docs/diffusers/api/utilities>
- Diffusers video processor docs:
  <https://huggingface.co/docs/diffusers/api/video_processor>
- Diffusers memory optimization guide:
  <https://huggingface.co/docs/diffusers/optimization/memory>
- Diffusers quantization overview:
  <https://huggingface.co/docs/diffusers/quantization/overview>
