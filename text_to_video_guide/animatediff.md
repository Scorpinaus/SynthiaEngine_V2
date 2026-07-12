# AnimateDiff Implementation Guide

This guide summarizes the official Hugging Face Diffusers AnimateDiff API and source for implementing text-to-video, controlled text-to-video, SDXL text-to-video, and video-to-video flows. It is based on the Diffusers AnimateDiff docs for `v0.38.0` and the linked source files under `diffusers/pipelines/animatediff`.

Official entry points:

- Docs: <https://huggingface.co/docs/diffusers/api/pipelines/animatediff>
- Docs source: <https://github.com/huggingface/diffusers/blob/v0.38.0/docs/source/en/api/pipelines/animatediff.md>
- Pipeline package source: <https://github.com/huggingface/diffusers/tree/v0.38.0/src/diffusers/pipelines/animatediff>

## Mental Model

AnimateDiff turns a compatible image diffusion model into a video model by adding a `MotionAdapter` to the UNet. For SD 1.4/1.5 workflows, the adapter is injected into a Stable Diffusion UNet and adds temporal motion modules around the existing ResNet and attention blocks. The base image model controls visual style and subject quality; the motion adapter controls temporal coherence and movement.

The usual output is an `AnimateDiffPipelineOutput` whose `frames` field is indexed as `output.frames[batch_index]`. The frame sequence can then be saved with helpers such as `export_to_gif(frames, "animation.gif")` or `export_to_video(frames, "output.mp4", fps=16)`.

## Pipeline Overview

| Class | Main use | Base family | Extra inputs |
| --- | --- | --- | --- |
| `AnimateDiffPipeline` | Text-to-video from a prompt | SD 1.4/1.5 compatible models | Optional IP-Adapter image/embeds, LoRA, FreeInit, FreeNoise |
| `AnimateDiffControlNetPipeline` | Text-to-video with dense per-frame control | SD 1.4/1.5 compatible models | `ControlNetModel`, `conditioning_frames` |
| `AnimateDiffSparseControlNetPipeline` | Text-to-video with sparse keyframe controls | SD 1.4/1.5 compatible models | `SparseControlNetModel`, `conditioning_frames`, `controlnet_frame_indices` |
| `AnimateDiffSDXLPipeline` | Text-to-video with SDXL | SDXL | SDXL prompt pairs, pooled embeds, SDXL size conditioning |
| `AnimateDiffVideoToVideoPipeline` | Edit or restyle an input video | SD 1.4/1.5 compatible models | `video`, `strength`, optional custom `timesteps`/`sigmas` |
| `AnimateDiffVideoToVideoControlNetPipeline` | Video-to-video plus ControlNet structure guidance | SD 1.4/1.5 compatible models | `video`, `conditioning_frames`, ControlNet controls |
| `AnimateDiffPipelineOutput` | Shared output object | All AnimateDiff pipelines | `frames` |

## Shared Setup

Install the usual Diffusers stack, plus optional packages for video I/O and ControlNet preprocessing:

```shell
pip install diffusers transformers accelerate safetensors imageio
pip install controlnet_aux
```

Use `torch.float16` on CUDA for most examples. The docs examples generally use:

- `MotionAdapter.from_pretrained(...)` for Diffusers-format motion adapters.
- `MotionAdapter.from_single_file(...)` for original-format checkpoints in `diffusers>=0.30.0`.
- `DDIMScheduler` with `clip_sample=False`, `timestep_spacing="linspace"`, `beta_schedule="linear"`, and `steps_offset=1` for classic AnimateDiff examples.
- `LCMScheduler` plus LCM LoRA for AnimateLCM speedups.
- `enable_vae_slicing()`, `enable_vae_tiling()`, and/or `enable_model_cpu_offload()` for memory pressure.

### Motion Adapters

Common documented adapters:

- `guoyww/animatediff-motion-adapter-v1-5-2`: SD 1.5-era adapter used by the basic text-to-video and motion LoRA examples.
- `guoyww/animatediff-motion-adapter-v1-5-3`: SD 1.5-era adapter used by SparseCtrl examples.
- `guoyww/animatediff-motion-adapter-sdxl-beta`: SDXL beta adapter. The docs call SDXL AnimateDiff experimental because only a beta motion adapter checkpoint is available.
- `wangfuyun/AnimateLCM`: motion module for AnimateLCM, often paired with an LCM LoRA and `LCMScheduler`.

Adapters must match the base model family. SD 1.5 adapters are for SD 1.4/1.5-derived checkpoints, not SDXL. SDXL requires the SDXL beta motion adapter.

### Output Handling

```python
output = pipe(...)
frames = output.frames[0]
export_to_gif(frames, "animation.gif")
```

`output_type` defaults to `"pil"` for the documented APIs. The SD 1.5 pipelines document `"pil"`, `"np"`, and `"pt"`/`torch.Tensor` options depending on the class. SDXL docs list PIL images or NumPy arrays. If `return_dict=False`, the pipeline returns a tuple whose first element is the generated frame list.

## AnimateDiffPipeline

Source: <https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/animatediff/pipeline_animatediff.py>

Use this for prompt-only text-to-video with an SD 1.4/1.5-compatible base model and a `MotionAdapter`.

### Minimal Example

```python
import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2",
    torch_dtype=torch.float16,
)

pipe = AnimateDiffPipeline.from_pretrained(
    model_id,
    motion_adapter=adapter,
    torch_dtype=torch.float16,
)
pipe.scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

output = pipe(
    prompt="a corgi walking through a sunny park, cinematic, high quality",
    negative_prompt="bad quality, worse quality",
    num_frames=16,
    num_inference_steps=25,
    guidance_scale=7.5,
    generator=torch.Generator("cpu").manual_seed(42),
)
export_to_gif(output.frames[0], "animation.gif")
```

### Components

Constructor inputs include:

- `vae`: `AutoencoderKL` for latent/image conversion.
- `text_encoder` and `tokenizer`: CLIP text components.
- `unet`: `UNet2DConditionModel` or `UNetMotionModel`. If a 2D UNet is passed, the pipeline builds a motion UNet from it and the adapter.
- `motion_adapter`: required for adding temporal motion modules.
- `scheduler`: DDIM, PNDM, LMS, Euler, Euler ancestral, or DPMSolver multistep family scheduler.
- Optional `feature_extractor` and `image_encoder` for IP-Adapter support.

The class inherits loading helpers for textual inversion, LoRA weights, saving LoRA weights, IP-Adapter loading, FreeInit, FreeNoise, and `from_single_file`.

### Key `__call__` Parameters

- `prompt`: string, list of strings, or embeddings via `prompt_embeds`. FreeNoise also allows a prompt dictionary in the source path when enabled.
- `num_frames`: default `16`; at 8 fps this is about 2 seconds.
- `height`, `width`: default from `unet.config.sample_size * vae_scale_factor`; must be divisible by 8.
- `num_inference_steps`: default `50`; more steps usually improves quality but slows generation.
- `guidance_scale`: default `7.5`; classifier-free guidance is active above 1.
- `negative_prompt`: omitted if using `negative_prompt_embeds`.
- `num_videos_per_prompt`: batch multiplier.
- `eta`: DDIM-only stochasticity parameter.
- `generator`: use a CPU `torch.Generator` for reproducible examples.
- `latents`: optional pre-generated latents shaped `(batch, channels, frames, height, width)` in latent resolution.
- `ip_adapter_image` / `ip_adapter_image_embeds`: optional IP-Adapter conditioning.
- `cross_attention_kwargs`: passed to attention processors; often used for LoRA scale.
- `clip_skip`: use earlier CLIP hidden states.
- `callback_on_step_end` and `callback_on_step_end_tensor_inputs`: inspect or modify tensors during denoising. Valid tensor inputs are constrained by the pipeline.
- `decode_chunk_size`: default `16`; controls how many frames are VAE-decoded at once.

### Gotchas

- The official docs say AnimateDiff tends to work better with finetuned Stable Diffusion models than with the raw base model.
- If your scheduler supports sample clipping, set `clip_sample=False`; the docs warn clipping can hurt generated samples.
- AnimateDiff checkpoints are sensitive to scheduler beta schedule; the docs recommend `beta_schedule="linear"`.
- `height` and `width` must be divisible by 8.
- Do not pass both `prompt` and `prompt_embeds`, both `negative_prompt` and `negative_prompt_embeds`, or both `ip_adapter_image` and `ip_adapter_image_embeds`.

## AnimateDiffControlNetPipeline

Source: <https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/animatediff/pipeline_animatediff_controlnet.py>

Use this when each output frame should follow dense structure guidance, such as depth, pose, edge, or other per-frame ControlNet maps.

### Dense ControlNet Example

```python
import torch
from controlnet_aux.processor import ZoeDetector
from diffusers import (
    AnimateDiffControlNetPipeline,
    AutoencoderKL,
    ControlNetModel,
    LCMScheduler,
    MotionAdapter,
)
from diffusers.utils import export_to_gif, load_video

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth",
    torch_dtype=torch.float16,
)
motion_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM")
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)

pipe = AnimateDiffControlNetPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    motion_adapter=motion_adapter,
    controlnet=controlnet,
    vae=vae,
).to(device="cuda", dtype=torch.float16)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
pipe.load_lora_weights(
    "wangfuyun/AnimateLCM",
    weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
    adapter_name="lcm-lora",
)
pipe.set_adapters(["lcm-lora"], [0.8])

depth_detector = ZoeDetector.from_pretrained("lllyasviel/Annotators").to("cuda")
source_video = load_video("input.gif")
conditioning_frames = [depth_detector(frame) for frame in source_video]

frames = pipe(
    prompt="a panda playing guitar on a small boat, realistic, high quality",
    negative_prompt="bad quality, worst quality",
    num_frames=len(conditioning_frames),
    num_inference_steps=10,
    guidance_scale=2.0,
    conditioning_frames=conditioning_frames,
    controlnet_conditioning_scale=1.0,
    generator=torch.Generator().manual_seed(42),
).frames[0]
export_to_gif(frames, "animatediff_controlnet.gif", fps=8)
```

### Components and Controls

Additional constructor input:

- `controlnet`: `ControlNetModel`, list/tuple of `ControlNetModel`, or `MultiControlNetModel`. With multiple ControlNets, residuals are combined, and scale parameters can be lists.

Additional `__call__` inputs:

- `conditioning_frames`: ControlNet images/maps. For a single ControlNet, pass the frame sequence for that ControlNet. For multiple ControlNets, pass a list structured so each ControlNet receives the right batched images.
- `controlnet_conditioning_scale`: scalar or list of scalars. Multiplies ControlNet outputs before adding them to UNet residuals.
- `guess_mode`: lets the ControlNet infer content with weak or absent prompts; docs recommend `guidance_scale` between 3 and 5 for guess mode.
- `control_guidance_start` / `control_guidance_end`: scalar or list fractions controlling when ControlNet starts and stops applying over the denoising schedule.

### Gotchas

- ControlNet input frames must align with `num_frames`. Generate or resize preprocessed maps consistently.
- The docs example notes `controlnet_aux` is needed for common preprocessors such as Zoe depth and OpenPose.
- Original ControlNet checkpoint files can be loaded with `ControlNetModel.from_single_file(...)`; Diffusers-format ControlNets can be loaded with `from_pretrained(...)`.
- This is a text-to-video pipeline guided by control images. If you also need to preserve or edit an original source video, use `AnimateDiffVideoToVideoControlNetPipeline`.

## AnimateDiffSparseControlNetPipeline

Source: <https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/animatediff/pipeline_animatediff_sparsectrl.py>

Use this for SparseCtrl, where only one or a few frames provide structure guidance and the model fills temporal gaps. The official docs list SparseCtrl Scribble and SparseCtrl RGB checkpoints.

### SparseCtrl Scribble Example

```python
import torch
from diffusers import AnimateDiffSparseControlNetPipeline
from diffusers.models import AutoencoderKL, MotionAdapter, SparseControlNetModel
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import export_to_gif, load_image

model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
motion_adapter_id = "guoyww/animatediff-motion-adapter-v1-5-3"
controlnet_id = "guoyww/animatediff-sparsectrl-scribble"
lora_adapter_id = "guoyww/animatediff-motion-lora-v1-5-3"

motion_adapter = MotionAdapter.from_pretrained(motion_adapter_id, torch_dtype=torch.float16).to("cuda")
controlnet = SparseControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16).to("cuda")
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")
scheduler = DPMSolverMultistepScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    beta_schedule="linear",
    algorithm_type="dpmsolver++",
    use_karras_sigmas=True,
)

pipe = AnimateDiffSparseControlNetPipeline.from_pretrained(
    model_id,
    motion_adapter=motion_adapter,
    controlnet=controlnet,
    vae=vae,
    scheduler=scheduler,
    torch_dtype=torch.float16,
).to("cuda")
pipe.load_lora_weights(lora_adapter_id, adapter_name="motion_lora")
pipe.fuse_lora(lora_scale=1.0)

conditioning_frames = [
    load_image("scribble_000.png"),
    load_image("scribble_008.png"),
    load_image("scribble_015.png"),
]
frames = pipe(
    prompt="an aerial view of a cyberpunk city, night time, neon lights, high quality",
    negative_prompt="low quality, worst quality, letterboxed",
    num_inference_steps=25,
    conditioning_frames=conditioning_frames,
    controlnet_frame_indices=[0, 8, 15],
    controlnet_conditioning_scale=1.0,
    generator=torch.Generator().manual_seed(1337),
).frames[0]
export_to_gif(frames, "sparsectrl.gif")
```

### SparseCtrl RGB Example

For RGB keyframe guidance, use `guoyww/animatediff-sparsectrl-rgb` and pass one or more RGB reference images:

```python
controlnet = SparseControlNetModel.from_pretrained(
    "guoyww/animatediff-sparsectrl-rgb",
    torch_dtype=torch.float16,
).to("cuda")

image = load_image("first_frame_reference.png")
frames = pipe(
    prompt="closeup face photo of man in black clothes, night city street, fireworks",
    negative_prompt="low quality, worst quality",
    num_inference_steps=25,
    conditioning_frames=image,
    controlnet_frame_indices=[0],
    controlnet_conditioning_scale=1.0,
).frames[0]
```

### Key Parameters

- `conditioning_frames`: one image or a list of sparse control frames.
- `controlnet_frame_indices`: frame indices where those controls apply. This must have the same length as `conditioning_frames`.
- `controlnet_conditioning_scale`: control strength.
- `guess_mode`: supported in the source signature.
- Most prompt, latent, IP-Adapter, LoRA, `clip_skip`, and callback parameters mirror `AnimateDiffPipeline`.

### Gotchas

- SparseCtrl is for sparse structure, not dense per-frame conditioning. Use regular ControlNet when every frame has a control map.
- `controlnet_frame_indices` is required for intentional placement. A single RGB keyframe at `[0]` guides the beginning; multiple indices can guide interpolation or storyboarding.
- The source creates an internal control mask for the specified frame indices; mismatched indices and frames will fail validation or produce unintended conditioning.

## AnimateDiffSDXLPipeline

Source: <https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/animatediff/pipeline_animatediff_sdxl.py>

Use this for SDXL text-to-video. The docs mark it as experimental because only a beta SDXL motion adapter checkpoint is available.

### SDXL Example

```python
import torch
from diffusers import AnimateDiffSDXLPipeline, DDIMScheduler
from diffusers.models import MotionAdapter
from diffusers.utils import export_to_gif

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-sdxl-beta",
    torch_dtype=torch.float16,
)
scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)

pipe = AnimateDiffSDXLPipeline.from_pretrained(
    model_id,
    motion_adapter=adapter,
    scheduler=scheduler,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

output = pipe(
    prompt="a panda surfing in the ocean, realistic, high quality",
    negative_prompt="low quality, worst quality",
    num_inference_steps=20,
    guidance_scale=8,
    width=1024,
    height=1024,
    num_frames=16,
)
export_to_gif(output.frames[0], "sdxl_animation.gif")
```

### SDXL-Specific Components

Constructor inputs include SDXL's two text encoders and tokenizers:

- `text_encoder`, `tokenizer`: CLIP ViT-L text path.
- `text_encoder_2`, `tokenizer_2`: CLIP bigG text/projection path.
- `force_zeros_for_empty_prompt`: default `True`; matches SDXL negative prompt behavior.
- Optional `image_encoder` and `feature_extractor` for IP-Adapter.

The class supports textual inversion, `from_single_file`, LoRA loading/saving, IP-Adapter loading, and SDXL-style guidance scale embeddings.

### Key `__call__` Parameters

SDXL includes the shared text-to-video parameters plus:

- `prompt_2` / `negative_prompt_2`: text sent to the second tokenizer/text encoder. If omitted, `prompt` / `negative_prompt` are reused.
- `timesteps` / `sigmas`: custom scheduler steps for schedulers that support them.
- `denoising_end`: fraction of denoising to run before stopping, useful for multi-pipeline mixture-of-denoisers setups.
- `guidance_scale`: default `5.0` in the SDXL pipeline.
- `pooled_prompt_embeds` / `negative_pooled_prompt_embeds`: SDXL pooled embeddings for advanced prompt weighting or reuse.
- `guidance_rescale`: rescale factor for classifier-free guidance to reduce overexposure with zero-terminal-SNR schedules.
- `original_size`, `target_size`, `crops_coords_top_left` and their negative counterparts: SDXL micro-conditioning controls.

### Gotchas

- Use an SDXL motion adapter with SDXL. SD 1.5 motion adapters are not interchangeable.
- The docs recommend 1024x1024 for SDXL; resolutions below 512 generally work poorly unless the checkpoint is trained for them.
- SDXL video generation is much heavier than SD 1.5. Enable VAE slicing and tiling, reduce `num_frames`, or lower resolution if VRAM is tight.

## AnimateDiffVideoToVideoPipeline

Source: <https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/animatediff/pipeline_animatediff_video2video.py>

Use this to edit, restyle, or generate a visually similar video from an input frame sequence. The source video is encoded to latents, noise is added according to `strength`, and AnimateDiff denoises toward the prompt.

### Video-to-Video Example

```python
import torch
from diffusers import AnimateDiffVideoToVideoPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif, load_video

model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2",
    torch_dtype=torch.float16,
)
pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(
    model_id,
    motion_adapter=adapter,
    torch_dtype=torch.float16,
)
pipe.scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

video = load_video("input.gif")
output = pipe(
    video=video,
    prompt="panda playing a guitar, on a boat, in the ocean, high quality",
    negative_prompt="bad quality, worse quality",
    guidance_scale=7.5,
    num_inference_steps=25,
    strength=0.5,
    generator=torch.Generator("cpu").manual_seed(42),
)
export_to_gif(output.frames[0], "vid2vid.gif")
```

### Key Parameters

- `video`: list of input frames/images.
- `strength`: default `0.8`; higher values depart more from the input video, lower values preserve it more.
- `enforce_inference_steps`: source signature option for controlling the relationship between `strength` and the scheduler step count.
- `timesteps` / `sigmas`: optional custom scheduler steps for schedulers that support them.
- `latents`: can replace `video`, but source validation rejects passing both.
- `decode_chunk_size`: controls VAE encode/decode chunking over frames.

All prompt, negative prompt, guidance, generator, IP-Adapter, LoRA, callback, and output controls mirror `AnimateDiffPipeline`.

### Gotchas

- `video` must be a list of images/frames. Use `diffusers.utils.load_video` or a loader that returns PIL frames.
- `strength` maps to how many denoising timesteps are used from the schedule. If it is too low, the prompt may barely affect the result; if too high, the input motion/composition may be lost.
- If input dimensions do not match the target `height` and `width`, the video processor resizes/preprocesses frames. Keep aspect ratio and frame size intentional.
- Do not pass both `video` and `latents`.

## AnimateDiffVideoToVideoControlNetPipeline

Source: <https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/animatediff/pipeline_animatediff_video2video_controlnet.py>

Use this when you need both the source video and a sequence of control images. The docs example uses OpenPose maps from a dance video so the output follows the source motion and composition while changing the subject.

### Video-to-Video ControlNet Example

```python
import torch
from controlnet_aux.processor import OpenposeDetector
from diffusers import (
    AnimateDiffVideoToVideoControlNetPipeline,
    AutoencoderKL,
    ControlNetModel,
    LCMScheduler,
    MotionAdapter,
)
from diffusers.utils import export_to_gif, load_video
from tqdm.auto import tqdm

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose",
    torch_dtype=torch.float16,
)
motion_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM")
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)

pipe = AnimateDiffVideoToVideoControlNetPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    motion_adapter=motion_adapter,
    controlnet=controlnet,
    vae=vae,
).to(device="cuda", dtype=torch.float16)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
pipe.load_lora_weights(
    "wangfuyun/AnimateLCM",
    weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
    adapter_name="lcm-lora",
)
pipe.set_adapters(["lcm-lora"], [0.8])

video = [frame.convert("RGB") for frame in load_video("dance.gif")]
open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators").to("cuda")
conditioning_frames = [open_pose(frame) for frame in tqdm(video)]

frames = pipe(
    video=video,
    prompt="astronaut in space, dancing",
    negative_prompt="bad quality, worst quality, jpeg artifacts, ugly",
    num_inference_steps=10,
    guidance_scale=2.0,
    strength=0.8,
    controlnet_conditioning_scale=0.75,
    conditioning_frames=conditioning_frames,
    generator=torch.Generator().manual_seed(42),
).frames[0]
export_to_gif(frames, "vid2vid_controlnet.gif", fps=8)
```

### Key Parameters

This pipeline combines the video-to-video controls and ControlNet controls:

- `video`: source frames.
- `strength`: source-video edit strength.
- `conditioning_frames`: per-frame ControlNet maps.
- `controlnet_conditioning_scale`: control strength.
- `guess_mode`, `control_guidance_start`, `control_guidance_end`: same semantics as the text-to-video ControlNet pipeline.
- `timesteps`, `sigmas`, `enforce_inference_steps`: video-to-video scheduler controls.
- Multiple ControlNets are supported through the same constructor/control-scale patterns as `AnimateDiffControlNetPipeline`.

### Gotchas

- Source frames and control frames should have matching frame count and compatible size. The docs resize generated frames back to the conditioning frame size after generation in the example.
- ControlNet guidance preserves structural details, while `strength` controls how much the original video latents survive. Tune both together.
- AnimateLCM examples use low guidance and few steps; classic DDIM examples usually need more steps and higher guidance.

## Motion LoRA, PEFT, AnimateLCM, FreeInit, and FreeNoise

### Motion LoRAs

The official docs describe Motion LoRAs as LoRAs that work with `guoyww/animatediff-motion-adapter-v1-5-2` and add specific camera/motion styles. Load them like other Diffusers LoRA weights:

```python
pipe.load_lora_weights(
    "guoyww/animatediff-motion-lora-zoom-out",
    adapter_name="zoom-out",
)
```

With PEFT installed, multiple Motion LoRAs can be combined:

```shell
pip install peft
```

```python
pipe.load_lora_weights("diffusers/animatediff-motion-lora-zoom-out", adapter_name="zoom-out")
pipe.load_lora_weights("diffusers/animatediff-motion-lora-pan-left", adapter_name="pan-left")
pipe.set_adapters(["zoom-out", "pan-left"], adapter_weights=[1.0, 1.0])
```

### AnimateLCM

AnimateLCM is documented as a motion module checkpoint plus LCM LoRA. It speeds inference by combining:

- `MotionAdapter.from_pretrained("wangfuyun/AnimateLCM")`
- `LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")`
- `pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name=..., adapter_name="lcm-lora")`
- Lower `num_inference_steps`, often around 6 to 10.
- Lower `guidance_scale`, often around 1.5 to 2.0.

AnimateLCM is documented as compatible with existing Motion LoRAs:

```python
pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="sd15_lora_beta.safetensors", adapter_name="lcm-lora")
pipe.load_lora_weights("guoyww/animatediff-motion-lora-tilt-up", adapter_name="tilt-up")
pipe.set_adapters(["lcm-lora", "tilt-up"], [1.0, 0.8])
```

### FreeInit

FreeInit can improve temporal consistency and overall quality at inference time by iteratively refining latent initialization noise:

```python
pipe.enable_free_init(method="butterworth", use_fast_sampling=True)
output = pipe(...)
pipe.disable_free_init()
```

The docs explicitly warn that FreeInit increases compute because it samples extra times depending on `num_iters`. `use_fast_sampling=True` improves speed with some quality tradeoff compared with full FreeInit sampling.

### FreeNoise

FreeNoise supports longer video generation for these AnimateDiff pipelines:

- `AnimateDiffPipeline`
- `AnimateDiffControlNetPipeline`
- `AnimateDiffVideoToVideoPipeline`
- `AnimateDiffVideoToVideoControlNetPipeline`

Enable it after loading:

```python
pipe.enable_free_noise(context_length=16, context_stride=4)
```

After enabling FreeNoise, the prompt can be a single string or a dictionary mapping frame indices to prompt strings. Intermediate prompts are interpolated, and the interpolation can be customized with `prompt_interpolation_callback`.

For memory pressure, the docs show:

```python
pipe.enable_free_noise_split_inference()
pipe.unet.enable_forward_chunking(16)
```

`enable_free_noise_split_inference` accepts `spatial_split_size` and `temporal_split_size`. Smaller split sizes reduce VRAM and increase runtime; larger split sizes are faster and use more memory.

## Memory and Performance Checklist

- Prefer `torch.float16` on CUDA.
- Use `pipe.enable_model_cpu_offload()` when model VRAM is tight and slower runtime is acceptable.
- Use `pipe.enable_vae_slicing()` broadly for lower VAE memory.
- Use `pipe.enable_vae_tiling()` for larger SDXL frames or high resolutions.
- Reduce `num_frames`, `height`, `width`, `num_videos_per_prompt`, or `num_inference_steps` when out of memory.
- Increase `decode_chunk_size` for speed if VAE memory allows; decrease it if decoding OOMs.
- For FreeNoise, use split inference and UNet forward chunking when generating long videos.
- For LCM/AnimateLCM, use `LCMScheduler`, load the LCM LoRA, lower steps, and lower guidance.

## Validation and Input Rules

Common validation constraints from docs/source:

- `height` and `width` must be divisible by 8.
- Pass either prompt text or prompt embeddings, not both.
- Pass either negative prompt text or negative prompt embeddings, not both.
- Prompt and negative prompt batches must match.
- Prompt and negative prompt embedding shapes must match when passed directly.
- Pass either `ip_adapter_image` or `ip_adapter_image_embeds`, not both.
- `ip_adapter_image_embeds` must be a list of 3D or 4D tensors.
- Video-to-video pipelines accept either `video` or `latents`, not both.
- SparseCtrl `controlnet_frame_indices` must correspond to the supplied `conditioning_frames`.
- Custom `timesteps` must be in descending order when used.

## Output Class

Source: <https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/animatediff/pipeline_output.py>

`AnimateDiffPipelineOutput` is a dataclass with:

```python
frames: torch.Tensor | np.ndarray | list[list[PIL.Image.Image]]
```

For PIL output, it is a nested list of length `batch_size`, where each inner list contains `num_frames` denoised PIL images. For tensor/NumPy output, the documented shape is:

```text
(batch_size, num_frames, channels, height, width)
```

Most examples use:

```python
frames = output.frames[0]
```

## Choosing the Right Pipeline

- Prompt-only SD 1.5 text-to-video: `AnimateDiffPipeline`.
- Per-frame depth, pose, canny, or other dense controls without a source video: `AnimateDiffControlNetPipeline`.
- One/few keyframe scribble or RGB controls: `AnimateDiffSparseControlNetPipeline`.
- SDXL text-to-video: `AnimateDiffSDXLPipeline`, with the beta SDXL motion adapter.
- Edit an existing video by prompt: `AnimateDiffVideoToVideoPipeline`.
- Edit an existing video while preserving pose/depth/edges/composition through controls: `AnimateDiffVideoToVideoControlNetPipeline`.

## Source Links

- Official AnimateDiff docs: <https://huggingface.co/docs/diffusers/api/pipelines/animatediff>
- Official docs markdown: <https://github.com/huggingface/diffusers/blob/v0.38.0/docs/source/en/api/pipelines/animatediff.md>
- `AnimateDiffPipeline`: <https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/animatediff/pipeline_animatediff.py>
- `AnimateDiffControlNetPipeline`: <https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/animatediff/pipeline_animatediff_controlnet.py>
- `AnimateDiffSparseControlNetPipeline`: <https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/animatediff/pipeline_animatediff_sparsectrl.py>
- `AnimateDiffSDXLPipeline`: <https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/animatediff/pipeline_animatediff_sdxl.py>
- `AnimateDiffVideoToVideoPipeline`: <https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/animatediff/pipeline_animatediff_video2video.py>
- `AnimateDiffVideoToVideoControlNetPipeline`: <https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/animatediff/pipeline_animatediff_video2video_controlnet.py>
- `AnimateDiffPipelineOutput`: <https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/animatediff/pipeline_output.py>
