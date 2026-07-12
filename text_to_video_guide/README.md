# Diffusers Text-to-Video Pipeline Guide

This folder collects implementation guides for the Hugging Face Diffusers
pipeline families that can generate video from text, or from text plus
additional image/video/control conditioning.

The guides were researched from official Diffusers documentation and source.
Most current families use the latest public Diffusers docs/current stable docs;
legacy pages note their deprecation/version status in their individual files.

## How to Use This Folder

Start with the family that matches the model checkpoint you want to run, then
check:

- Pipeline class selection: text-to-video, image-to-video, video-to-video, or
  controllable variants.
- Model IDs and checkpoint variants.
- Required components and dtype/device constraints.
- Important `__call__` parameters such as `prompt`, `negative_prompt`,
  `num_frames`, `height`, `width`, `guidance_scale`, scheduler settings, and
  conditioning inputs.
- Memory and speed features such as CPU offload, VAE tiling/slicing, group
  offload, quantization, and torch compile.
- Output handling with `export_to_video`, `export_to_gif`, or pipeline output
  classes.

## Current Pipeline Families

| Family | Guide | Primary classes | Main modes |
| --- | --- | --- | --- |
| Allegro | [allegro.md](allegro.md) | `AllegroPipeline` | Text-to-video |
| AnimateDiff | [animatediff.md](animatediff.md) | `AnimateDiffPipeline`, `AnimateDiffSDXLPipeline`, ControlNet/SparseCtrl/V2V variants | Text-to-video, controlled video, video-to-video |
| CogVideoX | [cogvideox.md](cogvideox.md) | `CogVideoXPipeline`, `CogVideoXImageToVideoPipeline`, `CogVideoXVideoToVideoPipeline`, `CogVideoXFunControlPipeline` | Text-to-video, image-to-video, video-to-video, controllable generation |
| Cosmos | [cosmos.md](cosmos.md) | `CosmosTextToWorldPipeline`, `CosmosVideoToWorldPipeline`, Cosmos 2/2.5 variants | Text-to-world/video, video-to-world, transfer/control |
| EasyAnimate | [easyanimate.md](easyanimate.md) | `EasyAnimatePipeline` and documented inpaint/control variants | Text-to-video, image/video-to-video, control-to-video |
| Helios | [helios.md](helios.md) | `HeliosPipeline`, `HeliosPyramidPipeline` | Text-to-video, image-to-video, video-to-video |
| HunyuanVideo | [hunyuan_video.md](hunyuan_video.md) | `HunyuanVideoPipeline` | Text-to-video |
| HunyuanVideo 1.5 | [hunyuan_video15.md](hunyuan_video15.md) | `HunyuanVideo15Pipeline`, `HunyuanVideo15ImageToVideoPipeline` | Text-to-video, image-to-video |
| Kandinsky 5.0 Video | [kandinsky5_video.md](kandinsky5_video.md) | `Kandinsky5T2VPipeline`, `Kandinsky5I2VPipeline` | Text-to-video, image-to-video |
| Latte | [latte.md](latte.md) | `LattePipeline` | Text-to-video |
| LTX-Video | [ltx_video.md](ltx_video.md) | `LTXPipeline`, `LTXImageToVideoPipeline`, conditioning variants | Text-to-video, image-to-video, text/image/video-to-video |
| LTX-2 | [ltx2.md](ltx2.md) | `LTX2Pipeline`, `LTX2ImageToVideoPipeline`, conditioning variants | Text-to-video, image-to-video, conditional video and audio-aware fields |
| Mochi | [mochi.md](mochi.md) | `MochiPipeline` | Text-to-video |
| Sana Video | [sana_video.md](sana_video.md) | `SanaVideoPipeline`, `SanaImageToVideoPipeline` | Text-to-video, image/text-to-video |
| SkyReels-V2 | [skyreels_v2.md](skyreels_v2.md) | `SkyReelsV2Pipeline`, diffusion-forcing T2V/I2V/V2V variants | Text-to-video, long-form generation, image/video-to-video |
| Wan | [wan.md](wan.md) | `WanPipeline`, `WanImageToVideoPipeline`, `WanVACEPipeline`, `WanVideoToVideoPipeline`, `WanAnimatePipeline` | Text-to-video, first/last-frame I2V, controllable any-to-video, video-to-video, animation |

## Legacy and Deprecated Families

| Family | Guide | Primary classes | Notes |
| --- | --- | --- | --- |
| ModelScope / Zeroscope Text-to-Video SD | [legacy_text_to_video_sd.md](legacy_text_to_video_sd.md) | `TextToVideoSDPipeline`, `VideoToVideoSDPipeline` | Deprecated in current Diffusers docs; useful for older ModelScope and Zeroscope workflows. |
| Text2Video-Zero | [text2video_zero.md](text2video_zero.md) | `TextToVideoZeroPipeline`, `TextToVideoZeroSDXLPipeline` | Deprecated/legacy zero-shot method built on Stable Diffusion text-to-image pipelines. |

## Implementation Notes

- Video pipelines are memory-heavy. Start with documented dtype/offload settings
  before increasing resolution, frame count, or batch size.
- Some guides refer to docs from `main` because the public stable page may not
  expose a newly added family yet. Those guides call out source/stable
  assumptions directly.
- Several families use gated or very large checkpoints. Check each model card
  for license, access, recommended dtype, and expected VRAM.
- For SynthaEngine integration, keep public task identifiers and workflow
  payload shapes stable. Add new model families as new workflow capabilities
  rather than renaming existing public fields.
