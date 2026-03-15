# SD1.5 Modular Diffusers Test Repo

This folder is a local Modular Diffusers repository for Phase 1 SD1.5 migration testing.

## Scope

- Text-to-image, img2img, and inpaint
- Custom `ModularPipelineBlocks` implementation
- SD1.5 component layout referenced from `runwayml/stable-diffusion-v1-5`
- Supports either text prompts or precomputed prompt embeddings
- Supports `pil`, `np`, and `latent` output modes

## Files

- `block.py`: custom SD1.5 text-to-image block
- `modular_config.json`: block loading config for the installed Diffusers modular runtime
- `modular_model_index.json`: component loading specs for the local modular repo

## Local Usage

```python
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained(
    r"C:\Users\Admin\DiffusersProject\SynthaEngine_codex\backend\modular_diffusers\sd15",
    trust_remote_code=True,
)
pipe.load_components(torch_dtype=torch.float16)
pipe.to("cuda")

images = pipe(
    prompt="a cinematic portrait of a robot explorer",
    negative_prompt="blurry, distorted",
    num_inference_steps=30,
    guidance_scale=7.5,
    height=512,
    width=512,
    generator=torch.Generator(device="cuda").manual_seed(1234),
    output="images",
)
```

## Simple Script

Run the standalone test inference script:

```powershell
.venv\Scripts\python backend\modular_diffusers\sd15\sd15_modular_text2img.py
```

Example with custom prompt and output path:

```powershell
.venv\Scripts\python backend\modular_diffusers\sd15\sd15_modular_text2img.py `
  --prompt "a watercolor castle on a hill at sunrise" `
  --output backend\modular_diffusers\sd15\outputs\castle.png
```

Run the standalone img2img script:

```powershell
.venv\Scripts\python backend\modular_diffusers\sd15\sd15_modular_img2img.py `
  --image input.png
```

Example with custom prompt and output path:

```powershell
.venv\Scripts\python backend\modular_diffusers\sd15\sd15_modular_img2img.py `
  --image input.png `
  --prompt "turn this into a moody oil painting" `
  --output backend\modular_diffusers\sd15\outputs\img2img_result.png
```

## Img2Img / Inpaint

The modular block supports img2img by passing:

- `image`
- `strength`

Example from Python:

```python
from PIL import Image
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained(
    r"C:\Users\Admin\DiffusersProject\SynthaEngine_codex\backend\modular_diffusers\sd15",
    trust_remote_code=True,
)
pipe.load_components(torch_dtype=torch.float16)
pipe.to("cuda")

init_image = Image.open("input.png").convert("RGB")
images = pipe(
    prompt="turn this into a moody oil painting",
    negative_prompt="blurry, distorted",
    image=init_image,
    strength=0.75,
    num_inference_steps=30,
    guidance_scale=7.5,
    height=512,
    width=512,
    generator=torch.Generator(device="cuda").manual_seed(1234),
    output="images",
)
```

For inpaint, also pass `mask_image`:

```python
from PIL import Image
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained(
    r"C:\Users\Admin\DiffusersProject\SynthaEngine_codex\backend\modular_diffusers\sd15",
    trust_remote_code=True,
)
pipe.load_components(torch_dtype=torch.float16)
pipe.to("cuda")

init_image = Image.open("input.png").convert("RGB")
mask_image = Image.open("mask.png").convert("L")
images = pipe(
    prompt="replace the masked area with a glowing portal",
    negative_prompt="blurry, distorted",
    image=init_image,
    mask_image=mask_image,
    strength=0.75,
    num_inference_steps=30,
    guidance_scale=7.5,
    height=512,
    width=512,
    generator=torch.Generator(device="cuda").manual_seed(1234),
    output="images",
)
```

## Notes

- This Diffusers installation expects `modular_config.json` for custom block loading.
- `ModularPipeline.from_pretrained(...)` is lazy; call `load_components()` before inference.
- You can replace `pipe.scheduler` after `load_components()` if you want to test scheduler variants.
- ControlNet, LoRA, and file-output orchestration are still not implemented in this modular test repo.
