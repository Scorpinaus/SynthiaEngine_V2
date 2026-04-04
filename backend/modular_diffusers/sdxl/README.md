# SDXL Modular Diffusers Test Repo

This folder is a local Modular Diffusers repository for an SDXL text-to-image migration path.

## Scope

- Text-to-image only in this first pass
- Reuses the official SDXL modular pipeline steps shipped in Diffusers `0.37.0`
- Uses the `stabilityai/stable-diffusion-xl-base-1.0` component layout by default
- Keeps the local repo structure aligned with `backend/modular_diffusers/sd15`

## Files

- `block.py`: local text2image block composition built from the official SDXL modular steps
- `modular_config.json`: custom block loading config for the local modular repo
- `modular_model_index.json`: component loading specs for the local modular repo
- `sdxl_modular_text2img.py`: simple standalone smoke script

## Local Usage

```python
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained(
    r"C:\Users\Admin\DiffusersProject\SynthaEngine_codex\backend\modular_diffusers\sdxl",
    trust_remote_code=True,
)
pipe.load_components(torch_dtype=torch.float16)
pipe.to("cuda")

images = pipe(
    prompt="a cinematic portrait of an astronaut in a neon jungle",
    negative_prompt="blurry, distorted, low quality",
    num_inference_steps=30,
    guidance_scale=5.0,
    width=1024,
    height=1024,
    generator=torch.Generator(device="cuda").manual_seed(1234),
    output="images",
)
```

## Simple Script

Run the standalone text-to-image script:

```powershell
.venv\Scripts\python backend\modular_diffusers\sdxl\sdxl_modular_text2img.py
```

Example with custom prompt and output path:

```powershell
.venv\Scripts\python backend\modular_diffusers\sdxl\sdxl_modular_text2img.py `
  --prompt "a hyper-detailed sci-fi city at sunrise" `
  --output backend\modular_diffusers\sdxl\outputs\city.png
```

## Notes

- `ModularPipeline.from_pretrained(...)` is lazy; call `load_components()` before inference.
- The local repo currently targets SDXL text2image only.
- `guider` and `image_processor` are created by the modular block definitions, so they are not listed in `modular_model_index.json`.
- `output="images"` returns the generated image list directly, while `output_type` controls PIL vs tensor vs latent decoding behavior.
