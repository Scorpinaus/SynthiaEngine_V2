# SDXL Modular Diffusers Text-to-Image Smoke Script

This folder contains simple SDXL smoke scripts that use the built-in SDXL
Modular Diffusers support from the installed `diffusers` package.

## Scope

- Text-to-image, img2img, and inpainting smoke paths
- Uses `ModularPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")`
  by default
- Keeps the local repo structure aligned with `backend/modular_diffusers/sd15`

## Files

- `sdxl_modular_text2img.py`: standalone smoke script using Diffusers' built-in
  SDXL modular pipeline mapping
- `sdxl_modular_img2img.py`: standalone img2img smoke script using the same
  built-in SDXL modular pipeline mapping
- `sdxl_modular_inpaint.py`: standalone inpaint smoke script using the same
  built-in SDXL modular pipeline mapping
- `sdxl_modular_controlnet_text2img.py`: standalone ControlNet text-to-image
  smoke script using the same built-in SDXL modular pipeline mapping
- `sdxl_modular_controlnet_img2img.py`: standalone ControlNet img2img smoke
  script using the same built-in SDXL modular pipeline mapping
- `sdxl_modular_controlnet_inpaint.py`: standalone ControlNet inpaint smoke
  script using the same built-in SDXL modular pipeline mapping
- `block.py`, `modular_config.json`, `modular_model_index.json`: local custom
  modular repo experiment files retained for reference

## Local Usage

```python
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
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

Run text-to-image with an experimental LoRA adapter:

```powershell
.venv\Scripts\python backend\modular_diffusers\sdxl\sdxl_modular_text2img.py `
  --prompt "a hyper-detailed sci-fi city at sunrise" `
  --lora path\to\style.safetensors `
  --lora-weight 0.8 `
  --output backend\modular_diffusers\sdxl\outputs\city_lora.png
```

Run text-to-image with experimental textual inversion embeddings:

```powershell
.venv\Scripts\python backend\modular_diffusers\sdxl\sdxl_modular_text2img.py `
  --prompt "a portrait in <my-style> style" `
  --textual-inversion path\to\clip_l_embedding.pt `
  --textual-inversion-token "<my-style>" `
  --textual-inversion-2 path\to\clip_g_embedding.pt `
  --textual-inversion-2-token "<my-style>" `
  --output backend\modular_diffusers\sdxl\outputs\city_ti.png
```

Run img2img:

```powershell
.venv\Scripts\python backend\modular_diffusers\sdxl\sdxl_modular_img2img.py `
  --image path\to\input.png `
  --prompt "turn this into a cinematic matte painting" `
  --output backend\modular_diffusers\sdxl\outputs\img2img.png
```

Run inpainting:

```powershell
.venv\Scripts\python backend\modular_diffusers\sdxl\sdxl_modular_inpaint.py `
  --image path\to\input.png `
  --mask-image path\to\mask.png `
  --prompt "replace the masked area with a glowing portal" `
  --output backend\modular_diffusers\sdxl\outputs\inpaint.png
```

Run ControlNet text-to-image:

```powershell
.venv\Scripts\python backend\modular_diffusers\sdxl\sdxl_modular_controlnet_text2img.py `
  --control-image path\to\canny.png `
  --prompt "a cinematic portrait following the edge map" `
  --output backend\modular_diffusers\sdxl\outputs\controlnet_text2img.png
```

Run ControlNet img2img:

```powershell
.venv\Scripts\python backend\modular_diffusers\sdxl\sdxl_modular_controlnet_img2img.py `
  --image path\to\input.png `
  --control-image path\to\canny.png `
  --prompt "turn this into a cinematic matte painting" `
  --output backend\modular_diffusers\sdxl\outputs\controlnet_img2img.png
```

Run ControlNet inpainting:

```powershell
.venv\Scripts\python backend\modular_diffusers\sdxl\sdxl_modular_controlnet_inpaint.py `
  --image path\to\input.png `
  --mask-image path\to\mask.png `
  --control-image path\to\canny.png `
  --prompt "replace the masked area while following the control image" `
  --output backend\modular_diffusers\sdxl\outputs\controlnet_inpaint.png
```

## Notes

- `ModularPipeline.from_pretrained(...)` is lazy; call `load_components()` before inference.
- The smoke scripts intentionally use Diffusers' built-in SDXL modular repository
  support instead of the local custom SDXL repo files.
- ControlNet smoke scripts default to `diffusers/controlnet-canny-sdxl-1.0`.
  Provide a preprocessed conditioning image that matches the selected ControlNet.
- LoRA and textual inversion support is experimental. Diffusers `ModularPipeline`
  does not expose these adapter APIs directly, so the smoke scripts add a local
  compatibility layer around Diffusers' SDXL LoRA and textual inversion loader mixins.
- SDXL textual inversion can require separate embeddings for the first and second
  text encoders. Use `--textual-inversion` for `tokenizer`/`text_encoder` and
  `--textual-inversion-2` for `tokenizer_2`/`text_encoder_2`.
- `output="images"` returns the generated image list directly, while `output_type` controls PIL vs tensor vs latent decoding behavior.
