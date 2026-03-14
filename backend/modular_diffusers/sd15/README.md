# SD1.5 Modular Diffusers Test Repo

This folder is a local Modular Diffusers repository for Phase 1 SD1.5 migration testing.

## Scope

- Text-to-image only
- Custom `ModularPipelineBlocks` implementation
- SD1.5 component layout referenced from `runwayml/stable-diffusion-v1-5`

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

## Notes

- This Diffusers installation expects `modular_config.json` for custom block loading.
- `ModularPipeline.from_pretrained(...)` is lazy; call `load_components()` before inference.
- This Phase 1 repo intentionally omits img2img, inpaint, ControlNet, LoRA, and file-output orchestration.
