# PixelDiT Modular Diffusers Prototype

This folder contains a custom Modular Diffusers pipeline for NVIDIA PixelDiT
text-to-image checkpoints. PixelDiT is not currently a native Diffusers
pipeline, so this package provides:

- A `PixelDiTTransformer2DModel` `ModelMixin` component.
- Sequential text-to-image modular blocks.
- A converter in `tools/convert_pixeldit_to_diffusers.py` that writes a local
  Diffusers-style repository from `pixeldit_t2i_v1.pth`.

The converted repository keeps components together:

```text
PixelDiT-Diffusers/
  config.json
  modular_model_index.json
  transformer/
    config.json
    diffusion_pytorch_model.safetensors
  text_encoder/
  tokenizer/
```

`text_encoder/` and `tokenizer/` must be populated from a local Gemma-compatible
Transformers model folder for prompt text inference. Tests and low-level block
validation can pass `prompt_embeds` directly without loading those components.

## Local Usage

```powershell
.venv\Scripts\python.exe tools\convert_pixeldit_to_diffusers.py `
  --input-dir D:\diffusion\diffusers\PixelDiT `
  --output-dir D:\diffusion\diffusers\PixelDiT-Diffusers `
  --text-encoder-source D:\diffusion\diffusers\gemma-2-2b-it `
  --tokenizer-source D:\diffusion\diffusers\gemma-2-2b-it
```

Then, from Python:

```python
import torch
from diffusers import ComponentsManager
from backend.modular_diffusers.pixeldit import PixelDiTModularPipeline

components_manager = ComponentsManager()
pipe = PixelDiTModularPipeline.from_pretrained(
    r"D:\diffusion\diffusers\PixelDiT-Diffusers",
    trust_remote_code=True,
    components_manager=components_manager,
    collection="pixeldit",
)

pipe.load_components(names="tokenizer")
pipe.load_components(names="text_encoder", torch_dtype=torch.bfloat16)
pipe.text_encoder.to("cpu")
pipe.load_components(names="transformer", torch_dtype=torch.bfloat16)
pipe.transformer.to("cuda", dtype=torch.bfloat16)

images = pipe(
    prompt="a glass greenhouse at sunrise",
    negative_prompt="low quality, blurry",
    use_chi_prompt=True,
    height=512,
    width=512,
    num_inference_steps=25,
    guidance_scale=2.75,
    output="images",
)
```

The `ComponentsManager` owns the loaded component registry for the `"pixeldit"`
collection, so additional Modular Diffusers pipelines can inspect or reuse the
same component objects with `components_manager.get_one(...)`.

For the 64 GB RAM / RTX 3060 12 GB path, use the companion runner:

```powershell
.venv\Scripts\python.exe tools\run_pixeldit_modular_components_manager.py `
  --model-dir D:\diffusion\diffusers\PixelDiT-Diffusers `
  --height 512 `
  --width 512 `
  --steps 25 `
  --guidance-scale 2.75 `
  --use-chi-prompt
```

It writes the PNG and a memory/timing JSON report to
`D:\diffusion\diffusers\PixelDiT-Diffusers\test_outputs` by default.

`--use-chi-prompt` applies PixelDiT's prompt-enhancement template from the
stage-3 training config to positive prompts before text encoding. Use
`--chi-prompt-file path\to\prompt.txt` to supply a custom template.

In sandboxed or locked-down environments, set `HF_MODULES_CACHE` to a writable
folder before calling `from_pretrained(..., trust_remote_code=True)`.

## Current Limits

- The public block surface accepts `sampling_algo="flow_dpm-solver"` and uses
  PixelDiT's flow DPM-Solver++ path with `time_uniform_flow` spacing.
- `num_images_per_prompt` is fixed to `1`.
- For 12 GB VRAM systems, keep the Gemma text encoder on CPU and move only
  `transformer` to CUDA in `bfloat16`; prompt embeddings are transferred to the
  transformer device after CPU encoding.
