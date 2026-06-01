# Local Low-Memory Modular Flux

This package is a local copy of the Diffusers `modular_pipelines/flux`
implementation with low-memory block replacements layered in. It is intentionally
kept outside `.venv` so Diffusers can be upgraded or compared against without
losing local changes.

## Entry Points

```python
from custom_pipelines.FluxModular import (
    FluxModularPipeline,
    FluxKontextModularPipeline,
    LowMemoryFluxAutoBlocks,
    LowMemoryFluxKontextAutoBlocks,
    enable_low_memory_flux_modular,
)
```

`FluxModularPipeline()` defaults to `LowMemoryFluxAutoBlocks`.
`FluxKontextModularPipeline()` defaults to `LowMemoryFluxKontextAutoBlocks`.

## Memory Changes

- T5 prompt encoding uses dynamic padding by default. Pass
  `max_sequence_length=512` to recover the stock behavior.
- Passing both `prompt_embeds` and `pooled_prompt_embeds` skips CLIP and T5.
- Text-to-image noise is generated directly in packed Flux latent layout.
- Text encoders and the transformer are eagerly offloaded before later phases
  when `low_memory_eager_offload=True`.
- A dedicated cleanup block runs between denoise and decode, offloading heavy
  modules and deleting denoise-only state before VAE decode begins.
- VAE slicing and tiling are enabled before decode.
- Decode runs in batches of `decode_chunk_size`, defaulting to `1`.
- Pass `vae_decode_device="cpu"` to keep VAE decode off the GPU, or leave it
  unset to decode on the pipeline execution device.
- `low_memory_transformer_buffers=True` replaces selected Flux transformer
  attention and single-block concatenations with reusable scratch tensors during
  denoise, then clears those tensors before VAE decode.
- Intermediate image, prompt, latent, and id tensors are pruned after they are no
  longer needed when `low_memory_prune_intermediates=True`.
- Flux Kontext image-conditioned denoise reuses a latent concatenation buffer
  instead of allocating a new `torch.cat` tensor every step.
- `low_memory_sequential_images=True` is handled at the pipeline level. When
  `num_images_per_prompt > 1`, each requested image is generated as an effective
  batch of one and aggregated in the same prompt-major order as Diffusers.

## Example

```python
import torch
from custom_pipelines.FluxModular import FluxModularPipeline, enable_low_memory_flux_modular

pipe = FluxModularPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)
pipe.load_components(torch_dtype=torch.bfloat16)
enable_low_memory_flux_modular(pipe)

image = pipe(
    prompt="a small robot reading a handwritten memory budget",
    height=1024,
    width=1024,
    num_inference_steps=28,
    num_images_per_prompt=2,
    low_memory_sequential_images=True,
    low_memory_transformer_buffers=True,
    decode_chunk_size=1,
).images[0]
```

For maximum RAM savings, precompute prompt embeddings once, persist them, and
construct the pipeline without loading text encoders for repeated runs.
For 12 GB VRAM or smaller GPUs, keep `num_images_per_prompt=1` or leave
`low_memory_sequential_images=True` so extra images are generated one at a time.
