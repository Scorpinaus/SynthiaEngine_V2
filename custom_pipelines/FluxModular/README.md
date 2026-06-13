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
Base `FluxModularPipeline` supports text-to-image, image-to-image, and
inpainting. Flux Kontext remains text/image-conditioned only in this local
package.

## Memory Changes

- T5 prompt encoding uses dynamic padding by default. Pass
  `max_sequence_length=512` to recover the stock behavior.
- Passing both `prompt_embeds` and `pooled_prompt_embeds` skips CLIP and T5.
- Text-to-image noise is generated directly in packed Flux latent layout.
- Flux inpainting uses the stock packed-latent mask blend: white mask pixels
  are repainted and black mask pixels preserve the source image.
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
- `low_memory_cuda_placement="auto"` stages active components on CUDA when they
  fit the available VRAM budget. If the full Flux transformer does not fit, the
  transformer forward streams blocks to CUDA in small groups while keeping the
  rest of the transformer on CPU.
- For 16 GB system-RAM experiments, use the benchmark harness with opt-in
  bitsandbytes quantization for `text_encoder_2` and `transformer`; the default
  package path remains unquantized.

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
    low_memory_cuda_placement="auto",
    low_memory_vram_reserve_margin="3GB",
    low_memory_transformer_stream_blocks="auto",
    decode_chunk_size=1,
).images[0]
```

Inpainting uses the same pipeline class. Provide an RGB source image and a
grayscale mask:

```python
image = pipe(
    prompt="replace the masked area with a tiny repair workshop",
    image=source_image,
    mask_image=mask_image,
    strength=0.8,
    height=1024,
    width=1024,
    num_inference_steps=28,
    low_memory_transformer_buffers=True,
    decode_chunk_size=1,
).images[0]
```

For maximum RAM savings, precompute prompt embeddings once, persist them, and
construct the pipeline without loading text encoders for repeated runs.
For 12 GB VRAM or smaller GPUs, keep `num_images_per_prompt=1` or leave
`low_memory_sequential_images=True` so extra images are generated one at a time.
For a validated 16 GB system-RAM command, use `--quantization bnb_4bit`,
`--system-ram-limit 16GB`, and CUDA VAE decode in
`testing\ModularFlux\measure_flux_modular.py`.

See [Flux 12 GB VRAM Runbook](../../docs/FLUX_12GB_VRAM.md) for the full
runbook of actions taken to make the local FluxModular path complete inference
under 12 GB VRAM and the documented 16 GB system-RAM experiment.
