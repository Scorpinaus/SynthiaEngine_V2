# Modular Flux Memory Tests

Manual benchmark harnesses for the local low-memory `custom_pipelines.FluxModular`
rewrite live here. These scripts are intentionally separate from the normal
backend workflow tests because real Flux runs can load large models and may
download weights if they are not already cached.

## Supported Cases

- `flux-text2img`
- `flux-img2img`
- `flux-embeds2img`
- `flux-img2img-embeds`
- `kontext-text2img`
- `kontext-image`
- `kontext-embeds2img`
- `kontext-image-embeds`

The local modular rewrite does not currently include Flux inpaint, fill, or
control variants.

## Example

```powershell
.venv\Scripts\python.exe testing\ModularFlux\measure_flux_modular.py `
  --case flux-text2img `
  --model black-forest-labs/FLUX.1-dev `
  --width 768 `
  --height 768 `
  --steps 8 `
  --runs 1 `
  --output-json outputs\modular_flux_tests\flux_text2img.json
```

For 12 GB VRAM or smaller GPUs, keep the defaults: `--num-images 1`,
`--load-strategy phased`, `--offload auto`,
`--low-memory-sequential-images`, transformer buffers enabled, and
`--decode-chunk-size 1`. The prompt cache is enabled by default and stores
embeddings on CPU. `--cuda-placement auto` stages active components on CUDA
when they fit and streams Flux transformer blocks when the full transformer does
not fit the available VRAM budget.

## Load Profiling

The harness loads Modular Diffusers components one at a time and records load
time, process RSS, sampled peak RSS, CUDA peak allocated memory, and CUDA peak
reserved memory for each component. The JSON report includes a `loads[].component_loads`
array with one entry per component and `loads[].phase_loads` with phase-level
summaries.

Load strategy:

```powershell
--load-strategy phased
--load-strategy eager
```

`phased` is the default for benchmarks. It loads prompt components first,
pre-encodes the prompt, releases tokenizers/text encoders, then loads generation
components such as the transformer and VAE. This targets the expensive initial
load overlap on 12 GB VRAM or smaller systems without changing the
`custom_pipelines\FluxModular` implementation. `eager` keeps the older behavior
of loading every component before inference.

Prompt embedding cache:

```powershell
--prompt-cache / --no-prompt-cache
--prompt-cache-device cpu
--prompt-cache-device device
```

With `--load-strategy phased`, compatible prompt embeddings are keyed by
pipeline family, model, revision, variant, prompt, secondary prompt, sequence
length, and dtype. Cache hits skip the prompt-component load and text-encoder
step for later compatible pipeline loads, which is most useful with
`--reload-per-case` or multi-case sweeps. The default `cpu` cache device avoids
holding prompt embeddings in VRAM while the transformer and VAE load.

Staged CUDA placement:

```powershell
--cuda-placement auto
--cuda-placement cuda
--cuda-placement cpu
--vram-reserve-margin 3GB
--transformer-stream-blocks auto
```

`auto` is the benchmark default. It moves active prompt/VAE components to CUDA
when they fit and uses block streaming for the Flux transformer if the whole
transformer does not fit after reserving the requested VRAM margin. `cuda`
forces whole-component CUDA placement and may OOM. `cpu` keeps the active
components on CPU for comparison.

Useful load-time switches:

```powershell
--low-cpu-mem-usage / --no-low-cpu-mem-usage
--offload-state-dict / --no-offload-state-dict
--offload-folder C:\tmp\flux-load-offload
--device-map cpu
--max-memory 0=10GB --max-memory cpu=48GB
--use-safetensors / --no-use-safetensors
--disable-mmap
```

`--disable-mmap` only applies to Diffusers model components such as the Flux
transformer and VAE. It is opt-in because it can help some HDD or network-drive
loads, but may increase peak RAM on other systems.
