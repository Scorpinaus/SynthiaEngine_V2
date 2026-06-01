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
`--offload auto`, `--low-memory-sequential-images`, transformer buffers enabled,
and `--decode-chunk-size 1`.
