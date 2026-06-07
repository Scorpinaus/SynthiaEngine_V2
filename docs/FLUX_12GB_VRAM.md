# Flux 12 GB VRAM Runbook

This document records the low-memory actions added around the local
`custom_pipelines.FluxModular` rewrite so Flux can complete inference on GPUs
with 12 GB VRAM or less. The goal is to reduce peak VRAM overlap first, then
make CPU/system-memory costs visible enough to tune.

## Target Constraint

The practical target is successful Flux text-to-image inference at 768 x 768 on
a 12 GB VRAM GPU. The current low-memory path prioritizes completion over raw
speed: large weights can remain CPU-resident and stream to CUDA in smaller
groups, which reduces OOM risk but can increase wall time and system RAM use.

## Main Memory Problems

Flux has three large memory pressure points:

- `text_encoder_2` / T5 prompt encoding can consume high system RAM and is slow
  to initialize on HDD-backed model folders.
- `FluxTransformer2DModel` dominates generation memory. Keeping it resident on
  CUDA together with prompt encoders or VAE decode is the easiest way to OOM on
  smaller cards.
- VAE decode can overlap with denoise leftovers unless the transformer and
  temporary tensors are explicitly released first.

The local rewrite attacks those overlaps by splitting inference into phases,
cleaning temporary state between phases, and only placing the active component
or active transformer block group on CUDA.

## Implemented Actions

### Phased Loading

The benchmark harness defaults to `--load-strategy phased`.

Phased loading runs the model in this order:

1. Load prompt components: tokenizers, CLIP, and T5.
2. Encode the prompt.
3. Release prompt components.
4. Load generation components: scheduler, transformer, and VAE.
5. Denoise.
6. Release denoise-only state before decode.
7. Decode images.

This avoids holding T5 and the Flux transformer at the same time during normal
benchmark execution. It does not reduce the size of either component; it reduces
the worst overlap.

### Prompt Encoder Release

After prompt embeddings are produced, prompt-only components are released before
generation components are loaded. This is especially important for
`text_encoder_2`, which is one of the largest non-transformer components in
Flux.

Prompt embeddings can be passed directly with `prompt_embeds` and
`pooled_prompt_embeds`, which skips CLIP and T5 inside the pipeline call.

### Prompt Embedding Cache

The benchmark harness enables `--prompt-cache` by default and stores cached
embeddings on CPU with `--prompt-cache-device cpu`.

The cache key includes the pipeline family, model path, revision, variant,
prompt text, secondary prompt, sequence length, and dtype. Compatible later runs
can skip prompt-component loading and prompt encoding. This helps most with
multi-case sweeps, reload-per-case tests, and repeated benchmark runs with the
same prompt.

### Dynamic T5 Prompt Length

The local FluxModular prompt path uses dynamic padding by default instead of
always padding T5 embeddings to the stock maximum sequence length. Short prompts
therefore carry fewer text tokens through prompt encoding and transformer
attention.

Use `max_sequence_length=512` only when the stock behavior is required.

### Packed Latents Generated Directly

Text-to-image latents are generated directly in Flux packed latent layout. This
avoids creating a temporary unpacked BCHW latent tensor and packing it
immediately afterward.

### Sequential Images

`low_memory_sequential_images=True` is the low-memory default. When
`num_images_per_prompt > 1`, images are generated one at a time and aggregated
afterward instead of increasing the effective denoise batch size.

For 12 GB VRAM testing, keep `--num-images 1` unless measuring the sequential
multi-image path directly.

### Denoise Cleanup Before Decode

A dedicated cleanup block runs between denoise and VAE decode. It deletes or
offloads denoise-only state before image decoding begins, including transformer
state, prompt embeddings, ids, and other intermediate tensors that are no
longer needed.

This reduces the dangerous overlap between transformer residues and VAE decode.

### VAE Decode Chunking

Decode uses `decode_chunk_size=1` by default. Multi-image outputs decode one
sample at a time, and VAE slicing/tiling are enabled before decode.

The VAE can be decoded on CPU or CUDA:

- `--vae-decode-device cpu` minimizes VRAM but can be slow.
- `--vae-decode-device cuda` is faster when enough VRAM remains after denoise
  cleanup.

For a 12 GB GPU, CUDA VAE decode is usually worth testing after the streamed
transformer path is stable.

### Reusable Transformer Buffers

`low_memory_transformer_buffers=True` reduces allocation churn in selected Flux
transformer paths by reusing scratch buffers during denoise, then clearing those
buffers before decode.

This targets repeated concatenation and temporary tensor allocation in the
hottest loop. It is a memory-fragmentation and allocator-pressure optimization,
not a quality-changing operation.

### Staged CUDA Placement

The low-memory path accepts:

```powershell
--cuda-placement auto
--vram-reserve-margin 3GB
--transformer-stream-blocks auto
```

With `auto`, active prompt and VAE components are moved to CUDA only if they fit
the available VRAM budget. For the Flux transformer, the system first checks
whether the whole transformer fits after reserving the margin. If it does not,
it enables transformer block streaming.

### Transformer Block Streaming

Block streaming keeps most transformer weights on CPU and moves small groups of
transformer blocks to CUDA during the forward pass. After each group runs, it is
moved back to CPU and CUDA cache pressure is reduced before the next group.

This is the core completion path for low-VRAM systems. It should not change
image quality by itself because it changes placement and execution order, not
weights, scheduler math, prompts, or latent values. Quality can change only if
other settings change dtype, quantization, seed, scheduler, prompt length, or
model weights.

Larger block groups reduce CPU-to-GPU transfer overhead but use more VRAM. If a
run succeeds with a low CUDA peak, increase `--transformer-stream-blocks` from
`auto` to values such as `8` to test speed improvements.

### Per-Component Load Profiling

The benchmark harness records phase and component load timings together with:

- process RSS before and after load
- sampled peak RSS
- CUDA peak allocated memory
- CUDA peak reserved memory
- device placement events

This matters because the low-VRAM path can shift pressure from VRAM to system
RAM and disk IO. On HDD-backed model folders, initial load and prompt encode can
still be slow even when VRAM is controlled.

## Recommended 12 GB Benchmark Command

For the local model folder at `D:\diffusion\diffusers\FLUX.1-dev`:

```powershell
.venv\Scripts\python.exe testing\ModularFlux\measure_flux_modular.py `
  --case flux-text2img `
  --model "D:\diffusion\diffusers\FLUX.1-dev" `
  --local-files-only `
  --width 768 `
  --height 768 `
  --steps 8 `
  --runs 1 `
  --load-strategy phased `
  --prompt-cache `
  --prompt-cache-device cpu `
  --cuda-placement auto `
  --vram-reserve-margin 3GB `
  --transformer-stream-blocks auto `
  --vae-decode-device cpu `
  --output-json outputs\modular_flux_tests\flux_text2img_768_streamed.json
```

After this succeeds, test faster decode:

```powershell
--vae-decode-device cuda --decode-chunk-size 1
```

If CUDA peak memory remains comfortably below the card limit, test larger
transformer stream groups:

```powershell
--transformer-stream-blocks 8
```

## 16 GB System RAM Experiment

The remaining system-RAM pressure comes mostly from keeping full Flux weights
CPU-resident for block streaming. The benchmark harness now has two opt-in
system-RAM controls:

```powershell
--system-ram-limit 16GB
--quantization bnb_4bit
```

`--system-ram-limit` records the configured RSS cap in JSON and marks load or
run profiles as `rss_limit_exceeded` if sampled process RSS goes above it.
`--quantization bnb_4bit` applies bitsandbytes 4-bit loading to
`text_encoder_2` and `transformer` only; CLIP, tokenizers, scheduler, and VAE
stay unquantized.

Successful local 16 GB system-RAM command on 2026-06-06:

```powershell
.venv\Scripts\python.exe testing\ModularFlux\measure_flux_modular.py `
  --case flux-text2img `
  --model "D:\diffusion\diffusers\FLUX.1-dev" `
  --local-files-only `
  --width 768 `
  --height 768 `
  --steps 8 `
  --runs 1 `
  --load-strategy phased `
  --prompt-cache `
  --prompt-cache-device cpu `
  --cuda-placement auto `
  --vram-reserve-margin 3GB `
  --transformer-stream-blocks auto `
  --vae-decode-device cuda `
  --decode-chunk-size 1 `
  --quantization bnb_4bit `
  --system-ram-limit 16GB `
  --low-cpu-mem-usage `
  --output-json outputs\modular_flux_tests\flux_text2img_768_8step_bnb4bit_16gb_cuda_decode.json
```

Result:

- status: success
- elapsed inference time: about 31 seconds
- peak sampled process RSS during load: about 11.0 GB
- peak sampled process RSS during inference: about 7.9 GB
- peak CUDA allocated: about 5.9 GB
- transformer placement: `block-stream`, CUDA, `blocks_per_group=5`

This confirms the local Flux text-to-image path can complete the same 768 x
768, 8-step workflow under a 16 GB system-RAM cap when bnb 4-bit transformer and
T5 loading are enabled. This is not numerically identical to the bf16 runbook
path because quantization changes weight representation.

Counterexamples from the same experiment:

- Adding `--offload-state-dict --offload-folder C:\tmp\flux-load-offload` kept
  load RSS under 16 GB but failed at inference with
  `NotImplementedError: Cannot copy out of meta tensor; no data!`. Do not combine
  bnb 4-bit phased prompt encoding with load-time state-dict offload until that
  incompatibility is fixed.
- The same 768 x 768 run with `--vae-decode-device cpu` completed denoise but
  did not return before a 15-minute command timeout, so CUDA VAE decode is the
  recommended 16 GB system-RAM path when VRAM allows it.

## Known Tradeoffs

- VRAM use is greatly reduced, but system RAM can remain high because transformer
  weights are CPU-resident during block streaming.
- A 16 GB system-RAM cap currently requires quantized transformer/T5 loading or
  another non-default weight storage strategy. Full bf16 CPU-resident block
  streaming remains around the 38 GB RSS range in the recorded baseline.
- HDD-backed model folders can make initial load and prompt encoding feel slow.
  Moving the model folder to SSD is a load-time optimization, not a quality
  change.
- CPU VAE decode is safe for VRAM but slow. CUDA VAE decode should be preferred
  when denoise cleanup leaves enough VRAM.
- Prompt cache currently avoids repeated in-process prompt work for compatible
  runs. A persistent disk prompt-embedding cache would further reduce repeated
  prompt encode cost across separate PowerShell invocations.

## Example Result

A successful 768 x 768, 8-step text-to-image run with block-stream placement
reported:

- elapsed time: about 160 seconds
- peak CUDA allocated: about 2.9 GB
- peak CUDA reserved: about 3.0 GB
- peak sampled process RSS: about 38 GB
- transformer placement: `block-stream`, CUDA, `blocks_per_group=4`

This confirms the current implementation is meeting the primary 12 GB VRAM goal.
The remaining optimization work is mostly about speed and system RAM pressure,
not avoiding CUDA OOM.
