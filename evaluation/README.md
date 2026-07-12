# Evaluation baselines

These scripts exercise upstream model pipelines without importing SynthaEngine.
They provide a control when application functions or installed packages change.
Generated artifacts are written below `evaluation/outputs/`, which is ignored by
Git.

## SDXL text-to-image

The requested base model, `stable-diffusion-xl-base-1-0`, is SDXL rather than
SD 1.5. Run its isolated baseline from the repository root:

```powershell
.\.venv\Scripts\python.exe evaluation\sdxl_txt2img_baseline.py
```

It uses only the local model at
`D:\diffusion\diffusers\stable-diffusion-xl-base-1-0` and writes a PNG plus JSON
environment/timing metadata under `evaluation/outputs/sdxl_txt2img_baseline/`.
For a quicker, lower-memory smoke run, use `--width 512 --height 512 --steps 5`.

## SD 1.5 text-to-image

From the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe evaluation\sd15_txt2img_baseline.py
```

The default model is the local Diffusers model at
`D:\diffusion\diffusers\raemumix_v90`, so the normal baseline run does not need
to download model weights. The script writes `baseline.png` and `baseline.json`;
the JSON records inputs, runtime/package versions, timings, and the PNG SHA-256
digest. Re-run the same command after a change and compare the metadata and
image. Exact pixels can vary across devices, drivers, and package versions, so
treat a changed digest as a signal to inspect rather than an automatic failure.

Useful overrides:

```powershell
.\.venv\Scripts\python.exe evaluation\sd15_txt2img_baseline.py --seed 42 --steps 30
.\.venv\Scripts\python.exe evaluation\sd15_txt2img_baseline.py --model D:\models\sd15
.\.venv\Scripts\python.exe evaluation\sd15_txt2img_baseline.py --device cpu
```

## SD 1.5 image-to-image

The default run creates and saves a deterministic input fixture:

```powershell
.\.venv\Scripts\python.exe evaluation\sd15_img2img_baseline.py
```

To use an existing image:

```powershell
.\.venv\Scripts\python.exe evaluation\sd15_img2img_baseline.py --input-image D:\images\input.png
```

## SD 1.5 inpainting

The default run creates deterministic input and mask fixtures. White mask pixels
are repainted and black pixels are preserved:

```powershell
.\.venv\Scripts\python.exe evaluation\sd15_inpaint_baseline.py
```

To use existing input and mask images:

```powershell
.\.venv\Scripts\python.exe evaluation\sd15_inpaint_baseline.py --input-image D:\images\input.png --mask-image D:\images\mask.png
```
