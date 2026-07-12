# SD1.5 Prompt Embedding Regression (July 2026)

## Failure

After upgrading the inference environment to Diffusers 0.39.0 and Transformers
5.x, SD1.5 text-to-image requests produced saturated, structureless images.
The failure reproduced through the complete `sd15.text2img` path with a fixed
seed, but not when the same model and parameters were passed directly to the
standard Diffusers pipeline.

## Root cause

SynthiaEngine's custom prompt encoder had two interacting defects:

1. `needs_embeddings` was a tuple rather than a boolean. A non-empty tuple is
   always truthy, so requests bypassed the intended routing predicate.
2. The clip-skip path only looked for
   `text_encoder.text_model.final_layer_norm`. Transformers 5.x exposes the
   layer as `text_encoder.final_layer_norm`, so the selected hidden state was
   sent to the UNet without CLIP's required final normalization.

In the reproduction, the malformed prompt embeddings had a standard deviation
of about 4.97 and a maximum absolute value of 857. Applying the correct final
LayerNorm reduced these to about 0.99 and 33 and restored coherent generation.

The apparent Diffusers regression was therefore an integration compatibility
bug exposed by the upgraded dependency set, not a scheduler, UNet, latent, or
VAE regression.

## Resolution

- Construct `needs_embeddings` as a boolean expression.
- Keep explicit clip-skip on the compatible custom path. Diffusers 0.39.0 also
  looks up the old nested attribute internally when it handles clip-skip, so
  delegating that case to native encoding raises `AttributeError` with
  Transformers 5.x.
- Prefer the flattened `final_layer_norm` attribute and retain the older nested
  attribute as a compatibility fallback.
- Regression-test both CLIP layouts and verify that ordinary prompts remain on
  the native Diffusers encoding path.
- Route SD1.5 img2img and inpainting through the same prompt-argument builder.
  When clip-skip or another custom feature produces embeddings, these pipelines
  receive mutually exclusive `prompt_embeds`/`negative_prompt_embeds` arguments
  with raw prompts and Diffusers-side `clip_skip` unset.

## Lessons

- Compare the full application path with a minimal upstream pipeline using the
  same model, seed, scheduler, steps, and CFG before investigating denoising.
- Treat custom prompt embeddings as a compatibility boundary. Test their shape,
  finiteness, scale, and normalization whenever Diffusers or Transformers is
  upgraded.
- Avoid broad custom paths when upstream behavior is sufficient. Native prompt
  encoding should remain the default for ordinary prompts without clip-skip.
- Dependency upgrades must be evaluated as a set: a Diffusers upgrade may also
  expose Transformers model-layout changes.
