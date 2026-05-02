"""Denoising blocks for SD1.5 modular workflows."""

from ._common import *  # noqa: F403

class SD15DenoiseStep(ModularPipelineBlocks):
    model_name = "sd15"

    @property
    def description(self) -> str:
        return "Run the SD1.5 denoising loop."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return SD15WorkflowUtils.denoise_components()

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("prompt_embeds", type_hint=torch.Tensor, required=True),
            InputParam("guidance_scale", type_hint=float, default=7.5),
            InputParam("latents", type_hint=torch.Tensor, required=True),
            InputParam("timesteps", type_hint=torch.Tensor, required=True),
            InputParam("eta", type_hint=float, default=0.0),
            InputParam("generator", type_hint=torch.Generator | list[torch.Generator] | None),
            InputParam("image_latents", type_hint=torch.Tensor | None),
            InputParam("latent_noise", type_hint=torch.Tensor | None),
            InputParam("mask", type_hint=torch.Tensor | None),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("latents", type_hint=torch.Tensor)]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        extra_step_kwargs = SD15WorkflowUtils.prepare_extra_step_kwargs(components, block_state.generator, block_state.eta)
        do_classifier_free_guidance = block_state.guidance_scale > 1.0
        image_latents = getattr(block_state, "image_latents", None)
        latent_noise = getattr(block_state, "latent_noise", None)
        mask = getattr(block_state, "mask", None)

        for timestep_index, timestep in enumerate(block_state.timesteps):
            latent_model_input = torch.cat([block_state.latents] * 2) if do_classifier_free_guidance else block_state.latents
            if hasattr(components.scheduler, "scale_model_input"):
                latent_model_input = components.scheduler.scale_model_input(latent_model_input, timestep)
            noise_pred = components.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=block_state.prompt_embeds,
                return_dict=False,
            )[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + block_state.guidance_scale * (noise_pred_text - noise_pred_uncond)
            block_state.latents = components.scheduler.step(
                noise_pred,
                timestep,
                block_state.latents,
                return_dict=False,
                **extra_step_kwargs,
            )[0]
            if mask is not None and image_latents is not None and latent_noise is not None:
                if timestep_index < len(block_state.timesteps) - 1:
                    next_timestep = block_state.timesteps[timestep_index + 1]
                    init_latents_proper = components.scheduler.add_noise(image_latents, latent_noise, next_timestep)
                else:
                    init_latents_proper = image_latents
                block_state.latents = init_latents_proper * (1 - mask) + block_state.latents * mask

        self.set_block_state(state, block_state)
        return components, state


