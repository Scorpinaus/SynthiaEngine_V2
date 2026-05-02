"""Timestep and latent preparation blocks for SD1.5 modular workflows."""

from ._common import *  # noqa: F403

class SD15Text2ImgLatentsStep(ModularPipelineBlocks):
    model_name = "sd15"

    @property
    def description(self) -> str:
        return "Prepare text-to-image timesteps and random latents."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return SD15WorkflowUtils.latent_components()

    @property
    def inputs(self) -> list[InputParam]:
        return SD15WorkflowUtils.common_generation_inputs()

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return SD15WorkflowUtils.latent_outputs()

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        device = SD15WorkflowUtils.execution_device(components)
        default_size = SD15WorkflowUtils.default_size(components)
        height = block_state.height or default_size
        width = block_state.width or default_size
        SD15WorkflowUtils.validate_dimensions(height, width)
        batch_size = SD15WorkflowUtils.batch_size_from_state(block_state)
        image_batch_size = batch_size * block_state.num_images_per_prompt
        SD15WorkflowUtils.validate_latents(components, block_state.latents, image_batch_size, height, width)
        timesteps, num_inference_steps = retrieve_timesteps(
            components.scheduler,
            num_inference_steps=block_state.num_inference_steps,
            device=device,
            timesteps=getattr(block_state, "timesteps", None),
            sigmas=getattr(block_state, "sigmas", None),
        )
        block_state.timesteps = timesteps
        block_state.num_inference_steps = num_inference_steps
        block_state.latents = SD15WorkflowUtils.prepare_text2img_latents(
            components, image_batch_size, height, width, block_state.prompt_embeds.dtype, device, block_state.generator, block_state.latents
        )
        block_state.height = height
        block_state.width = width
        self.set_block_state(state, block_state)
        return components, state


class SD15Img2ImgLatentsStep(ModularPipelineBlocks):
    model_name = "sd15"

    @property
    def description(self) -> str:
        return "Prepare img2img timesteps and noised image latents."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return SD15WorkflowUtils.latent_components(include_image=True, include_mask=True)

    @property
    def inputs(self) -> list[InputParam]:
        return SD15WorkflowUtils.common_generation_inputs(include_image=True)

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return SD15WorkflowUtils.latent_outputs(include_mask=True)

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        device = SD15WorkflowUtils.execution_device(components)
        default_size = SD15WorkflowUtils.default_size(components)
        height = block_state.height or default_size
        width = block_state.width or default_size
        SD15WorkflowUtils.validate_dimensions(height, width)
        batch_size = SD15WorkflowUtils.batch_size_from_state(block_state)
        image, _ = SD15WorkflowUtils.validate_and_resolve_image_inputs(block_state, require_mask=False)
        SD15WorkflowUtils.validate_image_batch(image, batch_size, "image")
        image_batch_size = batch_size * block_state.num_images_per_prompt
        SD15WorkflowUtils.validate_latents(components, block_state.latents, image_batch_size, height, width)
        timesteps, num_inference_steps = retrieve_timesteps(
            components.scheduler,
            num_inference_steps=block_state.num_inference_steps,
            device=device,
            timesteps=getattr(block_state, "timesteps", None),
            sigmas=getattr(block_state, "sigmas", None),
        )
        timesteps, num_inference_steps = SD15WorkflowUtils.get_timesteps_img2img(components.scheduler, num_inference_steps, block_state.strength)
        block_state.timesteps = timesteps
        block_state.num_inference_steps = num_inference_steps
        block_state.height = height
        block_state.width = width
        if block_state.latents is not None:
            block_state.latents = SD15WorkflowUtils.prepare_text2img_latents(
                components, image_batch_size, height, width, block_state.prompt_embeds.dtype, device, block_state.generator, block_state.latents
            )
            block_state.image_latents = None
            block_state.latent_noise = None
        else:
            latent_timestep = timesteps[:1].repeat(image_batch_size)
            latents, image_latents, latent_noise = SD15WorkflowUtils.prepare_img2img_latents(
                components,
                image,
                batch_size,
                block_state.num_images_per_prompt,
                height,
                width,
                block_state.prompt_embeds.dtype,
                device,
                block_state.generator,
                latent_timestep,
            )
            block_state.latents = latents
            block_state.image_latents = image_latents
            block_state.latent_noise = latent_noise
        block_state.mask = None
        self.set_block_state(state, block_state)
        return components, state


class SD15InpaintLatentsStep(ModularPipelineBlocks):
    model_name = "sd15"

    @property
    def description(self) -> str:
        return "Prepare inpaint timesteps, masks, and noised image latents."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return SD15WorkflowUtils.latent_components(include_image=True)

    @property
    def inputs(self) -> list[InputParam]:
        return SD15WorkflowUtils.common_generation_inputs(include_image=True, include_mask=True)

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return SD15WorkflowUtils.latent_outputs(include_mask=True)

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        device = SD15WorkflowUtils.execution_device(components)
        default_size = SD15WorkflowUtils.default_size(components)
        height = block_state.height or default_size
        width = block_state.width or default_size
        SD15WorkflowUtils.validate_dimensions(height, width)
        batch_size = SD15WorkflowUtils.batch_size_from_state(block_state)
        image, mask_image = SD15WorkflowUtils.validate_and_resolve_image_inputs(block_state, require_mask=True)
        SD15WorkflowUtils.validate_image_batch(image, batch_size, "image")
        SD15WorkflowUtils.validate_image_batch(mask_image, batch_size, "mask_image")
        image_batch_size = batch_size * block_state.num_images_per_prompt
        SD15WorkflowUtils.validate_latents(components, block_state.latents, image_batch_size, height, width)
        timesteps, num_inference_steps = retrieve_timesteps(
            components.scheduler,
            num_inference_steps=block_state.num_inference_steps,
            device=device,
            timesteps=getattr(block_state, "timesteps", None),
            sigmas=getattr(block_state, "sigmas", None),
        )
        timesteps, num_inference_steps = SD15WorkflowUtils.get_timesteps_img2img(components.scheduler, num_inference_steps, block_state.strength)
        block_state.timesteps = timesteps
        block_state.num_inference_steps = num_inference_steps
        block_state.height = height
        block_state.width = width
        if block_state.latents is not None:
            block_state.latents = SD15WorkflowUtils.prepare_text2img_latents(
                components, image_batch_size, height, width, block_state.prompt_embeds.dtype, device, block_state.generator, block_state.latents
            )
            block_state.image_latents = None
            block_state.latent_noise = None
        else:
            latent_timestep = timesteps[:1].repeat(image_batch_size)
            latents, image_latents, latent_noise = SD15WorkflowUtils.prepare_img2img_latents(
                components,
                image,
                batch_size,
                block_state.num_images_per_prompt,
                height,
                width,
                block_state.prompt_embeds.dtype,
                device,
                block_state.generator,
                latent_timestep,
            )
            block_state.latents = latents
            block_state.image_latents = image_latents
            block_state.latent_noise = latent_noise
        block_state.mask = SD15WorkflowUtils.prepare_mask_latents(
            components,
            mask_image,
            batch_size,
            block_state.num_images_per_prompt,
            height,
            width,
            block_state.prompt_embeds.dtype,
            device,
        )
        self.set_block_state(state, block_state)
        return components, state


