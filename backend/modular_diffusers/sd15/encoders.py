"""Input validation and encoder blocks for SD1.5 modular workflows."""

from ._common import *  # noqa: F403

class SD15InputValidationStep(ModularPipelineBlocks):
    model_name = "sd15"

    @property
    def description(self) -> str:
        return "Validate SD1.5 request inputs before lazy components are required."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("prompt", type_hint=str | list[str]),
            InputParam("negative_prompt", type_hint=str | list[str]),
            InputParam("prompt_embeds", type_hint=torch.Tensor | None),
            InputParam("negative_prompt_embeds", type_hint=torch.Tensor | None),
            InputParam("num_images_per_prompt", type_hint=int, default=1),
            InputParam("guidance_scale", type_hint=float, default=7.5),
            InputParam("height", type_hint=int),
            InputParam("width", type_hint=int),
            InputParam("latents", type_hint=torch.Tensor | None),
            InputParam("image", type_hint=PIL.Image.Image | list[PIL.Image.Image] | torch.Tensor | None),
            InputParam("mask_image", type_hint=PIL.Image.Image | list[PIL.Image.Image] | torch.Tensor | None),
            InputParam("strength", type_hint=float, default=0.8),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return []

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        SD15WorkflowUtils.validate_prompt_inputs(block_state)
        height = block_state.height or 512
        width = block_state.width or 512
        SD15WorkflowUtils.validate_dimensions(height, width)
        batch_size = SD15WorkflowUtils.batch_size_from_state(block_state)
        image_batch_size = batch_size * block_state.num_images_per_prompt
        SD15WorkflowUtils.validate_latents_without_components(
            block_state.latents,
            image_batch_size,
            height,
            width,
        )
        if block_state.image is not None or block_state.mask_image is not None:
            image, mask_image = SD15WorkflowUtils.validate_and_resolve_image_inputs(
                block_state,
                require_mask=block_state.mask_image is not None,
            )
            SD15WorkflowUtils.validate_image_batch(image, batch_size, "image")
            if mask_image is not None:
                SD15WorkflowUtils.validate_image_batch(mask_image, batch_size, "mask_image")
        return components, state


class SD15PromptEncodingStep(ModularPipelineBlocks):
    model_name = "sd15"

    @property
    def description(self) -> str:
        return "Encode text prompts or reuse precomputed prompt embeddings."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return SD15WorkflowUtils.prompt_components()

    @property
    def inputs(self) -> list[InputParam]:
        return SD15WorkflowUtils.prompt_input_params()

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return SD15WorkflowUtils.prompt_encoding_outputs()

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        SD15WorkflowUtils.validate_prompt_inputs(block_state)
        device = SD15WorkflowUtils.execution_device(components)
        text_encoder_dtype = getattr(components.text_encoder, "dtype", torch.float32)
        if getattr(block_state, "prompt_embeds", None) is not None:
            block_state.batch_size = int(block_state.prompt_embeds.shape[0])
            block_state.prompt_embeds = SD15WorkflowUtils.prepare_prompt_embeds(
                prompt_embeds=block_state.prompt_embeds,
                negative_prompt_embeds=getattr(block_state, "negative_prompt_embeds", None),
                device=device,
                dtype=text_encoder_dtype,
                num_images_per_prompt=block_state.num_images_per_prompt,
                guidance_scale=block_state.guidance_scale,
            )
        else:
            block_state.batch_size = SD15WorkflowUtils.batch_size_from_state(block_state)
            block_state.prompt_embeds = SD15WorkflowUtils.encode_prompt(
                components=components,
                prompt=block_state.prompt,
                negative_prompt=getattr(block_state, "negative_prompt", None),
                device=device,
                num_images_per_prompt=block_state.num_images_per_prompt,
                guidance_scale=block_state.guidance_scale,
            )
        self.set_block_state(state, block_state)
        return components, state


