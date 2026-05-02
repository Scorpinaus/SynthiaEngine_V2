"""Decoder blocks for SD1.5 modular workflows."""

from ._common import *  # noqa: F403

class SD15DecodeStep(ModularPipelineBlocks):
    model_name = "sd15"

    @property
    def description(self) -> str:
        return "Decode SD1.5 latents into the requested output format."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return SD15WorkflowUtils.decode_components()

    @property
    def inputs(self) -> list[InputParam]:
        return [InputParam("latents", type_hint=torch.Tensor, required=True), SD15WorkflowUtils.output_input_param()]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return SD15WorkflowUtils.decode_outputs()

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        block_state.images = SD15WorkflowUtils.decode_latents(components, block_state.latents, block_state.output_type)
        self.set_block_state(state, block_state)
        return components, state


