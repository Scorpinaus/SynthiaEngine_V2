"""Validation and prompt encoding blocks for PixelDiT."""

from __future__ import annotations

import torch
from diffusers.modular_pipelines import ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_CHI_PROMPT_LINES = [
    'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:',
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: ",
]
DEFAULT_CHI_PROMPT = "\n".join(DEFAULT_CHI_PROMPT_LINES)


def normalize_chi_prompt(chi_prompt: str | list[str] | None) -> str:
    if chi_prompt is None:
        return DEFAULT_CHI_PROMPT
    if isinstance(chi_prompt, list):
        return "\n".join(str(line) for line in chi_prompt)
    return str(chi_prompt)


def apply_chi_prompt(prompt: str | list[str], chi_prompt: str | list[str] | None = None) -> list[str]:
    prompt_list = prompt if isinstance(prompt, list) else [str(prompt)]
    prefix = normalize_chi_prompt(chi_prompt)
    return [prefix + str(item) for item in prompt_list]


def select_chi_token_window(value: torch.Tensor, model_max_length: int) -> torch.Tensor:
    if model_max_length < 2:
        raise ValueError("model_max_length must be at least 2 for CHI prompt selection.")
    sequence_length = int(value.shape[1])
    if sequence_length < model_max_length:
        raise ValueError("CHI-encoded sequence length must be at least model_max_length.")
    indices = [0] + list(range(sequence_length - model_max_length + 1, sequence_length))
    return value.index_select(1, torch.tensor(indices, device=value.device))


class PixelDiTInputValidationStep(ModularPipelineBlocks):
    model_name = "pixeldit"

    @property
    def description(self) -> str:
        return "Validate PixelDiT inputs before loading heavy components."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("prompt", type_hint=str | list[str]),
            InputParam("prompt_embeds", type_hint=torch.Tensor | None),
            InputParam("negative_prompt_embeds", type_hint=torch.Tensor | None),
            InputParam("height", type_hint=int | None),
            InputParam("width", type_hint=int | None),
            InputParam("num_inference_steps", type_hint=int, default=50),
            InputParam("guidance_scale", type_hint=float, default=2.75),
            InputParam("num_images_per_prompt", type_hint=int, default=1),
            InputParam("sampling_algo", type_hint=str, default="flow_dpm-solver"),
            InputParam("interval_guidance", type_hint=tuple[float, float], default=(0.0, 1.0)),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        if block_state.prompt is None and block_state.prompt_embeds is None:
            raise ValueError("PixelDiT requires `prompt` or `prompt_embeds`.")
        if block_state.prompt is not None and block_state.prompt_embeds is not None:
            raise ValueError("Pass either `prompt` or `prompt_embeds`, not both.")
        height = int(block_state.height or getattr(components.transformer.config, "image_size", 1024))
        width = int(block_state.width or height)
        patch_size = int(getattr(components.transformer.config, "patch_size", 16))
        if height < patch_size or width < patch_size:
            raise ValueError("PixelDiT height and width must be at least one patch.")
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(f"PixelDiT height and width must be divisible by patch_size={patch_size}.")
        if int(block_state.num_images_per_prompt) != 1:
            raise ValueError("PixelDiT modular prototype supports num_images_per_prompt=1.")
        if int(block_state.num_inference_steps) < 1:
            raise ValueError("num_inference_steps must be >= 1.")
        if str(block_state.sampling_algo) != "flow_dpm-solver":
            raise ValueError("PixelDiT currently accepts only sampling_algo='flow_dpm-solver'.")
        interval_guidance = tuple(block_state.interval_guidance or (0.0, 1.0))
        if len(interval_guidance) != 2 or interval_guidance[0] > interval_guidance[1]:
            raise ValueError("interval_guidance must be a two-value range with start <= end.")
        block_state.height = height
        block_state.width = width
        block_state.interval_guidance = (max(0.0, float(interval_guidance[0])), min(1.0, float(interval_guidance[1])))
        self.set_block_state(state, block_state)
        return components, state


class PixelDiTPromptEncodingStep(ModularPipelineBlocks):
    model_name = "pixeldit"

    @property
    def description(self) -> str:
        return "Encode Gemma prompts or reuse precomputed PixelDiT prompt embeddings."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("tokenizer", type_hint=AutoTokenizer, default_creation_method="from_pretrained"),
            ComponentSpec("text_encoder", type_hint=AutoModelForCausalLM, default_creation_method="from_pretrained"),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("prompt", type_hint=str | list[str]),
            InputParam("negative_prompt", type_hint=str | list[str], default=""),
            InputParam("prompt_embeds", type_hint=torch.Tensor | None),
            InputParam("negative_prompt_embeds", type_hint=torch.Tensor | None),
            InputParam("attention_mask", type_hint=torch.Tensor | None),
            InputParam("negative_attention_mask", type_hint=torch.Tensor | None),
            InputParam("num_images_per_prompt", type_hint=int, default=1),
            InputParam("use_chi_prompt", type_hint=bool, default=False),
            InputParam("chi_prompt", type_hint=str | list[str] | None, default=None),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("prompt_embeds", type_hint=torch.Tensor),
            OutputParam("negative_prompt_embeds", type_hint=torch.Tensor | None),
            OutputParam("attention_mask", type_hint=torch.Tensor | None),
            OutputParam("negative_attention_mask", type_hint=torch.Tensor | None),
            OutputParam("batch_size", type_hint=int),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        device = next(components.transformer.parameters()).device
        dtype = components.transformer.dtype
        max_length = int(getattr(components.transformer.config, "txt_max_length", 300))

        if block_state.prompt_embeds is not None:
            block_state.prompt_embeds = block_state.prompt_embeds.to(device=device, dtype=dtype)
            if block_state.negative_prompt_embeds is not None:
                block_state.negative_prompt_embeds = block_state.negative_prompt_embeds.to(device=device, dtype=dtype)
            if block_state.attention_mask is not None:
                block_state.attention_mask = block_state.attention_mask.to(device=device)
            if block_state.negative_attention_mask is not None:
                block_state.negative_attention_mask = block_state.negative_attention_mask.to(device=device)
            block_state.batch_size = int(block_state.prompt_embeds.shape[0])
            self.set_block_state(state, block_state)
            return components, state

        if components.tokenizer is None or components.text_encoder is None:
            raise ValueError("`tokenizer` and `text_encoder` components must be loaded when prompt_embeds are absent.")
        components.tokenizer.padding_side = "right"

        prompt = block_state.prompt if isinstance(block_state.prompt, list) else [str(block_state.prompt)]
        negative_prompt = block_state.negative_prompt
        negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [str(negative_prompt or "")]
        if len(negative_prompt) == 1 and len(prompt) > 1:
            negative_prompt = negative_prompt * len(prompt)
        if len(negative_prompt) != len(prompt):
            raise ValueError("negative_prompt batch size must match prompt batch size.")

        def encode(texts: list[str], *, use_chi_prompt: bool = False, chi_prompt: str | list[str] | None = None):
            encoder = components.text_encoder.get_decoder() if hasattr(components.text_encoder, "get_decoder") else components.text_encoder
            text_device = next(encoder.parameters()).device
            encode_max_length = max_length
            if use_chi_prompt:
                chi_prompt_text = normalize_chi_prompt(chi_prompt)
                texts = apply_chi_prompt(texts, chi_prompt_text)
                encode_max_length = len(components.tokenizer.encode(chi_prompt_text)) + max_length - 2
            tokenized = components.tokenizer(
                texts,
                max_length=encode_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(text_device)
            outputs = encoder(tokenized.input_ids, tokenized.attention_mask)
            embeds = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            attention_mask = tokenized.attention_mask
            if use_chi_prompt:
                embeds = select_chi_token_window(embeds, max_length)
                attention_mask = select_chi_token_window(attention_mask, max_length)
            return embeds.to(device=device, dtype=dtype), attention_mask.to(device=device)

        block_state.prompt_embeds, block_state.attention_mask = encode(
            prompt,
            use_chi_prompt=bool(block_state.use_chi_prompt),
            chi_prompt=block_state.chi_prompt,
        )
        block_state.negative_prompt_embeds, block_state.negative_attention_mask = encode(negative_prompt)
        block_state.batch_size = len(prompt)
        self.set_block_state(state, block_state)
        return components, state
