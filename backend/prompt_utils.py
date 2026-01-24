from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

import torch


DEFAULT_EMPHASIS = 1.1
DEFAULT_DEEMPHASIS = 1 / DEFAULT_EMPHASIS
NORMALIZE_PROMPT_WEIGHTS = True
NORMALIZATION_EPS = 1e-6


class WeightingPolicy(str, Enum):
    DIFFUSERS_LIKE = "diffusers-like"
    A1111_LIKE = "a1111-like"
    COMFYUI_LIKE = "comfyui-like"

    MULTIPLY = "multiply"  # legacy alias for A1111-like
    DELTA_FROM_EMPTY = "delta_from_empty"  # legacy alias for ComfyUI-like


_A1111_RE_ATTENTION = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)

_A1111_RE_BREAK = re.compile(r"\s*\bBREAK\b\s*", re.S)


@dataclass
class ParsedPrompt:
    tokens: list[int]
    weights: list[float]
    has_weights: bool


def parse_prompt_attention(
    text: str,
    base_weight: float = 1.0,
    weighting_policy: WeightingPolicy | str = WeightingPolicy.DIFFUSERS_LIKE,
) -> list[tuple[str, float]]:
    policy = _coerce_weighting_policy(weighting_policy)
    if policy is WeightingPolicy.DIFFUSERS_LIKE:
        return parse_prompt_attention_diffusers(text)
    if policy is WeightingPolicy.A1111_LIKE:
        return parse_prompt_attention_a1111(text)
    if policy is WeightingPolicy.COMFYUI_LIKE:
        return parse_prompt_attention_comfyui(text, base_weight=base_weight)
    raise ValueError(f"Unknown weighting_policy: {weighting_policy!r}")


def parse_prompt_attention_diffusers(text: str) -> list[tuple[str, float]]:
    return _parse_prompt_attention_a1111(text, enable_break=False)


def parse_prompt_attention_a1111(text: str) -> list[tuple[str, float]]:
    return _parse_prompt_attention_a1111(text, enable_break=True)


def parse_prompt_attention_comfyui(text: str, base_weight: float = 1.0) -> list[tuple[str, float]]:
    return _parse_prompt_attention_comfyui(text, base_weight=base_weight)


def tokenize_prompt(
    tokenizer,
    prompt: str,
    weighting_policy: WeightingPolicy | str = WeightingPolicy.DIFFUSERS_LIKE,
) -> ParsedPrompt:
    segments = parse_prompt_attention(prompt, weighting_policy=weighting_policy)
    tokens: list[int] = []
    weights: list[float] = []
    has_weights = False

    for text, weight in segments:
        if not text:
            continue
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        if not token_ids:
            continue
        tokens.extend(token_ids)
        weights.extend([weight] * len(token_ids))
        if weight != 1.0:
            has_weights = True

    return ParsedPrompt(tokens=tokens, weights=weights, has_weights=has_weights)


def build_prompt_embeddings(
    pipe,
    prompt: str,
    negative_prompt: str | None,
    clip_skip: int | None = None,
    lora_scale: float | None = None,
    normalize_weights: bool = NORMALIZE_PROMPT_WEIGHTS,
    weighting_policy: WeightingPolicy | str = WeightingPolicy.DIFFUSERS_LIKE,
) -> tuple[torch.Tensor | None, torch.Tensor | None, bool]:
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    device = text_encoder.device

    prompt_tokens = tokenize_prompt(tokenizer, prompt, weighting_policy=weighting_policy)
    negative_tokens = tokenize_prompt(tokenizer, negative_prompt or "", weighting_policy=weighting_policy)

    max_chunk = max(1, tokenizer.model_max_length - 2)
    needs_embeddings = (
        lora_scale is not None,
        prompt_tokens.has_weights
        or negative_tokens.has_weights
        or len(prompt_tokens.tokens) > max_chunk
        or len(negative_tokens.tokens) > max_chunk
    )

    if not needs_embeddings:
        return None, None, False

    scaled = False
    use_peft = False
    try:
        if lora_scale is not None:
            try:
                from diffusers.utils import USE_PEFT_BACKEND

                use_peft = bool(USE_PEFT_BACKEND)
            except Exception:
                use_peft = False

            if use_peft:
                try:
                    from diffusers.utils.peft_utils import scale_lora_layers

                    scale_lora_layers(text_encoder, lora_scale)
                    scaled = True
                except Exception:
                    scaled = False
            else:
                # Best-effort for non-PEFT LoRA. Diffusers doesn't "unscale" in this case.
                try:
                    from diffusers.models.lora import adjust_lora_scale_text_encoder

                    adjust_lora_scale_text_encoder(text_encoder, lora_scale)
                except Exception:
                    pass

        prompt_embeds = _encode_tokens(
            text_encoder,
            tokenizer,
            prompt_tokens.tokens,
            prompt_tokens.weights,
            device,
            max_chunk,
            clip_skip,
            normalize_weights,
            weighting_policy,
        )
        negative_prompt_embeds = _encode_tokens(
            text_encoder,
            tokenizer,
            negative_tokens.tokens,
            negative_tokens.weights,
            device,
            max_chunk,
            clip_skip,
            normalize_weights,
            weighting_policy,
        )
    finally:
        if lora_scale is not None and scaled and use_peft:
            try:
                from diffusers.utils.peft_utils import unscale_lora_layers

                unscale_lora_layers(text_encoder, lora_scale)
            except Exception:
                pass

    if prompt_embeds.shape[1] != negative_prompt_embeds.shape[1]:
        target_length = max(prompt_embeds.shape[1], negative_prompt_embeds.shape[1])
        prompt_embeds = _pad_embeddings(prompt_embeds, target_length)
        negative_prompt_embeds = _pad_embeddings(negative_prompt_embeds, target_length)
    return prompt_embeds, negative_prompt_embeds, True


def build_prompt_embeddings_diffusers(
    pipe,
    prompt: str,
    negative_prompt: str | None,
    clip_skip: int | None = None,
    lora_scale: float | None = None,
    normalize_weights: bool = NORMALIZE_PROMPT_WEIGHTS,
) -> tuple[torch.Tensor | None, torch.Tensor | None, bool]:
    return build_prompt_embeddings(
        pipe,
        prompt,
        negative_prompt,
        clip_skip=clip_skip,
        lora_scale=lora_scale,
        normalize_weights=normalize_weights,
        weighting_policy=WeightingPolicy.DIFFUSERS_LIKE,
    )


def build_prompt_embeddings_a1111(
    pipe,
    prompt: str,
    negative_prompt: str | None,
    clip_skip: int | None = None,
    lora_scale: float | None = None,
    normalize_weights: bool = NORMALIZE_PROMPT_WEIGHTS,
) -> tuple[torch.Tensor | None, torch.Tensor | None, bool]:
    return build_prompt_embeddings(
        pipe,
        prompt,
        negative_prompt,
        clip_skip=clip_skip,
        lora_scale=lora_scale,
        normalize_weights=normalize_weights,
        weighting_policy=WeightingPolicy.A1111_LIKE,
    )


def build_prompt_embeddings_comfyui(
    pipe,
    prompt: str,
    negative_prompt: str | None,
    clip_skip: int | None = None,
    lora_scale: float | None = None,
    normalize_weights: bool = NORMALIZE_PROMPT_WEIGHTS,
) -> tuple[torch.Tensor | None, torch.Tensor | None, bool]:
    return build_prompt_embeddings(
        pipe,
        prompt,
        negative_prompt,
        clip_skip=clip_skip,
        lora_scale=lora_scale,
        normalize_weights=normalize_weights,
        weighting_policy=WeightingPolicy.COMFYUI_LIKE,
    )


def _encode_tokens(
    text_encoder,
    tokenizer,
    tokens: list[int],
    weights: list[float],
    device: torch.device,
    chunk_size: int,
    clip_skip: int | None,
    normalize_weights: bool,
    weighting_policy: WeightingPolicy | str,
) -> torch.Tensor:
    policy = _coerce_weighting_policy(weighting_policy)
    if policy is WeightingPolicy.DIFFUSERS_LIKE:
        return _encode_tokens_diffusers_like(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            tokens=tokens,
            weights=weights,
            device=device,
            chunk_size=chunk_size,
            clip_skip=clip_skip,
            normalize_weights=normalize_weights,
        )
    if policy is WeightingPolicy.A1111_LIKE:
        return _encode_tokens_a1111_like(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            tokens=tokens,
            weights=weights,
            device=device,
            chunk_size=chunk_size,
            clip_skip=clip_skip,
            normalize_weights=normalize_weights,
        )
    if policy is WeightingPolicy.COMFYUI_LIKE:
        return _encode_tokens_comfyui_like(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            tokens=tokens,
            weights=weights,
            device=device,
            chunk_size=chunk_size,
            clip_skip=clip_skip,
        )
    raise ValueError(f"Unknown weighting_policy: {weighting_policy!r}")


def _encode_text_encoder(
    *,
    text_encoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    clip_skip: int | None,
) -> torch.Tensor:
    if clip_skip is None:
        return text_encoder(input_ids, attention_mask=attention_mask)[0]

    outputs = text_encoder(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hidden_states = outputs[-1]
    selected = hidden_states[-(clip_skip + 1)]
    final_ln = getattr(getattr(text_encoder, "text_model", None), "final_layer_norm", None)
    if final_ln is not None:
        selected = final_ln(selected)
    return selected


def _build_padded_chunk_input(
    *,
    tokenizer,
    chunk_tokens: list[int],
    chunk_weights: list[float],
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    if bos is None or eos is None:
        raise ValueError("Tokenizer is missing BOS/EOS token IDs.")

    # Pad with EOS to match CLIP max length (A1111-style).
    chunk_tokens = chunk_tokens[: max(0, max_length - 2)]
    input_ids = [bos] + chunk_tokens
    input_ids.extend([eos] * (max_length - len(input_ids)))
    input_ids_tensor = torch.tensor([input_ids])

    if not chunk_weights:
        return input_ids_tensor, None

    chunk_weights = chunk_weights[: max(0, max_length - 2)]
    weight_values = [1.0] + chunk_weights
    weight_values.extend([1.0] * (max_length - len(weight_values)))
    weight_tensor = torch.tensor(weight_values)
    return input_ids_tensor, weight_tensor


def _chunk_tokens_and_weights(
    tokens: list[int], weights: list[float], chunk_size: int
) -> list[tuple[list[int], list[float]]]:
    chunks: list[tuple[list[int], list[float]]] = []
    for start in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[start : start + chunk_size]
        chunk_weights = weights[start : start + chunk_size]
        chunks.append((chunk_tokens, chunk_weights))
    if not chunks:
        chunks = [([], [])]
    return chunks


def _encode_tokens_diffusers_like(
    *,
    text_encoder,
    tokenizer,
    tokens: list[int],
    weights: list[float],
    device: torch.device,
    chunk_size: int,
    clip_skip: int | None,
    normalize_weights: bool,
) -> torch.Tensor:
    if not tokens:
        tokens = []
        weights = []

    max_length = tokenizer.model_max_length
    chunks = _chunk_tokens_and_weights(tokens, weights, chunk_size)
    weighted_chunks: list[torch.Tensor] = []
    total_count = 0
    before_sumsq: torch.Tensor | None = None
    after_sumsq: torch.Tensor | None = None
    for chunk_tokens, chunk_weights in chunks:
        input_ids_tensor, weight_tensor = _build_padded_chunk_input(
            tokenizer=tokenizer,
            chunk_tokens=chunk_tokens,
            chunk_weights=chunk_weights,
            max_length=max_length,
        )
        input_ids_tensor = input_ids_tensor.to(device=device)

        use_mask = bool(getattr(getattr(text_encoder, "config", None), "use_attention_mask", False))
        attention_mask = torch.ones_like(input_ids_tensor) if use_mask else None

        chunk_embeds = _encode_text_encoder(
            text_encoder=text_encoder,
            input_ids=input_ids_tensor,
            attention_mask=attention_mask,
            clip_skip=clip_skip,
        )
        weighted = chunk_embeds

        if weight_tensor is not None:
            weight_tensor = weight_tensor.to(device=device, dtype=chunk_embeds.dtype)
            weighted = chunk_embeds * weight_tensor.unsqueeze(-1)

        weighted_chunks.append(weighted)
        if normalize_weights:
            if before_sumsq is None or after_sumsq is None:
                before_sumsq = torch.zeros((chunk_embeds.shape[0],), device=device, dtype=torch.float32)
                after_sumsq = torch.zeros_like(before_sumsq)
            before_sumsq += chunk_embeds.float().pow(2).sum(dim=(1, 2))
            after_sumsq += weighted.float().pow(2).sum(dim=(1, 2))
            total_count += chunk_embeds.shape[1] * chunk_embeds.shape[2]

    weighted_all = torch.cat(weighted_chunks, dim=1)
    if normalize_weights:
        if before_sumsq is not None and after_sumsq is not None and total_count > 0:
            original_rms = (before_sumsq / float(total_count)).sqrt().to(weighted_all.dtype)
            new_rms = (after_sumsq / float(total_count)).sqrt().to(weighted_all.dtype)
            weighted_all = weighted_all * (
                original_rms / new_rms.clamp(min=NORMALIZATION_EPS)
            ).unsqueeze(-1).unsqueeze(-1)

    return weighted_all


def _encode_tokens_a1111_like(
    *,
    text_encoder,
    tokenizer,
    tokens: list[int],
    weights: list[float],
    device: torch.device,
    chunk_size: int,
    clip_skip: int | None,
    normalize_weights: bool,
) -> torch.Tensor:
    if not tokens:
        tokens = []
        weights = []

    max_length = tokenizer.model_max_length
    chunks = _chunk_tokens_and_weights(tokens, weights, chunk_size)
    weighted_chunks: list[torch.Tensor] = []
    for chunk_tokens, chunk_weights in chunks:
        input_ids_tensor, weight_tensor = _build_padded_chunk_input(
            tokenizer=tokenizer,
            chunk_tokens=chunk_tokens,
            chunk_weights=chunk_weights,
            max_length=max_length,
        )
        input_ids_tensor = input_ids_tensor.to(device=device)

        use_mask = bool(getattr(getattr(text_encoder, "config", None), "use_attention_mask", False))
        attention_mask = torch.ones_like(input_ids_tensor) if use_mask else None

        chunk_embeds = _encode_text_encoder(
            text_encoder=text_encoder,
            input_ids=input_ids_tensor,
            attention_mask=attention_mask,
            clip_skip=clip_skip,
        )
        weighted = chunk_embeds

        if weight_tensor is not None:
            weight_tensor = weight_tensor.to(device=device, dtype=chunk_embeds.dtype)
            weighted = chunk_embeds * weight_tensor.unsqueeze(-1)

            if normalize_weights:
                original_rms = chunk_embeds.float().pow(2).mean(dim=(1, 2)).sqrt().to(chunk_embeds.dtype)
                new_rms = weighted.float().pow(2).mean(dim=(1, 2)).sqrt().to(chunk_embeds.dtype)
                weighted = weighted * (
                    original_rms / new_rms.clamp(min=NORMALIZATION_EPS)
                ).unsqueeze(-1).unsqueeze(-1)

        weighted_chunks.append(weighted)

    return torch.cat(weighted_chunks, dim=1)


def _encode_tokens_comfyui_like(
    *,
    text_encoder,
    tokenizer,
    tokens: list[int],
    weights: list[float],
    device: torch.device,
    chunk_size: int,
    clip_skip: int | None,
) -> torch.Tensor:
    if not tokens:
        tokens = []
        weights = []

    max_length = tokenizer.model_max_length
    chunks = _chunk_tokens_and_weights(tokens, weights, chunk_size)
    weighted_chunks: list[torch.Tensor] = []
    empty_embeds_cache: dict[int, torch.Tensor] = {}
    for chunk_tokens, chunk_weights in chunks:
        input_ids_tensor, weight_tensor = _build_padded_chunk_input(
            tokenizer=tokenizer,
            chunk_tokens=chunk_tokens,
            chunk_weights=chunk_weights,
            max_length=max_length,
        )
        input_ids_tensor = input_ids_tensor.to(device=device)

        use_mask = bool(getattr(getattr(text_encoder, "config", None), "use_attention_mask", False))
        attention_mask = torch.ones_like(input_ids_tensor) if use_mask else None

        chunk_embeds = _encode_text_encoder(
            text_encoder=text_encoder,
            input_ids=input_ids_tensor,
            attention_mask=attention_mask,
            clip_skip=clip_skip,
        )
        weighted = chunk_embeds

        if weight_tensor is not None:
            weight_tensor = weight_tensor.to(device=device, dtype=chunk_embeds.dtype)
            seq_len = input_ids_tensor.shape[1]
            empty_embeds = empty_embeds_cache.get(seq_len)
            if empty_embeds is None:
                empty_ids_tensor = input_ids_tensor.clone()
                empty_ids_tensor[:, 1:] = tokenizer.eos_token_id
                empty_embeds = _encode_text_encoder(
                    text_encoder=text_encoder,
                    input_ids=empty_ids_tensor,
                    attention_mask=attention_mask,
                    clip_skip=clip_skip,
                )
                empty_embeds_cache[seq_len] = empty_embeds
            weighted = (chunk_embeds - empty_embeds) * weight_tensor.unsqueeze(-1) + empty_embeds

        weighted_chunks.append(weighted)

    return torch.cat(weighted_chunks, dim=1)


def _coerce_weighting_policy(weighting_policy: WeightingPolicy | str) -> WeightingPolicy:
    if isinstance(weighting_policy, WeightingPolicy):
        if weighting_policy is WeightingPolicy.MULTIPLY:
            return WeightingPolicy.A1111_LIKE
        if weighting_policy is WeightingPolicy.DELTA_FROM_EMPTY:
            return WeightingPolicy.COMFYUI_LIKE
        return weighting_policy
    value = str(weighting_policy).strip().lower()
    aliases = {
        "diffusers": WeightingPolicy.DIFFUSERS_LIKE,
        "diffusers_like": WeightingPolicy.DIFFUSERS_LIKE,
        "diffusers-like": WeightingPolicy.DIFFUSERS_LIKE,
        "a1111": WeightingPolicy.A1111_LIKE,
        "a1111_like": WeightingPolicy.A1111_LIKE,
        "a1111-like": WeightingPolicy.A1111_LIKE,
        "comfy": WeightingPolicy.COMFYUI_LIKE,
        "comfyui": WeightingPolicy.COMFYUI_LIKE,
        "comfyui_like": WeightingPolicy.COMFYUI_LIKE,
        "comfyui-like": WeightingPolicy.COMFYUI_LIKE,
        "multiply": WeightingPolicy.A1111_LIKE,
        "delta_from_empty": WeightingPolicy.COMFYUI_LIKE,
    }
    if value in aliases:
        return aliases[value]
    try:
        coerced = WeightingPolicy(value)
        if coerced is WeightingPolicy.MULTIPLY:
            return WeightingPolicy.A1111_LIKE
        if coerced is WeightingPolicy.DELTA_FROM_EMPTY:
            return WeightingPolicy.COMFYUI_LIKE
        return coerced
    except ValueError as exc:
        raise ValueError(f"Unknown weighting_policy: {weighting_policy!r}") from exc


def _parse_prompt_attention_a1111(text: str, enable_break: bool) -> list[tuple[str, float]]:
    res: list[list[object]] = []
    round_brackets: list[int] = []
    square_brackets: list[int] = []

    round_bracket_multiplier = DEFAULT_EMPHASIS
    square_bracket_multiplier = 1 / DEFAULT_EMPHASIS

    def multiply_range(start_position: int, multiplier: float) -> None:
        for p in range(start_position, len(res)):
            res[p][1] = float(res[p][1]) * multiplier

    for match in _A1111_RE_ATTENTION.finditer(text):
        token = match.group(0)
        weight = match.group(1)

        if token.startswith("\\"):
            res.append([token[1:], 1.0])
        elif token == "(":
            round_brackets.append(len(res))
        elif token == "[":
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif token == ")" and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif token == "]" and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            if enable_break:
                parts = re.split(_A1111_RE_BREAK, token)
                for idx, part in enumerate(parts):
                    if idx > 0:
                        res.append(["BREAK", -1.0])
                    res.append([part, 1.0])
            else:
                res.append([token, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)
    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if not res:
        return [("", 1.0)]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] = f"{res[i][0]}{res[i + 1][0]}"
            res.pop(i + 1)
        else:
            i += 1

    merged: list[tuple[str, float]] = []
    for text_part, weight_part in res:
        merged.append((str(text_part), float(weight_part)))
    return merged


def _parse_prompt_attention_comfyui(text: str, base_weight: float) -> list[tuple[str, float]]:
    # ComfyUI's tokenizer primarily supports parentheses emphasis; treat square brackets as literal.
    text = text.replace("\\)", "\0\1").replace("\\(", "\0\2")

    def parse_parentheses(string: str) -> list[str]:
        result: list[str] = []
        current = ""
        nesting = 0
        for char in string:
            if char == "(":
                if nesting == 0:
                    if current:
                        result.append(current)
                    current = "("
                else:
                    current += char
                nesting += 1
            elif char == ")":
                nesting -= 1
                if nesting == 0:
                    result.append(current + ")")
                    current = ""
                else:
                    current += char
            else:
                current += char
        if current:
            result.append(current)
        return result

    def token_weights(string: str, current_weight: float) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        for part in parse_parentheses(string):
            weight = current_weight
            if len(part) >= 2 and part[0] == "(" and part[-1] == ")":
                inner = part[1:-1]
                colon = inner.rfind(":")
                weight *= DEFAULT_EMPHASIS
                if colon > 0:
                    try:
                        weight = float(inner[colon + 1 :])
                        inner = inner[:colon]
                    except ValueError:
                        pass
                out.extend(token_weights(inner, weight))
            else:
                out.append((part, current_weight))
        return out

    segments = [(t.replace("\0\1", ")").replace("\0\2", "("), w) for t, w in token_weights(text, base_weight)]
    return _merge_segments(segments)



def _merge_segments(segments: list[tuple[str, float]]) -> list[tuple[str, float]]:
    merged: list[tuple[str, float]] = []
    for text, weight in segments:
        if not text:
            continue
        if merged and merged[-1][1] == weight:
            merged[-1] = (merged[-1][0] + text, weight)
        else:
            merged.append((text, weight))
    return merged


def _pad_embeddings(embeds: torch.Tensor, target_length: int) -> torch.Tensor:
    current_length = embeds.shape[1]
    if current_length >= target_length:
        return embeds
    pad_length = target_length - current_length
    pad_token = embeds[:, -1:, :]
    pad = pad_token.repeat(1, pad_length, 1)
    return torch.cat([embeds, pad], dim=1)
