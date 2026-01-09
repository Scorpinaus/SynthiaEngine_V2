from __future__ import annotations

from dataclasses import dataclass

import torch


DEFAULT_EMPHASIS = 1.1
DEFAULT_DEEMPHASIS = 0.9
NORMALIZE_PROMPT_WEIGHTS = True
NORMALIZATION_EPS = 1e-6


@dataclass
class ParsedPrompt:
    tokens: list[int]
    weights: list[float]
    has_weights: bool


def parse_prompt_attention(text: str, base_weight: float = 1.0) -> list[tuple[str, float]]:
    segments: list[tuple[str, float]] = []
    buffer: list[str] = []
    idx = 0

    def flush_buffer(weight: float) -> None:
        if buffer:
            segment = "".join(buffer)
            segments.append((segment, weight))
            buffer.clear()

    while idx < len(text):
        char = text[idx]
        if char == "\\" and idx + 1 < len(text):
            buffer.append(text[idx + 1])
            idx += 2
            continue

        if char in "([":
            flush_buffer(base_weight)
            close_char = ")" if char == "(" else "]"
            end_idx = _find_matching_bracket(text, idx, char, close_char)
            if end_idx is None:
                buffer.append(char)
                idx += 1
                continue

            content = text[idx + 1 : end_idx]
            explicit = _extract_explicit_weight(content)
            if explicit is None:
                multiplier = DEFAULT_EMPHASIS if char == "(" else DEFAULT_DEEMPHASIS
                inner_text = content
            else:
                inner_text, multiplier = explicit

            segments.extend(parse_prompt_attention(inner_text, base_weight * multiplier))
            idx = end_idx + 1
            continue

        buffer.append(char)
        idx += 1

    flush_buffer(base_weight)
    return _merge_segments(segments)


def tokenize_prompt(tokenizer, prompt: str) -> ParsedPrompt:
    segments = parse_prompt_attention(prompt)
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
    normalize_weights: bool = NORMALIZE_PROMPT_WEIGHTS,
) -> tuple[torch.Tensor | None, torch.Tensor | None, bool]:
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    device = text_encoder.device

    prompt_tokens = tokenize_prompt(tokenizer, prompt)
    negative_tokens = tokenize_prompt(tokenizer, negative_prompt or "")

    max_chunk = max(1, tokenizer.model_max_length - 2)
    needs_embeddings = (
        prompt_tokens.has_weights
        or negative_tokens.has_weights
        or len(prompt_tokens.tokens) > max_chunk
        or len(negative_tokens.tokens) > max_chunk
    )

    if not needs_embeddings:
        return None, None, False

    prompt_embeds = _encode_tokens(
        text_encoder,
        tokenizer,
        prompt_tokens.tokens,
        prompt_tokens.weights,
        device,
        max_chunk,
        normalize_weights,
    )
    negative_prompt_embeds = _encode_tokens(
        text_encoder,
        tokenizer,
        negative_tokens.tokens,
        negative_tokens.weights,
        device,
        max_chunk,
        normalize_weights,
    )
    return prompt_embeds, negative_prompt_embeds, True


def _encode_tokens(
    text_encoder,
    tokenizer,
    tokens: list[int],
    weights: list[float],
    device: torch.device,
    chunk_size: int,
    normalize_weights: bool,
) -> torch.Tensor:
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    if bos is None or eos is None:
        raise ValueError("Tokenizer is missing BOS/EOS token IDs.")

    if not tokens:
        tokens = []
        weights = []

    chunks: list[tuple[list[int], list[float]]] = []
    for start in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[start : start + chunk_size]
        chunk_weights = weights[start : start + chunk_size]
        chunks.append((chunk_tokens, chunk_weights))

    if not chunks:
        chunks = [([], [])]

    embeddings: list[torch.Tensor] = []
    for chunk_tokens, chunk_weights in chunks:
        input_ids = [bos] + chunk_tokens + [eos]
        input_ids_tensor = torch.tensor([input_ids], device=device)
        encoder_output = text_encoder(input_ids_tensor)
        chunk_embeds = encoder_output[0]

        if chunk_weights:
            weight_values = [1.0] + chunk_weights + [1.0]
            weight_tensor = torch.tensor(weight_values, device=device, dtype=chunk_embeds.dtype)
            chunk_embeds = chunk_embeds * weight_tensor.unsqueeze(-1)
            if normalize_weights:
                mean_weight = weight_tensor.mean().clamp(min=NORMALIZATION_EPS)
                chunk_embeds = chunk_embeds / mean_weight

        embeddings.append(chunk_embeds)

    return torch.cat(embeddings, dim=1)


def _find_matching_bracket(text: str, start_idx: int, open_char: str, close_char: str) -> int | None:
    stack = [open_char]
    idx = start_idx + 1
    while idx < len(text):
        char = text[idx]
        if char == "\\" and idx + 1 < len(text):
            idx += 2
            continue
        if char in "([":
            stack.append(char)
        elif char in ")]":
            if not stack:
                return None
            last = stack.pop()
            if not stack and char == close_char and last == open_char:
                return idx
        idx += 1
    return None


def _extract_explicit_weight(text: str) -> tuple[str, float] | None:
    depth = 0
    colon_index: int | None = None
    idx = 0
    while idx < len(text):
        char = text[idx]
        if char == "\\" and idx + 1 < len(text):
            idx += 2
            continue
        if char in "([":
            depth += 1
        elif char in ")]":
            depth = max(0, depth - 1)
        elif char == ":" and depth == 0:
            colon_index = idx
        idx += 1

    if colon_index is None:
        return None

    weight_text = text[colon_index + 1 :].strip()
    inner_text = text[:colon_index]
    if not inner_text:
        return None
    try:
        weight_value = float(weight_text)
    except ValueError:
        return None
    return inner_text, weight_value


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
