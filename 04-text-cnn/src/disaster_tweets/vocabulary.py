"""Vocabulary construction and fixed-length token encoding."""

from __future__ import annotations

from collections import Counter

from .text import tokenize

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def build_vocabulary(
    texts: list[str],
    max_size: int = 20_000,
    min_frequency: int = 2,
) -> dict[str, int]:
    """Build a deterministic vocabulary using training texts only."""
    if max_size < 2:
        raise ValueError("max_size must be at least 2 to include special tokens.")
    if min_frequency < 1:
        raise ValueError("min_frequency must be positive.")

    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))

    candidates = [
        token
        for token, count in counts.items()
        if count >= min_frequency and token not in {PAD_TOKEN, UNK_TOKEN}
    ]
    candidates.sort(key=lambda token: (-counts[token], token))
    candidates = candidates[: max_size - 2]

    return {PAD_TOKEN: 0, UNK_TOKEN: 1, **{token: i + 2 for i, token in enumerate(candidates)}}


def encode_text(text: str, vocabulary: dict[str, int], max_length: int = 64) -> list[int]:
    """Convert text to token IDs, truncating and padding to max_length."""
    if max_length < 1:
        raise ValueError("max_length must be positive.")

    unknown_id = vocabulary[UNK_TOKEN]
    padding_id = vocabulary[PAD_TOKEN]
    token_ids = [vocabulary.get(token, unknown_id) for token in tokenize(text)[:max_length]]
    return token_ids + [padding_id] * (max_length - len(token_ids))
