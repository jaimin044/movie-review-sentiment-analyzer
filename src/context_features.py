"""Rule-based context features for review text."""

from __future__ import annotations

import re
from typing import Iterable


_TOKEN_PATTERN = re.compile(
    r"NOT_[A-Za-z0-9]+|[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?|[.!?,;:]"
)
_BOUNDARY_TOKENS = {
    "and",
    "although",
    "because",
    "but",
    "however",
    "nor",
    "or",
    "so",
    "though",
    "while",
    "yet",
}
_CONTRASTIVE_TOKENS = {"although", "but", "however", "yet"}
_NEGATION_TOKENS = {
    "cannot",
    "can't",
    "couldn't",
    "didn't",
    "doesn't",
    "don't",
    "hardly",
    "isn't",
    "neither",
    "never",
    "no",
    "nor",
    "not",
    "nothing",
    "shouldn't",
    "wasn't",
    "weren't",
    "without",
    "won't",
    "wouldn't",
}
_PUNCTUATION = {".", "!", "?", ",", ";", ":"}


def _tokenize(text: str) -> list[str]:
    return [
        token if token.startswith("NOT_") else token.lower()
        for token in _TOKEN_PATTERN.findall(text)
    ]


def _render(tokens: Iterable[str]) -> str:
    return " ".join(tokens)


def mark_negation_scope(text: str) -> str:
    """Prefix words in a negation scope with ``NOT_``.

    A scope starts after a negation cue and ends at punctuation or a clause
    boundary. The cue itself is retained without a prefix.
    """

    tokens = _tokenize(text)
    transformed: list[str] = []
    in_scope = False

    for token in tokens:
        if token in _PUNCTUATION:
            in_scope = False
            continue

        if token in _NEGATION_TOKENS:
            in_scope = True
            transformed.append(token)
            continue

        if token in _BOUNDARY_TOKENS:
            in_scope = False
            transformed.append(token)
            continue

        transformed.append(f"NOT_{token}" if in_scope else token)

    return _render(transformed)


def mark_contrastive_clauses(text: str) -> str:
    """Prefix clause tokens with their position around contrastive markers."""

    tokens = _tokenize(text)
    contrast_index = next(
        (index for index, token in enumerate(tokens) if token in _CONTRASTIVE_TOKENS),
        None,
    )
    if contrast_index is None:
        return _render(tokens)

    result: list[str] = []
    for index, token in enumerate(tokens):
        if token in _PUNCTUATION:
            continue
        if token in _CONTRASTIVE_TOKENS:
            result.append(token)
        elif index < contrast_index:
            result.append(f"BEFORE_BUT_{token}")
        else:
            result.append(f"AFTER_BUT_{token}")
    return _render(result)


def apply_context_features(text: str) -> str:
    """Apply negation and contrastive clause features to one review."""

    return mark_contrastive_clauses(mark_negation_scope(text))


def transform_reviews(reviews: Iterable[str]) -> list[str]:
    """Apply context features to an iterable of reviews."""

    return [apply_context_features(review) for review in reviews]
