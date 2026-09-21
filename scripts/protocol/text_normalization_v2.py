#!/usr/bin/env python3
"""Frozen text normalization for Clean Dev V2 and all model predictions."""

from __future__ import annotations

import re
import unicodedata


PROTOCOL_ID = "normalization_v2"
FULLWIDTH_TILDE = "\uFF5E"
ASCII_TILDE = "~"
DISALLOWED_CONTROLS = frozenset(
    {
        "\u200B",  # zero width space
        "\u200C",  # zero width non-joiner
        "\u200D",  # zero width joiner
        "\u200E",  # left-to-right mark
        "\u200F",  # right-to-left mark
        "\u202A",
        "\u202B",
        "\u202C",
        "\u202D",
        "\u202E",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
    }
)


def _canonical_whitespace(text: str) -> str:
    characters = []
    for character in text:
        if character.isspace() or unicodedata.category(character) == "Zs":
            characters.append(" ")
        else:
            characters.append(character)
    return re.sub(r" +", " ", "".join(characters)).strip(" ")


def normalize_text_v2(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    controls = sorted({character for character in text if character in DISALLOWED_CONTROLS})
    if controls:
        codepoints = ", ".join(f"U+{ord(character):04X}" for character in controls)
        raise ValueError(f"Disallowed zero-width or bidi controls: {codepoints}")
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace(FULLWIDTH_TILDE, ASCII_TILDE)
    normalized = _canonical_whitespace(normalized)
    return unicodedata.normalize("NFC", normalized)


__all__ = ["PROTOCOL_ID", "normalize_text_v2"]
