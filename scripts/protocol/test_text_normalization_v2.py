#!/usr/bin/env python3
"""Regression tests for normalization V2."""

from __future__ import annotations

from text_normalization_v2 import normalize_text_v2


def main() -> None:
    cases = {
        "爱你哟～": "爱你哟~",
        "我最好 我最美 我值得": "我最好 我最美 我值得",
        "  Не\n\tболды!  ": "Не болды!",
        "سېنى ياخشى كۆرىمەن～": "سېنى ياخشى كۆرىمەن~",
        "Сені жақсы көремін～": "Сені жақсы көремін~",
        "e\u0301": "é",
        "你好，世界！": "你好，世界！",
    }
    for source, expected in cases.items():
        actual = normalize_text_v2(source)
        assert actual == expected, (source, actual, expected)
        assert normalize_text_v2(actual) == actual
    try:
        normalize_text_v2("ئۇيغۇر\u200Dچە")
    except ValueError as error:
        assert "U+200D" in str(error)
    else:
        raise AssertionError("Zero-width joiner must not be silently removed")
    print("TEXT_NORMALIZATION_V2_TESTS_OK")


if __name__ == "__main__":
    main()
