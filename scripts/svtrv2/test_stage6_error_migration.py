#!/usr/bin/env python3
"""Regression checks for Stage 6 deterministic paired analysis helpers."""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_stage6_error_migration import (  # noqa: E402
    exact_mcnemar,
    levenshtein_alignment,
    operation_counts,
)


def check(reference: str, hypothesis: str, expected: dict[str, int]) -> None:
    counts = operation_counts(levenshtein_alignment(reference, hypothesis))
    actual = {name: counts[name] for name in ("sub", "del", "ins")}
    if actual != expected:
        raise AssertionError(f"{reference!r} -> {hypothesis!r}: {actual} != {expected}")


def main() -> None:
    check("abc", "abc", {"sub": 0, "del": 0, "ins": 0})
    check("abc", "axc", {"sub": 1, "del": 0, "ins": 0})
    check("abc", "ac", {"sub": 0, "del": 1, "ins": 0})
    check("ac", "abc", {"sub": 0, "del": 0, "ins": 1})
    check("ۇيغۇر", "ۇيغۈر", {"sub": 1, "del": 0, "ins": 0})
    if exact_mcnemar(0, 0) != 1.0:
        raise AssertionError("McNemar empty-discordance case failed")
    if not (0.0 <= exact_mcnemar(12, 3) <= 1.0):
        raise AssertionError("McNemar p-value outside [0,1]")
    print("STAGE6_ERROR_MIGRATION_REGRESSION_OK")


if __name__ == "__main__":
    main()
