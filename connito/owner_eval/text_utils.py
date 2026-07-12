"""Pure-stdlib text helpers for answer extraction.

Kept dependency-free (no torch/transformers/datasets) so the parsing logic is
unit-testable in a minimal environment and reusable across metric evaluators.
"""

from __future__ import annotations

import re

# Matches an integer or decimal, optionally signed, with thousands separators
# (commas). Examples: "42", "-3.5", "1,234", "$1,234.50" -> "1,234.50".
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_final_number(text: str | None) -> float | None:
    """Return the last numeric value mentioned in ``text`` as a float.

    GSM8K answers put the gold value last (after ``####``) and models trained to
    "show their work" emit the final answer last too, so taking the *last* match
    is the standard convention. Returns ``None`` when no number is present.
    """
    if not text:
        return None
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    cleaned = matches[-1].replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_gsm8k_gold(answer: str) -> float | None:
    """Extract the gold numeric answer from a GSM8K ``answer`` field.

    GSM8K marks the final answer with ``#### <value>``; fall back to the last
    number in the string if the marker is absent.
    """
    if answer is None:
        return None
    tail = answer.split("####")[-1]
    return extract_final_number(tail)


def numeric_eq(a: float | None, b: float | None, rel_tol: float = 1e-4, abs_tol: float = 1e-6) -> bool:
    """Tolerant numeric equality for comparing an extracted prediction to gold."""
    if a is None or b is None:
        return False
    diff = abs(a - b)
    return diff <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
