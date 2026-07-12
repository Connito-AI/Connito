"""Pure-logic tests for owner-eval answer extraction (no heavy deps)."""

import pytest

from connito.owner_eval.text_utils import (
    extract_final_number,
    extract_gsm8k_gold,
    numeric_eq,
)


@pytest.mark.parametrize("text,expected", [
    ("The answer is 42.", 42.0),
    ("So we get 1,234 apples", 1234.0),
    ("It costs $1,234.50 total", 1234.50),
    ("first 3 then 7 finally 9", 9.0),
    ("negative result: -15", -15.0),
    ("3.14159", 3.14159),
    ("no numbers here", None),
    ("", None),
    (None, None),
])
def test_extract_final_number(text, expected):
    assert extract_final_number(text) == expected


def test_extract_gsm8k_gold_uses_marker():
    answer = "Janet has 16 eggs ... she makes $18 per day.\n#### 18"
    assert extract_gsm8k_gold(answer) == 18.0


def test_extract_gsm8k_gold_falls_back_without_marker():
    assert extract_gsm8k_gold("the result is 7") == 7.0
    assert extract_gsm8k_gold(None) is None


@pytest.mark.parametrize("a,b,expected", [
    (42.0, 42.0, True),
    (42.0, 42.00001, True),       # within tolerance
    (42.0, 43.0, False),
    (None, 42.0, False),
    (42.0, None, False),
    (None, None, False),
    (0.0, 0.0, True),
])
def test_numeric_eq(a, b, expected):
    assert numeric_eq(a, b) is expected
