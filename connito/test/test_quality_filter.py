"""Unit tests for the opt-in data quality filter.

Exercises the heuristic `is_low_quality` predicate added to
`connito/shared/dataloader.py` and the matching `DataQualityCfg`
section under `connito/shared/config.py:DataCfg`.

The filter is opt-in (default `enabled=False`); these tests focus on:

  * The thresholds (min_length, repetition, junk) each fire on
    representative inputs.
  * Clean prose is *not* dropped.
  * Config defaults match the spec (200 / 0.5 / 0.2, disabled).
  * The adapter `_quality_filter_keep` is a true no-op when called
    with permissive thresholds — i.e. the filter never drops a good
    sample by accident.

No HuggingFace network access — these are pure-Python checks.
"""
from __future__ import annotations

import pytest

from connito.shared.config import DataCfg, DataQualityCfg
from connito.shared.dataloader import _quality_filter_keep, is_low_quality


# ---------------------------------------------------------------------------
# is_low_quality — per-rule firing
# ---------------------------------------------------------------------------
class TestIsLowQualityMinLength:
    def test_short_text_is_dropped(self) -> None:
        assert is_low_quality("short") is True

    def test_empty_text_is_dropped(self) -> None:
        assert is_low_quality("") is True

    def test_exactly_min_length_is_kept(self) -> None:
        # `"a" * 200` is 200 chars: meets min_length, low repetition
        # (only one word so the >=10-words branch is skipped), no junk.
        assert is_low_quality("a" * 200) is False

    def test_just_below_min_length_is_dropped(self) -> None:
        assert is_low_quality("a" * 199) is True

    def test_custom_min_length_threshold(self) -> None:
        text = "a" * 50
        assert is_low_quality(text, min_length=100) is True
        assert is_low_quality(text, min_length=10) is False


class TestIsLowQualityRepetition:
    def test_high_repetition_is_dropped(self) -> None:
        # 100 identical words → unique/total ≈ 0.01 → repetition ≈ 0.99.
        assert is_low_quality("hello " * 100) is True

    def test_normal_prose_is_kept(self) -> None:
        # "normal sentence " repeated has only two unique words but >=10
        # words total, so repetition_ratio = 1 - 2/(2*30) ≈ 0.967 → dropped.
        # The spec example uses a more varied sentence; use that here.
        text = (
            "The quick brown fox jumps over the lazy dog while a bright "
            "moon illuminates the silent meadow beyond the rolling hills "
            "and distant forest where ancient trees stand watch over the "
            "sleeping world below in peaceful slumber until morning."
        )
        assert is_low_quality(text) is False

    def test_repetition_disabled_below_ten_words(self) -> None:
        # 9 words, all identical — but the repetition branch is skipped
        # because the sample has < 10 words. The text is long enough to
        # pass min_length and has no junk, so it must be kept.
        text = ("xxxxxxxxxxxxxxxxxxxxxxxxxxxxx " * 9).strip()
        assert len(text.split()) == 9
        assert len(text) >= 200
        assert is_low_quality(text) is False

    def test_custom_repetition_threshold(self) -> None:
        # Construct a long sample with high repetition: 100 words drawn
        # from a 5-word vocabulary → 95% repetition. Pad to clear the
        # min_length floor so only the repetition rule decides.
        words = ["alpha", "bravo", "charlie", "delta", "echo"] * 20
        text = " ".join(words)
        assert len(text.split()) == 100
        assert len(set(text.split())) == 5
        # 100 words, 5 unique → repetition_ratio = 1 - 5/100 = 0.95
        assert is_low_quality(text) is True
        # With a permissive threshold the same sample passes repetition,
        # so the overall verdict flips to "keep" (no junk, long enough).
        assert is_low_quality(text, max_repetition_ratio=0.99) is False


class TestIsLowQualityJunk:
    def test_high_junk_is_dropped(self) -> None:
        # NUL / SOH / STX bytes are outside `string.printable + " \t\n"`.
        assert is_low_quality("\x00\x01\x02" * 100) is True

    def test_clean_text_is_kept(self) -> None:
        # Real-paragraph stand-in: ASCII prose with newlines and tabs.
        paragraph = (
            "Large language models trained with continuous pre-training "
            "benefit from balanced data mixtures across domains. When "
            "the corpus mixes web crawl and math-heavy sources, miners "
            "should ideally see clean, informative samples — boilerplate "
            "and broken encoding waste compute and slow convergence."
        )
        assert is_low_quality(paragraph) is False

    def test_custom_junk_threshold(self) -> None:
        # Sample is ~50% junk — passes default 0.2 threshold? No: drops.
        text = ("a" * 100) + ("\x00" * 100)
        assert is_low_quality(text) is True
        # With a very permissive junk threshold (0.99), keep it.
        assert is_low_quality(text, max_junk_ratio=0.99) is False


# ---------------------------------------------------------------------------
# _quality_filter_keep — HF `filter()` adapter
# ---------------------------------------------------------------------------
class TestQualityFilterKeepAdapter:
    def test_keep_returns_true_for_clean_sample(self) -> None:
        example = {"text": "a" * 500}
        assert (
            _quality_filter_keep(
                example,
                min_length=200,
                max_repetition_ratio=0.5,
                max_junk_ratio=0.2,
            )
            is True
        )

    def test_keep_returns_false_for_short_sample(self) -> None:
        example = {"text": "tiny"}
        assert (
            _quality_filter_keep(
                example,
                min_length=200,
                max_repetition_ratio=0.5,
                max_junk_ratio=0.2,
            )
            is False
        )

    def test_keep_handles_missing_text_key(self) -> None:
        # HF rows without a `text` column should be treated as empty
        # (length 0) and therefore DROPPED — never raise.
        assert (
            _quality_filter_keep(
                {},
                min_length=200,
                max_repetition_ratio=0.5,
                max_junk_ratio=0.2,
            )
            is False
        )


# ---------------------------------------------------------------------------
# DataQualityCfg — defaults & validation
# ---------------------------------------------------------------------------
class TestDataQualityCfg:
    def test_defaults_are_safe_opt_in(self) -> None:
        cfg = DataQualityCfg()
        assert cfg.enabled is False
        assert cfg.min_length == 200
        assert cfg.max_repetition_ratio == 0.5
        assert cfg.max_junk_ratio == 0.2

    def test_data_cfg_attaches_quality_filter_by_default(self) -> None:
        cfg = DataCfg()
        assert isinstance(cfg.quality_filter, DataQualityCfg)
        # Crucially, the default is OFF — so adding this field can't
        # accidentally change the training data pipeline for operators
        # who don't opt in.
        assert cfg.quality_filter.enabled is False

    def test_ratio_field_bounds(self) -> None:
        # max_repetition_ratio is bounded to [0, 1]; values outside
        # the range raise.
        with pytest.raises(ValueError):
            DataQualityCfg(max_repetition_ratio=1.5)
        with pytest.raises(ValueError):
            DataQualityCfg(max_junk_ratio=-0.1)

    def test_min_length_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            DataQualityCfg(min_length=0)

    def test_enabling_filter_overrides_default(self) -> None:
        cfg = DataQualityCfg(enabled=True, min_length=500)
        assert cfg.enabled is True
        assert cfg.min_length == 500
        # Other defaults remain in place.
        assert cfg.max_repetition_ratio == 0.5
        assert cfg.max_junk_ratio == 0.2


# ---------------------------------------------------------------------------
# Real paragraph sanity check — ensure the heuristic does NOT trip on
# normal text that miners actually want to train on.
# ---------------------------------------------------------------------------
def test_real_paragraph_is_kept() -> None:
    paragraph = (
        "Bittensor is a peer-to-peer protocol that incentivizes participants "
        "to contribute computational resources, model weights, and validation "
        "work to a shared machine intelligence network. Subnet 102 focuses on "
        "distributed mixture-of-experts pretraining, where miners train shards "
        "of a model and validators score the resulting weights against a held-"
        "out evaluation pool. Miners earn TAO rewards proportional to how much "
        "their submitted shards reduce validation loss relative to a baseline."
    )
    assert is_low_quality(paragraph) is False
