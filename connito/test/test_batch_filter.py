"""Unit tests for ``connito.miner.batch_filter.LossSpikeFilter``.

Covers:
  - Warmup: the first ``warmup`` batches are always accepted, regardless
    of how anomalous they look.
  - Outlier rejection: after warmup, a loss many sigma above the rolling
    mean is skipped.
  - Normal-range acceptance: small perturbations around the mean are
    accepted (so the filter doesn't accidentally throw away ordinary
    samples that happen to be slightly above mean).
  - Stats accounting: ``stats()`` reflects total_seen / skipped_count /
    skip_rate truthfully.
  - Pathological inputs:
      * sigma == 0 (all-identical losses) must not panic and must not
        flag any sample (any deviation would be infinitely many sigma).
      * Empty window before warmup completes — should_skip must not
        crash and must return False.
  - Constructor validation: invalid (window, warmup, z_threshold) inputs
    raise rather than silently producing a broken filter.

The filter has no external dependencies (no torch, no telemetry), so
these tests are pure Python and run in well under 1 second.
"""

from __future__ import annotations

import random

import pytest

from connito.miner.batch_filter import LossSpikeFilter


def test_warmup_never_skips() -> None:
    """During warmup, even a wildly anomalous batch must pass through."""
    f = LossSpikeFilter(window=200, z_threshold=4.0, warmup=50)
    # 49 ordinary batches + 1 huge "would be outlier" batch — but we're
    # still inside warmup so even this is accepted.
    skipped = [f.should_skip(1.0) for _ in range(49)]
    assert not any(skipped), "no batch should be skipped before warmup completes"
    assert f.should_skip(1000.0) is False, "outlier inside warmup must pass"
    assert f.skipped_count == 0
    assert f.total_seen == 50


def test_outlier_skipped_after_warmup() -> None:
    """A batch many sigma above the rolling mean is rejected post-warmup."""
    rng = random.Random(0xC0FFEE)
    f = LossSpikeFilter(window=200, z_threshold=4.0, warmup=50)

    # Establish a tight baseline: loss ~ N(1.0, 0.1).
    for _ in range(100):
        f.should_skip(1.0 + rng.gauss(0.0, 0.1))

    # A loss at 10.0 is ~90 sigma out — must be flagged.
    assert f.should_skip(10.0) is True
    assert f.skipped_count == 1


def test_normal_range_not_skipped() -> None:
    """Small perturbations around the rolling mean must NOT be skipped."""
    f = LossSpikeFilter(window=200, z_threshold=4.0, warmup=50)
    # Tight baseline at 1.0.
    for _ in range(100):
        f.should_skip(1.0)
    # 1.05 is within reach of pstdev — at this point sigma==0 so the
    # filter takes its "do not skip when sigma collapses" early return.
    # Either way, the answer must be False.
    assert f.should_skip(1.05) is False
    assert f.skipped_count == 0


def test_normal_range_with_variance_not_skipped() -> None:
    """When sigma > 0, samples within a few sigma of mean stay accepted."""
    rng = random.Random(7)
    f = LossSpikeFilter(window=200, z_threshold=4.0, warmup=50)
    # Build a baseline with real variance.
    for _ in range(200):
        f.should_skip(1.0 + rng.gauss(0.0, 0.1))

    # A 1-sigma sample (loss ~= 1.10) should NEVER be flagged at z=4.
    assert f.should_skip(1.10) is False
    # A 2-sigma sample (~1.20) is also far below z=4 threshold.
    assert f.should_skip(1.20) is False
    assert f.skipped_count == 0


def test_skipped_loss_not_added_to_window() -> None:
    """Outliers must not poison the rolling mean."""
    rng = random.Random(42)
    f = LossSpikeFilter(window=50, z_threshold=4.0, warmup=20)

    # Warmup at ~1.0.
    for _ in range(20):
        f.should_skip(1.0 + rng.gauss(0.0, 0.05))

    # Feed several huge outliers — all should be rejected and excluded
    # from the rolling statistics.
    pre_window_snapshot = list(f._losses)
    for _ in range(10):
        assert f.should_skip(100.0) is True

    # The window must contain only the warmup losses (no 100.0 entries).
    assert len(f._losses) == len(pre_window_snapshot)
    assert max(f._losses) < 2.0, "window must not contain any rejected outlier"
    assert f.skipped_count == 10


def test_stats_output_shape_and_values() -> None:
    """``stats()`` should report total_seen, skipped, skip_rate honestly."""
    f = LossSpikeFilter(window=100, z_threshold=4.0, warmup=10)
    # Warmup with tight baseline.
    for _ in range(50):
        f.should_skip(1.0)
    # Try a couple of outliers (will be skipped only when sigma > 0;
    # since all warmup losses == 1.0, sigma == 0 here and they pass).
    f.should_skip(2.0)
    f.should_skip(3.0)

    stats = f.stats()
    assert set(stats.keys()) == {"window_size", "skipped", "total_seen", "skip_rate"}
    assert stats["total_seen"] == 52
    assert stats["window_size"] <= 100
    assert stats["skip_rate"] == stats["skipped"] / stats["total_seen"]


def test_stats_skip_rate_with_zero_total_seen() -> None:
    """``skip_rate`` must not divide-by-zero on a fresh filter."""
    f = LossSpikeFilter()
    stats = f.stats()
    assert stats["total_seen"] == 0
    assert stats["skipped"] == 0
    assert stats["skip_rate"] == 0.0


def test_all_same_loss_sigma_zero_no_skips() -> None:
    """When the entire window is identical, sigma is 0 and no batch can
    be flagged — guarding against a divide-by-zero / inf-sigma bug."""
    f = LossSpikeFilter(window=50, z_threshold=4.0, warmup=10)
    for _ in range(100):
        # Feed exactly the same loss repeatedly. Even a slightly
        # different "outlier" must NOT be skipped because sigma==0.
        assert f.should_skip(2.5) is False
    # And immediately try a deviating value — still no skip while
    # sigma stays at 0.
    assert f.should_skip(2.5001) is False
    assert f.skipped_count == 0


def test_empty_window_does_not_crash() -> None:
    """A fresh filter (empty internal window) must handle should_skip
    without raising and must return False."""
    f = LossSpikeFilter(window=10, z_threshold=4.0, warmup=5)
    assert f.should_skip(1.0) is False
    assert f.total_seen == 1
    assert f.skipped_count == 0


def test_window_rolls_off_old_losses() -> None:
    """When `window` is filled, the oldest sample drops off so the rolling
    statistics track recent batches, not the entire training run."""
    f = LossSpikeFilter(window=5, z_threshold=4.0, warmup=3)
    # Fill with 1.0s up to and past the window.
    for _ in range(10):
        f.should_skip(1.0)
    # Internal deque is bounded by `window`.
    assert len(f._losses) == 5
    assert all(v == 1.0 for v in f._losses)


def test_constructor_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="window"):
        LossSpikeFilter(window=0)
    with pytest.raises(ValueError, match="window"):
        LossSpikeFilter(window=-1)


def test_constructor_rejects_invalid_z_threshold() -> None:
    with pytest.raises(ValueError, match="z_threshold"):
        LossSpikeFilter(z_threshold=0.0)
    with pytest.raises(ValueError, match="z_threshold"):
        LossSpikeFilter(z_threshold=-1.0)


def test_constructor_rejects_invalid_warmup() -> None:
    with pytest.raises(ValueError, match="warmup"):
        LossSpikeFilter(warmup=-1)
    with pytest.raises(ValueError, match="warmup"):
        LossSpikeFilter(window=10, warmup=100)


def test_default_z_threshold_is_conservative() -> None:
    """Sanity-check the documented default of z=4 (≈0.006% Gaussian tail)."""
    f = LossSpikeFilter()
    assert f.z_threshold == 4.0
    assert f.warmup == 50
    assert f.window == 200
