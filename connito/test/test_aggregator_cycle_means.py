"""Tests for `MinerSeries.cycle_means` and `MinerScoreAggregator.scores_over_window`.

These are the helpers the round-group election ballot uses to rank
miners over the previous cohort's 8-cycle window.
"""

from __future__ import annotations

from datetime import datetime, timezone

from connito.validator.aggregator import MinerScoreAggregator, MinerSeries


def _ts(seconds: int) -> datetime:
    """Deterministic UTC timestamps; cycle_means doesn't actually use them."""
    return datetime(2026, 1, 1, tzinfo=timezone.utc).replace(second=seconds)


def test_cycle_means_zero_fills_missing_cycles():
    """A miner with points in 2 of 8 cycles → mean = (s1+s2) / 8, not / 2."""
    series = MinerSeries(max_points=8, max_history_points=64)
    series.add(_ts(1), 4.0, round_id=100)   # cycle 1
    series.add(_ts(2), 8.0, round_id=200)   # cycle 2
    # Other cycles in the window get sentinel zero.
    round_id_to_cycle_index = {100: 1, 200: 2}
    cycles = list(range(0, 8))
    mean, min_per_cycle = series.cycle_means(round_id_to_cycle_index, cycles)
    # mean = (0 + 4 + 8 + 0 + 0 + 0 + 0 + 0) / 8 = 12/8 = 1.5
    assert mean == 1.5
    assert min_per_cycle == 0.0


def test_cycle_means_averages_within_a_cycle():
    """Multiple points tagged with the same cycle are averaged together."""
    series = MinerSeries(max_points=8, max_history_points=64)
    series.add(_ts(1), 2.0, round_id=100)
    series.add(_ts(2), 6.0, round_id=101)   # same cycle 1 (different round_id within cycle is unusual but legal)
    round_id_to_cycle_index = {100: 1, 101: 1}
    mean, _ = series.cycle_means(round_id_to_cycle_index, [1])
    assert mean == 4.0   # (2 + 6) / 2 inside cycle 1


def test_cycle_means_ignores_points_outside_window():
    series = MinerSeries(max_points=8, max_history_points=64)
    series.add(_ts(1), 100.0, round_id=999)   # cycle 99 — way outside window
    round_id_to_cycle_index = {999: 99}
    mean, min_per_cycle = series.cycle_means(round_id_to_cycle_index, [0, 1, 2])
    assert mean == 0.0
    assert min_per_cycle == 0.0


def test_cycle_means_returns_min_for_tiebreak():
    """Spec item 20: tiebreak uses min cycle score."""
    series = MinerSeries(max_points=8, max_history_points=64)
    series.add(_ts(1), 0.5, round_id=10)   # cycle 0
    series.add(_ts(2), 0.5, round_id=11)   # cycle 1
    series.add(_ts(3), 0.5, round_id=12)   # cycle 2
    round_id_to_cycle_index = {10: 0, 11: 1, 12: 2}
    mean, min_per_cycle = series.cycle_means(round_id_to_cycle_index, [0, 1, 2])
    assert mean == 0.5
    assert min_per_cycle == 0.5

    # Now add a fourth cycle with score 0.1 — min should drop.
    series.add(_ts(4), 0.1, round_id=13)
    round_id_to_cycle_index[13] = 3
    mean2, min_per_cycle2 = series.cycle_means(round_id_to_cycle_index, [0, 1, 2, 3])
    assert mean2 == (0.5 + 0.5 + 0.5 + 0.1) / 4
    assert min_per_cycle2 == 0.1


def test_cycle_means_skips_points_with_none_round_id():
    """Legacy points with round_id=None (pre-v2 schema) cannot be mapped to a cycle."""
    series = MinerSeries(max_points=8, max_history_points=64)
    series.add(_ts(1), 99.0, round_id=None)
    series.add(_ts(2), 4.0, round_id=100)
    round_id_to_cycle_index = {100: 0}
    mean, _ = series.cycle_means(round_id_to_cycle_index, [0, 1])
    # Only the round_id=100 point counts; cycle 0 mean = 4, cycle 1 mean = 0.
    assert mean == 2.0


def test_scores_over_window_returns_per_uid_tuples():
    agg = MinerScoreAggregator(max_points=8, max_history_points=64)
    agg.add_score(uid=1, hotkey="hk1", score=2.0, ts=_ts(1), round_id=100)
    agg.add_score(uid=2, hotkey="hk2", score=4.0, ts=_ts(2), round_id=100)
    round_id_to_cycle_index = {100: 0}
    out = agg.scores_over_window(
        uids=[1, 2, 99],   # 99 is unknown — should return (0, 0)
        round_id_to_cycle_index=round_id_to_cycle_index,
        cycles_in_window=[0],
    )
    assert out[1] == (2.0, 2.0)
    assert out[2] == (4.0, 4.0)
    assert out[99] == (0.0, 0.0)


def test_all_round_ids_collects_distinct_round_ids():
    agg = MinerScoreAggregator(max_points=8, max_history_points=64)
    agg.add_score(uid=1, hotkey="hk1", score=1.0, ts=_ts(1), round_id=100)
    agg.add_score(uid=2, hotkey="hk2", score=1.0, ts=_ts(2), round_id=200)
    agg.add_score(uid=1, hotkey="hk1", score=1.0, ts=_ts(3), round_id=200)   # dup round_id
    agg.add_score(uid=3, hotkey="hk3", score=1.0, ts=_ts(4), round_id=None)  # legacy
    assert agg.all_round_ids() == {100, 200}
