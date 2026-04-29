from datetime import datetime, timedelta, timezone

import pytest

from connito.validator.aggregator import MinerScoreAggregator


def _ts(offset_s: int = 0) -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=offset_s)


def test_fresh_uid_score_pairs_drops_stale_miners():
    agg = MinerScoreAggregator(max_points=8)

    agg.add_score(uid=1, hotkey="hk1", score=0.5, ts=_ts(0), round_id=10)
    agg.add_score(uid=2, hotkey="hk2", score=0.7, ts=_ts(1), round_id=11)
    agg.add_score(uid=3, hotkey="hk3", score=0.9, ts=_ts(2), round_id=12)

    fresh = agg.fresh_uid_score_pairs(current_round_id=12, freshness_window=2, how="avg")

    assert set(fresh.keys()) == {1, 2, 3}

    fresh = agg.fresh_uid_score_pairs(current_round_id=13, freshness_window=2, how="avg")
    assert set(fresh.keys()) == {2, 3}

    fresh = agg.fresh_uid_score_pairs(current_round_id=14, freshness_window=1, how="avg")
    assert set(fresh.keys()) == set()


def test_fresh_uid_score_pairs_uses_newest_round_id_per_miner():
    agg = MinerScoreAggregator(max_points=8)

    agg.add_score(uid=1, hotkey="hk1", score=0.1, ts=_ts(0), round_id=5)
    agg.add_score(uid=1, hotkey="hk1", score=0.9, ts=_ts(1), round_id=12)

    fresh = agg.fresh_uid_score_pairs(current_round_id=13, freshness_window=2, how="latest")
    assert fresh == {1: pytest.approx(0.9)}


def test_fresh_uid_score_pairs_drops_legacy_untagged_points():
    agg = MinerScoreAggregator(max_points=8)

    agg.add_score(uid=1, hotkey="hk1", score=0.5, ts=_ts(0), round_id=None)

    fresh = agg.fresh_uid_score_pairs(current_round_id=10, freshness_window=2, how="avg")
    assert fresh == {}


def test_fresh_uid_score_pairs_zero_window_only_current_round():
    agg = MinerScoreAggregator(max_points=8)

    agg.add_score(uid=1, hotkey="hk1", score=0.4, ts=_ts(0), round_id=9)
    agg.add_score(uid=2, hotkey="hk2", score=0.6, ts=_ts(1), round_id=10)

    fresh = agg.fresh_uid_score_pairs(current_round_id=10, freshness_window=0, how="avg")
    assert set(fresh.keys()) == {2}


def test_fresh_uid_score_pairs_rejects_negative_window():
    agg = MinerScoreAggregator(max_points=8)
    with pytest.raises(ValueError):
        agg.fresh_uid_score_pairs(current_round_id=1, freshness_window=-1)


def test_fresh_uid_score_pairs_avg_matches_unfiltered_for_recent():
    agg = MinerScoreAggregator(max_points=8)
    agg.add_score(uid=1, hotkey="hk1", score=0.2, ts=_ts(0), round_id=10)
    agg.add_score(uid=1, hotkey="hk1", score=0.8, ts=_ts(1), round_id=11)

    plain = agg.uid_score_pairs(how="avg")
    fresh = agg.fresh_uid_score_pairs(current_round_id=11, freshness_window=1, how="avg")

    assert fresh[1] == pytest.approx(plain[1])
