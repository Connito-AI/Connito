"""Unit tests for `build_submission_uid_weights` — the shared helper
that both the restart-replay path and the end-of-round (step 3 → 4)
path use to construct the chain-submission weight map.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from connito.validator.aggregator import MinerScoreAggregator
from connito.validator.evaluator import (
    WeightSubmissionPayload,
    build_submission_uid_weights,
)


def _eval_cfg(**overrides) -> SimpleNamespace:
    base = dict(
        weight_group_1_size=3,
        weight_group_1_share=0.98,
        weight_group_2_size=5,
        weight_group_2_share=0.02,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _cohort_state(
    *,
    a: tuple[int, ...] = (),
    b: tuple[int, ...] = (),
    c: tuple[int, ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        validation_group_a=a,
        validation_group_b=b,
        validation_group_c=c,
    )


def _aggregator_with_history(
    *,
    cur_rid: int,
    cycle_length: int,
    uids_full_history: list[int],
    uids_partial_history: list[int] | None = None,
    uids_one_record: list[int] | None = None,
) -> MinerScoreAggregator:
    """Build an aggregator where:

    * `uids_full_history`: 3 records at 3 distinct round_ids inside the
      G1 window (`cur_rid - 5*cycle_length .. cur_rid`) → clears G1
      (gate is ≥3 distinct round_ids).
    * `uids_partial_history`: 2 records at 2 distinct round_ids inside
      the G1 window → fails G1's ≥3 gate, still clears G2's ≥1 gate.
    * `uids_one_record`: 1 record at cur_rid only → fails G1, clears G2.
    """
    agg = MinerScoreAggregator(max_points=8, max_history_points=64)
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def _add(uid: int, rids: list[int], scores: list[float]) -> None:
        for i, (rid, score) in enumerate(zip(rids, scores)):
            agg.add_score(
                uid=uid,
                hotkey=f"hk{uid}",
                score=score,
                ts=ts.replace(microsecond=i + 1),
                round_id=rid,
            )

    base_score = 1.0
    for uid in uids_full_history:
        _add(
            uid,
            [cur_rid - 2 * cycle_length, cur_rid - cycle_length, cur_rid],
            [base_score, base_score, base_score + uid * 0.1],
        )
    for uid in uids_partial_history or []:
        _add(uid, [cur_rid - cycle_length, cur_rid], [0.5, 0.5])
    for uid in uids_one_record or []:
        _add(uid, [cur_rid], [0.5])
    return agg


def test_no_cohort_state_falls_back_to_aggregator_avg():
    """No `cohort_state` provided → aggregator avg directly. Covers the
    cold-start replay case where `cohort_state.json` does not exist."""
    agg = _aggregator_with_history(
        cur_rid=1000,
        cycle_length=100,
        uids_full_history=[1, 2],
    )
    payload = build_submission_uid_weights(score_aggregator=agg)
    assert payload.cohort_emission is False
    assert payload.g1_redirected_to_uid_zero is False
    assert payload.weight_group_1 == ()
    assert payload.weight_group_2 == ()
    assert set(payload.uid_weights) == {1, 2}


def test_cohort_path_emits_g1_g2_split_without_pending_round():
    """Helper accepts `cohort_state` directly — no Round wrapper required.
    With ≥3 distinct round_ids in the G1 window, applies the 98/2 top-3
    / top-5 split. uids 4/5 only have 2 distinct round_ids so they fail
    the G1 gate and drop to G2."""
    cur_rid = 1000
    cycle_length = 100
    agg = _aggregator_with_history(
        cur_rid=cur_rid,
        cycle_length=cycle_length,
        uids_full_history=[1, 2, 3],
        uids_partial_history=[4, 5, 6, 7],
    )
    payload = build_submission_uid_weights(
        score_aggregator=agg,
        cohort_state=_cohort_state(a=(1, 2, 3), b=(4, 5), c=(6, 7)),
        round_id=cur_rid,
        cycle_length=cycle_length,
        eval_cfg=_eval_cfg(),
    )
    assert payload.cohort_emission is True
    assert payload.g1_redirected_to_uid_zero is False
    # G1 picks top-3 by avg from A∪B that clears the ≥3 distinct round_ids
    # gate; uids 4/5 only have 2 distinct round_ids so they cannot reach G1
    # regardless of avg.
    assert set(payload.weight_group_1) == {1, 2, 3}
    assert set(payload.weight_group_2) <= {4, 5, 6, 7}
    assert pytest.approx(sum(payload.uid_weights.values()), abs=1e-6) == 1.0


def test_cohort_path_g1_admits_uid_with_three_distinct_round_ids_in_window():
    """The recency gate is "≥3 distinct round_ids within
    `5*cycle_length` blocks of `cur_rid`" — i.e. scored in at least 3
    of the last 5 cycles. UIDs with 2-or-fewer distinct round_ids in
    the window do not qualify, even if they have multiple records at
    the same round_id."""
    cur_rid = 1000
    cycle_length = 100
    agg = MinerScoreAggregator(max_points=8, max_history_points=64)
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # uid 1: three distinct round_ids inside the window, non-consecutive.
    for i, rid in enumerate([cur_rid - 4 * cycle_length, cur_rid - 2 * cycle_length, cur_rid]):
        agg.add_score(uid=1, hotkey="hk1", score=1.0,
                      ts=ts.replace(microsecond=i + 1), round_id=rid)
    # uid 2: two distinct round_ids — fails the ≥3 gate.
    for i, rid in enumerate([cur_rid - cycle_length, cur_rid]):
        agg.add_score(uid=2, hotkey="hk2", score=1.0,
                      ts=ts.replace(microsecond=i + 10), round_id=rid)
    # uid 3: three records but only 1 distinct round_id — fails.
    for i in range(3):
        agg.add_score(uid=3, hotkey="hk3", score=0.5,
                      ts=ts.replace(microsecond=i + 20), round_id=cur_rid)

    payload = build_submission_uid_weights(
        score_aggregator=agg,
        cohort_state=_cohort_state(a=(1, 2, 3)),
        round_id=cur_rid,
        cycle_length=cycle_length,
        eval_cfg=_eval_cfg(),
    )
    assert payload.cohort_emission is True
    assert payload.weight_group_1 == (1,)


def test_cohort_path_g1_excludes_uid_outside_window():
    """A UID with 3 distinct round_ids, all OUTSIDE the
    `5*cycle_length` window, does not qualify for G1. A UID with 3
    distinct round_ids inside the window does."""
    cur_rid = 1000
    cycle_length = 100
    # Window: [cur_rid - 500, cur_rid] = [500, 1000].
    agg = MinerScoreAggregator(max_points=8, max_history_points=64)
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # uid 1: three distinct round_ids, all BELOW the window.
    for i, rid in enumerate(
        [cur_rid - 10 * cycle_length, cur_rid - 8 * cycle_length, cur_rid - 6 * cycle_length]
    ):
        agg.add_score(uid=1, hotkey="hk1", score=1.0,
                      ts=ts.replace(microsecond=i + 1), round_id=rid)
    # uid 2: three distinct round_ids, all INSIDE the window.
    for i, rid in enumerate(
        [cur_rid - 4 * cycle_length, cur_rid - 2 * cycle_length, cur_rid]
    ):
        agg.add_score(uid=2, hotkey="hk2", score=1.0,
                      ts=ts.replace(microsecond=i + 10), round_id=rid)

    payload = build_submission_uid_weights(
        score_aggregator=agg,
        cohort_state=_cohort_state(a=(1, 2)),
        round_id=cur_rid,
        cycle_length=cycle_length,
        eval_cfg=_eval_cfg(),
    )
    assert payload.cohort_emission is True
    assert payload.weight_group_1 == (2,)


def test_cohort_path_empty_g1_redirects_to_uid_zero():
    """If no UID clears the Group 1 gates, the 98% share goes to uid=0
    so the validator stays at 100% emission."""
    cur_rid = 1000
    cycle_length = 100
    agg = _aggregator_with_history(
        cur_rid=cur_rid,
        cycle_length=cycle_length,
        uids_full_history=[],
        uids_partial_history=[4, 5, 6],
    )
    payload = build_submission_uid_weights(
        score_aggregator=agg,
        cohort_state=_cohort_state(c=(4, 5, 6)),
        round_id=cur_rid,
        cycle_length=cycle_length,
        eval_cfg=_eval_cfg(),
    )
    assert payload.cohort_emission is True
    assert payload.g1_redirected_to_uid_zero is True
    assert payload.weight_group_1 == (0,)
    assert pytest.approx(payload.uid_weights[0], abs=1e-6) == 0.98
    assert pytest.approx(sum(payload.uid_weights.values()), abs=1e-6) == 1.0


def test_cohort_path_without_eval_cfg_falls_back_to_avg():
    agg = _aggregator_with_history(cur_rid=1000, cycle_length=100, uids_full_history=[1, 2])
    payload = build_submission_uid_weights(
        score_aggregator=agg,
        cohort_state=_cohort_state(a=(1,), b=(2,)),
        round_id=1000,
        cycle_length=100,
        eval_cfg=None,
    )
    assert payload.cohort_emission is False


def test_cohort_path_without_round_id_falls_back_to_avg():
    """No anchor for the recency gate → fall back to avg."""
    agg = _aggregator_with_history(cur_rid=1000, cycle_length=100, uids_full_history=[1, 2])
    payload = build_submission_uid_weights(
        score_aggregator=agg,
        cohort_state=_cohort_state(a=(1,), b=(2,)),
        round_id=None,
        cycle_length=100,
        eval_cfg=_eval_cfg(),
    )
    assert payload.cohort_emission is False


def test_cohort_path_without_cycle_length_falls_back_to_avg():
    agg = _aggregator_with_history(cur_rid=1000, cycle_length=100, uids_full_history=[1, 2])
    payload = build_submission_uid_weights(
        score_aggregator=agg,
        cohort_state=_cohort_state(a=(1,), b=(2,)),
        round_id=1000,
        cycle_length=None,
        eval_cfg=_eval_cfg(),
    )
    assert payload.cohort_emission is False


def test_payload_is_a_frozen_dataclass():
    p = WeightSubmissionPayload(uid_weights={1: 1.0})
    with pytest.raises((TypeError, AttributeError)):
        p.weight_group_1 = (1, 2, 3)   # type: ignore[misc]
