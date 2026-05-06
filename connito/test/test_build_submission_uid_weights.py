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


def _aggregator_with_history(
    *,
    cur_rid: int,
    cycle_length: int,
    uids_full_history: list[int],
    uids_partial_history: list[int] | None = None,
    uids_one_record: list[int] | None = None,
) -> MinerScoreAggregator:
    """Build an aggregator where:

    * `uids_full_history`: 3 records each, including BOTH cur_rid and
      cur_rid - cycle_length → clears the Group 1 gates.
    * `uids_partial_history`: 2 records (cur_rid + cur_rid - cycle_length)
      so they clear Group 2's `>= 2` gate but not Group 1's `>= 3`.
    * `uids_one_record`: 1 record at cur_rid, clears neither gate.
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


def test_replay_path_falls_back_to_aggregator_avg():
    """No `pending_round` (restart replay) → aggregator avg directly."""
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


def test_legacy_round_without_cohort_state_falls_back_to_avg():
    """`pending_round.cohort_state is None` (legacy path) → aggregator avg."""
    agg = _aggregator_with_history(
        cur_rid=1000,
        cycle_length=100,
        uids_full_history=[1, 2],
    )
    pending = SimpleNamespace(
        cohort_state=None,
        round_id=1000,
        scores={},
        validation_group_a=(),
        validation_group_b=(),
        validation_group_c=(),
    )
    payload = build_submission_uid_weights(
        score_aggregator=agg,
        pending_round=pending,
        eval_cfg=_eval_cfg(),
        cycle_length=100,
    )
    assert payload.cohort_emission is False
    assert set(payload.uid_weights) == {1, 2}


def test_cohort_path_emits_g1_g2_split():
    """With a cohort state plus full history on A∪B, helper applies
    the 98/2 top-3 / top-5 split with the >=3 records + last-2-rounds gate."""
    cur_rid = 1000
    cycle_length = 100
    agg = _aggregator_with_history(
        cur_rid=cur_rid,
        cycle_length=cycle_length,
        uids_full_history=[1, 2, 3],          # → Group 1
        uids_partial_history=[4, 5, 6, 7],    # → Group 2 (>= 2 records)
    )
    pending = SimpleNamespace(
        cohort_state=object(),
        round_id=cur_rid,
        scores={},
        validation_group_a=(1, 2, 3),
        validation_group_b=(4, 5),
        validation_group_c=(6, 7),
    )
    payload = build_submission_uid_weights(
        score_aggregator=agg,
        pending_round=pending,
        eval_cfg=_eval_cfg(),
        cycle_length=cycle_length,
    )
    assert payload.cohort_emission is True
    assert payload.g1_redirected_to_uid_zero is False
    assert set(payload.weight_group_1) == {1, 2, 3}
    # G2 pool = (A ∪ B ∪ C) \ G1, restricted to >= 2 records → {4,5,6,7}
    assert set(payload.weight_group_2) <= {4, 5, 6, 7}
    assert pytest.approx(sum(payload.uid_weights.values()), abs=1e-6) == 1.0


def test_cohort_path_excludes_uids_missing_one_of_last_2_rounds():
    """A UID with 3 records but missing `cur_rid - cycle_length` does
    NOT clear the recency gate, even though `record_count >= 3`."""
    cur_rid = 1000
    cycle_length = 100
    agg = MinerScoreAggregator(max_points=8, max_history_points=64)
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # uid 1: 3 records, but the previous round is missing.
    for i, rid in enumerate([cur_rid - 3 * cycle_length, cur_rid - 2 * cycle_length, cur_rid]):
        agg.add_score(uid=1, hotkey="hk1", score=1.0,
                      ts=ts.replace(microsecond=i + 1), round_id=rid)
    # uid 2: full last-2-rounds history.
    for i, rid in enumerate([cur_rid - 2 * cycle_length, cur_rid - cycle_length, cur_rid]):
        agg.add_score(uid=2, hotkey="hk2", score=1.0,
                      ts=ts.replace(microsecond=i + 10), round_id=rid)

    pending = SimpleNamespace(
        cohort_state=object(),
        round_id=cur_rid,
        scores={},
        validation_group_a=(1, 2),
        validation_group_b=(),
        validation_group_c=(),
    )
    payload = build_submission_uid_weights(
        score_aggregator=agg,
        pending_round=pending,
        eval_cfg=_eval_cfg(),
        cycle_length=cycle_length,
    )
    assert payload.cohort_emission is True
    # Only uid 2 clears the gates.
    assert payload.weight_group_1 == (2,)


def test_cohort_path_empty_g1_redirects_to_uid_zero():
    """If no UID clears the Group 1 gates, the 98% share goes to uid=0
    so the validator stays at 100% emission."""
    cur_rid = 1000
    cycle_length = 100
    agg = _aggregator_with_history(
        cur_rid=cur_rid,
        cycle_length=cycle_length,
        uids_full_history=[],                  # nobody clears g1
        uids_partial_history=[4, 5, 6],        # g2 candidates exist
    )
    pending = SimpleNamespace(
        cohort_state=object(),
        round_id=cur_rid,
        scores={},
        validation_group_a=(),
        validation_group_b=(),
        validation_group_c=(4, 5, 6),
    )
    payload = build_submission_uid_weights(
        score_aggregator=agg,
        pending_round=pending,
        eval_cfg=_eval_cfg(),
        cycle_length=cycle_length,
    )
    assert payload.cohort_emission is True
    assert payload.g1_redirected_to_uid_zero is True
    assert payload.weight_group_1 == (0,)
    # Total stays at 100% emission — uid 0 collects the full 98%.
    assert pytest.approx(payload.uid_weights[0], abs=1e-6) == 0.98
    assert pytest.approx(sum(payload.uid_weights.values()), abs=1e-6) == 1.0


def test_cohort_path_without_eval_cfg_falls_back_to_avg():
    """Cohort state present but missing `eval_cfg` → fall back to avg
    rather than crashing on a missing knob."""
    agg = _aggregator_with_history(
        cur_rid=1000,
        cycle_length=100,
        uids_full_history=[1, 2],
    )
    pending = SimpleNamespace(
        cohort_state=object(),
        round_id=1000,
        scores={},
        validation_group_a=(1,),
        validation_group_b=(2,),
        validation_group_c=(),
    )
    payload = build_submission_uid_weights(
        score_aggregator=agg,
        pending_round=pending,
        eval_cfg=None,
        cycle_length=100,
    )
    assert payload.cohort_emission is False


def test_cohort_path_without_cycle_length_falls_back_to_avg():
    """Cohort state present but missing `cycle_length` → fall back to avg
    (the recency gate cannot be evaluated without it)."""
    agg = _aggregator_with_history(
        cur_rid=1000,
        cycle_length=100,
        uids_full_history=[1, 2],
    )
    pending = SimpleNamespace(
        cohort_state=object(),
        round_id=1000,
        scores={},
        validation_group_a=(1,),
        validation_group_b=(2,),
        validation_group_c=(),
    )
    payload = build_submission_uid_weights(
        score_aggregator=agg,
        pending_round=pending,
        eval_cfg=_eval_cfg(),
        cycle_length=None,
    )
    assert payload.cohort_emission is False


def test_payload_is_a_frozen_dataclass():
    """Defensive: callers should not be able to mutate the payload after
    construction."""
    p = WeightSubmissionPayload(uid_weights={1: 1.0})
    with pytest.raises((TypeError, AttributeError)):
        p.weight_group_1 = (1, 2, 3)   # type: ignore[misc]
