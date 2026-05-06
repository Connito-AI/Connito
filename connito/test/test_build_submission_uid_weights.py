"""Unit tests for `build_submission_uid_weights` — the shared helper
that both the restart-replay path and the end-of-round (step 3 → 4)
path use to construct the chain-submission weight map.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from connito.validator.evaluator import (
    WeightSubmissionPayload,
    build_submission_uid_weights,
)


class _StubAggregator:
    """Minimal stand-in for `MinerScoreAggregator`. Only exposes the
    `uid_score_pairs(how="avg")` method the helper reads."""

    def __init__(self, avg_scores: dict[int, float]):
        self._avg = dict(avg_scores)

    def uid_score_pairs(self, how: str = "avg") -> dict[int, float]:
        assert how == "avg", f"helper should always ask for avg, got {how!r}"
        return dict(self._avg)


def _eval_cfg(**overrides) -> SimpleNamespace:
    base = dict(
        weight_group_1_size=3,
        weight_group_1_share=0.97,
        weight_group_2_size=15,
        weight_group_2_share=0.03,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_replay_path_falls_back_to_aggregator_avg():
    """No `pending_round` (restart replay) → return aggregator avg directly."""
    avg = {1: 0.5, 2: 0.0, 3: 1.0}
    payload = build_submission_uid_weights(
        score_aggregator=_StubAggregator(avg),
    )
    assert payload.uid_weights == avg
    assert payload.cohort_emission is False
    assert payload.weight_group_1 == ()
    assert payload.weight_group_2 == ()


def test_legacy_round_without_cohort_state_falls_back_to_avg():
    """`pending_round.cohort_state is None` (legacy path) → aggregator avg."""
    avg = {1: 0.25, 2: 0.75}
    pending = SimpleNamespace(
        cohort_state=None,
        scores={},
        validation_group_a=(),
        validation_group_b=(),
        validation_group_c=(),
    )
    payload = build_submission_uid_weights(
        score_aggregator=_StubAggregator(avg),
        pending_round=pending,
        eval_cfg=_eval_cfg(),
    )
    assert payload.uid_weights == avg
    assert payload.cohort_emission is False


def test_cohort_path_emits_g1_g2_split():
    """With a cohort state and per-round scores, helper applies the
    97% / 3% top-3 / top-15 split."""
    avg = {uid: 0.0 for uid in range(20)}    # avg ignored on cohort branch
    pending = SimpleNamespace(
        cohort_state=object(),               # truthy, contents unused
        scores={1: 1.0, 2: 2.0, 3: 0.5, 4: 0.4, 5: 0.3, 6: 0.2, 7: 0.1},
        validation_group_a=(1, 2, 3),
        validation_group_b=(4, 5),
        validation_group_c=(6, 7),
    )
    payload = build_submission_uid_weights(
        score_aggregator=_StubAggregator(avg),
        pending_round=pending,
        eval_cfg=_eval_cfg(),
    )
    assert payload.cohort_emission is True
    # Top-3 of A∪B by per-round score = (2, 1, 3).
    assert set(payload.weight_group_1) == {1, 2, 3}
    # G2 pool is A∪B∪C minus G1 = {4, 5, 6, 7}; top-15 by score retains all.
    assert set(payload.weight_group_2) <= {4, 5, 6, 7}
    # G1 share splits 97% proportional to score; sum of all weights ~= 1.0
    assert pytest.approx(sum(payload.uid_weights.values()), abs=1e-6) == 1.0


def test_cohort_path_without_eval_cfg_falls_back_to_avg():
    """If a cohort state is present but `eval_cfg` is None, helper
    cannot compute the split — falls back to aggregator avg rather
    than crashing on a missing knob."""
    avg = {1: 0.5, 2: 0.5}
    pending = SimpleNamespace(
        cohort_state=object(),
        scores={1: 1.0, 2: 1.0},
        validation_group_a=(1,),
        validation_group_b=(2,),
        validation_group_c=(),
    )
    payload = build_submission_uid_weights(
        score_aggregator=_StubAggregator(avg),
        pending_round=pending,
        eval_cfg=None,
    )
    assert payload.uid_weights == avg
    assert payload.cohort_emission is False


def test_payload_is_a_frozen_dataclass():
    """Defensive: callers should not be able to mutate the payload's
    selection tuples after construction."""
    p = WeightSubmissionPayload(uid_weights={1: 1.0})
    with pytest.raises((TypeError, AttributeError)):
        p.weight_group_1 = (1, 2, 3)   # type: ignore[misc]
