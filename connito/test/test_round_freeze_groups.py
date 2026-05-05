"""Integration tests for the round-group overlay inside `Round.freeze`.

These exercise the path where the feature flag is on, a `cohort_state`
is provided (or absent for cold start), and the resulting `Round` has
its new group fields populated and its foreground roster overridden.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from connito.validator.aggregator import MinerScoreAggregator
from connito.validator.cohort_state import CohortState
from connito.validator.round import Round


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_model() -> nn.Module:
    m = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        m.weight.fill_(0.1)
    return m


def _config(*, flag: bool, my_hotkey: str = "vme") -> SimpleNamespace:
    """Validator config stub matching what `Round.freeze` reads."""
    return SimpleNamespace(
        chain=SimpleNamespace(hotkey_ss58=my_hotkey, netuid=7, network="mock"),
        task=SimpleNamespace(exp=SimpleNamespace(group_id=1)),
        evaluation=SimpleNamespace(
            enable_round_group_construction=flag,
            cohort_window_cycles=8,
            weight_group_1_size=3,
            weight_group_1_share=0.97,
            weight_group_2_size=15,
            weight_group_2_share=0.03,
            validation_group_a_size=3,
            validation_group_ab_total=13,
            validation_group_c_size=17,
            group_a_min_consensus=1,
            group_a_min_weight_per_validator=0.03,
            cohort_state_filename="cohort_state.json",
        ),
    )


def _metagraph(
    *,
    n_total: int,
    validator_hotkeys: list[str] | None = None,
    miner_hotkeys: list[str] | None = None,
    weight_matrix: list[list[float]] | None = None,
) -> SimpleNamespace:
    """Build a metagraph stub.

    `hotkeys` is laid out as `validator_hotkeys + miner_hotkeys`, padded
    with synthetic `m{i}` hotkeys to reach `n_total`. The weight matrix
    is shape `(n_total, n_total)`; validator rows (indices `0..len(vh)-1`)
    are populated from `weight_matrix`, all other rows are zero.
    """
    validator_hotkeys = validator_hotkeys or []
    miner_hotkeys = miner_hotkeys or []
    n_val = len(validator_hotkeys)
    n_explicit = n_val + len(miner_hotkeys)
    hotkeys = (
        list(validator_hotkeys)
        + list(miner_hotkeys)
        + [f"pad{i}" for i in range(n_total - n_explicit)]
    )
    weights = torch.zeros((n_total, n_total), dtype=torch.float32)
    if weight_matrix is not None:
        for v_idx, row in enumerate(weight_matrix):
            for m_idx, w in enumerate(row):
                weights[v_idx, m_idx] = w
    return SimpleNamespace(
        hotkeys=hotkeys,
        incentive=torch.zeros(n_total),
        weights=weights,
        S=torch.ones(n_total),
    )


def _stub_subtensor(metagraph, block: int = 100) -> SimpleNamespace:
    return SimpleNamespace(
        block=block,
        metagraph=lambda netuid=None: metagraph,
        network="mock",
    )


def _ckpt_stub(hk: str) -> SimpleNamespace:
    """Mimic ChainCheckpoint enough to clear the freeze_zero filter."""
    return SimpleNamespace(hf_repo_id="repo", hf_revision=hk)


def _freeze(
    *,
    config,
    metagraph,
    assignment,
    miners_with_checkpoint,
    cohort_state=None,
    score_aggregator=None,
    cycle_index=8,
    cycle_length=100,
    round_id=800,
):
    chain_checkpoints = {hk: _ckpt_stub(hk) for hk in miners_with_checkpoint}
    assignment_result = SimpleNamespace(
        assignment=assignment,
        miners_with_checkpoint=miners_with_checkpoint,
        chain_checkpoints_by_hotkey=chain_checkpoints,
    )
    subtensor = _stub_subtensor(metagraph, block=round_id)
    with patch("connito.shared.chain.get_chain_commits", return_value=[]), \
         patch(
             "connito.shared.cycle.get_combined_validator_seed", return_value="seed"
         ), \
         patch(
             "connito.shared.cycle.get_validator_miner_assignment",
             return_value=assignment_result,
         ):
        return Round.freeze(
            config=config,
            subtensor=subtensor,
            metagraph=metagraph,
            global_model=_make_model(),
            round_id=round_id,
            cycle_index=cycle_index,
            cycle_length=cycle_length,
            cohort_state=cohort_state,
            score_aggregator=score_aggregator,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_flag_off_leaves_new_fields_empty():
    """Legacy path: feature flag off → no group fields populated."""
    config = _config(flag=False)
    metagraph = _metagraph(
        n_total=15,
        validator_hotkeys=["vme"],
        miner_hotkeys=[f"m{i}" for i in range(5)],
    )
    assignment = {"vme": [f"m{i}" for i in range(5)]}
    miners = [f"m{i}" for i in range(5)]

    r = _freeze(
        config=config,
        metagraph=metagraph,
        assignment=assignment,
        miners_with_checkpoint=miners,
    )

    assert r.weight_group_1 == ()
    assert r.weight_group_2 == ()
    assert r.validation_group_a == ()
    assert r.validation_group_b == ()
    assert r.validation_group_c == ()
    assert r.cohort_epoch == 0
    assert r.cohort_state is None


def test_flag_on_cold_start_populates_groups_from_chain_consensus():
    """Cold start: no prior cohort state, no aggregator. Round still
    gets validation Groups A/B/C from chain-set consensus, with empty
    weight ballots (the first cohort cannot have a promotion election).
    """
    n_total = 35
    # 5 validators (v0..v3 + vme), the 4 vN at uids 0..3 emit weights;
    # vme is uid 4 (no weights from us this cohort). Miners m5..m34
    # at uids 5..34. v0..v3 all agree on miners 5,6,7 (high weight)
    # → Group A full, and on 21..30 (low weight) → Group B candidates.
    validator_hotkeys = ["v0", "v1", "v2", "v3", "vme"]
    miner_hotkeys = [f"m{i}" for i in range(5, n_total)]
    weights = []
    for _v in range(4):
        row = [0.0] * n_total
        for j, m in enumerate([5, 6, 7]):
            row[m] = 0.9 - j * 0.01
        for m in range(21, 31):
            row[m] = 0.05
        weights.append(row)
    metagraph = _metagraph(
        n_total=n_total,
        validator_hotkeys=validator_hotkeys,
        miner_hotkeys=miner_hotkeys,
        weight_matrix=weights,
    )
    config = _config(flag=True)
    assignment = {
        "v0": [f"m{i}" for i in range(5, 10)],
        "v1": [f"m{i}" for i in range(10, 15)],
        "v2": [f"m{i}" for i in range(15, 20)],
        "v3": [f"m{i}" for i in range(20, 25)],
        "vme": [f"m{i}" for i in range(25, n_total)],   # this validator's slice
    }
    miners = [f"m{i}" for i in range(5, n_total)]

    r = _freeze(
        config=config,
        metagraph=metagraph,
        assignment=assignment,
        miners_with_checkpoint=miners,
        cohort_state=None,
        cycle_index=0,
        cycle_length=100,
        round_id=0,
    )

    # Validation Group A has the 3 consensus miners (uids 5,6,7).
    assert set(r.validation_group_a) == {5, 6, 7}
    # |A| + |B| invariant.
    assert len(r.validation_group_a) + len(r.validation_group_b) == 13
    # Foreground is a subset of A∪B (per-validator partition of A∪B).
    ab_set = set(r.validation_group_a) | set(r.validation_group_b)
    assert set(r.foreground_uids) <= ab_set
    # Group C is disjoint from A∪B (per-validator partition of all \ A∪B).
    assert set(r.validation_group_c).isdisjoint(ab_set)
    # Background = (A ∪ B ∪ C) \ foreground.
    assert set(r.background_uids) == (
        set(r.validation_group_a) | set(r.validation_group_b) | set(r.validation_group_c)
    ) - set(r.foreground_uids)
    # |fg| + |bg| = full roster, no UID dropped.
    assert len(r.foreground_uids) + len(r.background_uids) == (
        len(r.validation_group_a) + len(r.validation_group_b) + len(r.validation_group_c)
    )
    # Cold start → empty weight ballots.
    assert r.weight_group_1 == ()
    assert r.weight_group_2 == ()
    # New CohortState is attached for run.py to persist.
    assert r.cohort_state is not None
    assert r.cohort_state.cohort_epoch == 0


def test_flag_on_within_cohort_holds_groups_constant():
    """Cycle 8k+5: cohort state already exists → Round.freeze reuses it
    rather than re-electing.
    """
    n = 30
    metagraph = _metagraph(
        n_total=n,
        validator_hotkeys=["vme"],
        miner_hotkeys=[f"m{i}" for i in range(1, n)],
    )
    config = _config(flag=True)
    assignment = {"vme": [f"m{i}" for i in range(20, 30)]}
    miners = [f"m{i}" for i in range(n)]

    held = CohortState(
        cohort_epoch=8,
        expert_group="1",
        weight_group_1=(0, 1, 2),
        weight_group_2=(3, 4, 5),
        validation_group_a=(0, 1, 2),
        validation_group_b=tuple(range(3, 13)),
        validation_group_c=tuple(range(20, 30)),
        last_election_round_id=800,
        highest_seen_cycle_index=12,
    )
    r = _freeze(
        config=config,
        metagraph=metagraph,
        assignment=assignment,
        miners_with_checkpoint=miners,
        cohort_state=held,
        cycle_index=13,         # mid-cohort (8..15)
        cycle_length=100,
        round_id=1300,
    )
    # Held groups should pass through unchanged.
    assert r.validation_group_a == held.validation_group_a
    assert r.validation_group_b == held.validation_group_b
    assert r.validation_group_c == held.validation_group_c
    assert r.weight_group_1 == held.weight_group_1
    assert r.weight_group_2 == held.weight_group_2
    assert r.cohort_state is held


def test_flag_on_election_at_boundary_promotes_top_scorers():
    """At cohort boundary, the previous cohort's top-3 of A∪B by mean
    local score lands in the next cohort's weight Group 1 ballot.
    """
    n = 30
    metagraph = _metagraph(
        n_total=n,
        validator_hotkeys=["vme"],
        miner_hotkeys=[f"m{i}" for i in range(1, n)],
    )
    config = _config(flag=True)
    assignment = {"vme": [f"m{i}" for i in range(20, n)]}
    miners = [f"m{i}" for i in range(1, n)]

    # Previous cohort: A = {0,1,2}, B = {3,4,5,6,7}. Score 0 highest.
    held = CohortState(
        cohort_epoch=0,
        expert_group="1",
        validation_group_a=(0, 1, 2),
        validation_group_b=(3, 4, 5, 6, 7),
        validation_group_c=tuple(range(20, 30)),
        highest_seen_cycle_index=7,
    )

    aggregator = MinerScoreAggregator(max_points=64, max_history_points=64)
    # Round_id 0..700 step 100 → cycles 0..7 with cycle_length=100.
    # High scorers get distinct scores so the ballot's exclude-ties rule
    # doesn't drop them (uid-tiebreak was removed deliberately).
    high_scores = {0: 1.0, 1: 0.9, 2: 0.8}
    low_scores = {uid: 0.1 + uid * 0.01 for uid in (3, 4, 5, 6, 7)}
    for cycle in range(8):
        rid = cycle * 100
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc).replace(microsecond=cycle)
        for uid, score in high_scores.items():
            aggregator.add_score(
                uid=uid, hotkey=f"m{uid}", score=score, ts=ts, round_id=rid
            )
        for uid, score in low_scores.items():
            aggregator.add_score(
                uid=uid, hotkey=f"m{uid}", score=score, ts=ts, round_id=rid
            )

    r = _freeze(
        config=config,
        metagraph=metagraph,
        assignment=assignment,
        miners_with_checkpoint=miners,
        cohort_state=held,
        score_aggregator=aggregator,
        cycle_index=8,        # boundary
        cycle_length=100,
        round_id=800,
    )

    # Top-3 of A∪B by mean = {0,1,2} → next cohort's weight Group 1.
    assert set(r.weight_group_1) == {0, 1, 2}
    # Weight Group 2 = top-2 of (B∪C)\G1 by mean. Group C had no scores
    # (not validated), so all C miners get 0.0. Group B miners scored
    # 0.1 each → those rank above the zeros.
    assert len(r.weight_group_2) == config.evaluation.weight_group_2_size or \
        len(r.weight_group_2) <= config.evaluation.weight_group_2_size
    # Cohort advanced.
    assert r.cohort_state is not None
    assert r.cohort_state.cohort_epoch == 8


def test_flag_on_with_missing_cycle_index_falls_back_to_legacy():
    """If cycle_index/cycle_length aren't supplied, the new path can't
    run — Round.freeze logs a warning and falls back to legacy.
    """
    n = 10
    metagraph = _metagraph(
        n_total=n,
        validator_hotkeys=["vme"],
        miner_hotkeys=[f"m{i}" for i in range(1, 6)],
    )
    config = _config(flag=True)
    assignment = {"vme": [f"m{i}" for i in range(1, 6)]}
    miners = [f"m{i}" for i in range(1, 6)]

    r = _freeze(
        config=config,
        metagraph=metagraph,
        assignment=assignment,
        miners_with_checkpoint=miners,
        cohort_state=None,
        cycle_index=None,    # missing
        cycle_length=None,   # missing
        round_id=100,
    )
    # No new fields populated.
    assert r.validation_group_a == ()
    assert r.weight_group_1 == ()
    assert r.cohort_state is None
