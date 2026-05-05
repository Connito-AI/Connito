"""Pure unit tests for connito/validator/round_groups.py.

No chain RPCs, no fixtures from `Round.freeze`. Each test exercises one
selection rule from the spec
(`_specs/round-group-construction-scheme.md`) against a synthetic
metagraph or score map.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from connito.validator.cohort_state import CohortState
from connito.validator.round_groups import (
    ElectionBallots,
    build_cohort_groups,
    cohort_epoch_for,
    compute_election_ballots,
    compute_foreground_partition,
    compute_group_a,
    compute_group_b,
    compute_group_c,
    compute_uid_weights,
    is_cohort_boundary,
    maybe_advance_cohort,
    read_chain_set_top_k,
    select_top_n_by_local_score,
    split_foreground_background,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(**overrides) -> SimpleNamespace:
    """Default config knobs matching the spec."""
    base = dict(
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
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _metagraph_with_weights(weight_matrix: list[list[float]]) -> SimpleNamespace:
    """Build a metagraph stub whose `.weights` is a 2-D tensor of shape
    `(n_validators, n_miners)`.
    """
    W = torch.tensor(weight_matrix, dtype=torch.float32)
    return SimpleNamespace(weights=W, hotkeys=[f"hk{i}" for i in range(W.shape[1])])


# ---------------------------------------------------------------------------
# Cohort boundary helpers
# ---------------------------------------------------------------------------


def test_is_cohort_boundary_at_multiples_of_window():
    assert is_cohort_boundary(0)
    assert is_cohort_boundary(8)
    assert is_cohort_boundary(16)
    assert not is_cohort_boundary(1)
    assert not is_cohort_boundary(7)
    assert not is_cohort_boundary(9)


def test_cohort_epoch_for_truncates_to_window():
    assert cohort_epoch_for(0) == 0
    assert cohort_epoch_for(7) == 0
    assert cohort_epoch_for(8) == 8
    assert cohort_epoch_for(15) == 8
    assert cohort_epoch_for(16) == 16


# ---------------------------------------------------------------------------
# read_chain_set_top_k
# ---------------------------------------------------------------------------


def test_read_chain_set_top_k_tallies_validator_count_and_total_weight():
    # 3 validators (uids 0,1,2), 5 miners (uids 0..4).
    weights = [
        [0.0, 0.0, 0.5, 0.3, 0.0],
        [0.0, 0.0, 0.4, 0.0, 0.2],
        [0.0, 0.0, 0.0, 0.6, 0.1],
    ]
    metagraph = _metagraph_with_weights(weights)
    tally = read_chain_set_top_k(
        metagraph, k=2, qualified_validator_uids=[0, 1, 2]
    )
    # Tally is (count, total_weight, max_weight_from_one_validator)
    assert tally[2][0] == 2 and tally[2][1] == pytest.approx(0.9) and tally[2][2] == pytest.approx(0.5)
    assert tally[3][0] == 2 and tally[3][1] == pytest.approx(0.9) and tally[3][2] == pytest.approx(0.6)
    assert tally[4][0] == 2 and tally[4][1] == pytest.approx(0.3) and tally[4][2] == pytest.approx(0.2)


def test_read_chain_set_top_k_respects_top_k_per_validator():
    # Validator 0 weights 4 miners; only top-2 should count.
    weights = [
        [0.0, 0.1, 0.2, 0.3, 0.4],   # top-2 = miners 4 and 3
    ]
    metagraph = _metagraph_with_weights(weights)
    tally = read_chain_set_top_k(metagraph, k=2, qualified_validator_uids=[0])
    assert set(tally.keys()) == {3, 4}


def test_read_chain_set_top_k_filters_by_eligible_miner_uids():
    weights = [[0.0, 0.0, 0.5, 0.3, 0.2]]
    metagraph = _metagraph_with_weights(weights)
    tally = read_chain_set_top_k(
        metagraph,
        k=3,
        qualified_validator_uids=[0],
        eligible_miner_uids={2, 4},
    )
    assert set(tally.keys()) == {2, 4}


def test_read_chain_set_top_k_empty_when_metagraph_lacks_weights():
    metagraph = SimpleNamespace(hotkeys=["a", "b"])  # no `weights` attr
    assert read_chain_set_top_k(metagraph, k=3, qualified_validator_uids=[0]) == {}


# ---------------------------------------------------------------------------
# Group A — consensus check
# ---------------------------------------------------------------------------


def test_compute_group_a_returns_uids_passing_both_gates():
    """min_consensus=1 + min_weight_per_validator=0.03; rank by total weight."""
    chain_top1 = {
        10: (3, 1.5, 0.6),
        11: (4, 2.0, 0.7),
        12: (2, 0.04, 0.02),    # max_w 0.02 below 3% floor → excluded
        13: (3, 1.2, 0.5),
    }
    g_a = compute_group_a(
        chain_top1, min_consensus=1, min_weight_per_validator=0.03, max_size=3
    )
    assert set(g_a) == {11, 10, 13}
    assert g_a[0] == 11   # highest total weight first


def test_compute_group_a_excludes_miners_below_weight_floor():
    """Miner with high count but no validator emitting > 3% is excluded."""
    chain_top1 = {
        10: (5, 0.10, 0.02),    # 5 validators, all tiny — fails 3% gate
        11: (1, 0.05, 0.05),    # only 1 validator but emits 5% — passes
    }
    g_a = compute_group_a(
        chain_top1, min_consensus=1, min_weight_per_validator=0.03, max_size=3
    )
    assert g_a == (11,)


def test_compute_group_a_min_consensus_can_be_raised():
    """The count floor is still tunable (legacy 3-validator behavior)."""
    chain_top1 = {
        10: (1, 0.9, 0.9),       # only 1 validator — fails count floor of 3
        11: (3, 1.2, 0.5),       # passes both gates
    }
    g_a = compute_group_a(
        chain_top1, min_consensus=3, min_weight_per_validator=0.03, max_size=3
    )
    assert g_a == (11,)


def test_compute_group_a_empty_when_no_one_passes_both_gates():
    chain_top1 = {
        10: (1, 0.02, 0.02),     # fails weight gate
        11: (0, 0.0, 0.0),       # empty
    }
    g_a = compute_group_a(
        chain_top1, min_consensus=1, min_weight_per_validator=0.03, max_size=3
    )
    assert g_a == ()


def test_compute_group_a_tie_break_by_total_weight_then_uid():
    chain_top1 = {
        10: (3, 1.0, 0.4),
        11: (3, 2.0, 0.7),    # higher total weight wins
        12: (3, 1.0, 0.4),    # same total as 10 → uid asc
    }
    g_a = compute_group_a(
        chain_top1, min_consensus=3, min_weight_per_validator=0.03, max_size=3
    )
    assert g_a == (11, 10, 12)


# ---------------------------------------------------------------------------
# Group B — invariant |A|+|B| = ab_total
# ---------------------------------------------------------------------------


def test_compute_group_b_grows_when_group_a_underfills():
    chain_top2 = {uid: (5, 1.0, 0.2) for uid in range(20, 40)}
    # A=2 → B should be 11 to keep |A|+|B|=13
    g_b = compute_group_b(chain_top2, group_a=(1, 2), ab_total=13)
    assert len(g_b) == 11

    # A=0 → B is 13
    g_b_full = compute_group_b(chain_top2, group_a=(), ab_total=13)
    assert len(g_b_full) == 13

    # A=3 → B=10
    g_b_normal = compute_group_b(chain_top2, group_a=(1, 2, 3), ab_total=13)
    assert len(g_b_normal) == 10


def test_compute_group_b_excludes_group_a_uids():
    chain_top2 = {uid: (5, 1.0, 0.2) for uid in range(10)}
    g_b = compute_group_b(chain_top2, group_a=(1, 2, 3), ab_total=13)
    assert 1 not in g_b and 2 not in g_b and 3 not in g_b


def test_compute_group_b_ranks_by_total_weight_then_count_then_uid():
    """Sort: total_weight desc → validator_count desc → uid asc.

    Total weight is the primary signal; validator count is a tiebreaker
    only when totals are equal.
    """
    chain_top2 = {
        20: (4, 1.0, 0.25),
        21: (5, 0.5, 0.10),   # higher count, but lowest total → ranks last
        22: (4, 2.0, 0.50),   # highest total → ranks first
        23: (4, 1.0, 0.25),   # same total as 20 → tiebreak uid asc → 20 first
    }
    g_b = compute_group_b(chain_top2, group_a=(), ab_total=4)
    assert g_b == (22, 20, 23, 21)


# ---------------------------------------------------------------------------
# Group C — distinct across validators
# ---------------------------------------------------------------------------


def test_compute_group_c_excludes_ab_uids():
    """Group C is partitioned over miners NOT in A∪B (the exploration tier)."""
    validator_seeds = {"v1": 1, "v2": 2}
    all_miner_hotkeys = [f"m{i}" for i in range(1, 9)]
    hotkey_to_uid = {f"m{i}": i for i in range(1, 9)}
    g_c = compute_group_c(
        validator_seeds=validator_seeds,
        all_miner_hotkeys=all_miner_hotkeys,
        ab_uids={1, 2},   # uids 1 and 2 are in A∪B
        my_hotkey="v1",
        hotkey_to_uid=hotkey_to_uid,
        max_size=17,
    )
    assert 1 not in g_c and 2 not in g_c
    # Whatever validator gets, must be from {3..8}
    assert set(g_c) <= {3, 4, 5, 6, 7, 8}


def test_compute_group_c_distinct_across_validators():
    validator_seeds = {"v1": 1, "v2": 2}
    all_miner_hotkeys = [f"m{i}" for i in range(1, 7)]
    hotkey_to_uid = {f"m{i}": i for i in range(1, 7)}
    c1 = compute_group_c(
        validator_seeds=validator_seeds,
        all_miner_hotkeys=all_miner_hotkeys,
        ab_uids=set(),
        my_hotkey="v1",
        hotkey_to_uid=hotkey_to_uid,
    )
    c2 = compute_group_c(
        validator_seeds=validator_seeds,
        all_miner_hotkeys=all_miner_hotkeys,
        ab_uids=set(),
        my_hotkey="v2",
        hotkey_to_uid=hotkey_to_uid,
    )
    # Each miner is assigned to exactly one validator → c1 ∩ c2 == ∅.
    assert set(c1).isdisjoint(set(c2))
    # Together they cover the full pool.
    assert set(c1) | set(c2) == {1, 2, 3, 4, 5, 6}


def test_compute_group_c_caps_at_max_size():
    """Each validator's slice is capped at `max_size` when enough validators
    exist to absorb the overflow.
    """
    # 4 validators, 100 miners, cap=17 each → at most 4*17=68 miners assigned
    # and each validator's slice capped at 17.
    validator_seeds = {f"v{i}": i for i in range(4)}
    all_miner_hotkeys = [f"m{i}" for i in range(100)]
    hotkey_to_uid = {f"m{i}": i for i in range(100)}
    g_c = compute_group_c(
        validator_seeds=validator_seeds,
        all_miner_hotkeys=all_miner_hotkeys,
        ab_uids=set(),
        my_hotkey="v1",
        hotkey_to_uid=hotkey_to_uid,
        max_size=17,
    )
    assert len(g_c) <= 17


def test_compute_foreground_partition_is_subset_of_ab():
    """Foreground only contains UIDs from A∪B."""
    validator_seeds = {"v1": 1, "v2": 2}
    all_miner_hotkeys = [f"m{i}" for i in range(1, 11)]
    hotkey_to_uid = {f"m{i}": i for i in range(1, 11)}
    fg = compute_foreground_partition(
        validator_seeds=validator_seeds,
        all_miner_hotkeys=all_miner_hotkeys,
        ab_uids={3, 4, 5},   # Group A∪B
        my_hotkey="v1",
        hotkey_to_uid=hotkey_to_uid,
    )
    assert set(fg) <= {3, 4, 5}


def test_compute_foreground_partition_distinct_across_validators_covers_ab():
    """Every A∪B miner ends up in exactly one validator's foreground."""
    validator_seeds = {"v1": 1, "v2": 2, "v3": 3}
    all_miner_hotkeys = [f"m{i}" for i in range(1, 14)]
    hotkey_to_uid = {f"m{i}": i for i in range(1, 14)}
    ab_uids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}   # 13 miners
    fg1 = compute_foreground_partition(
        validator_seeds=validator_seeds, all_miner_hotkeys=all_miner_hotkeys,
        ab_uids=ab_uids, my_hotkey="v1", hotkey_to_uid=hotkey_to_uid,
    )
    fg2 = compute_foreground_partition(
        validator_seeds=validator_seeds, all_miner_hotkeys=all_miner_hotkeys,
        ab_uids=ab_uids, my_hotkey="v2", hotkey_to_uid=hotkey_to_uid,
    )
    fg3 = compute_foreground_partition(
        validator_seeds=validator_seeds, all_miner_hotkeys=all_miner_hotkeys,
        ab_uids=ab_uids, my_hotkey="v3", hotkey_to_uid=hotkey_to_uid,
    )
    assert set(fg1).isdisjoint(set(fg2))
    assert set(fg1).isdisjoint(set(fg3))
    assert set(fg2).isdisjoint(set(fg3))
    assert set(fg1) | set(fg2) | set(fg3) == ab_uids


def test_compute_foreground_partition_empty_when_no_ab():
    fg = compute_foreground_partition(
        validator_seeds={"v1": 1},
        all_miner_hotkeys=[f"m{i}" for i in range(5)],
        ab_uids=set(),
        my_hotkey="v1",
        hotkey_to_uid={f"m{i}": i for i in range(5)},
    )
    assert fg == ()


# ---------------------------------------------------------------------------
# Election ballots — top-3 of A∪B, top-2 of B∪C \ G1
# ---------------------------------------------------------------------------


def test_compute_election_ballots_top3_of_ab_into_g1():
    scores = {
        1: (0.9, 0.8),    # high mean
        2: (0.7, 0.5),
        3: (0.5, 0.4),
        4: (0.3, 0.2),
        5: (0.1, 0.0),
    }
    ballots = compute_election_ballots(
        prev_validation_a=(1, 2, 3),
        prev_validation_b=(4, 5),
        prev_validation_c=(),
        scores_over_window=scores,
        group_1_size=3,
        group_2_size=2,
    )
    assert ballots.weight_group_1 == (1, 2, 3)


def test_compute_election_ballots_top2_of_bc_excludes_g1_winners():
    # Group A = {1,2}, Group B = {3,4}, Group C = {5,6,7}.
    # G1 ballot ⊆ A∪B; G2 ballot ⊆ (B∪C)\G1.
    scores = {
        1: (0.9, 0.9),
        2: (0.8, 0.8),
        3: (0.7, 0.7),    # high in B — wins G1, must NOT appear in G2
        4: (0.5, 0.5),
        5: (0.6, 0.6),
        6: (0.4, 0.4),
        7: (0.3, 0.3),
    }
    ballots = compute_election_ballots(
        prev_validation_a=(1, 2),
        prev_validation_b=(3, 4),
        prev_validation_c=(5, 6, 7),
        scores_over_window=scores,
        group_1_size=3,
        group_2_size=2,
    )
    assert set(ballots.weight_group_1) == {1, 2, 3}
    assert 3 not in ballots.weight_group_2
    assert ballots.weight_group_2 == (5, 4)   # next-best in (B∪C)\G1 by mean


def test_election_tie_break_uses_min_and_excludes_full_ties():
    """Mean ties → fall through to min; (mean, min) ties → exclude both
    rather than picking by UID. Ballot may under-fill.
    """
    scores = {
        10: (0.5, 0.4),
        11: (0.5, 0.2),
        12: (0.5, 0.4),   # same (mean, min) as 10 → both excluded
    }
    ballots = compute_election_ballots(
        prev_validation_a=(10, 11, 12),
        prev_validation_b=(),
        prev_validation_c=(),
        scores_over_window=scores,
        group_1_size=2,
        group_2_size=0,
    )
    assert ballots.weight_group_1 == (11,)


def test_election_all_tied_yields_empty_ballot():
    """Cold start equivalent: every candidate has the same (mean, min);
    no winner can be picked without an arbitrary tiebreaker, so the
    ballot is empty.
    """
    scores = {1: (0.0, 0.0), 2: (0.0, 0.0), 3: (0.0, 0.0)}
    ballots = compute_election_ballots(
        prev_validation_a=(1, 2, 3),
        prev_validation_b=(),
        prev_validation_c=(),
        scores_over_window=scores,
        group_1_size=2,
        group_2_size=0,
    )
    assert ballots.weight_group_1 == ()


# ---------------------------------------------------------------------------
# build_cohort_groups — full assembly
# ---------------------------------------------------------------------------


def test_build_cohort_groups_invariants():
    # 4 qualified validators, 30 miners, full participation in top-3 and top-15.
    n_miners = 30
    weights = []
    base_row = [0.0] * n_miners
    for i in range(3):
        base_row[i] = 0.9 - i * 0.1
    for i in range(3, 18):
        base_row[i] = 0.05
    for _ in range(4):
        weights.append(list(base_row))
    metagraph = SimpleNamespace(
        weights=torch.tensor(weights),
        hotkeys=[f"m{i}" for i in range(n_miners)],
    )
    validator_seeds = {f"v{i}": i for i in range(4)}
    all_miner_hotkeys = [f"m{i}" for i in range(n_miners)]
    hotkey_to_uid = {f"m{i}": i for i in range(n_miners)}
    cfg = _cfg()

    groups = build_cohort_groups(
        metagraph=metagraph,
        qualified_validator_uids=[0, 1, 2, 3],
        validator_seeds=validator_seeds,
        all_miner_hotkeys=all_miner_hotkeys,
        my_hotkey="v3",
        hotkey_to_uid=hotkey_to_uid,
        election_ballots=ElectionBallots(weight_group_1=(0, 1, 2), weight_group_2=()),
        cfg=cfg,
    )
    # All validators agreed — Group A is full.
    assert len(groups.validation.group_a) == 3
    # |A| + |B| invariant.
    assert len(groups.validation.group_a) + len(groups.validation.group_b) == 13
    # Group C drawn from non-A∪B pool.
    ab_set = set(groups.validation.group_a) | set(groups.validation.group_b)
    assert set(groups.validation.group_c).isdisjoint(ab_set)
    # Foreground is a subset of A∪B (per-validator partition).
    assert set(groups.foreground_uids) <= ab_set


def test_build_cohort_groups_consensus_failure_grows_b():
    """Each validator picks disjoint top-3 winners → no miner reaches
    `min_consensus=3` qualified validators in chain-set Group 1, so
    Group A is empty and Group B grows to 13 (the |A|+|B|=13 invariant).
    """
    n_miners = 30
    weights = []
    for v in range(4):
        row = [0.0] * n_miners
        for j, m in enumerate(range(v * 3, v * 3 + 3)):
            row[m] = 0.9 - j * 0.01
        for m in range(16, 26):
            row[m] = 0.05
        weights.append(row)
    metagraph = SimpleNamespace(
        weights=torch.tensor(weights),
        hotkeys=[f"m{i}" for i in range(n_miners)],
    )
    validator_seeds = {f"v{i}": i for i in range(4)}
    all_miner_hotkeys = [f"m{i}" for i in range(n_miners)]
    cfg = _cfg(group_a_min_consensus=3)

    groups = build_cohort_groups(
        metagraph=metagraph,
        qualified_validator_uids=[0, 1, 2, 3],
        validator_seeds=validator_seeds,
        all_miner_hotkeys=all_miner_hotkeys,
        my_hotkey="v3",
        hotkey_to_uid={f"m{i}": i for i in range(n_miners)},
        election_ballots=ElectionBallots(weight_group_1=(), weight_group_2=()),
        cfg=cfg,
    )
    assert len(groups.validation.group_a) == 0
    assert len(groups.validation.group_b) == 13


# ---------------------------------------------------------------------------
# maybe_advance_cohort — boundary detection + persistence shape
# ---------------------------------------------------------------------------


def _trivial_metagraph(n: int = 30) -> SimpleNamespace:
    return SimpleNamespace(
        weights=torch.zeros((1, n)),
        hotkeys=[f"m{i}" for i in range(n)],
    )


def test_maybe_advance_cohort_holds_within_window():
    state = CohortState(
        cohort_epoch=8,
        expert_group="g1",
        weight_group_1=(1, 2, 3),
        validation_group_a=(1, 2, 3),
        validation_group_b=tuple(range(4, 14)),
        validation_group_c=tuple(range(20, 30)),
        highest_seen_cycle_index=10,
    )
    out = maybe_advance_cohort(
        cycle_index=11,           # still within [8, 15]
        round_id=10000,
        cycle_length=100,
        current_state=state,
        score_aggregator=None,
        metagraph=_trivial_metagraph(),
        qualified_validator_uids=[],
        validator_seeds={},
        all_miner_hotkeys=[],
        my_hotkey="vme",
        hotkey_to_uid={},
        expert_group="g1",
        cfg=_cfg(),
    )
    assert out is state


def test_maybe_advance_cohort_advances_at_boundary():
    state = CohortState(
        cohort_epoch=0,
        expert_group="g1",
        validation_group_a=(),
        validation_group_b=(),
        validation_group_c=(),
        highest_seen_cycle_index=7,
    )
    out = maybe_advance_cohort(
        cycle_index=8,            # crossed boundary
        round_id=10000,
        cycle_length=100,
        current_state=state,
        score_aggregator=None,    # no history → empty ballots
        metagraph=_trivial_metagraph(),
        qualified_validator_uids=[],
        validator_seeds={"vme": 0},
        all_miner_hotkeys=[],
        my_hotkey="vme",
        hotkey_to_uid={},
        expert_group="g1",
        cfg=_cfg(),
    )
    assert out.cohort_epoch == 8
    assert out.highest_seen_cycle_index == 8


def test_maybe_advance_cohort_cold_start_returns_empty_ballots():
    out = maybe_advance_cohort(
        cycle_index=0,
        round_id=0,
        cycle_length=100,
        current_state=None,
        score_aggregator=None,
        metagraph=_trivial_metagraph(),
        qualified_validator_uids=[],
        validator_seeds={"vme": 0},
        all_miner_hotkeys=[],
        my_hotkey="vme",
        hotkey_to_uid={},
        expert_group="g1",
        cfg=_cfg(),
    )
    assert out.cohort_epoch == 0
    assert out.weight_group_1 == ()
    assert out.weight_group_2 == ()


def test_maybe_advance_cohort_clamps_rollback():
    state = CohortState(
        cohort_epoch=8,
        expert_group="g1",
        highest_seen_cycle_index=12,
    )
    out = maybe_advance_cohort(
        cycle_index=5,            # < highest_seen — rollback
        round_id=100,
        cycle_length=100,
        current_state=state,
        score_aggregator=None,
        metagraph=_trivial_metagraph(),
        qualified_validator_uids=[],
        validator_seeds={},
        all_miner_hotkeys=[],
        my_hotkey="vme",
        hotkey_to_uid={},
        expert_group="g1",
        cfg=_cfg(),
    )
    assert out is state   # refused to advance


# ---------------------------------------------------------------------------
# compute_uid_weights — 97% / 3% split
# ---------------------------------------------------------------------------


def test_compute_uid_weights_g1_proportional_to_score():
    """Group 1's 97% is split in proportion to local score (was equal)."""
    weights = compute_uid_weights(
        weight_group_1=(1, 2, 3),
        weight_group_2=(),
        local_scores={1: 1.0, 2: 2.0, 3: 1.0},   # 1:2:1 ratio
    )
    assert weights[1] == pytest.approx(0.97 * 0.25)
    assert weights[2] == pytest.approx(0.97 * 0.50)
    assert weights[3] == pytest.approx(0.97 * 0.25)
    assert sum(weights.values()) == pytest.approx(0.97)


def test_compute_uid_weights_g1_falls_back_to_equal_when_all_zero():
    weights = compute_uid_weights(
        weight_group_1=(1, 2, 3),
        weight_group_2=(),
        local_scores={1: 0.0, 2: 0.0, 3: 0.0},
    )
    for uid in (1, 2, 3):
        assert weights[uid] == pytest.approx(0.97 / 3)


def test_compute_uid_weights_g2_proportional_to_score():
    weights = compute_uid_weights(
        weight_group_1=(1,),
        weight_group_2=(10, 20),
        local_scores={1: 1.0, 10: 3.0, 20: 1.0},   # G2 ratio 3:1
        group_1_share=0.97,
        group_2_share=0.03,
    )
    assert weights[1] == pytest.approx(0.97)
    assert weights[10] == pytest.approx(0.03 * 0.75)
    assert weights[20] == pytest.approx(0.03 * 0.25)


def test_compute_uid_weights_g2_falls_back_to_equal_when_all_zero():
    weights = compute_uid_weights(
        weight_group_1=(),
        weight_group_2=(10, 20),
        local_scores={10: 0.0, 20: 0.0},
    )
    assert weights[10] == pytest.approx(0.03 / 2)
    assert weights[20] == pytest.approx(0.03 / 2)


def test_compute_uid_weights_handles_empty_groups():
    assert compute_uid_weights(
        weight_group_1=(),
        weight_group_2=(),
        local_scores={},
    ) == {}


# ---------------------------------------------------------------------------
# select_top_n_by_local_score — used at step 4 weight submission
# ---------------------------------------------------------------------------


def test_select_top_n_by_local_score_orders_by_score_desc():
    pool = [10, 20, 30, 40]
    scores = {10: 1.0, 20: 3.0, 30: 2.0, 40: 0.5}
    assert select_top_n_by_local_score(pool, scores, n=3) == (20, 30, 10)


def test_select_top_n_by_local_score_skips_zero_and_missing():
    """Miners not evaluated (no entry or score=0) are skipped, not included
    with a 0-weight slot.
    """
    pool = [10, 20, 30, 40]
    scores = {10: 0.5, 30: 0.0}   # 20 absent, 40 absent, 30 zero
    assert select_top_n_by_local_score(pool, scores, n=3) == (10,)


def test_select_top_n_by_local_score_uid_asc_tiebreak():
    pool = [10, 20, 30]
    scores = {10: 1.0, 20: 1.0, 30: 1.0}
    assert select_top_n_by_local_score(pool, scores, n=3) == (10, 20, 30)


def test_select_top_n_by_local_score_returns_fewer_when_pool_underfills():
    pool = [10, 20]
    scores = {10: 1.0, 20: 2.0}
    assert select_top_n_by_local_score(pool, scores, n=5) == (20, 10)


def test_select_top_n_by_local_score_empty_pool():
    assert select_top_n_by_local_score([], {1: 1.0}, n=3) == ()


# ---------------------------------------------------------------------------
# split_foreground_background — per-cohort fg/bg partition
# ---------------------------------------------------------------------------


def _state_with_groups(
    group_a, group_b, group_c, foreground=()
) -> CohortState:
    return CohortState(
        cohort_epoch=0,
        expert_group="g1",
        validation_group_a=tuple(group_a),
        validation_group_b=tuple(group_b),
        validation_group_c=tuple(group_c),
        foreground_uids=tuple(foreground),
    )


def test_split_foreground_background_uses_persisted_foreground():
    state = _state_with_groups(
        group_a=(1, 2, 3),
        group_b=(4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
        group_c=(20, 21, 22),
        foreground=(2, 5),    # this validator's slice of A∪B
    )
    fg, bg = split_foreground_background(state)
    assert fg == (2, 5)
    # Background = (A ∪ B ∪ C) \ foreground, A→B→C order
    assert bg == (1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 20, 21, 22)
    assert len(fg) + len(bg) == 16   # full roster preserved


def test_split_foreground_background_empty_foreground_keeps_full_roster_in_bg():
    state = _state_with_groups(
        group_a=(1, 2, 3), group_b=(4, 5), group_c=(20, 21),
        foreground=(),
    )
    fg, bg = split_foreground_background(state)
    assert fg == ()
    assert bg == (1, 2, 3, 4, 5, 20, 21)


def test_split_foreground_background_preserves_a_then_b_then_c_order_in_bg():
    state = _state_with_groups(
        group_a=(10, 11, 12), group_b=(20, 21), group_c=(30, 31),
        foreground=(11,),
    )
    fg, bg = split_foreground_background(state)
    assert fg == (11,)
    assert bg == (10, 12, 20, 21, 30, 31)


def test_split_foreground_background_dedupes_overlap():
    """Defensive: if Group C accidentally contains an A∪B UID
    (shouldn't happen by construction), the fg/bg partition still
    de-duplicates.
    """
    state = _state_with_groups(
        group_a=(1,), group_b=(2,), group_c=(2, 3),   # 2 erroneously dup
        foreground=(),
    )
    _fg, bg = split_foreground_background(state)
    assert bg == (1, 2, 3)   # 2 not duplicated
