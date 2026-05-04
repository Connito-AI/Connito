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
    compute_group_a,
    compute_group_b,
    compute_group_c,
    compute_stake_weighted_total,
    compute_uid_weights,
    is_cohort_boundary,
    maybe_advance_cohort,
    read_chain_set_top_k,
    split_validation_uids,
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
        group_a_min_consensus=3,
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
    # Validator 0 puts weight on miners {2:0.5, 3:0.3}
    # Validator 1 puts weight on miners {2:0.4, 4:0.2}
    # Validator 2 puts weight on miners {3:0.6, 4:0.1}
    weights = [
        [0.0, 0.0, 0.5, 0.3, 0.0],
        [0.0, 0.0, 0.4, 0.0, 0.2],
        [0.0, 0.0, 0.0, 0.6, 0.1],
    ]
    metagraph = _metagraph_with_weights(weights)
    tally = read_chain_set_top_k(
        metagraph, k=2, qualified_validator_uids=[0, 1, 2]
    )
    assert tally[2] == (2, pytest.approx(0.9))    # v0 + v1
    assert tally[3] == (2, pytest.approx(0.9))    # v0 + v2
    assert tally[4] == (2, pytest.approx(0.3))    # v1 + v2


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


def test_compute_group_a_returns_consensus_uids():
    chain_top1 = {
        10: (3, 1.5),
        11: (4, 2.0),
        12: (2, 1.0),    # below min_consensus
        13: (3, 1.2),
    }
    g_a = compute_group_a(chain_top1, min_consensus=3, max_size=3)
    assert set(g_a) == {11, 10, 13}
    assert g_a[0] == 11   # highest validator-count first


def test_compute_group_a_under_fills_when_consensus_short():
    chain_top1 = {10: (3, 1.0), 11: (2, 0.5), 12: (1, 0.1)}
    g_a = compute_group_a(chain_top1, min_consensus=3, max_size=3)
    assert g_a == (10,)   # only one passes the bar


def test_compute_group_a_empty_when_no_consensus():
    chain_top1 = {10: (1, 0.9), 11: (2, 0.5)}
    assert compute_group_a(chain_top1, min_consensus=3, max_size=3) == ()


def test_compute_group_a_tie_break_by_total_weight_then_uid():
    chain_top1 = {
        10: (3, 1.0),
        11: (3, 2.0),    # higher total weight wins
        12: (3, 1.0),    # same count+weight as 10 → uid asc
    }
    g_a = compute_group_a(chain_top1, min_consensus=3, max_size=3)
    assert g_a == (11, 10, 12)


# ---------------------------------------------------------------------------
# Group B — invariant |A|+|B| = ab_total
# ---------------------------------------------------------------------------


def test_compute_group_b_grows_when_group_a_underfills():
    chain_top2 = {uid: (5, 1.0) for uid in range(20, 40)}
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
    chain_top2 = {uid: (5, 1.0) for uid in range(10)}
    g_b = compute_group_b(chain_top2, group_a=(1, 2, 3), ab_total=13)
    assert 1 not in g_b and 2 not in g_b and 3 not in g_b


def test_compute_group_b_ranks_by_validator_count_then_weight_then_uid():
    chain_top2 = {
        20: (4, 1.0),
        21: (5, 0.5),    # higher count beats higher weight
        22: (4, 2.0),    # same count as 20 but higher weight
        23: (4, 1.0),    # same as 20 → tiebreak uid asc → 20 wins
    }
    g_b = compute_group_b(chain_top2, group_a=(), ab_total=4)
    assert g_b == (21, 22, 20, 23)


# ---------------------------------------------------------------------------
# Group C — distinct across validators
# ---------------------------------------------------------------------------


def test_compute_group_c_returns_my_assignment_slice_minus_excluded():
    assignment = {
        "v1": ["m1", "m2", "m3", "m4", "m5"],
        "v2": ["m6", "m7", "m8"],
    }
    hotkey_to_uid = {f"m{i}": i for i in range(1, 9)}
    g_c = compute_group_c(
        assignment=assignment,
        my_hotkey="v1",
        hotkey_to_uid=hotkey_to_uid,
        exclude={1, 2},   # uids 1 and 2 are in Group A or B
        max_size=17,
    )
    assert g_c == (3, 4, 5)


def test_compute_group_c_distinct_across_validators():
    assignment = {
        "v1": ["m1", "m2", "m3"],
        "v2": ["m4", "m5", "m6"],
    }
    hotkey_to_uid = {f"m{i}": i for i in range(1, 7)}
    c1 = compute_group_c(assignment=assignment, my_hotkey="v1",
                         hotkey_to_uid=hotkey_to_uid, exclude=set())
    c2 = compute_group_c(assignment=assignment, my_hotkey="v2",
                         hotkey_to_uid=hotkey_to_uid, exclude=set())
    assert set(c1).isdisjoint(set(c2))


def test_compute_group_c_caps_at_max_size():
    assignment = {"v1": [f"m{i}" for i in range(50)]}
    hotkey_to_uid = {f"m{i}": i for i in range(50)}
    g_c = compute_group_c(
        assignment=assignment, my_hotkey="v1",
        hotkey_to_uid=hotkey_to_uid, exclude=set(), max_size=17,
    )
    assert len(g_c) == 17


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
    # 4 qualified validators, 30 miners, full participation in top-1 and top-15.
    n_miners = 30
    weights = []
    # Each validator votes the same top-3 to weight Group 1, and the next 15 to Group 2.
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
    assignment = {
        "v0": ["m0", "m1", "m2", "m3"],
        "v1": ["m4", "m5", "m6"],
        "v2": ["m7", "m8"],
        "v3": [
            f"m{i}" for i in range(20, 30)
        ],
    }
    hotkey_to_uid = {f"m{i}": i for i in range(n_miners)}
    cfg = _cfg()

    groups = build_cohort_groups(
        metagraph=metagraph,
        qualified_validator_uids=[0, 1, 2, 3],
        eligible_miner_uids=set(range(n_miners)),
        assignment=assignment,
        my_hotkey="v3",
        hotkey_to_uid=hotkey_to_uid,
        election_ballots=ElectionBallots(weight_group_1=(0, 1, 2), weight_group_2=()),
        cfg=cfg,
    )
    # All validators agreed — Group A is full.
    assert len(groups.validation.group_a) == 3
    # |A| + |B| invariant.
    assert len(groups.validation.group_a) + len(groups.validation.group_b) == 13
    # Group C drawn from this validator's slice, excluding A and B.
    assert set(groups.validation.group_c).isdisjoint(set(groups.validation.group_a))
    assert set(groups.validation.group_c).isdisjoint(set(groups.validation.group_b))


def test_build_cohort_groups_consensus_failure_grows_b():
    """Each validator picks disjoint top-3 winners → no miner reaches
    `min_consensus=3` qualified validators in chain-set Group 1, so
    Group A is empty and Group B grows to 13 (the |A|+|B|=13 invariant).
    """
    n_miners = 30
    weights = []
    for v in range(4):
        row = [0.0] * n_miners
        # 3 high-weight votes, distinct per validator (no overlap → no consensus)
        for j, m in enumerate(range(v * 3, v * 3 + 3)):
            row[m] = 0.9 - j * 0.01
        # 10 low-weight shared votes so chain-set Group 2 (top-15) has
        # candidates that DO have consensus and can fill Group B.
        for m in range(16, 26):
            row[m] = 0.05
        weights.append(row)
    metagraph = SimpleNamespace(
        weights=torch.tensor(weights),
        hotkeys=[f"m{i}" for i in range(n_miners)],
    )
    assignment = {f"v{i}": [] for i in range(4)}
    assignment["v3"] = [f"m{i}" for i in range(20, 30)]
    cfg = _cfg(group_a_min_consensus=3)

    groups = build_cohort_groups(
        metagraph=metagraph,
        qualified_validator_uids=[0, 1, 2, 3],
        eligible_miner_uids=set(range(n_miners)),
        assignment=assignment,
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
        eligible_miner_uids=set(),
        assignment={},
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
        eligible_miner_uids=set(),
        assignment={"vme": []},
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
        eligible_miner_uids=set(),
        assignment={"vme": []},
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
        eligible_miner_uids=set(),
        assignment={},
        my_hotkey="vme",
        hotkey_to_uid={},
        expert_group="g1",
        cfg=_cfg(),
    )
    assert out is state   # refused to advance


# ---------------------------------------------------------------------------
# compute_uid_weights — 97% / 3% split
# ---------------------------------------------------------------------------


def test_compute_uid_weights_97_3_split_with_equal_g1():
    weights = compute_uid_weights(
        weight_group_1=(1, 2, 3),
        weight_group_2=(4, 5),
        local_scores={4: 1.0, 5: 1.0},
    )
    # Group 1 split equally: 0.97 / 3 each
    for uid in (1, 2, 3):
        assert weights[uid] == pytest.approx(0.97 / 3)
    # Group 2 split equally (scores tied)
    for uid in (4, 5):
        assert weights[uid] == pytest.approx(0.03 / 2)
    # Sums to 1.0 (no normalization needed at this layer)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_compute_uid_weights_g2_proportional_to_score():
    weights = compute_uid_weights(
        weight_group_1=(1,),
        weight_group_2=(10, 20),
        local_scores={10: 3.0, 20: 1.0},   # 3:1 ratio
        group_1_share=0.97,
        group_2_share=0.03,
    )
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
# split_validation_uids — foreground/background priority partition
# ---------------------------------------------------------------------------


def _state_with_groups(group_a, group_b, group_c) -> CohortState:
    return CohortState(
        cohort_epoch=0,
        expert_group="g1",
        validation_group_a=tuple(group_a),
        validation_group_b=tuple(group_b),
        validation_group_c=tuple(group_c),
    )


def test_split_validation_uids_foreground_is_ab_intersect_assignment():
    """Foreground = (A ∪ B) ∩ my_assignment; Group C never enters foreground."""
    state = _state_with_groups(
        group_a=(1, 2, 3),
        group_b=(4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
        group_c=(20, 21, 22, 23, 24),
    )
    # validator vme assigned uids {2, 5, 7, 21}: 2 in A, 5/7 in B, 21 in C.
    metagraph = SimpleNamespace(
        weights=torch.zeros((1, 30)),
        S=torch.ones(1),
    )
    fg, bg = split_validation_uids(
        state,
        metagraph=metagraph,
        qualified_validator_uids=[],
        my_assignment_uids={2, 5, 7, 21},
    )
    assert set(fg) == {2, 5, 7}        # 21 is in C, not in foreground
    assert 21 in set(bg)               # 21 lands in background
    # |fg| + |bg| = full roster (no UID is dropped).
    assert len(fg) + len(bg) == len(state.validation_group_a) + len(
        state.validation_group_b
    ) + len(state.validation_group_c)


def test_split_validation_uids_foreground_sorted_by_stake_weight_desc():
    """Higher stake-weighted total weight ranks first in foreground."""
    state = _state_with_groups(
        group_a=(1, 2, 3), group_b=(4,), group_c=()
    )
    # 2 validators with stakes 5.0 and 10.0; both vote on miners 1, 2.
    weights = [
        [0.0, 0.5, 0.2, 0.0, 0.0],  # validator 0
        [0.0, 0.1, 0.9, 0.0, 0.0],  # validator 1
    ]
    metagraph = SimpleNamespace(
        weights=torch.tensor(weights),
        S=torch.tensor([5.0, 10.0]),
    )
    # Stake-weighted: m1 = 5*0.5 + 10*0.1 = 3.5; m2 = 5*0.2 + 10*0.9 = 10.0
    # → m2 ranks before m1.
    fg, _bg = split_validation_uids(
        state,
        metagraph=metagraph,
        qualified_validator_uids=[0, 1],
        my_assignment_uids={1, 2},
    )
    assert fg == (2, 1)


def test_split_validation_uids_empty_foreground_when_no_overlap():
    """If none of A∪B is in this validator's assignment, foreground is empty
    and background is the full roster.
    """
    state = _state_with_groups(
        group_a=(1, 2, 3), group_b=(4, 5), group_c=(20, 21)
    )
    metagraph = SimpleNamespace(weights=torch.zeros((1, 30)), S=torch.ones(1))
    fg, bg = split_validation_uids(
        state,
        metagraph=metagraph,
        qualified_validator_uids=[],
        my_assignment_uids={20, 21},   # only Group C
    )
    assert fg == ()
    assert set(bg) == {1, 2, 3, 4, 5, 20, 21}


def test_split_validation_uids_background_preserves_a_then_b_then_c_order():
    state = _state_with_groups(
        group_a=(10, 11, 12), group_b=(20, 21), group_c=(30, 31)
    )
    metagraph = SimpleNamespace(weights=torch.zeros((1, 50)), S=torch.ones(1))
    fg, bg = split_validation_uids(
        state,
        metagraph=metagraph,
        qualified_validator_uids=[],
        my_assignment_uids={11},   # only one foreground UID
    )
    assert fg == (11,)
    # Background should be the remaining roster in A→B→C order.
    assert bg == (10, 12, 20, 21, 30, 31)


def test_compute_stake_weighted_total_sums_stake_times_weight():
    weights = [
        [0.0, 0.5, 0.2],
        [0.0, 0.1, 0.9],
    ]
    metagraph = SimpleNamespace(
        weights=torch.tensor(weights),
        S=torch.tensor([4.0, 2.0]),
    )
    out = compute_stake_weighted_total(
        metagraph,
        target_uids={1, 2},
        qualified_validator_uids=[0, 1],
    )
    # m1 = 4*0.5 + 2*0.1 = 2.2 ; m2 = 4*0.2 + 2*0.9 = 2.6
    assert out[1] == pytest.approx(2.2)
    assert out[2] == pytest.approx(2.6)


def test_compute_stake_weighted_total_zero_when_no_weights_attr():
    metagraph = SimpleNamespace()
    out = compute_stake_weighted_total(
        metagraph, target_uids={1, 2, 3}, qualified_validator_uids=[0, 1]
    )
    assert out == {1: 0.0, 2: 0.0, 3: 0.0}
