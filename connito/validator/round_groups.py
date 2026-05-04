"""Round-group construction for the tiered weight / validation scheme.

Spec: `_specs/round-group-construction-scheme.md`. This module is a pure
collection of selection helpers — no I/O, no chain calls — so it is
trivially unit-testable. All chain reads happen in `Round.freeze()` and
the orchestration layer; this module only consumes the snapshots.

Two axes:
  * **Weight groups** (Group 1 = 3 miners @ 97%, Group 2 = 15 miners @ 3%).
    Per-validator local ballots; what this validator emits on chain.
  * **Validation groups** (A = 3, B = up to 13 - |A|, C = 17). The
    `|A| + |B| = 13` invariant means Group A under-fill (consensus
    failure) grows Group B to compensate.

8-cycle cohort hold: validation Groups A/B/C and weight Groups 1/2 are
held constant for `cohort_window_cycles` (default 8) cycles, then
recomputed at the cohort boundary using the previous cohort's local
8-cycle score history (Promotion / demotion mechanics, items 14–21).

Note on circularity (spec edge case): weight Group 2 is sourced from
validation Groups B and C, and next cohort's validation Group B is the
chain-set Group 2. Within a single cycle there is no recursion — the
boundary computation reads frozen prior-cohort state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from connito.shared.app_logging import structlog

if TYPE_CHECKING:
    from connito.validator.aggregator import MinerScoreAggregator
    from connito.validator.cohort_state import CohortState

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ValidationGroups:
    """Held for the cohort. `|group_a| + |group_b|` is invariant 13."""

    group_a: tuple[int, ...]
    group_b: tuple[int, ...]
    group_c: tuple[int, ...]


@dataclass(frozen=True)
class WeightGroups:
    """This validator's local ballot, held for the cohort."""

    group_1: tuple[int, ...]
    group_2: tuple[int, ...]


@dataclass(frozen=True)
class CohortGroups:
    validation: ValidationGroups
    weight: WeightGroups


def is_cohort_boundary(cycle_index: int, window: int = 8) -> bool:
    return cycle_index % window == 0


def cohort_epoch_for(cycle_index: int, window: int = 8) -> int:
    return (cycle_index // window) * window


# ---------------------------------------------------------------------------
# Chain-set readers (operate on a metagraph snapshot already in memory)
# ---------------------------------------------------------------------------


def read_chain_set_top_k(
    metagraph,
    *,
    k: int,
    qualified_validator_uids: list[int],
    eligible_miner_uids: set[int] | None = None,
) -> dict[int, tuple[int, float]]:
    """For each qualified validator, take that validator's top-`k` miners
    by emitted weight, then tally `(count, total_weight)` per miner uid.

    Returns `{miner_uid: (validator_count, total_weight_received)}`. Sort
    by `(validator_count, total_weight)` descending to get the chain-set
    Group `k` ranking.

    Used twice per cohort boundary: `k=1` for chain-set Group 1 (consensus
    candidates for validation Group A) and `k=15` for chain-set Group 2
    (candidates for validation Group B). The legacy weight-prepend code at
    `round.py:233-287` is a near-duplicate of this — it's kept until the
    feature flag flips so existing tests stay green; PR 5 collapses both
    to one caller.
    """
    weights_attr = getattr(metagraph, "weights", None)
    if weights_attr is None:
        logger.debug(
            "round_groups.read_chain_set_top_k: metagraph.weights missing — "
            "returning empty tally (caller likely fetched lite=True)",
        )
        return {}

    try:
        W = weights_attr if isinstance(weights_attr, torch.Tensor) \
            else torch.as_tensor(weights_attr)
    except Exception:
        logger.warning(
            "round_groups.read_chain_set_top_k: failed to coerce metagraph.weights to tensor",
        )
        return {}

    if W.ndim != 2 or W.shape[0] == 0:
        return {}

    n_rows = int(W.shape[0])
    tally: dict[int, tuple[int, float]] = {}

    for v_uid in qualified_validator_uids:
        if v_uid < 0 or v_uid >= n_rows:
            continue
        row = W[v_uid]
        # Sparse in practice — most validators reward a small subset. Pull
        # nonzero indices, then take the top-k by weight on this row.
        nonzero_idx = torch.nonzero(row, as_tuple=True)[0].tolist()
        if not nonzero_idx:
            continue
        scored = [(int(m_uid), float(row[m_uid].item())) for m_uid in nonzero_idx]
        if eligible_miner_uids is not None:
            scored = [(m, w) for m, w in scored if m in eligible_miner_uids]
        if not scored:
            continue
        scored.sort(key=lambda mw: mw[1], reverse=True)
        for m_uid, w in scored[:k]:
            count, total = tally.get(m_uid, (0, 0.0))
            tally[m_uid] = (count + 1, total + w)

    return tally


# ---------------------------------------------------------------------------
# Validation group construction (per cohort boundary)
# ---------------------------------------------------------------------------


def compute_group_a(
    chain_set_top1: dict[int, tuple[int, float]],
    *,
    min_consensus: int = 3,
    max_size: int = 3,
) -> tuple[int, ...]:
    """Return up to `max_size` UIDs that have ≥ `min_consensus` qualified
    validators placing them in their weight Group 1.

    May under-fill — that's the Group A consensus-failure case (spec edge
    case). The freed slots are absorbed by Group B via `compute_group_b`.
    Tiebreak: higher total weight, then lower UID (deterministic).
    """
    eligible = [
        (uid, count, total)
        for uid, (count, total) in chain_set_top1.items()
        if count >= min_consensus
    ]
    eligible.sort(key=lambda x: (-x[1], -x[2], x[0]))
    return tuple(uid for uid, _, _ in eligible[:max_size])


def compute_group_b(
    chain_set_top2: dict[int, tuple[int, float]],
    *,
    group_a: tuple[int, ...],
    ab_total: int = 13,
    exclude: set[int] | None = None,
) -> tuple[int, ...]:
    """Top miners from chain-set Group 2 by validator-count (NOT
    stake-weighted), with weight-received as tiebreaker.

    Size = `ab_total - len(group_a)` so the `|A|+|B|=ab_total` invariant
    holds even when Group A under-fills.
    """
    target_size = max(0, ab_total - len(group_a))
    blocklist: set[int] = set(group_a)
    if exclude:
        blocklist |= exclude
    eligible = [
        (uid, count, total)
        for uid, (count, total) in chain_set_top2.items()
        if uid not in blocklist
    ]
    eligible.sort(key=lambda x: (-x[1], -x[2], x[0]))
    return tuple(uid for uid, _, _ in eligible[:target_size])


def compute_group_c(
    *,
    assignment: dict[str, list[str]],
    my_hotkey: str,
    hotkey_to_uid: dict[str, int],
    exclude: set[int],
    max_size: int = 17,
) -> tuple[int, ...]:
    """Per-validator Group C drawn from `validator_miner_assignment`.

    Excludes UIDs already in Groups A and B so a miner cannot occupy two
    validation slots in the same cohort. Distinct across validators by
    construction (each validator has a different `my_hotkey` slice).
    """
    my_slice = assignment.get(my_hotkey, [])
    out: list[int] = []
    for hk in my_slice:
        uid = hotkey_to_uid.get(hk)
        if uid is None or uid in exclude:
            continue
        out.append(uid)
        if len(out) >= max_size:
            break
    return tuple(out)


# ---------------------------------------------------------------------------
# Election ballot (cohort boundary, drives next cohort's weight groups)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ElectionBallots:
    """Local ballots produced at a cohort boundary from 8-cycle history."""

    weight_group_1: tuple[int, ...]
    weight_group_2: tuple[int, ...]


def _rank_by_mean_then_min_excluding_ties(
    candidates: dict[int, tuple[float, float]],
    take: int,
) -> tuple[int, ...]:
    """Sort by mean desc → min cycle score desc, then drop any candidate
    whose `(mean, min)` matches another candidate's.

    The lower-UID fallback from an earlier draft was removed: picking by
    UID would give a systematic advantage that has nothing to do with
    miner performance. Tied candidates instead miss the ballot, so the
    ballot may under-fill (return < `take`). Determinism is preserved —
    every validator with identical inputs excludes the same UIDs.

    Cold-start consequence: if every candidate has `(0.0, 0.0)`, all of
    them tie and the ballot is empty. This is the desired behavior; the
    next cohort election will run on real scores.
    """
    score_counts: dict[tuple[float, float], int] = {}
    for score in candidates.values():
        score_counts[score] = score_counts.get(score, 0) + 1
    unique = [
        (uid, score)
        for uid, score in candidates.items()
        if score_counts[score] == 1
    ]
    unique.sort(key=lambda kv: (-kv[1][0], -kv[1][1]))
    return tuple(uid for uid, _ in unique[:take])


def compute_election_ballots(
    *,
    prev_validation_a: tuple[int, ...],
    prev_validation_b: tuple[int, ...],
    prev_validation_c: tuple[int, ...],
    scores_over_window: dict[int, tuple[float, float]],
    group_1_size: int = 3,
    group_2_size: int = 15,
) -> ElectionBallots:
    """Compute this validator's local ballots at the cohort boundary.

    `scores_over_window[uid] = (mean, min_per_cycle)` over the previous
    cohort's 8-cycle window — see `MinerScoreAggregator.scores_over_window`.

    Group 1 ballot = top-`group_1_size` of (prev A ∪ prev B) by mean.
    Group 2 ballot = top-`group_2_size` of (prev B ∪ prev C) by mean,
    excluding anyone already on the Group 1 ballot.
    """
    ab_pool = set(prev_validation_a) | set(prev_validation_b)
    bc_pool = set(prev_validation_b) | set(prev_validation_c)

    ab_scores = {uid: scores_over_window.get(uid, (0.0, 0.0)) for uid in ab_pool}
    g1 = _rank_by_mean_then_min_excluding_ties(ab_scores, group_1_size)

    bc_remaining = {
        uid: scores_over_window.get(uid, (0.0, 0.0))
        for uid in bc_pool
        if uid not in g1
    }
    g2 = _rank_by_mean_then_min_excluding_ties(bc_remaining, group_2_size)

    return ElectionBallots(weight_group_1=g1, weight_group_2=g2)


# ---------------------------------------------------------------------------
# Single entry point used by Round.freeze()
# ---------------------------------------------------------------------------


def build_cohort_groups(
    *,
    metagraph,
    qualified_validator_uids: list[int],
    eligible_miner_uids: set[int],
    assignment: dict[str, list[str]],
    my_hotkey: str,
    hotkey_to_uid: dict[str, int],
    election_ballots: ElectionBallots,
    cfg,
) -> CohortGroups:
    """Compose validation Groups A/B/C and weight Groups 1/2 for a new cohort.

    Called only at the cohort boundary. Within a cohort the cached
    `CohortState` is reused; this function is what produces that state.

    Group A and Group B are derived from the chain-set tally of every
    qualified validator's election ballots (i.e. the previous cohort's
    weight Group 1 / Group 2 votes that landed on chain). The validator's
    own weight ballots come straight from `election_ballots`.
    """
    chain_top1 = read_chain_set_top_k(
        metagraph,
        k=cfg.weight_group_1_size,
        qualified_validator_uids=qualified_validator_uids,
        eligible_miner_uids=eligible_miner_uids,
    )
    chain_top2 = read_chain_set_top_k(
        metagraph,
        k=cfg.weight_group_2_size,
        qualified_validator_uids=qualified_validator_uids,
        eligible_miner_uids=eligible_miner_uids,
    )

    group_a = compute_group_a(
        chain_top1,
        min_consensus=cfg.group_a_min_consensus,
        max_size=cfg.validation_group_a_size,
    )
    group_b = compute_group_b(
        chain_top2,
        group_a=group_a,
        ab_total=cfg.validation_group_ab_total,
    )
    placed = set(group_a) | set(group_b)
    group_c = compute_group_c(
        assignment=assignment,
        my_hotkey=my_hotkey,
        hotkey_to_uid=hotkey_to_uid,
        exclude=placed,
        max_size=cfg.validation_group_c_size,
    )

    validation = ValidationGroups(group_a=group_a, group_b=group_b, group_c=group_c)
    weight = WeightGroups(
        group_1=election_ballots.weight_group_1,
        group_2=election_ballots.weight_group_2,
    )

    logger.info(
        "round_groups.build_cohort_groups",
        group_a=list(group_a),
        group_b_size=len(group_b),
        group_c_size=len(group_c),
        weight_group_1=list(weight.group_1),
        weight_group_2_size=len(weight.group_2),
    )

    return CohortGroups(validation=validation, weight=weight)


# ---------------------------------------------------------------------------
# Cohort lifecycle (boundary detection + ballot + new-state assembly)
# ---------------------------------------------------------------------------


def maybe_advance_cohort(
    *,
    cycle_index: int,
    round_id: int,
    cycle_length: int,
    current_state: "CohortState | None",
    score_aggregator: "MinerScoreAggregator | None",
    metagraph,
    qualified_validator_uids: list[int],
    eligible_miner_uids: set[int],
    assignment: dict[str, list[str]],
    my_hotkey: str,
    hotkey_to_uid: dict[str, int],
    expert_group: str,
    cfg,
) -> "CohortState":
    """Return the `CohortState` to use for this cycle.

    If `cycle_index` is still within the current cohort window, returns
    `current_state` unchanged (the 8-cycle hold). At a cohort boundary —
    or on first run — runs the election from the previous cohort's
    score history and assembles a fresh `CohortState`.

    Cold start (`current_state is None`) emits empty ballots and lets
    `build_cohort_groups` construct Groups A/B/C from chain state alone.
    Spec edge case "Cold start": promotion rules are skipped during the
    first 8 cycles, which is the natural consequence of empty ballots.

    Rollback guard: refuses to advance if `cycle_index` is below the
    persisted `highest_seen_cycle_index` — chain reorg or owner-API
    drift cannot rewind the cohort.
    """
    from connito.validator.cohort_state import CohortState

    window = cfg.cohort_window_cycles
    new_epoch = cohort_epoch_for(cycle_index, window)

    if current_state is not None:
        if cycle_index < current_state.highest_seen_cycle_index:
            logger.warning(
                "round_groups.maybe_advance_cohort: cycle_index < highest_seen — clamping",
                cycle_index=cycle_index,
                highest_seen=current_state.highest_seen_cycle_index,
            )
            return current_state
        if current_state.cohort_epoch == new_epoch:
            return current_state

    # Cohort boundary or first run.
    if current_state is None or score_aggregator is None:
        ballots = ElectionBallots(weight_group_1=(), weight_group_2=())
    else:
        cycles_in_window = list(
            range(current_state.cohort_epoch, current_state.cohort_epoch + window)
        )
        candidate_uids = (
            set(current_state.validation_group_a)
            | set(current_state.validation_group_b)
            | set(current_state.validation_group_c)
        )
        if candidate_uids:
            round_id_to_cycle_index = {
                rid: rid // cycle_length
                for rid in score_aggregator.all_round_ids()
            }
            scores_over_window = score_aggregator.scores_over_window(
                uids=candidate_uids,
                round_id_to_cycle_index=round_id_to_cycle_index,
                cycles_in_window=cycles_in_window,
            )
            ballots = compute_election_ballots(
                prev_validation_a=current_state.validation_group_a,
                prev_validation_b=current_state.validation_group_b,
                prev_validation_c=current_state.validation_group_c,
                scores_over_window=scores_over_window,
                group_1_size=cfg.weight_group_1_size,
                group_2_size=cfg.weight_group_2_size,
            )
        else:
            ballots = ElectionBallots(weight_group_1=(), weight_group_2=())

    new_groups = build_cohort_groups(
        metagraph=metagraph,
        qualified_validator_uids=qualified_validator_uids,
        eligible_miner_uids=eligible_miner_uids,
        assignment=assignment,
        my_hotkey=my_hotkey,
        hotkey_to_uid=hotkey_to_uid,
        election_ballots=ballots,
        cfg=cfg,
    )

    highest_seen = max(
        cycle_index,
        current_state.highest_seen_cycle_index if current_state else 0,
    )

    new_state = CohortState(
        cohort_epoch=new_epoch,
        expert_group=expert_group,
        weight_group_1=new_groups.weight.group_1,
        weight_group_2=new_groups.weight.group_2,
        validation_group_a=new_groups.validation.group_a,
        validation_group_b=new_groups.validation.group_b,
        validation_group_c=new_groups.validation.group_c,
        last_election_round_id=round_id,
        highest_seen_cycle_index=highest_seen,
    )

    logger.info(
        "round_groups.maybe_advance_cohort: new cohort",
        cohort_epoch=new_state.cohort_epoch,
        cycle_index=cycle_index,
        weight_group_1=list(new_state.weight_group_1),
        weight_group_2_size=len(new_state.weight_group_2),
        validation_group_a=list(new_state.validation_group_a),
        validation_group_b_size=len(new_state.validation_group_b),
        validation_group_c_size=len(new_state.validation_group_c),
    )
    return new_state


def split_validation_uids_into_foreground(
    state: "CohortState",
) -> tuple[int, ...]:
    """Foreground evaluation order: Group A → Group B → Group C.

    `Round.freeze()` overrides `foreground_uids` with this when the
    feature flag is on; `background_uids` becomes empty since the
    30-miner budget is fully covered.
    """
    return tuple([*state.validation_group_a, *state.validation_group_b, *state.validation_group_c])


def compute_uid_weights(
    *,
    weight_group_1: tuple[int, ...],
    weight_group_2: tuple[int, ...],
    local_scores: dict[int, float],
    group_1_share: float = 0.97,
    group_2_share: float = 0.03,
) -> dict[int, float]:
    """Build the chain-submission weight map per spec items 2 and 4.

    Group 1 (97%) is split *equally* among its members — spec item 2
    says only "split among its 3 members", so equal share is the
    natural reading and avoids amplifying a small score gap into a
    large emission gap at the top of the leaderboard.

    Group 2 (3%) is split *in proportion to local score*. If every
    Group 2 member scored 0.0 in the previous window, fall back to an
    equal split so the emission still goes out and the chain-set Group 2
    tally still moves.

    Everyone not in either group receives 0.0; the chain submitter
    normalizes to 1.0 before sending.
    """
    out: dict[int, float] = {}

    if weight_group_1:
        per_member = group_1_share / len(weight_group_1)
        for uid in weight_group_1:
            out[uid] = out.get(uid, 0.0) + per_member

    if weight_group_2:
        scores = [(uid, max(0.0, float(local_scores.get(uid, 0.0)))) for uid in weight_group_2]
        total = sum(s for _, s in scores)
        if total > 0.0:
            for uid, s in scores:
                out[uid] = out.get(uid, 0.0) + group_2_share * (s / total)
        else:
            per_member = group_2_share / len(weight_group_2)
            for uid in weight_group_2:
                out[uid] = out.get(uid, 0.0) + per_member

    return out
