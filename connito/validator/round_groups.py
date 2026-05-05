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
    foreground_uids: tuple[int, ...] = ()


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
) -> dict[int, tuple[int, float, float]]:
    """For each qualified validator, take that validator's top-`k` miners
    by emitted weight, then tally `(count, total_weight, max_weight)` per
    miner uid.

    Returns `{miner_uid: (validator_count, total_weight_received,
    max_weight_from_one_validator)}`.

    The `max_weight` field lets `compute_group_a` enforce a per-validator
    weight floor (e.g. "must receive > 3% from at least one validator")
    in addition to the count floor. Without it, a miner could land in
    Group A purely on count via many tiny weight emissions.

    Sort by `(total_weight, validator_count)` descending to get the
    chain-set Group `k` ranking — total weight is the primary signal,
    validator count is a secondary tiebreaker.

    Used twice per cohort boundary: `k=3` for chain-set Group 1 (consensus
    candidates for validation Group A) and `k=15` for chain-set Group 2
    (candidates for validation Group B).
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
    tally: dict[int, tuple[int, float, float]] = {}

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
            count, total, max_w = tally.get(m_uid, (0, 0.0, 0.0))
            tally[m_uid] = (count + 1, total + w, max(max_w, w))

    return tally


# ---------------------------------------------------------------------------
# Validation group construction (per cohort boundary)
# ---------------------------------------------------------------------------


def compute_group_a(
    chain_set_top1: dict[int, tuple[int, float, float]],
    *,
    min_consensus: int = 1,
    min_weight_per_validator: float = 0.03,
    max_size: int = 3,
) -> tuple[int, ...]:
    """Return up to `max_size` UIDs that pass two gates:

      1. **Count floor:** at least `min_consensus` qualified validators
         placed them in their weight Group 1.
      2. **Per-validator weight floor:** at least one of those validators
         emitted more than `min_weight_per_validator` (default 3%) of
         their weight to this miner. Prevents a miner from landing in
         Group A purely via many tiny emissions; Group A is meant to be
         the "this miner is genuinely top tier" set.

    May under-fill — that's the Group A consensus-failure case (spec edge
    case). The freed slots are absorbed by Group B via `compute_group_b`.
    Sort: total weight desc → validator count desc → lower UID.
    """
    eligible = [
        (uid, count, total, max_w)
        for uid, (count, total, max_w) in chain_set_top1.items()
        if count >= min_consensus and max_w > min_weight_per_validator
    ]
    eligible.sort(key=lambda x: (-x[2], -x[1], x[0]))
    return tuple(uid for uid, _, _, _ in eligible[:max_size])


def compute_group_b(
    chain_set_top2: dict[int, tuple[int, float, float]],
    *,
    group_a: tuple[int, ...],
    ab_total: int = 13,
    exclude: set[int] | None = None,
) -> tuple[int, ...]:
    """Top miners from chain-set Group 2 by total weight received (NOT
    stake-weighted), with validator-count as tiebreaker.

    Size = `ab_total - len(group_a)` so the `|A|+|B|=ab_total` invariant
    holds even when Group A under-fills. The `max_weight` field of the
    chain-set tally is ignored here — Group B has no per-validator
    weight floor.
    """
    target_size = max(0, ab_total - len(group_a))
    blocklist: set[int] = set(group_a)
    if exclude:
        blocklist |= exclude
    eligible = [
        (uid, count, total)
        for uid, (count, total, _max_w) in chain_set_top2.items()
        if uid not in blocklist
    ]
    eligible.sort(key=lambda x: (-x[2], -x[1], x[0]))
    return tuple(uid for uid, _, _ in eligible[:target_size])


def _partition_pool(
    *,
    validator_seeds: dict[str, int],
    miner_hotkeys: list[str],
    my_hotkey: str,
    hotkey_to_uid: dict[str, int],
    max_per_validator: int | None,
) -> tuple[int, ...]:
    """Run the seeded `assign_miners_to_validators` distribution over
    `miner_hotkeys` and return this validator's slice as UIDs.

    Wrapper over `connito.shared.cycle.assign_miners_to_validators` so
    the round-group scheme can re-partition arbitrary miner pools (A∪B
    for foreground, all-minus-A∪B for Group C) without re-issuing chain
    RPCs. `validator_seeds` and `miner_hotkeys` together fully determine
    the partition; every validator with the same inputs gets the same
    answer.
    """
    from connito.shared.cycle import assign_miners_to_validators
    if not miner_hotkeys or not validator_seeds:
        return ()
    assignment = assign_miners_to_validators(
        validator_seeds, list(miner_hotkeys),
        max_miners_per_validator=max_per_validator,
    )
    my_slice = assignment.get(my_hotkey, [])
    # `assign_miners_to_validators` treats `max_miners_per_validator` as a
    # soft cap — overflow miners that exceed total capacity get pushed to
    # `prefs[-1]`, which can put a validator over the cap. The round-group
    # scheme wants a hard cap (it bounds this validator's eval budget), so
    # truncate here. The slice order is deterministic from the seeded
    # distribution, so the truncation is stable across runs.
    if max_per_validator is not None and len(my_slice) > max_per_validator:
        my_slice = my_slice[:max_per_validator]
    return tuple(
        hotkey_to_uid[hk] for hk in my_slice if hk in hotkey_to_uid
    )


def compute_group_c(
    *,
    validator_seeds: dict[str, int],
    all_miner_hotkeys: list[str],
    ab_uids: set[int],
    my_hotkey: str,
    hotkey_to_uid: dict[str, int],
    max_size: int = 17,
) -> tuple[int, ...]:
    """Group C — the exploration tier, distinct across validators.

    Computed by partitioning the pool of *non-A∪B* miners across
    qualified validators using the same seeded distribution as
    `validator_miner_assignment`, then taking this validator's slice
    capped at `max_size`. Disjoint from A∪B by construction (the input
    pool excludes them).
    """
    non_ab_hotkeys = [
        hk for hk in all_miner_hotkeys
        if hotkey_to_uid.get(hk) not in ab_uids
    ]
    return _partition_pool(
        validator_seeds=validator_seeds,
        miner_hotkeys=non_ab_hotkeys,
        my_hotkey=my_hotkey,
        hotkey_to_uid=hotkey_to_uid,
        max_per_validator=max_size,
    )


def compute_foreground_partition(
    *,
    validator_seeds: dict[str, int],
    all_miner_hotkeys: list[str],
    ab_uids: set[int],
    my_hotkey: str,
    hotkey_to_uid: dict[str, int],
    max_size: int | None = None,
) -> tuple[int, ...]:
    """Foreground — per-validator partition of A∪B miners.

    Uses the same seeded `assign_miners_to_validators` distribution as
    `validator_miner_assignment`, but restricted to the chain-consensus
    pool (A∪B). Each validator gets a deterministic, balanced slice of
    A∪B regardless of whether those miners landed in its full-pool
    assignment.
    """
    ab_hotkeys = [
        hk for hk in all_miner_hotkeys
        if hotkey_to_uid.get(hk) in ab_uids
    ]
    return _partition_pool(
        validator_seeds=validator_seeds,
        miner_hotkeys=ab_hotkeys,
        my_hotkey=my_hotkey,
        hotkey_to_uid=hotkey_to_uid,
        max_per_validator=max_size,
    )


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
    validator_seeds: dict[str, int],
    all_miner_hotkeys: list[str],
    my_hotkey: str,
    hotkey_to_uid: dict[str, int],
    election_ballots: ElectionBallots,
    cfg,
) -> CohortGroups:
    """Compose validation Groups A/B/C, weight Groups 1/2, and foreground
    UIDs for a new cohort. Called only at a cohort boundary.

    * **Group A / Group B** — derived from the chain-set tally of every
      qualified validator's previous-cohort weight Group 1 / Group 2
      votes (read from `metagraph.weights`).
    * **Group C** — `assign_miners_to_validators` over `(all miners \\ A∪B)`,
      this validator's slice, capped at `cfg.validation_group_c_size`.
    * **Foreground** — `assign_miners_to_validators` over `A∪B`, this
      validator's slice. Per-validator distinct partition of the
      consensus tier. Background falls out as `(A∪B∪C) \\ foreground`.
    * **Weight Groups 1/2** — straight from this validator's local
      `election_ballots`.
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
        min_weight_per_validator=cfg.group_a_min_weight_per_validator,
        max_size=cfg.validation_group_a_size,
    )
    group_b = compute_group_b(
        chain_top2,
        group_a=group_a,
        ab_total=cfg.validation_group_ab_total,
    )
    ab_uids = set(group_a) | set(group_b)
    group_c = compute_group_c(
        validator_seeds=validator_seeds,
        all_miner_hotkeys=all_miner_hotkeys,
        ab_uids=ab_uids,
        my_hotkey=my_hotkey,
        hotkey_to_uid=hotkey_to_uid,
        max_size=cfg.validation_group_c_size,
    )
    foreground_uids = compute_foreground_partition(
        validator_seeds=validator_seeds,
        all_miner_hotkeys=all_miner_hotkeys,
        ab_uids=ab_uids,
        my_hotkey=my_hotkey,
        hotkey_to_uid=hotkey_to_uid,
        # No cap — A∪B is at most 13 miners; let assignment distribute
        # them evenly across qualified validators (~1-2 each on a
        # typical 7-13 validator subnet).
        max_size=None,
    )

    validation = ValidationGroups(group_a=group_a, group_b=group_b, group_c=group_c)
    weight = WeightGroups(
        group_1=election_ballots.weight_group_1,
        group_2=election_ballots.weight_group_2,
    )

    logger.info(
        "round_groups.build_cohort_groups",
        group_a=list(group_a),
        group_b=list(group_b),
        group_c=list(group_c),
        foreground_uids=list(foreground_uids),
        weight_group_1=list(weight.group_1),
        weight_group_2=list(weight.group_2),
    )

    return CohortGroups(
        validation=validation,
        weight=weight,
        foreground_uids=foreground_uids,
    )


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
    validator_seeds: dict[str, int],
    all_miner_hotkeys: list[str],
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
        validator_seeds=validator_seeds,
        all_miner_hotkeys=all_miner_hotkeys,
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
        foreground_uids=new_groups.foreground_uids,
        last_election_round_id=round_id,
        highest_seen_cycle_index=highest_seen,
    )

    logger.info(
        "round_groups.maybe_advance_cohort: new cohort",
        cohort_epoch=new_state.cohort_epoch,
        cycle_index=cycle_index,
        validation_group_a=list(new_state.validation_group_a),
        validation_group_b=list(new_state.validation_group_b),
        validation_group_c=list(new_state.validation_group_c),
        foreground_uids=list(new_state.foreground_uids),
        weight_group_1=list(new_state.weight_group_1),
        weight_group_2=list(new_state.weight_group_2),
    )
    return new_state


def split_foreground_background(
    state: "CohortState",
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return `(foreground_uids, background_uids)` for the round.

    Foreground is the per-validator A∪B partition stored on
    `CohortState.foreground_uids` (computed at the cohort boundary by
    `compute_foreground_partition`). Background is `(A ∪ B ∪ C) \\ foreground`,
    preserving A → B → C order so the workers process the consensus
    tier first within background as well.
    """
    foreground = tuple(state.foreground_uids)
    fg_set = set(foreground)
    full_roster = (
        *state.validation_group_a,
        *state.validation_group_b,
        *state.validation_group_c,
    )
    seen: set[int] = set()
    background_list: list[int] = []
    for uid in full_roster:
        if uid in fg_set or uid in seen:
            continue
        seen.add(uid)
        background_list.append(uid)
    return foreground, tuple(background_list)


def split_validation_uids_into_foreground(
    state: "CohortState",
) -> tuple[int, ...]:
    """Deprecated: returns the flat A→B→C concatenation used in earlier
    drafts. New callers should use `split_foreground_background`.
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

    Both Group 1 (`group_1_share`) and Group 2 (`group_2_share`) are
    split *in proportion to local score*. If every member of a group
    scored 0.0, fall back to an equal split so the share still goes
    out and the chain-set tally still moves.

    Everyone not in either group receives 0.0; the chain submitter
    normalizes to 1.0 before sending.
    """
    out: dict[int, float] = {}

    def _allocate(uids: tuple[int, ...], share: float) -> None:
        if not uids:
            return
        scores = [(uid, max(0.0, float(local_scores.get(uid, 0.0)))) for uid in uids]
        total = sum(s for _, s in scores)
        if total > 0.0:
            for uid, s in scores:
                out[uid] = out.get(uid, 0.0) + share * (s / total)
        else:
            per_member = share / len(uids)
            for uid in uids:
                out[uid] = out.get(uid, 0.0) + per_member

    _allocate(weight_group_1, group_1_share)
    _allocate(weight_group_2, group_2_share)
    return out


def select_top_n_by_local_score(
    uids: list[int] | tuple[int, ...],
    local_scores: dict[int, float],
    *,
    n: int,
) -> tuple[int, ...]:
    """Pick the top-`n` UIDs from `uids` ranked by `local_scores` desc,
    UID asc as tiebreak.

    Skips UIDs without a positive score (`<= 0.0` or absent) so the
    weight emission doesn't waste a slot on a miner that wasn't actually
    evaluated this round. Returns up to `n` UIDs — fewer when the pool
    has fewer evaluated miners than `n`.
    """
    scored = [
        (int(uid), float(local_scores.get(uid, 0.0)))
        for uid in uids
        if float(local_scores.get(uid, 0.0)) > 0.0
    ]
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    return tuple(uid for uid, _ in scored[:n])
