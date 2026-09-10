from __future__ import annotations

import copy
import gc
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from connito.shared.app_logging import structlog
from connito.shared.dataloader import get_dataloader
from connito.shared.evaluate import EvalDeadlineExceeded, evaluate_model
from connito.shared.helper import (
    MINER_CHECKPOINT_SUFFIXES,
    load_state_dict_from_path,
    parse_dynamic_filename,
)
from connito.shared.telemetry import (
    EvalFailureReason,
    VALIDATOR_MINER_VAL_LOSS,
    inc_eval_failure,
    set_miner_eval_status,
    track_eval_latency,
    track_model_load_latency,
)

logger = structlog.get_logger(__name__)


# Maps the short reason strings returned by `validate_miner_submission` onto
# the closed `EvalFailureReason` enum used by the
# `validator_miner_eval_failures_total` Counter and the
# `validator_miner_eval_status` Gauge. Each row gets its own miner-facing
# bucket so a miner reading the gateway can tell a bad signature apart from
# a hash mismatch or an expert-group / NaN-Inf violation.
_VALIDATION_FAIL_TO_REASON: dict[str, EvalFailureReason] = {
    "no_chain_commit": "no_chain_commit",
    "signature": "signature_invalid",
    "hash": "hash_mismatch",
    "expert_group_or_nan": "expert_group_or_nan",
    "unknown": "unknown",
}


def _record_eval_failure(uid: int, reason: EvalFailureReason | str) -> None:
    """Helper that pairs the failure Counter with the per-miner status Gauge
    so every call site stays in sync. Both writes are best-effort.
    """
    inc_eval_failure(int(uid), reason)
    set_miner_eval_status(int(uid), reason)


def cleanup_non_top_submissions(
    *,
    round_obj,  # connito.validator.round.Round
    submission_dir: Path,
    top_k: int,
) -> list[str]:
    """Delete miner submission files for UIDs that have been *processed*
    this round but are not in the top-`top_k` by *this round's* score.

    Ranking unions two sets, both owned by the round object so this hot
    path (it runs after every eval) never reads the global
    `MinerScoreAggregator` under a foreign lock:

      - `top_scored_uids_this_round` — top-`top_k` by *this round's*
        score. Merge takes its top-1 from here, so it must survive.
      - `baseline_winner_uid` — the miner `publish_round_baseline`
        would pick if the round finalized now. Same call publish makes,
        so the keep set and the selection can never disagree.

    A file is deleted iff its hotkey resolves to a UID that:
      - is in `round.failed_uids` (validation/timeout/exception, score=0
        — never top), or
      - is in `round.scored_uids` AND is not in the per-round top-k.

    Files for UIDs that have *not* yet been processed are explicitly
    skipped — this is the safety guarantee the eval workers rely on:
    bg-download has already written the shard to disk, but no eval has
    happened, so the file MUST stay until the worker can read it.
    Files belonging to hotkeys outside the round's roster (stale from a
    previous cycle, etc.) are also skipped here; the cycle-tail prune
    catches those.
    """
    submission_dir = Path(submission_dir)
    if not submission_dir.exists():
        return []

    scored, failed = round_obj.processed_uids_snapshot()
    processed = scored | failed
    if not processed:
        return []

    top_uids = round_obj.top_scored_uids_this_round(top_k)
    baseline_uid = round_obj.baseline_winner_uid()
    if baseline_uid is not None:
        top_uids = top_uids | {baseline_uid}
    delete_uids = failed | (scored - top_uids)
    if not delete_uids:
        return []

    # Map UID → hotkey for the deletion target set, and collect the
    # hotkeys of every roster UID that has *not* been processed yet so
    # we can refuse to touch their files even by accident (defense in
    # depth: a hotkey clash would already be impossible, but the explicit
    # filter makes the invariant readable at the deletion site).
    hotkeys_to_delete = {
        round_obj.uid_to_hotkey[uid]
        for uid in delete_uids
        if uid in round_obj.uid_to_hotkey
    }
    unprocessed_hotkeys = {
        hotkey
        for uid, hotkey in round_obj.uid_to_hotkey.items()
        if uid not in processed
    }
    if not hotkeys_to_delete:
        return []

    deleted: list[str] = []
    submission_files = [
        p for suffix in MINER_CHECKPOINT_SUFFIXES
        for p in submission_dir.glob(f"*{suffix}")
    ]
    for file_path in submission_files:
        if file_path.name.startswith(".tmp"):
            continue
        meta = parse_dynamic_filename(file_path.name)
        if not meta:
            continue
        hotkey = meta.get("hotkey")
        if hotkey in unprocessed_hotkeys:
            # Explicit safety: never delete a file for a miner whose
            # checkpoint has not been evaluated yet.
            continue
        if hotkey in hotkeys_to_delete:
            try:
                file_path.unlink(missing_ok=True)
                deleted.append(file_path.name)
            except Exception as e:
                logger.warning(
                    "cleanup_non_top_submissions: failed to delete file",
                    file=file_path.name, error=str(e),
                )
    return deleted


# Rank → score mapping used by `finalize_round_scores`. Geometric
# progression with ratio 1.5: top-1 in the round's delta ranking gets
# 2.25, runner-up 1.5, third 1.0; everyone else (and every failed /
# missing miner) gets 0.0. The geometric spacing concentrates more
# reward weight at the top — `top1 / top3 = 2.25` vs. the previous
# arithmetic mapping's `3 / 1 = 3` — while keeping the second-place
# miner closer to first than to third (`top2 / top1 = 0.667` vs.
# `top3 / top2 = 0.667`, equal ratios across tiers). Hard-coded rather
# than parameterized off `top_k_miners_to_reward` (which governs disk
# retention, not reward weight) because these values are part of the
# scoring contract — see PR #93.
_RANK_TO_SCORE: tuple[float, ...] = (2.25, 1.5, 1.0)


def finalize_round_scores(
    *,
    round_obj,  # connito.validator.round.Round
    score_aggregator,
    score_path=None,
) -> dict[int, float]:
    """Replace this round's per-miner aggregator entries with rank-based
    scores derived from `round.scores` (the delta-based per-round signal
    recorded by `mark_scored`).

    Drops every aggregator point tagged with `round.round_id` first so
    intermediate eval-time scores do not stack with the rank-based ones,
    then re-adds:

      - Top-1 by `round.scores` (delta desc): score 2.25.
      - Top-2: score 1.5.
      - Top-3: score 1.0.
      - Other scored UIDs (incl. delta=0): score 0.
      - Any UIDs whose `round.scores` value exactly equals another
        scored miner's: score 0 — a tied val_loss is evidence of a
        duplicated submission, so both sides are penalized regardless
        of where they would have ranked.
      - `validation_failed_uids` (hash/sig/expert_group/NaN-Inf, or a
        committed HF checkpoint confirmed not publicly retrievable):
        score 0.
      - `freeze_zero_uids` (no/invalid chain commit at freeze): score 0.

    Operational failures (download timeout, eval timeout, OOM, unexpected
    exception) live in `failed_uids` but NOT in `validation_failed_uids`,
    so finalize deliberately writes nothing for them — their prior EMA
    is preserved. The validator's lack of compute / bandwidth must not
    dock a miner's reward.

    Likewise, miners we never reached (submission never landed, or
    bg-eval ran out of time before claiming) are absent from every set
    and receive no entry. They keep their prior EMA and the next
    round's stalest-first prioritization gives them another shot.

    Miners whose `round.scores` value is 0.0 are explicitly excluded
    from the top-3 ranking so a "best of a bad bunch" miner cannot
    collect reward weight without actually improving over baseline.

    Returns ``{uid: rank_score}`` for the UIDs the function wrote, for
    logging.
    """
    # Snapshot all sets under the round's lock so the worker threads
    # cannot race a mark_scored / mark_failed against the read.
    scored, _failed = round_obj.processed_uids_snapshot()
    # `round.scores` is mutated under the same lock; copy it explicitly
    # rather than alias.
    with round_obj._lock:  # noqa: SLF001 — same module family
        round_scores = dict(round_obj.scores)
        validation_failed = set(round_obj.validation_failed_uids)
    freeze_zero = set(round_obj.freeze_zero_uids)
    freeze_hotkeys = dict(round_obj.freeze_zero_hotkeys)

    score_aggregator.drop_round(round_obj.round_id)

    # Rank only positive-delta miners — see the docstring's "best of a
    # bad bunch" clause.
    positive = [
        (uid, score) for uid, score in round_scores.items()
        if uid in scored and score > 0.0
    ]
    # Group by exact score value: any miner whose val_loss matches
    # another miner's gets 0 regardless of where they would have ranked.
    # `score = (baseline_loss - val_loss) ** 1.2` with float64 math —
    # exact equality between two miners is overwhelmingly evidence of a
    # duplicated submission, not legitimate parallel improvement, so
    # penalize both sides. Unique-score miners are then ranked normally
    # and slot into the 3/2/1 mapping by position.
    score_counts: dict[float, int] = {}
    for _, s in positive:
        score_counts[s] = score_counts.get(s, 0) + 1
    tied_uids = {uid for uid, s in positive if score_counts[s] > 1}
    unique_positive = [(uid, s) for uid, s in positive if score_counts[s] == 1]
    unique_positive.sort(key=lambda kv: (-kv[1], kv[0]))

    written: dict[int, float] = {}
    top_uids: set[int] = set()
    for rank, (uid, _) in enumerate(unique_positive):
        rank_score = _RANK_TO_SCORE[rank] if rank < len(_RANK_TO_SCORE) else 0.0
        hotkey = round_obj.uid_to_hotkey.get(uid)
        if hotkey is None:
            continue
        score_aggregator.add_score(
            uid=uid, hotkey=hotkey, score=rank_score, round_id=round_obj.round_id,
        )
        written[uid] = rank_score
        top_uids.add(uid)

    # Tied positive-delta miners — explicit 0 entry per uid.
    for uid in tied_uids:
        hotkey = round_obj.uid_to_hotkey.get(uid)
        if hotkey is None:
            continue
        score_aggregator.add_score(
            uid=uid, hotkey=hotkey, score=0.0, round_id=round_obj.round_id,
        )
        written[uid] = 0.0
        top_uids.add(uid)

    # Remaining scored UIDs (delta == 0 or beyond top-3): score 0.
    for uid in scored - top_uids:
        hotkey = round_obj.uid_to_hotkey.get(uid)
        if hotkey is None:
            continue
        score_aggregator.add_score(
            uid=uid, hotkey=hotkey, score=0.0, round_id=round_obj.round_id,
        )
        written[uid] = 0.0

    # Explicit validation failures — submission was off-spec.
    for uid in validation_failed:
        hotkey = round_obj.uid_to_hotkey.get(uid)
        if hotkey is None:
            continue
        score_aggregator.add_score(
            uid=uid, hotkey=hotkey, score=0.0, round_id=round_obj.round_id,
        )
        written[uid] = 0.0

    # Freeze-time invalid-checkpoint penalties. Skip any UID that ended
    # up in scored/validation_failed (cannot happen today, but keep the
    # override explicit if the freeze logic ever shifts).
    freeze_zero_only = freeze_zero - scored - validation_failed
    for uid in freeze_zero_only:
        hotkey = freeze_hotkeys.get(uid) or round_obj.uid_to_hotkey.get(uid)
        if hotkey is None:
            continue
        score_aggregator.add_score(
            uid=uid, hotkey=hotkey, score=0.0, round_id=round_obj.round_id,
        )
        written[uid] = 0.0

    # Telemetry — emit eval_status for the freeze_zero bucket here because
    # the per-uid `_record_eval_failure` call sites only fire when the eval
    # loop actually picks the miner up. Freeze-time invalid checkpoints
    # never reach that point, so without this loop the gateway has no signal
    # for "we knew at freeze you had no/invalid commit." validation_failed
    # statuses are already set at the validate_miner_submission call site
    # with the specific sub-reason (signature/hash/expert_group_or_nan), so
    # do not overwrite them here. Wrapped broadly — telemetry must never
    # block finalize.
    try:
        from connito.shared.telemetry import set_miner_eval_status as _set_status
        for uid in freeze_zero_only:
            _set_status(int(uid), "no_chain_commit")
    except Exception:
        pass

    if score_path is not None:
        try:
            score_aggregator.persist_atomic(score_path)
        except Exception as e:
            logger.warning(
                "finalize_round_scores: persist_atomic failed",
                round_id=round_obj.round_id, error=str(e),
            )

    # Flip the round's journal to `finalized=true` and rewrite it so the
    # post-finalize file on disk reflects the rank-based scores. The
    # journal stays on disk after this — pruned by age along with the
    # aggregator entries it backs (see `prune_before_round` callers in
    # run.py).
    journal_path = getattr(round_obj, "journal_path", None)
    if journal_path is not None:
        try:
            from connito.validator import round_journal as _rj
            scored_set, failed_set = round_obj.processed_uids_snapshot()
            with round_obj._lock:  # noqa: SLF001
                journal_scores = dict(round_obj.scores)
                journal_uid_to_hotkey = dict(round_obj.uid_to_hotkey)
            _rj.write_atomic(
                journal_path,
                _rj.RoundJournal(
                    round_id=round_obj.round_id,
                    uid_to_hotkey=journal_uid_to_hotkey,
                    scores=journal_scores,
                    scored_uids=tuple(sorted(scored_set)),
                    failed_uids=tuple(sorted(failed_set)),
                    validation_failed_uids=tuple(sorted(validation_failed)),
                    freeze_zero_uids=tuple(sorted(freeze_zero)),
                    freeze_zero_hotkeys=dict(freeze_hotkeys),
                    uid_to_commit=_rj.commit_map_from_checkpoints(
                        getattr(round_obj, "uid_to_chain_checkpoint", None) or {}
                    ),
                    # v3 round-level gauge inputs. A live Round exposes
                    # `background_uids`; the recovery stub carries
                    # `roster_size` forward from the journal it was hydrated
                    # from — so a re-finalize preserves them.
                    roster_size=int(
                        getattr(round_obj, "roster_size", 0)
                        or len(getattr(round_obj, "background_uids", ()))
                    ),
                    lifecycle_step=int(getattr(round_obj, "lifecycle_step", 0)),
                    uid_to_val_loss=dict(getattr(round_obj, "val_losses", None) or {}),
                    seed=str(getattr(round_obj, "seed", "") or ""),
                    finalized=True,
                ),
            )
            # The round is over, so its base snapshot can never be resumed
            # from again. `prune_before_round` is the backstop for a round
            # that never reached finalize.
            _rj.base_snapshot_path_for(
                Path(journal_path).parent.parent, round_obj.round_id
            ).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(
                "finalize_round_scores: journal flip-to-finalized failed",
                round_id=round_obj.round_id, error=str(e),
            )

    # --- Cycle-consistent per-miner telemetry (dashboard contract). -------
    # Emitted HERE — not from run.py's weight loop — for two reasons:
    # (1) the weight loop only iterates weight recipients, so the dashboard
    #     previously saw score snapshots for ~1 uid; every verdict uid gets
    #     one now; (2) the journal-recovery replay calls this function too,
    #     so a restart re-publishes the series without waiting for a fresh
    #     round. Best-effort throughout — telemetry must never block
    #     finalize or scoring.
    try:
        from connito.shared.telemetry import (
            set_miner_evaluated_commit,
            set_miner_last_scored_round,
            set_miner_round_delta,
            set_miner_score_snapshot,
        )

        _rid = int(round_obj.round_id)
        try:
            _latest_scores = score_aggregator.uid_score_pairs(how="latest")
            _avg_scores = score_aggregator.uid_score_pairs(how="avg")
        except Exception:
            _latest_scores, _avg_scores = {}, {}
        for uid in written:
            set_miner_last_scored_round(int(uid), _rid)
            try:
                _samples = score_aggregator.record_count(int(uid))
            except Exception:
                _samples = None
            set_miner_score_snapshot(
                int(uid),
                latest=_latest_scores.get(int(uid)),
                avg=_avg_scores.get(int(uid)),
                samples=_samples,
            )
        for uid in scored:
            set_miner_round_delta(int(uid), float(round_scores.get(uid, 0.0)))
        _ckpt_map = getattr(round_obj, "uid_to_chain_checkpoint", None) or {}
        for uid, _ckpt in _ckpt_map.items():
            _repo = getattr(_ckpt, "hf_repo_id", None)
            _rev = getattr(_ckpt, "hf_revision", None)
            if _repo and _rev:
                set_miner_evaluated_commit(int(uid), str(_repo), str(_rev), _rid)
    except Exception as e:
        logger.warning(
            "finalize_round_scores: telemetry emission failed",
            round_id=round_obj.round_id, error=str(e),
        )

    logger.info(
        "finalize_round_scores: round scored by rank",
        round_id=round_obj.round_id,
        top3={
            int(u): _RANK_TO_SCORE[r]
            for r, (u, _) in enumerate(unique_positive[:3])
        },
        scored_count=len(scored),
        tied_count=len(tied_uids),
        validation_failed_count=len(validation_failed),
        freeze_zero_count=len(freeze_zero - scored - validation_failed),
    )
    return written


@dataclass(frozen=True)
class WeightSubmissionPayload:
    """Structured payload returned by `build_submission_uid_weights`.

    `uid_weights` is what the chain submitter consumes; the
    `weight_group_*` fields are populated only when the cohort-style
    emission was used (otherwise empty), and exist purely so the caller
    can log them without recomputing the selection.
    `g1_redirected_to_uid_zero` is set when the empty-G1 guard fires —
    the caller logs that case under its own info line.
    `g1_stale_excluded` lists the A∪B UIDs that cleared the count and
    recency gates but were dropped by the freshness gate, so the caller
    can log exactly who lost a seat to staleness.
    """
    uid_weights: dict[int, float]
    weight_group_1: tuple[int, ...] = ()
    weight_group_2: tuple[int, ...] = ()
    cohort_emission: bool = False
    g1_redirected_to_uid_zero: bool = False
    g1_stale_excluded: tuple[int, ...] = ()


def build_submission_uid_weights(
    *,
    score_aggregator,
    cohort_state=None,
    round_id: int | None = None,
    cycle_length: int | None = None,
    eval_cfg=None,
) -> WeightSubmissionPayload:
    """Build the `{uid: weight}` payload for a single chain submission.

    Decoupled from any wrapper Round — the cohort fields are passed
    directly so callers without a Round (e.g. restart replay) can also
    drive cohort-style emission as long as a persisted `CohortState`
    is available on disk.

    Inputs needed for cohort-style emission:
      * `cohort_state` — provides `validation_group_a/b/c`.
      * `round_id` — anchor for the recency gate.
      * `cycle_length` — sizes the recency window (`5 * cycle_length`).
      * `eval_cfg` — reads `weight_group_*_size` and `weight_group_*_share`.

    Cohort emission rule (when all four are present):
      * Group 1 (`cfg.weight_group_1_share`): top-`weight_group_1_size`
        of A∪B by aggregator avg, restricted to UIDs with
        `record_count >= 3` AND scores recorded in at least 3 distinct
        round_ids within the last `5 * cycle_length` blocks — i.e.
        scored in at least 3 of the last 5 cycles. Tightens the prior
        2-of-5 gate so a miner needs sustained participation to anchor
        the validator's top-N ballot. On top of that, a **freshness
        gate**: the UID's most recent tagged point must be no older
        than `cfg.g1_max_stale_rounds` rounds. Empty-G1 guard: if no
        UID clears, redirect to `uid = 0` (subnet owner) so the
        validator stays at full emission.
      * Group 2 (`cfg.weight_group_2_share`): top-`weight_group_2_size`
        of A∪B∪C \\ G1 by aggregator avg, restricted to UIDs with
        `record_count >= 1` (no recency gate).

    With any of the four cohort inputs missing (cold-start replay
    before disk has a CohortState, legacy non-cohort rounds, etc.) the
    helper falls back to the score-aggregator avg directly.
    """
    avg_scores = score_aggregator.uid_score_pairs(how="avg")
    if (
        cohort_state is None
        or round_id is None
        or cycle_length is None
        or eval_cfg is None
    ):
        return WeightSubmissionPayload(uid_weights=avg_scores)

    from connito.validator import round_groups as _rg

    ab_uids = list(cohort_state.validation_group_a) + list(cohort_state.validation_group_b)
    abc_uids = ab_uids + list(cohort_state.validation_group_c)
    cur_rid = int(round_id)
    g1_window_min_rid = cur_rid - 5 * int(cycle_length)

    # Freshness gate. `avg_scores` is a rolling mean over *recorded
    # points* with no notion of elapsed rounds, so a UID that stops
    # being evaluated keeps its last average intact and holds its G1
    # seat until those points age out of the retention window. Observed
    # on mainnet: uid 158 took 31.5% of emission at round 8692978 on a
    # 2.25 earned four rounds earlier, and its avg sat unchanged at
    # 0.40625 across six consecutive rounds. Requiring a recent tagged
    # point makes "no current evidence about this miner" disqualifying
    # for a G1 seat, independent of what the stale average says.
    #
    # A 0.0 written by `finalize_round_scores` counts as evidence — the
    # gate is about whether the validator looked at the miner this
    # round, not about how well it did. `latest_round_id` returns None
    # for a UID with no tagged points (schema v1 legacy state), which
    # fails the gate: unattributable history cannot back a G1 seat.
    max_stale = int(getattr(eval_cfg, "g1_max_stale_rounds", 1))
    g1_min_fresh_rid = cur_rid - max_stale * int(cycle_length)

    def _has_recent_history(uid: int) -> bool:
        return (
            score_aggregator.record_count(uid) >= 3
            and score_aggregator.count_distinct_round_ids_in_range(
                uid, g1_window_min_rid, cur_rid,
            ) >= 3
        )

    def _is_fresh(uid: int) -> bool:
        latest_rid = score_aggregator.latest_round_id(uid)
        return latest_rid is not None and latest_rid >= g1_min_fresh_rid

    ab_qualified = [
        u for u in ab_uids if _has_recent_history(u) and _is_fresh(u)
    ]
    # Recorded purely so the caller can log who lost a seat to staleness;
    # does not affect selection.
    g1_stale_excluded = tuple(
        u for u in ab_uids if _has_recent_history(u) and not _is_fresh(u)
    )
    g1 = _rg.select_top_n_by_local_score(
        ab_qualified,
        avg_scores,
        n=eval_cfg.weight_group_1_size,
    )
    g1_redirected = False
    if not g1:
        g1 = (0,)
        g1_redirected = True
    g1_set = set(g1)
    g2_pool = [
        u for u in abc_uids
        if u not in g1_set
        and score_aggregator.record_count(u) >= 1
    ]
    g2 = _rg.select_top_n_by_local_score(
        g2_pool,
        avg_scores,
        n=eval_cfg.weight_group_2_size,
    )
    uid_weights = _rg.compute_uid_weights(
        weight_group_1=g1,
        weight_group_2=g2,
        local_scores=avg_scores,
        group_1_share=eval_cfg.weight_group_1_share,
        group_2_share=eval_cfg.weight_group_2_share,
    )
    return WeightSubmissionPayload(
        uid_weights=uid_weights,
        weight_group_1=g1,
        weight_group_2=g2,
        cohort_emission=True,
        g1_redirected_to_uid_zero=g1_redirected,
        g1_stale_excluded=g1_stale_excluded,
    )


def validate_miner_submission(
    *,
    round_obj,  # connito.validator.round.Round
    uid: int,
    model_path: str | Path,
    expert_group_assignment,
) -> str | None:
    """Run the existing `ChainCheckpoint.validate(...)` against a miner's
    on-disk submission before it is fed to `evaluate_one_miner`.

    Returns ``None`` on success. On failure returns a short reason string —
    one of ``no_chain_commit | signature | hash | expert_group | nan_inf``,
    or a generic ``"unknown"`` if the helper raised. The reason is intended
    to be plumbed into telemetry labels and log lines.

    The chain checkpoint is read from `round_obj.uid_to_chain_checkpoint`
    so this never re-fetches anything from the chain. The check itself is
    `ChainCheckpoint.validate(expert_group_assignment=…)`, which runs:

    - `_verify_signature` — the chain hotkey signed `model_hash`.
    - `_verify_hash` — the on-disk shard's hash matches the chain commit.
    - `_verify_expert_group` — every routed-expert key in the state dict
      belongs to the miner's assigned group, and no tensor contains NaN/Inf.
    """
    chain_checkpoint = round_obj.uid_to_chain_checkpoint.get(int(uid))
    if chain_checkpoint is None:
        return "no_chain_commit"

    # `validate()` reads the state dict from `chain_checkpoint.path`; point
    # it at the on-disk submission for this round.
    chain_checkpoint.path = Path(model_path)

    try:
        ok = chain_checkpoint.validate(expert_group_assignment=expert_group_assignment)
    except Exception as e:
        logger.warning(
            "validate_miner_submission: validate() raised",
            uid=int(uid), error=str(e), exc_info=True,
        )
        return "unknown"

    if ok:
        return None

    # `validate()` already logged a structured warning per failed sub-check.
    # Map the per-check booleans to a single short reason for telemetry.
    if not getattr(chain_checkpoint, "signature_verified", False):
        return "signature"
    if not getattr(chain_checkpoint, "hash_verified", False):
        return "hash"
    if not getattr(chain_checkpoint, "expert_group_verified", False):
        # _verify_expert_group folds the NaN/Inf scan in with the routing
        # check, so we cannot tell them apart from the booleans alone. The
        # underlying logger.warning at the failure site distinguishes them.
        return "expert_group_or_nan"
    return "unknown"


# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class MinerEvalJob:
    uid: int
    hotkey: str
    model_path: str
    step: int
    score: float = 0.0
    # Raw evaluation loss for this miner this round. Carried alongside the
    # delta-based `score` so the caller can journal it: `val_loss` is
    # published to Prometheus at eval time and is NOT recoverable from
    # `score` alone, because `delta = max(0.0, baseline - val_loss)` clamps
    # at zero — every miner scoring 0 (the majority in many rounds) would
    # be underivable. Without journaling it, a mid-round restart loses the
    # cycle's losses permanently.
    val_loss: float | None = None


# -------------------------- Pipeline Config -----------------------------------
MAX_CONCURRENT_DOWNLOADS = 4
EVAL_WORKERS = 1
DOWNLOAD_TIMEOUT_SEC = 60
EVAL_MAX_BATCHES = 50
# ------------------------------------------------------------------------------

# def load_model_from_path(path: str, base_model, device: torch.device) -> nn.Module:
#     sd = torch.load(path, map_location=torch.device("cpu"))["model_state_dict"]
#     model = copy.deepcopy(base_model)
#     model.load_state_dict(sd, strict=False)
#     return model.to(device)

@track_model_load_latency()
def load_model_from_path(path: str, base_model: nn.Module, device: torch.device) -> nn.Module:
    # `path` points to a miner-controlled checkpoint downloaded from HF.
    # `load_state_dict_from_path` accepts `.safetensors` (preferred — no
    # pickle path) or `.pt` (gated by `weights_only=True` so a malicious
    # `__reduce__` payload cannot execute on the validator host).
    sd = load_state_dict_from_path(path)

    if len(sd) == 0:
        raise ValueError(f"Checkpoint at {path} has empty model_state_dict")

    model = copy.deepcopy(base_model)

    # Load weights (strict=False so missing/unexpected are allowed)
    incompatible = model.load_state_dict(sd, strict=False)

    # Key diagnostics come from `load_state_dict`'s own report. Deriving them
    # beforehand needed `base_model.state_dict()`, which dequantizes every fp8
    # weight to fp32 — ~14.5 GB materialized per miner, for these log lines
    # alone. Mismatched shapes raise from `load_state_dict` itself, so they no
    # longer need a branch here.
    expert_not_in_base = [k for k in incompatible.unexpected_keys if "expert" in k]
    expert_in_base_not_common = [k for k in incompatible.missing_keys if "expert" in k]
    matched_keys = len(sd) - len(incompatible.unexpected_keys)

    if matched_keys == 0:
        logger.warning(
            "No compatible keys between checkpoint and base model — "
            "checkpoint is likely from a different architecture or naming convention",
            ckpt_key_count=len(sd),
            sample_ckpt_keys=sorted(k for k in sd if "expert" in k)[:5],
        )
    elif expert_not_in_base:
        logger.warning(
            "Expert keys in checkpoint not found in base model",
            expert_not_in_base=len(expert_not_in_base),
            sample_keys=sorted(expert_not_in_base)[:5],
        )
    else:
        logger.debug(
            "Key summary",
            matched_keys=matched_keys,
            expert_in_base_not_common=len(expert_in_base_not_common),
        )

    # # Extra helpful debug (optional)
    # if incompatible.missing_keys:
    #     print(f"[load_model] missing keys (first 50): {incompatible.missing_keys[:50]}")
    # if incompatible.unexpected_keys:
    #     print(f"[load_model] unexpected keys (first 50): {incompatible.unexpected_keys[:50]}")

    return model.to(device)


def _evaluate_on_fresh_loader_sync(
    *,
    config,
    tokenizer,
    combinded_seed: str,
    step: int,
    model: nn.Module,
    device: torch.device,
    max_eval_batches: int,
    rank: int | None = None,
    deadline_monotonic: float | None = None,
    cached_batches: list | None = None,
) -> dict:
    """Run `evaluate_model` on the calling thread against a fresh (or
    cached) eval batch source.

    Every caller in a round shares the same `combinded_seed`, so the
    baseline and all miner evals see the same batches — the deltas are
    comparable.

    If `cached_batches` is provided, iterates the cached list (CPU
    tensors, materialized once per round by `materialize_batches`) and
    skips `get_dataloader` entirely. This keeps HF streaming off the
    per-miner critical path — the trigger for the bg-eval lock-leak
    wedge observed in production logs. Falls back to a fresh streaming
    dataloader when the cache is absent (first miner of a fresh worker,
    etc.).
    """
    if cached_batches is not None:
        dataloader = cached_batches
    else:
        dataloader = get_dataloader(
            config=config,
            tokenizer=tokenizer,
            seed=combinded_seed,
            rank=0,
            world_size=config.dataloader.world_size,
        )
    try:
        @track_eval_latency()
        def _run():
            return evaluate_model(
                step, model, dataloader, device, max_eval_batches, rank,
                deadline_monotonic=deadline_monotonic,
            )
        return _run()
    finally:
        if cached_batches is None:
            del dataloader


def evaluate_one_miner_sync(
    *,
    config,
    model_path: str | Path,
    uid: int,
    hotkey: str,
    base_model: nn.Module,
    tokenizer,
    combined_seed: str,
    device: torch.device,
    baseline_loss: float,
    step: int,
    round_id: int | None = None,
    max_eval_batches: int = EVAL_MAX_BATCHES,
    rank: int | None = None,
    deadline_monotonic: float | None = None,
    cached_batches: list | None = None,
) -> "MinerEvalJob | None":
    """Synchronous variant of `evaluate_one_miner`.

    All GPU work — `load_model_from_path`, dataloader build, and
    `evaluate_model` — happens inside this single function so the caller
    can run the entire eval as one `asyncio.to_thread` task.

    That structure matters for cancellation: tasks scheduled via
    `asyncio.to_thread` are not cancellable, so when an outer
    `asyncio.wait_for` timeout fires, the awaiter is cancelled but the
    underlying thread keeps running. If the thread is partway through
    `copy.deepcopy(base_model)` or `model.to(device)`, the GPU memory it
    has allocated is still live; starting a second eval in parallel will
    OOM. Funnelling the whole eval through one `to_thread` task lets
    callers acquire any GPU lock INSIDE that thread — the lock release
    then tracks actual GPU completion (not awaiter cancellation), so the
    next eval naturally blocks on lock acquisition until the previous
    thread drains.

    `deadline_monotonic` is forwarded to `evaluate_model`, which checks
    it between batches and raises `EvalDeadlineExceeded` cleanly. The
    caller's `with` block then unwinds and releases the lock — covers
    the case where the eval genuinely stalls on GPU and `wait_for`
    cancellation alone wouldn't recover.
    """
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        miner_model = load_model_from_path(str(model_path), base_model, device)

        try:
            metrics = _evaluate_on_fresh_loader_sync(
                config=config,
                tokenizer=tokenizer,
                combinded_seed=combined_seed,
                step=step,
                model=miner_model,
                device=device,
                max_eval_batches=max_eval_batches,
                rank=rank,
                deadline_monotonic=deadline_monotonic,
                cached_batches=cached_batches,
            )
        finally:
            del miner_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        val_loss = float(metrics.get("val_loss", 100))
        delta = max(0.0, baseline_loss - val_loss)
        # `score` is the per-round delta-based signal stored on
        # `MinerEvalJob` and recorded in `round.scores` by the caller via
        # `mark_scored`. The aggregator is intentionally NOT updated here
        # — `finalize_round_scores` is the sole writer for this round's
        # aggregator entries (see PR #93 introducing rank-based scoring).
        score = delta ** 1.2
        # Publish per-miner val_loss to Prometheus so external aggregators
        # can render the leaderboard without a per-validator HTTP scrape.
        # Best-effort — Prometheus exposition is purely an observability
        # side-effect and must never block scoring.
        try:
            VALIDATOR_MINER_VAL_LOSS.labels(miner_uid=str(int(uid))).set(float(val_loss))
        except Exception:
            pass
        # Surface the eval outcome on the per-miner status gauge so miners
        # can self-serve the answer to "why was my val_loss empty?". A
        # non-finite loss does not abort scoring — downstream finalize will
        # naturally produce a 0 ranking score — but the gauge lets the
        # gateway distinguish "evaluated and clean" from "evaluated and the
        # eval blew up numerically", which the failure Counter doesn't.
        if math.isfinite(val_loss):
            set_miner_eval_status(int(uid), None)
        else:
            set_miner_eval_status(int(uid), "non_finite_loss")
        logger.info(
            "evaluate_one_miner: complete",
            uid=int(uid),
            hotkey=hotkey[:6],
            val_loss=round(val_loss, 4),
            baseline_loss=round(baseline_loss, 4),
            delta=round(delta, 4),
            score=round(score, 6),
            round_id=round_id,
        )
        return MinerEvalJob(
            uid=int(uid),
            hotkey=hotkey,
            model_path=str(model_path),
            step=int(step),
            score=float(score),
            val_loss=float(val_loss),
        )
    except EvalDeadlineExceeded as e:
        logger.warning(
            "evaluate_one_miner: deadline exceeded — bailing cleanly",
            uid=int(uid), hotkey=hotkey[:6], round_id=round_id, error=str(e),
        )
        _record_eval_failure(int(uid), "deadline")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None
    except torch.cuda.OutOfMemoryError:
        logger.error("evaluate_one_miner: OOM", uid=int(uid))
        _record_eval_failure(int(uid), "oom")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None
    except (ValueError, RuntimeError, EOFError) as e:
        # ValueError: load_model_from_path's "Unsupported checkpoint format" /
        # empty state_dict guard. RuntimeError / EOFError: torch.load rejecting
        # truncated or malformed payloads. All three signal an unreadable
        # state_dict on disk, which is what the miner needs to see.
        logger.exception("evaluate_one_miner: statedict parse failed", uid=int(uid), error=str(e))
        _record_eval_failure(int(uid), "statedict_parse_failed")
        return None
    except Exception as e:
        logger.exception("evaluate_one_miner: failed", uid=int(uid), error=str(e))
        _record_eval_failure(int(uid), "unknown")
        return None
