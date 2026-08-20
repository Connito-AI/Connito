"""Per-round state for the lifecycle (0)..(4); see
`docs/validator-round-construction.md`.

A `Round` is constructed once, at the start of each Submission phase
(step 0), and is immutable thereafter. The foreground pass and the two
background workers (download + eval) all anchor on the same `Round`.
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, NamedTuple

import torch
import torch.nn as nn

from connito.shared.app_logging import structlog
from connito.shared.telemetry import set_round_progress

if TYPE_CHECKING:
    from connito.shared.checkpoints import ChainCheckpoint
    from connito.validator.aggregator import MinerScoreAggregator
    from connito.validator.cohort_state import CohortState

logger = structlog.get_logger(__name__)


class RosterEntry(NamedTuple):
    """Lightweight (uid, hotkey) pair yielded by Round iteration helpers."""
    uid: int
    hotkey: str


@dataclass
class Round:
    """Immutable snapshot of round K's inputs + mutable per-worker state.

    The frozen pieces are written once by `Round.freeze` and never changed.
    The mutable pieces (downloaded_pool, scored_uids, failed_uids,
    weights_submitted) are guarded by an internal lock and are updated by
    the workers.

    `foreground_uids` is this validator's assignment slice; `background_uids`
    is every other miner with a chain checkpoint this cycle. `uid_to_hotkey`
    covers the union, so workers don't need to hold a metagraph reference
    to translate a UID back to a hotkey.
    """

    round_id: int
    seed: str
    validator_miner_assignment: dict[str, list[str]]
    foreground_uids: tuple[int, ...]
    background_uids: tuple[int, ...]
    uid_to_hotkey: dict[int, str]
    model_snapshot_cpu: dict[str, torch.Tensor]
    # Gates `_existing_submission` reuse in bg-download: without it a stale
    # file from a previous cycle short-circuits the fetch and is then silently
    # rejected by `gather_validation_job`.
    submission_block_range: tuple[int, int] | None = None
    # Captured at freeze so the eval path can validate submissions without
    # re-issuing chain RPCs.
    uid_to_chain_checkpoint: dict[int, "ChainCheckpoint"] = field(default_factory=dict)

    # Mutable, lock-guarded
    downloaded_pool: dict[int, Path] = field(default_factory=dict)
    scored_uids: set[int] = field(default_factory=set)
    # This round's scores only. Cleanup ranks off these rather than the
    # aggregator, whose history would pull a non-top-this-round miner into the
    # keep set.
    scores: dict[int, float] = field(default_factory=dict)
    # Journaled because `scores` clamps the delta at 0, so a restart could not
    # otherwise recover the cycle's losses.
    val_losses: dict[int, float] = field(default_factory=dict)
    claimed_uids: set[int] = field(default_factory=set)
    failed_uids: set[int] = field(default_factory=set)
    # Miner-fault failures: bad signature/hash/expert-group/NaN, no chain
    # commit, an unfetchable HF repo, or a freeze-time invalid checkpoint.
    # These get score=0 at finalize.
    #
    # `failed_uids ⊃ validation_failed_uids`. Operational failures
    # (timeout/OOM/download) live only in `failed_uids` and deliberately get
    # NO aggregator entry, so the miner keeps its prior average — the
    # validator's lack of compute or bandwidth must not dock a miner.
    validation_failed_uids: set[int] = field(default_factory=set)
    # Freeze-time invalid-checkpoint penalties. Hotkey map is captured
    # alongside because these UIDs may not appear in `uid_to_hotkey`
    # (which only covers roster miners with a valid checkpoint).
    freeze_zero_uids: set[int] = field(default_factory=set)
    freeze_zero_hotkeys: dict[int, str] = field(default_factory=dict)
    weights_submitted: bool = False
    # 0 freeze / 2 post-foreground / 3 eval-window. Journaled so startup
    # recovery can restore the round-level gauges only the live loop writes.
    lifecycle_step: int = 0

    # Round-group construction. Default to empty so consumers can branch on
    # `bool(weight_group_1)` instead of re-reading the feature flag.
    weight_group_1: tuple[int, ...] = ()
    weight_group_2: tuple[int, ...] = ()
    validation_group_a: tuple[int, ...] = ()
    validation_group_b: tuple[int, ...] = ()
    validation_group_c: tuple[int, ...] = ()
    cohort_epoch: int = 0
    # Transient: run.py reads this off and persists it after freeze returns.
    cohort_state: "CohortState | None" = field(default=None, repr=False, compare=False)

    # Every mark_* call writes the round's mutation state here, so a kill
    # before `finalize_round_scores` doesn't lose bg-eval work. All three
    # default None so tests can build a `Round` without `Round.freeze`.
    journal_path: "Path | None" = field(default=None, repr=False, compare=False)
    score_aggregator: "MinerScoreAggregator | None" = field(default=None, repr=False, compare=False)
    score_path: "Path | None" = field(default=None, repr=False, compare=False)

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    # ---------------- Construction ----------------
    BG_TOP_SCORED_PREPEND_COUNT: int = 5

    @classmethod
    def freeze(
        cls,
        *,
        config,
        subtensor,
        metagraph,
        global_model: nn.Module,
        round_id: int | None = None,
        submission_block_range: tuple[int, int] | None = None,
        last_evaluated: dict[int, datetime] | None = None,
        prior_avg_scores: dict[int, float] | None = None,
        cycle_index: int | None = None,
        cycle_length: int | None = None,
        cohort_state: "CohortState | None" = None,
        score_aggregator: "MinerScoreAggregator | None" = None,
        score_path: "Path | None" = None,
        checkpoint_path: "Path | None" = None,
    ) -> "Round":
        """Build a Round at Submission-phase start.

        Caller pre-fetches `metagraph` (sync or async, depending on the
        validator's subtensor type) and passes it in so this method has
        no opinion on the connection model. Captures the metagraph
        incentive snapshot and the global_model state_dict (CPU clone)
        before Merge(K) can mutate either.
        """
        from connito.shared.chain import get_chain_commits
        from connito.shared.cycle import (
            get_combined_validator_seed,
            get_validator_miner_assignment,
            get_validator_seed_from_commit,
        )

        # Fetch commits once and share them (plus the caller's metagraph) with
        # both helpers; each would otherwise repeat a get_all_commitments +
        # metagraph pair through the global subtensor lock.
        commits = get_chain_commits(config, subtensor)
        seed = get_combined_validator_seed(config, subtensor, commits=commits)
        assignment_result = get_validator_miner_assignment(
            config, subtensor, commits=commits, metagraph=metagraph,
        )
        assignment = assignment_result.assignment
        my_assignment_set = set(assignment.get(config.chain.hotkey_ss58, []))

        hotkey_to_uid = {hk: uid for uid, hk in enumerate(metagraph.hotkeys)}

        # `miners_with_checkpoint` is already incentive-ranked. Walk it once
        # and split into foreground (this validator's assignment) and
        # background (everyone else with a checkpoint).
        foreground: list[int] = []
        background: list[int] = []
        uid_to_hotkey: dict[int, str] = {}
        uid_to_chain_checkpoint: dict[int, "ChainCheckpoint"] = {}
        chain_checkpoints_by_hotkey = getattr(
            assignment_result, "chain_checkpoints_by_hotkey", {}
        ) or {}
        assigned_with_valid_ckpt: set[str] = set()
        for hk in assignment_result.miners_with_checkpoint:
            uid = hotkey_to_uid.get(hk)
            if uid is None:
                logger.warning("Round.freeze: hotkey not in metagraph; skipping", hotkey=hk[:6])
                continue
            uid_to_hotkey[uid] = hk
            ckpt = chain_checkpoints_by_hotkey.get(hk)
            if ckpt is not None:
                uid_to_chain_checkpoint[uid] = ckpt
            if ckpt is not None and ckpt.hf_repo_id and ckpt.hf_revision:
                assigned_with_valid_ckpt.add(hk)

            (foreground if hk in my_assignment_set else background).append(uid)

        rid = int(round_id) if round_id is not None else int(subtensor.block)

        # Neurons with no valid chain checkpoint, stamped score=0 at finalize.
        # Collected here because miners with no commit at all never reach the
        # eval workers and would otherwise be invisible.
        freeze_zero_uids: set[int] = set()
        freeze_zero_hotkeys: dict[int, str] = {}
        for hk in metagraph.hotkeys:
            if hk in assigned_with_valid_ckpt:
                continue
            uid = hotkey_to_uid.get(hk)
            if uid is None:
                continue
            freeze_zero_uids.add(uid)
            freeze_zero_hotkeys[uid] = hk
        if freeze_zero_uids:
            logger.info(
                "Round.freeze: invalid chain checkpoints — will record score=0 at finalize",
                round_id=rid,
                count=len(freeze_zero_uids),
                uids=sorted(freeze_zero_uids),
            )

        foreground_uids = tuple(foreground)

        # Legacy ordering, used only when round-group construction is off:
        # background = top-N by prior avg score, then a staleness tail.
        EPOCH = datetime.min.replace(tzinfo=timezone.utc)
        last_eval_map = last_evaluated or {}
        prior_scores = prior_avg_scores or {}

        bg_set = set(background)
        placed: set[int] = set(foreground)

        # (b) Top-N by prior-round avg score. Random tiebreak.
        scored_candidates = sorted(
            (
                (uid, prior_scores.get(uid, 0.0))
                for uid in bg_set
                if prior_scores.get(uid, 0.0) > 0.0 and uid not in placed
            ),
            key=lambda kv: (-kv[1], random.random()),
        )
        score_prepend_uids = [
            uid for uid, _ in scored_candidates[: cls.BG_TOP_SCORED_PREPEND_COUNT]
        ]
        placed.update(score_prepend_uids)

        # (c) Staleness tail — every remaining UID, oldest evaluation
        # first. Random tiebreak on equal staleness (e.g. all
        # never-evaluated UIDs share the EPOCH key).
        stale_tail = sorted(
            (uid for uid in background if uid not in placed),
            key=lambda uid: (last_eval_map.get(uid, EPOCH), random.random()),
        )

        background_uids = tuple([
            *score_prepend_uids,
            *stale_tail,
        ])

        # Detach + clone so later in-place mutations of global_model cannot
        # leak into the snapshot.
        snapshot = {
            k: v.detach().clone().cpu() for k, v in global_model.state_dict().items()
        }

        logger.info(
            "Round.freeze: roster locked",
            round_id=rid,
            roster_size=len(uid_to_hotkey),
            foreground_size=len(foreground_uids),
            background_size=len(background_uids),
        )
        logger.info(
            "Round.freeze: foreground",
            round_id=rid,
            count=len(foreground_uids),
            uids=list(foreground_uids),
        )
        logger.info(
            "Round.freeze: bg score prepend",
            round_id=rid,
            count=len(score_prepend_uids),
            uids=list(score_prepend_uids),
        )
        logger.info(
            "Round.freeze: bg staleness tail",
            round_id=rid,
            count=len(stale_tail),
            uids=list(stale_tail),
        )

        # Round-group overlay: the cohort advances every 8th cycle and the
        # roster becomes Group A + B + C in that order, replacing the legacy
        # ordering computed above. See docs/miner-validation-group-promotion.md.
        new_cohort_state: "CohortState | None" = None
        new_weight_group_1: tuple[int, ...] = ()
        new_weight_group_2: tuple[int, ...] = ()
        new_validation_a: tuple[int, ...] = ()
        new_validation_b: tuple[int, ...] = ()
        new_validation_c: tuple[int, ...] = ()
        new_cohort_epoch = 0
        flag_enabled = bool(getattr(
            getattr(config, "evaluation", None),
            "enable_round_group_construction",
            False,
        ))
        if flag_enabled:
            from connito.validator import round_groups

            effective_cycle_length = cycle_length
            effective_cycle_index = cycle_index
            if effective_cycle_index is None and effective_cycle_length:
                effective_cycle_index = rid // int(effective_cycle_length)
            if effective_cycle_index is None or not effective_cycle_length:
                logger.warning(
                    "Round.freeze: round-group flag on but cycle_index/cycle_length missing; "
                    "falling back to legacy foreground/background construction",
                    round_id=rid,
                )
            else:
                expert_group = str(getattr(getattr(config, "task", None), "exp", None).group_id) \
                    if getattr(getattr(config, "task", None), "exp", None) is not None else ""
                qualified_validator_uids = [
                    hotkey_to_uid[hk]
                    for hk in assignment.keys()
                    if hk in hotkey_to_uid
                ]
                # Validator seeds for the seeded `assign_miners_to_validators`
                # partitions used to construct Group C and Foreground.
                validator_seeds = get_validator_seed_from_commit(config, commits)
                new_cohort_state = round_groups.maybe_advance_cohort(
                    cycle_index=int(effective_cycle_index),
                    round_id=rid,
                    cycle_length=int(effective_cycle_length),
                    current_state=cohort_state,
                    score_aggregator=score_aggregator,
                    metagraph=metagraph,
                    qualified_validator_uids=qualified_validator_uids,
                    validator_seeds=validator_seeds,
                    all_miner_hotkeys=list(assignment_result.miners_with_checkpoint),
                    my_hotkey=config.chain.hotkey_ss58,
                    hotkey_to_uid=hotkey_to_uid,
                    expert_group=expert_group,
                    cfg=config.evaluation,
                )

                new_weight_group_1 = new_cohort_state.weight_group_1
                new_weight_group_2 = new_cohort_state.weight_group_2
                new_validation_a = new_cohort_state.validation_group_a
                new_validation_b = new_cohort_state.validation_group_b
                new_validation_c = new_cohort_state.validation_group_c
                new_cohort_epoch = new_cohort_state.cohort_epoch

                # foreground = this validator's seeded slice of A∪B;
                # background = the rest of A∪B∪C in A→B→C order.
                foreground_uids, background_uids = round_groups.split_foreground_background(
                    new_cohort_state
                )

                # Tail: committed miners outside the A/B/C roster and the
                # carry-over tier, appended last so they are evaluated only
                # with spare capacity. Staleness-first with a random tiebreak,
                # so equal-staleness UIDs rotate across cycles.
                abc_set = (
                    set(new_validation_a)
                    | set(new_validation_b)
                    | set(new_validation_c)
                )
                cohort_set = abc_set | set(foreground_uids)

                # Carry-over of last round's A/B into background. A no-op
                # within a cohort epoch; at a boundary the input state still
                # holds the previous epoch's groups, so those UIDs get one more
                # eval pass instead of losing their scores at the handoff.
                prev_ab_pool: list[int] = []
                if cohort_state is not None:
                    _seen_prev: set[int] = set()
                    for uid in (
                        *cohort_state.validation_group_a,
                        *cohort_state.validation_group_b,
                    ):
                        if uid in cohort_set or uid in _seen_prev:
                            continue
                        _seen_prev.add(uid)
                        prev_ab_pool.append(uid)
                prev_ab_set = set(prev_ab_pool)

                tail_pool: list[int] = []
                _seen_tail: set[int] = set()
                for uid in (*foreground, *background):
                    if (
                        uid in cohort_set
                        or uid in prev_ab_set
                        or uid in _seen_tail
                    ):
                        continue
                    _seen_tail.add(uid)
                    tail_pool.append(uid)
                tail_pool.sort(
                    key=lambda u: (last_eval_map.get(u, EPOCH), random.random())
                )
                if prev_ab_pool or tail_pool:
                    background_uids = tuple([
                        *background_uids,
                        *prev_ab_pool,
                        *tail_pool,
                    ])

                # Groups A and B may include UIDs outside this validator's
                # assignment slice, which the earlier loop never saw.
                for uid in (*foreground_uids, *background_uids):
                    if uid in uid_to_hotkey:
                        continue
                    if 0 <= uid < len(metagraph.hotkeys):
                        uid_to_hotkey[uid] = metagraph.hotkeys[uid]

                # `gather_validation_job` decides `is_assigned` from
                # `validator_miner_assignment[my_hotkey]`, so overlay-foreground
                # UIDs must be spliced in or they are dropped as unexpected
                # submissions and the whole eval window is wasted. Copy-on-write
                # so callers holding the chain helper's dict aren't mutated.
                my_hk = config.chain.hotkey_ss58
                spliced_assignment: dict[str, list[str]] = {
                    k: list(v) for k, v in assignment.items()
                }
                my_list = spliced_assignment.setdefault(my_hk, [])
                _seen_my_assigned: set[str] = set(my_list)
                for uid in foreground_uids:
                    hk = uid_to_hotkey.get(uid)
                    if hk is None or hk in _seen_my_assigned:
                        continue
                    my_list.append(hk)
                    _seen_my_assigned.add(hk)
                assignment = spliced_assignment

                logger.info(
                    "Round.freeze: round-group overlay applied",
                    round_id=rid,
                    cohort_epoch=new_cohort_epoch,
                    cycle_index=int(effective_cycle_index),
                    validation_group_a=list(new_validation_a),
                    validation_group_b=list(new_validation_b),
                    validation_group_c=list(new_validation_c),
                    foreground_uids=list(foreground_uids),
                    background_uids=list(background_uids),
                    prev_ab_carryover=list(prev_ab_pool),
                    tail_uids=list(tail_pool),
                    weight_group_1=list(new_weight_group_1),
                    weight_group_2=list(new_weight_group_2),
                )

        # No `checkpoint_path` means no journaling — used by tests that build
        # rounds directly.
        resolved_journal_path: "Path | None" = None
        if checkpoint_path is not None:
            from connito.validator import round_journal as _rj
            resolved_journal_path = _rj.journal_path_for(checkpoint_path, rid)

        new_round = cls(
            round_id=rid,
            seed=seed,
            validator_miner_assignment=assignment,
            foreground_uids=foreground_uids,
            background_uids=background_uids,
            uid_to_hotkey=uid_to_hotkey,
            model_snapshot_cpu=snapshot,
            submission_block_range=submission_block_range,
            uid_to_chain_checkpoint=uid_to_chain_checkpoint,
            freeze_zero_uids=freeze_zero_uids,
            freeze_zero_hotkeys=freeze_zero_hotkeys,
            weight_group_1=new_weight_group_1,
            weight_group_2=new_weight_group_2,
            validation_group_a=new_validation_a,
            validation_group_b=new_validation_b,
            validation_group_c=new_validation_c,
            cohort_epoch=new_cohort_epoch,
            cohort_state=new_cohort_state,
            journal_path=resolved_journal_path,
            score_aggregator=score_aggregator,
            score_path=score_path,
        )

        # Initial write, so recovery has the full finalize inputs even if
        # nothing else mutates this round.
        if resolved_journal_path is not None:
            try:
                new_round._persist_journal()
            except Exception as e:
                logger.warning(
                    "Round.freeze: initial journal write failed",
                    error=str(e), round_id=rid, path=str(resolved_journal_path),
                )

        return new_round

    # ---------------- Claim / score helpers ----------------
    def claim_for_foreground(self, uid: int) -> bool:
        with self._lock:
            if (
                uid in self.claimed_uids
                or uid in self.scored_uids
                or uid in self.failed_uids
            ):
                return False
            self.claimed_uids.add(uid)
            return True

    def claim_for_eval(self, uid: int) -> bool:
        with self._lock:
            if (
                uid in self.claimed_uids
                or uid in self.scored_uids
                or uid in self.failed_uids
            ):
                return False
            self.claimed_uids.add(uid)
            return True

    def release_claim(self, uid: int) -> None:
        with self._lock:
            self.claimed_uids.discard(uid)

    def _journal_snapshot_locked(self) -> dict | None:
        """Build the journal payload while holding `self._lock`. Returns
        `None` if journaling is not configured (legacy/test rounds).
        Caller must persist the snapshot OUTSIDE the lock so disk IO
        does not block other workers.
        """
        if self.journal_path is None:
            return None
        from connito.validator.round_journal import commit_map_from_checkpoints

        return {
            "round_id": self.round_id,
            "uid_to_hotkey": dict(self.uid_to_hotkey),
            "scores": dict(self.scores),
            "scored_uids": tuple(sorted(self.scored_uids)),
            "failed_uids": tuple(sorted(self.failed_uids)),
            "validation_failed_uids": tuple(sorted(self.validation_failed_uids)),
            "freeze_zero_uids": tuple(sorted(self.freeze_zero_uids)),
            "freeze_zero_hotkeys": dict(self.freeze_zero_hotkeys),
            "uid_to_commit": commit_map_from_checkpoints(self.uid_to_chain_checkpoint),
            "uid_to_val_loss": dict(self.val_losses),
            "roster_size": len(self.foreground_uids) + len(self.background_uids),
            "lifecycle_step": int(self.lifecycle_step),
            "finalized": False,
        }

    def _persist_journal(self) -> None:
        """Snapshot + write the journal atomically.

        Two-phase: take the snapshot under `self._lock`, then release
        the lock before writing to disk so concurrent eval threads are
        not blocked on IO. Any IO failure logs a warning and returns —
        a journal failure must never abort an evaluation.
        """
        if self.journal_path is None:
            return
        with self._lock:
            payload = self._journal_snapshot_locked()
        if payload is None:
            return
        try:
            from connito.validator import round_journal as _rj
            _rj.write_atomic(
                self.journal_path,
                _rj.RoundJournal(**payload),
            )
        except Exception as e:
            logger.warning(
                "Round: journal write failed",
                error=str(e), round_id=self.round_id, path=str(self.journal_path),
            )

    def _record_in_cycle_score(self, uid: int, hotkey: str, score: float) -> None:
        """Write the raw in-cycle score to the aggregator alongside the
        journal. `finalize_round_scores` calls
        `score_aggregator.drop_round(round_id)` first, so these raw
        entries get cleanly replaced with rank-based ones at finalize.
        Until finalize runs, the aggregator on disk carries the raw
        delta tagged with this round_id — slightly under-weights the
        miner vs. rank-based but survives a kill.
        """
        if self.score_aggregator is None:
            return
        try:
            self.score_aggregator.add_score(
                uid=uid, hotkey=hotkey, score=float(score), round_id=self.round_id,
            )
            if self.score_path is not None:
                self.score_aggregator.persist_atomic(self.score_path)
        except Exception as e:
            logger.warning(
                "Round: in-cycle aggregator write failed",
                error=str(e), uid=uid, round_id=self.round_id,
            )

    def mark_scored(
        self, uid: int, score: float = 0.0, val_loss: float | None = None
    ) -> None:
        """Record a successful evaluation. `score` is this-round's score
        (e.g. ``delta ** 1.2`` from `evaluate_one_miner`); it is stored
        in `self.scores` so per-round ranking — used by post-eval
        submission cleanup — never has to consult the global aggregator.

        `val_loss` is the raw evaluation loss. It is recorded purely so the
        journal can carry it: `validator_miner_val_loss` is published at
        eval time and cannot be reconstructed from `score`, because
        ``delta = max(0.0, baseline - val_loss)`` clamps at zero. Optional
        so existing callers and test fixtures keep working.

        Also writes to the per-round journal and (if configured) the
        score aggregator with the raw delta tagged with this round_id,
        so a kill before `finalize_round_scores` runs does not lose the
        evaluation. Both writes happen OUTSIDE `self._lock` to avoid
        blocking other workers on disk IO.
        """
        score_f = float(score)
        with self._lock:
            self.scored_uids.add(uid)
            self.scores[uid] = score_f
            if val_loss is not None:
                self.val_losses[uid] = float(val_loss)
            self.claimed_uids.discard(uid)
            hotkey = self.uid_to_hotkey.get(uid)
        self._persist_journal()
        if hotkey is not None:
            self._record_in_cycle_score(uid, hotkey, score_f)

    def top_scored_uids_this_round(self, top_k: int) -> set[int]:
        """Top-`top_k` UIDs by *this round's* score. Returns every scored
        UID when fewer than `top_k` have been scored. Ties are broken
        arbitrarily by UID (stable-sort fallback) — the caller only needs
        a set, not a ranking.
        """
        if top_k <= 0:
            return set()
        with self._lock:
            if not self.scores:
                return set()
            if len(self.scores) <= top_k:
                return set(self.scores.keys())
            ranked = sorted(
                self.scores.items(),
                key=lambda kv: (kv[1], -kv[0]),  # score desc, uid asc as tiebreak
                reverse=True,
            )
            return {uid for uid, _ in ranked[:top_k]}

    def mark_failed(self, uid: int) -> None:
        """Mark a UID as failed for operational reasons (download timeout,
        eval timeout, OOM, unexpected exception). Lands in `failed_uids`
        only; finalize will *not* write a score=0 for it — the miner's
        prior EMA is preserved.
        """
        with self._lock:
            self.failed_uids.add(uid)
            self.claimed_uids.discard(uid)
        self._persist_journal()

    def mark_validation_failed(self, uid: int) -> None:
        """Mark a UID as failed through the miner's own fault: an off-spec
        on-disk submission (hash/signature/expert_group/NaN-Inf mismatch
        detected by `validate_miner_submission`) or a committed HF
        checkpoint that is confirmed not publicly retrievable (repo
        deleted/private/gated). Lands in both `failed_uids` and
        `validation_failed_uids`; finalize records score=0 for it.
        """
        with self._lock:
            self.failed_uids.add(uid)
            self.validation_failed_uids.add(uid)
            self.claimed_uids.discard(uid)
        self._persist_journal()

    def publish_download(self, uid: int, path: Path) -> bool:
        with self._lock:
            if uid in self.scored_uids:
                return False
            self.downloaded_pool[uid] = path
            return True

    def pop_downloaded(self, uid: int) -> Path | None:
        with self._lock:
            return self.downloaded_pool.pop(uid, None)

    def has_downloaded(self, uid: int) -> bool:
        with self._lock:
            return uid in self.downloaded_pool

    def downloaded_pending_eval_count(self) -> int:
        """Number of UIDs whose checkpoint has been downloaded but is not
        yet picked up by an eval worker (claimed/scored/failed). Used by
        bg-download to backpressure: when this rises above the configured
        cap, downloads pause until bg-eval has drained the queue.
        """
        with self._lock:
            return sum(
                1 for uid in self.downloaded_pool
                if uid not in self.scored_uids
                and uid not in self.claimed_uids
                and uid not in self.failed_uids
            )

    def processed_uids_snapshot(self) -> tuple[set[int], set[int]]:
        """Lock-protected snapshot of (scored_uids, failed_uids) for callers
        that need a consistent view across both sets in the same instant.
        """
        with self._lock:
            return set(self.scored_uids), set(self.failed_uids)

    # ---------------- Iteration helpers ----------------
    @property
    def assigned_uids(self) -> tuple[int, ...]:
        """Alias for `foreground_uids` — the validator's assignment slice.
        Kept under a separate name so callers can express *intent* without
        coupling to the fact that today every assigned miner is also
        evaluated in foreground.
        """
        return self.foreground_uids

    @property
    def roster(self) -> tuple[RosterEntry, ...]:
        """Foreground first, then background, both already incentive-ordered."""
        return tuple(
            RosterEntry(uid=uid, hotkey=self.uid_to_hotkey[uid])
            for uid in (*self.foreground_uids, *self.background_uids)
        )

    def next_for_download(self) -> Iterable[RosterEntry]:
        """Yield roster UIDs (foreground first, then background) in priority
        order that are not yet downloaded, scored, or claimed. Re-checks state
        each iteration so pause/resume stays correct.

        Foreground UIDs are yielded first because they are this validator's
        assignment slice — `gather_validation_job` (called by
        `evaluate_foreground_round`) scans `miner_submission_path`, so until
        bg-download writes a foreground miner's shard to disk, foreground eval
        polls forever and finds nothing. Walking foreground first puts the
        priority work where it's needed.
        """
        for uid in (*self.foreground_uids, *self.background_uids):
            with self._lock:
                if (
                    uid in self.scored_uids
                    or uid in self.failed_uids
                    or uid in self.downloaded_pool
                    or uid in self.claimed_uids
                ):
                    continue
            yield RosterEntry(uid=uid, hotkey=self.uid_to_hotkey[uid])

    def next_for_eval(self) -> Iterable[RosterEntry]:
        """Yield (uid, hotkey) for every miner whose checkpoint is downloaded
        and not yet scored/claimed/failed, in foreground-then-background order."""
        with self._lock:
            candidates = {
                u for u in self.downloaded_pool
                if u not in self.scored_uids
                and u not in self.claimed_uids
                and u not in self.failed_uids
            }
        for uid in (*self.foreground_uids, *self.background_uids):
            if uid in candidates:
                yield RosterEntry(uid=uid, hotkey=self.uid_to_hotkey[uid])

    def unscored_roster_uids(self) -> list[RosterEntry]:
        """Assigned miners this validator did not score this round. Scoped
        to `foreground_uids` so it never returns miners that belong to
        other validators' assignments. No longer used for penalties (we
        only score=0 for invalid checkpoints, recorded inline at the
        validation site); kept for diagnostics and future use."""
        with self._lock:
            return [
                RosterEntry(uid=uid, hotkey=self.uid_to_hotkey[uid])
                for uid in self.foreground_uids
                if uid not in self.scored_uids
            ]

    # ---------------- Stats ----------------
    def stats(self) -> dict[str, int]:
        roster_size = len(self.foreground_uids) + len(self.background_uids)
        with self._lock:
            return {
                "roster": roster_size,
                "scored": len(self.scored_uids),
                "failed": len(self.failed_uids),
                "downloaded": len(self.downloaded_pool),
                "claimed": len(self.claimed_uids),
                "pending": roster_size - len(self.scored_uids) - len(self.failed_uids),
            }

    def publish_progress(self) -> None:
        """Publish this round's progress counters (scored / failed / pending)
        to Prometheus.

        Called from every path that advances the round — once at freeze, after
        each foreground evaluation, and after each background evaluation — so
        the counters move as soon as evaluation starts rather than only once
        the background eval window opens at Merge.

        `stats()` takes `self._lock`, so callers must NOT hold it. Every call
        site is outside the lock (the `mark_*` helpers release it before
        returning). Best-effort: `set_round_progress` swallows telemetry
        errors so this can never affect scoring.
        """
        stats = self.stats()
        set_round_progress(
            self.round_id,
            scored=stats["scored"],
            failed=stats["failed"],
            pending=stats["pending"],
        )


@dataclass
class RoundRef:
    """Mutable holder for the workers to follow as the main loop swaps rounds.

    Workers re-read `current` on every iteration so a swap takes effect
    without restarting the thread.
    """

    current: Round | None = None
    previous: Round | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def swap(self, new_current: Round) -> Round | None:
        with self._lock:
            old = self.current
            self.previous = old
            self.current = new_current
            return old
