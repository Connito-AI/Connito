"""Per-round state for the lifecycle (0)..(4) defined in
`_specs/background-submission-validation.md`.

A `Round` is constructed once, at the start of each Submission phase
(step 0), and is immutable thereafter. The foreground pass and the two
background workers (download + eval) all anchor on the same `Round`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, NamedTuple

import torch
import torch.nn as nn

from connito.shared.app_logging import structlog

if TYPE_CHECKING:
    from connito.shared.checkpoints import ChainCheckpoint
    from connito.validator.aggregator import MinerScoreAggregator

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
    # On-chain Submission phase block range for this round. bg-download uses
    # it to gate `_existing_submission` reuse — without this filter, a stale
    # .pt left over from a previous cycle would short-circuit the fresh
    # fetch and get published into downloaded_pool, but `gather_validation_job`
    # would silently reject it because its block falls outside the window.
    submission_block_range: tuple[int, int] | None = None
    # Per-uid `ChainCheckpoint` snapshot captured at freeze time so the eval
    # path can run `validate(expert_group_assignment=...)` (signature, hash,
    # expert-group ownership, NaN/Inf scan) without re-issuing chain RPCs.
    uid_to_chain_checkpoint: dict[int, "ChainCheckpoint"] = field(default_factory=dict)

    # Mutable, lock-guarded
    downloaded_pool: dict[int, Path] = field(default_factory=dict)
    scored_uids: set[int] = field(default_factory=set)
    claimed_uids: set[int] = field(default_factory=set)
    failed_uids: set[int] = field(default_factory=set)
    weights_submitted: bool = False

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    # ---------------- Construction ----------------
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
        score_aggregator: "MinerScoreAggregator | None" = None,
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
        )

        # Fetch head-block chain commits ONCE and pass to both helpers; they
        # would otherwise each issue a duplicate `get_all_commitments` +
        # `metagraph()` pair against the archive endpoint, serialized through
        # the global subtensor lock. Same for the metagraph already passed in
        # by the caller — `get_validator_miner_assignment` reuses it instead
        # of re-fetching head-block state.
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

        # Freeze-time penalty: every neuron in the metagraph that is not
        # in `assignment_result.miners_with_checkpoint` with a valid chain
        # commit gets score=0 under this round_id. Doing it here —
        # synchronously, on the main thread — keeps the worker threads
        # out of the aggregator and catches miners with no commit at all
        # (those never appear in `miners_with_checkpoint` and so would
        # be invisible to the bg-download path).
        if score_aggregator is not None:
            for hk in metagraph.hotkeys:
                if hk in assigned_with_valid_ckpt:
                    continue
                uid = hotkey_to_uid.get(hk)
                if uid is None:
                    continue
                score_aggregator.add_score(
                    uid=uid, hotkey=hk, score=0.0, round_id=rid,
                )
                logger.info(
                    "Round.freeze: invalid chain checkpoint — penalizing with score=0",
                    uid=uid, hotkey=hk[:6], round_id=rid,
                )

        foreground_uids = tuple(foreground)

        # Order *background* by staleness — longest-since-last-evaluated
        # first, never-evaluated UIDs treated as infinitely stale. This
        # ensures the long tail of the roster eventually rotates through
        # bg-eval instead of always favoring the same incentive-ranked
        # head. Foreground stays in incentive order; that's the priority
        # set by design and the per-round eval budget covers it. Each
        # validator has different `last_evaluated` so background also
        # spreads naturally across the subnet without an explicit shuffle.
        EPOCH = datetime.min.replace(tzinfo=timezone.utc)
        last_eval_map = last_evaluated or {}

        def _staleness_key(uid: int) -> datetime:
            return last_eval_map.get(uid, EPOCH)

        background.sort(key=_staleness_key)
        background_uids = tuple(background)

        # CPU-resident clone of global_model.state_dict(). Detach + clone +
        # move to CPU so subsequent in-place mutations of global_model
        # cannot leak into the snapshot.
        snapshot = {
            k: v.detach().clone().cpu() for k, v in global_model.state_dict().items()
        }

        logger.info(
            "Round.freeze: roster locked",
            round_id=rid,
            roster_size=len(uid_to_hotkey),
            foreground_size=len(foreground_uids),
            background_size=len(background_uids),
            foreground_uids=list(foreground_uids),
        )

        return cls(
            round_id=rid,
            seed=seed,
            validator_miner_assignment=assignment,
            foreground_uids=foreground_uids,
            background_uids=background_uids,
            uid_to_hotkey=uid_to_hotkey,
            model_snapshot_cpu=snapshot,
            submission_block_range=submission_block_range,
            uid_to_chain_checkpoint=uid_to_chain_checkpoint,
        )

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

    def mark_scored(self, uid: int) -> None:
        with self._lock:
            self.scored_uids.add(uid)
            self.claimed_uids.discard(uid)

    def mark_failed(self, uid: int) -> None:
        with self._lock:
            self.failed_uids.add(uid)
            self.claimed_uids.discard(uid)

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
