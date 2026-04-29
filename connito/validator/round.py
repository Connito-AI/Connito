"""Per-round state for the lifecycle (0)..(4) defined in
`_specs/background-submission-validation.md`.

A `Round` is constructed once, at the start of each Submission phase
(step 0), and is immutable thereafter. The foreground pass and the two
background workers (download + eval) all anchor on the same `Round`.
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, NamedTuple

import torch
import torch.nn as nn

from connito.shared.app_logging import structlog

if TYPE_CHECKING:
    from connito.shared.checkpoints import ChainCheckpoint

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

    # Per-round eval telemetry consumed by connito.validator.api. These are
    # written by evaluate_foreground_round (baseline_loss, once) and
    # evaluate_one_miner (val_loss_by_uid[uid], once per scored miner). Kept
    # on the Round so the API endpoint can read everything from a single
    # locked snapshot without reaching back into the eval pipeline.
    baseline_loss: float | None = None
    val_loss_by_uid: dict[int, float] = field(default_factory=dict)

    # Subnet shape captured at freeze time — feeds the "Total miners on
    # subnet" tile on the leaderboard frontend. Kept on the Round so the
    # API doesn't have to issue a fresh metagraph RPC per request.
    total_subnet_uids: int = 0  # every UID in the metagraph (validators + miners)
    validator_count: int = 0    # whitelisted validators (== len(assignment))

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
        for hk in assignment_result.miners_with_checkpoint:
            uid = hotkey_to_uid.get(hk)
            if uid is None:
                logger.warning("Round.freeze: hotkey not in metagraph; skipping", hotkey=hk[:6])
                continue
            uid_to_hotkey[uid] = hk
            ckpt = chain_checkpoints_by_hotkey.get(hk)
            if ckpt is not None:
                uid_to_chain_checkpoint[uid] = ckpt
            (foreground if hk in my_assignment_set else background).append(uid)

        foreground_uids = tuple(foreground)

        rid = int(round_id) if round_id is not None else int(subtensor.block)

        # Shuffle background deterministically per (validator, round) so each
        # validator hits HF in a different order — spreads download load
        # across the subnet instead of every validator racing for the same
        # top-incentive miners first. Stable within a round (workers see the
        # same order across re-reads) but varies cycle-to-cycle so no miner
        # is permanently stuck at the back of any one validator's queue.
        random.Random(f"{config.chain.hotkey_ss58}:{rid}").shuffle(background)
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

        # Subnet shape snapshot. `metagraph.hotkeys` covers every UID
        # (validators + miners); `assignment` is keyed by validator hotkey,
        # so its length is the whitelisted validator count.
        total_subnet_uids = len(metagraph.hotkeys)
        validator_count = len(assignment)

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
            total_subnet_uids=total_subnet_uids,
            validator_count=validator_count,
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

    # ---------------- Iteration helpers ----------------
    @property
    def assigned_uids(self) -> tuple[int, ...]:
        """Alias for `foreground_uids` — the validator's assignment slice.
        Kept under a separate name so callers (e.g. the missed-submission
        penalty pass) can express *intent* without coupling to the fact
        that today every assigned miner is also evaluated in foreground.
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
        to `foreground_uids` so the penalty pass does not zero out miners
        that belong to other validators' assignments."""
        with self._lock:
            return [
                RosterEntry(uid=uid, hotkey=self.uid_to_hotkey[uid])
                for uid in self.foreground_uids
                if uid not in self.scored_uids
            ]

    # ---------------- Eval telemetry stashes ----------------
    def set_baseline_loss(self, loss: float) -> None:
        """Record the round's baseline loss. Called once per round by
        ``evaluate_foreground_round`` after the baseline pass completes."""
        with self._lock:
            self.baseline_loss = float(loss)

    def record_val_loss(self, uid: int, val_loss: float) -> None:
        """Record per-miner validation loss. Called by ``evaluate_one_miner``
        for each successfully evaluated miner."""
        with self._lock:
            self.val_loss_by_uid[int(uid)] = float(val_loss)

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

    def snapshot(self) -> dict:
        """Lock-guarded snapshot of every field the /v1/state.json endpoint
        needs. One acquire/release covers the full read so the API never
        races with the eval workers updating scored_uids / val_loss_by_uid.
        """
        roster_size = len(self.foreground_uids) + len(self.background_uids)
        with self._lock:
            return {
                "round_id": self.round_id,
                "baseline_loss": self.baseline_loss,
                "foreground_uids": tuple(self.foreground_uids),
                "uid_to_hotkey": dict(self.uid_to_hotkey),
                "uid_to_chain_checkpoint": dict(self.uid_to_chain_checkpoint),
                "val_loss_by_uid": dict(self.val_loss_by_uid),
                "total_subnet_uids": self.total_subnet_uids,
                "validator_count": self.validator_count,
                "stats": {
                    "roster": roster_size,
                    "scored": len(self.scored_uids),
                    "failed": len(self.failed_uids),
                    "downloaded": len(self.downloaded_pool),
                    "claimed": len(self.claimed_uids),
                    "pending": roster_size - len(self.scored_uids) - len(self.failed_uids),
                },
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
