"""Per-round state for the lifecycle (0)..(4) defined in
`_specs/background-submission-validation.md`.

A `Round` is constructed once, at the start of each Submission phase
(step 0), and is immutable thereafter. The foreground pass and the two
background workers (download + eval) all anchor on the same `Round`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from connito.shared.app_logging import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RosterEntry:
    uid: int
    hotkey: str
    incentive: float


@dataclass
class Round:
    """Immutable snapshot of round K's inputs + mutable per-worker state.

    The frozen pieces (round_id, seed, validator_miner_assignment, roster,
    foreground_uids, background_uids, model_snapshot_cpu) are written once
    by `Round.freeze` and never changed. The mutable pieces
    (downloaded_pool, scored_uids, failed_uids, weights_submitted) are
    guarded by an internal lock and are updated by the workers.
    """

    round_id: int
    seed: str
    validator_miner_assignment: dict[str, list[str]]
    roster: tuple[RosterEntry, ...]
    foreground_uids: tuple[int, ...]
    background_uids: tuple[int, ...]
    model_snapshot_cpu: dict[str, torch.Tensor]

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
        lite_subtensor,
        global_model: nn.Module,
        top_n: int,
        round_id: int | None = None,
    ) -> "Round":
        """Build a Round at Submission-phase start.

        Captures the metagraph incentive snapshot and the global_model
        state_dict (CPU clone) before Merge(K) can mutate either.
        """
        from connito.shared.cycle import (
            get_combined_validator_seed,
            get_validator_miner_assignment,
        )

        seed = get_combined_validator_seed(config, subtensor)
        assignment = get_validator_miner_assignment(config, subtensor)
        my_assignment = assignment.get(config.chain.hotkey_ss58, [])

        metagraph = lite_subtensor.metagraph(netuid=config.chain.netuid)
        incentive = metagraph.incentive  # torch.Tensor
        hotkeys = list(metagraph.hotkeys)

        roster_entries: list[RosterEntry] = []
        for hk in my_assignment:
            try:
                uid = hotkeys.index(hk)
            except ValueError:
                logger.warning("Round.freeze: assigned hotkey not in metagraph; skipping", hotkey=hk[:6])
                continue
            try:
                inc = float(incentive[uid].item())
            except Exception:
                inc = 0.0
            roster_entries.append(RosterEntry(uid=uid, hotkey=hk, incentive=inc))

        # Sort by incentive desc; tie-break on uid for determinism.
        roster_entries.sort(key=lambda e: (-e.incentive, e.uid))
        roster = tuple(roster_entries)

        cap = max(0, int(top_n))
        foreground_uids = tuple(e.uid for e in roster[:cap])
        background_uids = tuple(e.uid for e in roster[cap:])

        # CPU-resident clone of global_model.state_dict(). Detach + clone +
        # move to CPU so subsequent in-place mutations of global_model
        # cannot leak into the snapshot.
        snapshot = {
            k: v.detach().clone().cpu() for k, v in global_model.state_dict().items()
        }

        rid = int(round_id) if round_id is not None else int(subtensor.block)

        logger.info(
            "Round.freeze: roster locked",
            round_id=rid,
            roster_size=len(roster),
            foreground_size=len(foreground_uids),
            background_size=len(background_uids),
            top_uids=[e.uid for e in roster[:cap]],
        )

        return cls(
            round_id=rid,
            seed=seed,
            validator_miner_assignment=assignment,
            roster=roster,
            foreground_uids=foreground_uids,
            background_uids=background_uids,
            model_snapshot_cpu=snapshot,
        )

    # ---------------- Claim / score helpers ----------------
    def claim_for_foreground(self, uid: int) -> bool:
        with self._lock:
            if uid in self.claimed_uids or uid in self.scored_uids:
                return False
            self.claimed_uids.add(uid)
            return True

    def claim_for_eval(self, uid: int) -> bool:
        with self._lock:
            if uid in self.claimed_uids or uid in self.scored_uids:
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
    def next_for_download(self) -> Iterable[RosterEntry]:
        """Yield background_uids in incentive order that are not yet
        downloaded, scored, or claimed. Re-checks state each iteration so
        pause/resume stays correct."""
        bg_set = set(self.background_uids)
        # Walk roster in original (incentive-desc) order to preserve priority.
        for entry in self.roster:
            if entry.uid not in bg_set:
                continue
            with self._lock:
                if (
                    entry.uid in self.scored_uids
                    or entry.uid in self.failed_uids
                    or entry.uid in self.downloaded_pool
                    or entry.uid in self.claimed_uids
                ):
                    continue
            yield entry

    def next_for_eval(self) -> Iterable[RosterEntry]:
        """Yield roster entries whose checkpoint is downloaded and not yet
        scored/claimed."""
        by_uid = {e.uid: e for e in self.roster}
        with self._lock:
            candidates = [u for u in self.downloaded_pool if u not in self.scored_uids and u not in self.claimed_uids]
        # Iterate in incentive order over candidates to keep behavior
        # deterministic across validators.
        ordered = [e for e in self.roster if e.uid in candidates and e.uid in by_uid]
        for entry in ordered:
            yield entry

    def unscored_roster_uids(self) -> list[RosterEntry]:
        with self._lock:
            return [e for e in self.roster if e.uid not in self.scored_uids]

    # ---------------- Stats ----------------
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "roster": len(self.roster),
                "scored": len(self.scored_uids),
                "failed": len(self.failed_uids),
                "downloaded": len(self.downloaded_pool),
                "claimed": len(self.claimed_uids),
                "pending": len(self.roster) - len(self.scored_uids) - len(self.failed_uids),
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
