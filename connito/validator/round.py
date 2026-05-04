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
from datetime import datetime, timezone
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
    # Per-uid score recorded by `mark_scored`. Scoped to *this round*
    # only — kept here so cleanup ranking does not have to read the
    # global `MinerScoreAggregator` (which mixes in scores from prior
    # rounds and would let history pull a non-top-this-round miner into
    # the keep set).
    scores: dict[int, float] = field(default_factory=dict)
    claimed_uids: set[int] = field(default_factory=set)
    failed_uids: set[int] = field(default_factory=set)
    # UIDs the miner is at fault for: explicit validation failures
    # (hash/signature/expert_group/NaN-Inf/no_chain_commit) or freeze-time
    # invalid checkpoints. These get score=0 in the aggregator at finalize.
    # `failed_uids ⊃ validation_failed_uids` — operational failures
    # (timeout/OOM/exception/download failure) are in `failed_uids` only
    # and intentionally receive *no* aggregator entry, so the miner keeps
    # its prior EMA. The validator's lack of compute/bandwidth must not
    # dock a miner's reward.
    validation_failed_uids: set[int] = field(default_factory=set)
    # Freeze-time invalid-checkpoint penalties. Hotkey map is captured
    # alongside because these UIDs may not appear in `uid_to_hotkey`
    # (which only covers roster miners with a valid checkpoint).
    freeze_zero_uids: set[int] = field(default_factory=set)
    freeze_zero_hotkeys: dict[int, str] = field(default_factory=dict)
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

        # Freeze-time penalty: every metagraph neuron that lacks a valid
        # chain checkpoint this round is recorded here so
        # `finalize_round_scores` can stamp it with score=0 in the
        # aggregator at end of round. Catching it on the main thread
        # also covers miners with no commit at all — those never appear
        # in `miners_with_checkpoint` and would otherwise be invisible
        # to the eval workers entirely.
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
            logger.info(
                "Round.freeze: invalid chain checkpoint — will record score=0 at finalize",
                uid=uid, hotkey=hk[:6], round_id=rid,
            )

        foreground_uids = tuple(foreground)

        # Order *background* in three segments. Earlier segments take
        # priority and are de-duplicated against later ones via `placed`,
        # which starts as the foreground set so foreground UIDs cannot
        # leak into any background segment:
        #   (a) UIDs receiving non-zero weight from a qualified validator
        #       (i.e. one whose hotkey is a key in
        #       `assignment_result.assignment`). Ranked by stake-weighted
        #       total weight across qualified validators, descending.
        #       Lets chain consensus pull a previously unseen miner ahead
        #       of this validator's own EMA leaders.
        #   (b) top-N background miners by their prior-round avg score
        #       (`prior_avg_scores`, the same metric that determines the
        #       weight submission), excluding anyone already in (a).
        #       Re-evaluating the current local leaders first protects
        #       the top of the leaderboard against a stale EMA — without
        #       this, a strong miner that hasn't been picked recently
        #       could keep its lead even after submitting a worse
        #       checkpoint. Capped at `BG_TOP_SCORED_PREPEND_COUNT` so
        #       it cannot crowd out the staleness rotation.
        #   (c) everyone else, sorted by staleness — longest-since-last-
        #       evaluated first, never-evaluated UIDs treated as
        #       infinitely stale. Rotates the long tail through bg-eval
        #       instead of always favoring the same incentive-ranked
        #       head. Each validator has different `last_evaluated` so
        #       the tail spreads naturally across the subnet.
        # All three segments break ties randomly (no UID-asc fallback)
        # so a low-numbered UID has no systematic head start.
        # Foreground stays in incentive order; that's the priority set
        # by design and the per-round eval budget covers it.
        EPOCH = datetime.min.replace(tzinfo=timezone.utc)
        last_eval_map = last_evaluated or {}
        prior_scores = prior_avg_scores or {}

        bg_set = set(background)
        placed: set[int] = set(foreground)

        # (a) Chain-weight prepend — UIDs other qualified validators are
        # already rewarding. Requires `metagraph.weights` to be populated;
        # the caller must fetch the metagraph with `lite=False`.
        weight_prepend_uids: list[int] = []
        weights_attr = getattr(metagraph, "weights", None)
        if weights_attr is not None:
            try:
                W = weights_attr if isinstance(weights_attr, torch.Tensor) \
                    else torch.as_tensor(weights_attr)
            except Exception:
                W = None
            if W is None or W.ndim != 2 or W.shape[0] == 0:
                logger.debug(
                    "Round.freeze: metagraph.weights empty or malformed — "
                    "skipping chain-weight prepend",
                    shape=getattr(weights_attr, "shape", None),
                )
                W = None
            if W is not None:
                stake_attr = getattr(metagraph, "S", None)
                S = None
                if stake_attr is not None:
                    try:
                        S = stake_attr if isinstance(stake_attr, torch.Tensor) \
                            else torch.as_tensor(stake_attr)
                    except Exception:
                        S = None

                consensus: dict[int, float] = {}
                n_rows, n_cols = int(W.shape[0]), int(W.shape[1])
                for v_hk in assignment.keys():
                    v_uid = hotkey_to_uid.get(v_hk)
                    if v_uid is None or v_uid >= n_rows:
                        continue
                    row = W[v_uid]
                    stake = (
                        float(S[v_uid].item())
                        if (S is not None and v_uid < int(S.shape[0]))
                        else 1.0
                    )
                    # Iterate only over non-zero entries (matrix is sparse
                    # in practice — most validators reward a small subset).
                    nonzero_idx = torch.nonzero(row, as_tuple=True)[0].tolist()
                    for m_uid in nonzero_idx:
                        if m_uid in bg_set and m_uid not in placed:
                            w = float(row[m_uid].item())
                            consensus[m_uid] = consensus.get(m_uid, 0.0) + stake * w

                weight_prepend_uids = [
                    uid for uid, _ in sorted(
                        consensus.items(),
                        key=lambda kv: (-kv[1], random.random()),
                    )
                ]
                placed.update(weight_prepend_uids)
        else:
            logger.debug(
                "Round.freeze: metagraph has no `weights` attribute — "
                "skipping chain-weight prepend (metagraph likely fetched with lite=True)",
            )

        # (b) Top-N by prior-round avg score, excluding anyone already
        # placed (foreground or (a)). Random tiebreak.
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
            *weight_prepend_uids,
            *score_prepend_uids,
            *stale_tail,
        ])

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
            bg_chain_weight_prepend=len(weight_prepend_uids),
            bg_score_prepend=len(score_prepend_uids),
            bg_staleness_tail=len(stale_tail),
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
            freeze_zero_uids=freeze_zero_uids,
            freeze_zero_hotkeys=freeze_zero_hotkeys,
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

    def mark_scored(self, uid: int, score: float = 0.0) -> None:
        """Record a successful evaluation. `score` is this-round's score
        (e.g. ``delta ** 1.2`` from `evaluate_one_miner`); it is stored
        in `self.scores` so per-round ranking — used by post-eval
        submission cleanup — never has to consult the global aggregator.
        """
        with self._lock:
            self.scored_uids.add(uid)
            self.scores[uid] = float(score)
            self.claimed_uids.discard(uid)

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

    def mark_validation_failed(self, uid: int) -> None:
        """Mark a UID as failed because its on-disk submission is off-spec
        (hash/signature/expert_group/NaN-Inf mismatch detected by
        `validate_miner_submission`). Lands in both `failed_uids` and
        `validation_failed_uids`; finalize records score=0 for it.
        """
        with self._lock:
            self.failed_uids.add(uid)
            self.validation_failed_uids.add(uid)
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
