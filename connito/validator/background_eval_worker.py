"""Step (3) of the round lifecycle: GPU-bound background evaluation.

Active only inside the (3) window: from end of Validate(K) to end of
Train(K+1). Pulls UIDs from `Round.downloaded_pool` and runs
`evaluate_one_miner` against this worker's own `eval_base_model` (loaded
once per round from `round.model_snapshot_cpu`, so Merge(K) cannot
change the round's reference state mid-evaluation).

GPU-lock yielding invariant: `gpu_eval_lock` is acquired only for the
narrow `load_state_dict` and `evaluate_one_miner` calls. It MUST NOT be
held across `await`, across `Event.wait`, or across iteration
boundaries.
"""

from __future__ import annotations

import asyncio
import copy
import gc
import math
import threading
import time
from pathlib import Path

import torch
import torch.nn as nn

from connito.shared.app_logging import structlog
from connito.shared.helper import load_state_dict_from_path
from connito.shared.telemetry import (
    VALIDATOR_BG_EVAL_LOCK_LEAK_TOTAL,
    VALIDATOR_BG_EVAL_RECYCLE_TOTAL,
    VALIDATOR_BG_EVAL_STUCK_LOCK_ITERATIONS,
    VALIDATOR_BG_WORKER_PAUSED,
    VALIDATOR_DEDUP_PAIRS_EVALUATED_TOTAL,
    VALIDATOR_DEDUP_WOULD_FLAG_TOTAL,
    note_round_series,
)
from connito.validator.dedup import (
    DEFAULT_DEDUP_THRESHOLD,
    average_state_dicts,
    compute_merge_penalty,
    delta_cosine,
    find_submission_path,
    is_redundant,
    recover_val_loss,
    select_pairs,
    shadow_report,
)
from connito.validator.evaluator import (
    EVAL_MAX_BATCHES,
    cleanup_non_top_submissions,
    evaluate_one_miner_sync,
    retention_top_k,
)
from connito.validator.round import RoundRef

# Recycling threshold: how many consecutive iterations may observe
# `gpu_eval_lock` held at the iteration boundary before we drop our
# `_eval_base_model` reference and re-park, forcing the main loop to
# re-seed us. ~3 iterations × 2s poll = ~6s; long enough that a slow-
# but-completing eval isn't penalized, short enough that a leaked lock
# doesn't silently wedge the worker for hours (as observed in
# notebooks/data/validator_A6000_v0.1.38.log).
DEFAULT_STUCK_LOCK_RECYCLE_THRESHOLD = 3

# Grace period added to the outer `wait_for` deadline. The in-thread
# `EvalDeadlineExceeded` fires at `now + per_miner_eval_timeout_sec`,
# but the eval loop only checks between batches — so we let the awaiter
# wait an extra GRACE seconds for the in-flight batch to drain and the
# `with lock` block to unwind. If the awaiter still times out, the
# thread is genuinely stuck and the recycler will catch it.
EVAL_DEADLINE_GRACE_SEC = 30.0

# Backoff delays (seconds) between dataloader build/materialize attempts
# inside `_load_round_snapshot`. Total budget ~40 s — short enough that
# the retry loop never bites into the eval window meaningfully (cycle is
# ~90 min), long enough to absorb transient HF blips of the sort that
# triggered the lock-leak wedges (single timeout, recovered seconds
# later). After exhausting retries, fall back to the degraded baseline
# path so the round still finalizes — just with poorer scoring fidelity
# rather than indefinite retries.
DATALOADER_BUILD_RETRY_DELAYS_SEC: tuple[float, ...] = (0.0, 10.0, 30.0)

logger = structlog.get_logger(__name__)


class BackgroundEvalWorker(threading.Thread):
    def __init__(
        self,
        *,
        config,
        round_ref: RoundRef,
        device: torch.device,
        tokenizer,
        merge_phase_active: threading.Event,
        eval_window_active: threading.Event,
        gpu_eval_lock: threading.Lock,
        expert_group_assignment,
        stop_event: threading.Event | None = None,
        poll_interval_sec: float = 2.0,
        stuck_lock_recycle_threshold: int = DEFAULT_STUCK_LOCK_RECYCLE_THRESHOLD,
    ) -> None:
        super().__init__(daemon=True, name="connito-bg-eval")
        self.config = config
        self.round_ref = round_ref
        self.device = device
        self.tokenizer = tokenizer
        self.merge_phase_active = merge_phase_active
        self.eval_window_active = eval_window_active
        self.gpu_eval_lock = gpu_eval_lock
        self.expert_group_assignment = expert_group_assignment
        self.stop_event = stop_event or threading.Event()
        self.poll_interval_sec = poll_interval_sec
        self.stuck_lock_recycle_threshold = stuck_lock_recycle_threshold
        # Model is handed in by the main loop (see set_eval_base_model)
        # right after foreground eval completes, instead of being
        # re-fetched from chain at startup.
        self._eval_base_model: nn.Module | None = None
        self._eval_base_model_lock = threading.Lock()
        self._loaded_round_id: int | None = None
        self._loaded_baseline_loss: float | None = None
        # Round-scoped cache of materialized eval batches. Built once in
        # `_load_round_snapshot` from the streaming dataloader, then
        # iterated by every miner's eval this round. Same combined seed
        # → same batches, so per-miner re-streaming was wasted work AND
        # the trigger for the HF-stall lock-leak.
        self._cached_batches: list | None = None
        # Stuck-lock detection: incremented every iteration that observes
        # `gpu_eval_lock` held at the top-of-loop assertion, reset to 0
        # on any successful (or even attempted) eval iteration. When the
        # streak crosses `stuck_lock_recycle_threshold`, we drop our
        # `_eval_base_model` reference and re-park; the next foreground
        # eval re-seeds us via `set_eval_base_model`.
        self._stuck_lock_streak: int = 0
        # Dedup shadow-pass state. Keyed by `_dedup_round_id` — NOT by
        # `_loaded_round_id`, which the stuck-lock recycler nulls for the
        # SAME round; keying dedup state to it would re-evaluate pairs
        # after a recycle. Reset only when a genuinely new round arrives
        # (see `_reset_dedup_state_if_new_round`).
        self._dedup_round_id: int | None = None
        self._scored_paths: dict[int, Path] = {}
        self._scored_val_loss: dict[int, float] = {}
        self._dedup_pairs_done: set[frozenset[int]] = set()
        self._dedup_budget_used: int = 0
        self._dedup_pairs_skipped: int = 0
        self._dedup_summary_emitted: bool = False

    # ---------------- Public lifecycle ----------------
    def stop(self) -> None:
        self.stop_event.set()

    def set_eval_base_model(self, model: nn.Module) -> None:
        """Hand the worker a model to use as its eval base.

        Called by the main loop after foreground eval completes, so the
        worker doesn't need to re-fetch and re-construct the model from
        chain. The state_dict is reloaded per round from
        `round.model_snapshot_cpu`, so what matters here is the model
        architecture, not its current weights.
        """
        model.to(self.device)
        model.eval()
        with self._eval_base_model_lock:
            self._eval_base_model = model

    def has_eval_base_model(self) -> bool:
        with self._eval_base_model_lock:
            return self._eval_base_model is not None

    # ---------------- Thread body ----------------
    def run(self) -> None:
        try:
            asyncio.run(self._loop())
        except Exception:
            logger.exception("BackgroundEvalWorker crashed")

    # ---------------- Internal ----------------
    async def _loop(self) -> None:
        # The eval_base_model is handed to us by the main loop via
        # set_eval_base_model() after foreground eval completes. Until
        # then we just gate-loop. This avoids duplicating chain fetches
        # + MoE construction on a separate thread at startup.
        logger.info(
            "BackgroundEvalWorker: started",
            device=str(self.device),
            poll_interval_sec=self.poll_interval_sec,
            per_miner_eval_timeout_sec=self.config.evaluation.per_miner_eval_timeout_sec,
        )

        # Rate-limit idle-state logs.
        IDLE_LOG_EVERY = 10  # ~20s at 2s poll interval
        idle_ticks = 0
        try:
            while not self.stop_event.is_set():
                # Lock-yielding invariant + stuck-lock recycler. If the
                # lock is held at the iteration boundary by an orphan,
                # increment the streak; if it persists past the
                # threshold, drop our `_eval_base_model` and re-park —
                # the next foreground eval will re-seed us via
                # `set_eval_base_model`. Without this, a single timeout
                # whose GPU thread never returns wedges the worker for
                # the rest of the process's life.
                if self._stuck_lock_check_and_maybe_recycle():
                    # We just recycled. Skip the rest of this iteration
                    # so we re-enter the gate loop and wait for re-seed.
                    await asyncio.sleep(self.poll_interval_sec)
                    continue

                round_obj = self.round_ref.current
                gated = (
                    round_obj is None
                    or self._eval_base_model is None
                    or self.merge_phase_active.is_set()
                    or not self.eval_window_active.is_set()
                )
                try:
                    VALIDATOR_BG_WORKER_PAUSED.labels(worker="eval").set(1 if gated else 0)
                except Exception:
                    pass
                if gated:
                    if idle_ticks % IDLE_LOG_EVERY == 0:
                        logger.debug(
                            "bg-eval: gated",
                            has_round=round_obj is not None,
                            has_eval_base_model=self._eval_base_model is not None,
                            merge_phase_active=self.merge_phase_active.is_set(),
                            eval_window_active=self.eval_window_active.is_set(),
                        )
                    idle_ticks += 1
                    await self._wait_clear()
                    continue

                # Reload state_dict on round transition.
                if round_obj.round_id != self._loaded_round_id:
                    await self._load_round_snapshot(round_obj)

                target = self._next_target(round_obj)
                if target is None:
                    # Log only on the transition into idle; stay quiet until
                    # new work arrives (idle_ticks resets to 0 on the next
                    # successful claim, re-arming this log for the next gap).
                    if idle_ticks == 0:
                        try:
                            stats = round_obj.stats()
                        except Exception:
                            stats = None
                        logger.info(
                            "bg-eval: no pending targets — going idle",
                            round_id=round_obj.round_id,
                            round_stats=stats,
                        )
                    idle_ticks += 1
                    # Dedup shadow pass: spend idle GPU time measuring
                    # merged-pair losses over the round's top scorers.
                    # One pair per idle tick, and only after a short
                    # quiescence (>= 2 ticks), so freshly-landed
                    # downloads always win the GPU over shadow work.
                    if idle_ticks >= 2:
                        await self._maybe_run_dedup_shadow(round_obj)
                    await asyncio.sleep(self.poll_interval_sec)
                    continue

                idle_ticks = 0
                uid, hotkey = target
                await self._evaluate_one(round_obj, uid=uid, hotkey=hotkey)
        finally:
            try:
                VALIDATOR_BG_WORKER_PAUSED.labels(worker="eval").set(0)
            except Exception:
                pass
            # Free GPU memory the worker held.
            try:
                del self._eval_base_model
                self._eval_base_model = None
                self._cached_batches = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def _next_target(self, round_obj) -> tuple[int, str] | None:
        for entry in round_obj.next_for_eval():
            return entry.uid, entry.hotkey
        return None

    async def _wait_clear(self) -> None:
        round_obj = self.round_ref.current
        logger.info(
            "bg-eval: deactivating — gates set, pausing evaluations",
            has_round=round_obj is not None,
            has_eval_base_model=self._eval_base_model is not None,
            merge_phase_active=self.merge_phase_active.is_set(),
            eval_window_active=self.eval_window_active.is_set(),
        )
        # Entering a gated stretch usually means the eval window closed
        # (or Merge started) — a natural point to summarize the round's
        # shadow pass. Idempotent via `_dedup_summary_emitted`.
        self._emit_dedup_summary(reason="gated")
        while not self.stop_event.is_set():
            round_obj = self.round_ref.current
            if (
                round_obj is not None
                and self._eval_base_model is not None
                and not self.merge_phase_active.is_set()
                and self.eval_window_active.is_set()
            ):
                logger.info("bg-eval: active — gates cleared, resuming evaluations")
                return
            await asyncio.sleep(0.5)

    async def _load_round_snapshot(self, round_obj) -> None:
        """Load the round's CPU snapshot into our GPU eval_base_model,
        materialize the round's eval batches once, and compute baseline.

        Holds `gpu_eval_lock` only for the duration of `load_state_dict`
        and the baseline forward pass. The dataloader is *materialized*
        — i.e., all batches are pulled into a CPU-side list — so every
        subsequent miner this round iterates from RAM instead of the HF
        streaming endpoint. Removes HF from the per-miner critical path
        and eliminates the lock-leak trigger seen in production
        (a hung HF read inside `for batch in dataloader:`).
        """
        # Lazy imports keep this module loadable without the heavy
        # datasets/pandas chain at module-import time (helps tests).
        from connito.shared.dataloader import get_dataloader, materialize_batches
        from connito.shared.evaluate import evaluate_model

        self._reset_dedup_state_if_new_round(round_obj)

        # Drop any previous round's cached batches before loading the new
        # round's. The CPU-side tensor list can run to several hundred MB
        # for a full 50-batch window — leaking it across rounds compounds
        # under restart-replay scenarios.
        self._cached_batches = None

        def _load() -> None:
            with self.gpu_eval_lock:
                self._eval_base_model.load_state_dict(round_obj.model_snapshot_cpu, strict=False)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

        await asyncio.to_thread(_load)

        # Build the dataloader and pull all batches into RAM up-front.
        # Both steps are wrapped in to_thread so the event loop stays
        # free; if HF is unreachable, the build/materialize raises and
        # we retry with backoff before falling back to an unscored
        # baseline. Transient HF blips (the trigger for the lock-leak
        # wedges) typically resolve in seconds; bounded retry keeps the
        # round in normal scoring mode without dragging it into
        # degraded mode on the first failed attempt.
        def _build_and_materialize() -> list:
            dl = get_dataloader(
                config=self.config,
                tokenizer=self.tokenizer,
                seed=round_obj.seed,
                rank=0,
                world_size=self.config.dataloader.world_size,
            )
            try:
                return materialize_batches(dl, EVAL_MAX_BATCHES)
            finally:
                del dl

        last_error: Exception | None = None
        for attempt, delay in enumerate(DATALOADER_BUILD_RETRY_DELAYS_SEC):
            if delay > 0:
                logger.info(
                    "bg-eval: backing off before retrying dataloader build",
                    attempt=attempt + 1,
                    of=len(DATALOADER_BUILD_RETRY_DELAYS_SEC),
                    delay_sec=delay,
                )
                await asyncio.sleep(delay)
            try:
                self._cached_batches = await asyncio.to_thread(_build_and_materialize)
                last_error = None
                break
            except Exception as e:
                last_error = e
                logger.warning(
                    "bg-eval: dataloader build/materialize failed",
                    attempt=attempt + 1,
                    of=len(DATALOADER_BUILD_RETRY_DELAYS_SEC),
                    error=str(e),
                )

        if last_error is not None:
            logger.warning(
                "bg-eval: dataloader build exhausted retries; using fallback baseline",
                attempts=len(DATALOADER_BUILD_RETRY_DELAYS_SEC),
                error=str(last_error),
            )
            self._cached_batches = None
            self._loaded_baseline_loss = 100.0
            self._loaded_round_id = round_obj.round_id
            return

        def _baseline() -> float:
            with self.gpu_eval_lock:
                metrics = evaluate_model(
                    0, self._eval_base_model, self._cached_batches,
                    self.device, EVAL_MAX_BATCHES, None,
                )
            return float(metrics.get("val_loss", 100))

        try:
            self._loaded_baseline_loss = await asyncio.to_thread(_baseline)
        except Exception as e:
            logger.warning("bg-eval: baseline failed; using fallback", error=str(e))
            self._loaded_baseline_loss = 100.0

        self._loaded_round_id = round_obj.round_id
        logger.info(
            "bg-eval: round snapshot loaded",
            round_id=round_obj.round_id,
            baseline_loss=round(self._loaded_baseline_loss or 0.0, 4),
            cached_batches=(
                len(self._cached_batches) if self._cached_batches is not None else 0
            ),
        )

    async def _evaluate_one(self, round_obj, *, uid: int, hotkey: str) -> None:
        if not round_obj.claim_for_eval(uid):
            return

        # Last-mile check: if Merge fired between the gate poll and the
        # claim, abort cleanly instead of running a fresh GPU eval
        # alongside `sync_grad_across_validators` / global optimizer
        # (the contention pattern that produced the wedge in
        # validator_A6000_v0.1.38.log at 23:32:43 → 23:37:43). Done
        # *before* `pop_downloaded` so the path stays on the pool and
        # the next iteration after the merge clears can pick it up.
        if self.merge_phase_active.is_set():
            logger.info(
                "bg-eval: aborting claim — merge_phase_active set after gate",
                uid=uid, hotkey=hotkey[:6], round_id=round_obj.round_id,
            )
            round_obj.release_claim(uid)
            return

        path = round_obj.pop_downloaded(uid)
        if path is None:
            round_obj.release_claim(uid)
            return

        timeout = float(self.config.evaluation.per_miner_eval_timeout_sec)
        baseline = self._loaded_baseline_loss if self._loaded_baseline_loss is not None else 100.0

        # Verify the on-disk submission against the chain commit (signed
        # hash, hash, expert-group ownership, NaN/Inf scan) BEFORE the
        # GPU eval. Off-spec submissions are dropped — same outcome as
        # the foreground path.
        from connito.shared.telemetry import inc_error
        from connito.validator.evaluator import validate_miner_submission

        fail_reason = await asyncio.to_thread(
            validate_miner_submission,
            round_obj=round_obj,
            uid=uid,
            model_path=path,
            expert_group_assignment=self.expert_group_assignment,
        )
        if fail_reason is not None:
            logger.warning(
                "bg-eval: submission failed validation — marking validation-failed",
                uid=uid, hotkey=hotkey[:6],
                round_id=round_obj.round_id,
                reason=fail_reason,
            )
            inc_error(component="bg_eval", kind="validation")
            # `mark_validation_failed` puts this UID into both
            # `failed_uids` (claim-blocking) and `validation_failed_uids`
            # so finalize writes score=0 for it. Operational failures
            # below use plain `mark_failed` and leave the EMA alone.
            round_obj.mark_validation_failed(uid)
            round_obj.publish_progress()
            self._prune_non_top(round_obj)
            return

        logger.info(
            "bg-eval: evaluating",
            round_id=round_obj.round_id,
            uid=uid, hotkey=hotkey[:6],
            model_path=str(path),
            timeout_sec=timeout,
        )

        # Run the entire eval inside one threadpool task and acquire
        # `gpu_eval_lock` INSIDE that thread. This couples lock release
        # to actual GPU completion (not awaiter cancellation): when
        # `wait_for` cancels the awaiter on timeout, the threadpool task
        # keeps running (`asyncio.to_thread` tasks aren't cancellable),
        # so the lock stays held until GPU work finishes. The next
        # eval's `to_thread(_run_eval)` call blocks on `gpu_eval_lock`
        # inside its own thread until the timed-out thread drains —
        # preventing two concurrent miner-model allocations on a single
        # GPU (the OOM cascade observed when cancellation released the
        # lock mid-thread).
        #
        # To cover the case where the eval genuinely stalls on GPU, we
        # forward an in-thread deadline to `evaluate_model`, which
        # raises `EvalDeadlineExceeded` between batches and unwinds the
        # `with lock:` block cleanly. The outer `wait_for` is given a
        # small grace window past that deadline so the in-thread check
        # fires first; if the awaiter still times out, the thread is
        # wedged and the recycler in `_loop` will catch it.
        deadline = time.monotonic() + timeout
        outer_timeout = timeout + EVAL_DEADLINE_GRACE_SEC

        def _run_eval():
            with self.gpu_eval_lock:
                return evaluate_one_miner_sync(
                    config=self.config,
                    model_path=path,
                    uid=uid,
                    hotkey=hotkey,
                    base_model=self._eval_base_model,
                    tokenizer=self.tokenizer,
                    combined_seed=round_obj.seed,
                    device=self.device,
                    baseline_loss=baseline,
                    step=round_obj.round_id,
                    round_id=round_obj.round_id,
                    deadline_monotonic=deadline,
                    cached_batches=self._cached_batches,
                )

        try:
            evaluated = await asyncio.wait_for(
                asyncio.to_thread(_run_eval), timeout=outer_timeout,
            )
        except asyncio.TimeoutError:
            # In-thread deadline didn't fire in time — thread is wedged
            # past the grace period. Lock is still held by the orphan;
            # the recycler will surface this when the streak crosses
            # the threshold.
            logger.warning(
                "bg-eval: outer wait_for fired — in-thread deadline did not "
                "unwind in grace period; gpu_eval_lock likely orphaned",
                uid=uid, hotkey=hotkey[:6],
                timeout_sec=outer_timeout,
                round_id=round_obj.round_id,
            )
            try:
                note_round_series(round_obj.round_id)
                VALIDATOR_BG_EVAL_LOCK_LEAK_TOTAL.labels(
                    round_id=str(round_obj.round_id)
                ).inc()
            except Exception:
                pass
            evaluated = None
        except Exception as e:
            logger.exception("bg-eval: failure", uid=uid, error=str(e))
            evaluated = None

        if evaluated is None:
            round_obj.mark_failed(uid)
            round_obj.publish_progress()
            self._prune_non_top(round_obj)
            return

        round_obj.mark_scored(uid, evaluated.score, val_loss=evaluated.val_loss)
        # Dedup shadow bookkeeping: remember where this miner's file
        # lives (pop_downloaded above removed the only uid→path map) and
        # its exact val_loss, so shadow pairs don't have to re-derive
        # either. Read-only breadcrumbs — never fed back into scoring.
        if round_obj.round_id == self._dedup_round_id:
            self._scored_paths[uid] = Path(path)
            ev_loss = evaluated.val_loss
            if isinstance(ev_loss, float) and math.isfinite(ev_loss):
                self._scored_val_loss[uid] = ev_loss
        logger.info(
            "bg-eval: success",
            round_id=round_obj.round_id,
            uid=uid, hotkey=hotkey[:6],
            score=round(evaluated.score, 6),
        )
        round_obj.publish_progress()
        self._prune_non_top(round_obj)

    def _stuck_lock_check_and_maybe_recycle(self) -> bool:
        """At the top of each iteration, probe `gpu_eval_lock`.

        If the non-blocking acquire succeeds, no one holds it; reset the
        streak counter and return False (proceed normally).

        If it fails, an orphan thread is holding the lock — most likely
        a previous eval whose `wait_for` cancelled the awaiter while the
        in-thread deadline hadn't yet fired. Increment the streak; once
        it crosses `stuck_lock_recycle_threshold` consecutive iterations
        we drop `_eval_base_model` and bump the recycle counter. The
        main loop re-seeds us via `set_eval_base_model` after the next
        foreground eval finishes, so we recover without a process
        restart.

        Returns True iff we recycled — the caller should re-enter the
        outer gate loop instead of trying to claim work this tick.
        """
        if self.gpu_eval_lock.acquire(blocking=False):
            self.gpu_eval_lock.release()
            if self._stuck_lock_streak != 0:
                self._stuck_lock_streak = 0
                try:
                    VALIDATOR_BG_EVAL_STUCK_LOCK_ITERATIONS.set(0)
                except Exception:
                    pass
            return False

        self._stuck_lock_streak += 1
        try:
            VALIDATOR_BG_EVAL_STUCK_LOCK_ITERATIONS.set(self._stuck_lock_streak)
        except Exception:
            pass
        logger.warning(
            "bg-eval: gpu_eval_lock appears held at iteration boundary; "
            "this is the lock-yielding invariant — investigate.",
            stuck_streak=self._stuck_lock_streak,
            recycle_threshold=self.stuck_lock_recycle_threshold,
        )

        if self._stuck_lock_streak < self.stuck_lock_recycle_threshold:
            return False

        # Threshold crossed. Drop our model reference and re-park; the
        # main loop owns re-seeding via `set_eval_base_model`. We do NOT
        # touch the lock itself — the orphan thread still holds it and
        # may eventually release; reaching in to free a `threading.Lock`
        # we don't own is undefined behavior.
        logger.error(
            "bg-eval: lock leak — recycling worker state",
            stuck_streak=self._stuck_lock_streak,
            recycle_threshold=self.stuck_lock_recycle_threshold,
        )
        with self._eval_base_model_lock:
            self._eval_base_model = None
        self._loaded_round_id = None
        self._loaded_baseline_loss = None
        self._cached_batches = None
        self._stuck_lock_streak = 0
        try:
            VALIDATOR_BG_EVAL_STUCK_LOCK_ITERATIONS.set(0)
            VALIDATOR_BG_EVAL_RECYCLE_TOTAL.inc()
        except Exception:
            pass
        return True

    def _prune_non_top(self, round_obj) -> None:
        """Drop on-disk submission files for miners that have been
        processed this round but are not in the top-`top_k_miners_to_reward`
        by *this round's* score (read from `round.scores`, not the global
        aggregator). Files for miners that have not yet been evaluated
        are explicitly retained — see `cleanup_non_top_submissions`.
        Keeps the merge-time top-1 set (top_k_miners_to_merge=1 ⊆
        top_k_miners_to_reward=3) and the archive-time top set, while
        reclaiming disk for everyone else.
        """
        try:
            deleted = cleanup_non_top_submissions(
                round_obj=round_obj,
                submission_dir=Path(self.config.ckpt.miner_submission_path),
                # Widened to dedup_top_k while the dedup filter is active
                # so the shadow pass can still read the files it pairs.
                top_k=retention_top_k(self.config),
            )
        except Exception as e:
            logger.warning("bg-eval: post-eval submission cleanup failed", error=str(e))
            return
        if deleted:
            logger.info(
                "bg-eval: pruned non-top miner submissions",
                round_id=round_obj.round_id,
                deleted=len(deleted),
                files=deleted,
            )

    # ---------------- Dedup shadow pass ----------------
    def _reset_dedup_state_if_new_round(self, round_obj) -> None:
        """Reset dedup bookkeeping when a genuinely NEW round arrives.

        Called from `_load_round_snapshot`, which also re-runs after a
        stuck-lock recycle for the SAME round — hence the explicit
        round-id key instead of piggybacking on `_loaded_round_id`.
        Emits the previous round's summary first (idempotent).
        """
        if self._dedup_round_id == round_obj.round_id:
            return
        self._emit_dedup_summary(reason="round_transition")
        self._dedup_round_id = round_obj.round_id
        self._scored_paths = {}
        self._scored_val_loss = {}
        self._dedup_pairs_done = set()
        self._dedup_budget_used = 0
        self._dedup_pairs_skipped = 0
        self._dedup_summary_emitted = False

    def _emit_dedup_summary(self, *, reason: str) -> None:
        """One `dedup-shadow: round summary` line per round (best effort)."""
        if self._dedup_summary_emitted or self._dedup_round_id is None:
            return
        if self._dedup_budget_used == 0 and self._dedup_pairs_skipped == 0:
            return  # nothing ran (mode off, or <2 positive miners) — stay quiet
        self._dedup_summary_emitted = True
        logger.info(
            "dedup-shadow: round summary",
            round_id=self._dedup_round_id,
            pairs_evaluated=len(self._dedup_pairs_done) - self._dedup_pairs_skipped,
            pairs_skipped=self._dedup_pairs_skipped,
            budget_used=self._dedup_budget_used,
            budget_max=int(getattr(self.config.evaluation, "dedup_max_pairs", 0)),
            reason=reason,
        )

    def _resolve_submission_path(self, round_obj, uid: int) -> Path | None:
        """uid → on-disk submission file, best effort.

        Bg-scored miners were captured in `_scored_paths`; foreground-
        scored miners never transit this worker, so fall back to the
        filename convention (hotkey + block inside the round's submission
        window). Either way the file may have been pruned since —
        callers treat None / missing file as "skip pair".
        """
        path = self._scored_paths.get(uid)
        if path is not None and path.exists():
            return path
        hotkey = round_obj.uid_to_hotkey.get(uid)
        if hotkey is None:
            return None
        return find_submission_path(
            Path(self.config.ckpt.miner_submission_path),
            hotkey,
            getattr(round_obj, "submission_block_range", None),
        )

    async def _maybe_run_dedup_shadow(self, round_obj) -> None:
        """Evaluate at most ONE merged pair of top submissions, then yield.

        Shadow mode: measurement + logging only. This method never calls
        `mark_scored`/`mark_failed`/`claim_for_eval`/`pop_downloaded` and
        never writes to the aggregator or journal — it cannot affect any
        miner's score or the round's lifecycle. One pair per idle tick
        keeps real miner evals strictly ahead of shadow work.
        """
        cfg = self.config.evaluation
        mode = getattr(cfg, "dedup_filter_mode", "off")
        if mode not in ("shadow", "enforce"):
            return
        if self._dedup_round_id != round_obj.round_id:
            return  # snapshot load (which resets dedup state) hasn't run yet
        max_pairs = int(getattr(cfg, "dedup_max_pairs", 0))
        if self._dedup_budget_used >= max_pairs:
            return
        # Same gates as a real eval, re-checked immediately before work.
        if (
            self.round_ref.current is not round_obj
            or self._loaded_round_id != round_obj.round_id
            or self._eval_base_model is None
            or self._cached_batches is None
            or self.merge_phase_active.is_set()
            or not self.eval_window_active.is_set()
        ):
            return

        pairs = select_pairs(
            round_obj.scores_snapshot(),
            top_k=int(getattr(cfg, "dedup_top_k", 0)),
            max_pairs=max_pairs - self._dedup_budget_used,
            exclude=self._dedup_pairs_done,
        )
        if not pairs:
            return
        uid_a, uid_b = pairs[0]
        # Mark the pair consumed up-front: a pair that fails (missing
        # file, eval error) is not retried — its files are only going to
        # get *less* available as pruning proceeds.
        self._dedup_pairs_done.add(frozenset((uid_a, uid_b)))
        self._dedup_budget_used += 1

        path_a = self._resolve_submission_path(round_obj, uid_a)
        path_b = self._resolve_submission_path(round_obj, uid_b)
        if path_a is None or path_b is None:
            self._dedup_pairs_skipped += 1
            logger.info(
                "dedup-shadow: pair skipped — submission file unavailable",
                round_id=round_obj.round_id,
                uid_a=uid_a, uid_b=uid_b,
                has_a=path_a is not None, has_b=path_b is not None,
            )
            return

        baseline = (
            self._loaded_baseline_loss
            if self._loaded_baseline_loss is not None
            else 100.0
        )
        scores = round_obj.scores_snapshot()

        def _side_loss(uid: int) -> tuple[float, str]:
            exact = self._scored_val_loss.get(uid)
            if exact is not None:
                return exact, "bg_exact"
            # Foreground-scored miners: invert score = delta**1.2. Their
            # delta was computed against the FOREGROUND baseline (same
            # snapshot + seed, different loader instance), so a small
            # offset vs our measurement of loss_avg is possible — tagged
            # so the what-if analysis can segment.
            return recover_val_loss(scores.get(uid, 0.0), baseline), "recovered"

        timeout = float(cfg.per_miner_eval_timeout_sec)
        deadline = time.monotonic() + timeout
        outer_timeout = timeout + EVAL_DEADLINE_GRACE_SEC

        def _run_pair() -> tuple[float, float, int, int]:
            # CPU work (load, cosine, average) outside the GPU lock —
            # the lock-yielding invariant wants the narrowest scope.
            from connito.shared.evaluate import evaluate_model

            sd_a = load_state_dict_from_path(path_a)
            sd_b = load_state_dict_from_path(path_b)
            cosine, n_keys = delta_cosine(sd_a, sd_b, round_obj.model_snapshot_cpu)
            merged_sd, asymmetric = average_state_dicts(sd_a, sd_b)
            del sd_a, sd_b
            with self.gpu_eval_lock:
                # Deepcopy per pair — NEVER mutate `_eval_base_model` in
                # place: real miner evals interleave between pairs and
                # deepcopy it assuming pristine round-snapshot weights.
                merged_model = copy.deepcopy(self._eval_base_model)
                try:
                    merged_model.load_state_dict(merged_sd, strict=False)
                    metrics = evaluate_model(
                        0, merged_model, self._cached_batches,
                        self.device, EVAL_MAX_BATCHES, None,
                        deadline_monotonic=deadline,
                    )
                finally:
                    del merged_model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            return float(metrics.get("val_loss", float("inf"))), cosine, n_keys, asymmetric

        try:
            loss_avg, cosine, n_keys, asymmetric = await asyncio.wait_for(
                asyncio.to_thread(_run_pair), timeout=outer_timeout,
            )
        except Exception as e:
            self._dedup_pairs_skipped += 1
            logger.warning(
                "dedup-shadow: pair eval failed",
                round_id=round_obj.round_id,
                uid_a=uid_a, uid_b=uid_b, error=str(e),
            )
            return

        loss_a, source_a = _side_loss(uid_a)
        loss_b, source_b = _side_loss(uid_b)
        report = shadow_report(
            loss_a=loss_a, loss_b=loss_b, loss_avg=loss_avg,
            baseline=baseline, cosine=cosine,
        )
        # Enforcement decides on the UNROUNDED penalty — `report`'s copy is
        # rounded to 6 dp, where a tiny negative becomes `-0.0` and would
        # satisfy `>= 0`.
        threshold = float(getattr(cfg, "dedup_threshold", DEFAULT_DEDUP_THRESHOLD))
        redundant = is_redundant(
            compute_merge_penalty(loss_a, loss_b, loss_avg), threshold,
        )
        enforced = False
        if mode == "enforce" and redundant:
            # Zero BOTH sides at finalize. `merge_penalty` is a property of
            # the pair, so it says one of the two added nothing the other
            # lacked — never which one.
            round_obj.mark_dedup_flagged(uid_a, uid_b)
            enforced = True
        logger.info(
            "dedup-shadow: pair result",
            round_id=round_obj.round_id,
            uid_a=uid_a, uid_b=uid_b,
            loss_source_a=source_a, loss_source_b=source_b,
            n_keys=n_keys, asymmetric_keys=asymmetric,
            mode=mode, threshold=threshold,
            redundant=redundant, enforced=enforced,
            **report,
        )
        try:
            VALIDATOR_DEDUP_PAIRS_EVALUATED_TOTAL.inc()
            for threshold, flagged in report["would_flag_not_better"].items():
                if flagged:
                    VALIDATOR_DEDUP_WOULD_FLAG_TOTAL.labels(threshold=threshold).inc()
        except Exception:
            pass
