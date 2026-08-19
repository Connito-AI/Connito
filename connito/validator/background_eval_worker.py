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
import threading
import time
from pathlib import Path

import torch
import torch.nn as nn

from connito.shared.app_logging import structlog
from connito.shared.telemetry import (
    VALIDATOR_BG_EVAL_LOCK_LEAK_TOTAL,
    VALIDATOR_BG_EVAL_RECYCLE_TOTAL,
    VALIDATOR_BG_EVAL_STUCK_LOCK_ITERATIONS,
    VALIDATOR_BG_WORKER_PAUSED,
    note_round_series,
)
from connito.validator.evaluator import (
    EVAL_MAX_BATCHES,
    cleanup_non_top_submissions,
    evaluate_one_miner_sync,
)
from connito.validator.round import RoundRef

# Consecutive iterations that may see `gpu_eval_lock` still held before we
# drop `_eval_base_model` and re-park for a re-seed. Long enough that a slow
# eval isn't penalized, short enough that a leaked lock can't wedge the worker
# for the life of the process.
DEFAULT_STUCK_LOCK_RECYCLE_THRESHOLD = 3

# Slack on the outer `wait_for` deadline. The in-thread deadline only fires
# between batches, so give the in-flight batch time to drain and release the
# lock; if the awaiter still times out, the thread is stuck and the recycler
# catches it.
EVAL_DEADLINE_GRACE_SEC = 30.0

# Backoff between dataloader build attempts. Absorbs transient HF blips
# without eating into the eval window; on exhaustion we fall back to the
# degraded baseline path so the round still finalizes.
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
        # Handed in by the main loop after foreground eval, rather than
        # re-fetched from chain at startup.
        self._eval_base_model: nn.Module | None = None
        self._eval_base_model_lock = threading.Lock()
        self._loaded_round_id: int | None = None
        self._loaded_baseline_loss: float | None = None
        # Eval batches materialized once per round. Every miner scores on the
        # same combined seed, so per-miner re-streaming is wasted work and a
        # source of HF stalls inside the GPU lock.
        self._cached_batches: list | None = None
        # Consecutive iterations that saw `gpu_eval_lock` held at the top of
        # the loop; crossing the threshold re-parks the worker for a re-seed.
        self._stuck_lock_streak: int = 0

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
        # Gate-loop until the main loop seeds us, so startup doesn't duplicate
        # the chain fetch and MoE construction on this thread.
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
                # The lock must never be held across an iteration boundary.
                # If an orphan holds it past the threshold, re-park for a
                # re-seed — otherwise one timed-out GPU thread wedges the
                # worker for the life of the process.
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

        # Build and materialize on a thread so the event loop stays free. A
        # bounded retry rides out transient HF failures instead of dropping
        # straight into degraded (unscored-baseline) mode.
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

        # Merge may have started between the gate poll and the claim; running
        # a GPU eval alongside allreduce/optimizer is the contention pattern
        # that wedges this worker. Checked before `pop_downloaded` so the path
        # stays in the pool for the next iteration.
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

        # `gpu_eval_lock` is acquired INSIDE the thread so its release is
        # coupled to GPU completion rather than awaiter cancellation:
        # `asyncio.to_thread` tasks keep running after `wait_for` gives up, and
        # releasing the lock there would let a second miner model allocate on
        # the same GPU and cascade into OOM.
        #
        # For a genuine GPU stall, the in-thread deadline forwarded to
        # `evaluate_model` raises between batches and unwinds `with lock:`
        # cleanly. The outer `wait_for` gets a grace window so that fires
        # first; if it still times out, the recycler in `_loop` catches it.
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
            # Wedged past the grace period; the orphan still holds the lock
            # and the recycler will surface it.
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

        # Drop the model reference and re-park; the main loop re-seeds us.
        # Never touch the lock — freeing a `threading.Lock` this thread does
        # not own is undefined behavior, and the orphan may still release it.
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
                top_k=int(self.config.evaluation.top_k_miners_to_reward),
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

