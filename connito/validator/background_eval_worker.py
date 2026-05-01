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
from pathlib import Path

import torch
import torch.nn as nn

from connito.shared.app_logging import structlog
from connito.shared.telemetry import (
    VALIDATOR_BG_WORKER_PAUSED,
    VALIDATOR_ROUND_MINERS_FAILED,
    VALIDATOR_ROUND_MINERS_PENDING,
    VALIDATOR_ROUND_MINERS_SCORED,
)
from connito.validator.evaluator import (
    EVAL_MAX_BATCHES,
    cleanup_non_top_submissions,
    evaluate_one_miner,
)
from connito.validator.round import RoundRef

logger = structlog.get_logger(__name__)


class BackgroundEvalWorker(threading.Thread):
    def __init__(
        self,
        *,
        config,
        round_ref: RoundRef,
        device: torch.device,
        tokenizer,
        score_aggregator,
        score_path,
        merge_phase_active: threading.Event,
        eval_window_active: threading.Event,
        gpu_eval_lock: threading.Lock,
        expert_group_assignment,
        stop_event: threading.Event | None = None,
        poll_interval_sec: float = 2.0,
    ) -> None:
        super().__init__(daemon=True, name="connito-bg-eval")
        self.config = config
        self.round_ref = round_ref
        self.device = device
        self.tokenizer = tokenizer
        self.score_aggregator = score_aggregator
        self.score_path = score_path
        self.merge_phase_active = merge_phase_active
        self.eval_window_active = eval_window_active
        self.gpu_eval_lock = gpu_eval_lock
        self.expert_group_assignment = expert_group_assignment
        self.stop_event = stop_event or threading.Event()
        self.poll_interval_sec = poll_interval_sec
        # Model is handed in by the main loop (see set_eval_base_model)
        # right after foreground eval completes, instead of being
        # re-fetched from chain at startup.
        self._eval_base_model: nn.Module | None = None
        self._eval_base_model_lock = threading.Lock()
        self._loaded_round_id: int | None = None
        self._loaded_baseline_loss: float | None = None

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
                # Invariant: do not enter waits while owning the lock.
                self._assert_lock_unheld_by_us()

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
        """Load the round's CPU snapshot into our GPU eval_base_model.

        Holds gpu_eval_lock only for the duration of load_state_dict.
        """
        # Lazy imports keep this module loadable without the heavy
        # datasets/pandas chain at module-import time (helps tests).
        from connito.shared.dataloader import get_dataloader
        from connito.shared.evaluate import evaluate_model

        def _load() -> None:
            with self.gpu_eval_lock:
                self._eval_base_model.load_state_dict(round_obj.model_snapshot_cpu, strict=False)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

        await asyncio.to_thread(_load)
        # Recompute the baseline loss for this round once.
        try:
            dataloader = await asyncio.to_thread(
                get_dataloader,
                config=self.config,
                tokenizer=self.tokenizer,
                seed=round_obj.seed,
                rank=0,
                world_size=self.config.dataloader.world_size,
            )
        except Exception as e:
            logger.warning("bg-eval: dataloader build failed; using fallback baseline", error=str(e))
            self._loaded_baseline_loss = 100.0
            self._loaded_round_id = round_obj.round_id
            return

        def _baseline() -> float:
            with self.gpu_eval_lock:
                metrics = evaluate_model(0, self._eval_base_model, dataloader, self.device, EVAL_MAX_BATCHES, None)
            return float(metrics.get("val_loss", 100))

        try:
            self._loaded_baseline_loss = await asyncio.to_thread(_baseline)
        except Exception as e:
            logger.warning("bg-eval: baseline failed; using fallback", error=str(e))
            self._loaded_baseline_loss = 100.0
        finally:
            del dataloader

        self._loaded_round_id = round_obj.round_id
        logger.info(
            "bg-eval: round snapshot loaded",
            round_id=round_obj.round_id,
            baseline_loss=round(self._loaded_baseline_loss or 0.0, 4),
        )

    async def _evaluate_one(self, round_obj, *, uid: int, hotkey: str) -> None:
        if not round_obj.claim_for_eval(uid):
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
                "bg-eval: submission failed validation — marking failed",
                uid=uid, hotkey=hotkey[:6],
                round_id=round_obj.round_id,
                reason=fail_reason,
            )
            inc_error(component="bg_eval", kind="validation")
            self.score_aggregator.add_score(uid=int(uid), hotkey=hotkey, score=0, round_id=round_obj.round_id)
            round_obj.mark_failed(uid)
            self._record_metrics(round_obj, scored_inc=False)
            self._prune_non_top(round_obj)
            return

        logger.info(
            "bg-eval: evaluating",
            round_id=round_obj.round_id,
            uid=uid, hotkey=hotkey[:6],
            model_path=str(path),
            timeout_sec=timeout,
        )

        # Wrap the GPU-touching work so the lock is held only for the eval
        # itself. evaluate_one_miner is async but the heavy GPU ops happen
        # inside asyncio.to_thread; we acquire the lock around the whole
        # call to make the lock-yielding contract explicit.
        async def _eval_with_lock() -> "MinerEvalJob | None":
            # Acquire briefly; release right after the call returns.
            acquired = await asyncio.to_thread(self.gpu_eval_lock.acquire)
            if not acquired:
                return None
            try:
                return await evaluate_one_miner(
                    config=self.config,
                    model_path=path,
                    uid=uid,
                    hotkey=hotkey,
                    base_model=self._eval_base_model,
                    tokenizer=self.tokenizer,
                    combined_seed=round_obj.seed,
                    device=self.device,
                    score_aggregator=self.score_aggregator,
                    baseline_loss=baseline,
                    step=round_obj.round_id,
                    round_id=round_obj.round_id,
                    round_obj=round_obj,
                    score_path=self.score_path,
                )
            finally:
                self.gpu_eval_lock.release()

        try:
            evaluated = await asyncio.wait_for(_eval_with_lock(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("bg-eval: timeout", uid=uid, hotkey=hotkey[:6], timeout_sec=timeout)
            evaluated = None
        except Exception as e:
            logger.exception("bg-eval: failure", uid=uid, error=str(e))
            evaluated = None

        if evaluated is None:
            round_obj.mark_failed(uid)
            self._record_metrics(round_obj, scored_inc=False)
            self._prune_non_top(round_obj)
            return

        round_obj.mark_scored(uid, evaluated.score)
        logger.info(
            "bg-eval: success",
            round_id=round_obj.round_id,
            uid=uid, hotkey=hotkey[:6],
            score=round(evaluated.score, 6),
        )
        # Per-miner persistence happens inside evaluate_one_miner via score_path.
        self._record_metrics(round_obj, scored_inc=True)
        self._prune_non_top(round_obj)

    def _assert_lock_unheld_by_us(self) -> None:
        # Best-effort invariant check. `Lock.locked()` is true if anyone
        # holds it; we cannot ask "do *we* hold it?" via threading.Lock,
        # so we guard with try-acquire-release: if non-blocking acquire
        # succeeds we know we did not hold it (and we hand the lock back
        # immediately).
        if self.gpu_eval_lock.acquire(blocking=False):
            self.gpu_eval_lock.release()
            return
        # Lock is held by someone — that someone might be us if a code
        # path forgot to release. Log loudly so tests can catch it.
        logger.warning(
            "bg-eval: gpu_eval_lock appears held at iteration boundary; "
            "this is the lock-yielding invariant — investigate."
        )

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

    @staticmethod
    def _record_metrics(round_obj, *, scored_inc: bool) -> None:
        try:
            stats = round_obj.stats()
            VALIDATOR_ROUND_MINERS_SCORED.labels(round_id=str(round_obj.round_id)).set(stats["scored"])
            VALIDATOR_ROUND_MINERS_FAILED.labels(round_id=str(round_obj.round_id)).set(stats["failed"])
            VALIDATOR_ROUND_MINERS_PENDING.labels(round_id=str(round_obj.round_id)).set(stats["pending"])
        except Exception:
            pass
