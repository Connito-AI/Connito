"""Step (1) of the round lifecycle: download miner HF checkpoints in the
background, in incentive order, into the round's `downloaded_pool`.

This worker is network-only — disk writes + HF reads — so it does not
contend with foreground evaluation (which only reads from disk and runs
on GPU). It is paused while:
  - the main loop is in the Merge phase (`merge_phase_active` set), so
    HF upload + allreduce can hold the available bandwidth, or
  - the download window has closed (`download_window_closed` set), which
    the main loop sets when it begins waiting for MinerCommit1 of the
    next round and clears at the next freeze.

It does not gate on the foreground pass: foreground reads from
`miner_submission_path`, which this worker is responsible for filling,
so the two MUST run concurrently or foreground would never discover any
miner to evaluate.
"""

from __future__ import annotations

import asyncio
import shutil
import threading
from pathlib import Path

import bittensor

from connito.shared.app_logging import structlog
from connito.shared.helper import parse_dynamic_filename
from connito.shared.hf_distribute import download_checkpoint_from_hf
from connito.shared.telemetry import (
    CHECKPOINT_DOWNLOAD_BYTES,
    VALIDATOR_BG_WORKER_PAUSED,
    VALIDATOR_ROUND_MINERS_FAILED,
    VALIDATOR_ROUND_MINERS_PENDING,
    inc_eval_failure,
)
from connito.validator.evaluator import resolve_miner_hf_target
from connito.validator.round import RoundRef

logger = structlog.get_logger(__name__)


class BackgroundDownloadWorker(threading.Thread):
    def __init__(
        self,
        *,
        config,
        round_ref: RoundRef,
        merge_phase_active: threading.Event,
        download_window_closed: threading.Event | None = None,
        stop_event: threading.Event | None = None,
        poll_interval_sec: float = 6.0,
    ) -> None:
        super().__init__(daemon=True, name="connito-bg-download")
        self.config = config
        self.round_ref = round_ref
        self.merge_phase_active = merge_phase_active
        self.download_window_closed = download_window_closed or threading.Event()
        self.stop_event = stop_event or threading.Event()
        self.poll_interval_sec = poll_interval_sec
        self._subtensor: bittensor.Subtensor | None = None

    # ---------------- Public lifecycle ----------------
    def stop(self) -> None:
        self.stop_event.set()

    # ---------------- Thread body ----------------
    def run(self) -> None:
        try:
            asyncio.run(self._loop())
        except Exception:
            logger.exception("BackgroundDownloadWorker crashed")

    # ---------------- Internal ----------------
    async def _loop(self) -> None:
        try:
            self._subtensor = await asyncio.to_thread(
                bittensor.Subtensor, network=self.config.chain.network,
            )
        except Exception as e:
            logger.warning("BackgroundDownloadWorker: failed to open subtensor; exiting", error=str(e))
            return

        logger.info(
            "BackgroundDownloadWorker: started",
            network=self.config.chain.network,
            poll_interval_sec=self.poll_interval_sec,
        )

        # Rate-limit idle-state logs to roughly once every IDLE_LOG_EVERY ticks.
        IDLE_LOG_EVERY = 5
        idle_ticks = 0
        try:
            while not self.stop_event.is_set():
                round_obj = self.round_ref.current
                if round_obj is None:
                    if idle_ticks % IDLE_LOG_EVERY == 0:
                        logger.debug("bg-download: idle — no current round")
                    idle_ticks += 1
                    await asyncio.sleep(self.poll_interval_sec)
                    continue

                # Snapshot pause state for telemetry.
                paused = (
                    self.merge_phase_active.is_set()
                    or self.download_window_closed.is_set()
                )
                try:
                    VALIDATOR_BG_WORKER_PAUSED.labels(worker="download").set(1 if paused else 0)
                except Exception:
                    pass
                if paused:
                    if idle_ticks % IDLE_LOG_EVERY == 0:
                        logger.info(
                            "bg-download: paused",
                            merge_phase_active=self.merge_phase_active.is_set(),
                            download_window_closed=self.download_window_closed.is_set(),
                        )
                    idle_ticks += 1
                    await self._wait_clear()
                    continue

                # Pick the next UID to download.
                target = self._next_target(round_obj)
                if target is None:
                    # Log only on the transition into idle; stay quiet until
                    # new work arrives. idle_ticks resets to 0 on the next
                    # successful download below, re-arming this log for the
                    # next gap. Without this, an empty queue spammed
                    # ~once-per-30s for the whole rest of the cycle.
                    if idle_ticks == 0:
                        try:
                            stats = round_obj.stats()
                        except Exception:
                            stats = None
                        logger.info(
                            "bg-download: no pending targets — going idle",
                            round_id=getattr(round_obj, "round_id", None),
                            round_stats=stats,
                        )
                    idle_ticks += 1
                    await asyncio.sleep(self.poll_interval_sec)
                    continue

                idle_ticks = 0
                uid, hotkey = target
                await self._download_one(round_obj, uid=uid, hotkey=hotkey)
        finally:
            try:
                VALIDATOR_BG_WORKER_PAUSED.labels(worker="download").set(0)
            except Exception:
                pass

    def _next_target(self, round_obj) -> tuple[int, str] | None:
        for entry in round_obj.next_for_download():
            return entry.uid, entry.hotkey
        return None

    async def _wait_clear(self) -> None:
        # Coarse polling: wake every 0.5s so stop and gate transitions
        # propagate without spinning.
        logger.info(
            "bg-download: deactivated — gates blocked, pausing downloads",
            merge_phase_active=self.merge_phase_active.is_set(),
            download_window_closed=self.download_window_closed.is_set(),
        )
        while not self.stop_event.is_set():
            if (
                not self.merge_phase_active.is_set()
                and not self.download_window_closed.is_set()
            ):
                logger.info("bg-download: active — gates cleared, resuming downloads")
                return
            await asyncio.sleep(0.5)

    async def _download_one(self, round_obj, *, uid: int, hotkey: str) -> None:
        timeout = float(self.config.evaluation.per_miner_download_timeout_sec)
        # We walk foreground_uids first then background_uids; the single
        # download thread plus next_for_download's claimed/scored/failed
        # filters keep us from racing with foreground eval. publish_download
        # is a no-op if the UID has already been scored.
        subtensor = self._subtensor
        if subtensor is None:
            return
        try:
            target = await asyncio.to_thread(
                resolve_miner_hf_target,
                config=self.config,
                subtensor=subtensor,
                hotkey=hotkey,
            )
            if target is None:
                logger.debug("bg-download: no HF target for miner; skipping", uid=uid, hotkey=hotkey[:6])
                round_obj.mark_failed(uid)
                self._update_pending_metric(round_obj)
                return

            repo_id, revision = target
            expert_group_id = self.config.task.exp.group_id
            filename = f"model_expgroup_{expert_group_id}.pt"
            submission_dir = Path(self.config.ckpt.miner_submission_path)
            submission_dir.mkdir(parents=True, exist_ok=True)

            # Skip if a submission for this hotkey already exists locally
            # (HTTP path or earlier hydration may have written it).
            existing = self._existing_submission(submission_dir, hotkey)
            if existing is not None:
                logger.info(
                    "bg-download: submission already on disk; reusing",
                    uid=uid, hotkey=hotkey[:6], path=str(existing),
                )
                round_obj.publish_download(uid, existing)
                self._update_pending_metric(round_obj)
                return

            tmp_dir = submission_dir / f".tmp_bg_dl_{hotkey}"
            block = self._subtensor.block if self._subtensor is not None else 0
            dest_name = f"hotkey_{hotkey}_block_{block}.pt"
            dest = submission_dir / dest_name

            logger.info(
                "bg-download: fetching",
                uid=uid, hotkey=hotkey[:6],
                repo_id=repo_id,
                revision=(revision[:8] if revision else None),
                timeout_sec=timeout,
            )
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(
                        download_checkpoint_from_hf,
                        repo_id=repo_id,
                        revision=revision,
                        filenames=[filename],
                        dest_dir=tmp_dir,
                        token_env_var=self.config.hf.token_env_var,
                    ),
                    timeout=timeout,
                )
                (tmp_dir / filename).replace(dest)
            except asyncio.TimeoutError:
                logger.warning("bg-download: timeout", uid=uid, hotkey=hotkey[:6], timeout_sec=timeout)
                inc_eval_failure(int(uid), "timeout")
                round_obj.mark_failed(uid)
                self._record_failure_metric(round_obj)
                return
            except Exception as e:
                logger.warning("bg-download: failed", uid=uid, hotkey=hotkey[:6], error=str(e))
                # HF-side or network-layer failures all surface here; bucket
                # them under "rpc" so timeouts above stay distinguishable.
                inc_eval_failure(int(uid), "rpc")
                round_obj.mark_failed(uid)
                self._record_failure_metric(round_obj)
                return
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            round_obj.publish_download(uid, dest)
            self._update_pending_metric(round_obj)
            try:
                size_bytes = dest.stat().st_size
            except OSError:
                size_bytes = None
            if size_bytes is not None:
                try:
                    CHECKPOINT_DOWNLOAD_BYTES.observe(size_bytes)
                except Exception:
                    pass
            logger.info(
                "bg-download: success",
                uid=uid, hotkey=hotkey[:6],
                repo_id=repo_id,
                revision=(revision[:8] if revision else None),
                dest=str(dest),
                size_bytes=size_bytes,
            )
        except Exception as e:
            logger.exception("bg-download: unexpected failure", uid=uid, error=str(e))

    def _existing_submission(self, submission_dir: Path, hotkey: str) -> Path | None:
        for path in submission_dir.glob("*.pt"):
            if path.name.startswith(".tmp"):
                continue
            meta = parse_dynamic_filename(path.name)
            if meta and meta.get("hotkey") == hotkey:
                return path
        return None

    @staticmethod
    def _update_pending_metric(round_obj) -> None:
        try:
            stats = round_obj.stats()
            VALIDATOR_ROUND_MINERS_PENDING.labels(round_id=str(round_obj.round_id)).set(stats["pending"])
        except Exception:
            pass

    @staticmethod
    def _record_failure_metric(round_obj) -> None:
        try:
            stats = round_obj.stats()
            VALIDATOR_ROUND_MINERS_FAILED.labels(round_id=str(round_obj.round_id)).set(stats["failed"])
            VALIDATOR_ROUND_MINERS_PENDING.labels(round_id=str(round_obj.round_id)).set(stats["pending"])
        except Exception:
            pass
