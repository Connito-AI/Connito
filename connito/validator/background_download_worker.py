"""Step (1) of the round lifecycle: download miner HF checkpoints in the
background, in incentive order, into the round's `downloaded_pool`.

This worker is network-only. It is paused while:
  - the foreground holds GPU/HF (`foreground_active` set), or
  - the main loop is in the Merge phase (`merge_phase_active` set).

It does not gate on `eval_window_active`; download continues across the
entire round (Submission → Train(K+1)) so the eval worker has work
queued when its window opens.
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
    VALIDATOR_BG_WORKER_PAUSED,
    VALIDATOR_ROUND_MINERS_FAILED,
    VALIDATOR_ROUND_MINERS_PENDING,
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
        foreground_active: threading.Event,
        merge_phase_active: threading.Event,
        stop_event: threading.Event | None = None,
        poll_interval_sec: float = 6.0,
    ) -> None:
        super().__init__(daemon=True, name="connito-bg-download")
        self.config = config
        self.round_ref = round_ref
        self.foreground_active = foreground_active
        self.merge_phase_active = merge_phase_active
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

        try:
            while not self.stop_event.is_set():
                round_obj = self.round_ref.current
                if round_obj is None:
                    await asyncio.sleep(self.poll_interval_sec)
                    continue

                # Snapshot pause state for telemetry.
                paused = self.foreground_active.is_set() or self.merge_phase_active.is_set()
                try:
                    VALIDATOR_BG_WORKER_PAUSED.labels(worker="download").set(1 if paused else 0)
                except Exception:
                    pass
                if paused:
                    await self._wait_clear()
                    continue

                # Pick the next UID to download.
                target = self._next_target(round_obj)
                if target is None:
                    # Nothing to do for the current round; sleep and re-check.
                    await asyncio.sleep(self.poll_interval_sec)
                    continue

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
        while not self.stop_event.is_set():
            if not self.foreground_active.is_set() and not self.merge_phase_active.is_set():
                return
            await asyncio.sleep(0.5)

    async def _download_one(self, round_obj, *, uid: int, hotkey: str) -> None:
        timeout = float(self.config.evaluation.per_miner_download_timeout_sec)
        # background_uids and foreground_uids are disjoint, and there is a
        # single download thread, so we don't need to claim the UID here.
        # publish_download is a no-op if the UID has already been scored.
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
                round_obj.publish_download(uid, existing)
                self._update_pending_metric(round_obj)
                return

            tmp_dir = submission_dir / f".tmp_bg_dl_{hotkey}"
            block = self._subtensor.block if self._subtensor is not None else 0
            dest_name = f"hotkey_{hotkey}_block_{block}.pt"
            dest = submission_dir / dest_name

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
                round_obj.mark_failed(uid)
                self._record_failure_metric(round_obj)
                return
            except Exception as e:
                logger.warning("bg-download: failed", uid=uid, hotkey=hotkey[:6], error=str(e))
                round_obj.mark_failed(uid)
                self._record_failure_metric(round_obj)
                return
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            round_obj.publish_download(uid, dest)
            self._update_pending_metric(round_obj)
            logger.info("bg-download: published", uid=uid, hotkey=hotkey[:6], dest=str(dest))
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
