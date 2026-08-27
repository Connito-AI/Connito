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
from connito.shared.helper import find_submission_for_hotkey
from connito.shared.hf_distribute import (
    HFFileMissingError,
    HFRepoUnavailableError,
    check_file_publicly_fetchable,
    download_checkpoint_from_hf_subprocess,
)
from connito.shared.telemetry import (
    CHECKPOINT_DOWNLOAD_BYTES,
    VALIDATOR_BG_WORKER_PAUSED,
    VALIDATOR_ROUND_MINERS_FAILED,
    VALIDATOR_ROUND_MINERS_PENDING,
    inc_eval_failure,
    set_miner_eval_status,
)
from connito.validator.round import RoundRef

logger = structlog.get_logger(__name__)

# Maximum number of UIDs that may sit in `Round.downloaded_pool` waiting
# for bg-eval to pick them up before bg-download stops fetching new
# checkpoints. Without this cap, bg-download will happily pull every
# miner's shard onto disk even when bg-eval is many minutes behind, which
# wastes HF bandwidth and (more importantly) inflates the on-disk backlog
# the cycle-tail prune has to tear down. Re-checked every poll so the cap
# self-clears once eval drains the queue.
DOWNLOAD_PENDING_EVAL_CAP = 5


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

                # Backpressure on bg-eval: stop pulling more checkpoints
                # while bg-eval already has DOWNLOAD_PENDING_EVAL_CAP+ UIDs
                # queued. Counted under Round's lock so a concurrent
                # publish/pop can't skew the read.
                pending_eval = round_obj.downloaded_pending_eval_count()
                if pending_eval > DOWNLOAD_PENDING_EVAL_CAP:
                    # Log once on the rising edge into the cap; stay quiet
                    # until a successful download resets idle_ticks (same
                    # pattern as the "no pending targets" branch below).
                    if idle_ticks == 0:
                        logger.info(
                            "bg-download: pausing — eval backlog above cap",
                            pending_eval=pending_eval,
                            cap=DOWNLOAD_PENDING_EVAL_CAP,
                            round_id=getattr(round_obj, "round_id", None),
                        )
                    idle_ticks += 1
                    await asyncio.sleep(self.poll_interval_sec)
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
        try:
            ckpt = round_obj.uid_to_chain_checkpoint.get(uid)
            if ckpt is None or not (ckpt.hf_repo_id and ckpt.hf_revision):
                logger.debug("bg-download: no HF target for miner; skipping", uid=uid, hotkey=hotkey[:6])                
                round_obj.mark_failed(uid)
                self._update_pending_metric(round_obj)
                return

            repo_id, revision = ckpt.hf_repo_id, ckpt.hf_revision
            expert_group_id = self.config.task.exp.group_id
            # Only fetch `.safetensors` — pickle-free, no code-execution
            # surface. Legacy `.pt` submissions are no longer probed: doing
            # so doubled HF round-trips on every miner that hadn't uploaded
            # at all (every miss became two 404s), and `.pt` is deprecated
            # for new miner submissions.
            candidate_filenames = [f"model_expgroup_{expert_group_id}.safetensors"]
            submission_dir = Path(self.config.ckpt.miner_submission_path)
            submission_dir.mkdir(parents=True, exist_ok=True)

            # Skip if a submission for this hotkey already exists locally
            # (e.g. validator restarted mid-round and the file is still on
            # disk). The match is gated on block ∈ this round's submission
            # window — without that filter, a leftover file from a previous
            # cycle would short-circuit the fresh fetch and get published,
            # but `gather_validation_job` would silently reject it for
            # being out-of-window.
            existing = find_submission_for_hotkey(
                submission_dir, hotkey, round_obj.submission_block_range,
            )
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

            logger.info(
                "bg-download: fetching",
                uid=uid, hotkey=hotkey[:6],
                repo_id=repo_id,
                revision=(revision[:8] if revision else None),
                timeout_sec=timeout,
                candidates=candidate_filenames,
            )

            downloaded_filename: str | None = None
            last_error: Exception | None = None
            # Count candidates whose failure was definitively
            # miner-attributable (repo/revision gone, file absent at the
            # committed revision) — as opposed to network trouble. Only when
            # EVERY candidate missed this way do we consider blaming the
            # miner below.
            definitive_misses = 0
            try:
                for candidate in candidate_filenames:
                    # Clear tmp_dir between attempts so a partial download
                    # from a missing-file failure can't pollute the next try.
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    try:
                        # Subprocess isolation: a fresh `spawn`ed child runs
                        # `hf_hub_download` and is fully terminable on timeout.
                        # Previously we ran the download in a private
                        # ThreadPoolExecutor and deliberately leaked the worker
                        # thread on timeout (`shutdown(wait=False)`); over a
                        # multi-day validator that, combined with
                        # `huggingface_hub`'s xet backend churning its own
                        # internal worker pool, degraded the shared HF session
                        # to the point that present-on-HF files began returning
                        # spurious "no candidate file" (see PR description for
                        # the per-(uid,repo,revision) comparison against yuma).
                        # The child has no shared state with the parent, so
                        # one bad download cannot poison the next.
                        await asyncio.to_thread(
                            download_checkpoint_from_hf_subprocess,
                            repo_id=repo_id,
                            revision=revision,
                            filenames=[candidate],
                            dest_dir=tmp_dir,
                            token_env_var=self.config.hf.token_env_var,
                            timeout_sec=timeout,
                        )
                        downloaded_filename = candidate
                        break
                    except TimeoutError:
                        # Hard timeout means the validator's network is the
                        # problem, not a missing file — don't waste budget
                        # retrying with a different suffix. The subprocess
                        # variant raises the builtin TimeoutError (the old
                        # thread variant used concurrent.futures.TimeoutError);
                        # both subclass nothing in common, so we explicitly
                        # match builtin TimeoutError here.
                        logger.warning(
                            "bg-download: timeout",
                            uid=uid, hotkey=hotkey[:6], timeout_sec=timeout,
                        )
                        inc_eval_failure(int(uid), "timeout")
                        round_obj.mark_failed(uid)
                        self._record_failure_metric(round_obj)
                        return
                    except HFRepoUnavailableError as e:
                        # Repo-level miss (deleted / private / gated /
                        # revision rewritten) — affects every candidate, so
                        # don't waste round-trips on the rest.
                        last_error = e
                        definitive_misses = len(candidate_filenames)
                        break
                    except HFFileMissingError as e:
                        last_error = e
                        definitive_misses += 1
                        logger.debug(
                            "bg-download: candidate not present, trying next",
                            uid=uid, hotkey=hotkey[:6],
                            candidate=candidate, error=str(e),
                        )
                        continue
                    except Exception as e:
                        last_error = e
                        logger.debug(
                            "bg-download: candidate failed, trying next",
                            uid=uid, hotkey=hotkey[:6],
                            candidate=candidate, error=str(e),
                        )
                        continue

                if downloaded_filename is None:
                    # Miner-fault attribution: the download stack said the
                    # repo/revision/file is definitively gone (typed HF
                    # errors, not network trouble) for EVERY candidate. Before
                    # blaming the miner, get a second opinion from an
                    # unauthenticated probe — if the file is publicly
                    # fetchable without our token, the failure was on our
                    # side (expired token, proxy, HF session state) and the
                    # miner must keep its prior average. Both signals have to
                    # agree before we zero anyone.
                    if (
                        definitive_misses >= len(candidate_filenames)
                        and getattr(
                            self.config.evaluation,
                            "repo_unavailable_is_miner_fault",
                            False,
                        )
                    ):
                        publicly_fetchable = await asyncio.to_thread(
                            check_file_publicly_fetchable,
                            repo_id=repo_id,
                            revision=revision,
                            filename=candidate_filenames[0],
                        )
                        if publicly_fetchable is False:
                            logger.warning(
                                "bg-download: committed checkpoint is not publicly "
                                "retrievable (repo deleted/private/gated or file "
                                "gone) — miner fault, will record score=0 at finalize",
                                uid=uid, hotkey=hotkey[:6],
                                repo_id=repo_id,
                                revision=(revision[:8] if revision else None),
                                last_error=str(last_error) if last_error else None,
                            )
                            inc_eval_failure(int(uid), "repo_unavailable")
                            set_miner_eval_status(int(uid), "repo_unavailable")
                            round_obj.mark_validation_failed(uid)
                            self._record_failure_metric(round_obj)
                            return
                        logger.warning(
                            "bg-download: typed HF miss but public probe did not "
                            "confirm — treating as operational (validator-side)",
                            uid=uid, hotkey=hotkey[:6],
                            repo_id=repo_id,
                            probe_result=publicly_fetchable,
                            last_error=str(last_error) if last_error else None,
                        )
                    # Surface enough state for the "no candidate file" miss to be
                    # diagnosable after the fact. The plain `str(last_error)` we
                    # used to log was empty in ~80% of misses (HF's
                    # `EntryNotFoundError.__str__` can return ""), which left no
                    # way to tell apart "file genuinely absent at this revision"
                    # from "auth error" or "network blip". Now we also emit the
                    # exception type, its repr, and the (repo_id, revision)
                    # pair the chain handed us, so cross-validator divergence
                    # on the same UID can be traced back to which revision each
                    # validator's substrate read returned.
                    last_error_type = (
                        type(last_error).__name__ if last_error is not None else None
                    )
                    last_error_repr = repr(last_error) if last_error is not None else None
                    logger.warning(
                        "bg-download: no candidate file found in HF repo",
                        uid=uid, hotkey=hotkey[:6],
                        candidates=candidate_filenames,
                        repo_id=repo_id,
                        revision=(revision[:8] if revision else None),
                        last_error=str(last_error) if last_error else None,
                        last_error_type=last_error_type,
                        last_error_repr=last_error_repr,
                    )
                    # HF-side or network-layer failures all surface here; bucket
                    # them under "rpc" so timeouts above stay distinguishable.
                    inc_eval_failure(int(uid), "rpc")
                    round_obj.mark_failed(uid)
                    self._record_failure_metric(round_obj)
                    return

                # Preserve the source extension so downstream loaders can
                # dispatch by suffix (.safetensors vs .pt).
                dest_name = (
                    f"hotkey_{hotkey}_block_{block}{Path(downloaded_filename).suffix}"
                )
                dest = submission_dir / dest_name
                (tmp_dir / downloaded_filename).replace(dest)
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
