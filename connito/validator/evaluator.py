from __future__ import annotations

import asyncio
import copy
import gc
from dataclasses import dataclass
from pathlib import Path

import bittensor
import torch
import torch.nn as nn

from connito.shared.app_logging import structlog
from connito.shared.dataloader import get_dataloader
from connito.shared.evaluate import evaluate_model
from connito.shared.helper import get_model_hash
from connito.shared.telemetry import track_eval_latency, track_model_load_latency

logger = structlog.get_logger(__name__)


# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class MinerEvalJob:
    uid: int
    hotkey: str
    model_path: str
    step: int


# -------------------------- Pipeline Config -----------------------------------
MAX_CONCURRENT_DOWNLOADS = 4
EVAL_WORKERS = 1
DOWNLOAD_TIMEOUT_SEC = 60
EVAL_MAX_BATCHES = 50
# ------------------------------------------------------------------------------

# def load_model_from_path(path: str, base_model, device: torch.device) -> nn.Module:
#     sd = torch.load(path, map_location=torch.device("cpu"))["model_state_dict"]
#     model = copy.deepcopy(base_model)
#     model.load_state_dict(sd, strict=False)
#     return model.to(device)

@track_model_load_latency()
def load_model_from_path(path: str, base_model: nn.Module, device: torch.device) -> nn.Module:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format at {path}: {type(ckpt).__name__}")

    if len(sd) == 0:
        raise ValueError(f"Checkpoint at {path} has empty model_state_dict")

    model = copy.deepcopy(base_model)

    # Keys in each state_dict (before loading)
    base_sd = base_model.state_dict()
    base_keys = set(base_sd.keys())
    ckpt_keys = set(sd.keys())

    # 1) Params that are the same across both dicts (intersection).
    #    (Optional: filter to ones with matching shapes too.)
    common_keys = base_keys & ckpt_keys
    common_same_shape = {k for k in common_keys if base_sd[k].shape == sd[k].shape}

    # 2) Keys containing 'expert' that exist in the checkpoint but NOT in the base model
    expert_not_in_base = {k for k in ckpt_keys - base_keys if "expert" in k}

    # 3) "expert" keys in base_model but NOT in checkpoint/common_keys
    expert_in_base_not_common = {k for k in (base_keys - common_keys) if "expert" in k}

    if len(common_same_shape) == 0:
        logger.warning(
            "No compatible keys between checkpoint and base model — "
            "checkpoint is likely from a different architecture or naming convention",
            ckpt_key_count=len(ckpt_keys),
            base_key_count=len(base_keys),
            sample_ckpt_keys=sorted(k for k in ckpt_keys if "expert" in k)[:5],
            sample_base_keys=sorted(k for k in base_keys if "expert" in k)[:5],
        )
    elif expert_not_in_base:
        logger.warning(
            "Expert keys in checkpoint not found in base model",
            expert_not_in_base=len(expert_not_in_base),
            sample_keys=sorted(expert_not_in_base)[:5],
        )
    elif common_same_shape != common_keys:
        logger.warning(
            "Some common keys have mismatched shapes",
            common_keys=len(common_keys),
            common_same_shape=len(common_same_shape),
            shape_mismatch=len(common_keys - common_same_shape),
            sample_mismatched=sorted(common_keys - common_same_shape)[:5],
        )
    else:
        logger.debug(
            "Key summary",
            common_keys=len(common_keys),
            common_same_shape=len(common_same_shape),
            expert_in_base_not_common=len(expert_in_base_not_common),
        )

    # Load weights (strict=False so missing/unexpected are allowed)
    incompatible = model.load_state_dict(sd, strict=False)

    # # Extra helpful debug (optional)
    # if incompatible.missing_keys:
    #     print(f"[load_model] missing keys (first 50): {incompatible.missing_keys[:50]}")
    # if incompatible.unexpected_keys:
    #     print(f"[load_model] unexpected keys (first 50): {incompatible.unexpected_keys[:50]}")

    return model.to(device)


async def _evaluate_on_fresh_loader(
    *,
    config,
    tokenizer,
    combinded_seed: str,
    step: int,
    model: nn.Module,
    device: torch.device,
    max_eval_batches: int,
    rank: int | None = None,
) -> dict:
    """Build a fresh eval dataloader and run evaluate_model on the given model.

    Every caller shares the same `combinded_seed`, so the baseline and all
    miner evals see the same batches — the deltas are comparable.
    """
    dataloader = await asyncio.to_thread(
        get_dataloader,
        config=config,
        tokenizer=tokenizer,
        seed=combinded_seed,
        rank=0,
        world_size=config.dataloader.world_size,
    )

    @track_eval_latency()
    def _run():
        return evaluate_model(step, model, dataloader, device, max_eval_batches, rank)

    try:
        return await asyncio.to_thread(_run)
    finally:
        del dataloader


async def evaluate_one_miner(
    *,
    config,
    model_path: str | Path,
    uid: int,
    hotkey: str,
    base_model: nn.Module,
    tokenizer,
    combined_seed: str,
    device: torch.device,
    score_aggregator,
    baseline_loss: float,
    step: int,
    round_id: int | None = None,
    max_eval_batches: int = EVAL_MAX_BATCHES,
    rank: int | None = None,
) -> "MinerEvalJob | None":
    """Evaluate a single miner and record the score.

    Shared between foreground (`stream_gather_and_evaluate`) and the
    `BackgroundEvalWorker`. `base_model` is treated as read-only — caller
    is responsible for not mutating it across calls so successive miners
    see an identical baseline.

    Returns a `MinerEvalJob` on success (so the caller can later use
    `model_path` for gradient aggregation), or None on failure.
    """
    job = MinerEvalJob(uid=int(uid), hotkey=hotkey, model_path=str(model_path), step=int(step))
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        miner_model = await asyncio.to_thread(load_model_from_path, str(model_path), base_model, device)

        try:
            metrics = await _evaluate_on_fresh_loader(
                config=config,
                tokenizer=tokenizer,
                combinded_seed=combined_seed,
                step=step,
                model=miner_model,
                device=device,
                max_eval_batches=max_eval_batches,
                rank=rank,
            )
        finally:
            del miner_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        val_loss = float(metrics.get("val_loss", 100))
        delta = max(0.0, baseline_loss - val_loss)
        score = delta ** 1.2
        score_aggregator.add_score(uid=int(uid), hotkey=hotkey, score=score, round_id=round_id)
        logger.info(
            "evaluate_one_miner: complete",
            uid=int(uid),
            hotkey=hotkey[:6],
            val_loss=round(val_loss, 4),
            baseline_loss=round(baseline_loss, 4),
            delta=round(delta, 4),
            score=round(score, 6),
            round_id=round_id,
        )
        return job
    except torch.cuda.OutOfMemoryError:
        logger.error("evaluate_one_miner: OOM", uid=int(uid))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None
    except Exception as e:
        logger.exception("evaluate_one_miner: failed", uid=int(uid), error=str(e))
        return None


def resolve_miner_hf_target(
    *,
    config,
    subtensor: bittensor.Subtensor,
    hotkey: str,
) -> tuple[str, str] | None:
    """Resolve a miner's (hf_repo_id, hf_revision) from the chain commits.

    Returns None when the miner has no chain commit, no HF coords, or
    when fetching commits fails.
    """
    from connito.shared.checkpoints import build_chain_checkpoints_from_previous_phase

    try:
        chain_checkpoints = build_chain_checkpoints_from_previous_phase(
            config=config, subtensor=subtensor, for_role="miner",
        )
    except Exception as e:
        logger.warning("resolve_miner_hf_target: failed to fetch chain checkpoints", error=str(e))
        return None

    for ckpt in chain_checkpoints.checkpoints:
        if ckpt.hotkey != hotkey:
            continue
        if not (ckpt.hf_repo_id and ckpt.hf_revision):
            return None
        return ckpt.hf_repo_id, ckpt.hf_revision
    return None


async def evaluator_worker(
    name: str,
    config,
    jobs_q: asyncio.Queue[MinerEvalJob],
    aggregator: MinerScoreAggregator,
    device: torch.device,
    base_model: nn.Module,
    tokenizer,
    combinded_seed: str,
    baseline_loss: float,
    max_eval_batches: int = EVAL_MAX_BATCHES,
    rank: int | None = None,
):
    import gc

    while True:
        job = await jobs_q.get()
        if job is None:  # type: ignore
            jobs_q.task_done()
            logger.debug(f"{name}: shutdown signal received.")
            break

        try:
            snap = torch.cuda.memory_snapshot()
            if not snap:
                logger.warning(f"{name}: no CUDA memory segments found")
            else:
                logger.debug(f"{name}: CUDA memory segments", count=len(snap))

            # Clear memory before loading
            gc.collect()
            torch.cuda.empty_cache()

            logger.debug(f"{name}: Evaluating hotkey={job.hotkey}")

            # Load model (potentially blocking) in a thread

            model = await asyncio.to_thread(load_model_from_path, job.model_path, base_model, device)

            logger.info(
                f"{name}: starting evaluation",
                hotkey=job.hotkey,
                base_hash=get_model_hash(base_model.state_dict(), hex=True)[:6],
                merged_hash=get_model_hash(model.state_dict(), hex=True)[:6],
            )

            metrics = await _evaluate_on_fresh_loader(
                config=config,
                tokenizer=tokenizer,
                combinded_seed=combinded_seed,
                step=job.step,
                model=model,
                device=device,
                max_eval_batches=max_eval_batches,
                rank=rank,
            )

            # Score = (baseline_loss - miner_loss) ** 1.5. Direct loss-space delta
            # isolates the miner's contribution over the un-merged baseline. Clamped
            # at 0 because (-x) ** 1.5 returns a complex number in Python.
            val_loss = float(metrics.get("val_loss", 100))
            delta = max(0.0, baseline_loss - val_loss)
            score = delta ** 1.2
            aggregator.add_score(job.uid, job.hotkey, score)
            logger.info(
                f"{name}: evaluation complete",
                uid=job.uid,
                hotkey=job.hotkey[:6],
                val_loss=round(val_loss, 4),
                baseline_loss=round(baseline_loss, 4),
                delta=round(delta, 4),
                score=round(score, 6),
            )

            # Explicit cleanup
            del model, metrics
            gc.collect()
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            logger.error(f"{name}: OOM for uid={job.uid}")
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            logger.exception(f"{name}: Evaluation failed for uid={job.uid}: {e}")
        finally:
            jobs_q.task_done()


async def run_evaluation(
    config, step, device, miners, score_aggregator, base_model: nn.Module, tokenizer, combinded_seed
):
    import gc

    # --- Baseline: evaluate the un-merged base model on the same eval stream so
    # each miner's score can be expressed as an improvement delta over it. ---
    baseline_metrics = await _evaluate_on_fresh_loader(
        config=config,
        tokenizer=tokenizer,
        combinded_seed=combinded_seed,
        step=step,
        model=base_model,
        device=device,
        max_eval_batches=EVAL_MAX_BATCHES,
    )
    baseline_loss = float(baseline_metrics.get("val_loss", 100))
    logger.info(
        "Baseline evaluation complete (no miner merged)",
        val_loss=round(baseline_loss, 4),
    )

    del baseline_metrics
    gc.collect()
    torch.cuda.empty_cache()

    miners_q: asyncio.Queue[MinerEvalJob] = asyncio.Queue()

    # Enqueue miners
    for m in miners:
        await miners_q.put(m)

    # Spin up evaluator workers
    eval_workers = [
        asyncio.create_task(
            evaluator_worker(
                f"evaluator-{i+1}", config, miners_q, score_aggregator, device,
                base_model, tokenizer, combinded_seed, baseline_loss=baseline_loss,
            )
        )
        for i in range(EVAL_WORKERS)
    ]

    # Wait for all miners to be processed
    await miners_q.join()

    # Signal evaluator workers to stop
    for _ in eval_workers:
        await miners_q.put(None)

    await asyncio.gather(*eval_workers)


async def stream_gather_and_evaluate(
    config,
    subtensor: bittensor.Subtensor,
    step: int,
    device: torch.device,
    score_aggregator,
    base_model: nn.Module,
    tokenizer,
    combined_seed: str,
    end_block: int,
    validator_miner_assignment: dict[str, list[str]],
    poll_interval_sec: float = 6.0,
) -> list[MinerEvalJob]:
    """
    Stream-evaluate miner submissions as they land during the combined
    Submission + Validate window.

    Runs the baseline once, spins up a single evaluator worker, and polls
    `config.ckpt.miner_submission_path` via gather_validation_job every
    `poll_interval_sec` seconds. New qualifying submissions are enqueued
    as they appear (deduped by hotkey). Polling stops once
    `subtensor.block > end_block`; the queue is then drained and the
    worker shut down.

    `validator_miner_assignment` is provided by the caller (computed once
    at submission start) so the penalty pass after evaluation can reuse
    the exact same set of miners, avoiding drift from a later recompute.

    Returns the list of MinerEvalJobs that were evaluated so downstream
    aggregation can operate on the same set.
    """
    # Deferred import — gather_validation_job depends on config schemas that
    # would otherwise create a circular import at module load.
    from connito.shared.cycle import gather_validation_job, hydrate_miner_submissions_from_hf

    # Pull HF-committed miner checkpoints in the background while baseline and
    # polling run. All HF coords are on chain by Submission phase (miners upload
    # during MinerCommit2), so a single pass is enough; the HTTP
    # /submit-checkpoint path fills in miners who couldn't use HF. Written files
    # land atomically (os.replace) so the poll loop picks them up as soon as
    # each shard lands without racing partial data.
    async def _hydrate_and_log() -> None:
        # Dedicated Subtensor for the hydration thread — sharing the caller's
        # subtensor here triggers websockets ConcurrencyError, since the main
        # coroutine concurrently drives subtensor.block / gather_validation_job
        # against the same WS connection.
        try:
            hydration_subtensor = await asyncio.to_thread(
                bittensor.Subtensor, network=subtensor.network
            )
        except Exception as e:
            logger.warning("Streaming eval: failed to open hydration subtensor, skipping HF hydration", error=str(e))
            return
        try:
            hydrated = await asyncio.to_thread(
                hydrate_miner_submissions_from_hf,
                config,
                hydration_subtensor,
                validator_miner_assignment,
            )
            logger.info("Streaming eval: hydrated miner submissions from HF", count=hydrated)
        except Exception as e:
            logger.warning("Streaming eval: HF hydration failed, continuing with HTTP only", error=str(e))

    hydration_task = asyncio.create_task(_hydrate_and_log())

    # --- Baseline: unmerged base model scored once, reused for all deltas ---
    baseline_metrics = await _evaluate_on_fresh_loader(
        config=config,
        tokenizer=tokenizer,
        combinded_seed=combined_seed,
        step=step,
        model=base_model,
        device=device,
        max_eval_batches=EVAL_MAX_BATCHES,
    )
    baseline_loss = float(baseline_metrics.get("val_loss", 100))
    del baseline_metrics
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(
        "Streaming evaluation baseline complete",
        val_loss=round(baseline_loss, 4),
        end_block=end_block,
        current_block=subtensor.block,
    )

    jobs_q: asyncio.Queue[MinerEvalJob] = asyncio.Queue()
    enqueued_hotkeys: set[str] = set()
    all_jobs: list[MinerEvalJob] = []

    worker = asyncio.create_task(
        evaluator_worker(
            "evaluator-streaming",
            config,
            jobs_q,
            score_aggregator,
            device,
            base_model,
            tokenizer,
            combined_seed,
            baseline_loss=baseline_loss,
        )
    )

    try:
        while subtensor.block <= end_block:
            try:
                jobs = gather_validation_job(
                    config,
                    subtensor,
                    step=step,
                    validator_miner_assignment=validator_miner_assignment,
                )
            except Exception as e:
                logger.warning("stream_evaluate: gather_validation_job failed", error=str(e))
                jobs = []

            new_count = 0
            for job in jobs:
                if job.hotkey in enqueued_hotkeys:
                    continue
                enqueued_hotkeys.add(job.hotkey)
                all_jobs.append(job)
                await jobs_q.put(job)
                new_count += 1

            if new_count:
                logger.info(
                    "Streaming eval: enqueued new submissions",
                    enqueued=new_count,
                    total_enqueued=len(all_jobs),
                    queued_waiting=jobs_q.qsize(),
                    current_block=subtensor.block,
                    end_block=end_block,
                )

            if subtensor.block > end_block:
                break
            await asyncio.sleep(poll_interval_sec)
    finally:
        # Don't extend the Submission phase waiting on a slow HF peer. Anything
        # already downloaded was picked up by the poll loop via filename scan;
        # anything mid-download at phase end is abandoned and will be retried
        # next cycle. Cancel doesn't kill the underlying worker thread from
        # asyncio.to_thread, but the result is discarded so the coroutine exits
        # cleanly and no "Task was destroyed" warning fires.
        if not hydration_task.done():
            hydration_task.cancel()
        try:
            await hydration_task
        except (asyncio.CancelledError, Exception):
            pass
        # Drain whatever is still in the queue before stopping the worker.
        await jobs_q.join()
        await jobs_q.put(None)  # type: ignore[arg-type]
        await worker

    logger.info(
        "Streaming evaluation finished",
        evaluated=len(all_jobs),
        final_block=subtensor.block,
        end_block=end_block,
    )
    return all_jobs


async def evaluate_foreground_round(
    *,
    config,
    round_obj,  # connito.validator.round.Round
    subtensor: bittensor.Subtensor,
    step: int,
    device: torch.device,
    score_aggregator,
    base_model: nn.Module,
    tokenizer,
    end_block: int,
    poll_interval_sec: float = 6.0,
    per_miner_eval_timeout_sec: float | None = None,
) -> list[MinerEvalJob]:
    """Foreground (step 2): evaluate the round's top-N miners during
    Submission + Validate.

    Walks `round_obj.foreground_uids` only; downloads miner checkpoints
    from HF and from the existing HTTP submission path; calls
    `evaluate_one_miner` for each. UIDs that exceed the per-miner budget
    or fail to download by `end_block` are left unclaimed so the
    `BackgroundEvalWorker` can pick them up in step 3.
    """
    from connito.shared.cycle import gather_validation_job, hydrate_miner_submissions_from_hf

    # Kick off HF hydration in the background — same pattern as the legacy
    # stream_gather_and_evaluate.
    async def _hydrate() -> None:
        try:
            hydration_subtensor = await asyncio.to_thread(
                bittensor.Subtensor, network=subtensor.network
            )
        except Exception as e:
            logger.warning("foreground eval: failed to open hydration subtensor", error=str(e))
            return
        try:
            hydrated = await asyncio.to_thread(
                hydrate_miner_submissions_from_hf,
                config,
                hydration_subtensor,
                round_obj.validator_miner_assignment,
            )
            logger.info("foreground eval: hydrated miner submissions from HF", count=hydrated)
        except Exception as e:
            logger.warning("foreground eval: HF hydration failed", error=str(e))

    hydration_task = asyncio.create_task(_hydrate())

    # Baseline once against the round's input model (= live `base_model`,
    # which equals round.model_snapshot_cpu since the foreground runs
    # before Merge(K)).
    baseline_metrics = await _evaluate_on_fresh_loader(
        config=config,
        tokenizer=tokenizer,
        combinded_seed=round_obj.seed,
        step=step,
        model=base_model,
        device=device,
        max_eval_batches=EVAL_MAX_BATCHES,
    )
    baseline_loss = float(baseline_metrics.get("val_loss", 100))
    del baseline_metrics
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    foreground_set = set(round_obj.foreground_uids)
    completed: list[MinerEvalJob] = []

    try:
        while subtensor.block <= end_block:
            try:
                discovered = gather_validation_job(
                    config,
                    subtensor,
                    step=step,
                    validator_miner_assignment=round_obj.validator_miner_assignment,
                )
            except Exception as e:
                logger.warning("foreground eval: gather_validation_job failed", error=str(e))
                discovered = []

            # Walk top-N in incentive order; pick up any whose checkpoint
            # has landed and is not yet claimed/scored.
            by_uid: dict[int, MinerEvalJob] = {j.uid: j for j in discovered if j.uid in foreground_set}
            progressed = False
            for entry in round_obj.roster:
                if entry.uid not in foreground_set:
                    continue
                if entry.uid not in by_uid:
                    continue
                if not round_obj.claim_for_foreground(entry.uid):
                    continue
                job = by_uid[entry.uid]
                progressed = True
                eval_coro = evaluate_one_miner(
                    config=config,
                    model_path=job.model_path,
                    uid=entry.uid,
                    hotkey=entry.hotkey,
                    base_model=base_model,
                    tokenizer=tokenizer,
                    combined_seed=round_obj.seed,
                    device=device,
                    score_aggregator=score_aggregator,
                    baseline_loss=baseline_loss,
                    step=step,
                    round_id=round_obj.round_id,
                )
                try:
                    if per_miner_eval_timeout_sec:
                        evaluated = await asyncio.wait_for(eval_coro, timeout=per_miner_eval_timeout_sec)
                    else:
                        evaluated = await eval_coro
                except asyncio.TimeoutError:
                    logger.warning(
                        "foreground eval: per-miner timeout — leaving for background spillover",
                        uid=entry.uid, hotkey=entry.hotkey[:6],
                    )
                    round_obj.release_claim(entry.uid)
                    continue
                except Exception as e:
                    logger.exception("foreground eval: unexpected failure", uid=entry.uid, error=str(e))
                    round_obj.release_claim(entry.uid)
                    continue

                if evaluated is None:
                    round_obj.release_claim(entry.uid)
                    continue
                round_obj.mark_scored(entry.uid)
                completed.append(evaluated)

            # Stop once every top-N UID is scored or the phase boundary hits.
            scored_top_n = sum(1 for u in foreground_set if u in round_obj.scored_uids)
            if scored_top_n >= len(foreground_set):
                break

            if subtensor.block > end_block:
                break
            if not progressed:
                await asyncio.sleep(poll_interval_sec)
    finally:
        if not hydration_task.done():
            hydration_task.cancel()
        try:
            await hydration_task
        except (asyncio.CancelledError, Exception):
            pass

    logger.info(
        "foreground eval: complete",
        round_id=round_obj.round_id,
        top_n=len(foreground_set),
        scored=len(completed),
        spilled=len(foreground_set) - len(completed),
    )
    return completed
