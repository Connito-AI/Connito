from __future__ import annotations

import asyncio
import copy
import gc
import os
from dataclasses import dataclass
from pathlib import Path

import bittensor
import torch
import torch.nn as nn

from connito.shared.app_logging import structlog
from connito.shared.dataloader import get_dataloader
from connito.shared.evaluate import evaluate_model
from connito.shared.telemetry import track_eval_latency, track_model_load_latency

logger = structlog.get_logger(__name__)


# -----------------------------------------------------------------------------
def validate_miner_submission(
    *,
    round_obj,  # connito.validator.round.Round
    uid: int,
    model_path: str | Path,
    expert_group_assignment,
) -> str | None:
    """Run the existing `ChainCheckpoint.validate(...)` against a miner's
    on-disk submission before it is fed to `evaluate_one_miner`.

    Returns ``None`` on success. On failure returns a short reason string —
    one of ``no_chain_commit | signature | hash | expert_group | nan_inf``,
    or a generic ``"unknown"`` if the helper raised. The reason is intended
    to be plumbed into telemetry labels and log lines.

    The chain checkpoint is read from `round_obj.uid_to_chain_checkpoint`
    so this never re-fetches anything from the chain. The check itself is
    `ChainCheckpoint.validate(expert_group_assignment=…)`, which runs:

    - `_verify_signature` — the chain hotkey signed `model_hash`.
    - `_verify_hash` — the on-disk shard's hash matches the chain commit.
    - `_verify_expert_group` — every routed-expert key in the state dict
      belongs to the miner's assigned group, and no tensor contains NaN/Inf.
    """
    chain_checkpoint = round_obj.uid_to_chain_checkpoint.get(int(uid))
    if chain_checkpoint is None:
        return "no_chain_commit"

    # `validate()` reads the state dict from `chain_checkpoint.path`; point
    # it at the on-disk submission for this round.
    chain_checkpoint.path = Path(model_path)

    try:
        ok = chain_checkpoint.validate(expert_group_assignment=expert_group_assignment)
    except Exception as e:
        logger.warning(
            "validate_miner_submission: validate() raised",
            uid=int(uid), error=str(e), exc_info=True,
        )
        return "unknown"

    if ok:
        return None

    # `validate()` already logged a structured warning per failed sub-check.
    # Map the per-check booleans to a single short reason for telemetry.
    if not getattr(chain_checkpoint, "signature_verified", False):
        return "signature"
    if not getattr(chain_checkpoint, "hash_verified", False):
        return "hash"
    if not getattr(chain_checkpoint, "expert_group_verified", False):
        # _verify_expert_group folds the NaN/Inf scan in with the routing
        # check, so we cannot tell them apart from the booleans alone. The
        # underlying logger.warning at the failure site distinguishes them.
        return "expert_group_or_nan"
    return "unknown"


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
    score_path: str | os.PathLike | None = None,
) -> "MinerEvalJob | None":
    """Evaluate a single miner and record the score.

    Shared between foreground (`evaluate_foreground_round`) and the
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
        if score_path is not None:
            try:
                score_aggregator.persist_atomic(score_path)
            except Exception as e:
                logger.warning("evaluate_one_miner: persist_atomic failed", uid=int(uid), error=str(e))
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
    expert_group_assignment,
    poll_interval_sec: float = 6.0,
    per_miner_eval_timeout_sec: float | None = None,
    score_path: str | os.PathLike | None = None,
) -> list[MinerEvalJob]:
    """Foreground (step 2): evaluate the round's top-N miners during
    Submission + Validate.

    Walks `round_obj.foreground_uids` only and calls `evaluate_one_miner`
    for each. Miner checkpoints are made available locally by the
    `BackgroundDownloadWorker` (HF); this function does not pull from HF
    itself. UIDs that exceed the per-miner budget or fail to land by
    `end_block` are left unclaimed so the `BackgroundEvalWorker` can pick
    them up in step 3.
    """
    # Lazy imports — connito.shared.cycle imports this module, so a top-
    # level import would create a cycle.
    from connito.shared.cycle import BITTENSOR_BLOCK_TIME_SECONDS, gather_validation_job

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

    logger.info(
        "foreground eval: starting",
        round_id=round_obj.round_id,
        foreground_uids=list(round_obj.foreground_uids),
        end_block=end_block,
        current_block=subtensor.block,
        baseline_loss=round(baseline_loss, 4),
        per_miner_eval_timeout_sec=per_miner_eval_timeout_sec,
    )

    poll_idx = 0
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

        # Walk foreground UIDs in incentive order; pick up any whose
        # checkpoint has landed and is not yet claimed/scored.
        by_uid: dict[int, MinerEvalJob] = {j.uid: j for j in discovered if j.uid in foreground_set}
        scored_count = sum(1 for u in foreground_set if u in round_obj.scored_uids)
        current_block = subtensor.block
        logger.info(
            "foreground eval: poll",
            round_id=round_obj.round_id,
            poll_idx=poll_idx,
            current_block=current_block,
            blocks_remaining=max(0, end_block - current_block),
            discovered_total=len(discovered),
            discovered_in_foreground=len(by_uid),
            ready_uids=sorted(by_uid.keys()),
            scored=scored_count,
            foreground_total=len(foreground_set),
        )
        poll_idx += 1
        progressed = False
        phase_deadline_crossed = False
        for uid in round_obj.foreground_uids:
            if uid not in by_uid:
                continue
            # Hard-stop before claiming if Validate has ended. Without
            # this, the inner for-loop walks every foreground UID before
            # the outer `subtensor.block > end_block` check fires, so a
            # 5-miner round can spill ~5 × per_miner_eval_timeout_sec
            # past end_block.
            block_now = subtensor.block
            if block_now > end_block:
                phase_deadline_crossed = True
                break
            if not round_obj.claim_for_foreground(uid):
                continue
            job = by_uid[uid]
            hotkey = round_obj.uid_to_hotkey[uid]
            progressed = True

            # Verify the on-disk submission against the chain commit (signed
            # hash, hash, expert-group ownership, NaN/Inf scan) BEFORE the
            # GPU eval. A failure here means the submission is off-spec —
            # mark the miner failed so the missed-submission penalty pass
            # zeroes their score for the round.
            fail_reason = await asyncio.to_thread(
                validate_miner_submission,
                round_obj=round_obj,
                uid=uid,
                model_path=job.model_path,
                expert_group_assignment=expert_group_assignment,
            )
            if fail_reason is not None:
                logger.warning(
                    "foreground eval: submission failed validation — marking failed",
                    uid=uid, hotkey=hotkey[:6],
                    round_id=round_obj.round_id,
                    reason=fail_reason,
                )
                from connito.shared.telemetry import inc_error
                inc_error(component="foreground_eval", kind="validation")
                round_obj.mark_failed(uid)
                continue

            # Cap the per-miner eval at min(configured_timeout, time_to_end_block)
            # so a long-running eval can't itself overrun the phase boundary.
            sec_to_end_block = max(0.0, (end_block - block_now) * BITTENSOR_BLOCK_TIME_SECONDS)
            effective_timeout: float | None = sec_to_end_block
            if per_miner_eval_timeout_sec is not None:
                effective_timeout = min(per_miner_eval_timeout_sec, sec_to_end_block)
            if effective_timeout <= 0:
                round_obj.release_claim(uid)
                phase_deadline_crossed = True
                break

            eval_coro = evaluate_one_miner(
                config=config,
                model_path=job.model_path,
                uid=uid,
                hotkey=hotkey,
                base_model=base_model,
                tokenizer=tokenizer,
                combined_seed=round_obj.seed,
                device=device,
                score_aggregator=score_aggregator,
                baseline_loss=baseline_loss,
                step=step,
                round_id=round_obj.round_id,
                score_path=score_path,
            )
            try:
                evaluated = await asyncio.wait_for(eval_coro, timeout=effective_timeout)
            except asyncio.TimeoutError:
                logger.warning(
                    "foreground eval: per-miner timeout — marking failed",
                    uid=uid, hotkey=hotkey[:6],
                    timeout_sec=round(effective_timeout, 2),
                )
                round_obj.mark_failed(uid)
                continue
            except Exception as e:
                logger.exception("foreground eval: unexpected failure", uid=uid, error=str(e))
                round_obj.mark_failed(uid)
                continue

            if evaluated is None:
                round_obj.mark_failed(uid)
                continue
            round_obj.mark_scored(uid)
            completed.append(evaluated)

        # Stop once every top-N UID is scored or the phase boundary hits.
        scored_top_n = sum(1 for u in foreground_set if u in round_obj.scored_uids)
        if scored_top_n >= len(foreground_set):
            break

        if phase_deadline_crossed or subtensor.block > end_block:
            logger.info(
                "foreground eval: validate phase ended — stopping",
                round_id=round_obj.round_id,
                end_block=end_block,
                current_block=subtensor.block,
                scored=scored_top_n,
                foreground_total=len(foreground_set),
            )
            break
        if not progressed:
            await asyncio.sleep(poll_interval_sec)

    logger.info(
        "foreground eval: complete",
        round_id=round_obj.round_id,
        top_n=len(foreground_set),
        scored=len(completed),
        spilled=len(foreground_set) - len(completed),
    )
    return completed
