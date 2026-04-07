from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass

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


async def evaluator_worker(
    name: str,
    config,
    jobs_q: asyncio.Queue[MinerEvalJob],
    aggregator: MinerScoreAggregator,
    device: torch.device,
    base_model: nn.Module,
    tokenizer,
    combinded_seed: str,
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

            eval_dataloader = await asyncio.to_thread(
                get_dataloader,
                config=config,
                tokenizer=tokenizer,
                seed=combinded_seed,
                rank=0,
                world_size=config.dataloader.world_size,
            )

            @track_eval_latency()
            def run_eval_with_telemetry():
                return evaluate_model(job.step, model, eval_dataloader, device, max_eval_batches, rank)

            metrics = await asyncio.to_thread(run_eval_with_telemetry)

            # choose a primary score (here 'accuracy'); adjust if your evaluate_model returns other keys
            score = float(metrics.get("val_loss", 100))
            aggregator.add_score(job.uid, job.hotkey, score)
            logger.info(
                f"{name}: evaluation complete",
                uid=job.uid,
                hotkey=job.hotkey[:6],
                val_loss=round(score, 4),
            )

            # Explicit cleanup
            del eval_dataloader, model, metrics
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
    # Device & dataloader (MOCK). Replace eval_dataloader with a real one.
    miners_q: asyncio.Queue[MinerEvalJob] = asyncio.Queue()

    # Enqueue miners
    for m in miners:
        await miners_q.put(m)

    # Spin up evaluator workers
    eval_workers = [
        asyncio.create_task(
            evaluator_worker(
                f"evaluator-{i+1}", config, miners_q, score_aggregator, device, base_model, tokenizer, combinded_seed
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
