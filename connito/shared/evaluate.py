from __future__ import annotations

import gc
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from connito.shared.app_logging import structlog
from connito.shared.modeling.residual import record_gate_usage

logger = structlog.getLogger(__name__)

tqdm(disable=True, total=0)

# Positions per lm_head chunk: about 1 GiB of fp32 logits at Qwen's 248,320 vocab, as measured on the L40S floor.
_CE_CHUNK_TOKENS = 1024


def _chunk_ce(lm_head: nn.Module, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(lm_head(hidden).float(), labels, reduction="sum")


def chunked_lm_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Next-token cross-entropy averaged over labelled tokens, as `model(**batch).loss`, without full logits."""
    hidden = model.model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask")).last_hidden_state
    hidden = hidden[:, :-1].reshape(-1, hidden.shape[-1])
    labels = batch["labels"][:, 1:].reshape(-1)
    # Checkpointed, so backward keeps one chunk's logits at a time instead of the whole batch's.
    total = sum(
        checkpoint(_chunk_ce, model.lm_head, hidden[i : i + _CE_CHUNK_TOKENS], labels[i : i + _CE_CHUNK_TOKENS],
                   use_reentrant=False)
        for i in range(0, labels.numel(), _CE_CHUNK_TOKENS)
    )
    return total / labels.ne(-100).sum()


class EvalDeadlineExceeded(RuntimeError):
    """Raised by `evaluate_model` when its `deadline_monotonic` is crossed.

    Distinct from `asyncio.TimeoutError` so callers running the eval
    inside a thread (and therefore unable to be cancelled by
    `asyncio.wait_for`) can surface the deadline as a normal exception
    that unwinds locks via `with`/`finally`, instead of letting an
    awaiter cancellation orphan an in-flight GPU thread.
    """


def evaluate_model(
    step: int,
    model: nn.Module,
    eval_dataloader,
    device: torch.device,
    max_eval_batches: int | None = 50,
    rank: int | None = None,
    deadline_monotonic: float | None = None,
) -> dict[str, float]:
    """
    Run a lightweight eval pass and return scalar metrics.

    Parameters
    ----------
    step : int
        Training step for logging context.
    model : nn.Module
        Fully-assembled model placed on the correct device.
    eval_dataloader :
        Iterable of evaluation batches (dicts of Tensors).
    device : torch.device
        Device to run evaluation on.
    max_eval_batches : Optional[int]
        Optional cap on the number of batches to evaluate.

    Returns
    -------
    Dict[str, float]
        e.g., {"val_loss": 2.345, "gate_usage": 0.07}; `gate_usage` is the residual gate's mean Σq over real tokens.
    """
    model.to(device)
    model.eval()
    loss_sum: float = 0.0
    # Count of batches that produced a finite loss. The previous
    # implementation skipped NaN-loss batches from `loss_sum` but still
    # included them in the divisor, so a miner that crafts weights to
    # overflow logits to inf/NaN under bf16 autocast on a fraction `p` of
    # batches would report `(1 - p) * honest_loss` instead of
    # `honest_loss` — gaming the val_loss downward and inflating their
    # reward. Divide by `scored_batches` instead so a NaN/Inf batch
    # contributes nothing to either side of the ratio.
    scored_batches: int = 0
    nan_batches: int = 0
    batch_step = -1
    with torch.no_grad(), record_gate_usage(model) as usage:
        for batch_step, batch in enumerate(iterable=eval_dataloader):
            # Per-batch deadline check. Raised before we start GPU work
            # for this batch so the caller's `with lock:` unwinds without
            # leaving an in-flight allocation. Granularity is one batch —
            # the eval loop cannot interrupt mid-forward — so the
            # effective bound is `deadline + one_batch_wall_time`.
            if deadline_monotonic is not None and time.monotonic() > deadline_monotonic:
                raise EvalDeadlineExceeded(
                    f"evaluate_model deadline exceeded at batch={batch_step} "
                    f"step={step} scored_batches={scored_batches}"
                )
            device_batch = {}
            for key in batch.keys():
                device_batch[key] = batch[key].to(model.device)

            if device_batch.get("attention_mask") is None and "input_ids" in device_batch:
                device_batch["attention_mask"] = torch.ones_like(device_batch["input_ids"])
            usage.mask = device_batch["attention_mask"]

            autocast_device = "cuda" if device.type == "cuda" else "cpu"
            with torch.amp.autocast(autocast_device, dtype=torch.bfloat16):
                loss = chunked_lm_loss(model, device_batch)

            if torch.isnan(loss) or torch.isinf(loss):
                # NaN/Inf batches add nothing to the sum and do NOT increment
                # `scored_batches`, so they drop out of the divisor as well.
                nan_batches += 1
            else:
                loss_sum += float(loss.item())
                scored_batches += 1

            del device_batch, loss
            gc.collect()

            if max_eval_batches is not None and batch_step >= max_eval_batches:
                break

        # INFO, not debug: `nan_batches` is the only production-visible
        # signal distinguishing "miner scored on all batches" from
        # "miner's average silently excludes NaN batches" (degenerate
        # eval rows and overflow-on-OOD submissions both land here).
        # Ops must be able to read this off a live validator's logs.
        logger.info(
            "eval loss",
            loss_sum=round(loss_sum, 4),
            scored_batches=scored_batches,
            nan_batches=nan_batches,
            step=step,
        )

    # Every batch was NaN/Inf — the miner's checkpoint is malicious or
    # broken. Return `+inf` so the caller's `delta = max(0, baseline -
    # val_loss)` clamps to 0 (score=0). Returning 0.0 would have given
    # them maximum delta.
    if scored_batches == 0:
        if nan_batches > 0:
            logger.warning(
                "evaluate_model: every eval batch produced NaN/Inf loss",
                nan_batches=nan_batches, step=step,
            )
        return {
            "val_loss": float("inf"),
            "nan_batches": nan_batches,
            "scored_batches": 0,
            "gate_usage": usage.mean(),
        }

    return {
        "val_loss": loss_sum / scored_batches,
        "nan_batches": nan_batches,
        "scored_batches": scored_batches,
        "gate_usage": usage.mean(),
    }
