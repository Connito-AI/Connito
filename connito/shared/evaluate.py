from __future__ import annotations

import gc

import torch
from torch import nn
from tqdm import tqdm

from connito.shared.app_logging import structlog

logger = structlog.getLogger(__name__)

tqdm(disable=True, total=0)


def evaluate_model(
    step: int,
    model: nn.Module,
    eval_dataloader,
    device: torch.device,
    max_eval_batches: int | None = 50,
    rank: int | None = None,
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
        e.g., {"val_loss": 2.345}
    """
    model.to(device)
    model.eval()
    loss_sum: float = 0.0
    aux_loss_sum: float = 0.0
    # Count of batches that produced a finite loss. The previous
    # implementation skipped NaN-loss batches from `loss_sum` but still
    # included them in the divisor, so a miner that crafts weights to
    # overflow logits to inf/NaN under bf16 autocast on a fraction `p` of
    # batches would report `(1 - p) * honest_loss` instead of
    # `honest_loss` — gaming the val_loss downward and inflating their
    # reward. Divide by `scored_batches` instead so a NaN/Inf batch
    # contributes nothing to either side of the ratio. Also skip
    # `aux_loss_sum` on NaN/Inf batches so a related variant (subtract
    # aux_loss but skip loss) cannot replicate the same exploit.
    scored_batches: int = 0
    nan_batches: int = 0
    batch_step = -1
    with torch.no_grad():
        for batch_step, batch in enumerate(iterable=eval_dataloader):
            device_batch = {}
            for key in batch.keys():
                device_batch[key] = batch[key].to(model.device)

            if device_batch.get("attention_mask") is None and "input_ids" in device_batch:
                device_batch["attention_mask"] = torch.ones_like(device_batch["input_ids"])

            autocast_device = "cuda" if device.type == "cuda" else "cpu"
            eval_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            with torch.amp.autocast(autocast_device, dtype=eval_dtype):
                outputs = model(**device_batch)

                if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
                    # NaN/Inf batches contribute 0 to both sums and do
                    # NOT increment `scored_batches`, so they drop out
                    # of the divisor as well. Explicit no-op `+= 0`
                    # keeps the parallel structure with the else-branch.
                    loss_sum += 0.0
                    aux_loss_sum += 0.0
                    nan_batches += 1
                else:
                    loss_sum += float(outputs.loss.item())
                    aux_loss_sum += (
                        float(outputs.aux_loss.item())
                        if hasattr(outputs, "aux_loss") and outputs.aux_loss is not None
                        else 0.0
                    )
                    scored_batches += 1

            del device_batch, outputs
            gc.collect()

            if max_eval_batches is not None and batch_step >= max_eval_batches:
                break

        logger.debug(
            "eval loss",
            loss_sum=round(loss_sum, 4),
            aux_loss_sum=round(aux_loss_sum, 4),
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
            "val_aux_loss": 0.0,
            "nan_batches": nan_batches,
            "scored_batches": 0,
        }

    if nan_batches > 0:
        logger.warning(
            "evaluate_model: NaN/Inf batches dropped from loss avg",
            nan_batches=nan_batches,
            scored_batches=scored_batches,
            step=step,
        )

    val_loss = (loss_sum - aux_loss_sum) / scored_batches
    val_aux_loss = aux_loss_sum / scored_batches
    return {
        "val_loss": val_loss,
        "val_aux_loss": val_aux_loss,
        "nan_batches": nan_batches,
        "scored_batches": scored_batches,
    }
