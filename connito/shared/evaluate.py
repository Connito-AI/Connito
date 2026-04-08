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

                if not torch.isnan(outputs.loss):
                    loss_sum += float(outputs.loss.item())

                aux_loss_sum += (
                    float(outputs.aux_loss.item())
                    if hasattr(outputs, "aux_loss") and outputs.aux_loss is not None
                    else 0
                )

            del device_batch, outputs
            gc.collect()

            if max_eval_batches is not None and batch_step >= max_eval_batches:
                break

        logger.debug(
            "eval loss",
            loss_sum=round(loss_sum, 4),
            aux_loss_sum=round(aux_loss_sum, 4),
            batches=batch_step,
            step=step,
        )
    
    # Avoid zero division if dataloader was empty or evaluated only 1 batch (batch_step is 0-indexed)
    total_batches = batch_step + 1 if batch_step is not None else 1
    return {"val_loss": (loss_sum - aux_loss_sum) / total_batches, "val_aux_loss": aux_loss_sum / total_batches}
