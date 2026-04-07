import gc
import logging
from collections.abc import Iterable, Sequence
from typing import Any

import torch

from connito.shared.config import MinerConfig, ValidatorConfig
from connito.shared.expert_manager import get_weight_sum

# Configure the basic logging setup
logger = logging.getLogger(__name__)


def free_cuda_models(
    models: torch.nn.Module | Iterable[torch.nn.Module],
    optimizers: Iterable[torch.optim.Optimizer] | None = None,
    devices: Sequence[int | str | torch.device] | None = None,
    move_to_cpu_first: bool = True,
    clear_ipc: bool = True,
    sync_before: bool = True,
):
    """
    Release GPU memory held by models/optimizers.

    What it does:
      1) (optional) synchronize CUDA to let queued work finish
      2) move models to CPU (optional but recommended)
      3) drop grads and optimizer state
      4) delete references and run GC
      5) empty CUDA caching allocator (and IPC cache if requested)

    Args:
      models: A model or iterable of models to free.
      optimizers: Any optimizers associated with those models. If provided,
                  their per-param state will be freed.
      devices: CUDA devices to flush caches for (defaults to all visible or the
               current device if none is set).
      move_to_cpu_first: Move models to CPU before deletion (avoids device
                         context surprises and async frees).
      clear_ipc: Also clear CUDA IPC allocations (relevant with multi-process).
      sync_before: Call torch.cuda.synchronize() before freeing (safer when you
                   just launched kernels that touch these tensors).

    Notes:
      - torch.cuda.empty_cache() only returns *unused cached blocks* to the driver.
        Any still-referenced tensor keeps its memory. That’s why we delete refs first.
      - If you used torch.compile, you may also wish to reset compile caches:
            import torch._dynamo as dynamo; dynamo.reset()
        (This is optional and uses internal APIs; only do it if you know you need it.)
    """

    logger.info("Free cuda models start")
    # Normalize inputs
    if isinstance(models, torch.nn.Module):
        models = [models]
    models = list(models)
    optimizers = list(optimizers) if optimizers is not None else []

    # 1) Sync so we don't free buffers that are still in use by queued kernels
    if sync_before and torch.cuda.is_available():
        try:
            if devices is None:
                torch.cuda.synchronize()  # current device (or all streams on it)
            else:
                for d in devices:
                    torch.cuda.synchronize(d)
        except Exception as e:
            logger.warning("CUDA sync failed during cleanup", error=str(e))

    # 2) Move models to CPU (helps ensure param storages detach from CUDA)
    if move_to_cpu_first:
        for m in models:
            try:
                m.cpu()
            except Exception as e:
                logger.warning("Failed to move model to CPU during cleanup", error=str(e))

    # 3) Clear grads and optimizer state
    for m in models:
        try:
            for p in m.parameters():
                p.grad = None
        except Exception as e:
            logger.warning("Failed to clear gradients during cleanup", error=str(e))

    for opt in optimizers:
        try:
            # Clear per-param state (exp_avg, exp_avg_sq, momentum buffers, etc.)
            opt.state.clear()
        except Exception as e:
            logger.warning("Failed to clear optimizer state during cleanup", error=str(e))

    # 4) Drop strong references and collect
    #    (Caller is expected to drop their own references too!)
    del models
    del optimizers
    gc.collect()

    # 5) Return unused cached blocks to the driver (and clear IPC if requested)
    if torch.cuda.is_available():
        try:
            if devices is None:
                torch.cuda.empty_cache()
                if clear_ipc:
                    torch.cuda.ipc_collect()
            else:
                for d in devices:
                    with torch.cuda.device(d):
                        torch.cuda.empty_cache()
                        if clear_ipc:
                            torch.cuda.ipc_collect()
        except Exception as e:
            logger.warning("Failed to empty CUDA cache during cleanup", error=str(e))

    logger.info("Free cuda models complete")


def get_status(
    config: MinerConfig | ValidatorConfig,
    model: torch.nn.Module,
    step: int,
    training_time: float,
    total_training_time: float,
    inner_optimizer: torch.optim.Optimizer | None = None,
    inner_opt_step: int | None = None,
    global_opt_step: int | None = None,
    loss_batch: torch.Tensor | None | Any = None,
    aux_loss_batch: torch.Tensor | None | Any = None,
) -> dict[str, Any]:
    """
    Build a dictionary of training metrics for monitoring and logging.
    Times are reported in hours, throughput in tokens/second.
    """

    if inner_opt_step is None and global_opt_step is None:
        raise ValueError("At least one of inner_opt_step or global_opt_step must be provided.")

    total_batch_size: int = 0
    if inner_opt_step is not None:
        total_batch_size = config.task.exp.data.batch_size * config.local_par.world_size
        total_samples = inner_opt_step * total_batch_size
        total_tokens = total_samples * config.task.exp.data.sequence_length

    _, expert_sum = get_weight_sum(model, shared=False)
    if not torch.isfinite(expert_sum):
        logger.warning("Non-finite expert parameter sum detected in get_status; substituting 0.0", step=step)
        expert_sum = torch.tensor(0.0, dtype=torch.float32, device=expert_sum.device)

    expert_sum_value = float(expert_sum.detach().cpu().item())

    # Extract current learning rate (assume one param group or take first)

    metrics: dict[str, Any] = {
        "step": step,
        "inner_opt_step": inner_opt_step,
        "global_opt_step": global_opt_step,
        "lr": (
            next(iter(group["lr"] for group in inner_optimizer.param_groups)) if inner_optimizer is not None else None
        ),
        "param_sum": expert_sum_value,
    }

    if inner_opt_step is not None:
        metrics = metrics | {
            "total_samples": total_samples,
            "total_tokens": total_tokens,
        }

    if training_time > 0 and total_training_time > 0:
        metrics = metrics | {
            "inner_step_time_hours": training_time / 3600,
            "total_training_time_hours": total_training_time / 3600,
        }
        if inner_opt_step is not None:
            metrics = metrics | {
                "tokens_per_second": (config.task.exp.data.sequence_length * total_batch_size) / training_time,
            }


    if loss_batch is not None and loss_batch != 0:
        if aux_loss_batch is not None and aux_loss_batch != 0:
            metrics = metrics | {
                "loss": float(loss_batch.item() - aux_loss_batch.item()),
                "perplexity": float(torch.exp(loss_batch - aux_loss_batch).item()),
                "aux_loss": float(aux_loss_batch.item()),
            }
        else:
            metrics = metrics | {
                "loss": float(loss_batch.item()),
                "perplexity": float(torch.exp(loss_batch).item()),
            }
            
    # Injection: System/Training Tracking Details
    if hasattr(config, "task") and hasattr(config.task, "expert_group_name"):
        metrics["expert_group_name"] = config.task.expert_group_name
        
    if hasattr(config, "local_par") and hasattr(config.local_par, "gradient_accumulation_steps"):
        metrics["gradient_accumulation_steps"] = config.local_par.gradient_accumulation_steps
        
    # Attempt to calculate simple grad norm on the fly safely 
    try:
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        metrics["grad_norm"] = total_norm ** 0.5
    except Exception as e:
        pass
        
    return metrics

