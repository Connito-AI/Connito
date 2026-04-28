from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

# Allow the CUDA allocator to use expandable memory segments so fragmented
# reserved-but-unallocated blocks don't cause spurious OOMs during AdamW
# moment buffer allocation. Must be set before any CUDA tensor is created.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _repo_imports() -> dict[str, Any]:
    from connito.shared.app_logging import configure_logging, structlog
    from connito.shared.checkpoint_helper import compile_full_state_dict_from_path, save_checkpoint
    from connito.shared.checkpoints import select_best_checkpoint
    from connito.shared.config import ValidatorConfig
    from connito.shared.dataloader import get_dataloader
    from connito.shared.evaluate import evaluate_model
    from connito.shared.expert_manager import ExpertManager
    from connito.shared.helper import get_model_hash
    from connito.shared.model import freeze_parameters
    from connito.shared.modeling.mycelia import get_base_model, get_base_tokenizer

    configure_logging()
    return {
        "ValidatorConfig": ValidatorConfig,
        "ExpertManager": ExpertManager,
        "compile_full_state_dict_from_path": compile_full_state_dict_from_path,
        "evaluate_model": evaluate_model,
        "freeze_parameters": freeze_parameters,
        "get_base_model": get_base_model,
        "get_base_tokenizer": get_base_tokenizer,
        "get_dataloader": get_dataloader,
        "get_model_hash": get_model_hash,
        "logger": structlog.get_logger(__name__),
        "save_checkpoint": save_checkpoint,
        "select_best_checkpoint": select_best_checkpoint,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose miner-vs-validator evaluation mismatches offline by replaying "
            "the validator eval path on a fixed seeded batch set."
        )
    )
    parser.add_argument("--config", required=True, help="Path to validator config.yaml")
    parser.add_argument(
        "--miner-checkpoint",
        help="Path to miner submission file (.pt) or checkpoint directory to evaluate",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        help=(
            "Optional validator baseline checkpoint directory/file. If omitted, the script uses the latest "
            "checkpoint under config.ckpt.checkpoint_path, falling back to the raw base model template."
        ),
    )
    parser.add_argument(
        "--seed",
        required=True,
        help="Combined validator seed used to rebuild the same streaming eval shard",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=50,
        help="Maximum batch index passed to evaluate_model; matches validator semantics (default: 50)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override, e.g. cuda:0 or cpu. Defaults to config.model.device when available.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write the full JSON report",
    )
    parser.add_argument(
        "--run-roundtrip-smoke",
        action="store_true",
        help=(
            "Run a one-batch optimizer step, save a checkpoint with the production helper, reload it, "
            "and compare losses on the same captured batches."
        ),
    )
    parser.add_argument(
        "--run-offline-cycle",
        action="store_true",
        help=(
            "Simulate an offline miner cycle: start from the baseline model, run a few local training steps, "
            "perform miner-local evaluation, save a checkpoint, reload it validator-style, and compare losses."
        ),
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=1,
        help=(
            "Number of local optimizer steps to run in offline-cycle mode. "
            "Use 0 for the fastest no-train save/reload/eval parity pass (default: 1)"
        ),
    )
    parser.add_argument(
        "--train-seed",
        default=None,
        help=(
            "Optional seed for the miner training/local-eval dataloader in offline-cycle mode. "
            "If omitted, the miner loader uses its default unseeded behavior."
        ),
    )
    parser.add_argument(
        "--train-optimizer",
        choices=("sgd", "adamw"),
        default="sgd",
        help=(
            "Optimizer used in offline-cycle mode. 'adamw' is closer to miner training but may OOM on large models; "
            "'sgd' is the lower-memory default for parity debugging."
        ),
    )
    parser.add_argument(
        "--use-lr-schedule",
        action="store_true",
        help=(
            "Attach a cosine-warmup LR schedule to the offline training loop (matches real miner training). "
            "Reads warmup_steps / total_steps from config.sched."
        ),
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping max-norm for the offline training loop (default: 1.0; set 0 to disable).",
    )
    parser.add_argument(
        "--save-trained-checkpoint",
        default=None,
        help=(
            "If set, the checkpoint produced by --run-offline-cycle is written to this directory so it "
            "can be passed back as --miner-checkpoint for a separate validator-style evaluation run."
        ),
    )
    return parser.parse_args()


def _resolve_device(config: Any, requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)

    configured = getattr(config.model, "device", None)
    if configured:
        if configured.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("Configured CUDA device unavailable; falling back to CPU", configured_device=configured)
            return torch.device("cpu")
        return torch.device(configured)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_config_read_only(config_path: str) -> Any:
    repo = _repo_imports()
    with open(config_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return repo["ValidatorConfig"](**data)


def _set_model_device_attr(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    try:
        setattr(model, "device", device)
    except AttributeError:
        pass
    return model


def _clone_batch_to_cpu(batch: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().cpu().clone()
        else:
            cloned[key] = value
    return cloned


def _capture_batches(dataloader, max_eval_batches: int) -> list[dict[str, Any]]:
    target_batches = max_eval_batches + 1
    captured: list[dict[str, Any]] = []
    for batch in dataloader:
        captured.append(_clone_batch_to_cpu(batch))
        if len(captured) >= target_batches:
            break

    if not captured:
        raise RuntimeError("Failed to capture any batches from the dataloader")

    return captured


def _capture_batches_from_iterator(batch_iter, max_eval_batches: int) -> list[dict[str, Any]]:
    target_batches = max_eval_batches + 1
    captured: list[dict[str, Any]] = []
    while len(captured) < target_batches:
        try:
            batch = next(batch_iter)
        except StopIteration as exc:
            if not captured:
                raise RuntimeError("Failed to capture any batches from iterator") from exc
            break
        captured.append(_clone_batch_to_cpu(batch))

    return captured


def _extract_state_dict(checkpoint_path: Path, expert_group_id: int) -> tuple[dict[str, torch.Tensor], str]:
    repo = _repo_imports()
    if checkpoint_path.is_file():
        raw = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(raw, dict) and "model_state_dict" in raw and isinstance(raw["model_state_dict"], dict):
            return raw["model_state_dict"], "file:model_state_dict"
        if isinstance(raw, dict):
            return raw, "file:raw_state_dict"
        raise ValueError(f"Unsupported checkpoint file format: {checkpoint_path}")

    if checkpoint_path.is_dir():
        return (
            repo["compile_full_state_dict_from_path"](checkpoint_path, expert_groups=[expert_group_id, "shared"]),
            "directory:compiled",
        )

    raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")


def _state_dict_compatibility(base_model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    base_sd = base_model.state_dict()
    base_keys = set(base_sd.keys())
    incoming_keys = set(state_dict.keys())
    common_keys = base_keys & incoming_keys
    common_same_shape = {key for key in common_keys if base_sd[key].shape == state_dict[key].shape}
    incompatible = base_model.load_state_dict(state_dict, strict=False)

    return {
        "incoming_key_count": len(incoming_keys),
        "base_key_count": len(base_keys),
        "common_key_count": len(common_keys),
        "common_same_shape_count": len(common_same_shape),
        "missing_key_count": len(incompatible.missing_keys),
        "unexpected_key_count": len(incompatible.unexpected_keys),
        "shape_mismatch_count": len(common_keys - common_same_shape),
        "sample_missing_keys": list(incompatible.missing_keys[:10]),
        "sample_unexpected_keys": list(incompatible.unexpected_keys[:10]),
        "sample_shape_mismatches": sorted(common_keys - common_same_shape)[:10],
    }


def _load_model_for_eval(
    base_template: torch.nn.Module,
    checkpoint_path: Path,
    expert_group_id: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    repo = _repo_imports()
    state_dict, source = _extract_state_dict(checkpoint_path, expert_group_id)
    model = copy.deepcopy(base_template)
    compatibility = _state_dict_compatibility(model, state_dict)
    return model, {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_source": source,
        "model_hash": repo["get_model_hash"](model.state_dict(), hex=True),
        **compatibility,
    }


def _move_model_to_cpu(model: torch.nn.Module) -> torch.nn.Module:
    model = model.to(torch.device("cpu"))
    return _set_model_device_attr(model, torch.device("cpu"))


def _precision_settings(config: Any, device: torch.device) -> tuple[bool, torch.dtype, str]:
    precision = getattr(config.model, "precision", "fp16-mixed")
    if precision == "bf16-mixed" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        precision = "fp16-mixed"
    autocast_enabled = device.type == "cuda" and precision in ("fp16-mixed", "bf16-mixed")
    autocast_dtype = torch.float16 if precision == "fp16-mixed" else torch.bfloat16
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    return autocast_enabled, autocast_dtype, autocast_device


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _evaluate_fixed_batches(
    label: str,
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
    device: torch.device,
    max_eval_batches: int,
) -> dict[str, Any]:
    repo = _repo_imports()
    model = _set_model_device_attr(model.to(device), device)
    metrics = repo["evaluate_model"](
        step=0,
        model=model,
        eval_dataloader=batches,
        device=device,
        max_eval_batches=max_eval_batches,
        rank=0,
    )
    result = {
        "label": label,
        "val_loss": float(metrics["val_loss"]),
        "val_aux_loss": float(metrics.get("val_aux_loss", 0.0)),
        "model_hash": repo["get_model_hash"](model.state_dict(), hex=True),
    }
    return result


def _default_baseline_path(config: Any) -> Path | None:
    latest = _repo_imports()["select_best_checkpoint"](primary_dir=config.ckpt.checkpoint_path)
    if latest is None or latest.path is None:
        return None
    return Path(latest.path)


def _release_model(model: torch.nn.Module | None) -> None:
    if model is None:
        return
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_roundtrip_smoke(
    config: Any,
    expert_manager: Any,
    base_template: torch.nn.Module,
    batches: list[dict[str, Any]],
    device: torch.device,
    max_eval_batches: int,
) -> dict[str, Any]:
    repo = _repo_imports()
    model = copy.deepcopy(base_template)
    model = repo["freeze_parameters"](
        model=model,
        expert_manager=expert_manager,
        expert_group_id=config.task.exp.group_id,
        upcast_trainable=True,
    )
    model = _set_model_device_attr(model.to(device), device)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        return {"skipped": True, "reason": "No trainable parameters found for the active expert group"}

    optimizer = torch.optim.AdamW(trainable_params, lr=config.opt.lr, weight_decay=0.1, betas=(0.9, 0.95))
    train_batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batches[0].items()}

    autocast_enabled, autocast_dtype, autocast_device = _precision_settings(config, device)

    model.train()
    optimizer.zero_grad()
    with torch.amp.autocast(autocast_device, enabled=autocast_enabled, dtype=autocast_dtype):
        outputs = model(**train_batch)
        loss = outputs.loss.float()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    in_memory_eval = _evaluate_fixed_batches(
        label="roundtrip_in_memory",
        model=model,
        batches=batches,
        device=device,
        max_eval_batches=max_eval_batches,
    )

    with tempfile.TemporaryDirectory(prefix="connito-roundtrip-") as tmpdir:
        ckpt_dir = Path(tmpdir) / "checkpoint"
        repo["save_checkpoint"](
            checkpoint_path=ckpt_dir,
            model=model,
            rank=0,
            loss=float(loss.item()),
            save_global_state=False,
            save_model_by_expert_group=True,
            expert_manager=expert_manager,
            active_expert_group_id=config.task.exp.group_id,
        )
        reloaded_model, load_summary = _load_model_for_eval(
            base_template=base_template,
            checkpoint_path=ckpt_dir,
            expert_group_id=config.task.exp.group_id,
            device=device,
        )
        reloaded_eval = _evaluate_fixed_batches(
            label="roundtrip_reloaded",
            model=reloaded_model,
            batches=batches,
            device=device,
            max_eval_batches=max_eval_batches,
        )

    delta = abs(in_memory_eval["val_loss"] - reloaded_eval["val_loss"])
    _release_model(reloaded_model)
    _release_model(model)

    return {
        "skipped": False,
        "train_loss": float(loss.item()),
        "in_memory_eval": in_memory_eval,
        "reloaded_eval": reloaded_eval,
        "reload_summary": load_summary,
        "eval_loss_abs_delta": delta,
    }


def _run_offline_cycle(
    config: Any,
    expert_manager: Any,
    tokenizer: Any,
    base_training_model: torch.nn.Module,
    validator_eval_batches: list[dict[str, Any]],
    validator_baseline_eval: dict[str, Any],
    device: torch.device,
    max_eval_batches: int,
    train_steps: int,
    train_seed: str | None,
    train_optimizer: str,
    save_checkpoint_path: str | None = None,
    use_lr_schedule: bool = False,
    grad_clip: float = 1.0,
) -> dict[str, Any]:
    repo = _repo_imports()
    if train_steps < 0:
        return {"skipped": True, "reason": "train_steps must be >= 0"}

    # Flush allocator so any cached eval-phase CUDA memory is returned before
    # the training model + AdamW moment buffers are allocated.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = copy.deepcopy(base_training_model)
    model = repo["freeze_parameters"](
        model=model,
        expert_manager=expert_manager,
        expert_group_id=config.task.exp.group_id,
        # upcast_trainable=True (fp32 master weights) is what the real miner uses with
        # AdamW, but it costs ~6 GB extra VRAM for fp32 params + moments + grads.
        # On a validator GPU that already holds the 38 GB baseline model there is no room.
        # We get AdamW's adaptive-LR benefit (the key convergence difference vs SGD) without
        # fp32 upcasting; GradScaler below compensates for fp16 gradient underflow.
        upcast_trainable=False,
    )
    model = _set_model_device_attr(model.to(device), device)

    # Free any fragmented CUDA allocator cache now that the model is resident,
    # so AdamW moment buffer allocation has the maximum contiguous room.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        _release_model(model)
        return {"skipped": True, "reason": "No trainable parameters found for the active expert group"}

    if train_optimizer == "adamw":
        # foreach=False avoids the _foreach_sqrt batched-temp-buffer allocation that
        # OOMs on a 40 GB GPU already holding the 38 GB partial model.  The update rule
        # and convergence behaviour are identical; only which CUDA kernel is dispatched
        # changes (scalar ops instead of batched ops).
        optimizer = torch.optim.AdamW(
            trainable_params, lr=config.opt.lr, weight_decay=0.1, betas=(0.9, 0.95), foreach=False
        )
    else:
        optimizer = torch.optim.SGD(trainable_params, lr=config.opt.lr, momentum=0.0)

    scheduler = None
    if use_lr_schedule and train_steps > 0:
        try:
            from transformers import get_cosine_schedule_with_warmup
            warmup_steps = int(getattr(getattr(config, "sched", None), "warmup_steps", 10) or 10)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=min(warmup_steps, train_steps),
                num_training_steps=train_steps,
            )
            logger.info("offline_cycle: LR schedule enabled", warmup_steps=warmup_steps, total_steps=train_steps)
        except ImportError:
            logger.warning("offline_cycle: transformers not available, skipping LR schedule")

    train_loader = repo["get_dataloader"](
        config=config,
        tokenizer=tokenizer,
        seed=train_seed,
        rank=0,
        world_size=config.task.exp.data.world_size,
    )
    train_iter = iter(train_loader)
    autocast_enabled, autocast_dtype, autocast_device = _precision_settings(config, device)

    # GradScaler: only meaningful for fp16-mixed + AdamW (mirrors production miner).
    # SGD path skips it — SGD doesn't need fp32 upcast or scaling.
    scaler_enabled = (
        train_optimizer == "adamw"
        and autocast_enabled
        and getattr(config.model, "precision", "") == "fp16-mixed"
    )
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    logger.info(
        "offline_cycle: optimizer ready",
        optimizer=train_optimizer,
        lr=config.opt.lr,
        scaler_enabled=scaler_enabled,
        upcast_trainable=(train_optimizer == "adamw"),
    )

    train_losses: list[float] = []
    trained_batches = 0
    model.train()
    for _ in range(train_steps):
        try:
            train_batch = next(train_iter)
        except StopIteration:
            break

        batch_device = _batch_to_device(train_batch, device)
        optimizer.zero_grad()
        with torch.amp.autocast(autocast_device, enabled=autocast_enabled, dtype=autocast_dtype):
            outputs = model(**batch_device)
            loss = outputs.loss

        if not torch.isfinite(loss):
            logger.warning("offline_cycle: non-finite loss, skipping batch", step=trained_batches + 1)
            del batch_device, outputs, loss
            continue

        if scaler_enabled:
            # Production miner path: scale → backward → unscale → clip → step
            scaler.scale(loss).backward()
            del outputs, batch_device  # free activations before optimizer allocs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = loss.float()
            loss.backward()
            del outputs, batch_device
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        train_losses.append(float(loss.float().item()))
        trained_batches += 1
        if trained_batches == 1 or trained_batches % 10 == 0:
            window = train_losses[-10:]
            logger.info(
                "offline_cycle training",
                step=trained_batches,
                loss=round(train_losses[-1], 4),
                rolling_avg_10=round(sum(window) / len(window), 4),
            )
        del batch_device, outputs, loss
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if train_steps > 0 and trained_batches == 0:
        _release_model(model)
        return {"skipped": True, "reason": "Train dataloader produced no batches"}

    miner_local_eval_batches = _capture_batches_from_iterator(train_iter, max_eval_batches=max_eval_batches)
    if not miner_local_eval_batches:
        _release_model(model)
        return {"skipped": True, "reason": "Could not capture miner local-eval batches after training"}

    in_memory_hash = repo["get_model_hash"](model.state_dict(), hex=True)
    miner_local_in_memory = _evaluate_fixed_batches(
        label="miner_local_in_memory",
        model=model,
        batches=miner_local_eval_batches,
        device=device,
        max_eval_batches=max_eval_batches,
    )
    validator_eval_in_memory = _evaluate_fixed_batches(
        label="validator_eval_in_memory",
        model=model,
        batches=validator_eval_batches,
        device=device,
        max_eval_batches=max_eval_batches,
    )

    with tempfile.TemporaryDirectory(prefix="connito-offline-cycle-") as tmpdir:
        ckpt_dir = Path(tmpdir) / "checkpoint"
        checkpoint_loss = train_losses[-1] if train_losses else 0.0
        repo["save_checkpoint"](
            checkpoint_path=ckpt_dir,
            model=model,
            rank=0,
            loss=checkpoint_loss,
            save_global_state=False,
            save_model_by_expert_group=True,
            expert_manager=expert_manager,
            active_expert_group_id=config.task.exp.group_id,
        )
        saved_state = repo["compile_full_state_dict_from_path"](
            ckpt_dir,
            expert_groups=[config.task.exp.group_id, "shared"],
        )
        saved_hash = repo["get_model_hash"](saved_state, hex=True)

        # Persist to disk if requested so the caller can later pass it as
        # --miner-checkpoint for a standalone validator-style evaluation.
        persistent_ckpt_path: str | None = None
        if save_checkpoint_path:
            import shutil
            dest = Path(save_checkpoint_path)
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copytree(str(ckpt_dir), str(dest), dirs_exist_ok=True)
            persistent_ckpt_path = str(dest)
            logger.info("offline_cycle: checkpoint saved", path=persistent_ckpt_path)

        reloaded_model, reload_summary = _load_model_for_eval(
            base_template=base_training_model,
            checkpoint_path=ckpt_dir,
            expert_group_id=config.task.exp.group_id,
            device=device,
        )

    miner_local_reloaded = _evaluate_fixed_batches(
        label="miner_local_reloaded",
        model=reloaded_model,
        batches=miner_local_eval_batches,
        device=device,
        max_eval_batches=max_eval_batches,
    )
    validator_eval_reloaded = _evaluate_fixed_batches(
        label="validator_eval_reloaded",
        model=reloaded_model,
        batches=validator_eval_batches,
        device=device,
        max_eval_batches=max_eval_batches,
    )
    reloaded_hash = repo["get_model_hash"](reloaded_model.state_dict(), hex=True)

    result = {
        "skipped": False,
        "train_steps_requested": train_steps,
        "train_steps_completed": trained_batches,
        "train_seed": train_seed,
        "train_optimizer": train_optimizer,
        "use_lr_schedule": use_lr_schedule,
        "grad_clip": grad_clip,
        "train_losses": train_losses,
        "train_loss_mean": round(sum(train_losses) / len(train_losses), 6) if train_losses else None,
        "train_loss_final": round(train_losses[-1], 6) if train_losses else None,
        "saved_checkpoint_path": persistent_ckpt_path,
        "training_applied": trained_batches > 0,
        "miner_local_eval_batch_count": len(miner_local_eval_batches),
        "validator_eval_batch_count": len(validator_eval_batches),
        "in_memory_model_hash": in_memory_hash,
        "saved_checkpoint_hash": saved_hash,
        "reloaded_model_hash": reloaded_hash,
        "miner_local_in_memory": miner_local_in_memory,
        "validator_eval_in_memory": validator_eval_in_memory,
        "miner_local_reloaded": miner_local_reloaded,
        "validator_eval_reloaded": validator_eval_reloaded,
        "validator_baseline_eval": validator_baseline_eval,
        "reload_summary": reload_summary,
        "discrepancy": {
            "save_reload_delta_on_local_batches": abs(
                miner_local_in_memory["val_loss"] - miner_local_reloaded["val_loss"]
            ),
            "save_reload_delta_on_validator_batches": abs(
                validator_eval_in_memory["val_loss"] - validator_eval_reloaded["val_loss"]
            ),
            "local_vs_validator_in_memory_delta": (
                miner_local_in_memory["val_loss"] - validator_eval_in_memory["val_loss"]
            ),
            "local_vs_validator_reloaded_delta": (
                miner_local_reloaded["val_loss"] - validator_eval_reloaded["val_loss"]
            ),
            "validator_score_from_reloaded": max(
                0.0,
                validator_baseline_eval["val_loss"] - validator_eval_reloaded["val_loss"],
            ) ** 1.2,
        },
    }

    _release_model(reloaded_model)
    _release_model(model)
    return result


def main() -> int:
    args = parse_args()
    if not args.miner_checkpoint and not args.run_offline_cycle:
        raise SystemExit("Provide --miner-checkpoint and/or enable --run-offline-cycle")

    repo = _repo_imports()
    logger = repo["logger"]
    config = _load_config_read_only(args.config)
    device = _resolve_device(config, args.device)

    expert_manager = repo["ExpertManager"](config)
    tokenizer = repo["get_base_tokenizer"](config)
    base_template = repo["get_base_model"](
        config=config,
        expert_manager=expert_manager,
        group_ids=[config.task.exp.group_id],
        partial=True,
    )
    if base_template is None:
        raise RuntimeError("Failed to build base model template")
    base_template = _set_model_device_attr(base_template, torch.device("cpu"))

    dataloader = repo["get_dataloader"](
        config=config,
        tokenizer=tokenizer,
        seed=args.seed,
        rank=0,
        world_size=config.dataloader.world_size,
    )
    batches = _capture_batches(dataloader, max_eval_batches=args.max_eval_batches)

    miner_checkpoint_path = Path(args.miner_checkpoint).expanduser().resolve() if args.miner_checkpoint else None
    baseline_checkpoint_path = (
        Path(args.baseline_checkpoint).expanduser().resolve()
        if args.baseline_checkpoint
        else _default_baseline_path(config)
    )

    base_template_eval = _evaluate_fixed_batches(
        label="base_template",
        model=copy.deepcopy(base_template),
        batches=batches,
        device=device,
        max_eval_batches=args.max_eval_batches,
    )

    if baseline_checkpoint_path is not None:
        baseline_model, baseline_load = _load_model_for_eval(
            base_template=base_template,
            checkpoint_path=baseline_checkpoint_path,
            expert_group_id=config.task.exp.group_id,
            device=device,
        )
        baseline_eval = _evaluate_fixed_batches(
            label="baseline_checkpoint",
            model=baseline_model,
            batches=batches,
            device=device,
            max_eval_batches=args.max_eval_batches,
        )
        miner_overlay_base = _move_model_to_cpu(baseline_model)
    else:
        baseline_load = None
        baseline_eval = base_template_eval
        miner_overlay_base = base_template

    miner_load = None
    miner_eval = None
    scoring = None
    if miner_checkpoint_path is not None:
        miner_model, miner_load = _load_model_for_eval(
            base_template=miner_overlay_base,
            checkpoint_path=miner_checkpoint_path,
            expert_group_id=config.task.exp.group_id,
            device=device,
        )
        miner_eval = _evaluate_fixed_batches(
            label="miner_checkpoint",
            model=miner_model,
            batches=batches,
            device=device,
            max_eval_batches=args.max_eval_batches,
        )
        _release_model(miner_model)

        baseline_loss = baseline_eval["val_loss"]
        miner_loss = miner_eval["val_loss"]
        delta = max(0.0, baseline_loss - miner_loss)
        score = delta ** 1.2
        scoring = {
            "baseline_loss": baseline_loss,
            "miner_loss": miner_loss,
            "delta": delta,
            "score": score,
            "beats_baseline": miner_loss < baseline_loss,
        }

    offline_cycle = None
    if args.run_offline_cycle:
        # Flush the CUDA allocator cache so eval-phase memory is fully reclaimed
        # before the training model + AdamW moments are allocated.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        offline_cycle = _run_offline_cycle(
            config=config,
            expert_manager=expert_manager,
            tokenizer=tokenizer,
            base_training_model=miner_overlay_base,
            validator_eval_batches=batches,
            validator_baseline_eval=baseline_eval,
            device=device,
            max_eval_batches=args.max_eval_batches,
            train_steps=args.train_steps,
            train_seed=args.train_seed,
            train_optimizer=args.train_optimizer,
            save_checkpoint_path=args.save_trained_checkpoint,
            use_lr_schedule=args.use_lr_schedule,
            grad_clip=args.grad_clip,
        )

    if baseline_checkpoint_path is not None:
        _release_model(miner_overlay_base)

    result: dict[str, Any] = {
        "config_path": str(Path(args.config).expanduser().resolve()),
        "device": str(device),
        "expert_group_id": config.task.exp.group_id,
        "seed": args.seed,
        "captured_batch_count": len(batches),
        "max_eval_batches_argument": args.max_eval_batches,
        "base_template_eval": base_template_eval,
        "baseline_eval": baseline_eval,
        "baseline_load": baseline_load,
        "miner_eval": miner_eval,
        "miner_load": miner_load,
        "scoring": scoring,
        "offline_cycle": offline_cycle,
    }

    if args.run_roundtrip_smoke:
        result["roundtrip_smoke"] = _run_roundtrip_smoke(
            config=config,
            expert_manager=expert_manager,
            base_template=base_template,
            batches=batches,
            device=device,
            max_eval_batches=args.max_eval_batches,
        )

    report = json.dumps(result, indent=2, sort_keys=False)
    print(report)

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report + "\n", encoding="utf-8")
        logger.info("Wrote diagnosis report", path=str(output_path))

    _release_model(base_template)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())