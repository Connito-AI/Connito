from __future__ import annotations

import re
import time
import traceback
from pathlib import Path

import bittensor
import torch
from torch import nn

from connito.shared.app_logging import structlog
from connito.shared.chain import (
    SignedModelHashChainCommit,
    get_chain_commits,
)
from connito.shared.checkpoint_helper import compile_full_state_dict_from_path, load_checkpoint
from connito.shared.checkpoints import (
    ChainCheckpoints,
    ModelCheckpoint,
    build_chain_checkpoints,
    build_chain_checkpoints_from_previous_phase,
    delete_old_checkpoints,
    select_best_checkpoint,
)
from connito.shared.client import download_model
from connito.shared.config import MinerConfig, ValidatorConfig, WorkerConfig
from connito.shared.cycle import PhaseNames
from connito.shared.expert_manager import (
    ExpertManager,
    get_layer_expert_id,
    ExpertAssignments
)
from connito.shared.helper import get_model_hash, get_nested_attr
from connito.shared.modeling.mycelia import get_base_model
from connito.shared.schema import verify_message

logger = structlog.get_logger(__name__)


def grad_hook(name):
    def h(grad):
        if grad is not None and not torch.isfinite(grad).all():
            print("❌ grad NaN/Inf at", name)
            raise RuntimeError(name)
        return grad

    return h

def freeze_parameters(
    model: nn.Module,
    expert_manager: ExpertManager,
    expert_group_id: int,
    upcast_trainable: bool = False,
) -> nn.Module:
    """
    Freeze all parameters except those belonging to expert_group_id.

    Two modes depending on how experts are stored in the model:

    - Per-expert names (e.g. ``experts.7.gate_up_proj``): expert_id is
      parsed from the parameter name; only the specific experts assigned to
      expert_group_id are kept trainable.

    - Stacked tensors (e.g. ``experts.gate_up_proj``, shape
      [num_local_experts, ...]): no expert_id is embedded in the name.
      In this case every expert-layer param for layers assigned to
      expert_group_id is kept trainable (the whole stacked tensor trains
      together; individual expert slices cannot be selectively frozen).
    """
    assignment = expert_manager.expert_group_assignment.get(expert_group_id, {})

    # Detect which mode the model uses by scanning for a param that has an
    # expert index in its name.
    uses_per_expert_names = any(
        get_layer_expert_id(name)[1] is not None
        for name, _ in model.named_parameters()
    )

    logger.debug(
        "freeze_parameters: detected naming mode",
        mode="per-expert" if uses_per_expert_names else "stacked",
        expert_group_id=expert_group_id,
        assigned_layers=sorted(assignment.keys()),
    )

    for name, param in model.named_parameters():
        layer_id, expert_id = get_layer_expert_id(name)
        
        # Check specifically for 3D fused expert blocks (e.g., Qwen3-VL-MoE)
        is_3d_expert_block = bool(re.search(r"layers\.\d+\.mlp\.experts\.(?:gate_up_proj|down_proj)", name))

        if uses_per_expert_names:
            # ── per-expert mode ──────────────────────────────────────────────
            if layer_id is not None and expert_id is not None:
                allowed = {
                    eid for eid, _ in assignment.get(layer_id, [])
                }
                param.requires_grad_(expert_id in allowed)
            else:
                param.requires_grad_(False)
        else:
            # ── stacked mode ─────────────────────────────────────────────────
            # Trainable iff: the param belongs to a routed expert layer AND
            # that layer has at least one expert assigned to this group.
            # Shared experts (shared_experts.*) are always frozen — they are
            # not group-specific.
            is_routed_expert_param = "expert" in name and "shared_expert" not in name and layer_id is not None
            layer_has_assignment = layer_id in assignment if layer_id is not None else False
            param.requires_grad_(is_routed_expert_param and layer_has_assignment)

    # Optionally upcast trainable parameters to float32 for stable mixed-precision optimization.
    # Needed for AdamW (moment estimates need fp32 precision), but not for SGD.
    upcast_count = 0
    if upcast_trainable:
        for p in model.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
                upcast_count += 1

    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    total = sum(1 for _ in model.parameters())
    logger.debug(
        "freeze_parameters: done",
        trainable=trainable,
        total=total,
        upcast_trainable_to_fp32=upcast_count,
        mode="per-expert" if uses_per_expert_names else "stacked",
    )
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()

    return model


def get_model_from_checkpoint(
    rank: int, config: MinerConfig | ValidatorConfig, expert_manager: ExpertManager, partial: bool = False,
    checkpoint_device: torch.device | None = None,
) -> tuple[nn.Module, ModelCheckpoint]:
    resume = get_nested_attr(config, "ckpt.resume_from_ckpt", False)
    group_ids = [config.task.exp.group_id] if partial else None
    logger.info(
        "Loading base model for checkpoint",
        mode="partial" if partial else "full",
        group_ids=group_ids or "all",
    )
    # get base model
    model = get_base_model(
        config,
        expert_manager=expert_manager,
        group_ids=group_ids,
        partial=partial,
    )

    # load from checkpoint
    if resume:
        latest_checkpoint = select_best_checkpoint(
            primary_dir=config.ckpt.validator_checkpoint_path,
            secondary_dir=config.ckpt.checkpoint_path,
            resume=config.ckpt.resume_from_ckpt,
        )

        if resume and latest_checkpoint is not None and latest_checkpoint.path:
            load_checkpoint(
                config=config,
                checkpoint_path=latest_checkpoint.path,
                model=model,
                rank=rank,
                device=checkpoint_device if checkpoint_device is not None else config.model.device,
                expert_groups=[config.task.exp.group_id, "shared"] if partial else None,
            )
        else:
            logger.info("Tried to resume from checkpoint, but no checkpoint found.")

    _device = checkpoint_device if checkpoint_device is not None else config.model.device
    
    precision = getattr(config.model, "precision", "fp16-mixed")
    if precision == "bf16-mixed" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        precision = "fp16-mixed"
    model_dtype = torch.bfloat16 if precision == "bf16-mixed" else torch.float16

    model = model.to(device=_device, dtype=model_dtype)
    model.gradient_checkpointing_enable()
    return model, latest_checkpoint

def load_model(
    rank: int,
    config: MinerConfig | ValidatorConfig,
    expert_manager: ExpertManager,
    subtensor: bittensor.Subtensor,
    phase_manager: "PhaseManager",
    wallet: bittensor.Wallet,
    current_checkpoint: ModelCheckpoint | None = None,
    partial: bool = False,
    checkpoint_device: torch.device | None = None,
) -> tuple[nn.Module, dict]:
    """
    Main entry point used by miners (and potentially validator itself).
    1) Ask the chain for an active validator endpoint.
    2) If available, ping and fetch current model.
    3) Else, initialize a default model.
    """
    # download new model from chain into file

    if current_checkpoint is None:
        current_checkpoint = select_best_checkpoint(
            primary_dir=config.ckpt.validator_checkpoint_path,
            secondary_dir=config.ckpt.checkpoint_path,
        )

    fetch_model_from_chain_validator(
        current_model_meta=current_checkpoint,
        config=config,
        subtensor=subtensor,
        phase_manager=phase_manager,
        wallet=wallet,
        expert_group_ids=[config.task.exp.group_id],
        expert_group_assignment=expert_manager.expert_group_assignment
    )

    return get_model_from_checkpoint(rank=rank, config=config, expert_manager=expert_manager, partial=partial, checkpoint_device=checkpoint_device)


def fetch_model_from_chain_validator(
    current_model_meta: ModelCheckpoint | None,
    config: WorkerConfig,
    subtensor: bittensor.Subtensor,
    phase_manager: "PhaseManager",
    wallet: bittensor.Wallet,
    expert_group_ids: list[int | str],
    expert_group_assignment: ExpertAssignments
) -> dict | None:
    """
    Fetches a model from the chain validator if it's has the right commit format from the previous phase commits (validator_commit_1 & validator_commit_2) and newer than the current model.
    """
    try:
        owner_hotkey = subtensor.get_subnet_owner_hotkey(netuid=config.chain.netuid)
    except Exception as e:
        logger.warning("Could not resolve SN owner hotkey", error=str(e))
        owner_hotkey = None

    chain_checkpoints = build_chain_checkpoints_from_previous_phase(
        config=config,
        subtensor=subtensor,
        phase_manager=phase_manager,
        for_role="validator",
        owner_hotkey=owner_hotkey,
    )

    # --- Filter to only newer than current model ---
    if current_model_meta is not None: 
        chain_checkpoints = ChainCheckpoints(
            checkpoints=[ckpt for ckpt in chain_checkpoints.checkpoints if ckpt > current_model_meta]
        )
        
    should_download = len(chain_checkpoints.checkpoints) > 0

    logger.info(
        "Fetching model from chain",
        should_download=should_download,
        chain_checkpoints=chain_checkpoints,
        current_model_meta=current_model_meta,
    )

    # --- Download model if available ---
    if should_download and chain_checkpoints:
        download_success = False
        retries = 0
        max_retries = 3
        base_delay_s = 5  # backoff base

        while (not download_success) and (retries < max_retries):
            for chain_checkpoint in chain_checkpoints.checkpoints:
                logger.info(f"Downloading from chain: uid = {chain_checkpoint.uid}", chain_checkpoint=chain_checkpoint)

                # Resolve URL if not provided; fall back to ip/port + default route
                # Best-effort defaults; customize if your API differs
                protocol = getattr(getattr(config, "miner", object()), "protocol", "http")
                if chain_checkpoint.ip and chain_checkpoint.port:
                    url = f"{protocol}://{chain_checkpoint.ip}:{chain_checkpoint.port}/get-checkpoint"
                else:
                    logger.warning("Skipping meta without URL or ip:port: %s", chain_checkpoint)
                    continue

                out_folder = Path(config.ckpt.validator_checkpoint_path) / (
                    f"uid_{chain_checkpoint.uid}_hotkey_{chain_checkpoint.hotkey}_globalver_{chain_checkpoint.global_ver}"
                )

                out_folder.mkdir(parents=True, exist_ok=True)

                for expert_group_id in expert_group_ids:
                    if isinstance(expert_group_id, int):
                        out_file = f"model_expgroup_{expert_group_id}.pt"
                    elif expert_group_id == "shared":
                        out_file = "model_shared.pt"
                    else:
                        logger.warning("Invalid expert_group_id, skipping:", expert_group_id=expert_group_id)
                        continue

                    out_path = out_folder / out_file
                    try:
                        download_model(
                            url=url,
                            my_hotkey=wallet.hotkey,  # type: ignore
                            target_hotkey_ss58=chain_checkpoint.hotkey,
                            block=subtensor.block,
                            expert_group_id=expert_group_id,
                            token=getattr(config.cycle, "token", ""),
                            out_dir=out_path,
                        )

                        chain_checkpoint.path = out_folder
                        validated = chain_checkpoint.validate(expert_group_assignment = expert_group_assignment)

                        if not validated:
                            logger.warning(
                                "❌ Downloaded checkpoint failed validation",
                                out_path=out_path,
                                current_model_version=current_model_meta.global_ver
                                if current_model_meta
                                else None,
                                current_model_hash=current_model_meta.model_hash if current_model_meta else None,
                            )
                            continue
                        # If download + verification succeed, consider it a success
                        download_success = validated

                        current_model_version = chain_checkpoint.global_ver
                        current_model_hash = chain_checkpoint.model_hash
                        
                        logger.info(
                            "✅ Downloaded checkpoint (verified)",
                            out_path=out_path,
                            current_model_version=current_model_version,
                            current_model_hash=current_model_hash,
                            validation_success=validated,
                        )

                        delete_old_checkpoints(
                            checkpoint_path=Path(config.ckpt.validator_checkpoint_path),
                            topk=config.ckpt.checkpoint_topk,
                        )

                        return chain_checkpoint
                    except Exception as e:
                        logger.warning("Download failed", url=url, error=str(e), exc_info=True)

            if not download_success:
                retries += 1
                if retries < max_retries:
                    delay = base_delay_s * (2 ** (retries - 1))
                    logger.info("Retrying", delay=delay, retries=retries + 1, max_retries=max_retries)
                    time.sleep(delay)

        if not download_success:
            logger.error(f"❌ All download attempts failed after {retries} retries.")

            return None


def reload_model_inplace(
    config: ValidatorConfig,
    global_model: nn.Module,
    expert_manager: ExpertManager,
    device: torch.device,
    subtensor: bittensor.Subtensor,
    phase_manager: "PhaseManager",
    wallet: bittensor.Wallet,
) -> bool:
    """
    Pull the latest committed validator checkpoint from a peer and load it
    into *global_model* in-place.  Called at the start of the next cycle
    when this validator was excluded from the allreduce (no miner assigned,
    or inf/nan gradients).  By then the participating validators have
    finished merge + optimizer + save + committed new hashes via
    validator_commit_1/2, so build_chain_checkpoints_from_previous_phase
    finds the fresh model.
    Returns True on success, False on any failure (caller keeps stale model).
    """
    logger.info("Pulling model from peer validator to re-sync excluded validator")
    try:
        fetch_model_from_chain_validator(
            current_model_meta=None,  # None = always try; no version gate
            config=config,
            subtensor=subtensor,
            phase_manager=phase_manager,
            wallet=wallet,
            expert_group_ids=[config.task.exp.group_id, "shared"],
            expert_group_assignment=expert_manager.expert_group_assignment,
        )
    except Exception as e:
        logger.warning("Peer sync: fetch_model_from_chain_validator failed", error=str(e))
        return False

    latest = select_best_checkpoint(
        primary_dir=config.ckpt.validator_checkpoint_path,
        secondary_dir=config.ckpt.checkpoint_path,
    )
    if latest is None or latest.path is None:
        logger.warning("Peer sync: no checkpoint found after download")
        return False

    try:
        sd = compile_full_state_dict_from_path(
            latest.path,
            expert_groups=[config.task.exp.group_id, "shared"],
        )
        if not sd:
            logger.warning("Peer sync: downloaded checkpoint has empty state dict")
            return False
        global_model.load_state_dict(sd, strict=False)
        global_model.to(device)
        logger.info(
            "Peer sync: loaded checkpoint into global_model",
            path=str(latest.path),
            global_ver=latest.global_ver,
            model_hash=get_model_hash(sd, hex=True)[:6],
        )
        return True
    except Exception as e:
        logger.warning("Peer sync: failed to load state dict into global_model", error=str(e))
        return False
