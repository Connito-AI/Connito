from __future__ import annotations

import re
import time
import traceback
from concurrent.futures import TimeoutError as FuturesTimeoutError
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
from connito.shared.config import MinerConfig, ValidatorConfig, WorkerConfig
from connito.shared.hf_distribute import download_checkpoint_from_hf_with_timeout
from connito.shared.cycle import (
    PhaseNames,
    get_blocks_from_previous_phase_from_api,
    get_validator_seed_from_commit,
)
from connito.shared.expert_manager import (
    ExpertManager,
    get_layer_expert_id,
    ExpertAssignments
)
from connito.shared.helper import get_model_hash, get_nested_attr
from connito.shared.memory import cleanup
from connito.shared.modeling.mycelia import get_base_model
from connito.shared.schema import verify_message

logger = structlog.get_logger(__name__)


def _build_download_targets(expert_group_ids: list[int | str]) -> list[tuple[int | str, str]]:
    targets: list[tuple[int | str, str]] = []
    for expert_group_id in expert_group_ids:
        if isinstance(expert_group_id, int):
            targets.append((expert_group_id, f"model_expgroup_{expert_group_id}.pt"))
        elif expert_group_id == "shared":
            # `model_shared` is no longer persisted or distributed; backbone state
            # is reconstructed from `config.model.model_path` at instantiation.
            # Tolerated here for backward compatibility with callers that still
            # pass the sentinel, but it's a no-op.
            logger.debug(
                "expert_group_id='shared' is deprecated and ignored — backbone is loaded from from_pretrained",
            )
        else:
            logger.warning("Invalid expert_group_id, skipping", expert_group_id=expert_group_id)
    return targets


def _clear_download_targets(out_folder: Path, filenames: list[str]) -> None:
    for filename in filenames:
        (out_folder / filename).unlink(missing_ok=True)


def _download_checkpoint_from_hf_with_timeout(
    *,
    repo_id: str,
    revision: str,
    filenames: list[str],
    dest_dir: Path,
    token_env_var: str | None,
    timeout_sec: float | None,
) -> None:
    download_checkpoint_from_hf_with_timeout(
        repo_id=repo_id,
        revision=revision,
        filenames=filenames,
        dest_dir=dest_dir,
        token_env_var=token_env_var or "HF_TOKEN",
        timeout_sec=timeout_sec,
    )


def grad_hook(name):
    def h(grad):
        if grad is not None and not torch.isfinite(grad).all():
            print("❌ grad NaN/Inf at", name)
            raise RuntimeError(name)
        return grad

    return h


def enable_gradient_checkpointing(model: nn.Module, reentrant: bool = False) -> None:
    """
    Enable HuggingFace gradient checkpointing on `model` when supported.

    Activations for transformer blocks are recomputed during backward instead
    of being stored, trading ~20% throughput for a large memory reduction.
    Operators can then raise the per-device batch size to net positive on
    tokens/sec per GB.

    Notes
    -----
    - `use_reentrant=False` selects the newer non-reentrant API
      (faster and recommended). Pass `reentrant=True` only if the
      backend mis-handles non-reentrant (rare on modern transformers).
    - `model.config.use_cache` must be False while grad checkpointing
      is on; we set it here defensively even when `use_cache` is already
      False.
    - When ordered against `torch.compile` (Unit 10), grad checkpointing
      must be enabled BEFORE `torch.compile(model)`.
    - No-op (warning logged, no crash) if `model` does not expose
      `gradient_checkpointing_enable`.
    """
    if not hasattr(model, "gradient_checkpointing_enable"):
        logger.warning(
            "Model does not support gradient_checkpointing_enable; skipping",
            model_type=type(model).__name__,
        )
        return

    # `use_cache=True` is incompatible with gradient checkpointing — the
    # KV cache assumes activations are kept around, but grad ckpt deletes
    # them. Disable defensively on the model config when available.
    model_config = getattr(model, "config", None)
    if model_config is not None and getattr(model_config, "use_cache", False):
        model_config.use_cache = False
        logger.info("Disabled model.config.use_cache for gradient checkpointing")

    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": reentrant},
        )
    except TypeError:
        # Older transformers releases do not accept
        # `gradient_checkpointing_kwargs`. Fall back to the legacy call —
        # the older default (reentrant=True) is the only available
        # behavior there.
        model.gradient_checkpointing_enable()
        logger.info(
            "Gradient checkpointing enabled (legacy fallback; "
            "transformers does not support gradient_checkpointing_kwargs)",
            requested_reentrant=reentrant,
        )
        return

    logger.info("Gradient checkpointing enabled", reentrant=reentrant)

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
    load_global_checkpoint: bool = True,
) -> tuple[nn.Module, ModelCheckpoint | None]:
    # `load_global_checkpoint=False` lets a caller (currently: the validator
    # at cold start) build the base model without overlaying any on-disk
    # expert state. The pretrained backbone + experts from `get_base_model`
    # are returned as-is; mid-cycle peer resync via `reload_model_inplace`
    # is unaffected.
    resume = get_nested_attr(config, "ckpt.resume_from_ckpt", False) and load_global_checkpoint
    group_ids = [config.task.exp.group_id] if partial else None
    logger.info(
        "Loading base model for checkpoint",
        mode="partial" if partial else "full",
        group_ids=group_ids or "all",
        load_global_checkpoint=load_global_checkpoint,
    )
    # get base model
    model = get_base_model(
        config,
        expert_manager=expert_manager,
        group_ids=group_ids,
        partial=partial,
    )

    latest_checkpoint: ModelCheckpoint | None = None
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
                # Only the active expert group is read from disk. Any legacy
                # `model_shared.*` next to it is ignored — backbone state is
                # already in `model` from `get_base_model`'s pretrained load
                # (full path via `from_pretrained`, partial path via the
                # full→partial port in mycelia.get_base_model), and we no
                # longer trust on-disk shared weights (they were a source of
                # cross-validator divergence; see PR description).
                expert_groups=[config.task.exp.group_id] if partial else None,
            )
        else:
            logger.info("Tried to resume from checkpoint, but no checkpoint found.")

    _device = checkpoint_device if checkpoint_device is not None else config.model.device
    
    precision = getattr(config.model, "precision", "fp16-mixed")
    if precision == "bf16-mixed" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        precision = "fp16-mixed"
    model_dtype = torch.bfloat16 if precision == "bf16-mixed" else torch.float16

    model = model.to(device=_device, dtype=model_dtype)
    # Gradient checkpointing is opt-in via `task.exp.data.use_gradient_checkpointing`.
    # Default off — trades ~20% throughput for a large memory reduction, so
    # operators turn it on only when they want to raise batch size to net
    # positive on tokens/sec per GB. Callers (train.py / validator/run.py)
    # may also call `enable_gradient_checkpointing` after this returns; the
    # helper is idempotent — HuggingFace's flag is a simple bool set on
    # every block.
    if get_nested_attr(config, "task.exp.data.use_gradient_checkpointing", False):
        enable_gradient_checkpointing(
            model,
            reentrant=get_nested_attr(
                config, "task.exp.data.gradient_checkpointing_reentrant", False
            ),
        )
    return model, latest_checkpoint

def load_model(
    rank: int,
    config: MinerConfig | ValidatorConfig,
    expert_manager: ExpertManager,
    subtensor: bittensor.Subtensor,
    wallet: bittensor.Wallet,
    current_checkpoint: ModelCheckpoint | None = None,
    partial: bool = False,
    checkpoint_device: torch.device | None = None,
    load_global_checkpoint: bool = True,
) -> tuple[nn.Module, ModelCheckpoint | None]:
    """
    Main entry point used by miners (and potentially validator itself).
    1) Ask the chain for an active validator endpoint.
    2) If available, ping and fetch current model.
    3) Else, initialize a default model.

    `load_global_checkpoint=False` skips the on-disk expert-state overlay
    in `get_model_from_checkpoint` (the chain fetch above still runs so
    future peer-resyncs have shards on disk).
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
        wallet=wallet,
        expert_group_ids=[config.task.exp.group_id],
        expert_group_assignment=expert_manager.expert_group_assignment
    )

    return get_model_from_checkpoint(
        rank=rank, config=config, expert_manager=expert_manager,
        partial=partial, checkpoint_device=checkpoint_device,
        load_global_checkpoint=load_global_checkpoint,
    )


def fetch_model_from_chain_validator(
    current_model_meta: ModelCheckpoint | None,
    config: WorkerConfig,
    subtensor: bittensor.Subtensor,
    wallet: bittensor.Wallet,
    expert_group_ids: list[int | str],
    expert_group_assignment: ExpertAssignments,
    allowed_hotkeys: set[str] | None = None,
    download_timeout_sec: float | None = None,
) -> dict | None:
    """
    Fetches a model from the chain validator if it's has the right commit format from the previous phase commits (validator_commit_1 & validator_commit_2) and newer than the current model.

    When ``allowed_hotkeys`` is provided, chain checkpoints whose hotkey is not
    in the set are dropped before download. Used by peer sync to require the
    source be one of the assignment validators.

    When ``download_timeout_sec`` is set, each per-checkpoint
    ``download_checkpoint_from_hf`` call is bounded by that wall clock and a
    timeout is treated as a download failure (retried/skipped to the next
    candidate by the existing retry budget).
    """
    try:
        owner_hotkey = subtensor.get_subnet_owner_hotkey(netuid=config.chain.netuid)
    except Exception as e:
        logger.warning("Could not resolve SN owner hotkey", error=str(e))
        owner_hotkey = None

    chain_checkpoints = build_chain_checkpoints_from_previous_phase(
        config=config, subtensor=subtensor, for_role="validator", owner_hotkey=owner_hotkey
    )

    if allowed_hotkeys is not None:
        before = len(chain_checkpoints.checkpoints)
        kept = [
            ckpt for ckpt in chain_checkpoints.checkpoints
            if ckpt.hotkey is not None and ckpt.hotkey in allowed_hotkeys
        ]
        dropped = before - len(kept)
        if dropped:
            logger.info(
                "fetch_model_from_chain_validator: dropped checkpoints not in allowed_hotkeys",
                dropped=dropped,
                kept=len(kept),
                allowed_hotkey_count=len(allowed_hotkeys),
            )
        chain_checkpoints = ChainCheckpoints(checkpoints=kept)

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
        max_retries = 2
        retry_delay_s = 10

        while (not download_success) and (retries < max_retries):
            for chain_checkpoint in chain_checkpoints.checkpoints:
                logger.info(f"Downloading from chain: uid = {chain_checkpoint.uid}", chain_checkpoint=chain_checkpoint)

                out_folder = Path(config.ckpt.validator_checkpoint_path) / (
                    f"uid_{chain_checkpoint.uid}_hotkey_{chain_checkpoint.hotkey}_globalver_{chain_checkpoint.global_ver}"
                )
                out_folder.mkdir(parents=True, exist_ok=True)

                targets = _build_download_targets(expert_group_ids)
                filenames = [filename for _, filename in targets]
                if not filenames:
                    continue

                if not (chain_checkpoint.hf_repo_id and chain_checkpoint.hf_revision):
                    logger.warning(
                        "Chain checkpoint missing HF coordinates; skipping",
                        uid=chain_checkpoint.uid,
                        hotkey=chain_checkpoint.hotkey,
                    )
                    continue

                _clear_download_targets(out_folder, filenames)

                try:
                    _download_checkpoint_from_hf_with_timeout(
                        repo_id=chain_checkpoint.hf_repo_id,
                        revision=chain_checkpoint.hf_revision,
                        filenames=filenames,
                        dest_dir=out_folder,
                        token_env_var=config.hf.token_env_var,
                        timeout_sec=download_timeout_sec,
                    )
                except FuturesTimeoutError:
                    # The thread running download_checkpoint_from_hf cannot be
                    # killed from Python — it stays orphaned until the OS-level
                    # socket times out — but the main loop is freed and moves
                    # on to the next candidate / retry, which is the whole
                    # point of this guard.
                    logger.warning(
                        "Checkpoint download from HF timed out",
                        uid=chain_checkpoint.uid,
                        hotkey=chain_checkpoint.hotkey,
                        hf_repo_id=chain_checkpoint.hf_repo_id,
                        hf_revision=chain_checkpoint.hf_revision,
                        timeout_sec=download_timeout_sec,
                    )
                    continue
                except Exception as e:
                    logger.warning(
                        "Checkpoint download from HF failed",
                        uid=chain_checkpoint.uid,
                        hotkey=chain_checkpoint.hotkey,
                        hf_repo_id=chain_checkpoint.hf_repo_id,
                        hf_revision=chain_checkpoint.hf_revision,
                        error=str(e),
                        exc_info=True,
                    )
                    continue

                chain_checkpoint.path = out_folder
                validated = chain_checkpoint.validate(expert_group_assignment=expert_group_assignment)

                if not validated:
                    logger.warning(
                        "Downloaded checkpoint failed validation",
                        out_folder=out_folder,
                        current_model_version=current_model_meta.global_ver if current_model_meta else None,
                        current_model_hash=current_model_meta.model_hash if current_model_meta else None,
                    )
                    continue

                download_success = True
                current_model_version = chain_checkpoint.global_ver
                current_model_hash = chain_checkpoint.model_hash

                logger.info(
                    "Downloaded checkpoint (verified)",
                    out_folder=out_folder,
                    hf_repo_id=chain_checkpoint.hf_repo_id,
                    hf_revision=chain_checkpoint.hf_revision,
                    current_model_version=current_model_version,
                    current_model_hash=current_model_hash,
                )

                delete_old_checkpoints(
                    checkpoint_path=Path(config.ckpt.validator_checkpoint_path),
                    topk=config.ckpt.checkpoint_topk,
                )

                return chain_checkpoint

            if not download_success:
                retries += 1
                if retries < max_retries:
                    logger.info("Retrying", delay=retry_delay_s, retries=retries + 1, max_retries=max_retries)
                    time.sleep(retry_delay_s)

        if not download_success:
            logger.error(f"❌ All download attempts failed after {retries} retries.")

            return None


def reload_model_inplace(
    config: ValidatorConfig,
    global_model: nn.Module,
    expert_manager: ExpertManager,
    device: torch.device,
    subtensor: bittensor.Subtensor,
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
    t_start = time.monotonic()

    # Restrict the peer-sync source set to the same validator pool used by
    # validator-miner assignment (validators that committed a
    # ValidatorChainCommit for this expert_group). Any chain entry from a
    # non-assignment hotkey — e.g. a stake-0 / unverified commit — is
    # rejected before the HF download so a bad source can't stall the cycle.
    try:
        commits = get_chain_commits(config, subtensor)
        validator_seeds = get_validator_seed_from_commit(config, commits)
        allowed_hotkeys = set(validator_seeds.keys())
    except Exception as e:
        logger.warning(
            "Peer sync: could not resolve assignment validator whitelist; aborting",
            error=str(e),
        )
        return False

    if not allowed_hotkeys:
        logger.warning(
            "Peer sync: no assignment validators found on chain; aborting"
        )
        return False

    download_timeout_sec = float(config.evaluation.per_miner_download_timeout_sec)

    logger.info(
        "Peer sync: fetching latest checkpoint from chain",
        primary_dir=str(config.ckpt.validator_checkpoint_path),
        secondary_dir=str(config.ckpt.checkpoint_path),
        allowed_hotkey_count=len(allowed_hotkeys),
        download_timeout_sec=download_timeout_sec,
    )
    t_fetch = time.monotonic()
    try:
        fetch_model_from_chain_validator(
            current_model_meta=None,  # None = always try; no version gate
            config=config,
            subtensor=subtensor,
            wallet=wallet,
            expert_group_ids=[config.task.exp.group_id],
            expert_group_assignment=expert_manager.expert_group_assignment,
            allowed_hotkeys=allowed_hotkeys,
            download_timeout_sec=download_timeout_sec,
        )
    except Exception as e:
        logger.warning(
            "Peer sync: fetch_model_from_chain_validator failed",
            error=str(e),
            elapsed_sec=round(time.monotonic() - t_fetch, 2),
        )
        return False
    logger.info(
        "Peer sync: chain fetch complete",
        elapsed_sec=round(time.monotonic() - t_fetch, 2),
    )

    logger.info("Peer sync: selecting best local checkpoint")
    latest = select_best_checkpoint(
        primary_dir=config.ckpt.validator_checkpoint_path,
        secondary_dir=config.ckpt.checkpoint_path,
    )
    if latest is None or latest.path is None:
        logger.warning(
            "Peer sync: no checkpoint found after download",
            primary_dir=str(config.ckpt.validator_checkpoint_path),
            secondary_dir=str(config.ckpt.checkpoint_path),
        )
        return False
    logger.info(
        "Peer sync: selected best checkpoint",
        path=str(latest.path),
        global_ver=latest.global_ver,
    )

    # Reclaim cached allocator memory and CPU heap before loading the multi-GB
    # state dict. Peer sync runs while bg-eval / training buffers are still
    # live; without this the transient allocation peak during torch.load +
    # load_state_dict has been seen to OOM-kill the process silently.
    cleanup(global_model)

    try:
        logger.info("Peer sync: loading state dict from disk", path=str(latest.path))
        t_load = time.monotonic()
        sd = compile_full_state_dict_from_path(
            latest.path,
            expert_groups=[config.task.exp.group_id],
        )
        load_elapsed = round(time.monotonic() - t_load, 2)
        if not sd:
            logger.warning(
                "Peer sync: downloaded checkpoint has empty state dict",
                path=str(latest.path),
                elapsed_sec=load_elapsed,
            )
            return False
        model_hash = get_model_hash(sd, hex=True)[:6]
        logger.info(
            "Peer sync: state dict loaded",
            num_keys=len(sd),
            elapsed_sec=load_elapsed,
            model_hash=model_hash,
        )

        logger.info("Peer sync: copying state dict into global_model")
        t_copy = time.monotonic()
        result = global_model.load_state_dict(sd, strict=False)
        copy_elapsed = round(time.monotonic() - t_copy, 2)
        # Drop the CPU copy and reclaim heap before the success log so the
        # caller's next cycle starts with the full memory budget back.
        del sd
        cleanup(global_model)
        logger.info(
            "Peer sync: loaded checkpoint into global_model",
            path=str(latest.path),
            global_ver=latest.global_ver,
            model_hash=model_hash,
            missing_keys=len(result.missing_keys),
            unexpected_keys=len(result.unexpected_keys),
            copy_elapsed_sec=copy_elapsed,
            total_elapsed_sec=round(time.monotonic() - t_start, 2),
        )
        return True
    except Exception as e:
        logger.warning(
            "Peer sync: failed to load state dict into global_model",
            error=str(e),
            path=str(latest.path),
        )
        return False
