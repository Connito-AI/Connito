from __future__ import annotations

import shutil
import time
import traceback
from pathlib import Path

import bittensor
import torch
from torch import nn

from connito.shared.app_logging import structlog
from connito.shared.chain import (
    SignedModelHashChainCommit,
)
from connito.shared.checkpoint_helper import TensorSpec, group_shard, verify_residual_shard
from connito.shared.checkpoints import (
    ChainCheckpoints,
    ModelCheckpoint,
    build_chain_checkpoints,
    build_chain_checkpoints_from_previous_phase,
    delete_old_checkpoints,
    local_checkpoints_best_first,
    select_best_checkpoint,
)
from connito.shared.config import MinerConfig, ValidatorConfig, WorkerConfig
from connito.shared.hf_distribute import download_checkpoint_from_hf_with_timeout
from connito.shared.cycle import (
    PhaseNames,
    get_blocks_from_previous_phase_from_api,
)
from connito.shared.helper import expert_group_shard_name, get_nested_attr
from connito.shared.modeling.mycelia import get_base_model
from connito.shared.modeling.residual import install_residual_experts, residual_shapes
from connito.shared.schema import verify_message

logger = structlog.get_logger(__name__)


def _build_download_targets(expert_group_ids: list[int | str]) -> list[tuple[int | str, str]]:
    targets: list[tuple[int | str, str]] = []
    for expert_group_id in expert_group_ids:
        if isinstance(expert_group_id, int):
            targets.append((expert_group_id, expert_group_shard_name(expert_group_id)))
        elif expert_group_id == "shared":
            # `model_shared` is no longer persisted or distributed; backbone state
            # is reconstructed from the model family's pinned checkpoint at instantiation.
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


def grad_hook(name):
    def h(grad):
        if grad is not None and not torch.isfinite(grad).all():
            print("❌ grad NaN/Inf at", name)
            raise RuntimeError(name)
        return grad

    return h

def get_model_from_checkpoint(
    config: MinerConfig | ValidatorConfig,
    checkpoint_device: torch.device | None = None,
    load_global_checkpoint: bool = True,
) -> tuple[nn.Module, ModelCheckpoint | None]:
    # `load_global_checkpoint=False` lets a caller (currently: the validator
    # at cold start) build the base model without overlaying any on-disk
    # expert state. The pretrained backbone + experts from `get_base_model`
    # are returned as-is.
    use_pretrained_only = bool(get_nested_attr(config, "ckpt.use_pretrained_only", False))
    resume = (
        get_nested_attr(config, "ckpt.resume_from_ckpt", False)
        and load_global_checkpoint
        and not use_pretrained_only
    )
    logger.info("Loading base model for checkpoint", load_global_checkpoint=load_global_checkpoint)
    model = get_base_model(config)
    install_residual_experts(model, config)

    latest_checkpoint: ModelCheckpoint | None = None
    # load from checkpoint
    if resume:
        candidates = local_checkpoints_best_first(config.ckpt.validator_checkpoint_path, config.ckpt.checkpoint_path)
        # Returned as before, the newest on disk: the resync check compares the next newest with it.
        latest_checkpoint = candidates[0] if candidates else None
        # The newest that verifies is the start. One refused (untrained, off spec, an empty download folder) must
        # not cost the miner its own trained checkpoint below it; none verifying leaves the fresh draw, which trains.
        spec = residual_shapes(config)
        for candidate in candidates:
            state_dict = verify_residual_shard(group_shard(candidate.path, config.task.exp.group_id), spec)
            if state_dict is not None:
                model.load_state_dict(state_dict, strict=False)
                logger.info("Resumed the residual", path=str(candidate.path))
                break
        else:
            logger.info("Keeping the fresh residual: no local checkpoint is a start", candidates=len(candidates))

    _device = checkpoint_device if checkpoint_device is not None else config.model.device
    model = model.to(device=_device)
    # Non-reentrant: the setting the L40S hardware floor was measured with.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, latest_checkpoint

def load_model(
    config: MinerConfig | ValidatorConfig,
    subtensor: bittensor.Subtensor,
    wallet: bittensor.Wallet,
    current_checkpoint: ModelCheckpoint | None = None,
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

    use_pretrained_only = bool(get_nested_attr(config, "ckpt.use_pretrained_only", False))

    if use_pretrained_only:
        logger.info(
            "ckpt.use_pretrained_only=True — skipping chain fetch + on-disk overlay; "
            "returning the pretrained base model only",
        )
    else:
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
            expected=residual_shapes(config),
        )

    return get_model_from_checkpoint(
        config=config, checkpoint_device=checkpoint_device,
        load_global_checkpoint=load_global_checkpoint,
    )


def fetch_model_from_chain_validator(
    current_model_meta: ModelCheckpoint | None,
    config: WorkerConfig,
    subtensor: bittensor.Subtensor,
    wallet: bittensor.Wallet,
    expert_group_ids: list[int | str],
    expected: dict[str, TensorSpec],
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
        config=config, subtensor=subtensor, for_role="validator", owner_hotkey=owner_hotkey
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
        max_retries = 2
        retry_delay_s = 10

        while (not download_success) and (retries < max_retries):
            for chain_checkpoint in chain_checkpoints.checkpoints:
                logger.info(f"Downloading from chain: uid = {chain_checkpoint.uid}", chain_checkpoint=chain_checkpoint)

                out_folder = Path(config.ckpt.validator_checkpoint_path) / (
                    f"uid_{chain_checkpoint.uid}_hotkey_{chain_checkpoint.hotkey}_globalver_{chain_checkpoint.global_ver}"
                )

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

                out_folder.mkdir(parents=True, exist_ok=True)
                _clear_download_targets(out_folder, filenames)

                try:
                    download_checkpoint_from_hf_with_timeout(
                        repo_id=chain_checkpoint.hf_repo_id,
                        revision=chain_checkpoint.hf_revision,
                        filenames=filenames,
                        dest_dir=out_folder,
                        token_env_var=config.hf.token_env_var or "HF_TOKEN",
                        timeout_sec=None,
                    )
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
                    # A folder left behind is the newest checkpoint on disk, and a miner would try to start from it.
                    shutil.rmtree(out_folder, ignore_errors=True)
                    continue

                chain_checkpoint.path = out_folder
                validated = chain_checkpoint.validate(expected)

                if not validated:
                    logger.warning(
                        "Downloaded checkpoint failed validation",
                        out_folder=out_folder,
                        current_model_version=current_model_meta.global_ver if current_model_meta else None,
                        current_model_hash=current_model_meta.model_hash if current_model_meta else None,
                    )
                    shutil.rmtree(out_folder, ignore_errors=True)
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

