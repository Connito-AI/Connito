import asyncio
import copy
import gc
import os
from dotenv import load_dotenv

load_dotenv()

import secrets
from typing import Any

import bittensor
import torch
import torch.nn as nn
from hivemind.averaging import DecentralizedAverager
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerBase

from connito.miner.train_helper import get_status
from connito.shared.app_logging import configure_logging, log_phase, structlog
from connito.shared.chain import (
    SignedModelHashChainCommit,
    ValidatorChainCommit,
    _submit_fallback_weights,
    commit_status,
    setup_chain_worker,
    submit_weights,
)
from connito.shared.checkpoint_helper import (
    load_checkpoint,
    save_checkpoint,
)
from connito.shared.checkpoints import (
    ModelCheckpoint,
    archive_top_miner_submissions,
    build_local_checkpoint,
    delete_old_checkpoints,
    select_best_checkpoint,
)
from connito.shared.config import ValidatorConfig, parse_args
from connito.shared.cycle import check_phase_expired, gather_validation_job, get_combined_validator_seed, wait_till
from connito.shared.dataloader import get_dataloader
from connito.shared.evaluate import evaluate_model
from connito.shared.expert_manager import (
    ExpertManager,
    get_weight_sum,
    populate_global_grads_from_local,
)
from connito.shared.helper import get_model_hash, get_nested_attr, sum_model_gradients
from connito.shared.metrics import MetricLogger
from connito.shared.model import load_model
from connito.shared.modeling.mycelia import get_base_tokenizer
from connito.sn_owner.cycle import PhaseNames, PhaseManager
from connito.validator.aggregator import MinerScoreAggregator
from connito.validator.evaluator import (
    MinerEvalJob,
    load_model_from_path,
    run_evaluation,
)
from connito.validator.inter_validator_connection import (
    build_averagers_from_buff,
    build_grad_buff_from_model,
    connect_with_peers,
    pack_grads,
    unpack_to_grads,
)
from connito.shared.telemetry import (
    TelemetryManager,
    VALIDATOR_AVG_STEP_STATUS,
    SystemStatePoller
)
from datetime import datetime

configure_logging()
logger = structlog.get_logger(__name__)


def _cuda_mem_report(tag: str = "", device: int | None = None) -> None:
    if not torch.cuda.is_available():
        print(f"[{tag}] CUDA not available")
        return

    if device is None:
        device = torch.cuda.current_device()

    torch.cuda.synchronize(device)

    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)

    free, total = torch.cuda.mem_get_info(device)  # bytes

    def mb(x):
        return x / 1024**2

    log_phase(
        f"[{tag}] cuda:{device}",
        allocated=f"{mb(allocated):.1f}MB",
        reserved=f"{mb(reserved):.1f}MB",
        free=f"{mb(free):.1f}MB",
        total=f"{mb(total):.1f}MB",
        alloc_pct=f"{allocated/total*100:.1f}%",
        reserved_pct=f"{reserved/total*100:.1f}%",
    )


def cleanup(global_model, base_model) -> None:
    """
    Cleans up the distributed training environment.
    """
    _cuda_mem_report("VRAM before GPU cleanup")

    # Move models off GPU
    torch.cuda.synchronize()
    global_model.to("cpu")
    base_model.to("cpu")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    _cuda_mem_report("VRAM after GPU cleanup")


def setup_training(
    config,
    rank: int,
    device: torch.device,
    tokenizer: PreTrainedTokenizerBase,
    subtensor: bittensor.Subtensor,
    wallet: bittensor.Wallet,
    current_model_meta: ModelCheckpoint | None,
) -> tuple[
    torch.nn.Module,  # model
    torch.nn.Module,  # global_model
    torch.optim.Optimizer,  # outer_optimizer
    torch.amp.GradScaler,  # outer_scaler
    int,  # start_step
    "ExpertManager",  # em
    StatefulDataLoader,
]:
    """
    Build model(s), experts layout, optimizers, scheduler, scaler, and optionally resume from a checkpoint.
    """
    # === checkpoint info ===
    latest_checkpoint = select_best_checkpoint(primary_dir=config.ckpt.checkpoint_path)
    resume = latest_checkpoint is not None
    latest_checkpoint_path = latest_checkpoint.path if latest_checkpoint else None

    # === model & Experts manager ===
    logger.debug("setup training - load model and expert manager")
    expert_manager = ExpertManager(config)
    # base_model: full model (all experts) — used for evaluation
    base_model, model_meta = load_model(rank, config, expert_manager, subtensor, wallet, current_model_meta, partial=False, checkpoint_device=torch.device("cpu"))
    # global_model: partial model (only assigned experts) — used for optimization
    global_model, _ = load_model(rank, config, expert_manager, subtensor, wallet, current_model_meta, partial=True, checkpoint_device=torch.device("cpu"))

    # === optimizers ===
    logger.debug("setup training - load optimizer")
    outer_optimizer = torch.optim.SGD(
        [p for p in global_model.parameters() if p.requires_grad],
        lr=config.opt.outer_lr,
        momentum=config.opt.outer_momentum,
        nesterov=True,
    )

    # === scaler ===
    logger.debug("setup training - load scaler")
    outer_scaler = torch.amp.GradScaler(
        "cuda", enabled=(get_nested_attr(config, "model.precision", "") == "fp16-mixed")
    )

    # === dataloader ===
    logger.debug("setup training - load dataloader")
    train_dataloader = get_dataloader(
        config, rank=rank, world_size=config.task.exp.data.world_size, tokenizer=tokenizer
    )

    # === load checkpoint (if any) ===
    logger.debug(
        "setup training - load past checkpoint"
    )  # outer_optimizer is static, so dont really need to load checkpoint
    if get_nested_attr(config, "resume_from_ckpt", False) and resume and latest_checkpoint_path:
        _ = load_checkpoint(
            config=config,
            checkpoint_path=latest_checkpoint_path,
            outer_optimizer=outer_optimizer,
            outer_scaler=outer_scaler,
            rank=rank,
            device=device,
            data_loader=train_dataloader,
        )

    logger.info(
        "Training setup complete",
        resumed=resume,
        outer_lr=config.opt.outer_lr,
        device=str(device),
    )
    return (
        base_model,
        global_model,
        outer_optimizer,
        outer_scaler,
        model_meta.global_ver if model_meta else 0,
        expert_manager,
        train_dataloader,
    )


async def aggregate_miner_gradient_change(
    config: ValidatorConfig,
    global_model: nn.Module,
    device: torch.device,
    rank: int,
    outer_optimizer: torch.optim.Optimizer,
    miner_jobs: list[MinerEvalJob],
    score_aggregator: MinerScoreAggregator,
):
    global_model.to(device)
    miner_models: dict[str, nn.Module] = {}
    for miner_job in miner_jobs:
        if score_aggregator.is_in_top(uid=str(miner_job.uid), cutoff=config.run.top_k_miners_to_merge, how="avg"):
            miner_models[miner_job.uid] = await asyncio.to_thread(
                load_model_from_path, miner_job.model_path, global_model, device
            )

    # each validator is only expected to validate 1 expert group at a time
    for _, miner_model in miner_models.items():
        pre_grad_sum = sum_model_gradients(global_model)
        populate_global_grads_from_local(global_model, miner_model, weight=1 / len(miner_models))
        post_grad_sum = sum_model_gradients(global_model)
        logger.info(
            "Miner gradient aggregated",
            pre_grad_sum=round(pre_grad_sum, 6),
            post_grad_sum=round(post_grad_sum, 6),
            grad_delta=round(post_grad_sum - pre_grad_sum, 6),
        )

def sync_grad_across_validators(
    config: ValidatorConfig,
    group_averagers: dict[str | int, DecentralizedAverager],
    group_grad_buff_meta: dict[str | int, Any],
    model,
):
    for group_id, avg in group_averagers.items():
        if avg.total_size <= 0:
            logger.debug("Skipping averager — no peers", group=group_id, mode=avg.mode, total_size=avg.total_size)
            continue

        pack_grads(group_grad_buff_meta[group_id], model)

        grad_sum = sum_model_gradients(model)

        group_bits = avg.get_group_bits()

        peer_count = max(0, avg.total_size)

        logger.info(
            "Starting gradient sync across validators",
            group=group_id,
            mode=avg.mode,
            matchmaking_key=f"{avg.prefix}/{group_bits}",
            peers_visible=peer_count,
        )
        if peer_count == 0:
            logger.warning("No visible peers for gradient sync", group=group_id)
        logger.debug(
            "Averager details",
            group=group_id,
            target_group_size=getattr(avg, "target_group_size", None),
            min_group_size=getattr(avg, "min_group_size", None),
            client_mode=getattr(avg, "client_mode", None),
        )
        
        avg_step = None
        for attempt in range(1, config.run.averager_step_max_retries + 1):
            try:
                avg_step = avg.step(
                    gather={"grad_sum": grad_sum},
                    timeout=config.run.averager_step_timeout_sec,
                    allow_retries=False,
                    wait=True,
                    # scheduled_time=scheduled_time.timestamp()
                )
                VALIDATOR_AVG_STEP_STATUS.labels(status="success").inc()
                break
            except TimeoutError as e:
                logger.warning(f"Averager - Timeout during avg.step (attempt {attempt}/{config.run.averager_step_max_retries}): {e}")
                VALIDATOR_AVG_STEP_STATUS.labels(status="timeout").inc()
            except Exception as e:
                logger.warning(f"Averager - Unexpected error during avg.step (attempt {attempt}/{config.run.averager_step_max_retries}): {e}")
                VALIDATOR_AVG_STEP_STATUS.labels(status="error").inc()
                break

        unpack_to_grads(group_grad_buff_meta[group_id], model)

        after_sum = sum_model_gradients(model)
        logger.info(
            "Gradient sync complete" if avg_step else "Gradient sync failed — no group found",
            group=group_id,
            mode=avg.mode,
            before_grad_sum=round(grad_sum, 6),
            after_grad_sum=round(after_sum, 6),
        )

    return


def run_global_optimization(
    model: nn.Module,
    global_model: nn.Module,
    device: torch.device,
    rank: int,
    outer_optimizer: torch.optim.Optimizer,
    miner_jobs: list[MinerEvalJob],
    score_aggregator: MinerScoreAggregator,
):
    # --- sync + outer step ---
    # keep global model on device for syncing/stepping, then move back to CPU
    for state in outer_optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device, non_blocking=True)

    global_model.to(device)

    old_shared_name, old_shared_sum = get_weight_sum(global_model, shared=True)
    old_expert_name, old_expert_sum = get_weight_sum(global_model, shared=False)

    logger.debug("start syncing shared weights")

    outer_optimizer.step()
    outer_optimizer.zero_grad()

    # copy updated global (partial) weights back into the base (full) model
    # strict=False because global_model has fewer experts than base_model
    with torch.no_grad():
        model.load_state_dict(global_model.state_dict(), strict=False)

    new_shared_name, new_shared_sum = get_weight_sum(global_model, shared=True)
    new_expert_name, new_expert_sum = get_weight_sum(global_model, shared=False)

    shared_delta = round(float(new_shared_sum - old_shared_sum), 6)
    expert_delta = round(float(new_expert_sum - old_expert_sum), 6)
    
    logger.info(
        "Outer optimizer step complete",
        shared_param=old_shared_name,
        shared_delta=shared_delta,
        expert_param=old_expert_name,
        expert_delta=expert_delta,
    )

    for state in outer_optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cpu()

    torch.cuda.empty_cache()


def run(rank: int, world_size: int, config: ValidatorConfig) -> None:
    """
    The worker function for training in a distributed setting.

    Args:
        rank (int): The rank of the process.
        world_size (int): The total number of processes.
        config (Config): The configuration object for the training.

    Returns:
        None
    """
    # Start the integrated Prometheus telemetry server
    # Port 8200+rank to avoid conflicts with other services on this host
    telemetry_port = 8200 + rank
    TelemetryManager().start_server(port=telemetry_port)
    
    if rank == 0:
        config.write()

    torch.cuda.memory._record_memory_history(enabled=True)

    # === create checkpoint directory ===
    os.makedirs(config.ckpt.base_checkpoint_path, exist_ok=True)
    os.makedirs(config.ckpt.checkpoint_path, exist_ok=True)
    os.makedirs(config.log.base_metric_path, exist_ok=True)
    os.makedirs(config.ckpt.miner_submission_path, exist_ok=True)

    # === set up chain worker ===
    wallet, subtensor = setup_chain_worker(config, serve=False)

    # === set logging ===
    metric_logger = MetricLogger(config, rank)

    # === mis ===
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    tokenizer = get_base_tokenizer(config)

    eval_dataloader = get_dataloader(
        config,
        rank=0,
        world_size=config.dataloader.world_size,
        tokenizer=tokenizer,
    )

    # === set up training ===
    (
        base_model,
        global_model,
        outer_optimizer,
        outer_scaler,
        start_step,
        expert_manager,
        train_dataloader,
    ) = setup_training(config, rank, device, tokenizer, subtensor, wallet, current_model_meta=None)

    global_opt_step = start_step

    # === set up score aggregator ===
    score_path = config.ckpt.checkpoint_path / "score_aggregator.json"
    if score_path.exists():
        try:
            with open(score_path, "r") as f:
                score_aggregator = MinerScoreAggregator.from_json(f.read())
            logger.info("Loaded previous MinerScoreAggregator state from disk")
        except Exception as e:
            logger.warning(f"Failed to load score_aggregator.json, starting fresh: {e}")
            score_aggregator = MinerScoreAggregator()
    else:
        score_aggregator = MinerScoreAggregator()

    # === set up averager ===
    group_grad_buff_meta = build_grad_buff_from_model(
        model=global_model, expert_group_assignment=expert_manager.expert_group_assignment
    )
    # Only keep this validator's expert group and shared; drop other groups
    active_group_id = config.task.exp.group_id
    excluded = [gid for gid in group_grad_buff_meta if gid != active_group_id and gid != "shared"]
    for gid in excluded:
        logger.info("Disabling averager for non-active expert group", excluded_group_id=gid, active_group_id=active_group_id)
        del group_grad_buff_meta[gid]

    dht = connect_with_peers(config, wallet, subtensor)

    group_averagers = build_averagers_from_buff(group_buff_metas=group_grad_buff_meta, dht=dht)

    # Start telemetry sidecar poller
    poller = SystemStatePoller(
        subtensor=subtensor, 
        phase_manager=PhaseManager(config, subtensor),
        group_averagers=group_averagers,
        interval_sec=12.0
    )
    poller.start()


    # === commit status ===
    commit_status(
        config,
        wallet,
        subtensor,
        ValidatorChainCommit(
            model_hash=None,
            global_ver=global_opt_step,
            expert_group=config.task.exp.group_id,
            miner_seed=0,  # this should reveal later
            block=subtensor.block,
        ),
    )

    # === training ===
    loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
    aux_loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
    training_time = 0
    total_training_time = 0

    outer_optimizer.zero_grad()

    current_model_hash = None

    try:
        while True:

            # for each step, we run 1 backward
            # for each inner_opt_step, we run local optimization; gradient_accumulation_steps = 1 real step
            # for each global_opt_interval number of inner_opt_step, we synchronise weight from different ddp worker, and then run global optimization

            # === Wait till commit phase to submit random seed ===
            phase_response = wait_till(config, PhaseNames.miner_commit_1)
            global_opt_step = phase_response.phase_start_block
            logger.info("(0) Commit new seed for next validation")

            # Submit fallback weights if last_update is stale (past max_weight_age)
            max_weight_age = int(config.cycle.cycle_length)
            metagraph = subtensor.metagraph(netuid=config.chain.netuid)
            my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
            last_update = metagraph.last_update[my_uid].item()
            weight_age = subtensor.block - last_update
            if weight_age > max_weight_age:
                logger.info("Weights stale, submitting fallback", weight_age=weight_age, max_weight_age=max_weight_age)
                _submit_fallback_weights(config, wallet, subtensor, wait_for_inclusion=True, wait_for_finalization=True)

            commit_status(
                config,
                wallet,
                subtensor,
                ValidatorChainCommit(
                    model_hash=current_model_hash,
                    global_ver=global_opt_step,
                    expert_group=config.task.exp.group_id,
                    miner_seed=secrets.randbits(16),
                    block=subtensor.block,
                ),
            )

            check_phase_expired(subtensor, phase_response)

            # === Wait till validation phase to start the validation procedure ===
            phase_response = wait_till(config, PhaseNames.validate)
            
            # === Get miner ===
            logger.info("(1) Gathering miner jobs")
            miner_jobs = gather_validation_job(config, subtensor, step=global_opt_step)
            logger.debug("Miner jobs collected", step=global_opt_step, count=len(miner_jobs))
            if len(miner_jobs) == 0:
                logger.warning("No miner jobs to evaluate", step=global_opt_step)

            # === Get miner model and evaluate the miners ===
            logger.info("(2) Evaluating miners")
            cleanup(global_model, base_model)
            asyncio.run(
                run_evaluation(
                    config=config,
                    step=global_opt_step,
                    device=device,  # operate at cuda
                    miners=miner_jobs,
                    score_aggregator=score_aggregator,
                    base_model=global_model.to("cpu"),
                    tokenizer=tokenizer,
                    combinded_seed=get_combined_validator_seed(config, subtensor),
                )
            )

            cleanup(global_model, base_model)
            
            # Penalize assigned miners that missed their submission
            from connito.shared.cycle import get_validator_miner_assignment
            validator_miner_assignment = get_validator_miner_assignment(config, subtensor)
            miner_assignment = validator_miner_assignment.get(config.chain.hotkey_ss58, [])
            submitted_uids = {job.uid for job in miner_jobs}
            metagraph = subtensor.metagraph(netuid=config.chain.netuid)
            for expected_hotkey in miner_assignment:
                try:
                    uid = metagraph.hotkeys.index(expected_hotkey)
                    if uid not in submitted_uids:
                        score_aggregator.add_score(uid=str(uid), hotkey=expected_hotkey, score=0.0)
                        logger.info("Penalizing missing submission", uid=uid, hotkey=expected_hotkey[:6], score=0.0)
                except ValueError:
                    continue

            uid_scores = score_aggregator.uid_score_pairs()
            logger.info(
                "Evaluation results",
                miners_evaluated=len(uid_scores),
                scores={uid: round(s, 4) for uid, s in uid_scores.items()},
            )

            # Persist aggregator state locally for restarts
            try:
                with open(score_path, "w") as f:
                    f.write(score_aggregator.to_json())
            except Exception as e:
                logger.warning(f"Failed to save score_aggregator.json: {e}")

            # === aggragate miner gradient change locally ===
            # Use global_model (partial) as template for loading miner checkpoints (also partial)
            logger.info("(3) Aggregating miner gradient change locally")
            asyncio.run(
                aggregate_miner_gradient_change(
                    config=config,
                    global_model=global_model.to("cpu"),
                    device=torch.device("cpu"),  # all gradient aggregation done on cpu
                    rank=rank,
                    outer_optimizer=outer_optimizer,
                    miner_jobs=miner_jobs,
                    score_aggregator=score_aggregator,
                )
            )

            cleanup(global_model, base_model)

            check_phase_expired(subtensor, phase_response)

            # === wait till merging phase and aggragate miner gradient change ===
            phase_response = wait_till(config, PhaseNames.merge)
            logger.info("(4) Syncing gradient across validators")
            sync_grad_across_validators(config=config, group_averagers=group_averagers, group_grad_buff_meta=group_grad_buff_meta, model=global_model)

            # === global optimizer ===
            logger.info("(5) Running global model optimization step")

            org_model_hash = get_model_hash(global_model.state_dict(), hex=True)

            run_global_optimization(
                model=base_model,
                global_model=global_model.to("cpu"),
                device=device,
                rank=rank,
                outer_optimizer=outer_optimizer,
                miner_jobs=miner_jobs,
                score_aggregator=score_aggregator,
            )

            logger.info(
                "Optimization step complete",
                org_model_hash=org_model_hash,
                new_model_hash=get_model_hash(global_model.state_dict(), hex=True)[:6],
                new_base_model_hash=get_model_hash(base_model.state_dict(), hex=True)[:6],
            )

            cleanup(global_model, base_model)

            # === save checkpoint ===
            logger.info("(6) Saving checkpoint")
            ckpt_path = config.ckpt.checkpoint_path / f"globalver_{int(global_opt_step)}"

            save_checkpoint(
                checkpoint_path=ckpt_path,
                model=global_model,
                outer_optimizer=outer_optimizer,
                loss=loss_batch.item(),
                outer_scaler=outer_scaler,
                data_loader=train_dataloader,
                save_global_state=rank == 0,
                rank=rank,
                expert_manager=expert_manager,
                save_model_by_expert_group=True,
                strict_sharding=get_nested_attr(config, "ckpt.strict_sharding", False),
                active_expert_group_id=config.task.exp.group_id,
            )

            check_phase_expired(subtensor, phase_response)

            # === Comit to chain for new model ===
            model_ckpt = build_local_checkpoint(ckpt_path)
            if model_ckpt is not None:

                model_ckpt.expert_group = config.task.exp.group_id
                model_ckpt.sign_hash(wallet=wallet)
                current_model_hash = model_ckpt.model_hash
                phase_response = wait_till(config, PhaseNames.validator_commit_1)
                logger.info("(7) Commit new signed_model_hash for next validation")
                commit_status(
                    config,
                    wallet,
                    subtensor,
                    SignedModelHashChainCommit(
                        signed_model_hash=model_ckpt.signed_model_hash,
                    ),
                )

                check_phase_expired(subtensor, phase_response)

                phase_response = wait_till(config, PhaseNames.validator_commit_2)
                logger.info("(8) Commit model_hash for next validation")
                commit_status(
                    config,
                    wallet,
                    subtensor,
                    ValidatorChainCommit(
                        model_hash=model_ckpt.model_hash,
                        global_ver=model_ckpt.global_ver,
                        expert_group=config.task.exp.group_id,
                        miner_seed=0,
                        block=subtensor.block,
                    ),
                )

                if config.ckpt.checkpoint_topk is not None:
                    ckpt_deleted = delete_old_checkpoints(config.ckpt.checkpoint_path, config.ckpt.checkpoint_topk)
                    if ckpt_deleted:
                        logger.debug(f"Deleted old checkpoints: {ckpt_deleted}")

            # === Set weight to chain ===
            logger.info("(9) Set weight to chain")
            uid_weights = score_aggregator.uid_score_pairs(how="avg")
            logger.info(
                "Submitting weights derived from avg scores",
                top_weights={str(k): round(v, 4) for k, v in sorted(uid_weights.items(), key=lambda item: item[1], reverse=True)[:5]}
            )
            submit_weights(
                config=config,
                wallet=wallet,
                subtensor=subtensor,
                uid_weights=uid_weights,
                normalize=True,
                top_k=config.run.top_k_miners_to_reward,
            )

            # === archive top-k submissions, delete the rest ===
            logger.info("(11) Archiving top miner submissions")
            archive_top_miner_submissions(
                submission_dir=config.ckpt.miner_submission_path,
                archive_dir=config.ckpt.miner_submission_archive_path,
                score_aggregator=score_aggregator,
                top_k=config.run.top_k_miners_to_reward,
            )

            # === validation and log metric ===
            logger.info("(10) Running local evaluation")
            val_metric = evaluate_model(
                rank=rank,
                step=global_opt_step,
                model=global_model.to("cpu"),
                eval_dataloader=eval_dataloader,
                device=device,
            )

            metrics = (
                get_status(
                    config=config,
                    model=base_model,
                    step=global_opt_step,
                    training_time=training_time,
                    total_training_time=total_training_time,
                    inner_opt_step=None,
                    global_opt_step=global_opt_step,
                    loss_batch=loss_batch,
                    aux_loss_batch=aux_loss_batch,
                )
                | val_metric
            )

            metric_logger.log(metrics)
            cleanup(global_model, base_model)

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received, shutting down validator loop")
        poller.stop()
        cleanup(global_model, base_model)
        metric_logger.close()
        for _, a in group_averagers.items():
            a.shutdown()
        raise
    except Exception:
        logger.error("Quit training", exc_info=True)
        poller.stop()
        cleanup(global_model, base_model)
        metric_logger.close()
        for _, a in group_averagers.items():
            a.shutdown()

        if rank == 0:
            torch.save(global_model.state_dict(), "mycelia_final.pt")


if __name__ == "__main__":
    args = parse_args()

    if args.path:
        config = ValidatorConfig.from_path(args.path, auto_update_config=args.auto_update_config)
    else:
        config = ValidatorConfig()

    run(0, 1, config)
