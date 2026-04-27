import asyncio
import ctypes
import gc
import math
import os
import threading
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from dotenv import load_dotenv

load_dotenv()

from typing import Any


def _get_build_version() -> tuple[str, str]:
    """Return (version, git_sha).

    Precedence for `version`:
      1. CONNITO_GIT_VERSION env (baked into the Docker image by CI; matches
         the docker tag — e.g. "1.2.3", "master", "staging").
      2. `git describe --tags --always` in a source checkout (e.g. "v1.2.3-5-gabc1234").
      3. pyproject.toml version via installed metadata (e.g. "0.1.0").

    Precedence for `git_sha`:
      1. CONNITO_GIT_SHA env (baked into the Docker image).
      2. `git rev-parse HEAD` in a source checkout.
      3. "unknown".
    """
    import subprocess
    from pathlib import Path

    def _git(*args) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=Path(__file__).resolve().parent,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return ""

    version = os.environ.get("CONNITO_GIT_VERSION", "")
    if not version or version == "unknown":
        version = _git("describe", "--tags", "--always", "--dirty")
    if not version:
        try:
            version = _pkg_version("subnet-moe")
        except PackageNotFoundError:
            version = "unknown"

    sha = os.environ.get("CONNITO_GIT_SHA", "")
    if not sha or sha == "unknown":
        sha = _git("rev-parse", "HEAD") or "unknown"

    return version, sha

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
    VALIDATOR_COMMIT_MAX_HF_REPO_ID_CHARS,
    validate_validator_chain_commit_payload,
    setup_chain_worker,
)
from connito.shared.checkpoint_helper import (
    cleanup_temporary_checkpoint_dirs,
    load_checkpoint,
    save_checkpoint,
)
from connito.shared.checkpoints import (
    ModelCheckpoint,
    archive_top_miner_submissions,
    build_local_checkpoint,
    delete_old_checkpoints,
    prune_miner_submission_files,
    select_best_checkpoint,
)
from connito.shared.config import ValidatorConfig, parse_args
from connito.shared.hf_distribute import (
    get_hf_upload_readiness,
    resolve_hf_repo_ids,
    upload_checkpoint_to_hf,
)
from connito.shared.cycle import (
    check_phase_expired,
    wait_till,
)
from connito.shared.dataloader import get_dataloader
from connito.shared.expert_manager import (
    ExpertManager,
    get_weight_sum,
    populate_global_grads_from_local,
)
from connito.shared.helper import get_model_hash, get_nested_attr, sum_model_gradients
from connito.shared.metrics import MetricLogger
from connito.shared.model import load_model, reload_model_inplace
from connito.shared.modeling.mycelia import get_base_tokenizer
from connito.sn_owner.cycle import PhaseNames, PhaseManager
from connito.validator.aggregator import MinerScoreAggregator
from connito.validator.background_download_worker import BackgroundDownloadWorker
from connito.validator.background_eval_worker import BackgroundEvalWorker
from connito.validator.chain_submitter import ChainSubmitter
from connito.validator.evaluator import (
    MinerEvalJob,
    evaluate_foreground_round,
    load_model_from_path,
)
from connito.validator.round import Round, RoundRef
HF_CHAIN_REVISION_LENGTH = 7


def validate_hf_distribution_config(config: ValidatorConfig) -> tuple[str | None, str | None]:
    hf_upload_repo_id, hf_chain_repo_id = resolve_hf_repo_ids(
        config.hf,
        max_chain_repo_chars=VALIDATOR_COMMIT_MAX_HF_REPO_ID_CHARS,
    )

    if not (hf_upload_repo_id and hf_chain_repo_id):
        return hf_upload_repo_id, hf_chain_repo_id

    validate_validator_chain_commit_payload(
        ValidatorChainCommit(
            model_hash="0" * 64,
            global_ver=0,
            expert_group=config.task.exp.group_id,
            hf_repo_id=hf_chain_repo_id,
            hf_revision="0" * HF_CHAIN_REVISION_LENGTH,
        )
    )

    if config.hf.uses_explicit_checkpoint_repo():
        logger.info(
            "Using configured HF checkpoint repo",
            upload_checkpoint_repo=hf_upload_repo_id,
            advertised_checkpoint_repo=hf_chain_repo_id,
        )
    else:
        logger.info(
            "Using default HF checkpoint repo derived from authenticated user",
            upload_checkpoint_repo=hf_upload_repo_id,
            advertised_checkpoint_repo=hf_chain_repo_id,
        )

    return hf_upload_repo_id, hf_chain_repo_id


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
    VALIDATOR_ROUND_LIFECYCLE_STEP,
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


try:
    _LIBC = ctypes.CDLL("libc.so.6")
    _LIBC.malloc_trim.argtypes = [ctypes.c_size_t]
    _LIBC.malloc_trim.restype = ctypes.c_int
except OSError:
    _LIBC = None


def _release_cpu_ram() -> None:
    """Ask glibc to return freed arenas to the OS."""
    if _LIBC is not None:
        try:
            _LIBC.malloc_trim(0)
        except Exception:
            pass


def cleanup(global_model) -> None:
    """
    Reclaim cached allocator memory. global_model stays resident on GPU.
    """
    _cuda_mem_report("VRAM before GPU cleanup")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    _release_cpu_ram()

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
    # global_model: partial model (only assigned experts) — used for optimization and evaluation
    global_model, model_meta = load_model(
        rank, config, expert_manager, subtensor, wallet, current_model_meta,
        partial=True, checkpoint_device=device,
    )

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
) -> list[str]:
    # global_model is expected to already live on `device` (GPU).
    this_round_uids = {job.uid for job in miner_jobs}

    # Drop zero-score miners before ranking so they can never be merged, even
    # when fewer than top_k miners have a positive score this round. Use the
    # latest (this-round) score, not the rolling avg, so a single bad round
    # is enough to exclude a miner regardless of their history.
    latest_scores = score_aggregator.uid_score_pairs(how="latest")
    scored_jobs = [job for job in miner_jobs if latest_scores.get(job.uid, 0.0) > 0]
    skipped_zero_uids = [job.uid for job in miner_jobs if job not in scored_jobs]
    if skipped_zero_uids:
        logger.info("Excluding zero-score miners from merge", uids=skipped_zero_uids)

    top_jobs = [
        job for job in scored_jobs
        if score_aggregator.is_in_top(
            uid=job.uid,
            cutoff=config.evaluation.top_k_miners_to_merge,
            how="avg",
            among=this_round_uids,
        )
    ]
    weight = 1 / max(1, len(top_jobs))
    merged_uids: list[str] = []

    # Stream one miner at a time: load → aggregate into global_model → release.
    # Keeping all top-k miner models resident on CPU simultaneously was the
    # single largest transient RAM spike in the cycle.
    for job in top_jobs:
        miner_model = await asyncio.to_thread(
            load_model_from_path, job.model_path, global_model, device
        )
        try:
            pre_grad_sum = sum_model_gradients(global_model)
            populate_global_grads_from_local(global_model, miner_model, weight=weight)
            post_grad_sum = sum_model_gradients(global_model)
            # Check element-wise for inf/nan rather than testing the sum,
            # because abs().sum() in bf16 can overflow to inf even when
            # individual gradient elements are merely large but finite.
            grad_has_nonfinite = any(
                torch.any(torch.isinf(p.grad) | torch.isnan(p.grad)).item()
                for p in global_model.parameters()
                if p.grad is not None
            )
            if grad_has_nonfinite:
                logger.warning(
                    "Non-finite gradient elements after merging miner — zeroing all gradients and skipping miner",
                    uid=job.uid,
                    pre_grad_sum=pre_grad_sum,
                    post_grad_sum=post_grad_sum,
                )
                # Zero out all accumulated .grad tensors so the poisoned
                # gradient does not propagate to the allreduce or optimizer.
                for p in global_model.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                logger.info(
                    "Miner gradient aggregated",
                    uid=job.uid,
                    pre_grad_sum=round(pre_grad_sum, 6),
                    post_grad_sum=round(post_grad_sum, 6),
                    grad_delta=round(post_grad_sum - pre_grad_sum, 6),
                )
                merged_uids.append(str(job.uid))
        finally:
            del miner_model
            gc.collect()
            _release_cpu_ram()

    return merged_uids

def sync_grad_across_validators(
    config: ValidatorConfig,
    group_averagers: dict[str | int, DecentralizedAverager],
    group_grad_buff_meta: dict[str | int, Any],
    model,
):
    for group_id, avg in group_averagers.items():
        # avg.total_size is the number of tensor *elements* in the grad buffer,
        # not the peer count. Skip only if the buffer is empty (should never happen).
        if avg.total_size <= 0:
            logger.debug("Skipping averager — grad buffer is empty", group=group_id, mode=avg.mode)
            continue

        pack_grads(group_grad_buff_meta[group_id], model)

        grad_sum = sum_model_gradients(model)

        group_bits = avg.get_group_bits()

        logger.info(
            "Starting gradient sync across validators",
            group=group_id,
            mode=avg.mode,
            matchmaking_key=f"{avg.prefix}/{group_bits}",
            grad_buffer_elements=avg.total_size,
        )
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
                    gather={"grad_sum": grad_sum, "hotkey": config.chain.hotkey_ss58},
                    timeout=config.run.averager_step_timeout_sec,
                    allow_retries=True,
                    wait=True,
                    # scheduled_time=scheduled_time.timestamp()
                )
                gathered = {}
                if hasattr(avg_step, "items"):
                    gathered = {
                        str(peer): {
                            "hotkey": vals.get("hotkey") if isinstance(vals, dict) else None,
                            "grad_sum": vals.get("grad_sum") if isinstance(vals, dict) else vals,
                        }
                        for peer, vals in avg_step.items()
                    }
                logger.info(
                    "Averager step succeeded",
                    group=group_id,
                    our_hotkey=config.chain.hotkey_ss58[-6:],
                    our_grad_sum=round(grad_sum, 6),
                    peers=gathered,
                    group_size=len(gathered),
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
    global_model: nn.Module,
    device: torch.device,
    rank: int,
    outer_optimizer: torch.optim.Optimizer,
    miner_jobs: list[MinerEvalJob],
    score_aggregator: MinerScoreAggregator,
):
    # global_model and outer_optimizer state are expected to already live on `device` (GPU).
    old_shared_name, old_shared_sum = get_weight_sum(global_model, shared=True)
    old_expert_name, old_expert_sum = get_weight_sum(global_model, shared=False)

    logger.debug("start syncing shared weights")

    outer_optimizer.step()
    outer_optimizer.zero_grad()

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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(rank: int, world_size: int, config: ValidatorConfig, pkg_version: str = "") -> None:
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
        logger.info("Loaded config", config=config.model_dump_json(indent=2))
        config.write()

    # CUDA allocation history recording leaks RAM on long-running loops —
    # enable only when profiling via run.record_cuda_mem_history in config.
    if config.run.record_cuda_mem_history:
        torch.cuda.memory._record_memory_history(enabled=True)

    # === create checkpoint directory ===
    os.makedirs(config.ckpt.base_checkpoint_path, exist_ok=True)
    os.makedirs(config.ckpt.checkpoint_path, exist_ok=True)
    os.makedirs(config.log.base_metric_path, exist_ok=True)
    os.makedirs(config.ckpt.miner_submission_path, exist_ok=True)

    # === set up chain worker ===
    # subtensor: archive connection — required by callers that issue
    # historical block queries (Round.freeze, setup_training/load_model,
    # reload_model_inplace, evaluate_foreground_round, eval factory).
    # lite_subtensor: sync Subtensor for head-only reads (metagraph,
    # current block, peer connect, phase checks).
    # chain_submitter: owns an AsyncSubtensor + AsyncRunner; handles every
    # non-blocking commit_status / set_weights call for this validator.
    validate_hf_distribution_config(config)
    wallet, subtensor, lite_subtensor = setup_chain_worker(config, serve=False)
    chain_submitter = ChainSubmitter(
        config,
        wallet,
        normalize=True,
        top_k=config.evaluation.top_k_miners_to_reward,
    )

    # === set logging ===
    metric_logger = MetricLogger(config, rank)

    # === mis ===
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    tokenizer = get_base_tokenizer(config)

    # eval_dataloader is built lazily inside the eval step so its worker
    # processes / prefetched batches don't stay resident across the whole cycle.

    # === set up training ===
    (
        global_model,
        outer_optimizer,
        outer_scaler,
        start_step,
        expert_manager,
        train_dataloader,
    ) = setup_training(config, rank, device, tokenizer, subtensor, wallet, current_model_meta=None)

    global_opt_step = start_step
    # Tracks whether this validator participated in the last allreduce.
    # If False at the start of the next cycle, pull the updated model from a
    # peer validator before continuing.
    _participated_in_merge = True

    # === set up score aggregator ===
    score_window = config.evaluation.score_window
    score_path = config.ckpt.checkpoint_path / "score_aggregator.json"
    if pkg_version == "v0.1.2":
        logger.info("Skipping historic score_aggregator load for v0.1.2", pkg_version=pkg_version)
        score_aggregator = MinerScoreAggregator(max_points=score_window)
    elif score_path.exists():
        try:
            with open(score_path, "r") as f:
                score_aggregator = MinerScoreAggregator.from_json(f.read(), max_points=score_window)
            logger.info("Loaded previous MinerScoreAggregator state from disk")
        except Exception as e:
            logger.warning(f"Failed to load score_aggregator.json, starting fresh: {e}")
            score_aggregator = MinerScoreAggregator(max_points=score_window)
    else:
        score_aggregator = MinerScoreAggregator(max_points=score_window)

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

    dht = connect_with_peers(config, wallet, lite_subtensor)

    group_averagers = build_averagers_from_buff(group_buff_metas=group_grad_buff_meta, dht=dht)

    # Start telemetry sidecar poller
    poller = SystemStatePoller(
        subtensor=lite_subtensor,
        phase_manager=PhaseManager(config, lite_subtensor),
        group_averagers=group_averagers,
        interval_sec=12.0
    )
    poller.start()


    # === commit status === (non-blocking; queued on chain_submitter)
    chain_submitter.async_commit(ValidatorChainCommit(
        model_hash=None,
        global_ver=global_opt_step,
        expert_group=config.task.exp.group_id,
    ))

    # === training ===
    loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
    aux_loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
    training_time = 0
    total_training_time = 0

    outer_optimizer.zero_grad()

    current_model_hash = None

    if config.ckpt.cleanup_stale_temporary_checkpoints:
        cleanup_temporary_checkpoint_dirs(config.ckpt.checkpoint_path)

    # === Round-lifecycle scaffolding ===
    # foreground_active: set while the foreground evaluation pass holds GPU/HF.
    # merge_phase_active: set for the entire Merge phase plus briefly around HF upload.
    # eval_window_active: set after Merge(K) completes so the eval worker may
    #   evaluate round K's downloaded miners; cleared at the top of the next
    #   cycle right before submit_weights for round K.
    # download_window_closed: set when the main loop begins waiting for
    #   MinerCommit1 of the next round (round K's downloads are dead weight
    #   past that point); cleared at the next freeze. Pauses bg-download
    #   from MinerCommit1(K+1) → Submission(K+1).
    # gpu_eval_lock: held by the eval worker only across its load_state_dict
    #   and evaluate_one_miner calls (yielded everywhere else; see plan).
    foreground_active = threading.Event()
    merge_phase_active = threading.Event()
    eval_window_active = threading.Event()
    download_window_closed = threading.Event()
    gpu_eval_lock = threading.Lock()
    round_ref = RoundRef()

    download_worker: BackgroundDownloadWorker | None = None
    eval_worker: BackgroundEvalWorker | None = None
    if config.evaluation.background_worker_enabled:
        def _eval_model_factory():
            # Runs on a worker thread (via asyncio.to_thread). Open a
            # dedicated archive Subtensor here so its WebSocket is not
            # shared with the main loop's subtensor — the substrate WS is
            # not safe across threads — and load_model issues historical
            # block queries that only an archive node can serve.
            eval_subtensor = bittensor.Subtensor(network=config.chain.network)
            new_model, _ = load_model(
                rank, config, expert_manager, eval_subtensor, wallet, None,
                partial=True, checkpoint_device=device,
            )
            return new_model

        download_worker = BackgroundDownloadWorker(
            config=config,
            round_ref=round_ref,
            foreground_active=foreground_active,
            merge_phase_active=merge_phase_active,
            download_window_closed=download_window_closed,
        )
        eval_worker = BackgroundEvalWorker(
            config=config,
            round_ref=round_ref,
            model_factory=_eval_model_factory,
            device=device,
            tokenizer=tokenizer,
            score_aggregator=score_aggregator,
            score_path=score_path,
            merge_phase_active=merge_phase_active,
            eval_window_active=eval_window_active,
            gpu_eval_lock=gpu_eval_lock,
        )
        download_worker.start()
        eval_worker.start()
        logger.info("Background download + eval workers started")

    logger.info("ChainSubmitter ready")

    try:
        while True:

            # for each step, we run 1 backward
            # for each inner_opt_step, we run local optimization; gradient_accumulation_steps = 1 real step
            # for each global_opt_interval number of inner_opt_step, we synchronise weight from different ddp worker, and then run global optimization

            # === (0) Re-sync from peer if we were excluded last cycle ===
            if not _participated_in_merge:
                logger.info(
                    "(0) Re-syncing model from peer validator (was excluded from allreduce last cycle)"
                )
                success = reload_model_inplace(
                    config=config,
                    global_model=global_model,
                    expert_manager=expert_manager,
                    device=device,
                    subtensor=subtensor,
                    wallet=wallet,
                )
                if success:
                    logger.info("(0) Peer sync successful — model updated")
                else:
                    logger.warning(
                        "(0) Peer sync failed — continuing with current model; "
                        "weight quality may be reduced this cycle"
                    )
                _participated_in_merge = True  # reset regardless; try allreduce next cycle

            # === Wait till commit phase to submit random seed ===
            # Round K's downloads are stale past this point — the eval window
            # is closed by Merge and weights are about to be submitted below.
            # Pause bg-download until the next freeze re-opens it.
            download_window_closed.set()
            phase_response = wait_till(config, PhaseNames.miner_commit_1)
            global_opt_step = phase_response.phase_start_block
            logger.info("(0) Commit new seed for next validation")

            # === (4) Close the (3) window and submit weights for the round
            # whose evaluation just ended. This used to live at the bottom of
            # the cycle (after ValidatorCommit2); it now happens here, at end
            # of Train(K+1) = start of MinerCommit1(K+1), so the weights
            # reflect both (2) foreground and (3) background scores.
            eval_window_active.clear()
            pending_round: Round | None = round_ref.current
            if pending_round is not None and not pending_round.weights_submitted:
                # Missed-submission penalty pass against the frozen roster.
                for entry in pending_round.unscored_roster_uids():
                    score_aggregator.add_score(
                        uid=entry.uid,
                        hotkey=entry.hotkey,
                        score=0.0,
                        round_id=pending_round.round_id,
                    )
                    logger.info(
                        "Penalizing missing submission",
                        uid=entry.uid, hotkey=entry.hotkey[:6], score=0.0,
                        round_id=pending_round.round_id,
                    )

                logger.info(
                    "(4) Handing weight submission to background submitter",
                    round_id=pending_round.round_id,
                )
                uid_weights = score_aggregator.uid_score_pairs(how="avg")
                # Fire-and-forget. ChainSubmitter sets
                # pending_round.weights_submitted once the chain accepts the call.
                chain_submitter.async_submit_weight(pending_round, uid_weights)

                try:
                    score_aggregator.persist_atomic(score_path)
                except Exception as e:
                    logger.warning("Failed to persist score_aggregator after submit_weights", error=str(e))

            # Submit fallback weights if last_update is stale (past max_weight_age).
            # Fetch the metagraph once per cycle — it holds per-neuron tensors and
            # is reused later for penalizing missing submissions.
            max_weight_age = int(config.cycle.cycle_length)
            metagraph = lite_subtensor.metagraph(netuid=config.chain.netuid)
            my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
            last_update = metagraph.last_update[my_uid].item()
            current_block = lite_subtensor.get_current_block()
            weight_age = current_block - last_update
            if weight_age > max_weight_age:
                logger.info("Weights stale, submitting fallback (non-blocking)",
                            weight_age=weight_age, max_weight_age=max_weight_age)
                # Non-blocking; ChainSubmitter serializes this with the
                # commit_status that follows, so order is preserved.
                chain_submitter.async_submit_fallback_weights()

            chain_submitter.async_commit(ValidatorChainCommit(
                model_hash=current_model_hash,
                global_ver=global_opt_step,
                expert_group=config.task.exp.group_id,
            ))

            check_phase_expired(lite_subtensor, phase_response)

            # === Wait till Submission phase; freeze the round and start
            # foreground evaluation of the top-N (step 2). The round is the
            # unit of work for the rest of the lifecycle: download worker
            # picks up its background_uids, eval worker waits for
            # eval_window_active to open after Merge.
            phase_response = wait_till(config, PhaseNames.submission)

            logger.info(
                "(0) Submission phase entered — freezing round",
                submission_start=phase_response.phase_start_block,
                submission_end=phase_response.phase_end_block,
                current_block=lite_subtensor.block,
            )

            cleanup(global_model)

            # (0) Lock and prioritize: build the round roster (incentive-ranked,
            # restricted to this validator's assignment), capture the seed, and
            # snapshot global_model.state_dict() to CPU before Merge can mutate it.
            new_round = Round.freeze(
                config=config,
                subtensor=subtensor,
                metagraph=metagraph,
                global_model=global_model,
                round_id=phase_response.phase_start_block,
            )
            round_ref.swap(new_current=new_round)
            download_window_closed.clear()
            try:
                VALIDATOR_ROUND_LIFECYCLE_STEP.labels(round_id=str(new_round.round_id)).set(0)
            except Exception:
                pass

            # (2) Foreground evaluation: top-N miners only, by incentive,
            # bounded by per_miner_eval_timeout_sec; spillover lands in (3).
            foreground_active.set()
            try:
                miner_jobs = asyncio.run(
                    evaluate_foreground_round(
                        config=config,
                        round_obj=new_round,
                        subtensor=subtensor,
                        step=global_opt_step,
                        device=device,
                        score_aggregator=score_aggregator,
                        base_model=global_model,
                        tokenizer=tokenizer,
                        end_block=phase_response.phase_end_block,
                        per_miner_eval_timeout_sec=float(config.evaluation.per_miner_eval_timeout_sec),
                    )
                )
            finally:
                foreground_active.clear()
            try:
                VALIDATOR_ROUND_LIFECYCLE_STEP.labels(round_id=str(new_round.round_id)).set(2)
            except Exception:
                pass

            phase_response = wait_till(config, PhaseNames.validate)

            logger.info("(2) Foreground evaluation complete", evaluated=len(miner_jobs))
            if len(miner_jobs) == 0:
                logger.warning("No foreground miners evaluated", round_id=new_round.round_id)

            cleanup(global_model)

            # Logging — show scores for foreground miners only; the (3)
            # background scores accumulate after Merge.
            submitted_uids = {job.uid for job in miner_jobs}
            all_latest = score_aggregator.uid_score_pairs(how="latest")
            round_scores = {uid: round(s, 4) for uid, s in all_latest.items() if uid in submitted_uids}
            logger.info(
                "Foreground evaluation results",
                miners_evaluated=len(submitted_uids),
                round_id=new_round.round_id,
                scores=round_scores,
            )

            # Persist aggregator state atomically.
            try:
                score_aggregator.persist_atomic(score_path)
            except Exception as e:
                logger.warning(f"Failed to persist score_aggregator: {e}")

            # === aggragate miner gradient change locally ===
            # Use global_model (partial) as template for loading miner checkpoints (also partial)
            logger.info("(3) Aggregating miner gradient change locally")
            merged_uids = asyncio.run(
                aggregate_miner_gradient_change(
                    config=config,
                    global_model=global_model,
                    device=device,  # gradient aggregation runs on GPU
                    rank=rank,
                    outer_optimizer=outer_optimizer,
                    miner_jobs=miner_jobs,
                    score_aggregator=score_aggregator,
                )
            )

            grad_sum_after_aggregation = sum_model_gradients(global_model)
            # Use element-wise check: the sum can overflow bf16 to inf even
            # when no individual element is actually non-finite.
            grad_has_nonfinite_elements = any(
                torch.any(torch.isinf(p.grad) | torch.isnan(p.grad)).item()
                for p in global_model.parameters()
                if p.grad is not None
            )
            grad_is_valid = bool(merged_uids) and not grad_has_nonfinite_elements

            logger.info(
                "Aggregated miner gradients locally",
                merged_uids=merged_uids,
                grad_sum=round(grad_sum_after_aggregation, 6) if math.isfinite(grad_sum_after_aggregation) else str(grad_sum_after_aggregation),
                grad_is_valid=grad_is_valid,
                model_hash=get_model_hash(global_model.state_dict(), hex=True)[:6],
            )

            if not grad_is_valid:
                logger.warning(
                    "Invalid gradient state after local aggregation — "
                    "skipping allreduce and optimizer this cycle; "
                    "will pull updated model from peer at start of next cycle",
                    merged_uids=merged_uids,
                    grad_sum=grad_sum_after_aggregation,
                )
                outer_optimizer.zero_grad()  # ensure clean state

            cleanup(global_model)

            check_phase_expired(lite_subtensor, phase_response)

            # === wait till merging phase and aggregate miner gradient change ===
            phase_response = wait_till(config, PhaseNames.merge)

            # Suspend both background workers for the entire Merge window —
            # they share GPU and DHT resources with sync_grad_across_validators
            # and run_global_optimization.
            merge_phase_active.set()
            try:
                if grad_is_valid:
                    logger.info("(4) Syncing gradient across validators")
                    sync_grad_across_validators(
                        config=config,
                        group_averagers=group_averagers,
                        group_grad_buff_meta=group_grad_buff_meta,
                        model=global_model,
                    )

                    # === global optimizer ===
                    logger.info("(5) Running global model optimization step")

                    org_model_hash = get_model_hash(global_model.state_dict(), hex=True)

                    run_global_optimization(
                        global_model=global_model,
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
                    )
                    _participated_in_merge = True
                else:
                    logger.info(
                        "(4/5) Skipping gradient sync and optimizer — "
                        "no valid gradient contribution this cycle"
                    )
                    _participated_in_merge = False

                cleanup(global_model)

                # === save checkpoint ===
                logger.info("(6) Saving checkpoint")
                ckpt_path = config.ckpt.checkpoint_path / f"globalver_{int(global_opt_step)}"

                presave_keep = None
                if config.ckpt.checkpoint_topk is not None:
                    presave_keep = max(config.ckpt.checkpoint_topk - 1, 0)
                if presave_keep is not None:
                    presave_deleted = delete_old_checkpoints(config.ckpt.checkpoint_path, presave_keep)
                    if presave_deleted:
                        logger.info(
                            "Pruned older checkpoints before save",
                            keep=presave_keep,
                            deleted=presave_deleted,
                        )

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
            finally:
                merge_phase_active.clear()

            # (3) Open the eval window for the round we just merged. The eval
            # worker uses round.model_snapshot_cpu (taken at freeze time) so
            # the post-Merge mutation of global_model does not affect it.
            eval_window_active.set()
            try:
                VALIDATOR_ROUND_LIFECYCLE_STEP.labels(round_id=str(new_round.round_id)).set(3)
            except Exception:
                pass

            check_phase_expired(lite_subtensor, phase_response)

            # === Comit to chain for new model ===
            model_ckpt = build_local_checkpoint(ckpt_path)
            if model_ckpt is not None:

                model_ckpt.expert_group = config.task.exp.group_id
                model_ckpt.sign_hash(wallet=wallet)
                current_model_hash = model_ckpt.model_hash
                phase_response = wait_till(config, PhaseNames.validator_commit_1)
                logger.info("(7) Commit new signed_model_hash for next validation (non-blocking)")
                chain_submitter.async_commit(SignedModelHashChainCommit(
                    signed_model_hash=model_ckpt.signed_model_hash,
                ))

                check_phase_expired(lite_subtensor, phase_response)

                # Upload checkpoint to HuggingFace so miners can pull it during
                # the Distribute phase. The returned revision SHA pins the exact
                # bytes miners will download, even if :main advances afterward.
                hf_upload_repo_id, hf_chain_repo_id = resolve_hf_repo_ids(
                    config.hf,
                    max_chain_repo_chars=VALIDATOR_COMMIT_MAX_HF_REPO_ID_CHARS,
                )
                hf_revision: str | None = None
                if hf_upload_repo_id and hf_chain_repo_id and hf_upload_repo_id != hf_chain_repo_id:
                    logger.info(
                        "HF upload repo differs from chain-advertised repo",
                        upload_checkpoint_repo=hf_upload_repo_id,
                        advertised_checkpoint_repo=hf_chain_repo_id,
                    )
                hf_ready, hf_reason = get_hf_upload_readiness(
                    repo_id=hf_upload_repo_id,
                    token_env_var=config.hf.token_env_var,
                )
                if model_ckpt.path is None:
                    logger.warning(
                        "No checkpoint path available for HF upload",
                        upload_checkpoint_repo=hf_upload_repo_id,
                        advertised_checkpoint_repo=hf_chain_repo_id,
                    )
                elif hf_ready:
                    # Pause the background workers while we hold the HF
                    # bandwidth — the download worker also pulls from HF and
                    # would contend on the same network/disk.
                    merge_phase_active.set()
                    try:
                        hf_revision = upload_checkpoint_to_hf(
                            ckpt_dir=model_ckpt.path,
                            repo_id=hf_upload_repo_id,
                            token_env_var=config.hf.token_env_var,
                            commit_message=(
                                f"global_ver={model_ckpt.global_ver} "
                                f"expert_group={config.task.exp.group_id}"
                            ),
                        )
                    except Exception as e:
                        logger.error(
                            "HF checkpoint upload failed; miners will use validator HTTP fallback",
                            upload_checkpoint_repo=hf_upload_repo_id,
                            advertised_checkpoint_repo=hf_chain_repo_id,
                            error=str(e),
                            exc_info=True,
                        )
                    finally:
                        merge_phase_active.clear()
                else:
                    logger.warning(
                        "HF checkpoint upload unavailable; miners will use validator HTTP fallback",
                        upload_checkpoint_repo=hf_upload_repo_id,
                        advertised_checkpoint_repo=hf_chain_repo_id,
                        reason=hf_reason,
                        has_ckpt_path=model_ckpt.path is not None,
                    )

                phase_response = wait_till(config, PhaseNames.validator_commit_2)
                logger.info("(8) Commit model_hash for next validation (non-blocking)")
                chain_submitter.async_commit(ValidatorChainCommit(
                    model_hash=model_ckpt.model_hash,
                    global_ver=model_ckpt.global_ver if _participated_in_merge else 0,  # only update global_ver if we participated in the merge
                    expert_group=config.task.exp.group_id,
                    hf_repo_id=hf_chain_repo_id if hf_revision else None,
                    hf_revision=(hf_revision[:HF_CHAIN_REVISION_LENGTH] if hf_revision else None),
                ))

                if config.ckpt.checkpoint_topk is not None:
                    ckpt_deleted = delete_old_checkpoints(config.ckpt.checkpoint_path, config.ckpt.checkpoint_topk)
                    if ckpt_deleted:
                        logger.debug(f"Deleted old checkpoints: {ckpt_deleted}")

            # === (9) Set weight to chain ===
            # Relocated to the top of the next iteration's MinerCommit1 block
            # so it can incorporate the (3) background scores collected from
            # end-of-Validate(K) through end-of-Train(K+1).

            # === archive top-k submissions, delete the rest ===
            if config.ckpt.archive_submissions:
                logger.info("(10) Archiving top miner submissions")
                archive_top_miner_submissions(
                    submission_dir=config.ckpt.miner_submission_path,
                    archive_dir=config.ckpt.miner_submission_archive_path,
                    score_aggregator=score_aggregator,
                    top_k=config.evaluation.top_k_miners_to_reward,
                    max_archive=config.ckpt.miner_submission_archive_max_files,
                )

            deleted = prune_miner_submission_files(
                config.ckpt.miner_submission_path,
                current_block=lite_subtensor.block,
                cycle_length=config.cycle.cycle_length,
                max_age_cycles=0,
            )
            logger.info(
                "(10) Pruned aged miner submissions after cycle",
                deleted=len(deleted),
                current_block=lite_subtensor.block,
                cycle_length=config.cycle.cycle_length,
                max_age_cycles=0,
            )

            # === validation and log metric ===
            # Local evaluation step disabled to reduce per-cycle RAM/compute load.
            logger.info("(11) Local evaluation disabled, skipping")

            metrics = get_status(
                config=config,
                model=global_model,
                step=global_opt_step,
                training_time=training_time,
                total_training_time=total_training_time,
                inner_opt_step=None,
                global_opt_step=global_opt_step,
                loss_batch=loss_batch,
                aux_loss_batch=aux_loss_batch,
            )

            metric_logger.log(metrics)
            cleanup(global_model)

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received, shutting down validator loop")
        # Stop the producer first so the eval worker drains its remaining
        # claims; then stop the eval worker; finally stop the chain_submitter
        # so any in-flight chain RPCs get cancelled cleanly.
        if download_worker is not None:
            download_worker.stop()
        if eval_worker is not None:
            eval_worker.stop()
        if download_worker is not None:
            download_worker.join(timeout=30)
        if eval_worker is not None:
            eval_worker.join(timeout=30)
        chain_submitter.stop()
        poller.stop()
        cleanup(global_model)
        metric_logger.close()
        for _, a in group_averagers.items():
            a.shutdown()
        raise
    except Exception:
        logger.error("Quit training", exc_info=True)
        if download_worker is not None:
            download_worker.stop()
        if eval_worker is not None:
            eval_worker.stop()
        if download_worker is not None:
            download_worker.join(timeout=30)
        if eval_worker is not None:
            eval_worker.join(timeout=30)
        chain_submitter.stop()
        poller.stop()
        cleanup(global_model)
        metric_logger.close()
        for _, a in group_averagers.items():
            a.shutdown()

        if rank == 0:
            torch.save(global_model.state_dict(), "mycelia_final.pt")


if __name__ == "__main__":
    args = parse_args()

    pkg_version, git_sha = _get_build_version()
    print(f"Connito validator — version={pkg_version}  git_sha={git_sha[:12]}", flush=True)
    logger.info("Validator starting", version=pkg_version, git_sha=git_sha[:12])

    if getattr(args, "test", False):
        from connito.shared.cycle import set_test_mode
        set_test_mode(True)

    if args.path:
        config = ValidatorConfig.from_path(args.path, auto_update_config=args.auto_update_config)
    else:
        config = ValidatorConfig()

    run(0, 1, config)
