import copy
import gc
import math
import os
import secrets
import signal
import threading
import time
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
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
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerBase

from connito.miner.train_helper import get_status
from connito.shared.app_logging import configure_logging, structlog
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
    prune_submissions_outside_window,
    select_best_checkpoint,
)
from connito.shared.config import ValidatorConfig, parse_args
from connito.shared.hf_distribute import (
    resolve_hf_repo_ids,
)
from connito.shared.cycle import (
    check_phase_expired,
    get_blocks_from_previous_phase_from_api,
    get_phase_from_api,
    wait_till,
)
from connito.shared.dataloader import get_dataloader
from connito.shared.expert_manager import (
    ExpertManager,
    get_weight_sum,
)
from connito.shared.helper import get_nested_attr, load_state_dict_from_path
from connito.shared.metrics import MetricLogger
from connito.shared.model import load_model
from connito.shared.modeling.mycelia import get_base_tokenizer
from connito.shared.modeling.quantization import apply_from_config
from connito.sn_owner.cycle import PhaseNames, PhaseManager
from connito.validator.aggregator import MinerScoreAggregator
from connito.validator import cohort_state as cohort_state_module
from connito.validator.background_download_worker import BackgroundDownloadWorker
from connito.validator.background_eval_worker import BackgroundEvalWorker
from connito.validator.chain_submitter import ChainSubmitter, observer_mode_enabled
from connito.validator.evaluator import (
    build_submission_uid_weights,
    finalize_round_scores,
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


from connito.shared.telemetry import (
    TelemetryManager,
    VALIDATOR_AVG_STEP_STATUS,
    VALIDATOR_COHORT_EPOCH,
    VALIDATOR_CURRENT_ROUND_ID,
    VALIDATOR_GLOBAL_OPT_STEP,
    VALIDATOR_HEARTBEAT_TOTAL,
    VALIDATOR_MINER_WEIGHT_SUBMITTED,
    VALIDATOR_ROUND_LIFECYCLE_STEP,
    SystemStatePoller,
    set_miner_assignment_role,
    set_miner_cohort_group,
    set_miner_last_observed_commit_block,
    set_validator_identity,
    note_round_series,
    evict_round_series_before,
    track_metagraph_sync_latency,
)
from datetime import datetime

configure_logging()
logger = structlog.get_logger(__name__)


from connito.shared.memory import cleanup, release_cpu_ram


# Default base port for the Prometheus exporter. Overridable per host via
# the `CONNITO_TELEMETRY_PORT` env var — see `resolve_telemetry_port`.
DEFAULT_TELEMETRY_BASE_PORT = 8200


def resolve_telemetry_port(rank: int, env: dict[str, str] | None = None) -> int:
    """Resolve the port for the Prometheus exporter.

    `CONNITO_TELEMETRY_PORT` is a **base** port, not an absolute one: the
    effective port is `base + rank`. That preserves the semantics of the
    8200 default it replaces, and keeps a multi-rank deployment on one host
    collision-free — an absolute override would point every rank at the same
    port and all but one would fail to bind, which is exactly the bug this
    override exists to fix. For the validator `rank` is always 0, so
    `CONNITO_TELEMETRY_PORT=8201` yields 8201.

    Operators need this when the default 8200 is already taken on the host:
    before this was wired up the env var existed in the image but was never
    read, so the exporter kept trying 8200, failed with "Address already in
    use", and the validator ran on with no telemetry at all.

    Invalid input falls back to the default with a warning rather than
    raising — a typo in an operator's `.env` must not take a validator off
    chain over a telemetry setting.
    """
    env = os.environ if env is None else env
    raw = str(env.get("CONNITO_TELEMETRY_PORT", "") or "").strip()
    base = DEFAULT_TELEMETRY_BASE_PORT
    if raw:
        try:
            parsed = int(raw)
            if not (1 <= parsed <= 65535):
                raise ValueError(f"port out of range: {parsed}")
            base = parsed
        except ValueError as e:
            logger.warning(
                "Invalid CONNITO_TELEMETRY_PORT — falling back to default base port",
                value=raw,
                default_base=DEFAULT_TELEMETRY_BASE_PORT,
                error=str(e),
            )
    port = base + int(rank)
    if not (1 <= port <= 65535):
        logger.warning(
            "Resolved telemetry port out of range — falling back to default",
            base=base, rank=rank, resolved=port,
            default_base=DEFAULT_TELEMETRY_BASE_PORT,
        )
        base = DEFAULT_TELEMETRY_BASE_PORT
        port = base + int(rank)
    if base != DEFAULT_TELEMETRY_BASE_PORT:
        logger.info(
            "Telemetry port overridden via CONNITO_TELEMETRY_PORT",
            base=base, rank=rank, port=port,
        )
    return port


@track_metagraph_sync_latency()
def _sync_lite_metagraph(subtensor, netuid: int):
    """Validator-side metagraph fetch via lite_subtensor.

    Wrapped here (rather than at the call site) so the
    ``track_metagraph_sync_latency`` decorator times every fetch and stamps
    ``validator_metagraph_last_sync_timestamp`` on success.
    """
    return subtensor.metagraph(netuid=netuid)


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


def _install_signal_logging() -> None:
    """Funnel SIGTERM / SIGHUP into the same `KeyboardInterrupt` path SIGINT
    already takes, so docker-initiated stops run the existing shutdown block
    in `run()` (background workers, chain_submitter, poller, averagers, …).

    The previous implementation restored `SIG_DFL` and re-raised the signal.
    For SIGTERM that meant "terminate immediately" with no Python exception —
    the `except KeyboardInterrupt` / `except Exception` arms in `run()` never
    fired, so nothing was stopped cleanly. Watchtower then timed out after 120s
    and dockerd was left with a zombie PID 1 (orphaned hivemind libp2p +
    background-worker threads, no init to reap them) which couldn't be removed.
    Raising `KeyboardInterrupt` reuses the SIGINT shutdown path verbatim.

    Caveat: if the main thread is parked inside a C extension when the signal
    arrives (hivemind averager step, a torch op, etc.), the exception only
    propagates once control returns to Python. The shutdown block itself still
    needs per-step time bounds for that, but those are separate work.
    """
    def _handler(signum: int, frame) -> None:
        try:
            name = signal.Signals(signum).name
        except (ValueError, KeyError):
            name = str(signum)
        logger.warning(
            "Validator received signal — initiating shutdown",
            signal=name,
            signum=signum,
        )
        raise KeyboardInterrupt

    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            # Signals can't be installed from non-main threads; harmless here
            # because we install at module import time, but guard regardless.
            pass


# Phases where a round's eval window can still be open. MinerCommit1/2 are
# excluded for liveness: `build_chain_checkpoints_from_previous_phase` blocks
# on `wait_till(next_phase)` during a commit phase, which would stall startup.
_RESUMABLE_PHASES = frozenset({
    PhaseNames.validate,
    PhaseNames.merge,
    PhaseNames.validator_commit_1,
    PhaseNames.validator_commit_2,
    PhaseNames.distribute,
    PhaseNames.train,
})


def _finalize_journal_file(journal_file, score_aggregator, score_path) -> bool:
    """Replay one unfinalized journal through `finalize_round_scores`.

    Shared by the startup-recovery sweep and by the deferred finalize that
    runs when `resume_open_round` declines the round it held back, so both
    paths produce identical aggregator state.
    """
    from connito.validator import round_journal as _rj_recover
    from connito.validator.round_journal import _RecoveryRound

    try:
        journal = _rj_recover.load(journal_file)
        if journal is None or journal.finalized:
            return False
        logger.info(
            "Startup recovery: replaying unfinalized round journal",
            path=str(journal_file),
            round_id=journal.round_id,
            scored=len(journal.scored_uids),
            failed=len(journal.failed_uids),
            validation_failed=len(journal.validation_failed_uids),
            freeze_zero=len(journal.freeze_zero_uids),
        )
        finalize_round_scores(
            round_obj=_RecoveryRound.from_journal(journal, journal_file),
            score_aggregator=score_aggregator,
            score_path=score_path,
        )
        logger.info("Startup recovery: finalized journal", round_id=journal.round_id)
        return True
    except Exception as e:
        logger.warning(
            "Startup recovery: failed to replay journal",
            path=str(journal_file), error=str(e),
        )
        return False


def resume_open_round(
    *,
    config,
    subtensor,
    lite_subtensor,
    global_model: nn.Module,
    score_aggregator,
    score_path,
    round_ref: RoundRef,
    eval_worker,
    eval_window_active: threading.Event,
    download_window_closed: threading.Event,
) -> int | None:
    """Rebuild the round whose eval window is still open and hand it to the workers.

    Reuses `Round.freeze` for everything chain-derived; only the base model
    cannot be reproduced, so it is read back from the file freeze wrote. Every
    refusal below exists to keep one round's deltas on one base.

    Returns the resumed round_id, or None if refused (caller then finalizes).
    """
    from connito.validator import round_journal as _rj

    def _decline(reason: str, **fields) -> None:
        # Most restarts land outside an eval window and decline here, so this
        # is info, not warning. Without it a refusal is indistinguishable from
        # a resume that never ran.
        logger.info("resume: declined", reason=reason, **fields)

    phase = get_phase_from_api(config)
    if phase is None or phase.phase_name not in _RESUMABLE_PHASES:
        return _decline("phase not resumable", phase=getattr(phase, "phase_name", None))
    previous = get_blocks_from_previous_phase_from_api(config)
    # The API serves each phase as a [start, end] pair.
    sub_range = (previous or {}).get(PhaseNames.submission)
    if not sub_range or len(sub_range) < 2:
        return _decline("no submission range from phase API", phase=phase.phase_name)
    sub_start, sub_end = int(sub_range[0]), int(sub_range[1])

    checkpoint_path = Path(config.ckpt.checkpoint_path)
    journal = _rj.load(_rj.journal_path_for(checkpoint_path, sub_start))
    if journal is None or journal.finalized or journal.roster_size <= 0:
        return _decline(
            "journal unusable",
            round_id=sub_start,
            finalized=journal.finalized if journal else None,
            roster_size=journal.roster_size if journal else None,
        )
    remaining = journal.roster_size - len(journal.scored_uids) - len(journal.failed_uids)
    if remaining <= 0:
        return _decline("roster already complete", round_id=sub_start)

    base_path = _rj.base_snapshot_path_for(checkpoint_path, sub_start)
    if not base_path.exists():
        logger.warning(
            "resume: no base snapshot — refusing to resume",
            round_id=sub_start, path=str(base_path),
        )
        return None
    base_params = torch.load(base_path, map_location="cpu", weights_only=True)

    current_cohort_state = None
    if config.evaluation.enable_round_group_construction:
        _exp = getattr(getattr(config, "task", None), "exp", None)
        current_cohort_state = cohort_state_module.load(
            Path(config.ckpt.checkpoint_path) / config.evaluation.cohort_state_filename,
            expected_expert_group=str(_exp.group_id) if _exp is not None else "",
        )

    # `checkpoint_path=None` so freeze skips its initial journal write, which
    # would briefly overwrite the on-disk verdicts with empty sets.
    resumed = Round.freeze(
        config=config,
        subtensor=subtensor,
        metagraph=lite_subtensor.metagraph(netuid=config.chain.netuid, lite=False),
        global_model=global_model,
        round_id=sub_start,
        submission_block_range=(sub_start, sub_end),
        last_evaluated=score_aggregator.last_evaluated_per_uid(),
        prior_avg_scores=score_aggregator.uid_score_pairs(how="avg"),
        # The live PhaseResponse describes cycle K+1; let freeze derive K.
        cycle_index=None,
        cycle_length=phase.cycle_length,
        cohort_state=current_cohort_state,
        score_aggregator=score_aggregator,
        score_path=score_path,
        checkpoint_path=None,
        advance_cohort=False,
    )

    if journal.seed and resumed.seed != journal.seed:
        logger.warning(
            "resume: seed mismatch — refusing to resume",
            round_id=sub_start, journal_seed=journal.seed, rebuilt_seed=resumed.seed,
        )
        return None

    resumed.model_snapshot_cpu = base_params
    resumed.journal_path = _rj.journal_path_for(checkpoint_path, sub_start)

    # A uid that deregistered and re-registered mid-cycle must not inherit the
    # old miner's verdict: `add_score` resets a uid's whole history on hotkey
    # mismatch, so mis-attribution is not a local error.
    stale = {
        uid for uid, hk in journal.uid_to_hotkey.items()
        if resumed.uid_to_hotkey.get(uid) not in (None, hk)
    }
    with resumed._lock:  # noqa: SLF001
        resumed.scored_uids |= set(journal.scored_uids)
        resumed.failed_uids |= set(journal.failed_uids) | stale
        resumed.validation_failed_uids |= set(journal.validation_failed_uids)
        resumed.scores.update(journal.scores)
        resumed.val_losses.update(journal.uid_to_val_loss)
        # Round K's freeze-time decision is authoritative — a union would
        # over-penalize uids that only look uncommitted now.
        resumed.freeze_zero_uids = set(journal.freeze_zero_uids)
        resumed.freeze_zero_hotkeys = dict(journal.freeze_zero_hotkeys)
        resumed.lifecycle_step = int(journal.lifecycle_step)
    resumed._persist_journal()  # noqa: SLF001

    round_ref.swap(new_current=resumed)
    download_window_closed.clear()
    if eval_worker is not None and not eval_worker.has_eval_base_model():
        eval_worker.set_eval_base_model(copy.deepcopy(global_model))
    eval_window_active.set()

    try:
        # Registers the round's label for later eviction; `publish_progress`
        # emits the series but does not register it.
        note_round_series(resumed.round_id)
        resumed.publish_progress()
    except Exception:
        pass

    logger.info(
        "resume: round resumed",
        round_id=resumed.round_id, phase=phase.phase_name, remaining=remaining,
        already_scored=len(resumed.scored_uids), stale_hotkeys=len(stale),
    )
    return resumed.round_id


def _shutdown_background_workers(
    download_worker: "BackgroundDownloadWorker | None",
    eval_worker: "BackgroundEvalWorker | None",
    join_timeout_sec: float = 30.0,
) -> None:
    """Signal both background workers to stop and wait for them to exit.

    Logs each step so an operator can see which worker is still running
    when the join times out.
    """
    logger.info("Shutdown: signaling background workers to stop")
    if download_worker is not None:
        download_worker.stop()
    if eval_worker is not None:
        eval_worker.stop()

    for worker in (download_worker, eval_worker):
        if worker is None:
            continue
        logger.info(
            "Shutdown: joining background worker",
            thread_name=worker.name,
            timeout_sec=join_timeout_sec,
        )
        worker.join(timeout=join_timeout_sec)
        if worker.is_alive():
            logger.warning(
                "Shutdown: background worker did not exit within timeout",
                thread_name=worker.name,
                timeout_sec=join_timeout_sec,
            )
        else:
            logger.info("Shutdown: background worker joined", thread_name=worker.name)


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
    # global_model: partial model (only assigned experts) — used for optimization and evaluation.
    # `load_global_checkpoint=True`: overlay the newest on-disk `globalver_*`
    # expert state, which now holds the round baseline. That directory is the
    # only local copy of the model, so skipping it restarts from pretrained.
    global_model, model_meta = load_model(
        rank, config, expert_manager, subtensor, wallet, current_model_meta,
        partial=True, checkpoint_device=device,
        load_global_checkpoint=True,
    )
    apply_from_config(global_model, config, expert_manager, role="validator")


    # === dataloader ===
    logger.debug("setup training - load dataloader")
    train_dataloader = get_dataloader(
        config, rank=rank, world_size=config.task.exp.data.world_size, tokenizer=tokenizer
    )

    # === load checkpoint (if any) ===
    logger.debug(
        "setup training - load past checkpoint"
    )
    if get_nested_attr(config, "resume_from_ckpt", False) and resume and latest_checkpoint_path:
        _ = load_checkpoint(
            config=config,
            checkpoint_path=latest_checkpoint_path,
            rank=rank,
            device=device,
            data_loader=train_dataloader,
        )

    logger.info(
        "Training setup complete",
        resumed=resume,
        device=str(device),
    )
    return (
        global_model,
        model_meta.global_ver if model_meta else 0,
        expert_manager,
        train_dataloader,
    )


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
    telemetry_port = resolve_telemetry_port(rank)
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
    # historical block queries (Round.freeze, setup_training/load_model).
    # lite_subtensor: sync Subtensor for head-only reads (metagraph,
    # current block, peer connect, phase checks).
    # chain_submitter: owns an AsyncSubtensor + AsyncRunner; handles every
    # non-blocking commit_status / set_weights call for this validator.
    validate_hf_distribution_config(config)
    wallet, subtensor, lite_subtensor = setup_chain_worker(config, serve=False)
    # Round-group emission produces up to 18 weights (3 G1 + 15 G2) and
    # `compute_uid_weights` is already the canonical set — applying the
    # legacy `top_k_miners_to_reward=3` truncation in `_normalize_uid_weights`
    # would drop every Group 2 entry (each ~0.2% of stake) and leave only
    # the 3 Group 1 winners on chain. Skip the cap when the new scheme is on.
    chain_submitter = ChainSubmitter(
        config,
        wallet,
        normalize=True,
        top_k=(
            None
            if config.evaluation.enable_round_group_construction
            else config.evaluation.top_k_miners_to_reward
        ),
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
        start_step,
        expert_manager,
        train_dataloader,
    ) = setup_training(config, rank, device, tokenizer, subtensor, wallet, current_model_meta=None)

    global_opt_step = start_step
    # Coordinates of the baseline published at finalize, read by the next
    # ValidatorCommit. Empty means "nothing to advertise this cycle".
    baseline_ref: dict[str, object] = {}

    # === set up score aggregator ===
    score_window = config.evaluation.score_window
    # On-disk retention per miner — kept independent of score_window so
    # avg/sum/ema (the metric driving weight submission) still cap reads
    # at score_window. Larger here means more historical points are
    # retained on disk for diagnostics without changing scoring.
    # Hard-coded for now; promote to a config field once we settle on a
    # default that won't change cross-validator behavior.
    score_history_window: int = 80
    score_path = config.ckpt.checkpoint_path / "score_aggregator.json"
    if pkg_version == "v0.2.3":
        # One-time wipe: drop any prior aggregator state on disk so the v0.2.3
        # rollout starts every validator with a clean score history. Subsequent
        # restarts on v0.2.3 fall through the `score_path.exists()` branch and
        # load whatever this version has persisted.
        logger.info("Clearing historic score_aggregator for v0.2.3", pkg_version=pkg_version)
        score_path.unlink(missing_ok=True)
        score_aggregator = MinerScoreAggregator(
            max_points=score_window,
            max_history_points=score_history_window,
        )
    elif score_path.exists():
        try:
            with open(score_path, "r") as f:
                score_aggregator = MinerScoreAggregator.from_json(
                    f.read(),
                    max_points=score_window,
                    max_history_points=score_history_window,
                )
            _loaded_latest = score_aggregator.uid_score_pairs(how="latest")
            _loaded_avg = score_aggregator.uid_score_pairs(how="avg")
            logger.info(
                "Loaded previous MinerScoreAggregator state from disk",
                uids=len(_loaded_latest),
                latest_scores={int(u): float(s) for u, s in sorted(_loaded_latest.items())},
                avg_scores={int(u): float(s) for u, s in sorted(_loaded_avg.items())},
            )
        except Exception as e:
            logger.warning(f"Failed to load score_aggregator.json, starting fresh: {e}")
            score_aggregator = MinerScoreAggregator(
                max_points=score_window,
                max_history_points=score_history_window,
            )
    else:
        score_aggregator = MinerScoreAggregator(
            max_points=score_window,
            max_history_points=score_history_window,
        )

    # === startup recovery: replay any unfinalized round journals ===
    # If a previous run died before `finalize_round_scores` could run, the
    # per-round journal holds the partial state — see `_finalize_journal_file`.
    # The one round whose eval window may still be open is held back and
    # offered to `resume_open_round` once the model and workers exist; if that
    # declines it is finalized then instead. Finalize must run exactly once per
    # round, because `add_score` stamps `_utc_now()` and a second pass would
    # re-timestamp the round's points and reshuffle the scoring window.
    _skipped_live_journal = None
    try:
        from connito.validator import round_journal as _rj_recover
        _journals = _rj_recover.scan(config.ckpt.checkpoint_path)
        _live_round_id = None
        try:
            _prev = get_blocks_from_previous_phase_from_api(config)
            if _prev and PhaseNames.submission in _prev:
                _live_round_id = int(_prev[PhaseNames.submission][0])
        except Exception:
            pass
        _recovered = 0
        for _journal_file in _journals:
            if _live_round_id is not None and _journal_file.name == (
                _rj_recover.journal_path_for(config.ckpt.checkpoint_path, _live_round_id).name
            ):
                _skipped_live_journal = _journal_file
                logger.info(
                    "Startup recovery: skipping live round — deferring to resume",
                    round_id=_live_round_id,
                )
                continue
            if _finalize_journal_file(_journal_file, score_aggregator, score_path):
                _recovered += 1
        if _recovered:
            logger.info(
                "Startup recovery: complete",
                journals_finalized=_recovered,
                journals_seen=len(_journals),
            )

        # Re-emit the most recent finalized round's telemetry.
        #
        # Runs unconditionally, and this is the point: the replay loop above
        # only touches *unfinalized* journals, and replaying one marks it
        # finalized. So a second restart finds nothing to replay and used to
        # emit nothing at all — leaving the dashboard blank for the whole
        # last completed cycle until the next round's evaluations arrived
        # (observed 2026-07-31: two Watchtower restarts 25 min apart, every
        # per-miner family at zero series for 17 minutes).
        #
        # This is a METRICS-ONLY pass — it never re-runs finalize. Re-running
        # finalize would keep the aggregator's point set correct (drop_round
        # runs first) but would re-stamp those points with fresh timestamps,
        # reshuffling the "last N by timestamp" rolling average that drives
        # weight submission. See `republish_telemetry_from_journal`.
        #
        # Journals are scanned ascending, so the last finalized one is the
        # most recent round — which is also the one just replayed on a first
        # restart, making the re-emit a harmless idempotent gauge write.
        try:
            _newest_finalized = None
            for _journal_file in reversed(_journals):
                _j = _rj_recover.load(_journal_file)
                if _j is not None and _j.finalized:
                    _newest_finalized = _j
                    break
            if _newest_finalized is not None:
                _republished = _rj_recover.republish_telemetry_from_journal(
                    _newest_finalized, score_aggregator=score_aggregator,
                )
                logger.info(
                    "Startup recovery: republished telemetry for last finalized round",
                    round_id=_newest_finalized.round_id,
                    uids=_republished,
                    schema_version=_newest_finalized.schema_version,
                    val_losses=len(_newest_finalized.uid_to_val_loss),
                )
        except Exception as e:
            logger.warning(
                "Startup recovery: telemetry republish failed", error=str(e),
            )
    except Exception as e:
        logger.warning(
            "Startup recovery: scan failed", error=str(e),
        )

    # Resolve this validator's UID so the poller can emit vtrust / consensus
    # for our own slot. Failing this lookup keeps the metagraph block of the
    # poller inert (validator_uid=None) rather than crashing startup.
    validator_uid: int | None
    try:
        bootstrap_metagraph = _sync_lite_metagraph(lite_subtensor, config.chain.netuid)
        validator_uid = bootstrap_metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    except Exception as e:
        logger.warning(
            "Could not resolve validator UID for telemetry; metagraph metrics will be inert",
            error=str(e),
        )
        validator_uid = None

    # Stamp identity onto the connito_validator info metric so every Prom scrape
    # carries which validator emitted it, and stash git_version for the
    # /v1/state.json meta block. _get_build_version() reads CONNITO_GIT_VERSION
    # / CONNITO_GIT_SHA env vars (baked into the Docker image) with a git-cli
    # fallback in source checkouts.
    git_version, git_sha = _get_build_version()
    try:
        set_validator_identity(
            hotkey=wallet.hotkey.ss58_address,
            uid=validator_uid,
            version=git_version,
            netuid=int(config.chain.netuid),
            observer=observer_mode_enabled(),
        )
    except Exception as e:
        logger.warning("Failed to stamp connito_validator_info; continuing", error=str(e))

    # Start telemetry sidecar poller
    poller = SystemStatePoller(
        subtensor=lite_subtensor,
        phase_manager=PhaseManager(config, lite_subtensor),
        netuid=config.chain.netuid,
        validator_uid=validator_uid,
        interval_sec=12.0,
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

    current_model_hash = None

    if config.ckpt.cleanup_stale_temporary_checkpoints:
        cleanup_temporary_checkpoint_dirs(config.ckpt.checkpoint_path)

    # === Round-lifecycle scaffolding ===
    # merge_phase_active: set for the entire Merge phase plus briefly around HF upload.
    #   Pauses bg-download (HF bandwidth contention with the validator's own
    #   HF upload) and bg-eval (GPU contention with allreduce / optimizer step).
    # eval_window_active: set when the round freezes so the eval worker may
    #   evaluate round K's downloaded miners; cleared at the top of the next
    #   cycle right before submit_weights for round K.
    # download_window_closed: set when the main loop begins waiting for
    #   MinerCommit1 of the next round (round K's downloads are dead weight
    #   past that point); cleared at the next freeze. Pauses bg-download
    #   from MinerCommit1(K+1) → Submission(K+1).
    # gpu_eval_lock: held by the eval worker only across its load_state_dict
    #   and evaluate_one_miner calls (yielded everywhere else; see plan).
    merge_phase_active = threading.Event()
    eval_window_active = threading.Event()
    download_window_closed = threading.Event()
    gpu_eval_lock = threading.Lock()
    round_ref = RoundRef()

    download_worker: BackgroundDownloadWorker | None = None
    eval_worker: BackgroundEvalWorker | None = None
    resumed_round_id: int | None = None
    if config.evaluation.background_worker_enabled:
        download_worker = BackgroundDownloadWorker(
            config=config,
            round_ref=round_ref,
            merge_phase_active=merge_phase_active,
            download_window_closed=download_window_closed,
        )
        # bg-eval idles until the main loop hands it a copy of global_model,
        # which now happens as soon as the round freezes.
        eval_worker = BackgroundEvalWorker(
            config=config,
            round_ref=round_ref,
            device=device,
            tokenizer=tokenizer,
            merge_phase_active=merge_phase_active,
            eval_window_active=eval_window_active,
            gpu_eval_lock=gpu_eval_lock,
            expert_group_assignment=expert_manager.expert_group_assignment,
        )
        download_worker.start()
        eval_worker.start()
        logger.info(
            "Background workers launched",
            download_thread=download_worker.name,
            download_ident=download_worker.ident,
            eval_thread=eval_worker.name,
            eval_ident=eval_worker.ident,
        )

        # Pick up a round whose eval window was still open at restart. Must be
        # here, not in the loop: the loop opens with `wait_till(miner_commit_1,
        # -5)`, which would block through the whole window. The workers are
        # already parked on their gates, so handing them the round starts them
        # during that wait; the loop's finalize then closes it out.
        try:
            resumed_round_id = resume_open_round(
                config=config,
                subtensor=subtensor,
                lite_subtensor=lite_subtensor,
                global_model=global_model,
                score_aggregator=score_aggregator,
                score_path=score_path,
                round_ref=round_ref,
                eval_worker=eval_worker,
                eval_window_active=eval_window_active,
                download_window_closed=download_window_closed,
            )
        except Exception as e:
            logger.warning("resume: failed — continuing without", error=str(e), exc_info=True)

    # Resume declined, failed, or never ran (workers disabled): finalize the
    # journal held back above, so the outcome matches the pre-resume behaviour
    # exactly. Unconditional so a held-back journal can never leak unfinalized.
    if resumed_round_id is None and _skipped_live_journal is not None:
        _finalize_journal_file(_skipped_live_journal, score_aggregator, score_path)

    logger.info("ChainSubmitter ready")


    try:
        while True:
            # Liveness signal: alert on rate(validator_main_loop_heartbeat_total[5m]) == 0
            VALIDATOR_HEARTBEAT_TOTAL.inc()

            # for each step, we run 1 backward
            # for each inner_opt_step, we run local optimization; gradient_accumulation_steps = 1 real step
            # for each global_opt_interval number of inner_opt_step, we synchronise weight from different ddp worker, and then run global optimization

            # === Wait till commit phase to submit random seed ===
            # block_offset=-5 (was -15) trims dead time at the top of the
            # loop — finalize + submit only need a few blocks of headroom
            # before MinerCommit1, not 15.
            phase_response = wait_till(config, PhaseNames.miner_commit_1, block_offset=-5)
            logger.info("Commit new seed for next validation")

            # === (4) Finalize round-K scoring and submit weights.
            #
            # Close the (3) bg-eval window FIRST so no in-flight eval can
            # add a new entry to `round.scores` after `finalize_round_scores`
            # has snapshotted it. The archive/prune step that lives lower
            # in this block also runs while the window is closed — same
            # invariant we used to rely on, just hoisted up.
            #
            # `finalize_round_scores` is the sole writer to the global
            # aggregator for this round_id: it computes ranks from the
            # delta-based per-round signal in `round.scores`, drops any
            # stale aggregator points tagged with this round_id, and
            # writes 3/2/1 for the top-3 (with delta>0), 0 for everyone
            # else (incl. failed evals and freeze-time invalid checkpoints).
            eval_window_active.clear()
            pending_round: Round | None = round_ref.current
            scheduled_round_weights = False
            if pending_round is not None and not pending_round.weights_submitted:
                finalize_round_scores(
                    round_obj=pending_round,
                    score_aggregator=score_aggregator,
                    score_path=score_path,
                )
                # Off the main loop: a ~3 GB upload must not sit between here
                # and MinerCommit1. Daemon so it can never hold up shutdown.
                from connito.validator.distribute import publish_round_baseline

                threading.Thread(
                    target=publish_round_baseline,
                    kwargs={"round_obj": pending_round, "config": config, "out": baseline_ref},
                    name="publish-baseline", daemon=True,
                ).start()
                # Drop history older than 8 cycle lengths so the aggregator
                # only carries the recent window the cohort election + weight
                # avg actually look at.
                _cycle_len = int(phase_response.cycle_length)
                _min_round_id = int(pending_round.round_id) - 8 * _cycle_len
                _dropped = score_aggregator.prune_before_round(_min_round_id)
                if _dropped:
                    try:
                        score_aggregator.persist_atomic(score_path)
                    except Exception as e:
                        logger.warning(
                            "score_aggregator.persist_atomic after prune failed",
                            error=str(e),
                        )
                # Prune per-round journals on the same cutoff so leftover
                # files don't grow unbounded.
                try:
                    from connito.validator import round_journal as _rj_prune
                    _journals_dropped = _rj_prune.prune_before_round(
                        config.ckpt.checkpoint_path, _min_round_id,
                    )
                    if _journals_dropped:
                        logger.info(
                            "round_journal: pruned old journals",
                            dropped=_journals_dropped,
                            min_round_id=_min_round_id,
                        )
                except Exception as e:
                    logger.warning(
                        "round_journal.prune_before_round failed",
                        error=str(e),
                    )
                # Evict per-round Prometheus labelsets on the same cutoff so
                # metric retention matches on-disk retention (without this,
                # every round leaves permanent {round_id} series behind).
                try:
                    _series_evicted = evict_round_series_before(_min_round_id)
                    if _series_evicted:
                        logger.info(
                            "telemetry: evicted stale per-round series",
                            rounds=_series_evicted,
                            min_round_id=_min_round_id,
                        )
                except Exception as e:
                    logger.warning(
                        "telemetry.evict_round_series_before failed",
                        error=str(e),
                    )
                logger.info(
                    "(4) Handing weight submission to background submitter",
                    round_id=pending_round.round_id,
                )
                payload = build_submission_uid_weights(
                    score_aggregator=score_aggregator,
                    cohort_state=pending_round.cohort_state,
                    round_id=pending_round.round_id,
                    cycle_length=_cycle_len,
                    eval_cfg=config.evaluation,
                )
                uid_weights = payload.uid_weights
                if payload.g1_redirected_to_uid_zero:
                    logger.info(
                        "(4) g1 empty — redirecting weight_group_1 share to uid=0",
                        round_id=pending_round.round_id,
                        ab_uids=list(pending_round.validation_group_a)
                        + list(pending_round.validation_group_b),
                    )
                if payload.cohort_emission:
                    logger.info(
                        "(4) round-group avg-score emission",
                        round_id=pending_round.round_id,
                        weight_group_1=list(payload.weight_group_1),
                        weight_group_2=list(payload.weight_group_2),
                    )
                # Mirror the about-to-submit weights into Prometheus so
                # external aggregators don't have to scrape `/v1/state.json`
                # to learn what each validator votes on chain. Entries are
                # written only for UIDs we actually weight, so a miner the
                # validator has never scored has *no* sample rather than a
                # zero (preserves prior EMA semantics).
                #
                # The per-miner score snapshots (latest / avg / samples /
                # emitted_at) are NOT published here anymore — they moved to
                # `finalize_round_scores`, which covers every verdict uid
                # (not just weight recipients) and re-publishes via the
                # journal-recovery replay after a restart.
                for _uid, _weight in uid_weights.items():
                    try:
                        VALIDATOR_MINER_WEIGHT_SUBMITTED.labels(
                            miner_uid=str(_uid),
                        ).set(float(_weight))
                    except Exception:
                        pass
                # Fire-and-forget. ChainSubmitter sets
                # pending_round.weights_submitted once the chain accepts the call.
                chain_submitter.async_submit_weight(pending_round, uid_weights)
                scheduled_round_weights = True

            # Submit fallback weights if last_update is stale (past max_weight_age)
            # AND we did not just schedule a fresh round-weight submission. The
            # round's set_weights will bump last_update once it lands, which is
            # exactly what the fallback would do — and racing both extrinsics on
            # the same wallet caused substrate "Invalid Transaction" / "Priority
            # is too low" errors and let the (older) fallback weights overwrite
            # the round's weights on chain. If the round's submit fails, next
            # cycle's stale-weights check catches it (no race that cycle).
            max_weight_age = int(config.cycle.cycle_length)
            # `lite=False` so `metagraph.weights` is populated, matching the
            # shape we re-fetch below right before `Round.freeze`. This fetch
            # is only used for the fallback-weights staleness check that
            # follows; the freeze-time fetch is refreshed separately because
            # ~80 blocks of phases pass between here and Submission.
            metagraph = lite_subtensor.metagraph(netuid=config.chain.netuid, lite=False)
            my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
            last_update = metagraph.last_update[my_uid].item()
            current_block = lite_subtensor.get_current_block()
            weight_age = current_block - last_update
            if scheduled_round_weights:
                logger.debug(
                    "Skipping fallback weights this cycle (round weights already scheduled)",
                    weight_age=weight_age,
                    max_weight_age=max_weight_age,
                )
            elif weight_age > max_weight_age:
                logger.info("Weights stale, submitting fallback (non-blocking)",
                            weight_age=weight_age, max_weight_age=max_weight_age)
                # Non-blocking; ChainSubmitter serializes this with the
                # commit_status that follows, so order is preserved.
                chain_submitter.async_submit_fallback_weights()

            phase_response = wait_till(config, PhaseNames.miner_commit_1)
            global_opt_step = phase_response.phase_start_block

            # The (3) eval window was closed at the top of this block before
            # `finalize_round_scores`. Archive/prune below runs with bg-eval
            # gated, preserving the file-race protection that used to live
            # at this point in the loop.
            #
            # Fresh 16-bit random seed each cycle. Read by every validator at
            # the next Submission start via `get_combined_validator_seed`,
            # which sha256s the sorted concat — so cohort-wide assignment
            # rotates each cycle even when miner/validator membership is
            # static. 16 bits = up to 5 decimal digits, ≤9 bytes of JSON; the
            # downstream sha256 supplies the entropy `assign_miners_to_validators`
            # actually needs, so going wider just costs commit-budget bytes
            # for no shuffle-quality gain.
            new_miner_seed = secrets.randbits(16)
            chain_submitter.async_commit(ValidatorChainCommit(
                model_hash=current_model_hash,
                global_ver=global_opt_step,
                expert_group=config.task.exp.group_id,
                miner_seed=new_miner_seed,
            ))

            if config.ckpt.archive_submissions:
                logger.info("Archiving top miner submissions")
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
                "Pruned aged miner submissions after cycle",
                deleted=len(deleted),
                current_block=lite_subtensor.block,
                cycle_length=config.cycle.cycle_length,
                max_age_cycles=0,
            )

            check_phase_expired(lite_subtensor, phase_response)

            # === Wait till Submission phase and freeze the round. The round is
            # the unit of work for the rest of the lifecycle: the download
            # worker picks up its roster, the eval worker scores it.
            phase_response = wait_till(config, PhaseNames.submission)

            logger.info(
                "(0) Submission phase entered — freezing round",
                submission_start=phase_response.phase_start_block,
                submission_end=phase_response.phase_end_block,
                current_block=lite_subtensor.block,
            )

            cleanup(global_model)

            # Round-group construction scheme (gated by
            # config.evaluation.enable_round_group_construction). When the
            # flag is on, load the held cohort state so Round.freeze can
            # advance it at the cohort boundary or reuse it within one.
            # Spec: _specs/round-group-construction-scheme.md.
            cohort_state_path = None
            current_cohort_state = None
            if config.evaluation.enable_round_group_construction:
                cohort_state_path = (
                    Path(config.ckpt.checkpoint_path) / config.evaluation.cohort_state_filename
                )
                _task = getattr(config, "task", None)
                _exp = getattr(_task, "exp", None) if _task is not None else None
                expected_expert_group = str(_exp.group_id) if _exp is not None else ""
                try:
                    current_cohort_state = cohort_state_module.load(
                        cohort_state_path,
                        expected_expert_group=expected_expert_group,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to load cohort_state.json — starting fresh cohort",
                        error=str(e),
                        path=str(cohort_state_path),
                    )
                    current_cohort_state = None

            # Refresh metagraph immediately before freezing. The earlier
            # fetch happened during MinerCommit1 (~80 blocks ago), so its
            # `last_update` / `weights` view is stale by a full submission
            # period. Re-fetching here gives `Round.freeze` the most recent
            # chain-weight and staleness signals available.
            metagraph = lite_subtensor.metagraph(netuid=config.chain.netuid, lite=False)

            # (0) Lock and prioritize: build the round roster in A -> B -> C
            # order, then the previous round's A/B carry-over, then a
            # staleness tail (see Round.freeze). Capture the seed and snapshot
            # global_model to CPU.
            new_round = Round.freeze(
                config=config,
                subtensor=subtensor,
                metagraph=metagraph,
                global_model=global_model,
                round_id=phase_response.phase_start_block,
                submission_block_range=(
                    phase_response.phase_start_block,
                    phase_response.phase_end_block,
                ),
                last_evaluated=score_aggregator.last_evaluated_per_uid(),
                # Re-eval the current leaders first inside background so
                # a stale EMA can't keep a regressed miner on top.
                prior_avg_scores=score_aggregator.uid_score_pairs(how="avg"),
                cycle_index=phase_response.cycle_index,
                cycle_length=phase_response.cycle_length,
                cohort_state=current_cohort_state,
                score_aggregator=score_aggregator,
                score_path=score_path,
                checkpoint_path=Path(config.ckpt.checkpoint_path),
            )

            # Publish the active round id to Prometheus so external
            # aggregators can key per-miner score / val_loss readings to
            # a specific round without parsing labels off the lifecycle
            # gauge. Best-effort.
            try:
                VALIDATOR_CURRENT_ROUND_ID.set(float(new_round.round_id))
            except Exception:
                pass

            # Dashboard telemetry: publish per-miner cohort group, this
            # validator's assignment role, and last-observed-commit block for
            # the round we just froze. All values are read off `new_round` (no
            # extra chain/RPC work) and emitted for EVERY metagraph uid so the
            # gateway sees a fresh value per miner each round — stale group /
            # assignment membership never lingers across cohort epochs. One
            # broad try/except: telemetry must never break the round loop.
            try:
                _group_code_by_uid: dict[int, int] = {}
                for _uid in new_round.validation_group_a:
                    _group_code_by_uid[int(_uid)] = 1
                for _uid in new_round.validation_group_b:
                    _group_code_by_uid[int(_uid)] = 2
                for _uid in new_round.validation_group_c:
                    _group_code_by_uid[int(_uid)] = 3

                _roster_set = {int(u) for u in new_round.background_uids}

                # Tail = miners on this validator's roster but outside the
                # formal A/B/C tiers. Distinct from code 0 ("none"), which
                # means no roster status at all. The dashboard uses this to
                # render "evaluated opportunistically" instead of leaving
                # these miners indistinguishable from unrostered ones.
                for _uid in _roster_set - _group_code_by_uid.keys():
                    _group_code_by_uid[_uid] = 4

                for _uid in range(len(metagraph.hotkeys)):
                    set_miner_cohort_group(_uid, _group_code_by_uid.get(_uid, 0))
                    # Role 1 (foreground) is retired; every rostered miner is 2.
                    set_miner_assignment_role(_uid, 2 if _uid in _roster_set else 0)

                # Last block at which we confirmed a miner's valid chain commit.
                for _uid, _ckpt in new_round.uid_to_chain_checkpoint.items():
                    if getattr(_ckpt, "hf_repo_id", None) and getattr(_ckpt, "hf_revision", None):
                        set_miner_last_observed_commit_block(int(_uid), new_round.round_id)

                VALIDATOR_COHORT_EPOCH.set(float(new_round.cohort_epoch))
            except Exception as _e:
                logger.warning("Failed to emit dashboard round telemetry", error=str(_e))

            # Persist the (possibly newly advanced) cohort state to disk
            # BEFORE round_ref.swap so a crash between freeze and swap can
            # replay deterministically (the next process picks up the same
            # cohort epoch and groups).
            if config.evaluation.enable_round_group_construction and new_round.cohort_state is not None:
                try:
                    cohort_state_module.persist_atomic(
                        cohort_state_path, new_round.cohort_state
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to persist cohort_state.json",
                        error=str(e),
                        path=str(cohort_state_path),
                    )
            # Belt-and-suspenders: drop any leftover submission file whose
            # block falls outside this round's window. The end-of-cycle
            # prune is normally enough, but a validator restart that
            # crashed mid-cycle (or any path that skips that prune) leaves
            # stale .pt files behind — bg-download's find_submission_for_hotkey
            # would then short-circuit the fresh fetch and publish the
            # stale path, which gather_validation_job silently rejects.
            try:
                deleted = prune_submissions_outside_window(
                    folder_path=config.ckpt.miner_submission_path,
                    submission_block_range=new_round.submission_block_range,
                )
                if deleted:
                    logger.info(
                        "Pruned out-of-window submissions at round freeze",
                        deleted=len(deleted),
                        round_id=new_round.round_id,
                        submission_block_range=new_round.submission_block_range,
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to prune out-of-window submissions at round freeze",
                    error=str(exc),
                )
            round_ref.swap(new_current=new_round)
            download_window_closed.clear()
            # bg-eval needs an architecture template and an open window; per-round
            # state comes from `round.model_snapshot_cpu`, taken at freeze. Mirrors
            # the resume path. Opening here rather than after Merge gives the
            # worker the whole round now that nothing else competes for the GPU.
            if eval_worker is not None and not eval_worker.has_eval_base_model():
                eval_worker.set_eval_base_model(copy.deepcopy(global_model))
            eval_window_active.set()
            try:
                note_round_series(new_round.round_id)
                new_round.lifecycle_step = 0
                VALIDATOR_ROUND_LIFECYCLE_STEP.labels(round_id=str(new_round.round_id)).set(0)
                # Seed the progress counters at freeze so the round exists in
                # the metric from the moment it is frozen (scored=0, failed=0,
                # pending=roster) instead of appearing only once the first
                # evaluation lands. Consumers that switch on "newest round with
                # a non-zero scored value" are unaffected by the zero.
                new_round.publish_progress()
            except Exception:
                pass

            try:
                note_round_series(new_round.round_id)
                new_round.lifecycle_step = 2
                VALIDATOR_ROUND_LIFECYCLE_STEP.labels(round_id=str(new_round.round_id)).set(2)
            except Exception:
                pass

            phase_response = wait_till(config, PhaseNames.validate)

            cleanup(global_model)

            # Persist aggregator state atomically.
            try:
                score_aggregator.persist_atomic(score_path)
            except Exception as e:
                logger.warning(f"Failed to persist score_aggregator: {e}")

            # === wait till merge phase ===
            # Nothing is merged any more; the baseline published at
            # MinerCommit1 is this validator's next model. The phase itself
            # stays because the central phase API owns its boundaries and
            # miners read the same schedule — we just do no work in it, which
            # frees the window for the background workers.
            check_phase_expired(lite_subtensor, phase_response)
            phase_response = wait_till(config, PhaseNames.merge)

            # Still populated: the ValidatorCommit block below is what clears
            # `baseline_ref`, and it runs after this point.
            baseline_path = baseline_ref.get("path")

            # Held across the load and save only, not the whole phase: both
            # mutate state the background workers read.
            merge_phase_active.set()
            try:
                if baseline_path:
                    logger.info(
                        "Adopting round baseline as the new model",
                        round_id=baseline_ref.get("round_id"),
                        uid=baseline_ref.get("uid"),
                    )
                    # Same primitives as `evaluator.load_model_from_path`, but
                    # applied in place — a deepcopy here would double model
                    # VRAM for nothing. `strict=False` because the file carries
                    # only the active expert group; backbone and helper-group
                    # keys are legitimately absent and keep their values.
                    sd = load_state_dict_from_path(baseline_path)
                    incompatible = global_model.load_state_dict(sd, strict=False)
                    matched_keys = len(sd) - len(incompatible.unexpected_keys)
                    del sd
                    if matched_keys == 0:
                        logger.error(
                            "Round baseline shares no keys with the model; "
                            "model unchanged this cycle",
                            path=baseline_path,
                        )
                    else:
                        logger.info("Round baseline adopted", matched_keys=matched_keys)
                else:
                    logger.warning(
                        "No baseline published this round; keeping the current model"
                    )

                cleanup(global_model)

                # === save checkpoint ===
                logger.info("Saving checkpoint")
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
                    loss=loss_batch.item(),
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

            try:
                note_round_series(new_round.round_id)
                new_round.lifecycle_step = 3
                VALIDATOR_ROUND_LIFECYCLE_STEP.labels(round_id=str(new_round.round_id)).set(3)
            except Exception:
                pass

            check_phase_expired(lite_subtensor, phase_response)

            # === Comit to chain for new model ===
            model_ckpt = build_local_checkpoint(ckpt_path)
            if model_ckpt is not None:

                model_ckpt.expert_group = config.task.exp.group_id
                if observer_mode_enabled():
                    # The signature's only consumer is the commit below, which
                    # observer mode suppresses — and this is the last thing in
                    # the validator that needs the hotkey's *private* key.
                    # Skipping it lets an observer run on a public-only
                    # keyfile, so a live validator's key never has to be copied
                    # onto the test host at all. Hash anyway: `model_hash` is
                    # read on the next line and drives eval, and `sign_hash`
                    # was what triggered it.
                    if model_ckpt.model_hash is None:
                        model_ckpt.hash_model()
                else:
                    model_ckpt.sign_hash(wallet=wallet)
                current_model_hash = model_ckpt.model_hash
                # Dashboard telemetry: the model's global optimization version
                # (chain-committed `global_ver`) is the "steps" the leaderboard
                # charts plot against. Best-effort.
                try:
                    VALIDATOR_GLOBAL_OPT_STEP.set(float(model_ckpt.global_ver))
                except Exception:
                    pass
                # Advertise the baseline published at finalize, not the merged
                # model. `publish_round_baseline` uploaded it ~142 blocks ago,
                # so it is normally done by now; if it is not (still uploading,
                # or lost to a restart) commit no HF coordinates and miners keep
                # what they have — `fetch_model_from_chain_validator` skips a
                # checkpoint with no repo/revision.
                _, hf_chain_repo_id = resolve_hf_repo_ids(
                    config.hf,
                    max_chain_repo_chars=VALIDATOR_COMMIT_MAX_HF_REPO_ID_CHARS,
                )
                baseline = dict(baseline_ref)
                baseline_ref.clear()
                hf_revision = baseline.get("revision")
                if hf_revision and baseline.get("model_hash"):
                    # The hash MUST travel with the revision: miners verify the
                    # downloaded bytes against `model_hash`, so advertising the
                    # baseline's revision beside the merged model's hash would
                    # make every miner reject it.
                    commit_ckpt = ModelCheckpoint(model_hash=baseline["model_hash"])
                    if not observer_mode_enabled():
                        commit_ckpt.sign_hash(wallet=wallet)
                    logger.info(
                        "Advertising round baseline",
                        repo_id=hf_chain_repo_id,
                        revision=hf_revision[:HF_CHAIN_REVISION_LENGTH],
                        round_id=baseline.get("round_id"),
                        uid=baseline.get("uid"),
                    )
                else:
                    commit_ckpt = model_ckpt
                    logger.warning(
                        "No baseline to advertise this cycle; committing without HF coordinates",
                        has_revision=bool(hf_revision),
                    )

                phase_response = wait_till(config, PhaseNames.validator_commit_1)
                logger.info("Commit new signed_model_hash for next validation (non-blocking)")
                chain_submitter.async_commit(SignedModelHashChainCommit(
                    signed_model_hash=commit_ckpt.signed_model_hash,
                ))

                check_phase_expired(lite_subtensor, phase_response)

                phase_response = wait_till(config, PhaseNames.validator_commit_2)
                logger.info("Commit model_hash for next validation (non-blocking)")
                chain_submitter.async_commit(ValidatorChainCommit(
                    model_hash=commit_ckpt.model_hash,
                    global_ver=global_opt_step,
                    expert_group=config.task.exp.group_id,
                    hf_repo_id=hf_chain_repo_id if hf_revision else None,
                    hf_revision=(hf_revision[:HF_CHAIN_REVISION_LENGTH] if hf_revision else None),
                ))

                if config.ckpt.checkpoint_topk is not None:
                    ckpt_deleted = delete_old_checkpoints(config.ckpt.checkpoint_path, config.ckpt.checkpoint_topk)
                    if ckpt_deleted:
                        logger.debug(f"Deleted old checkpoints: {ckpt_deleted}")

            # === (4) Set weight to chain ===
            # Relocated to the top of the next iteration's MinerCommit1 block
            # so it can incorporate the (3) background scores collected from
            # end-of-Validate(K) through end-of-Train(K+1).

            # === Close download window before next-cycle MinerCommit1 ===
            # Wait until 30 blocks before the next MinerCommit1 so bg-download
            # stops pulling round-K submissions inside the quiet window just
            # before the new cycle begins. The archive + prune of those files
            # has been moved to right after MinerCommit1 begins (above), so
            # bg-eval can keep scoring round-K's miners through this window
            # without racing the cleanup.
            wait_till(config, PhaseNames.miner_commit_1, block_offset=-15)
            download_window_closed.set()

            # === validation and log metric ===
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
        _shutdown_background_workers(download_worker, eval_worker)
        chain_submitter.stop()
        poller.stop()
        cleanup(global_model)
        metric_logger.close()
        raise
    except Exception:
        logger.error("Quit training", exc_info=True)
        _shutdown_background_workers(download_worker, eval_worker)
        chain_submitter.stop()
        poller.stop()
        cleanup(global_model)
        metric_logger.close()

        if rank == 0:
            torch.save(global_model.state_dict(), "mycelia_final.pt")


if __name__ == "__main__":
    args = parse_args()

    pkg_version, git_sha = _get_build_version()
    print(f"Connito validator — version={pkg_version}  git_sha={git_sha[:12]}", flush=True)
    logger.info("Validator starting", version=pkg_version, git_sha=git_sha[:12])
    _install_signal_logging()

    if getattr(args, "test", False):
        from connito.shared.cycle import set_test_mode
        set_test_mode(True)

    if args.path:
        config = ValidatorConfig.from_path(args.path, auto_update_config=args.auto_update_config)
    else:
        config = ValidatorConfig()

    run(0, 1, config, pkg_version=pkg_version)
