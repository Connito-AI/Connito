import shutil
import tempfile
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from queue import Queue
from threading import Lock, Thread

from dotenv import load_dotenv

load_dotenv()


import bittensor

from connito.miner.ema_tracker import EMA_SHADOW_FILENAME
from connito.shared.app_logging import configure_logging, structlog
from connito.shared.chain import (
    CHAIN_COMMIT_MAX_HF_REPO_ID_CHARS,
    MinerChainCommit,
    SignedModelHashChainCommit,
    commit_status,
)
from connito.shared.checkpoints import (
    ModelCheckpoint,
    select_best_checkpoint,
)
from connito.shared.expert_manager import ExpertManager
from connito.shared.config import MinerConfig, parse_args
from connito.shared.chain import setup_chain_worker
from connito.shared.cycle import PhaseResponse, check_phase_expired, wait_till
from connito.shared.hf_distribute import (
    get_hf_upload_readiness,
    resolve_hf_repo_ids,
    upload_checkpoint_to_hf,
)
from connito.shared.model import fetch_model_from_chain_validator
from connito.shared.telemetry import inc_error
from connito.sn_owner.cycle import PhaseNames

# Short SHA prefix written to the chain. Matches the validator convention so
# HF short-SHA resolution behaves the same on both sides.
HF_CHAIN_REVISION_LENGTH = 7

configure_logging()
logger = structlog.get_logger(__name__)


def _classify_upload_error(exc: BaseException) -> str:
    """Map an upload exception to a small set of `inc_error` kind labels.

    Keeps cardinality bounded; refine the buckets here as we learn what
    actually surfaces in production logs. Returns one of:
      - "timeout" — TimeoutError or any exception whose name/message mentions a timeout
      - "rpc" — HF / requests / urllib HTTP transport failures
      - "unknown" — anything else (config, filesystem, programmer error)
    """
    if isinstance(exc, TimeoutError):
        return "timeout"
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "timeout" in name or "timed out" in msg or "timeout" in msg:
        return "timeout"
    if any(k in name for k in ("http", "connection", "request", "hub", "hf")):
        return "rpc"
    if any(k in msg for k in ("connection", "network", "503", "502", "504", "reset", "unreachable", "dns")):
        return "rpc"
    return "unknown"


# --- Job definitions ---


class JobType(Enum):
    DOWNLOAD = auto()
    COMMIT = auto()


@dataclass
class Job:
    job_type: JobType
    payload: dict | None = None
    phase_response: PhaseResponse | None = None


@dataclass
class SharedState:
    current_model_version: int | None = None
    current_model_hash: str | None = None
    latest_checkpoint_path: str | None = None
    lock: Lock = field(default_factory=Lock, repr=False)


class FileNotReadyError(RuntimeError):
    pass


# --- Scheduler service ---
def scheduler_service(
    config,
    download_queue: Queue,
    commit_queue: Queue,
    poll_fallback_block: int = 3,
):
    """
    Periodically checks whether to start download/commit phases and enqueues jobs.
    """
    while True:
        # --------- DOWNLOAD SCHEDULING ---------
        phase_response = wait_till(config, phase_name=PhaseNames.distribute, poll_fallback_block=poll_fallback_block)
        download_queue.put(Job(job_type=JobType.DOWNLOAD, phase_response=phase_response))

        # --------- COMISSION SCHEDULING ---------
        phase_response = wait_till(
            config, phase_name=PhaseNames.miner_commit_1, poll_fallback_block=poll_fallback_block
        )
        commit_queue.put(
            Job(
                job_type=JobType.COMMIT,
                phase_response=phase_response,
            )
        )


# --- Workers ---
def download_worker(
    config,
    wallet,
    expert_manager,
    download_queue: Queue,
    current_model_meta,
    current_model_hash,
    shared_state: SharedState,
    subtensor=None,
):
    """
    Consumes DOWNLOAD jobs and runs the download phase logic.
    """
    if subtensor is None:
        subtensor = bittensor.Subtensor(config.chain.network)
    while True:
        job = download_queue.get()
        if job is None:  # poison pill — clean shutdown
            download_queue.task_done()
            logger.info(f"<{PhaseNames.distribute}> shutdown signal received.")
            return
        try:
            # Read current version/hash snapshot
            current_model_meta = select_best_checkpoint(
                primary_dir=config.ckpt.validator_checkpoint_path,
                secondary_dir=config.ckpt.checkpoint_path,
                resume=config.ckpt.resume_from_ckpt,
            )

            if current_model_meta is not None:
                current_model_meta.model_hash = current_model_hash

            chain_checkpoint = fetch_model_from_chain_validator(
                current_model_meta,
                config,
                subtensor,
                wallet,
                expert_group_ids=[config.task.exp.group_id],
                expert_group_assignment = expert_manager.expert_group_assignment
            )

            if (
                chain_checkpoint is None
                or chain_checkpoint.global_ver is None
                or chain_checkpoint.model_hash is None
            ):
                raise FileNotReadyError(f"No required download job: {chain_checkpoint}")

            logger.info(f"<{PhaseNames.distribute}> downloaded model metadata from chain: {chain_checkpoint}.")

            # Update shared state with new version/hash
            current_model_meta = select_best_checkpoint(
                primary_dir=config.ckpt.validator_checkpoint_path,
                secondary_dir=config.ckpt.checkpoint_path,
                resume=config.ckpt.resume_from_ckpt,
            )

            with shared_state.lock:
                shared_state.current_model_version = current_model_meta.global_ver
                shared_state.current_model_hash = current_model_meta.model_hash

        except FileNotReadyError as e:
            logger.info(f"<{PhaseNames.distribute}>: {e}")

        except Exception as e:
            logger.error(f"<{PhaseNames.distribute}> Error while handling job", error=str(e), exc_info=True)

        finally:
            if job.phase_response is not None:
                check_phase_expired(subtensor, job.phase_response)
            download_queue.task_done()
            logger.info(f"<{PhaseNames.distribute}> task completed.")


def _ema_commit_enabled(config) -> bool:
    """True when miner config requests committing the EMA snapshot."""
    ema_cfg = getattr(config, "ema", None)
    if ema_cfg is None:
        return False
    return bool(getattr(ema_cfg, "enabled", False) and getattr(ema_cfg, "commit_ema_snapshot", False))


def _materialize_ema_snapshot_dir(
    src_ckpt_dir: Path,
    expert_group_id: int,
) -> Path | None:
    """Stage an EMA-substituted copy of the expert-group shard in a temp dir.

    Reads ``ema_shadow.safetensors`` (written by ``train.py`` at checkpoint
    time) and writes it out as ``model_expgroup_{id}.safetensors`` inside a
    fresh temp directory. The returned directory is what gets hashed and
    uploaded in place of the live last-step shard.

    Returns ``None`` when the EMA shadow file is absent (e.g. first checkpoint
    before any optimizer step landed). Callers should fall back to the
    live checkpoint in that case so the miner still submits something.

    The caller is responsible for cleaning up the returned directory.
    """
    src_ckpt_dir = Path(src_ckpt_dir)
    ema_path = src_ckpt_dir / EMA_SHADOW_FILENAME
    if not ema_path.exists():
        logger.warning(
            f"commit_ema_snapshot=True but {EMA_SHADOW_FILENAME} not found; "
            "falling back to live checkpoint for this round",
            checkpoint_dir=str(src_ckpt_dir),
        )
        return None

    # The EMA shadow only contains trainable params (typically the
    # expert-group shard's experts). The validator's download path expects
    # `model_expgroup_{id}.safetensors`, so we materialize the EMA tensors
    # under that filename inside a fresh temp dir. Any non-expert tensors
    # in the shadow are simply written too — the validator only loads
    # tensor keys it recognizes from this group.
    tmp_dir = Path(tempfile.mkdtemp(prefix="ema_commit_"))
    try:
        import torch
        from safetensors.torch import load_file, save_file
        shadow_state = load_file(str(ema_path), device="cpu")
        # Match the dtype used by the live shard so the EMA-staged upload
        # isn't unexpectedly 2x larger. The live shard's writer
        # (`save_state_dict_by_expert_group`) uses
        # `next(model.parameters()).dtype` — typically fp16 because the
        # backbone is fp16-listed first. Trainable params are upcast to
        # fp32 during training, so the shadow is fp32; downcast back to
        # fp16 (or bf16 when configured) here to keep upload size on par
        # with the non-EMA path. Cast via .to() handles the precision
        # tradeoff cleanly; values that overflow fp16 will saturate but
        # that's the same behavior the live shard already accepts.
        live_shard_path_st = src_ckpt_dir / f"model_expgroup_{expert_group_id}.safetensors"
        live_shard_path_pt = src_ckpt_dir / f"model_expgroup_{expert_group_id}.pt"
        target_dtype: torch.dtype | None = None
        try:
            if live_shard_path_st.exists():
                live_sd = load_file(str(live_shard_path_st), device="cpu")
            elif live_shard_path_pt.exists():
                live_sd = torch.load(str(live_shard_path_pt), map_location="cpu", weights_only=True)
                live_sd = live_sd.get("model_state_dict", live_sd) if isinstance(live_sd, dict) else {}
            else:
                live_sd = None
            if live_sd:
                # Use the dtype of any floating-point tensor in the live shard.
                for _v in live_sd.values():
                    if torch.is_tensor(_v) and torch.is_floating_point(_v):
                        target_dtype = _v.dtype
                        break
        except Exception:
            target_dtype = None

        if target_dtype is None:
            target_dtype = torch.float16  # safe default matching legacy save_dtype

        casted_state = {
            k: (
                v.to(dtype=target_dtype).contiguous()
                if torch.is_floating_point(v)
                else v.contiguous()
            )
            for k, v in shadow_state.items()
        }
        out_path = tmp_dir / f"model_expgroup_{expert_group_id}.safetensors"
        save_file(casted_state, str(out_path))
        logger.info(
            "Staged EMA snapshot for commit",
            staging_dir=str(tmp_dir),
            num_params=len(casted_state),
            expert_group=expert_group_id,
            target_dtype=str(target_dtype),
        )
        return tmp_dir
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.error(
            "Failed to materialize EMA snapshot; falling back to live checkpoint",
            error=str(e),
            exc_info=True,
        )
        return None


def _prepare_checkpoint_for_commit(
    config,
    wallet,
    shared_state: SharedState,
) -> tuple[ModelCheckpoint, Path | None]:
    """Pick the latest local checkpoint, sign it, and publish the path to
    shared state.

    When ``config.ema.commit_ema_snapshot=True`` and an ``ema_shadow.safetensors``
    file exists alongside the latest checkpoint, the EMA shadow is staged in
    a temporary directory and used as the submission target. The hash is
    then computed from the EMA tensors so the signed hash and uploaded
    files agree.

    Returns ``(checkpoint, ema_temp_dir)`` — ``ema_temp_dir`` is non-None
    only when an EMA staging dir was created and must be cleaned up by the
    caller (see ``commit_worker``).
    """
    latest_checkpoint = select_best_checkpoint(
        primary_dir=config.ckpt.checkpoint_path, resume=config.ckpt.resume_from_ckpt
    )
    if latest_checkpoint is None or latest_checkpoint.path is None:
        raise FileNotReadyError("Not checkpoint found, skip commit.")

    latest_checkpoint.expert_group = config.task.exp.group_id

    ema_temp_dir: Path | None = None
    if _ema_commit_enabled(config):
        staged = _materialize_ema_snapshot_dir(
            src_ckpt_dir=Path(latest_checkpoint.path),
            expert_group_id=config.task.exp.group_id,
        )
        if staged is not None:
            # Repoint the checkpoint at the EMA staging directory. The
            # subsequent `sign_hash()` re-hashes from this new path, so
            # the on-chain signed hash matches what we upload.
            latest_checkpoint.path = staged
            ema_temp_dir = staged
            # `hash_model` lazily fills `model_hash` from `self.path`; clear
            # any prior value so signing re-derives it from EMA tensors.
            latest_checkpoint.model_hash = None

    latest_checkpoint.sign_hash(wallet=wallet)

    with shared_state.lock:
        shared_state.latest_checkpoint_path = latest_checkpoint.path

    return latest_checkpoint, ema_temp_dir


def _commit_signed_model_hash(
    config,
    wallet,
    subtensor,
    latest_checkpoint: ModelCheckpoint,
) -> None:
    logger.info(
        f"<{PhaseNames.miner_commit_1}> committing",
        model_version=latest_checkpoint.global_ver,
        hash=latest_checkpoint.model_hash,
        path=latest_checkpoint.path,
    )
    commit_status(
        config,
        wallet,
        subtensor,
        SignedModelHashChainCommit(
            signed_model_hash=latest_checkpoint.signed_model_hash,
        ),
    )


def _upload_checkpoint_to_hf_safe(
    config,
    latest_checkpoint: ModelCheckpoint,
) -> tuple[str | None, str | None]:
    """Resolve the miner's HF repo and upload the checkpoint directory.

    Returns ``(chain_repo_id, revision)`` — both ``None`` if the HF transport
    isn't configured or the upload fails. HF is the only submission path:
    a failure here means the miner will be missing for this round.
    """
    try:
        hf_upload_repo_id, hf_chain_repo_id = resolve_hf_repo_ids(
            config.hf,
            max_chain_repo_chars=CHAIN_COMMIT_MAX_HF_REPO_ID_CHARS,
        )
    except Exception as e:
        inc_error("checkpoint_upload", _classify_upload_error(e))
        logger.error(
            f"<{PhaseNames.miner_commit_1}> HF repo id resolution failed; miner will be missing for this round",
            error=str(e),
            exc_info=True,
        )
        return None, None

    hf_ready, hf_reason = get_hf_upload_readiness(
        repo_id=hf_upload_repo_id,
        token_env_var=config.hf.token_env_var,
    )
    if not (hf_ready and latest_checkpoint.path is not None):
        logger.error(
            f"<{PhaseNames.miner_commit_1}> HF upload unavailable; miner will be missing for this round",
            reason=hf_reason,
            upload_checkpoint_repo=hf_upload_repo_id,
            has_ckpt_path=latest_checkpoint.path is not None,
        )
        return None, None

    try:
        hf_revision = upload_checkpoint_to_hf(
            ckpt_dir=latest_checkpoint.path,
            repo_id=hf_upload_repo_id,
            token_env_var=config.hf.token_env_var,
            commit_message=(
                f"miner submission global_ver={latest_checkpoint.global_ver} "
                f"expert_group={config.task.exp.group_id}"
            ),
            # Validators fetch this validator's expert-group shard. New
            # miners ship `.safetensors` (no pickle, no code-execution
            # surface); the validator's download worker (PR #98) tries
            # `.safetensors` first and falls back to `.pt`, so we include
            # both extensions during the migration window so a miner
            # upgrading mid-cycle whose latest checkpoint is still `.pt`
            # doesn't end up with an empty upload. `model_shared.*` is
            # intentionally excluded — validators only fetch the expert-
            # group shard, and skipping it keeps uploads small.
            allow_patterns=[
                f"model_expgroup_{config.task.exp.group_id}.safetensors",
                f"model_expgroup_{config.task.exp.group_id}.pt",
            ],
        )
    except Exception as e:
        inc_error("checkpoint_upload", _classify_upload_error(e))
        logger.error(
            f"<{PhaseNames.miner_commit_1}> HF upload failed; miner will be missing for this round",
            upload_checkpoint_repo=hf_upload_repo_id,
            error=str(e),
            exc_info=True,
        )
        return None, None

    return hf_chain_repo_id, hf_revision


def _commit_model_hash(
    config,
    wallet,
    subtensor,
    latest_checkpoint: ModelCheckpoint,
    hf_chain_repo_id: str | None,
    hf_revision: str | None,
) -> None:
    """Emit the miner_commit_2 payload. Omits block and inner_opt so the
    serialized JSON stays within the 128-byte chain budget shared with the
    validator commit.
    """
    short_revision = hf_revision[:HF_CHAIN_REVISION_LENGTH] if hf_revision else None
    logger.info(
        f"<{PhaseNames.miner_commit_2}> committing",
        model_version=latest_checkpoint.global_ver,
        hash=latest_checkpoint.model_hash,
        path=latest_checkpoint.path,
        hf_repo_id=hf_chain_repo_id if hf_revision else None,
        hf_revision=short_revision,
    )
    commit_status(
        config,
        wallet,
        subtensor,
        MinerChainCommit(
            expert_group=config.task.exp.group_id,
            model_hash=latest_checkpoint.model_hash,
            global_ver=latest_checkpoint.global_ver,
            hf_repo_id=hf_chain_repo_id if hf_revision else None,
            hf_revision=short_revision,
        ),
    )


def commit_worker(
    config,
    commit_queue: Queue,
    wallet,
    shared_state: SharedState,
    subtensor=None,
):
    """Consume COMMIT jobs. For each cycle: sign+publish the checkpoint hash
    (miner_commit_1), upload to HF, then commit the hash+HF coords
    (miner_commit_2). Each step lives in its own helper for readability.
    """
    if subtensor is None:
        subtensor = bittensor.Subtensor(config.chain.network)
    while True:
        job = commit_queue.get()
        if job is None:  # poison pill — clean shutdown
            commit_queue.task_done()
            logger.info(f"<{PhaseNames.miner_commit_1}> shutdown signal received.")
            return
        ema_temp_dir: Path | None = None
        try:
            latest_checkpoint, ema_temp_dir = _prepare_checkpoint_for_commit(
                config, wallet, shared_state,
            )
            _commit_signed_model_hash(config, wallet, subtensor, latest_checkpoint)
            check_phase_expired(subtensor, job.phase_response)

            # HF upload runs between the two commits so the revision is known
            # by the time we write miner_commit_2. Failure returns (None, None)
            # and the chain commit goes out without r/rv — the miner is then
            # missing for this round and gets the zero-score penalty.
            hf_chain_repo_id, hf_revision = _upload_checkpoint_to_hf_safe(config, latest_checkpoint)

            phase_response = wait_till(config, PhaseNames.miner_commit_2)
            _commit_model_hash(
                config, wallet, subtensor, latest_checkpoint,
                hf_chain_repo_id, hf_revision,
            )
            check_phase_expired(subtensor, phase_response)

        except FileNotReadyError as e:
            logger.warning(f"<{PhaseNames.miner_commit_1}> File not ready error: {e}")

        except Exception as e:
            logger.error(f"<{PhaseNames.miner_commit_1}> Error while handling job", error=str(e), exc_info=True)

        finally:
            # Clean up EMA staging dir if one was created. Done in finally
            # so a mid-commit error doesn't leak temp directories.
            if ema_temp_dir is not None:
                shutil.rmtree(ema_temp_dir, ignore_errors=True)
            commit_queue.task_done()


# --- Wiring it all together ---
def run_system(config, wallet, expert_manager, current_model_version: int = 0, current_model_hash: str = "xxx", subtensor=None):
    if subtensor is None:
        subtensor = bittensor.Subtensor(config.chain.network)

    download_queue = Queue()
    commit_queue = Queue()
    shared_state = SharedState(current_model_version, current_model_hash)

    # Non-daemon threads so they can be joined cleanly on shutdown.
    download_thread = Thread(
        target=download_worker,
        args=(config, wallet, expert_manager, download_queue, current_model_version, current_model_hash, shared_state, subtensor),
        daemon=False,
    )
    commit_thread = Thread(
        target=commit_worker,
        args=(config, commit_queue, wallet, shared_state, subtensor),
        daemon=False,
    )

    download_thread.start()
    commit_thread.start()

    try:
        # Scheduler runs in the foreground; blocks until interrupted or it errors.
        scheduler_service(
            config=config,
            download_queue=download_queue,
            commit_queue=commit_queue,
        )
    finally:
        # Send poison pills so each worker loop exits cleanly.
        download_queue.put(None)
        commit_queue.put(None)

        _JOIN_TIMEOUT_S = 30
        download_thread.join(timeout=_JOIN_TIMEOUT_S)
        commit_thread.join(timeout=_JOIN_TIMEOUT_S)

        logger.info("run_system: all worker threads have exited.")


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose debug logging enabled!")

    if args.path:
        config = MinerConfig.from_path(args.path, auto_update_config=args.auto_update_config)
    else:
        config = MinerConfig()

    config.write()

    wallet, subtensor, _lite_subtensor = setup_chain_worker(config)

    expert_manager = ExpertManager(config)

    run_system(config, wallet, expert_manager, subtensor=subtensor)
