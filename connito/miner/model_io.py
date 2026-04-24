import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from queue import Queue
from threading import Lock, Thread

import os
from dotenv import load_dotenv

load_dotenv()


import bittensor

from connito.shared.app_logging import configure_logging, structlog
from connito.shared.chain import (
    CHAIN_COMMIT_MAX_HF_REPO_ID_CHARS,
    MinerChainCommit,
    SignedModelHashChainCommit,
    commit_status,
)
from connito.shared.checkpoint_helper import (
    compile_full_state_dict_from_path,
)
from connito.shared.checkpoints import (
    ModelCheckpoint,
    select_best_checkpoint,
)
from connito.shared.expert_manager import ExpertManager
from connito.shared.client import submit_model
from connito.shared.config import MinerConfig, parse_args
from connito.shared.chain import setup_chain_worker
from connito.shared.cycle import PhaseResponse, check_phase_expired, wait_till, search_model_submission_destination
from connito.shared.helper import get_model_hash
from connito.shared.hf_distribute import (
    get_hf_upload_readiness,
    resolve_hf_repo_ids,
    upload_checkpoint_to_hf,
)
from connito.shared.model import fetch_model_from_chain_validator
from connito.sn_owner.cycle import PhaseNames

# Short SHA prefix written to the chain. Matches the validator convention so
# HF short-SHA resolution behaves the same on both sides.
HF_CHAIN_REVISION_LENGTH = 7

configure_logging()
logger = structlog.get_logger(__name__)


# --- Job definitions ---


class JobType(Enum):
    DOWNLOAD = auto()
    SUBMIT = auto()
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
    # True iff the *current* latest_checkpoint_path has been successfully
    # published to HuggingFace this cycle. Reset to False every time
    # latest_checkpoint_path is rotated so the submit_worker never skips HTTP
    # submission based on a previous cycle's upload.
    latest_checkpoint_hf_uploaded: bool = False
    lock: Lock = field(default_factory=Lock, repr=False)


class FileNotReadyError(RuntimeError):
    pass


# --- Scheduler service ---
def scheduler_service(
    config,
    download_queue: Queue,
    commit_queue: Queue,
    submit_queue: Queue,
    poll_fallback_block: int = 3,
):
    """
    Periodically checks whether to start download/submit phases and enqueues jobs.
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

        # --------- SUBMISSION SCHEDULING ---------
        phase_response = wait_till(config, phase_name=PhaseNames.submission, poll_fallback_block=poll_fallback_block)
        submit_queue.put(Job(job_type=JobType.SUBMIT, phase_response=phase_response))


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


def _prepare_checkpoint_for_commit(
    config,
    wallet,
    shared_state: SharedState,
) -> ModelCheckpoint:
    """Pick the latest local checkpoint, sign it, and publish the path to
    shared state. The HF-uploaded flag is reset here because a new
    latest_checkpoint_path supersedes any prior cycle's upload.
    """
    latest_checkpoint = select_best_checkpoint(
        primary_dir=config.ckpt.checkpoint_path, resume=config.ckpt.resume_from_ckpt
    )
    if latest_checkpoint is None or latest_checkpoint.path is None:
        raise FileNotReadyError("Not checkpoint found, skip commit.")

    latest_checkpoint.expert_group = config.task.exp.group_id
    latest_checkpoint.sign_hash(wallet=wallet)

    with shared_state.lock:
        shared_state.latest_checkpoint_path = latest_checkpoint.path
        shared_state.latest_checkpoint_hf_uploaded = False

    return latest_checkpoint


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
    isn't configured or the upload fails. Every failure is logged and the
    caller falls through to the HTTP /submit-checkpoint path, so HF is always
    additive, never load-bearing.
    """
    try:
        hf_upload_repo_id, hf_chain_repo_id = resolve_hf_repo_ids(
            config.hf,
            max_chain_repo_chars=CHAIN_COMMIT_MAX_HF_REPO_ID_CHARS,
        )
    except Exception as e:
        logger.error(
            f"<{PhaseNames.miner_commit_1}> HF repo id resolution failed; falling back to HTTP submission",
            error=str(e),
            exc_info=True,
        )
        return None, None

    hf_ready, hf_reason = get_hf_upload_readiness(
        repo_id=hf_upload_repo_id,
        token_env_var=config.hf.token_env_var,
    )
    if not (hf_ready and latest_checkpoint.path is not None):
        logger.warning(
            f"<{PhaseNames.miner_commit_1}> HF upload unavailable; validator will use HTTP submission fallback",
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
        )
    except Exception as e:
        logger.error(
            f"<{PhaseNames.miner_commit_1}> HF upload failed; validator will use HTTP submission fallback",
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
        try:
            latest_checkpoint = _prepare_checkpoint_for_commit(config, wallet, shared_state)
            _commit_signed_model_hash(config, wallet, subtensor, latest_checkpoint)
            check_phase_expired(subtensor, job.phase_response)

            # HF upload runs between the two commits so the revision is known
            # by the time we write miner_commit_2. Failure returns (None, None)
            # and the chain commit goes out without r/rv — validator then pulls
            # via HTTP /submit-checkpoint as before.
            hf_chain_repo_id, hf_revision = _upload_checkpoint_to_hf_safe(config, latest_checkpoint)

            # Signal submit_worker that HF has the bytes. Guard with the path
            # so a stale flag can't survive a checkpoint rotation: if another
            # thread has already moved latest_checkpoint_path forward, we know
            # this cycle was superseded and don't overwrite.
            if hf_revision is not None:
                with shared_state.lock:
                    if shared_state.latest_checkpoint_path == latest_checkpoint.path:
                        shared_state.latest_checkpoint_hf_uploaded = True

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
            commit_queue.task_done()


def submit_worker(
    config,
    submit_queue: Queue,
    wallet,
    shared_state: SharedState,
    subtensor=None,
):
    """
    Consumes SUBMIT jobs and runs the submission phase logic.
    """
    if subtensor is None:
        subtensor = bittensor.Subtensor(config.chain.network)
    while True:
        job = submit_queue.get()
        if job is None:  # poison pill — clean shutdown
            submit_queue.task_done()
            logger.info(f"<{PhaseNames.submission}> shutdown signal received.")
            return
        try:
            with shared_state.lock:
                latest_checkpoint_path = shared_state.latest_checkpoint_path
                hf_uploaded = shared_state.latest_checkpoint_hf_uploaded

            if latest_checkpoint_path is None:
                raise FileNotReadyError("Not checkpoint found, skip submission.")

            # HF has the bytes. Validator pulls the miner's shard directly from
            # HuggingFace based on the MinerChainCommit coords, so the HTTP
            # /submit-checkpoint POST would just duplicate a multi-GB upload.
            if hf_uploaded:
                logger.info(
                    f"<{PhaseNames.submission}> HF upload already succeeded this cycle — skipping HTTP submission",
                    path=latest_checkpoint_path,
                )
                continue

            destination_axon = search_model_submission_destination(
                wallet=wallet,
                config=config,
                subtensor=subtensor,
            )

            if destination_axon is None:
                logger.info(
                    f"<{PhaseNames.submission}> No validator assigned to this miner — skipping submission",
                    miner_hotkey=wallet.hotkey.ss58_address[:6],
                )
                continue

            if not destination_axon.ip or destination_axon.ip == "0.0.0.0" or not destination_axon.port:
                logger.warning(
                    f"<{PhaseNames.submission}> Assigned validator has no axon served — skipping submission",
                    validator_hotkey=destination_axon.hotkey[:6] if destination_axon.hotkey else None,
                    ip=destination_axon.ip,
                    port=destination_axon.port,
                )
                continue

            block = subtensor.block

            submit_model(
                url=f"http://{destination_axon.ip}:{destination_axon.port}/submit-checkpoint",
                token="",
                my_hotkey=wallet.hotkey,  # type: ignore
                target_hotkey_ss58=destination_axon.hotkey,
                block=block,
                model_path=f"{latest_checkpoint_path}/model_expgroup_{config.task.exp.group_id}.pt",
                expert_groups=[config.task.exp.group_id],
            )

            model_hash = get_model_hash(
                compile_full_state_dict_from_path(latest_checkpoint_path, expert_groups=[config.task.exp.group_id])
            )

            logger.info(
                f"<{PhaseNames.submission}> submitted model",
                destination={destination_axon.hotkey},
                block=block,
                hash=model_hash,
                path=latest_checkpoint_path / "model_expgroup_{config.task.exp.group_id}.pt",
            )

        except FileNotReadyError as e:
            logger.warning(f"<{PhaseNames.submission}> File not ready error: {e}")

        except Exception as e:
            logger.error(f"<{PhaseNames.submission}> Error while handling job", error=str(e), exc_info=True)

        finally:
            if job.phase_response is not None:
                check_phase_expired(subtensor, job.phase_response)
            submit_queue.task_done()
            logger.info(f"<{PhaseNames.submission}> task completed.")


# --- Wiring it all together ---
def run_system(config, wallet, expert_manager, current_model_version: int = 0, current_model_hash: str = "xxx", subtensor=None):
    if subtensor is None:
        subtensor = bittensor.Subtensor(config.chain.network)

    download_queue = Queue()
    commit_queue = Queue()
    submit_queue = Queue()
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
    submit_thread = Thread(
        target=submit_worker,
        args=(config, submit_queue, wallet, shared_state, subtensor),
        daemon=False,
    )

    download_thread.start()
    commit_thread.start()
    submit_thread.start()

    try:
        # Scheduler runs in the foreground; blocks until interrupted or it errors.
        scheduler_service(
            config=config,
            download_queue=download_queue,
            commit_queue=commit_queue,
            submit_queue=submit_queue,
        )
    finally:
        # Send poison pills so each worker loop exits cleanly.
        download_queue.put(None)
        commit_queue.put(None)
        submit_queue.put(None)

        _JOIN_TIMEOUT_S = 30
        download_thread.join(timeout=_JOIN_TIMEOUT_S)
        commit_thread.join(timeout=_JOIN_TIMEOUT_S)
        submit_thread.join(timeout=_JOIN_TIMEOUT_S)

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
