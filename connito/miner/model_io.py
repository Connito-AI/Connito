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
from connito.shared.chain import MinerChainCommit, SignedModelHashChainCommit, commit_status
from connito.shared.checkpoint_helper import (
    compile_full_state_dict_from_path,
)
from connito.shared.checkpoints import (
    select_best_checkpoint,
)
from connito.shared.expert_manager import ExpertManager
from connito.shared.client import submit_model
from connito.shared.config import MinerConfig, parse_args
from connito.shared.chain import setup_chain_worker
from connito.shared.cycle import PhaseResponse, check_phase_expired, wait_till, search_model_submission_destination
from connito.shared.helper import get_model_hash
from connito.shared.model import fetch_model_from_chain_validator
from connito.sn_owner.cycle import PhaseNames

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
                raise FileNotReadyError(f"No qualifying download destination: {chain_checkpoint}")

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
            logger.warning(f"<{PhaseNames.distribute}> File not ready error: {e}")

        except Exception as e:
            logger.error(f"<{PhaseNames.distribute}> Error while handling job", error=str(e), exc_info=True)

        finally:
            if job.phase_response is not None:
                check_phase_expired(subtensor, job.phase_response)
            download_queue.task_done()
            logger.info(f"<{PhaseNames.distribute}> task completed.")


def commit_worker(
    config,
    commit_queue: Queue,
    wallet,
    shared_state: SharedState,
    subtensor=None,
):
    """
    Consumes COMMIT model and runs the submission phase logic.
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
            latest_checkpoint = select_best_checkpoint(
                primary_dir=config.ckpt.checkpoint_path, resume=config.ckpt.resume_from_ckpt
            )
            latest_checkpoint.expert_group = config.task.exp.group_id
            latest_checkpoint.sign_hash(wallet=wallet)

            with shared_state.lock:
                shared_state.latest_checkpoint_path = latest_checkpoint.path

            if latest_checkpoint is None or latest_checkpoint.path is None:
                raise FileNotReadyError("Not checkpoint found, skip commit.")

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

            check_phase_expired(subtensor, job.phase_response)

            phase_response = wait_till(config, PhaseNames.miner_commit_2)

            logger.info(
                f"<{PhaseNames.miner_commit_2}> committing",
                model_version=latest_checkpoint.global_ver,
                hash=latest_checkpoint.model_hash,
                path=latest_checkpoint.path,
            )

            commit_status(
                config,
                wallet,
                subtensor,
                MinerChainCommit(
                    expert_group=config.task.exp.group_id,
                    model_hash=latest_checkpoint.model_hash,
                    block=subtensor.block,
                    global_ver=latest_checkpoint.global_ver,
                    inner_opt=latest_checkpoint.inner_opt,
                ),
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

            if latest_checkpoint_path is None:
                raise FileNotReadyError("Not checkpoint found, skip submission.")

            destination_axon = search_model_submission_destination(
                wallet=wallet,
                config=config,
                subtensor=subtensor,
            )

            if destination_axon is None:
                logger.warning(
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

    wallet, subtensor = setup_chain_worker(config)

    expert_manager = ExpertManager(config)

    run_system(config, wallet, expert_manager, subtensor=subtensor)
