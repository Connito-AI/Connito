from __future__ import annotations

import hashlib
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import bittensor
import requests
from pydantic import BaseModel

from connito.shared.app_logging import configure_logging, log_phase, structlog
from connito.shared.chain import (
    MinerChainCommit,
    ValidatorChainCommit,
    WorkerChainCommit,
    get_chain_commits,
    serve_axon,
)
from connito.shared.config import MinerConfig, ValidatorConfig, WorkerConfig
from connito.shared.helper import h256_int, parse_dynamic_filename
from connito.shared.hf_distribute import download_checkpoint_from_hf
from connito.validator.evaluator import MinerEvalJob

configure_logging()
logger = structlog.get_logger(__name__)


BITTENSOR_BLOCK_TIME_SECONDS: int = 12

def _get_with_retry(
    url: str,
    *,
    timeout: int = 10,
    retries: int = 3,
    backoff: int = 2,
) -> requests.Response | None:
    attempt = 0
    non_retryable = {400, 401, 403, 404, 405, 409, 422}

    while attempt <= retries:
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code >= 400:
                body_snippet = resp.text[:500] if resp.text else ""
                if resp.status_code in non_retryable or attempt == retries:
                    logger.error(
                        "HTTP error calling %s (status=%s). Body (first 500 chars): %r",
                        url,
                        resp.status_code,
                        body_snippet,
                    )
                    return None
                logger.warning(
                    "HTTP error, will retry",
                    url=url,
                    status_code=resp.status_code,
                    attempt=attempt + 1,
                )
            else:
                if attempt > 0:
                    logger.info("Request succeeded after retry", url=url, attempt=attempt + 1)
                return resp
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as net_err:
            logger.warning(
                "Network error calling %s, will retry",
                url,
                error=str(net_err),
                attempt=attempt + 1,
            )
        except requests.exceptions.RequestException as req_err:
            logger.warning(
                "Request error calling %s, will retry",
                url,
                error=str(req_err),
                attempt=attempt + 1,
            )

        attempt += 1
        if attempt <= retries:
            sleep_s = backoff**attempt
            logger.debug("Retrying after backoff", url=url, sleep_seconds=sleep_s, attempt=attempt + 1)
            time.sleep(sleep_s)

    logger.error("Request failed after retries", url=url, total_attempts=retries + 1)
    return None

class PhaseResponseLite(BaseModel):
    phase_name: str
    phase_start_block: int
    phase_end_block: int
class PhaseResponse(BaseModel):
    block: int
    cycle_length: int  # how long is one cycle
    cycle_index: int  # which cycle are we in
    cycle_block_index: int  # how far in block are we into a cycle
    phase_name: str  # what is the name of the current phase
    phase_index: int  # what is the id of the phase
    phase_start_block: int  # the start block of the phase
    phase_end_block: int  # the end block of the phase
    blocks_into_phase: int  # how far in block are we in the current phase
    blocks_remaining_in_phase: int  # how manuy block left in the phase


@dataclass
class PhaseNames:
    distribute: str = "Distribute"  # miner download from validator
    train: str = "Train"  # miner trian
    miner_commit_1: str = "MinerCommit1"  # miner commit signed_model_hash and vlaidators commit seed
    miner_commit_2: str = "MinerCommit2"  # miner commit model_hash
    submission: str = "Submission"  # miner submit model to validator
    validate: str = "Validate"  # validator validate
    merge: str = "Merge"  # validator merge
    validator_commit_1: str = "ValidatorCommit1"  # validator commit signed_model_hash
    validator_commit_2: str = "ValidatorCommit2"  # validator commit model_hash


def wait_till(config: MinerConfig | ValidatorConfig, phase_name: str, poll_fallback_block: int = 3) -> PhaseResponse:
    ready = False
    phase_response: PhaseResponse | None = None
    first_print = True
    while not ready:
        ready, blocks_till, phase_response = should_act(config, phase_name, retry_blocks=poll_fallback_block)
        if ready is False and blocks_till > 0:
            sleep_sec = min(blocks_till, max(poll_fallback_block, blocks_till * 0.9)) * BITTENSOR_BLOCK_TIME_SECONDS

            check_time = datetime.now() + timedelta(seconds=sleep_sec)
            check_time_str = check_time.strftime("%H:%M:%S")

            expect_time = datetime.now() + timedelta(seconds=blocks_till * BITTENSOR_BLOCK_TIME_SECONDS)
            expect_time_str = expect_time.strftime("%H:%M:%S")

            if first_print: log_phase(f"<{phase_name}> to begin in {blocks_till} blocks, at {expect_time_str}")
            first_print = False
            time.sleep(sleep_sec)

    if phase_response is None:
        logger.warning(f"wait_till: loop exited with ready=True but phase_response is None for phase '{phase_name}'")
    log_phase(
        f"<{phase_name}> has started, {phase_response.blocks_into_phase} blocks passed,"
        f" {phase_response.blocks_remaining_in_phase} blocks left in phase."
    )
    return phase_response


def check_phase_expired(subtensor: bittensor.Subtensor, phase_response: PhaseResponse) -> bool:
    current_block = subtensor.block
    blocks_remaining = phase_response.phase_end_block - current_block
    if current_block > phase_response.phase_end_block:
        logger.warning(
            f"<{phase_response.phase_name}> phase did not complete on time",
            current_block=current_block,
            phase_end_block=phase_response.phase_end_block,
            diff=blocks_remaining,
        )
        return True
    
    if blocks_remaining >= 0:
        log_phase(
            f"<{phase_response.phase_name}> phase completed on time",
            blocks_remaining=blocks_remaining,
        )
    
    return False


def should_act(config: MinerConfig | ValidatorConfig, phase_name: str, retry_blocks: int) -> tuple[bool, int, PhaseResponse | None]:
    phase_response: PhaseResponse | None = get_phase_from_api(config)

    if phase_response is None:
        ready = False
    else:
        ready = phase_response.phase_name == phase_name

    blocks_till_next_phase = get_blocks_until_next_phase_from_api(config)

    if blocks_till_next_phase is None:
        blocks_till = retry_blocks
    else:
        blocks_till = blocks_till_next_phase[phase_name][2]

    return ready, blocks_till, phase_response


def search_model_submission_destination(
    wallet: bittensor.Wallet, config: MinerConfig, subtensor: bittensor.Subtensor
) -> bittensor.Axon:
    
    validator_miner_assignment = get_validator_miner_assignment(config, subtensor)

    assigned_validator_hotkey = None
    for validator, miners in validator_miner_assignment.items():
        if wallet.hotkey.ss58_address in miners:
            assigned_validator_hotkey = validator
            break

    if assigned_validator_hotkey is None:
        return None

    metagraph = subtensor.metagraph(netuid=config.chain.netuid)
    uid = metagraph.hotkeys.index(assigned_validator_hotkey)

    logger.debug("Resolved validator axon", hotkey=f"{assigned_validator_hotkey[:4]}...{assigned_validator_hotkey[-4:]}", uid=uid)

    return metagraph.axons[uid]


def assign_miners_to_validators(
    validators: dict[str, Any],  # {validator_id: seed}
    miners: list[str],
    max_miners_per_validator: int | None = None,
) -> dict[str, list[str]]:
    n_v = len(validators)
    n_m = len(miners)

    if n_v == 0:
        logger.warning("No validators provided, returning empty assignments")
        return {}

    # --- 0) Combined seed (hash of all validator seeds)
    combined_seed_str = "".join(str(validators[v]) for v in sorted(validators.keys()))
    combined_seed = hashlib.sha256(combined_seed_str.encode()).hexdigest()

    # --- 1) Balanced capacities
    base = n_m // n_v
    rem = n_m % n_v
    v_ids = list(validators.keys())

    ranked_for_bonus = sorted(
        v_ids,
        key=lambda vid: h256_int("cap_bonus", validators[vid], combined_seed),
        reverse=True,
    )
    capacities = {vid: base for vid in v_ids}
    for vid in ranked_for_bonus[:rem]:
        capacities[vid] += 1

    # Cap each validator's capacity
    if max_miners_per_validator is not None:
        for vid in v_ids:
            capacities[vid] = min(capacities[vid], max_miners_per_validator)

    # --- 2) Deterministic miner order seeded by combined validator seed
    miners_sorted = sorted(miners, key=lambda mid: h256_int("miner_order", mid, combined_seed))

    # --- 3) Preference per miner (based on validator seed + combined seed)
    def validator_prefs(mid: str) -> list[str]:
        return sorted(
            v_ids,
            key=lambda vid: h256_int("preference", mid, validators[vid], combined_seed),
            reverse=True,
        )

    # --- 4) Assign miners evenly, respecting capacities
    assignment: dict[str, list[str]] = {vid: [] for vid in v_ids}
    for mid in miners_sorted:
        prefs = validator_prefs(mid)
        for vid in prefs:
            if capacities[vid] > 0:
                assignment[vid].append(mid)
                capacities[vid] -= 1
                break
        else:
            # Should never happen if capacities sum == len(miners)
            assignment[prefs[-1]].append(mid)

    return assignment


def get_combined_validator_seed(config: WorkerConfig, subtensor: bittensor.Subtensor) -> str:
    """
    Deterministically combine validator seeds into a single hex string.

    We sort validator IDs so the result is independent of dict iteration order.
    """
    commits: tuple[WorkerChainCommit, bittensor.Neuron] = get_chain_commits(config, subtensor)

    validator_seeds = get_validator_seed_from_commit(config, commits)
    if not validator_seeds:
        logger.warning("No validator seeds found on chain — returning fallback seed '0'")
        return hashlib.sha256(b"0").hexdigest()

    combined_seed_str = "".join(str(validator_seeds[v]) for v in sorted(validator_seeds.keys()))
    return hashlib.sha256(combined_seed_str.encode()).hexdigest()


def get_validator_miner_assignment(config: WorkerConfig, subtensor: bittensor.Subtensor):
    from connito.shared.checkpoints import build_chain_checkpoints_from_previous_phase

    commits: tuple[WorkerChainCommit, bittensor.Neuron] = get_chain_commits(config, subtensor)
    validator_seeds = get_validator_seed_from_commit(config, commits)
    miners = get_miners_from_commit(config, commits)

    # Filter miners to only those with a chain checkpoint from the previous commit phase
    chain_checkpoints = build_chain_checkpoints_from_previous_phase(
        config=config, subtensor=subtensor, for_role="miner",
    )
    miners_with_checkpoint = {ckpt.hotkey for ckpt in chain_checkpoints.checkpoints}
    excluded_miners = [m for m in miners if m not in miners_with_checkpoint]
    if excluded_miners and len(excluded_miners) == len(miners):
        logger.warning(
            "get_validator_miner_assignment: all miners excluded — none have chain checkpoint",
            excluded_count=len(excluded_miners),
            excluded_miners=[f"{hk[:4]}...{hk[-4:]}" for hk in excluded_miners],
        )
    elif excluded_miners:
        logger.debug(
            "get_validator_miner_assignment: excluding miners without chain checkpoint",
            excluded_count=len(excluded_miners),
            excluded_miners=[f"{hk[:4]}...{hk[-4:]}" for hk in excluded_miners],
        )
    miners = [m for m in miners if m in miners_with_checkpoint]

    logger.debug(
        "get_validator_miner_assignment: inputs",
        expert_group_id=config.task.exp.group_id,
        validator_count=len(validator_seeds),
        miner_count=len(miners),
        validator_seeds={f"{hk[:4]}...{hk[-4:]}": seed for hk, seed in validator_seeds.items()},
        miners=[f"{hk[:4]}...{hk[-4:]}" for hk in miners],
    )

    validator_miner_assignment = assign_miners_to_validators(
        validator_seeds, miners, max_miners_per_validator=config.cycle.max_miners_per_validator,
    )

    logger.info(
        "Validator-miner assignment",
        validators=len(validator_miner_assignment),
        assignment={
            f"{v[:4]}...{v[-4:]}": [f"{m[:4]}...{m[-4:]}" for m in ms]
            for v, ms in validator_miner_assignment.items()
        },
    )
    return validator_miner_assignment


def get_validator_seed_from_commit(config, commits):
    def _derive_validator_assignment_seed(commit: ValidatorChainCommit, hotkey: str) -> int:
        # Keep `miner_seed` in the schema contract, but treat omitted values as
        # the default assignment seed to preserve the compact chain payload.
        legacy_seed = getattr(commit, "miner_seed", None)
        if legacy_seed is not None:
            return int(legacy_seed)
        return 0

    validator_seeds: dict[str, int] = {
        neuron.hotkey: _derive_validator_assignment_seed(commit, neuron.hotkey)
        for commit, neuron in commits
        if isinstance(commit, ValidatorChainCommit)
        and getattr(commit, "expert_group", None) == config.task.exp.group_id
    }
    return validator_seeds


def get_miners_from_commit(config, commits):
    miners: list[str] = [
        neuron.hotkey
        for commit, neuron in commits
        if isinstance(commit, MinerChainCommit) and getattr(commit, "expert_group", None) == config.task.exp.group_id
    ]

    return miners


def get_phase_from_api(config: WorkerConfig) -> PhaseResponse | None:
    """
    Determine current phase based on block schedule.

    Returns:
        str: one of ["training", "submission", "waiting"]
    """
    base_url = config.cycle.owner_url
    url = f"{base_url}/get_phase"

    resp = _get_with_retry(url, timeout=config.cycle.api_timeout_sec, retries=config.cycle.api_retries, backoff=config.cycle.api_backoff_sec)
    if resp is None:
        return None

    try:
        return PhaseResponse(**resp.json())
    except (ValueError, TypeError) as e:
        # ValueError: JSON decode problems
        # TypeError: PhaseResponse(**...) got unexpected/missing fields
        logger.exception("Bad response payload from %s: %s", url, e)
        return None


def get_blocks_until_next_phase_from_api(config: WorkerConfig) -> dict[str, tuple[int, int, int]] | None:
    """
    Determine current phase based on block schedule.

    Returns:
        str: one of ["training", "submission", "waiting"]
    """
    base_url = config.cycle.owner_url
    url = f"{base_url}/blocks_until_next_phase"

    resp = _get_with_retry(url, timeout=config.cycle.api_timeout_sec, retries=config.cycle.api_retries, backoff=config.cycle.api_backoff_sec)
    if resp is None:
        return None

    try:
        return resp.json()
    except ValueError as e:
        # JSON decoding failed
        logger.exception("Invalid JSON from %s: %s", url, e)
        return None


def get_blocks_from_previous_phase_from_api(config: WorkerConfig) -> dict | None:
    """
    Determine current phase based on block schedule.

    Returns:
        str: one of ["training", "submission", "waiting"]
    """
    base_url = config.cycle.owner_url
    url = f"{base_url}/previous_phase_blocks"

    resp = _get_with_retry(url, timeout=config.cycle.api_timeout_sec, retries=config.cycle.api_retries, backoff=config.cycle.api_backoff_sec)
    if resp is None:
        return None

    try:
        return resp.json()
    except ValueError as e:
        # JSON decoding failed
        logger.exception("Invalid JSON from %s: %s", url, e)
        return None

def get_validator_whitelist_from_api(config) -> set[str]:
    """Fetch the validator whitelist from the owner phase service."""
    base_url = config.cycle.owner_url
    url = f"{base_url}/get_validator_whitelist"

    resp = _get_with_retry(
        url,
        timeout=config.cycle.api_timeout_sec,
        retries=config.cycle.api_retries,
        backoff=config.cycle.api_backoff_sec,
    )
    if resp is None:
        logger.warning("Failed to fetch validator whitelist from owner API")
        return set()

    try:
        hotkeys = resp.json()
        logger.debug("fetched validator whitelist", count=len(hotkeys))
        return set(hotkeys)
    except (ValueError, TypeError) as e:
        logger.exception("Invalid JSON from %s: %s", url, e)
        return set()
def get_allowed_version_range(config: WorkerConfig) -> tuple[int | None, int | None]:
    """
    Return (min_allowed_version, max_allowed_version) for global_ver filtering.

    max_allowed_version: start block of the most recent MinerCommit1 phase.
    min_allowed_version: max_allowed_version - 1.5 * cycle_length.

    Both are derived from the owner phase service API.
    Returns (None, None) if the API is unavailable.
    """
    current_phase = get_phase_from_api(config)
    if current_phase is not None and current_phase.phase_name == PhaseNames.miner_commit_1:
        max_version = current_phase.phase_start_block
        cycle_length = current_phase.cycle_length
        min_version = max_version - int(cycle_length * config.cycle.version_range_cycles)
        logger.debug(
            "get_allowed_version_range: currently in MinerCommit1",
            min_version=min_version, max_version=max_version, cycle_length=cycle_length,
        )
        return min_version, max_version

    previous_ranges = get_blocks_from_previous_phase_from_api(config)
    if previous_ranges is None:
        logger.warning("get_allowed_version_range: could not fetch previous phase ranges")
        return None, None

    miner_commit_1_range = previous_ranges.get(PhaseNames.miner_commit_1)
    if miner_commit_1_range is None:
        logger.warning("get_allowed_version_range: MinerCommit1 not found in previous phase ranges")
        return None, None

    max_version = miner_commit_1_range[0]  # start block

    # derive cycle_length as sum of all phase lengths
    cycle_length = sum((end - start + 1) for start, end in previous_ranges.values())

    min_version = max_version - int(cycle_length * config.cycle.version_range_cycles)
    logger.debug(
        "get_allowed_version_range",
        min_version=min_version, max_version=max_version, cycle_length=cycle_length,
    )
    return min_version, max_version


def get_init_peer_id(config: WorkerConfig) -> str | None:
    """
    Determine current phase based on block schedule.

    Returns:
        str: one of ["training", "submission", "waiting"]
    """
    base_url = config.cycle.owner_url
    url = f"{base_url}/get_init_peer_id"

    resp = _get_with_retry(url, timeout=config.cycle.api_timeout_sec, retries=config.cycle.api_retries, backoff=config.cycle.api_backoff_sec)
    if resp is None:
        return None

    try:
        return resp.json()
    except ValueError as e:
        # JSON decoding failed
        logger.exception("Invalid JSON from %s: %s", url, e)
        return None

def load_submission_files(folder: str = "miner_submission"):
    """
    Scans a folder for .pt files and returns:
        { filename: {parsed key/values} }
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path.resolve()}")

    files_dict = {}
    for file_name in folder_path.glob("*.pt"):
        if file_name.name.startswith("._tmp"):
            continue
        meta = parse_dynamic_filename(file_name.name)
        if meta is None:
            continue
        files_dict[file_name.name] = meta

    return files_dict


def hydrate_miner_submissions_from_hf(
    config: ValidatorConfig,
    subtensor: bittensor.Subtensor,
    validator_miner_assignment: dict[str, list[str]],
) -> int:
    """Pull HF-committed miner checkpoints into the local submission dir.

    Miners that upload to HuggingFace during MinerCommit2 advertise
    ``(hf_repo_id, hf_revision)`` in their chain commit. By Submission phase
    those coords are on chain, so the validator can pull the shard directly
    instead of waiting on the miner's HTTP ``/submit-checkpoint`` push. The
    downloaded file is written with the same ``hotkey_*_block_*.pt`` naming
    the HTTP path uses so downstream scanning is transport-agnostic. Miners
    without HF coords, or whose HF download fails, are handled by the
    existing HTTP path — no extra code path.

    Returns the number of miners hydrated this call.
    """
    # Local imports: checkpoints.py imports this module, so a module-level
    # import of build_chain_checkpoints_from_previous_phase is circular.
    from connito.shared.checkpoints import build_chain_checkpoints_from_previous_phase

    miner_assignment = set(validator_miner_assignment.get(config.chain.hotkey_ss58, []))
    if not miner_assignment:
        return 0

    submission_dir = Path(config.ckpt.miner_submission_path)
    submission_dir.mkdir(parents=True, exist_ok=True)

    # A miner with any existing submission file is skipped — don't clobber an
    # HTTP upload already received, and don't re-download on subsequent polls.
    existing_hotkeys: set[str] = set()
    for file_path in submission_dir.glob("*.pt"):
        if file_path.name.startswith(".tmp"):
            continue
        meta = parse_dynamic_filename(file_path.name)
        if meta and "hotkey" in meta:
            existing_hotkeys.add(meta["hotkey"])

    try:
        chain_checkpoints = build_chain_checkpoints_from_previous_phase(
            config=config, subtensor=subtensor, for_role="miner",
        )
    except Exception as e:
        logger.warning("hydrate_miner_submissions_from_hf: failed to build chain checkpoints", error=str(e))
        return 0

    expert_group_id = config.task.exp.group_id
    filename_in_hf = f"model_expgroup_{expert_group_id}.pt"

    hydrated = 0
    for ckpt in chain_checkpoints.checkpoints:
        if ckpt.hotkey is None or ckpt.hotkey not in miner_assignment:
            continue
        if ckpt.hotkey in existing_hotkeys:
            continue
        if not (ckpt.hf_repo_id and ckpt.hf_revision):
            continue

        tmp_dir = submission_dir / f".tmp_hf_{ckpt.hotkey}"
        dest_name = f"hotkey_{ckpt.hotkey}_block_{subtensor.block}.pt"
        dest = submission_dir / dest_name
        try:
            tmp_dir.mkdir(parents=True, exist_ok=True)
            download_checkpoint_from_hf(
                repo_id=ckpt.hf_repo_id,
                revision=ckpt.hf_revision,
                filenames=[filename_in_hf],
                dest_dir=tmp_dir,
                token_env_var=config.hf.token_env_var,
            )
            # Atomic rename so gather_validation_job never sees a partial file.
            (tmp_dir / filename_in_hf).replace(dest)
            existing_hotkeys.add(ckpt.hotkey)
            hydrated += 1
            logger.info(
                "Hydrated miner submission from HF",
                hotkey=ckpt.hotkey[:6],
                uid=ckpt.uid,
                hf_repo_id=ckpt.hf_repo_id,
                hf_revision=ckpt.hf_revision,
                dest=str(dest),
            )
        except Exception as e:
            logger.warning(
                "Failed to hydrate miner submission from HF",
                hotkey=ckpt.hotkey[:6] if ckpt.hotkey else None,
                hf_repo_id=ckpt.hf_repo_id,
                hf_revision=ckpt.hf_revision,
                error=str(e),
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return hydrated


def gather_validation_job(
    config: ValidatorConfig,
    subtensor: bittensor.Subtensor,
    step: int,
    validator_miner_assignment: dict[str, list[str]],
) -> list[MinerEvalJob]:
    miner_assignment = validator_miner_assignment.get(config.chain.hotkey_ss58, [])

    if not miner_assignment:
        logger.warning("No miners assigned to this validator", hotkey=config.chain.hotkey_ss58)
    else:
        logger.debug("assigned_miners", miner_assignment=miner_assignment)

    miner_submission_files = load_submission_files(str(config.ckpt.miner_submission_path))
    _prev_phase_api = get_blocks_from_previous_phase_from_api(config)
    if _prev_phase_api is None:
        logger.warning("gather_validation_job: could not fetch previous phase info from API — skipping evaluation this cycle")
        return []
    previous_phase_range = _prev_phase_api.get(PhaseNames.submission)

    hotkeys = subtensor.metagraph(netuid=config.chain.netuid).hotkeys
    miner_jobs = []
    qualifying_hotkeys: set[str] = set()
    outdated_submissions = []
    unexpected_submissions = []
    for file_name, submission_meta in miner_submission_files.items():
        # Guard against filenames that don't match the uid_*_hotkey_*_block_*.pt
        # template (partial uploads, stale files from older miner versions,
        # manually placed debug checkpoints). Without this, a single bad file
        # raises KeyError and aborts the whole scan, losing valid submissions.
        if "hotkey" not in submission_meta or "block" not in submission_meta:
            logger.warning(
                "Skipping submission file: missing required filename fields",
                file_name=file_name,
                parsed_keys=sorted(submission_meta.keys()),
            )
            continue
        is_assigned = submission_meta["hotkey"] in miner_assignment
        in_previous_phase = previous_phase_range is not None and (previous_phase_range[0] <= submission_meta["block"] <= previous_phase_range[1])
        if is_assigned and in_previous_phase:
            logger.debug("Found qualifying submission file", file_name=file_name, submission_meta=submission_meta)
            qualifying_hotkeys.add(submission_meta["hotkey"])
            miner_jobs.append(
                MinerEvalJob(
                    uid=hotkeys.index(submission_meta["hotkey"]),
                    hotkey=submission_meta["hotkey"],
                    model_path=config.ckpt.miner_submission_path / file_name,
                    step=step,
                )
            )
        else:
            if not in_previous_phase:
                outdated_submissions.append(
                    {
                        "file_name": file_name,
                        "hotkey": submission_meta["hotkey"],
                        "block": submission_meta["block"],
                        # "reason": reason,
                    }
                )

            elif not is_assigned:
                unexpected_submissions.append(
                    {
                        "file_name": file_name,
                        "hotkey": submission_meta["hotkey"],
                        "block": submission_meta["block"],
                        # "reason": reason,
                    }
                )


            else:
                reason = "unknown"

    missing_hotkeys = [hotkey for hotkey in miner_assignment if hotkey not in qualifying_hotkeys]


    if missing_hotkeys:
        logger.debug(
            "Missing miner submissions",
            missing_hotkeys=missing_hotkeys,
            assigned_count=len(miner_assignment),
            received_count=len(qualifying_hotkeys),
        )
    
    if unexpected_submissions:
        logger.debug(
            "Rejected submission: In phase but unassigned miner",
            unexpected_count=len(unexpected_submissions),
            unexpected_submissions=unexpected_submissions,
        )

    return miner_jobs
