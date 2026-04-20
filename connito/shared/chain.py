from __future__ import annotations

import json
import math
import threading
import time
from typing import Literal

import bittensor
try:
    from websockets.exceptions import ConnectionClosedError
except Exception:  # pragma: no cover - optional dependency shape varies
    ConnectionClosedError = None
from pydantic import BaseModel, ConfigDict, Field

from connito.shared.app_logging import structlog
from connito.shared.config import WorkerConfig
from connito.shared.telemetry import track_chain_commit_latency, count_rpc_errors

logger = structlog.get_logger(__name__)

# Global lock for subtensor WebSocket access to prevent concurrent recv calls
_subtensor_lock = threading.Lock()

# Retry policy for set_weights RPC calls
_WEIGHT_SUBMIT_MAX_RETRIES: int = 2
_WEIGHT_SUBMIT_BACKOFF_S: float = 2.0



# --- Status structure and submission (for miner validator communication)---
class WorkerChainCommit(BaseModel):
    pass 
class SignedModelHashChainCommit(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    signed_model_hash: str | None = Field(default=None, alias="m")


class ValidatorChainCommit(WorkerChainCommit):
    model_config = ConfigDict(populate_by_name=True)
    signed_model_hash: str | None = Field(default=None, alias="m")
    model_hash: str | None = Field(default=None, alias="h")
    global_ver: int | None = Field(default=None, alias="v")
    expert_group: int | None = Field(default=None, alias="e")
    miner_seed: int | None = Field(default=None, alias="s")
    block: int | None = Field(default=None, alias="b")


class MinerChainCommit(WorkerChainCommit):
    model_config = ConfigDict(populate_by_name=True)
    block: int | None = Field(default=None, alias="b")
    expert_group: int | None = Field(default=None, alias="e")
    signed_model_hash: str | None = Field(default=None, alias="m")
    model_hash: str | None = Field(default=None, alias="h")
    global_ver: int | None = Field(default=0, alias="v")
    inner_opt: int | None = Field(default=0, alias="i")

@track_chain_commit_latency()
@count_rpc_errors()
def commit_status(
    config: WorkerConfig,
    wallet: bittensor.Wallet,
    subtensor: bittensor.Subtensor,
    status: ValidatorChainCommit | MinerChainCommit | SignedModelHashChainCommit,
) -> None:
    """
    Commit the worker status to chain.

    If encrypted=False:
        - Uses subtensor.set_commitment (plain metadata, immediately visible).

    If encrypted=True:
        - Timelock-encrypts the status JSON using Drand.
        - Stores it via the Commitments pallet so it will be revealed later
          when the target Drand round is reached.

    Assumes:
        - config.chain.netuid: subnet netuid
        - config.chain.timelock_rounds_ahead: how many Drand rounds in the future
          you want the data to be revealed (fallback to 200 if missing).
    """
    # Serialize status first; same input for both plain + encrypted paths
    data_dict = status.model_dump(by_alias=True)

    data = json.dumps(data_dict)

    success = subtensor.set_commitment(wallet=wallet, netuid=config.chain.netuid, data=data, raise_error=False)

    if not success:
        logger.warning("Failed to commit status to chain", status=data_dict)
    else:
        logger.info("Committed status to chain", block = subtensor.block, status=data_dict)

    return data_dict


def get_chain_commits(
    config: WorkerConfig,
    subtensor: bittensor.Subtensor,
    wait_to_decrypt: bool = False,
    block: int | None = None,
    signature_commit: bool = False,
) -> tuple[WorkerChainCommit, bittensor.Neuron]:
    try:
        all_commitments = subtensor.get_all_commitments(
            netuid=config.chain.netuid, block=block,
        )
        metagraph = subtensor.metagraph(netuid=config.chain.netuid, block=block)
        current_block = block if block is not None else subtensor.block
    except Exception as err:
        err_msg = str(err)
        if block is not None and "State discarded" in err_msg:
            logger.warning(
                "Historical chain state unavailable on current node; retrying with latest head",
                requested_block=block,
                network=config.chain.network,
                netuid=config.chain.netuid,
                error=err_msg,
            )
            all_commitments = subtensor.get_all_commitments(netuid=config.chain.netuid, block=None)
            metagraph = subtensor.metagraph(netuid=config.chain.netuid, block=None)
            current_block = subtensor.block
        else:
            raise
    max_weight_age = int(config.cycle.cycle_length)

    from connito.shared.cycle import get_validator_whitelist_from_api  # noqa: E402 — lazy import to avoid circular dependency with cycle.py
    whitelisted_validators = get_validator_whitelist_from_api(config)

    parsed = []

    for hotkey, commit in all_commitments.items():
        uid = metagraph.hotkeys.index(hotkey)
        neuron = metagraph.neurons[uid]
        age = current_block - int(getattr(neuron, "last_update", 0))

        try:
            status_dict = json.loads(commit)

            if signature_commit:
                chain_commit = SignedModelHashChainCommit.model_validate(status_dict)
            else:
                is_whitelisted = hotkey in whitelisted_validators
                weight_age = current_block - neuron.last_update
                is_weight_fresh = weight_age <= max_weight_age

                is_validator = is_whitelisted and is_weight_fresh

                if not is_validator:
                    reasons = []
                    if not is_whitelisted:
                        reasons.append("not in validator whitelist")
                    if not is_weight_fresh:
                        reasons.append(
                            f"stale weights (age={weight_age} > max={max_weight_age})"
                        )
                    logger.debug(
                        "role gating: classified as miner",
                        hotkey=hotkey,
                        uid=uid,
                        reasons=reasons,
                        is_whitelisted=is_whitelisted,
                        weight_age=weight_age,
                        last_update=neuron.last_update,
                    )

                chain_commit = (
                    ValidatorChainCommit.model_validate(status_dict)
                    if is_validator
                    else MinerChainCommit.model_validate(status_dict)
                )
                logger.debug(
                    "Parsed chain commit via role gating",
                    hotkey=hotkey,
                    uid=uid,
                    is_whitelisted=is_whitelisted,
                    age_blocks=age,
                    max_weight_age=max_weight_age,
                    parsed_as=("validator" if is_validator else "miner"),
                    status_keys=sorted(status_dict.keys()),
                )

        except Exception as e:
            commit_preview = commit if isinstance(commit, str) else str(commit)
            log_fn = logger.warning if age <= max_weight_age else logger.debug
            log_fn(
                "Failed to parse chain commit",
                hotkey=hotkey,
                uid=uid,
                is_whitelisted=hotkey in whitelisted_validators,
                age_blocks=age,
                max_weight_age=max_weight_age,
                error=str(e),
                commit_preview=commit_preview[:240],
            )
            chain_commit = None

        parsed.append((chain_commit, neuron))

    return parsed


# --- setup chain worker ---
def setup_chain_worker(config, subtensor=None, serve=True):
    wallet = bittensor.Wallet(name=config.chain.coldkey_name, hotkey=config.chain.hotkey_name)
    if subtensor is None:
        logger.debug("setup_chain_worker: creating new Subtensor connection", network=config.chain.network)
        subtensor = bittensor.Subtensor(network=config.chain.network)
    else:
        logger.debug("setup_chain_worker: reusing existing Subtensor connection", network=config.chain.network)
    if serve:
        serve_axon(
            config=config,
            wallet=wallet,
            subtensor=subtensor,
        )
    return wallet, subtensor


def serve_axon(config: WorkerConfig, wallet: bittensor.Wallet, subtensor: bittensor.Subtensor):
    axon = bittensor.Axon(wallet=wallet, external_port=config.chain.port, ip=config.chain.ip)
    axon.serve(netuid=config.chain.netuid, subtensor=subtensor)
    logger.info(
        "Axon served on chain",
        ip=config.chain.ip,
        port=config.chain.port,
        hotkey=wallet.hotkey.ss58_address,
        netuid=config.chain.netuid,
        network=config.chain.network,
    )


def _submit_fallback_weights(
    config: WorkerConfig,
    wallet: bittensor.Wallet,
    subtensor: bittensor.Subtensor,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
    fallback_miners: list[int] | None = None,
) -> bool:
    """Try previous weights from chain, otherwise submit uniform weights.

    Validator UIDs are excluded — weights are only set on miners. When
    ``fallback_miners`` is provided, it is used directly as the miner group
    for the uniform-weights path (bypasses metagraph-wide miner derivation).
    """
    from connito.shared.cycle import get_validator_whitelist_from_api  # noqa: E402 — lazy import to avoid circular dependency with cycle.py

    metagraph = subtensor.metagraph(netuid=config.chain.netuid)
    my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    full_neuron = subtensor.neuron_for_uid(uid=my_uid, netuid=config.chain.netuid)
    prev_weights = {uid: float(w) for uid, w in full_neuron.weights} if full_neuron else {}

    validator_hotkeys = get_validator_whitelist_from_api(config)
    validator_uids = {
        metagraph.hotkeys.index(hk) for hk in validator_hotkeys if hk in metagraph.hotkeys
    }

    fallback_miner_uids = (
        {int(uid) for uid in fallback_miners} if fallback_miners is not None else None
    )

    if prev_weights:
        miner_prev_weights = {
            uid: w for uid, w in prev_weights.items()
            if int(uid) not in validator_uids
            and (fallback_miner_uids is None or int(uid) in fallback_miner_uids)
        }
        dropped = len(prev_weights) - len(miner_prev_weights)

        values = list(miner_prev_weights.values())
        is_even = len(values) >= 2 and (max(values) - min(values)) < 1e-9

        if miner_prev_weights and not is_even:
            logger.info(
                "Falling back to previous weights from chain (miners only)",
                count=len(miner_prev_weights),
                dropped_validator_uids=dropped,
            )
            return submit_weights(config, wallet, subtensor, miner_prev_weights, normalize=True,
                                  wait_for_inclusion=wait_for_inclusion,
                                  wait_for_finalization=wait_for_finalization)
        if is_even:
            logger.info(
                "Previous on-chain weights are uniform; treating as no prev_weights and recomputing",
                count=len(miner_prev_weights),
            )
        else:
            logger.warning("No miner weights remain after excluding validators, falling through to uniform")

    if fallback_miners is not None:
        miner_uids = [int(uid) for uid in fallback_miners]
    else:
        n = metagraph.n.item()
        miner_uids = [uid for uid in range(n) if uid not in validator_uids]
    if not miner_uids:
        logger.warning("No miner UIDs available (all excluded as validators), skipping uniform weight set")
        return False

    logger.warning(
        "No previous weights found on chain, submitting uniform weights to miners",
        miner_count=len(miner_uids),
        excluded_validator_count=len(validator_uids),
    )
    weight = 1.0 / len(miner_uids)
    result = subtensor.set_weights(
        wallet=wallet,
        netuid=config.chain.netuid,
        uids=miner_uids,
        weights=[weight] * len(miner_uids),
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )
    success = result[0] if isinstance(result, tuple) else bool(result)
    if success:
        logger.info("Uniform weights set successfully", count=len(miner_uids))
    else:
        logger.warning("Failed to set uniform weights")
    return success


# --- Chain weight submission ---
@track_chain_commit_latency()
@count_rpc_errors()
def submit_weights(
    config: WorkerConfig,
    wallet: bittensor.Wallet,
    subtensor: bittensor.Subtensor,
    uid_weights: dict[str, float],
    normalize: bool = True,
    top_k: int | None = None,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
    fallback_miners: list[int] | None = None,
) -> bool:
    """
    Submit weights to the chain for this subnet.

    Notes
    -----
    - `uid_weights` maps uid -> weight.
    - If `top_k` is set, only the top-k weights are kept (by value) before normalization.
    - If `normalize=True`, weights are normalized to sum to 1.
    - Zero/negative or non-finite weights are dropped.
    - `fallback_miners` is forwarded to `_submit_fallback_weights` and used as
      the miner group for the uniform-weights fallback path.
    """
    # Filter invalid weights
    filtered: list[tuple[int, float]] = []
    for uid, w in uid_weights.items():
        if w is None or not math.isfinite(w) or w <= 0:
            continue
        filtered.append((int(uid), float(w)))

    if not filtered:
        logger.warning("No valid weights to submit, falling back to default weights", uids=len(uid_weights))
        return _submit_fallback_weights(config, wallet, subtensor,
                                        wait_for_inclusion=wait_for_inclusion,
                                        wait_for_finalization=wait_for_finalization,
                                        fallback_miners=fallback_miners)

    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be > 0 when provided.")
        filtered = sorted(filtered, key=lambda x: x[1], reverse=True)[:top_k]

    uids_f, weights_f = zip(*filtered)
    weights_list = list(weights_f)

    if normalize:
        total = sum(weights_list)
        if total <= 0:
            logger.warning("Weight sum <= 0, skipping submit", total=total)
            return False
        weights_list = [w / total for w in weights_list]

    kwargs = dict(
        wallet=wallet,
        netuid=config.chain.netuid,
        uids=list(uids_f),
        weights=weights_list,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    max_retries = _WEIGHT_SUBMIT_MAX_RETRIES
    backoff_s = _WEIGHT_SUBMIT_BACKOFF_S

    for attempt in range(max_retries + 1):
        try:
            with _subtensor_lock:
                try:
                    result = subtensor.set_weights(**kwargs)
                except TypeError:
                    # Older/newer bittensor signatures may not support wait flags.
                    kwargs.pop("wait_for_inclusion", None)
                    kwargs.pop("wait_for_finalization", None)
                    result = subtensor.set_weights(**kwargs)

            success = result[0] if isinstance(result, tuple) else bool(result)
            if not success:
                logger.warning("Failed to set weights on chain", netuid=config.chain.netuid, count=len(weights_list))
            else:
                logger.info(
                    "Set weights on chain",
                    netuid=config.chain.netuid,
                    count=len(weights_list),
                    block=subtensor.block,
                    weights={int(uid): round(w, 4) for uid, w in zip(uids_f, weights_list, strict=True)},
                )

            return success
        except Exception as exc:
            msg = str(exc)
            retryable = isinstance(exc, TimeoutError)
            if ConnectionClosedError is not None and isinstance(exc, ConnectionClosedError):
                retryable = True
            if "keepalive ping timeout" in msg or "ConnectionClosedError" in msg:
                retryable = True

            if attempt < max_retries and retryable:
                logger.warning(
                    "set_weights failed; retrying",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=msg,
                )
                try:
                    # Recreate subtensor to refresh the WS connection.
                    subtensor = bittensor.Subtensor(network=config.chain.network)
                except Exception as refresh_exc:
                    logger.warning("Failed to refresh subtensor", error=str(refresh_exc))
                time.sleep(backoff_s * (attempt + 1))
                continue

            logger.warning("set_weights failed; giving up", error=msg)
            return False
