from __future__ import annotations

import asyncio
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
try:
    from async_substrate_interface.errors import SubstrateRequestException
except Exception:  # pragma: no cover - optional dependency shape varies
    SubstrateRequestException = None
from pydantic import BaseModel, ConfigDict, Field

from connito.shared.app_logging import structlog
from connito.shared.config import WorkerConfig
from connito.shared.telemetry import track_chain_commit_latency, count_rpc_errors

logger = structlog.get_logger(__name__)

# The Bittensor commitments pallet caps this subnet's per-hotkey commit payload
# — validator and miner commits share the same budget and the same per-field
# ceiling on the HF repo id. Both paths go through validate_chain_commit_payload.
CHAIN_COMMIT_MAX_BYTES = 128
CHAIN_COMMIT_MAX_HF_REPO_ID_CHARS = 32
# Back-compat aliases for callers that imported the old names.
VALIDATOR_COMMIT_MAX_BYTES = CHAIN_COMMIT_MAX_BYTES
VALIDATOR_COMMIT_MAX_HF_REPO_ID_CHARS = CHAIN_COMMIT_MAX_HF_REPO_ID_CHARS
MINER_COMMIT_MAX_BYTES = CHAIN_COMMIT_MAX_BYTES
MINER_COMMIT_MAX_HF_REPO_ID_CHARS = CHAIN_COMMIT_MAX_HF_REPO_ID_CHARS

# Global lock for subtensor WebSocket access to prevent concurrent recv calls
_subtensor_lock = threading.Lock()

# Retry policy for set_weights RPC calls
_WEIGHT_SUBMIT_MAX_RETRIES: int = 2
_WEIGHT_SUBMIT_BACKOFF_S: float = 2.0

# Retry policy for chain read RPCs in get_chain_commits — covers transient
# upstream-node failures (SubstrateRequestException "Internal error", WS drop,
# timeout) that would otherwise crash the validator at Round.freeze.
_CHAIN_READ_MAX_RETRIES: int = 5
_CHAIN_READ_BACKOFF_S: float = 2.0

# How many miner UIDs the peer-consensus fallback emits even-weight to.
# Matches the validator's `weight_group_1_size` default so the fallback
# emission shape mirrors a successful round's G1 selection.
_FALLBACK_TOP_N_PEERS: int = 3

# Every weight submission (regular round + fallback peer-top-3)
# reserves this fraction for the subnet owner (uid=0). The fallback's
# uid=0 escalation path (all peers in even-weight state) already emits
# 100% to uid=0 and is unaffected.
SUBNET_OWNER_UID: int = 0
SUBNET_OWNER_WEIGHT_SHARE: float = 0.05


def reserve_subnet_owner_share(
    uid_weights: dict[int, float],
    *,
    share: float = SUBNET_OWNER_WEIGHT_SHARE,
    owner_uid: int = SUBNET_OWNER_UID,
) -> dict[int, float]:
    """Return a new ``{uid: weight}`` dict where ``owner_uid`` receives
    at least ``share`` of the total. All other entries are scaled by
    ``(1 - share)``; if ``owner_uid`` is already present, its prior
    weight is also scaled and then ``share`` is added on top.

    Assumes the input sums to ~1.0; the output is normalized to the
    same total.

    Edge cases:
      - Empty input → ``{owner_uid: 1.0}``.
      - ``share <= 0`` → input returned unchanged.
      - ``share >= 1`` → ``{owner_uid: 1.0}`` (rest collapses to zero).

    Note: this helper does NOT defend against ``top_k`` filtering on
    the downstream submitter — if ``top_k`` is set tightly enough that
    ``owner_uid``'s 5% gets cut, the share is lost. The validator's
    current ``ChainSubmitter`` uses ``top_k=None`` so this is fine in
    production; reconsider if that changes.
    """
    if share <= 0.0:
        return dict(uid_weights)
    if share >= 1.0:
        return {owner_uid: 1.0}
    if not uid_weights:
        return {owner_uid: 1.0}
    scale = 1.0 - share
    out: dict[int, float] = {int(uid): float(w) * scale for uid, w in uid_weights.items()}
    out[owner_uid] = out.get(owner_uid, 0.0) + share
    return out



# --- Status structure and submission (for miner validator communication)---
class WorkerChainCommit(BaseModel):
    pass 
class SignedModelHashChainCommit(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    signed_model_hash: str | None = Field(default=None, alias="m")


class ValidatorChainCommit(WorkerChainCommit):
    model_config = ConfigDict(populate_by_name=True)
    model_hash: str | None = Field(default=None, alias="h")
    global_ver: int | None = Field(default=None, alias="v")
    expert_group: int | None = Field(default=None, alias="e")
    miner_seed: int | None = Field(default=None, alias="s")
    # HuggingFace is the checkpoint transport: the validator uploads the
    # checkpoint directory to `hf_repo_id`, and `hf_revision` carries a short
    # commit SHA prefix so miners pull a pinned snapshot even if the repo
    # advances. model_hash still verifies integrity post-download.
    hf_repo_id: str | None = Field(default=None, alias="r")
    hf_revision: str | None = Field(default=None, alias="rv")


class MinerChainCommit(WorkerChainCommit):
    model_config = ConfigDict(populate_by_name=True)
    # block and inner_opt default to None (not 0) so they're excluded from the
    # serialized payload when the caller leaves them out — miner commits have
    # to share the 128-byte budget with HF coords, and every field that isn't
    # load-bearing pushes the HF repo id length allowance down. Older commits
    # that still set them parse fine.
    block: int | None = Field(default=None, alias="b")
    expert_group: int | None = Field(default=None, alias="e")
    signed_model_hash: str | None = Field(default=None, alias="m")
    model_hash: str | None = Field(default=None, alias="h")
    global_ver: int | None = Field(default=None, alias="v")
    inner_opt: int | None = Field(default=None, alias="i")
    # HuggingFace is the only submission transport: the miner uploads the
    # checkpoint directory to `hf_repo_id` and `hf_revision` pins a short commit
    # SHA prefix so validators pull the exact bytes the miner advertised.
    hf_repo_id: str | None = Field(default=None, alias="r")
    hf_revision: str | None = Field(default=None, alias="rv")


def serialize_chain_status(
    status: ValidatorChainCommit | MinerChainCommit | SignedModelHashChainCommit,
) -> tuple[dict, str]:
    data_dict = status.model_dump(by_alias=True, exclude_none=True)
    data = json.dumps(data_dict, separators=(",", ":"))
    return data_dict, data


def validate_chain_commit_payload(
    status: ValidatorChainCommit | MinerChainCommit,
    max_bytes: int = CHAIN_COMMIT_MAX_BYTES,
    max_hf_repo_id_chars: int = CHAIN_COMMIT_MAX_HF_REPO_ID_CHARS,
) -> tuple[dict, str]:
    """Serialize a commit and assert it fits the chain's payload budget.

    Shared by validator and miner commits — the Bittensor commitments pallet
    doesn't distinguish between the two, so the budgets are identical. The
    per-field HF repo id cap gives a clearer error than the raw byte-count
    check when a user configures a pathologically long repo.
    """
    data_dict, data = serialize_chain_status(status)
    hf_repo_id = getattr(status, "hf_repo_id", None)
    if hf_repo_id and len(hf_repo_id) > max_hf_repo_id_chars:
        raise ValueError(
            "HF repo id is too long for the chain payload budget: "
            f"{len(hf_repo_id)} > {max_hf_repo_id_chars}"
        )

    payload_bytes = len(data.encode())
    if payload_bytes > max_bytes:
        raise ValueError(
            f"Chain commit exceeds payload budget: {payload_bytes} > {max_bytes} bytes"
        )

    return data_dict, data


# Back-compat wrappers for existing callers. Kept so external imports don't
# break; new code should use validate_chain_commit_payload directly.
def validate_validator_chain_commit_payload(
    status: ValidatorChainCommit,
    max_bytes: int = CHAIN_COMMIT_MAX_BYTES,
) -> tuple[dict, str]:
    return validate_chain_commit_payload(status, max_bytes=max_bytes)


def validate_miner_chain_commit_payload(
    status: MinerChainCommit,
    max_bytes: int = CHAIN_COMMIT_MAX_BYTES,
) -> tuple[dict, str]:
    return validate_chain_commit_payload(status, max_bytes=max_bytes)

async def acommit_status(
    config: WorkerConfig,
    wallet: bittensor.Wallet,
    async_subtensor: "bittensor.AsyncSubtensor",
    status: ValidatorChainCommit | MinerChainCommit | SignedModelHashChainCommit,
) -> dict:
    """Async equivalent of `commit_status` for use against an AsyncSubtensor."""
    if isinstance(status, ValidatorChainCommit | MinerChainCommit):
        data_dict, data = validate_chain_commit_payload(status)
    else:
        data_dict, data = serialize_chain_status(status)

    try:
        success = await async_subtensor.set_commitment(
            wallet=wallet, netuid=config.chain.netuid, data=data, raise_error=False,
        )
    except Exception as exc:
        logger.warning("acommit_status: set_commitment raised", error=str(exc))
        return data_dict

    if not success:
        logger.warning("Failed to commit status to chain (async)", status=data_dict)
    else:
        logger.info("Committed status to chain (async)", status=data_dict)
    return data_dict


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
    if isinstance(status, ValidatorChainCommit | MinerChainCommit):
        data_dict, data = validate_chain_commit_payload(status)
    else:
        data_dict, data = serialize_chain_status(status)

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
    # Retry transient upstream-node errors (e.g. SubstrateRequestException
    # "Internal error" from get_chain_head, WS disconnects, timeouts). Without
    # this, a single bad RPC response at Round.freeze kills the validator.
    # The "State discarded" branch is a one-shot fallback to head when the
    # archive node has pruned the requested historical block — it's tracked
    # separately so it doesn't burn retry budget.
    max_retries = _CHAIN_READ_MAX_RETRIES
    backoff_s = _CHAIN_READ_BACKOFF_S
    fetch_block = block

    for attempt in range(max_retries + 1):
        try:
            all_commitments = subtensor.get_all_commitments(
                netuid=config.chain.netuid, block=fetch_block,
            )
            metagraph = subtensor.metagraph(netuid=config.chain.netuid, block=fetch_block)
            current_block = fetch_block if fetch_block is not None else subtensor.block
            break
        except Exception as err:
            err_msg = str(err)

            if fetch_block is not None and "State discarded" in err_msg:
                logger.warning(
                    "Historical chain state unavailable on current node; retrying with latest head",
                    requested_block=fetch_block,
                    network=config.chain.network,
                    netuid=config.chain.netuid,
                    error=err_msg,
                )
                fetch_block = None
                continue

            retryable = isinstance(err, TimeoutError)
            if SubstrateRequestException is not None and isinstance(err, SubstrateRequestException):
                retryable = True
            if ConnectionClosedError is not None and isinstance(err, ConnectionClosedError):
                retryable = True
            if "Internal error" in err_msg or "keepalive ping timeout" in err_msg or "ConnectionClosedError" in err_msg:
                retryable = True

            if attempt < max_retries and retryable:
                logger.warning(
                    "get_chain_commits: chain RPC failed; retrying",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=err_msg,
                )
                try:
                    subtensor = bittensor.Subtensor(network=config.chain.network)
                except Exception as refresh_exc:
                    logger.warning(
                        "Failed to refresh archive subtensor", error=str(refresh_exc),
                    )
                time.sleep(backoff_s * (attempt + 1))
                continue

            raise
    max_weight_age = int(config.cycle.cycle_length)

    from connito.shared.cycle import get_validator_whitelist_from_api  # noqa: E402 — lazy import to avoid circular dependency with cycle.py
    whitelisted_validators = get_validator_whitelist_from_api(config)
    hotkey_to_uid = {metagraph_hotkey: uid for uid, metagraph_hotkey in enumerate(metagraph.hotkeys)}

    parsed = []

    for hotkey, commit in all_commitments.items():
        uid = hotkey_to_uid.get(hotkey)
        if uid is None:
            logger.debug(
                "Skipping commit from hotkey not in metagraph (likely deregistered)",
                hotkey=hotkey,
                block=current_block,
            )
            continue
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


def get_validator_chain_commits(
    config: WorkerConfig,
    subtensor: bittensor.Subtensor,
    *,
    block: int | None = None,
    signature_commit: bool = False,
    metagraph=None,
) -> list[tuple[ValidatorChainCommit | SignedModelHashChainCommit | None, "bittensor.Neuron"]]:
    """Like ``get_chain_commits`` but only pulls commits for whitelisted
    validators (per the owner API).

    Saves a ``get_all_commitments(netuid=)`` over ~256 UIDs by issuing one
    ``get_commitment(netuid=, uid=)`` per whitelisted validator instead.
    Skips role-gating entirely (every hotkey we query is already known to
    be a validator).

    ``metagraph`` may be passed in (e.g. a lite metagraph already fetched
    by the caller) to avoid a duplicate metagraph RPC. Hotkey→UID mapping
    uses the supplied metagraph as-is; if you need historical UID
    assignment, pass a metagraph fetched at ``block``.
    """
    from connito.shared.cycle import get_validator_whitelist_from_api  # lazy: avoids cycle import cycle

    whitelist = get_validator_whitelist_from_api(config)
    if not whitelist:
        logger.warning("get_validator_chain_commits: empty validator whitelist")
        return []

    if metagraph is None:
        metagraph = subtensor.metagraph(netuid=config.chain.netuid, lite=True)
    hotkey_to_uid = {hk: uid for uid, hk in enumerate(metagraph.hotkeys)}

    parsed: list[tuple[ValidatorChainCommit | SignedModelHashChainCommit | None, "bittensor.Neuron"]] = []
    for hotkey in whitelist:
        uid = hotkey_to_uid.get(hotkey)
        if uid is None:
            continue
        try:
            raw = subtensor.get_commitment(
                netuid=config.chain.netuid, uid=uid, block=block,
            )
        except Exception as e:
            logger.warning(
                "get_validator_chain_commits: per-uid fetch failed",
                uid=uid, hotkey=hotkey, error=str(e),
            )
            continue
        if not raw:
            continue
        try:
            status_dict = json.loads(raw)
            if signature_commit:
                chain_commit = SignedModelHashChainCommit.model_validate(status_dict)
            else:
                chain_commit = ValidatorChainCommit.model_validate(status_dict)
        except Exception as e:
            logger.debug(
                "get_validator_chain_commits: parse failed",
                uid=uid, hotkey=hotkey, error=str(e),
            )
            chain_commit = None
        parsed.append((chain_commit, metagraph.neurons[uid]))

    return parsed


# --- setup chain worker ---
def setup_chain_worker(config, subtensor=None, lite_subtensor=None, serve=True):
    """Create the chain connections this worker needs.

    Returns ``(wallet, subtensor, lite_subtensor)``:

    - ``subtensor`` uses ``config.chain.network`` and must be an archive node
      because ``get_chain_commits`` issues historical ``block=N`` queries.
    - ``lite_subtensor`` uses ``config.chain.lite_network`` (defaults to
      ``finney``) for operations that only need the current head.
    """
    wallet = bittensor.Wallet(name=config.chain.coldkey_name, hotkey=config.chain.hotkey_name)
    if subtensor is None:
        logger.debug("setup_chain_worker: creating archive Subtensor connection", network=config.chain.network)
        subtensor = bittensor.Subtensor(network=config.chain.network)
    else:
        logger.debug("setup_chain_worker: reusing existing archive Subtensor connection", network=config.chain.network)

    if lite_subtensor is None:
        lite_network = config.chain.lite_network
        if lite_network and lite_network != config.chain.network:
            logger.debug("setup_chain_worker: creating lite Subtensor connection", lite_network=lite_network)
            lite_subtensor = bittensor.Subtensor(network=lite_network)
        else:
            # lite_network explicitly matches archive — single connection.
            lite_subtensor = subtensor

    if serve:
        serve_axon(
            config=config,
            wallet=wallet,
            subtensor=lite_subtensor,
        )
    return wallet, subtensor, lite_subtensor


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


def _peer_miner_contribution(
    vneuron,
    validator_uids: set[int],
    my_uid: int,
) -> dict[int, float] | None:
    """Return one peer's stake-weighted contribution to the fallback
    aggregation, or ``None`` if the peer should be skipped.

    A peer is skipped when:
      - the neuron lookup returned ``None``,
      - the peer has zero stake,
      - none of the peer's positive weights land on miners
        (everything was on validators or self), or
      - the peer's miner weights are themselves in an even-weight
        fallback shape (≥2 non-zero weights all equal) — counting
        them would just amplify a circular signal.

    A single-UID vote is treated as differentiated (it's a concrete
    pick, not a uniform spread).
    """
    if vneuron is None:
        return None
    v_stake = float(getattr(vneuron, "stake", 0.0))
    if v_stake <= 0:
        return None
    peer_miner_weights: dict[int, float] = {}
    for uid, w in getattr(vneuron, "weights", []):
        uid_i = int(uid)
        if uid_i in validator_uids or uid_i == my_uid:
            continue
        fw = float(w)
        if fw <= 0:
            continue
        peer_miner_weights[uid_i] = fw
    if not peer_miner_weights:
        return None
    values = list(peer_miner_weights.values())
    is_even = (
        len(values) >= 2
        and (max(values) - min(values)) < 1e-9
    )
    if is_even:
        return None
    return {uid: fw for uid, fw in peer_miner_weights.items()}


def _pick_top_n_miner_uids(
    miner_score: dict[int, float],
    top_n: int,
) -> list[tuple[int, float]]:
    """Sort ``{uid: score}`` desc by score, ties broken by uid asc, and
    slice to ``top_n``. Returns ``[(uid, score), ...]`` for both the
    UID list and per-UID logging.
    """
    return sorted(
        miner_score.items(), key=lambda kv: (-kv[1], kv[0]),
    )[:top_n]


def _submit_fallback_weights(
    config: WorkerConfig,
    wallet: bittensor.Wallet,
    subtensor: bittensor.Subtensor,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> bool:
    """Submit fallback weights from a sync ``Subtensor``.

    Aggregate stake-weighted votes from other validators whose
    submissions are themselves differentiated (i.e. not in an
    even-weight fallback shape) and emit even weight to the top-3
    miners. If no peer is differentiated, emit 100% on uid=0 — the
    validator still earns this cycle's emission rather than echoing a
    subnet-wide even-weight cascade.

    Note: we deliberately do **not** reuse our own previous on-chain
    weights. Those could be a stale snapshot from a buggy cycle or
    from our own earlier fallback emission; a fresh look at peer
    consensus is always safer than perpetuating our last submission.

    The sync/async split exists only at the I/O boundary; the pure
    decisions live in `_peer_miner_contribution` and
    `_pick_top_n_miner_uids`.
    """
    from connito.shared.cycle import get_validator_whitelist_from_api  # noqa: E402 — lazy import to avoid circular dependency with cycle.py

    metagraph = subtensor.metagraph(netuid=config.chain.netuid)
    my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)

    validator_hotkeys = get_validator_whitelist_from_api(config)
    validator_uids = {
        metagraph.hotkeys.index(hk) for hk in validator_hotkeys if hk in metagraph.hotkeys
    }

    miner_score: dict[int, float] = {}
    any_peer_differentiated = False
    for vuid in validator_uids - {my_uid}:
        vneuron = subtensor.neuron_for_uid(uid=vuid, netuid=config.chain.netuid)
        contrib = _peer_miner_contribution(vneuron, validator_uids, my_uid)
        if contrib is None:
            continue
        any_peer_differentiated = True
        for uid_i, score in contrib.items():
            miner_score[uid_i] = miner_score.get(uid_i, 0.0) + score

    top_peers = _pick_top_n_miner_uids(miner_score, _FALLBACK_TOP_N_PEERS)
    miner_uids = [uid for uid, _ in top_peers]
    if not miner_uids:
        logger.warning(
            "Fallback: all other validators in even-weight state; "
            "submitting 100% weight to uid=0 instead of echoing",
            other_validator_count=len(validator_uids - {my_uid}),
            any_peer_differentiated=any_peer_differentiated,
        )
        result = subtensor.set_weights(
            wallet=wallet,
            netuid=config.chain.netuid,
            uids=[0],
            weights=[1.0],
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
        success = result[0] if isinstance(result, tuple) else bool(result)
        if not success:
            logger.warning("Failed to set uid=0 fallback weight")
        return success

    # Each miner gets an equal share of (1 - SUBNET_OWNER_WEIGHT_SHARE).
    miner_weights_raw = {uid: 1.0 / len(miner_uids) for uid in miner_uids}
    final = reserve_subnet_owner_share(miner_weights_raw)
    submit_uids = list(final.keys())
    submit_weights_list = list(final.values())
    logger.warning(
        "Fallback: submitting even weight to top stake-weighted miners "
        "from differentiated peers (with subnet-owner share)",
        top_uids=miner_uids,
        top_scores=[round(s, 6) for _, s in top_peers],
        excluded_validator_count=len(validator_uids),
        subnet_owner_share=SUBNET_OWNER_WEIGHT_SHARE,
    )
    result = subtensor.set_weights(
        wallet=wallet,
        netuid=config.chain.netuid,
        uids=submit_uids,
        weights=submit_weights_list,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )
    success = result[0] if isinstance(result, tuple) else bool(result)
    if success:
        logger.info("Fallback weights set successfully", count=len(miner_uids))
    else:
        logger.warning("Failed to set fallback weights")
    return success


async def _asubmit_fallback_weights(
    config: WorkerConfig,
    wallet: bittensor.Wallet,
    async_subtensor: "bittensor.AsyncSubtensor",
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> bool:
    """Async equivalent of `_submit_fallback_weights`.

    Same shape, same decisions, same helper calls — only the I/O is
    awaited.
    """
    from connito.shared.cycle import get_validator_whitelist_from_api  # noqa: E402

    metagraph = await async_subtensor.metagraph(netuid=config.chain.netuid)
    my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)

    validator_hotkeys = get_validator_whitelist_from_api(config)
    validator_uids = {
        metagraph.hotkeys.index(hk) for hk in validator_hotkeys if hk in metagraph.hotkeys
    }

    miner_score: dict[int, float] = {}
    any_peer_differentiated = False
    for vuid in validator_uids - {my_uid}:
        vneuron = await async_subtensor.neuron_for_uid(uid=vuid, netuid=config.chain.netuid)
        contrib = _peer_miner_contribution(vneuron, validator_uids, my_uid)
        if contrib is None:
            continue
        any_peer_differentiated = True
        for uid_i, score in contrib.items():
            miner_score[uid_i] = miner_score.get(uid_i, 0.0) + score

    top_peers = _pick_top_n_miner_uids(miner_score, _FALLBACK_TOP_N_PEERS)
    miner_uids = [uid for uid, _ in top_peers]
    if not miner_uids:
        logger.warning(
            "Fallback (async): all other validators in even-weight state; "
            "submitting 100% weight to uid=0 instead of echoing",
            other_validator_count=len(validator_uids - {my_uid}),
            any_peer_differentiated=any_peer_differentiated,
        )
        result = await async_subtensor.set_weights(
            wallet=wallet,
            netuid=config.chain.netuid,
            uids=[0],
            weights=[1.0],
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
        success = result[0] if isinstance(result, tuple) else bool(result)
        if not success:
            logger.warning("Failed to set uid=0 fallback weight (async)")
        return success

    # Mirror of the sync helper: even share across top-3, then reserve
    # SUBNET_OWNER_WEIGHT_SHARE for uid=0.
    miner_weights_raw = {uid: 1.0 / len(miner_uids) for uid in miner_uids}
    final = reserve_subnet_owner_share(miner_weights_raw)
    submit_uids = list(final.keys())
    submit_weights_list = list(final.values())
    logger.warning(
        "Fallback (async): submitting even weight to top stake-weighted "
        "miners from differentiated peers (with subnet-owner share)",
        top_uids=miner_uids,
        subnet_owner_share=SUBNET_OWNER_WEIGHT_SHARE,
    )
    result = await async_subtensor.set_weights(
        wallet=wallet,
        netuid=config.chain.netuid,
        uids=submit_uids,
        weights=submit_weights_list,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )
    success = result[0] if isinstance(result, tuple) else bool(result)
    if success:
        logger.info("Fallback weights set successfully (async)", count=len(miner_uids))
    else:
        logger.warning("Failed to set fallback weights (async)")
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
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> bool:
    """
    Submit weights to the chain for this subnet.

    Notes
    -----
    - `uid_weights` maps uid -> weight.
    - If `top_k` is set, only the top-k weights are kept (by value) before normalization.
    - If `normalize=True`, weights are normalized to sum to 1.
    - Zero/negative or non-finite weights are dropped.
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
                                        wait_for_finalization=wait_for_finalization)

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
                    # set_weights only needs the current head, so reconnect to
                    # the lite endpoint rather than the heavier archive node.
                    subtensor = bittensor.Subtensor(network=config.chain.lite_network)
                except Exception as refresh_exc:
                    logger.warning("Failed to refresh subtensor", error=str(refresh_exc))
                time.sleep(backoff_s * (attempt + 1))
                continue

            logger.warning("set_weights failed; giving up", error=msg)
            return False


def _normalize_uid_weights(
    uid_weights: dict[int | str, float],
    *,
    normalize: bool,
    top_k: int | None,
) -> tuple[list[int], list[float]] | None:
    """Filter, top-k, and normalize a uid → weight dict.

    Returns (uids, weights) ready for `set_weights`, or None if no valid
    weights remained after filtering.
    """
    filtered: list[tuple[int, float]] = []
    for uid, w in uid_weights.items():
        if w is None or not math.isfinite(w) or w <= 0:
            continue
        filtered.append((int(uid), float(w)))

    if not filtered:
        return None

    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be > 0 when provided.")
        filtered = sorted(filtered, key=lambda x: x[1], reverse=True)[:top_k]

    uids_f, weights_f = zip(*filtered, strict=True)
    weights_list = list(weights_f)

    if normalize:
        total = sum(weights_list)
        if total <= 0:
            return None
        weights_list = [w / total for w in weights_list]

    return list(uids_f), weights_list


async def submit_weights_async(
    config: WorkerConfig,
    wallet: bittensor.Wallet,
    async_subtensor: "bittensor.AsyncSubtensor",
    uid_weights: dict[int | str, float],
    normalize: bool = True,
    top_k: int | None = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> bool:
    """Async equivalent of `submit_weights` for use against an AsyncSubtensor.

    On transient WS failures the caller's AsyncSubtensor is reused for
    retries — its WebSocket reconnects on its own; reopening it from
    inside this helper would race the caller's other in-flight calls.
    """
    prepared = _normalize_uid_weights(uid_weights, normalize=normalize, top_k=top_k)
    if prepared is None:
        logger.warning(
            "submit_weights_async: no valid weights to submit",
            uids=len(uid_weights),
        )
        return False
    uids, weights_list = prepared

    kwargs = dict(
        wallet=wallet,
        netuid=config.chain.netuid,
        uids=uids,
        weights=weights_list,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    max_retries = _WEIGHT_SUBMIT_MAX_RETRIES
    backoff_s = _WEIGHT_SUBMIT_BACKOFF_S

    for attempt in range(max_retries + 1):
        try:
            try:
                result = await async_subtensor.set_weights(**kwargs)
            except TypeError:
                kwargs.pop("wait_for_inclusion", None)
                kwargs.pop("wait_for_finalization", None)
                result = await async_subtensor.set_weights(**kwargs)

            success = result[0] if isinstance(result, tuple) else bool(result)
            if not success:
                logger.warning(
                    "submit_weights_async: chain rejected weights",
                    netuid=config.chain.netuid, count=len(weights_list),
                )
            else:
                logger.info(
                    "submit_weights_async: set weights on chain",
                    netuid=config.chain.netuid,
                    count=len(weights_list),
                    weights={int(uid): round(w, 4) for uid, w in zip(uids, weights_list, strict=True)},
                )
            return success
        except Exception as exc:
            msg = str(exc)
            retryable = isinstance(exc, (TimeoutError, asyncio.TimeoutError))
            if ConnectionClosedError is not None and isinstance(exc, ConnectionClosedError):
                retryable = True
            if "keepalive ping timeout" in msg or "ConnectionClosedError" in msg:
                retryable = True

            if attempt < max_retries and retryable:
                logger.warning(
                    "submit_weights_async: retrying",
                    attempt=attempt + 1, max_retries=max_retries, error=msg,
                )
                await asyncio.sleep(backoff_s * (attempt + 1))
                continue

            logger.warning("submit_weights_async: giving up", error=msg)
            return False

    return False
