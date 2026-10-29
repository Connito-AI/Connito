"""Blend the prompt validator's weights, published on prompt-record, into the training validator's.

The round's weight submission passes its miner weights through `ablend_prompt_weights` before the subnet-owner
share, behind `evaluation.prompt_blend`; the fallbacks never blend. Any doubt about the payload submits training only.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass

import requests
import urllib3

from connito.shared.app_logging import structlog
from connito.shared.btauth import wallet_signer
from connito.shared.chain import reserve_subnet_owner_share
from connito.shared.helper import get_with_retry

logger = structlog.get_logger(__name__)

WEIGHTS_ROUTE = "/v1/validator/weights/latest"  # prompt-record #68: signed, validators and the owner
TEST_WEIGHTS_ROUTE = "/v1/test/validator/weights/latest"  # the same on the server's test tables
FETCH_TIMEOUT_S = 5  # per connect or read
READ_BYTES = 4096  # read in small pieces, so the deadline is checked while a server trickles
MAX_BODY_BYTES = 4 * 1024 * 1024
CYCLE_API_TIMEOUT_S = 5
BLEND_DEADLINE_S = 15.0  # the whole blend: metagraph read, fetch, cycle-api and the whitelist
DEADLINE_GRACE_S = 1.0  # the async side waits this past the deadline for the thread's own answer
TRAINING_CYCLES_PER_PROMPT_CYCLE = 13  # prompt-record's prompt cycle, as the prompt validator's scoring_cycle_k
MAX_AGE_PROMPT_CYCLES = 2
TRAINING_ONLY = "prompt blend: submitting training weights only"


class BlendRefused(Exception):
    """The reason this round submits training weights only."""


class _Late(Exception):
    """The blend finished past the deadline its caller waits by."""


@dataclass(frozen=True)
class PromptWeights:
    cycle_id: int
    rules_sha256: str
    validator_hotkey: str
    weights: dict[str, float]


def _number(value) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BlendRefused("malformed payload: a weight or the total is not a number")
    try:
        number = float(value)
    except OverflowError:
        raise BlendRefused("malformed payload: a number too large") from None
    if not math.isfinite(number) or number < 0:
        raise BlendRefused("malformed payload: a number not finite or below 0")
    return number


def parse_prompt_weights(payload) -> PromptWeights:
    """prompt-record's latest weights (#68): {cycle_id, validator_hotkey, published_at, rules_sha256,
    weights: {hotkey: w}, total, prompt_weights, prompt_total}. Only `weights` is used, not normalized; the prompt
    miners' `prompt_weights` and `prompt_total` are not read."""
    if not isinstance(payload, dict):
        raise BlendRefused("malformed payload: not an object")
    cycle_id, rules, hotkey, weights = (payload.get(k) for k in
                                        ("cycle_id", "rules_sha256", "validator_hotkey", "weights"))
    if not isinstance(cycle_id, int) or isinstance(cycle_id, bool):
        raise BlendRefused("malformed payload: cycle_id")
    if not isinstance(rules, str) or not isinstance(hotkey, str) or not isinstance(payload.get("published_at"), str):
        raise BlendRefused("malformed payload: rules_sha256, validator_hotkey or published_at")
    if not isinstance(weights, dict) or not all(isinstance(hk, str) for hk in weights):
        raise BlendRefused("malformed payload: weights")
    parsed = {hk: _number(w) for hk, w in weights.items()}
    if not math.isclose(sum(parsed.values()), _number(payload.get("total")), rel_tol=1e-6, abs_tol=1e-9):
        raise BlendRefused("malformed payload: the weights don't add up to total")
    if not sum(parsed.values()):
        raise BlendRefused("no weights published (total 0)")
    return PromptWeights(cycle_id, rules, hotkey, parsed)


def fetch_prompt_weights(url: str, route: str, wallet, deadline: float) -> PromptWeights:
    """One signed GET, its body read by `deadline` (time.monotonic) and within MAX_BODY_BYTES."""
    headers = wallet_signer(wallet)("GET", route, b"") | {"Accept-Encoding": "identity"}  # signed as sent
    body = bytearray()
    try:
        with requests.get(url.rstrip("/") + route, headers=headers, timeout=FETCH_TIMEOUT_S, stream=True,
                          allow_redirects=False) as resp:
            if resp.status_code == 404:
                raise BlendRefused("nothing published yet (404)")
            if resp.status_code != 200:
                raise BlendRefused(f"fetch failed: HTTP {resp.status_code}")
            # read1 returns what has arrived rather than waiting for a full piece.
            while chunk := resp.raw.read1(READ_BYTES):
                body += chunk
                if len(body) > MAX_BODY_BYTES:
                    raise BlendRefused(f"body over {MAX_BODY_BYTES} bytes")
                if time.monotonic() > deadline:
                    raise BlendRefused("fetch past the blend's deadline")
    except (requests.RequestException, urllib3.exceptions.HTTPError) as e:
        raise BlendRefused(f"fetch failed: {type(e).__name__}") from None
    try:
        payload = json.loads(body)
    except ValueError:
        raise BlendRefused("malformed payload: not JSON") from None
    return parse_prompt_weights(payload)


def _cycle_api(config, route: str):
    """One short attempt, as the blend has its own deadline; None if it fails."""
    timeout = min(config.cycle.api_timeout_sec, CYCLE_API_TIMEOUT_S)
    resp = get_with_retry(f"{config.cycle.owner_url}{route}", timeout=timeout, retries=0)
    try:
        return None if resp is None else resp.json()
    except ValueError:
        return None


def current_prompt_cycle(config) -> int:
    """cycle-api's prompt cycle, as the prompt validator numbers it: the local cycle config isn't production's."""
    phase = _cycle_api(config, "/get_phase")
    if phase is None:
        raise BlendRefused("cycle-api unreachable")
    return int(phase["cycle_index"]) // TRAINING_CYCLES_PER_PROMPT_CYCLE


def _validator_uids(config, uid_of: dict[str, int]) -> set[int]:
    # Unlike get_validator_whitelist_from_api, an outage refuses rather than reading as no validators.
    whitelist = _cycle_api(config, "/get_validator_whitelist")
    if not isinstance(whitelist, list) or not whitelist:
        raise BlendRefused("validator whitelist unavailable")
    return {uid_of[hk] for hk in whitelist if hk in uid_of}


def _normalized(weights: dict[int, float]) -> dict[int, float]:
    total = sum(weights.values())
    return {uid: w / total for uid, w in weights.items()}


def blend(train: dict[int, float], prompt: dict[int, float], share: float) -> dict[int, float]:
    """(1 − share)·train + share·prompt, each normalized first, then the sum normalized."""
    out = {uid: (1.0 - share) * w for uid, w in _normalized(train).items()}
    for uid, w in _normalized(prompt).items():
        out[uid] = out.get(uid, 0.0) + share * w
    return _normalized(out)


@dataclass(frozen=True)
class _Prompt:
    published: PromptWeights
    by_uid: dict[int, float]
    off_metagraph: int
    on_validators: int


def _prompt_uid_weights(config, cfg, wallet, metagraph, deadline: float) -> _Prompt:
    """The checked payload and its weights by uid, less hotkeys off the metagraph and validators' uids."""
    if not cfg.prompt_blend_validator_hotkey:
        raise BlendRefused("no prompt_blend_validator_hotkey configured")
    if not cfg.prompt_blend_rules_sha256:
        raise BlendRefused("no prompt_blend_rules_sha256 configured")
    route = TEST_WEIGHTS_ROUTE if cfg.prompt_blend_test_route else WEIGHTS_ROUTE
    published = fetch_prompt_weights(cfg.prompt_blend_url, route, wallet, deadline)
    if published.validator_hotkey != cfg.prompt_blend_validator_hotkey:
        raise BlendRefused(f"published by {published.validator_hotkey}, not the configured prompt validator")
    if published.rules_sha256 not in cfg.prompt_blend_rules_sha256:
        raise BlendRefused(f"rules_sha256 {published.rules_sha256} not allowed")
    # Test cycles are opened and closed by hand, numbered from 1,000,000,000: cycle-api can't date them.
    if not cfg.prompt_blend_test_route:
        current = current_prompt_cycle(config)
        if not 0 <= current - published.cycle_id <= MAX_AGE_PROMPT_CYCLES:
            raise BlendRefused(f"cycle_id {published.cycle_id} too far from the current prompt cycle {current}")
    uid_of = {hk: uid for uid, hk in enumerate(metagraph.hotkeys)}
    # A validator's uid earns nothing from miner weights, and weight on our own uid would be a self-vote.
    excluded = _validator_uids(config, uid_of) | {uid_of.get(wallet.hotkey.ss58_address, -1)}
    on_metagraph = {uid_of[hk]: w for hk, w in published.weights.items() if hk in uid_of and w > 0}
    by_uid = {uid: w for uid, w in on_metagraph.items() if uid not in excluded}
    off_metagraph = sum(1 for hk in published.weights if hk not in uid_of)
    if sum(by_uid.values()) <= 0:
        raise BlendRefused(f"no prompt weight left ({off_metagraph} hotkeys not on the metagraph)")
    return _Prompt(published, by_uid, off_metagraph, len(on_metagraph) - len(by_uid))


def _compact(weights: dict[int, float]) -> dict[int, float]:
    return {uid: round(w, 4) for uid, w in sorted(weights.items(), key=lambda kv: -kv[1]) if w > 0}


def prompt_blend_mode(config) -> str:
    # A config with no evaluation section (a test's stand-in) is off.
    return getattr(getattr(config, "evaluation", None), "prompt_blend", "off")


def blend_prompt_weights(config, wallet, train: dict[int, float], metagraph, deadline: float) -> dict[int, float]:
    """The miner weights to submit: `train` unless the switch is on and the published prompt weights pass every check.

    A failure is logged with its reason and submits `train`. The caller stops waiting soon after `deadline`
    (time.monotonic): a late result is discarded, logged at debug only, and raises _Late for the caller to report.
    """
    cfg = config.evaluation
    mode = cfg.prompt_blend
    try:
        clean = {uid: w for uid, w in train.items() if math.isfinite(w) and w > 0}
        if not clean:
            raise BlendRefused("no training weight to blend with")
        prompt = _prompt_uid_weights(config, cfg, wallet, metagraph, deadline)
        blended = blend(clean, prompt.by_uid, cfg.prompt_blend_share)
        if time.monotonic() > deadline:
            raise _Late
    except Exception as e:
        if time.monotonic() > deadline:
            logger.debug("prompt blend: result discarded after the deadline", mode=mode)
            raise _Late from None
        reason = str(e) if isinstance(e, BlendRefused) else f"error: {type(e).__name__}"
        logger.warning(TRAINING_ONLY, mode=mode, reason=reason)
        return train
    fields = dict(
        cycle_id=prompt.published.cycle_id,
        prompt_hotkeys={uid: metagraph.hotkeys[uid] for uid in sorted(prompt.by_uid)},  # to check uid ↔ hotkey
        dropped_off_metagraph=prompt.off_metagraph,
        dropped_on_validators=prompt.on_validators,
        dropped_training=len(train) - len(clean),
        share=cfg.prompt_blend_share,
        blended=_compact(blended),
        with_owner_share=_compact(reserve_subnet_owner_share(blended)),
    )
    if mode == "dry_run":
        logger.info("prompt blend dry run: submitting training weights only", **fields)
        return train
    logger.info("prompt blend: submitting blended weights", **fields)
    return blended


async def ablend_prompt_weights(config, wallet, train: dict[int, float], async_subtensor) -> dict[int, float]:
    """`blend_prompt_weights` off the event loop after a metagraph read, both within one BLEND_DEADLINE_S.

    A thread still running past it may run on briefly, silent.
    """
    mode = prompt_blend_mode(config)
    if mode == "off":
        return train
    deadline = time.monotonic() + BLEND_DEADLINE_S
    try:
        metagraph = await asyncio.wait_for(
            async_subtensor.metagraph(netuid=config.chain.netuid), deadline - time.monotonic(),
        )
        return await asyncio.wait_for(
            asyncio.to_thread(blend_prompt_weights, config, wallet, train, metagraph, deadline),
            deadline - time.monotonic() + DEADLINE_GRACE_S,
        )
    except Exception as e:
        late = isinstance(e, asyncio.TimeoutError | _Late)
        reason = f"past the {BLEND_DEADLINE_S:g} s deadline" if late else f"async: {type(e).__name__}"
        logger.warning(TRAINING_ONLY, mode=mode, reason=reason)
        return train
