"""Lightweight FastAPI server exposing this validator's per-round state.

Single endpoint: ``GET /v1/state.json`` — returns the current phase, round
stats, and the leaderboard from this validator's perspective.

Wired into the validator process by ``connito/validator/run.py``; runs on
its own daemon thread (uvicorn event loop). All data is read from
in-memory references — no chain RPC, no database. Reads are best-effort
and may be slightly stale relative to the main loop; that's acceptable
for an admin / observability endpoint.

Critical caveat for consumers of this endpoint: a single validator only
authoritatively scores its own ``foreground_uids`` slice (per
``Round.freeze`` in connito/validator/round.py). Each leaderboard row
carries an ``in_assignment`` flag so callers can render off-assignment
rows distinctly. A network-wide leaderboard requires aggregating across
every whitelisted validator's ``/v1/state.json``.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, Request

from connito.shared.app_logging import structlog
from connito.shared.telemetry import SUBNET_CURRENT_BLOCK

logger = structlog.get_logger(__name__)


# ============================================================================
# App construction
# ============================================================================

def build_app(
    *,
    config,
    round_ref,                  # connito.validator.round.RoundRef
    score_aggregator,           # connito.validator.aggregator.MinerScoreAggregator
    phase_manager,              # connito.sn_owner.cycle.PhaseManager
    lite_subtensor,
    wallet,
    validator_uid: int | None,
    baseline_loss_history=None, # connito.validator.baseline_history.BaselineLossHistory
    git_version: str | None = None,  # build tag, surfaced in /v1/state.json meta
) -> FastAPI:
    """Build a FastAPI app with the /v1/state.json endpoint.

    Runtime references are stashed on ``app.state`` so request handlers
    read them without module globals. Pass real refs once at startup.
    """
    app = FastAPI(
        title="Connito Validator State API",
        # Admin endpoint — no Swagger UI, no schema exposure.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    app.state.config = config
    app.state.round_ref = round_ref
    app.state.score_aggregator = score_aggregator
    app.state.phase_manager = phase_manager
    app.state.lite_subtensor = lite_subtensor
    app.state.wallet = wallet
    app.state.validator_uid = validator_uid
    app.state.baseline_loss_history = baseline_loss_history
    app.state.git_version = git_version

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/state.json")
    async def state(request: Request) -> dict[str, Any]:
        return _build_state_snapshot(request.app.state)

    return app


# ============================================================================
# Snapshot composition — each section is best-effort and isolated
# ============================================================================

PAYLOAD_VERSION = "1.0"


def _build_state_snapshot(s) -> dict[str, Any]:
    return {
        # Envelope metadata first — lets multi-validator aggregators detect
        # schema-skew between validators on different builds before parsing
        # the rest of the payload.
        "meta": _build_meta_section(s),
        "validator": _build_validator_section(s),
        "subnet": _build_subnet_section(s),
        "phase": _build_phase_section(s),
        "round": _build_round_section(s),
        "leaderboard": _build_leaderboard_section(s),
    }


def _build_meta_section(s) -> dict[str, Any]:
    """Aggregator-facing payload metadata. Bump ``payload_version`` on any
    breaking schema change so downstream consumers can fail fast."""
    return {
        "payload_version": PAYLOAD_VERSION,
        "connito_version": getattr(s, "git_version", None),
        "snapshot_ts": time.time(),
    }


def _build_validator_section(s) -> dict[str, Any]:
    try:
        hotkey = s.wallet.hotkey.ss58_address if s.wallet else None
    except Exception:
        hotkey = None
    return {"hotkey": hotkey, "uid": s.validator_uid}


def _build_subnet_section(s) -> dict[str, Any]:
    """Subnet shape — netuid (static) + total_miners and validator_count
    (frozen onto Round at the start of each cycle)."""
    netuid: int | None = None
    try:
        netuid = int(getattr(s.config.chain, "netuid", None))
    except (TypeError, ValueError, AttributeError):
        netuid = None

    rd = s.round_ref.current if s.round_ref else None
    if rd is None:
        return {"netuid": netuid, "total_miners": None, "validator_count": None}

    snap = rd.snapshot()
    total_uids = snap.get("total_subnet_uids") or 0
    validator_count = snap.get("validator_count") or 0
    # "Total miners" = every UID minus the whitelisted validators.
    total_miners = max(0, total_uids - validator_count) if total_uids else None
    return {
        "netuid": netuid,
        "total_miners": total_miners,
        "validator_count": validator_count or None,
    }


# Phase-name → actor mapping. PhaseNames are string-typed (see
# connito/shared/cycle.py:PhaseNames), so equality-string lookups are stable
# as long as the enum values stay in sync.
_VALIDATOR_PHASES = frozenset({
    "Validate", "Merge", "ValidatorCommit1", "ValidatorCommit2",
})
_MINER_PHASES = frozenset({
    "Distribute", "Train", "MinerCommit1", "MinerCommit2", "Submission",
})


def _actor_for_phase(name: str) -> str:
    if name in _VALIDATOR_PHASES:
        return "Validator"
    if name in _MINER_PHASES:
        return "Miner"
    return "Unknown"


def _compute_upcoming_phases(phase_manager, head_block: int) -> list[tuple[str, int, int]]:
    """Local mirror of ``PhaseManager.blocks_until_next_phase`` that uses a
    caller-supplied ``head_block`` instead of ``self.subtensor.block``.

    The original calls into the shared ``lite_subtensor``'s websocket — fine
    on the main thread but a race against this validator's main loop when
    invoked from the API request thread (caused a fatal
    ``websockets.ConcurrencyError`` in production). This pure-computation
    variant uses only static phase config + the cached head_block so the API
    thread never touches the shared websocket.

    Returns ``[(name, start_block, end_block), ...]`` for every phase in the
    cycle, in declaration order.
    """
    cycle_len = phase_manager.cycle_length
    cycle_block_index = head_block % cycle_len
    cycle_start_block = head_block - cycle_block_index

    out: list[tuple[str, int, int]] = []
    start = 0
    for phase in phase_manager.phases:
        phase_start = start
        if phase_start >= cycle_block_index:
            start_block = cycle_start_block + phase_start
        else:
            start_block = cycle_start_block + cycle_len + phase_start
        end_block = start_block + phase["length"] - 1
        out.append((phase["name"], start_block, end_block))
        start += phase["length"]
    return out


def _build_phase_section(s) -> dict[str, Any]:
    # Read the cached head block from the Prometheus gauge that
    # SystemStatePoller refreshes every ~12s. Avoids a concurrent websocket
    # `recv()` against the shared lite_subtensor (the main loop's reads of
    # `lite_subtensor.block` raced this previously and crashed the validator
    # with `websockets.ConcurrencyError`).
    head_block: int | None = None
    try:
        cached = SUBNET_CURRENT_BLOCK._value.get()
        if cached:
            head_block = int(cached)
    except Exception as e:
        logger.debug("api: failed to read cached head block", error=str(e))

    if head_block is None or s.phase_manager is None:
        return {
            "name": None, "index": None, "head_block": head_block,
            "started_at_block": None, "ends_at_block": None,
            "blocks_remaining": None, "cycle_index": None, "cycle_length": None,
            "upcoming": [],
        }

    try:
        # `get_phase(block)` is safe — pass the explicit block and PhaseManager
        # uses it directly without re-fetching from chain. (SystemStatePoller
        # already calls it the same way from a background thread.)
        pr = s.phase_manager.get_phase(head_block)
    except Exception as e:
        logger.debug("api: failed to resolve current phase", error=str(e))
        return {
            "name": None, "index": None, "head_block": head_block,
            "started_at_block": None, "ends_at_block": None,
            "blocks_remaining": None, "cycle_index": None, "cycle_length": None,
            "upcoming": [],
        }

    upcoming: list[dict[str, Any]] = []
    try:
        # Local pure-computation; no websocket touch (see _compute_upcoming_phases).
        all_phases = _compute_upcoming_phases(s.phase_manager, head_block)
        candidates = [
            (name, start_block)
            for (name, start_block, _end_block) in all_phases
            if start_block > pr.phase_start_block
        ]
        candidates.sort(key=lambda x: x[1])
        upcoming = [
            {"name": name, "start_block": start, "actor": _actor_for_phase(name)}
            for name, start in candidates[:3]
        ]
    except Exception as e:
        logger.debug("api: failed to enumerate upcoming phases", error=str(e))

    return {
        "name": pr.phase_name,
        "index": pr.phase_index,
        "started_at_block": pr.phase_start_block,
        "ends_at_block": pr.phase_end_block,
        "blocks_remaining": pr.blocks_remaining_in_phase,
        "head_block": head_block,
        "cycle_index": pr.cycle_index,
        "cycle_length": pr.cycle_length,
        "upcoming": upcoming,
    }


def _build_round_section(s) -> dict[str, Any]:
    history: list[dict[str, Any]] = []
    if s.baseline_loss_history is not None:
        try:
            history = s.baseline_loss_history.snapshot()
        except Exception as e:
            logger.debug("api: baseline_loss_history.snapshot failed", error=str(e))

    rd = s.round_ref.current if s.round_ref else None
    if rd is None:
        return {
            "id": None, "baseline_loss": None, "stats": None,
            "baseline_loss_history": history,
        }
    snap = rd.snapshot()
    return {
        "id": snap["round_id"],
        "baseline_loss": snap["baseline_loss"],
        "stats": snap["stats"],
        "baseline_loss_history": history,
    }


def _build_leaderboard_section(s) -> list[dict[str, Any]]:
    rd = s.round_ref.current if s.round_ref else None
    if rd is None:
        return []

    snap = rd.snapshot()
    foreground = set(snap["foreground_uids"])
    uid_to_hotkey = snap["uid_to_hotkey"]
    val_loss_by_uid = snap["val_loss_by_uid"]
    chain_ckpts = snap["uid_to_chain_checkpoint"]
    baseline_loss = snap["baseline_loss"]

    # `latest` = most recent score per uid (this round's score if scored).
    # `avg` = the rolling weight this validator submits to chain
    # (mirrors connito/validator/run.py:async_submit_weight which uses
    # uid_score_pairs(how="avg")). Two queries, one lock acquire each.
    try:
        latest_scores = s.score_aggregator.uid_score_pairs(how="latest") if s.score_aggregator else {}
    except Exception as e:
        logger.debug("api: score_aggregator.uid_score_pairs(latest) failed", error=str(e))
        latest_scores = {}
    try:
        avg_scores = s.score_aggregator.uid_score_pairs(how="avg") if s.score_aggregator else {}
    except Exception as e:
        logger.debug("api: score_aggregator.uid_score_pairs(avg) failed", error=str(e))
        avg_scores = {}

    rows: list[dict[str, Any]] = []
    for uid, hotkey in uid_to_hotkey.items():
        ck = chain_ckpts.get(uid)
        val_loss = val_loss_by_uid.get(uid)
        delta_loss: float | None = None
        if val_loss is not None and baseline_loss is not None:
            delta_loss = max(0.0, baseline_loss - val_loss)
        rows.append({
            "uid": uid,
            "hotkey": hotkey,
            "score": latest_scores.get(uid),
            "delta_loss": delta_loss,
            "val_loss": val_loss,
            "weight_submitted": avg_scores.get(uid),
            "hf_repo_id": getattr(ck, "hf_repo_id", None),
            "hf_revision": getattr(ck, "hf_revision", None),
            "in_assignment": uid in foreground,
        })

    # Score desc; rows with no score sink to the bottom.
    rows.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0.0)))
    return rows


# ============================================================================
# Server lifecycle — runs uvicorn on a daemon thread
# ============================================================================

class StateAPIServer:
    """Run a uvicorn server on a daemon thread with clean shutdown.

    Use ``start()`` to launch and ``stop(timeout=...)`` to drain. Mirrors
    the ``SystemStatePoller`` lifecycle so the validator's main loop can
    treat it the same way during exception/cleanup paths.
    """

    def __init__(self, app: FastAPI, host: str, port: int) -> None:
        self._uvicorn_config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
            # Don't install signal handlers — uvicorn would otherwise try to
            # capture SIGINT/SIGTERM, stealing them from the validator's
            # main process which already manages KeyboardInterrupt.
            workers=1,
        )
        self._server = uvicorn.Server(self._uvicorn_config)
        self._server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        self._thread = threading.Thread(
            target=self._server.run,
            name="connito-validator-api",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()
        logger.info(
            "Validator state API started",
            host=self._uvicorn_config.host,
            port=self._uvicorn_config.port,
        )

    def stop(self, timeout: float = 5.0) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning(
                "Validator state API thread did not exit within timeout",
                timeout=timeout,
            )
