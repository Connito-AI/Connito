"""Lightweight FastAPI server exposing this validator's per-round state.

Single endpoint: ``GET /v1/state.json`` — returns *only* the per-miner
loss signals and assignment metadata that aren't already published to
Prometheus or readable from chain. Specifically:

  - per-miner ``val_loss`` / ``delta_loss`` (high-cardinality time-series
    that we deliberately keep out of Prometheus to bound scrape cost)
  - the latest ``score`` and rolling-EMA ``weight_submitted`` per miner
    (the latter mirrors ``validator_miner_weight_submitted`` in
    Prometheus, included here to keep the leaderboard renderable from
    one fetch)
  - ``in_assignment`` — whether the miner is in this validator's
    deterministic foreground slice this round, i.e. whether the score
    is authoritative from this validator vs a background drive-by.

Aggregators that want hotkeys, HF repo/revision, phase / cycle state,
or baseline-loss history should pull those from the canonical sources
(``subtensor.metagraph``, ``get_chain_commits``, Prometheus' subnet/eval
gauges). They are intentionally NOT served here — keeping the validator
API responsibility narrow makes protocol changes that affect chain or
Prometheus fields invisible to this endpoint.

Wired into the validator process by ``connito/validator/run.py``; runs
on its own daemon thread (uvicorn event loop). All data is read from
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

logger = structlog.get_logger(__name__)


# ============================================================================
# App construction
# ============================================================================

def build_app(
    *,
    config,
    round_ref,                  # connito.validator.round.RoundRef
    score_aggregator,           # connito.validator.aggregator.MinerScoreAggregator
    wallet,
    validator_uid: int | None,
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
    app.state.wallet = wallet
    app.state.validator_uid = validator_uid
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

# Bumped to 2.0 in the slim-API rework: dropped subnet / phase / hotkey /
# hf_repo / baseline_loss_history fields. Aggregators that previously
# read those from this payload must source them from chain (metagraph,
# chain commits) or Prometheus (subnet_*, validator_eval_loss,
# validator_miner_weight_submitted). Major bump signals the schema is
# not backwards-compatible with v1.
PAYLOAD_VERSION = "2.0"


def _build_state_snapshot(s) -> dict[str, Any]:
    return {
        # Envelope metadata first — lets multi-validator aggregators detect
        # schema-skew between validators on different builds before parsing
        # the rest of the payload.
        "meta": _build_meta_section(s),
        "validator": _build_validator_section(s),
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


def _build_round_section(s) -> dict[str, Any]:
    """Round identity + baseline_loss only.

    Baseline is kept here (not in Prometheus) so a frontend rendering
    the leaderboard can derive ``delta_loss = max(0, baseline - val_loss)``
    consistently across all rows in one fetch — Prometheus
    ``validator_eval_loss`` gives the same value but with separate
    scrape staleness vs. the leaderboard's val_loss values.
    """
    rd = s.round_ref.current if s.round_ref else None
    if rd is None:
        return {"id": None, "baseline_loss": None}
    snap = rd.snapshot()
    return {
        "id": snap["round_id"],
        "baseline_loss": snap["baseline_loss"],
    }


def _build_leaderboard_section(s) -> list[dict[str, Any]]:
    """Per-miner leaderboard rows.

    Keeps only fields that aren't trivially derivable from chain
    (hotkey, hf_repo_id, hf_revision) or Prometheus (cycle/phase data).
    The remaining fields — per-miner loss, latest score, rolling EMA,
    in_assignment — are either too high-cardinality for Prometheus or
    only meaningful as part of a same-snapshot row, so they live here.
    """
    rd = s.round_ref.current if s.round_ref else None
    if rd is None:
        return []

    snap = rd.snapshot()
    foreground = set(snap["foreground_uids"])
    uid_to_hotkey = snap["uid_to_hotkey"]
    val_loss_by_uid = snap["val_loss_by_uid"]
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
    for uid in uid_to_hotkey:
        val_loss = val_loss_by_uid.get(uid)
        delta_loss: float | None = None
        if val_loss is not None and baseline_loss is not None:
            delta_loss = max(0.0, baseline_loss - val_loss)
        rows.append({
            "uid": uid,
            "score": latest_scores.get(uid),
            "delta_loss": delta_loss,
            "val_loss": val_loss,
            "weight_submitted": avg_scores.get(uid),
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
