"""Connito Telemetry Gateway — FastAPI aggregator.

Polls one or more Connito validators' ``/v1/state.json`` endpoints on a
fixed interval, caches the responses in memory, and serves a single
``/api/leaderboard`` endpoint to the frontend SPA. Adds CORS so the SPA
can fetch from a different origin.

Single-validator MVP: the lone validator's response is served directly.
The reducer slot for multi-validator aggregation (mean score / mean
weight per UID) is in ``_reduce_multi`` — wired but stubbed; replace
when a second validator comes online.

----------------------------------------------------------------------
Run locally with uvicorn:

    cd observability/api_gateway
    pip install -r requirements.txt
    VALIDATOR_URLS="http://127.0.0.1:8300" uvicorn main:app --host 0.0.0.0 --port 8400

Run via Docker:

    cd observability/api_gateway
    docker build -t connito-telemetry-gateway .
    docker run -p 8400:8400 -e VALIDATOR_URLS="http://10.0.0.1:8300" connito-telemetry-gateway

Configuration (env vars):
    VALIDATOR_URLS              Comma-separated base URLs, no trailing /v1/...
                                e.g. "http://10.0.0.1:8300,http://10.0.0.2:8300"
    POLL_INTERVAL_SECONDS       Poll cadence (default: 12.0)
    POLL_TIMEOUT_SECONDS        Per-request timeout (default: 10.0)
    STALE_AFTER_FAILURES        Mark cache stale after N consecutive
                                failures (default: 3)
    CORS_ALLOW_ORIGINS          "*" or comma-separated list (default: "*")
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    validator_urls: str = "http://127.0.0.1:8300"
    poll_interval_seconds: float = 12.0
    poll_timeout_seconds: float = 10.0
    stale_after_failures: int = 3
    cors_allow_origins: str = "*"


def _parse_urls(raw: str) -> list[str]:
    return [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]


settings = Settings()
VALIDATOR_URLS: list[str] = _parse_urls(settings.validator_urls)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("connito.gateway")


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    payload: dict[str, Any] | None = None
    last_success_ts: float | None = None
    consecutive_failures: int = 0
    last_error: str | None = None


cache: dict[str, CacheEntry] = {url: CacheEntry() for url in VALIDATOR_URLS}


def _is_stale(entry: CacheEntry) -> bool:
    return entry.consecutive_failures >= settings.stale_after_failures


# ---------------------------------------------------------------------------
# Poller
# ---------------------------------------------------------------------------

async def _poll_one(client: httpx.AsyncClient, url: str) -> None:
    target = f"{url}/v1/state.json"
    entry = cache[url]
    try:
        resp = await client.get(target, timeout=settings.poll_timeout_seconds)
        resp.raise_for_status()
        entry.payload = resp.json()
        entry.last_success_ts = time.time()
        entry.consecutive_failures = 0
        entry.last_error = None
    except Exception as e:
        entry.consecutive_failures += 1
        entry.last_error = f"{type(e).__name__}: {e}"
        logger.warning(
            "poll failed url=%s consecutive=%d error=%s",
            url, entry.consecutive_failures, entry.last_error,
        )


async def _poller_loop() -> None:
    """Forever-running background task. Sleeps between sweeps; per sweep,
    polls every configured validator concurrently."""
    async with httpx.AsyncClient() as client:
        while True:
            try:
                await asyncio.gather(
                    *(_poll_one(client, url) for url in VALIDATOR_URLS),
                    return_exceptions=False,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("poller sweep crashed; continuing")
            await asyncio.sleep(settings.poll_interval_seconds)


# ---------------------------------------------------------------------------
# Lifespan — start/stop the poller alongside the FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001 — FastAPI signature
    if not VALIDATOR_URLS:
        logger.error("VALIDATOR_URLS is empty; gateway will serve no data")
        task: asyncio.Task | None = None
    else:
        logger.info("Starting poller for: %s (interval=%ss)",
                    VALIDATOR_URLS, settings.poll_interval_seconds)
        task = asyncio.create_task(_poller_loop(), name="connito-gateway-poller")

    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("_poller_loop did not exit cleanly")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Connito Telemetry Gateway",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)

_cors_origins = (
    ["*"] if settings.cors_allow_origins.strip() == "*"
    else _parse_urls(settings.cors_allow_origins)
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    # Wildcard origins are incompatible with credentials per the CORS spec;
    # keep credentials off so allow_origins=["*"] actually works in browsers.
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _meta(*, stale: bool, last_success_ts: float | None,
          stale_reason: str | None, served_from: str | None = None) -> dict[str, Any]:
    return {
        "validator_count": len(VALIDATOR_URLS),
        "polled_validator_count": sum(1 for c in cache.values() if c.payload is not None),
        "last_success_ts": last_success_ts,
        "poll_interval_seconds": settings.poll_interval_seconds,
        "stale": stale,
        "stale_reason": stale_reason,
        "served_from": served_from,
    }


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    """Container readiness probe. Does NOT depend on validator reachability —
    the gateway itself is healthy as long as the process is up."""
    return {"status": "ok", "validator_count": len(VALIDATOR_URLS)}


@app.get("/api/leaderboard")
async def leaderboard() -> dict[str, Any]:
    if not VALIDATOR_URLS:
        raise HTTPException(503, detail="No validators configured (VALIDATOR_URLS empty)")

    if len(VALIDATOR_URLS) == 1:
        url, entry = next(iter(cache.items()))
        if entry.payload is None:
            raise HTTPException(
                503,
                detail=f"No data yet from validator (last error: {entry.last_error})",
            )
        stale = _is_stale(entry)
        return {
            "data": entry.payload,
            "meta": _meta(
                stale=stale,
                last_success_ts=entry.last_success_ts,
                stale_reason=entry.last_error if stale else None,
                served_from=url,
            ),
        }

    return _reduce_multi()


def _reduce_multi() -> dict[str, Any]:
    """Multi-validator aggregation slot.

    Today: returns the freshest validator's payload (correct for an MVP
    when at most one validator is reachable). When you onboard validator
    N+1, replace this with the proper reducer:

      - leaderboard rows: union of UIDs; per UID:
          score             = mean(latest scores across validators)
          weight_submitted  = mean(weights across validators)
          delta_loss        = mean(delta_loss where reported)
          val_loss          = mean(val_loss where reported)
          hf_repo_id        = mode (most common); fall back to any
          hf_revision       = same
          in_assignment     = OR across validators (if ANY validator
                              had this miner in its foreground, true)
      - phase: take from the validator with the highest head_block
      - round.stats: sum across validators' counts (or use the leader)

    Until that's wired, callers of /api/leaderboard get the freshest
    single-validator view + a meta.served_from field naming the source.
    """
    fresh = sorted(
        ((url, c) for url, c in cache.items() if c.payload is not None),
        key=lambda x: x[1].last_success_ts or 0.0,
        reverse=True,
    )
    if not fresh:
        raise HTTPException(
            503,
            detail="No validator has returned data yet",
        )

    url, entry = fresh[0]
    all_stale = all(_is_stale(c) for c in cache.values())
    return {
        "data": entry.payload,
        "meta": _meta(
            stale=all_stale,
            last_success_ts=entry.last_success_ts,
            stale_reason=entry.last_error if all_stale else None,
            served_from=url,
        ),
    }
