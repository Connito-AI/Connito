"""Client for the owner API's active-task endpoints.

The owner decides which expert group the subnet trains and serves that from
`cycle-api`, alongside `/get_phase`:

    GET /active_task         -> {name, group_id, bundle_sha256}
    GET /active_task_bundle  -> the above plus config, expert_assignment,
                                shard_policy

Nodes poll the light endpoint once per cycle and compare `name` to the task
they are running. The endpoints take no arguments and answer only for *now* —
no end cycle, no next task — so a transition is detectable only once it has
happened. **Nothing here raises:** every fetch returns None on failure and the
caller keeps the task it already has, because an owner-API outage must never
stop a node mining.

Schema drift is the risk this module carries: the server is a separate private
repo, so the single shared definition that keeps `PhaseResponse` honest does
not exist here. The models below ignore unknown fields, so the server can add
one without a fleet-wide client release, and `bundle_sha256` is re-computed
locally and checked, so a serialization disagreement refuses the transition
instead of materializing a payload we did not really agree on. Removing or
renaming a field is still a breaking change needing a coordinated release.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from connito.shared.app_logging import configure_logging, structlog
from connito.shared.config import CycleCfg
from connito.shared.helper import get_with_retry

configure_logging()
logger = structlog.get_logger(__name__)

# Written last, after every payload file, so an interrupted materialize is
# retried rather than trusted.
STAMP_FILE = ".bundle_sha256"

# Both models ignore unknown fields — that is what lets cycle-api add one
# without a coordinated client release.
_IGNORE_UNKNOWN = ConfigDict(extra="ignore")


class ActiveTask(BaseModel):
    """The light poll. Carries no end cycle and no next task, by design."""

    model_config = _IGNORE_UNKNOWN

    name: str
    group_id: int
    bundle_sha256: str


class TaskBundle(BaseModel):
    model_config = _IGNORE_UNKNOWN

    name: str
    group_id: int
    config: dict[str, Any]
    expert_assignment: dict[str, Any]
    shard_policy: dict[str, Any] | None = None
    bundle_sha256: str

    def canonical_sha256(self) -> str:
        """Re-compute the server's hash from the payload we received.

        Mirrors cycle-api's `_canonical_sha256`: the same four keys in the same
        canonical form (sorted, no whitespace). `group_id` and `bundle_sha256`
        are excluded — they are derived from `config` and from this hash.
        """
        payload = {
            "name": self.name,
            "config": self.config,
            "expert_assignment": self.expert_assignment,
            "shard_policy": self.shard_policy,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class _OwnerUrl:
    """Just enough of a config for `_fetch` when none has been built yet."""

    def __init__(self, owner_url: str | None) -> None:
        defaults = CycleCfg.model_fields
        self.cycle = SimpleNamespace(
            owner_url=owner_url or defaults["owner_url"].default,
            api_timeout_sec=defaults["api_timeout_sec"].default,
            api_retries=defaults["api_retries"].default,
            api_backoff_sec=defaults["api_backoff_sec"].default,
        )


def _fetch(config, path: str, model: type[BaseModel]):
    """GET `path` from the owner API and parse it, or return None.

    Shares `helper.get_with_retry` with `get_phase_from_api`, so timeout,
    retry, backoff and the non-retryable status list stay identical.
    """
    url = f"{config.cycle.owner_url}/{path}"
    resp = get_with_retry(
        url,
        timeout=config.cycle.api_timeout_sec,
        retries=config.cycle.api_retries,
        backoff=config.cycle.api_backoff_sec,
    )
    if resp is None:
        return None
    try:
        return model(**resp.json())
    except (ValueError, TypeError) as exc:
        # ValueError: JSON decode; TypeError: missing/unexpected fields.
        logger.error("bad active-task payload", url=url, error=str(exc))
        return None


def get_active_task(config) -> ActiveTask | None:
    """Which expert group the subnet is training right now."""
    return _fetch(config, "active_task", ActiveTask)


def resolve_active_task_name(config_path: str | Path) -> str | None:
    """The task an entrypoint should start on, or None to use the config's own.

    Entrypoints call this *before* building their config and pass the result to
    `WorkerConfig.from_path`, so the name is right from the start and nothing is
    ever derived from a stale one. Config itself never reaches for the network —
    it only receives an answer.

    Reads `cycle.owner_url` straight from the YAML because there is no config
    object yet; an absent or unreadable file falls through to the shipped
    default, which is the same URL every node uses.
    """
    try:
        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        owner_url = (raw.get("cycle") or {}).get("owner_url")
    except (OSError, yaml.YAMLError):
        owner_url = None

    active = get_active_task(_OwnerUrl(owner_url))
    if active is None:
        logger.warning("Owner API unreachable — starting on the task from config")
        return None
    return active.name


def get_active_task_bundle(config) -> TaskBundle | None:
    """The active task's full payload, refused if it fails its own hash."""
    bundle = _fetch(config, "active_task_bundle", TaskBundle)
    if bundle is None:
        return None
    computed = bundle.canonical_sha256()
    if computed != bundle.bundle_sha256:
        logger.error(
            "task bundle failed its own hash — refusing it",
            task=bundle.name,
            served=bundle.bundle_sha256,
            computed=computed,
        )
        return None
    return bundle


def materialize_task(bundle: TaskBundle, dest_root: Path) -> Path:
    """Write a bundle to `dest_root/<name>/`, mirroring `expert_groups/<name>/`.

    Idempotent: a directory already stamped with this bundle's hash is left
    alone. Otherwise the payload is staged in a sibling `.tmp_` directory and
    swapped in, so a crash mid-write can never leave a half-written task where
    the loader will find it, and a failed write leaves the previous task intact.
    """
    dest_root = Path(dest_root)
    task_dir = dest_root / bundle.name
    try:
        if (task_dir / STAMP_FILE).read_text(encoding="utf-8").strip() == bundle.bundle_sha256:
            return task_dir
    except OSError:
        pass  # absent or unreadable stamp -> (re)materialize

    dest_root.mkdir(parents=True, exist_ok=True)
    staging = dest_root / f".tmp_{bundle.name}_{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    try:
        (staging / "config.yaml").write_text(
            yaml.safe_dump(bundle.config, sort_keys=False), encoding="utf-8"
        )
        (staging / "expert_assignment.json").write_text(
            json.dumps(bundle.expert_assignment), encoding="utf-8"
        )
        if bundle.shard_policy is not None:
            (staging / "shard_policy.json").write_text(
                json.dumps(bundle.shard_policy), encoding="utf-8"
            )
        (staging / STAMP_FILE).write_text(bundle.bundle_sha256, encoding="utf-8")

        replaced = dest_root / f".old_{bundle.name}_{os.getpid()}"
        if task_dir.exists():
            os.replace(task_dir, replaced)
        os.replace(staging, task_dir)
        shutil.rmtree(replaced, ignore_errors=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    logger.info(
        "materialized task",
        task=bundle.name,
        group_id=bundle.group_id,
        path=str(task_dir),
        bundle_sha256=bundle.bundle_sha256,
    )
    return task_dir
