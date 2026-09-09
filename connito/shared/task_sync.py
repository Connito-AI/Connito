"""Client for the owner API's active-task endpoints.

The owner decides which expert group the subnet trains, and serves that
decision from `cycle-api` alongside `/get_phase`:

    GET /active_task         -> {name, group_id, bundle_sha256}
    GET /active_task_bundle  -> the above plus config, expert_assignment,
                                shard_policy

Nodes poll the light endpoint once per cycle and compare `name` against the
task they are running. A change means fetch the bundle, materialize it, and
reconfigure. The endpoints take no arguments and answer only for *now* — there
is no end cycle and no next task — so a transition is detectable only once it
has happened, and there is deliberately nothing to count down to.

**Nothing here raises.** Every fetch returns `None` on any failure and the
caller keeps running the task it already has. An owner-API outage must never
stop a node mining.

**Schema drift is the risk this module carries.** The server lives in a
separate private repo, so the single shared definition that keeps
`PhaseResponse` honest does not exist here. Two things mitigate it: the models
below ignore unknown fields, so the server can add fields without a fleet-wide
client release; and `bundle_sha256` is re-computed locally and checked, which
turns a serialization disagreement into a refused transition instead of a
silently wrong one. Removing or renaming a field remains a breaking change
needing a coordinated release.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from connito.shared.app_logging import configure_logging, structlog
from connito.shared.cycle import _get_with_retry

configure_logging()
logger = structlog.get_logger(__name__)

CONFIG_FILE = "config.yaml"
ASSIGNMENT_FILE = "expert_assignment.json"
SHARD_POLICY_FILE = "shard_policy.json"
# Written last, after every payload file, and read to decide whether a
# re-materialize is needed at all.
STAMP_FILE = ".bundle_sha256"


class _Payload(BaseModel):
    """Base for the served models: tolerate fields we do not know about yet."""

    model_config = ConfigDict(extra="ignore")


class ActiveTask(_Payload):
    """The light poll. Carries no end cycle and no next task, by design."""

    name: str
    group_id: int
    bundle_sha256: str


class TaskBundle(_Payload):
    name: str
    group_id: int
    config: dict[str, Any]
    expert_assignment: dict[str, Any]
    shard_policy: dict[str, Any] | None = None
    bundle_sha256: str

    def canonical_sha256(self) -> str:
        """Re-compute the server's hash from the payload we received.

        Mirrors cycle-api's `_canonical_sha256` exactly: the same four keys, in
        the same canonical JSON form (sorted keys, no whitespace). `group_id`
        and `bundle_sha256` are deliberately excluded — they are derived from
        `config` and from this hash respectively. If the two repos ever
        disagree about this, every transition fails loudly here rather than
        materializing a payload we did not really agree on.
        """
        payload = {
            "name": self.name,
            "config": self.config,
            "expert_assignment": self.expert_assignment,
            "shard_policy": self.shard_policy,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _fetch(config, path: str, model: type[BaseModel]):
    """GET `path` from the owner API and parse it, or return None.

    Uses `cycle._get_with_retry` so the timeout/retry/backoff behaviour and the
    non-retryable status list stay identical to `get_phase_from_api`.
    """
    url = f"{config.cycle.owner_url}/{path}"
    resp = _get_with_retry(
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


def get_active_task_bundle(config) -> TaskBundle | None:
    """The active task's full payload, with its hash verified.

    A bundle whose contents do not hash to the `bundle_sha256` the server sent
    is rejected: the transition is refused and the node keeps its current task.
    """
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


def is_materialized(bundle: TaskBundle, dest_root: Path) -> bool:
    """True when `dest_root/<name>/` already holds exactly this bundle."""
    stamp = Path(dest_root) / bundle.name / STAMP_FILE
    try:
        return stamp.read_text(encoding="utf-8").strip() == bundle.bundle_sha256
    except OSError:
        return False


def materialize_task(bundle: TaskBundle, dest_root: Path) -> Path:
    """Write a bundle to `dest_root/<name>/` and return that directory.

    Idempotent: a directory already stamped with this bundle's hash is left
    alone. Otherwise the payload is written to a staging directory and swapped
    into place, so a crash mid-write can never leave a half-written task where
    the loader will find it. The stamp is written last, so an interrupted swap
    is retried rather than trusted.
    """
    dest_root = Path(dest_root)
    task_dir = dest_root / bundle.name
    if is_materialized(bundle, dest_root):
        return task_dir

    dest_root.mkdir(parents=True, exist_ok=True)
    staging = dest_root / f".tmp-{bundle.name}-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    try:
        (staging / CONFIG_FILE).write_text(
            yaml.safe_dump(bundle.config, sort_keys=False), encoding="utf-8"
        )
        (staging / ASSIGNMENT_FILE).write_text(
            json.dumps(bundle.expert_assignment), encoding="utf-8"
        )
        if bundle.shard_policy is not None:
            (staging / SHARD_POLICY_FILE).write_text(
                json.dumps(bundle.shard_policy), encoding="utf-8"
            )
        (staging / STAMP_FILE).write_text(bundle.bundle_sha256, encoding="utf-8")

        replaced = dest_root / f".old-{bundle.name}-{os.getpid()}"
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
