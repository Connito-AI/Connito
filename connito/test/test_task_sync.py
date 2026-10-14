"""Client for the owner API's active-task endpoints, checked against the server's own hash.

`active_task*.json` come from cycle-api's bundle builder, pending its deploy (recapture then);
`active_task_bundle_v1.json` is a deployed response under the old, expert_assignment hash.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from connito.shared import task_sync
from connito.shared.config import CycleCfg
from connito.shared.task_sync import (
    ActiveTask,
    resolve_active_task_name,
    TaskBundle,
    get_active_task,
    get_active_task_bundle,
    materialize_task,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _cycle() -> CycleCfg:
    """The real config section, so the tests exercise the real defaults."""
    return CycleCfg(owner_url="https://cycle-api.example", api_retries=0, api_backoff_sec=0)


def _stub_response(monkeypatch, payload) -> None:
    monkeypatch.setattr(
        task_sync, "get_with_retry", lambda *a, **k: SimpleNamespace(json=lambda: payload)
    )


def _explode(*args, **kwargs):
    raise OSError("disk full")


def _bundle(**overrides) -> TaskBundle:
    data = _fixture("active_task_bundle") | overrides
    return TaskBundle(**data)


# ------------------------------------------------------------------- contract

def test_models_parse_the_deployed_responses():
    light = ActiveTask(**_fixture("active_task"))
    bundle = TaskBundle(**_fixture("active_task_bundle"))
    assert light.name == bundle.name
    assert light.group_id == bundle.group_id
    # The light endpoint's hash is what clients compare to decide to re-fetch.
    assert light.bundle_sha256 == bundle.bundle_sha256
    assert bundle.config["residual"], "the residual settings travel in the bundle"


def test_bundle_hash_matches_the_server():
    bundle = TaskBundle(**_fixture("active_task_bundle"))
    # Recomputed here, from cycle-api's own bundle builder with the C1 change (recapture after C1 deploys).
    assert bundle.canonical_sha256() == bundle.bundle_sha256


def test_a_bundle_hashed_with_expert_assignment_is_refused(monkeypatch):
    """A server still on the old hash: the payload parses, its hash does not match."""
    _stub_response(monkeypatch, _fixture("active_task_bundle_v1"))
    assert get_active_task_bundle(_cycle()) is None


def test_unknown_fields_are_ignored():
    # The server must be able to add a field without a fleet-wide client
    # release; only removing or renaming one is a breaking change.
    payload = _fixture("active_task") | {"some_future_field": {"nested": 1}}
    assert ActiveTask(**payload).name == _fixture("active_task")["name"]


# ---------------------------------------------------------------------- fetch

def test_returns_none_when_the_api_is_unreachable(monkeypatch):
    monkeypatch.setattr(task_sync, "get_with_retry", lambda *a, **k: None)
    # An owner-API outage must never stop a node; the caller keeps its task.
    assert get_active_task(_cycle()) is None
    assert get_active_task_bundle(_cycle()) is None


@pytest.mark.parametrize("payload", [{"name": "x"}, {"unrelated": 1}, []])
def test_malformed_payload_returns_none_without_raising(monkeypatch, payload):
    _stub_response(monkeypatch, payload)
    assert get_active_task(_cycle()) is None


def test_bundle_failing_its_own_hash_is_refused(monkeypatch):
    tampered = _fixture("active_task_bundle")
    tampered["config"]["residual"]["k"] = 2   # hash no longer matches
    _stub_response(monkeypatch, tampered)
    # Refusing beats materializing a payload the two repos disagree about.
    assert get_active_task_bundle(_cycle()) is None


def test_valid_bundle_is_returned(monkeypatch):
    _stub_response(monkeypatch, _fixture("active_task_bundle"))
    bundle = get_active_task_bundle(_cycle())
    assert bundle is not None and bundle.group_id == 9


# --------------------------------------------------------------- materialize

def test_materialize_writes_the_payload(tmp_path: Path):
    bundle = _bundle()
    task_dir = materialize_task(bundle, tmp_path)

    assert task_dir == tmp_path / bundle.name
    assert yaml.safe_load((task_dir / "config.yaml").read_text()) == bundle.config
    # No shard policy served, so the file is absent rather than "null".
    assert sorted(p.name for p in task_dir.iterdir()) == [task_sync.STAMP_FILE, "config.yaml"]


def test_shard_policy_is_written_when_served(tmp_path: Path):
    policy = {"allenai/c4": {"path_prefix": "en/"}}
    bundle = _bundle(shard_policy=policy)
    bundle.bundle_sha256 = bundle.canonical_sha256()
    task_dir = materialize_task(bundle, tmp_path)
    assert json.loads((task_dir / "shard_policy.json").read_text()) == policy


def test_materialize_is_idempotent(tmp_path: Path):
    bundle = _bundle()
    task_dir = materialize_task(bundle, tmp_path)
    (task_dir / "marker").write_text("survives")

    materialize_task(bundle, tmp_path)
    # An unchanged bundle must not churn the directory every cycle.
    assert (task_dir / "marker").read_text() == "survives"


def test_a_changed_bundle_replaces_the_directory(tmp_path: Path):
    materialize_task(_bundle(), tmp_path)
    stale = tmp_path / "exp_pmc_residual" / "expert_assignment.json"
    stale.write_text("from a previous release")

    changed = _bundle(config={"group_id": 9, "residual": {"k": 2}})
    changed.bundle_sha256 = changed.canonical_sha256()
    task_dir = materialize_task(changed, tmp_path)

    assert yaml.safe_load((task_dir / "config.yaml").read_text()) == changed.config
    assert not stale.exists(), "a replaced task must not keep the old files"


def test_a_crash_mid_write_leaves_no_half_written_task(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(task_sync.json, "dumps", _explode)
    with pytest.raises(OSError):
        materialize_task(_bundle(shard_policy={"allenai/c4": {}}), tmp_path)

    # Nothing at the path the loader reads, and no staging left behind.
    assert not (tmp_path / "exp_pmc_residual").exists()
    assert list(tmp_path.iterdir()) == []


def test_a_failed_write_leaves_the_existing_task_intact(tmp_path: Path, monkeypatch):
    materialize_task(_bundle(), tmp_path)
    task_dir = tmp_path / "exp_pmc_residual"

    # config.yaml is written before shard_policy.json, so a non-atomic write
    # would leave the new config in place with no stamp to say so.
    changed = _bundle(config={"group_id": 9}, shard_policy={"allenai/c4": {}})
    changed.bundle_sha256 = changed.canonical_sha256()

    monkeypatch.setattr(task_sync.json, "dumps", _explode)
    with pytest.raises(OSError):
        materialize_task(changed, tmp_path)

    served = _fixture("active_task_bundle")
    assert yaml.safe_load((task_dir / "config.yaml").read_text()) == served["config"]
    assert not (task_dir / "shard_policy.json").exists()
    assert (task_dir / task_sync.STAMP_FILE).read_text() == served["bundle_sha256"]


# ------------------------------------------------------------------ resolver

def test_resolver_returns_the_served_name(tmp_path, monkeypatch):
    _stub_response(monkeypatch, _fixture("active_task"))
    cfg = tmp_path / "config.yaml"
    cfg.write_text("cycle:\n  owner_url: https://cycle-api.example\n")
    assert resolve_active_task_name(cfg) == "exp_pmc_residual"


def test_resolver_returns_none_when_the_api_is_unreachable(tmp_path, monkeypatch):
    monkeypatch.setattr(task_sync, "get_with_retry", lambda *a, **k: None)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("cycle: {}\n")
    # None means "start on whatever the config says" — never halt.
    assert resolve_active_task_name(cfg) is None


def test_resolver_survives_an_unreadable_config(tmp_path, monkeypatch):
    _stub_response(monkeypatch, _fixture("active_task"))
    # No config file at all: fall through to the shipped owner_url rather than
    # raising, since every node uses the same one anyway.
    assert resolve_active_task_name(tmp_path / "missing.yaml") == "exp_pmc_residual"
