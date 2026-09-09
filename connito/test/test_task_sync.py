"""Client for the owner API's active-task endpoints.

The fixtures under `fixtures/` are verbatim responses captured from the
deployed cycle-api, so `test_bundle_hash_matches_the_server` is a real
cross-repo check: it re-computes the canonical hash with this repo's code and
asserts it equals the one the server actually sent. The server lives in a
separate private repo with its own release cadence, and a boundary is the
moment the two must agree — this is what makes a serialization drift fail here
instead of during a live transition.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from connito.shared import task_sync
from connito.shared.task_sync import (
    ActiveTask,
    TaskBundle,
    get_active_task,
    get_active_task_bundle,
    materialize_task,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _config() -> SimpleNamespace:
    """Only the four fields `_fetch` reads, so no chain or wallet is needed."""
    return SimpleNamespace(
        cycle=SimpleNamespace(
            owner_url="https://cycle-api.example",
            api_timeout_sec=1,
            api_retries=0,
            api_backoff_sec=0,
        )
    )


def _stub_response(monkeypatch, payload) -> None:
    monkeypatch.setattr(
        task_sync, "_get_with_retry", lambda *a, **k: SimpleNamespace(json=lambda: payload)
    )


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
    assert bundle.expert_assignment, "routing table must not be empty"


def test_bundle_hash_matches_the_server():
    bundle = TaskBundle(**_fixture("active_task_bundle"))
    # Recomputed here, from a response the deployed server really sent.
    assert bundle.canonical_sha256() == bundle.bundle_sha256


def test_unknown_fields_are_ignored():
    # The server must be able to add a field without a fleet-wide client
    # release; only removing or renaming one is a breaking change.
    payload = _fixture("active_task") | {"some_future_field": {"nested": 1}}
    assert ActiveTask(**payload).name == _fixture("active_task")["name"]


# ---------------------------------------------------------------------- fetch

def test_returns_none_when_the_api_is_unreachable(monkeypatch):
    monkeypatch.setattr(task_sync, "_get_with_retry", lambda *a, **k: None)
    # An owner-API outage must never stop a node; the caller keeps its task.
    assert get_active_task(_config()) is None
    assert get_active_task_bundle(_config()) is None


@pytest.mark.parametrize("payload", [{"name": "x"}, {"unrelated": 1}, []])
def test_malformed_payload_returns_none_without_raising(monkeypatch, payload):
    _stub_response(monkeypatch, payload)
    assert get_active_task(_config()) is None


def test_bundle_failing_its_own_hash_is_refused(monkeypatch):
    tampered = _fixture("active_task_bundle")
    tampered["expert_assignment"] = {"1": [[0, 999]]}   # hash no longer matches
    _stub_response(monkeypatch, tampered)
    # Refusing beats materializing a payload the two repos disagree about.
    assert get_active_task_bundle(_config()) is None


def test_valid_bundle_is_returned(monkeypatch):
    _stub_response(monkeypatch, _fixture("active_task_bundle"))
    bundle = get_active_task_bundle(_config())
    assert bundle is not None and bundle.group_id == 4


# --------------------------------------------------------------- materialize

def test_materialize_writes_the_payload(tmp_path: Path):
    bundle = _bundle()
    task_dir = materialize_task(bundle, tmp_path)

    assert task_dir == tmp_path / bundle.name
    assert yaml.safe_load((task_dir / "config.yaml").read_text()) == bundle.config
    assert json.loads((task_dir / "expert_assignment.json").read_text()) == bundle.expert_assignment
    # The deployed task has no shard policy; the file must then be absent
    # rather than written as "null".
    assert not (task_dir / "shard_policy.json").exists()


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
    stale = tmp_path / "exp_nemotron_c4" / "stale.txt"
    stale.write_text("from the previous bundle")

    changed = _bundle(expert_assignment={"1": [[0, 5]]})
    changed.bundle_sha256 = changed.canonical_sha256()
    task_dir = materialize_task(changed, tmp_path)

    assert json.loads((task_dir / "expert_assignment.json").read_text()) == {"1": [[0, 5]]}
    assert not stale.exists(), "a replaced task must not keep the old files"


def test_a_crash_mid_write_leaves_no_half_written_task(tmp_path: Path, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(task_sync.json, "dumps", boom)
    with pytest.raises(OSError):
        materialize_task(_bundle(), tmp_path)

    # Nothing at the path the loader reads, and no staging left behind.
    assert not (tmp_path / "exp_nemotron_c4").exists()
    assert list(tmp_path.iterdir()) == []


def test_a_failed_write_leaves_the_existing_task_intact(tmp_path: Path, monkeypatch):
    materialize_task(_bundle(), tmp_path)
    changed = _bundle(expert_assignment={"1": [[0, 5]]})
    changed.bundle_sha256 = changed.canonical_sha256()

    monkeypatch.setattr(task_sync.json, "dumps", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    with pytest.raises(OSError):
        materialize_task(changed, tmp_path)

    # The task that was already there is still intact and still loadable.
    assert json.loads(
        (tmp_path / "exp_nemotron_c4" / "expert_assignment.json").read_text()
    ) == _fixture("active_task_bundle")["expert_assignment"]
