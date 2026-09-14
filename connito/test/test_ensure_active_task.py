"""`ensure_active_task` closes boot's gap: `from_path` cannot fetch a task it
has not got (config stays off the network) and falls back to the shipped
default, so the entrypoints call this right after to fetch, write and switch.

Run with `python -m pytest connito/test/test_ensure_active_task.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from connito.shared import task_sync
from connito.shared.config import MinerConfig

SHIPPED = "exp_nemotron_c4"
TARGET = "exp_owner_named"
HELPER = "exp_helper"


def _task_dir(root: Path, name: str, group_id: int) -> None:
    body = yaml.safe_load(Path(f"expert_groups/{SHIPPED}/config.yaml").read_text())
    body["group_id"] = group_id
    d = root / "expert_groups" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(yaml.safe_dump(body))
    (d / "expert_assignment.json").write_text(json.dumps({"0": [[0, 10], [1, 11]]}))


@pytest.fixture
def config(tmp_path: Path) -> MinerConfig:
    """Booted as `from_path(active_task=TARGET)` would when TARGET is not on
    disk: on the shipped default."""
    _task_dir(tmp_path, SHIPPED, group_id=4)
    _task_dir(tmp_path, HELPER, group_id=2)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "run": {"root_path": str(tmp_path)},
        "chain": {"hotkey_ss58": "test-hk", "coldkey_ss58": "test-ck", "uid": 0},
    }))
    return MinerConfig.from_path(cfg_path, active_task=TARGET, auto_update_config=True)


@pytest.fixture
def owner(monkeypatch, tmp_path):
    """A stubbed owner whose bundle, when materialized, puts TARGET on disk."""
    calls: list[str] = []

    def _materialize(bundle, root):
        calls.append("materialize")
        _task_dir(Path(root).parent, bundle.name, group_id=7)
        return Path(root) / bundle.name

    monkeypatch.setattr(task_sync, "get_active_task_bundle", lambda cycle: SimpleNamespace(name=TARGET))
    monkeypatch.setattr(task_sync, "materialize_task", _materialize)
    return calls


def test_boot_fell_back_so_the_task_is_fetched_and_switched_onto(config, owner) -> None:
    assert config.task.expert_group_name == SHIPPED  # the fallback

    task_sync.ensure_active_task(config, TARGET)

    assert owner == ["materialize"]
    assert config.task.expert_group_name == TARGET
    assert config.task.exp.group_id == 7
    assert config.task.path.name == TARGET


def test_nothing_happens_when_boot_already_landed_on_the_owners_task(config, owner) -> None:
    task_sync.ensure_active_task(config, SHIPPED)

    assert owner == []
    assert config.task.expert_group_name == SHIPPED


def test_nothing_happens_when_the_owner_was_unreachable(config, owner) -> None:
    task_sync.ensure_active_task(config, None)

    assert owner == []


def test_an_unfetchable_bundle_keeps_the_fallback(config, owner, monkeypatch) -> None:
    monkeypatch.setattr(task_sync, "get_active_task_bundle", lambda cycle: None)

    task_sync.ensure_active_task(config, TARGET)

    assert owner == []
    assert config.task.expert_group_name == SHIPPED


def test_a_failed_write_keeps_the_fallback_without_raising(config, monkeypatch) -> None:
    monkeypatch.setattr(task_sync, "get_active_task_bundle", lambda cycle: SimpleNamespace(name=TARGET))

    def boom(bundle, root):
        raise OSError("disk full")

    monkeypatch.setattr(task_sync, "materialize_task", boom)

    task_sync.ensure_active_task(config, TARGET)

    assert config.task.expert_group_name == SHIPPED
    assert config.task.exp.group_id == 4


# --- sync_active_task: the poll the loops call ------------------------------

def test_the_owner_naming_another_task_switches_and_reports_it(config, owner, monkeypatch) -> None:
    monkeypatch.setattr(task_sync, "get_active_task", lambda cycle: SimpleNamespace(name=TARGET))

    assert task_sync.sync_active_task(config) is True
    assert config.task.expert_group_name == TARGET
    assert owner == ["materialize"]


def test_an_unreachable_owner_reports_no_switch(config, owner, monkeypatch) -> None:
    monkeypatch.setattr(task_sync, "get_active_task", lambda cycle: None)

    assert task_sync.sync_active_task(config) is False
    assert config.task.expert_group_name == SHIPPED
    assert owner == []


def test_the_owner_naming_our_task_reports_no_switch(config, owner, monkeypatch) -> None:
    monkeypatch.setattr(task_sync, "get_active_task", lambda cycle: SimpleNamespace(name=SHIPPED))

    assert task_sync.sync_active_task(config) is False
    assert owner == []
