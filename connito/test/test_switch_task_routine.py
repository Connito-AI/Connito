"""`_maybe_switch_task` — how a running validator moves onto a new task.

It no longer switches in place. The eval model is the full 64-expert topology,
and building the new one beside the live one needs more memory than any
validator in the fleet has, so the process ends and the container's restart
policy brings it back on the new task.

Three properties here are silent when broken: the bundle must reach disk
*before* the process ends, the restart signal must not be swallowed by the
`except Exception` that guards the fetch, and an owner that says nothing must
leave the node exactly where it is.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from connito.shared.config import ValidatorConfig
from connito.validator import run
from connito.validator.run import _RestartForTask

SHIPPED = "exp_nemotron_c4"   # the locked default; the config boots on this
TARGET = "exp_switch_target"
HELPER = "exp_helper"          # task.helper_group_id defaults to 2


def _task_dir(root: Path, name: str, group_id: int, org_base: int) -> None:
    body = yaml.safe_load(Path(f"expert_groups/{SHIPPED}/config.yaml").read_text())
    body["group_id"] = group_id
    d = root / "expert_groups" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(yaml.safe_dump(body))
    # {layer: [[my_idx, org_idx], ...]}, distinct org ids per group.
    (d / "expert_assignment.json").write_text(
        json.dumps({"0": [[0, org_base], [1, org_base + 1]]})
    )


@pytest.fixture
def config(tmp_path: Path) -> ValidatorConfig:
    _task_dir(tmp_path, SHIPPED, group_id=4, org_base=10)
    _task_dir(tmp_path, TARGET, group_id=7, org_base=20)
    _task_dir(tmp_path, HELPER, group_id=2, org_base=30)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "run": {"root_path": str(tmp_path)},
        "chain": {"hotkey_ss58": "test-hk", "coldkey_ss58": "test-ck", "uid": 0},
    }))
    return ValidatorConfig.from_path(cfg_path, active_task=None, auto_update_config=True)


@pytest.fixture
def eval_window() -> threading.Event:
    return threading.Event()


def _poll(config, eval_window, monkeypatch, *, active, bundle=None):
    """Run `_maybe_switch_task` against a stubbed owner.

    Returns the tasks it asked to materialize, so a test can tell whether the
    bundle reached disk before the process ended.
    """
    materialized: list[tuple[str, Path]] = []
    monkeypatch.setattr(run, "get_active_task", lambda cycle: active and SimpleNamespace(name=active))
    monkeypatch.setattr(run, "get_active_task_bundle", lambda cycle: bundle and SimpleNamespace(name=bundle))
    monkeypatch.setattr(run, "materialize_task", lambda b, root: materialized.append((b.name, Path(root))))
    run._maybe_switch_task(config, eval_window_active=eval_window)
    return materialized


def test_the_owner_naming_another_task_restarts(config, eval_window, monkeypatch) -> None:
    with pytest.raises(_RestartForTask) as excinfo:
        _poll(config, eval_window, monkeypatch, active=TARGET, bundle=TARGET)

    # The task it is rebooting onto, so the log and any handler can name it.
    assert excinfo.value.args == (TARGET,)


def test_the_bundle_is_on_disk_before_the_process_ends(config, eval_window, monkeypatch) -> None:
    """The reboot resolves the task through `ensure_active_task`. Materializing
    first is what makes that idempotent instead of a second fetch — and what
    stops an owner that goes down in between stranding the node."""
    materialized: list[tuple[str, Path]] = []
    monkeypatch.setattr(run, "get_active_task", lambda cycle: SimpleNamespace(name=TARGET))
    monkeypatch.setattr(run, "get_active_task_bundle", lambda cycle: SimpleNamespace(name=TARGET))
    monkeypatch.setattr(run, "materialize_task", lambda b, root: materialized.append((b.name, Path(root))))

    with pytest.raises(_RestartForTask):
        run._maybe_switch_task(config, eval_window_active=eval_window)

    assert materialized == [(TARGET, config.task.base_path)]


def test_the_eval_window_is_closed_before_restarting(config, eval_window, monkeypatch) -> None:
    """So the worker claims no further miners while the shutdown block joins it."""
    eval_window.set()

    with pytest.raises(_RestartForTask):
        _poll(config, eval_window, monkeypatch, active=TARGET, bundle=TARGET)

    assert not eval_window.is_set()


def test_the_restart_signal_survives_an_except_exception(config, eval_window, monkeypatch) -> None:
    """The load-bearing property of the whole design.

    `_RestartForTask` derives from `KeyboardInterrupt`, so it is a
    `BaseException`. Were it an ordinary `Exception`, the guard around the
    bundle fetch would swallow it, the log would claim the switch merely
    failed, and the validator would sit on the old task forever.
    """
    monkeypatch.setattr(run, "get_active_task", lambda cycle: SimpleNamespace(name=TARGET))
    monkeypatch.setattr(run, "get_active_task_bundle", lambda cycle: SimpleNamespace(name=TARGET))
    monkeypatch.setattr(run, "materialize_task", lambda b, root: None)

    with pytest.raises(_RestartForTask):
        try:
            run._maybe_switch_task(config, eval_window_active=eval_window)
        except Exception:  # noqa: BLE001 - deliberately mirrors the guard
            pytest.fail("the restart signal was caught as an ordinary Exception")


def test_an_unreachable_owner_keeps_the_current_task(config, eval_window, monkeypatch) -> None:
    """No answer is not a change of answer: nothing fetched, nothing restarted."""
    materialized = _poll(config, eval_window, monkeypatch, active=None, bundle=TARGET)

    assert materialized == []
    assert config.task.expert_group_name == SHIPPED


def test_the_owner_naming_our_task_fetches_nothing(config, eval_window, monkeypatch) -> None:
    """Had the bundle been fetched, its task would have been materialized."""
    materialized = _poll(config, eval_window, monkeypatch, active=SHIPPED, bundle=TARGET)

    assert materialized == []
    assert config.task.expert_group_name == SHIPPED


def test_a_bad_bundle_keeps_the_current_task(config, eval_window, monkeypatch) -> None:
    """The fetch returns None for a payload that fails its hash. That is a
    reason to stay put and retry next cycle, not to end the process."""
    materialized = _poll(config, eval_window, monkeypatch, active=TARGET, bundle=None)

    assert materialized == []
    assert config.task.expert_group_name == SHIPPED
