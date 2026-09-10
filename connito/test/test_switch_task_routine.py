"""`_switch_task` — the one place a running validator changes task.

Two properties worth pinning, both silent when broken: the gate must be
checked *before* config moves, and a task whose assignment will not load must
roll config back. See the routine's docstring for why each matters.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import yaml

from connito.shared.config import ValidatorConfig
from connito.validator.background_eval_worker import BackgroundEvalWorker
from connito.validator.run import _switch_task

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
    return ValidatorConfig.from_path(cfg_path, auto_update_config=True)


@pytest.fixture
def gates() -> tuple[threading.Event, threading.Event]:
    return threading.Event(), threading.Event()


@pytest.fixture
def eval_worker(gates) -> BackgroundEvalWorker:
    eval_window, merge = gates
    return BackgroundEvalWorker(
        config=SimpleNamespace(
            evaluation=SimpleNamespace(per_miner_eval_timeout_sec=1.0),
        ),
        round_ref=MagicMock(),
        device=torch.device("cpu"),
        tokenizer=MagicMock(),
        merge_phase_active=merge,
        eval_window_active=eval_window,
        gpu_eval_lock=threading.Lock(),
        expert_group_assignment={4: {0: [(0, 10), (1, 11)]}},
    )


def _switch(config, eval_worker, gates, to: str):
    eval_window, merge = gates
    return _switch_task(
        config, to,
        eval_worker=eval_worker,
        eval_window_active=eval_window,
        merge_phase_active=merge,
    )


def test_config_and_routing_table_move_together(config, eval_worker, gates) -> None:
    manager = _switch(config, eval_worker, gates, TARGET)

    assert config.task.expert_group_name == TARGET
    assert config.task.exp.group_id == 7
    # The new group's table, plus the helper that rides along with every task.
    assert set(manager.expert_group_assignment) == {7, 2}
    assert manager.expert_group_assignment[7][0] == [(0, 20), (1, 21)]


def test_the_eval_worker_is_handed_the_new_table(config, eval_worker, gates) -> None:
    manager = _switch(config, eval_worker, gates, TARGET)

    assert eval_worker._expert_group_assignment is manager.expert_group_assignment


@pytest.mark.parametrize("gate", ["eval_window", "merge"])
def test_switch_is_refused_mid_round(config, eval_worker, gates, gate) -> None:
    eval_window, merge = gates
    (eval_window if gate == "eval_window" else merge).set()

    with pytest.raises(RuntimeError, match="mid-round"):
        _switch(config, eval_worker, gates, TARGET)


def test_a_refused_switch_moves_nothing(config, eval_worker, gates) -> None:
    """The gate must be checked before config is touched, not after."""
    eval_window, _ = gates
    eval_window.set()
    before = (config.task.expert_group_name, config.task.exp.group_id)

    with pytest.raises(RuntimeError):
        _switch(config, eval_worker, gates, TARGET)

    assert (config.task.expert_group_name, config.task.exp.group_id) == before
    assert 4 in eval_worker._expert_group_assignment


def test_an_unloadable_task_rolls_config_back(config, eval_worker, gates, tmp_path) -> None:
    """Config must not be left naming a group whose table failed to load."""
    # A task folder the config can read but ExpertManager cannot: config.yaml
    # present, expert_assignment.json missing.
    broken = tmp_path / "expert_groups" / "exp_broken"
    broken.mkdir(parents=True)
    (broken / "config.yaml").write_text(
        (tmp_path / "expert_groups" / TARGET / "config.yaml").read_text()
    )

    with pytest.raises(Exception):
        _switch(config, eval_worker, gates, "exp_broken")

    assert config.task.expert_group_name == SHIPPED
    assert config.task.exp.group_id == 4
    assert config.task.path.name == SHIPPED
    # And the worker was never handed a half-built table.
    assert 4 in eval_worker._expert_group_assignment
