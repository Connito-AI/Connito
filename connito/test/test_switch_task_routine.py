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
from unittest.mock import MagicMock, patch

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
    return ValidatorConfig.from_path(cfg_path, active_task=None, auto_update_config=True)


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


def _stub_builder(config, rank, device, expert_manager):
    """Stands in for `_build_eval_model`: a tiny module and a shard path,
    without loading DeepSeek. Tagged with the group so tests can tell
    whose model came back."""
    model = torch.nn.Linear(2, 2)
    model.group_id = config.task.exp.group_id
    shard = Path(config.ckpt.checkpoint_path) / "pretrained" / f"model_expgroup_{model.group_id}.safetensors"
    return model, shard


def _switch(config, eval_worker, gates, to: str, build_model=_stub_builder):
    eval_window, merge = gates
    with patch("connito.validator.run._build_eval_model", build_model):
        return _switch_task(
            config, to,
            rank=0, device=torch.device("cpu"),
            eval_worker=eval_worker,
            eval_window_active=eval_window,
            merge_phase_active=merge,
        )


def test_config_and_routing_table_move_together(config, eval_worker, gates) -> None:
    manager = _switch(config, eval_worker, gates, TARGET).expert_manager

    assert config.task.expert_group_name == TARGET
    assert config.task.exp.group_id == 7
    # The new group's table, plus the helper that rides along with every task.
    assert set(manager.expert_group_assignment) == {7, 2}
    assert manager.expert_group_assignment[7][0] == [(0, 20), (1, 21)]


def test_the_eval_worker_is_handed_the_new_table(config, eval_worker, gates) -> None:
    manager = _switch(config, eval_worker, gates, TARGET).expert_manager

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


def test_the_switch_hands_back_a_fresh_baseline_ref(config, eval_worker, gates) -> None:
    """`run` must rebind to a different dict than a still-uploading publish
    holds: `publish_round_baseline` writes `out` a second time once the ~3 GB
    upload lands, routinely after the switch window, so clearing in place
    would put the previous group's shard back. Fresh means empty and new."""
    first = _switch(config, eval_worker, gates, TARGET).baseline_ref
    second = _switch(config, eval_worker, gates, SHIPPED).baseline_ref

    assert first == {} and second == {}
    assert first is not second


# --- tier 3: the switch rebuilds the model --------------------------------

def test_the_switch_hands_the_worker_the_new_groups_model(config, eval_worker, gates) -> None:
    old = torch.nn.Linear(2, 2)
    eval_worker.set_eval_base_model(old)

    state = _switch(config, eval_worker, gates, TARGET)

    assert state.eval_model.group_id == 7
    assert eval_worker._eval_base_model is state.eval_model
    assert state.base_shard.name == "model_expgroup_7.safetensors"
    assert 7 in eval_worker._expert_group_assignment


def test_the_model_is_built_from_the_new_groups_table(config, eval_worker, gates) -> None:
    """Config moves first, then the manager, then the model — a builder that
    saw the old table would host the old group's experts."""
    seen: list[set[int]] = []

    def spy(cfg, rank, device, expert_manager):
        seen.append(set(expert_manager.expert_group_assignment))
        return _stub_builder(cfg, rank, device, expert_manager)

    _switch(config, eval_worker, gates, TARGET, build_model=spy)

    assert seen == [{7, 2}]  # the new group plus the helper


def test_the_old_model_is_still_serving_while_the_new_one_builds(config, eval_worker, gates) -> None:
    """Built before released: two models resident briefly, so a failed build
    has something to fall back to."""
    old = torch.nn.Linear(2, 2)
    eval_worker.set_eval_base_model(old)
    during: list[object] = []

    def spy(cfg, rank, device, expert_manager):
        during.append(eval_worker._eval_base_model)
        return _stub_builder(cfg, rank, device, expert_manager)

    _switch(config, eval_worker, gates, TARGET, build_model=spy)

    assert during == [old]


def test_a_failed_build_rolls_everything_back(config, eval_worker, gates) -> None:
    old = torch.nn.Linear(2, 2)
    eval_worker.set_eval_base_model(old)

    def boom(cfg, rank, device, expert_manager):
        raise RuntimeError("no such model")

    with pytest.raises(RuntimeError, match="no such model"):
        _switch(config, eval_worker, gates, TARGET, build_model=boom)

    assert config.task.expert_group_name == SHIPPED
    assert config.task.exp.group_id == 4
    assert eval_worker._eval_base_model is old
    assert 4 in eval_worker._expert_group_assignment


# --- tier 3: the trigger ------------------------------------------------------

def _poll(config, eval_worker, gates, monkeypatch, *, active, bundle=None, build_model=_stub_builder,
          round_ref=None):
    """Run `_maybe_switch_task` against a stubbed owner. Returns the state it
    handed back and the (task, root) pairs it asked to materialize."""
    from types import SimpleNamespace

    from connito.validator import run
    from connito.validator.round import RoundRef

    eval_window, merge = gates
    round_ref = round_ref or RoundRef()
    materialized: list[tuple[str, Path]] = []
    monkeypatch.setattr(run, "get_active_task", lambda cycle: active and SimpleNamespace(name=active))
    monkeypatch.setattr(run, "get_active_task_bundle", lambda cycle: bundle and SimpleNamespace(name=bundle))
    monkeypatch.setattr(run, "materialize_task", lambda b, root: materialized.append((b.name, Path(root))))
    monkeypatch.setattr(run, "_build_eval_model", build_model)
    state = run._maybe_switch_task(
        config, rank=0, device=torch.device("cpu"), eval_worker=eval_worker,
        eval_window_active=eval_window, merge_phase_active=merge,
        round_ref=round_ref, gpu_eval_lock=threading.Lock(),
    )
    return state, materialized


def _round_in_flight(gates):
    """A round mid-evaluation: the window is open and the ref holds it."""
    from types import SimpleNamespace

    from connito.validator.round import RoundRef

    gates[0].set()
    return RoundRef(current=SimpleNamespace(round_id=9000))


def test_the_owner_naming_another_task_switches_to_it(config, eval_worker, gates, monkeypatch) -> None:
    state, materialized = _poll(config, eval_worker, gates, monkeypatch, active=TARGET, bundle=TARGET)

    assert state is not None and state.eval_model.group_id == 7
    assert config.task.expert_group_name == TARGET
    assert materialized == [(TARGET, config.task.base_path)]


def test_the_round_in_flight_is_dropped_with_its_task(config, eval_worker, gates, monkeypatch) -> None:
    """Its task is over: no more claims (window closed) and nothing to
    finalize (ref cleared), so no weights go out for it."""
    round_ref = _round_in_flight(gates)

    _poll(config, eval_worker, gates, monkeypatch, active=TARGET, bundle=TARGET, round_ref=round_ref)

    assert round_ref.current is None
    assert not gates[0].is_set()


def test_an_unreachable_owner_keeps_the_current_task(config, eval_worker, gates, monkeypatch) -> None:
    """No answer is not a change of answer: nothing is fetched, let alone switched."""
    state, materialized = _poll(config, eval_worker, gates, monkeypatch, active=None, bundle=TARGET)

    assert state is None
    assert config.task.expert_group_name == SHIPPED
    assert materialized == []


def test_the_owner_naming_our_task_fetches_nothing(config, eval_worker, gates, monkeypatch) -> None:
    """Had the bundle been fetched, its task would have been materialized."""
    state, materialized = _poll(config, eval_worker, gates, monkeypatch, active=SHIPPED, bundle=TARGET)

    assert state is None and materialized == []
    assert config.task.expert_group_name == SHIPPED


def test_a_bad_bundle_keeps_the_current_task(config, eval_worker, gates, monkeypatch) -> None:
    """The bundle fetch refuses a payload that fails its hash by returning None."""
    state, materialized = _poll(config, eval_worker, gates, monkeypatch, active=TARGET, bundle=None)

    assert state is None
    assert config.task.expert_group_name == SHIPPED
    assert materialized == []


def test_a_failed_build_keeps_the_current_task_and_model(config, eval_worker, gates, monkeypatch) -> None:
    old = torch.nn.Linear(2, 2)
    eval_worker.set_eval_base_model(old)

    def boom(cfg, rank, device, expert_manager):
        raise RuntimeError("no such model")

    round_ref = _round_in_flight(gates)

    state, _ = _poll(config, eval_worker, gates, monkeypatch, active=TARGET, bundle=TARGET, build_model=boom,
                     round_ref=round_ref)

    assert state is None
    assert config.task.expert_group_name == SHIPPED
    assert eval_worker._eval_base_model is old
    # The round carries on, on the model it had.
    assert round_ref.current is not None
    assert gates[0].is_set()
