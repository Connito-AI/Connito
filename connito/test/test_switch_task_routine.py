"""`_switch_task` — the one place a running validator changes task.

Three properties worth pinning, all silent when broken: the gate must be
checked *before* config moves, a task whose assignment will not load must
roll config back, and the model must be the same object afterwards — a
switch is not a rebuild. See the routine's docstring for why each matters.
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


class _Model(torch.nn.Module):
    """One parameter, so a restore from the old shard is observable."""

    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(2))


@pytest.fixture
def model(eval_worker) -> _Model:
    m = _Model()
    eval_worker.set_eval_base_model(m)
    return m


OLD_SHARD = Path("model_expgroup_4.safetensors")
# What the checkpoint read hands back. Empty so the precision restore walks
# nothing; the tests below care that it travels from the read to the writer.
CHECKPOINT: dict = {}
# What the old shard "holds": pretrained values for the model's parameter.
PRETRAINED = {"w": torch.zeros(2, dtype=torch.bfloat16)}


def _stub_writer(state_dict, expert_manager, group_id, dest_dir, save_dtype):
    """Stands in for `_write_pretrained_shard`: the path it would return,
    tagged with the group so tests can tell whose shard came back."""
    return dest_dir / f"model_expgroup_{group_id}.safetensors"


def _switch(config, eval_worker, gates, to: str, *, model, write_shard=_stub_writer,
            read_experts=lambda path, layer_map: {}):
    eval_window, merge = gates
    with (
        patch("connito.validator.run.load_state_dict_from_path", lambda path: dict(PRETRAINED)),
        patch("connito.validator.run.load_pretrained_expert_tensors", read_experts),
        patch("connito.validator.run._write_pretrained_shard", write_shard),
    ):
        return _switch_task(
            config, to,
            eval_model=model, base_shard=OLD_SHARD,
            eval_worker=eval_worker,
            eval_window_active=eval_window,
            merge_phase_active=merge,
        )


def test_config_and_routing_table_move_together(config, eval_worker, gates, model) -> None:
    manager = _switch(config, eval_worker, gates, TARGET, model=model).expert_manager

    assert config.task.expert_group_name == TARGET
    assert config.task.exp.group_id == 7
    # The new group's table, plus the helper that rides along with every task.
    assert set(manager.expert_group_assignment) == {7, 2}
    assert manager.expert_group_assignment[7][0] == [(0, 20), (1, 21)]


def test_the_eval_worker_is_handed_the_new_table(config, eval_worker, gates, model) -> None:
    manager = _switch(config, eval_worker, gates, TARGET, model=model).expert_manager

    assert eval_worker._expert_group_assignment is manager.expert_group_assignment


@pytest.mark.parametrize("gate", ["eval_window", "merge"])
def test_switch_is_refused_mid_round(config, eval_worker, gates, model, gate) -> None:
    eval_window, merge = gates
    (eval_window if gate == "eval_window" else merge).set()

    with pytest.raises(RuntimeError, match="mid-round"):
        _switch(config, eval_worker, gates, TARGET, model=model)


def test_a_refused_switch_moves_nothing(config, eval_worker, gates, model) -> None:
    """The gate must be checked before config — or the model — is touched."""
    eval_window, _ = gates
    eval_window.set()
    before = (config.task.expert_group_name, config.task.exp.group_id)

    with pytest.raises(RuntimeError):
        _switch(config, eval_worker, gates, TARGET, model=model)

    assert (config.task.expert_group_name, config.task.exp.group_id) == before
    assert 4 in eval_worker._expert_group_assignment
    assert torch.equal(model.w, torch.ones(2))


def test_an_unloadable_task_rolls_config_back(config, eval_worker, gates, model, tmp_path) -> None:
    """Config must not be left naming a group whose table failed to load."""
    # A task folder the config can read but ExpertManager cannot: config.yaml
    # present, expert_assignment.json missing.
    broken = tmp_path / "expert_groups" / "exp_broken"
    broken.mkdir(parents=True)
    (broken / "config.yaml").write_text(
        (tmp_path / "expert_groups" / TARGET / "config.yaml").read_text()
    )

    with pytest.raises(Exception):
        _switch(config, eval_worker, gates, "exp_broken", model=model)

    assert config.task.expert_group_name == SHIPPED
    assert config.task.exp.group_id == 4
    assert config.task.path.name == SHIPPED
    # And the worker was never handed a half-built table.
    assert 4 in eval_worker._expert_group_assignment


def test_the_switch_hands_back_a_fresh_baseline_ref(config, eval_worker, gates, model) -> None:
    """`run` must rebind to a different dict than a still-uploading publish
    holds: `publish_round_baseline` writes `out` a second time once the ~3 GB
    upload lands, routinely after the switch window, so clearing in place
    would put the previous group's shard back. Fresh means empty and new."""
    first = _switch(config, eval_worker, gates, TARGET, model=model).baseline_ref
    second = _switch(config, eval_worker, gates, SHIPPED, model=model).baseline_ref

    assert first == {} and second == {}
    assert first is not second


# --- tier 3: the switch keeps the model ---------------------------------------

def test_the_model_is_the_same_object_after_the_switch(config, eval_worker, gates, model) -> None:
    """Under full topology the model declares every expert whatever the
    task, so a switch has nothing to build — and a second build beside the
    live model would not fit on the card."""
    state = _switch(config, eval_worker, gates, TARGET, model=model)

    assert state.eval_model is model
    assert eval_worker._eval_base_model is model
    assert state.base_shard.name == "model_expgroup_7.safetensors"
    assert 7 in eval_worker._expert_group_assignment


def test_the_old_groups_experts_go_back_to_pretrained(config, eval_worker, gates, model) -> None:
    """The last miner's overlay is still in the model; under full routing
    every expert takes part in the forward, so it must be undone."""
    _switch(config, eval_worker, gates, TARGET, model=model)

    assert torch.equal(model.w, torch.zeros(2))


def test_the_new_shard_is_cut_from_the_new_groups_table(config, eval_worker, gates, model) -> None:
    """Config moves first, then the manager, then the shard — the checkpoint
    is read for the new group's experts, in the dtype of the shard it replaces."""
    read: list[dict] = []
    written: list[tuple] = []

    def spy_read(path, layer_map):
        read.append(layer_map)
        return CHECKPOINT

    def spy_write(state_dict, expert_manager, group_id, dest_dir, save_dtype):
        written.append((state_dict, set(expert_manager.expert_group_assignment), group_id, save_dtype))
        return _stub_writer(state_dict, expert_manager, group_id, dest_dir, save_dtype)

    _switch(config, eval_worker, gates, TARGET, model=model, read_experts=spy_read, write_shard=spy_write)

    assert read == [{0: [(0, 20), (1, 21)]}]
    assert written == [(CHECKPOINT, {7, 2}, 7, torch.bfloat16)]
    assert written[0][0] is CHECKPOINT   # the shard is cut from what was read


def test_a_failed_shard_write_rolls_config_back(config, eval_worker, gates, model) -> None:
    def boom(state_dict, expert_manager, group_id, dest_dir, save_dtype):
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError, match="disk full"):
        _switch(config, eval_worker, gates, TARGET, model=model, write_shard=boom)

    assert config.task.expert_group_name == SHIPPED
    assert config.task.exp.group_id == 4
    assert eval_worker._eval_base_model is model
    assert 4 in eval_worker._expert_group_assignment


# --- tier 3: the trigger ------------------------------------------------------

def _poll(config, eval_worker, gates, monkeypatch, *, model, active, bundle=None, write_shard=_stub_writer,
          round_ref=None):
    """Run `_maybe_switch_task` against a stubbed owner. Returns the state it
    handed back and the (task, root) pairs it asked to materialize."""
    from connito.validator import run
    from connito.validator.round import RoundRef

    eval_window, merge = gates
    round_ref = round_ref or RoundRef()
    materialized: list[tuple[str, Path]] = []
    monkeypatch.setattr(run, "get_active_task", lambda cycle: active and SimpleNamespace(name=active))
    monkeypatch.setattr(run, "get_active_task_bundle", lambda cycle: bundle and SimpleNamespace(name=bundle))
    monkeypatch.setattr(run, "materialize_task", lambda b, root: materialized.append((b.name, Path(root))))
    monkeypatch.setattr(run, "load_state_dict_from_path", lambda path: dict(PRETRAINED))
    monkeypatch.setattr(run, "load_pretrained_expert_tensors", lambda path, layer_map: {})
    monkeypatch.setattr(run, "_write_pretrained_shard", write_shard)
    state = run._maybe_switch_task(
        config, eval_model=model, base_shard=OLD_SHARD, eval_worker=eval_worker,
        eval_window_active=eval_window, merge_phase_active=merge,
        round_ref=round_ref, gpu_eval_lock=threading.Lock(),
    )
    return state, materialized


def _round_in_flight(gates):
    """A round mid-evaluation: the window is open and the ref holds it."""
    from connito.validator.round import RoundRef

    gates[0].set()
    return RoundRef(current=SimpleNamespace(round_id=9000))


def test_the_owner_naming_another_task_switches_to_it(config, eval_worker, gates, model, monkeypatch) -> None:
    state, materialized = _poll(config, eval_worker, gates, monkeypatch, model=model, active=TARGET, bundle=TARGET)

    assert state is not None and state.base_shard.name == "model_expgroup_7.safetensors"
    assert config.task.expert_group_name == TARGET
    assert materialized == [(TARGET, config.task.base_path)]


def test_the_round_in_flight_is_dropped_with_its_task(config, eval_worker, gates, model, monkeypatch) -> None:
    """Its task is over: no more claims (window closed) and nothing to
    finalize (ref cleared), so no weights go out for it."""
    round_ref = _round_in_flight(gates)

    _poll(config, eval_worker, gates, monkeypatch, model=model, active=TARGET, bundle=TARGET, round_ref=round_ref)

    assert round_ref.current is None
    assert not gates[0].is_set()


def test_an_unreachable_owner_keeps_the_current_task(config, eval_worker, gates, model, monkeypatch) -> None:
    """No answer is not a change of answer: nothing is fetched, let alone switched."""
    state, materialized = _poll(config, eval_worker, gates, monkeypatch, model=model, active=None, bundle=TARGET)

    assert state is None
    assert config.task.expert_group_name == SHIPPED
    assert materialized == []


def test_the_owner_naming_our_task_fetches_nothing(config, eval_worker, gates, model, monkeypatch) -> None:
    """Had the bundle been fetched, its task would have been materialized."""
    state, materialized = _poll(config, eval_worker, gates, monkeypatch, model=model, active=SHIPPED, bundle=TARGET)

    assert state is None and materialized == []
    assert config.task.expert_group_name == SHIPPED


def test_a_bad_bundle_keeps_the_current_task(config, eval_worker, gates, model, monkeypatch) -> None:
    """The bundle fetch refuses a payload that fails its hash by returning None."""
    state, materialized = _poll(config, eval_worker, gates, monkeypatch, model=model, active=TARGET, bundle=None)

    assert state is None
    assert config.task.expert_group_name == SHIPPED
    assert materialized == []


def test_a_failed_switch_keeps_the_current_task_and_model(config, eval_worker, gates, model, monkeypatch) -> None:
    def boom(state_dict, expert_manager, group_id, dest_dir, save_dtype):
        raise RuntimeError("disk full")

    round_ref = _round_in_flight(gates)

    state, _ = _poll(config, eval_worker, gates, monkeypatch, model=model, active=TARGET, bundle=TARGET,
                     write_shard=boom, round_ref=round_ref)

    assert state is None
    assert config.task.expert_group_name == SHIPPED
    assert eval_worker._eval_base_model is model
    # The round carries on, on the model it had.
    assert round_ref.current is not None
    assert gates[0].is_set()
