"""`_switch_task` — the one place a running validator changes task.

Properties worth pinning, all silent when broken: the gate must be checked
*before* config moves; the model must be the same object afterwards (a switch
is not a rebuild); every switch reinstalls the residual from the new task's
K/A/τ and writes that fresh install as the base shard; a failure leaves config
and the model's residual on the old task; and the GPU lock covers only the
reinstall and the write. Tiny real models, CPU only.
"""
from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import yaml
from safetensors.torch import load_file

from connito.shared import checkpoint_helper, checkpoints
from connito.shared import model as shared_model
from connito.shared.config import ValidatorConfig
from connito.shared.modeling.residual import residual_shapes
from connito.validator import run
from connito.validator.background_eval_worker import BackgroundEvalWorker
from connito.validator.round import RoundRef

SHIPPED = "exp_pmc_residual"   # the locked default; the config boots on this
TARGET = "exp_switch_target"   # K = 2
SAME = "exp_same_residual"     # the shipped K/A/τ, another group
GATE = "exp_other_gate"        # the shipped K, another A and τ
TOKENS = torch.randint(0, 128, (2, 12), generator=torch.Generator().manual_seed(1))


def _task_dir(root: Path, name: str, group_id: int, **residual) -> None:
    body = yaml.safe_load(Path(f"expert_groups/{SHIPPED}/config.yaml").read_text())
    body["group_id"] = group_id
    # An open-ish gate everywhere, so a residual visibly moves the logits.
    body["residual"] = {"k": 1, "threshold": 0.5, "temperature": 0.5, **residual}
    d = root / "expert_groups" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(yaml.safe_dump(body))


@pytest.fixture
def config(tmp_path: Path, family_name: str) -> ValidatorConfig:
    _task_dir(tmp_path, SHIPPED, group_id=4)
    _task_dir(tmp_path, TARGET, group_id=7, k=2)
    _task_dir(tmp_path, SAME, group_id=5)
    _task_dir(tmp_path, GATE, group_id=6, threshold=0.3, temperature=0.2)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "run": {"root_path": str(tmp_path)},
        "chain": {"hotkey_ss58": "test-hk", "coldkey_ss58": "test-ck", "uid": 0},
    }))
    cfg = ValidatorConfig.from_path(cfg_path, active_task=None, auto_update_config=True)
    cfg.model.family = family_name
    return cfg


@pytest.fixture
def gates() -> tuple[threading.Event, threading.Event]:
    return threading.Event(), threading.Event()


@pytest.fixture
def lock() -> threading.Lock:
    return threading.Lock()


@pytest.fixture
def worker(gates, lock) -> BackgroundEvalWorker:
    eval_window, merge = gates
    return BackgroundEvalWorker(
        config=SimpleNamespace(evaluation=SimpleNamespace(per_miner_eval_timeout_sec=1.0)),
        round_ref=MagicMock(), device=torch.device("cpu"), tokenizer=MagicMock(),
        merge_phase_active=merge, eval_window_active=eval_window, gpu_eval_lock=lock,
    )


@pytest.fixture
def model(config, worker) -> torch.nn.Module:
    """The validator's model at boot, already handed to its eval worker."""
    m, _ = shared_model.get_model_from_checkpoint(
        config=config, checkpoint_device=torch.device("cpu"), load_global_checkpoint=False,
    )
    worker.set_eval_base_model(m)
    return m


@pytest.fixture(autouse=True)
def never_verified(monkeypatch):
    """The base shard is the validator's own `down_proj` = 0 install; the untrained rule would refuse it."""
    def refuse(*a, **k):
        raise AssertionError("the validator's base shard went through verify_residual_shard")

    for module in (checkpoint_helper, checkpoints, shared_model):
        monkeypatch.setattr(module, "verify_residual_shard", refuse)


def _switch(config, gates, lock, to: str, *, model) -> run.TaskScopedState:
    eval_window, merge = gates
    return run._switch_task(
        config, to, eval_model=model,
        eval_window_active=eval_window, merge_phase_active=merge, gpu_eval_lock=lock,
    )


def _residual(model) -> dict[str, torch.Tensor]:
    return {n: p for n, p in model.named_parameters() if ".residual_" in n}


def _spec(model) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    return {n: (tuple(p.shape), p.dtype) for n, p in _residual(model).items()}


def _routers(model) -> list[torch.nn.Module]:
    return [layer.mlp.residual_router for layer in model.model.layers if hasattr(layer.mlp, "residual_router")]


def _logits(model) -> torch.Tensor:
    with torch.no_grad():
        return model.eval()(input_ids=TOKENS).logits


def _trained(model) -> dict[str, torch.Tensor]:
    """A miner's residual for the model's current K: every tensor nonzero."""
    gen = torch.Generator().manual_seed(2)
    return {n: (torch.randn(p.shape, generator=gen) * 0.1).to(p.dtype) for n, p in _residual(model).items()}


def _same(a: list, b: list) -> bool:
    return len(a) == len(b) and all(x is y for x, y in zip(a, b, strict=True))


# --- boot -------------------------------------------------------------------------


def test_the_validator_boots_without_an_expert_assignment(config) -> None:
    """Task folders hold only config.yaml since the bundle dropped `expert_assignment.json`."""
    assert sorted(p.name for p in config.task.path.iterdir()) == ["config.yaml"]

    state = run.setup_training(config, torch.device("cpu"))

    assert state.base_shard == Path(config.ckpt.checkpoint_path) / "pretrained" / "model_expgroup_4.safetensors"
    assert state.base_shard.is_file()


# --- the gate and the config ----------------------------------------------------


def test_config_and_base_shard_move_to_the_new_task(config, gates, lock, model) -> None:
    state = _switch(config, gates, lock, TARGET, model=model)

    assert (config.task.expert_group_name, config.task.exp.group_id) == (TARGET, 7)
    assert state.base_shard == Path(config.ckpt.checkpoint_path) / "pretrained" / "model_expgroup_7.safetensors"
    assert state.base_shard.is_file()


@pytest.mark.parametrize("gate", ["eval_window", "merge"])
def test_switch_is_refused_mid_round(config, gates, lock, model, gate) -> None:
    eval_window, merge = gates
    (eval_window if gate == "eval_window" else merge).set()

    with pytest.raises(RuntimeError, match="mid-round"):
        _switch(config, gates, lock, TARGET, model=model)


def test_a_refused_switch_moves_nothing(config, gates, lock, model) -> None:
    """The gate must be checked before config — or the model — is touched."""
    gates[0].set()
    before = (config.task.expert_group_name, config.task.exp.group_id)
    routers = _routers(model)

    with pytest.raises(RuntimeError):
        _switch(config, gates, lock, TARGET, model=model)

    assert (config.task.expert_group_name, config.task.exp.group_id) == before
    assert _same(_routers(model), routers)


def test_an_unloadable_task_rolls_config_back_and_keeps_the_residual(config, gates, lock, model, tmp_path) -> None:
    """A task config.yaml without a group_id is refused before the model is touched."""
    _task_dir(tmp_path, "exp_broken", group_id=8)
    path = tmp_path / "expert_groups" / "exp_broken" / "config.yaml"
    path.write_text(yaml.safe_dump({k: v for k, v in yaml.safe_load(path.read_text()).items() if k != "group_id"}))
    routers = _routers(model)

    with pytest.raises(ValueError, match="group_id"):
        _switch(config, gates, lock, "exp_broken", model=model)

    assert (config.task.expert_group_name, config.task.exp.group_id) == (SHIPPED, 4)
    assert config.task.path.name == SHIPPED
    assert _same(_routers(model), routers)


def test_the_switch_hands_back_a_fresh_baseline_ref(config, gates, lock, model) -> None:
    """`run` must rebind to a different dict than a still-uploading publish
    holds: `publish_round_baseline` writes `out` a second time once the ~3 GB
    upload lands, routinely after the switch window, so clearing in place
    would put the previous group's shard back. Fresh means empty and new."""
    first = _switch(config, gates, lock, TARGET, model=model).baseline_ref
    second = _switch(config, gates, lock, SHIPPED, model=model).baseline_ref

    assert first == {} and second == {}
    assert first is not second


# --- the residual -----------------------------------------------------------------


def test_the_model_is_the_same_object_after_the_switch(config, gates, lock, model, worker) -> None:
    """A second build beside the live model would not fit on the card."""
    state = _switch(config, gates, lock, TARGET, model=model)

    assert state.eval_model is model
    assert worker._eval_base_model is model


def test_a_different_k_installs_the_new_shapes_and_writes_them_as_the_base(config, gates, lock, model) -> None:
    state = _switch(config, gates, lock, TARGET, model=model)

    spec = residual_shapes(config)
    assert _spec(model) == spec
    assert all(router.weight.shape[0] == 2 for router in _routers(model))
    shard = load_file(str(state.base_shard))
    assert {k: (tuple(t.shape), t.dtype) for k, t in shard.items()} == spec
    assert all(not t.any() for k, t in shard.items() if "down_proj" in k)


def test_the_same_residual_settings_still_give_a_fresh_residual(config, gates, lock, model) -> None:
    """The last miner's residual must not survive into the new task's base."""
    base = _logits(model)
    routers = _routers(model)
    model.load_state_dict(_trained(model), strict=False)
    assert not torch.equal(_logits(model), base)

    state = _switch(config, gates, lock, SAME, model=model)

    assert not any(a is b for a, b in zip(_routers(model), routers, strict=True))
    assert all(not p.any() for n, p in _residual(model).items() if "down_proj" in n)
    assert torch.equal(_logits(model), base)
    shard = load_file(str(state.base_shard))
    assert all(not t.any() for k, t in shard.items() if "down_proj" in k)


def test_a_threshold_and_temperature_change_takes_effect(config, gates, lock, model) -> None:
    """The hook binds A and τ at install, so the switch reinstalls even when K is unchanged."""
    trained = _trained(model)
    model.load_state_dict(trained, strict=False)
    before = _logits(model)

    _switch(config, gates, lock, GATE, model=model)
    model.load_state_dict(trained, strict=False)

    fresh, _ = shared_model.get_model_from_checkpoint(
        config=config, checkpoint_device=torch.device("cpu"), load_global_checkpoint=False,
    )
    fresh.load_state_dict(trained, strict=False)
    assert not torch.equal(_logits(model), before)
    assert torch.equal(_logits(model), _logits(fresh))


def test_a_failed_shard_write_puts_config_and_residual_back(config, gates, lock, model, worker, monkeypatch) -> None:
    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(run, "save_residual_shard", boom)
    spec = residual_shapes(config)

    with pytest.raises(RuntimeError, match="disk full"):
        _switch(config, gates, lock, TARGET, model=model)

    assert (config.task.expert_group_name, config.task.exp.group_id) == (SHIPPED, 4)
    assert _spec(model) == spec
    assert worker._eval_base_model is model
    assert not lock.locked()


# --- the lock -----------------------------------------------------------------------


def test_with_the_lock_held_the_switch_cannot_touch_the_model(config, gates, lock, model) -> None:
    """A miner eval holding the lock finishes on the residual it started with."""
    routers = _routers(model)
    lock.acquire()
    switch = threading.Thread(target=_switch, args=(config, gates, lock, TARGET), kwargs={"model": model})
    switch.start()
    switch.join(timeout=1.0)
    try:
        assert switch.is_alive()
        assert _same(_routers(model), routers)
    finally:
        lock.release()
    switch.join(timeout=30)

    assert not switch.is_alive()
    assert all(router.weight.shape[0] == 2 for router in _routers(model))


def test_only_the_reinstall_and_the_write_hold_the_lock(config, gates, lock, model, monkeypatch) -> None:
    """The worker's rule: the lock covers what touches the model, never the bundle, the data or the config."""
    held: dict[str, bool] = {}

    def spy(name, fn):
        def wrapped(*a, **k):
            held[name] = lock.locked()
            return fn(*a, **k)
        return wrapped

    monkeypatch.setattr(run, "get_active_task", lambda cycle: SimpleNamespace(name=TARGET))
    monkeypatch.setattr(run, "get_active_task_bundle", spy("bundle", lambda cycle: SimpleNamespace(name=TARGET)))
    monkeypatch.setattr(run, "materialize_task", spy("materialize", lambda bundle, root: None))
    monkeypatch.setattr(ValidatorConfig, "switch_active_task", spy("config", ValidatorConfig.switch_active_task))
    monkeypatch.setattr(run, "install_residual_experts", spy("install", run.install_residual_experts))
    monkeypatch.setattr(run, "save_residual_shard", spy("write", run.save_residual_shard))

    assert _poll(config, gates, lock, model=model) is not None
    assert held == {"bundle": False, "materialize": False, "config": False, "install": True, "write": True}


class _RecordingLock:
    """A real lock that logs each acquire and release, so a gap in a critical section shows."""

    def __init__(self, events: list[str]) -> None:
        self._lock, self._events = threading.Lock(), events

    def __enter__(self):
        self._lock.acquire()
        self._events.append("acquire")
        return self

    def __exit__(self, *exc) -> None:
        self._events.append("release")
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()


def test_a_failed_write_rolls_back_before_a_waiting_eval_can_run(config, gates, model, monkeypatch) -> None:
    """One critical section from install to rollback: no eval scores a miner on a half-switched residual."""
    events: list[str] = []
    lock = _RecordingLock(events)
    install = run.install_residual_experts

    def eval_waiting():
        with lock:
            events.append("eval")

    waiting = threading.Thread(target=eval_waiting)

    def spy_install(model_, config_):
        events.append(f"install k={config_.task.exp.residual.k}")
        install(model_, config_)

    def boom(*a, **k):
        events.append("write")
        waiting.start()
        raise RuntimeError("disk full")

    monkeypatch.setattr(run, "install_residual_experts", spy_install)
    monkeypatch.setattr(run, "save_residual_shard", boom)

    with pytest.raises(RuntimeError, match="disk full"):
        _switch(config, gates, lock, TARGET, model=model)
    waiting.join(timeout=30)

    assert events == ["acquire", "install k=2", "write", "install k=1", "release", "acquire", "eval", "release"]


def test_the_rollback_reads_no_file(config, gates, lock, model, monkeypatch) -> None:
    """Re-reading the old task's config.yaml could fail and strand config on the new task."""
    spec = residual_shapes(config)

    def boom(*a, **k):
        (config.task.base_path / SHIPPED / "config.yaml").unlink()
        raise RuntimeError("disk full")

    monkeypatch.setattr(run, "save_residual_shard", boom)

    with pytest.raises(RuntimeError, match="disk full"):
        _switch(config, gates, lock, TARGET, model=model)

    assert (config.task.expert_group_name, config.task.exp.group_id) == (SHIPPED, 4)
    assert config.task.path.name == SHIPPED and Path(config.ckpt.checkpoint_path).name == SHIPPED
    assert _spec(model) == spec


# --- the trigger ----------------------------------------------------------------------


def _poll(config, gates, lock, *, model, round_ref=None):
    eval_window, merge = gates
    return run._maybe_switch_task(
        config, eval_model=model,
        eval_window_active=eval_window, merge_phase_active=merge,
        round_ref=round_ref or RoundRef(), gpu_eval_lock=lock,
    )


@pytest.fixture
def owner(monkeypatch):
    """Stub the owner API; returns the (task, root) pairs it was asked to materialize."""
    materialized: list[tuple[str, Path]] = []

    def answer(*, active, bundle):
        monkeypatch.setattr(run, "get_active_task", lambda cycle: active and SimpleNamespace(name=active))
        monkeypatch.setattr(run, "get_active_task_bundle", lambda cycle: bundle and SimpleNamespace(name=bundle))
        monkeypatch.setattr(run, "materialize_task", lambda b, root: materialized.append((b.name, Path(root))))
        return materialized

    return answer


def _round_in_flight(gates) -> RoundRef:
    """A round mid-evaluation: the window is open and the ref holds it."""
    gates[0].set()
    return RoundRef(current=SimpleNamespace(round_id=9000))


def test_the_owner_naming_another_task_switches_to_it(config, gates, lock, model, owner) -> None:
    materialized = owner(active=TARGET, bundle=TARGET)

    state = _poll(config, gates, lock, model=model)

    assert state is not None and state.base_shard.name == "model_expgroup_7.safetensors"
    assert config.task.expert_group_name == TARGET
    assert materialized == [(TARGET, config.task.base_path)]


def test_the_round_in_flight_is_dropped_with_its_task(config, gates, lock, model, owner) -> None:
    """Its task is over: no more claims (window closed) and nothing to
    finalize (ref cleared), so no weights go out for it."""
    owner(active=TARGET, bundle=TARGET)
    round_ref = _round_in_flight(gates)

    _poll(config, gates, lock, model=model, round_ref=round_ref)

    assert round_ref.current is None
    assert not gates[0].is_set()


def test_an_unreachable_owner_keeps_the_current_task(config, gates, lock, model, owner) -> None:
    """No answer is not a change of answer: nothing is fetched, let alone switched."""
    materialized = owner(active=None, bundle=TARGET)

    assert _poll(config, gates, lock, model=model) is None
    assert config.task.expert_group_name == SHIPPED
    assert materialized == []


def test_the_owner_naming_our_task_fetches_nothing(config, gates, lock, model, owner) -> None:
    """Had the bundle been fetched, its task would have been materialized."""
    materialized = owner(active=SHIPPED, bundle=TARGET)

    assert _poll(config, gates, lock, model=model) is None
    assert materialized == []
    assert config.task.expert_group_name == SHIPPED


def test_a_bad_bundle_keeps_the_current_task(config, gates, lock, model, owner) -> None:
    """The bundle fetch refuses a payload that fails its hash by returning None."""
    materialized = owner(active=TARGET, bundle=None)

    assert _poll(config, gates, lock, model=model) is None
    assert config.task.expert_group_name == SHIPPED
    assert materialized == []


def test_a_failed_switch_keeps_the_current_task_and_model(config, gates, lock, model, worker, owner,
                                                          monkeypatch) -> None:
    def boom(*a, **k):
        raise RuntimeError("disk full")

    owner(active=TARGET, bundle=TARGET)
    monkeypatch.setattr(run, "save_residual_shard", boom)
    round_ref = _round_in_flight(gates)
    spec = residual_shapes(config)

    assert _poll(config, gates, lock, model=model, round_ref=round_ref) is None

    assert config.task.expert_group_name == SHIPPED
    assert worker._eval_base_model is model
    assert _spec(model) == spec
    # The round carries on, on the model it had.
    assert round_ref.current is not None
    assert gates[0].is_set()
