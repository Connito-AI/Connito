"""One model, shared in place by every miner eval.

The property that makes it safe is not a lock — it is that every accepted
submission is exactly the residual spec, so each load fully overwrites the
one before it.
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from connito.shared.config import ResidualCfg
from connito.shared.modeling.residual import residual_shapes
from connito.shared.telemetry import VALIDATOR_MINER_EVAL_FAILURES, VALIDATOR_MINER_EVAL_STATUS
from connito.validator.background_eval_worker import BackgroundEvalWorker
from connito.validator.evaluator import load_model_from_path

CPU = torch.device("cpu")


class _Two(nn.Module):
    def __init__(self, val: float) -> None:
        super().__init__()
        self.a = nn.Linear(4, 2, bias=False)
        self.b = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.a.weight.fill_(val)
            self.b.weight.fill_(val)


def _shard(path: Path, val: float, keys=("a.weight", "b.weight"), shape=(2, 4)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: torch.full(shape, val) for k in keys}, str(path))
    return path


def _weights(m) -> tuple[torch.Tensor, torch.Tensor]:
    return m.a.weight.detach(), m.b.weight.detach()


# --- in place ------------------------------------------------------------------

def test_a_miner_is_loaded_into_the_base_model_itself(tmp_path):
    base = _Two(0.1)
    shard = _shard(tmp_path / "m1.safetensors", 0.5)

    out = load_model_from_path(str(shard), base, CPU)

    assert out is base, "no copy"
    for w in _weights(base):
        torch.testing.assert_close(w, torch.full((2, 4), 0.5))


def test_consecutive_complete_miners_do_not_contaminate(tmp_path):
    base = _Two(0.1)
    load_model_from_path(str(_shard(tmp_path / "m1.safetensors", 0.3)), base, CPU)
    load_model_from_path(str(_shard(tmp_path / "m2.safetensors", 0.6)), base, CPU)

    for w in _weights(base):
        torch.testing.assert_close(w, torch.full((2, 4), 0.6))


def test_a_subset_would_contaminate_which_is_why_completeness_is_enforced(tmp_path):
    """Documents the failure the rule prevents: the load itself cannot tell."""
    base = _Two(0.1)
    load_model_from_path(str(_shard(tmp_path / "m1.safetensors", 0.3)), base, CPU)
    load_model_from_path(str(_shard(tmp_path / "m2.safetensors", 0.6, keys=("a.weight",))), base, CPU)

    a, b = _weights(base)
    torch.testing.assert_close(a, torch.full((2, 4), 0.6))
    torch.testing.assert_close(b, torch.full((2, 4), 0.3))  # miner 1's leftover


def test_a_failed_load_is_overwritten_by_the_next_miner(tmp_path):
    base = _Two(0.1)
    bad = _shard(tmp_path / "bad.safetensors", 0.9, shape=(3, 3))  # shape mismatch raises
    with pytest.raises(RuntimeError):
        load_model_from_path(str(bad), base, CPU)

    load_model_from_path(str(_shard(tmp_path / "m2.safetensors", 0.6)), base, CPU)

    for w in _weights(base):
        torch.testing.assert_close(w, torch.full((2, 4), 0.6))


# --- the worker reads the residual spec from config at each round load ---------

@pytest.fixture
def stub_eval(monkeypatch):
    monkeypatch.setattr("connito.shared.dataloader.get_dataloader", lambda **kw: [])
    monkeypatch.setattr("connito.shared.dataloader.materialize_batches", lambda *a, **kw: [1])
    monkeypatch.setattr("connito.shared.evaluate.evaluate_model", lambda *a, **kw: {"val_loss": 1.0})


def _residual(k: int) -> ResidualCfg:
    return ResidualCfg(k=k, threshold=0.5, temperature=0.5)


def _worker(model: nn.Module, family_name: str) -> BackgroundEvalWorker:
    w = BackgroundEvalWorker(
        config=SimpleNamespace(
            evaluation=SimpleNamespace(per_miner_eval_timeout_sec=1.0),
            dataloader=SimpleNamespace(world_size=1),
            model=SimpleNamespace(family=family_name),
            task=SimpleNamespace(exp=SimpleNamespace(residual=_residual(1))),
        ),
        round_ref=MagicMock(), device=CPU, tokenizer=MagicMock(),
        merge_phase_active=threading.Event(), eval_window_active=threading.Event(),
        gpu_eval_lock=threading.Lock(),
    )
    w.set_eval_base_model(model)
    return w


def test_the_worker_expects_the_tasks_residual_spec_from_each_round_on(family_name, tmp_path, stub_eval):
    w = _worker(_Two(0.1), family_name)
    shard = _shard(tmp_path / "round_9.safetensors", 0.4)

    asyncio.run(w._load_round_base(SimpleNamespace(round_id=9, base_shard=shard, seed="s")))
    assert w._expected == residual_shapes(w.config)

    w.config.task.exp.residual = _residual(2)  # a task switch between rounds
    asyncio.run(w._load_round_base(SimpleNamespace(round_id=10, base_shard=shard, seed="s")))

    assert w._expected == residual_shapes(w.config)
    assert any(".residual_experts.1." in key for key in w._expected)


def test_the_worker_validates_against_that_spec(family_name, monkeypatch, tmp_path):
    w = _worker(_Two(0.1), family_name)
    w._expected = residual_shapes(w.config)
    monkeypatch.setattr(w, "_prune_non_top", lambda round_obj: None)
    seen = {}

    def _capture(*, round_obj, uid, model_path, expected):
        seen["expected"] = expected
        return "bad_residual_shard"  # non-None short-circuits before the GPU eval

    monkeypatch.setattr("connito.validator.evaluator.validate_miner_submission", _capture)
    round_obj = MagicMock(round_id=100)
    round_obj.claim_for_eval.return_value = True
    round_obj.pop_downloaded.return_value = tmp_path / "model.safetensors"

    asyncio.run(w._evaluate_one(round_obj, uid=1, hotkey="hk_abcdef"))

    assert seen["expected"] == residual_shapes(w.config)


def test_a_refused_submission_reaches_the_miner_as_code_14(family_name, monkeypatch, tmp_path):
    w = _worker(_Two(0.1), family_name)
    w._expected = residual_shapes(w.config)
    monkeypatch.setattr(w, "_prune_non_top", lambda round_obj: None)
    monkeypatch.setattr("connito.validator.evaluator.validate_miner_submission", lambda **kw: "bad_residual_shard")
    round_obj = MagicMock(round_id=100)
    round_obj.claim_for_eval.return_value = True
    round_obj.pop_downloaded.return_value = tmp_path / "model.safetensors"
    failures = VALIDATOR_MINER_EVAL_FAILURES.labels(miner_uid="4242", reason="bad_residual_shard")
    before = failures._value.get()

    asyncio.run(w._evaluate_one(round_obj, uid=4242, hotkey="hk_abcdef"))

    assert VALIDATOR_MINER_EVAL_STATUS.labels(miner_uid="4242")._value.get() == 14
    assert failures._value.get() == before + 1
