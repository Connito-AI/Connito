"""One model, shared in place by every miner eval.

The property that makes it safe is not a lock — it is that every accepted
submission covers exactly the round's base key set, so each load fully
overwrites the one before it.
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

from connito.shared.telemetry import EVAL_STATUS_CODES
from connito.validator.background_eval_worker import BackgroundEvalWorker
from connito.validator.evaluator import (
    _VALIDATION_FAIL_TO_REASON,
    load_model_from_path,
    validate_miner_submission,
)

CPU = torch.device("cpu")
FULL = {"a.weight", "b.weight"}


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


# --- the completeness rule -----------------------------------------------------

def _round_with_valid_commit(uid: int = 1) -> SimpleNamespace:
    ckpt = SimpleNamespace(
        validate=lambda **kw: True, path=None,
        signature_verified=True, hash_verified=True, expert_group_verified=True,
    )
    return SimpleNamespace(uid_to_chain_checkpoint={uid: ckpt})


def test_an_incomplete_submission_is_rejected(tmp_path):
    subset = _shard(tmp_path / "m.safetensors", 0.5, keys=("a.weight",))

    reason = validate_miner_submission(
        round_obj=_round_with_valid_commit(), uid=1, model_path=subset,
        expert_group_assignment={}, expected_keys=FULL,
    )

    assert reason == "incomplete_expert_set"


def test_a_complete_submission_passes(tmp_path):
    full = _shard(tmp_path / "m.safetensors", 0.5)
    assert validate_miner_submission(
        round_obj=_round_with_valid_commit(), uid=1, model_path=full,
        expert_group_assignment={}, expected_keys=FULL,
    ) is None


def test_the_check_is_skipped_without_a_base(tmp_path):
    subset = _shard(tmp_path / "m.safetensors", 0.5, keys=("a.weight",))
    assert validate_miner_submission(
        round_obj=_round_with_valid_commit(), uid=1, model_path=subset,
        expert_group_assignment={}, expected_keys=None,
    ) is None


def test_the_reason_is_visible_to_miners():
    assert _VALIDATION_FAIL_TO_REASON["incomplete_expert_set"] == "incomplete_expert_set"
    assert "incomplete_expert_set" in EVAL_STATUS_CODES.values()


# --- the worker records the base's key set at freeze --------------------------

@pytest.fixture
def stub_eval(monkeypatch):
    monkeypatch.setattr("connito.shared.dataloader.get_dataloader", lambda **kw: [])
    monkeypatch.setattr("connito.shared.dataloader.materialize_batches", lambda *a, **kw: [1])
    monkeypatch.setattr("connito.shared.evaluate.evaluate_model", lambda *a, **kw: {"val_loss": 1.0})


def _worker(model: nn.Module) -> BackgroundEvalWorker:
    w = BackgroundEvalWorker(
        config=SimpleNamespace(
            evaluation=SimpleNamespace(per_miner_eval_timeout_sec=1.0),
            dataloader=SimpleNamespace(world_size=1),
        ),
        round_ref=MagicMock(), device=CPU, tokenizer=MagicMock(),
        merge_phase_active=threading.Event(), eval_window_active=threading.Event(),
        gpu_eval_lock=threading.Lock(), expert_group_assignment={},
    )
    w.set_eval_base_model(model)
    return w


def test_the_worker_expects_the_base_shards_key_set(tmp_path, stub_eval):
    w = _worker(_Two(0.1))
    shard = _shard(tmp_path / "round_9.safetensors", 0.4)

    asyncio.run(w._load_round_base(SimpleNamespace(round_id=9, base_shard=shard, seed="s")))

    assert w._expected_expert_keys == FULL
