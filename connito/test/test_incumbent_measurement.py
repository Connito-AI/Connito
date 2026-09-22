"""Measuring the published baseline against the one it replaced.

Measurement only: the incumbent (the chain-advertised baseline a round's
submissions trained from) is fetched, scored on the round's batches, and
compared with what `select_baseline_uid` publishes. These pin that the
comparison is right and that measuring can neither disturb the round nor
measure the wrong model.
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

from connito.validator.background_eval_worker import BackgroundEvalWorker
from connito.validator.round import baseline_selection_report

# ---------------------------------------------------------------- the report


def test_a_proven_miners_worse_file_is_a_regression_the_ratchet_avoids():
    """The case the measurement exists for: rolling average picks A, whose
    file is worse than the incumbent, while C beat the incumbent."""
    r = baseline_selection_report(
        val_losses={1: 1.95, 2: 1.90, 3: 1.85},
        prior_avg_scores={1: 2.0, 2: 1.5, 4: 1.0},
        incumbent_val_loss=1.92,
    )
    assert r["legacy_uid"] == 1 and r["legacy_regressed"] is True
    assert r["best_uid"] == 3 and r["ratchet_uid"] == 3
    assert r["legacy_is_best"] is False


def test_the_incumbent_is_kept_when_nothing_beats_it_and_wins_ties():
    worse = baseline_selection_report({1: 1.95, 2: 1.93}, {}, incumbent_val_loss=1.92)
    tie = baseline_selection_report({1: 1.92}, {}, incumbent_val_loss=1.92)
    assert worse["ratchet_uid"] is None and tie["ratchet_uid"] is None


def test_without_an_incumbent_the_ratchet_is_the_rounds_best_and_regression_unknown():
    r = baseline_selection_report({1: 1.9, 2: 1.8}, {}, incumbent_val_loss=None)
    assert r["ratchet_uid"] == 2 and r["legacy_regressed"] is None


def test_an_empty_round_reports_nothing_picked():
    r = baseline_selection_report({}, {1: 2.0}, incumbent_val_loss=1.9)
    assert r["legacy_uid"] is None and r["best_uid"] is None and r["ratchet_uid"] is None
    assert r["legacy_regressed"] is None


# ---------------------------------------------------------------- the worker


class _Tiny(nn.Module):
    def __init__(self, val: float) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.lin.weight.fill_(val)


def _worker(model: nn.Module) -> BackgroundEvalWorker:
    worker = BackgroundEvalWorker(
        config=SimpleNamespace(
            evaluation=SimpleNamespace(per_miner_eval_timeout_sec=1.0),
            dataloader=SimpleNamespace(world_size=1),
        ),
        round_ref=MagicMock(),
        device=torch.device("cpu"),
        tokenizer=MagicMock(),
        merge_phase_active=threading.Event(),
        eval_window_active=threading.Event(),
        gpu_eval_lock=threading.Lock(),
        expert_group_assignment={},
    )
    worker.set_eval_base_model(model)
    return worker


@pytest.fixture
def evals(monkeypatch):
    """The loss is the resident weight, so a test can tell which model was scored."""
    calls = []

    def _evaluate(step, model, *a, **kw):
        calls.append(float(model.lin.weight.mean()))
        return {"val_loss": calls[-1]}

    monkeypatch.setattr("connito.shared.dataloader.get_dataloader", lambda **kw: [])
    monkeypatch.setattr("connito.shared.dataloader.materialize_batches", lambda *a, **kw: [1])
    monkeypatch.setattr("connito.shared.evaluate.evaluate_model", _evaluate)
    return calls


def _shard(path: Path, val: float, key: str = "lin.weight") -> Path:
    save_file({key: torch.full((2, 4), val)}, str(path))
    return path


def _round(tmp_path: Path, **kw) -> SimpleNamespace:
    base = _shard(tmp_path / "base.safetensors", 0.5)
    return SimpleNamespace(round_id=100, base_shard=base, seed="s",
                           incumbent_path=None, incumbent_val_loss=None, finalized=False, **kw)


def test_the_incumbent_is_scored_once_on_its_own_weights(tmp_path, evals):
    worker = _worker(_Tiny(0.1))
    rnd = _round(tmp_path)
    asyncio.run(worker._load_round_base(rnd))
    assert rnd.incumbent_val_loss is None  # not downloaded yet: nothing to do
    asyncio.run(worker._maybe_eval_incumbent(rnd))

    rnd.incumbent_path = _shard(tmp_path / "incumbent.safetensors", 0.3)
    asyncio.run(worker._maybe_eval_incumbent(rnd))
    asyncio.run(worker._maybe_eval_incumbent(rnd))

    assert rnd.incumbent_val_loss == pytest.approx(0.3)
    assert evals == [pytest.approx(0.5), pytest.approx(0.3)]  # baseline, then incumbent, once


def test_an_incumbent_with_a_different_key_set_is_not_measured(tmp_path, evals):
    """A partial overlay would score a hybrid of it and the previous model."""
    worker = _worker(_Tiny(0.1))
    rnd = _round(tmp_path)
    asyncio.run(worker._load_round_base(rnd))
    rnd.incumbent_path = _shard(tmp_path / "other.safetensors", 0.3, key="other.weight")
    asyncio.run(worker._maybe_eval_incumbent(rnd))

    assert rnd.incumbent_val_loss is None
    torch.testing.assert_close(worker._eval_base_model.lin.weight, torch.full((2, 4), 0.5))


def test_a_finalized_round_is_not_written(tmp_path, evals):
    worker = _worker(_Tiny(0.1))
    rnd = _round(tmp_path)
    asyncio.run(worker._load_round_base(rnd))
    rnd.finalized = True
    rnd.incumbent_path = _shard(tmp_path / "incumbent.safetensors", 0.3)
    asyncio.run(worker._maybe_eval_incumbent(rnd))
    assert rnd.incumbent_val_loss is None


# ---------------------------------------------------------------- the fetch


def _fetch(monkeypatch, ckpt):
    from connito.validator.incumbent import fetch_round_incumbent

    monkeypatch.setattr("bittensor.Subtensor", lambda **kw: MagicMock())
    monkeypatch.setattr("connito.shared.model.fetch_model_from_chain_validator", lambda **kw: ckpt)
    rnd = SimpleNamespace(round_id=100, incumbent_path=None)
    cfg = SimpleNamespace(task=SimpleNamespace(exp=SimpleNamespace(group_id=5)),
                          chain=SimpleNamespace(network="test"))
    fetch_round_incumbent(rnd, cfg, expert_group_assignment={})
    return rnd


def test_the_fetch_points_the_round_at_the_downloaded_shard(tmp_path, monkeypatch):
    _shard(tmp_path / "model_expgroup_5.safetensors", 0.3)
    ckpt = SimpleNamespace(path=tmp_path, global_ver=1, hf_revision="abc", model_hash="h")
    assert _fetch(monkeypatch, ckpt).incumbent_path == tmp_path / "model_expgroup_5.safetensors"


def test_nothing_advertised_or_a_failing_fetch_leaves_the_round_alone(monkeypatch):
    assert _fetch(monkeypatch, None).incumbent_path is None

    def _boom(**kw):
        raise RuntimeError("chain down")

    monkeypatch.setattr("connito.shared.model.fetch_model_from_chain_validator", _boom)
    from connito.validator.incumbent import fetch_round_incumbent

    rnd = SimpleNamespace(round_id=1, incumbent_path=None)
    cfg = SimpleNamespace(task=SimpleNamespace(exp=SimpleNamespace(group_id=5)),
                          chain=SimpleNamespace(network="test"))
    fetch_round_incumbent(rnd, cfg, expert_group_assignment={})  # must not raise
    assert rnd.incumbent_path is None
