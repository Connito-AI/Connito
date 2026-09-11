"""The round's base is a shard path, not a copy of the model.

The backbone is frozen for the life of the subnet
(`shared/model.freeze_parameters`), so the active group's pretrained shard
plus that backbone is the whole base. This pins that the worker really
rebuilds from the file, whatever it was left holding.
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
def stub_eval(monkeypatch):
    """Past the load: the dataloader and the baseline forward pass."""
    monkeypatch.setattr("connito.shared.dataloader.get_dataloader", lambda **kw: [])
    monkeypatch.setattr("connito.shared.dataloader.materialize_batches", lambda *a, **kw: [1])
    monkeypatch.setattr(
        "connito.shared.evaluate.evaluate_model", lambda *a, **kw: {"val_loss": 1.0}
    )


def _round(base_shard: Path) -> SimpleNamespace:
    return SimpleNamespace(round_id=100, base_shard=base_shard, seed="s")


def test_the_base_comes_from_the_shard_not_the_resident_weights(tmp_path, stub_eval):
    """A worker left holding the previous round's weights must end up on the
    new round's — otherwise every delta is measured against the wrong base."""
    shard = tmp_path / "model_expgroup_1.safetensors"
    save_file({"lin.weight": torch.full((2, 4), 0.7)}, str(shard))
    worker = _worker(_Tiny(val=0.1))

    asyncio.run(worker._load_round_base(_round(shard)))

    torch.testing.assert_close(
        worker._eval_base_model.lin.weight, torch.full((2, 4), 0.7)
    )
