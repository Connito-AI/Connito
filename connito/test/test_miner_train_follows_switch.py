"""The training loop restarts on the new task when the owner moves on.

`train_worker` has no seams, so this drives the real loop with every
collaborator stubbed at the module: a two-parameter model, a two-batch
dataloader, no chain, no telemetry server. What is real is the loop body and
the hook under test.

Run with `python -m pytest connito/test/test_miner_train_follows_switch.py`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from connito.miner import train
from connito.shared.config import MinerConfig

SHIPPED = "exp_pmc_residual"
TARGET = "exp_switch_target"


def _task_dir(root: Path, name: str, group_id: int) -> None:
    body = yaml.safe_load(Path(f"expert_groups/{SHIPPED}/config.yaml").read_text())
    body["group_id"] = group_id
    d = root / "expert_groups" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(yaml.safe_dump(body))


@pytest.fixture
def config(tmp_path: Path) -> MinerConfig:
    _task_dir(tmp_path, SHIPPED, group_id=4)
    _task_dir(tmp_path, TARGET, group_id=7)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "run": {"root_path": str(tmp_path)},
        "chain": {"hotkey_ss58": "test-hk", "coldkey_ss58": "test-ck", "uid": 0},
    }))
    cfg = MinerConfig.from_path(cfg_path, active_task=None, auto_update_config=True)
    cfg.local_par.gradient_accumulation_steps = 1  # every batch is an inner step
    cfg.log.metric_interval = 10**9              # local evaluation at inner step 0 only
    cfg.ckpt.checkpoint_interval = None          # no checkpoint saves
    return cfg


class _Tiny(torch.nn.Module):
    """A causal LM in shape only: what `chunked_lm_loss` reads, `model(...).last_hidden_state` and `lm_head`."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(4, 2)
        self.lm_head = torch.nn.Linear(2, 4)

    @property
    def device(self) -> torch.device:
        return self.lm_head.weight.device

    def model(self, input_ids, attention_mask=None):
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


def _batches(n: int) -> list[dict]:
    return [{"input_ids": torch.tensor([[0, 1, 2]]), "labels": torch.tensor([[0, 1, 2]])} for _ in range(n)]


@pytest.fixture
def harness(monkeypatch):
    """Stub everything around the loop; record what the hook must drive."""
    seen = SimpleNamespace(setups=[], freed=[], poller_stops=0, logger_closes=0)

    def _setup_training(config, rank, device, tokenizer, subtensor, wallet, current_model_meta):
        seen.setups.append((config.task.expert_group_name, config.task.exp.group_id))
        model = _Tiny()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        return model, opt, SimpleNamespace(step=lambda: None), _batches(2), None

    class _Poller:
        def __init__(self, **kw): pass
        def start(self): pass
        def stop(self): seen.poller_stops += 1

    class _MetricLogger:
        def __init__(self, *a, **k): pass
        def log(self, *a, **k): pass
        def close(self): seen.logger_closes += 1

    real_get_nested_attr = train.get_nested_attr

    def _get_nested_attr(obj, path, default=None):
        if path == "ckpt.enable_peer_resync":
            return False  # keep the checkpoint-reload branch out of the way
        return real_get_nested_attr(obj, path, default)

    monkeypatch.setattr(train, "setup_training", _setup_training)
    monkeypatch.setattr(train, "get_dataloader", lambda *a, **k: [])
    monkeypatch.setattr(train, "setup_chain_worker", lambda config: (None, object(), None))
    monkeypatch.setattr(train, "SystemStatePoller", _Poller)
    monkeypatch.setattr(train, "PhaseManager", lambda config, subtensor: None)
    monkeypatch.setattr(train, "MetricLogger", _MetricLogger)
    monkeypatch.setattr(train, "TelemetryManager", lambda: SimpleNamespace(start_server=lambda **k: None))
    monkeypatch.setattr(train, "get_base_tokenizer", lambda config: None)
    monkeypatch.setattr(train, "get_status", lambda **k: {})
    monkeypatch.setattr(train, "model_health", lambda model, step: {})
    monkeypatch.setattr(train, "get_model_hash", lambda sd, hex=True: "h")
    monkeypatch.setattr(train, "sum_model_gradients", lambda model: 0.0)
    monkeypatch.setattr(train, "get_nested_attr", _get_nested_attr)
    monkeypatch.setattr(train, "free_cuda_models", lambda models, **k: seen.freed.extend(models))
    return seen


def test_a_task_change_restarts_the_loop_on_the_new_task(config, harness, monkeypatch) -> None:
    """First inner step: the owner has moved on. The loop frees what it holds
    and comes back through setup on the switched config; the old dataloader
    is never resumed."""
    polls = []

    def _sync(cfg):
        polls.append(cfg.task.expert_group_name)
        if len(polls) == 1:
            cfg.switch_active_task(TARGET)
            return True
        return False

    monkeypatch.setattr(train, "sync_active_task", _sync)

    train.train_worker(rank=0, world_size=1, config=config)

    assert harness.setups == [(SHIPPED, 4), (TARGET, 7)]
    assert len(harness.freed) == 1 and isinstance(harness.freed[0], _Tiny)
    assert harness.poller_stops == 1 and harness.logger_closes == 1
    assert polls[0] == SHIPPED and all(p == TARGET for p in polls[1:])


def test_no_change_means_one_setup_and_no_restart(config, harness, monkeypatch) -> None:
    monkeypatch.setattr(train, "sync_active_task", lambda cfg: False)

    train.train_worker(rank=0, world_size=1, config=config)

    assert harness.setups == [(SHIPPED, 4)]
    assert harness.freed == [] and harness.poller_stops == 0
