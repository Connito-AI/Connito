"""The round's base is a shard path, not a copy of the model.

The base model never trains, so the validator's own fresh residual (`down_proj`
= 0, so base + residual = base) is the whole round base. These pin that boot
writes it, that the worker really rebuilds from the file whatever the last
miner left, and that the file is never put through the untrained-expert rule.
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
from safetensors.torch import load_file, save_file

from connito.shared import checkpoint_helper, checkpoints
from connito.shared import model as shared_model
from connito.shared.checkpoint_helper import save_residual_shard, verify_residual_shard
from connito.shared.config import ResidualCfg
from connito.shared.modeling.residual import residual_shapes
from connito.validator import run
from connito.validator.background_eval_worker import BackgroundEvalWorker
from connito.validator.evaluator import load_model_from_path

TOKENS = torch.randint(0, 128, (2, 12), generator=torch.Generator().manual_seed(1))


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
            task=SimpleNamespace(exp=SimpleNamespace(data=SimpleNamespace(dataset_sources=[], sequence_length=8))),
        ),
        round_ref=MagicMock(),
        device=torch.device("cpu"),
        tokenizer=MagicMock(),
        merge_phase_active=threading.Event(),
        eval_window_active=threading.Event(),
        gpu_eval_lock=threading.Lock(),
    )
    worker.set_eval_base_model(model)
    return worker


@pytest.fixture
def stub_eval(monkeypatch):
    """Past the load: the residual spec, the dataloader and the baseline forward pass."""
    monkeypatch.setattr("connito.validator.background_eval_worker.residual_shapes", lambda config: {})
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


def _shard_and_round(tmp_path, shape) -> SimpleNamespace:
    shard = tmp_path / "model_expgroup_1.safetensors"
    if shape is not None:
        save_file({"lin.weight": torch.zeros(shape)}, str(shard))
    return _round(shard)


@pytest.mark.parametrize("shape", [None, (3, 4)], ids=["missing", "other_shape"])
def test_a_base_shard_that_will_not_load_leaves_the_worker_on_the_fallback_baseline(tmp_path, stub_eval, shape):
    """A shard for another K raises inside `load_state_dict`; it must not end the eval thread."""
    worker = _worker(_Tiny(val=0.1))

    asyncio.run(worker._load_round_base(_shard_and_round(tmp_path, shape)))

    assert (worker._loaded_baseline_loss, worker._loaded_round_id) == (100.0, 100)
    # Submissions are still checked against this round's spec (`stub_eval` serves {}), not a stale one.
    assert worker._expected == {}


def test_no_spec_this_round_refuses_no_miner(tmp_path, stub_eval, monkeypatch):
    """A validator-side fault deriving the spec must not mark, zero or penalize any miner."""
    def unreachable(config):
        raise OSError("config.json unreachable")

    validated, recorded = [], []
    monkeypatch.setattr("connito.validator.background_eval_worker.residual_shapes", unreachable)
    monkeypatch.setattr("connito.validator.evaluator.validate_miner_submission",
                        lambda **kw: validated.append(kw) or "bad_residual_shard")
    monkeypatch.setattr("connito.validator.evaluator._record_eval_failure", lambda uid, reason: recorded.append(reason))
    round_obj = MagicMock(round_id=100, base_shard=tmp_path / "base.safetensors", seed="s")
    round_obj.claim_for_eval.return_value = True
    round_obj.pop_downloaded.return_value = tmp_path / "miner.safetensors"
    worker = _worker(_Tiny(val=0.1))
    worker._expected = {"stale": "spec of the previous K"}

    asyncio.run(worker._load_round_base(round_obj))
    asyncio.run(worker._evaluate_one(round_obj, uid=3, hotkey="hk"))

    round_obj.mark_failed.assert_called_once_with(3)
    round_obj.mark_validation_failed.assert_not_called()
    round_obj.mark_scored.assert_not_called()
    assert validated == [] and recorded == []


# --- the validator's own residual as the round base -------------------------------


def _validator_config(name: str, root: Path) -> SimpleNamespace:
    """The validator's fp8 scope; an open-ish gate so a miner's residual visibly moves the logits."""
    return SimpleNamespace(
        role="validator",
        model=SimpleNamespace(family=name, device="cpu", quantization="fp8", quantization_validator="experts"),
        task=SimpleNamespace(exp=SimpleNamespace(
            group_id=9, residual=ResidualCfg(k=1, threshold=0.5, temperature=0.5),
        )),
        # A folder that does not exist yet: boot must create it.
        ckpt=SimpleNamespace(resume_from_ckpt=False, use_pretrained_only=False, checkpoint_path=root / "ckpt"),
    )


def _logits(model) -> torch.Tensor:
    with torch.no_grad():
        return model.eval()(input_ids=TOKENS).logits


@pytest.fixture
def never_verified(monkeypatch):
    """The untrained-expert rule refuses the base shard by design, so no path may route it there."""
    def refuse(*a, **k):
        raise AssertionError("the validator's base shard went through verify_residual_shard")

    for module in (checkpoint_helper, checkpoints, shared_model):
        monkeypatch.setattr(module, "verify_residual_shard", refuse)


def test_boot_writes_its_fresh_residual_as_the_base_shard(family_name, tmp_path, never_verified):
    config = _validator_config(family_name, tmp_path)

    state = run.setup_training(config, torch.device("cpu"))

    assert state.base_shard == tmp_path / "ckpt" / "pretrained" / "model_expgroup_9.safetensors"
    spec = residual_shapes(config)
    shard = load_file(str(state.base_shard))
    assert {k: (tuple(t.shape), t.dtype) for k, t in shard.items()} == spec
    assert all(not t.any() for k, t in shard.items() if "down_proj" in k)
    # Why it is loaded directly: the rule a submission must pass refuses it.
    assert verify_residual_shard(state.base_shard, spec) is None


def test_the_next_rounds_base_undoes_the_last_miner(family_name, tmp_path, stub_eval, never_verified):
    state = run.setup_training(_validator_config(family_name, tmp_path), torch.device("cpu"))
    model = state.eval_model
    base = _logits(model)
    miner = {n: torch.randn_like(p) * 0.1 for n, p in model.named_parameters() if ".residual_" in n}
    save_file(miner, str(tmp_path / "miner.safetensors"))
    load_model_from_path(str(tmp_path / "miner.safetensors"), model, torch.device("cpu"))
    assert not torch.equal(_logits(model), base)

    asyncio.run(_worker(model)._load_round_base(_round(state.base_shard)))

    assert torch.equal(_logits(model), base)


def test_the_residual_shard_writer_creates_its_folder(family_name, tmp_path):
    model, _ = shared_model.get_model_from_checkpoint(
        config=_validator_config(family_name, tmp_path), load_global_checkpoint=False,
    )

    path = save_residual_shard(model, tmp_path / "a" / "b", 3)

    assert path == tmp_path / "a" / "b" / "model_expgroup_3.safetensors" and path.is_file()


def test_validators_without_a_winner_commit_one_hash(family_name, tmp_path):
    """Different fresh draws, one no-winner hash: those validators pool their stake in the miners' vote."""
    shards = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        config = _validator_config(family_name, tmp_path / str(seed))
        shards.append(run.setup_training(config, torch.device("cpu")).base_shard)
    first, second = (load_file(str(shard)) for shard in shards)
    assert any(not torch.equal(first[k], second[k]) for k in first)

    commits = [run._round_commit({}, shard) for shard in shards]
    assert commits[0] == commits[1] and commits[0].hf_revision is None
    # A revision without a hash advertises nothing either.
    assert run._round_commit({"revision": "abc"}, shards[0]) == commits[0]


def test_a_published_baseline_is_advertised_and_taken(tmp_path):
    published = {"revision": "0123456789abcdef", "model_hash": "feed", "round_id": 1, "uid": 2}

    assert run._round_commit(published, tmp_path / "never_read.safetensors") == run.RoundCommit(
        "feed", "0123456789abcdef",
    )
    assert published == {}
