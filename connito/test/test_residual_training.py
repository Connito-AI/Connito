"""The miner's training loss and loop on residual experts, on tiny models of each family.

CPU only, nothing downloaded: `python -m pytest connito/test/test_residual_training.py`.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from transformers import AutoTokenizer

from connito.miner import train, train_helper
from connito.shared import dataloader, evaluate
from connito.shared import model as shared_model
from connito.shared.config import MinerConfig, ResidualCfg
from connito.shared.evaluate import chunked_lm_loss
from connito.shared.modeling.residual import freeze_all_but_residual

VOCAB = 128


def _config(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        role="miner",
        model=SimpleNamespace(family=name, device="cpu", quantization="off",
                              quantization_miner="off"),
        task=SimpleNamespace(exp=SimpleNamespace(residual=ResidualCfg(k=1, threshold=0.5, temperature=0.5))),
        ckpt=SimpleNamespace(resume_from_ckpt=False, use_pretrained_only=False),
    )


def _model(name: str, dtype: torch.dtype = torch.bfloat16, draw_down: bool = True) -> torch.nn.Module:
    """The miner's model; `draw_down` makes down_proj non-zero so every residual tensor gets a gradient at once."""
    model, _ = shared_model.get_model_from_checkpoint(rank=0, config=_config(name), load_global_checkpoint=False)
    freeze_all_but_residual(model)
    if dtype is torch.float32:
        model.float()
    with torch.no_grad():
        for name_, param in model.named_parameters():
            if draw_down and ".residual_experts." in name_ and "down_proj" in name_:
                param.normal_(std=0.02)
    return model.train()


def _batch(seed: int = 0) -> dict[str, torch.Tensor]:
    ids = torch.randint(0, VOCAB, (3, 40), generator=torch.Generator().manual_seed(seed))
    mask = torch.ones_like(ids)
    mask[0, 25:] = 0  # a short row
    mask[1, 1:] = 0   # a row that is all padding but its first token
    # The last row is full, so with chunks of 7 the trailing partial chunk (5 positions) is all labelled.
    return {"input_ids": ids, "attention_mask": mask, "labels": ids.masked_fill(mask == 0, -100)}


def _grads(model) -> dict[str, torch.Tensor]:
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}


# bf16 without autocast: the CE itself must still run on fp32 logits, as the full loss's does.
@pytest.mark.parametrize("dtype, autocast", [(torch.float32, False), (torch.bfloat16, True), (torch.bfloat16, False)])
def test_chunked_loss_equals_the_full_loss_in_value_and_gradients(family_name, dtype, autocast, monkeypatch):
    monkeypatch.setattr(evaluate, "_CE_CHUNK_TOKENS", 7)  # 117 positions: chunks that don't divide them
    model, batch = _model(family_name, dtype), _batch()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        full = model(**batch).loss
    full.backward()
    expected = _grads(model)
    model.zero_grad()

    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        loss = chunked_lm_loss(model, batch)
    loss.backward()

    # Both means add the same fp32 terms in a different order, so they can land a few fp32 ulps apart.
    torch.testing.assert_close(loss, full)
    assert all(g.any() for g in expected.values())
    # Equal up to BLAS reduction order, which differs with the chunk shapes and across machines.
    for name, grad in _grads(model).items():
        torch.testing.assert_close(grad.float(), expected[name].float())


def test_chunked_loss_without_grad_equals_the_full_loss(family_name, monkeypatch):
    monkeypatch.setattr(evaluate, "_CE_CHUNK_TOKENS", 7)
    model, batch = _model(family_name, torch.float32).eval(), _batch()

    with torch.no_grad():
        torch.testing.assert_close(chunked_lm_loss(model, batch), model(**batch).loss)


def test_backward_keeps_no_vocab_wide_tensor(family_name):
    """Only one chunk's logits exist at a time, recomputed in backward; nothing as wide as the vocab is saved."""
    model, saved = _model(family_name), []

    def pack(t):
        saved.append(tuple(t.shape))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = chunked_lm_loss(model, _batch())
    loss.backward()

    assert saved
    assert [s for s in saved if s and s[-1] == VOCAB] == []


def test_a_batch_with_no_labelled_token_gives_nan_as_the_full_loss_does(family_name):
    model, batch = _model(family_name, torch.float32).eval(), _batch()
    batch["labels"] = torch.full_like(batch["labels"], -100)

    with torch.no_grad():
        assert math.isnan(chunked_lm_loss(model, batch).item())
        assert math.isnan(model(**batch).loss.item())


def test_evaluate_model_scores_what_the_full_logits_loss_scored(family_name, monkeypatch):
    """Each batch's loss equals `model(**batch).loss` in the same autocast; a batch with no labels still drops out."""
    model, full, scored, chunked = _model(family_name), [], [], evaluate.chunked_lm_loss

    def both(model, batch):
        full.append(model(**batch).loss.item())
        scored.append(chunked(model, batch))
        return scored[-1]

    monkeypatch.setattr(evaluate, "chunked_lm_loss", both)
    unlabelled = _batch(3) | {"labels": torch.full_like(_batch(3)["labels"], -100)}
    batches = [_batch(0), _batch(1), _batch(2), unlabelled]

    metrics = evaluate.evaluate_model(step=0, model=model, eval_dataloader=batches, device=torch.device("cpu"),
                                      max_eval_batches=None)

    assert [loss.item() for loss in scored[:3]] == pytest.approx(full[:3])
    assert (metrics["scored_batches"], metrics["nan_batches"]) == (3, 1)
    assert metrics["val_loss"] == pytest.approx(sum(full[:3]) / 3)


def _miner_config(root: Path) -> MinerConfig:
    task = root / "expert_groups" / "exp_pmc_residual"
    task.mkdir(parents=True)
    (task / "config.yaml").write_text(Path("expert_groups/exp_pmc_residual/config.yaml").read_text())
    (root / "config.yaml").write_text(yaml.safe_dump({
        "run": {"root_path": str(root)},
        "chain": {"hotkey_ss58": "test-hk", "coldkey_ss58": "test-ck", "uid": 0},
    }))
    config = MinerConfig.from_path(root / "config.yaml", active_task=None, auto_update_config=True)
    config.local_par.gradient_accumulation_steps = 1
    config.log.metric_interval = 10**9  # local evaluation at inner step 0 only
    config.ckpt.checkpoint_interval = None
    config.ckpt.enable_peer_resync = False
    return config


def _run_loop(model, batches, config: MinerConfig, monkeypatch, **stubs) -> SimpleNamespace:
    """Drive the real `train_worker` over `batches` with everything around the loop stubbed; record what it did."""
    seen = SimpleNamespace(losses=[], logged=[], rebuilt=[])

    def loss(model, batch):
        out = chunked_lm_loss(model, batch)
        assert torch.isfinite(out)
        seen.losses.append((torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu"), out.item()))
        return out

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    seen.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    setup = (model, optimizer, seen.scheduler, None, batches, None)
    for name, stub in {
        "setup_training": lambda *a, **k: setup,
        "chunked_lm_loss": loss,
        "get_dataloader": lambda *a, **k: seen.rebuilt.append(k["batch_size"]) or [],
        "setup_chain_worker": lambda config: (None, object(), None),
        "SystemStatePoller": lambda **k: SimpleNamespace(start=lambda: None, stop=lambda: None),
        "PhaseManager": lambda config, subtensor: None,
        "MetricLogger": lambda *a, **k: SimpleNamespace(log=lambda *a, **k: None, close=lambda: None),
        "TelemetryManager": lambda: SimpleNamespace(start_server=lambda **k: None),
        "get_base_tokenizer": lambda config: None,
        "get_status": lambda **k: seen.logged.append(float(k["loss_batch"])) or {},
        "model_health": lambda model, step: {},
        "sync_active_task": lambda config: False,
        **stubs,
    }.items():
        monkeypatch.setattr(train, name, stub)
    train.train_worker(rank=0, world_size=1, config=config)
    return seen


def test_the_training_loop_moves_only_the_residual_in_bf16(family_name, tmp_path, monkeypatch):
    """The real loop from the null-residual init: two labelled batches train, a batch with no label is skipped."""
    model = _model(family_name, draw_down=False)
    unlabelled = _batch(1) | {"labels": torch.full_like(_batch(1)["labels"], -100)}
    base = {n: t.clone() for n, t in [*model.named_parameters(), *model.named_buffers()] if ".residual_" not in n}
    residual = {n: p.clone() for n, p in model.named_parameters() if ".residual_" in n}

    seen = _run_loop(model, [_batch(0), unlabelled, _batch(2)], _miner_config(tmp_path), monkeypatch)

    assert [(on, dtype) for on, dtype, _ in seen.losses] == [(True, torch.bfloat16)] * 2
    assert seen.scheduler.last_epoch == 2
    assert seen.rebuilt == [2]  # the next epoch's loader, at the miner's micro-batch
    after = dict([*model.named_parameters(), *model.named_buffers()])
    assert all(torch.equal(after[n].view(torch.uint8), t.view(torch.uint8)) for n, t in base.items())
    # down_proj moves at step 1; gate, up and the router follow at step 2.
    assert [n for n, t in residual.items() if torch.equal(after[n], t)] == []


def test_accumulated_micro_batches_are_averaged(family_name, tmp_path, monkeypatch):
    config = _miner_config(tmp_path)
    config.local_par.gradient_accumulation_steps = 2

    seen = _run_loop(_model(family_name), [_batch(0), _batch(2)], config, monkeypatch)

    assert seen.scheduler.last_epoch == 1
    mean = sum(loss for *_, loss in seen.losses) / 2
    assert seen.logged and seen.logged == pytest.approx([mean] * len(seen.logged))


# One error for each recovery branch: the network errors, and anything else naming an HF server error.
@pytest.mark.parametrize("error", [ConnectionError("read timed out"), Exception("HF server error 503")])
def test_a_network_error_in_local_eval_rebuilds_the_loader_at_the_miners_micro_batch(family_name, error, tmp_path,
                                                                                    monkeypatch):
    def offline(**kwargs):
        raise error

    monkeypatch.setattr(train.time, "sleep", lambda seconds: None)
    seen = _run_loop(_model(family_name), [_batch(0)], _miner_config(tmp_path), monkeypatch,
                     evaluate_model=offline, wait_till=lambda *a, **k: None)

    assert seen.rebuilt == [2, 2]  # after the network error, then for the next epoch


def test_a_model_with_no_residual_experts_is_refused():
    with pytest.raises(ValueError, match="no residual experts"):
        freeze_all_but_residual(torch.nn.Linear(2, 2))


def test_the_train_loader_takes_the_miners_micro_batch_and_the_eval_loader_the_tasks(tiny_qwen_checkpoint, monkeypatch):
    rows = [{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}] * 4
    monkeypatch.setattr(dataloader.DefaultStreamingTorchDataset, "get_tokenised_dataset",
                        classmethod(lambda cls, **k: rows))
    data = SimpleNamespace(world_size=1, rank=0, vali_fraction=0.1, per_device_train_batch_size=1, num_workers=0)
    config = SimpleNamespace(task=SimpleNamespace(exp=SimpleNamespace(data=data)))
    tokenizer = AutoTokenizer.from_pretrained(tiny_qwen_checkpoint)

    def rows_per_batch(**kw) -> int:
        return next(iter(dataloader.get_dataloader(config, tokenizer, **kw)))["input_ids"].shape[0]

    assert (rows_per_batch(train=False), rows_per_batch(batch_size=2)) == (1, 2)


def test_setup_training_loads_at_the_miners_micro_batch(family_name, tmp_path, monkeypatch):
    config, seen = _miner_config(tmp_path), []
    monkeypatch.setattr(train, "ExpertManager", lambda config: None)
    monkeypatch.setattr(train, "load_model", lambda *a: (_model(family_name, draw_down=False), None))
    monkeypatch.setattr(train, "get_dataloader", lambda config, **kw: seen.append(kw["batch_size"]) or [])

    train.setup_training(config, 0, torch.device("cpu"), None, None, None, None)

    assert seen == [2]
    assert config.task.exp.data.per_device_train_batch_size == 1  # the validator's eval batch is untouched


def test_tokens_per_step_count_every_accumulated_micro_batch(tmp_path):
    config = _miner_config(tmp_path)
    config.local_par.micro_batch_size, config.local_par.gradient_accumulation_steps = 3, 5

    status = train_helper.get_status(config=config, step=0, training_time=0, total_training_time=0, inner_opt_step=2)

    assert status["total_tokens"] == 2 * 3 * 5 * config.task.exp.data.sequence_length
