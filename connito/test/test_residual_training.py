"""The miner's training loss on residual experts, on tiny models of each family.

CPU only, nothing downloaded: `python -m pytest connito/test/test_residual_training.py`.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from connito.shared import evaluate
from connito.shared import model as shared_model
from connito.shared.config import ResidualCfg
from connito.shared.evaluate import chunked_lm_loss
from connito.shared.modeling.residual import freeze_all_but_residual

VOCAB = 128


def _config(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        role="miner",
        model=SimpleNamespace(family=name, torch_compile=False, device="cpu", quantization="off",
                              quantization_miner="off"),
        task=SimpleNamespace(exp=SimpleNamespace(residual=ResidualCfg(k=1, threshold=0.5, temperature=0.5))),
        ckpt=SimpleNamespace(resume_from_ckpt=False, use_pretrained_only=False),
    )


def _model(name: str, dtype: torch.dtype = torch.bfloat16) -> torch.nn.Module:
    """The miner's model, with down_proj drawn non-zero so every residual tensor gets a gradient."""
    model, _ = shared_model.get_model_from_checkpoint(rank=0, config=_config(name), load_global_checkpoint=False)
    freeze_all_but_residual(model)
    if dtype is torch.float32:
        model.float()
    with torch.no_grad():
        for name_, param in model.named_parameters():
            if ".residual_experts." in name_ and "down_proj" in name_:
                param.normal_(std=0.02)
    return model.train()


def _batch() -> dict[str, torch.Tensor]:
    ids = torch.randint(0, VOCAB, (3, 40), generator=torch.Generator().manual_seed(0))
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
