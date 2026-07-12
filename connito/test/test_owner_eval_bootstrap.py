"""Tests for model-source dispatch in the owner-eval bootstrap."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from connito.owner_eval import bootstrap


def test_base_source_uses_base_loader_not_chain(monkeypatch):
    sentinel = object()
    calls = {"base": 0}

    def fake_base(cfg, em, dev):
        calls["base"] += 1
        return sentinel

    # If the chain path were taken it would import connito.shared.model; assert
    # we never get there by failing loudly if load_base_model isn't used.
    monkeypatch.setattr(bootstrap, "load_base_model", fake_base)

    cfg = SimpleNamespace(eval_pipeline=SimpleNamespace(model_source="base"))
    model, ckpt = bootstrap.load_latest_full_model(
        config=cfg, expert_manager=None, subtensor=None, wallet=None,
        device=torch.device("cpu"),
    )
    assert model is sentinel
    assert ckpt is None
    assert calls["base"] == 1


def test_default_source_is_chain(monkeypatch):
    # When model_source is absent, default to chain (production) — verified by
    # the base loader NOT being called and the chain import being attempted.
    monkeypatch.setattr(bootstrap, "load_base_model",
                        lambda *a, **k: pytest.fail("should not use base loader"))
    cfg = SimpleNamespace(eval_pipeline=SimpleNamespace())  # no model_source attr

    # Stub the lazily-imported load_model so we don't pull the chain stack.
    import sys
    import types as _types

    class FakeModel:
        def eval(self):
            return self

    fake = FakeModel()
    stub = _types.ModuleType("connito.shared.model")
    stub.load_model = lambda **k: (fake, "CKPT")
    monkeypatch.setitem(sys.modules, "connito.shared.model", stub)

    model, ckpt = bootstrap.load_latest_full_model(
        config=cfg, expert_manager=None, subtensor="ST", wallet="W",
        device=torch.device("cpu"),
    )
    assert model is fake
    assert ckpt == "CKPT"
