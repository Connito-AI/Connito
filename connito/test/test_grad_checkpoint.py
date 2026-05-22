"""
Tests for the gradient-checkpointing toggle (Unit 12).

Covers:
1. Helper passes `use_reentrant=False` by default and `True` when asked.
2. Older transformers releases that don't accept `gradient_checkpointing_kwargs`
   fall back to the legacy zero-arg call.
3. Helper is a graceful no-op when the model has no
   `gradient_checkpointing_enable` method.
4. Helper disables `model.config.use_cache` because cache + grad ckpt
   are incompatible.
5. The new `DataCfg` fields default to the spec-required values.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from connito.shared.config import DataCfg
from connito.shared.model import enable_gradient_checkpointing


def _make_mock_model(
    *,
    accepts_kwargs: bool = True,
    has_method: bool = True,
    use_cache: bool | None = False,
) -> MagicMock:
    """
    Build a minimal mock with the bits `enable_gradient_checkpointing`
    inspects:
      - `gradient_checkpointing_enable` callable (optional)
      - `config.use_cache` attribute (optional)
    """
    model = MagicMock(spec=[])  # empty spec; we add attributes explicitly

    if has_method:
        def gce(*args, **kwargs):
            if kwargs and not accepts_kwargs:
                raise TypeError(
                    "gradient_checkpointing_enable() got an unexpected keyword "
                    "argument 'gradient_checkpointing_kwargs'"
                )
            return None

        model.gradient_checkpointing_enable = MagicMock(side_effect=gce)

    if use_cache is not None:
        # `SimpleNamespace` so attribute assignment works like a real
        # `transformers.PretrainedConfig` would.
        model.config = SimpleNamespace(use_cache=use_cache)

    return model


def test_enable_gradient_checkpointing_default_non_reentrant() -> None:
    model = _make_mock_model(accepts_kwargs=True)

    enable_gradient_checkpointing(model)

    model.gradient_checkpointing_enable.assert_called_once_with(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )


def test_enable_gradient_checkpointing_reentrant_true_passes_through() -> None:
    model = _make_mock_model(accepts_kwargs=True)

    enable_gradient_checkpointing(model, reentrant=True)

    model.gradient_checkpointing_enable.assert_called_once_with(
        gradient_checkpointing_kwargs={"use_reentrant": True},
    )


def test_enable_gradient_checkpointing_legacy_transformers_fallback() -> None:
    """
    Older transformers releases reject `gradient_checkpointing_kwargs`.
    Helper should retry with a zero-arg call instead of bubbling the
    TypeError.
    """
    model = _make_mock_model(accepts_kwargs=False)

    enable_gradient_checkpointing(model, reentrant=False)

    calls = model.gradient_checkpointing_enable.call_args_list
    assert len(calls) == 2, calls
    # First call: with kwargs (rejected by mock).
    assert calls[0].kwargs == {
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }
    # Second call: legacy no-arg fallback.
    assert calls[1].args == () and calls[1].kwargs == {}


def test_enable_gradient_checkpointing_no_method_is_warning_only() -> None:
    """
    Helper must not crash when `gradient_checkpointing_enable` is absent
    — operators may run with a model class that doesn't support it (e.g.
    a wrapped optimizer test stub).
    """
    # Plain SimpleNamespace has no `gradient_checkpointing_enable` —
    # the bare-object code path is what the helper has to tolerate. We
    # use SimpleNamespace rather than `MagicMock(spec=[])` because
    # `MagicMock` auto-creates attributes if the spec changes; the
    # explicit object makes the contract obvious.
    bare = SimpleNamespace()
    assert not hasattr(bare, "gradient_checkpointing_enable")

    # Should be a no-op with no exception.
    enable_gradient_checkpointing(bare)


def test_enable_gradient_checkpointing_disables_use_cache() -> None:
    """
    `use_cache=True` is incompatible with grad checkpointing — the helper
    flips it to False so we don't silently train with a stale KV cache
    blowing up activation memory anyway.
    """
    model = _make_mock_model(accepts_kwargs=True, use_cache=True)
    assert model.config.use_cache is True

    enable_gradient_checkpointing(model)

    assert model.config.use_cache is False
    model.gradient_checkpointing_enable.assert_called_once()


def test_enable_gradient_checkpointing_keeps_use_cache_false() -> None:
    """When `use_cache` is already False, the helper still calls the HF
    enable method and leaves the flag alone."""
    model = _make_mock_model(accepts_kwargs=True, use_cache=False)

    enable_gradient_checkpointing(model)

    assert model.config.use_cache is False
    model.gradient_checkpointing_enable.assert_called_once()


def test_enable_gradient_checkpointing_no_config_attr() -> None:
    """Helper must not crash when `model.config` is missing (e.g. a plain
    `nn.Module`). It should still enable checkpointing."""
    model = _make_mock_model(accepts_kwargs=True, use_cache=None)

    enable_gradient_checkpointing(model)

    model.gradient_checkpointing_enable.assert_called_once_with(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )


def test_datacfg_defaults_grad_checkpointing_off() -> None:
    """Opt-in only: both new fields default to False — operators must
    flip them explicitly to take the throughput hit."""
    cfg = DataCfg()
    assert cfg.use_gradient_checkpointing is False
    assert cfg.gradient_checkpointing_reentrant is False


def test_datacfg_explicit_grad_checkpointing_on() -> None:
    """Field round-trips through the pydantic schema."""
    cfg = DataCfg(
        use_gradient_checkpointing=True,
        gradient_checkpointing_reentrant=True,
    )
    assert cfg.use_gradient_checkpointing is True
    assert cfg.gradient_checkpointing_reentrant is True


# -------------------------------
# Real-model smoke test (skipped in unit-test runs by default).
# -------------------------------
def test_enable_gradient_checkpointing_on_gpt2_smoke() -> None:
    """
    End-to-end recipe step #3: load a tiny model and assert
    `enable_gradient_checkpointing` runs without exception. Skipped
    when transformers isn't importable so the unit suite stays
    network-free; HuggingFace will need to be cached locally for this
    to run.
    """
    transformers = pytest.importorskip("transformers")
    try:
        # `from_config` keeps this network-free if the model isn't cached;
        # it instantiates random weights from the architecture spec.
        cfg = transformers.GPT2Config(n_layer=2, n_head=2, n_embd=64)
        model = transformers.GPT2LMHeadModel(cfg)
    except Exception as exc:  # pragma: no cover — fully environment dependent
        pytest.skip(f"GPT-2 init unavailable in this environment: {exc}")

    # Sanity: cache flag starts True on stock GPT-2.
    assert model.config.use_cache is True

    enable_gradient_checkpointing(model)

    assert model.config.use_cache is False
    # `gradient_checkpointing` is set on every transformer block.
    assert any(
        getattr(block, "gradient_checkpointing", False)
        for block in model.transformer.h
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
