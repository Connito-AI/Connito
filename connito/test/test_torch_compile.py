"""Unit tests for the torch.compile opt-in path in the miner training loop.

Covers the four scenarios from Unit 10:
  1. Config defaults (compile off, mode=reduce-overhead, dynamic=False).
  2. With `torch_compile=True`, the helper calls `torch.compile` with the
     mode/dynamic args read from config — invoked on `model.forward` so
     that downstream `state_dict()` / `named_parameters()` keys (used by
     checkpoint sharding and cross-validator hashing) keep their original
     un-prefixed names.
  3. When `torch.compile` raises, the helper logs and leaves the model's
     forward unchanged — training proceeds without compilation.
  4. DDP wrapping order: when the model is already wrapped in
     `DistributedDataParallel`, the compile call wraps the DDP module's
     forward so the compiled artifact composes with DDP per torch
     guidance.

These tests mock `torch.compile` so they execute without GPU and without
the Inductor toolchain.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from connito.miner.train import maybe_torch_compile
from connito.shared.config import MinerConfig


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------

def test_config_defaults_torch_compile_disabled() -> None:
    """torch.compile must default to OFF — operators opt in explicitly."""
    cfg = MinerConfig()
    assert cfg.model.torch_compile is False


def test_config_default_torch_compile_mode_is_reduce_overhead() -> None:
    cfg = MinerConfig()
    assert cfg.model.torch_compile_mode == "reduce-overhead"


def test_config_default_torch_compile_dynamic_is_false() -> None:
    cfg = MinerConfig()
    assert cfg.model.torch_compile_dynamic is False


def test_config_accepts_all_compile_modes() -> None:
    """Literal-typed field must accept every documented Inductor mode."""
    valid_modes = (
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    )
    for mode in valid_modes:
        cfg = MinerConfig()
        cfg.model.torch_compile_mode = mode  # type: ignore[assignment]
        assert cfg.model.torch_compile_mode == mode


# ---------------------------------------------------------------------------
# 2. Mocked torch.compile is called with the right args
# ---------------------------------------------------------------------------

def test_maybe_compile_is_a_noop_when_disabled() -> None:
    """With `torch_compile=False`, `torch.compile` must not be invoked."""
    cfg = MinerConfig()
    cfg.model.torch_compile = False
    model = nn.Linear(4, 4)
    # Bound methods compare equal across accesses iff they bind the same
    # function on the same instance; `is` would fail because each attribute
    # access materializes a fresh bound-method object.
    original_forward = model.forward

    with patch("connito.miner.train.torch.compile") as mocked:
        out = maybe_torch_compile(model, cfg)

    mocked.assert_not_called()
    assert out is model
    assert model.forward == original_forward


def test_maybe_compile_invokes_torch_compile_on_forward_with_config_args() -> None:
    cfg = MinerConfig()
    cfg.model.torch_compile = True
    cfg.model.torch_compile_mode = "reduce-overhead"
    cfg.model.torch_compile_dynamic = False

    model = nn.Linear(4, 4)
    original_forward = model.forward

    def fake_compiled(*args, **kwargs):
        return "compiled-forward"

    with patch("connito.miner.train.torch.compile", side_effect=fake_compiled) as mocked:
        out = maybe_torch_compile(model, cfg)

    mocked.assert_called_once_with(original_forward, mode="reduce-overhead", dynamic=False)
    # Helper returns the original module unchanged; only its `forward` is
    # swapped to the compiled artifact so state_dict() keys stay un-prefixed.
    assert out is model
    assert model.forward == "compiled-forward"


def test_maybe_compile_preserves_state_dict_keys_unprefixed() -> None:
    """state_dict() keys must NOT gain a `_orig_mod.` prefix after compile,
    because the checkpoint sharder and the validator-side hash comparison
    key on the original parameter names.
    """
    cfg = MinerConfig()
    cfg.model.torch_compile = True

    model = nn.Linear(4, 4)
    expected_keys = set(model.state_dict().keys())

    with patch("connito.miner.train.torch.compile", side_effect=lambda fn, **kw: fn):
        maybe_torch_compile(model, cfg)

    assert set(model.state_dict().keys()) == expected_keys


def test_maybe_compile_propagates_max_autotune_mode_and_dynamic_flag() -> None:
    cfg = MinerConfig()
    cfg.model.torch_compile = True
    cfg.model.torch_compile_mode = "max-autotune"
    cfg.model.torch_compile_dynamic = True

    model = nn.Linear(4, 4)
    original_forward = model.forward

    with patch("connito.miner.train.torch.compile", side_effect=lambda fn, **kw: fn) as mocked:
        maybe_torch_compile(model, cfg)

    mocked.assert_called_once_with(original_forward, mode="max-autotune", dynamic=True)


# ---------------------------------------------------------------------------
# 3. Fallback on compile failure
# ---------------------------------------------------------------------------

def test_maybe_compile_falls_back_to_eager_on_exception() -> None:
    """Compile failure must not break training — leave the model's forward
    untouched and return the original model.
    """
    cfg = MinerConfig()
    cfg.model.torch_compile = True

    model = nn.Linear(4, 4)
    original_forward = model.forward

    with patch(
        "connito.miner.train.torch.compile",
        side_effect=RuntimeError("Inductor missing"),
    ) as mocked:
        out = maybe_torch_compile(model, cfg)

    mocked.assert_called_once()
    # Returned model is the original, and its forward is unchanged (compare
    # bound methods with `==`; `is` would fail on every re-access).
    assert out is model
    assert model.forward == original_forward
    # Forward still works as a sanity check.
    y = out(torch.zeros(1, 4))
    assert y.shape == (1, 4)


def test_maybe_compile_swallows_arbitrary_compile_exceptions() -> None:
    """Any exception class raised by torch.compile must be caught."""
    cfg = MinerConfig()
    cfg.model.torch_compile = True

    model = nn.Linear(4, 4)
    original_forward = model.forward

    class _CompilerCrash(Exception):
        pass

    with patch(
        "connito.miner.train.torch.compile",
        side_effect=_CompilerCrash("dynamo barfed"),
    ):
        out = maybe_torch_compile(model, cfg)

    assert out is model
    assert model.forward == original_forward


# ---------------------------------------------------------------------------
# 4. DDP wrapping order
# ---------------------------------------------------------------------------

def test_maybe_compile_compiles_ddp_module_forward_not_inner_forward() -> None:
    """When the training loop hands a DDP-wrapped model to maybe_torch_compile,
    the helper must call torch.compile on the DDP wrapper's `.forward`
    (not the inner module's). This matches PyTorch's recommended
    composition order: compile the wrapper, so allreduce + compile
    cooperate.
    """
    cfg = MinerConfig()
    cfg.model.torch_compile = True
    cfg.model.torch_compile_mode = "reduce-overhead"

    inner = nn.Linear(4, 4)

    # Use a real module to host an attribute-settable `.forward`.
    class _FakeDDP(nn.Module):
        def __init__(self, module: nn.Module) -> None:
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):  # pragma: no cover - mocked
            return self.module(*args, **kwargs)

    ddp_wrapper = _FakeDDP(inner)

    with patch("connito.miner.train.torch.compile", side_effect=lambda fn, **kw: fn) as mocked:
        out = maybe_torch_compile(ddp_wrapper, cfg)

    # Critical: the FIRST positional arg is the DDP wrapper's forward,
    # never the unwrapped inner module's forward — verifies the
    # documented compose order. Compare via the bound method's underlying
    # function + instance to be robust to bound-method object identity.
    mocked.assert_called_once()
    args, kwargs = mocked.call_args
    compiled_target = args[0]
    assert compiled_target.__self__ is ddp_wrapper
    assert compiled_target.__self__ is not inner
    assert compiled_target.__func__ is _FakeDDP.forward
    assert compiled_target.__func__ is not nn.Linear.forward
    assert kwargs == {"mode": "reduce-overhead", "dynamic": False}
    assert out is ddp_wrapper


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
