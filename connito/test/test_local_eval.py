"""Tests for :class:`connito.miner.local_eval.LocalEvaluator` (Unit 8).

The mini-eval pipeline is purely observational — it shouldn't mutate
gradients, shouldn't crash when the eval throws, and should be opt-in
via ``MinerConfig.local_eval.enabled``. The tests here cover those
contracts plus the schema of the returned dict so the training-loop
caller can rely on it without runtime ``KeyError``s.

We avoid pulling in the full ``MinerConfig`` (which would require a
running Bittensor chain) — instead we hand-roll a ``SimpleNamespace``
that exposes the attributes ``LocalEvaluator`` actually reads. This
keeps the test isolated to the unit under test.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from connito.miner.local_eval import LocalEvaluator, _seed_to_hex
from connito.shared.config import LocalEvalCfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TinyEvalModel(nn.Module):
    """A model that returns a constant loss so eval output is deterministic.

    Mirrors the shape evaluate_model expects: forward returns an object
    with `.loss` (tensor) and `.aux_loss` (tensor). The constant-loss
    setup lets us assert that the eval pipeline averaged correctly
    without depending on a real tokenizer or HF dataset.
    """

    def __init__(self, fixed_loss: float = 1.5, n_finite: int | None = None) -> None:
        super().__init__()
        # A real (trainable) parameter so we can assert grad-state isn't
        # mutated by the mini-eval.
        self.linear = nn.Linear(4, 4)
        self.fixed_loss = float(fixed_loss)
        self.n_finite = n_finite
        self._calls = 0
        self.device_attr = torch.device("cpu")

    @property
    def device(self) -> torch.device:  # evaluate_model reads .device
        return self.device_attr

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        # evaluate_model calls model.to(device); update our exposed attr
        # so the inner `.to(model.device)` call lands on CPU.
        for arg in args:
            if isinstance(arg, torch.device):
                self.device_attr = arg
        if "device" in kwargs and isinstance(kwargs["device"], torch.device):
            self.device_attr = kwargs["device"]
        return result

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        self._calls += 1
        if self.n_finite is not None and self._calls > self.n_finite:
            loss = torch.tensor(float("nan"))
        else:
            loss = torch.tensor(self.fixed_loss)
        return SimpleNamespace(loss=loss, aux_loss=torch.tensor(0.0))


def _fake_batches(n: int) -> list[dict]:
    """Returns ``n`` minimal batches understood by evaluate_model."""
    return [{"input_ids": torch.tensor([[1, 2, 3]])} for _ in range(n)]


def _make_config(
    *,
    enabled: bool = True,
    interval_steps: int = 100,
    max_batches: int = 5,
    seeds: list[int] | None = None,
    use_baseline_compare: bool = False,
) -> SimpleNamespace:
    """Build a SimpleNamespace that quacks like MinerConfig for the
    attributes LocalEvaluator touches: ``local_eval.*``, ``task.*``,
    and indirectly ``task.exp.data.world_size`` via get_dataloader.

    The real MinerConfig requires a Bittensor chain connection, which
    isn't available in unit tests — this lets us exercise the
    LocalEvaluator's gating and dispatch without those dependencies.
    """
    seeds = seeds if seeds is not None else [42]
    return SimpleNamespace(
        local_eval=SimpleNamespace(
            enabled=enabled,
            interval_steps=interval_steps,
            max_batches=max_batches,
            seeds=list(seeds),
            use_baseline_compare=use_baseline_compare,
            baseline_checkpoint_path=None,
        ),
        task=SimpleNamespace(
            expert_group_name="exp_test",
            exp=SimpleNamespace(
                data=SimpleNamespace(world_size=1, rank=0),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Config defaults — opt-in shape
# ---------------------------------------------------------------------------

def test_localevalcfg_defaults_are_opt_in():
    """``enabled=False`` so existing miner runs are bit-for-bit unaffected."""
    cfg = LocalEvalCfg()
    assert cfg.enabled is False
    assert cfg.interval_steps == 500
    assert cfg.max_batches == 10
    assert cfg.seeds == [42]
    assert cfg.use_baseline_compare is False
    assert cfg.baseline_checkpoint_path is None


def test_localevalcfg_accepts_overrides():
    cfg = LocalEvalCfg(
        enabled=True,
        interval_steps=200,
        max_batches=20,
        seeds=[1, 2, 3],
        use_baseline_compare=True,
        baseline_checkpoint_path="/tmp/baseline.pt",
    )
    assert cfg.enabled is True
    assert cfg.interval_steps == 200
    assert cfg.max_batches == 20
    assert cfg.seeds == [1, 2, 3]
    assert cfg.use_baseline_compare is True
    assert cfg.baseline_checkpoint_path == "/tmp/baseline.pt"


# ---------------------------------------------------------------------------
# Seed helper
# ---------------------------------------------------------------------------

def test_seed_to_hex_is_deterministic():
    a = _seed_to_hex(42)
    b = _seed_to_hex(42)
    assert a == b
    assert len(a) == 64  # SHA-256 hex digest
    assert isinstance(int(a[:8], 16), int)  # consumable by get_dataloader


def test_seed_to_hex_differs_per_seed():
    assert _seed_to_hex(1) != _seed_to_hex(2)
    assert _seed_to_hex(0) != _seed_to_hex(42)


# ---------------------------------------------------------------------------
# Interval gating — Unit 8 spec says "eval not called when step < interval"
# ---------------------------------------------------------------------------

def test_should_run_disabled_by_default():
    """Default config (`enabled=False`) must never fire `should_run`."""
    cfg = _make_config(enabled=False)
    evaluator = LocalEvaluator(
        config=cfg,
        tokenizer=MagicMock(),
        device=torch.device("cpu"),
        rank=0,
    )
    for step in (0, 1, 100, 500, 1000, 10_000):
        assert evaluator.should_run(step) is False


def test_should_run_gating_when_enabled():
    """When enabled, should_run only returns True on multiples of
    `interval_steps`, and never at step 0."""
    cfg = _make_config(enabled=True, interval_steps=100)
    evaluator = LocalEvaluator(
        config=cfg,
        tokenizer=MagicMock(),
        device=torch.device("cpu"),
        rank=0,
    )
    assert evaluator.should_run(0) is False
    assert evaluator.should_run(50) is False
    assert evaluator.should_run(99) is False
    assert evaluator.should_run(100) is True
    assert evaluator.should_run(101) is False
    assert evaluator.should_run(200) is True
    assert evaluator.should_run(450) is False
    assert evaluator.should_run(500) is True


def test_2_step_mock_training_does_not_trigger_eval():
    """Spec: a 2-step mock training with interval=100 must never call eval."""
    cfg = _make_config(enabled=True, interval_steps=100)
    evaluator = LocalEvaluator(
        config=cfg,
        tokenizer=MagicMock(),
        device=torch.device("cpu"),
        rank=0,
    )
    eval_calls: list[int] = []
    with patch("connito.miner.local_eval.evaluate_model") as mock_eval, patch(
        "connito.miner.local_eval.get_dataloader"
    ) as mock_dl:
        mock_eval.side_effect = lambda **kwargs: eval_calls.append(kwargs["step"]) or {
            "val_loss": 1.0,
            "scored_batches": 1,
            "nan_batches": 0,
        }
        mock_dl.return_value = iter(_fake_batches(5))
        model = TinyEvalModel()

        for step in (1, 2):
            if evaluator.should_run(step):
                evaluator.run(model=model, step=step)

    assert eval_calls == [], "eval must NOT fire when step < interval"
    assert mock_eval.call_count == 0


# ---------------------------------------------------------------------------
# Run schema — returned dict must contain the keys the trainer logs
# ---------------------------------------------------------------------------

def test_run_returns_expected_schema():
    cfg = _make_config(enabled=True, interval_steps=10, max_batches=5, seeds=[42])
    with patch("connito.miner.local_eval.get_dataloader") as mock_dl:
        mock_dl.return_value = _fake_batches(6)
        evaluator = LocalEvaluator(
            config=cfg,
            tokenizer=MagicMock(),
            device=torch.device("cpu"),
            rank=0,
        )
        model = TinyEvalModel(fixed_loss=2.0)
        result = evaluator.run(model=model, step=10)

    expected_keys = {
        "local_val_loss",
        "local_delta",
        "local_perplexity",
        "local_scored_batches",
        "local_nan_batches",
        "local_eval_step",
    }
    assert expected_keys.issubset(result.keys()), f"Missing keys: {expected_keys - result.keys()}"
    # With a constant-loss model the average val_loss across batches
    # equals the constant we configured.
    assert math.isclose(result["local_val_loss"], 2.0, abs_tol=1e-5)
    assert result["local_scored_batches"] > 0
    assert result["local_nan_batches"] == 0
    assert result["local_eval_step"] == 10
    # No baseline configured → delta is NaN.
    assert math.isnan(result["local_delta"])
    # perplexity = exp(2.0) ≈ 7.389
    assert math.isclose(result["local_perplexity"], math.exp(2.0), abs_tol=1e-3)


def test_run_aggregates_over_multiple_seeds():
    """Two seeds → val_loss is averaged across them."""
    cfg = _make_config(enabled=True, interval_steps=10, max_batches=3, seeds=[1, 2])
    with patch("connito.miner.local_eval.get_dataloader") as mock_dl:
        mock_dl.side_effect = lambda *a, **kw: _fake_batches(4)
        evaluator = LocalEvaluator(
            config=cfg,
            tokenizer=MagicMock(),
            device=torch.device("cpu"),
            rank=0,
        )
        model = TinyEvalModel(fixed_loss=3.0)
        result = evaluator.run(model=model, step=10)

    # Two seeds × constant 3.0 → avg should still be 3.0.
    assert math.isclose(result["local_val_loss"], 3.0, abs_tol=1e-5)
    # `evaluate_model` checks the limit AFTER scoring (`batch_step >=
    # max_eval_batches` post-increment), so it scores max_batches+1
    # batches per seed: 4 batches × 2 seeds = 8 scored batches total.
    assert result["local_scored_batches"] == 8


# ---------------------------------------------------------------------------
# Mutation safety — must not touch gradients
# ---------------------------------------------------------------------------

def test_run_does_not_mutate_model_gradients():
    """The mini-eval is observational. After ``run`` returns, the
    model's gradient buffers must be exactly what they were on entry.
    """
    cfg = _make_config(enabled=True, interval_steps=10, max_batches=3, seeds=[42])
    with patch("connito.miner.local_eval.get_dataloader") as mock_dl:
        mock_dl.return_value = _fake_batches(4)
        evaluator = LocalEvaluator(
            config=cfg,
            tokenizer=MagicMock(),
            device=torch.device("cpu"),
            rank=0,
        )
        model = TinyEvalModel(fixed_loss=1.0)
        # Stamp a known gradient pattern onto the model.
        for p in model.parameters():
            p.requires_grad_(True)
            p.grad = torch.full_like(p, 0.25)

        snapshots = {id(p): p.grad.detach().clone() for p in model.parameters()}
        evaluator.run(model=model, step=10)

        for p in model.parameters():
            assert p.grad is not None, "evaluator must not clear gradient buffers"
            assert torch.equal(p.grad, snapshots[id(p)]), (
                "mini-eval mutated a gradient buffer — must be observational"
            )


def test_run_restores_train_mode():
    """``evaluate_model`` flips the model to .eval(); the evaluator
    must restore it for the caller."""
    cfg = _make_config(enabled=True, interval_steps=10, max_batches=3, seeds=[42])
    with patch("connito.miner.local_eval.get_dataloader") as mock_dl:
        mock_dl.return_value = _fake_batches(4)
        evaluator = LocalEvaluator(
            config=cfg,
            tokenizer=MagicMock(),
            device=torch.device("cpu"),
            rank=0,
        )
        model = TinyEvalModel()
        model.train()
        assert model.training is True

        evaluator.run(model=model, step=10)
        assert model.training is True, "model.train() not restored after mini-eval"


def test_run_preserves_eval_mode_when_called_from_eval():
    """If the caller happens to be in eval mode already, don't force a
    flip back to train — preserve the mode we found."""
    cfg = _make_config(enabled=True, interval_steps=10, max_batches=3, seeds=[42])
    with patch("connito.miner.local_eval.get_dataloader") as mock_dl:
        mock_dl.return_value = _fake_batches(4)
        evaluator = LocalEvaluator(
            config=cfg,
            tokenizer=MagicMock(),
            device=torch.device("cpu"),
            rank=0,
        )
        model = TinyEvalModel()
        model.eval()
        assert model.training is False

        evaluator.run(model=model, step=10)
        assert model.training is False


# ---------------------------------------------------------------------------
# NaN-batch handling — must correctly count NaN batches via evaluate_model
# ---------------------------------------------------------------------------

def test_run_handles_partial_nan_batches():
    """Half-finite/half-NaN model: val_loss == finite_loss (NaN batches
    drop out of the divisor per evaluate_model's contract), and the
    aggregated dict surfaces the NaN counts."""
    cfg = _make_config(enabled=True, interval_steps=10, max_batches=6, seeds=[42])
    with patch("connito.miner.local_eval.get_dataloader") as mock_dl:
        mock_dl.return_value = _fake_batches(8)
        evaluator = LocalEvaluator(
            config=cfg,
            tokenizer=MagicMock(),
            device=torch.device("cpu"),
            rank=0,
        )
        # Model: first 4 batches finite (loss=2.0), rest NaN.
        model = TinyEvalModel(fixed_loss=2.0, n_finite=4)
        result = evaluator.run(model=model, step=10)

    # 4 finite at 2.0 → val_loss = 2.0 (NaN batches NOT padding the divisor).
    assert math.isclose(result["local_val_loss"], 2.0, abs_tol=1e-5)
    assert result["local_scored_batches"] == 4
    # We asked for `max_batches=6` so evaluate_model yields 6 + 1 sentinel,
    # of which 2 land in the NaN region (calls 5 and 6).
    assert result["local_nan_batches"] >= 1


def test_run_handles_fully_nan_model():
    """A model that NaNs every batch produces +inf val_loss (per the
    `evaluate_model` contract that protects against gaming)."""
    cfg = _make_config(enabled=True, interval_steps=10, max_batches=3, seeds=[42])
    with patch("connito.miner.local_eval.get_dataloader") as mock_dl:
        mock_dl.return_value = _fake_batches(5)
        evaluator = LocalEvaluator(
            config=cfg,
            tokenizer=MagicMock(),
            device=torch.device("cpu"),
            rank=0,
        )
        model = TinyEvalModel(fixed_loss=1.0, n_finite=0)
        result = evaluator.run(model=model, step=10)

    assert math.isinf(result["local_val_loss"]), "fully-NaN model must yield +inf val_loss"
    assert result["local_scored_batches"] == 0


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def test_run_with_baseline_publishes_delta():
    """When ``use_baseline_compare=True`` and a baseline model is
    supplied, the result dict's delta is finite and equals
    ``baseline_loss - miner_loss``."""
    cfg = _make_config(
        enabled=True,
        interval_steps=10,
        max_batches=3,
        seeds=[42],
        use_baseline_compare=True,
    )
    with patch("connito.miner.local_eval.get_dataloader") as mock_dl:
        mock_dl.side_effect = lambda *a, **kw: _fake_batches(4)
        baseline = TinyEvalModel(fixed_loss=3.0)
        miner_model = TinyEvalModel(fixed_loss=2.0)
        evaluator = LocalEvaluator(
            config=cfg,
            tokenizer=MagicMock(),
            device=torch.device("cpu"),
            rank=0,
            baseline_model=baseline,
        )
        result = evaluator.run(model=miner_model, step=10)

    # baseline 3.0 − miner 2.0 = 1.0 (positive == miner is better).
    assert math.isclose(result["local_delta"], 1.0, abs_tol=1e-5)
    # Baseline cached after first call.
    assert evaluator._baseline_loss_by_seed.get(42) is not None
    assert math.isclose(evaluator._baseline_loss_by_seed[42], 3.0, abs_tol=1e-5)


def test_baseline_evaluated_once_per_seed():
    """Calling ``run`` twice should evaluate the baseline only on the
    first call — subsequent calls reuse the cache."""
    cfg = _make_config(
        enabled=True,
        interval_steps=10,
        max_batches=3,
        seeds=[42],
        use_baseline_compare=True,
    )
    baseline_calls = []
    miner_calls = []

    real_evaluate = __import__(
        "connito.shared.evaluate", fromlist=["evaluate_model"]
    ).evaluate_model

    def _tracking_evaluate(*args, **kwargs):
        model = kwargs.get("model") or (args[1] if len(args) > 1 else None)
        result = real_evaluate(*args, **kwargs)
        if isinstance(model, TinyEvalModel):
            if model.fixed_loss == 3.0:
                baseline_calls.append(1)
            else:
                miner_calls.append(1)
        return result

    with patch("connito.miner.local_eval.get_dataloader") as mock_dl, patch(
        "connito.miner.local_eval.evaluate_model", side_effect=_tracking_evaluate
    ):
        mock_dl.side_effect = lambda *a, **kw: _fake_batches(4)
        baseline = TinyEvalModel(fixed_loss=3.0)
        miner_model = TinyEvalModel(fixed_loss=2.0)
        evaluator = LocalEvaluator(
            config=cfg,
            tokenizer=MagicMock(),
            device=torch.device("cpu"),
            rank=0,
            baseline_model=baseline,
        )
        evaluator.run(model=miner_model, step=10)
        evaluator.run(model=miner_model, step=20)

    assert len(baseline_calls) == 1, (
        f"baseline must be evaluated exactly once per seed; got {len(baseline_calls)}"
    )
    assert len(miner_calls) == 2, (
        f"miner must be evaluated once per run; got {len(miner_calls)}"
    )


def test_run_without_baseline_skips_delta():
    """``use_baseline_compare=False`` and no baseline model → delta is NaN."""
    cfg = _make_config(
        enabled=True,
        interval_steps=10,
        max_batches=3,
        seeds=[42],
        use_baseline_compare=False,
    )
    with patch("connito.miner.local_eval.get_dataloader") as mock_dl:
        mock_dl.return_value = _fake_batches(4)
        evaluator = LocalEvaluator(
            config=cfg,
            tokenizer=MagicMock(),
            device=torch.device("cpu"),
            rank=0,
            baseline_model=None,
        )
        model = TinyEvalModel(fixed_loss=2.0)
        result = evaluator.run(model=model, step=10)

    assert math.isnan(result["local_delta"])
    # Internal state: no baseline cached.
    assert evaluator._baseline_loss_by_seed == {}


# ---------------------------------------------------------------------------
# Empty-seeds edge case
# ---------------------------------------------------------------------------

def test_empty_seeds_returns_nan_result():
    """An empty `seeds` list shouldn't crash — return NaN result so the
    trainer logs a row but doesn't blow up."""
    cfg = _make_config(enabled=True, interval_steps=10, seeds=[])
    evaluator = LocalEvaluator(
        config=cfg,
        tokenizer=MagicMock(),
        device=torch.device("cpu"),
        rank=0,
    )
    model = TinyEvalModel()
    result = evaluator.run(model=model, step=10)
    assert math.isnan(result["local_val_loss"])
    assert result["local_scored_batches"] == 0
    assert result["local_nan_batches"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
