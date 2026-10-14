import pytest
import torch
import torch.nn as nn

from connito.shared import evaluate
from connito.shared.evaluate import evaluate_model


def _scripted(monkeypatch, losses):
    """Make evaluate_model's loss for each batch come from `losses`, in order; record the autocast dtype."""
    seen = []

    def loss(model, batch):
        seen.append(torch.get_autocast_dtype("cuda" if batch["input_ids"].is_cuda else "cpu"))
        return torch.tensor(losses[len(seen) - 1], device=batch["input_ids"].device)

    monkeypatch.setattr(evaluate, "chunked_lm_loss", loss)
    return seen


def _evaluate(n_batches, device=None, **kw):
    device = device or torch.device("cpu")
    model = nn.Linear(1, 1).to(device)
    model.device = device
    dataloader = [{"input_ids": torch.tensor([[1]])} for _ in range(n_batches)]
    return evaluate_model(step=0, model=model, eval_dataloader=dataloader, device=device, **kw)


def test_evaluator_runs_in_bf16(monkeypatch):
    """Both families are published in bf16; fp16 would overflow them, so there is no fallback."""
    seen = _scripted(monkeypatch, [2.4388])

    metrics = _evaluate(1)

    assert seen == [torch.bfloat16]
    assert metrics["val_loss"] == pytest.approx(2.4388)


def test_evaluator_skips_nan_batches_from_divisor(monkeypatch):
    """A miner that NaNs some batches must not get the divisor padded by
    the failed batches. Returned val_loss = mean over scored (finite)
    batches only, not the gamed `(scored / total) * honest_loss`.
    """
    n_finite, n_nan = 4, 6
    _scripted(monkeypatch, [2.0] * n_finite + [float("nan")] * n_nan)

    metrics = _evaluate(n_finite + n_nan, max_eval_batches=None)

    # Pre-fix behavior would have been `(2.0 * 4) / 10 = 0.8`. The fix
    # divides by `scored_batches = 4`, recovering the honest mean.
    assert metrics["val_loss"] == 2.0
    assert metrics["scored_batches"] == n_finite
    assert metrics["nan_batches"] == n_nan


def test_evaluator_returns_inf_when_every_batch_nans(monkeypatch):
    """100% NaN miner must NOT score val_loss=0 (which would be the
    maximum task gain). Return +inf so its score is -inf."""
    import math
    _scripted(monkeypatch, [float("nan")] * 5)

    metrics = _evaluate(5, max_eval_batches=None)

    assert math.isinf(metrics["val_loss"])
    assert metrics["scored_batches"] == 0
    assert metrics["nan_batches"] == 5


def test_evaluator_skips_inf_batches_too(monkeypatch):
    """bf16 overflow lands as +inf before propagating to NaN. The fix
    treats Inf and NaN equivalently: both drop from the divisor."""
    _scripted(monkeypatch, [float("inf")] * 3 + [2.0] * 2)

    metrics = _evaluate(5, max_eval_batches=None)

    # Only the 2 finite batches contribute.
    assert metrics["val_loss"] == 2.0
    assert metrics["scored_batches"] == 2
    assert metrics["nan_batches"] == 3
