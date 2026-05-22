"""Tests for `bench.local_scorer`.

Strategy: we exercise the scoring math + JSON shape with a tiny custom
`nn.Module` that returns scripted losses, plus an in-memory dataloader.
We avoid the real `get_dataloader` (which streams from HuggingFace and
costs a real network call) and the real DeepSeek-V2-Lite model (too
heavy for CI). The validator-equivalent code path being tested is the
loss aggregation + delta + score calculation; that's the part most
sensitive to typos and easiest to break.

Heavy end-to-end tests that exercise the full HF + DeepSeek-V2-Lite
pipeline are gated by `RUN_LOCAL_SCORER_E2E=1` and skipped by default.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from bench.local_scorer import (
    _normalize_seed,
    _resolve_device,
    _resolve_seeds,
    score_checkpoint,
    score_checkpoint_for_seed,
)


# -----------------------------------------------------------------------------
# Test fixtures: tiny "model" + scripted dataloader
# -----------------------------------------------------------------------------
class _ScriptedLossModel(nn.Module):
    """Minimal nn.Module that returns a scripted loss for each forward call.

    Uses a real nn.Parameter so `copy.deepcopy` produces a model whose
    forward call is independent — this matters because the scorer
    deep-copies `base_model` once per (baseline, miner) eval and we need
    each copy's `_calls` counter to start fresh.
    """

    def __init__(self, *, losses: list[float], device: torch.device, aux: float = 0.0):
        super().__init__()
        # Real parameter so `state_dict()` is non-empty and
        # `load_state_dict(strict=False)` doesn't bail on emptiness.
        self.dummy = nn.Parameter(torch.zeros(1, device=device))
        self._losses = list(losses)
        self._dev = device
        self._aux = aux
        self._call_idx = 0

    @property
    def device(self) -> torch.device:
        return self._dev

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        loss_value = self._losses[self._call_idx % len(self._losses)]
        self._call_idx += 1
        loss = torch.tensor(loss_value, device=self._dev)
        aux = torch.tensor(self._aux, device=self._dev)
        # Return an object with `.loss` and `.aux_loss` attributes, matching
        # what `evaluate_model` / `_make_per_batch_eval` consume.
        return SimpleNamespace(loss=loss, aux_loss=aux)


def _fake_dataloader(n_batches: int = 6) -> list[dict[str, torch.Tensor]]:
    """In-memory dataloader. Tensors shaped to match what the validator's
    dataloader emits (input_ids only — attention_mask is auto-derived in
    the eval loop).
    """
    return [{"input_ids": torch.tensor([[1, 2, 3]])} for _ in range(n_batches)]


def _fake_config_and_tokenizer(
    *,
    monkeypatch: pytest.MonkeyPatch,
    dataloader_factory,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    """Build a minimal stub config + tokenizer and patch `get_dataloader`
    + `_per_source_losses` to return the in-memory dataloader.

    The scorer reads `config.dataloader.world_size` and threads `seed`
    through to `get_dataloader`. Both are mocked here so no real HF
    network call happens.
    """
    config = SimpleNamespace(
        dataloader=SimpleNamespace(world_size=1),
        task=SimpleNamespace(
            exp=SimpleNamespace(
                data=SimpleNamespace(dataset_sources=None),
                group_id=0,
            ),
        ),
    )
    tokenizer = SimpleNamespace(pad_token="<pad>", eos_token="<eos>")

    monkeypatch.setattr(
        "bench.local_scorer.get_dataloader",
        lambda **kwargs: dataloader_factory(),
    )
    return config, tokenizer


# -----------------------------------------------------------------------------
# Pure-function helpers
# -----------------------------------------------------------------------------
def test_normalize_seed_accepts_int_and_hex():
    """Validator combined-seeds are hex (sha256). Accept either an int or
    a string. Strings are lowercased to keep JSON output stable across
    runs."""
    assert _normalize_seed(0xDEADBEEF) == f"{0xDEADBEEF:064x}"
    assert _normalize_seed("DEADBEEF") == "deadbeef"
    assert _normalize_seed("0xDEADBEEF") == "deadbeef"


def test_normalize_seed_rejects_non_hex():
    """The dataloader does `int(seed[:8], 16)` — a non-hex first 8 chars
    would raise there. Catch it at parse time instead."""
    with pytest.raises(ValueError):
        _normalize_seed("not-a-hex-seed-zzz")


def test_resolve_device_cpu_fallback():
    """No CUDA / no --device → cpu. Test environments without a GPU should
    still be able to run the local scorer (the model just runs slower)."""
    with patch("bench.local_scorer.torch.cuda.is_available", return_value=False):
        dev = _resolve_device(None)
    assert dev.type == "cpu"


def test_resolve_device_explicit_override():
    """--device cpu must always honor the operator's choice even on a
    cuda-equipped box."""
    assert _resolve_device("cpu").type == "cpu"


def test_resolve_seeds_validator_seeds_takes_precedence():
    """--validator-seeds is the most explicit; it must win over
    --multi-seed and --seed."""
    args = SimpleNamespace(
        validator_seeds="abc,def,123456",
        multi_seed=10,
        seed="999",
    )
    assert _resolve_seeds(args) == ["abc", "def", "123456"]


def test_resolve_seeds_multi_seed_pads_with_random():
    """--multi-seed 3 with no --seed → 3 random seeds. With --seed, the
    explicit seed is first and the remainder are random — this lets a
    caller reproduce one specific score and sample around it."""
    args = SimpleNamespace(validator_seeds=None, multi_seed=3, seed=None)
    seeds = _resolve_seeds(args)
    assert len(seeds) == 3
    assert all(len(s) == 64 for s in seeds), "random seeds are sha256-length hex"


# -----------------------------------------------------------------------------
# Scoring math
# -----------------------------------------------------------------------------
def test_score_zero_when_baseline_equals_miner(monkeypatch: pytest.MonkeyPatch):
    """Same model evaluated as baseline AND miner → val_loss is identical
    → delta is 0 → score is 0.

    This is the most important invariant: the scorer must not silently
    grade a miner above the baseline when the miner IS the baseline.
    Catches off-by-one bugs in the loss aggregation.
    """
    device = torch.device("cpu")
    config, tokenizer = _fake_config_and_tokenizer(
        monkeypatch=monkeypatch, dataloader_factory=lambda: _fake_dataloader(6),
    )

    # Single nn.Module instance is used by both baseline and miner code paths.
    # The scorer deep-copies it twice and loads each state_dict back in.
    model = _ScriptedLossModel(losses=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0], device=device)
    sd = model.state_dict()

    result = score_checkpoint_for_seed(
        config=config,
        tokenizer=tokenizer,
        base_model=model,
        miner_state_dict=sd,
        baseline_state_dict=sd,
        seed="deadbeef",
        device=device,
        max_eval_batches=5,
    )

    assert result.baseline_loss == pytest.approx(2.0)
    assert result.miner_val_loss == pytest.approx(2.0)
    assert result.delta == pytest.approx(0.0)
    assert result.score == pytest.approx(0.0)


def test_score_positive_delta_when_miner_better(monkeypatch: pytest.MonkeyPatch):
    """Miner beats baseline → positive delta → positive score = delta ** 1.2.

    Hard-codes baseline=2.0, miner=1.0 so the expected score `1.0 ** 1.2 = 1.0`
    is easy to spot-check.
    """
    device = torch.device("cpu")

    # Build two different models — different scripted losses for baseline vs miner
    baseline_model = _ScriptedLossModel(
        losses=[2.0] * 10, device=device,
    )
    miner_model = _ScriptedLossModel(
        losses=[1.0] * 10, device=device,
    )

    config, tokenizer = _fake_config_and_tokenizer(
        monkeypatch=monkeypatch, dataloader_factory=lambda: _fake_dataloader(6),
    )

    # We can't load one model's state_dict into the other (they're the
    # same class with the same dummy param), so instead of using
    # state_dict swaps, we patch `copy.deepcopy` to dispatch.
    deepcopies = {"baseline": baseline_model, "miner": miner_model}
    call_order = ["baseline", "miner"]
    call_idx = {"n": 0}

    def _deepcopy_dispatch(model):
        which = call_order[call_idx["n"] % 2]
        call_idx["n"] += 1
        return deepcopies[which]

    with patch("bench.local_scorer.copy.deepcopy", side_effect=_deepcopy_dispatch):
        result = score_checkpoint_for_seed(
            config=config,
            tokenizer=tokenizer,
            base_model=baseline_model,  # ignored due to deepcopy patch
            miner_state_dict={},  # strict=False → noop
            baseline_state_dict={},
            seed="deadbeef",
            device=device,
            max_eval_batches=5,
        )

    assert result.baseline_loss == pytest.approx(2.0)
    assert result.miner_val_loss == pytest.approx(1.0)
    assert result.delta == pytest.approx(1.0)
    assert result.score == pytest.approx(1.0 ** 1.2)


def test_score_zero_when_miner_worse_than_baseline(monkeypatch: pytest.MonkeyPatch):
    """Miner does worse → delta clamps to 0 (max(0, ...)) → score 0.

    Critical: the validator MUST NOT give a worse-than-baseline miner
    any positive reward. If our local replica leaks negative deltas
    through, downstream optimization units will chase the wrong target.
    """
    device = torch.device("cpu")
    baseline_model = _ScriptedLossModel(losses=[1.0] * 10, device=device)
    miner_model = _ScriptedLossModel(losses=[5.0] * 10, device=device)

    config, tokenizer = _fake_config_and_tokenizer(
        monkeypatch=monkeypatch, dataloader_factory=lambda: _fake_dataloader(6),
    )

    deepcopies = [baseline_model, miner_model]
    call_idx = {"n": 0}

    def _deepcopy_dispatch(model):
        out = deepcopies[call_idx["n"] % 2]
        call_idx["n"] += 1
        return out

    with patch("bench.local_scorer.copy.deepcopy", side_effect=_deepcopy_dispatch):
        result = score_checkpoint_for_seed(
            config=config, tokenizer=tokenizer,
            base_model=baseline_model,
            miner_state_dict={}, baseline_state_dict={},
            seed="deadbeef", device=device, max_eval_batches=5,
        )

    assert result.baseline_loss == pytest.approx(1.0)
    assert result.miner_val_loss == pytest.approx(5.0)
    assert result.delta == pytest.approx(0.0)
    assert result.score == pytest.approx(0.0)


def test_aux_loss_is_subtracted_from_val_loss(monkeypatch: pytest.MonkeyPatch):
    """Validator's val_loss = (loss_sum - aux_loss_sum) / scored_batches.
    Aux_loss=0.5 with loss=2.0 → val_loss should be 1.5, not 2.0.

    Catches a class of bugs where someone naively forwards `outputs.loss`
    without subtracting the routing aux loss. Real DeepSeek-V2-Lite forward
    passes return both.
    """
    device = torch.device("cpu")
    model = _ScriptedLossModel(losses=[2.0] * 10, device=device, aux=0.5)
    sd = model.state_dict()

    config, tokenizer = _fake_config_and_tokenizer(
        monkeypatch=monkeypatch, dataloader_factory=lambda: _fake_dataloader(6),
    )

    result = score_checkpoint_for_seed(
        config=config, tokenizer=tokenizer,
        base_model=model,
        miner_state_dict=sd, baseline_state_dict=sd,
        seed="deadbeef", device=device, max_eval_batches=5,
    )

    # (2.0 - 0.5) = 1.5 per batch; both baseline and miner report 1.5;
    # delta=0; score=0. The aux=0.5 makes the val_loss reflect (2.0 - 0.5) = 1.5
    # instead of 2.0.
    assert result.baseline_loss == pytest.approx(1.5)
    assert result.miner_val_loss == pytest.approx(1.5)


def test_multi_seed_summary_populated(monkeypatch: pytest.MonkeyPatch):
    """Multi-seed mode populates `multi_seed_summary` with mean/std/min/max
    + per-seed breakdown. Validates the JSON schema contract."""
    device = torch.device("cpu")
    model = _ScriptedLossModel(losses=[2.0] * 10, device=device)
    sd = model.state_dict()

    config, tokenizer = _fake_config_and_tokenizer(
        monkeypatch=monkeypatch, dataloader_factory=lambda: _fake_dataloader(6),
    )

    report = score_checkpoint(
        config=config, tokenizer=tokenizer,
        base_model=model,
        miner_state_dict=sd, baseline_state_dict=sd,
        seeds=["aaaa", "bbbb", "cccc"],
        device=device, max_eval_batches=5,
    )

    assert report["seeds_used"] == ["aaaa", "bbbb", "cccc"]
    assert report["multi_seed_summary"] is not None
    summary = report["multi_seed_summary"]
    assert "mean_score" in summary
    assert "std_score" in summary
    assert "min_score" in summary
    assert "max_score" in summary
    assert len(summary["per_seed"]) == 3
    for entry in summary["per_seed"]:
        assert {"seed", "val_loss", "delta", "score"} <= entry.keys()


def test_single_seed_multi_seed_summary_is_none(monkeypatch: pytest.MonkeyPatch):
    """Single seed → `multi_seed_summary` should be null in the report.
    Downstream callers branch on this — null means "look at the top-level
    fields directly", not-null means "use the aggregated summary".
    """
    device = torch.device("cpu")
    model = _ScriptedLossModel(losses=[2.0] * 10, device=device)
    sd = model.state_dict()

    config, tokenizer = _fake_config_and_tokenizer(
        monkeypatch=monkeypatch, dataloader_factory=lambda: _fake_dataloader(6),
    )

    report = score_checkpoint(
        config=config, tokenizer=tokenizer,
        base_model=model,
        miner_state_dict=sd, baseline_state_dict=sd,
        seeds=["aaaa"],
        device=device, max_eval_batches=5,
    )
    assert report["multi_seed_summary"] is None


def test_json_schema_serializable(monkeypatch: pytest.MonkeyPatch):
    """The report must round-trip through json.dumps without TypeError.
    If a tensor / numpy scalar leaks in, this fails. The CLI uses
    `json.dumps(default=str)` as a safety net, but it's cleaner to
    enforce native floats at the boundary."""
    device = torch.device("cpu")
    model = _ScriptedLossModel(losses=[2.0] * 10, device=device)
    sd = model.state_dict()

    config, tokenizer = _fake_config_and_tokenizer(
        monkeypatch=monkeypatch, dataloader_factory=lambda: _fake_dataloader(6),
    )

    report = score_checkpoint(
        config=config, tokenizer=tokenizer,
        base_model=model,
        miner_state_dict=sd, baseline_state_dict=sd,
        seeds=["aaaa"],
        device=device, max_eval_batches=5,
    )

    serialized = json.dumps(report)
    reparsed = json.loads(serialized)
    assert reparsed["baseline_loss"] == report["baseline_loss"]
    assert reparsed["score"] == report["score"]
    # Required top-level keys per the spec
    for required in (
        "baseline_loss", "miner_val_loss", "delta", "score",
        "scored_batches", "nan_batches", "per_batch_loss",
        "per_source_loss", "seeds_used", "multi_seed_summary",
    ):
        assert required in reparsed, f"missing required key: {required}"


def test_nan_batches_drop_from_divisor(monkeypatch: pytest.MonkeyPatch):
    """Reuses the validator's NaN-batch handling: NaN/Inf batches don't
    enter `scored_batches` so the mean is over finite batches only.
    Mirrors `connito/shared/evaluate.py:99-112`."""
    import math

    device = torch.device("cpu")
    # 4 finite-loss batches + 2 NaN batches → mean over the finite 4 only
    losses = [2.0, 2.0, float("nan"), 2.0, float("nan"), 2.0]
    model = _ScriptedLossModel(losses=losses, device=device)
    sd = model.state_dict()

    config, tokenizer = _fake_config_and_tokenizer(
        monkeypatch=monkeypatch, dataloader_factory=lambda: _fake_dataloader(6),
    )

    result = score_checkpoint_for_seed(
        config=config, tokenizer=tokenizer,
        base_model=model,
        miner_state_dict=sd, baseline_state_dict=sd,
        seed="deadbeef", device=device, max_eval_batches=10,
    )

    # `evaluate_model` (baseline path) and `_make_per_batch_eval` (miner
    # path) both skip NaN batches; the mean over the 4 finite batches is 2.0.
    assert result.miner_val_loss == pytest.approx(2.0)
    # `evaluate_model` returns nan_batches in its dict, but the scorer
    # only surfaces the miner side's count for the report.
    assert result.nan_batches == 2
    assert result.scored_batches == 4


def test_per_batch_loss_length_matches_scored_batches(monkeypatch: pytest.MonkeyPatch):
    """The per-batch loss list should have exactly `scored_batches`
    entries. Catches a regression where NaN batches accidentally get
    appended as 0.0 placeholders."""
    device = torch.device("cpu")
    losses = [1.5, float("inf"), 2.5, float("nan"), 3.5]
    model = _ScriptedLossModel(losses=losses, device=device)
    sd = model.state_dict()

    config, tokenizer = _fake_config_and_tokenizer(
        monkeypatch=monkeypatch, dataloader_factory=lambda: _fake_dataloader(5),
    )

    result = score_checkpoint_for_seed(
        config=config, tokenizer=tokenizer,
        base_model=model,
        miner_state_dict=sd, baseline_state_dict=sd,
        seed="deadbeef", device=device, max_eval_batches=10,
    )

    assert len(result.per_batch_loss) == result.scored_batches


# -----------------------------------------------------------------------------
# CLI smoke
# -----------------------------------------------------------------------------
def test_cli_help_exits_zero():
    """`python -m bench.local_scorer --help` must exit 0 and show usage.
    Caught a class of bugs where someone forgets a required arg in the
    parser and `--help` crashes."""
    result = subprocess.run(
        [sys.executable, "-m", "bench.local_scorer", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"--help exited {result.returncode}, stderr:\n{result.stderr}"
    assert "local_scorer" in result.stdout or "local_scorer" in result.stderr or "checkpoint" in result.stdout.lower()


# -----------------------------------------------------------------------------
# E2E heavy test — gated by env var
# -----------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.skipif(
    os.getenv("RUN_LOCAL_SCORER_E2E") != "1",
    reason="E2E test hits HuggingFace + needs a full DeepSeek-V2-Lite. Set RUN_LOCAL_SCORER_E2E=1.",
)
def test_e2e_real_model():
    """Real end-to-end against the configured base model + HF dataloader.
    Disabled by default — provided so an operator can run it locally
    before shipping changes that touch the dataloader or model loading.
    """
    pytest.skip("placeholder — fill in once a tiny test checkpoint is provisioned in CI")
