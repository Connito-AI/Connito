"""The 2026-07-10 crash: a transient HF Hub failure inside the foreground
baseline dataloader build (`_evaluate_on_fresh_loader_sync` →
`load_streaming_shard` → `requests.ConnectionError`) escaped
`evaluate_foreground_round`, reached run()'s top-level handler, and killed
the validator — losing round 8590274 entirely.

These tests pin the fix: the baseline build retries with backoff and, on
exhaustion, `evaluate_foreground_round` returns cleanly (skipping the
foreground pass) instead of raising.
"""

from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace

import pytest


def _install_stub_if_unavailable(mod_path: str, attrs: dict) -> None:
    try:
        __import__(mod_path)
        return
    except Exception:
        pass
    mod = types.ModuleType(mod_path)
    for name, value in attrs.items():
        setattr(mod, name, value)
    sys.modules[mod_path] = mod


_install_stub_if_unavailable(
    "connito.shared.dataloader",
    {"get_dataloader": lambda **k: None, "materialize_batches": lambda dl, n: []},
)
_install_stub_if_unavailable(
    "connito.shared.evaluate",
    {"evaluate_model": lambda *a, **k: {"val_loss": 100.0}},
)

import connito.validator.evaluator as evaluator  # noqa: E402
from connito.validator.round import Round  # noqa: E402


def _make_round(*, foreground_uids: tuple[int, ...] = ()) -> Round:
    return Round(
        round_id=42,
        seed="test-seed",
        validator_miner_assignment={},
        foreground_uids=foreground_uids,
        background_uids=(),
        uid_to_hotkey={uid: f"HK{uid}" for uid in foreground_uids},
        model_snapshot_cpu={},
        journal_path=None,
        score_aggregator=None,
        score_path=None,
    )


def _call(monkeypatch, *, baseline_fn, completed_out=None):
    """Invoke evaluate_foreground_round with minimal stubs and no sleeps."""
    monkeypatch.setattr(
        evaluator, "FOREGROUND_BASELINE_RETRY_DELAYS_SEC", (0.0, 0.0, 0.0),
    )
    monkeypatch.setattr(evaluator, "_evaluate_on_fresh_loader_sync", baseline_fn)
    # subtensor.block > end_block so the polling loop never runs on the
    # success path; the failure path returns before either is consulted.
    subtensor = SimpleNamespace(block=1_001)
    return asyncio.run(
        evaluator.evaluate_foreground_round(
            config=SimpleNamespace(),
            round_obj=_make_round(),
            subtensor=subtensor,
            step=1,
            device="cpu",
            base_model=None,
            tokenizer=None,
            end_block=1_000,
            expert_group_assignment={},
            per_miner_eval_timeout_sec=5.0,
            completed_out=completed_out,
        )
    )


def test_baseline_hf_failure_degrades_instead_of_raising(monkeypatch) -> None:
    """Repro of the crash: every baseline attempt raises ConnectionError.
    Before the fix this propagated out (process-fatal); now the round's
    foreground pass is skipped and an empty result is returned."""
    calls = {"n": 0}

    def _always_fails(**kwargs):
        calls["n"] += 1
        raise ConnectionError("huggingface.co Read timed out")

    result = _call(monkeypatch, baseline_fn=_always_fails)
    assert result == []
    assert calls["n"] == 3  # one attempt per configured delay


def test_baseline_failure_returns_completed_out_alias(monkeypatch) -> None:
    """When the caller pre-allocates `completed_out` (run.py does, so
    partial scores survive cancellation), the degraded path must return
    that same list, not a fresh one."""
    sink: list = []

    def _always_fails(**kwargs):
        raise ConnectionError("boom")

    result = _call(monkeypatch, baseline_fn=_always_fails, completed_out=sink)
    assert result is sink


def test_baseline_transient_failure_recovers_on_retry(monkeypatch) -> None:
    """First attempt fails (the transient blip), second succeeds — the
    round proceeds with a real baseline instead of being skipped."""
    calls = {"n": 0}

    def _flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("transient")
        return {"val_loss": 4.0}

    result = _call(monkeypatch, baseline_fn=_flaky)
    # Empty foreground set + block past end_block → completes with no jobs,
    # but crucially it got PAST the baseline (no skip, no raise).
    assert result == []
    assert calls["n"] == 2


def test_baseline_success_unchanged(monkeypatch) -> None:
    calls = {"n": 0}

    def _ok(**kwargs):
        calls["n"] += 1
        return {"val_loss": 4.2}

    result = _call(monkeypatch, baseline_fn=_ok)
    assert result == []
    assert calls["n"] == 1  # no spurious retries on success
