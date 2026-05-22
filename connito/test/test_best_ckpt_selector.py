"""Tests for the multi-seed best-checkpoint selector (Unit 15).

The selector is the miner-side opt-in optimization that picks the
checkpoint with the lowest mean val_loss across multiple seeds before
the MinerCommit2 submission. These tests verify the selection logic
without spinning up a real model — the heavy bits (model loading,
dataloader build, ``evaluate_model``) are monkeypatched.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from connito.miner.best_ckpt_selector import (
    BestCheckpointSelector,
    discover_candidate_checkpoints,
)
from connito.shared.config import CommitCfg, CommitSelectionCfg


# ----------------------------------------------------------------------
# Defaults / dataclass tests
# ----------------------------------------------------------------------


class TestCommitSelectionCfgDefaults:
    def test_disabled_by_default(self):
        """The feature is opt-in. Default config keeps the existing
        commit-worker behavior bit-for-bit so a miner upgrade alone
        cannot change selection semantics."""
        cfg = CommitSelectionCfg()
        assert cfg.enabled is False

    def test_candidate_count_default_is_three(self):
        cfg = CommitSelectionCfg()
        assert cfg.candidate_count == 3

    def test_include_ema_default_is_true(self):
        """Unit 13's EMA snapshot is included by default when present;
        the selector tolerates its absence so a miner running Unit 15
        without Unit 13 still benefits."""
        cfg = CommitSelectionCfg()
        assert cfg.include_ema is True

    def test_local_eval_seeds_default(self):
        cfg = CommitSelectionCfg()
        assert cfg.local_eval_seeds == [42, 1337, 314159]

    def test_max_eval_batches_default(self):
        cfg = CommitSelectionCfg()
        assert cfg.max_eval_batches == 20

    def test_commit_cfg_wraps_selection(self):
        cfg = CommitCfg()
        assert isinstance(cfg.selection, CommitSelectionCfg)
        assert cfg.selection.enabled is False

    def test_seeds_default_is_defensive_copy(self):
        """Two CommitSelectionCfg instances must not share a seed list —
        a `default=[...]` (instead of default_factory) would cause
        mutation in one to leak into the other."""
        a = CommitSelectionCfg()
        b = CommitSelectionCfg()
        a.local_eval_seeds.append(99)
        assert 99 not in b.local_eval_seeds


# ----------------------------------------------------------------------
# Helpers: tiny model + mock loader for selector tests
# ----------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Bare-bones model with state_dict semantics for selection tests.

    We don't actually run forward — the selector path under test goes
    through monkeypatched ``evaluate_model`` and a monkeypatched
    ``_load_into_base``, so the model only needs to exist and be a
    distinct Python object per candidate.
    """

    def __init__(self, marker: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.fill_(marker)

    def forward(self, *args, **kwargs):  # pragma: no cover - never called
        raise RuntimeError("test should never call forward")


def _make_config(
    seeds: list[int] | None = None,
    max_batches: int = 5,
) -> MagicMock:
    """Build a minimal duck-typed config matching what the selector reads."""
    cfg = MagicMock()
    cfg.task.exp.group_id = 0
    cfg.task.exp.data.world_size = 1
    cfg.task.exp.data.rank = 0
    return cfg


# ----------------------------------------------------------------------
# Selection logic tests
# ----------------------------------------------------------------------


def test_select_picks_lowest_mean_val_loss(monkeypatch):
    """Three candidates, decreasing val_loss across the three. The
    selector must pick the one with the lowest mean across seeds.
    """
    # Per-candidate per-seed losses. Candidate 2 has the lowest mean.
    losses_by_path = {
        "ckpt_a": [3.0, 3.0, 3.0],  # mean=3.0
        "ckpt_b": [2.0, 2.0, 2.0],  # mean=2.0  <-- winner
        "ckpt_c": [2.5, 2.5, 2.5],  # mean=2.5
    }
    seed_idx = {0: 0, 1: 1, 2: 2}  # map for seeds [42, 1337, 314159]

    # State machine: which candidate is currently loaded.
    state = {"current": None}

    def fake_load(self, path: Path):
        state["current"] = path.name
        return _TinyModel()

    def fake_evaluate_model(*, model, **kwargs):
        # `seed` arg isn't visible here, but we use a closure counter
        # that maps call index to the seed index (selector iterates
        # seeds in order). Each candidate is evaluated for all 3
        # seeds in a row.
        path_name = state["current"]
        seed_call = state.setdefault(f"call_{path_name}", 0)
        loss = losses_by_path[path_name][seed_call]
        state[f"call_{path_name}"] = seed_call + 1
        return {"val_loss": loss}

    # Patch the evaluator and the loader. The dataloader builder is
    # replaced with a no-op iterator so `get_dataloader(...)` doesn't
    # try to hit HuggingFace.
    monkeypatch.setattr(
        "connito.miner.best_ckpt_selector.evaluate_model", fake_evaluate_model
    )
    monkeypatch.setattr(
        "connito.miner.best_ckpt_selector.get_dataloader",
        lambda **kwargs: iter([]),
    )
    monkeypatch.setattr(
        BestCheckpointSelector, "_load_into_base", fake_load, raising=True
    )

    selector = BestCheckpointSelector(
        config=_make_config(),
        tokenizer=MagicMock(),
        base_model=_TinyModel(),
        device=torch.device("cpu"),
        seeds=[42, 1337, 314159],
        max_batches=5,
    )
    candidates = [Path("ckpt_a"), Path("ckpt_b"), Path("ckpt_c")]
    winner, report = selector.select(candidates)

    assert winner == Path("ckpt_b")
    assert report["winner"] == "ckpt_b"
    assert report["winner_mean_val_loss"] == pytest.approx(2.0)
    assert len(report["results"]) == 3
    # Per-seed entries reflect the mock losses.
    b_result = next(r for r in report["results"] if r["path"] == "ckpt_b")
    assert [p["val_loss"] for p in b_result["per_seed"]] == [2.0, 2.0, 2.0]


def test_multi_seed_averaging_picks_robust_candidate(monkeypatch):
    """A candidate that's the lowest on a single seed but high-variance
    must lose to a candidate with a higher peak but lower mean across
    seeds. This is the load-bearing claim for multi-seed selection.
    """
    # ckpt_peaky: one seed at 1.0 (spike), two at 5.0 → mean = 3.67
    # ckpt_robust: three seeds at 3.0 → mean = 3.0  <-- wins
    losses_by_path = {
        "ckpt_peaky": [1.0, 5.0, 5.0],
        "ckpt_robust": [3.0, 3.0, 3.0],
    }
    state = {"current": None}

    def fake_load(self, path: Path):
        state["current"] = path.name
        return _TinyModel()

    def fake_evaluate_model(*, model, **kwargs):
        path_name = state["current"]
        seed_call = state.setdefault(f"call_{path_name}", 0)
        loss = losses_by_path[path_name][seed_call]
        state[f"call_{path_name}"] = seed_call + 1
        return {"val_loss": loss}

    monkeypatch.setattr(
        "connito.miner.best_ckpt_selector.evaluate_model", fake_evaluate_model
    )
    monkeypatch.setattr(
        "connito.miner.best_ckpt_selector.get_dataloader",
        lambda **kwargs: iter([]),
    )
    monkeypatch.setattr(
        BestCheckpointSelector, "_load_into_base", fake_load, raising=True
    )

    selector = BestCheckpointSelector(
        config=_make_config(),
        tokenizer=MagicMock(),
        base_model=_TinyModel(),
        device=torch.device("cpu"),
        seeds=[1, 2, 3],
        max_batches=5,
    )
    winner, report = selector.select([Path("ckpt_peaky"), Path("ckpt_robust")])

    assert winner == Path("ckpt_robust")
    peaky_result = next(r for r in report["results"] if r["path"] == "ckpt_peaky")
    robust_result = next(r for r in report["results"] if r["path"] == "ckpt_robust")
    assert peaky_result["mean_val_loss"] == pytest.approx(11.0 / 3.0)
    assert robust_result["mean_val_loss"] == pytest.approx(3.0)
    assert robust_result["mean_val_loss"] < peaky_result["mean_val_loss"]


def test_empty_candidate_list_raises_value_error():
    selector = BestCheckpointSelector(
        config=_make_config(),
        tokenizer=MagicMock(),
        base_model=_TinyModel(),
        device=torch.device("cpu"),
        seeds=[42],
        max_batches=5,
    )
    with pytest.raises(ValueError, match="candidate_paths is empty"):
        selector.select([])


def test_empty_seeds_raises_value_error(monkeypatch):
    """A selector constructed with no seeds cannot evaluate anything —
    surface the misconfiguration loudly so the caller can fall back."""
    monkeypatch.setattr(
        "connito.miner.best_ckpt_selector.evaluate_model",
        lambda **kwargs: {"val_loss": 1.0},
    )
    monkeypatch.setattr(
        "connito.miner.best_ckpt_selector.get_dataloader",
        lambda **kwargs: iter([]),
    )
    monkeypatch.setattr(
        BestCheckpointSelector,
        "_load_into_base",
        lambda self, path: _TinyModel(),
        raising=True,
    )
    selector = BestCheckpointSelector(
        config=_make_config(),
        tokenizer=MagicMock(),
        base_model=_TinyModel(),
        device=torch.device("cpu"),
        seeds=[],
        max_batches=5,
    )
    with pytest.raises(ValueError, match="no seeds configured"):
        selector.select([Path("only")])


def test_candidate_with_load_error_still_completes(monkeypatch):
    """If one candidate fails to load (corrupt shard, missing file,
    etc.) the selector records it as +inf and keeps going. The round
    must complete against the remaining candidates instead of
    crashing the entire commit."""
    state = {"current": None}

    def fake_load(self, path: Path):
        state["current"] = path.name
        if path.name == "broken":
            raise RuntimeError("corrupt safetensors header")
        return _TinyModel()

    losses_by_path = {"good_a": [2.0, 2.0], "good_b": [1.5, 1.5]}

    def fake_evaluate_model(*, model, **kwargs):
        path_name = state["current"]
        seed_call = state.setdefault(f"call_{path_name}", 0)
        loss = losses_by_path[path_name][seed_call]
        state[f"call_{path_name}"] = seed_call + 1
        return {"val_loss": loss}

    monkeypatch.setattr(
        "connito.miner.best_ckpt_selector.evaluate_model", fake_evaluate_model
    )
    monkeypatch.setattr(
        "connito.miner.best_ckpt_selector.get_dataloader",
        lambda **kwargs: iter([]),
    )
    monkeypatch.setattr(
        BestCheckpointSelector, "_load_into_base", fake_load, raising=True
    )

    selector = BestCheckpointSelector(
        config=_make_config(),
        tokenizer=MagicMock(),
        base_model=_TinyModel(),
        device=torch.device("cpu"),
        seeds=[1, 2],
        max_batches=5,
    )
    winner, report = selector.select(
        [Path("good_a"), Path("broken"), Path("good_b")]
    )

    assert winner == Path("good_b")
    broken_result = next(r for r in report["results"] if r["path"] == "broken")
    assert math.isinf(broken_result["mean_val_loss"])
    assert broken_result.get("load_error") is True


def test_inf_val_loss_loses_to_finite_candidate(monkeypatch):
    """A candidate that NaNs/Infs out (every batch non-finite →
    evaluate_model returns +inf for that seed) must lose to a finite
    candidate. Same clamp behavior as the validator-side
    `max(0, baseline - val_loss)`."""
    state = {"current": None}

    def fake_load(self, path: Path):
        state["current"] = path.name
        return _TinyModel()

    def fake_evaluate_model(*, model, **kwargs):
        path_name = state["current"]
        seed_call = state.setdefault(f"call_{path_name}", 0)
        state[f"call_{path_name}"] = seed_call + 1
        if path_name == "diverged":
            return {"val_loss": float("inf")}
        return {"val_loss": 4.0}

    monkeypatch.setattr(
        "connito.miner.best_ckpt_selector.evaluate_model", fake_evaluate_model
    )
    monkeypatch.setattr(
        "connito.miner.best_ckpt_selector.get_dataloader",
        lambda **kwargs: iter([]),
    )
    monkeypatch.setattr(
        BestCheckpointSelector, "_load_into_base", fake_load, raising=True
    )

    selector = BestCheckpointSelector(
        config=_make_config(),
        tokenizer=MagicMock(),
        base_model=_TinyModel(),
        device=torch.device("cpu"),
        seeds=[1, 2],
        max_batches=5,
    )
    winner, report = selector.select([Path("diverged"), Path("finite")])

    assert winner == Path("finite")
    diverged_result = next(r for r in report["results"] if r["path"] == "diverged")
    assert math.isinf(diverged_result["mean_val_loss"])


# ----------------------------------------------------------------------
# discover_candidate_checkpoints helper
# ----------------------------------------------------------------------


def test_discover_candidates_returns_empty_for_missing_dir(tmp_path):
    """A miner that hasn't trained anything yet — no checkpoint dir —
    must get an empty list back so the commit worker falls back to the
    legacy `latest checkpoint` path instead of crashing on missing files.
    """
    nope = tmp_path / "nope"
    candidates = discover_candidate_checkpoints(
        checkpoint_dir=nope, candidate_count=3,
    )
    assert candidates == []


def test_discover_candidates_picks_most_recent(tmp_path):
    """``build_local_checkpoints`` orders by (globalver, inner_opt)
    descending, so the most recent training checkpoints win even when
    older directories are present."""
    # Trainer naming convention: `globalver_<N>_inneropt_<M>/`.
    (tmp_path / "globalver_1_inneropt_100").mkdir()
    (tmp_path / "globalver_1_inneropt_200").mkdir()
    (tmp_path / "globalver_2_inneropt_50").mkdir()
    (tmp_path / "globalver_2_inneropt_150").mkdir()

    candidates = discover_candidate_checkpoints(
        checkpoint_dir=tmp_path, candidate_count=2, include_ema=False,
    )
    assert len(candidates) == 2
    names = [p.name for p in candidates]
    # Most recent two by (globalver, inner_opt) ordering.
    assert names == ["globalver_2_inneropt_150", "globalver_2_inneropt_50"]


def test_discover_candidates_includes_ema_when_present(tmp_path):
    """EMA snapshot is prepended to the candidate list when present
    and ``include_ema=True``."""
    (tmp_path / "globalver_1_inneropt_100").mkdir()
    ema = tmp_path / "ema.safetensors"
    ema.write_bytes(b"x")  # presence-only — selector loads its bytes

    candidates = discover_candidate_checkpoints(
        checkpoint_dir=tmp_path, candidate_count=3, include_ema=True,
    )
    # EMA listed first so it gets evaluated before training checkpoints
    # when iteration order matters for tie-breaking.
    assert candidates[0] == ema
    assert any(p.name == "globalver_1_inneropt_100" for p in candidates)


def test_discover_candidates_skips_ema_when_disabled(tmp_path):
    """``include_ema=False`` must not pull the EMA snapshot in even if
    it exists on disk — operators who explicitly disable Unit 13
    integration should see a clean list of training checkpoints only."""
    (tmp_path / "globalver_1_inneropt_100").mkdir()
    (tmp_path / "ema.safetensors").write_bytes(b"x")

    candidates = discover_candidate_checkpoints(
        checkpoint_dir=tmp_path, candidate_count=3, include_ema=False,
    )
    assert all(p.name != "ema.safetensors" for p in candidates)


def test_discover_candidates_skips_missing_ema(tmp_path):
    """``include_ema=True`` with no ema.safetensors must NOT crash —
    operators running Unit 15 without Unit 13 see only training
    checkpoints, no error."""
    (tmp_path / "globalver_1_inneropt_100").mkdir()

    candidates = discover_candidate_checkpoints(
        checkpoint_dir=tmp_path, candidate_count=3, include_ema=True,
    )
    assert all(p.name != "ema.safetensors" for p in candidates)
    assert len(candidates) == 1


def test_discover_candidates_caps_at_candidate_count(tmp_path):
    """``candidate_count`` is the maximum number of training
    checkpoints to score. Older checkpoints beyond the cap are
    dropped so the per-commit selection cost stays bounded."""
    for i in range(1, 6):
        (tmp_path / f"globalver_1_inneropt_{i*100}").mkdir()

    candidates = discover_candidate_checkpoints(
        checkpoint_dir=tmp_path, candidate_count=2, include_ema=False,
    )
    assert len(candidates) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
