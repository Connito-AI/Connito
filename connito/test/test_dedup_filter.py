"""Unit tests for the dedup (duplicate-submission) shadow filter.

Pure-function tests for `connito.validator.dedup`, plus a light
integration test of the background worker's shadow pass using the
established sys.modules stubbing + real-`Round` construction patterns
(see test_persist_bg_eval_scores.py / test_validate_miner_submission.py).
"""

from __future__ import annotations

import asyncio
import sys
import threading
import types
from pathlib import Path

import pytest
import torch


# ---------------------------------------------------------------------------
# Heavy-module stubs (same pattern as test_validate_miner_submission.py) so
# importing the worker doesn't require the datasets/pandas chain.
# ---------------------------------------------------------------------------
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

from connito.validator.dedup import (  # noqa: E402
    average_state_dicts,
    delta_cosine,
    find_submission_path,
    recover_val_loss,
    select_pairs,
    shadow_report,
)


# ---------------------------------------------------------------------------
# average_state_dicts
# ---------------------------------------------------------------------------
def test_average_state_dicts_mean_and_asymmetric_keys() -> None:
    sd_a = {"w": torch.ones(4), "a_only": torch.zeros(2)}
    sd_b = {"w": torch.full((4,), 3.0), "b_only": torch.zeros(3)}
    merged, asymmetric = average_state_dicts(sd_a, sd_b)
    assert set(merged) == {"w"}
    assert torch.equal(merged["w"], torch.full((4,), 2.0))
    assert asymmetric == 2


def test_average_state_dicts_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        average_state_dicts({"w": torch.ones(4)}, {"w": torch.ones(5)})


def test_average_state_dicts_preserves_dtype() -> None:
    sd_a = {"w": torch.ones(4, dtype=torch.float16)}
    sd_b = {"w": torch.full((4,), 2.0, dtype=torch.float16)}
    merged, _ = average_state_dicts(sd_a, sd_b)
    assert merged["w"].dtype == torch.float16


# ---------------------------------------------------------------------------
# delta_cosine
# ---------------------------------------------------------------------------
def _base() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {f"layer{i}": torch.randn(16, generator=g) for i in range(3)}


def test_delta_cosine_exact_copy_is_one() -> None:
    base = _base()
    g = torch.Generator().manual_seed(1)
    sub = {k: v + torch.randn(v.shape, generator=g) for k, v in base.items()}
    cos, n_keys = delta_cosine(sub, {k: v.clone() for k, v in sub.items()}, base)
    assert n_keys == 3
    assert cos == pytest.approx(1.0, abs=1e-6)


def test_delta_cosine_noisy_copy_stays_high() -> None:
    base = _base()
    g = torch.Generator().manual_seed(2)
    delta = {k: torch.randn(v.shape, generator=g) for k, v in base.items()}
    sub_a = {k: base[k] + delta[k] for k in base}
    # Copy + 1% noise: the copier's evasion move.
    sub_b = {
        k: sub_a[k] + 0.01 * torch.randn(base[k].shape, generator=g) for k in base
    }
    cos, _ = delta_cosine(sub_a, sub_b, base)
    assert cos > 0.99


def test_delta_cosine_independent_deltas_low() -> None:
    base = {f"layer{i}": torch.zeros(512) for i in range(3)}
    g = torch.Generator().manual_seed(3)
    sub_a = {k: torch.randn(v.shape, generator=g) for k, v in base.items()}
    sub_b = {k: torch.randn(v.shape, generator=g) for k, v in base.items()}
    cos, _ = delta_cosine(sub_a, sub_b, base)
    assert abs(cos) < 0.2  # independent gaussian deltas decorrelate


def test_delta_cosine_zero_delta_or_no_overlap() -> None:
    base = _base()
    same = {k: v.clone() for k, v in base.items()}
    cos, n_keys = delta_cosine(same, same, base)
    assert (cos, n_keys) == (0.0, 3)  # zero-norm deltas -> defined 0.0
    cos, n_keys = delta_cosine({"x": torch.ones(2)}, {"y": torch.ones(2)}, base)
    assert (cos, n_keys) == (0.0, 0)


# ---------------------------------------------------------------------------
# select_pairs
# ---------------------------------------------------------------------------
def test_select_pairs_positive_only_rank_order_and_cap() -> None:
    scores = {1: 0.5, 2: 0.9, 3: 0.0, 4: -1.0, 5: float("inf"), 6: 0.7}
    pairs = select_pairs(scores, top_k=3, max_pairs=10)
    # Ranked positives: 2 (0.9), 6 (0.7), 1 (0.5); uid 3/4/5 excluded.
    assert pairs == [(2, 6), (1, 2), (1, 6)]
    assert select_pairs(scores, top_k=3, max_pairs=2) == [(2, 6), (1, 2)]


def test_select_pairs_exclusion_and_degenerate() -> None:
    scores = {1: 0.5, 2: 0.9}
    assert select_pairs(scores, top_k=5, max_pairs=10) == [(1, 2)]
    assert select_pairs(scores, top_k=5, max_pairs=10, exclude={frozenset((1, 2))}) == []
    assert select_pairs({1: 0.5}, top_k=5, max_pairs=10) == []
    assert select_pairs({}, top_k=5, max_pairs=10) == []


# ---------------------------------------------------------------------------
# shadow_report + recover_val_loss
# ---------------------------------------------------------------------------
def test_shadow_report_duplicate_pair_flags() -> None:
    # Near-copy: merged loss == both sides.
    r = shadow_report(loss_a=3.0, loss_b=3.0, loss_avg=3.0, baseline=4.0, cosine=1.0)
    assert r["merge_penalty"] == pytest.approx(0.0)
    # "not strictly better" flags at every tau > 0 and (boundary) tau=0.
    assert all(r["would_flag_not_better"].values())
    assert all(r["would_flag_band"].values())


def test_shadow_report_diverse_pair_helping_merge() -> None:
    # Same-basin honest pair: merged strictly better than both.
    r = shadow_report(loss_a=3.0, loss_b=3.1, loss_avg=2.8, baseline=4.0, cosine=0.1)
    assert r["merge_penalty"] == pytest.approx(-0.2)
    assert not any(r["would_flag_not_better"].values())
    assert not any(r["would_flag_band"].values())


def test_shadow_report_loss_barrier_pair() -> None:
    # Distinct pair with a loss barrier: merged clearly WORSE than both.
    r = shadow_report(loss_a=3.0, loss_b=3.1, loss_avg=3.6, baseline=4.0, cosine=0.1)
    assert r["merge_penalty"] == pytest.approx(0.6)
    # One-sided predicate flags it (merge not better)...
    assert all(r["would_flag_not_better"].values())
    # ...but the band predicate correctly does not (|0.6| > every tau).
    assert not any(r["would_flag_band"].values())


def test_recover_val_loss_roundtrip() -> None:
    baseline = 4.2
    for val_loss in (2.5, 3.9, 4.19):
        score = (baseline - val_loss) ** 1.2
        assert recover_val_loss(score, baseline) == pytest.approx(val_loss, abs=1e-9)


# ---------------------------------------------------------------------------
# find_submission_path
# ---------------------------------------------------------------------------
def test_find_submission_path_matches_hotkey_and_block_window(tmp_path: Path) -> None:
    (tmp_path / "hotkey_HK1_block_100.safetensors").touch()
    (tmp_path / "hotkey_HK1_block_999.safetensors").touch()
    (tmp_path / "hotkey_HK2_block_150.safetensors").touch()
    (tmp_path / ".tmp_hotkey_HK1_block_120.safetensors").touch()

    found = find_submission_path(tmp_path, "HK1", (50, 200))
    assert found is not None and found.name == "hotkey_HK1_block_100.safetensors"
    # Blocks 100 and 999 both fall outside (500, 600) — no match.
    assert find_submission_path(tmp_path, "HK1", (500, 600)) is None
    # Unknown hotkey — no match even without a window.
    assert find_submission_path(tmp_path, "HK3", None) is None


# ---------------------------------------------------------------------------
# Worker shadow-pass integration (no GPU, no real eval)
# ---------------------------------------------------------------------------
from types import SimpleNamespace  # noqa: E402

from connito.validator.background_eval_worker import BackgroundEvalWorker  # noqa: E402
from connito.validator.round import Round, RoundRef  # noqa: E402


def _make_worker(tmp_path: Path, round_obj: Round, *, mode: str = "shadow") -> BackgroundEvalWorker:
    config = SimpleNamespace(
        evaluation=SimpleNamespace(
            dedup_filter_mode=mode,
            dedup_top_k=5,
            dedup_max_pairs=10,
            per_miner_eval_timeout_sec=30,
            top_k_miners_to_reward=3,
        ),
        ckpt=SimpleNamespace(miner_submission_path=tmp_path),
    )
    ref = RoundRef()
    ref.swap(round_obj)
    worker = BackgroundEvalWorker(
        config=config,
        round_ref=ref,
        device=torch.device("cpu"),
        tokenizer=None,
        merge_phase_active=threading.Event(),
        eval_window_active=threading.Event(),
        gpu_eval_lock=threading.Lock(),
        expert_group_assignment={},
    )
    worker.eval_window_active.set()
    # Pretend the round snapshot was loaded.
    worker._eval_base_model = torch.nn.Linear(2, 2)
    worker._loaded_round_id = round_obj.round_id
    worker._loaded_baseline_loss = 4.0
    worker._cached_batches = []
    worker._reset_dedup_state_if_new_round(round_obj)
    return worker


def _make_round(tmp_path: Path, scores: dict[int, float], *, round_id: int = 7) -> Round:
    round_obj = Round(
        round_id=round_id,
        seed="test-seed",
        validator_miner_assignment={},
        foreground_uids=tuple(scores.keys()),
        background_uids=(),
        uid_to_hotkey={uid: f"HK{uid}" for uid in scores},
        model_snapshot_cpu={"w": torch.zeros(4)},
        journal_path=None,
        score_aggregator=None,
        score_path=None,
    )
    for uid, s in scores.items():
        round_obj.claim_for_eval(uid)
        round_obj.mark_scored(uid, s)
    return round_obj


def _write_submission(tmp_path: Path, hotkey: str, block: int, sd: dict) -> Path:
    import safetensors.torch as st

    path = tmp_path / f"hotkey_{hotkey}_block_{block}.safetensors"
    st.save_file(sd, str(path))
    return path


def test_shadow_pass_evaluates_pairs_and_never_touches_scoring(tmp_path: Path) -> None:
    round_obj = _make_round(tmp_path, {1: 0.9, 2: 0.5})
    base = round_obj.model_snapshot_cpu
    _write_submission(tmp_path, "HK1", 10, {"w": base["w"] + torch.ones(4)})
    _write_submission(tmp_path, "HK2", 11, {"w": base["w"] + torch.full((4,), 2.0)})

    worker = _make_worker(tmp_path, round_obj)
    # Canned merged-model eval so no GPU / model machinery is needed.
    import connito.shared.evaluate as ev

    ev.evaluate_model = lambda *a, **k: {"val_loss": 3.5}

    scored_before, failed_before = round_obj.processed_uids_snapshot()
    asyncio.run(worker._maybe_run_dedup_shadow(round_obj))

    assert worker._dedup_pairs_done == {frozenset((1, 2))}
    assert worker._dedup_budget_used == 1
    assert worker._dedup_pairs_skipped == 0
    # Read-only wrt scoring: sets and scores untouched.
    scored_after, failed_after = round_obj.processed_uids_snapshot()
    assert (scored_after, failed_after) == (scored_before, failed_before)
    assert round_obj.scores_snapshot() == {1: 0.9, 2: 0.5}
    # Second tick: pair already done, nothing new.
    asyncio.run(worker._maybe_run_dedup_shadow(round_obj))
    assert worker._dedup_budget_used == 1


def test_shadow_pass_respects_mode_and_gates(tmp_path: Path) -> None:
    round_obj = _make_round(tmp_path, {1: 0.9, 2: 0.5})
    worker = _make_worker(tmp_path, round_obj, mode="off")
    asyncio.run(worker._maybe_run_dedup_shadow(round_obj))
    assert worker._dedup_budget_used == 0

    worker = _make_worker(tmp_path, round_obj)
    worker.merge_phase_active.set()
    asyncio.run(worker._maybe_run_dedup_shadow(round_obj))
    assert worker._dedup_budget_used == 0  # merge gate wins

    worker = _make_worker(tmp_path, round_obj)
    worker.eval_window_active.clear()
    asyncio.run(worker._maybe_run_dedup_shadow(round_obj))
    assert worker._dedup_budget_used == 0  # window gate wins


def test_shadow_pass_single_positive_miner_is_noop(tmp_path: Path) -> None:
    round_obj = _make_round(tmp_path, {1: 0.9, 2: 0.0})
    worker = _make_worker(tmp_path, round_obj)
    asyncio.run(worker._maybe_run_dedup_shadow(round_obj))
    assert worker._dedup_budget_used == 0


def test_shadow_pass_missing_file_skips_pair(tmp_path: Path) -> None:
    round_obj = _make_round(tmp_path, {1: 0.9, 2: 0.5})
    # No submission files written at all.
    worker = _make_worker(tmp_path, round_obj)
    asyncio.run(worker._maybe_run_dedup_shadow(round_obj))
    assert worker._dedup_budget_used == 1
    assert worker._dedup_pairs_skipped == 1


def test_dedup_state_resets_on_new_round_only(tmp_path: Path) -> None:
    round_a = _make_round(tmp_path, {1: 0.9, 2: 0.5})
    worker = _make_worker(tmp_path, round_a)
    worker._dedup_pairs_done.add(frozenset((1, 2)))
    worker._dedup_budget_used = 1
    # Same round id (recycler path): state survives.
    worker._reset_dedup_state_if_new_round(round_a)
    assert worker._dedup_budget_used == 1
    # New round id: state resets.
    round_b = _make_round(tmp_path, {3: 0.4, 4: 0.3}, round_id=8)
    worker._reset_dedup_state_if_new_round(round_b)
    assert worker._dedup_budget_used == 0
    assert worker._dedup_pairs_done == set()


def test_retention_top_k_widens_when_dedup_active() -> None:
    from connito.validator.evaluator import retention_top_k

    cfg_off = SimpleNamespace(evaluation=SimpleNamespace(
        top_k_miners_to_reward=3, dedup_filter_mode="off", dedup_top_k=5))
    cfg_on = SimpleNamespace(evaluation=SimpleNamespace(
        top_k_miners_to_reward=3, dedup_filter_mode="shadow", dedup_top_k=5))
    assert retention_top_k(cfg_off) == 3
    assert retention_top_k(cfg_on) == 5
