"""Tests for weight-delta sketches and near-duplicate zeroing.

The attack: one operator submits N slightly-perturbed copies of the same
model from N UIDs. Each copy's val_loss differs in the last decimals, so
the exact-equality tie rule in `finalize_round_scores` never fires and
the copies sweep the top-3 rank slots. The defense compares weight-delta
sketches (submission minus base model, sampled at seed-derived
coordinates): copies stay near-parallel to their source no matter how
the perturbation lands, while independently trained deltas do not.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from connito.validator.aggregator import MinerScoreAggregator
from connito.validator.evaluator import finalize_round_scores
from connito.validator.round import Round
from connito.validator.submission_fingerprint import (
    find_near_duplicate_clusters,
    sketch_state_dict,
)


def _base_sd(gen: torch.Generator) -> dict[str, torch.Tensor]:
    return {
        "experts.0.w1": torch.randn(64, 32, generator=gen),
        "experts.0.w2": torch.randn(32, 64, generator=gen),
        "experts.0.bias": torch.randn(64, generator=gen),
    }


def _trained(base: dict, gen: torch.Generator, scale: float = 0.05) -> dict:
    return {k: v + scale * torch.randn(v.shape, generator=gen) for k, v in base.items()}


def _perturbed(sd: dict, gen: torch.Generator, scale: float = 0.002) -> dict:
    return {k: v + scale * torch.randn(v.shape, generator=gen) for k, v in sd.items()}


@pytest.fixture()
def gen() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(1234)
    return g


# ---------------------------------------------------------------------------
# Sketch properties
# ---------------------------------------------------------------------------

def test_sketch_is_deterministic(gen):
    base = _base_sd(gen)
    sd = _trained(base, gen)
    a = sketch_state_dict(sd, base, seed="round-seed", budget=1024)
    b = sketch_state_dict(sd, base, seed="round-seed", budget=1024)
    assert torch.equal(a, b)


def test_sketch_no_common_keys_returns_none(gen):
    base = _base_sd(gen)
    assert sketch_state_dict({"other.key": torch.randn(4)}, base, seed="s") is None


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def test_copy_perturb_stays_parallel_independent_does_not(gen):
    base = _base_sd(gen)
    honest_a = _trained(base, gen)
    honest_b = _trained(base, gen)          # independent training
    copier = _perturbed(honest_a, gen)      # copy of A + small noise

    fp_a = sketch_state_dict(honest_a, base, seed="round-seed", budget=2048)
    fp_b = sketch_state_dict(honest_b, base, seed="round-seed", budget=2048)
    fp_c = sketch_state_dict(copier, base, seed="round-seed", budget=2048)

    assert _cos(fp_a, fp_c) > 0.99          # perturbed copy: near-parallel
    assert abs(_cos(fp_a, fp_b)) < 0.5      # independent runs: far apart


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def test_clusters_flag_copy_group_only(gen):
    base = _base_sd(gen)
    honest_a = _trained(base, gen)
    honest_b = _trained(base, gen)
    copier_1 = _perturbed(honest_a, gen)
    copier_2 = _perturbed(honest_a, gen)

    fps = {
        10: sketch_state_dict(honest_a, base, seed="s", budget=2048),
        20: sketch_state_dict(honest_b, base, seed="s", budget=2048),
        30: sketch_state_dict(copier_1, base, seed="s", budget=2048),
        40: sketch_state_dict(copier_2, base, seed="s", budget=2048),
    }
    clusters = find_near_duplicate_clusters(fps, cos_threshold=0.95)
    assert clusters == [{10, 30, 40}]


def test_zero_delta_submissions_are_ignored(gen):
    base = _base_sd(gen)
    fps = {
        1: sketch_state_dict(base, base, seed="s", budget=2048),  # base copy: ~0 delta
        2: sketch_state_dict(base, base, seed="s", budget=2048),
    }
    assert find_near_duplicate_clusters(fps, cos_threshold=0.95) == []


def test_mismatched_sketch_lengths_never_compare(gen):
    fps = {1: torch.randn(128), 2: torch.randn(256)}
    assert find_near_duplicate_clusters(fps, cos_threshold=0.0) == []


# ---------------------------------------------------------------------------
# finalize_round_scores integration
# ---------------------------------------------------------------------------

def _make_round(uid_to_hotkey: dict[int, str]) -> Round:
    return Round(
        round_id=5000,
        seed="test-seed",
        validator_miner_assignment={},
        foreground_uids=tuple(uid_to_hotkey.keys()),
        background_uids=(),
        uid_to_hotkey=dict(uid_to_hotkey),
        model_snapshot_cpu={},
    )


def _make_config(enabled: bool = True, threshold: float = 0.95) -> SimpleNamespace:
    return SimpleNamespace(evaluation=SimpleNamespace(
        near_duplicate_detection_enabled=enabled,
        near_duplicate_cos_threshold=threshold,
    ))


def _score_with_fingerprints(gen):
    """3 miners: uid 1 honest (best score), uids 2 & 3 a copy pair whose
    scores straddle uid 1 — without detection they take ranks 1 and 3."""
    base = _base_sd(gen)
    honest = _trained(base, gen)
    source = _trained(base, gen)
    copy_of_source = _perturbed(source, gen)

    round_obj = _make_round({1: "hk1", 2: "hk2", 3: "hk3"})
    kw = dict(seed="test-seed", budget=2048)
    round_obj.mark_scored(1, score=0.5, fingerprint=sketch_state_dict(honest, base, **kw))
    round_obj.mark_scored(2, score=0.6, fingerprint=sketch_state_dict(source, base, **kw))
    round_obj.mark_scored(3, score=0.4, fingerprint=sketch_state_dict(copy_of_source, base, **kw))
    return round_obj


def test_finalize_zeroes_near_duplicate_cluster(gen):
    round_obj = _score_with_fingerprints(gen)
    agg = MinerScoreAggregator(max_points=8, max_history_points=64)

    written = finalize_round_scores(
        round_obj=round_obj, score_aggregator=agg, config=_make_config(),
    )

    assert written[1] == 2.25   # honest miner promoted to rank 1
    assert written[2] == 0.0    # copy pair zeroed
    assert written[3] == 0.0


def test_finalize_detection_disabled_keeps_legacy_ranking(gen):
    round_obj = _score_with_fingerprints(gen)
    agg = MinerScoreAggregator(max_points=8, max_history_points=64)

    written = finalize_round_scores(
        round_obj=round_obj, score_aggregator=agg, config=_make_config(enabled=False),
    )

    assert written[2] == 2.25   # copies rank normally when disabled
    assert written[1] == 1.5
    assert written[3] == 1.0


def test_finalize_without_config_skips_detection(gen):
    """The journal-recovery path passes config=None and must not blow up
    on a stub round object that has no `fingerprints` attribute."""
    round_obj = _score_with_fingerprints(gen)
    agg = MinerScoreAggregator(max_points=8, max_history_points=64)

    written = finalize_round_scores(round_obj=round_obj, score_aggregator=agg)

    assert written[2] == 2.25


def test_finalize_handles_missing_fingerprints(gen):
    """UIDs without a sketch (restart mid-round, sketch failure) still
    rank; detection just cannot see them."""
    round_obj = _make_round({1: "hk1", 2: "hk2"})
    round_obj.mark_scored(1, score=0.5)          # no fingerprint
    round_obj.mark_scored(2, score=0.6)          # no fingerprint
    agg = MinerScoreAggregator(max_points=8, max_history_points=64)

    written = finalize_round_scores(
        round_obj=round_obj, score_aggregator=agg, config=_make_config(),
    )

    assert written[2] == 2.25
    assert written[1] == 1.5
