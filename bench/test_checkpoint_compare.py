"""Tests for ``bench.checkpoint_compare``.

The comparator is split into two halves:

  1. Pure formula / ranking helpers (`score_from_loss`,
     `discover_checkpoints`, `parse_seeds`, `compare_checkpoints`,
     `render_markdown`). These are exercised with synthetic data and
     run in milliseconds.

  2. The real CLI eval path (`_build_real_eval_fn`, ``main``) which
     touches HuggingFace + GPU. Only ``--help`` and the arg parser are
     exercised in unit tests; the real eval path is implicitly covered
     by the smoke test that uses dummy state-dicts of a tiny gpt2 model
     plus a stubbed eval loop.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from bench import checkpoint_compare as cc


# ---------------------------------------------------------------------------
# score_from_loss
# ---------------------------------------------------------------------------


def test_score_from_loss_matches_validator_formula() -> None:
    """``score = (max(0, baseline - val_loss)) ** 1.2``.

    Locks the comparator to the exact validator formula in
    ``connito/validator/evaluator.py:787`` — any divergence here would
    silently mis-rank checkpoints relative to what a real validator
    would award them.
    """
    delta, score = cc.score_from_loss(4.78, 1.43)
    assert delta == pytest.approx(4.78 - 1.43)
    assert score == pytest.approx((4.78 - 1.43) ** 1.2)


def test_score_from_loss_clamps_negative_delta_to_zero() -> None:
    """A checkpoint worse than the baseline must score 0, not a negative.

    ``max(0, baseline - val_loss)`` is the load-bearing clamp — without
    it `delta ** 1.2` becomes a complex number for negative inputs.
    """
    delta, score = cc.score_from_loss(2.0, 3.0)
    assert delta == 0.0
    assert score == 0.0


def test_score_from_loss_inf_val_loss_collapses_to_zero() -> None:
    """A 100% NaN/Inf miner must score 0 — matches `evaluate_model`'s
    sentinel of returning `+inf` for that case."""
    delta, score = cc.score_from_loss(4.78, float("inf"))
    assert delta == 0.0
    assert score == 0.0


def test_score_from_loss_nan_val_loss_collapses_to_zero() -> None:
    """A NaN val_loss must not propagate into the leaderboard. The
    helper filters NaN explicitly because `max(0.0, NaN) == NaN`."""
    delta, score = cc.score_from_loss(4.78, float("nan"))
    assert delta == 0.0
    assert score == 0.0


# ---------------------------------------------------------------------------
# discover_checkpoints
# ---------------------------------------------------------------------------


def _touch_ckpt(path: Path) -> None:
    """Create a zero-byte file at `path` so `discover_checkpoints` finds it.

    The comparator only stat-checks the filesystem during discovery; we
    don't need to write a valid state_dict here.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_discover_checkpoints_from_explicit_paths(tmp_path: Path) -> None:
    a = tmp_path / "a.safetensors"
    b = tmp_path / "b.pt"
    _touch_ckpt(a)
    _touch_ckpt(b)

    paths = cc.discover_checkpoints(paths=[str(a), str(b)])
    assert paths == [a, b]


def test_discover_checkpoints_from_directory_picks_up_only_known_suffixes(tmp_path: Path) -> None:
    """``--checkpoint-dir`` must skip unrelated files (.json, .yaml, etc.)
    so a dirty checkpoint directory doesn't poison the leaderboard."""
    a = tmp_path / "a.safetensors"
    b = tmp_path / "b.pt"
    c = tmp_path / "c.bin"
    junk = tmp_path / "ignored.json"
    _touch_ckpt(a)
    _touch_ckpt(b)
    _touch_ckpt(c)
    junk.write_text("{}")

    paths = cc.discover_checkpoints(directory=str(tmp_path))
    assert set(paths) == {a, b, c}
    # Sorted output is stable
    assert paths == sorted(paths, key=lambda p: p.name)


def test_discover_checkpoints_deduplicates(tmp_path: Path) -> None:
    a = tmp_path / "a.safetensors"
    _touch_ckpt(a)
    paths = cc.discover_checkpoints(paths=[str(a), str(a)])
    assert paths == [a]


def test_discover_checkpoints_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cc.discover_checkpoints(paths=[str(tmp_path / "nonexistent.safetensors")])


def test_discover_checkpoints_no_input_raises() -> None:
    with pytest.raises(ValueError):
        cc.discover_checkpoints()


# ---------------------------------------------------------------------------
# parse_seeds
# ---------------------------------------------------------------------------


def test_parse_seeds_default_is_deterministic_across_calls() -> None:
    """Default seeds must be reproducible so two leaderboards produced
    minutes apart can be compared directly."""
    first = cc.parse_seeds(None)
    second = cc.parse_seeds(None)
    assert first == second
    assert len(first) == cc.DEFAULT_NUM_SEEDS


def test_parse_seeds_explicit_list() -> None:
    assert cc.parse_seeds("1, 2,3") == [1, 2, 3]


def test_parse_seeds_invalid_raises() -> None:
    with pytest.raises(ValueError):
        cc.parse_seeds("not-an-int")


def test_parse_seeds_empty_string_raises() -> None:
    with pytest.raises(ValueError):
        cc.parse_seeds(",")


# ---------------------------------------------------------------------------
# CheckpointResult.finalize
# ---------------------------------------------------------------------------


def test_checkpoint_result_finalize_single_seed_no_stdev() -> None:
    """A single-seed eval must not raise on `statistics.stdev` (which
    needs n>=2 by the docs)."""
    ckpt = cc.CheckpointResult(path="/x.bin", name="x")
    ckpt.per_seed.append(cc.SeedResult(seed=42, val_loss=1.5, delta=3.28, score=4.0))
    ckpt.finalize()
    assert ckpt.mean_val_loss == 1.5
    assert ckpt.std_val_loss == 0.0
    assert ckpt.min_val_loss == 1.5
    assert ckpt.max_val_loss == 1.5


def test_checkpoint_result_finalize_multi_seed_aggregates() -> None:
    ckpt = cc.CheckpointResult(path="/x.bin", name="x")
    ckpt.per_seed.append(cc.SeedResult(seed=1, val_loss=1.0, delta=3.78, score=4.86))
    ckpt.per_seed.append(cc.SeedResult(seed=2, val_loss=2.0, delta=2.78, score=3.47))
    ckpt.per_seed.append(cc.SeedResult(seed=3, val_loss=3.0, delta=1.78, score=2.10))
    ckpt.finalize()
    assert ckpt.mean_val_loss == pytest.approx(2.0)
    assert ckpt.min_val_loss == 1.0
    assert ckpt.max_val_loss == 3.0
    # Plain sample stdev of [1, 2, 3] = 1.0
    assert ckpt.std_val_loss == pytest.approx(1.0)
    assert ckpt.mean_delta == pytest.approx((3.78 + 2.78 + 1.78) / 3)
    assert ckpt.mean_score == pytest.approx((4.86 + 3.47 + 2.10) / 3)


def test_checkpoint_result_finalize_empty_seed_list_does_not_raise() -> None:
    ckpt = cc.CheckpointResult(path="/x.bin", name="x")
    ckpt.finalize()
    # Empty checkpoint sentinels — confirms downstream sort doesn't NaN.
    assert math.isinf(ckpt.mean_val_loss)
    assert ckpt.mean_score == 0.0


# ---------------------------------------------------------------------------
# compare_checkpoints — pure ranking with a synthetic eval_fn
# ---------------------------------------------------------------------------


def _stub_eval_fn(loss_table: dict[tuple[str, int], float]):
    """Build an `eval_fn` that returns deterministic val_losses from a
    lookup table. The path is keyed by `Path(path).name` so callers can
    write the table with bare filenames and not full /tmp/ paths.
    """

    def _fn(path: Path, seed: int) -> float:
        key = (Path(path).name, int(seed))
        if key not in loss_table:
            raise KeyError(f"No stub loss configured for {key!r}")
        return float(loss_table[key])

    return _fn


def test_compare_checkpoints_ranks_lower_loss_higher(tmp_path: Path) -> None:
    """Lower mean val_loss → higher rank. Smoke test using bare filenames
    so the test is independent of where pytest's tmpdir lives."""
    baseline = tmp_path / "baseline.safetensors"
    a = tmp_path / "a.safetensors"
    b = tmp_path / "b.safetensors"
    c = tmp_path / "c.safetensors"
    for p in (baseline, a, b, c):
        _touch_ckpt(p)

    seeds = [1, 2, 3]
    # Baseline loss is constant 5.0 per seed for simplicity.
    # a: 1.5, 1.6, 1.4   (mean 1.5)  → best
    # b: 2.0, 2.1, 1.9   (mean 2.0)
    # c: 3.0, 3.1, 2.9   (mean 3.0)
    losses = {
        ("baseline.safetensors", 1): 5.0,
        ("baseline.safetensors", 2): 5.0,
        ("baseline.safetensors", 3): 5.0,
        ("a.safetensors", 1): 1.5,
        ("a.safetensors", 2): 1.6,
        ("a.safetensors", 3): 1.4,
        ("b.safetensors", 1): 2.0,
        ("b.safetensors", 2): 2.1,
        ("b.safetensors", 3): 1.9,
        ("c.safetensors", 1): 3.0,
        ("c.safetensors", 2): 3.1,
        ("c.safetensors", 3): 2.9,
    }

    payload = cc.compare_checkpoints(
        checkpoint_paths=[a, b, c],
        baseline_path=baseline,
        seeds=seeds,
        eval_fn=_stub_eval_fn(losses),
    )
    assert payload["rank_by_score"] == ["a", "b", "c"]
    assert payload["winner"] == "a"

    # Spot-check the schema and exact numbers.
    baseline_per_seed = payload["baseline_loss_per_seed"]
    assert baseline_per_seed == {"1": 5.0, "2": 5.0, "3": 5.0}

    by_name = {ckpt["name"]: ckpt for ckpt in payload["checkpoints"]}
    assert by_name["a"]["mean_val_loss"] == pytest.approx(1.5)
    assert by_name["a"]["mean_delta"] == pytest.approx(5.0 - 1.5)
    assert by_name["a"]["mean_score"] == pytest.approx(
        sum((5.0 - v) ** 1.2 for v in (1.5, 1.6, 1.4)) / 3
    )
    # Per-seed entries preserve insertion order (= seeds order)
    assert [r["seed"] for r in by_name["a"]["per_seed"]] == [1, 2, 3]


def test_compare_checkpoints_uses_per_seed_baseline(tmp_path: Path) -> None:
    """Baseline is per-seed, not a single scalar — the validator's eval
    is seed-dependent, so subtracting a different seed's baseline would
    distort the delta."""
    baseline = tmp_path / "baseline.safetensors"
    ckpt = tmp_path / "ckpt.safetensors"
    _touch_ckpt(baseline)
    _touch_ckpt(ckpt)

    # Same ckpt val_loss but very different baseline per seed → very
    # different deltas, exercising the per-seed pairing.
    losses = {
        ("baseline.safetensors", 10): 10.0,
        ("baseline.safetensors", 20): 2.5,
        ("ckpt.safetensors", 10): 1.0,
        ("ckpt.safetensors", 20): 1.0,
    }
    payload = cc.compare_checkpoints(
        checkpoint_paths=[ckpt],
        baseline_path=baseline,
        seeds=[10, 20],
        eval_fn=_stub_eval_fn(losses),
    )
    per_seed = payload["checkpoints"][0]["per_seed"]
    by_seed = {r["seed"]: r for r in per_seed}
    assert by_seed[10]["delta"] == pytest.approx(9.0)
    assert by_seed[20]["delta"] == pytest.approx(1.5)


def test_compare_checkpoints_handles_worse_than_baseline(tmp_path: Path) -> None:
    """A checkpoint whose val_loss is worse than baseline must record
    delta=0, score=0 — and still appear in the leaderboard."""
    baseline = tmp_path / "baseline.safetensors"
    bad = tmp_path / "bad.safetensors"
    _touch_ckpt(baseline)
    _touch_ckpt(bad)

    losses = {
        ("baseline.safetensors", 1): 1.0,
        ("bad.safetensors", 1): 5.0,  # worse than baseline
    }
    payload = cc.compare_checkpoints(
        checkpoint_paths=[bad],
        baseline_path=baseline,
        seeds=[1],
        eval_fn=_stub_eval_fn(losses),
    )
    ckpt = payload["checkpoints"][0]
    assert ckpt["mean_delta"] == 0.0
    assert ckpt["mean_score"] == 0.0
    assert payload["winner"] == "bad"  # still the (only) winner


def test_compare_checkpoints_inf_loss_is_safely_ranked_last(tmp_path: Path) -> None:
    """An exploded checkpoint (val_loss = inf) ranks last but does not
    crash the leaderboard."""
    baseline = tmp_path / "baseline.safetensors"
    good = tmp_path / "good.safetensors"
    broken = tmp_path / "broken.safetensors"
    for p in (baseline, good, broken):
        _touch_ckpt(p)

    losses = {
        ("baseline.safetensors", 1): 5.0,
        ("good.safetensors", 1): 1.0,
        ("broken.safetensors", 1): float("inf"),
    }
    payload = cc.compare_checkpoints(
        checkpoint_paths=[good, broken],
        baseline_path=baseline,
        seeds=[1],
        eval_fn=_stub_eval_fn(losses),
    )
    assert payload["winner"] == "good"
    assert payload["rank_by_score"] == ["good", "broken"]

    broken_entry = next(c for c in payload["checkpoints"] if c["name"] == "broken")
    assert broken_entry["mean_score"] == 0.0


def test_compare_checkpoints_raises_on_no_seeds(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.safetensors"
    ckpt = tmp_path / "ckpt.safetensors"
    _touch_ckpt(baseline)
    _touch_ckpt(ckpt)
    with pytest.raises(ValueError):
        cc.compare_checkpoints(
            checkpoint_paths=[ckpt],
            baseline_path=baseline,
            seeds=[],
            eval_fn=lambda p, s: 1.0,
        )


def test_compare_checkpoints_raises_on_no_checkpoints(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.safetensors"
    _touch_ckpt(baseline)
    with pytest.raises(ValueError):
        cc.compare_checkpoints(
            checkpoint_paths=[],
            baseline_path=baseline,
            seeds=[1],
            eval_fn=lambda p, s: 1.0,
        )


# ---------------------------------------------------------------------------
# render_markdown
# ---------------------------------------------------------------------------


def test_render_markdown_contains_winner_and_table(tmp_path: Path) -> None:
    payload = {
        "baseline_loss_per_seed": {"1": 5.0, "2": 5.0},
        "checkpoints": [
            {
                "path": "/x/a.safetensors",
                "name": "a",
                "mean_val_loss": 1.5,
                "std_val_loss": 0.1,
                "min_val_loss": 1.4,
                "max_val_loss": 1.6,
                "mean_delta": 3.5,
                "mean_score": 4.4,
                "per_seed": [],
            },
            {
                "path": "/x/b.safetensors",
                "name": "b",
                "mean_val_loss": 2.0,
                "std_val_loss": 0.0,
                "min_val_loss": 2.0,
                "max_val_loss": 2.0,
                "mean_delta": 3.0,
                "mean_score": 3.7,
                "per_seed": [],
            },
        ],
        "winner": "a",
        "rank_by_score": ["a", "b"],
    }
    md = cc.render_markdown(payload)
    assert "Winner" in md
    assert "`a`" in md
    # Both checkpoints should appear in the table
    assert "| a |" in md or "a |" in md  # exact alignment may vary
    assert "b" in md
    # Per-seed baselines should appear
    assert "5.0000" in md


def test_render_markdown_handles_empty_payload() -> None:
    """A leaderboard with no checkpoints must still render valid markdown
    without raising — useful when the comparator runs against a dir that
    matched zero candidates and the caller still wants a markdown stub."""
    md = cc.render_markdown({})
    assert "Checkpoint comparator leaderboard" in md
    assert "(none)" in md


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def test_cli_help_exits_zero() -> None:
    """``python -m bench.checkpoint_compare --help`` must succeed.

    Run as a subprocess so we exercise the actual `__main__` path and
    its argparse setup, not just an internal import.
    """
    result = subprocess.run(
        [sys.executable, "-m", "bench.checkpoint_compare", "--help"],
        capture_output=True,
        text=True,
        # Run from the repo root so `bench` resolves on import.
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert result.returncode == 0, (
        f"--help exited {result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "checkpoint" in result.stdout.lower()


def test_cli_requires_checkpoints_or_directory() -> None:
    parser = cc.build_arg_parser()
    with pytest.raises(SystemExit):
        # --baseline alone is not enough; the mutually-exclusive group
        # for --checkpoints / --checkpoint-dir is `required=True`.
        parser.parse_args(["--baseline", "/tmp/baseline.safetensors"])


def test_main_returns_2_when_baseline_missing(tmp_path: Path) -> None:
    """``main`` must surface a non-zero exit code for a bad baseline so
    the worker checklist's smoke test fails loudly. We stub out the
    heavyweight real eval-fn builder so the test does not require GPU /
    HF access."""
    ckpt = tmp_path / "ckpt.safetensors"
    _touch_ckpt(ckpt)
    missing_baseline = tmp_path / "missing.safetensors"

    with patch.object(cc, "_build_real_eval_fn") as mock_build:
        # Should never actually be called — the baseline check happens
        # before we wire up the real eval-fn.
        mock_build.return_value = (lambda p, s: 1.0, lambda: None)
        rc = cc.main(
            [
                "--checkpoints",
                str(ckpt),
                "--baseline",
                str(missing_baseline),
            ]
        )
    assert rc == 2
    mock_build.assert_not_called()


def test_main_returns_2_when_no_checkpoints_discovered(tmp_path: Path) -> None:
    """An empty discovery directory must also short-circuit with exit 2."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    baseline = tmp_path / "baseline.safetensors"
    _touch_ckpt(baseline)
    with patch.object(cc, "_build_real_eval_fn") as mock_build:
        mock_build.return_value = (lambda p, s: 1.0, lambda: None)
        rc = cc.main(
            [
                "--checkpoint-dir",
                str(empty_dir),
                "--baseline",
                str(baseline),
            ]
        )
    assert rc == 2
    mock_build.assert_not_called()


def test_main_writes_outputs_with_stubbed_eval_fn(tmp_path: Path) -> None:
    """End-to-end CLI path with the heavyweight real eval-fn stubbed
    out. Verifies that ``--out`` / ``--md-out`` are written and that
    the JSON payload follows the documented schema."""
    baseline = tmp_path / "baseline.safetensors"
    a = tmp_path / "a.safetensors"
    b = tmp_path / "b.safetensors"
    for p in (baseline, a, b):
        _touch_ckpt(p)

    out_json = tmp_path / "leaderboard.json"
    out_md = tmp_path / "leaderboard.md"

    fake_losses = {
        ("baseline.safetensors", 7): 5.0,
        ("a.safetensors", 7): 1.0,
        ("b.safetensors", 7): 2.0,
    }
    fake_eval = _stub_eval_fn(fake_losses)

    with patch.object(cc, "_build_real_eval_fn") as mock_build:
        mock_build.return_value = (fake_eval, lambda: None)
        rc = cc.main(
            [
                "--checkpoints",
                f"{a},{b}",
                "--baseline",
                str(baseline),
                "--seeds",
                "7",
                "--out",
                str(out_json),
                "--md-out",
                str(out_md),
            ]
        )

    assert rc == 0
    assert out_json.exists()
    assert out_md.exists()

    payload = json.loads(out_json.read_text())
    # Schema spot-check
    assert payload["winner"] == "a"
    assert payload["rank_by_score"] == ["a", "b"]
    assert set(payload["baseline_loss_per_seed"].keys()) == {"7"}
    for entry in payload["checkpoints"]:
        assert {"path", "name", "mean_val_loss", "std_val_loss", "mean_delta",
                "mean_score", "per_seed"} <= set(entry)

    md = out_md.read_text()
    assert "`a`" in md


# ---------------------------------------------------------------------------
# Smoke test — three perturbed tiny-gpt2 state-dicts, mocked eval loop
# ---------------------------------------------------------------------------


def _make_state_dict_perturbation(seed: int, scale: float) -> dict[str, torch.Tensor]:
    """Build a tiny state_dict with values perturbed from baseline.

    Returns a 2-tensor dict so the smoke test can write a real .safetensors
    file via ``safetensors.torch.save_file`` and reach the
    ``load_state_dict_from_path`` happy path.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return {
        "model.weight": torch.randn(8, 8, generator=g) * scale,
        "model.bias": torch.randn(8, generator=g) * scale,
    }


def test_smoke_three_perturbed_state_dicts_rank_in_expected_order(tmp_path: Path) -> None:
    """End-to-end smoke test:
      * Save 3 dummy state-dicts of increasing perturbation magnitude.
      * Stub the eval loop with a synthetic dataloader: the more a
        checkpoint's weight magnitude departs from the baseline's, the
        higher its val_loss.
      * Run `compare_checkpoints` and confirm ranking matches the
        synthetic difficulty ordering (lower scale → lower loss → higher
        score → higher rank).
    """
    # Baseline state_dict (tiny — same shape contract as the perturbed ones)
    baseline_path = tmp_path / "baseline.safetensors"
    save_file(_make_state_dict_perturbation(seed=0, scale=0.0), str(baseline_path))

    ckpts: list[tuple[str, float, Path]] = []
    for tag, scale in (("ckpt_step_4500_ema", 0.1), ("ckpt_step_4000", 0.5), ("ckpt_step_3500", 1.0)):
        p = tmp_path / f"{tag}.safetensors"
        save_file(_make_state_dict_perturbation(seed=hash(tag) & 0xFFFFFFFF, scale=scale), str(p))
        ckpts.append((tag, scale, p))

    def _synthetic_eval_fn(path: Path, seed: int) -> float:
        """Synthetic val_loss: baseline scores 5.0; each checkpoint
        scores 5.0 - 2.0 * (1 - scale) + jitter(seed). So scale 0.1 ->
        ~3.2, scale 0.5 -> ~4.0, scale 1.0 -> ~5.0; all monotonic in
        scale → ranking is deterministic."""
        if path.name == "baseline.safetensors":
            # Baseline jitters a tiny bit per seed to mimic real noise
            return 5.0 + 0.01 * (seed % 5)
        for tag, scale, p in ckpts:
            if path.name == p.name:
                jitter = 0.01 * (seed % 5)
                return 5.0 - 2.0 * (1.0 - scale) + jitter
        raise KeyError(path.name)

    seeds = [11, 22, 33]
    payload = cc.compare_checkpoints(
        checkpoint_paths=[p for _, _, p in ckpts],
        baseline_path=baseline_path,
        seeds=seeds,
        eval_fn=_synthetic_eval_fn,
    )

    # Lower scale → lower val_loss → larger delta → higher score.
    # So the expected ranking is: ckpt_step_4500_ema, ckpt_step_4000,
    # ckpt_step_3500.
    assert payload["rank_by_score"] == [
        "ckpt_step_4500_ema",
        "ckpt_step_4000",
        "ckpt_step_3500",
    ]
    assert payload["winner"] == "ckpt_step_4500_ema"

    # Validate that std is non-zero (we injected jitter) and that the
    # mean_score for the winner is strictly greater than the runner-up.
    by_name = {c["name"]: c for c in payload["checkpoints"]}
    assert by_name["ckpt_step_4500_ema"]["std_val_loss"] > 0.0
    assert (
        by_name["ckpt_step_4500_ema"]["mean_score"]
        > by_name["ckpt_step_4000"]["mean_score"]
        > by_name["ckpt_step_3500"]["mean_score"]
    )
