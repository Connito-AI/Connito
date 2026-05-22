"""Smoke tests for `bench.sweep_runner`.

These tests exercise the harness WITHOUT any real training:

* YAML loading + validation (`SweepConfig.from_yaml`).
* Grid expansion (`expand_grid`).
* Dotted-path overrides (`apply_override`).
* Multi-seed stub scoring + aggregation (`score_trial`, `aggregate_seed_results`).
* Leaderboard render — JSON + Markdown — sort order, top callout.
* Output dir layout: `sweep_config.json`, `trials/<id>.json`,
  `leaderboard.json`, `leaderboard.md`.
* CLI: `--dry-run`, `--report`, `--max-trials`.
* Graceful handling when `bench.local_scorer` is missing.
"""

from __future__ import annotations

import io
import json
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import yaml

# Make the worktree root importable as `bench` when pytest is invoked
# from elsewhere. The `conftest` mechanism would also work but this
# keeps the test file standalone-runnable via `python -m pytest`.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.sweep_runner import (  # noqa: E402
    GridAxis,
    LocalScorerCfg,
    MiniTrainCfg,
    SeedResult,
    SweepConfig,
    TrialResult,
    _load_base_config,
    aggregate_seed_results,
    apply_override,
    build_trial_config,
    build_trial_matrix,
    expand_grid,
    load_trial_results,
    main,
    prepare_output_dir,
    render_dry_run_matrix,
    render_leaderboard,
    run_sweep,
    score_from_loss,
    score_trial,
    stub_score_for_overrides,
    trial_id_for,
    write_leaderboard,
    write_trial_result,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_sweep_dict(output_dir: Path, **overrides) -> dict:
    base = {
        "sweep_name": "smoke_test",
        "output_dir": str(output_dir),
        "mini_train": {"steps": 1, "dry_run_stub_model": True},
        "local_scorer": {"max_batches": 1, "seeds": [42, 1337], "baseline_loss": 4.78},
        "grid": [
            {"param": "opt.lr", "values": [1e-5, 3e-5]},
        ],
    }
    base.update(overrides)
    return base


def _write_yaml(tmp_path: Path, body: dict, name: str = "sweep.yaml") -> Path:
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(body, f)
    return p


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSweepConfigSchema:
    def test_from_dict_minimal(self, tmp_path: Path) -> None:
        cfg = SweepConfig.from_dict(_minimal_sweep_dict(tmp_path / "out"))
        assert cfg.sweep_name == "smoke_test"
        assert cfg.output_dir == tmp_path / "out"
        assert cfg.mini_train.steps == 1
        assert cfg.mini_train.dry_run_stub_model is True
        assert cfg.local_scorer.seeds == [42, 1337]
        assert len(cfg.grid) == 1
        assert cfg.grid[0].param == "opt.lr"
        assert cfg.grid[0].values == [1e-5, 3e-5]

    def test_from_yaml_round_trip(self, tmp_path: Path) -> None:
        body = _minimal_sweep_dict(tmp_path / "out")
        yaml_path = _write_yaml(tmp_path, body)
        cfg = SweepConfig.from_yaml(yaml_path)
        assert cfg.sweep_name == "smoke_test"
        assert cfg.local_scorer.baseline_loss == 4.78

    def test_from_yaml_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            SweepConfig.from_yaml(tmp_path / "nope.yaml")

    def test_missing_sweep_name_raises(self, tmp_path: Path) -> None:
        body = _minimal_sweep_dict(tmp_path / "out")
        body.pop("sweep_name")
        with pytest.raises(ValueError, match="sweep_name"):
            SweepConfig.from_dict(body)

    def test_missing_output_dir_raises(self, tmp_path: Path) -> None:
        body = _minimal_sweep_dict(tmp_path / "out")
        body.pop("output_dir")
        with pytest.raises(ValueError, match="output_dir"):
            SweepConfig.from_dict(body)

    def test_empty_grid_raises(self, tmp_path: Path) -> None:
        body = _minimal_sweep_dict(tmp_path / "out")
        body["grid"] = []
        with pytest.raises(ValueError, match="grid"):
            SweepConfig.from_dict(body)

    def test_grid_axis_must_have_param_and_values(self, tmp_path: Path) -> None:
        body = _minimal_sweep_dict(tmp_path / "out")
        body["grid"] = [{"param": "opt.lr"}]
        with pytest.raises(ValueError, match="values"):
            SweepConfig.from_dict(body)

    def test_grid_axis_empty_values_raises(self, tmp_path: Path) -> None:
        body = _minimal_sweep_dict(tmp_path / "out")
        body["grid"] = [{"param": "opt.lr", "values": []}]
        with pytest.raises(ValueError, match="non-empty list"):
            SweepConfig.from_dict(body)

    def test_unknown_top_level_keys_are_ignored(self, tmp_path: Path) -> None:
        body = _minimal_sweep_dict(tmp_path / "out")
        body["future_key"] = "ignored"
        # Should not raise — forward-compat for future additions.
        SweepConfig.from_dict(body)

    def test_unknown_mini_train_keys_are_dropped(self, tmp_path: Path) -> None:
        body = _minimal_sweep_dict(tmp_path / "out")
        body["mini_train"]["future_subkey"] = 42
        cfg = SweepConfig.from_dict(body)
        # The unknown subkey was filtered before reaching the dataclass.
        assert not hasattr(cfg.mini_train, "future_subkey")


# ---------------------------------------------------------------------------
# Grid expansion + overrides
# ---------------------------------------------------------------------------


class TestGridExpansion:
    def test_single_axis(self) -> None:
        grid = [GridAxis(param="opt.lr", values=[1e-5, 3e-5])]
        out = expand_grid(grid)
        assert out == [{"opt.lr": 1e-5}, {"opt.lr": 3e-5}]

    def test_cartesian_two_axes(self) -> None:
        grid = [
            GridAxis(param="opt.lr", values=[1e-5, 3e-5]),
            GridAxis(param="opt.weight_decay", values=[0.0, 0.1]),
        ]
        out = expand_grid(grid)
        assert len(out) == 4
        combos = {(d["opt.lr"], d["opt.weight_decay"]) for d in out}
        assert combos == {(1e-5, 0.0), (1e-5, 0.1), (3e-5, 0.0), (3e-5, 0.1)}

    def test_cartesian_three_axes_count(self) -> None:
        grid = [
            GridAxis(param="a", values=[1, 2, 3]),
            GridAxis(param="b", values=[1, 2]),
            GridAxis(param="c", values=[1, 2, 3, 4]),
        ]
        assert len(expand_grid(grid)) == 24

    def test_empty_grid_yields_baseline(self) -> None:
        assert expand_grid([]) == [{}]

    def test_grid_axis_validation(self) -> None:
        with pytest.raises(ValueError):
            GridAxis(param="", values=[1])
        with pytest.raises(ValueError):
            GridAxis(param="x", values=[])

    def test_build_trial_matrix_respects_max_trials(self) -> None:
        cfg = SweepConfig.from_dict(_minimal_sweep_dict(Path("/tmp/x")))
        matrix = build_trial_matrix(cfg, max_trials=1)
        assert len(matrix) == 1

    def test_build_trial_matrix_no_cap(self) -> None:
        cfg = SweepConfig.from_dict(_minimal_sweep_dict(Path("/tmp/x")))
        matrix = build_trial_matrix(cfg)
        assert len(matrix) == 2  # 1 axis x 2 values


class TestApplyOverride:
    def test_single_level(self) -> None:
        d: dict = {}
        apply_override(d, "lr", 1e-5)
        assert d == {"lr": 1e-5}

    def test_nested(self) -> None:
        d: dict = {}
        apply_override(d, "opt.lr", 1e-5)
        assert d == {"opt": {"lr": 1e-5}}

    def test_deep_nesting(self) -> None:
        d: dict = {}
        apply_override(d, "opt.scheduler.warmup", 100)
        assert d == {"opt": {"scheduler": {"warmup": 100}}}

    def test_does_not_clobber_siblings(self) -> None:
        d: dict = {"opt": {"weight_decay": 0.1}}
        apply_override(d, "opt.lr", 1e-5)
        assert d == {"opt": {"weight_decay": 0.1, "lr": 1e-5}}

    def test_overrides_existing_leaf(self) -> None:
        d: dict = {"opt": {"lr": 1e-5}}
        apply_override(d, "opt.lr", 3e-5)
        assert d["opt"]["lr"] == 3e-5

    def test_replaces_scalar_intermediate(self) -> None:
        # An intermediate that is not a dict is silently replaced — this
        # is preferable to raising because the override grammar is a
        # write-through API.
        d: dict = {"opt": 1.0}
        apply_override(d, "opt.lr", 3e-5)
        assert d == {"opt": {"lr": 3e-5}}

    def test_empty_path_raises(self) -> None:
        with pytest.raises(ValueError):
            apply_override({}, "", 1)


# ---------------------------------------------------------------------------
# Base config + trial config resolution
# ---------------------------------------------------------------------------


class TestLoadBaseConfig:
    def test_none_yields_empty_dict(self) -> None:
        assert _load_base_config(None) == {}

    def test_yaml_path_loads_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "base.yaml"
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump({"opt": {"lr": 1e-5}, "sched": {"warmup_steps": 0}}, f)
        out = _load_base_config(p)
        assert out == {"opt": {"lr": 1e-5}, "sched": {"warmup_steps": 0}}

    def test_missing_yaml_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _load_base_config(tmp_path / "nope.yaml")

    def test_non_mapping_yaml_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "base.yaml"
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(["not", "a", "mapping"], f)
        with pytest.raises(ValueError, match="mapping"):
            _load_base_config(p)


class TestBuildTrialConfig:
    def test_overrides_layered_on_base(self) -> None:
        base = {"opt": {"lr": 1e-5, "weight_decay": 0.1}}
        out = build_trial_config(base, {"opt.lr": 3e-5})
        assert out == {"opt": {"lr": 3e-5, "weight_decay": 0.1}}

    def test_does_not_mutate_base(self) -> None:
        base = {"opt": {"lr": 1e-5}}
        build_trial_config(base, {"opt.lr": 3e-5})
        # Base unchanged.
        assert base == {"opt": {"lr": 1e-5}}

    def test_empty_overrides_yields_deep_copy(self) -> None:
        base = {"opt": {"lr": 1e-5}}
        out = build_trial_config(base, {})
        assert out == base
        assert out is not base
        assert out["opt"] is not base["opt"]

    def test_multiple_overrides_compose(self) -> None:
        base = {"opt": {"lr": 1e-5}}
        out = build_trial_config(
            base,
            {"opt.lr": 3e-5, "opt.weight_decay": 0.1, "sched.warmup_steps": 100},
        )
        assert out == {
            "opt": {"lr": 3e-5, "weight_decay": 0.1},
            "sched": {"warmup_steps": 100},
        }


# ---------------------------------------------------------------------------
# Trial id
# ---------------------------------------------------------------------------


class TestTrialId:
    def test_baseline_id_stable(self) -> None:
        assert trial_id_for({}) == "trial_baseline"

    def test_deterministic(self) -> None:
        a = trial_id_for({"opt.lr": 1e-5})
        b = trial_id_for({"opt.lr": 1e-5})
        assert a == b

    def test_distinct_overrides_distinct_ids(self) -> None:
        a = trial_id_for({"opt.lr": 1e-5})
        b = trial_id_for({"opt.lr": 3e-5})
        assert a != b

    def test_path_sanitization(self) -> None:
        tid = trial_id_for({"opt/lr": "foo/bar"})
        # Should not contain raw `/` characters (would create subdirs).
        assert "/" not in tid


# ---------------------------------------------------------------------------
# Stub scorer + aggregation
# ---------------------------------------------------------------------------


class TestStubScorer:
    def test_deterministic_per_overrides_seed(self) -> None:
        a = stub_score_for_overrides({"opt.lr": 1e-5}, seed=42, baseline_loss=4.78)
        b = stub_score_for_overrides({"opt.lr": 1e-5}, seed=42, baseline_loss=4.78)
        assert a == b

    def test_different_seed_different_loss(self) -> None:
        a, _, _ = stub_score_for_overrides({"opt.lr": 1e-5}, 42, 4.78)
        b, _, _ = stub_score_for_overrides({"opt.lr": 1e-5}, 1337, 4.78)
        assert a != b

    def test_different_overrides_different_loss(self) -> None:
        a, _, _ = stub_score_for_overrides({"opt.lr": 1e-5}, 42, 4.78)
        b, _, _ = stub_score_for_overrides({"opt.lr": 3e-5}, 42, 4.78)
        assert a != b

    def test_loss_in_expected_range(self) -> None:
        val_loss, _, _ = stub_score_for_overrides({"opt.lr": 1e-5}, 42, 4.78)
        assert 1.0 <= val_loss <= 5.0

    def test_score_formula(self) -> None:
        # score_from_loss must mirror validator behavior.
        delta, score = score_from_loss(3.0, 4.78)
        assert math.isclose(delta, 1.78)
        assert math.isclose(score, 1.78**1.2)
        # Clamp when val_loss > baseline.
        delta, score = score_from_loss(10.0, 4.78)
        assert delta == 0.0
        assert score == 0.0

    def test_score_non_finite_loss(self) -> None:
        delta, score = score_from_loss(float("inf"), 4.78)
        assert delta == 0.0 and score == 0.0
        delta, score = score_from_loss(float("nan"), 4.78)
        assert delta == 0.0 and score == 0.0


class TestAggregateSeedResults:
    def _seed_results(self, losses: list[float], baseline: float = 4.78) -> list[SeedResult]:
        out: list[SeedResult] = []
        for i, l in enumerate(losses):
            d, s = score_from_loss(l, baseline)
            out.append(SeedResult(seed=42 + i, val_loss=l, delta=d, score=s))
        return out

    def test_mean_and_std(self) -> None:
        sr = self._seed_results([2.0, 3.0, 4.0])
        r = aggregate_seed_results("t", {}, sr, baseline_loss=4.78)
        assert math.isclose(r.mean_val_loss, 3.0)
        # Population stddev of {2,3,4} == sqrt(2/3)
        assert math.isclose(r.std_val_loss, math.sqrt(2 / 3))
        assert r.error is None

    def test_all_non_finite_marks_error(self) -> None:
        sr = self._seed_results([float("inf"), float("nan")])
        r = aggregate_seed_results("t", {}, sr, baseline_loss=4.78)
        assert r.error == "all_seeds_non_finite"

    def test_mean_score_uses_average_of_per_seed_scores(self) -> None:
        sr = self._seed_results([2.0, 4.0], baseline=4.78)
        r = aggregate_seed_results("t", {}, sr, baseline_loss=4.78)
        expected = sum(s.score for s in sr) / 2
        assert math.isclose(r.mean_score, expected)

    def test_rescores_per_seed_against_baseline(self) -> None:
        # Pass seed_results with placeholder delta/score=0.0 and verify
        # aggregator recomputes them from val_loss against the provided
        # baseline. This is the contract that makes the
        # `baseline_loss` parameter load-bearing.
        sr = [
            SeedResult(seed=42, val_loss=2.0, delta=0.0, score=0.0),
            SeedResult(seed=1337, val_loss=4.0, delta=0.0, score=0.0),
        ]
        r = aggregate_seed_results("t", {}, sr, baseline_loss=4.78)
        # delta = 4.78 - val_loss for each seed.
        assert math.isclose(sr[0].delta, 4.78 - 2.0)
        assert math.isclose(sr[1].delta, 4.78 - 4.0)
        # score = delta ** 1.2.
        assert math.isclose(sr[0].score, (4.78 - 2.0) ** 1.2)
        # mean_score reflects the recomputed scores, not the placeholders.
        assert r.mean_score > 0.0

    def test_baseline_loss_change_propagates_through_aggregation(self) -> None:
        # Same per-seed val_loss with two different baselines yields
        # different aggregated scores.
        sr_a = [SeedResult(seed=42, val_loss=2.0, delta=0.0, score=0.0)]
        sr_b = [SeedResult(seed=42, val_loss=2.0, delta=0.0, score=0.0)]
        r_a = aggregate_seed_results("a", {}, sr_a, baseline_loss=4.78)
        r_b = aggregate_seed_results("b", {}, sr_b, baseline_loss=3.0)
        assert r_a.mean_score > r_b.mean_score
        assert math.isclose(r_a.mean_score, (4.78 - 2.0) ** 1.2)
        assert math.isclose(r_b.mean_score, (3.0 - 2.0) ** 1.2)


# ---------------------------------------------------------------------------
# score_trial
# ---------------------------------------------------------------------------


class TestScoreTrial:
    def test_stub_path(self, tmp_path: Path) -> None:
        cfg = SweepConfig.from_dict(_minimal_sweep_dict(tmp_path / "out"))
        r = score_trial(cfg, {"opt.lr": 1e-5}, scorer_module=None, force_stub=True)
        assert r.error is None
        assert len(r.seeds) == len(cfg.local_scorer.seeds)
        # All per-seed losses should be finite under the stub.
        assert all(math.isfinite(s.val_loss) for s in r.seeds)

    def test_real_path_raises_not_implemented_captured(self, tmp_path: Path) -> None:
        body = _minimal_sweep_dict(tmp_path / "out")
        body["mini_train"]["dry_run_stub_model"] = False
        cfg = SweepConfig.from_dict(body)
        # Force-stub=False, dry_run_stub_model=False — should hit the
        # NotImplementedError path and capture it into `error`.
        r = score_trial(cfg, {"opt.lr": 1e-5}, scorer_module=None, force_stub=False)
        assert r.error is not None
        assert "not_implemented" in r.error
        # resolved_config should be captured even on error so the trial
        # JSON records what the failed run would have consumed.
        assert r.resolved_config == {"opt": {"lr": 1e-5}}

    def test_resolved_config_layered_on_base(self, tmp_path: Path) -> None:
        # Provide an explicit base_config YAML and verify the override
        # is layered on top of it on every trial.
        base_yaml = tmp_path / "base.yaml"
        with open(base_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump({"opt": {"lr": 1e-5, "weight_decay": 0.1}}, f)
        body = _minimal_sweep_dict(tmp_path / "out")
        body["base_config"] = str(base_yaml)
        cfg = SweepConfig.from_dict(body)
        r = score_trial(cfg, {"opt.lr": 3e-5}, scorer_module=None, force_stub=True)
        # Override layered on top of base.
        assert r.resolved_config == {"opt": {"lr": 3e-5, "weight_decay": 0.1}}

    def test_missing_base_config_yields_error(self, tmp_path: Path) -> None:
        body = _minimal_sweep_dict(tmp_path / "out")
        body["base_config"] = str(tmp_path / "nope.yaml")
        cfg = SweepConfig.from_dict(body)
        r = score_trial(cfg, {"opt.lr": 1e-5}, scorer_module=None, force_stub=True)
        assert r.error is not None
        assert "base_config_load_failed" in r.error

    def test_resolved_config_captured_on_stub_path(self, tmp_path: Path) -> None:
        cfg = SweepConfig.from_dict(_minimal_sweep_dict(tmp_path / "out"))
        r = score_trial(cfg, {"opt.lr": 3e-5}, scorer_module=None, force_stub=True)
        assert r.error is None
        # With base_config=None and a single override.
        assert r.resolved_config == {"opt": {"lr": 3e-5}}


# ---------------------------------------------------------------------------
# Leaderboard render
# ---------------------------------------------------------------------------


class TestLeaderboard:
    def _make_results(self) -> list[TrialResult]:
        # Manually craft results with known ordering.
        def _make(score: float, name: str) -> TrialResult:
            tr = TrialResult(
                trial_id=name,
                overrides={"opt.lr": 1.0},
                mean_val_loss=4.78 - score ** (1 / 1.2),
                std_val_loss=0.01,
                mean_delta=score ** (1 / 1.2),
                mean_score=score,
            )
            tr.seeds = [SeedResult(seed=42, val_loss=tr.mean_val_loss, delta=tr.mean_delta, score=score)]
            return tr

        return [_make(0.1, "a"), _make(0.5, "b"), _make(0.3, "c")]

    def test_sort_descending_by_score(self) -> None:
        results = self._make_results()
        leaderboard, _ = render_leaderboard(results, "smoke", top_callout=3)
        ranked_ids = [r["trial_id"] for r in leaderboard["ranked"]]
        assert ranked_ids == ["b", "c", "a"]

    def test_failed_trials_sink_to_bottom(self) -> None:
        results = self._make_results()
        bad = TrialResult(trial_id="bad", overrides={}, error="boom")
        results.append(bad)
        leaderboard, md = render_leaderboard(results, "smoke")
        ranked_ids = [r["trial_id"] for r in leaderboard["ranked"]]
        assert ranked_ids[-1] == "bad"
        assert "**ERROR**" in md or "boom" in md

    def test_top_callout_respects_count(self) -> None:
        results = self._make_results()
        leaderboard, _ = render_leaderboard(results, "smoke", top_callout=2)
        assert leaderboard["top_callout"] == 2

    def test_markdown_has_table(self) -> None:
        results = self._make_results()
        _, md = render_leaderboard(results, "smoke")
        assert "| Rank |" in md
        assert "## Top" in md or "## Full ranking" in md
        # Sweep name surfaced as a header.
        assert "smoke" in md

    def test_empty_results(self) -> None:
        leaderboard, md = render_leaderboard([], "empty_sweep")
        assert leaderboard["num_trials"] == 0
        assert "empty_sweep" in md


# ---------------------------------------------------------------------------
# Output dir + persistence
# ---------------------------------------------------------------------------


class TestOutputDir:
    def test_prepare_output_dir_creates_layout(self, tmp_path: Path) -> None:
        cfg = SweepConfig.from_dict(_minimal_sweep_dict(tmp_path / "out"))
        out = prepare_output_dir(cfg)
        assert out.exists()
        assert (out / "trials").exists()
        assert (out / "sweep_config.json").exists()
        with open(out / "sweep_config.json", encoding="utf-8") as f:
            stamped = json.load(f)
        assert stamped["sweep_name"] == "smoke_test"
        assert stamped["grid"][0]["param"] == "opt.lr"

    def test_write_and_load_trial_result(self, tmp_path: Path) -> None:
        cfg = SweepConfig.from_dict(_minimal_sweep_dict(tmp_path / "out"))
        out = prepare_output_dir(cfg)
        tr = TrialResult(
            trial_id="t1",
            overrides={"opt.lr": 1e-5},
            resolved_config={"opt": {"lr": 1e-5}},
            mean_val_loss=2.5,
            std_val_loss=0.1,
            mean_delta=2.28,
            mean_score=2.28**1.2,
            seeds=[SeedResult(seed=42, val_loss=2.5, delta=2.28, score=2.28**1.2)],
        )
        write_trial_result(out, tr)
        loaded = load_trial_results(out)
        assert len(loaded) == 1
        assert loaded[0].trial_id == "t1"
        assert math.isclose(loaded[0].mean_val_loss, 2.5)
        # resolved_config must round-trip through JSON.
        assert loaded[0].resolved_config == {"opt": {"lr": 1e-5}}

    def test_load_trial_results_skips_malformed(self, tmp_path: Path) -> None:
        cfg = SweepConfig.from_dict(_minimal_sweep_dict(tmp_path / "out"))
        out = prepare_output_dir(cfg)
        (out / "trials" / "bogus.json").write_text("not json", encoding="utf-8")
        # Should not raise; bad file is skipped.
        assert load_trial_results(out) == []

    def test_write_leaderboard_produces_both_files(self, tmp_path: Path) -> None:
        cfg = SweepConfig.from_dict(_minimal_sweep_dict(tmp_path / "out"))
        out = prepare_output_dir(cfg)
        tr = TrialResult(
            trial_id="t1",
            overrides={"opt.lr": 1e-5},
            mean_val_loss=2.5,
            mean_delta=2.28,
            mean_score=2.28**1.2,
        )
        json_path, md_path = write_leaderboard(out, [tr], cfg.sweep_name)
        assert json_path.exists()
        assert md_path.exists()
        assert "smoke_test" in md_path.read_text()


# ---------------------------------------------------------------------------
# run_sweep e2e (stub)
# ---------------------------------------------------------------------------


class TestRunSweepE2E:
    def test_minimal_grid_two_trials(self, tmp_path: Path) -> None:
        body = _minimal_sweep_dict(tmp_path / "out")
        cfg = SweepConfig.from_dict(body)
        out_dir, results = run_sweep(cfg, force_stub=True)
        assert out_dir == tmp_path / "out"
        assert len(results) == 2
        assert all(r.error is None for r in results)
        # Files on disk.
        assert (out_dir / "leaderboard.json").exists()
        assert (out_dir / "leaderboard.md").exists()
        assert (out_dir / "sweep_config.json").exists()
        # One trial file per trial.
        trial_files = list((out_dir / "trials").glob("*.json"))
        assert len(trial_files) == 2

    def test_max_trials_caps_run(self, tmp_path: Path) -> None:
        body = _minimal_sweep_dict(tmp_path / "out")
        body["grid"] = [{"param": "opt.lr", "values": [1e-6, 1e-5, 1e-4, 1e-3]}]
        cfg = SweepConfig.from_dict(body)
        _, results = run_sweep(cfg, max_trials=2, force_stub=True)
        assert len(results) == 2

    def test_parallel_runs_match_serial(self, tmp_path: Path) -> None:
        body = _minimal_sweep_dict(tmp_path / "out_serial")
        cfg_serial = SweepConfig.from_dict(body)
        _, serial_results = run_sweep(cfg_serial, force_stub=True, parallel=1)

        body_p = _minimal_sweep_dict(tmp_path / "out_parallel")
        cfg_parallel = SweepConfig.from_dict(body_p)
        _, parallel_results = run_sweep(cfg_parallel, force_stub=True, parallel=2)

        # Parallel can finish out of order, so sort by trial_id before
        # comparing.
        def key(r: TrialResult) -> str:
            return r.trial_id

        s = sorted(serial_results, key=key)
        p = sorted(parallel_results, key=key)
        assert len(s) == len(p)
        for a, b in zip(s, p, strict=True):
            assert a.trial_id == b.trial_id
            # Stub is deterministic, so values match exactly.
            assert math.isclose(a.mean_val_loss, b.mean_val_loss)
            assert math.isclose(a.mean_score, b.mean_score)

    def test_progress_callback(self, tmp_path: Path) -> None:
        cfg = SweepConfig.from_dict(_minimal_sweep_dict(tmp_path / "out"))
        calls: list[tuple[int, int]] = []
        run_sweep(
            cfg,
            force_stub=True,
            progress_cb=lambda i, total, r: calls.append((i, total)),
        )
        assert len(calls) == 2
        assert calls[-1][1] == 2

    def test_empty_grid_yields_empty_leaderboard(self, tmp_path: Path) -> None:
        # SweepConfig forbids empty grid; bypass via direct construction.
        cfg = SweepConfig(
            sweep_name="empty",
            output_dir=tmp_path / "out",
            grid=[],
        )
        _, results = run_sweep(cfg, force_stub=True)
        # Empty grid → single baseline trial via expand_grid([]) == [{}].
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Dry-run + CLI
# ---------------------------------------------------------------------------


class TestDryRunMatrix:
    def test_render_lists_all_trials(self, tmp_path: Path) -> None:
        cfg = SweepConfig.from_dict(_minimal_sweep_dict(tmp_path / "out"))
        text = render_dry_run_matrix(cfg)
        assert "Sweep: smoke_test" in text
        assert "opt.lr" in text
        assert "1e-05" in text or "1.0e-05" in text or "1e-5" in text
        # 2 trials.
        assert "Trials (2)" in text

    def test_max_trials_caps_dry_run(self, tmp_path: Path) -> None:
        cfg = SweepConfig.from_dict(_minimal_sweep_dict(tmp_path / "out"))
        text = render_dry_run_matrix(cfg, max_trials=1)
        assert "Trials (1)" in text


class TestCLI:
    def test_dry_run_prints_and_exits_zero(self, tmp_path: Path) -> None:
        body = _minimal_sweep_dict(tmp_path / "out")
        yaml_path = _write_yaml(tmp_path, body)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--config", str(yaml_path), "--dry-run"])
        assert rc == 0
        text = buf.getvalue()
        assert "smoke_test" in text
        # Dry-run is pure preview — no filesystem side effects.
        assert not (tmp_path / "out").exists()

    def test_report_with_no_trials_returns_nonzero(self, tmp_path: Path) -> None:
        body = _minimal_sweep_dict(tmp_path / "out")
        yaml_path = _write_yaml(tmp_path, body)
        # Create empty output dir without running first.
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / "trials").mkdir()
        buf = io.StringIO()
        rc = main(["--config", str(yaml_path), "--report"])
        assert rc != 0

    def test_full_run_then_report_re_renders(self, tmp_path: Path) -> None:
        body = _minimal_sweep_dict(tmp_path / "out")
        yaml_path = _write_yaml(tmp_path, body)
        # First, do an actual run with --force-stub.
        rc = main(["--config", str(yaml_path), "--force-stub"])
        assert rc == 0
        leaderboard_path = tmp_path / "out" / "leaderboard.json"
        assert leaderboard_path.exists()
        first_payload = json.loads(leaderboard_path.read_text())

        # Delete and re-render via --report.
        leaderboard_path.unlink()
        rc = main(["--config", str(yaml_path), "--report"])
        assert rc == 0
        assert leaderboard_path.exists()
        # Re-rendered content matches the original.
        second_payload = json.loads(leaderboard_path.read_text())
        assert first_payload["num_trials"] == second_payload["num_trials"]
        first_ids = [r["trial_id"] for r in first_payload["ranked"]]
        second_ids = [r["trial_id"] for r in second_payload["ranked"]]
        assert first_ids == second_ids

    def test_missing_config_arg_raises(self) -> None:
        with pytest.raises(SystemExit):
            main([])

    def test_invalid_parallel_raises(self, tmp_path: Path) -> None:
        body = _minimal_sweep_dict(tmp_path / "out")
        yaml_path = _write_yaml(tmp_path, body)
        with pytest.raises(SystemExit):
            main(["--config", str(yaml_path), "--parallel", "0"])

    def test_max_trials_negative_raises(self, tmp_path: Path) -> None:
        body = _minimal_sweep_dict(tmp_path / "out")
        yaml_path = _write_yaml(tmp_path, body)
        with pytest.raises(SystemExit):
            main(["--config", str(yaml_path), "--max-trials", "-1"])


# ---------------------------------------------------------------------------
# Example YAML files validate
# ---------------------------------------------------------------------------


class TestExampleConfigs:
    @pytest.mark.parametrize(
        "name",
        ["example_lr_sweep.yaml", "example_optimizer_sweep.yaml", "example_clip_sweep.yaml"],
    )
    def test_example_yaml_loads(self, name: str) -> None:
        path = _HERE / "sweep_configs" / name
        assert path.exists(), f"missing example config {path}"
        cfg = SweepConfig.from_yaml(path)
        assert cfg.sweep_name
        assert len(cfg.grid) >= 1
        # Output dir must be set.
        assert str(cfg.output_dir)

    def test_example_lr_sweep_grid_size(self) -> None:
        path = _HERE / "sweep_configs" / "example_lr_sweep.yaml"
        cfg = SweepConfig.from_yaml(path)
        # 6 lrs x 3 weight decays = 18 trials.
        assert len(build_trial_matrix(cfg)) == 18

    def test_example_clip_sweep_grid_size(self) -> None:
        path = _HERE / "sweep_configs" / "example_clip_sweep.yaml"
        cfg = SweepConfig.from_yaml(path)
        assert len(build_trial_matrix(cfg)) == 4


# ---------------------------------------------------------------------------
# Local-scorer absence is non-fatal
# ---------------------------------------------------------------------------


class TestMissingLocalScorer:
    def test_score_trial_works_when_local_scorer_missing(self, tmp_path: Path) -> None:
        # bench.local_scorer is intentionally not committed in this unit;
        # the harness must run on stubs regardless. The stub path does
        # not touch local_scorer at all.
        cfg = SweepConfig.from_dict(_minimal_sweep_dict(tmp_path / "out"))
        r = score_trial(cfg, {"opt.lr": 1e-5}, scorer_module=None, force_stub=True)
        assert r.error is None
