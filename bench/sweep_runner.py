"""YAML-driven hyperparameter sweep harness for the Connito miner.

Runs mini training trials across a Cartesian grid of HP overrides, scores
each resulting checkpoint with a validator-replica eval, and emits a
sorted leaderboard (JSON + Markdown). The goal is evidence-driven
hyperparameter tuning before pushing changes that affect `miner_val_loss`
on chain.

Module status
-------------
This module is intentionally scaffolded so that the harness itself
(YAML schema, grid expansion, CLI, leaderboard) is independently
testable WITHOUT real training. Real training is performed via the
optional `_run_real_training_trial` path which is only reachable when
`mini_train.dry_run_stub_model` is False and `torch` + `transformers`
are importable. The default smoke-test path uses a deterministic stub
that returns synthetic per-trial losses keyed on the override
dictionary, which is sufficient to exercise:

- YAML loading + validation (`SweepConfig`).
- Grid expansion (`expand_grid`).
- Dotted-path config overrides (`apply_override`).
- Multi-seed scoring + aggregation (`score_trial`).
- Leaderboard render (JSON + Markdown).
- Output dir layout (`prepare_output_dir`, `--report`).

Integration with Unit 1 (`bench.local_scorer`)
----------------------------------------------
When `bench.local_scorer` is available on the path AND the trial is a
real (non-stub) run, the harness will invoke its public scoring entry
point if present. When not available, the harness falls back to a
minimal inline path that calls `connito.shared.evaluate:evaluate_model`
directly with the same numeric semantics as the validator (see
`connito/validator/evaluator.py:787` for the score formula).

CLI
---
``python -m bench.sweep_runner --config <yaml>``
``python -m bench.sweep_runner --config <yaml> --dry-run``
``python -m bench.sweep_runner --config <yaml> --max-trials 4``
``python -m bench.sweep_runner --config <yaml> --parallel 2``
``python -m bench.sweep_runner --report --config <yaml>``
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

# Logging
# -------
# The project standard is structlog (see connito.shared.app_logging).
# We use the stdlib logger here so the bench harness has zero
# import-time dependency on the heavyweight `connito` package — the
# smoke-test path must run in environments where torch/transformers are
# unavailable (CI, sandbox).
logger = logging.getLogger("bench.sweep_runner")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(os.environ.get("BENCH_LOG_LEVEL", "INFO"))
    logger.propagate = False


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class MiniTrainCfg:
    """Configuration for the mini training run within a single trial."""

    steps: int = 100
    base_model: str = "deepseek-ai/DeepSeek-V2-Lite"
    expert_group: str = "exp_math"
    # If True, skip real training and use a deterministic stub scorer
    # (smoke-test / CI mode). The CLI `--dry-run` flag also forces this
    # behavior independently.
    dry_run_stub_model: bool = False


@dataclass
class LocalScorerCfg:
    """Configuration for the local validator-replica eval."""

    max_batches: int = 20
    seeds: list[int] = field(default_factory=lambda: [42, 1337, 314159])
    # Synthetic baseline loss for the score formula
    # `score = max(0, baseline - val_loss) ** 1.2`. Falls back to the
    # observed top-of-leaderboard baseline (4.78 from the dashboard) so
    # smoke tests can still produce a non-zero delta.
    baseline_loss: float = 4.78


@dataclass
class GridAxis:
    """A single axis of the Cartesian sweep grid."""

    param: str
    values: list[Any]

    def __post_init__(self) -> None:
        if not isinstance(self.param, str) or not self.param:
            raise ValueError(f"grid axis 'param' must be a non-empty str, got {self.param!r}")
        if not isinstance(self.values, list) or len(self.values) == 0:
            raise ValueError(
                f"grid axis 'values' for {self.param!r} must be a non-empty list, "
                f"got {self.values!r}"
            )


@dataclass
class SweepConfig:
    """Validated representation of the sweep YAML."""

    sweep_name: str
    output_dir: Path
    base_config: str | None = None
    mini_train: MiniTrainCfg = field(default_factory=MiniTrainCfg)
    local_scorer: LocalScorerCfg = field(default_factory=LocalScorerCfg)
    grid: list[GridAxis] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SweepConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"sweep config not found: {path}")
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SweepConfig:
        if not isinstance(raw, dict):
            raise ValueError(f"sweep config root must be a mapping, got {type(raw).__name__}")

        sweep_name = raw.get("sweep_name")
        if not sweep_name or not isinstance(sweep_name, str):
            raise ValueError("sweep config must contain a non-empty string `sweep_name`")

        output_dir = raw.get("output_dir")
        if not output_dir or not isinstance(output_dir, str):
            raise ValueError("sweep config must contain a non-empty string `output_dir`")

        mt_raw = raw.get("mini_train") or {}
        if not isinstance(mt_raw, dict):
            raise ValueError("`mini_train` must be a mapping if present")
        mini_train = MiniTrainCfg(**{k: v for k, v in mt_raw.items() if k in MiniTrainCfg.__annotations__})

        ls_raw = raw.get("local_scorer") or {}
        if not isinstance(ls_raw, dict):
            raise ValueError("`local_scorer` must be a mapping if present")
        local_scorer = LocalScorerCfg(
            **{k: v for k, v in ls_raw.items() if k in LocalScorerCfg.__annotations__}
        )

        grid_raw = raw.get("grid") or []
        if not isinstance(grid_raw, list) or len(grid_raw) == 0:
            raise ValueError("sweep config must contain a non-empty `grid` list")
        grid: list[GridAxis] = []
        for i, axis in enumerate(grid_raw):
            if not isinstance(axis, dict):
                raise ValueError(f"grid[{i}] must be a mapping, got {type(axis).__name__}")
            if "param" not in axis or "values" not in axis:
                raise ValueError(f"grid[{i}] must contain `param` and `values` keys")
            grid.append(GridAxis(param=axis["param"], values=list(axis["values"])))

        return cls(
            sweep_name=sweep_name,
            output_dir=Path(output_dir),
            base_config=raw.get("base_config"),
            mini_train=mini_train,
            local_scorer=local_scorer,
            grid=grid,
        )


# ---------------------------------------------------------------------------
# Grid expansion
# ---------------------------------------------------------------------------


def expand_grid(grid: list[GridAxis]) -> list[dict[str, Any]]:
    """Expand a list of grid axes into a list of (param -> value) dicts.

    Returns the Cartesian product preserving the input axis order. If
    `grid` is empty, returns a single trivial trial `[{}]` so the caller
    can still run the unmodified base config. The single-empty-dict
    shape keeps the iteration loop uniform.
    """
    if not grid:
        return [{}]
    value_lists = [axis.values for axis in grid]
    names = [axis.param for axis in grid]
    out: list[dict[str, Any]] = []
    for combo in itertools.product(*value_lists):
        out.append({name: val for name, val in zip(names, combo, strict=True)})
    return out


def apply_override(cfg_dict: dict[str, Any], dotted_path: str, value: Any) -> dict[str, Any]:
    """Apply a dotted-path override to a nested config dict in-place.

    Creates intermediate mappings if they do not exist. This is the
    counterpart to the YAML schema where users write e.g. `opt.lr` to
    target the `opt` block's `lr` field on a `WorkerConfig`-shaped dict.

    Returns the mutated dict (also mutates in-place) so callers can
    chain.
    """
    if not dotted_path:
        raise ValueError("override dotted_path must be non-empty")
    parts = dotted_path.split(".")
    cur: Any = cfg_dict
    for key in parts[:-1]:
        existing = cur.get(key)
        if not isinstance(existing, dict):
            existing = {}
            cur[key] = existing
        cur = existing
    cur[parts[-1]] = value
    return cfg_dict


def trial_id_for(overrides: dict[str, Any]) -> str:
    """Deterministic short id for a trial keyed on its overrides."""
    if not overrides:
        return "trial_baseline"
    canonical = json.dumps(overrides, sort_keys=True, default=str)
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:10]
    # Short human-readable suffix from the first param's name/value
    first_param, first_value = next(iter(overrides.items()))
    safe = f"{first_param.replace('.', '_')}-{first_value}".replace("/", "_")
    safe = "".join(c for c in safe if c.isalnum() or c in "._-")[:32]
    return f"trial_{safe}_{digest}"


def _load_base_config(base_config: str | Path | None) -> dict[str, Any]:
    """Load the base config dict that overrides are layered on top of.

    Resolution order:
    * If `base_config` is a path to a YAML file, load it.
    * Otherwise, fall back to an empty mapping. The real-training path
      will then call `WorkerConfig(**resolved)` which fills in defaults
      from `connito/shared/config.py`. Doing the merge as a plain dict
      (rather than instantiating a `WorkerConfig` here) keeps the
      smoke-test path import-light.
    """
    if base_config is None:
        return {}
    path = Path(base_config)
    if not path.exists():
        raise FileNotFoundError(
            f"base_config path does not exist: {base_config}"
        )
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"base_config YAML must be a mapping, got {type(raw).__name__}"
        )
    return raw


def build_trial_config(
    base: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Apply every override onto a deep copy of the base config.

    Single call site for `apply_override` — invoked for every trial
    regardless of stub-vs-real path so the resolved config is always
    captured on `TrialResult.resolved_config` for reproducibility.
    """
    resolved = copy.deepcopy(base)
    for dotted_path, value in overrides.items():
        apply_override(resolved, dotted_path, value)
    return resolved


# ---------------------------------------------------------------------------
# Trial result schema
# ---------------------------------------------------------------------------


@dataclass
class SeedResult:
    """Per-seed eval result for a single trial."""

    seed: int
    val_loss: float
    delta: float
    score: float


@dataclass
class TrialResult:
    """Aggregated result for a single trial across all seeds."""

    trial_id: str
    overrides: dict[str, Any]
    # Materialized config dict produced by applying `overrides` on top
    # of the base config (see `build_trial_config`). Captured on every
    # trial so the on-disk trial JSON records the EXACT config the
    # mini-training run consumed — important for reproducibility once
    # the real-training scaffold is wired.
    resolved_config: dict[str, Any] = field(default_factory=dict)
    seeds: list[SeedResult] = field(default_factory=list)
    mean_val_loss: float = math.nan
    std_val_loss: float = math.nan
    mean_delta: float = math.nan
    mean_score: float = math.nan
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "overrides": self.overrides,
            "resolved_config": self.resolved_config,
            "seeds": [asdict(s) for s in self.seeds],
            "mean_val_loss": self.mean_val_loss,
            "std_val_loss": self.std_val_loss,
            "mean_delta": self.mean_delta,
            "mean_score": self.mean_score,
            "error": self.error,
        }


def aggregate_seed_results(
    trial_id: str,
    overrides: dict[str, Any],
    seed_results: list[SeedResult],
    baseline_loss: float,
) -> TrialResult:
    """Aggregate per-seed eval results into a `TrialResult`.

    Every per-seed `delta` and `score` is RE-COMPUTED here from
    `SeedResult.val_loss` and the configured `baseline_loss`, in-place
    on the SeedResult instances, so this function is the single source
    of truth for the `score = max(0, baseline - val_loss) ** 1.2`
    formula. Upstream scorers may set delta/score to placeholder values
    (or 0.0) and rely on aggregation to fill them in against whatever
    `local_scorer.baseline_loss` is configured for the sweep. This
    decouples raw eval (which only knows val_loss) from the scoring
    parameter (which may change between sweeps).

    `mean_score` is computed from the mean of per-seed scores (not from
    the mean val_loss) so a single high-variance seed cannot dominate.
    This matches the validator's per-round behavior — every round
    produces its own score — but at the per-trial level we average those
    scores to reflect robustness across seeds.
    """
    # Re-score every seed against the configured baseline. Mutate
    # in-place so callers serializing `seed_results` afterwards see
    # consistent per-seed delta/score.
    for sr in seed_results:
        sr.delta, sr.score = score_from_loss(sr.val_loss, baseline_loss)

    finite = [s for s in seed_results if math.isfinite(s.val_loss)]
    if not finite:
        return TrialResult(
            trial_id=trial_id,
            overrides=overrides,
            seeds=seed_results,
            error="all_seeds_non_finite",
        )

    losses = [s.val_loss for s in finite]
    deltas = [s.delta for s in finite]
    scores = [s.score for s in finite]

    n = len(losses)
    mean_loss = sum(losses) / n
    var = sum((x - mean_loss) ** 2 for x in losses) / n if n > 0 else 0.0
    std_loss = math.sqrt(var)

    return TrialResult(
        trial_id=trial_id,
        overrides=overrides,
        seeds=seed_results,
        mean_val_loss=mean_loss,
        std_val_loss=std_loss,
        mean_delta=sum(deltas) / n,
        mean_score=sum(scores) / n,
    )


def score_from_loss(val_loss: float, baseline_loss: float) -> tuple[float, float]:
    """Mirror validator scoring: returns (delta, score).

    Matches `connito/validator/evaluator.py:787` —
    ``delta = max(0, baseline - val_loss); score = delta ** 1.2``.
    """
    if not math.isfinite(val_loss):
        return 0.0, 0.0
    delta = max(0.0, baseline_loss - val_loss)
    score = delta**1.2
    return delta, score


# ---------------------------------------------------------------------------
# Scorer adapters
# ---------------------------------------------------------------------------


def _try_import_local_scorer() -> Any | None:
    """Best-effort import of Unit 1's local_scorer.

    Returns the imported module if available, else None. Logs a single
    info line on either path.
    """
    try:
        import bench.local_scorer as ls  # type: ignore  # noqa: F401

        logger.info("Using bench.local_scorer for scoring")
        return ls
    except ImportError:
        logger.info(
            "bench.local_scorer not available — falling back to inline minimal "
            "scoring path (connito.shared.evaluate.evaluate_model)"
        )
        return None


def stub_score_for_overrides(
    overrides: dict[str, Any],
    seed: int,
    baseline_loss: float,
) -> tuple[float, float, float]:
    """Deterministic stub for smoke tests / dry-run.

    Computes a synthetic val_loss as a deterministic function of the
    override dict + seed. The mapping is designed so distinct overrides
    yield distinct losses while remaining bounded, which is enough to
    exercise the leaderboard sort + multi-seed aggregation.
    """
    key = json.dumps([sorted(overrides.items(), key=lambda kv: kv[0]), int(seed)], default=str)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    # First 8 bytes → [0, 1) fraction.
    frac = int.from_bytes(digest[:8], "big") / 2**64
    # Bound val_loss to a plausible range [1.0, 5.0] so deltas vs the
    # configured baseline (default 4.78) span both positive and zero.
    val_loss = 1.0 + 4.0 * frac
    delta, score = score_from_loss(val_loss, baseline_loss)
    return val_loss, delta, score


def _run_real_training_trial(
    mini_train: MiniTrainCfg,
    local_scorer: LocalScorerCfg,
    overrides: dict[str, Any],
    resolved_config: dict[str, Any],
    scorer_module: Any | None,
) -> list[SeedResult]:
    """Real-training trial path.

    Only reachable when `mini_train.dry_run_stub_model` is False AND
    torch + transformers + connito are importable. Defers the heavy
    imports until inside this function so the smoke-test path does not
    drag them in.

    `resolved_config` is the merged base + override dict produced by
    `build_trial_config`; the real-training path consumes it directly
    (e.g. `WorkerConfig(**resolved_config)`).

    Notes
    -----
    This unit (Unit 3) lays the scaffold. The actual training-loop body
    is intentionally minimal — it loads the base model, applies the HP
    overrides via the live `WorkerConfig`, runs N optimizer steps over
    a small streamed batch, saves the resulting state_dict, and then
    invokes the local scorer. Most of the heavy lifting (data loading,
    model assembly, gradient clipping, etc.) is delegated to the
    existing miner/train helpers — see `connito/miner/train.py`. Hooks
    are provided so a follow-up unit can swap in a more elaborate
    micro-training recipe without changing the harness CLI.
    """
    # Heavy imports are kept inside the function to avoid pulling them
    # into the smoke-test path.
    import torch  # noqa: F401  # type: ignore
    from connito.shared.config import WorkerConfig  # type: ignore  # noqa: F401
    from connito.shared.evaluate import evaluate_model  # type: ignore  # noqa: F401

    # The full implementation of mini-training is left to a follow-up
    # because it depends on real GPUs and a real expert-group checkpoint.
    # We surface a NotImplementedError so callers don't silently get
    # bogus results — but we still return per-seed structures shaped
    # like the success path so the integration is easier to land later.
    raise NotImplementedError(
        "Real-training trial path is scaffolded but not yet wired. Set "
        "`mini_train.dry_run_stub_model: true` (or pass `--dry-run`) "
        "to exercise the harness end-to-end with the deterministic "
        "stub scorer."
    )


def score_trial(
    cfg: SweepConfig,
    overrides: dict[str, Any],
    scorer_module: Any | None,
    force_stub: bool = False,
    base_config_dict: dict[str, Any] | None = None,
) -> TrialResult:
    """Run + score a single trial.

    `base_config_dict` is the parsed `cfg.base_config` YAML loaded once
    by the driver and threaded through to every trial; we pass it in
    rather than re-reading the file per-trial. If None, the driver
    will resolve it via `_load_base_config`.

    The dotted-path `overrides` are applied via `build_trial_config`
    onto a deep copy of the base config, producing the
    `resolved_config` dict that the real-training path consumes and
    that is persisted on `TrialResult` for reproducibility.

    If `force_stub` is True (e.g. CLI `--force-stub` or
    `mini_train.dry_run_stub_model`), the stub scorer is used. Otherwise
    `_run_real_training_trial` is invoked, which may raise
    NotImplementedError on systems where the real-training scaffold has
    not been wired yet — this is captured into `TrialResult.error` so
    the leaderboard render still completes.
    """
    tid = trial_id_for(overrides)
    seeds = cfg.local_scorer.seeds or [42]
    use_stub = force_stub or cfg.mini_train.dry_run_stub_model

    if base_config_dict is None:
        try:
            base_config_dict = _load_base_config(cfg.base_config)
        except (FileNotFoundError, ValueError) as e:
            logger.error("trial %s: base_config load failed: %s", tid, e)
            return TrialResult(
                trial_id=tid,
                overrides=overrides,
                error=f"base_config_load_failed: {e}",
            )

    resolved_config = build_trial_config(base_config_dict, overrides)

    seed_results: list[SeedResult] = []
    try:
        if use_stub:
            for seed in seeds:
                val_loss, delta, score = stub_score_for_overrides(
                    overrides, seed, cfg.local_scorer.baseline_loss
                )
                seed_results.append(SeedResult(seed=seed, val_loss=val_loss, delta=delta, score=score))
        else:
            seed_results = _run_real_training_trial(
                cfg.mini_train,
                cfg.local_scorer,
                overrides,
                resolved_config,
                scorer_module,
            )
    except NotImplementedError as e:
        logger.error("trial %s failed: %s", tid, e)
        return TrialResult(
            trial_id=tid,
            overrides=overrides,
            resolved_config=resolved_config,
            error=f"not_implemented: {e}",
        )
    except Exception as e:  # pragma: no cover — defensive
        logger.exception("trial %s failed with exception", tid)
        return TrialResult(
            trial_id=tid,
            overrides=overrides,
            resolved_config=resolved_config,
            error=f"exception: {type(e).__name__}: {e}",
        )

    result = aggregate_seed_results(tid, overrides, seed_results, cfg.local_scorer.baseline_loss)
    result.resolved_config = resolved_config
    return result


# ---------------------------------------------------------------------------
# Output / leaderboard
# ---------------------------------------------------------------------------


def prepare_output_dir(cfg: SweepConfig) -> Path:
    """Create (or reuse) the output dir for this sweep and stamp it."""
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "trials").mkdir(parents=True, exist_ok=True)
    # Persist the resolved sweep config so a later `--report` can
    # recover the same baseline + seeds + grid without re-parsing the
    # source YAML.
    cfg_resolved = {
        "sweep_name": cfg.sweep_name,
        "output_dir": str(cfg.output_dir),
        "base_config": cfg.base_config,
        "mini_train": asdict(cfg.mini_train),
        "local_scorer": asdict(cfg.local_scorer),
        "grid": [{"param": a.param, "values": a.values} for a in cfg.grid],
    }
    with open(out / "sweep_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_resolved, f, indent=2, default=str)
    return out


def write_trial_result(out_dir: Path, result: TrialResult) -> Path:
    """Persist a single trial's full result to <out>/trials/<id>.json."""
    trial_path = out_dir / "trials" / f"{result.trial_id}.json"
    trial_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trial_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    return trial_path


def load_trial_results(out_dir: Path) -> list[TrialResult]:
    """Reconstruct `TrialResult` objects from `<out>/trials/*.json`."""
    trial_dir = out_dir / "trials"
    if not trial_dir.exists():
        return []
    results: list[TrialResult] = []
    for path in sorted(trial_dir.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning("skipping malformed trial file %s: %s", path, e)
            continue
        seeds = [SeedResult(**s) for s in raw.get("seeds", [])]
        results.append(
            TrialResult(
                trial_id=raw["trial_id"],
                overrides=raw.get("overrides", {}),
                resolved_config=raw.get("resolved_config", {}),
                seeds=seeds,
                mean_val_loss=float(raw.get("mean_val_loss", math.nan)),
                std_val_loss=float(raw.get("std_val_loss", math.nan)),
                mean_delta=float(raw.get("mean_delta", math.nan)),
                mean_score=float(raw.get("mean_score", math.nan)),
                error=raw.get("error"),
            )
        )
    return results


def _score_key(r: TrialResult) -> float:
    """Sort key for leaderboard: higher mean_score wins; NaN/error last."""
    if r.error or not math.isfinite(r.mean_score):
        return float("-inf")
    return float(r.mean_score)


def render_leaderboard(
    results: list[TrialResult],
    sweep_name: str,
    top_callout: int = 5,
) -> tuple[dict[str, Any], str]:
    """Render leaderboard as (json_dict, markdown_str).

    Results are sorted by `mean_score` descending. Failed trials sink
    to the bottom with `mean_score = -inf`.
    """
    sorted_results = sorted(results, key=_score_key, reverse=True)

    leaderboard_json: dict[str, Any] = {
        "sweep_name": sweep_name,
        "num_trials": len(sorted_results),
        "top_callout": min(top_callout, len(sorted_results)),
        "ranked": [r.to_dict() for r in sorted_results],
    }

    lines: list[str] = []
    lines.append(f"# Sweep leaderboard: `{sweep_name}`")
    lines.append("")
    lines.append(f"Trials: {len(sorted_results)}")
    lines.append("")

    if sorted_results:
        # Top callout
        top = sorted_results[: min(top_callout, len(sorted_results))]
        lines.append(f"## Top {len(top)}")
        lines.append("")
        for i, r in enumerate(top, start=1):
            override_str = ", ".join(f"{k}={v}" for k, v in r.overrides.items()) or "(baseline)"
            if r.error:
                lines.append(
                    f"{i}. `{r.trial_id}` — {override_str} — **ERROR**: {r.error}"
                )
            else:
                lines.append(
                    f"{i}. `{r.trial_id}` — {override_str} — "
                    f"mean_score={r.mean_score:.6f} "
                    f"mean_val_loss={r.mean_val_loss:.4f} "
                    f"(±{r.std_val_loss:.4f}) "
                    f"mean_delta={r.mean_delta:.4f}"
                )
        lines.append("")

    # Full table
    lines.append("## Full ranking")
    lines.append("")
    lines.append("| Rank | Trial | Overrides | mean_score | mean_val_loss | std_val_loss | mean_delta | error |")
    lines.append("|---:|:---|:---|---:|---:|---:|---:|:---|")
    for rank, r in enumerate(sorted_results, start=1):
        override_str = "; ".join(f"`{k}`={v}" for k, v in r.overrides.items()) or "_baseline_"
        if r.error:
            lines.append(
                f"| {rank} | `{r.trial_id}` | {override_str} | _err_ | _err_ | _err_ | _err_ | {r.error} |"
            )
        else:
            lines.append(
                f"| {rank} | `{r.trial_id}` | {override_str} | "
                f"{r.mean_score:.6f} | {r.mean_val_loss:.4f} | "
                f"{r.std_val_loss:.4f} | {r.mean_delta:.4f} | |"
            )

    return leaderboard_json, "\n".join(lines) + "\n"


def write_leaderboard(out_dir: Path, results: list[TrialResult], sweep_name: str) -> tuple[Path, Path]:
    """Write leaderboard.json + leaderboard.md to `out_dir`."""
    leaderboard_json, leaderboard_md = render_leaderboard(results, sweep_name)
    json_path = out_dir / "leaderboard.json"
    md_path = out_dir / "leaderboard.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(leaderboard_json, f, indent=2, default=str)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(leaderboard_md)
    return json_path, md_path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_trial_matrix(cfg: SweepConfig, max_trials: int | None = None) -> list[dict[str, Any]]:
    """Expand the grid and cap at `max_trials` (None = no cap)."""
    matrix = expand_grid(cfg.grid)
    if max_trials is not None and max_trials >= 0:
        matrix = matrix[:max_trials]
    return matrix


def run_sweep(
    cfg: SweepConfig,
    *,
    max_trials: int | None = None,
    parallel: int = 1,
    force_stub: bool = False,
    progress_cb: Callable[[int, int, TrialResult], None] | None = None,
) -> tuple[Path, list[TrialResult]]:
    """Run the full sweep, persisting per-trial results + leaderboard.

    Returns (output_dir, results) where `results` is the full list of
    TrialResults in the order they were scored.
    """
    out_dir = prepare_output_dir(cfg)
    matrix = build_trial_matrix(cfg, max_trials=max_trials)
    if not matrix:
        logger.warning("sweep %s: empty trial matrix, nothing to do", cfg.sweep_name)
        write_leaderboard(out_dir, [], cfg.sweep_name)
        return out_dir, []

    scorer_module = None if force_stub or cfg.mini_train.dry_run_stub_model else _try_import_local_scorer()
    logger.info(
        "sweep %s: %d trial(s) (parallel=%d, stub=%s)",
        cfg.sweep_name,
        len(matrix),
        parallel,
        force_stub or cfg.mini_train.dry_run_stub_model,
    )

    # Load base config once and reuse across trials.
    try:
        base_config_dict = _load_base_config(cfg.base_config)
    except (FileNotFoundError, ValueError) as e:
        # Surface as a per-trial error rather than a sweep-wide crash so
        # the leaderboard still gets written (empty, all-error).
        logger.error("sweep %s: base_config load failed: %s — every trial will error", cfg.sweep_name, e)
        base_config_dict = None

    results: list[TrialResult] = []

    def _one(overrides: dict[str, Any]) -> TrialResult:
        t0 = time.monotonic()
        r = score_trial(
            cfg,
            overrides,
            scorer_module,
            force_stub=force_stub,
            base_config_dict=base_config_dict,
        )
        write_trial_result(out_dir, r)
        logger.info(
            "trial %s scored in %.2fs (mean_score=%s mean_val_loss=%s err=%s)",
            r.trial_id,
            time.monotonic() - t0,
            f"{r.mean_score:.6f}" if math.isfinite(r.mean_score) else "nan",
            f"{r.mean_val_loss:.4f}" if math.isfinite(r.mean_val_loss) else "nan",
            r.error,
        )
        return r

    if parallel <= 1:
        for i, overrides in enumerate(matrix, start=1):
            r = _one(overrides)
            results.append(r)
            if progress_cb is not None:
                progress_cb(i, len(matrix), r)
    else:
        # Parallel trials. Threads (not processes) so we share the
        # imported scorer_module and don't re-pay heavyweight import
        # cost N times. The trial work itself is GPU/IO-bound so the
        # GIL is rarely contended.
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            future_to_idx = {
                ex.submit(_one, overrides): idx
                for idx, overrides in enumerate(matrix, start=1)
            }
            for fut in as_completed(future_to_idx):
                r = fut.result()
                results.append(r)
                if progress_cb is not None:
                    progress_cb(len(results), len(matrix), r)

    write_leaderboard(out_dir, results, cfg.sweep_name)
    return out_dir, results


def render_dry_run_matrix(cfg: SweepConfig, max_trials: int | None = None) -> str:
    """Render the planned trial matrix as a human-readable string."""
    matrix = build_trial_matrix(cfg, max_trials=max_trials)
    lines = [
        f"Sweep: {cfg.sweep_name}",
        f"Output dir: {cfg.output_dir}",
        f"Base config: {cfg.base_config or '(connito defaults)'}",
        f"Mini-train: steps={cfg.mini_train.steps} "
        f"base_model={cfg.mini_train.base_model} "
        f"expert_group={cfg.mini_train.expert_group} "
        f"stub={cfg.mini_train.dry_run_stub_model}",
        f"Local scorer: max_batches={cfg.local_scorer.max_batches} "
        f"seeds={cfg.local_scorer.seeds} "
        f"baseline_loss={cfg.local_scorer.baseline_loss}",
        f"Grid axes ({len(cfg.grid)}):",
    ]
    for axis in cfg.grid:
        lines.append(f"  - {axis.param}: {axis.values}")
    lines.append(f"Trials ({len(matrix)}):")
    for i, overrides in enumerate(matrix, start=1):
        tid = trial_id_for(overrides)
        if overrides:
            override_str = ", ".join(f"{k}={v}" for k, v in overrides.items())
        else:
            override_str = "(baseline)"
        lines.append(f"  {i:>3}. {tid}  {override_str}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m bench.sweep_runner",
        description="YAML-driven hyperparameter sweep harness for the Connito miner.",
    )
    p.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to sweep config YAML.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Expand the trial matrix and print it. Pure preview — no "
            "filesystem side effects and no scoring is performed."
        ),
    )
    p.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Cap the trial matrix to N trials (after grid expansion).",
    )
    p.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Run N trials in parallel. Default 1; respect GPU memory before raising.",
    )
    p.add_argument(
        "--report",
        action="store_true",
        help=(
            "Re-render leaderboard.json + leaderboard.md from existing "
            "trial results in <output_dir>/trials/."
        ),
    )
    p.add_argument(
        "--force-stub",
        action="store_true",
        help=(
            "Force the deterministic stub scorer regardless of "
            "`mini_train.dry_run_stub_model`. Used by tests."
        ),
    )
    return p


def _validate_args(args: argparse.Namespace) -> None:
    if not args.config:
        raise SystemExit("--config is required")
    if args.parallel < 1:
        raise SystemExit("--parallel must be >= 1")
    if args.max_trials is not None and args.max_trials < 0:
        raise SystemExit("--max-trials must be >= 0")


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    _validate_args(args)

    cfg = SweepConfig.from_yaml(args.config)

    if args.dry_run:
        # Pure preview — no filesystem side effects. Callers wanting
        # the output dir stamped should invoke an actual run (which
        # creates `<output_dir>/sweep_config.json` via
        # `prepare_output_dir`) or call that helper directly.
        print(render_dry_run_matrix(cfg, max_trials=args.max_trials))
        return 0

    if args.report:
        out_dir = Path(cfg.output_dir)
        if not out_dir.exists():
            print(f"--report: output_dir {out_dir} does not exist", file=sys.stderr)
            return 2
        results = load_trial_results(out_dir)
        if not results:
            print(
                f"--report: no trial results found under {out_dir / 'trials'}",
                file=sys.stderr,
            )
            return 2
        json_path, md_path = write_leaderboard(out_dir, results, cfg.sweep_name)
        print(f"Re-rendered leaderboard:\n  {json_path}\n  {md_path}")
        return 0

    out_dir, results = run_sweep(
        cfg,
        max_trials=args.max_trials,
        parallel=args.parallel,
        force_stub=args.force_stub,
    )
    print(f"Sweep complete: {len(results)} trial(s) → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
