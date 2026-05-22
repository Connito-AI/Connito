"""Training-loss diagnostic analyzer (Unit 5/18).

This module parses miner training logs (CSV produced by
``connito.shared.metrics.MetricLogger`` and/or a Prometheus ``/metrics``
snapshot scrape) and emits a readable markdown (and optionally HTML)
report flagging common training pathologies:

* divergence / plateau (loss trend over a rolling window),
* loss spikes (>4 sigma outliers),
* gradient health (norm magnitude + variance),
* token-throughput regressions (memory pressure / IO stalls),
* MoE expert-load imbalance,
* routing-entropy collapse,
* learning-rate schedule deviation vs cosine reference,
* per-source loss split (Unit 7 metrics if present).

The analyzer is deliberately dependency-light: it only requires numpy /
pandas at runtime, and lazily imports matplotlib only when an HTML
report with an embedded plot is requested. This keeps it usable inside
the same containers the miner runs in without dragging GUI backends.

Typical usage::

    python -m bench.loss_curve_analyzer \\
        --metrics-csv metrics/run_xyz.csv \\
        --prometheus-snapshot /tmp/scrape.txt \\
        --out loss_analysis.md \\
        --out-html loss_analysis.html

The CLI is intentionally permissive: missing columns simply mean the
corresponding diagnostic is skipped (marked ``skipped`` in the report).
Designed to never crash on partially populated logs from short runs.
"""

from __future__ import annotations

import argparse
import base64
import io
import math
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Public verdict / diagnostic dataclasses
# ---------------------------------------------------------------------------

VERDICT_OK = "ok"
VERDICT_WARN = "warn"
VERDICT_BAD = "bad"
VERDICT_SKIP = "skipped"

# Verdict severity ordering — higher is worse. Used by ``Report.summary``
# to roll the most severe per-section verdict up into the overall health.
_VERDICT_RANK: dict[str, int] = {
    VERDICT_SKIP: -1,
    VERDICT_OK: 0,
    VERDICT_WARN: 1,
    VERDICT_BAD: 2,
}


@dataclass
class Diagnostic:
    """One section of the diagnostic report.

    Attributes
    ----------
    name: short title shown in the report (e.g. ``"Loss trend"``).
    verdict: one of ``VERDICT_OK / VERDICT_WARN / VERDICT_BAD / VERDICT_SKIP``.
    details: bullet-list strings rendered verbatim in markdown.
    suggestion: one-liner pointing at the most likely knob to tweak; can
        link into the Unit 6 ``OPTIMIZATION_PLAYBOOK`` once it lands.
    metrics: optional structured numbers tests can assert on without
        having to scrape the report markdown.
    """

    name: str
    verdict: str
    details: list[str] = field(default_factory=list)
    suggestion: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def severity(self) -> int:
        return _VERDICT_RANK.get(self.verdict, -2)


@dataclass
class Report:
    """Collected diagnostics + helpers to render the final document."""

    diagnostics: list[Diagnostic] = field(default_factory=list)
    metrics_csv: str | None = None
    prometheus_snapshot: str | None = None
    n_rows: int = 0
    window: int = 50
    loss_sparkline: str = ""
    loss_png_b64: str | None = None  # only populated when an HTML report is requested

    # ----- aggregation -----
    @property
    def overall_verdict(self) -> str:
        """Roll per-section severities into the overall health verdict."""

        worst = VERDICT_OK
        for d in self.diagnostics:
            if d.severity() > _VERDICT_RANK[worst]:
                worst = d.verdict
        # If every section was skipped, prefer ``skipped`` over ``ok`` so the
        # reader doesn't mistake an empty run for a healthy one.
        if all(d.verdict == VERDICT_SKIP for d in self.diagnostics) and self.diagnostics:
            return VERDICT_SKIP
        return worst

    def flagged(self) -> list[Diagnostic]:
        """Subset of diagnostics whose verdict is warn or bad."""

        return [d for d in self.diagnostics if d.verdict in (VERDICT_WARN, VERDICT_BAD)]


# ---------------------------------------------------------------------------
# CSV / Prometheus loading
# ---------------------------------------------------------------------------

# CSV columns we know how to read. Anything else is preserved but unused.
_KNOWN_LOSS_COLS = ("loss", "training_loss", "miner_training_loss", "train_loss")
_KNOWN_VAL_LOSS_COLS = ("val_loss", "validation_loss", "miner_val_loss")
_KNOWN_STEP_COLS = ("inner_opt_step", "step", "global_opt_step")
_KNOWN_LR_COLS = ("lr", "learning_rate", "miner_learning_rate")
_KNOWN_GRAD_COLS = ("grad_norm", "gradient_norm", "miner_gradient_norm")
_KNOWN_TPS_COLS = ("tokens_per_second", "miner_tokens_per_sec", "tokens_per_sec")
_KNOWN_AUX_COLS = ("aux_loss", "moe_aux_loss")
_KNOWN_ROUTING_ENTROPY_COLS = ("moe_routing_entropy", "moe_topk_routing_entropy", "routing_entropy")

# Per-source loss split — Unit 7 emits these columns when its metric
# adapter is wired into the miner. They are optional and silently skipped
# when absent.
_KNOWN_C4_LOSS_COLS = ("c4_loss", "loss_c4", "source_c4_loss")
_KNOWN_NEMOTRON_LOSS_COLS = ("nemotron_loss", "loss_nemotron", "source_nemotron_loss")


def _first_present(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    """Return the first column from ``names`` that exists in ``df``."""

    for name in names:
        if name in df.columns:
            return name
    return None


def load_metrics_csv(path: str | Path) -> pd.DataFrame:
    """Load a MetricLogger CSV into a pandas DataFrame.

    Returns an empty DataFrame (and warns) if the file is missing or empty
    rather than raising; downstream diagnostics gracefully degrade.
    """

    p = Path(path)
    if not p.exists():
        logger.warning(f"metrics CSV not found at {p}; returning empty frame")
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
    except pd.errors.EmptyDataError:
        logger.warning(f"metrics CSV at {p} is empty; returning empty frame")
        return pd.DataFrame()
    except Exception as exc:
        logger.warning(f"failed to parse metrics CSV at {p}: {exc}; returning empty frame")
        return pd.DataFrame()
    # Coerce known numeric columns; ignore failures so string-only rows
    # (e.g. expert_group_name) don't blow up the cast.
    for col in df.columns:
        if df[col].dtype == object:
            coerced = pd.to_numeric(df[col], errors="coerce")
            # Only swap if at least one value coerced cleanly — keeps purely
            # categorical columns intact (expert_group_name, etc.).
            if coerced.notna().any():
                df[col] = coerced
    return df


# Prometheus exposition format parser. Intentionally minimal — we only
# care about lines that look like ``metric_name{labels...} value`` and we
# ignore HELP/TYPE comments and histograms (their _bucket lines are
# decomposed but their aggregate _sum/_count remain). The full parser in
# ``prometheus_client.parser`` would also work, but rolling our own keeps
# the diagnostic dependency-free and easy to test.
_PROM_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?"
    r"\s+(?P<value>-?[\d\.eE+inafNAN]+)"
    r"(?:\s+\d+)?$"  # optional timestamp; ignored
)
_PROM_LABEL_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)=\"((?:[^\"\\]|\\.)*)\"")


@dataclass
class PromSeries:
    name: str
    labels: dict[str, str]
    value: float


def load_prometheus_snapshot(path: str | Path) -> list[PromSeries]:
    """Parse a Prometheus exposition-format scrape into a flat list.

    Returns an empty list if the file doesn't exist or can't be parsed —
    same forgiving behaviour as ``load_metrics_csv``.
    """

    p = Path(path)
    if not p.exists():
        logger.warning(f"prometheus snapshot not found at {p}; skipping")
        return []
    try:
        text = p.read_text()
    except Exception as exc:
        logger.warning(f"failed to read prometheus snapshot {p}: {exc}; skipping")
        return []

    out: list[PromSeries] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = _PROM_LINE_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        labels_raw = m.group("labels") or ""
        labels: dict[str, str] = {}
        for lm in _PROM_LABEL_RE.finditer(labels_raw):
            labels[lm.group(1)] = lm.group(2)
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        out.append(PromSeries(name=name, labels=labels, value=value))
    return out


# ---------------------------------------------------------------------------
# Individual diagnostics
# ---------------------------------------------------------------------------


def _safe_series(df: pd.DataFrame, col: str | None) -> np.ndarray:
    """Return the column as a finite float ndarray (NaN/Inf dropped)."""

    if col is None or col not in df.columns:
        return np.array([], dtype=float)
    arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def _linregress_slope(y: np.ndarray) -> tuple[float, float]:
    """Plain-NumPy linear regression slope + intercept.

    SciPy is already a project dependency, but keeping this in pure NumPy
    sidesteps having to import the heavier ``scipy.stats`` just for a
    slope. Returns ``(slope, intercept)`` of ``y`` vs ``range(len(y))``.
    Returns ``(0.0, 0.0)`` for inputs shorter than 2 points.
    """

    n = y.size
    if n < 2:
        return 0.0, 0.0
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = float(((x - x_mean) ** 2).sum())
    if denom == 0.0:
        return 0.0, float(y_mean)
    slope = float(((x - x_mean) * (y - y_mean)).sum() / denom)
    intercept = float(y_mean - slope * x_mean)
    return slope, intercept


def diagnose_loss_trend(df: pd.DataFrame, *, window: int) -> Diagnostic:
    """Linear-regression slope + mean/std over the last ``window`` steps.

    Flags ``bad`` when the slope is positive for >100 steps (divergence)
    or when ``|slope|`` is below ``1e-4`` for >500 steps (plateau).
    """

    col = _first_present(df, _KNOWN_LOSS_COLS)
    loss = _safe_series(df, col)
    if loss.size == 0:
        return Diagnostic(
            name="Loss trend",
            verdict=VERDICT_SKIP,
            details=["No loss column found in CSV (looked for `loss`, `training_loss`, ...)."]
        )

    tail = loss[-max(window, 2):]
    slope, _ = _linregress_slope(tail)
    mean = float(tail.mean())
    std = float(tail.std(ddof=0))

    diag = Diagnostic(
        name="Loss trend",
        verdict=VERDICT_OK,
        metrics={
            "loss_mean": mean,
            "loss_std": std,
            "loss_slope": slope,
            "window": int(tail.size),
        },
    )

    # Long-horizon divergence: slope positive across the last 100+ steps.
    # We re-run the regression on the full tail of size min(len, 100) so
    # the user-supplied ``window`` doesn't accidentally hide a 200-step
    # divergence at default window=50.
    diverge_tail = loss[-max(100, window):]
    diverge_slope, _ = _linregress_slope(diverge_tail)
    plateau_tail = loss[-max(500, window):]
    plateau_slope, _ = _linregress_slope(plateau_tail)

    # Distinguish noisy-but-flat from real divergence: positive slope above
    # the plateau threshold (1e-4/step) qualifies as divergence, anything
    # below stays in the plateau / OK branches.
    if loss.size >= 100 and diverge_slope > 1e-4:
        diag.verdict = VERDICT_BAD
        diag.details.append(
            f"Loss slope over last {min(loss.size, max(100, window))} steps is positive "
            f"({diverge_slope:+.6f}/step). Likely divergence."
        )
        diag.suggestion = "Lower LR (try 0.5x), tighten grad clip, or reduce micro-batch tokens."
    elif loss.size >= 500 and abs(plateau_slope) < 1e-4:
        diag.verdict = VERDICT_WARN
        diag.details.append(
            f"Loss slope across last {min(loss.size, max(500, window))} steps is ~0 "
            f"({plateau_slope:+.6f}/step). Plateau."
        )
        diag.suggestion = "Increase LR briefly (warm-up restart) or add curriculum data."
    else:
        diag.details.append(
            f"Slope over last {tail.size} steps = {slope:+.6f}/step "
            f"(mean={mean:.4f}, std={std:.4f}). No divergence/plateau flag."
        )
        if slope < 0:
            diag.details.append("Loss is decreasing — good.")
    diag.metrics["divergence_slope_100"] = diverge_slope
    diag.metrics["plateau_slope_500"] = plateau_slope
    return diag


def diagnose_loss_spikes(df: pd.DataFrame, *, sigma_threshold: float = 4.0) -> Diagnostic:
    """Count batches where ``loss > mean + sigma_threshold * std``.

    The 4-sigma cutoff matches the spec; spikes beyond that almost always
    indicate either malformed batches or a brittle optimizer state.
    """

    col = _first_present(df, _KNOWN_LOSS_COLS)
    loss = _safe_series(df, col)
    if loss.size < 10:
        return Diagnostic(
            name="Loss spikes",
            verdict=VERDICT_SKIP,
            details=["Need at least 10 finite loss samples to flag spikes."],
        )

    mean = float(loss.mean())
    std = float(loss.std(ddof=0))
    threshold = mean + sigma_threshold * std
    spike_mask = loss > threshold
    spike_count = int(spike_mask.sum())

    # Pull step numbers if present for the listing.
    step_col = _first_present(df, _KNOWN_STEP_COLS)
    spike_locations: list[str] = []
    if step_col is not None:
        # Recompute mask against the un-NaN-filtered series so step
        # indexing lines up with rows.
        raw_loss = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        steps = pd.to_numeric(df[step_col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(raw_loss)
        # Walk only finite rows.
        for s, lv in zip(steps[finite], raw_loss[finite], strict=False):
            if lv > threshold:
                spike_locations.append(f"step={int(s) if math.isfinite(s) else '?'} loss={lv:.4f}")
    else:
        for idx, lv in enumerate(loss):
            if lv > threshold:
                spike_locations.append(f"row={idx} loss={lv:.4f}")

    diag = Diagnostic(
        name="Loss spikes",
        verdict=VERDICT_OK,
        metrics={
            "spike_count": spike_count,
            "threshold": threshold,
            "mean": mean,
            "std": std,
        },
    )
    if spike_count == 0:
        diag.details.append(
            f"No batches > mean+{sigma_threshold:.0f}sigma (= {threshold:.4f}) in {loss.size} samples."
        )
        return diag

    # Even one spike is worth noting; > 0.5% of batches we treat as bad.
    if spike_count / loss.size > 0.005:
        diag.verdict = VERDICT_BAD
        diag.suggestion = "Tighten gradient clipping; inspect data loader for malformed batches."
    else:
        diag.verdict = VERDICT_WARN
        diag.suggestion = "Add gradient clipping if not already enabled; inspect tokenization."
    listed = spike_locations[:10]
    diag.details.append(
        f"{spike_count} spikes (> {threshold:.4f}). First {len(listed)}: "
        + "; ".join(listed)
        + ("..." if spike_count > len(listed) else "")
    )
    return diag


def diagnose_gradient_health(df: pd.DataFrame) -> Diagnostic:
    """Check gradient norm magnitude + variance for instability signals.

    Flags ``bad`` when any ``grad_norm`` exceeds 100 (very likely
    instability), or when the std/mean ratio exceeds 3 (high variance).
    """

    col = _first_present(df, _KNOWN_GRAD_COLS)
    arr = _safe_series(df, col)
    if arr.size == 0:
        return Diagnostic(
            name="Gradient health",
            verdict=VERDICT_SKIP,
            details=["No `grad_norm` column found in CSV."],
        )

    mx = float(arr.max())
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    ratio = float(std / mean) if mean > 0 else 0.0

    diag = Diagnostic(
        name="Gradient health",
        verdict=VERDICT_OK,
        metrics={
            "grad_norm_max": mx,
            "grad_norm_mean": mean,
            "grad_norm_std": std,
            "grad_norm_cv": ratio,
        },
    )
    if mx > 100:
        diag.verdict = VERDICT_BAD
        diag.details.append(
            f"Max grad_norm = {mx:.2f} (> 100). Instability risk."
        )
        diag.suggestion = "Add grad_clip=1.0 in optimizer; lower LR."
    elif ratio > 3.0:
        diag.verdict = VERDICT_WARN
        diag.details.append(
            f"grad_norm std/mean = {ratio:.2f} (> 3.0). High variance."
        )
        diag.suggestion = "Investigate sources of unstable gradients (data, dtype mismatch)."
    else:
        diag.details.append(
            f"grad_norm: max={mx:.2f}, mean={mean:.4f}, std={std:.4f}, cv={ratio:.2f}. Healthy."
        )
    return diag


def diagnose_throughput(df: pd.DataFrame) -> Diagnostic:
    """Throughput regression detector.

    Compares the mean tokens/sec across the first vs second half of the
    log. A >30% drop is flagged ``warn`` (memory pressure / IO stalls).
    """

    col = _first_present(df, _KNOWN_TPS_COLS)
    arr = _safe_series(df, col)
    if arr.size < 4:
        return Diagnostic(
            name="Token throughput trend",
            verdict=VERDICT_SKIP,
            details=["Not enough tokens/sec samples (>=4 required)."],
        )

    half = arr.size // 2
    first_mean = float(arr[:half].mean())
    second_mean = float(arr[half:].mean())
    drop = 0.0 if first_mean <= 0 else float((first_mean - second_mean) / first_mean)

    diag = Diagnostic(
        name="Token throughput trend",
        verdict=VERDICT_OK,
        metrics={
            "tps_first_half_mean": first_mean,
            "tps_second_half_mean": second_mean,
            "tps_relative_drop": drop,
        },
    )
    if drop > 0.30:
        diag.verdict = VERDICT_WARN
        diag.details.append(
            f"Tokens/sec dropped {drop * 100:.1f}% from {first_mean:.0f} -> {second_mean:.0f}. "
            "Memory pressure or IO stall."
        )
        diag.suggestion = "Inspect VRAM trend, reduce sequence length, or restart dataloader workers."
    else:
        diag.details.append(
            f"Tokens/sec: first half {first_mean:.0f}, second half {second_mean:.0f} "
            f"(delta {drop * 100:+.1f}%). Stable."
        )
    return diag


def diagnose_expert_load(prom: Sequence[PromSeries]) -> Diagnostic:
    """Coefficient of variation across MoE expert load.

    Reads ``moe_expert_load{layer_idx, expert_idx}`` from the Prometheus
    snapshot. Flags ``warn`` when CoV > 0.5 (uneven routing).
    """

    series = [s for s in prom if s.name == "moe_expert_load"]
    if not series:
        return Diagnostic(
            name="Expert load imbalance",
            verdict=VERDICT_SKIP,
            details=["No `moe_expert_load` samples in prometheus snapshot."],
        )

    loads = np.array([s.value for s in series], dtype=float)
    loads = loads[np.isfinite(loads)]
    if loads.size == 0 or loads.sum() == 0:
        return Diagnostic(
            name="Expert load imbalance",
            verdict=VERDICT_SKIP,
            details=["Expert load samples are all zero or NaN."],
        )

    mean = float(loads.mean())
    std = float(loads.std(ddof=0))
    cov = float(std / mean) if mean > 0 else 0.0

    diag = Diagnostic(
        name="Expert load imbalance",
        verdict=VERDICT_OK,
        metrics={
            "expert_load_mean": mean,
            "expert_load_std": std,
            "expert_load_cov": cov,
            "experts_observed": int(loads.size),
        },
    )
    if cov > 0.5:
        diag.verdict = VERDICT_WARN
        diag.details.append(
            f"CoV across {loads.size} expert/layer pairs = {cov:.2f} (> 0.5). Routing imbalance."
        )
        diag.suggestion = "Raise router aux-loss weight; check for dead experts."
    else:
        diag.details.append(
            f"CoV across {loads.size} expert/layer pairs = {cov:.2f}. Balanced."
        )
    return diag


def diagnose_routing_entropy(df: pd.DataFrame, prom: Sequence[PromSeries]) -> Diagnostic:
    """Watch routing entropy. Decreasing entropy = routing collapse.

    Uses the CSV column when available (richer time series). Falls back
    to the single point from the Prometheus snapshot when the CSV doesn't
    carry routing entropy.
    """

    col = _first_present(df, _KNOWN_ROUTING_ENTROPY_COLS)
    arr = _safe_series(df, col)

    if arr.size >= 4:
        slope, _ = _linregress_slope(arr)
        last_mean = float(arr[-max(arr.size // 5, 1):].mean())
        diag = Diagnostic(
            name="Routing entropy trend",
            verdict=VERDICT_OK,
            metrics={
                "routing_entropy_slope": slope,
                "routing_entropy_last": last_mean,
            },
        )
        if slope < -1e-4 and arr.size >= 50:
            diag.verdict = VERDICT_BAD
            diag.details.append(
                f"Routing entropy decreasing ({slope:+.6f}/step). Routing collapsing."
            )
            diag.suggestion = "Boost aux-loss; consider router-z-loss; reset expert weights."
        else:
            diag.details.append(
                f"Routing entropy slope {slope:+.6f}/step, recent mean {last_mean:.4f}. OK."
            )
        return diag

    # Prometheus fallback — only a single point but still worth surfacing.
    point = next((s.value for s in prom if s.name in ("moe_routing_entropy", "moe_topk_routing_entropy")), None)
    if point is None:
        return Diagnostic(
            name="Routing entropy trend",
            verdict=VERDICT_SKIP,
            details=["No routing-entropy column in CSV nor sample in prometheus snapshot."],
        )
    return Diagnostic(
        name="Routing entropy trend",
        verdict=VERDICT_OK,
        details=[f"Single prometheus sample: routing entropy = {float(point):.4f}. No trend available."],
        metrics={"routing_entropy_point": float(point)},
    )


def _expected_cosine_lr(step: int, base_lr: float, total_steps: int, min_lr: float = 0.0) -> float:
    """Reference cosine LR schedule. Used purely for visual comparison."""

    if total_steps <= 0:
        return base_lr
    progress = min(max(step / total_steps, 0.0), 1.0)
    return float(min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress)))


def diagnose_lr_schedule(df: pd.DataFrame) -> Diagnostic:
    """Check if observed LR roughly follows a cosine decay.

    We don't know the user's intended schedule for sure, so we compare to
    a cosine fit and flag ``warn`` only when the RMS deviation exceeds 20%
    of the max LR seen. That tolerance keeps false positives down for
    linear-decay or constant-LR runs while still catching e.g. an LR that
    silently froze due to a bug.
    """

    lr_col = _first_present(df, _KNOWN_LR_COLS)
    arr = _safe_series(df, lr_col)
    if arr.size < 8:
        return Diagnostic(
            name="LR schedule sanity",
            verdict=VERDICT_SKIP,
            details=["Need at least 8 LR samples to compare to a cosine reference."],
        )

    base_lr = float(arr.max())
    expected = np.array(
        [_expected_cosine_lr(i, base_lr, arr.size - 1) for i in range(arr.size)],
        dtype=float,
    )
    rms = float(np.sqrt(np.mean((arr - expected) ** 2)))
    rel_dev = rms / base_lr if base_lr > 0 else 0.0

    diag = Diagnostic(
        name="LR schedule sanity",
        verdict=VERDICT_OK,
        metrics={
            "lr_base": base_lr,
            "lr_rms_dev_vs_cosine": rms,
            "lr_relative_dev": rel_dev,
            "lr_first": float(arr[0]),
            "lr_last": float(arr[-1]),
        },
    )
    if arr.size >= 8 and float(arr.std(ddof=0)) == 0.0:
        diag.verdict = VERDICT_WARN
        diag.details.append(
            f"LR is constant at {arr[0]:.2e}. If a schedule was intended, it never engaged."
        )
        diag.suggestion = "Verify scheduler.step() is called each optimizer step."
    elif rel_dev > 0.20:
        diag.verdict = VERDICT_WARN
        diag.details.append(
            f"Observed LR diverges from cosine reference (rel_dev={rel_dev:.2%}). "
            f"Schedule probably linear / custom — verify intent."
        )
        diag.suggestion = "Inspect scheduler configuration; ensure warmup + decay match plan."
    else:
        diag.details.append(
            f"LR follows cosine within {rel_dev:.2%}. Start={arr[0]:.2e}, end={arr[-1]:.2e}."
        )
    return diag


def diagnose_source_split(df: pd.DataFrame) -> Diagnostic:
    """Per-source loss split (C4 vs Nemotron) if Unit 7 metrics present."""

    c4_col = _first_present(df, _KNOWN_C4_LOSS_COLS)
    nemo_col = _first_present(df, _KNOWN_NEMOTRON_LOSS_COLS)
    if c4_col is None and nemo_col is None:
        return Diagnostic(
            name="Per-source loss split",
            verdict=VERDICT_SKIP,
            details=["Unit 7 per-source loss columns not present (skipped)."],
        )

    c4 = _safe_series(df, c4_col)
    nemo = _safe_series(df, nemo_col)

    diag = Diagnostic(
        name="Per-source loss split",
        verdict=VERDICT_OK,
        metrics={},
    )
    if c4.size:
        diag.metrics["c4_loss_mean"] = float(c4.mean())
        diag.metrics["c4_loss_last"] = float(c4[-1])
    if nemo.size:
        diag.metrics["nemotron_loss_mean"] = float(nemo.mean())
        diag.metrics["nemotron_loss_last"] = float(nemo[-1])

    if c4.size and nemo.size:
        diff = abs(float(c4.mean()) - float(nemo.mean()))
        diag.metrics["abs_source_loss_gap"] = diff
        diag.details.append(
            f"C4 mean={c4.mean():.4f}, Nemotron mean={nemo.mean():.4f}, |gap|={diff:.4f}."
        )
        if diff > 1.0:
            diag.verdict = VERDICT_WARN
            diag.suggestion = "Investigate source mix or per-source LR; one corpus may dominate."
    else:
        seen = []
        if c4.size:
            seen.append(f"C4 mean={c4.mean():.4f}")
        if nemo.size:
            seen.append(f"Nemotron mean={nemo.mean():.4f}")
        diag.details.append(", ".join(seen) + ". One source present; gap unavailable.")
    return diag


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

# Tiny ASCII sparkline. Useful for the markdown output where embedding a
# PNG is awkward. 8 levels keeps it readable in monospaced terminals.
_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: Sequence[float], width: int = 60) -> str:
    """Return an ASCII sparkline of up to ``width`` blocks for ``values``."""

    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return ""
    if len(vals) > width:
        # Downsample by averaging chunks so we keep shape rather than
        # picking every Nth value (which can hide spikes).
        chunk = len(vals) / width
        sampled = []
        for i in range(width):
            lo = int(round(i * chunk))
            hi = int(round((i + 1) * chunk))
            if hi <= lo:
                hi = lo + 1
            sampled.append(sum(vals[lo:hi]) / max(1, hi - lo))
        vals = sampled
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return _SPARK_CHARS[0] * len(vals)
    chars = []
    n = len(_SPARK_CHARS) - 1
    for v in vals:
        idx = int(round((v - lo) / (hi - lo) * n))
        idx = max(0, min(n, idx))
        chars.append(_SPARK_CHARS[idx])
    return "".join(chars)


def _verdict_glyph(v: str) -> str:
    """Map verdict strings onto colour names (red/yellow/green) for the report.

    We use words rather than emoji per repo convention: assistants should
    avoid emojis in committed files.
    """

    return {
        VERDICT_OK: "green",
        VERDICT_WARN: "yellow",
        VERDICT_BAD: "red",
        VERDICT_SKIP: "gray",
    }.get(v, "gray")


def _render_loss_png_base64(df: pd.DataFrame) -> str | None:
    """Best-effort PNG plot of the loss curve, returned base64-encoded.

    Returns ``None`` if matplotlib isn't installed or the loss column is
    missing. Used only for the optional HTML output; the markdown path
    uses the ASCII sparkline instead.
    """

    col = _first_present(df, _KNOWN_LOSS_COLS)
    if col is None:
        return None
    arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None

    try:
        import matplotlib  # type: ignore[import-not-found]

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception as exc:
        logger.warning(f"matplotlib unavailable, embedding PNG skipped: {exc}")
        return None

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(arr, linewidth=1.2)
    ax.set_xlabel("sample index")
    ax.set_ylabel("loss")
    ax.set_title("Training loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _format_metrics_block(metrics: dict[str, Any]) -> str:
    if not metrics:
        return ""
    lines: list[str] = []
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"  - `{k}` = {v:.6g}")
        else:
            lines.append(f"  - `{k}` = {v}")
    return "\n".join(lines)


def render_markdown(report: Report) -> str:
    overall = report.overall_verdict
    parts: list[str] = []
    parts.append("# Training loss diagnostic report")
    parts.append("")
    parts.append(f"**Overall health:** `{overall}` ({_verdict_glyph(overall)})")
    parts.append("")
    parts.append("## Inputs")
    parts.append(f"- Metrics CSV: `{report.metrics_csv or 'n/a'}`")
    parts.append(f"- Prometheus snapshot: `{report.prometheus_snapshot or 'n/a'}`")
    parts.append(f"- Rows analysed: {report.n_rows}")
    parts.append(f"- Rolling window: {report.window}")
    parts.append("")

    if report.loss_sparkline:
        parts.append("## Loss sparkline")
        parts.append("```")
        parts.append(report.loss_sparkline)
        parts.append("```")
        parts.append("")

    parts.append("## Diagnostics")
    if not report.diagnostics:
        parts.append("_No diagnostics produced._")
    for d in report.diagnostics:
        parts.append(f"### {d.name} — `{d.verdict}` ({_verdict_glyph(d.verdict)})")
        for line in d.details:
            parts.append(f"- {line}")
        if d.metrics:
            parts.append("- metrics:")
            parts.append(_format_metrics_block(d.metrics))
        if d.suggestion:
            parts.append(f"- **suggestion**: {d.suggestion}")
        parts.append("")

    parts.append("## Recommendation")
    flagged = report.flagged()
    if not flagged:
        parts.append("- No actionable findings. Continue current training schedule.")
    else:
        worst = max(flagged, key=lambda d: d.severity())
        parts.append(
            f"- Highest-severity finding: **{worst.name}** (`{worst.verdict}`). "
            f"Suggested next knob: {worst.suggestion or 'see section above'}."
        )
        parts.append(
            "- See `OPTIMIZATION_PLAYBOOK.md` (Unit 6) for the full mapping "
            "from diagnostic verdict to recommended optimization knob."
        )
    return "\n".join(parts) + "\n"


def render_html(report: Report) -> str:
    """Render an HTML version of the report using a tiny inline template.

    Self-contained: no JS, no external CSS — only minimal inline styles so
    it renders consistently in a browser or via ``file://``.
    """

    overall = report.overall_verdict
    rows: list[str] = []
    for d in report.diagnostics:
        details_html = "".join(f"<li>{_escape_html(line)}</li>" for line in d.details)
        metrics_html = ""
        if d.metrics:
            metric_items = "".join(
                f"<li><code>{_escape_html(str(k))}</code> = {_escape_html(_fmt_metric_value(v))}</li>"
                for k, v in d.metrics.items()
            )
            metrics_html = f"<details><summary>metrics</summary><ul>{metric_items}</ul></details>"
        suggestion_html = ""
        if d.suggestion:
            suggestion_html = f"<p><strong>Suggestion:</strong> {_escape_html(d.suggestion)}</p>"
        # Escape verdict before injecting into HTML class attribute / text;
        # the Diagnostic.verdict field is typed `str`, so a future caller
        # passing arbitrary strings must not be able to break the markup.
        verdict_attr = _escape_html(d.verdict)
        rows.append(
            f"<section class='diag verdict-{verdict_attr}'>"
            f"<h3>{_escape_html(d.name)} — <code>{verdict_attr}</code></h3>"
            f"<ul>{details_html}</ul>{metrics_html}{suggestion_html}"
            f"</section>"
        )

    png_html = ""
    if report.loss_png_b64:
        png_html = (
            "<figure><img alt='Training loss curve' "
            f"src='data:image/png;base64,{report.loss_png_b64}' "
            "style='max-width:800px;border:1px solid #ddd;'/></figure>"
        )
    elif report.loss_sparkline:
        png_html = f"<pre class='spark'>{_escape_html(report.loss_sparkline)}</pre>"

    flagged = report.flagged()
    recommendation_html: str
    if not flagged:
        recommendation_html = "<p>No actionable findings. Continue current training schedule.</p>"
    else:
        worst = max(flagged, key=lambda d: d.severity())
        recommendation_html = (
            f"<p>Highest-severity finding: <strong>{_escape_html(worst.name)}</strong> "
            f"(<code>{worst.verdict}</code>). Suggested knob: "
            f"{_escape_html(worst.suggestion or 'see section above')}.</p>"
            "<p>See <code>OPTIMIZATION_PLAYBOOK.md</code> (Unit 6).</p>"
        )

    overall_attr = _escape_html(overall)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Loss diagnostic report</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 2em auto;
          max-width: 900px; line-height: 1.5; }}
  h1, h2, h3 {{ color: #222; }}
  pre, code {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; }}
  pre.spark {{ background: #f6f6f6; padding: 0.6em; }}
  section.diag {{ border-left: 4px solid #aaa; padding-left: 1em;
                  margin: 1em 0; }}
  .verdict-ok {{ border-left-color: #2ca02c; }}
  .verdict-warn {{ border-left-color: #f1c40f; }}
  .verdict-bad {{ border-left-color: #e74c3c; }}
  .verdict-skipped {{ border-left-color: #777; }}
  .overall.verdict-ok {{ color: #2ca02c; }}
  .overall.verdict-warn {{ color: #f1c40f; }}
  .overall.verdict-bad {{ color: #e74c3c; }}
  .overall.verdict-skipped {{ color: #777; }}
</style>
</head>
<body>
<h1>Training loss diagnostic report</h1>
<p>Overall health: <strong class="overall verdict-{overall_attr}">{overall_attr}</strong>
  ({_verdict_glyph(overall)})</p>
<h2>Inputs</h2>
<ul>
  <li>Metrics CSV: <code>{_escape_html(str(report.metrics_csv or 'n/a'))}</code></li>
  <li>Prometheus snapshot: <code>{_escape_html(str(report.prometheus_snapshot or 'n/a'))}</code></li>
  <li>Rows analysed: {report.n_rows}</li>
  <li>Rolling window: {report.window}</li>
</ul>
<h2>Loss curve</h2>
{png_html}
<h2>Diagnostics</h2>
{''.join(rows) if rows else '<p>No diagnostics produced.</p>'}
<h2>Recommendation</h2>
{recommendation_html}
</body>
</html>
"""


def _escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fmt_metric_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def analyze(
    metrics_csv: str | Path | None,
    prometheus_snapshot: str | Path | None,
    window: int = 50,
    *,
    embed_png: bool = False,
) -> Report:
    """Run all diagnostics against the given inputs and return a Report."""

    df = load_metrics_csv(metrics_csv) if metrics_csv else pd.DataFrame()
    prom = load_prometheus_snapshot(prometheus_snapshot) if prometheus_snapshot else []

    report = Report(
        metrics_csv=str(metrics_csv) if metrics_csv else None,
        prometheus_snapshot=str(prometheus_snapshot) if prometheus_snapshot else None,
        n_rows=int(len(df)),
        window=int(window),
    )

    # Sparkline of the loss column (if any). Computed once because it's
    # used in both markdown and HTML output paths.
    loss_col = _first_present(df, _KNOWN_LOSS_COLS)
    if loss_col is not None:
        report.loss_sparkline = _sparkline(_safe_series(df, loss_col).tolist())

    report.diagnostics = [
        diagnose_loss_trend(df, window=window),
        diagnose_loss_spikes(df),
        diagnose_gradient_health(df),
        diagnose_throughput(df),
        diagnose_lr_schedule(df),
        diagnose_routing_entropy(df, prom),
        diagnose_expert_load(prom),
        diagnose_source_split(df),
    ]

    if embed_png:
        report.loss_png_b64 = _render_loss_png_base64(df)

    return report


def write_outputs(report: Report, *, out_md: Path, out_html: Path | None) -> None:
    # Force UTF-8 — markdown sparkline (U+2581 ... U+2588) and em-dashes
    # in section titles fail with PlatformEncoding (cp1252) on Windows
    # or in containers with LANG=C.
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(report), encoding="utf-8")
    logger.info(f"Markdown report written to {out_md}")
    if out_html is not None:
        out_html.parent.mkdir(parents=True, exist_ok=True)
        out_html.write_text(render_html(report), encoding="utf-8")
        logger.info(f"HTML report written to {out_html}")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench.loss_curve_analyzer",
        description="Diagnose training loss anomalies from MetricLogger CSV / Prometheus snapshot.",
    )
    p.add_argument("--metrics-csv", type=Path, default=None, help="Primary CSV log file.")
    p.add_argument(
        "--prometheus-snapshot",
        type=Path,
        default=None,
        help="Optional Prometheus /metrics scrape (text exposition format).",
    )
    p.add_argument(
        "--out", type=Path, default=Path("loss_analysis.md"),
        help="Output markdown report path (default: ./loss_analysis.md).",
    )
    p.add_argument(
        "--out-html", type=Path, default=None,
        help="Optional HTML report path with embedded PNG.",
    )
    p.add_argument(
        "--window", type=int, default=50,
        help="Rolling window for trend detection (default: 50 steps).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.metrics_csv is None and args.prometheus_snapshot is None:
        logger.error(
            "Pass at least one of --metrics-csv / --prometheus-snapshot. Nothing to analyse."
        )
        return 2

    report = analyze(
        metrics_csv=args.metrics_csv,
        prometheus_snapshot=args.prometheus_snapshot,
        window=args.window,
        embed_png=args.out_html is not None,
    )
    write_outputs(report, out_md=args.out, out_html=args.out_html)
    flagged = report.flagged()
    logger.info(
        f"Analysis complete: overall={report.overall_verdict}, flagged={len(flagged)} "
        f"of {len(report.diagnostics)} sections."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
