"""Tests for ``bench.loss_curve_analyzer``.

The fixture at ``bench/sample_data/sample_metrics.csv`` is constructed to
exercise every diagnostic at once: declining-then-plateau-then-diverging
loss, three sigma-4+ spikes, a 150-norm gradient blowup, a ~35%
throughput drop in the second half, decreasing routing entropy, and both
per-source columns present. We assert verdicts and key metrics rather
than exact strings so the tests survive cosmetic copy edits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench import loss_curve_analyzer as lca

SAMPLE_CSV = Path(__file__).parent / "sample_data" / "sample_metrics.csv"
SAMPLE_PROM = Path(__file__).parent / "sample_data" / "sample_prometheus.txt"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    return lca.load_metrics_csv(SAMPLE_CSV)


@pytest.fixture(scope="module")
def sample_prom() -> list[lca.PromSeries]:
    return lca.load_prometheus_snapshot(SAMPLE_PROM)


# ---------------------------------------------------------------------------
# Smoke / loader tests
# ---------------------------------------------------------------------------


def test_sample_csv_loads(sample_df: pd.DataFrame) -> None:
    assert not sample_df.empty
    # Required columns exercised by the synthetic fixture.
    for col in (
        "loss", "grad_norm", "lr", "tokens_per_second",
        "moe_routing_entropy", "c4_loss", "nemotron_loss",
    ):
        assert col in sample_df.columns, f"missing column {col} in fixture"


def test_missing_csv_returns_empty_frame(tmp_path: Path) -> None:
    df = lca.load_metrics_csv(tmp_path / "does_not_exist.csv")
    assert df.empty


def test_empty_csv_returns_empty_frame(tmp_path: Path) -> None:
    p = tmp_path / "empty.csv"
    p.write_text("")
    assert lca.load_metrics_csv(p).empty


def test_prometheus_snapshot_loads(sample_prom: list[lca.PromSeries]) -> None:
    names = {s.name for s in sample_prom}
    assert "moe_expert_load" in names
    assert "miner_training_loss" in names
    # Spot check label extraction on one labelled series.
    load_samples = [s for s in sample_prom if s.name == "moe_expert_load"]
    assert load_samples, "expected moe_expert_load samples"
    for s in load_samples:
        assert "layer_idx" in s.labels
        assert "expert_idx" in s.labels


def test_prometheus_missing_returns_empty(tmp_path: Path) -> None:
    assert lca.load_prometheus_snapshot(tmp_path / "no.txt") == []


# ---------------------------------------------------------------------------
# Individual diagnostics on the fixture
# ---------------------------------------------------------------------------


def test_diagnose_loss_trend_flags_divergence(sample_df: pd.DataFrame) -> None:
    # The fixture diverges across the final ~200 steps. With window=50 the
    # default trend slice catches the divergence segment cleanly.
    diag = lca.diagnose_loss_trend(sample_df, window=50)
    assert diag.name == "Loss trend"
    assert diag.verdict == lca.VERDICT_BAD
    # The "divergence_slope_100" metric must be positive.
    assert diag.metrics["divergence_slope_100"] > 0
    assert diag.suggestion  # non-empty


def test_diagnose_loss_trend_plateau() -> None:
    # Pure-plateau synthetic series — slope basically zero across all 800
    # samples, so the plateau branch fires (and divergence does not).
    rng = np.random.default_rng(1)
    plateau = pd.DataFrame({"loss": 1.4 + rng.normal(0.0, 0.005, size=800)})
    diag = lca.diagnose_loss_trend(plateau, window=50)
    assert diag.verdict == lca.VERDICT_WARN
    assert abs(diag.metrics["plateau_slope_500"]) < 1e-4


def test_diagnose_loss_trend_healthy() -> None:
    # Steadily decreasing loss — neither divergence nor plateau. Use 80
    # samples to stay under the 100-step divergence guard so we don't
    # cross the plateau threshold either.
    rng = np.random.default_rng(0)
    healthy = pd.DataFrame({
        "loss": np.linspace(4.0, 2.0, 80) + rng.normal(0.0, 0.01, size=80),
    })
    diag = lca.diagnose_loss_trend(healthy, window=50)
    assert diag.verdict == lca.VERDICT_OK


def test_diagnose_loss_trend_no_loss_column() -> None:
    diag = lca.diagnose_loss_trend(pd.DataFrame({"step": [1, 2, 3]}), window=50)
    assert diag.verdict == lca.VERDICT_SKIP


def test_diagnose_loss_spikes_flags_spikes(sample_df: pd.DataFrame) -> None:
    diag = lca.diagnose_loss_spikes(sample_df)
    # Three injected spikes -> verdict at minimum warn.
    assert diag.verdict in (lca.VERDICT_WARN, lca.VERDICT_BAD)
    assert diag.metrics["spike_count"] >= 1


def test_diagnose_loss_spikes_no_spikes() -> None:
    rng = np.random.default_rng(2)
    clean = pd.DataFrame({
        "loss": 2.0 + rng.normal(0.0, 0.02, size=200),
    })
    diag = lca.diagnose_loss_spikes(clean)
    assert diag.verdict == lca.VERDICT_OK
    assert diag.metrics["spike_count"] == 0


def test_diagnose_gradient_health_flags_max_over_100(sample_df: pd.DataFrame) -> None:
    diag = lca.diagnose_gradient_health(sample_df)
    assert diag.verdict == lca.VERDICT_BAD
    assert diag.metrics["grad_norm_max"] >= 100


def test_diagnose_gradient_health_healthy() -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"grad_norm": np.clip(rng.normal(0.5, 0.1, size=100), 0.01, None)})
    diag = lca.diagnose_gradient_health(df)
    assert diag.verdict == lca.VERDICT_OK


def test_diagnose_throughput_flags_drop(sample_df: pd.DataFrame) -> None:
    diag = lca.diagnose_throughput(sample_df)
    # Fixture drops ~35%; spec says >30% triggers warn.
    assert diag.verdict == lca.VERDICT_WARN
    assert diag.metrics["tps_relative_drop"] > 0.30


def test_diagnose_throughput_stable() -> None:
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"tokens_per_second": rng.normal(4500, 50, size=100)})
    diag = lca.diagnose_throughput(df)
    assert diag.verdict == lca.VERDICT_OK


def test_diagnose_expert_load_flags_imbalance(sample_prom: list[lca.PromSeries]) -> None:
    diag = lca.diagnose_expert_load(sample_prom)
    assert diag.verdict == lca.VERDICT_WARN
    assert diag.metrics["expert_load_cov"] > 0.5


def test_diagnose_expert_load_no_data() -> None:
    diag = lca.diagnose_expert_load([])
    assert diag.verdict == lca.VERDICT_SKIP


def test_diagnose_routing_entropy_decreasing(sample_df: pd.DataFrame) -> None:
    diag = lca.diagnose_routing_entropy(sample_df, [])
    assert diag.verdict == lca.VERDICT_BAD
    assert diag.metrics["routing_entropy_slope"] < 0


def test_diagnose_routing_entropy_prom_fallback() -> None:
    df = pd.DataFrame()  # no CSV columns
    prom = [lca.PromSeries(name="moe_topk_routing_entropy", labels={}, value=1.8)]
    diag = lca.diagnose_routing_entropy(df, prom)
    assert diag.verdict == lca.VERDICT_OK
    assert diag.metrics["routing_entropy_point"] == 1.8


def test_diagnose_routing_entropy_skipped() -> None:
    diag = lca.diagnose_routing_entropy(pd.DataFrame(), [])
    assert diag.verdict == lca.VERDICT_SKIP


def test_diagnose_lr_schedule_cosine_ok() -> None:
    # Build a clean cosine LR and verify it's flagged ok.
    n = 200
    base_lr = 1e-3
    lrs = [
        1e-5 + 0.5 * (base_lr - 1e-5) * (1 + np.cos(np.pi * i / (n - 1)))
        for i in range(n)
    ]
    df = pd.DataFrame({"lr": lrs})
    diag = lca.diagnose_lr_schedule(df)
    assert diag.verdict == lca.VERDICT_OK


def test_diagnose_lr_schedule_constant_warns() -> None:
    df = pd.DataFrame({"lr": [1e-4] * 50})
    diag = lca.diagnose_lr_schedule(df)
    assert diag.verdict == lca.VERDICT_WARN


def test_diagnose_source_split_present(sample_df: pd.DataFrame) -> None:
    diag = lca.diagnose_source_split(sample_df)
    # Both columns present in fixture.
    assert diag.verdict in (lca.VERDICT_OK, lca.VERDICT_WARN)
    assert "c4_loss_mean" in diag.metrics
    assert "nemotron_loss_mean" in diag.metrics


def test_diagnose_source_split_skipped() -> None:
    diag = lca.diagnose_source_split(pd.DataFrame({"step": [1, 2, 3]}))
    assert diag.verdict == lca.VERDICT_SKIP


# ---------------------------------------------------------------------------
# End-to-end report
# ---------------------------------------------------------------------------


def test_analyze_end_to_end_sample_data() -> None:
    report = lca.analyze(SAMPLE_CSV, SAMPLE_PROM, window=50)
    # We expect at least a divergence flag, a spike flag, a gradient flag,
    # a throughput flag, an expert-load flag, and a routing flag.
    flagged_names = {d.name for d in report.flagged()}
    assert "Loss trend" in flagged_names
    assert "Loss spikes" in flagged_names
    assert "Gradient health" in flagged_names
    assert "Token throughput trend" in flagged_names
    assert "Expert load imbalance" in flagged_names
    assert "Routing entropy trend" in flagged_names
    # Overall must be bad given the divergence + gradient + routing-collapse signals.
    assert report.overall_verdict == lca.VERDICT_BAD


def test_render_markdown_includes_sections() -> None:
    report = lca.analyze(SAMPLE_CSV, SAMPLE_PROM, window=50)
    md = lca.render_markdown(report)
    # Spot-check structure rather than exact strings — copy may change.
    assert md.startswith("# Training loss diagnostic report")
    assert "## Diagnostics" in md
    assert "## Recommendation" in md
    assert "Loss trend" in md
    assert "Gradient health" in md
    # Sparkline section should appear because the fixture has loss data.
    assert "## Loss sparkline" in md


def test_render_html_self_contained() -> None:
    report = lca.analyze(SAMPLE_CSV, SAMPLE_PROM, window=50, embed_png=False)
    html = lca.render_html(report)
    assert html.startswith("<!doctype html>")
    # We did not embed PNG, but the ASCII sparkline must still appear.
    assert "spark" in html
    # No external JS / CSS imports.
    assert "<script" not in html.lower()
    assert "http://" not in html
    assert "https://" not in html


def test_cli_writes_report(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    rc = lca.main([
        "--metrics-csv", str(SAMPLE_CSV),
        "--prometheus-snapshot", str(SAMPLE_PROM),
        "--out", str(out),
        "--window", "50",
    ])
    assert rc == 0
    assert out.exists()
    content = out.read_text()
    assert "Training loss diagnostic report" in content


def test_cli_writes_html(tmp_path: Path) -> None:
    md_out = tmp_path / "report.md"
    html_out = tmp_path / "report.html"
    rc = lca.main([
        "--metrics-csv", str(SAMPLE_CSV),
        "--out", str(md_out),
        "--out-html", str(html_out),
        "--window", "50",
    ])
    assert rc == 0
    assert html_out.exists()
    text = html_out.read_text()
    assert "<!doctype html>" in text


def test_cli_requires_input(tmp_path: Path) -> None:
    rc = lca.main(["--out", str(tmp_path / "x.md")])
    assert rc == 2
