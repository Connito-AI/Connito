"""Connito Subnet Overview — Grafana dashboard generator.

Generates the JSON for the "Connito Subnet Overview" board. Every metric
and label referenced here must exist in connito/shared/telemetry.py —
treat those two files as a contract: any rename / removal there must be
mirrored here, or the panel renders empty.

Generate JSON (either form works):

    pip install grafanalib==0.7.1
    cd observability/grafana/dashboards

    # Form 1 — grafanalib CLI. The `<name>.<gtype>.py` filename convention
    # tells the CLI to look for a variable named `dashboard` in this file.
    generate-dashboard -o connito_subnet_overview.json connito_subnet_overview.dashboard.py

    # Form 2 — run as a script (uses the __main__ block at the bottom).
    python connito_subnet_overview.dashboard.py > connito_subnet_overview.json
"""

from grafanalib.core import (
    BarChart,
    Dashboard,
    GaugePanel,
    GridPos,
    RowPanel,
    Stat,
    Target,
    Threshold,
    Time,
    TimeSeries,
)


# Match the datasource name configured in
# observability/grafana/provisioning/datasources/. If the provisioned name
# differs (e.g. uses a UID), update here.
DATASOURCE = "Prometheus"


def _row(title: str, y: int) -> RowPanel:
    """Section header that spans the full grid width."""
    return RowPanel(title=title, gridPos=GridPos(h=1, w=24, x=0, y=y))


# ============================================================================
# Row 1: Macro Subnet Health (y = 0..8)
# ============================================================================

current_phase_block = Stat(
    title="Current Phase & Block",
    description="Current subtensor block and the active cycle phase index.",
    dataSource=DATASOURCE,
    targets=[
        Target(expr="subnet_current_block", legendFormat="Block", refId="A"),
        Target(expr="subnet_current_phase_index", legendFormat="Phase", refId="B"),
    ],
    gridPos=GridPos(h=8, w=8, x=0, y=1),
    colorMode="value",
    graphMode="none",
    reduceCalc="lastNotNull",
    orientation="horizontal",
    textMode="value_and_name",
)

subnet_vtrust = GaugePanel(
    title="Subnet VTrust",
    description="Validator trust score for this validator's UID. Range 0..1; higher is healthier.",
    dataSource=DATASOURCE,
    targets=[Target(expr="subnet_validator_vtrust", refId="A")],
    gridPos=GridPos(h=8, w=8, x=8, y=1),
    min=0,
    max=1,
    thresholdMarkers=True,
    # Tune once the steady-state range of a healthy validator on this subnet
    # is known. Defaults are conservative.
    thresholds=[
        Threshold("red",    index=0, value=0.0),
        Threshold("orange", index=1, value=0.4),
        Threshold("green",  index=2, value=0.7),
    ],
)

deregistrations = TimeSeries(
    title="Deregistrations (rate, 5m)",
    description="UIDs deregistered per second, averaged over a 5-minute window.",
    dataSource=DATASOURCE,
    targets=[
        Target(
            expr="rate(subnet_uid_deregistrations_total[5m])",
            legendFormat="UIDs/s",
            refId="A",
        ),
    ],
    gridPos=GridPos(h=8, w=8, x=16, y=1),
    drawStyle="bars",
    unit="short",
    lineWidth=1,
    fillOpacity=80,
)


# ============================================================================
# Row 2: Validator Operations (y = 9..17)
# ============================================================================

heartbeat = TimeSeries(
    title="Validator Heartbeat",
    description=(
        "Iterations per second through the validator outer loop. "
        "Alert on `rate(...[5m]) == 0` for >2 cycles to catch a stalled validator."
    ),
    dataSource=DATASOURCE,
    targets=[
        Target(
            expr="rate(validator_main_loop_heartbeat_total[5m])",
            legendFormat="iterations/s",
            refId="A",
        ),
    ],
    gridPos=GridPos(h=8, w=12, x=0, y=10),
    drawStyle="line",
    unit="ops",
    lineWidth=2,
    fillOpacity=10,
)

eval_failures_by_reason = BarChart(
    title="Evaluation Failures by Reason (1h)",
    description=(
        "Total miner evaluation failures over the last hour, broken down by "
        "the closed EvalFailureReason enum (timeout / corrupt / oom / "
        "checksum / rpc / unknown)."
    ),
    dataSource=DATASOURCE,
    targets=[
        Target(
            expr="sum by (reason) (increase(validator_miner_eval_failures_total[1h]))",
            legendFormat="{{reason}}",
            refId="A",
        ),
    ],
    gridPos=GridPos(h=8, w=12, x=12, y=10),
)


# ============================================================================
# Row 3: Miner Performance (y = 18..26)
# ============================================================================

total_throughput = TimeSeries(
    title="Total Subnet Throughput",
    description="Aggregate tokens/sec across every reporting miner.",
    dataSource=DATASOURCE,
    targets=[
        Target(
            expr="sum(miner_tokens_per_sec)",
            legendFormat="tokens/s",
            refId="A",
        ),
    ],
    gridPos=GridPos(h=8, w=12, x=0, y=19),
    drawStyle="line",
    unit="short",
    lineWidth=2,
    fillOpacity=20,
)

# ---- PANEL 7: deliberately commented out ------------------------------------
# `moe_dead_experts_count` was removed from connito/shared/telemetry.py during
# the telemetry/dashboard-overhaul work. The panel slot at gridPos(h=8, w=12,
# x=12, y=19) is reserved.
#
# Pick one of:
#   (a) drop the panel entirely (the layout still works at 6 panels).
#   (b) substitute another populated miner-side signal — e.g.:
#         expr="avg(miner_training_loss)"   legendFormat="loss"
#         expr="avg(miner_perplexity)"       legendFormat="perplexity"
#         expr="avg(miner_gradient_norm)"    legendFormat="grad norm"
#       (do NOT use moe_aux_loss — it's hardcoded to 0 at train.py:365 today,
#       so the panel would be flat at zero.)
#   (c) re-add MOE_DEAD_EXPERTS to telemetry.py + the wiring at the forward
#       pass, then uncomment the panel below.
#
# dead_experts = TimeSeries(
#     title="Dead Experts Count",
#     description="Experts that received zero tokens in the latest batch.",
#     dataSource=DATASOURCE,
#     targets=[
#         Target(expr="moe_dead_experts_count", legendFormat="dead experts", refId="A"),
#     ],
#     gridPos=GridPos(h=8, w=12, x=12, y=19),
#     drawStyle="line",
#     unit="short",
# )
# -----------------------------------------------------------------------------


dashboard = Dashboard(
    title="Connito Subnet Overview",
    description=(
        "Top-level health view for a Connito validator: macro subnet state, "
        "validator liveness and eval-failure attribution, miner throughput. "
        "Source-of-truth for metrics: connito/shared/telemetry.py."
    ),
    tags=["connito", "bittensor", "subnet"],
    timezone="browser",
    refresh="30s",
    time=Time("now-1h", "now"),
    schemaVersion=39,
    panels=[
        _row("Macro Subnet Health", y=0),
        current_phase_block,
        subnet_vtrust,
        deregistrations,

        _row("Validator Operations (The Referee)", y=9),
        heartbeat,
        eval_failures_by_reason,

        _row("Miner Performance (The Workers)", y=18),
        total_throughput,
        # dead_experts,  # see commented stub above
    ],
).auto_panel_ids()


if __name__ == "__main__":
    # Allow `python connito_subnet_overview.dashboard.py > out.json` as an
    # alternative to the grafanalib CLI.
    import json
    import sys
    from grafanalib._gen import DashboardEncoder

    json.dump(
        dashboard.to_json_data(),
        sys.stdout,
        sort_keys=True,
        indent=2,
        cls=DashboardEncoder,
    )
    sys.stdout.write("\n")
