import time
import requests
import threading

from connito.shared.telemetry import (
    TelemetryManager,
    SUBNET_CURRENT_BLOCK,
    VALIDATOR_MINER_SCORE,
    VALIDATOR_AVG_STEP_STATUS,
    track_eval_latency,
    count_rpc_errors
)

def test_telemetry_flow():
    print("1. Starting Prometheus Server...")
    # Use a custom port to avoid any collision for testing
    mgr = TelemetryManager()
    mgr.start_server(port=8111)

    print("2. Populating Gauges and Counters...")
    SUBNET_CURRENT_BLOCK.set(12345)
    
    # Check labelled metrics
    VALIDATOR_MINER_SCORE.labels(miner_uid="12").set(0.85)
    VALIDATOR_AVG_STEP_STATUS.labels(status="success").inc()
    VALIDATOR_AVG_STEP_STATUS.labels(status="success").inc()

    print("3. Testing Decorators...")
    @track_eval_latency()
    @count_rpc_errors()
    def fake_evaluation(fail=False):
        time.sleep(0.1)  # Simulate workload
        if fail:
            raise ConnectionError("Timeout hit")
            
    # Success path
    fake_evaluation(fail=False)
    
    # Error path
    try:
        fake_evaluation(fail=True)
    except ConnectionError:
        print("   Caught expected error for metric counting")

    print("4. Asserting HTTP Export...")
    # Give server a brief moment to stabilize
    time.sleep(0.5)
    
    resp = requests.get("http://localhost:8111/metrics")
    assert resp.status_code == 200, "Prometheus HTTP server not responding"
    
    content = resp.text
    
    assert 'subnet_current_block 12345.0' in content, "Missing block gauge"
    assert 'validator_miner_score{miner_uid="12"} 0.85' in content, "Missing labeled miner score"
    assert 'validator_avg_step_status_total{status="success"} 2.0' in content, "Missing labeled averager count"
    assert 'validator_eval_latency_seconds_count 2.0' in content, "Missing latency histogram count"
    
    if 'chain_rpc_errors_total_total 1.0' not in content and 'chain_rpc_errors_total 1.0' not in content:
        print("Received HTTP content:\n", content)
        assert False, "Missing exception counter"
    
    print("✅ All phase 1 tests passed successfully!")


def test_new_observability_metrics_registered():
    """Smoke test: every metric introduced for the observability audit
    appears on the default Prometheus registry under its expected name.

    Catches accidental renames or accidental removals before they reach
    operators' dashboards.
    """
    from prometheus_client import REGISTRY

    # Importing the module registers the metrics on REGISTRY at import time.
    from connito.shared.telemetry import (  # noqa: F401
        BG_DOWNLOAD_RESULT_TOTAL,
        BG_EVAL_RESULT_TOTAL,
        HF_TRANSFER_BYTES_TOTAL,
        HF_TRANSFER_LATENCY_SECONDS,
        HF_TRANSFER_RESULT_TOTAL,
        VALIDATOR_BASELINE_LOSS,
        VALIDATOR_CYCLES_COMPLETED_TOTAL,
        VALIDATOR_ERRORS_TOTAL,
        VALIDATOR_INFO,
        VALIDATOR_LAST_CYCLE_TIMESTAMP,
        VALIDATOR_MINER_CHAIN_WEIGHT,
        VALIDATOR_MINER_DELTA_LOSS,
        VALIDATOR_MINER_EVAL_LATENCY_SECONDS,
        VALIDATOR_MINER_SCORE_AVG,
        VALIDATOR_MINER_SCORE_POINTS,
        VALIDATOR_PARTICIPATED_IN_MERGE,
        VALIDATOR_PHASE_DURATION_SECONDS,
        VALIDATOR_ROUND_MINER_LANE,
        VALIDATOR_ROUND_ROSTER_SIZE,
        inc_error,
    )

    # Counter metrics get the `_total` suffix on scrape but are registered
    # under the bare name; the expected set below matches the registered
    # form so the assertion is suffix-agnostic.
    expected = {
        "validator_cycles_completed",
        "validator_last_cycle_timestamp",
        "validator_participated_in_merge",
        "validator_info",
        "validator_phase_duration_seconds",
        "validator_errors",
        "validator_round_roster_size",
        "validator_round_miner_lane",
        "validator_baseline_loss",
        "validator_miner_delta_loss",
        "validator_miner_eval_latency_seconds",
        "validator_miner_score_avg",
        "validator_miner_score_points",
        "validator_miner_chain_weight",
        "hf_transfer_bytes",
        "hf_transfer_latency_seconds",
        "hf_transfer_result",
        "bg_download_result",
        "bg_eval_result",
    }
    registered = {col.describe()[0].name for col in REGISTRY._collector_to_names.keys() if hasattr(col, "describe") and col.describe()}
    missing = expected - registered
    assert not missing, f"telemetry metrics missing from registry: {sorted(missing)}"

    # Smoke: helper compiles a labelled counter without raising.
    inc_error("hf_upload", "network")
    inc_error("foreground_eval", "oom")


if __name__ == "__main__":
    test_telemetry_flow()
    test_new_observability_metrics_registered()
