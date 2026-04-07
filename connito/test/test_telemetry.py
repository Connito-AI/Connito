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

if __name__ == "__main__":
    test_telemetry_flow()
