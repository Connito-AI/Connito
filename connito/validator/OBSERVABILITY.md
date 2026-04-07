# Validator Observability & Telemetry Guide

The validator acts as the backbone of the decentralized network, scoring miners and syncing gradients via the DHT. Tracking its performance is critical to ensure you emit correct consensus.

## Requirements
- Docker
- Docker Compose V2

## Getting Started


By default, telemetry is enabled. To disable it, run your validator with `ENABLE_TELEMETRY=false`.
The validator telemetry server automatically exposes metrics locally on port `8000` (scaling as `8000 + rank` for multi-process environments) while running.

To visualize this data securely via Grafana, leverage the provided stack:

1. Start your validator:
   ```bash
   python main.py <args>
   ```

2. Start the observability containers:
   ```bash
   cd ../../observability
   docker compose up -d
   ```

## Connecting Grafana
1. Open `http://localhost:3000` in a browser.
2. Login using `admin` / `admin`.
3. Add a Prometheus Data Source set to `http://localhost:9090` (Connections > Data Sources > Add).
4. Start building panels under the **Dashboards** interface.

## Key Validator Metrics to Monitor
The following PromQL queries can be used to monitor the heartbeat of your validator node and the active behavior of the network topology:

| Category | PromQL Query | Description |
|----------|--------------|-------------|
| **Miner Leaderboard** | `validator_miner_score` | Group by `{miner_uid=...}` to see how miners rank against each other historically. |
| **DHT Peer Sync** | `validator_dht_peers_count` | See the number of active Averagers visible in your local Hivemind network topology. |
| **Gradient Stability** | `validator_avg_step_status_total` | Counters natively partitioned by `{status="success" | "timeout" | "error"}`. |
| **Inference/Eval Speed** | `rate(validator_eval_latency_seconds_sum[5m])` | Time in seconds taken to evaluate incoming miner jobs. |
| **Hardware Constraint** | `validator_vram_allocated_bytes` | Real-time CUDA memory tracker per GPU layer. |
| **Blockchain Sync** | `subnet_blocks_remaining_in_phase` | Blocks until the next cycle transition (Distribute -> Validate -> Merge). Useful to catch chain RPC drift. |
