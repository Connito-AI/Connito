# Miner Observability & Telemetry Guide

Your standalone miner automatically exports rich runtime metrics via a passive local HTTP server. This allows you to track memory usage, training loss, and chain state safely without interrupting PyTorch execution.

## Requirements
- Docker
- Docker Compose V2

## Getting Started


By default, telemetry is enabled. To disable it, run your miner with `ENABLE_TELEMETRY=false`.
The telemetry server automatically starts on port `8100` (specifically `8100 + local_rank` for multi-GPU setups) when you run the miner.

To view these metrics locally, you can use the pre-configured Prometheus and Grafana stack located in the root repository.

1. Start your miner as usual:
   ```bash
   python main.py <args>
   ```

2. In a new terminal, launch the observability stack:
   ```bash
   cd ../../observability
   docker compose up -d
   ```

## Visualizing in Grafana
1. Open your browser and navigate to `http://localhost:3000`.
2. Login with username `admin` and password `admin`.
3. Add a Data Source: **Connections > Add Data Source > Prometheus**.
4. Set the URL to `http://localhost:9090` and click "Save & Test".
5. Navigate to **Dashboards > New Dashboard** and add visualizations.

## Key Miner Metrics to Monitor
You can paste the following PromQL queries into Grafana panels to instantly build charts:

| Metric Aspect | PromQL Query | Description |
|---|---|---|
| **Training Loss** | `miner_training_loss` | Live training loss per optimization step. Drops steadily. |
| **Learning Rate** | `miner_learning_rate` | Tracks your optimizer LR schedule curve. |
| **Throughput** | `miner_local_step_rate` | Steps per second completed locally. |
| **VRAM Usage** | `validator_vram_allocated_bytes` | Total CUDA memory allocated on your machine. Useful to anticipate and debug OOM errors. |
| **Chain Latency** | `rate(chain_commit_latency_seconds_sum[1m])` | Time overhead taken to commit weights or status to the chain. |
| **Cycle Progress** | `subnet_blocks_remaining_in_phase` | Countdown of blocks until the network transitions phases. |
