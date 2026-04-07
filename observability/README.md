# BlockZero Observability Overview

We have fully integrated passive Prometheus tracing into the Core Miner and Validator pipelines.

## 1. Quick Start Locally
To visualize test runs of your nodes, simply run:
```bash
cd observability
docker compose up -d
```
*Note: Make sure docker is installed.*

This will start:
1. **Prometheus Data Store** on `http://localhost:9090`
2. **Grafana Dashboard** on `http://localhost:3000`

### 2. View in Grafana
1. Navigate in your browser to `http://localhost:3000`.
2. Login with generic credentials username: `admin` / password: `admin`
3. By default, Grafana must be connected to Prometheus. Nav to **Data Sources > Add Data Source > Prometheus**
4. Set the Prometheus Server URL to `http://localhost:9090` and hit "Save & Test".

## 3. Recommended Panels to Create inside Grafana
Because everything exported has standardized metric names, you can instantly assemble panels with the following promQL queries:

| Metric | PromQL Query | Description |
|---|---|---|
| **VRAM Usage** | `validator_vram_allocated_bytes` | Track memory spikes (spikes here explain OOM errors). |
| **Miner Loss** | `miner_training_loss` | Live stream of the optimizer step loss. |
| **Validation Scores**| `validator_miner_score` | Group by `{miner_uid=...}` to see how miners fight for alpha. |
| **Sync Connectivity**| `validator_dht_peers_count` | See how many DHT peers Averagers can see on your subnet. |
| **Chain Latency** | `rate(chain_commit_latency_seconds_sum[1m]) / rate(chain_commit_latency_seconds_count[1m])` | Real-time moving average of Bittensor extrinsics. |
| **Phase Block Stats**| `subnet_blocks_remaining_in_phase` | Visual countdown until cycle triggers Merge/Commit stages. |

## Why this architecture?
1. **Zero Downtime**: Instead of interrupting `run.py` to sleep/query data, decorators stream results to a Singleton Registry, and a sidecar `SystemStatePoller` handles heavy lifting. 
2. **Network Partitioned**: If port 8000 drops, the validator continues seamlessly because Prometheus uses an external HTTP pull-based mechanism—no pipeline crashes.
