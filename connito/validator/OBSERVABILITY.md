# Validator Observability & Telemetry Guide

The validator acts as the backbone of the decentralized network, scoring miners and syncing gradients via the DHT. Tracking its performance is critical to ensure you emit correct consensus.

## Requirements
- Docker
- Docker Compose V2

## Getting Started

By default, telemetry is enabled. To disable it, run your validator with `ENABLE_TELEMETRY=false`.
The validator telemetry server automatically binds to port `8200 + rank` (rank-0 = `8200`); the validator Docker compose stack already runs on `network_mode: host` and the existing healthcheck (`curl http://localhost:8200/metrics`) confirms the exporter is live.

To visualize this data via Grafana, use the bundled stack:

1. Start your validator (host install or `connito/validator/docker/docker compose up -d`).

2. Start the observability containers:
   ```bash
   cd ../../observability
   docker compose up -d
   ```
   Prometheus is now on `http://localhost:19090`; Grafana is on `http://localhost:3033` (`admin` / `admin`).

3. Multi-host: edit `observability/prometheus.yml` to point at `<validator-ip>:8200` instead of `localhost:8200`, open port `8200/tcp` between hosts, and restart the Prometheus container. Put TLS / basic auth in front of the metrics port if the network between hosts is untrusted (the metric stream includes per-miner scores and the validator's hotkey).

## Connecting Grafana
1. Open `http://localhost:3033` in a browser.
2. Login using `admin` / `admin`.
3. Add a Prometheus Data Source set to `http://localhost:19090` (Connections > Data Sources > Add).
4. Start building panels under the **Dashboards** interface.

## Key Validator Metrics to Monitor

### Health and progress
| Question | PromQL |
|---|---|
| Is the validator alive? | `time() - validator_last_cycle_timestamp` (alert if > 2× cycle length) |
| Cycles completing? | `rate(validator_cycles_completed_total[1h])` |
| Did it merge into consensus? | `validator_participated_in_merge` (alert if `< 1` for > 1 cycle) |
| Which version is running? | `validator_info` (labels: `version`, `git_sha`, `hotkey_ss58`, `expert_group`) |

### Where time is spent
| Question | PromQL |
|---|---|
| Mean phase duration | `sum by (phase) (rate(validator_phase_duration_seconds_sum[5m])) / sum by (phase) (rate(validator_phase_duration_seconds_count[5m]))` |
| Per-miner eval p95 | `histogram_quantile(0.95, sum by (le, lane) (rate(validator_miner_eval_latency_seconds_bucket[5m])))` |

### Errors
| Question | PromQL |
|---|---|
| Error rate by component | `sum by (component, kind) (rate(validator_errors_total[5m]))` |
| HF transport failures | `sum by (direction, kind) (rate(hf_transfer_result_total{result="failure"}[5m]))` |
| Background download outcomes | `sum by (result) (rate(bg_download_result_total[5m]))` |
| Background eval outcomes | `sum by (result) (rate(bg_eval_result_total[5m]))` |

### Round + lanes
| Question | PromQL |
|---|---|
| Foreground roster this round | `validator_round_miner_lane{lane="foreground"}` |
| Background roster this round | `validator_round_miner_lane{lane="background"}` |
| Roster sizes by lane | `validator_round_roster_size` |
| Pending miners over time | `validator_round_miners_pending` |

### Scoring + reward signal
| Question | PromQL |
|---|---|
| Baseline loss for the round | `validator_baseline_loss` |
| Per-miner delta loss | `validator_miner_delta_loss` |
| Latest miner score | `validator_miner_score` |
| Rolling-avg score (used for chain weights) | `validator_miner_score_avg` |
| Final chain weight submitted | `validator_miner_chain_weight` |
| Top-N miners by reward | `topk(10, validator_miner_chain_weight)` |

### Hardware + chain RPC
| Question | PromQL |
|---|---|
| VRAM | `validator_vram_allocated_bytes` |
| DHT peers | `validator_dht_peers_count` |
| Allreduce status | `rate(validator_avg_step_status_total[5m])` |
| Blockchain sync | `subnet_blocks_remaining_in_phase` |
| Chain commit latency | `rate(chain_commit_latency_seconds_sum[5m])` |
| Weight submission outcomes | `rate(chain_weight_set_success[5m])`, `rate(chain_weight_set_failure[5m])` |
