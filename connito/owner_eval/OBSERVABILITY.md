# Owner Eval — Backend API Consumption Guide

How the backend / leaderboard API surfaces the owner eval pipeline's results.

## Architecture (pull model)

```
 owner_eval daemon                central Prometheus              backend API / dashboard
 ┌────────────────┐  scrape /metrics  ┌──────────────┐  PromQL over HTTP  ┌──────────────┐
 │ exposes :8400  │ ◀──────────────── │ job:         │ ◀───────────────── │ /api/v1/query│
 │ owner_eval_*   │   every 5–15s     │ mycelia_     │                    │ render panels│
 │ gauges/counter │                   │ owner_eval   │                    └──────────────┘
 └────────────────┘                   └──────────────┘
```

The daemon does **not** push anywhere and the backend does **not** talk to the daemon
directly. The daemon exposes a Prometheus text endpoint on `eval_pipeline.telemetry_port`
(default **8400**); Prometheus scrapes it; the backend queries **Prometheus**, never the
daemon. This mirrors how validator/miner metrics already flow.

### Prometheus must scrape the daemon

Add the daemon as a target (already in `observability/prometheus.yml` for local dev; the
production Prometheus needs the same target pointed at the daemon's real host):

```yaml
  - job_name: 'mycelia_owner_eval'
    static_configs:
      - targets: ['<daemon-host>:8400']   # eval_pipeline.telemetry_port
```

Until this target exists and reports `up`, the backend cannot see any owner_eval data.

## Metric reference (the contract)

Every series also carries Prometheus-added labels `job="mycelia_owner_eval"` and
`instance="<host>:8400"`.

| Series | Type | Labels | Meaning |
|---|---|---|---|
| `owner_eval_metric` | Gauge | `metric` | One scalar result per run, keyed by `metric`. Values below. |
| `owner_eval_status` | Gauge | `metric` (evaluator name) | `1` = evaluator's last run succeeded, `0` = it raised. |
| `owner_eval_run_info` | Gauge (=1) | `model_revision`, `cycle_index` | Identity of the latest run. Join target for context. |
| `owner_eval_last_run_timestamp` | Gauge | — | Unix seconds of the last completed run (freshness). |
| `owner_eval_loop_heartbeat_total` | Counter | — | Daemon poll iterations (liveness). |

**`owner_eval_metric` keys** (the `metric` label) — one Prometheus sample each:

| `metric` | Meaning | Range |
|---|---|---|
| `gsm8k_task_acc` | GSM8K exact-match accuracy | 0–1 |
| `gsm8k_task_n` | GSM8K samples evaluated | int |
| `mmlu_acc` | MMLU accuracy | 0–1 |
| `mmlu_n` | MMLU samples evaluated | int |
| `gsm8k_ppl` | GSM8K perplexity (`exp(val_loss)`) | ≥1, lower better |
| `gsm8k_ppl_val_loss` | mean LM loss behind the ppl | ≥0 |

New metrics appear automatically as new `metric` label values — the backend should treat
the key set as **open** (render whatever keys are present) rather than hard-coding it.

**`owner_eval_status` keys** are the *evaluator* names — `gsm8k_ppl`, `gsm8k_task`, `mmlu`
— not the per-scalar keys above (one evaluator can emit several scalars).

## Querying Prometheus (what the backend calls)

Prometheus HTTP API, typically `http://<prometheus>:9090`:

- **Instant** (current value): `GET /api/v1/query?query=<PromQL>`
- **Range** (history/trend): `GET /api/v1/query_range?query=<PromQL>&start=<unix>&end=<unix>&step=<sec>`
- **Targets health**: `GET /api/v1/targets?state=active`

### Response shape (instant query)

```json
{
  "status": "success",
  "data": {
    "resultType": "vector",
    "result": [
      { "metric": {"__name__":"owner_eval_metric","metric":"mmlu_acc",
                   "job":"mycelia_owner_eval","instance":"host:8400"},
        "value": [1780671984.769, "0.5"] }
    ]
  }
}
```

`value` is `[<unix_ts>, "<stringified number>"]` — parse the second element as a float.

### Latest leaderboard values

```
owner_eval_metric
```
Returns all current scalars in one call; group by the `metric` label to populate the board.

### Attach model revision + cycle to each value (instant)

```
owner_eval_metric
  * on(instance) group_left(model_revision, cycle_index) owner_eval_run_info
```
Multiplies by `owner_eval_run_info` (always `1`) and grafts its `model_revision` /
`cycle_index` labels onto every metric — so each displayed score is tagged with which model
produced it.

### Trend over time (range query)

```
owner_eval_metric{metric="mmlu_acc"}
```
`owner_eval_metric` has **stable labels**, so a `query_range` gives a clean time series per
metric for charting progress across runs.

### Health / freshness / liveness

```
up{job="mycelia_owner_eval"}                      # 1 = Prometheus is scraping the daemon
time() - owner_eval_last_run_timestamp            # seconds since last completed run (staleness)
owner_eval_status                                 # per-evaluator 1/0 success
rate(owner_eval_loop_heartbeat_total[10m]) > 0    # daemon poll loop alive
```

Suggested alerts: `up == 0` (daemon/scrape down), `time() - owner_eval_last_run_timestamp >
6 * 3600` (no run in ~last interval window), `owner_eval_status == 0` (an evaluator failing),
`rate(owner_eval_loop_heartbeat_total[15m]) == 0` (daemon hung).

### Example calls

```bash
# current scores, tagged with model revision + cycle
curl -G 'http://prometheus:9090/api/v1/query' \
  --data-urlencode 'query=owner_eval_metric * on(instance) group_left(model_revision,cycle_index) owner_eval_run_info'

# mmlu accuracy over the last 30 days, daily points
curl -G 'http://prometheus:9090/api/v1/query_range' \
  --data-urlencode 'query=owner_eval_metric{metric="mmlu_acc"}' \
  --data-urlencode "start=$(date -d -30days +%s)" \
  --data-urlencode "end=$(date +%s)" \
  --data-urlencode 'step=86400'
```

```python
import requests

PROM = "http://prometheus:9090"

def latest_scores():
    q = ('owner_eval_metric * on(instance) '
         'group_left(model_revision,cycle_index) owner_eval_run_info')
    r = requests.get(f"{PROM}/api/v1/query", params={"query": q}, timeout=5).json()
    out = []
    for s in r["data"]["result"]:
        m = s["metric"]
        out.append({
            "metric": m["metric"],
            "value": float(s["value"][1]),
            "model_revision": m.get("model_revision"),
            "cycle_index": m.get("cycle_index"),
        })
    return out
```

## Semantics & caveats

- **Gauges are last-value.** `owner_eval_metric` is overwritten each run; "history" comes
  from Prometheus's stored samples (retention), queried with `query_range`. There is no
  separate history endpoint.
- **`owner_eval_run_info` is an info metric** (value always `1`; `model_revision` /
  `cycle_index` carried as labels, replaced each run — bounded cardinality, one series at a
  time). Use the `group_left` join for *current* context. It is **not** a reliable way to
  reconstruct which revision produced a *historical* value (its labelset changes over time).
  If the backend needs durable per-model-version history, **snapshot** `owner_eval_metric` +
  `owner_eval_run_info` into its own store whenever `cycle_index` advances. (Alternatively
  the pipeline could add `model_revision` as a label on `owner_eval_metric` itself, at the
  cost of a new series per model version — ask if you want that.)
- **`model_revision`** is `globalver_<n>` for chain-fetched models, or `base` when the
  daemon runs in `model_source: base` (canary) mode — filter out `base` for production
  boards if a canary is ever pointed at the same Prometheus.
- **n changes don't change series**; `gsm8k_task_n` / `mmlu_n` report the sample count each
  run so the backend can show the confidence basis (e.g. "acc 0.41 over n=500").

## Reproduce locally (canary)

```bash
# 1. run the daemon against the base model (no wallet), publishing tiny-n results
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m connito.owner_eval.run \
    --path owner_eval.canary.yaml          # model_source: base, n=2, port 8400

# 2. confirm the publish side
curl -s localhost:8400/metrics | grep owner_eval

# 3. point a Prometheus at it (observability/prometheus.yml already has the job)
docker run --rm -d --name prom --network host \
  -v "$PWD/observability/prometheus.yml:/etc/prometheus/prometheus.yml" prom/prometheus:latest

# 4. query as the backend would
curl -s 'localhost:9090/api/v1/targets?state=active' | grep mycelia_owner_eval   # => up
curl -s 'localhost:9090/api/v1/query?query=owner_eval_metric'
```
