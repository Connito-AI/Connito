import os
import functools
import threading
import torch
import time
from typing import Callable, Any, Literal

import psutil
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Info

from connito.shared.app_logging import structlog
logger = structlog.get_logger(__name__)

class TelemetryManager:
    """
    Singleton manager to ensure Prometheus HTTP server is only started once
    per process, protecting against port collisions and multiple initializations.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TelemetryManager, cls).__new__(cls)
                cls._instance._server_started = False
        return cls._instance

    def start_server(self, port: int = 8000):
        if str(os.environ.get("ENABLE_TELEMETRY", "true")).lower() not in ("true", "1", "yes"):
            logger.info("Telemetry disabled via ENABLE_TELEMETRY flag.")
            return
        with self._lock:
            if not self._server_started:
                try:
                    start_http_server(port)
                    self._server_started = True
                    logger.info("Prometheus metrics server started", port=port)
                except Exception as e:
                    logger.error("Failed to start Prometheus server", port=port, error=str(e))


# ==============================================================================
# Metric Definitions
# ==============================================================================

# Identity — stamps every scrape with which validator emitted these metrics.
# Set once at startup via set_validator_identity(); central Prometheus joins
# this with measurement metrics via PromQL `* on(instance) group_left(...)`.
CONNITO_VALIDATOR_INFO = Info(
    "connito_validator",
    "Identity of the validator emitting these metrics (one labelset per process)",
)


def set_validator_identity(
    *, hotkey: str, uid: int | None, version: str, netuid: int, observer: bool = False,
) -> None:
    """Stamp the validator's identity onto the ``connito_validator_info``
    metric. Call once at validator startup, immediately after ``validator_uid``
    resolution. Safe to re-call (e.g. on UID change after a deregister/re-
    register cycle) — ``Info.info()`` replaces the labelset atomically.
    """
    CONNITO_VALIDATOR_INFO.info({
        "hotkey": str(hotkey),
        # uid==None when the bootstrap metagraph fetch failed; emit as -1 so
        # downstream queries can still match without crashing on null.
        "uid": str(uid if uid is not None else -1),
        "version": str(version),
        "netuid": str(netuid),
        # An observer shares a hotkey with the live validator, so hotkey and
        # uid alone do not distinguish this scrape from that one.
        "observer": "1" if observer else "0",
    })


# Infrastructure / Cycle (Gauges & Histograms)
SUBNET_CURRENT_BLOCK = Gauge("subnet_current_block", "Current block on local subtensor")
SUBNET_PHASE_INDEX = Gauge("subnet_current_phase_index", "Enum index of active phase")
SUBNET_BLOCKS_REMAINING = Gauge("subnet_blocks_remaining_in_phase", "Blocks left before phase transition")
SUBNET_VALIDATOR_VTRUST = Gauge("subnet_validator_vtrust", "Validator trust value for this validator's UID")
SUBNET_VALIDATOR_CONSENSUS = Gauge("subnet_validator_consensus", "Consensus value for this validator's UID")
SUBNET_UID_DEREGISTRATIONS_TOTAL = Counter(
    "subnet_uid_deregistrations_total",
    "UIDs that disappeared from the metagraph between consecutive polls",
)

GPU_VRAM_ALLOCATED_BYTES = Gauge("validator_vram_allocated_bytes", "VRAM allocated by operations", ["device"])
GPU_VRAM_PEAK_ALLOCATED_BYTES = Gauge("validator_vram_peak_allocated_bytes", "Peak VRAM allocated by operations", ["device"])
GPU_UTILIZATION_PERCENT = Gauge("system_gpu_utilization_percent", "GPU Utilization percent", ["device"])
SYSTEM_CPU_UTILIZATION_PERCENT = Gauge(
    "system_cpu_utilization_percent",
    "Host CPU utilization percent (psutil aggregate, sampled by SystemStatePoller)",
)
DHT_PEER_COUNT = Gauge("validator_dht_peers_count", "Total peers tracked in the averager network")
DATALOADER_QUEUE_DEPTH = Gauge("system_dataloader_queue_depth", "Data pipeline depth")
MODEL_PARAMETER_COUNT = Gauge("system_model_parameter_count", "Total loaded parameter count")

# Validator (Gauges & Counters)
VALIDATOR_ACTIVE_MINER_EVALS = Gauge("validator_active_miner_evaluations", "Number of miner_jobs being evaluated")
VALIDATOR_MINER_SCORE = Gauge("validator_miner_score", "Validation score assigned to a miner", ["miner_uid"])
# Rolling EMA score actually voted on chain (i.e. the value
# `score_aggregator.uid_score_pairs(how="avg")` returns and that
# `chain_submitter.async_submit_weight` consumes). Distinct from
# `validator_miner_score`, which is the latest *raw* per-round score
# fed into the aggregator. Set per-round right before chain submission;
# absent for UIDs the validator hasn't scored yet (no entry rather than 0).
VALIDATOR_MINER_WEIGHT_SUBMITTED = Gauge(
    "validator_miner_weight_submitted",
    "Rolling EMA voted on chain for a miner",
    ["miner_uid"],
)
# Per-miner validation loss measured against this validator's foreground
# eval set. High-cardinality (one series per miner UID) but bounded by
# subnet size (~100s); same shape as `validator_miner_score`. Set inside
# `evaluate_one_miner` immediately after the val_loss is computed.
# Aggregators compute `delta_loss = max(0, validator_baseline_loss -
# validator_miner_val_loss)` as needed.
VALIDATOR_MINER_VAL_LOSS = Gauge(
    "validator_miner_val_loss",
    "Per-miner validation loss measured against this validator's foreground eval set",
    ["miner_uid"],
)
# Round-level baseline loss: this validator's eval loss against the
# pre-merge global model, computed once per round at the start of the
# foreground pass (see `evaluate_foreground_round`). Single value (no
# labels) — the latest write wins. Distinct from
# `validator_eval_loss{expert_group=...}` which tracks training-side
# eval loss reported via MetricLogger.
VALIDATOR_BASELINE_LOSS = Gauge(
    "validator_baseline_loss",
    "Round baseline loss against this validator's foreground eval set",
)
# Per-round baseline loss, labeled by the round it belongs to. The unlabeled
# gauge above is overwritten every round, so a scraper can only sample it and
# a cycle's baseline ends up timing-dependent; this labeled family lets the
# gateway attribute the right baseline to the right round, freeze it after
# finalize, and naturally exclude a warming-up validator that has no value for
# a given round. Evicted on the same cutoff as the other per-round families
# (see evict_round_series_before). The unlabeled family is retained for
# backward compat during rollout.
VALIDATOR_BASELINE_LOSS_BY_ROUND = Gauge(
    "validator_baseline_loss_by_round",
    "Round baseline loss (this validator's foreground eval set), labeled by the "
    "round it belongs to. Stable per round; the unlabeled validator_baseline_loss "
    "is retained for backward compat.",
    ["round_id"],
)
# Numeric ID of the current round, set when `Round.freeze` returns and
# the round becomes active. Lets aggregators key per-miner score and
# val_loss readings to a specific round without parsing the round_id
# label off `validator_round_lifecycle_step`.
VALIDATOR_CURRENT_ROUND_ID = Gauge(
    "validator_current_round_id",
    "Numeric ID of the round this validator is currently evaluating",
)
VALIDATOR_SCORE_STD = Gauge("validator_score_std", "Spread of miner scores")
VALIDATOR_AVG_STEP_STATUS = Counter("validator_avg_step_status", "Averager sync step stats", ["status"])
# Pre-init the fixed status enum so the counter is visible in /metrics from
# process start instead of only after the first averager step. prometheus_client
# omits labeled metrics that have never had .labels() called.
for _status in ("success", "timeout", "error"):
    VALIDATOR_AVG_STEP_STATUS.labels(status=_status)
VALIDATOR_EVAL_LOSS = Gauge("validator_eval_loss", "Evaluation loss", ["expert_group"])
VALIDATOR_EVAL_BATCH_COUNT = Counter("validator_eval_batch_count", "Evaluation batch count")
VALIDATOR_HEARTBEAT_TOTAL = Counter(
    "validator_main_loop_heartbeat_total",
    "Validator main loop iterations completed; alert on rate() going to zero",
)
VALIDATOR_METAGRAPH_LAST_SYNC_TS = Gauge(
    "validator_metagraph_last_sync_timestamp",
    "Unix timestamp of the most recent successful metagraph sync",
)
VALIDATOR_MINER_EVAL_FAILURES = Counter(
    "validator_miner_eval_failures_total",
    "Failures encountered while evaluating a miner submission, by reason",
    ["miner_uid", "reason"],
)
# Per-miner aggregator snapshot. Set right before chain submission alongside
# VALIDATOR_MINER_WEIGHT_SUBMITTED so a single Prometheus scrape carries
# everything the leaderboard needs without re-deriving from per-round samples.
# Absent for UIDs the validator has never scored (no entry rather than 0).
VALIDATOR_MINER_SCORE_LATEST = Gauge(
    "validator_miner_score_latest",
    "Most recent per-round rank score the aggregator holds for a miner",
    ["miner_uid"],
)
VALIDATOR_MINER_SCORE_AVG = Gauge(
    "validator_miner_score_avg",
    "Rolling average of per-round rank scores within the aggregator window",
    ["miner_uid"],
)
VALIDATOR_MINER_SCORE_SAMPLES = Gauge(
    "validator_miner_score_samples",
    "Number of score samples retained for a miner within the aggregator window",
    ["miner_uid"],
)
# Unix-seconds timestamp of the moment this validator last published a score
# snapshot for the miner. Set by `set_miner_score_snapshot` alongside
# VALIDATOR_MINER_SCORE_LATEST/_AVG/_SAMPLES, so it advances exactly once per
# cycle at the chain-submit boundary. The gateway computes "score age" as
# `time() - validator_miner_score_latest_emitted_at` — `timestamp(score_latest)`
# alone is misleading because Prometheus updates the sample-timestamp on every
# scrape (every ~5s) even when the underlying value hasn't changed, making the
# score look freshly emitted when in fact it hasn't been updated for almost a
# whole cycle. This gauge is the "value last changed" signal the dashboard
# needs to render meaningful "scored Xm ago" labels.
VALIDATOR_MINER_SCORE_LATEST_EMITTED_AT = Gauge(
    "validator_miner_score_latest_emitted_at",
    "Unix seconds when this validator last published a score_latest snapshot for the miner",
    ["miner_uid"],
)
# Last-known per-miner eval outcome on THIS validator. Integer-coded so the
# gateway can render miner-facing strings without label cardinality blowing
# up (one series per miner_uid, value = code from EVAL_STATUS_CODES below).
# 0 == ok; non-zero codes correspond to the failure reason the validator
# observed most recently. Stable across rounds — the gateway is expected
# to read `last_over_time(...)` so this remains queryable even on cycles
# where the miner is not in the eval set.
VALIDATOR_MINER_EVAL_STATUS = Gauge(
    "validator_miner_eval_status",
    "Current per-miner eval status code (0=ok; see EVAL_STATUS_CODES)",
    ["miner_uid"],
)

# Dashboard-facing telemetry (validator-side emission for the leaderboard UI).
# Model global optimization version this validator is training. Single value;
# distinct from the loop's `global_opt_step` (which carries a block number).
VALIDATOR_GLOBAL_OPT_STEP = Gauge(
    "validator_global_opt_step",
    "Model global optimization version (global_ver) this validator is training",
)
# Current cohort epoch index for the round-group construction scheme. Single
# value; advances at each cohort boundary (every 8th cycle).
VALIDATOR_COHORT_EPOCH = Gauge(
    "validator_cohort_epoch",
    "Current cohort epoch index for the round-group construction scheme",
)
# Per-miner cohort validation group for the active round. Integer-coded (like
# eval_status) so the gateway renders the letter without label-cardinality
# churn when a miner rotates groups. Reset for every metagraph uid each round
# (0 for miners in no group) so stale membership never lingers across epochs.
VALIDATOR_MINER_COHORT_GROUP = Gauge(
    "validator_miner_cohort_group",
    "Cohort validation group for a miner this round (see COHORT_GROUP_CODES)",
    ["miner_uid"],
)
# Per-miner assignment role for THIS validator's roster this round. Each
# validator emits its own slice; the gateway unions across validators by the
# scrape instance label. Reset for every metagraph uid each round.
VALIDATOR_MINER_ASSIGNMENT_ROLE = Gauge(
    "validator_miner_assignment_role",
    "This validator's assignment role for a miner this round (see ASSIGNMENT_ROLE_CODES)",
    ["miner_uid"],
)
# Round (freeze) block at which this validator last observed a valid chain
# commit for the miner. Answers "do validators see my submission?" — set to the
# round_id of the most recent freeze where the miner had a valid (hf_repo_id,
# hf_revision) commit.
VALIDATOR_MINER_LAST_OBSERVED_COMMIT_BLOCK = Gauge(
    "validator_miner_last_observed_commit_block",
    "Round (freeze) block at which this validator last observed a valid chain commit for the miner",
    ["miner_uid"],
)

# --- Cycle-consistent per-miner attribution (dashboard contract) -----------
# The gateway attributes every per-miner sample to the exact evaluation
# round via these three families. All are set from `finalize_round_scores`
# (including its journal-recovery replay path) so a validator restart
# re-publishes them without waiting for a fresh round.
VALIDATOR_MINER_LAST_SCORED_ROUND_ID = Gauge(
    "validator_miner_last_scored_round_id",
    "round_id of the last round in which this validator wrote a finalize "
    "verdict (scored, tie-zeroed, validation-failed, or freeze-zero) for the miner",
    ["miner_uid"],
)
VALIDATOR_MINER_ROUND_DELTA = Gauge(
    "validator_miner_round_delta",
    "Raw per-round improvement signal ((baseline - val_loss) ** 1.2, >= 0) "
    "from the miner's most recent evaluated round. Distinct from "
    "validator_miner_score_latest, which is the finalized podium rank score.",
    ["miner_uid"],
)
VALIDATOR_MINER_EVALUATED_COMMIT_INFO = Gauge(
    "validator_miner_evaluated_commit_info",
    "round_id in which the labeled (hf_repo_id, hf_revision) was frozen and "
    "evaluated for the miner. At most one labelset per miner_uid (old "
    "labelsets are evicted on change).",
    ["miner_uid", "hf_repo_id", "hf_revision"],
)
# uid -> (hf_repo_id, hf_revision) currently exposed on
# VALIDATOR_MINER_EVALUATED_COMMIT_INFO. Guarded by _COMMIT_INFO_LOCK; used
# to evict the previous labelset when a miner's commit changes, keeping the
# "<= 1 labelset per uid" invariant. After a restart both this dict and the
# registry start empty, so correctness holds without persistence.
_COMMIT_INFO_LOCK = threading.Lock()
_COMMIT_INFO_LABELS: dict[str, tuple[str, str]] = {}

# Per-round lifecycle (background submission validation)
VALIDATOR_ROUND_LIFECYCLE_STEP = Gauge(
    "validator_round_lifecycle_step",
    "Current lifecycle step (0-4) for the round identified by round_id",
    ["round_id"],
)
VALIDATOR_ROUND_MINERS_PENDING = Gauge(
    "validator_round_miners_pending",
    "Roster miners not yet scored for the round",
    ["round_id"],
)
VALIDATOR_ROUND_MINERS_SCORED = Gauge(
    "validator_round_miners_scored",
    "Roster miners scored so far for the round",
    ["round_id"],
)
VALIDATOR_ROUND_MINERS_FAILED = Gauge(
    "validator_round_miners_failed",
    "Roster miners that failed download/eval for the round",
    ["round_id"],
)
# round_id label values ever emitted on the per-round families above (and on
# VALIDATOR_BG_EVAL_LOCK_LEAK_TOTAL). Call sites register via
# `note_round_series`; `evict_round_series_before` removes stale labelsets on
# the same cutoff run.py already uses to prune journals/aggregator entries —
# without this, every round leaves four-plus permanent series behind.
_ROUND_SERIES_LOCK = threading.Lock()
_EMITTED_ROUND_IDS: set[int] = set()
VALIDATOR_BG_WORKER_PAUSED = Gauge(
    "validator_bg_worker_paused",
    "1 while a background worker is paused on merge_phase_active / eval_window / download_window",
    ["worker"],
)
VALIDATOR_BG_EVAL_LOCK_LEAK_TOTAL = Counter(
    "validator_bg_eval_lock_leak_total",
    "Bg-eval timeouts that left gpu_eval_lock held by an in-flight thread",
    ["round_id"],
)
VALIDATOR_BG_EVAL_STUCK_LOCK_ITERATIONS = Gauge(
    "validator_bg_eval_stuck_lock_iterations",
    "Consecutive bg-eval iterations observing gpu_eval_lock held at iteration boundary; "
    "0 in steady state, escalates to a recycle when threshold is crossed",
)
VALIDATOR_BG_EVAL_RECYCLE_TOTAL = Counter(
    "validator_bg_eval_recycle_total",
    "Times bg-eval dropped its eval_base_model after a stuck-lock streak",
)

# Miner (Gauges)
MINER_TRAINING_LOSS = Gauge("miner_training_loss", "Local model training loss", ["expert_group"])
MINER_GRAD_NORM = Gauge("miner_gradient_norm", "Gradient norm per step")
MINER_LEARNING_RATE = Gauge("miner_learning_rate", "Current learning rate")
MINER_LOCAL_STEP_RATE = Gauge("miner_local_step_rate", "Rate of completed iterations (steps/sec)")
MINER_TOKENS_PER_SEC = Gauge("miner_tokens_per_sec", "Throughput in tokens per second")
MINER_GRAD_ACCUM_STEPS = Gauge("miner_grad_accum_steps", "Gradient accumulation steps effectuated")

# MoE / Expert Routing (Gauges)
MOE_EXPERT_LOAD = Gauge("moe_expert_load", "Tokens routed to each expert", ["layer_idx", "expert_idx"])
MOE_AUX_LOSS = Gauge("moe_aux_loss", "Router load-balance loss")
MOE_EXPERTS_ACTIVE = Gauge("moe_experts_active_count", "Number of experts that received tokens in batch")
MOE_ROUTING_ENTROPY = Gauge("moe_topk_routing_entropy", "Diversity of routing decisions")
MOE_EXPERT_UTILIZATION = Gauge("moe_expert_utilization_ratio", "Utilization proportion per group/layer", ["group_idx", "layer_idx"])
MINER_PERPLEXITY = Gauge("miner_perplexity", "Training perplexity (exp of loss)")
MINER_TOTAL_TOKENS = Gauge("miner_total_tokens", "Cumulative tokens processed since run start")
MINER_TOTAL_SAMPLES = Gauge("miner_total_samples", "Cumulative samples processed since run start")
MINER_STEP_TIME_HOURS = Gauge("miner_step_time_hours", "Wall-clock time of the last inner step (hours)")
MINER_TOTAL_TRAINING_TIME_HOURS = Gauge("miner_total_training_time_hours", "Total accumulated training time (hours)")
MINER_PARAM_SUM = Gauge("miner_param_sum", "Sum of expert parameter values (health check)")

# Histograms (Latency & Sizes)
EVAL_LATENCY_SECONDS = Histogram("validator_eval_latency_seconds", "Latency of run_evaluation()")
MODEL_LOAD_LATENCY_SECONDS = Histogram("validator_model_load_latency_seconds", "Latency of load_model_from_path()")
CHAIN_COMMIT_LATENCY_SECONDS = Histogram("chain_commit_latency_seconds", "Time taken to commit to Bittensor")
CHECKPOINT_SAVE_LATENCY_SECONDS = Histogram("miner_checkpoint_save_latency_seconds", "Time taken to save and submit checkpoint")
CHECKPOINT_FETCH_LATENCY_SECONDS = Histogram("chain_checkpoint_fetch_duration_seconds", "How long downloading miner checkpoints takes")
CHAIN_CYCLE_LATENCY_SECONDS = Histogram("chain_cycle_duration_seconds", "Time per full chain cycle")
METAGRAPH_SYNC_LATENCY_SECONDS = Histogram(
    "validator_metagraph_sync_latency_seconds",
    "Latency of metagraph sync calls",
)
# Buckets cover ~1MB through ~10GB to fit miner checkpoint payloads. The
# prometheus_client default buckets are tuned for seconds and would all fall
# into the +Inf bucket here, making the histogram useless.
CHECKPOINT_DOWNLOAD_BYTES = Histogram(
    "validator_checkpoint_download_bytes",
    "Size of miner checkpoint payloads downloaded by the validator (bytes)",
    buckets=(1e6, 1e7, 5e7, 1e8, 5e8, 1e9, 5e9, 1e10, float("inf")),
)

# System & Errors
RPC_ERRORS_TOTAL = Counter("chain_rpc_errors_total", "Bittensor RPC/timeout errors")
CHAIN_WEIGHT_SET_SUCCESS = Counter("chain_weight_set_success", "Successful weight settings")
CHAIN_WEIGHT_SET_FAILURE = Counter("chain_weight_set_failure", "Failed weight settings")
ERRORS_TOTAL = Counter("connito_errors_total", "Errors counted by component and kind", ["component", "kind"])


EvalFailureReason = Literal[
    # Legacy buckets — kept callable for back-compat with older call sites
    # and dashboards. New code should prefer the miner-facing reasons below.
    "timeout", "corrupt", "oom", "checksum", "rpc", "unknown",
    # Miner-facing failure categories. Each maps to a stable integer code in
    # EVAL_STATUS_CODES so the gateway can render them without label changes.
    "deadline",
    "no_chain_commit",
    "signature_invalid",
    "hash_mismatch",
    "expert_group_or_nan",
    "non_finite_loss",
    "download_failed",
    "statedict_parse_failed",
    "repo_unavailable",
]
_EVAL_FAILURE_REASONS: frozenset[str] = frozenset({
    "timeout", "corrupt", "oom", "checksum", "rpc", "unknown",
    "deadline",
    "no_chain_commit", "signature_invalid", "hash_mismatch",
    "expert_group_or_nan", "non_finite_loss",
    "download_failed", "statedict_parse_failed",
    "repo_unavailable",
})

# Stable integer codes surfaced by VALIDATOR_MINER_EVAL_STATUS. Treat as a
# public contract — the gateway joins on these to produce miner-facing labels,
# and changing a code retroactively reinterprets old samples in Prometheus.
EVAL_STATUS_OK: int = 0
EVAL_STATUS_CODES: dict[int, str] = {
    0: "ok",
    1: "non_finite_loss",
    2: "statedict_parse_failed",
    3: "signature_invalid",
    4: "hash_mismatch",
    5: "expert_group_or_nan",
    6: "no_chain_commit",
    7: "download_failed",
    8: "oom",
    9: "timeout",
    10: "deadline_exceeded",
    11: "rpc_error",
    12: "repo_unavailable",
    99: "unknown",
}
_EVAL_REASON_TO_STATUS_CODE: dict[str, int] = {
    "non_finite_loss": 1,
    "statedict_parse_failed": 2,
    "signature_invalid": 3,
    "hash_mismatch": 4,
    "expert_group_or_nan": 5,
    "no_chain_commit": 6,
    "download_failed": 7,
    "oom": 8,
    "timeout": 9,
    "deadline": 10,
    "rpc": 11,
    "repo_unavailable": 12,
    # Legacy aliases — fold into the closest miner-facing code so old
    # call sites continue producing meaningful status values.
    "corrupt": 2,
    "checksum": 4,
    "unknown": 99,
}


def inc_error(component: str, kind: str) -> None:
    ERRORS_TOTAL.labels(component=component, kind=kind).inc()


def inc_eval_failure(miner_uid: int | str, reason: EvalFailureReason | str) -> None:
    """Record a miner eval failure. Unknown reasons are coerced to 'unknown' to keep cardinality bounded."""
    safe_reason = reason if reason in _EVAL_FAILURE_REASONS else "unknown"
    VALIDATOR_MINER_EVAL_FAILURES.labels(miner_uid=str(miner_uid), reason=safe_reason).inc()


def set_miner_eval_status(miner_uid: int | str, reason: EvalFailureReason | str | None) -> None:
    """Update the per-miner eval status gauge. Pass ``reason=None`` to mark
    the latest eval as OK (code 0). Unknown reasons coerce to code 99 so the
    gateway can still render *something* rather than dropping the sample.

    Best-effort — never raises. Telemetry must not influence scoring.
    """
    try:
        if reason is None:
            code = EVAL_STATUS_OK
        else:
            code = _EVAL_REASON_TO_STATUS_CODE.get(str(reason), 99)
        VALIDATOR_MINER_EVAL_STATUS.labels(miner_uid=str(miner_uid)).set(float(code))
    except Exception:
        pass


def set_miner_last_scored_round(miner_uid: int | str, round_id: int) -> None:
    """Record the round_id of the last finalize verdict for this miner.

    Best-effort — never raises. Telemetry must not influence scoring.
    """
    try:
        VALIDATOR_MINER_LAST_SCORED_ROUND_ID.labels(miner_uid=str(miner_uid)).set(
            float(int(round_id))
        )
    except Exception:
        pass


def set_miner_round_delta(miner_uid: int | str, delta: float) -> None:
    """Record the raw per-round improvement signal for an evaluated miner.

    Best-effort — never raises.
    """
    try:
        VALIDATOR_MINER_ROUND_DELTA.labels(miner_uid=str(miner_uid)).set(float(delta))
    except Exception:
        pass


def set_miner_evaluated_commit(
    miner_uid: int | str, hf_repo_id: str, hf_revision: str, round_id: int
) -> None:
    """Expose which (hf_repo_id, hf_revision) was frozen + evaluated for the
    miner, valued with the round_id it belongs to.

    Enforces at most ONE labelset per miner_uid: when the commit changes, the
    previous labelset is removed from the registry before the new one is set,
    so the gateway never sees two competing commit rows for a uid. The
    KeyError guard covers the post-restart case (tracking dict repopulated
    while the registry series was already re-created) and double-eviction.

    Best-effort — never raises.
    """
    try:
        uid = str(miner_uid)
        repo = str(hf_repo_id or "")
        rev = str(hf_revision or "")
        if not repo or not rev:
            return
        with _COMMIT_INFO_LOCK:
            prev = _COMMIT_INFO_LABELS.get(uid)
            if prev is not None and prev != (repo, rev):
                try:
                    VALIDATOR_MINER_EVALUATED_COMMIT_INFO.remove(uid, prev[0], prev[1])
                except KeyError:
                    pass
            VALIDATOR_MINER_EVALUATED_COMMIT_INFO.labels(
                miner_uid=uid, hf_repo_id=repo, hf_revision=rev
            ).set(float(int(round_id)))
            _COMMIT_INFO_LABELS[uid] = (repo, rev)
    except Exception:
        pass


def note_round_series(round_id: int) -> None:
    """Register a round_id whose label value was emitted on a per-round
    family, so `evict_round_series_before` can remove it later.

    Best-effort — never raises.
    """
    try:
        with _ROUND_SERIES_LOCK:
            _EMITTED_ROUND_IDS.add(int(round_id))
    except Exception:
        pass


def set_round_progress(
    round_id: int, *, scored: int, failed: int, pending: int
) -> None:
    """Publish a round's evaluation-progress counters.

    Shared by every path that advances a round: the freeze-time initial
    publish, foreground eval (during Submission), and the background eval
    worker (after Merge). This previously lived as a private static method
    on `BackgroundEvalWorker`, so the counters only started moving once the
    background eval window opened at Merge — foreground evals accumulated
    in `scored_uids` with nothing publishing them, and the dashboard's
    "Evaluated N of M" panel sat on the *previous* round's final value for
    the first ~14 minutes of every round, then jumped straight to the
    foreground total.

    Registers the round_id via `note_round_series` so these labelsets are
    evicted on the normal cutoff.

    Best-effort — never raises. Telemetry must not influence scoring.
    """
    try:
        note_round_series(int(round_id))
        rid = str(int(round_id))
        VALIDATOR_ROUND_MINERS_SCORED.labels(round_id=rid).set(float(scored))
        VALIDATOR_ROUND_MINERS_FAILED.labels(round_id=rid).set(float(failed))
        VALIDATOR_ROUND_MINERS_PENDING.labels(round_id=rid).set(float(pending))
    except Exception:
        pass


def set_baseline_loss(round_id: int, baseline_loss: float) -> None:
    """Publish a round's foreground-eval baseline loss to BOTH the unlabeled
    gauge (backward compat) and the per-round labeled family.

    Registers the round_id for eviction so the labeled series is pruned on
    the same cutoff as the other per-round families. Round creation already
    registers the id via note_round_series; the extra registration here is
    idempotent (a set) and keeps this helper self-contained.

    Best-effort — telemetry must never block scoring.
    """
    try:
        value = float(baseline_loss)
        VALIDATOR_BASELINE_LOSS.set(value)
        note_round_series(int(round_id))
        VALIDATOR_BASELINE_LOSS_BY_ROUND.labels(round_id=str(int(round_id))).set(value)
    except Exception:
        pass


def evict_round_series_before(min_round_id: int) -> int:
    """Remove per-round labelsets for every tracked round_id below the
    cutoff. Called from run.py's journal/aggregator prune block with the
    same cutoff, so metric retention matches on-disk retention.

    Only rounds emitted by THIS process are tracked (the set is in-memory);
    series left over from a previous process incarnation don't exist in the
    fresh registry either, so nothing is leaked across restarts.

    Returns the number of round_ids evicted. Best-effort — never raises.
    """
    removed = 0
    try:
        cutoff = int(min_round_id)
        with _ROUND_SERIES_LOCK:
            stale = [r for r in _EMITTED_ROUND_IDS if r < cutoff]
            for r in stale:
                rid = str(r)
                for family in (
                    VALIDATOR_ROUND_LIFECYCLE_STEP,
                    VALIDATOR_ROUND_MINERS_PENDING,
                    VALIDATOR_ROUND_MINERS_SCORED,
                    VALIDATOR_ROUND_MINERS_FAILED,
                    VALIDATOR_BG_EVAL_LOCK_LEAK_TOTAL,
                    VALIDATOR_BASELINE_LOSS_BY_ROUND,
                ):
                    try:
                        family.remove(rid)
                    except KeyError:
                        pass
                _EMITTED_ROUND_IDS.discard(r)
                removed += 1
    except Exception:
        return removed
    return removed


def set_miner_score_snapshot(
    miner_uid: int | str,
    *,
    latest: float | None,
    avg: float | None,
    samples: int | None,
) -> None:
    """Publish the aggregator's per-miner snapshot (latest / avg / sample
    count) to Prometheus. Each arg is independent — pass ``None`` to skip
    that gauge for this uid. Best-effort.

    Also stamps ``VALIDATOR_MINER_SCORE_LATEST_EMITTED_AT`` with the current
    wall-clock time whenever ``latest`` is published, so the gateway can
    derive a meaningful "score age" from the value's last-change time
    rather than Prometheus's scrape time (which advances every ~5s even
    when the gauge value hasn't changed).
    """
    try:
        now_ts: float | None = None
        if latest is not None:
            now_ts = time.time()
            VALIDATOR_MINER_SCORE_LATEST.labels(miner_uid=str(miner_uid)).set(float(latest))
            VALIDATOR_MINER_SCORE_LATEST_EMITTED_AT.labels(miner_uid=str(miner_uid)).set(now_ts)
        if avg is not None:
            VALIDATOR_MINER_SCORE_AVG.labels(miner_uid=str(miner_uid)).set(float(avg))
        if samples is not None:
            VALIDATOR_MINER_SCORE_SAMPLES.labels(miner_uid=str(miner_uid)).set(float(samples))
    except Exception:
        pass


# CONTRACT: COHORT_GROUP_CODES and ASSIGNMENT_ROLE_CODES are mirrored by the
# telemetry gateway (same pattern as EVAL_STATUS_CODES). Changing a code
# retroactively reinterprets old Prometheus samples — extend, do not renumber.
# Code 4 ("tail") = miners on this validator's foreground/background roster but
# outside the formal A/B/C tiers (the staleness pool). Distinct from 0 ("none"),
# which means the validator has no roster status for this miner this round.
COHORT_GROUP_CODES: dict[int, str] = {0: "none", 1: "A", 2: "B", 3: "C", 4: "tail"}
ASSIGNMENT_ROLE_CODES: dict[int, str] = {0: "unassigned", 1: "foreground", 2: "background"}


def set_miner_cohort_group(miner_uid: int | str, code: int) -> None:
    """Set the per-miner cohort-group gauge (see COHORT_GROUP_CODES). Pass 0
    for miners in no group this round. Best-effort — never raises.
    """
    try:
        VALIDATOR_MINER_COHORT_GROUP.labels(miner_uid=str(miner_uid)).set(float(code))
    except Exception:
        pass


def set_miner_assignment_role(miner_uid: int | str, code: int) -> None:
    """Set this validator's per-miner assignment-role gauge (see
    ASSIGNMENT_ROLE_CODES). Pass 0 for miners not on this validator's roster
    this round. Best-effort — never raises.
    """
    try:
        VALIDATOR_MINER_ASSIGNMENT_ROLE.labels(miner_uid=str(miner_uid)).set(float(code))
    except Exception:
        pass


def set_miner_last_observed_commit_block(miner_uid: int | str, block: int | float) -> None:
    """Record the round (freeze) block at which this validator last observed a
    valid chain commit for the miner. Best-effort — never raises.
    """
    try:
        VALIDATOR_MINER_LAST_OBSERVED_COMMIT_BLOCK.labels(miner_uid=str(miner_uid)).set(float(block))
    except Exception:
        pass


# ==============================================================================
# Decorators for Passive Tracing
# ==============================================================================

def track_eval_latency():
    """Tracks latency of miner validation evaluation"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with EVAL_LATENCY_SECONDS.time():
                return func(*args, **kwargs)
        return wrapper
    return decorator

def track_model_load_latency():
    """Tracks latency of pulling/loading miner state dicts"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with MODEL_LOAD_LATENCY_SECONDS.time():
                return func(*args, **kwargs)
        return wrapper
    return decorator

def track_chain_commit_latency():
    """Tracks latency of submitting weights or committing status"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with CHAIN_COMMIT_LATENCY_SECONDS.time():
                return func(*args, **kwargs)
        return wrapper
    return decorator

def track_metagraph_sync_latency():
    """Tracks latency of metagraph sync calls and stamps the last-sync timestamp on success."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with METAGRAPH_SYNC_LATENCY_SECONDS.time():
                result = func(*args, **kwargs)
            VALIDATOR_METAGRAPH_LAST_SYNC_TS.set(time.time())
            return result
        return wrapper
    return decorator

def count_rpc_errors():
    """Counts unhandled exceptions/RPC dropouts silently while re-raising them"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Naively cast everything as an RPC error count, or you could filter by exception type
                RPC_ERRORS_TOTAL.inc()
                raise e
        return wrapper
    return decorator


# ==============================================================================
# Background Poller for System State & Infrastructure Metrics
# ==============================================================================

class SystemStatePoller(threading.Thread):
    """
    A sidecar thread that sleeps natively and only wakes to sample
    the bittensor chain phase, DHT sizes, and GPU/CPU variables without
    blocking main worker threads.

    Metagraph-derived metrics (vtrust, consensus, deregistration churn) are
    expensive RPC calls and are throttled to once every
    ``metagraph_poll_every_n_polls`` ticks (default: every 5th poll, ~once per
    minute at the 12s default cadence).
    """
    def __init__(
        self,
        subtensor=None,
        phase_manager=None,
        group_averagers=None,
        netuid: int | None = None,
        validator_uid: int | None = None,
        interval_sec: float = 12.0,
        metagraph_poll_every_n_polls: int = 5,
    ):
        super().__init__(daemon=True)
        self.interval = interval_sec
        self.subtensor = subtensor
        self.phase_manager = phase_manager
        self.group_averagers = group_averagers
        self.netuid = netuid
        self.validator_uid = validator_uid
        self.metagraph_poll_every_n_polls = max(1, int(metagraph_poll_every_n_polls))
        self._stop_event = threading.Event()
        # Dedicated subtensor for this thread to avoid websocket collisions
        # with the caller's subtensor. Created lazily on first poll.
        self._local_subtensor = None
        self._poll_count: int = 0
        # Holds the prior tick's UID set so we can diff for deregistrations.
        # Stays empty until the first metagraph sync runs successfully; we
        # skip emitting the deregistration counter on that first tick.
        self._prior_uids: set[int] = set()
        # psutil.cpu_percent(interval=None) returns 0.0 on its very first call
        # because it has no previous sample to diff against. Prime it here so
        # the first real poll already has a usable baseline.
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            try:
                self._poll()
            except Exception as e:
                logger.debug(f"Telemetry sidecar hit an error: {e}")
            self._poll_count += 1
            self._stop_event.wait(self.interval)

    def _poll(self):
        # 1. Update Chain Block & Phase Variables
        if self.subtensor:
            try:
                # Dedicated connection for this thread to avoid websocket collisions.
                # Created once and reused across polls.
                if self._local_subtensor is None:
                    import bittensor
                    self._local_subtensor = bittensor.Subtensor(network=self.subtensor.network)
                block = self._local_subtensor.get_current_block()
                SUBNET_CURRENT_BLOCK.set(block)

                if self.phase_manager:
                    phase_resp = self.phase_manager.get_phase(block)
                    SUBNET_PHASE_INDEX.set(phase_resp.phase_index)
                    SUBNET_BLOCKS_REMAINING.set(phase_resp.blocks_remaining_in_phase)
            except Exception as e:
                logger.debug(f"Failed to fetch phase state inside poller: {e}")

        # 2. Track DHT peer sizes if Averagers exist (validator only)
        if self.group_averagers:
            total_peers = 0
            for avg in self.group_averagers.values():
                if hasattr(avg, 'total_size'):
                    total_peers += max(0, avg.total_size)
            DHT_PEER_COUNT.set(total_peers)

        # 3. Track explicit CUDA VRAM
        if torch.cuda.is_available():
            for dev_idx in range(torch.cuda.device_count()):
                try:
                    alloc = torch.cuda.memory_allocated(dev_idx)
                    peak = torch.cuda.max_memory_allocated(dev_idx)
                    GPU_VRAM_ALLOCATED_BYTES.labels(device=str(dev_idx)).set(alloc)
                    GPU_VRAM_PEAK_ALLOCATED_BYTES.labels(device=str(dev_idx)).set(peak)
                except Exception:
                    pass

        # 4. Host CPU utilization (cheap; runs every tick)
        try:
            SYSTEM_CPU_UTILIZATION_PERCENT.set(psutil.cpu_percent(interval=None))
        except Exception:
            pass

        # 5. Throttled metagraph fetch (vtrust / consensus / deregistration churn).
        # Fetching the metagraph is a multi-second RPC, so we only do it every
        # Nth poll. We also fire on the very first tick (poll_count == 0) so
        # dashboards aren't blank at startup.
        is_metagraph_tick = (
            self._poll_count == 0
            or self._poll_count % self.metagraph_poll_every_n_polls == 0
        )
        if is_metagraph_tick and self._local_subtensor is not None and self.netuid is not None:
            self._poll_metagraph()

    def _poll_metagraph(self) -> None:
        try:
            with METAGRAPH_SYNC_LATENCY_SECONDS.time():
                metagraph = self._local_subtensor.metagraph(self.netuid)
            VALIDATOR_METAGRAPH_LAST_SYNC_TS.set(time.time())
        except Exception as e:
            logger.debug(f"Failed to fetch metagraph in poller: {e}")
            return

        # Deregistration diff. Skip on the first successful sync — we have no
        # prior set to diff against, so every UID would falsely look "new".
        try:
            current_uids: set[int] = {int(u) for u in metagraph.uids.tolist()}
            if self._prior_uids:
                deregistered = self._prior_uids - current_uids
                if deregistered:
                    SUBNET_UID_DEREGISTRATIONS_TOTAL.inc(len(deregistered))
            self._prior_uids = current_uids
        except Exception as e:
            logger.debug(f"Failed to diff metagraph UIDs: {e}")

        # Self vtrust / consensus (only meaningful for this validator's UID).
        if self.validator_uid is None:
            return
        try:
            uid = int(self.validator_uid)
            if 0 <= uid < len(metagraph.uids):
                vtrust = float(metagraph.validator_trust[uid].item())
                consensus = float(metagraph.consensus[uid].item())
                SUBNET_VALIDATOR_VTRUST.set(vtrust)
                SUBNET_VALIDATOR_CONSENSUS.set(consensus)
        except Exception as e:
            logger.debug(f"Failed to read self vtrust/consensus: {e}")
