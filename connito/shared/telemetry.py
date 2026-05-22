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


def set_validator_identity(*, hotkey: str, uid: int | None, version: str, netuid: int) -> None:
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

# Per-layer routing-entropy histogram. Observe one sample per (rank, layer)
# every metric_interval steps. Buckets cover the typical range for a
# `num_experts=8, top_k=2` router: uniform-random routing peaks near
# ln(8) ≈ 2.08, full collapse onto a single expert is 0.0. When the bulk of
# samples drifts below ~0.5 (see `router_collapse_alert_entropy_threshold`)
# the router is concentrating on a handful of experts and the aux_loss_coef
# should be raised back toward 1.0. Cardinality: groups × ranks × MoE layers,
# bounded by the model architecture (DeepSeek-V2-Lite has 26 hidden layers,
# of which ~25 are MoE — well within Prometheus headroom).
MOE_ROUTING_ENTROPY_HIST = Histogram(
    "moe_routing_entropy_distribution",
    "Routing entropy per layer",
    labelnames=("expert_group", "rank", "layer"),
    buckets=(0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, float("inf")),
)
# Per-expert token-load distribution. Each observe() is the fraction of
# routed tokens that landed on a given expert in the most recent batch.
# Uniform routing across 8 experts → 1/8 = 0.125 per expert; full collapse
# → one expert at 1.0 and the rest at 0.0. Pair with
# `moe_routing_entropy_distribution` to detect "high entropy but one expert
# still dominates" (e.g. mass concentrated near the 0.5 bucket boundary).
MOE_EXPERT_LOAD_HIST = Histogram(
    "moe_expert_load_distribution",
    "Per-expert token load distribution",
    labelnames=("expert_group", "rank", "expert_id"),
    buckets=(0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1.0, float("inf")),
)
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
]
_EVAL_FAILURE_REASONS: frozenset[str] = frozenset({
    "timeout", "corrupt", "oom", "checksum", "rpc", "unknown",
    "deadline",
    "no_chain_commit", "signature_invalid", "hash_mismatch",
    "expert_group_or_nan", "non_finite_loss",
    "download_failed", "statedict_parse_failed",
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
    """
    try:
        if latest is not None:
            VALIDATOR_MINER_SCORE_LATEST.labels(miner_uid=str(miner_uid)).set(float(latest))
        if avg is not None:
            VALIDATOR_MINER_SCORE_AVG.labels(miner_uid=str(miner_uid)).set(float(avg))
        if samples is not None:
            VALIDATOR_MINER_SCORE_SAMPLES.labels(miner_uid=str(miner_uid)).set(float(samples))
    except Exception:
        pass


# ==============================================================================
# MoE routing telemetry helpers
# ==============================================================================
#
# These helpers observe the MoE router's actual behaviour at training time so
# operators can detect router collapse (one expert absorbs most tokens) as
# soon as it starts. They are best-effort: any exception (model lacks an MoE
# block, hooks fail to attach on a custom layout, no batches have been routed
# yet) is swallowed silently. Telemetry must never block or break training.
#
# The validator scores miners by `(loss_sum - aux_loss_sum) / scored_batches`
# (see connito/shared/evaluate.py:146), so the LM loss term dominates the
# reward. Lowering `moe.router_aux_loss_coef` (1.0 → 0.01) frees gradient
# capacity for the LM loss but increases collapse risk — that risk is what
# these histograms surface.

# Attribute name used to stash the most recent router observation on each
# MoE block. Picked to be unlikely to collide with any module field.
_MOE_ROUTING_OBS_ATTR = "_connito_routing_obs"
# Attribute used to retain the original `route_tokens_to_experts` method
# after monkey-patching, so we can re-attach idempotently without stacking
# wrappers and so future detach work is possible.
_MOE_ROUTING_ORIGINAL_FN_ATTR = "_connito_routing_original_fn"


def _make_routing_capture_wrapper(moe_block: Any, original_fn: Any) -> Any:
    """Return a function that calls ``original_fn`` and caches the topk
    indices it returned onto ``moe_block`` under
    ``_MOE_ROUTING_OBS_ATTR``.

    PyTorch ``register_forward_hook`` only fires on ``Module.__call__``,
    so it cannot intercept a method that the parent ``forward`` invokes
    internally. The MoE block's main ``forward()`` returns the routed
    hidden states (not the topk indices we need for entropy), so we wrap
    the ``route_tokens_to_experts`` method itself. The wrapped function
    preserves the original signature and return value byte-for-byte so
    nothing in the routing path changes — we only piggyback a CPU
    snapshot for later metric emission.
    """

    def wrapped(router_logits: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        result = original_fn(router_logits, *args, **kwargs)
        try:
            if isinstance(result, tuple) and len(result) >= 1:
                topk_idx = result[0]
                if torch.is_tensor(topk_idx):
                    num_experts = int(
                        getattr(moe_block, "num_experts", None)
                        or getattr(getattr(moe_block, "config", None), "n_routed_experts", 0)
                        or 0
                    )
                    if num_experts > 0:
                        # Snapshot on the SAME device the routing tensor
                        # already lives on. A `.to("cpu", copy=True)` here
                        # would force a synchronous device-to-host copy on
                        # every forward pass — for DeepSeek-V2-Lite that
                        # is ~25 D2H stalls per microbatch, which silently
                        # caps training throughput. We accumulate on the
                        # original device (the bincount in emit() also
                        # runs on-device, cheap) and only cross to CPU
                        # when we actually need to emit a metric.
                        setattr(
                            moe_block,
                            _MOE_ROUTING_OBS_ATTR,
                            {
                                "topk_idx": topk_idx.detach().clone(),
                                "num_experts": num_experts,
                            },
                        )
        except Exception:
            # Telemetry must never break training. Drop the sample silently.
            pass
        return result

    return wrapped


def _iter_moe_blocks(model: Any) -> list[tuple[int, Any]]:
    """Walk `model.named_modules()` and return `(layer_idx, moe_block)`
    pairs for every MoE block carrying a `route_tokens_to_experts` method.

    The layer index is extracted from the module path
    (`model.layers.<i>.mlp`); blocks without a parseable index are skipped
    so we never observe a metric without a meaningful `layer` label.
    """
    import re

    # Match `layers.<i>` anywhere in the dotted module path. We anchor on
    # a word boundary instead of a leading dot because the path may start
    # with `layers.` (when the model itself is the root, e.g. the unit
    # test fake) or `model.layers.` (the production DeepSeek-V2-Lite
    # layout). Either way the integer immediately after `layers.` is the
    # layer index we want.
    layer_re = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
    found: list[tuple[int, Any]] = []
    try:
        for name, module in model.named_modules():
            if not hasattr(module, "route_tokens_to_experts"):
                continue
            m = layer_re.search(name)
            if m is None:
                continue
            found.append((int(m.group(1)), module))
    except Exception:
        pass
    return found


def attach_moe_routing_telemetry_hooks(model: Any) -> list[Any]:
    """Install the routing-capture wrapper on every MoE block's
    `route_tokens_to_experts` method.

    Returns the list of moe blocks that were newly wrapped, so callers
    can introspect the count for logging. Idempotent: blocks already
    carrying a saved original function are skipped so re-calling does
    not stack wrappers.
    """
    wrapped_blocks: list[Any] = []
    for _layer_idx, moe_block in _iter_moe_blocks(model):
        try:
            if getattr(moe_block, _MOE_ROUTING_ORIGINAL_FN_ATTR, None) is not None:
                continue
            original_fn = moe_block.route_tokens_to_experts
            setattr(moe_block, _MOE_ROUTING_ORIGINAL_FN_ATTR, original_fn)
            moe_block.route_tokens_to_experts = _make_routing_capture_wrapper(
                moe_block, original_fn,
            )
            wrapped_blocks.append(moe_block)
        except Exception:
            # Best-effort: any failure to wrap one block must not block
            # the others or training itself.
            continue
    return wrapped_blocks


def emit_moe_routing_telemetry(
    model: Any,
    *,
    expert_group: str,
    rank: int | str,
) -> None:
    """Compute and observe routing entropy / per-expert load histograms.

    For each MoE block with a fresh observation, compute:
      * `H(p) = -Σ p_i log p_i` over the per-expert token-fraction vector,
        observed onto `MOE_ROUTING_ENTROPY_HIST{expert_group, rank, layer}`.
      * Per-expert load `p_i`, observed onto
        `MOE_EXPERT_LOAD_HIST{expert_group, rank, expert_id}`.

    `expert_group` and `rank` are the same labels already used by other
    miner-side metrics (e.g. `miner_training_loss{expert_group=...}`), and
    `layer` and `expert_id` are emitted as strings to keep label types
    stable across architectures. Best-effort: any per-block exception is
    swallowed so partial telemetry still lands.
    """
    blocks = _iter_moe_blocks(model)
    if not blocks:
        return
    rank_str = str(rank)
    group_str = str(expert_group)
    for layer_idx, moe_block in blocks:
        obs = getattr(moe_block, _MOE_ROUTING_OBS_ATTR, None)
        if obs is None:
            continue
        try:
            topk_idx = obs["topk_idx"]
            num_experts = int(obs["num_experts"])
            if num_experts <= 0 or topk_idx.numel() == 0:
                continue
            # Per-expert token count: bincount over the flat topk index
            # tensor. Each token contributes `top_k` slots (one per
            # selected expert), so the sum equals num_tokens * top_k.
            # We run bincount on whatever device the stash lives on (GPU
            # if training on GPU) and only cross to CPU on the small
            # `probs` vector (size = num_experts, typically 8), so a
            # single emit triggers one tiny D2H copy per layer instead
            # of one full topk_idx D2H copy per forward pass.
            flat = topk_idx.reshape(-1).to(torch.long)
            counts = torch.bincount(flat, minlength=num_experts).to(torch.float64)
            total = float(counts.sum().item())
            if total <= 0.0:
                continue
            probs = counts / total
            # Routing entropy in nats: -Σ p log p, with 0 log 0 = 0.
            log_probs = torch.where(
                probs > 0,
                torch.log(probs.clamp(min=1e-12)),
                torch.zeros_like(probs),
            )
            entropy = float(-(probs * log_probs).sum().item())
            probs_cpu = probs.detach().to("cpu").tolist()
            MOE_ROUTING_ENTROPY_HIST.labels(
                expert_group=group_str,
                rank=rank_str,
                layer=str(layer_idx),
            ).observe(entropy)
            for expert_id, load in enumerate(probs_cpu):
                MOE_EXPERT_LOAD_HIST.labels(
                    expert_group=group_str,
                    rank=rank_str,
                    expert_id=str(expert_id),
                ).observe(float(load))
        except Exception:
            # Per-block failure must not skip the remaining blocks.
            continue


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
