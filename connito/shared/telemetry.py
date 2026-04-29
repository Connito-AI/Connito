import os
import functools
import threading
import torch
import time
from typing import Callable, Any, Literal

import psutil
from prometheus_client import start_http_server, Counter, Gauge, Histogram

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
VALIDATOR_SCORE_STD = Gauge("validator_score_std", "Spread of miner scores")
VALIDATOR_AVG_STEP_STATUS = Counter("validator_avg_step_status", "Averager sync step stats", ["status"])
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


EvalFailureReason = Literal["timeout", "corrupt", "oom", "checksum", "rpc", "unknown"]
_EVAL_FAILURE_REASONS: frozenset[str] = frozenset(
    {"timeout", "corrupt", "oom", "checksum", "rpc", "unknown"}
)


def inc_error(component: str, kind: str) -> None:
    ERRORS_TOTAL.labels(component=component, kind=kind).inc()


def inc_eval_failure(miner_uid: int | str, reason: EvalFailureReason | str) -> None:
    """Record a miner eval failure. Unknown reasons are coerced to 'unknown' to keep cardinality bounded."""
    safe_reason = reason if reason in _EVAL_FAILURE_REASONS else "unknown"
    VALIDATOR_MINER_EVAL_FAILURES.labels(miner_uid=str(miner_uid), reason=safe_reason).inc()


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
