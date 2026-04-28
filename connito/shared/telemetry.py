import os
import functools
import threading
import torch
import time
from typing import Callable, Any

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
GPU_VRAM_ALLOCATED_BYTES = Gauge("validator_vram_allocated_bytes", "VRAM allocated by operations", ["device"])
GPU_VRAM_PEAK_ALLOCATED_BYTES = Gauge("validator_vram_peak_allocated_bytes", "Peak VRAM allocated by operations", ["device"])
GPU_UTILIZATION_PERCENT = Gauge("system_gpu_utilization_percent", "GPU Utilization percent", ["device"])
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

# Histograms (Latency)
EVAL_LATENCY_SECONDS = Histogram("validator_eval_latency_seconds", "Latency of run_evaluation()")
MODEL_LOAD_LATENCY_SECONDS = Histogram("validator_model_load_latency_seconds", "Latency of load_model_from_path()")
CHAIN_COMMIT_LATENCY_SECONDS = Histogram("chain_commit_latency_seconds", "Time taken to commit to Bittensor")
CHECKPOINT_SAVE_LATENCY_SECONDS = Histogram("miner_checkpoint_save_latency_seconds", "Time taken to save and submit checkpoint")
CHECKPOINT_FETCH_LATENCY_SECONDS = Histogram("chain_checkpoint_fetch_duration_seconds", "How long downloading miner checkpoints takes")
CHAIN_CYCLE_LATENCY_SECONDS = Histogram("chain_cycle_duration_seconds", "Time per full chain cycle")

# System & Errors
RPC_ERRORS_TOTAL = Counter("chain_rpc_errors_total", "Bittensor RPC/timeout errors")
CHAIN_WEIGHT_SET_SUCCESS = Counter("chain_weight_set_success", "Successful weight settings")
CHAIN_WEIGHT_SET_FAILURE = Counter("chain_weight_set_failure", "Failed weight settings")
ERRORS_TOTAL = Counter("validator_errors_total", "Errors counted by component and kind", ["component", "kind"])


def inc_error(component: str, kind: str) -> None:
    ERRORS_TOTAL.labels(component=component, kind=kind).inc()

# ==============================================================================
# Validator health, per-phase timing, errors, foreground/background visibility,
# baseline loss, expanded per-miner scoring, and HF transport metrics.
# ==============================================================================

# Health
VALIDATOR_CYCLES_COMPLETED_TOTAL = Counter(
    "validator_cycles_completed_total",
    "Number of full validator cycles completed since process start",
)
VALIDATOR_LAST_CYCLE_TIMESTAMP = Gauge(
    "validator_last_cycle_timestamp",
    "Unix-seconds timestamp of the most recently completed validator cycle",
)
VALIDATOR_PARTICIPATED_IN_MERGE = Gauge(
    "validator_participated_in_merge",
    "1 if the validator contributed to the most recent allreduce, else 0",
)
VALIDATOR_INFO = Gauge(
    "validator_info",
    "Static label-only info gauge (always 1) describing this validator process",
    ["version", "git_sha", "hotkey_ss58", "expert_group"],
)

# Per-phase timing
VALIDATOR_PHASE_DURATION_SECONDS = Histogram(
    "validator_phase_duration_seconds",
    "Wall-clock duration of each validator phase block",
    ["phase"],
)

# Errors
VALIDATOR_ERRORS_TOTAL = Counter(
    "validator_errors_total",
    "Validator-side error count, partitioned by component and kind",
    ["component", "kind"],
)

# Foreground / background lane visibility
VALIDATOR_ROUND_ROSTER_SIZE = Gauge(
    "validator_round_roster_size",
    "Number of miners in each lane (foreground/background) for the round",
    ["round_id", "lane"],
)
VALIDATOR_ROUND_MINER_LANE = Gauge(
    "validator_round_miner_lane",
    "1 while the (uid, lane) pair is part of the active round; cleared when round ends",
    ["round_id", "miner_uid", "lane"],
)

# Baseline + per-miner eval visibility
VALIDATOR_BASELINE_LOSS = Gauge(
    "validator_baseline_loss",
    "Baseline (un-merged) eval loss for the round, used as the reference for delta scoring",
    ["round_id"],
)
VALIDATOR_MINER_DELTA_LOSS = Gauge(
    "validator_miner_delta_loss",
    "baseline_loss - val_loss for the miner this round (raw input to the score formula)",
    ["round_id", "miner_uid", "lane"],
)
VALIDATOR_MINER_EVAL_LATENCY_SECONDS = Histogram(
    "validator_miner_eval_latency_seconds",
    "Per-miner evaluation latency, tagged by lane",
    ["lane"],
)

# Per-miner scoring (extends VALIDATOR_MINER_SCORE which holds the latest score)
VALIDATOR_MINER_SCORE_AVG = Gauge(
    "validator_miner_score_avg",
    "Rolling-average miner score actually used when computing chain weights",
    ["miner_uid"],
)
VALIDATOR_MINER_SCORE_POINTS = Gauge(
    "validator_miner_score_points",
    "Number of scoring points retained for this miner (capped at score_window)",
    ["miner_uid"],
)
VALIDATOR_MINER_CHAIN_WEIGHT = Gauge(
    "validator_miner_chain_weight",
    "Final normalized chain weight submitted for this miner in the most recent cycle",
    ["miner_uid"],
)

# HF transport
HF_TRANSFER_BYTES_TOTAL = Counter(
    "hf_transfer_bytes_total",
    "Bytes transferred via HuggingFace, partitioned by direction and kind",
    ["direction", "kind"],
)
HF_TRANSFER_LATENCY_SECONDS = Histogram(
    "hf_transfer_latency_seconds",
    "Wall-clock latency of HuggingFace transfers",
    ["direction", "kind"],
)
HF_TRANSFER_RESULT_TOTAL = Counter(
    "hf_transfer_result_total",
    "HuggingFace transfer outcomes",
    ["direction", "kind", "result"],
)

# Background worker outcomes
BG_DOWNLOAD_RESULT_TOTAL = Counter(
    "bg_download_result_total",
    "BackgroundDownloadWorker per-attempt outcomes",
    ["result"],
)
BG_EVAL_RESULT_TOTAL = Counter(
    "bg_eval_result_total",
    "BackgroundEvalWorker per-attempt outcomes",
    ["result"],
)


def inc_error(component: str, kind: str = "unknown") -> None:
    """Bump the validator_errors_total counter with a standardized label set.

    `component` ∈ {hf_upload, hf_download, foreground_eval, bg_download,
    bg_eval, chain_commit, allreduce, peer_sync, weight_submit, eval_load}.
    `kind` ∈ {timeout, oom, network, validation, unknown}. Use `unknown` when
    the call site doesn't know more than "something raised."
    """
    VALIDATOR_ERRORS_TOTAL.labels(component=component, kind=kind).inc()


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
    the bittensor chain phase, DHT sizes, and GPU variables without
    blocking main worker threads.
    """
    def __init__(self, subtensor=None, phase_manager=None, group_averagers=None, interval_sec: float = 12.0):
        super().__init__(daemon=True)
        self.interval = interval_sec
        self.subtensor = subtensor
        self.phase_manager = phase_manager
        self.group_averagers = group_averagers
        self._stop_event = threading.Event()
        # Dedicated subtensor for this thread to avoid websocket collisions
        # with the caller's subtensor. Created lazily on first poll.
        self._local_subtensor = None
        
    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            try:
                self._poll()
            except Exception as e:
                logger.debug(f"Telemetry sidecar hit an error: {e}")
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
