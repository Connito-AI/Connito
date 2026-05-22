from __future__ import annotations

import gc
import time

import torch
from torch import nn
from tqdm import tqdm

from connito.shared.app_logging import structlog
from connito.shared.telemetry import (
    MINER_VAL_LOSS_BATCH_HISTOGRAM,
    MINER_VAL_LOSS_BY_SOURCE,
    MINER_VAL_SCORED_BATCHES_BY_SOURCE,
)

logger = structlog.getLogger(__name__)

tqdm(disable=True, total=0)


class EvalDeadlineExceeded(RuntimeError):
    """Raised by `evaluate_model` when its `deadline_monotonic` is crossed.

    Distinct from `asyncio.TimeoutError` so callers running the eval
    inside a thread (and therefore unable to be cancelled by
    `asyncio.wait_for`) can surface the deadline as a normal exception
    that unwinds locks via `with`/`finally`, instead of letting an
    awaiter cancellation orphan an in-flight GPU thread.
    """


def evaluate_model(
    step: int,
    model: nn.Module,
    eval_dataloader,
    device: torch.device,
    max_eval_batches: int | None = 50,
    rank: int | None = None,
    deadline_monotonic: float | None = None,
    source_attribution: dict[int, str] | None = None,
    expert_group: str | None = None,
) -> dict[str, float]:
    """
    Run a lightweight eval pass and return scalar metrics.

    Parameters
    ----------
    step : int
        Training step for logging context.
    model : nn.Module
        Fully-assembled model placed on the correct device.
    eval_dataloader :
        Iterable of evaluation batches (dicts of Tensors).
    device : torch.device
        Device to run evaluation on.
    max_eval_batches : Optional[int]
        Optional cap on the number of batches to evaluate.
    source_attribution : Optional[dict[int, str]]
        Optional mapping ``batch_idx -> source_name`` (e.g. ``"c4"``,
        ``"nemotron"``). When provided, per-source accumulators are
        tracked and emitted to Prometheus via
        ``MINER_VAL_LOSS_BY_SOURCE`` / ``MINER_VAL_LOSS_BATCH_HISTOGRAM``
        / ``MINER_VAL_SCORED_BATCHES_BY_SOURCE``, and the returned dict
        gains a ``by_source`` key. Batches whose index is absent from the
        map fall under the source label ``"unknown"``. Backwards-compatible:
        when ``None`` the legacy aggregate-only schema is returned and no
        per-source metrics are emitted.
    expert_group : Optional[str]
        Label value for the ``expert_group`` Prometheus dimension when
        per-source metrics are emitted. Defaults to ``"unknown_group"``
        to match `MetricLogger` convention. Ignored when
        ``source_attribution`` is ``None``.

    Returns
    -------
    Dict[str, float]
        e.g., {"val_loss": 2.345}. When ``source_attribution`` is passed,
        the dict also carries ``"by_source": {<src>: {"val_loss": ...,
        "scored_batches": ..., "val_aux_loss": ..., "nan_batches": ...}}``.
    """
    model.to(device)
    model.eval()
    loss_sum: float = 0.0
    aux_loss_sum: float = 0.0
    # Count of batches that produced a finite loss. The previous
    # implementation skipped NaN-loss batches from `loss_sum` but still
    # included them in the divisor, so a miner that crafts weights to
    # overflow logits to inf/NaN under bf16 autocast on a fraction `p` of
    # batches would report `(1 - p) * honest_loss` instead of
    # `honest_loss` — gaming the val_loss downward and inflating their
    # reward. Divide by `scored_batches` instead so a NaN/Inf batch
    # contributes nothing to either side of the ratio. Also skip
    # `aux_loss_sum` on NaN/Inf batches so a related variant (subtract
    # aux_loss but skip loss) cannot replicate the same exploit.
    scored_batches: int = 0
    nan_batches: int = 0
    batch_step = -1

    # Per-source accumulators. Only populated when the caller passes
    # `source_attribution`; absent otherwise so legacy callers retain
    # the exact pre-existing return schema.
    track_sources = source_attribution is not None
    per_source: dict[str, dict[str, float]] = {}

    def _bucket(src: str) -> dict[str, float]:
        if src not in per_source:
            per_source[src] = {
                "loss_sum": 0.0,
                "aux_loss_sum": 0.0,
                "scored_batches": 0,
                "nan_batches": 0,
            }
        return per_source[src]

    with torch.no_grad():
        for batch_step, batch in enumerate(iterable=eval_dataloader):
            # Per-batch deadline check. Raised before we start GPU work
            # for this batch so the caller's `with lock:` unwinds without
            # leaving an in-flight allocation. Granularity is one batch —
            # the eval loop cannot interrupt mid-forward — so the
            # effective bound is `deadline + one_batch_wall_time`.
            if deadline_monotonic is not None and time.monotonic() > deadline_monotonic:
                raise EvalDeadlineExceeded(
                    f"evaluate_model deadline exceeded at batch={batch_step} "
                    f"step={step} scored_batches={scored_batches}"
                )
            device_batch = {}
            for key in batch.keys():
                device_batch[key] = batch[key].to(model.device)

            if device_batch.get("attention_mask") is None and "input_ids" in device_batch:
                device_batch["attention_mask"] = torch.ones_like(device_batch["input_ids"])

            autocast_device = "cuda" if device.type == "cuda" else "cpu"
            eval_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            with torch.amp.autocast(autocast_device, dtype=eval_dtype):
                outputs = model(**device_batch)

                source_name = (
                    str(source_attribution.get(batch_step, "unknown"))
                    if track_sources
                    else None
                )

                if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
                    # NaN/Inf batches contribute 0 to both sums and do
                    # NOT increment `scored_batches`, so they drop out
                    # of the divisor as well. Explicit no-op `+= 0`
                    # keeps the parallel structure with the else-branch.
                    nan_batches += 1
                    if track_sources:
                        _bucket(source_name)["nan_batches"] += 1
                else:
                    loss_val = float(outputs.loss.item())
                    aux_val = (
                        float(outputs.aux_loss.item())
                        if hasattr(outputs, "aux_loss") and outputs.aux_loss is not None
                        else 0.0
                    )
                    loss_sum += loss_val
                    aux_loss_sum += aux_val
                    scored_batches += 1
                    if track_sources:
                        bucket = _bucket(source_name)
                        bucket["loss_sum"] += loss_val
                        bucket["aux_loss_sum"] += aux_val
                        bucket["scored_batches"] += 1
                        # Per-batch histogram observation. `loss_val` is
                        # the raw per-batch loss (already verified finite
                        # in the branch above). We observe the
                        # aux-loss-adjusted value to match what the
                        # caller's `delta = max(0, baseline - val_loss)`
                        # eventually compares against — keeping the
                        # histogram aligned with the gauge / aggregator.
                        try:
                            MINER_VAL_LOSS_BATCH_HISTOGRAM.labels(
                                expert_group=str(expert_group or "unknown_group"),
                                rank=str(rank if rank is not None else "0"),
                                source=source_name,
                            ).observe(loss_val - aux_val)
                        except Exception:
                            # Telemetry must never break scoring. Swallow
                            # and continue — same policy as the per-source
                            # gauges in `_finalize_per_source`.
                            pass

            del device_batch, outputs
            gc.collect()

            if max_eval_batches is not None and batch_step >= max_eval_batches:
                break

        logger.debug(
            "eval loss",
            loss_sum=round(loss_sum, 4),
            aux_loss_sum=round(aux_loss_sum, 4),
            scored_batches=scored_batches,
            nan_batches=nan_batches,
            step=step,
        )

    # Every batch was NaN/Inf — the miner's checkpoint is malicious or
    # broken. Return `+inf` so the caller's `delta = max(0, baseline -
    # val_loss)` clamps to 0 (score=0). Returning 0.0 would have given
    # them maximum delta.
    if scored_batches == 0:
        if nan_batches > 0:
            logger.warning(
                "evaluate_model: every eval batch produced NaN/Inf loss",
                nan_batches=nan_batches, step=step,
            )
        result: dict[str, float] = {
            "val_loss": float("inf"),
            "val_aux_loss": 0.0,
            "nan_batches": nan_batches,
            "scored_batches": 0,
        }
        if track_sources:
            result["by_source"] = _finalize_per_source(
                per_source,
                expert_group=expert_group,
                rank=rank,
            )
        return result

    val_loss = (loss_sum - aux_loss_sum) / scored_batches
    val_aux_loss = aux_loss_sum / scored_batches
    result = {
        "val_loss": val_loss,
        "val_aux_loss": val_aux_loss,
        "nan_batches": nan_batches,
        "scored_batches": scored_batches,
    }
    if track_sources:
        result["by_source"] = _finalize_per_source(
            per_source,
            expert_group=expert_group,
            rank=rank,
        )
    return result


def _finalize_per_source(
    per_source: dict[str, dict[str, float]],
    *,
    expert_group: str | None,
    rank: int | None,
) -> dict[str, dict[str, float]]:
    """Convert raw per-source accumulators into a finalized summary
    and emit them to Prometheus.

    Mirrors the aggregate finalization: ``val_loss = (loss_sum -
    aux_loss_sum) / scored_batches``, with ``+inf`` returned when a
    source produced no scored batches (so `max(0, baseline - val_loss)`
    in the caller still clamps to 0).

    Telemetry emission is best-effort — Prometheus failures never
    surface to the caller, because telemetry must never influence
    scoring.
    """
    label_group = str(expert_group or "unknown_group")
    label_rank = str(rank if rank is not None else "0")

    finalized: dict[str, dict[str, float]] = {}
    for source_name, bucket in per_source.items():
        scored = int(bucket["scored_batches"])
        nan = int(bucket["nan_batches"])
        if scored == 0:
            src_val_loss = float("inf")
            src_val_aux_loss = 0.0
        else:
            src_val_loss = (bucket["loss_sum"] - bucket["aux_loss_sum"]) / scored
            src_val_aux_loss = bucket["aux_loss_sum"] / scored

        finalized[source_name] = {
            "val_loss": src_val_loss,
            "val_aux_loss": src_val_aux_loss,
            "scored_batches": scored,
            "nan_batches": nan,
        }

        try:
            # `MINER_VAL_LOSS_BATCH_HISTOGRAM` is observed inside the
            # per-batch loop (one observation per scored batch) so the
            # histogram correctly captures the per-batch loss
            # distribution — emitting the mean once here would collapse
            # the distribution to a single point. We only set the
            # per-source aggregate gauge / scored-batches counter
            # here. Gauges accept ``+inf`` cleanly (just stamps that
            # value into the gauge); a 100%-NaN source therefore
            # surfaces as an explicit `+inf` reading in Prometheus —
            # which is exactly what callers want for the
            # `max(0, baseline - val_loss)` clamp logic.
            MINER_VAL_LOSS_BY_SOURCE.labels(
                expert_group=label_group,
                rank=label_rank,
                source=source_name,
            ).set(src_val_loss)
            MINER_VAL_SCORED_BATCHES_BY_SOURCE.labels(
                expert_group=label_group,
                rank=label_rank,
                source=source_name,
            ).set(scored)
        except Exception:
            # Telemetry must never break scoring. Swallow and continue.
            pass

    return finalized


def get_source_attribution(
    dataloader,
    max_batches: int,
    source_names: list[str] | None = None,
) -> dict[int, str]:
    """Build a ``batch_idx -> source_name`` mapping for a streaming
    eval pass.

    True per-sample attribution from an HF
    ``interleave_datasets``-driven loader is not possible: the
    interleaved iterable consumes from upstream sources eagerly inside
    a C-backed iterator and exposes no per-sample provenance field. The
    closest signal would be inspecting the dataset's ``info`` attribute,
    which only describes the merged dataset and does not carry which
    source produced each example.

    Limitation: as a result, this helper is intentionally a best-effort
    stub. When the caller can supply ``source_names`` (e.g. from the
    config's ``dataset_sources`` list, in the same order
    ``interleave_datasets`` was given them), we return a deterministic
    round-robin assignment weighted by the source order. Callers who
    need true per-source attribution should instead run two separate
    eval passes — one per source — and pass a uniform
    ``source_attribution`` dict for each pass.

    Returns
    -------
    dict[int, str]
        Mapping from batch index (0..max_batches-1) to source label.
        Empty when ``source_names`` is None.

    Examples
    --------
    >>> # Fallback: two-pass eval. Caller builds a dataloader per source
    >>> # and passes a uniform attribution.
    >>> attr_c4 = {i: "c4" for i in range(50)}
    >>> attr_nm = {i: "nemotron" for i in range(50)}
    """
    if not source_names:
        return {}

    # Round-robin assignment. Not true attribution, but produces a
    # well-defined batch_idx -> source mapping with the same cardinality
    # of source labels as the underlying interleave. Downstream consumers
    # (Unit 8 mini-eval, Unit 15 best-ckpt-selector) are expected to use
    # the two-pass strategy for accurate per-source val_loss.
    n_sources = len(source_names)
    return {i: source_names[i % n_sources] for i in range(max_batches)}
