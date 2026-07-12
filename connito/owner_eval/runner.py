"""Run the enabled evaluator suite over a loaded model and emit Prometheus metrics."""

from __future__ import annotations

from typing import Any

from connito.shared.app_logging import structlog
from connito.shared import telemetry
from connito.owner_eval.registry import build_enabled

# Importing the metrics package registers the built-in evaluators.
import connito.owner_eval.metrics  # noqa: F401

logger = structlog.get_logger(__name__)


def run_eval_suite(model: Any, tokenizer: Any, device: Any, config: Any,
                   model_revision: str, cycle_index: int) -> dict[str, float]:
    """Evaluate every enabled metric and publish results to Prometheus.

    One failing evaluator never aborts the rest: its exception is logged, its
    status gauge set to 0, and an error counted. Returns the aggregated
    ``{metric_key: value}`` dict (useful for tests / one-shot runs).
    """
    results: dict[str, float] = {}
    evaluators = build_enabled(config)
    logger.info("running owner eval suite", count=len(evaluators),
                metrics=[e.name for e in evaluators], cycle_index=cycle_index,
                model_revision=model_revision)

    for ev in evaluators:
        try:
            with telemetry.EVAL_LATENCY_SECONDS.time():
                metric_values = ev.evaluate(model, tokenizer, device, config)
            for key, value in metric_values.items():
                telemetry.set_owner_eval_metric(key, float(value))
                results[key] = float(value)
            telemetry.set_owner_eval_status(ev.name, ok=True)
            logger.info("evaluator complete", metric=ev.name, values=metric_values)
        except Exception as exc:  # noqa: BLE001 — one bad metric must not sink the suite
            logger.warning("evaluator failed", metric=ev.name, error=str(exc), exc_info=True)
            telemetry.set_owner_eval_status(ev.name, ok=False)
            telemetry.inc_error("owner_eval", ev.name)

    telemetry.set_owner_eval_run_info(model_revision=model_revision, cycle_index=cycle_index)
    return results
