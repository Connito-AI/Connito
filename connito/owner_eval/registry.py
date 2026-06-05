"""Evaluator registry — the pluggability core of the owner eval pipeline.

An evaluator is any object exposing a ``name`` and an ``evaluate(model,
tokenizer, device, config) -> dict[str, float]`` method. Returning a dict lets a
single evaluator emit several scalars (e.g. an accuracy and its sample count),
each published as its own Prometheus gauge sample by the runner.

Add a metric by subclassing ``connito.owner_eval.metrics.base.BaseEvaluator``,
decorating it with ``@register``, importing it from
``connito/owner_eval/metrics/__init__.py`` (so the decorator runs), and adding
its ``name`` to ``config.eval_pipeline.enabled_metrics``.

This module is intentionally dependency-light (no torch/transformers) so it can
be imported and unit-tested in isolation.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from connito.shared.app_logging import structlog

logger = structlog.get_logger(__name__)


@runtime_checkable
class Evaluator(Protocol):
    name: str

    def evaluate(self, model: Any, tokenizer: Any, device: Any, config: Any) -> dict[str, float]:
        ...


# name -> Evaluator class. Populated by @register at import time.
REGISTRY: dict[str, type] = {}


def register(cls: type) -> type:
    """Class decorator that registers an evaluator under its ``name``."""
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"Evaluator {cls!r} must define a non-empty class attribute 'name'")
    if name in REGISTRY and REGISTRY[name] is not cls:
        raise ValueError(f"Duplicate evaluator name {name!r}: {REGISTRY[name]!r} vs {cls!r}")
    REGISTRY[name] = cls
    return cls


def build_enabled(config: Any) -> list[Evaluator]:
    """Instantiate the evaluators named in ``config.eval_pipeline.enabled_metrics``.

    Preserves config order; silently skips unknown names (with a warning) so a
    typo in one metric name doesn't take down the whole suite.
    """
    enabled = list(getattr(config.eval_pipeline, "enabled_metrics", []))
    built: list[Evaluator] = []
    for name in enabled:
        cls = REGISTRY.get(name)
        if cls is None:
            logger.warning("unknown evaluator in enabled_metrics; skipping", metric=name,
                           known=sorted(REGISTRY))
            continue
        built.append(cls())
    return built
