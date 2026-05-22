"""Benchmarking and diagnostic utilities for the Connito miner.

Each module under ``bench`` is meant to be runnable standalone via
``python -m bench.<module>`` and to consume artefacts produced by the
training pipeline (CSV metric logs from ``connito.shared.metrics.MetricLogger``
or Prometheus ``/metrics`` scrapes).

The package is intentionally lightweight: only NumPy / pandas / matplotlib
are allowed as heavy dependencies, and ``matplotlib`` use must be guarded so
the diagnostics still run in headless / minimal envs (e.g. CI containers
without GUI backends).
"""

from __future__ import annotations

__all__: list[str] = []
