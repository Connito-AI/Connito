"""Standalone owner-run evaluation pipeline.

A daemon (``connito.owner_eval.run``) that the subnet owner runs independently of
any miner or validator. Every N cycles it downloads the latest merged full model
that a validator published to HuggingFace, runs a pluggable benchmark suite
(GSM8K perplexity, GSM8K task accuracy, MMLU accuracy to start), and emits the
results as Prometheus metrics for the leaderboard dashboard to scrape.

New metrics are added by dropping an ``Evaluator`` subclass into
``connito/owner_eval/metrics/`` and registering it with ``@register`` — see
``connito.owner_eval.registry``.
"""
