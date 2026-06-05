"""Importing this package registers all built-in evaluators via ``@register``.

Add a new metric module here so its registration side-effect runs.
"""

from connito.owner_eval.metrics import gsm8k_ppl, gsm8k_task, mmlu  # noqa: F401

__all__ = ["gsm8k_ppl", "gsm8k_task", "mmlu"]
