"""Base evaluator + shared dataset/scoring helpers for owner eval metrics."""

from __future__ import annotations

from typing import Any

import torch

from connito.shared.app_logging import structlog
from connito.owner_eval.text_utils import (  # re-exported for metric modules/tests
    extract_final_number,
    extract_gsm8k_gold,
    numeric_eq,
)

logger = structlog.get_logger(__name__)

__all__ = [
    "BaseEvaluator",
    "prep_tokenizer",
    "load_hf_split",
    "make_lm_batches",
    "loglikelihood",
    "extract_final_number",
    "extract_gsm8k_gold",
    "numeric_eq",
]


class BaseEvaluator:
    """Common base for registered evaluators.

    Subclasses set ``name`` (the Prometheus metric label / registry key) and
    implement ``evaluate``. ``default_n_samples`` is a per-class fallback used
    when the metric isn't listed in ``config.eval_pipeline.n_samples_per_metric``.
    """

    name: str = ""
    default_n_samples: int = 100

    def n_samples(self, config: Any) -> int:
        per_metric = getattr(config.eval_pipeline, "n_samples_per_metric", {}) or {}
        return int(per_metric.get(self.name, getattr(config.eval_pipeline, "default_n_samples", self.default_n_samples)))

    def evaluate(self, model: Any, tokenizer: Any, device: Any, config: Any) -> dict[str, float]:
        raise NotImplementedError


def prep_tokenizer(tokenizer):
    """Ensure a pad token exists (DeepSeek's tokenizer ships without one)."""
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_hf_split(path: str, name: str | None, split: str, n: int, seed: int | None = None):
    """Load up to ``n`` rows of an HF dataset split (non-streaming; n is small).

    When ``seed`` is given, the split is shuffled with that fixed seed before
    selecting ``n`` rows. Use this for datasets whose row order is itself
    structured — e.g. ``cais/mmlu`` "all" is ordered by subject, so taking the
    first n rows would only cover the alphabetically-earliest subjects. A seeded
    shuffle yields a representative sample across subjects while staying
    deterministic across runs.
    """
    from datasets import load_dataset

    ds = load_dataset(path, name, split=split) if name else load_dataset(path, split=split)
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    n = min(n, len(ds))
    return ds.select(range(n))


def make_lm_batches(texts, tokenizer, seq_len: int, batch_size: int):
    """Tokenize ``texts`` into causal-LM batches compatible with ``evaluate_model``.

    Yields dicts of ``{input_ids, attention_mask, labels}`` (labels == input_ids
    with pad positions masked to -100), the same shape the validator's
    DataCollatorForLanguageModeling(mlm=False) produces.
    """
    batches = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        enc = tokenizer(
            chunk,
            truncation=True,
            max_length=seq_len,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        batches.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })
    return batches


@torch.no_grad()
def loglikelihood(model, tokenizer, prompt: str, continuation: str, device, normalize: bool = True) -> float:
    """Teacher-forced log-likelihood of ``continuation`` given ``prompt``.

    Single forward pass; sums the log-probabilities the model assigns to the
    continuation tokens. When ``normalize`` is set, divides by the number of
    continuation tokens (length-normalised, the standard for comparing MMLU
    choices of differing token length).
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    full_ids = tokenizer(prompt + continuation, return_tensors="pt")["input_ids"][0]
    cont_len = full_ids.shape[0] - prompt_ids.shape[0]
    if cont_len <= 0:
        return float("-inf")

    input_ids = full_ids.unsqueeze(0).to(device)
    logits = model(input_ids=input_ids).logits  # [1, T, V]
    # logits[:, t] predicts token t+1; gather over the continuation span.
    log_probs = torch.log_softmax(logits[0, :-1].float(), dim=-1)  # [T-1, V]
    targets = full_ids[1:].to(device)  # [T-1]
    cont_log_probs = log_probs[-cont_len:].gather(1, targets[-cont_len:].unsqueeze(1)).squeeze(1)
    total = float(cont_log_probs.sum().item())
    return total / cont_len if normalize else total
