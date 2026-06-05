"""MMLU accuracy: score each of the 4 choices by log-likelihood, argmax vs gold."""

from __future__ import annotations

from typing import Any

import torch

from connito.shared.app_logging import structlog
from connito.owner_eval.registry import register
from connito.owner_eval.metrics.base import BaseEvaluator, prep_tokenizer, load_hf_split, loglikelihood

logger = structlog.get_logger(__name__)

_CHOICE_LABELS = ["A", "B", "C", "D"]


def _format_prompt(question: str, choices: list[str]) -> str:
    lines = [f"Question: {question}"]
    for label, choice in zip(_CHOICE_LABELS, choices):
        lines.append(f"{label}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


@register
class MMLUAccuracy(BaseEvaluator):
    name = "mmlu"

    @torch.no_grad()
    def evaluate(self, model: Any, tokenizer: Any, device: Any, config: Any) -> dict[str, float]:
        prep_tokenizer(tokenizer)
        # Seeded shuffle so an n < full set spans all 57 subjects representatively
        # (the test split is ordered by subject) and stays fixed across runs.
        rows = load_hf_split(
            "cais/mmlu", "all", "test", self.n_samples(config),
            seed=config.eval_pipeline.sample_seed,
        )
        normalize = config.eval_pipeline.mmlu_length_normalize

        correct = 0
        n = 0
        for r in rows:
            choices = list(r["choices"])
            prompt = _format_prompt(r["question"], choices)
            scores = [
                loglikelihood(model, tokenizer, prompt, f" {label}", device, normalize=normalize)
                for label in _CHOICE_LABELS[:len(choices)]
            ]
            pred = int(max(range(len(scores)), key=lambda i: scores[i]))
            correct += int(pred == int(r["answer"]))
            n += 1

        acc = correct / n if n else 0.0
        return {"mmlu_acc": acc, "mmlu_n": float(n)}
