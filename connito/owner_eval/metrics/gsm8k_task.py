"""GSM8K task score: greedy-decode each question, exact-match the final number."""

from __future__ import annotations

from typing import Any

import torch

from connito.shared.app_logging import structlog
from connito.owner_eval.registry import register
from connito.owner_eval.metrics.base import (
    BaseEvaluator,
    prep_tokenizer,
    load_hf_split,
    extract_final_number,
    extract_gsm8k_gold,
    numeric_eq,
)

logger = structlog.get_logger(__name__)

# Minimal zero-shot framing; nudges the model to emit a final answer we can parse.
_PROMPT_TEMPLATE = "Question: {q}\nAnswer:"


def _build_prompt(question: str) -> str:
    return _PROMPT_TEMPLATE.format(q=question)


@register
class GSM8KTaskScore(BaseEvaluator):
    name = "gsm8k_task"

    @torch.no_grad()
    def evaluate(self, model: Any, tokenizer: Any, device: Any, config: Any) -> dict[str, float]:
        prep_tokenizer(tokenizer)
        rows = load_hf_split("openai/gsm8k", "main", "test", self.n_samples(config))
        max_new_tokens = config.eval_pipeline.gsm8k_max_new_tokens

        correct = 0
        n = 0
        for r in rows:
            n += 1
            enc = tokenizer(_build_prompt(r["question"]), return_tensors="pt").to(device)
            prompt_len = enc["input_ids"].shape[1]
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            completion = tokenizer.decode(gen[0, prompt_len:], skip_special_tokens=True)
            pred = extract_final_number(completion)
            gold = extract_gsm8k_gold(r["answer"])
            correct += int(numeric_eq(pred, gold))

        acc = correct / n if n else 0.0
        return {"gsm8k_task_acc": acc, "gsm8k_task_n": float(n)}
