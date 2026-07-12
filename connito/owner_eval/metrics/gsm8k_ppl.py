"""GSM8K perplexity metric: exp(mean LM loss) over GSM8K test samples."""

from __future__ import annotations

import math
from typing import Any

from connito.shared.evaluate import evaluate_model
from connito.owner_eval.registry import register
from connito.owner_eval.metrics.base import BaseEvaluator, make_lm_batches, prep_tokenizer, load_hf_split


@register
class GSM8KPerplexity(BaseEvaluator):
    name = "gsm8k_ppl"

    def evaluate(self, model: Any, tokenizer: Any, device: Any, config: Any) -> dict[str, float]:
        prep_tokenizer(tokenizer)
        rows = load_hf_split("openai/gsm8k", "main", "test", self.n_samples(config))
        texts = [f"{r['question']}\n{r['answer']}" for r in rows]
        batches = make_lm_batches(
            texts, tokenizer,
            seq_len=config.eval_pipeline.eval_seq_length,
            batch_size=config.eval_pipeline.eval_batch_size,
        )
        metrics = evaluate_model(
            step=0,
            model=model,
            eval_dataloader=batches,
            device=device,
            max_eval_batches=None,
            rank=0,
        )
        val_loss = float(metrics["val_loss"])
        ppl = math.exp(val_loss) if math.isfinite(val_loss) else float("inf")
        return {"gsm8k_ppl": ppl, "gsm8k_ppl_val_loss": val_loss}
