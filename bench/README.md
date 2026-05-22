# bench/local_scorer.py — validator-replica scorer

Local executable that replicates `connito/validator/evaluator.py`'s
per-miner scoring so you can:

1. Score a candidate miner checkpoint against the current baseline
   before committing it on chain.
2. Sample N random eval seeds to see how variable a miner's score is
   across the eval pool a validator might draw.
3. Pinpoint whether a checkpoint regresses on C4, Nemotron-CC-Math, or
   both (per-source mode).

## Scoring formula

Mirrors `connito/validator/evaluator.py:780-787`:

```
delta = max(0.0, baseline_loss - miner_val_loss)
score = delta ** 1.2
```

Where:

- `baseline_loss` = `evaluate_model` on the validator's global / baseline
  model against a 50-batch slice of the C4+Nemotron mix.
- `miner_val_loss = (loss_sum - aux_loss_sum) / scored_batches`. NaN/Inf
  batches drop out of `scored_batches` so a miner can't game the divisor.

Reward weight per round is rank-based (top-1 → 2.25, top-2 → 1.5,
top-3 → 1.0, otherwise 0); this scorer reports only the per-miner delta
+ score. The rank step happens upstream on chain.

## Usage

```bash
# Score a single seed
python -m bench.local_scorer \
    --checkpoint path/to/miner.safetensors \
    --baseline-checkpoint path/to/baseline.safetensors \
    --seed deadbeef0123 \
    --out report.json

# Sample 10 random seeds and report mean/std/min/max score
python -m bench.local_scorer \
    --checkpoint miner.safetensors \
    --baseline-checkpoint baseline.safetensors \
    --multi-seed 10 \
    --out variance.json

# Use specific validator combined seeds (e.g. pulled from chain)
python -m bench.local_scorer \
    --checkpoint miner.safetensors \
    --baseline-checkpoint baseline.safetensors \
    --validator-seeds abc123,def456,789xyz

# Per-source loss breakdown (doubles eval cost)
python -m bench.local_scorer \
    --checkpoint miner.safetensors \
    --baseline-checkpoint baseline.safetensors \
    --per-source
```

## CLI flags

| Flag | Default | Description |
| --- | --- | --- |
| `--checkpoint <path>` | required | Miner checkpoint to score (.safetensors / .pt). |
| `--baseline-checkpoint <path>` | required | Baseline (validator global) checkpoint. |
| `--seed <hex>` | random | Single eval seed. |
| `--validator-seeds <a,b,c>` | unset | Comma-separated list of explicit seeds. |
| `--multi-seed N` | 0 | Sample N random seeds; reports mean/std/min/max. |
| `--max-batches N` | 50 | Per-eval batch cap (matches validator's `EVAL_MAX_BATCHES`). |
| `--device cuda:0\|cpu` | `cuda:0` if avail else `cpu` | Torch device. |
| `--config <yaml>` | default `MinerConfig()` | Validator/miner YAML config. |
| `--expert-group <name>` | from config | Override `task.expert_group_name`. |
| `--per-source` | off | Compute per-source loss (C4 vs Nemotron). |
| `--out <path>` | stdout only | Write JSON report to file. |

## JSON output schema

```json
{
  "baseline_loss": 4.78,
  "miner_val_loss": 1.43,
  "delta": 3.35,
  "score": 4.36,
  "scored_batches": 50,
  "nan_batches": 0,
  "per_batch_loss": [1.41, 1.45, ...],
  "per_source_loss": {
    "allenai/c4": 1.42,
    "nvidia/Nemotron-CC-Math-v1": 1.44
  },
  "seeds_used": ["abc...", "def..."],
  "multi_seed_summary": {
    "mean_score": 4.30,
    "std_score": 0.12,
    "min_score": 4.10,
    "max_score": 4.55,
    "per_seed": [{"seed": "abc...", "val_loss": 1.43, "delta": 3.35, "score": 4.36}, ...]
  }
}
```

`multi_seed_summary` is `null` when only a single seed was evaluated.

## Limitations

- **Per-source loss is approximate.** `interleave_datasets` strips the
  source label from each yielded batch, so the per-source mode runs a
  separate eval pass per source (no interleave). The mixed-source
  `miner_val_loss` is still the canonical number; per-source is
  diagnostic. Costs 2× eval time.
- **Block-hash component of `combined_seed` not reproduced.** If you
  want to score against the *exact* seed a specific validator used,
  fetch that validator's `combined_seed` from chain commits and pass
  it via `--seed`. Otherwise random seeds suffice to estimate variance.
- **Eval pool randomization knobs follow the config**, including
  `eval_source_shuffle_buffer` (50_000) and `eval_source_skip_max`
  (50_000). Override these in the YAML config if you want a smaller
  pool for faster local iteration.

## Testing

```bash
pytest bench/                       # fast unit tests
RUN_LOCAL_SCORER_E2E=1 pytest bench/  # add the heavy E2E
```

## Why this exists

Every downstream optimization unit (LR sweeps, attention impl changes,
gradient-clip tuning, data-mix experiments, etc.) needs an offline
proxy for "would this checkpoint have won the round?" The on-chain
validator is the only ground truth, but it runs once per cycle and you
can't sweep against it. This scorer is that offline proxy.
