# Miner optimization playbook

Operator playbook for tuning a Connito SN102 miner to climb the leaderboard. Pairs with `docs/SCORING_INTERNALS.md` for the validator-side mechanics.

---

## TL;DR

```
score_per_round = (max(0, baseline_loss - your_val_loss)) ** 1.2
rank-based:     top-1 -> 2.25 | top-2 -> 1.5 | top-3 -> 1.0 | else 0
chain weight:   8-round avg-over-window, top-3 split 98%, next-5 split 2%
```

Three rules:

1. **Lower your val_loss against the eval pool** (allenai/c4@en + nvidia/Nemotron-CC-Math-v1@4plus, 50 batches, seq_len 4096).
2. **Never score 0** (NaN/Inf, hash/sig fail, expert-group violation, exactly-tied val_loss with another miner).
3. **Be robust across many seeds**, not just one — per-validator weight variance is large; a single peak does not survive.

See section "Knobs by impact" for what to turn on, in what order, with rough delta-gain estimates.

---

## Knobs by impact

Estimated `Δ val_loss` ranges are based on internal sweeps and the dashboard reference (`baseline ≈ 4.78`, top val_loss ≈ 1.43). Treat them as rough — your mileage depends on starting point.

| Rank | Knob | Estimated Δ val_loss | Cost | Sub-section |
| --- | --- | --- | --- | --- |
| 1 | Flash Attention 2 | indirect: -0.1 to -0.3 via more training tokens | low (deps) | A |
| 2 | Sequence packing | indirect: -0.1 to -0.4 via more useful tokens/batch | low | B |
| 3 | EMA checkpoint (vs last step) | -0.05 to -0.2 | very low | C |
| 4 | Best-checkpoint selection before submit | -0.05 to -0.2 (variance reduction) | low | D |
| 5 | Tune `router_aux_loss_coef` DOWN | small, but free since validator subtracts it | none | E |
| 6 | LR with warmup, cosine | -0.1 to -0.3 | medium (tuning) | F |
| 7 | Loss-spike batch filter | -0.02 to -0.1 + stability | low | G |
| 8 | Data quality filter (Nemotron+C4) | -0.05 to -0.2 | medium | H |

### A. Flash Attention 2

`pip install flash-attn --no-build-isolation` and pass `attn_implementation="flash_attention_2"` to the model factory. 30-50% throughput gain at seq_len=4096. Bf16/fp16 only (matches the validator's eval dtype, see `connito/shared/evaluate.py:95-96`). More tokens trained in the same wall-clock budget -> measurably lower val_loss.

Verify with a smoke test before committing the change:

```bash
python -c "import torch; from transformers import AutoModelForCausalLM; \
  m = AutoModelForCausalLM.from_pretrained('your-base', attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16); \
  print('flash_attn ok')"
```

Rollback: drop the `attn_implementation` kwarg.

### B. Sequence packing

Trains on more useful tokens per batch by concatenating short samples up to `sequence_length` with proper attention masks. 2-3x effective batch size for short-text-heavy mixes like C4 (median ~500 tokens) plus Nemotron (highly variable). Implementations: TRL's `ConstantLengthDataset`, HF's `pack` argument, or hand-rolled.

Be careful: cross-sample attention bleed inflates training loss. Use a 4D attention mask or position_ids that reset on sample boundary.

### C. EMA checkpoint

Track an EMA of model parameters during training (`decay=0.999`) and submit the EMA weights, not the latest optimizer step. Almost always lower val_loss than last step. Add ~1x model memory cost on the host.

Minimal recipe:

```python
ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
# inside step loop, after optimizer.step():
for k, v in model.state_dict().items():
    ema[k].mul_(0.999).add_(v.detach().float(), alpha=0.001)
# at checkpoint time:
torch.save(ema, "ema.pt")  # submit this
```

### D. Best-checkpoint selection before submit

Per-validator weight variance is large (dashboard 2026-05-22 shows substantial validator-to-validator divergence even for UID 79). Counter this by:

1. Training to N candidate checkpoints (last K saves, plus EMA).
2. Running each through a **local eval** that mimics the validator's pool (see `bench/local_scorer.py`).
3. Submitting the one with the best **mean val_loss across many seeds**, not the best on any single seed.

Multi-seed mean is the metric to optimize, not single-seed peak.

### E. Tune `router_aux_loss_coef` DOWN

```python
# connito/shared/config.py:283-296
class MoECfg(BaseConfig):
    aux_load_balance: bool = True
    router_aux_loss_coef: float = 1.0
```

The validator subtracts `aux_loss` from val_loss anyway (`val_loss = (loss_sum - aux_loss_sum) / scored_batches`, `connito/shared/evaluate.py:146`). So tuning this knob does NOT affect your scored val_loss directly — but it DOES affect training dynamics:

- High `router_aux_loss_coef` (1.0+) -> balanced routing, stable training, but the auxiliary term competes with cross-entropy in the gradient.
- Low (0.01-0.1) -> the model can route greedily, potentially specializing experts harder, but risks dead-expert pathologies.

Recommendation: start at 0.01-0.1 with `aux_load_balance=True`. Add expert-utilization monitoring to catch dead experts.

### F. LR with warmup

```python
# config.py:298-301
class OptimizerCfg(BaseConfig):
    lr: float = 1e-5
    outer_lr: float = 0.7
    outer_momentum: float = 0.9
```

Default `lr=1e-5` is conservative for an MoE. Try:

- Inner: `2e-5` to `5e-5` with linear warmup over the first ~5% of `total_steps` (`schedule.warmup_steps`, `config.py:319-323`), then cosine to ~`1e-6`.
- Outer: stick to defaults unless you understand the gradient agg loop.

**Always run a gradient-norm sanity check.** `grad_norm > 100` is a NaN-risk warning sign. Cap with `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.

### G. Loss-spike batch filter

A single bad batch can deposit a step that wrecks the model's val_loss for the rest of the cycle. Filter:

```python
running_mean = ... # EMA of recent batch losses
if batch_loss > running_mean * 3 or not math.isfinite(batch_loss):
    optimizer.zero_grad()
    continue  # skip this batch
```

### H. Data quality filter

The eval pool is C4+Nemotron. Match training to that distribution:

- C4: filter out boilerplate (URL-heavy, navigation pages, ad text).
- Nemotron-CC-Math-v1: filter to `4plus` quality bucket.

Reduce mismatch by tracking per-domain val_loss locally — your training loss may be improving on the wrong sub-distribution.

---

## Recipe order

What to turn on first, in what sequence, with rollback criteria.

1. **Day 0: smoke-test the baseline pipeline.** Run `bench/local_scorer.py` on the unmodified miner; record `(baseline_val_loss, top_uid_val_loss_estimate)`. If your val_loss is `+inf` or NaN, fix that before touching anything else (see "Don't-do list" below).
2. **Day 1: Flash Attention 2.** Smoke-test as above. Rollback if `val_loss diff > 0.01` from the prior baseline (suggests a different numerical path; investigate).
3. **Day 2: sequence packing.** Rollback if perplexity-on-clean-eval degrades (suggests cross-sample bleed).
4. **Day 3: EMA checkpoint.** Always-on; almost never hurts.
5. **Day 4: LR tuning.** Run a 2x2 sweep (lr in {1e-5, 3e-5} x warmup in {0, 5%}) via `bench/sweep_runner.py`; keep the best by 8-seed mean.
6. **Day 5: data quality filter.** A/B against the unfiltered run.
7. **Day 6: best-checkpoint selection.** Wrap the submit step with `bench/checkpoint_compare.py`.
8. **Day 7+: monitor live.** Dashboard delta, per-validator variance, per-round rank.

---

## Don't-do list

| Don't | Why | Mechanism reference |
| --- | --- | --- |
| Remove all stochasticity from your forward pass | Two miners with deterministic+identical val_loss both get 0 from the tie penalty | `connito/validator/evaluator.py:232-261` |
| Train on the eval data verbatim | Validator seed (and thus shuffle+skip offset) changes every round; memorizing one realization gains nothing | `connito/shared/dataloader.py:240-275` (shuffle+skip), `connito/shared/cycle.py:523-588` (seed) |
| Let `grad_norm > 100` | Triggers NaN/Inf -> validator drops the batch -> if every batch is bad, val_loss = +inf -> score 0 | `connito/shared/evaluate.py:99-104, 133-144` |
| Skip the chain commit | No commit at freeze -> `freeze_zero_uids` -> hard 0 | `connito/validator/evaluator.py:286-294` |
| Submit weights with NaN/Inf tensors | `_verify_expert_group` folds the NaN/Inf scan in; rejection reason `expert_group_or_nan` | `connito/validator/evaluator.py:558-562` |
| Submit a different expert group's experts | Same rejection path | same |
| Submit only on round boundary | Bg-download timeout (`per_miner_download_timeout_sec = 180`, `config.py:853`) and per-miner eval timeout (300s, `config.py:854`) preserve EMA but waste a round. Submit early in the cycle. | `connito/validator/evaluator.py:823-832` |
| Use a >50% LR jump mid-training without warmup | Loss spikes; spike-filter or step skip needed | "Knob G" above |
| Crank `router_aux_loss_coef` to 0 expecting better val_loss | Validator subtracts aux_loss before scoring; benefit is in training stability only | `connito/shared/evaluate.py:107-112, 146` |

---

## Reverse-engineering top miners

Top-3 miners (dashboard 2026-05-22):

| UID | HF repo | Notes |
| --- | --- | --- |
| 79 | `Noburo/co79@e73fe4d` | val_loss 1.4303 (delta 3.3516, chain weight 0.288) |
| 99 | `imaman520/baobae5` | Top-2 territory |
| 8 | `putty77/p77-3` | Top-3 territory |

Inspection workflow:

```bash
# Pin to the exact commit to match what the validator evaluated.
huggingface-cli download Noburo/co79 --revision e73fe4d --local-dir ./co79
huggingface-cli download imaman520/baobae5 --local-dir ./baobae5
huggingface-cli download putty77/p77-3   --local-dir ./p77-3

# Inspect:
ls -la ./co79/
cat ./co79/config.json
cat ./co79/tokenizer_config.json

# Check commit history for training hints
huggingface-cli download Noburo/co79 --revision e73fe4d --local-dir-use-symlinks False \
    --include README.md training_config.yaml *.json
```

Look for:

- `config.json` — `attn_implementation`, `torch_dtype`, MoE hyperparams (`num_experts`, `num_experts_per_tok`, `partial_topk`, `full_topk`).
- `tokenizer_config.json` — model_max_length should align with eval seq_len=4096; padding_side; special tokens.
- `tokenizer.json` / `tokenizer.model` — same tokenizer as base.
- Commit messages on the HF Hub web view often hint at training schedule / LR / epoch count.
- Expert weight statistics — `python -c "import torch, safetensors.torch as sft; sd = sft.load_file('co79/model.safetensors'); print({k: (v.mean().item(), v.std().item()) for k, v in sd.items() if 'expert' in k})"` to compare to baseline expert weights and see where the miner concentrated training.

You cannot replicate the miner's checkpoint directly (chain commits enforce hotkey signature), but you CAN learn from their training choices.

---

## Live monitoring

Dashboard: <https://dashboard-dev.connito.ai/>

Key columns to watch:

| Column | Meaning | Action |
| --- | --- | --- |
| `baseline_loss` | Current cycle's validator baseline val_loss | Track over time; rising baseline = whole subnet improving = your edge eroding |
| `your_val_loss` | Your val_loss in the most recent eval round | Lower is better; target: at least 1.0 below baseline |
| `Δ = baseline - val_loss` | Your raw delta (pre-exponent) | Need this in top-3 of the round to earn 2.25/1.5/1.0 |
| `rank` | This round's rank from the per-validator delta ordering | Top-3 -> aggregator entry; else 0 |
| `chain_weight` | Average over the 8-round window, share of total emission | What actually pays TAO |
| `per-validator weights` | The same UID's chain weight as reported by each of N validators | High variance = per-validator drift; tighten with multi-seed robustness |
| `eval_failures_total` | Counter labeled by reason: `hash`/`signature`/`expert_group_or_nan`/`deadline`/`oom`/`statedict_parse_failed`/... | Any non-zero here for your UID needs immediate root-cause |
| `validator_miner_eval_status` | Per-UID gauge: `None` (clean), `no_chain_commit`, `non_finite_loss`, etc. | Use to self-diagnose without HTTP scraping the validators |

The `validator_baseline_loss` Prometheus gauge and `validator_miner_val_loss` per-UID gauge are set at `connito/validator/evaluator.py:792-795, 974-977`. Both are best-effort and never block scoring.

---

## Tooling reference

These benchmarking scripts under `bench/` ship as Units 1-5 of the same fleet. They may not exist yet in your local worktree at the time you read this; they will land via parallel PRs.

| Script | Purpose | Owner unit |
| --- | --- | --- |
| `bench/local_scorer.py` | Replicates the validator scoring path (`evaluate_model` + same dataloader settings) so a miner can produce a local `(val_loss, score, rank-estimate)` without burning a chain round | Unit 1 |
| `bench/eval_pool_explorer.py` | Dumps the eval samples that a given `combined_seed` would draw, so a miner can sanity-check distribution overlap with training data | Unit 2 |
| `bench/sweep_runner.py` | LR / warmup / aux-coef sweep harness with 8-seed mean as the objective | Unit 3 |
| `bench/checkpoint_compare.py` | Side-by-side val_loss across N candidate checkpoints (last-K + EMA + checkpoint-snapshots) | Unit 4 |
| `bench/loss_curve_analyzer.py` | Parses miner training logs and surfaces spike batches, divergence points, candidate-rollback steps | Unit 5 |

Usage pattern: train -> `bench/checkpoint_compare.py` -> pick best -> upload to HF -> chain commit -> watch dashboard.

---

## Troubleshooting

| Symptom | Diagnosis | Fix |
| --- | --- | --- |
| `eval_status = non_finite_loss` for your UID | Every eval batch produced NaN/Inf in bf16 autocast | Lower LR, add grad-norm clip, verify weights are not subnormal. See `connito/shared/evaluate.py:99-104`. |
| `eval_status = expert_group_or_nan` | NaN/Inf tensor in submitted state_dict OR a routed-expert key outside your assigned group | `python -c "import safetensors.torch as sft; sd=sft.load_file('model.safetensors'); print({k: (torch.isnan(v).any().item(), torch.isinf(v).any().item()) for k,v in sd.items()})"` then re-train from last good ckpt with clipping. |
| `eval_status = hash` | On-disk shard doesn't match the chain commit's `model_hash` | Verify upload completed; re-hash + re-commit. See `_verify_hash` in `ChainCheckpoint.validate`. |
| `eval_status = signature` | Chain hotkey did not sign `model_hash` correctly | Wallet / signing tooling misconfiguration; verify with `btcli wallet inspect`. |
| `eval_status = no_chain_commit` | Did not commit before freeze | Commit earlier in the cycle. |
| `eval_status = deadline` | Eval ran past `per_miner_eval_timeout_sec = 300`s | Your model is slower than peers; reduce model size, enable Flash Attention 2, or accept that some validators will time out and EMA-preserve you. |
| Score = 0 with `delta > 0` (Δ visible on dashboard) | TIED val_loss with another miner | Add stochasticity to your forward pass (e.g. enable dropout-style noise during the eval scoring). See `evaluator.py:232-261`. |
| Score = 0, never makes top-3 in the round | val_loss not low enough | Lower val_loss. See "Knobs by impact". |
| Chain weight inconsistent across validators | Per-validator divergence (see SCORING_INTERNALS section 9) | Optimize for multi-seed mean, not single-seed peak. |
| Sudden drop in chain weight after a good run | EMA washout — a 0.0 entry from a missed round, or a tied/failed round, replaced an older 2.25 in the 8-point window | Wait it out (window is FIFO); always submit on time so you stay in `scored_uids`; never tie. |
