# Scoring internals

How the Connito SN102 validator turns a miner's submitted checkpoint into a chain weight. This is the full, exhaustive reverse-engineering of the validator scoring path with line references into `connito/validator/evaluator.py`, `connito/shared/evaluate.py`, `connito/shared/dataloader.py`, `connito/validator/aggregator.py`, and `connito/shared/config.py`.

Operators tuning a miner should pair this with `docs/OPTIMIZATION_PLAYBOOK.md`.

---

## 1. The scoring formula

The per-round, per-miner score is computed inside `evaluate_one_miner_sync` at `connito/validator/evaluator.py:780-787`:

```python
val_loss = float(metrics.get("val_loss", 100))
delta    = max(0.0, baseline_loss - val_loss)
score    = delta ** 1.2
```

Three things to notice:

| Symbol | Meaning | Source |
| --- | --- | --- |
| `baseline_loss` | `val_loss` produced by running the validator's **input model** (its own pre-merge global model) over the same eval batches | `connito/validator/evaluator.py:954-964` (`evaluate_foreground_round`) |
| `val_loss` | The miner's val_loss on the same eval batches (formula in section 3) | `connito/shared/evaluate.py:146` |
| `score` | The per-round, per-miner raw signal that feeds `round.scores` and ultimately `finalize_round_scores` | `connito/validator/evaluator.py:787` |

`max(0, ...)` clamps any miner whose val_loss is worse than baseline to delta = 0. The `** 1.2` exponent concavifies further: a 2x larger delta becomes a 2.30x larger raw score (`2 ** 1.2 ≈ 2.297`), and a 3x larger delta becomes a 3.74x larger score. This penalizes ties and rewards a wider margin.

Importantly, `evaluate_one_miner_sync` does NOT write to the `MinerScoreAggregator`. It only returns a `MinerEvalJob` whose `.score` field is the raw delta**1.2. The aggregator update is centralized in `finalize_round_scores` (see section 6) so all round-level entries can be replaced atomically with rank-based scores at end of round.

## 2. Rank-based mapping

After every miner in the round has been evaluated, `finalize_round_scores` (`connito/validator/evaluator.py:163-366`) walks `round.scores`, drops the previous aggregator points tagged with this `round_id`, and re-adds rank-based entries:

```python
# connito/validator/evaluator.py:160
_RANK_TO_SCORE: tuple[float, ...] = (2.25, 1.5, 1.0)
```

| Rank in round (by raw `delta**1.2` desc) | Aggregator score written |
| --- | --- |
| 1 | 2.25 |
| 2 | 1.5 |
| 3 | 1.0 |
| 4..N | 0.0 |

The geometric progression ratio is 1.5 (`2.25 / 1.5 = 1.5`, `1.5 / 1.0 = 1.5`). See the docstring at `connito/validator/evaluator.py:149-160` for the rationale: equal ratios across tiers keep second-place close to first, while the top-1/top-3 cap of 2.25 caps the gap.

The raw delta**1.2 value is used **only for ranking** within the round; the aggregator never sees it directly. This means:

- A miner who beats baseline by 0.001 but ranks first gets 2.25.
- A miner who beats baseline by 1.5 but ranks fourth gets 0.0.

Improving val_loss is therefore worthless unless it lifts you into the top 3.

## 3. The val_loss formula

`connito/shared/evaluate.py:28-154` computes `val_loss` over up to `max_eval_batches=50` batches (`connito/validator/evaluator.py:580`):

```python
# connito/shared/evaluate.py:60-72
loss_sum: float = 0.0
aux_loss_sum: float = 0.0
scored_batches: int = 0
nan_batches: int = 0
```

For each batch:

```python
# connito/shared/evaluate.py:96-112
with torch.amp.autocast(autocast_device, dtype=eval_dtype):
    outputs = model(**device_batch)
    if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
        nan_batches += 1
    else:
        loss_sum    += float(outputs.loss.item())
        aux_loss_sum += float(outputs.aux_loss.item()) if hasattr(outputs, "aux_loss") and outputs.aux_loss is not None else 0.0
        scored_batches += 1
```

At end of pass (`connito/shared/evaluate.py:133-153`):

```python
if scored_batches == 0:
    return {"val_loss": float("inf"), ...}   # every batch was NaN/Inf -> score 0

val_loss     = (loss_sum - aux_loss_sum) / scored_batches
val_aux_loss = aux_loss_sum / scored_batches
```

Key implications:

| Behavior | Mechanism | Line |
| --- | --- | --- |
| Aux loss subtracted | `val_loss = (loss_sum - aux_loss_sum) / scored_batches` | `evaluate.py:146` |
| NaN/Inf batches dropped from BOTH numerator and divisor | `if isnan or isinf: nan_batches += 1` (no contribution to `loss_sum`, `aux_loss_sum`, or `scored_batches`) | `evaluate.py:99-104` |
| All-NaN/Inf submission -> `val_loss = +inf` -> delta clamps to 0 | `if scored_batches == 0: return {"val_loss": float("inf"), ...}` | `evaluate.py:133-144` |
| Eval runs under bf16 autocast (or fp16 fallback) | `eval_dtype = torch.bfloat16 if cuda.is_bf16_supported() else torch.float16` | `evaluate.py:95-96` |

The NaN-batch dropping was introduced to neuter a specific exploit: previously NaN batches were skipped from `loss_sum` but still counted in the divisor, so a checkpoint that produced NaN on fraction `p` of batches and honest loss elsewhere reported `(1 - p) * honest_loss` — gaming the val_loss downward. Both numerator and denominator now drop NaN batches symmetrically (see the inline comment at `evaluate.py:64-72`).

The **aux_loss subtraction** is the load-balance auxiliary used by MoE routers. Validators back it out so a miner cannot inflate val_loss by setting `router_aux_loss_coef` to 0 during eval — the validator's eval sees the model's actual `outputs.aux_loss` and subtracts it whatever the miner did. This means a miner can crank `router_aux_loss_coef` up during training (for routing stability) without paying for it in the leaderboard.

## 4. The eval data pool

The same `combined_seed` is used for both baseline and every miner in the round, so deltas are comparable.

### Seed derivation

`get_combined_validator_seed` at `connito/shared/cycle.py:523-588` builds the per-round seed:

```python
combined_seed_str = committed_part + block_hash
return hashlib.sha256(combined_seed_str.encode()).hexdigest()
```

where:

- `committed_part`: concatenation of per-validator `miner_seed` chain commits, sorted by hotkey (`cycle.py:583-585`). Read by miners during the commit window — exploitable on its own.
- `block_hash`: block hash of the LAST block of the most recent completed `MinerCommit2` phase (`cycle.py:569`). Validators do not control block production, so this component is robust against validator-miner collusion.

The SHA-256 mix is the seed every miner sees this round. It changes each round.

### Dataset sources

Default mix at `connito/shared/dataloader.py:158-175`:

```python
[
    {"path": "allenai/c4",                 "name": "en",    "weight": 0.5, "text_column": "text"},
    {"path": "nvidia/Nemotron-CC-Math-v1", "name": "4plus", "weight": 0.5, "text_column": "text"},
]
```

50/50 interleave is fed into `interleave_datasets(...)`. C4 supplies general English, Nemotron-CC-Math-v1 supplies mathematical text — a miner that only trains on one side will lose ground on the other.

### Shuffle + skip

Two stages, both seeded off `int_seed = int(combined_seed[:8], 16)` (`dataloader.py:217-218`):

```python
# dataloader.py:240-275
shuffle_buffer = getattr(config.task.exp.data, "eval_source_shuffle_buffer", 0)   # default 50_000
skip_max       = getattr(config.task.exp.data, "eval_source_skip_max",       0)   # default 50_000

dataset_splits = [ds.shuffle(seed=int_seed, buffer_size=shuffle_buffer) for ds in dataset_splits]

skip_rng = random.Random(int_seed)
offsets = [skip_rng.randrange(0, skip_max) for _ in dataset_splits]
dataset_splits = [ds.skip(offset) for ds, offset in zip(dataset_splits, offsets)]
```

This pair is what makes memorizing the eval pool infeasible:

- `.shuffle(seed, buffer_size=B)` permutes shard order AND buffer-shuffles within a B-wide window.
- `.skip(N)` advances the read into the BODY of the lead shard so the reachable pool grows along (shard permuted to lead) x (offset into that shard).

Inline comments at `dataloader.py:226-275` walk through the prior exploit: without the skip, every eval round reached only the head of each source's stream (~2,000 distinct samples ever drawn from the default mix), small enough for a miner to memorize.

### Fractional subsampling

Applied AFTER shuffle+skip, BEFORE sharding:

```python
# dataloader.py:287-299
if seed is not None and fraction is not None and fraction < 1.0:
    max_int   = 2**256 - 1
    threshold = int(max_int * fraction)
    filter_fn = partial(_fractional_index_filter, seed=seed, threshold=threshold)
    split     = split.filter(filter_fn, with_indices=True)
```

`config.task.exp.data.vali_fraction` defaults to `0.1` (`config.py:227`), so ~10% of the streamed positions pass the filter.

### Batch count and shape

```python
# connito/validator/evaluator.py:580
EVAL_MAX_BATCHES = 50
```

50 batches at `per_device_train_batch_size=1` (default in `DataCfg`, `config.py:223`) with `sequence_length=4096` (`config.py:222`) -> ~50 * 4096 = ~200K eval tokens per round per validator.

## 5. Score=0 traps

The full list of ways a miner gets `score = 0` (or worse, has the EMA still pulled down because their previous good rounds are washed out by a zero):

| Trap | Mechanism | File:line |
| --- | --- | --- |
| Hash mismatch | `_verify_hash` fails inside `ChainCheckpoint.validate(...)` | `evaluator.py:556-557` -> `validate_miner_submission` returns `"hash"` |
| Bad signature | `_verify_signature` fails -> reason `"signature"` | `evaluator.py:554-555` |
| Expert group violation OR NaN/Inf tensor in state dict | `_verify_expert_group` (folds NaN/Inf scan in with routing key check) -> reason `"expert_group_or_nan"` | `evaluator.py:558-562` |
| No chain commit at freeze | `freeze_zero_uids` (validation set populated at `Round.freeze`) | `evaluator.py:286-294` |
| Unknown validate() exception | `validate_miner_submission` returns `"unknown"` | `evaluator.py:542-547` |
| Every eval batch produced NaN/Inf -> `val_loss = +inf` -> delta clamps to 0 | `evaluate.py:133-144` |
| Delta == 0 (val_loss >= baseline_loss) | `max(0.0, baseline_loss - val_loss)` | `evaluator.py:781` + ranking filter `evaluator.py:221-224` (`score > 0.0`) |
| **TIED val_loss exactly equal to another miner's** | `tied_uids = {uid for uid, s in positive if score_counts[s] > 1}` -> both sides score 0 regardless of rank | `evaluator.py:232-235`, `evaluator.py:252-261` |
| Beyond rank 3 | `_RANK_TO_SCORE` has only 3 slots | `evaluator.py:160`, `evaluator.py:242` |

The tie penalty is enforced with float64 exact equality. Inline comment at `evaluator.py:226-231`: "exact equality between two miners is overwhelmingly evidence of a duplicated submission, not legitimate parallel improvement." The full tie path:

```python
# evaluator.py:232-261
score_counts: dict[float, int] = {}
for _, s in positive:
    score_counts[s] = score_counts.get(s, 0) + 1
tied_uids       = {uid for uid, s in positive if score_counts[s] > 1}
unique_positive = [(uid, s) for uid, s in positive if score_counts[s] == 1]
# ...
for uid in tied_uids:
    score_aggregator.add_score(uid=uid, hotkey=hotkey, score=0.0, round_id=round_obj.round_id)
```

## 6. Operational failures preserve the EMA

A class of failures **does not** dock the miner — their prior EMA is preserved:

| Failure | What happens | File:line |
| --- | --- | --- |
| Eval deadline exceeded (`EvalDeadlineExceeded`) | `_record_eval_failure(uid, "deadline")` and `evaluate_one_miner_sync` returns `None`; uid lands in `round.failed_uids` but NOT `round.validation_failed_uids` | `evaluator.py:823-832` |
| OOM (`torch.cuda.OutOfMemoryError`) | `_record_eval_failure(uid, "oom")`, returns `None`, same treatment as above | `evaluator.py:833-839` |
| Statedict parse failed (`ValueError | RuntimeError | EOFError`) | `_record_eval_failure(uid, "statedict_parse_failed")`, returns `None` | `evaluator.py:840-847` |
| Unknown exception | `_record_eval_failure(uid, "unknown")`, returns `None` | `evaluator.py:848-851` |
| Download timeout | bg-download never lands the shard; uid is absent from `scored_uids` and from `validation_failed_uids` | `connito/validator/background_download_worker.py` (the cleanup path at `evaluator.py:65-146` deletes the file but does not score it) |

`finalize_round_scores` writes nothing for these UIDs (`evaluator.py:189-192`). Their EMA is preserved by `drop_round` only dropping THIS round's points and not touching prior rounds (`aggregator.py:246-257`). The validator's lack of compute / bandwidth must not penalize the miner — this is explicit in the docstring at `evaluator.py:189-192`.

## 7. Aggregation: 8-round window and rank-EMA

### MinerSeries window

```python
# connito/validator/aggregator.py:37-38, 122-123
@dataclass
class MinerSeries:
    max_points: int = 8  # default mirrors config.evaluation.score_window
    # ...
    if len(pts) > self.max_points:
        pts = pts[-self.max_points:]
```

`max_points` mirrors `config.evaluation.score_window = 8` (`config.py:850`). Every aggregation (`sum`, `avg`, `ema`) caps to the last 8 points per miner.

### EMA

`MinerScoreAggregator.ema(...)` at `connito/validator/aggregator.py:319-335`:

```python
def ema(self, uid, alpha=0.2, start=None, end=None):
    pts = self.get_history(uid, start, end)
    if len(pts) > self._max_points:
        pts = pts[-self._max_points:]
    if not pts:
        return 0.0
    ema_val = pts[0][1]
    for _, v in pts[1:]:
        ema_val = alpha * v + (1 - alpha) * ema_val
    return float(ema_val)
```

Standard `alpha=0.2` EMA over the last 8 points. The aggregator value used to drive chain weight emission is `avg` by default (`uid_score_pairs(how="avg")` in `build_submission_uid_weights`, `evaluator.py:424`), but every metric (`latest`, `sum`, `avg`, `ema`) is exposed.

### Weight emission: group 1 vs group 2

`build_submission_uid_weights` at `connito/validator/evaluator.py:387-478` builds the chain submission:

```python
# connito/validator/evaluator.py:445-449
g1 = _rg.select_top_n_by_local_score(
    ab_qualified, avg_scores, n=eval_cfg.weight_group_1_size,  # default 3
)
# evaluator.py:460-464
g2 = _rg.select_top_n_by_local_score(
    g2_pool, avg_scores, n=eval_cfg.weight_group_2_size,       # default 5
)
# evaluator.py:465-471
uid_weights = _rg.compute_uid_weights(
    weight_group_1=g1,
    weight_group_2=g2,
    local_scores=avg_scores,
    group_1_share=eval_cfg.weight_group_1_share,               # default 0.98
    group_2_share=eval_cfg.weight_group_2_share,               # default 0.02
)
```

`compute_uid_weights` at `round_groups.py:648-683`:

- Each group's `share` is split among its members **in proportion to their `avg` aggregator score** (or equally if every member scored 0).
- Top-3 (`g1`) gets 98% of the chain weight, split in proportion to avg score.
- Next-5 (`g2`) split 2%.
- Everyone else: 0.

G1 has a recency gate: a UID must have `record_count >= 2` AND be present in BOTH the current round AND the previous round (`evaluator.py:440-444`). If no UID clears this gate, the share redirects to UID 0 (the subnet owner) so the validator stays at full emission rather than burning it.

## 8. Full constants table

Every parameter that affects the scoring pipeline. Pull from `connito/shared/config.py` and `connito/validator/evaluator.py`:

| Constant | Value | Source |
| --- | --- | --- |
| `EVAL_MAX_BATCHES` | 50 | `evaluator.py:580` |
| `MAX_CONCURRENT_DOWNLOADS` | 4 | `evaluator.py:577` |
| `EVAL_WORKERS` | 1 | `evaluator.py:578` |
| `DOWNLOAD_TIMEOUT_SEC` | 60 | `evaluator.py:579` |
| `_RANK_TO_SCORE` | `(2.25, 1.5, 1.0)` | `evaluator.py:160` |
| `score_exponent` | 1.2 (inline) | `evaluator.py:787` |
| `vali_fraction` | 0.1 | `config.py:227` |
| `sequence_length` | 4096 | `config.py:222` |
| `per_device_train_batch_size` | 1 | `config.py:223` |
| `world_size` | 10 | `config.py:224` |
| `eval_source_shuffle_buffer` | 50000 | `config.py:246` |
| `eval_source_skip_max` | 50000 | `config.py:270` |
| `top_k_miners_to_merge` | 1 | `config.py:848` |
| `top_k_miners_to_reward` | 3 | `config.py:849` |
| `score_window` | 8 | `config.py:850` |
| `foreground_top_n` | 5 | `config.py:851` |
| `per_miner_download_timeout_sec` | 180 | `config.py:853` |
| `per_miner_eval_timeout_sec` | 300 | `config.py:854` |
| `enable_round_group_construction` | True | `config.py:862` |
| `cohort_window_cycles` | 8 | `config.py:864` |
| `weight_group_1_size` | 3 | `config.py:865` |
| `weight_group_1_share` | 0.98 | `config.py:866` |
| `weight_group_2_size` | 5 | `config.py:867` |
| `weight_group_2_share` | 0.02 | `config.py:868` |
| `validation_group_a_size` | 3 | `config.py:869` |
| `validation_group_ab_total` | 13 | `config.py:870` |
| `validation_group_c_size` | 17 | `config.py:871` |
| `group_a_min_consensus` | 1 | `config.py:872` |
| `group_a_min_weight_per_validator` | 0.03 | `config.py:873` |
| Default datasets | `allenai/c4@en` (0.5) + `nvidia/Nemotron-CC-Math-v1@4plus` (0.5) | `dataloader.py:158-175` |
| Eval autocast dtype | bf16 (cuda+bf16 supported) else fp16 | `evaluate.py:95-96` |
| EMA alpha | 0.2 (default in `MinerScoreAggregator.ema`) | `aggregator.py:319-335` |
| `HF_HUB_DOWNLOAD_TIMEOUT` | 120 (env default) | `dataloader.py:37` |

## 9. Per-validator divergence

The 2026-05-22 dashboard snapshot shows large per-validator weight variance for the same UID:

- Top miner UID 79: baseline 4.7820, val_loss 1.4303, delta 3.3516, chain weight 0.288 averaged across validators.
- HF checkpoint: `Noburo/co79@e73fe4d`.

Even though every validator uses the same `combined_seed`, validators can drift on:

- **Independently-tracked baselines** — each validator runs its own model through the eval pool, so the baseline differs.
- **Network jitter / HF Hub latency** — a miner's checkpoint may not land in time for some validators, who then evaluate the *previous* checkpoint or skip the miner entirely.
- **Per-validator config drift** — operators may differ on `eval_source_shuffle_buffer`, `eval_source_skip_max`, `vali_fraction` defaults across versions; the inline comments at `config.py:240-271` flag this as a known "coordinated-rollout discipline" issue.
- **Cohort assignment** — `validation_group_a/b/c` differ per validator and only A∪B is in the G1 pool, so a miner not in A∪B for a given validator gets at most G2's 2% from that validator.
- **EMA window asymmetry** — restart wipes some validators' aggregator history (subject to `score_path` recovery at `evaluator.py:312-314`), so the 8-point window can mean different things per validator immediately after a restart.

**Implication for miners**: optimizing for a single (seed, shard) realization is overfitting. Multi-seed robustness — sampling many `(combined_seed, skip_offset)` realizations during local benchmarking — is the only way to reduce per-validator variance.

## 10. End-to-end flow

A single validator's round, end to end:

1. `Round.freeze` partitions the roster, sets the per-validator `combined_seed`, builds the foreground UID set (`round.py`, called from `run.py` at submission phase).
2. `evaluate_foreground_round` runs the baseline once against the validator's input model (`evaluator.py:954-964`), then iterates foreground miners as their checkpoints land.
3. For each miner: `validate_miner_submission` -> `validate_one_miner_sync` -> `_evaluate_on_fresh_loader_sync` -> `evaluate_model` -> `val_loss` -> `delta**1.2` -> stored in `round.scores` via `mark_scored`.
4. After foreground (or after bg-eval clears the rest of the roster), `finalize_round_scores` (`evaluator.py:163-366`):
   - Drops every prior aggregator entry for this `round_id`.
   - Re-adds rank-based 2.25/1.5/1.0 for top-3 unique-positive deltas.
   - Writes 0.0 for ties, beyond-top-3, validation failures, and freeze-zero UIDs.
   - Writes nothing for operational failures.
5. `build_submission_uid_weights` computes the `{uid: weight}` chain payload using `avg` over the 8-point window.
6. The chain submitter takes the payload, normalizes to sum=1, and emits `set_weights` on chain.
