# How scoring works

Every block of this document maps a miner-visible field to the code that
produces it. The chain weights you actually earn TAO emission on are a long
way downstream of the raw `val_loss` printed in the validator logs, so trace
through each step.

## The per-round signal: `val_loss` and `delta_loss`

When a validator picks up a miner submission, it runs:

```python
# connito/validator/evaluator.py:evaluate_one_miner_sync
val_loss = float(metrics.get("val_loss", 100))   # measured on the held-out slice
delta    = max(0.0, baseline_loss - val_loss)    # baseline is THIS validator's pre-merge model loss
score    = delta ** 1.2                          # per-round raw signal
```

- **`baseline_loss`** is computed once per round against the round's *input*
  global model (before any merging this cycle). Every miner in the same round
  is compared against this same baseline. Code:
  `connito/validator/evaluator.py:evaluate_foreground_round` (`baseline_loss`
  assignment, ~line 964).
- **`val_loss`** is the miner's loss on the same eval batches. If the eval
  blew up to NaN/Inf, `evaluate_model` returns `+inf` so `delta` clamps to
  0. Code: `connito/shared/evaluate.py:evaluate_model`.
- **`delta_loss`** is the gap. The `** 1.2` exponent makes top miners
  separate more sharply than a linear delta would, but is monotonic — a
  better `val_loss` always produces a higher `score`.

The raw `score` is stored on the `MinerEvalJob` but is **not** what ends up on
chain — see "Rank-based finalization" below.

## When `val_loss` and `delta_loss` are null

The dashboard surfaces nulls in the following code paths. None of these are
bugs; they each map to a specific validator state.

1. **No chain commit** — the miner has no `MinerChainCommit` from
   MinerCommit2 of the previous cycle (or it lacks `hf_repo_id`/`hf_revision`).
   The miner never enters `uid_to_chain_checkpoint` and never gets evaluated.
   `eval_status_label` is `no_chain_commit`. Code:
   `connito/validator/round.py:Round.freeze` (step 3, `freeze_zero_uids`),
   `connito/validator/evaluator.py:validate_miner_submission` (returns
   `"no_chain_commit"` if `chain_checkpoint is None`).
2. **Signature / hash / expert-group / NaN-Inf failed at the checkpoint
   verifier** — the on-disk shard's hash doesn't match the chain commit, the
   miner's hotkey didn't sign the model hash, the shard contains experts from
   a group the miner is not assigned to, or the shard contains a NaN/Inf
   tensor. `evaluate_one_miner` is **not called** in any of these cases, so
   no `val_loss` is produced. Code:
   `connito/shared/checkpoints.py:ChainCheckpoint.validate`,
   `connito/validator/evaluator.py:validate_miner_submission`.
3. **Non-finite `val_loss`** — the eval ran but returned `+inf` (e.g.,
   every batch produced a NaN loss). `val_loss` is logged as `inf` and the
   miner's `eval_status_label` is `non_finite_loss`. The aggregator still
   records a `score = 0` for the round. Code:
   `connito/validator/evaluator.py` line 802-805.
4. **Operational failure** — download timeout, GPU OOM, eval timeout, or
   unexpected exception. The miner is marked `failed_uids` (not
   `validation_failed_uids`), and `finalize_round_scores` writes **nothing**
   for the round — the prior EMA is preserved. The dashboard sees no new
   `val_loss` sample for that round. Code:
   `connito/validator/evaluator.py:evaluate_one_miner_sync` (the
   `TimeoutError` / `OutOfMemoryError` / `RuntimeError` except blocks).
5. **Round-tail timeout** — the foreground evaluation hits the phase
   boundary before reaching every miner. Unreached miners stay in
   `failed_uids` (operationally), again with no new sample.

> If you are looking at a dashboard `val_loss` and it is missing, the
> distinction between (1)/(2) and (4)/(5) matters: (1) and (2) are
> miner-fixable; (4) and (5) are validator-side and resolve themselves over
> the next few cycles.

## Rank-based finalization (what actually enters the aggregator)

At the end of each round, `finalize_round_scores` discards the raw `delta ** 1.2`
values and replaces them with a fixed rank-based mapping. Only the top-3
miners get non-zero score points; everyone else gets 0.

```python
# connito/validator/evaluator.py:_RANK_TO_SCORE
(2.25, 1.5, 1.0)
```

- Top-1 (lowest `val_loss` with `delta > 0`) → **score = 2.25**
- Top-2 → **score = 1.5**
- Top-3 → **score = 1.0**
- Everyone else who was evaluated → **score = 0.0**
- Two miners with exactly the same `val_loss` → **both get score = 0.0** (a
  duplicated submission is the much more likely cause than two miners
  legitimately producing bit-identical losses).
- Miners with `delta == 0` (didn't beat baseline) → **score = 0.0** regardless
  of rank.

Tied scores between unique miners do not break this rule — only literal
exact-equal `val_loss` values trigger the duplication penalty.

Code: `connito/validator/evaluator.py:finalize_round_scores`.

## The rolling aggregator: `score_latest`, `score_avg`, `score_samples`

Per-round rank scores are stored in a `MinerScoreAggregator` keyed by UID. The
aggregator keeps a rolling window of the last **8 samples** per miner
(`config.evaluation.score_window`).

Dashboard fields:

- **`score_latest`** — the most recent rank-score the aggregator holds
  (Prometheus gauge `validator_miner_score_latest`).
- **`score_avg`** — arithmetic mean of the rolling window
  (`validator_miner_score_avg`). This is the metric that drives chain weight
  emission.
- **`score_samples`** — number of points in the rolling window, 0 to 8
  (`validator_miner_score_samples`).

When a UID's hotkey changes (deregister + re-register), the aggregator
**resets** that UID's entire history. The miner must accumulate new samples
from scratch.

Code: `connito/validator/aggregator.py:MinerScoreAggregator.add_score`
(line 199, hotkey-change reset is at line 234-237),
`connito/shared/telemetry.py:set_miner_score_snapshot`.

History older than `8 × cycle_length` blocks is pruned at the end of every
cycle (`connito/validator/run.py:prune_before_round`).

## From `score_avg` to chain weights: `chain_weight_stake_weighted`

After Validate, the validator builds a `{uid: weight}` payload that goes onto
chain via `set_weights`. The payload is split into two groups:

```python
# connito/shared/config.py:EvalCfg
weight_group_1_share = 0.98   # 98% of this validator's emission
weight_group_2_share = 0.02   # 2% of this validator's emission
weight_group_1_size  = 3      # G1 holds at most 3 miners
weight_group_2_size  = 5      # G2 holds at most 5 miners
```

**Weight Group 1 (98 %)** — top-3 of validation groups A ∪ B by `score_avg`,
subject to:

1. `record_count >= 2` (the miner has at least 2 samples in the aggregator).
2. The miner has a score recorded for **both** of the last 2 rounds
   (`round_id == current_round_id` and `round_id == current_round_id - cycle_length`).
   This is the "recency gate" — a miner that missed either round drops off G1
   for this cycle.

If no UID clears the recency gate, the entire 98 % share is **redirected to
UID 0** (subnet owner). This is the **no-winner-in-group fallback**, not a
"burn":

> The chain emission is not destroyed. It is paid to the subnet owner because
> the validator could not confidently pick a top-3 from miners that have been
> scoring consistently in the last 2 rounds. From the chain's perspective UID
> 0 receiving 98 % looks identical to a burn, but the operational meaning is
> "no qualified winner this round."

Code:
`connito/validator/evaluator.py:build_submission_uid_weights` (g1 selection
and redirect at lines 440-453),
`connito/validator/run.py:1064-1070` (the "g1 empty — redirecting" log line),
`connito/shared/chain.py:reserve_subnet_owner_share` (a separate 5 % owner
share applied to fallback-weight paths, distinct from G1 redirection).

**Weight Group 2 (2 %)** — top-5 of A ∪ B ∪ C \ G1 by `score_avg`, subject
to `record_count >= 1`. No recency gate.

Within each group, the share is split **proportional to `score_avg`** (not
equal). If every member of a group scored 0, the share is split equally
across members.

Code: `connito/validator/round_groups.py:compute_uid_weights`,
`connito/validator/round_groups.py:select_top_n_by_local_score`.

**`chain_weight_stake_weighted`** as surfaced on the dashboard is the
post-Yuma-consensus value: for each miner, sum each validator's submitted
weight times that validator's stake, then normalize. The dashboard derives
this from `metagraph.weights` and stake; it is not directly recorded by the
validator. The validator emits per-validator weights; the chain (and the
Yuma consensus mechanic) aggregates them.

This is why miners see "validator RT", "Yuma", "Rizzo", and "owner" rows
diverging: each validator's local `score_avg` produces a slightly different
G1/G2 split, the consensus row averages over all of them, and a single
validator can disagree with the consensus during the period before its
local history has caught up.

## The `score` top-level field (removed)

Older versions of the validator API exposed a top-level `score` per miner. It
has been removed because there were two distinct quantities both called
`score`:

1. The per-round `delta ** 1.2` raw signal (only meaningful in-round).
2. The rolling rank-based aggregator value (the one that drives weights).

Conflating them caused confusion when miners with a low `val_loss` still saw
`score = 0`: the round had ranked them outside top-3, or they had hit the
duplicate-`val_loss` penalty. Current dashboards expose `score_latest` and
`score_avg` instead, which are unambiguous.

## Why a miner with low `val_loss` can still have `score = 0`

The most common cases, in order of frequency:

1. **Not in the round's top 3 by `val_loss`.** The rank-based mapping is
   geometric and cuts off after rank 3. Rank 4 is 0.0.
2. **`delta == 0`** — the miner did not beat the baseline. Even if their
   `val_loss` is lower than other submissions, if it is above the validator's
   own pre-merge baseline the delta clamps to 0 and the round score is 0.
3. **Tied `val_loss` with another miner** — both sides get 0 regardless of
   rank (duplicate-submission heuristic).
4. **Recency gate failure** — the miner has plenty of history but missed the
   most recent round, so they drop off Weight Group 1 even though
   `score_avg` is high.
5. **Hotkey rotation** — the UID's hotkey changed at some point and history
   was reset; the miner has fewer than the required samples.

Code: `connito/validator/evaluator.py:finalize_round_scores`,
`connito/validator/evaluator.py:build_submission_uid_weights`.

## The "97 % burn" question, end-to-end

The chain shows ~97 % of a validator's emission going to UID 0 when one of
these things is happening on that validator:

1. **No miner cleared the G1 recency gate** — `build_submission_uid_weights`
   redirects the 98 % G1 share to UID 0. 2 % G2 is still split among
   evaluated miners. This is the most common cause on a freshly-restarted
   validator with no rolling history, and on a quiet expert group where
   miners are not consistently submitting.
2. **Stale-weights fallback** — the validator did not submit weights in
   `cycle_length` blocks (e.g., it was down). `_submit_fallback_weights`
   aggregates peer consensus and emits to top-3 peer-favored miners, plus
   reserving a separate **5 % to UID 0** (`SUBNET_OWNER_WEIGHT_SHARE`). This
   is distinct from G1 redirection.
3. **All peers are themselves in even-weight fallback state** — emit 100 %
   to UID 0 to avoid amplifying a subnet-wide cascade. Code:
   `connito/shared/chain.py:_submit_fallback_weights`.

> None of these are "burning." UID 0 (the subnet owner) receives the TAO. The
> dashboard convention of calling this "burn" is loose; the on-chain effect
> is owner emission.

Code: `connito/shared/chain.py` (lines 568-586 for the no-differentiated-peer
case, line 590 for the 5 % owner share applied to the fallback path),
`connito/validator/run.py` (G1 redirection log line, ~line 1064).
