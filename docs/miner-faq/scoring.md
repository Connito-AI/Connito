# How scoring works

Every block of this document maps a miner-visible field to the code that
produces it. The chain weights you actually earn TAO emission on are a long
way downstream of the raw `val_loss` printed in the validator logs, so trace
through each step.

## The per-round score: task gain, c4 rise and gate usage

When a validator scores a miner submission, it loads the miner's residual onto
the frozen base, evaluates it on the round's task batches and on the c4 replay
batches, and computes a raw score, then subtracts the base's own raw score:

```python
# connito/validator/evaluator.py:evaluate_one_miner_sync (simplified)
task_gain = baseline_loss - val_loss               # > 0: better than pretrained on the task
raw = (task_gain
       - a * max(0, c4_loss - baseline_c4_loss)    # replay_penalty
       - b * abs(task_usage - target)              # usage_penalty
       - c * c4_usage)                             # replay_usage_penalty
score = raw - base_raw                             # base_raw: the base's own raw this round
# defaults (locked validator config): a = b = c = 1.0, target = 0.2
```

- **`baseline_loss`** is the loss of the **pretrained base model** on the
  round's task eval batches (pmc for `exp_pmc_residual`), measured once per
  round with the validator's own fresh residual loaded (`down_proj` = 0, so
  it equals the base exactly). `val_loss` is the base plus the miner's
  residual on the same batches. The dashboard's `delta_loss` is this task
  gain.
- **Gate usage** is how much the residual fires: the gate summed over the K
  experts (Σ_j q_j), averaged over MoE layers and real (non-padding) tokens.
  It is measured during the same task and c4 evals; batches whose loss is
  dropped as non-finite don't count. The score asks the gate to fire on about
  `target` (0.2) of task tokens and to stay quiet on c4.
- **`base_raw`** is the same formula applied to the base on its own baseline
  evals. Its task gain and c4 rise are 0, so only its gate-usage terms count,
  using the base's measured usage. A miner identical to the base therefore
  scores exactly 0, and **a score above 0 means a raw score above the
  base's**. Only a score above 0 is ranked or published.
- **The c4 cutoff.** If `c4_loss − baseline_c4_loss` exceeds the locked
  `evaluation.max_replay_loss_increase` (default 0.02), or any c4 batch is
  non-finite, or the rise is NaN, the score is `−inf`, `val_loss` is reported
  as `inf`, and the status is 15 (`replay_loss_increase`). Such a miner is
  never ranked, never published, and kept off the podium (the round's top-3
  shards that a validator archives to its HF archive repo, when one is
  configured; finite scores are archived even when negative, only `−inf` is
  excluded).
- **Any other non-finite score** (for example every task batch NaN) is `−inf`
  with status 1, `non_finite_loss`.
- **A task with no c4 replay source** has no c4 terms, for the miner or the
  base: raw = task gain − b·|task_usage − target|.
- **On a fallback round** (the base could not be measured) `base_raw` is 0.

The weights are locked `EvalCfg` fields: `replay_loss_weight` (a),
`task_usage_weight` (b), `task_usage_target` (target) and
`replay_usage_weight` (c). There is no exponent on the gain.

The score is stored on the `MinerEvalJob` but is **not** what ends up on
chain — see "Rank-based finalization" below.

Code: `connito/validator/evaluator.py:evaluate_one_miner_sync`,
`connito/validator/evaluator.py:usage_penalties`,
`connito/validator/background_eval_worker.py:BackgroundEvalWorker._load_round_base`,
`connito/shared/modeling/residual.py:record_gate_usage`,
`connito/shared/config.py:EvalCfg`.

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
2. **Signature / hash / residual spec failed at the checkpoint verifier** —
   the on-disk shard's hash doesn't match the chain commit, the miner's
   hotkey didn't sign the model hash, or the shard is not exactly the
   residual spec (wrong keys, shapes or dtypes, a NaN/Inf tensor, or an
   untrained expert; status 14, `bad_residual_shard`). `evaluate_one_miner`
   is **not called** in any of these cases, so no `val_loss` is produced.
   Code:
   `connito/shared/checkpoints.py:ChainCheckpoint.validate`,
   `connito/validator/evaluator.py:validate_miner_submission`.
3. **Non-finite `val_loss`** — the eval ran but returned `+inf` (e.g.,
   every batch produced a NaN loss). `val_loss` is logged as `inf` and the
   miner's score is `−inf` and its `eval_status_label` is `non_finite_loss`.
   The aggregator still records a rank score of 0 for the round. A miner
   stopped by the c4 cutoff also shows `val_loss = inf` and score `−inf`, with
   `eval_status_label = replay_loss_increase`. Code:
   `connito/validator/evaluator.py:evaluate_one_miner_sync`.
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

At the end of each round, `finalize_round_scores` discards the raw scores and
replaces them with a fixed rank-based mapping. Only the top-3 miners **with a
score above 0** get non-zero score points; everyone else gets 0.

```python
# connito/validator/evaluator.py:_RANK_TO_SCORE
(2.25, 1.5, 1.0)
```

- Top-1 (highest score above 0) → **score = 2.25**
- Top-2 → **score = 1.5**
- Top-3 → **score = 1.0**
- Everyone else who was evaluated → **score = 0.0**
- Two miners with exactly the same score → **both get score = 0.0** (a
  duplicated submission is the much more likely cause than two miners
  legitimately producing bit-identical results).
- Miners with a score of 0 or below (not better than the pretrained base
  after the penalties), or `−inf` (stopped by the c4 cutoff) → **score = 0.0**
  regardless of rank.

Close scores between different miners do not trigger this rule — only
literal exact-equal scores do.

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

History older than `8 × cycle_length` blocks is pruned each time a round is
finalized (`connito/validator/aggregator.py:MinerScoreAggregator.prune_before_round`,
called from the finalize step in `connito/validator/run.py`).

## From `score_avg` to chain weights: `chain_weight_stake_weighted`

After Validate, the validator builds a `{uid: weight}` payload that goes onto
chain via `set_weights`. Every submission first reserves **25 %** for UID 0,
the subnet owner (`SUBNET_OWNER_WEIGHT_SHARE = 0.25`); the miners' payload is
split into two groups that share the remaining 75 %:

```python
# connito/shared/config.py:EvalCfg
weight_group_1_share = 0.98   # 98% of the miners' 75% = 73.5% on the wire
weight_group_2_share = 0.02   # 2% of the miners' 75%  = 1.5% on the wire
weight_group_1_size  = 3      # G1 holds at most 3 miners
weight_group_2_size  = 5      # G2 holds at most 5 miners
```

**Weight Group 1 (98 % of the miners' share, 73.5 % on the wire)** — top-3 of
validation groups A ∪ B by `score_avg`, subject to:

1. `record_count >= 3` (the miner has at least 3 samples in the aggregator).
2. The miner has scores recorded in **at least 3 distinct rounds** within the
   last 5 cycles (round ids from `current_round_id − 5 × cycle_length` to
   `current_round_id`). This is the "recency gate".
3. Its `score_avg` is **above 0**. Rank scores are 2.25 / 1.5 / 1.0 / 0, so
   an average of 0 means no top-3 placing with a positive round score in the
   window.

If no UID qualifies, the whole G1 share is **redirected to UID 0** (subnet
owner). With the 25 % owner reserve on top, UID 0 then receives 98.5 % of the
submission. This is the **no-winner-in-group fallback**, not a "burn":

> The chain emission is not destroyed. It is paid to the subnet owner because
> the validator could not pick a top-3 from miners that have been scoring
> consistently and positively. From the chain's perspective UID 0 receiving
> 98.5 % looks identical to a burn, but the operational meaning is "no
> qualified winner this round."

Code:
`connito/validator/evaluator.py:build_submission_uid_weights` (g1 selection
and redirect),
`connito/validator/round_groups.py:select_top_n_by_local_score` (skips a
`score_avg` of 0 or below),
`connito/validator/run.py:1064-1070` (the "g1 empty — redirecting" log line),
`connito/shared/chain.py:reserve_subnet_owner_share` (the 25 % owner reserve,
applied to every regular submission in
`connito/validator/chain_submitter.py:ChainSubmitter._submit_weight_one` and
to both fallback paths in `connito/shared/chain.py`).

**Weight Group 2 (2 % of the miners' share, 1.5 % on the wire)** — top-5 of A
∪ B ∪ C \ G1 by `score_avg`, subject to `record_count >= 1` and a `score_avg`
above 0. No recency gate. If G2 is empty too, the chain submitter normalizes
the weights, so UID 0 receives all of this validator's emission.

Within each group, the share is split **proportional to `score_avg`** (not
equal).

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

1. The per-round raw score (task gain minus penalties; only meaningful in-round).
2. The rolling rank-based aggregator value (the one that drives weights).

Conflating them caused confusion when miners with a low `val_loss` still saw
`score = 0`: the round had ranked them outside top-3, or they had hit the
duplicate-score penalty. Current dashboards expose `score_latest` and
`score_avg` instead, which are unambiguous.

## Why a miner with low `val_loss` can still have `score = 0`

The most common cases, in order of frequency:

1. **Its raw score did not beat the base's.** A low `val_loss` only buys the
   task gain; the gate-usage and c4 penalties are subtracted from it, and the
   result is compared with the base's own raw score. If the residual's gate
   fires more on c4, or further from the 0.2 target on the task, than the
   base's does, the extra penalty can cancel the gain. A round score of 0 or
   below gets rank score 0. See "Gate usage and the base's raw score".
2. **Not in the round's top 3 by score.** The rank-based mapping is geometric
   and cuts off after rank 3. Rank 4 is 0.0.
3. **The c4 cutoff stopped it** (status 15): its c4 loss rose more than 0.02
   above the base's. The score is `−inf`.
4. **Exactly the same score as another miner** — both sides get 0 regardless
   of rank (duplicate-submission heuristic).
5. **Recency gate failure** — the miner has history but fewer than 3
   distinct scored rounds in the last 5 cycles, so it drops off Weight Group 1
   even though `score_avg` is high.
6. **Hotkey rotation** — the UID's hotkey changed at some point and history
   was reset; the miner has fewer than the required samples.

Code: `connito/validator/evaluator.py:evaluate_one_miner_sync`,
`connito/validator/evaluator.py:finalize_round_scores`,
`connito/validator/evaluator.py:build_submission_uid_weights`.

## The "97 % burn" question, end-to-end

UID 0 always receives the 25 % owner reserve. The chain shows nearly all of a
validator's emission going to UID 0 (98.5 % or more) when one of these things
is happening on that validator:

1. **No miner cleared the G1 recency gate** — `build_submission_uid_weights`
   redirects the G1 share to UID 0, which then holds 98.5 % with the owner
   reserve. The 1.5 % G2 share is still split among evaluated miners. This is
   the most common cause on a freshly-restarted validator with no rolling
   history, and on a quiet task where miners are not consistently submitting.
2. **No miner has a positive score average** — no miner placed in a round's
   top 3 with a round score above 0 within its last 8 samples (for example,
   every round score was 0 or below), so every `score_avg` is 0 and
   `select_top_n_by_local_score` skips them all. G1 is empty and its share
   goes to UID 0; G2 is empty too, so after normalization UID 0 receives the
   whole emission.
3. **Stale-weights fallback** — the validator did not submit weights in
   `cycle_length` blocks (e.g., it was down). `_submit_fallback_weights`
   aggregates peer consensus and emits to top-3 peer-favored miners, with the
   same **25 % reserved for UID 0** (`SUBNET_OWNER_WEIGHT_SHARE`). This is
   distinct from G1 redirection.
4. **All peers are themselves in even-weight fallback state** — emit 100 %
   to UID 0 to avoid amplifying a subnet-wide cascade. Code:
   `connito/shared/chain.py:_submit_fallback_weights`.

> None of these are "burning." UID 0 (the subnet owner) receives the TAO. The
> dashboard convention of calling this "burn" is loose; the on-chain effect
> is owner emission.

Code: `connito/shared/chain.py:_submit_fallback_weights` and
`_asubmit_fallback_weights` (the no-differentiated-peer case and the owner
reserve on the fallback paths), `connito/shared/chain.py:reserve_subnet_owner_share`,
`connito/validator/run.py` (the G1 redirection log line).

## Which miner becomes next round's starting point

Besides scoring, each validator picks one miner per round whose residual it
publishes as the **round baseline**: the shard every miner restarts from in
the next Distribute. This is separate from rewards; it decides what the
subnet builds on.

The rule, applied when the round is finalized:

1. **Only miners with a score above 0 this round qualify.** The score is
   measured from the base's own raw score, so the pretrained base scores 0
   and a score above 0 means a raw score above the base's; a miner stopped by
   the c4 cutoff scores `−inf` and never qualifies.
2. Among those, the best-ranked **proven** miner wins: one of the top 3 by
   track record (`score_avg`) that submitted this round.
3. If none of the proven miners qualifies, the qualifying miner with the
   highest score this round wins. UID breaks an exact tie, so two
   validators never disagree.
4. **If no miner qualifies, nothing is published.** The validator commits a
   hash with no HF coordinates, and miners keep training from what they
   have.
5. **A round on the fallback baseline publishes nothing.** If the validator
   could not measure the base model's loss (its base shard failed to load,
   the baseline came back non-finite, or the c4 check could not run on a task
   that has a replay source), it scores the round against a fixed 100.0, logs
   "round on the fallback baseline; no winner will be published", and
   publishes no winner. There is no c4 eval on such a round, so no c4 terms
   and no c4 cutoff, and `base_raw` is 0. Rank rewards for that round still
   apply.

So the published residual is always one whose raw score beat the pretrained
base's on that round's data, penalties included, and passed the c4 cutoff. The
publishing validator logs `publish_baseline: published` with the winner's
`score`, or `publish_baseline: no miner beat the baseline`.

Code: `connito/validator/round.py:select_baseline_uid`,
`connito/validator/round.py:Round.baseline_winner_uid`,
`connito/validator/distribute.py:publish_round_baseline`.

## Gate usage and the base's raw score

The raw score charges the residual for where its gate fires, not only for the
loss it wins: `− |task_usage − 0.2| − c4_usage` (default weights). The base is
charged the same way, so the score only rewards a residual that does better
than the base on these terms as well as on the task.

**The base sits at about −0.2.** Its gate usage, with the validator's fresh
residual, is about 0.04–0.05 on task and c4 tokens alike. A gate with no
preference between task and c4 tokens, at any usage u up to 0.2, pays
`|u − 0.2| + u = 0.2` exactly, so `base_raw` is about −0.2 (measured −0.200 to
−0.208).

**What was measured** (production scoring code, real shards, two eval seeds):

- A 3-step test shard from the reference training code scored **+0.028 and
  +0.039**.
- Two full-Train rehearsal shards scored **+0.043 to +0.064**. They beat the
  base on the task by 0.05 to 0.06 nats; their gate usage rose on both splits
  (to about 0.08 task and 0.09 c4), which costs about as much as it saves.
- A deliberately leaky shard (its residual scaled ×32) failed the c4 cutoff:
  `−inf`, status 15.

**How to score higher.** Once the task gain is in, the usage terms are where
the room is: a gate that fires on task tokens (toward 0.2) and stays quiet on
c4 lowers both penalties below the base's. The reference miner trains on
cross-entropy only, with no term that separates the gate's behaviour on task
and c4 tokens, so it leaves that room unused. How to use it is up to you;
there is no tested recipe.

**Validators differ a little.** Each validator draws its own fresh residual,
so its base usage varies by about ±0.005 and `base_raw` with it. The same
shard can score up to about 0.008 differently on two validators.

The weights and target are locked validator config and can change between
releases, so read the current values in `connito/shared/config.py:EvalCfg`
and watch your own terms on the `validator_miner_score_term` and
`validator_miner_gate_usage` gauges.

Code: `connito/validator/evaluator.py:evaluate_one_miner_sync`,
`connito/validator/evaluator.py:usage_penalties`,
`connito/shared/modeling/residual.py:record_gate_usage`.

## Where to see your score terms

Each validator publishes every term of your round score per miner:

- `validator_miner_score_term{miner_uid, term}`, with `term` one of
  `task_gain`, `replay_penalty`, `usage_penalty`, `replay_usage_penalty`. The
  penalties are positive magnitudes that the score subtracts. The replay
  terms are present only when the c4 eval ran.
- `validator_miner_gate_usage{miner_uid, split}`, with `split` `task` or
  `replay`.
- `validator_miner_round_delta{miner_uid}`: the round score itself (raw minus
  the base's raw). It can be negative, and is `−Inf` when the c4 cutoff
  stopped you.
- `validator_round_base_raw_score` (the last measured round's `base_raw`) and
  `validator_round_base_raw_score_by_round{round_id}`. Neither is set on a
  fallback round.
- `validator_miner_replay_loss_increase{miner_uid}`: your c4 rise over the
  base.

In the validator's log, `evaluate_one_miner: complete` carries `val_loss`,
`baseline_loss`, `replay_loss_increase`, `task_usage`, `replay_usage`,
`task_gain`, `replay_penalty`, `usage_penalty`, `replay_usage_penalty`,
`raw`, `base_raw` and `score`. A miner stopped by the c4 cutoff gets
`evaluate_one_miner: c4 isolation gate failed; not ranked` with `reason`,
`replay_loss_increase` and the `tolerance=` in force. The round's base usage
and raw score appear on the `bg-eval: round snapshot loaded` line as
`baseline_task_usage`, `baseline_replay_usage` and `base_raw_score`.

Code: `connito/shared/telemetry.py` (`VALIDATOR_MINER_SCORE_TERM`,
`VALIDATOR_MINER_GATE_USAGE`, `VALIDATOR_MINER_ROUND_DELTA`,
`VALIDATOR_ROUND_BASE_RAW_SCORE`),
`connito/validator/evaluator.py:evaluate_one_miner_sync`.
