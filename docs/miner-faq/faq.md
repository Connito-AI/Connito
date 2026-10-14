# Miner FAQ

The most common questions miners ask, with answers grounded in code. Each
entry ends with a `Code:` pointer; longer answers link to a dedicated doc.

## Why is the subnet "burning" 97 % of emission?

**Short answer:** It's not a burn. It's the empty-G1 fallback — when no
miner clears the recency gate (a score recorded in *both* of the last 2
rounds), the validator redirects the 98 % Weight Group 1 share to UID 0
(the subnet owner). UID 0 receives the TAO; nothing is destroyed. There is
also a separate 5 % owner share applied to peer-consensus fallback paths
(`reserve_subnet_owner_share`). See `scoring.md` for the long version.

**When the redirect fires:** validator just restarted (no rolling history),
a quiet task, or a cycle where every assigned miner submitted invalid
or missed entirely.

Code: `connito/validator/evaluator.py:build_submission_uid_weights` (the g1
redirect), `connito/validator/run.py` (the "g1 empty — redirecting" log line).

## Why is my `val_loss` null and `delta_loss` Unknown?

**Short answer:** one of five things, all listed in `scoring.md` under
"When `val_loss` and `delta_loss` are null":

1. No chain commit (`no_chain_commit`).
2. Pre-eval validation failure: `signature_invalid` / `hash_mismatch` /
   `bad_residual_shard` (the shard is not exactly the task's residual).
3. `non_finite_loss` — eval ran, returned NaN/Inf. A miner zeroed by the c4
   isolation check also shows `val_loss = inf`, with `replay_loss_increase`.
4. Operational failure on the validator side (download trouble, timeout,
   OOM). No new sample is written and prior EMA is preserved.
5. The round ended before the validator reached your miner.

The `eval_status_label` field tells you which of (1)-(3) you hit. (4) and
(5) produce no new sample and usually no label change.

Code: `connito/validator/evaluator.py:evaluate_one_miner_sync`,
`connito/validator/evaluator.py:validate_miner_submission`.

## Why is my score 0 even with a low `val_loss`?

**Short answer:** `finalize_round_scores` keeps only the top 3 miners per
round and assigns them 2.25 / 1.5 / 1.0; everyone else is 0. If your
`val_loss` is low but not in the top 3 *for this validator's round*, you
get 0 for the round.

Other ways to land at 0 with a finite task loss:

- `delta == 0`: you didn't beat the baseline, which is the **pretrained base
  model** (Qwen with no residual) on this round's data.
- The **c4 isolation check** zeroed you: your residual raised the loss on
  general web text by more than the tolerance (status 15,
  `replay_loss_increase`). Your `val_loss` then shows `inf`.
- Your `val_loss` is *exactly equal* to another miner's — both get 0 (the
  duplicated-submission heuristic). Two miners running the stock code from
  the same published winner on the same data can hit this.
- You hit the validator on a round where only your foreground was eval'd
  and other validators happened to find better miners.

Code: `connito/validator/evaluator.py:finalize_round_scores`,
`connito/validator/evaluator.py:_RANK_TO_SCORE = (2.25, 1.5, 1.0)`,
`connito/validator/evaluator.py:evaluate_one_miner_sync`.

## Which task am I on, and how is that assigned?

**The task is served by the subnet owner, not chosen by you.** At startup, and
again at each Distribute, your miner asks the owner's phase service which task
is active, downloads that task's bundle (hash-checked) into
`expert_groups/<name>/`, and loads it. The current task is
`exp_pmc_residual`:

```yaml
task:
  expert_group_name: exp_pmc_residual   # fallback when the API is unreachable
```

The task's `config.yaml` defines its `group_id` (9), its datasets, and its
residual settings (K, A, τ). The validator filters its miner roster by
`expert_group == config.task.exp.group_id` from each `MinerChainCommit`, so
the chain commit you write in MinerCommit2 must carry the active task's
`group_id`. A stock miner does this automatically.

There is **no self-selection**. The entire subnet runs one task at a time; a
miner committing under any other group is invisible to validators and earns
nothing. When the owner switches tasks, miners and validators that follow the
API switch together.

Code: `connito/shared/config.py:TaskCfg`,
`connito/shared/task_sync.py:sync_active_task`,
`connito/shared/cycle.py:get_miners_from_commit`,
`expert_groups/exp_pmc_residual/config.yaml`.

## How do I commit a checkpoint correctly?

Stock miner does this for you in `connito/miner/model_io.py:commit_worker`.
The sequence:

1. **MinerCommit1:** `_prepare_checkpoint_for_commit` picks
   `select_best_checkpoint(config.ckpt.checkpoint_path)` and signs the
   `model_hash` with the miner's hotkey. `commit_status` writes
   `SignedModelHashChainCommit(signed_model_hash=...)` to chain.
2. **MinerCommit2:** `_upload_checkpoint_to_hf_safe` uploads
   `model_expgroup_{group_id}.safetensors` from the checkpoint dir to HF and
   captures the revision SHA. Then `_commit_model_hash` writes
   `MinerChainCommit(expert_group, model_hash, global_ver, hf_repo_id,
   hf_revision)` to chain, with `hf_revision` truncated to 7 chars.

The filename **must** be `model_expgroup_{group_id}.safetensors`. A `.pt`
file is refused.

A commit is "valid" when:

- `signature_verified` (signed by the miner's hotkey);
- the shard is **exactly the task's residual**: the residual expert and router
  keys for all 40 layers, at the protocol shapes and dtypes (experts bf16,
  router fp32), nothing else, all finite, every expert trained;
- `hash_verified` (downloaded file hashes to the committed `model_hash`).

Code: `connito/miner/model_io.py:commit_worker`,
`connito/shared/checkpoints.py:ModelCheckpoint.validate`,
`connito/shared/checkpoint_helper.py:verify_residual_shard`.

## Why is validator RT / Yuma / Rizzo / Owner showing different weights?

Validators maintain independent score histories. Each one builds its own
top-3 by `score_avg` over its own rolling 8-sample window, against its own
foreground assignment slice. Their on-chain weight submissions differ
because their local data differs.

**Yuma consensus** is the chain's *stake-weighted aggregation* of every
validator's submission. It will not match any single validator exactly —
that's the point.

Divergence is expected during:

- Validator startup (8 cycles to fill the score window).
- After a recency gate forces a single validator into the UID-0 redirect.
- During task switches or eval-set changes (a brief
  consensus-divergence window per upgrade).

Code: `connito/shared/chain.py:submit_weights`,
`connito/validator/aggregator.py:MinerScoreAggregator`.

## What is the cycle, and what does each phase mean for me?

See `phases.md` for the full walkthrough. Short version:

- **Distribute** — download the round's published winner (a residual shard),
  if validators published one.
- **Train** — train the residual experts on top of the frozen base. This is
  the bulk of the cycle.
- **MinerCommit1** — sign your shard's hash; commit the signature to chain.
- **MinerCommit2** — upload your shard to HF; commit `(hash, repo,
  revision)` to chain.
- **Submission, Validate** — validators score your residual against the
  pretrained base, on the task data and on the c4 check. You are idle.
- **Merge** — nothing is merged; validators are idle too.
- **ValidatorCommit1/2** — validators commit the round winner they
  published. Next cycle's Distribute pulls from there.

The phase lengths are served by the owner phase service at
`cycle-api.connito.ai`, the single source of truth for the current phase. Ask
`GET /get_phase` rather than computing them from local config.

Code: `connito/shared/cycle.py:PhaseManager`,
`connito/shared/cycle.py:wait_till`.

## How is `chain_weight_stake_weighted` computed?

The validator emits a per-miner weight `w_v(uid)` on chain via
`set_weights`. The chain (and the dashboard) computes:

```
chain_weight_stake_weighted(uid) = sum over v of stake(v) * w_v(uid)
                                   / sum over v of stake(v)
```

where `v` ranges over all weight-setting validators. The dashboard then
normalizes across miners.

The old top-level `score` field was removed because it conflated the
per-round `delta ** 1.2` signal with the rolling rank-based aggregator
value. Use `score_latest`, `score_avg`, `score_samples` instead. See
`scoring.md`.

Code: `connito/shared/chain.py:submit_weights`,
`connito/validator/run.py:1100-1116`.

## Why do newly registered miners sometimes get rewards without a HF repo?

> **TODO: code says X, community says Y.** The Discord claim (from
> "hauwk") is that newly registered miners inherit the previous miner's
> "temp history" at the same UID. The code I have available **contradicts
> this**: `MinerScoreAggregator.add_score` explicitly resets the UID's
> history whenever the stored hotkey for that UID differs from the new
> hotkey:
>
> ```python
> # connito/validator/aggregator.py:add_score, line 234-237
> elif state.hotkey != hotkey:
>     # Hotkey changed -> reset scores for this uid
>     state.hotkey = hotkey
>     state.series.clear()
> ```
>
> And `miner-validation-group-promotion.md` documents: "Hotkey rotation:
> When a UID's hotkey changes, `MinerScoreAggregator.add_score` resets
> that UID's history."
>
> So according to the code, **a newly registered miner at a UID does not
> inherit the previous miner's score history.** If the chat-claimed
> behavior is being observed in production, plausible alternative
> explanations:
>
> 1. The new hotkey was *not* fully detected by some validators in the
>    transition cycle — `set_hotkey` is called on `add_score`, so the
>    reset only fires when the validator next scores the UID. Until then
>    the in-memory aggregator still holds the old hotkey's history.
> 2. A validator's `score_aggregator.json` on disk pre-dates a UID's
>    deregistration but was loaded after re-registration — the next
>    `add_score` would reset, but `submit_weights` between load and reset
>    would emit on stale data.
> 3. The chain-set ballots from peers are doing the lifting: a new
>    miner can appear in Group A/B via *other validators'* chain-set
>    weights (which themselves haven't reset yet), without ever being
>    locally scored. See `miner-validation-group-promotion.md`.
>
> Recommend the bot flag this as "behavior the code does not predict; if
> reproducible, file an issue with `score_aggregator.json` snapshots."

Code: `connito/validator/aggregator.py:add_score` (the hotkey-reset block).

## What does each `eval_status_label` mean and how do I fix it?

See `eval-status.md` for the full table. Quick reference:

- `ok` (0) — evaluated cleanly. No action.
- `non_finite_loss` (1) — eval produced NaN/Inf. Your residual diverged;
  lower the learning rate.
- `statedict_parse_failed` (2) — the shard verified but would not load onto
  the model. Rare; re-upload the stock file.
- `signature_invalid` (3) — signed with the wrong key or hash drift. Re-commit.
- `hash_mismatch` (4) — disk file ≠ chain hash. Don't push twice to the same
  revision.
- `no_chain_commit` (6) — MinerCommit2 missing HF coords. Check `HF_TOKEN`.
- `oom` (8) / `deadline_exceeded` (10) — validator-side hardware or
  time budget. No miner action.
- `repo_unavailable` (12) — your committed repo or revision is gone or not
  public. Keep it public and in place.
- `bad_residual_shard` (14) — the shard is not exactly the task's residual
  (a corrupt, truncated or empty file, extra/missing keys, wrong shape or
  dtype, `.pt`, NaN, or an untrained expert). Upload the stock `save_residual_shard` output.
- `replay_loss_increase` (15) — the c4 isolation check zeroed you. Vary your
  training data every round.
- Codes 5 and 13 are retired (they belonged to the old expert-group shards)
  and never set any more.

The labels are authored in **validator code**
(`connito/shared/telemetry.py:EVAL_STATUS_CODES`); the dashboard reads them
from the `validator_miner_eval_status` Prometheus gauge. Validation failures
(codes 3, 4, 6 and 14) are recorded on that gauge too.

## How big a GPU do I need?

**One 48 GB GPU** (RTX A6000, L40S or larger). The miner holds the whole base
model, `Qwen/Qwen3.6-35B-A3B`, with its routed experts stored in fp8, and
trains only the small residual experts beside it. Measured at the defaults
(sequence length 4096, `local_par.micro_batch_size` 2 ×
`gradient_accumulation_steps` 2):

- RTX A6000: peak 39.5 GiB allocated / 41.2 GiB reserved over a 300-step run;
  39.53 GiB / 39.8–40.0 GiB reserved for the miners in a full rehearsal.
- L40S (44.4 GiB usable): the measured floor. Micro-batch 2 fits.
- A 24 GB or 40 GB card is too small.

You also need about 100 GB of free disk for the 67 GiB (72 GB) checkpoint, a
host with about 62 GB of RAM or more, an NVIDIA driver ≥ 570, and a Python
with its headers (`Python.h`). The model must sit on **one** GPU; it cannot
be split across several.

If you're getting OOM during Train, lower `local_par.micro_batch_size` (to 1)
and raise `local_par.gradient_accumulation_steps` to keep 4 rows per
optimizer step. Don't reduce `sequence_length`: it is locked at the task's
4096, the length validators score at.

Code: `connito/shared/config.py:ParallelismCfg`,
`expert_groups/exp_pmc_residual/config.yaml`. Details in `miner-config.md`,
"Hardware and environment".

## Should I upload `.safetensors` or `.pt`?

**`.safetensors`, and only `.safetensors`.** A validator reads the shard's
header with `safe_open` and refuses any other file as `bad_residual_shard`
(status 14), including a `.pt`. A `.pt` is a Python pickle, which can run code
on load, so it is never loaded.

The stock miner writes `model_expgroup_{group_id}.safetensors` with
`save_residual_shard` and uploads only that file, so a stock miner is
automatically correct. If you use a custom upload script, upload exactly that
file and name.

Validators publish the round winner in the same format, and the miner
downloads `model_expgroup_{group_id}.safetensors` from them.

Code: `connito/shared/checkpoint_helper.py:verify_residual_shard`,
`connito/shared/helper.py:expert_group_shard_name`,
`connito/miner/model_io.py:_upload_checkpoint_to_hf_safe`.

## How long does it take a new miner to earn rewards?

Hard floor: 2 cycles. The Weight Group 1 ballot requires a score in both
the current round AND the round `cycle_length` blocks ago — a miner that
just started has at most 1 round of history and cannot clear the
recency gate.

Soft floor: ~3-8 cycles before the rolling `score_avg` starts to
stabilize. The aggregator window is 8 samples; until you fill it, your
avg fluctuates heavily and may bounce in and out of A/B/C placement.

If you are not in any validator's foreground slice (it is randomized per
cycle), background eval picks you up via the staleness queue — but
foreground is faster.

Code: `connito/validator/evaluator.py:build_submission_uid_weights`
(recency gate at line 442-444),
`connito/validator/aggregator.py:MinerScoreAggregator` (score_window=8),
`connito/validator/round.py:Round.freeze` (background staleness ordering).

## Are validators using `gemini-3.1-flash-lite` for anything?

Not as far as I can tell from the validator code. Validators run a local
PyTorch evaluation of `Qwen/Qwen3.6-35B-A3B` plus each miner's residual
experts — there is no LLM-API call path in `connito/validator/`. The Discord
support bot uses Gemini, but that is a separate process outside this
repository and does not influence scoring or chain weights.

This question was tagged as "unrelated" in the original prompt, so we'll
leave the bot's own model choice as a deployment decision rather than a
scoring concern.

## What if I want to run several miners from one machine?

The subnet runs one task at a time, and every miner trains that task's
residual. Running two miners means two miner processes, two HF repos, and two
hotkeys — they look like two separate miners to the subnet.

Each miner needs its own 48 GB GPU: the base model must sit on one GPU, and
two miners cannot share one. Point both at the same Hugging Face cache so the
67 GiB checkpoint is stored once.

Give each its own data order (see "Why was my score zeroed after restarting
from the published winner?"). Two stock miners starting from the same winner
read identical batches; if they produce the same shard, their `val_loss` is
exactly equal and both score 0 as a duplicate.

Code: `connito/shared/config.py:TaskCfg`,
`connito/validator/evaluator.py:finalize_round_scores`.

## What does a miner train, and what is in the shard?

The base model, `Qwen/Qwen3.6-35B-A3B` at revision
`995ad96eacd98c81ed38be0c5b274b04031597b0`, is **frozen**. Miners train small
**residual experts** that sit beside each of its 40 MoE blocks and add to the
block's output:

    h = h_base(x) + Σ_j q_j(x) · E_j(x),   q_j = sigmoid((sigmoid(w_j · x) − A) / τ)

- `E_j` is a SwiGLU MLP (2048 → 512 → 2048); `w_j` is its router row. The gate
  `q_j` decides per token how much the expert adds.
- K experts per block, threshold A and temperature τ come from the task
  (`exp_pmc_residual`: K = 1, A = 0.8, τ = 0.05).
- A fresh residual has `down_proj` = 0, so it starts exactly at the base.
- The loss is next-token cross-entropy over every labelled token: task rows
  (pmc) and c4 replay rows, which keep the residual quiet on general text.
  There is no auxiliary loss.

**The shard** (`model_expgroup_9.safetensors` for this task) holds, for every
layer L in 0–39 and expert j in 0..K−1:

- `model.layers.{L}.mlp.residual_experts.{j}.{gate,up,down}_proj.weight`, bf16;
- `model.layers.{L}.mlp.residual_router.weight`, [K, 2048], fp32.

At K = 1 that is 160 tensors, about 252 MB. Nothing else is allowed in the
file (status 14 otherwise).

Code: `connito/shared/modeling/residual.py:install_residual_experts`,
`connito/shared/modeling/residual.py:residual_shapes`,
`connito/shared/evaluate.py:chunked_lm_loss`.

## Why was my score zeroed after restarting from the published winner?

Most likely because you trained on the same data again. **The stock miner
replays the same training stream every round**: its loader has no seed, it is
rebuilt at every restart (including the restart from each round's winner),
and every one-GPU miner reads the same slice (rank 0 of 10). Training the
winner again on data it has already seen overfits, and the c4 isolation check
catches it.

Measured, two miners starting from the same published winner:

- **same stream again** (stock): task loss worse than the starting shard, c4
  loss +0.064 / +0.066 above the base: both zeroed (status 15), nothing
  published. A full two-round rehearsal on the stock code gave the same
  result in round 2 (c4 +0.061 / +0.059).
- **each source shuffled per (hotkey, round)**: task loss −0.0014 / −0.0047
  below the starting shard, c4 −0.024 / −0.025: a winner was published.

**Fix:** vary or shuffle your data every round, for example by shuffling each
streaming source with a seed built from your hotkey and the round. Do not just
pass `seed=` to `get_dataloader`; that switches to the validators' eval path
and leaves you a few hundred sequences. Details and where to hook it:
`miner-config.md`, "Vary your training data every round".

Code: `connito/miner/train.py:setup_training`,
`connito/shared/dataloader.py:DefaultStreamingTorchDataset.get_tokenised_dataset`,
`connito/validator/evaluator.py:evaluate_one_miner_sync`.

## Can I split the model over several GPUs?

**No. Load the base model onto one GPU.** On a 6× RTX A6000 host, Qwen loaded
with transformers `device_map="auto"` over three GPUs returned wrong logits
with no error or warning: loss 12.42 on a test sentence instead of 2.04, every
run. Two GPUs happened to give the right answer, but the cause (cross-device
races in the fla kernels and the final logits copy) means no split is
trusted.

So the loader refuses a model spread over more than one device, with
`the base model is split over devices [...]. Load it onto one GPU`. A single
48 GB card holds the whole model with its routed experts in fp8.

If you have several GPUs, run one miner per GPU (each with its own hotkey and
HF repo), or pin the miner to one GPU with `CUDA_VISIBLE_DEVICES`.
`local_par.world_size > 1` is not part of the supported miner profile.

Code: `connito/shared/modeling/mycelia.py:get_base_model` (the split check),
`connito/shared/config.py:ParallelismCfg`.

## Known issues in the reference miner

The stock miner is a reference implementation. These are known and do not
lose you score on their own:

1. **The same training data every round.** The train stream restarts at every
   `setup_training`, and all one-GPU miners read rank 0's slice. This one
   *does* cost score; see "Why was my score zeroed after restarting from the
   published winner?".
2. **Skipped micro-batches.** With the stock data stream, 13–28% of
   micro-batches in the rehearsal logged `Non-finite loss detected, skipping
   batch` and were skipped. With shuffled data none were. 50 such batches in a
   row stop training with a `FloatingPointError`.
3. **An extra model load at the first round.** A miner starting from its own
   fresh residual loads the model (about 125 s on an A6000), then reloads it
   once more at the first inner step (56–61 s): it starts with no checkpoint
   recorded (`current_model_meta` is None), and the resync check in the
   training loop reloads whenever that is None. Starting from a published
   winner took 77–107 s with no reload.
4. **A `TypeError` on Ctrl-C.** On shutdown, a failure inside the cleanup path
   is logged through a standard-library logger with a keyword argument it
   doesn't accept (`connito/miner/train_helper.py:76`), so a `TypeError`
   traceback can follow the real error. The process still stops.

Code: `connito/miner/train.py:train_worker`,
`connito/miner/train_helper.py`, `connito/shared/model.py:get_model_from_checkpoint`.

## What is the c4 check?

Besides scoring your residual on the task data, every validator scores it on
**c4** (general English web text, `allenai/c4`), the task's replay source. It
compares your c4 loss with the pretrained base's on the same batches (about
65,000 real tokens per round). The point: a residual that helps on medical
papers but damages ordinary text should not win.

- If `your c4 loss − base c4 loss` exceeds the locked
  `evaluation.max_replay_loss_increase` (default 0.02 nats), or any c4 batch
  is non-finite, your `val_loss` is set to `inf`, your round score is 0, and
  your status is 15, `replay_loss_increase`. You are never published as the
  round winner.
- The validator's log line `c4 isolation gate failed` shows `reason`,
  `replay_loss_increase` and the `tolerance=` in force. The
  `validator_miner_replay_loss_increase{miner_uid}` gauge shows your
  increase every round.

Honest training moves c4 loss **down**, because the c4 replay rows (20% of the
training mix) keep their labels: −0.025 over a 300-step run, and −0.021 to
−0.025 for fresh-data miners in the rehearsal. What trips the check in
practice is overfitting on repeated data (+0.06 measured).

Code: `connito/validator/evaluator.py:evaluate_one_miner_sync`,
`connito/shared/config.py:EvalCfg`,
`expert_groups/exp_pmc_residual/config.yaml` (the `replay: true` source).
