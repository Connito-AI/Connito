# Miner FAQ

The 15 most common questions miners ask, with answers grounded in code. Each
entry ends with a `Code:` pointer; longer answers link to a dedicated doc.

## Why is the subnet "burning" 97 % of emission?

**Short answer:** It's not a burn. It's the empty-G1 fallback — when no
miner clears the recency gate (a score recorded in *both* of the last 2
rounds), the validator redirects the 98 % Weight Group 1 share to UID 0
(the subnet owner). UID 0 receives the TAO; nothing is destroyed. There is
also a separate 5 % owner share applied to peer-consensus fallback paths
(`reserve_subnet_owner_share`). See `scoring.md` for the long version.

**When the redirect fires:** validator just restarted (no rolling history),
quiet expert group, or a cycle where every assigned miner submitted invalid
or missed entirely.

Code: `connito/validator/evaluator.py:build_submission_uid_weights` (g1
redirect, lines 450-453), `connito/validator/run.py:1064-1070`.

## Why is my `val_loss` null and `delta_loss` Unknown?

**Short answer:** one of five things, all listed in `scoring.md` under
"When `val_loss` and `delta_loss` are null":

1. No chain commit (`no_chain_commit`).
2. Pre-eval validation failure: `signature_invalid` / `hash_mismatch` /
   `expert_group_or_nan`.
3. `non_finite_loss` — eval ran, returned NaN/Inf.
4. Operational failure on the validator side (timeout, OOM, RPC error).
   No new sample is written and prior EMA is preserved.
5. The phase boundary closed before the validator reached your miner.

The `eval_status_label` field tells you which of (1)-(3) you hit. (4) and
(5) produce no new sample and no label change.

Code: `connito/validator/evaluator.py:evaluate_one_miner_sync` (line 780-805
for the val_loss/finite check), `connito/validator/evaluator.py:validate_miner_submission`.

## Why is my score 0 even with a low `val_loss`?

**Short answer:** `finalize_round_scores` keeps only the top 3 miners per
round and assigns them 2.25 / 1.5 / 1.0; everyone else is 0. If your
`val_loss` is low but not in the top 3 *for this validator's round*, you
get 0 for the round.

Other ways to land at 0 with a finite `val_loss`:

- `delta == 0` (you didn't beat the validator's baseline loss).
- Your `val_loss` is *exactly equal* to another miner's — both get 0 (the
  duplicated-submission heuristic).
- You hit the validator on a round where only your foreground was eval'd
  and other validators happened to find better miners.

Code: `connito/validator/evaluator.py:finalize_round_scores`,
`connito/validator/evaluator.py:_RANK_TO_SCORE = (2.25, 1.5, 1.0)`.

## Which expert group am I in, and how is that assigned?

Your expert group is set in your local config, but the value is **locked**:

```yaml
task:
  expert_group_name: exp_nemotron_c4   # locked — resets if you change it
```

The directory's `config.yaml` defines `group_id` (currently `4`). The validator
filters its miner roster by `expert_group == config.task.exp.group_id` from each
`MinerChainCommit`, so the chain commit you write in MinerCommit2 must agree
with the directory you trained under.

There is **no chain-level placement and no self-selection**. Loading a config
with `auto_update_config` resets `expert_group_name` to the built-in default and
rewrites your YAML, so editing it does not move you to another group. The entire
subnet runs one group at a time; a miner committing under any other group is
invisible to validators and earns nothing. Group changes are announced in
advance and take effect when you upgrade to the release that carries them.

Code: `connito/shared/config.py:TaskCfg` (`_LOCKED_FIELDS`),
`connito/shared/cycle.py:get_miners_from_commit`,
`expert_groups/exp_nemotron_c4/config.yaml`.

## How do I commit a checkpoint correctly?

Stock miner does this for you in `connito/miner/model_io.py:commit_worker`.
The sequence:

1. **MinerCommit1:** `_prepare_checkpoint_for_commit` picks
   `select_best_checkpoint(config.ckpt.checkpoint_path)` and signs the
   `model_hash` with the miner's hotkey. `commit_status` writes
   `SignedModelHashChainCommit(signed_model_hash=...)` to chain.
2. **MinerCommit2:** `_upload_checkpoint_to_hf_safe` uploads the
   checkpoint dir to HF (`allow_patterns=["model_expgroup_{N}.safetensors",
   "model_expgroup_{N}.pt"]`). Capture the revision SHA. Then
   `_commit_model_hash` writes
   `MinerChainCommit(expert_group, model_hash, global_ver, hf_repo_id,
   hf_revision)` to chain, with `hf_revision` truncated to 7 chars.

The filename **must** be `model_expgroup_{group_id}.safetensors`
(preferred) or `.pt`. The validator's `_build_download_targets` looks for
exactly this pattern.

A commit is "valid" when:

- `signature_verified` (signed by the miner's hotkey).
- `hash_verified` (downloaded file hashes to the committed `model_hash`).
- `expert_group_verified` (no foreign-expert keys, no NaN/Inf in any
  tensor).

Code: `connito/miner/model_io.py:commit_worker`,
`connito/shared/checkpoints.py:ChainCheckpoint.validate`.

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
- During expert-group migrations or eval-set changes (a brief
  consensus-divergence window per upgrade).

Code: `connito/shared/chain.py:submit_weights`,
`connito/validator/aggregator.py:MinerScoreAggregator`.

## What is the cycle, and what does each phase mean for me?

See `phases.md` for the full walkthrough. Short version:

- **Distribute** — download the latest global checkpoint.
- **Train** — train your expert group (the bulk of the cycle: ~60 min).
- **MinerCommit1** — sign your checkpoint hash; commit signature to chain.
- **MinerCommit2** — upload your shard to HF; commit `(hash, repo,
  revision)` to chain.
- **Submission, Validate, Merge** — validators evaluate, score, and merge
  your weights. You are idle.
- **ValidatorCommit1/2** — validators upload and commit the new global
  model. Next cycle's Distribute will pull from there.

Cycle length is ~500 blocks (≈100 minutes at 12s/block) per the deployed
`PhaseManager`. The owner phase service at `cycle-api.connito.ai` is the
single source of truth for the current phase.

Code: `connito/sn_owner/cycle.py:PhaseManager`,
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

- `ok` — evaluated cleanly. No action.
- `non_finite_loss` — eval produced NaN/Inf. Check your model for numerical
  instability.
- `statedict_parse_failed` — file corrupt/truncated. Re-upload.
- `signature_invalid` — signed with the wrong key or hash drift. Re-commit.
- `hash_mismatch` — disk file ≠ chain hash. Don't push twice to the same
  revision.
- `expert_group_or_nan` — shard has foreign experts or NaN/Inf tensors.
  Verify `task.exp.group_id` consistency and training stability.
- `no_chain_commit` — MinerCommit2 missing HF coords. Check `HF_TOKEN`.
- `download_failed` — validator couldn't pull from HF. Make the repo public.
- `oom` / `timeout` / `deadline_exceeded` / `rpc_error` — validator-side
  hardware/network/chain issues. No miner action.

The labels are authored in **validator code**
(`connito/shared/telemetry.py:EVAL_STATUS_CODES`); the dashboard reads them
from the `validator_miner_eval_status` Prometheus gauge.

## How big a GPU do I need?

The miner trains a *partial* model: only its expert group's MoE parameters
are trainable. Total memory is much smaller than a full DeepSeek-V2-Lite
fine-tune.

A 24 GB consumer GPU (3090 / 4090) is a reasonable minimum at the default
config (`fp16-mixed`, `batch_size=4`, `sequence_length=1024`,
`gradient_accumulation_steps=4`). Validators run on 40 GB GPUs because
they have to load the full model for evaluation plus host the eval
dataloader's buffers.

If you're getting OOM during Train, reduce `task.exp.data.batch_size`
in the expert-group config; don't reduce `sequence_length` (it is a
locked field and the validator's eval runs at 1024, so a different value
loses comparability).

Code: `connito/shared/config.py:ParallelismCfg`,
`expert_groups/exp_nemotron_c4/config.yaml`.

## Should I upload `.safetensors` or `.pt`?

**`.safetensors`.** Reasons:

1. `MINER_CHECKPOINT_SUFFIXES = (".safetensors", ".pt")` — validators
   prefer `.safetensors` (no pickle path, no code-execution surface).
2. The validator falls back to `.pt` for backward compatibility, but
   loads it with `torch.load(weights_only=True)` which gates off some
   legitimate `.pt` files.
3. Stock miner upload sends **both** in the same HF push, so a new miner
   is automatically safe.

The Lultime `.pt` 404 issue (validator stopped uploading `.pt`, miner
downloader hardcoded `.pt`) is the inverse problem and is documented in
`troubleshooting.md`.

Code: `connito/shared/helper.py:MINER_CHECKPOINT_SUFFIXES`,
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
PyTorch evaluation using DeepSeek-V2-Lite as the eval model architecture
— there is no LLM-API call path in `connito/validator/`. The Discord
support bot uses Gemini, but that is a separate process outside this
repository and does not influence scoring or chain weights.

This question was tagged as "unrelated" in the original prompt, so we'll
leave the bot's own model choice as a deployment decision rather than a
scoring concern.

## What if I want to run multiple expert groups from one machine?

The stock miner only trains one expert group at a time (single
`config.task.expert_group_name`). Running two groups requires two miner
processes, two HF repos, and two hotkeys — they look like two separate
miners to the subnet. The training and HF storage of each is independent.

`load_all_expert_groups` is a config flag that loads the full assignment
map but does not change the trained subset; the miner still only trains
the experts assigned to `config.task.exp.group_id`.

Code: `connito/shared/config.py:TaskCfg`,
`connito/shared/expert_manager.py:ExpertManager.load_expert_group_assignment`.
