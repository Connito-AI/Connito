# Troubleshooting

Quick symptom → cause → fix table. Skim for your symptom, jump to the
relevant section.

| Symptom | Likely cause | Section below |
|---------|--------------|---------------|
| HTTP 403 from `cycle-api.connito.ai` when polling phase | Cloudflare WAF flagging the request | [HTTP 403 from phase API](#http-403-from-cycle-apiconnitoai-cloudflare-waf) |
| Miner never starts from a published winner | Nothing published yet, or the published shard was refused | [Fresh residual every round](#miner-keeps-starting-from-its-own-fresh-residual) |
| Training stops at setup, or the loss doesn't move | No residual installed, diverged residual, or flat training | [Training refuses to start](#training-refuses-to-start-or-the-loss-doesnt-move) |
| Miner submitted but never evaluated | Not on the active task's group, or `no_chain_commit` | [Submitted but never evaluated](#submitted-but-never-evaluated) |
| `score_avg` flat for 5+ hours | Stuck on a single rank, or operationally failing every round | [Score didn't change for 5+ hours](#score_avg-didnt-change-for-5-hours) |
| Status 15, `replay_loss_increase`, `val_loss = inf` | Residual raised the c4 loss; usually the same data trained again | [Zeroed by the c4 check](#score-zeroed-by-the-c4-check-status-15) |
| Status 14, `bad_residual_shard` | Shard is not exactly the task's residual | See `eval-status.md` |
| `eval_status_label` shows `hash_mismatch` | Re-uploaded the same revision after committing | See `eval-status.md` |
| `the base model is split over devices` at startup | Model spread over several GPUs | [Split over GPUs](#loader-refuses-a-model-split-over-several-gpus) |
| fla warns "Triton is not supported on current platform" | Python without `Python.h` | [fla Triton warning](#fla-warns-that-triton-is-not-supported) |
| Validator dashboards show different `chain_weight_stake_weighted` per validator | Expected — each validator has independent `score_avg` history | [Validators show different weights](#validators-show-different-chain_weight_stake_weighted) |

---

## HTTP 403 from `cycle-api.connito.ai` (Cloudflare WAF)

**Symptom:**

```
HTTP error calling https://cycle-api.connito.ai:443/get_phase (status=403)
```

**Cause:** Cloudflare's WAF in front of the phase API is flagging the
request as automated. Common triggers:

- Datacenter / VPN IP in a region that has been flagged.
- Default `python-requests` User-Agent.
- Too many sequential identical-shape requests in a short window.

**What the code does:** `get_with_retry` treats 403 as **non-retryable**
and returns `None`, so `get_phase_from_api` returns `None`. `wait_till` then
cannot make progress until the API responds — both miner and validator
will idle on `get_phase_from_api`.

**Fix:**

1. Confirm it's a Cloudflare 403 (not an upstream issue): `curl -i
   https://cycle-api.connito.ai/get_phase` — Cloudflare 403 has a
   `cf-ray` header.
2. **Run from a non-flagged egress IP.** Residential or
   well-reputed cloud IPs usually pass; freshly-allocated cloud IPs or
   known VPN ranges often fail.
3. Contact the subnet owner if persistent — they can whitelist your IP
   range against the WAF rule.

Code: `connito/shared/helper.py:get_with_retry` (non-retryable status list
includes 403), `connito/shared/cycle.py:get_phase_from_api`.

## Miner keeps starting from its own fresh residual

**Symptom:** every round, the miner logs

```
Keeping the fresh residual: no local checkpoint is a start
```

or Distribute logs `Fetching model from chain` with `should_download=False`,
and the miner never trains from a published round winner.

**Causes, in order of likelihood:**

1. **Nothing was published.** Validators publish a round winner only if some
   miner beat the pretrained base on that round's data and passed the c4
   isolation check; on the first round of a task, or a round where nobody
   qualified, they commit a hash with **no HF coordinates**, and there is
   nothing to download. This is normal. With nothing published, the miner
   resumes its own last trained checkpoint, or keeps its fresh draw if it
   has none.
2. **The published shard was refused.** The miner checks a downloaded
   winner exactly as validators check submissions (keys, shapes, dtypes,
   finite, trained). A shard for another K, for example right after a task
   switch, fails and is deleted. Look for `residual shard refused` in the
   miner log. A refused winner is downloaded again on each later fetch (about
   252 MB) until a newer one is published.
3. **Your local checkpoints are gone.** A wiped `checkpoints/` dir or a
   renamed `run_name` means there is nothing of your own to resume from.

**Fix:** for (1) nothing; train from your own residual and vary your data.
For (2), make sure the miner is on the active task (it follows the owner API
at Distribute). For (3), keep `run_name` and the checkpoint dir stable.

Code: `connito/shared/model.py:fetch_model_from_chain_validator`,
`connito/shared/model.py:get_model_from_checkpoint`,
`connito/shared/checkpoint_helper.py:verify_residual_shard`.

## Training refuses to start, or the loss doesn't move

**What normal looks like.** At step 0 the residual adds nothing (its
`down_proj` is 0), so the first losses are the pretrained base's. On this
task's eval data the base scored about 1.5–1.7 on pmc and 2.3–2.5 on c4 across
measured rounds; training batches mix both, so early batch losses in that
range are expected. Over a 300-step run the pmc eval loss fell by about 0.07,
most of it in the first 50 steps.

**Training stops at setup** with
`the model has no residual experts (.residual_ parameters) to train`: the
residual was not installed, which means the model was built outside
`get_model_from_checkpoint` (a custom loader). Use the stock path.

**Training stops mid-round** with `FloatingPointError: Non-finite loss
persisted for 50 consecutive batches`: the residual diverged, or the data
stream is producing only near-empty rows. Lower the learning rate, restart
from your last good checkpoint, and vary your data (see "Score zeroed by the
c4 check"). A few `Non-finite loss detected, skipping batch` warnings are
tolerated.

**The loss doesn't move.** Check the miner's `miner_param_sum` gauge: it sums
the residual parameters and moves on every optimizer step while training
works. If it is flat, confirm the miner is in the Train phase and is not
reloading every step (`Should reload model` in the log on every step means it
keeps finding a newer checkpoint on disk and rebuilds from it).

Code: `connito/shared/modeling/residual.py:freeze_all_but_residual`,
`connito/miner/train.py:train_worker`,
`connito/shared/model.py:get_model_from_checkpoint`.

## Submitted but never evaluated

**Symptom:** Your miner ran MinerCommit2 successfully, but on the
dashboard you have no `val_loss` for the round and your
`eval_status_label` is stale.

**Cause (in order of frequency):**

1. **You are not on the active task.** Validators filter their roster by
   `expert_group == config.task.exp.group_id`, the `group_id` of the task the
   owner API serves (9 for `exp_pmc_residual`). A miner still committing
   under an older task's group is never evaluated.

   Check: the `expert_group` in your miner's MinerCommit2 log line, against
   the active task's `config.yaml`.

   Fix: run a current miner release; it follows the owner API's active task
   at startup and at each Distribute. Do not pin `task.expert_group_name`.

2. **Your hotkey is not in the validator's assignment slice.** Each
   validator only evaluates `foreground_top_n = 5` miners per cycle
   (foreground), plus background eval if the worker has capacity. With
   ~100 miners and a few validators, a given miner is foreground-evaluated
   on each validator every ~5-10 cycles. Background eval covers the long
   tail in `staleness` order (longest-since-last-evaluated first).

   Fix: wait. You will get into the staleness tail eventually.

3. **Your chain commit is missing HF coords.** If
   `_upload_checkpoint_to_hf_safe` returned `(None, None)`, your
   `MinerChainCommit` has `hf_repo_id = None`. `Round.freeze` puts you in
   `freeze_zero_uids`. `eval_status_label = no_chain_commit`. See
   `eval-status.md`.

   Fix: confirm `HF_TOKEN` has write access to `hf.checkpoint_repo`.
   Validate the repo exists.

Code: `connito/shared/cycle.py:get_miners_from_commit`,
`connito/validator/round.py:Round.freeze`.

## `score_avg` didn't change for 5+ hours

**Symptom:** Dashboard `score_avg` stays the same value for many cycles.

Diagnostics:

1. **Did `score_samples` change?**
   - Yes, but values are the same — you're consistently scoring the same
     rank (likely 0). See `scoring.md` for why a low `val_loss` can still
     produce `score = 0`.
   - No, `score_samples` is flat — you are not being evaluated. See
     "Submitted but never evaluated" above.

2. **Did `score_latest` change?**
   - Yes — at least one validator is scoring you, but the rolling window
     hasn't reflected it yet (the avg is over the last 8 samples). Wait
     more cycles for the moving average to catch up.
   - No — same diagnosis as `score_samples` flat.

3. **Was there a validator outage?** Check the validator's
   `validator_main_loop_heartbeat_total` rate on Prometheus. If it's flat,
   the validator hung and no scoring happened this cycle.

4. **Is the validator on the empty-G1 fallback?** If `score_avg` is
   non-zero but you see `validator_miner_weight_submitted` showing UID 0
   at 98 %, the validator could not pick a top-3 by recency — your
   `score_avg` is fine but the weight share is going to the owner anyway.
   See `scoring.md` "97 % burn" section.

Code: `connito/validator/aggregator.py:MinerScoreAggregator`,
`connito/validator/evaluator.py:build_submission_uid_weights`.

## Validators show different `chain_weight_stake_weighted`

**Symptom:** "Validator RT" panel shows you at 5 %, "Yuma" shows 2 %, "Rizzo"
shows 0 %, owner panel shows 1 %.

**This is expected** during normal operation. Reasons:

1. **Independent score histories.** Each validator maintains its own
   `MinerScoreAggregator`. Validator A may have evaluated you twice in the
   last 8 cycles; validator B four times. The `score_avg` they each see is
   genuinely different.
2. **Independent assignment slices.** Validator A's foreground assignment
   probably doesn't overlap with B's. Different miners are evaluated
   first, different rounds produce different rankings.
3. **Yuma consensus is the *aggregated* row.** Yuma is the chain's stake-
   weighted consensus across all validators' submitted weights — it
   doesn't match any single validator's view; that's what it's for.

When the divergence is **not** expected: if Yuma shows you at 0 % but every
individual validator shows you at >0 %, the issue is on the consensus
calculation (or your stake-weighting interpretation), not the validators.

Code: `connito/shared/chain.py:submit_weights`,
`connito/validator/aggregator.py:MinerScoreAggregator.uid_score_pairs`.

## My score didn't update for a round

**Symptom:** a validator's log shows the round finalized, but
`score_aggregator.json` was not updated for your UID.

**Cause:** scores are written once per round, by `finalize_round_scores`, at
the start of the next cycle's MinerCommit1. Nothing is merged in the Merge
phase; it does not touch scores. If your score didn't update for the round:

1. Check your UID is in `validator-1 | … round_id=… written_uids=…`. If
   it's not in `written_uids`, you weren't in scored_uids and your prior
   EMA is preserved (operational failure path: download trouble, timeout,
   OOM).
2. If you are in `written_uids` but the value is 0.0, you ranked outside
   the top 3, didn't beat the pretrained baseline, were zeroed by the c4
   check, or hit the duplicate-`val_loss` penalty.

Code: `connito/validator/evaluator.py:finalize_round_scores`.

## My miner committed, but the validator used an older commit or none

**Symptom:** your miner logged its MinerCommit1 and MinerCommit2 commits, but
the round shows `signature_invalid` (3), `no_chain_commit` (6), or a score for
an older shard.

**Cause:** validators do not read your commits whenever they arrive. At round
freeze they read the chain **as of two blocks**: the end of MinerCommit1 (your
signed hash) and the end of MinerCommit2 (your model hash and HF
coordinates). Whatever your commitments were at those blocks is what gets
scored:

- **Signed hash committed after MinerCommit1 ended:** the validator pairs the
  previous signed hash with the new model hash, and the signature check fails
  (`signature_invalid`).
- **Model hash committed after MinerCommit2 ended:** the validator sees your
  previous MinerCommit2 commit, or none (`no_chain_commit`).
- **HF upload failed:** the MinerCommit2 commit went out without HF
  coordinates (`no_chain_commit`).

**Fix:** let each commit land inside its phase. The stock miner waits on the
chain-driven phases (`wait_till(config, PhaseNames.miner_commit_1)` and
`…miner_commit_2`); a custom miner on a wall-clock timer, or one whose HF
upload runs past MinerCommit2, misses the window. Look for `<MinerCommit2>
committing` in your log and check its block against the phase's end block
from `GET /get_phase`.

Code: `connito/shared/checkpoints.py:build_chain_checkpoints_from_previous_phase`,
`connito/shared/cycle.py:get_validator_miner_assignment`,
`connito/miner/model_io.py:commit_worker`.

## Score zeroed by the c4 check (status 15)

**Symptom:** status 15, `replay_loss_increase`, with `val_loss = inf`, often
right after your miner restarted from a published round winner. Your task loss
may still have beaten the baseline.

**Cause:** your residual raised the loss on general English text (c4) by more
than the locked tolerance, `evaluation.max_replay_loss_increase` (default
0.02 nats), compared with the pretrained base on the same batches. The usual
reason is **training on the same data again**: the stock miner replays one
fixed stream every round, and every stock miner reads the same slice. Measured
from the same published winner, the stock stream raised c4 loss by
+0.064 / +0.066 and both miners were zeroed; shuffling each source per
(hotkey, round) lowered it by 0.024 / 0.025 and the winner was published.

**Fix:**
1. Vary or shuffle your training data every round, e.g. shuffle each
   streaming source with a seed from your hotkey and the round. Do not pass
   `seed=` to `get_dataloader` for training. See `miner-config.md`, "Vary
   your training data every round".
2. Keep the task's c4 replay rows in your mix, with their labels.
3. Read the validator log line `c4 isolation gate failed` (`reason`,
   `replay_loss_increase`, `tolerance=`) or the
   `validator_miner_replay_loss_increase` gauge to see your margin.

Code: `connito/validator/evaluator.py:evaluate_one_miner_sync`,
`connito/miner/train.py:setup_training`.

## Loader refuses a model split over several GPUs

**Symptom:** the miner or validator stops at startup with

```
the base model is split over devices [...]. Load it onto one GPU: split over several GPUs (transformers device_map), Qwen gives wrong outputs.
```

**Cause:** the base model ended up on more than one device, for example
because something loaded it with transformers `device_map="auto"` on a
multi-GPU host. That is refused on purpose: split over three GPUs, Qwen
returned wrong logits with no error (loss 12.42 instead of 2.04 on a test
sentence), and two GPUs only happened to be right.

**Fix:** run on one 48 GB GPU. On a multi-GPU host, pin the process to one
GPU with `CUDA_VISIBLE_DEVICES=<n>`, and run one miner per GPU, each with its
own hotkey.

Code: `connito/shared/modeling/mycelia.py:get_base_model`.

## fla warns that Triton is not supported

**Symptom:** at startup or at the first training step:

```
Triton is not supported on current platform, roll back to CPU
```

Training then runs, but falls back to a slower path.

**Cause:** the model's Gated DeltaNet layers run on flash-linear-attention
(fla) Triton kernels, which compile a small C launcher at first use. Without
the Python headers (`Python.h`), the compile fails, and fla only warns and
falls back. A system Python without its `-dev` package is the usual cause.

**Fix:** build the environment on a Python that ships its headers. A
uv-managed Python does; `--managed-python` makes uv use it even when a system
Python 3.10 exists (on Ubuntu 22.04, `uv venv -p 3.10` alone can pick the
system one, which has no headers):

```bash
uv venv --seed --managed-python -p 3.10 .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

On a uv release without `--managed-python`, run `uv python install 3.10`
first and pass `--python-preference only-managed` instead.

On a distribution Python, install its development package (for example
`python3.10-dev`) instead. Also check the NVIDIA driver is 570 or newer.

Code: `requirements.txt` (`flash-linear-attention`).
