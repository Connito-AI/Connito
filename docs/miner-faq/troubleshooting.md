# Troubleshooting

Quick symptom → cause → fix table. Skim for your symptom, jump to the
relevant section.

| Symptom | Likely cause | Section below |
|---------|--------------|---------------|
| HTTP 403 from `cycle-api.connito.ai` when polling phase | Cloudflare WAF flagging the request | [HTTP 403 from phase API](#http-403-from-cycle-apiconnitoai-cloudflare-waf) |
| Miner trains from base every cycle | Distribute download 404 / wrong file extension | [Miner trains from base every cycle](#miner-trains-from-base-every-cycle-download-404-on-pt) |
| Training batch loss starts at ~20 instead of <1 | Base model loaded; previous training discarded | [Batch loss starts at 20](#training-batch-loss-starts-at-20-instead-of--1) |
| Miner submitted but never evaluated | No validator covers your expert group, or `no_chain_commit` | [Submitted but never evaluated](#submitted-but-never-evaluated) |
| `score_avg` flat for 5+ hours | Stuck on a single rank, or operationally failing every round | [Score didn't change for 5+ hours](#score_avg-didnt-change-for-5-hours) |
| `eval_status_label` shows `download_failed` | HF repo private, file missing at committed revision, or timeout | See `eval-status.md` |
| `eval_status_label` shows `hash_mismatch` | Re-uploaded the same revision after committing | See `eval-status.md` |
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

**What the code does:** `_get_with_retry` treats 403 as **non-retryable**
(line 88 of `connito/shared/cycle.py`) and returns `None`. `wait_till` then
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

Code: `connito/shared/cycle.py:_get_with_retry` (non-retryable status list
includes 403), `connito/shared/cycle.py:get_phase_from_api`.

## Miner trains from base every cycle (download 404 on `.pt`)

**Symptom:**

```
<Distribute> downloaded model metadata from chain: None
FileNotReadyError: No required download job
```

Followed by `(0) Setup training` re-loading `deepseek-ai/DeepSeek-V2-Lite`
from scratch, and Train logs showing initial batch loss in the high teens.

**Cause:** `_build_download_targets` hard-codes
`f"model_expgroup_{expert_group_id}.pt"` (line 51 of
`connito/shared/model.py`). If the validator's HF upload only contains the
`.safetensors` version of the shard at that revision, the `.pt` download
404s and the miner has no checkpoint to load.

**Fix (one of):**

1. **Upgrade to the latest miner code.** Confirm `_build_download_targets`
   in your fork includes `.safetensors` as a fallback. Stock validator
   upload paths in the current code upload **both** `.safetensors` and `.pt`
   (`allow_patterns=["model_expgroup_*.pt", "model_expgroup_*.safetensors"]`
   — see `connito/shared/hf_distribute.py:139-147`); a 404 indicates either
   a stale validator on the old `.safetensors`-only upload path or a
   miner-side downloader pinned to the wrong extension.
2. **Verify the HF source.** Open the validator's HF repo at the chain-committed
   revision and confirm `model_expgroup_{N}.safetensors` exists there. If
   it doesn't, the validator's upload also failed and the miner can wait
   for the next cycle's commit.

> **TODO: code in `connito/shared/model.py:_build_download_targets` only
> requests `.pt`. The validator path uploads both; the miner downloader
> should be updated to request `.safetensors` first and fall back to
> `.pt`. Until that change ships, the symptom recurs on any cycle where the
> validator drops the `.pt` upload.**

Code: `connito/shared/model.py:_build_download_targets`,
`connito/shared/hf_distribute.py:download_checkpoint_from_hf`.

## Training batch loss starts at ~20 instead of <1

**Symptom:** Miner logs show:

```
batch loss loss=21.43 inner_opt_step=0
```

at the start of Train.

**Cause:** The model was loaded fresh from `deepseek-ai/DeepSeek-V2-Lite`
instead of from a recent checkpoint. The base model has not seen the
subnet's training data and produces high initial loss.

**Why this happens:**

1. The Distribute phase failed to find a chain-committed checkpoint (no
   recent validator commit with valid signature).
2. The Distribute phase succeeded but the HF download 404'd (see
   `.pt` 404 trap above).
3. Local `config.ckpt.checkpoint_path` was wiped and `resume_from_ckpt =
   true` could not find a local fallback either.

**Fix:**

- Check the previous cycle's `<Distribute> downloaded model metadata from
  chain: ...` log line. If it says `None`, no chain commit was found — was
  the validator down last cycle? Wait one cycle and retry.
- If the download 404'd, see the previous section.
- If neither — check `expert_groups/<group_name>/config.yaml` and your
  miner config agree on `group_id`. A mismatch means
  `freeze_parameters` produces zero trainable params and the model never
  actually trains; you see a finite but high loss because base DeepSeek
  is being evaluated, not your trained version.

Code: `connito/miner/train.py:setup_training` (the `freeze_parameters` →
"No trainable parameters found" warning around line 177-184).

## Submitted but never evaluated

**Symptom:** Your miner ran MinerCommit2 successfully, but on the
dashboard you have no `val_loss` for the round and your
`eval_status_label` is stale.

**Cause (in order of frequency):**

1. **No validator is running your expert group.** Validators filter their
   roster by `expert_group == config.task.exp.group_id`. If you committed
   to `group_id = 5` and every validator is running `group_id = 0` or `1`,
   no one evaluates you.

   Check: dashboard's per-validator panel will show which `expert_group`
   each validator runs.

   Fix: switch `task.expert_group_name` to a group at least one validator
   covers.

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

## Validator status `merge` ran but my score didn't update

**Symptom:** Validator log shows `(5) Run global optimization` completing,
but `score_aggregator.json` was not updated for your UID.

**Cause:** The Merge phase aggregates *gradients*, not *scores*. Scores
were finalized earlier (Validate phase). If your score didn't update for
the round:

1. Check your UID is in `validator-1 | … round_id=… written_uids=…`. If
   it's not in `written_uids`, you weren't in scored_uids and your prior
   EMA is preserved (operational failure path).
2. If you are in `written_uids` but the value is 0.0, you ranked outside
   the top 3 or hit the duplicate-`val_loss` penalty.

Code: `connito/validator/evaluator.py:finalize_round_scores`.

## My miner uploaded but the validator log says `outdated_submissions`

**Symptom:** Validator log:

```
Skipping submission file: outdated
file_name=hotkey_…_block_8120000.pt
```

**Cause:** Your submission's `block` is outside the previous Submission
phase's block range. This usually means you uploaded too early (during the
cycle that just ended) or your local clock is drifting badly.

**Fix:** Make sure MinerCommit2 fires inside the chain phase boundary, not
on a wall-clock timer. The stock miner uses `wait_till(config,
PhaseNames.miner_commit_2)` which is chain-block driven — only a custom
miner would deviate.

Code: `connito/shared/cycle.py:gather_validation_job` (the
`in_previous_phase` filter, line 1045-1059).
