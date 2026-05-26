# `eval_status_label` reference

Every miner has a per-validator status code on Prometheus
(`validator_miner_eval_status{miner_uid="…"}`). The numeric code is set by
the validator each time a miner is evaluated (or attempted to be evaluated)
and is **stable across rounds** — the dashboard reads `last_over_time(...)`
so a miner who is not in the current round still has a queryable status from
their last evaluation.

## Source of truth

The labels and their integer codes are **authored in the validator code**,
specifically `connito/shared/telemetry.py`. The dashboard / API mirrors them.

```python
# connito/shared/telemetry.py:EVAL_STATUS_CODES
{
    0:  "ok",
    1:  "non_finite_loss",
    2:  "statedict_parse_failed",
    3:  "signature_invalid",
    4:  "hash_mismatch",
    5:  "expert_group_or_nan",
    6:  "no_chain_commit",
    7:  "download_failed",
    8:  "oom",
    9:  "timeout",
    10: "deadline_exceeded",
    11: "rpc_error",
    99: "unknown",
}
```

`EVAL_STATUS_CODES` is a public contract: changing a code retroactively
re-interprets every old sample in Prometheus, so the gateway can safely join
on numeric value → string label. If the gateway shows a label not in this
table, it has been added in the dashboard layer and is not a validator
concern.

Code: `connito/shared/telemetry.py:EVAL_STATUS_CODES`,
`connito/shared/telemetry.py:set_miner_eval_status`.

## `ok` (code 0)

The miner's checkpoint was downloaded, validated, evaluated, and produced a
finite `val_loss`. The miner is in the round's score map. This is the only
"good" code.

**Set by:** `connito/validator/evaluator.py:evaluate_one_miner_sync`, line
803 — `set_miner_eval_status(int(uid), None)` after the `math.isfinite(val_loss)`
check passes.

**Miner action:** none. If `score_avg` is still 0, see `scoring.md` for why
a successfully-evaluated miner can still rank outside the top 3.

## `non_finite_loss` (code 1)

The miner was downloaded, validated, and the eval ran — but `val_loss` came
back as NaN or Inf. The miner's round score is 0.

**Condition in code:**
```python
# connito/validator/evaluator.py line 802-805
if math.isfinite(val_loss):
    set_miner_eval_status(int(uid), None)
else:
    set_miner_eval_status(int(uid), "non_finite_loss")
```

A non-finite `val_loss` usually means the model produced NaN logits during
the forward pass. The shard passed the `_verify_expert_group` NaN/Inf scan at
load time, so the trigger is something that happens during inference — most
likely a numerical instability in the trained weights themselves under the
validator's `fp16-mixed` autocast precision.

**Miner action:**
- Watch your local training `loss` — if it is finite but trending high (e.g.,
  starting at 20 and not dropping below 5 by the end of Train), the
  optimizer has not actually been training your experts.
- Verify your local `evaluate_model` call returns a finite `val_loss` on the
  same checkpoint before uploading.
- Confirm you have not loaded a stale `.pt` from a previous architecture.

## `statedict_parse_failed` (code 2)

The `.safetensors` or `.pt` file on disk could not be parsed into a tensor
dict. Either truncated, malformed, or an unsupported format.

**Condition in code:**
```python
# connito/validator/evaluator.py line 840-846
except (ValueError, RuntimeError, EOFError) as e:
    # ValueError: empty state_dict / unsupported format guard
    # RuntimeError, EOFError: torch.load rejecting truncated/malformed payloads
    _record_eval_failure(int(uid), "statedict_parse_failed")
```

**Miner action:**
- Verify your HF repo actually contains `model_expgroup_{N}.safetensors` (or
  `.pt`) at the revision you committed. A 0-byte file passes the HF download
  but fails parse.
- If you're using a custom upload script, run `safetensors.torch.load_file`
  locally on the file before uploading.

## `signature_invalid` (code 3)

The miner's `signed_model_hash` on chain does not verify against the
`model_hash` and the miner's hotkey. Either the miner signed with the wrong
key, or the on-disk file's hash does not match what was signed.

**Condition in code:** `ChainCheckpoint._verify_signature` returns `False`.
The validator returns `"signature"` from `validate_miner_submission`, which
maps to `signature_invalid` via `_VALIDATION_FAIL_TO_REASON`.
Code: `connito/shared/checkpoints.py:_verify_signature`,
`connito/validator/evaluator.py:validate_miner_submission`.

**Miner action:**
- Make sure `_prepare_checkpoint_for_commit` is called with the same hotkey
  that committed `MinerChainCommit` to chain.
- The signing happens in `connito/shared/checkpoints.py:ModelCheckpoint.sign_hash`.
  Stock miner code does this correctly — only a custom upload script can
  desync the signature.

## `hash_mismatch` (code 4)

The SHA-256 hash of the downloaded `state_dict` on disk does not match the
`model_hash` the miner committed in MinerCommit2.

**Condition in code:** `ChainCheckpoint._verify_hash` returns `False`. The
validator returns `"hash"` from `validate_miner_submission`.
Code: `connito/shared/checkpoints.py:_verify_hash`.

**Miner action:**
- Make sure you commit MinerCommit2 **after** the HF upload completes (stock
  miner does this; `_commit_model_hash` is the last step of
  `commit_worker`).
- If you re-upload the file between MinerCommit2 and the validator
  downloading it, the hashes will diverge. Don't push to the same HF
  revision twice.
- Verify the model hash is computed via
  `connito/shared/helper.py:get_model_hash` (stock); a custom hash function
  will silently produce mismatches.

## `expert_group_or_nan` (code 5)

Combined failure bucket — either the miner uploaded a shard containing
experts that don't belong to its assigned `expert_group`, **or** the shard
contains a NaN/Inf tensor. The two checks share a code because the
underlying `_verify_expert_group` helper folds them together; the structured
warning log at the failure site distinguishes them.

**Condition in code:** `ChainCheckpoint._verify_expert_group` returns
`False`. Either:
- A routed-expert tensor key in the state dict has an expert ID outside the
  miner's `expert_group_assignment`, or
- Any tensor in the state dict contains a non-finite value.

Code: `connito/shared/checkpoints.py:_verify_expert_group` (line 197),
`connito/validator/evaluator.py:validate_miner_submission` line 558-562.

**Miner action:**
- Verify `config.task.expert_group_name` matches the directory the miner
  trains under, and that `config.task.exp.group_id` is consistent across
  config / chain commit / uploaded filename.
- Check for NaN gradients in training logs. The miner already has a
  consecutive-NaN guard
  (`config.train.max_consecutive_non_finite_batches`, default 50) — if you
  hit that threshold, the training loop restarts; do not bypass the guard
  by uploading the partially-corrupted checkpoint.

## `no_chain_commit` (code 6)

The miner has no usable `MinerChainCommit` for the previous cycle, or the
commit is missing `hf_repo_id` / `hf_revision`. The miner never enters the
eval roster.

**Condition in code:** Two paths produce this label:

1. **Freeze-time bulk write.** `Round.freeze` builds
   `freeze_zero_uids` for every UID in the metagraph that is *not* in
   `assigned_with_valid_ckpt`. After `finalize_round_scores`, every
   freeze-zero UID is tagged `no_chain_commit`. Code:
   `connito/validator/evaluator.py:finalize_round_scores` (lines 295-310 in
   the telemetry block), `connito/validator/round.py:Round.freeze` (step 3).
2. **Per-miner eval-time check.** If the chain checkpoint is somehow missing
   when the validator tries to verify it, `validate_miner_submission`
   returns `"no_chain_commit"`. Code:
   `connito/validator/evaluator.py:validate_miner_submission` line 532-534.

**Miner action:**
- Confirm your miner ran MinerCommit2 in the previous cycle. Look for
  `<MinerCommit2> committing` in the miner logs.
- Confirm the HF upload step succeeded. If `_upload_checkpoint_to_hf_safe`
  returns `(None, None)`, the chain commit goes out without HF coords and
  the validator treats it as missing.
- Check that your config's `task.exp.group_id` matches at least one
  validator's group — a miner committing to `group_id = 5` will never be
  evaluated if no validator is running that group.

## `download_failed` (code 7)

The background download worker tried to pull
`model_expgroup_{N}.safetensors` (or `.pt`) from the miner's HF repo and
failed. Either the repo is private without the right token, the file does
not exist at the committed revision, or the network call timed out.

**Condition in code:** `BackgroundDownloadWorker._fetch` raises, the
exception is caught and `download_failed` is recorded.
Code: `connito/validator/background_download_worker.py` (search for
`download_failed`).

**Miner action:**
- Make sure your HF repo is public, or that validators have a token with
  read access. Stock validators use `HF_TOKEN`; private repos that depend on
  a validator-specific token will not work.
- Verify the file at the *exact short SHA* you committed. The validator
  uses the 7-char revision prefix from chain (`HF_CHAIN_REVISION_LENGTH`).
  If you committed `abc1234` but the file only exists at `abc1234abc...`'s
  full SHA on a *different* branch, the resolution fails.
- Default per-miner download timeout is
  `config.evaluation.per_miner_download_timeout_sec = 180`. Persistent
  >3-minute downloads will fail.

## `oom` (code 8)

The validator GPU ran out of memory while loading or evaluating the miner's
checkpoint.

**Condition in code:** `torch.cuda.OutOfMemoryError` is caught in
`evaluate_one_miner_sync` (line 833-839). The miner is marked
operationally-failed; the round score is **not** zeroed — the prior EMA is
preserved.

**Miner action:** none. This is the validator's hardware problem. If it
happens repeatedly on multiple validators, the miner's checkpoint may have a
pathologically large tensor (e.g., a parameter at full fp32 when fp16 was
expected). Inspect your `safetensors` dtypes.

## `timeout` (code 9)

The per-miner evaluation exceeded `per_miner_eval_timeout_sec` (default 300
seconds). The miner's prior EMA is preserved — no zero-score.

**Condition in code:** `asyncio.TimeoutError` raised by `asyncio.wait_for`
around `evaluate_one_miner`. Code:
`connito/validator/evaluator.py:evaluate_foreground_round` line 1103-1110.

**Miner action:** see `download_failed` (same kinds of network slowness
also cause eval-side slowness).

## `deadline_exceeded` (code 10)

The eval was bailed cleanly because the round's phase boundary was about to
expire. Distinct from `timeout` — this is the validator deliberately
stopping rather than a hard SLA breach.

**Condition in code:** `EvalDeadlineExceeded` raised by
`connito/shared/evaluate.py:evaluate_model` when `deadline_monotonic` has
passed. Code: `evaluate_one_miner_sync` line 823-828.

**Miner action:** none. The validator was being polite about the phase
boundary; nothing the miner did caused this.

## `rpc_error` (code 11)

Subtensor RPC failure during a chain-side operation related to this miner.

**Condition in code:** Generic `inc_eval_failure(..., "rpc")` from a chain
call site. Less common than the others.

**Miner action:** none — chain transport issue.

## `unknown` (code 99)

Fallback bucket for any failure reason not mapped to a specific code.
Includes the legacy `"corrupt"` / `"checksum"` reasons from older code
paths, plus any unhandled exception in `evaluate_one_miner_sync`.

**Miner action:** check your validator's logs for the `evaluate_one_miner:
failed` line with `exc_info=True`. The stack trace will name the actual
cause.

## How statuses combine across validators

Each validator emits its own gauge. A miner with status `ok` on Validator A
and `download_failed` on Validator B is *legitimately* in both states — the
validators have different network paths to HF, and their `last_evaluated`
timestamps differ. The gateway typically picks the most recent status across
all validators for the miner's UID.

If the dashboard's aggregated view shows the wrong status, the bug is in the
gateway's aggregation layer — the per-validator Prometheus gauges are
correct.
