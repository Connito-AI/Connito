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
    5:  "expert_group_or_nan",     # retired; never reused
    6:  "no_chain_commit",
    7:  "download_failed",
    8:  "oom",
    9:  "timeout",
    10: "deadline_exceeded",
    11: "rpc_error",
    12: "repo_unavailable",
    13: "incomplete_expert_set",   # retired; never reused
    14: "bad_residual_shard",
    15: "replay_loss_increase",
    99: "unknown",
}
```

`EVAL_STATUS_CODES` is a public contract: changing a code retroactively
re-interprets every old sample in Prometheus, so the gateway can safely join
on numeric value → string label. Retired codes keep their number and are
never reused. If the gateway shows a label not in this table, it has been
added in the dashboard layer and is not a validator concern.

**Validation failures reach the gauge.** When a submission fails validation
before it is scored — `signature_invalid` (3), `hash_mismatch` (4),
`no_chain_commit` (6) or `bad_residual_shard` (14) — the validator records
that code on the miner's status gauge, so the dashboard shows why.

Code: `connito/shared/telemetry.py:EVAL_STATUS_CODES`,
`connito/shared/telemetry.py:set_miner_eval_status`.

## `ok` (code 0)

The miner's shard was downloaded, validated, evaluated on the task data and
on c4, passed the c4 cutoff, and produced a finite round score. The miner is
in the round's score map. This is the only "good" code.

**`ok` does not mean a positive score.** The round score is the task gain
over the pretrained base minus the c4-rise and gate-usage penalties, and it
can be negative. Only a score above 0 earns a rank reward or can be
published; shards trained with the reference training code have measured
−0.14 to −0.17.

**Set by:** `connito/validator/evaluator.py:evaluate_one_miner_sync`, after the
c4 cutoff passes and the score is finite.

**Miner action:** none for the status itself. If `score_avg` is still 0, see
`scoring.md` ("Why a miner with low `val_loss` can still have `score = 0`"
and "Gate usage: why a better loss can still score below 0"), and read your
terms on the `validator_miner_score_term` gauge.

## `non_finite_loss` (code 1)

The miner was downloaded, validated, and the eval ran — but the round score
came out NaN or Inf (for example every task batch gave a NaN loss, so
`val_loss` is `inf`). The score is set to `−inf`: the miner is not ranked
and gets rank score 0.

**Condition in code:** in `evaluate_one_miner_sync`, after the c4 cutoff:
```python
if gated or not math.isfinite(score):
    score = float("-inf")
...
if gated:                         # c4 cutoff failed
    set_miner_eval_status(int(uid), "replay_loss_increase")
elif math.isfinite(score):
    set_miner_eval_status(int(uid), None)
else:
    set_miner_eval_status(int(uid), "non_finite_loss")
```

The shard had already passed the residual check, which refuses any NaN/Inf
tensor, so the trigger is something that happens during inference: the
trained residual produces NaN or Inf activations on the eval data (validators
run the model in bf16).

**Miner action:**
- Watch your local training loss: a loss that turns NaN, or the miner's
  `Non-finite loss detected, skipping batch` warnings on most batches, means
  the residual has diverged. Lower the learning rate and restart from the
  last good checkpoint.
- Run your local eval on the same shard before uploading and confirm it is
  finite.

## `statedict_parse_failed` (code 2)

The shard passed validation but could not be loaded onto the model for
scoring. This is rare now.

**A truncated, malformed or 0-byte file is not code 2.** Every submission is
validated before it is scored, and validation reads the file with
`safe_open`: a file that cannot be read fails there ("residual shard refused:
unreadable") and reports **code 14, `bad_residual_shard`**. Code 2 is left for
a file that verified and then failed to load.

**Condition in code:** `evaluate_one_miner_sync` catches `ValueError`,
`RuntimeError` or `EOFError` while loading the verified file onto the model
and records `statedict_parse_failed`.

**Miner action:**
- Re-upload the stock `model_expgroup_{group_id}.safetensors` and commit the
  new revision.
- If you're using a custom upload script, run `safetensors.torch.load_file`
  locally on the file before uploading. Only `.safetensors` is accepted.

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

## `bad_residual_shard` (code 14)

The shard is not exactly the residual the task asks for. A submission must be
the task's residual experts and routers and nothing else, in the protocol's
shapes and dtypes, all finite, and actually trained. The validator refuses it
before scoring; the miner gets rank score 0 for the round.

**Condition in code:** `verify_residual_shard` returns `None`, checking in
this order. A file it cannot read at all (truncated, malformed, 0 bytes, not
safetensors) is refused first, logged as "residual shard refused:
unreadable".

1. **The file header must equal the spec exactly**, read before any tensor:
   the same keys, shapes and dtypes. For K residual experts per layer on all
   40 layers, the keys are
   `model.layers.{L}.mlp.residual_experts.{j}.{gate,up,down}_proj.weight`
   (bf16) and `model.layers.{L}.mlp.residual_router.weight` ([K, 2048],
   fp32). An extra key (for example old `mlp.experts.N.*` expert-group
   tensors), a missing key, a wrong shape, a bf16 router, a shard built for
   another K, or a `.pt` file all fail here, in about 1 ms.
2. **Every tensor must be finite** (no NaN or Inf).
3. **Every expert must be trained:** its `down_proj` is not all zero, and its
   gate and up projections are not both all zero. An untrained residual
   scores exactly like the base and could never learn if it were published.

Code: `connito/shared/checkpoint_helper.py:verify_residual_shard`,
`connito/shared/modeling/residual.py:residual_shapes`,
`connito/validator/evaluator.py:validate_miner_submission`.

**Miner action:**
- Upload the shard the stock miner writes (`save_residual_shard`), unmodified,
  as `model_expgroup_{group_id}.safetensors`.
- Make sure your node is on the current task: a shard for another task's K is
  off spec.
- Check your training actually ran: a shard saved before the first optimizer
  step still has `down_proj` = 0.

**Retired codes.** Code 5 `expert_group_or_nan` and code 13
`incomplete_expert_set` belonged to the old expert-group shards, which are no
longer accepted. They are never set any more and never reused, but old
dashboard samples may still show them. Their failures now report as code 14.

## `no_chain_commit` (code 6)

The miner has no usable `MinerChainCommit` for the previous cycle, or the
commit is missing `hf_repo_id` / `hf_revision`. The miner never enters the
eval roster.

**Condition in code:** Two paths produce this label:

1. **Freeze-time bulk write.** `Round.freeze` builds
   `freeze_zero_uids` for every UID in the metagraph that is *not* in
   `assigned_with_valid_ckpt`. After `finalize_round_scores`, every
   freeze-zero UID is tagged `no_chain_commit`. Code:
   `connito/validator/evaluator.py:finalize_round_scores`,
   `connito/validator/round.py:Round.freeze` (step 3).
2. **Per-miner eval-time check.** If the chain checkpoint is somehow missing
   when the validator tries to verify it, `validate_miner_submission`
   returns `"no_chain_commit"`. Code:
   `connito/validator/evaluator.py:validate_miner_submission`.

**Miner action:**
- Confirm your miner ran MinerCommit2 in the previous cycle. Look for
  `<MinerCommit2> committing` in the miner logs.
- Confirm the HF upload step succeeded. If `_upload_checkpoint_to_hf_safe`
  returns `(None, None)`, the chain commit goes out without HF coords and
  the validator treats it as missing.
- Check that your node is on the active task. A miner committing under a
  `group_id` that no validator is running is never evaluated; the task and
  its `group_id` are served by the owner API, so an up-to-date miner follows
  them automatically.

## `download_failed` (code 7)

The validator could not download `model_expgroup_{group_id}.safetensors`
from the miner's HF repo: the repo is private without the right token, the
file does not exist at the committed revision, or the network call failed.

**Condition in code:** the current validator does not write code 7 to the
status gauge. A failed download is counted on
`validator_miner_eval_failures_total{reason="rpc"}` (file not found, HF or
network error) or `{reason="timeout"}` (hard timeout), the miner is marked
operationally failed for the round (prior average preserved), and the status
gauge keeps its previous value. A file that is definitively gone and not
publicly fetchable is code 12, `repo_unavailable`, instead.
Code: `connito/validator/background_download_worker.py:BackgroundDownloadWorker`.

**Miner action:**
- Make sure your HF repo is public. Stock validators use `HF_TOKEN`; private
  repos that depend on a validator-specific token will not work.
- Verify the file at the *exact short SHA* you committed. The validator
  uses the 7-char revision prefix from chain (`HF_CHAIN_REVISION_LENGTH`).
  If you committed `abc1234` but the file only exists at `abc1234abc...`'s
  full SHA on a *different* branch, the resolution fails.
- Default per-miner download timeout is
  `config.evaluation.per_miner_download_timeout_sec = 180`. Persistent
  >3-minute downloads will fail.

## `oom` (code 8)

The validator GPU ran out of memory while loading or evaluating the miner's
shard.

**Condition in code:** `torch.cuda.OutOfMemoryError` is caught in
`evaluate_one_miner_sync`. The miner is marked operationally-failed; the
miner gets **no** rank score for the round — the prior EMA is preserved.

**Miner action:** none. This is the validator's hardware problem. A miner's
residual has a fixed size and fixed dtypes (the shard is refused otherwise),
so it cannot make the eval larger.

## `timeout` (code 9)

The shard download hit its hard timeout
(`evaluation.per_miner_download_timeout_sec`, default 180 s). The miner's
prior EMA is preserved — no zero-score.

**Condition in code:** the download worker catches `TimeoutError` and counts
`validator_miner_eval_failures_total{reason="timeout"}`; the status gauge is
not changed. An *eval* that runs past its time budget is code 10,
`deadline_exceeded`, instead.
Code: `connito/validator/background_download_worker.py:BackgroundDownloadWorker`.

**Miner action:** see `download_failed` (same kinds of network slowness
also cause eval-side slowness).

## `deadline_exceeded` (code 10)

The eval was bailed cleanly because it ran past the per-miner eval budget
(`evaluation.per_miner_eval_timeout_sec`, default 300 s). The miner's prior
EMA is preserved. A miner's eval (task eval plus c4 check) takes about 98 s on an
RTX A6000, so this points at a slow or overloaded validator.

**Condition in code:** `EvalDeadlineExceeded` raised by
`connito/shared/evaluate.py:evaluate_model` between batches when
`deadline_monotonic` has passed, caught in `evaluate_one_miner_sync`.

**Miner action:** none. The validator was being polite about the phase
boundary; nothing the miner did caused this.

## `rpc_error` (code 11)

A transport failure while fetching this miner's submission.

**Condition in code:** `inc_eval_failure(..., "rpc")` counts it on
`validator_miner_eval_failures_total{reason="rpc"}`, for example when no
candidate file could be downloaded from the miner's HF repo. The current
validator does not write code 11 to the status gauge.

**Miner action:** usually none. If it repeats on every validator, check that
your committed repo@revision holds `model_expgroup_{group_id}.safetensors` and
is public (see `download_failed`).

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

## `repo_unavailable` (code 12)

The miner's committed HF repo, revision or file is definitively gone: the repo
was deleted, made private or gated, or the revision was rewritten. Unlike
`download_failed` (a transient or validator-side failure, which preserves the
miner's prior average), this is treated as the **miner's fault** and the
miner gets rank score 0 for the round.

**Condition in code:** the background download worker gets a definitive "not
found" from Hugging Face for every candidate file (typed HF errors, not
network trouble), **and** an unauthenticated probe confirms the file is not
publicly fetchable. Both signals must agree; if the public probe can fetch
the file, the miss is treated as validator-side instead. The behaviour is
controlled by `evaluation.repo_unavailable_is_miner_fault` (default on).

Code: `connito/validator/background_download_worker.py` (search for
`repo_unavailable`), `connito/shared/config.py:EvalCfg`.

**Miner action:**
- Keep the repo public and keep the committed revision in place until the
  round has been scored. Do not delete, rename, or force-push over a revision
  you committed to chain.
- Open `https://huggingface.co/<repo>/tree/<revision>` logged out: if you
  can't see `model_expgroup_{group_id}.safetensors` there, validators can't
  either.

## `replay_loss_increase` (code 15)

The c4 cutoff stopped the miner. Besides the task eval, every validator
scores each miner on a c4 replay split (general web text, about 65,000 real
tokens per round) and compares the miner's c4 loss with the pretrained base's
on the same batches. A residual that improves the task but damages general
text fails here. The miner's score is `−inf` and its `val_loss` is reported
as `inf`: it is never ranked (rank score 0) and never published as the round
winner.

A c4 rise below the cutoff is not free: it is subtracted from the score as
`replay_penalty`, along with the gate's usage on c4 tokens.

**Condition in code:** in `evaluate_one_miner_sync`, the miner fails the
cutoff if

- `c4_loss − base_c4_loss` exceeds the locked
  `evaluation.max_replay_loss_increase` (default 0.02 nats), or the rise is
  NaN, or
- any c4 batch came back non-finite.

The validator logs `evaluate_one_miner: c4 isolation gate failed; not ranked`
with `reason` (`increase` or `non_finite_batches`), `replay_loss_increase`
and `tolerance=` (the value in force). The per-miner increase is also on the
`validator_miner_replay_loss_increase{miner_uid}` gauge.

Code: `connito/validator/evaluator.py:evaluate_one_miner_sync`,
`connito/shared/config.py:EvalCfg`.

**Miner action:**
- **Do not train on the same data every round.** Restarting from the
  published winner and training on the same stream again overfits: in a
  measured run, the c4 loss rose by +0.06 and both miners were stopped. With
  each source shuffled per (hotkey, round), the c4 loss fell instead. See
  `miner-config.md`, "Vary your training data every round".
- Keep the c4 replay rows in your training mix with their labels; they are
  what keeps the residual quiet on general text.
- Check the `validator_miner_replay_loss_increase` gauge for your UID to see
  how far over the tolerance you were.
