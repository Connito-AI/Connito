# Cycle phases reference

The cycle is a state machine. Every block of every cycle is in exactly one
phase. Phase order is fixed and lengths are configured in
`config.cycle.*_period` (the same fields are used by miners, validators, and
the owner phase service, so they must agree).

## Source of truth for "current phase"

The owner phase service at `https://cycle-api.connito.ai/get_phase` is the
single source of truth that all workers consult. It computes the phase
deterministically from the current chain block height using the
`PhaseManager` walk:

```python
# connito/sn_owner/cycle.py:PhaseManager.get_phase
cycle_index       = block // cycle_length
cycle_block_index = block % cycle_length
# then walk the ordered phase list, finding the phase that contains
# cycle_block_index
```

The dashboard API mirrors what the phase service returns. The chain itself
does not store a "current phase" field; it stores block heights only.

If `cycle-api.connito.ai` is unreachable, `get_phase_from_api` returns
`None` and `wait_till` cannot make progress — both miners and validators
will idle until the API comes back. Both workers will keep retrying with
backoff (`api_retries = 5`, `api_backoff_sec = 2`).

Code: `connito/sn_owner/cycle.py:PhaseManager`,
`connito/shared/cycle.py:get_phase_from_api`,
`connito/sn_owner/phase_service.py:read_phase`.

## Phase order and default lengths

```
Distribute (20)  → Train (300)  → MinerCommit1 (10)  → MinerCommit2 (10)
   → Submission (80)  → Validate (10)  → Merge (50)
   → ValidatorCommit1 (10)  → ValidatorCommit2 (10)

Total = 500 blocks ≈ 100 minutes at 12 s/block
```

(`cycle.cycle_length: 448` in the locked defaults; the per-phase periods in
the latest code sum to 500. The validator's `PhaseManager` recomputes
`cycle_length` from the sum of `phase["length"]` at startup, so the actual
cycle is 500 blocks. The locked `cycle_length: 448` value is a historical
default that is no longer authoritative.)

> If the owner's deployed lengths differ from what the miner's config has,
> the deployed values win. Phase boundaries are computed against the
> deployed `PhaseManager`; the miner's local `config.cycle.*_period` only
> affects `wait_till`'s sleep math, not the chain-block timing.

Code: `connito/shared/config.py:CycleCfg`,
`connito/sn_owner/cycle.py:PhaseManager.init_phases`.

## Per-phase: what miners and validators do

### Distribute (20 blocks ≈ 4 min)

**Miner:** download the latest global checkpoint.

- Pull every recent `ValidatorChainCommit` from chain, filter to
  `expert_group == config.task.exp.group_id`, and pick one with a valid
  `signed_model_hash`.
- Download `model_expgroup_{N}.safetensors` from
  `validator_hf_repo_id @ validator_hf_revision` (revision is the 7-char
  short SHA the validator committed).
- Reconstruct the backbone from `deepseek-ai/DeepSeek-V2-Lite`
  (`from_pretrained`); only the expert shard came from chain.

**Validator:** mostly idle. May be finishing the previous cycle's tail
(weight submission, archive prune) or doing a peer-resync if
`_participated_in_merge` was False last cycle.

Code: `connito/miner/model_io.py:download_worker`,
`connito/shared/model.py:fetch_model_from_chain_validator`.

### Train (300 blocks ≈ 60 min)

**Miner:** run AdamW inner-optimizer steps on the miner's expert-group
parameters. The training loop is in `connito/miner/train.py:train_worker`.
Each `inner_opt_step` runs `gradient_accumulation_steps` forward+backward
passes, then one optimizer step.

The miner does **not** receive validator feedback during Train; it trains on
the most recent global checkpoint from Distribute against its expert-group
dataset (`expert_groups/{name}/dataset.py`).

**Validator:** blocked in `wait_till(PhaseNames.miner_commit_1)`. Idle from
the validator's perspective.

Code: `connito/miner/train.py:train_worker`, the main `for step, batch in
enumerate(train_dataloader)` loop (~line 321).

### MinerCommit1 (10 blocks ≈ 2 min)

**Miner:** sign the freshly-trained checkpoint and commit
`signed_model_hash` to chain. No HF upload yet — only the signature.

The signing message is the model hash; the signing key is the miner's
hotkey. This is what validators later verify against
`ChainCheckpoint._verify_signature`.

**Validator:** top-of-loop housekeeping. Stale-weights fallback if needed,
re-publish `ValidatorChainCommit(model_hash, global_ver, expert_group)`
with last cycle's model hash as a chain heartbeat.

Code: `connito/miner/model_io.py:_commit_signed_model_hash`,
`connito/validator/run.py` (~line 686).

### MinerCommit2 (10 blocks ≈ 2 min)

**Miner:** HF upload happens *first*, then chain commit. Specifically:

1. Resolve `hf_upload_repo_id` from `config.hf.checkpoint_repo`.
2. Call `upload_checkpoint_to_hf` to push the expert-group shard. Returns a
   short SHA (`hf_revision`).
3. Commit `MinerChainCommit(model_hash, global_ver, expert_group,
   hf_repo_id, hf_revision)` to chain.

If the HF upload fails, the chain commit goes out **without** HF coords and
the miner is missing for the round (validators cannot find the checkpoint).

**Validator:** still idle. Blocked in `wait_till(PhaseNames.submission)`.

Code: `connito/miner/model_io.py:_upload_checkpoint_to_hf_safe`,
`connito/miner/model_io.py:_commit_model_hash`.

### Submission (80 blocks ≈ 16 min)

**Validator:** the busy phase. `Round.freeze` runs at the start (builds the
eval roster — validation groups, foreground/background split, model
snapshot to CPU). `stream_gather_and_evaluate` then evaluates the assigned
top-N miners one at a time, computing `val_loss` against the round's
baseline.

The miner→validator assignment is computed once at the start of Submission
and reused for the missed-submission penalty pass in Validate.

**Miner:** idle — the miner's work for the cycle is already done. The
miner's training loop continues, but its outputs won't be used until the
next cycle's MinerCommit2.

Code: `connito/validator/run.py` (~line 719),
`connito/validator/round.py:Round.freeze`,
`connito/validator/evaluator.py:evaluate_foreground_round`.

### Validate (10 blocks ≈ 2 min)

**Validator:**
1. **Missed-submission penalty pass.** Every hotkey in this validator's
   assignment that did not submit gets `score = 0` for the round.
2. **Finalize scores.** `finalize_round_scores` overwrites the per-round
   raw `delta ** 1.2` values with the rank-based mapping (top-3 only).
3. **Persist `score_aggregator.json`.**
4. **Aggregate miner gradients.** Stream the top-K miners' checkpoints
   into a single combined gradient on `global_model.grad`. NaN/Inf-guarded.

**Miner:** idle.

Code: `connito/validator/run.py` (~line 750-848),
`connito/validator/evaluator.py:finalize_round_scores`.

### Merge (50 blocks ≈ 10 min)

**Validator:** the DHT-coordinated cross-validator step. Hivemind
`DecentralizedAverager.step()` averages each validator's `global_model.grad`
buffer for the active expert group and the `shared` group. Then the outer
SGD step runs (`outer_lr = 0.7`, `outer_momentum = 0.9`, Nesterov).

If `grad_is_valid` was False after Validate (no usable miner contributions
or NaN/Inf in the merged grad), Merge is **skipped** and the validator
flags itself for peer-resync next cycle.

A new checkpoint is saved to disk at
`config.ckpt.checkpoint_path / globalver_{global_opt_step}`.

**Miner:** idle.

Code: `connito/validator/run.py` (~line 849-918).

### ValidatorCommit1 (10 blocks ≈ 2 min)

**Validator:**
1. Build a `ModelCheckpoint`, sign its hash with the validator's hotkey.
2. Commit `SignedModelHashChainCommit(signed_model_hash)` to chain.
3. **Upload to HF.** `upload_checkpoint_to_hf` pushes the global checkpoint
   to the validator's HF repo and captures the revision SHA. This is the
   bytes miners pull during the next cycle's Distribute.

**Miner:** idle.

Code: `connito/validator/run.py` (~line 929-992).

### ValidatorCommit2 (10 blocks ≈ 2 min)

**Validator:** reveal the model hash on chain and pin the HF coords.

`commit_status(ValidatorChainCommit(model_hash, global_ver, expert_group,
hf_repo_id, hf_revision))` — `global_ver` is `0` if `_participated_in_merge`
is False, signaling to peers that this validator's checkpoint should not be
authoritative.

After this phase, the cycle's *visible* work is complete. The validator's
post-cycle tail (`submit_weights` to chain, archive/prune submissions,
metric log) runs immediately after ValidatorCommit2 but is not gated by a
separate phase — it has to finish before the next cycle's MinerCommit1.

**Miner:** idle.

Code: `connito/validator/run.py` (~line 994-1057),
`connito/validator/run.py:submit_weights` call site (~line 1014-1028).

## How a miner knows what phase they're in

Three signals are available; in order of preference:

1. **The phase API.** `GET https://cycle-api.connito.ai/get_phase` returns a
   `PhaseResponse` with `phase_name`, `phase_start_block`, `phase_end_block`,
   `blocks_into_phase`, `blocks_remaining_in_phase`. This is what
   `wait_till` consults.
2. **The validator log line.** `log_phase("<PhaseName> reached ...")` is
   emitted every time `wait_till` returns, with the chain block at return
   time. The validator's structured logs include the phase name on every
   step.
3. **The dashboard.** Mirrors the API in (1).

Do not try to compute the phase from the block number alone — the cycle
length and per-phase lengths are operationally tunable and the
authoritative computation lives in the deployed `PhaseManager`.

## Cross-cycle state

Between cycles, only one piece of state survives the validator main loop:
`_participated_in_merge`. It tracks whether the validator successfully
contributed to last cycle's Hivemind allreduce. If False, next cycle's
MinerCommit1 phase will pull a fresh model from a peer instead of using the
local one.

Miners have no equivalent cross-cycle state — each cycle is independent.

Code: `connito/validator/run.py` (search `_participated_in_merge`).
