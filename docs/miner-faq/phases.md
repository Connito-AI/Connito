# Cycle phases reference

The cycle is a state machine. Every block of every cycle is in exactly one
phase. The phase order is fixed; the phase lengths come from the owner phase
service, which every miner and validator consults.

## Source of truth for "current phase"

The owner phase service at `https://cycle-api.connito.ai/get_phase` is the
single source of truth that all workers consult. It computes the phase
deterministically from the current chain block height using the
`PhaseManager` walk:

```python
# connito/shared/cycle.py:PhaseManager.get_phase
cycle_index       = block // cycle_length
cycle_block_index = block % cycle_length
# then walk the ordered phase list, finding the phase that contains
# cycle_block_index
```

The dashboard API mirrors what the phase service returns. The chain itself
does not store a "current phase" field; it stores block heights only.

If `cycle-api.connito.ai` is unreachable, `get_phase_from_api` returns
`None` after retrying with backoff (`api_retries = 5`, `api_backoff_sec = 2`).
A miner's phase waits keep polling and resume when the API answers. A
validator waiting for the run-up to MinerCommit1 exits instead (log
`Quit training`, exit code 0) and recovers when it is restarted after the API
is back; see `troubleshooting.md`, "cycle-api unreachable: the validator
exits, the miner waits".

The `/get_phase` endpoint is served by the subnet owner's cycle-api service,
which is not part of this repository. `PhaseManager` mirrors the same block
arithmetic locally, so you can read the phase logic here.

Code: `connito/shared/cycle.py:PhaseManager`,
`connito/shared/cycle.py:get_phase_from_api`.

## Phase order and lengths

```
Distribute → Train → MinerCommit1 → MinerCommit2 → Submission
   → Validate → Merge → ValidatorCommit1 → ValidatorCommit2
```

**The lengths are served, not configured.** The owner's phase service decides
how many blocks each phase lasts, and it can change them. To see the live
lengths, call `GET https://cycle-api.connito.ai/get_phase`: the response
carries `phase_start_block`, `phase_end_block` and
`blocks_remaining_in_phase` for the current phase. A block is about 12 s.

The `config.cycle.*_period` and `cycle.cycle_length` values in your local
YAML are fallback defaults for local tooling. They are locked (reset on load)
and do not move the real phase boundaries, which are computed by the
deployed service from chain block height. Do not tune your miner against
them.

Train is by far the longest phase: it is the miner's whole training budget
for the round.

Code: `connito/shared/config.py:CycleCfg`,
`connito/shared/cycle.py:PhaseManager.init_phases`,
`connito/shared/cycle.py:wait_till`.

## Per-phase: what miners and validators do

### Distribute

**Miner:** fetch the round's published winner, if there is one.

- Read every recent `ValidatorChainCommit` from chain, keep those whose
  `expert_group` equals the active task's `group_id` and that are newer than
  what the miner already has, and pick one with a valid `signed_model_hash`.
- Download `model_expgroup_{group_id}.safetensors` from
  `validator_hf_repo_id @ validator_hf_revision` (the 7-char short SHA the
  validator committed). That file is a residual shard: the winner's residual
  experts and routers, nothing else.
- Verify it: hash against the chain commit, then the exact residual spec
  (keys, shapes, dtypes), finite values, and trained experts. A shard that
  passes becomes the miner's next starting point. A validator that published
  no winner commits no HF coordinates, and the miner keeps what it has.
- The miner also checks the owner API for the active task here; a task switch
  takes effect from this phase.

The frozen base model (`Qwen/Qwen3.6-35B-A3B` at its pinned revision) is
loaded from the local Hugging Face cache at miner startup; only the residual
shard comes from chain.

**Validator:** mostly idle. Its background eval worker may still be scoring
the previous round's miners.

Code: `connito/miner/model_io.py:download_worker`,
`connito/shared/model.py:fetch_model_from_chain_validator`.

### Train

**Miner:** run AdamW steps on the residual experts and their routers only;
every base weight is frozen. The loss is next-token cross-entropy over every
labelled token (task rows and c4 replay rows alike), computed in chunks over
the LM head. Each `inner_opt_step` runs `local_par.gradient_accumulation_steps`
micro-batches of `local_par.micro_batch_size` rows (defaults 2 × 2), then one
optimizer step.

The miner does **not** receive validator feedback during Train. It trains from
its starting point (the published winner, its own last checkpoint, or its own
fresh draw) on the task's data sources, as listed in the task's `config.yaml`.
Vary the data you train on every round; see `miner-config.md`, "Vary your
training data every round".

**Validator:** the background eval worker keeps scoring the previous round's
miners. At the start of Train the validator checks the owner API for a new
active task and switches in place if there is one.

Code: `connito/miner/train.py:train_worker`,
`connito/shared/evaluate.py:chunked_lm_loss`.

### MinerCommit1

**Miner:** sign the freshly trained shard and commit `signed_model_hash` to
chain. No HF upload yet — only the signature.

The signing message is the model hash; the signing key is the miner's
hotkey. This is what validators later verify against
`ChainCheckpoint._verify_signature`.

**Validator:** 5 blocks before this phase it finalizes the previous round:
rank-based scores, the round winner published to HF (in a background
thread), and chain weights (or stale-weights fallback weights if its weights
are too old). At the phase start it commits a fresh `ValidatorChainCommit`
heartbeat with a new miner seed and prunes old submissions.

Code: `connito/miner/model_io.py:_commit_signed_model_hash`,
`connito/validator/run.py` (the MinerCommit1 block of the main loop),
`connito/validator/evaluator.py:finalize_round_scores`,
`connito/validator/distribute.py:publish_round_baseline`.

### MinerCommit2

**Miner:** HF upload happens *first*, then chain commit. Specifically:

1. Resolve `hf_upload_repo_id` from `config.hf.checkpoint_repo`.
2. Upload `model_expgroup_{group_id}.safetensors` (the residual shard, about
   252 MB at K = 1). Returns a short SHA (`hf_revision`).
3. Commit `MinerChainCommit(model_hash, global_ver, expert_group,
   hf_repo_id, hf_revision)` to chain.

If the HF upload fails, the chain commit goes out **without** HF coords and
the miner is missing for the round (validators cannot find the shard).

**Validator:** still idle. Blocked in `wait_till(PhaseNames.submission)`.

Code: `connito/miner/model_io.py:_upload_checkpoint_to_hf_safe`,
`connito/miner/model_io.py:_commit_model_hash`.

### Submission

**Validator:** the busy phase. `Round.freeze` runs at the start and builds the
eval roster (validation groups, foreground/background split). The download
worker pulls each assigned miner's shard, and the eval worker scores them one
at a time on the GPU:

1. Once per round, it loads its own fresh residual (equal to the pretrained
   base) and measures the **baseline losses**: the task eval split and the c4
   replay split.
2. For each miner, it verifies the shard (signature, exact residual spec,
   hash), loads it onto the base, and measures the task loss, the c4 loss and
   the residual gate's usage on both, on the same batches.
3. A miner whose c4 loss rises more than the tolerance above the base's
   scores `−inf` (status 15). Otherwise its raw score is the task gain
   (`baseline_loss − val_loss`) minus the c4-rise and gate-usage penalties,
   and its score is that minus the base's own raw score, so the base scores
   0.

**Miner:** idle — the miner's work for the round is already done. The training
loop may keep running, but its output won't be used until the next cycle's
MinerCommit2.

Code: `connito/validator/round.py:Round.freeze`,
`connito/validator/background_eval_worker.py:BackgroundEvalWorker`,
`connito/validator/evaluator.py:evaluate_one_miner_sync`.

### Validate

**Validator:** frees GPU memory held by the main loop and persists the score
history (`score_aggregator.json`). Scoring of the round's miners continues in
the background worker, which deletes the submissions it no longer needs; the
round is finalized at the next MinerCommit1.

**Miner:** idle.

Code: `connito/validator/run.py` (the Validate block),
`connito/validator/background_eval_worker.py:BackgroundEvalWorker`.

### Merge

**Validator:** nothing is merged. Each round's winner is published as a whole
residual shard, and every validator scores every round against the pretrained
base. The phase stays because the owner's phase service owns its boundaries
and miners read the same schedule; validators do no work in it, which frees
the window for the background workers.

**Miner:** idle.

Code: `connito/validator/run.py` (the Merge block and its comment).

### ValidatorCommit1

**Validator:**
1. Build a `ModelCheckpoint` for the round winner this validator published
   (or, with no winner, for a hash that every validator of the task computes
   the same way) and sign its hash with the validator's hotkey.
2. Commit `SignedModelHashChainCommit(signed_model_hash)` to chain.

**Miner:** idle.

Code: `connito/validator/run.py` (the ValidatorCommit1 block),
`connito/validator/distribute.py:unadvertised_model_hash`.

### ValidatorCommit2

**Validator:** reveal the model hash on chain and pin the HF coords.

`commit_status(ValidatorChainCommit(model_hash, global_ver, expert_group,
hf_repo_id, hf_revision))`. With a published winner, `hf_repo_id` and
`hf_revision` point at it and miners fetch it in the next Distribute. With
no winner, both are empty and miners have nothing to fetch.

**Miner:** idle.

Code: `connito/validator/run.py` (the ValidatorCommit2 block).

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

**Validator.** The round in progress is journaled to disk (scores, the
roster, and whether the round is on the fallback baseline), so a validator
that restarts mid-round resumes it instead of starting over. A round resumed
on the fallback baseline still publishes no winner. The score history lives in
`score_aggregator.json`.

**Miner.** The miner's state across cycles is its local checkpoints and the
last published winner it downloaded. On restart it resumes from the newest
local checkpoint that passes the residual spec, or keeps a fresh draw if none
does.

Code: `connito/validator/round_journal.py`,
`connito/validator/aggregator.py:MinerScoreAggregator`,
`connito/shared/model.py:get_model_from_checkpoint`.
