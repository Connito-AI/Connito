# Validator cycle phases

What `connito/validator/run.py` does in each phase of one cycle. The phase order is fixed by `connito/sn_owner/cycle.py::PhaseManager` and phase names live in `connito/shared/cycle.py::PhaseNames`. The validator advances by blocking on `wait_till(config, <phase_name>)` and only acts at phase boundaries.

`run.py:NNN` references point at the relevant lines in `connito/validator/run.py`.

## Phase order (one cycle)

```
Distribute → Train → MinerCommit1 → MinerCommit2 → Submission
          → Validate → Merge → ValidatorCommit1 → ValidatorCommit2
```

The validator's main loop is a single `while True:` (`run.py:657`) inside a `try:` block (`run.py:656-1091`) that walks through the phases below. Each cycle runs through all of them in order; on `KeyboardInterrupt` (e.g. Watchtower SIGTERM) the loop exits cleanly between phases, so no chain commit is lost mid-cycle. The phase-1 streaming evaluation in particular runs to the phase boundary, so a SIGTERM during Submission stops at the next `await` rather than mid-upload.

## Per-phase logic

### Distribute / Train / MinerCommit2

**Validator is mostly passive — these belong to the miners** (download checkpoint → train → commit hashes). Within the validator's main loop, where the validator actually is during each phase depends on its progress through the previous cycle's tail:

- **Distribute** (start of cycle): the validator is typically finishing the previous cycle's tail steps 9-11 below (`submit_weights` → archive/prune submissions → metric log), then evaluating the `(0) Re-syncing model from peer` branch if `_participated_in_merge` was False last cycle.
- **Train**: validator is blocked in `wait_till(PhaseNames.miner_commit_1)` (`run.py:686`).
- **MinerCommit2**: validator is blocked in `wait_till(PhaseNames.submission)` (`run.py:719`) — its own MinerCommit1 work has already completed by this point.

### MinerCommit1 — `wait_till(PhaseNames.miner_commit_1)` (`run.py:686`)

Top-of-loop housekeeping and the validator's own seed commit:

- **(0) Peer resync if excluded last cycle.** If `_participated_in_merge` was `False` at the end of the previous cycle (no valid gradient → skipped allreduce), pull a fresh model from a peer validator via `reload_model_inplace`. (`run.py:663-683`)
- **Set `global_opt_step = phase_response.phase_start_block`.** This is the version stamp used by checkpoints and chain commits this cycle.
- **Stale-weights fallback.** Fetch the metagraph for this cycle, find the validator's UID, and if `block - last_update > cycle_length`, call `_submit_fallback_weights(...)` so the validator does not get deregistered for going dark. (`run.py:693-700`)
- **`commit_status(ValidatorChainCommit(model_hash=current_model_hash, global_ver=global_opt_step, expert_group=...))`** (`run.py:702-711`). Re-publishes the model hash committed at the end of last cycle (set in step 7 below) so miners and peer validators see a fresh chain heartbeat for this cycle. The `(0) Commit new seed for next validation` log line is metaphorical — `ValidatorChainCommit` has a `miner_seed` field (`shared/chain.py:55`) but it is not populated here; the actual evaluation seed is derived later via `get_combined_validator_seed(config, subtensor)` at submission time. On the very first cycle after a restart, `current_model_hash` is `None` (set at `run.py:651`).
- **`check_phase_expired`** before moving on — if we somehow drifted past the phase boundary, error out rather than perform misaligned work.

### Submission — `wait_till(PhaseNames.submission)` (`run.py:719`)

Streaming miner evaluation runs **for the entire Submission window**:

- **(1) `stream_gather_and_evaluate`** (`run.py:735-748`). Starts as soon as miners can submit, evaluates one miner at a time, and stops at `phase_response.phase_end_block`. Slow uploads still get scored if they land before the boundary.
- The seed used for evaluation is `get_combined_validator_seed(config, subtensor)` — same seed for all validators, computed from chain state, so scoring is reproducible across validators.
- The miner→validator assignment (`get_validator_miner_assignment`) is computed **once at submission start** and reused later for the missed-submission penalty pass. Recomputing later in the cycle would query a different block and could yield a different assignment.
- Memory: `cleanup(global_model)` runs before evaluation begins to release transient allocator state from the previous cycle.

### Validate — `wait_till(PhaseNames.validate)` (`run.py:750`)

Score aggregation and local gradient merge:

- **(2) Penalize missing miners.** For every hotkey in this validator's miner assignment that did not submit, call `score_aggregator.add_score(uid, hotkey, score=0.0)`. Reuses the assignment captured at submission start. (`run.py:758-771`)
- **Log scores** (latest-this-round + rolling avg).
- **Persist `score_aggregator.json`** to `config.ckpt.checkpoint_path` so a restart mid-cycle resumes with the right rolling history. (`run.py:794-799`)
- **(3) `aggregate_miner_gradient_change`** (`run.py:801-814`). Stream miner checkpoints onto the GPU one at a time, populate `global_model.grad` with a uniform-weight mean of the top-k miners (where rank is by rolling-avg score, restricted to miners with a positive *latest* score). Streaming was the fix for the largest transient RAM spike of the cycle.
- **NaN/Inf guard.** After each miner is merged, element-wise check `global_model.parameters()` for non-finite grads. If any element is non-finite, zero all grads and skip that miner. After the full merge, if `merged_uids` is empty or any grad is non-finite, set `grad_is_valid = False` — no allreduce, no optimizer step, and `_participated_in_merge` flips to `False` so next cycle re-syncs from a peer. (`run.py:817-842`)
- `cleanup(global_model)` again.

### Merge — `wait_till(PhaseNames.merge)` (`run.py:849`)

The DHT-coordinated cross-validator step. Only runs if `grad_is_valid`:

- **(4) `sync_grad_across_validators`** (`run.py:851-858`). Hivemind `DecentralizedAverager.step()` for the active expert group's grad buffer plus the `shared` group. Group formation is bounded by `ValidatorRunCfg.averager_step_timeout_sec`. Other expert groups' averagers were dropped at startup (`run.py:610-615`).
- **(5) `run_global_optimization`** (`run.py:860-872`). Outer SGD step (lr `config.opt.outer_lr`, momentum `config.opt.outer_momentum`, Nesterov) on the averaged grads. Sets `_participated_in_merge = True`.
- If `grad_is_valid` is false: **skip 4 and 5**, set `_participated_in_merge = False`, and the next cycle's phase 0 will pull a fresh model from a peer.
- **(6) Save checkpoint** to `config.ckpt.checkpoint_path / f"globalver_{global_opt_step}"` (`run.py:889-918`). Pre-save retention runs first (`delete_old_checkpoints` keeping `checkpoint_topk - 1`) so the new checkpoint is the (k)th. The `score_aggregator.json` sidecar is preserved across this prune.

### ValidatorCommit1 — `wait_till(PhaseNames.validator_commit_1)` (`run.py:929`)

Sign and announce the new model:

- **`build_local_checkpoint`** + `model_ckpt.sign_hash(wallet)`. `current_model_hash` is updated for use by next cycle's MinerCommit1.
- **(7) `commit_status(SignedModelHashChainCommit(signed_model_hash=...))`.** This is the validator's commit-reveal of the new model hash. (`run.py:929-938`)
- **HuggingFace upload** (`run.py:942-992`). Resolve the upload + chain-advertised repo ids from `config.hf`. If `get_hf_upload_readiness` says we can write, call `upload_checkpoint_to_hf(...)` and capture the returned revision SHA — this pins the exact bytes miners will pull during the next Distribute, even if `:main` advances afterward. On failure, log and continue: miners fall back to the validator HTTP `/get-checkpoint` endpoint served by `connito.validator.server`.

### ValidatorCommit2 — `wait_till(PhaseNames.validator_commit_2)` (`run.py:994`)

Reveal the model hash and post chain weights:

- **(8) `commit_status(ValidatorChainCommit(model_hash, global_ver, expert_group, hf_repo_id, hf_revision))`** (`run.py:994-1007`). The `hf_revision` is truncated to `HF_CHAIN_REVISION_LENGTH = 7` chars (capped overall by `VALIDATOR_COMMIT_MAX_HF_REPO_ID_CHARS`). `global_ver` is set to `0` if `_participated_in_merge` is False — this signals to peers that this validator did not contribute and its checkpoint should not be authoritative.
- **Post-save retention**: `delete_old_checkpoints(checkpoint_topk)`.

### After ValidatorCommit2 (runs into Distribute / Train of next cycle)

These steps are not gated by another `wait_till`; they run immediately after the ValidatorCommit2 commit and finish before the next iteration's MinerCommit1 wait:

- **(9) `submit_weights`** (`run.py:1014-1028`). Pulls `score_aggregator.uid_score_pairs(how="avg")`, normalizes, restricts to top `config.evaluation.top_k_miners_to_reward`, and submits via `submit_weights`. This is the actual TAO emissions signal.
- **(10) Archive + prune miner submissions** (`run.py:1030-1053`). If `config.ckpt.archive_submissions`, copy the top-k miner submission files into `miner_submission_archive_path` (capped at `miner_submission_archive_max_files`). Then `prune_miner_submission_files(..., max_age_cycles=0)` deletes everything else from this cycle's submission staging dir.
- **(11) Local evaluation is intentionally disabled** to reduce per-cycle RAM/compute (`run.py:1055-1057`). Only `get_status` + `metric_logger.log` runs to record the cycle's loss/timing metrics.
- `cleanup(global_model)` and back to the top of the loop.

## Cross-cutting concerns

- **`check_phase_expired(subtensor, phase_response)`** is called after long-running steps (commit, evaluation aggregation, checkpoint save) to detect drift past the phase boundary and abort the cycle rather than perform a misaligned commit.
- **`SystemStatePoller`** runs as a sidecar thread (12 s interval, started at `run.py:622-628`) and exposes Prometheus metrics for the current phase, blocks remaining, averager peer count, and VRAM. It is independent of the main loop and shuts down on `KeyboardInterrupt`.
- **`_participated_in_merge`** is the one piece of cross-cycle state that survives the loop body. It's set False whenever this validator's grad was invalid or allreduce was skipped; the next cycle's phase 0 reads it to decide whether to re-sync from a peer, and ValidatorCommit2 reads it to decide whether to advertise its new `global_ver` or `0`.
- **Shutdown paths** (`run.py:1074-1091`). Both branches stop the telemetry poller, release GPU memory, close the metric logger, and shut down every Hivemind averager.
  - `KeyboardInterrupt` (e.g. Watchtower SIGTERM) re-raises after cleanup so the process exits non-zero and the supervisor knows it was asked to stop.
  - Any other `Exception` is logged + swallowed, and rank 0 writes an emergency `mycelia_final.pt` state-dict dump in the current working directory before the process exits.
