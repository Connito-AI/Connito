# Long functions: inventory and decomposition proposals

**Analysis only. No code in this branch is changed** — the single commit adds
this document and nothing else.

**Baseline:** `origin/staging_v2` @ `4db4d8d`.

Every function whose body exceeds **65 effective lines**, with a proposed
decomposition for each. "Effective" means executable lines: total span minus
blank lines, comment-only lines and the docstring. A 90-line function that is
40 lines of docstring is not a long function, and counting it as one trains
people to write worse docstrings.

## Contents

- [§1 How this was measured](#1-how-this-was-measured)
- [§2 The inventory](#2-the-inventory)
- [§3 Before refactoring anything, read this](#3-before-refactoring-anything-read-this)
- [§4 Detailed proposals](#4-detailed-proposals)
- [§5 The remaining functions, by pattern](#5-the-remaining-functions-by-pattern)
- [§6 Suggested sequencing](#6-suggested-sequencing)

---

## 1. How this was measured

Every `.py` file parsed with `ast`; each `FunctionDef` / `AsyncFunctionDef`
(including methods and nested functions) measured as
`end_lineno - lineno + 1`, minus blanks, comment-only lines and the docstring.

Alongside length, four structural metrics, because length alone is a poor
signal — a flat 80-line sequence of `logger.info` calls is easier to read than
a 40-line function nested five deep:

| Metric | What it means | Why it matters |
|---|---|---|
| **depth** | maximum nesting of `if`/`for`/`while`/`with`/`try` | The strongest predictor of "I cannot hold this in my head". Depth ≥ 5 is where extraction pays most. |
| **branches** | count of `If`/`For`/`While`/`Try`/`BoolOp`/`ExceptHandler` | Proxy for the number of paths a test must cover. |
| **locals** | distinct names assigned in the body | The real cost of extraction: every local shared across a proposed seam becomes a parameter or a field. This is what decides *whether a split is cheap*. |
| **args** | positional + keyword-only parameters | A function with 15 parameters already has a missing object. |

48 functions exceed the threshold. The scan is reproducible from the tree; it
is pure AST and reads no runtime state.

---

## 2. The inventory

Ordered by effective length. The top six account for more lines than the other
42 combined.

| eff | total | depth | branches | locals | args | location |
|---:|---:|---:|---:|---:|---:|---|
| 844 | 1286 | 6 | 119 | 90 | 4 | `connito/validator/run.py:748` `run` |
| 410 | 538 | 6 | 61 | 57 | 3 | `connito/miner/train.py:281` `train_worker` |
| 288 | 409 | 5 | 37 | 41 | 15 | `connito/validator/round.py:144` `Round.freeze` |
| 224 | 302 | 5 | 47 | 30 | 6 | `connito/shared/checkpoint_helper.py:99` `save_state_dict_by_expert_group` |
| 219 | 320 | 2 | 39 | 32 | 8 | `connito/shared/dataloader.py:176` `DefaultStreamingTorchDataset.get_tokenised_dataset` |
| 192 | 262 | 3 | 19 | 18 | 12 | `connito/validator/evaluator.py:990` `evaluate_foreground_round` |
| 168 | 270 | 3 | 37 | 20 | 3 | `connito/validator/evaluator.py:163` `finalize_round_scores` |
| 167 | 244 | 5 | 22 | 17 | 4 | `connito/validator/background_download_worker.py:211` `BackgroundDownloadWorker._download_one` |
| 140 | 167 | 5 | 30 | 8 | 5 | `connito/shared/checkpoints.py:354` `ChainCheckpoints.filter_checkpoints` |
| 132 | 172 | 4 | 21 | 16 | 8 | `connito/shared/model.py:314` `fetch_model_from_chain_validator` |
| 132 | 190 | 4 | 23 | 14 | 5 | `connito/shared/modeling/mycelia.py:145` `get_base_model` |
| 126 | 148 | 5 | 24 | 25 | 5 | `connito/shared/chain.py:262` `get_chain_commits` |
| 122 | 149 | 2 | 10 | 14 | 6 | `connito/shared/model.py:488` `reload_model_inplace` |
| 118 | 170 | 4 | 13 | 15 | 7 | `connito/miner/train.py:106` `setup_training` |
| 109 | 124 | 5 | 18 | 17 | 8 | `connito/shared/modeling/custom_deepseek_v2_lite.py:180` `CustomDeepseekV2Experts._load_from_state_dict` |
| 104 | 127 | 4 | 20 | 18 | 3 | `connito/shared/modeling/custom_deepseek_v2_lite.py:340` `CustomDeepseekV2Moe.__init__` |
| 100 | 149 | 2 | 10 | 7 | 4 | `connito/validator/background_eval_worker.py:381` `BackgroundEvalWorker._evaluate_one` |
| 98 | 124 | 4 | 13 | 7 | 5 | `connito/validator/run.py:583` `sync_grad_across_validators` |
| 97 | 124 | 4 | 20 | 10 | 15 | `connito/shared/checkpoint_helper.py:403` `save_checkpoint` |
| 95 | 137 | 4 | 14 | 17 | 4 | `connito/shared/cycle.py:209` `wait_till` |
| 93 | 141 | 3 | 13 | 5 | 15 | `connito/validator/evaluator.py:790` `evaluate_one_miner_sync` |
| 88 | 102 | 5 | 18 | 15 | 9 | `connito/shared/client.py:30` `download_model` |
| 87 | 104 | 2 | 3 | 10 | 13 | `connito/validator/round_groups.py:505` `maybe_advance_cohort` |
| 86 | 111 | 4 | 22 | 11 | 8 | `connito/shared/chain.py:703` `submit_weights` |
| 85 | 96 | 5 | 14 | 14 | 4 | `connito/validator/inter_validator_connection.py:215` `build_grad_buff_from_model` |
| 84 | 107 | 4 | 23 | 16 | 6 | `connito/shared/modeling/custom_deepseek_v2_lite.py:744` `_apply_pretrained_tensor_to_partial` |
| 82 | 89 | 3 | 12 | 3 | 6 | `connito/shared/checkpoints.py:627` `build_chain_checkpoints` |
| 80 | 109 | 2 | 9 | 14 | 4 | `connito/shared/cycle.py:600` `get_validator_miner_assignment` |
| 78 | 103 | 3 | 19 | 21 | 5 | `connito/shared/checkpoints.py:961` `archive_top_miner_submissions` |
| 77 | 94 | 4 | 12 | 12 | 4 | `connito/shared/cycle.py:1016` `gather_validation_job` |
| 77 | 98 | 5 | 19 | 7 | 1 | `connito/validator/background_download_worker.py:90` `BackgroundDownloadWorker._loop` |
| 76 | 105 | 5 | 11 | 9 | 6 | `connito/validator/run.py:477` `aggregate_miner_gradient_change` |
| 75 | 94 | 3 | 20 | 8 | 10 | `connito/miner/train_helper.py:119` `get_status` |
| 75 | 109 | 2 | 10 | 3 | 2 | `connito/validator/background_eval_worker.py:271` `BackgroundEvalWorker._load_round_snapshot` |
| 74 | 95 | 2 | 8 | 8 | 4 | `connito/shared/checkpoints.py:717` `build_chain_checkpoints_from_previous_phase` |
| 74 | 131 | 4 | 14 | 7 | 7 | `connito/shared/evaluate.py:28` `evaluate_model` |
| 74 | 135 | 3 | 10 | 2 | 7 | `connito/shared/hf_distribute.py:495` `download_checkpoint_from_hf_subprocess` |
| 73 | 82 | 3 | 8 | 8 | 6 | `connito/sn_owner/dht_init.py:18` `init_dht_and_peer_id` |
| 70 | 95 | 5 | 18 | 6 | 1 | `connito/validator/background_eval_worker.py:149` `BackgroundEvalWorker._loop` |
| 70 | 89 | 1 | 2 | 8 | 7 | `connito/validator/run.py:376` `setup_training` |
| 69 | 95 | 2 | 6 | 15 | 5 | `connito/shared/chain.py:520` `_submit_fallback_weights` |
| 69 | 81 | 2 | 6 | 15 | 5 | `connito/shared/chain.py:617` `_asubmit_fallback_weights` |
| 68 | 96 | 2 | 15 | 10 | 3 | `connito/shared/cycle.py:918` `hydrate_miner_submissions_from_hf` |
| 68 | 97 | 0 | 0 | 9 | 8 | `connito/validator/round_groups.py:401` `build_cohort_groups` |
| 67 | 80 | 3 | 13 | 8 | 8 | `connito/shared/chain.py:853` `submit_weights_async` |
| 67 | 79 | 1 | 1 | 9 | 2 | `connito/test/test_hf_distribution_safety.py:303` `test_hydrate_miner_submissions_from_hf_writes_assigned_miners_only` |
| 66 | 90 | 3 | 10 | 7 | 2 | `connito/shared/expert_manager.py:152` `ExpertManager.load_expert_group_assignment` |
| 66 | 88 | 2 | 7 | 18 | 0 | `connito/test/test_eval_source_skip.py:141` `main` |

Two entries deserve an immediate footnote:

- `connito/shared/client.py:30 download_model` (88 lines) is **dead code** —
  the whole file is unreachable and is deleted by the cleanup on
  `chore/dead-code-cleanup-v2`. If that lands first, this list drops to 47.
- `connito/test/test_eval_source_skip.py:141 main` and
  `test_hf_distribution_safety.py:303` are a verification script and a single
  test. Long tests are a different problem with a different cost; they are
  listed for completeness and are not analysed below.

---

## 3. Before refactoring anything, read this

This codebase has three properties that change what "safe refactor" means, and
ignoring them is how a tidy-up becomes an incident.

**Consensus is byte-sensitive.** Validators must draw identical eval batches
and produce identical scores from the same seed. `get_tokenised_dataset`,
`Round.freeze`, `filter_checkpoints` and `finalize_round_scores` all sit on
that path. A refactor that changes iteration order, dict ordering, float
accumulation order or the sequence of `.shuffle()`/`.skip()` calls will not
fail a unit test — it will silently split the fleet's weights. For these,
"behaviour-preserving" means *bit-identical*, and the refactor should be
validated by running the old and new implementations against the same seed and
diffing the materialised batches, not by reading the diff.

**The chain block height is the clock.** Nothing schedules on wall time. Any
extraction out of `run()` must keep each `wait_till` call in the same order and
must not introduce a code path where a phase boundary is missed — a validator
that misses `MinerCommit1` loses the whole cycle.

**Restart recovery depends on partial state.** `Round`, `RoundJournal`,
`CohortState` and `MinerScoreAggregator` are written so a SIGKILL mid-round can
be replayed. Extraction that changes *when* a field is assigned relative to a
persist call changes what a recovered process sees.

There is also no CI test job — `.github/workflows/docker-publish.yml` only
builds the image — so nothing catches a regression except the person running
`pytest connito/test` by hand.

### Test coverage of the candidates

Refactor safety tracks coverage closely. Modules referencing each symbol:

| Function | Test modules | Refactor risk |
|---|---|---|
| `run` | 14 | Low per-extraction — but see the phase-order caveat above |
| `Round.freeze` | 5 | Low; `test_round_freeze_groups.py` covers the overlay well |
| `finalize_round_scores` | 4 | Low; rank/tie behaviour is pinned in detail |
| `fetch_model_from_chain_validator` | 3 | Low |
| `evaluate_model` | 3 | Low |
| `save_state_dict_by_expert_group`, `get_chain_commits` | 2 | Medium |
| `get_tokenised_dataset`, `filter_checkpoints`, `_download_one`, `get_base_model`, `evaluate_foreground_round`, `reload_model_inplace`, `wait_till`, `maybe_advance_cohort`, `build_cohort_groups`, `submit_weights` | 1 | Medium–high |
| **`train_worker`** | **0** | **High — the largest uncovered function in the tree** |

`train_worker` (410 effective lines) has no test touching it at all. That is
the single most important fact in this document: it is second-largest, it owns
the miner's hot loop, and any change to it is unverifiable by the suite.

---

## 4. Detailed proposals

### 4.1 `connito/validator/run.py:748 run` — 844 lines, depth 6, 90 locals

The whole validator. One function containing process setup, worker wiring and
an infinite phase state machine.

**Existing structure.** Lines 748–1123 are setup; 1124–2033 is a single `try`
whose body is `while True:` — the cycle. Inside the loop the phase boundaries
are already explicit, because every one of them is a `wait_till` call:

| Lines | Phase segment | Size |
|---|---|---|
| 1127–1292 | pre-`MinerCommit1`: submit the *previous* round's weights | 165 |
| 1293–1346 | `MinerCommit1`: commit new seed, prune/archive submissions | 54 |
| 1348–1608 | `Submission`: build the round, run foreground eval | 261 |
| 1610–1697 | `Validate`: aggregate miner gradients, validate them | 88 |
| 1699–1844 | `Merge`: allreduce with peers, open the eval window | 146 |
| 1846–1959 | `ValidatorCommit1/2`: save, upload, commit the global model | 114 |
| 1960–2002 | end of cycle: close the download window, log metrics | 43 |

**Proposal — one function per phase, in the order they already appear.**

```
def run(config, rank, world_size):
    session = _build_session(config, rank)      # 748–1123, the setup block
    try:
        while True:
            cycle = CycleState()
            _submit_pending_round_weights(session, cycle)   # 1127–1292
            _commit_seed_and_prune(session, cycle)          # 1293–1346
            _run_submission_phase(session, cycle)           # 1348–1608
            _run_validate_phase(session, cycle)             # 1610–1697
            _run_merge_phase(session, cycle)                # 1699–1844
            _commit_global_model(session, cycle)            # 1846–1959
            _close_cycle(session, cycle)                    # 1960–2002
    except ...:                                             # 2004–2033 unchanged
```

**The obstacle, stated honestly: 90 locals.** This is why the function is still
one piece. Extraction only works if those locals get a home first, and they
divide cleanly in two:

- **Session-lifetime** (created once, live forever): `config`, `wallet`,
  `subtensor`, `lite_subtensor`, `chain_submitter`, `metric_logger`, `device`,
  `tokenizer`, `global_model`, `outer_optimizer`, `expert_manager`,
  `score_aggregator`, `dht`, `group_averagers`, `download_worker`,
  `eval_worker`, `poller`, and the concurrency primitives `merge_phase_active`,
  `eval_window_active`, `download_window_closed`, `gpu_eval_lock`, `round_ref`,
  `sync_grad_executor`. → a frozen `ValidatorSession` dataclass.
- **Cycle-lifetime** (reassigned every iteration): `phase_response`,
  `global_opt_step`, `new_round`, `pending_round`, `miner_jobs`,
  `merged_uids`, `metagraph`, `my_uid`, `current_model_hash`,
  `scheduled_round_weights`, `_participated_in_merge`, the loss accumulators.
  → a mutable `CycleState` dataclass, re-created each iteration.

Once those two objects exist, every phase function takes exactly
`(session, cycle)` and the extraction is mechanical.

**Sequence.** Do not attempt this in one commit.

1. Introduce `ValidatorSession` and have `_build_session` return it. `run()`
   still does everything; it just reads `session.x` instead of `x`. Pure
   rename, reviewable, no behaviour change.
2. Introduce `CycleState` the same way.
3. Extract the phases one at a time, smallest first — `_close_cycle` (43
   lines), then `_commit_seed_and_prune` (54). Each is its own commit.
4. `_run_submission_phase` (261 lines) is still over threshold after
   extraction and needs a second pass of its own; leave it until the pattern
   has been proven on the small ones.

**Do not** reorder `wait_till` calls, and do not move `check_phase_expired`
relative to them.

### 4.2 `connito/miner/train.py:281 train_worker` — 410 lines, depth 6, 0 tests

**Existing structure.** 294–351 is setup; 352–818 is one `try` whose body is a
single `for step, batch in enumerate(dataloader)` spanning 409 lines. The loop
body is not tangled — it is ten sequential guarded blocks, each a distinct
concern:

| Lines | Guard | Concern |
|---|---|---|
| 377–455 | `if (...)` phase gate | forward, loss, backward |
| 458–573 | `if not is_start_step and is_inner_optimizer_step` | inner optimizer step, clipping, non-finite handling |
| 576–592 | `if (...)` | step bookkeeping |
| 595–662 | `if is_inner_optimizer_step and ... % metric_interval == 0` | metric logging |
| 666–706 | `if (...)` | checkpoint save |
| 715–754 | `if is_inner_optimizer_step and ckpt.enable_peer_resync` | peer resync |
| 757–763 | `if is_inner_optimizer_step` | step counter |

**Proposal.** Each guarded block becomes a named function taking a
`TrainStepContext`; the loop body becomes seven calls. The guards stay in the
loop so the control flow remains visible:

```
for step, batch in enumerate(train_dataloader):
    ctx = _step_context(session, step)
    if ctx.should_train:        _forward_backward(session, ctx)
    if ctx.is_optimizer_step:   _inner_optimizer_step(session, ctx)
    if ctx.should_log:          _log_step_metrics(session, ctx)
    if ctx.should_checkpoint:   _save_checkpoint(session, ctx)
    if ctx.should_resync:       _peer_resync(session, ctx)
```

**Prerequisite, not optional: write tests first.** With zero coverage, every
extraction here is unverifiable. Before touching it, add tests for at least the
non-finite-gradient path (`consecutive_non_finite_batches` /
`max_consecutive_non_finite_batches`) and the optimizer-step boundary
(`gradient_accumulation_steps`), which is where an off-by-one would be both
easy to introduce and invisible.

**Free win available while in here.** Lines 443–451 compute `grad_total` and
`sample_grads` and never read either — a full `sum_model_gradients` walk plus
up to ten `.norm().item()` GPU syncs, on every step. It is the only dead code
in the repo with a measurable cost. Delete it as its own commit *before* the
refactor, so the benchmark that justifies it is uncontaminated.

### 4.3 `connito/validator/round.py:144 Round.freeze` — 288 lines, 15 args

A classmethod with fifteen parameters. The parameter count is the finding: at
fifteen, the call site is unreadable and the object it wants is obvious.

**Existing structure.**

| Lines | Concern |
|---|---|
| 183–217 | read chain commits, seed, validator↔miner assignment |
| 218–244 | resolve round id; compute freeze-zero UIDs |
| 245–320 | order the background queue (prior-score prepend, stale tail) |
| 327–510 | the round-group overlay — declarations at 327–338, then **166 lines inside one `if flag_enabled:` at 339–504** |
| 511–552 | construct the `Round`, attach the journal |

**Proposal.**

1. Extract the `if flag_enabled:` body — lines **339–504** — verbatim into
   `_build_group_overlay(...) -> CohortGroups | None`. It is already fenced by a single flag and is the largest block; this
   alone takes the function from 288 to ~120 lines. `test_round_freeze_groups.py`
   covers exactly this path, so the extraction is verifiable today.
2. Extract `245–320` into `_order_background_queue(...) -> tuple[int, ...]`.
   Pure selection over already-fetched data — no I/O, so it can be unit-tested
   directly, like `round_groups.py` already is.
3. Collapse the fifteen parameters into a `FreezeInputs` dataclass. Do this
   *after* the extractions, when the true dependencies of each piece are visible.

### 4.4 `connito/shared/checkpoint_helper.py:99 save_state_dict_by_expert_group` — 224 lines

Four sequential stages with no interleaving — the easiest large win here.

| Lines | Stage | Extract to |
|---|---|---|
| 123–140 | validate args, resolve `group_ids` | `_resolve_group_ids(...)` |
| 142–196 | build expert lookup tables, detect duplicate/ambiguous assignments | `_build_expert_lookups(...) -> ExpertLookups` |
| 198–268 | route each tensor into its group | `_route_tensors_to_groups(...)` |
| 270–355 | validate the result (overlap, unassigned, empty-shard gates) | `_validate_sharding(...)` |
| 357–399 | serialise to `.safetensors`, return paths | `_write_shards(...)` |

Each stage's output is the next stage's only input, so the seams need no shared
mutable state. `test_checkpoint_sharding.py` pins the strict-sharding rejection
rules, which is precisely the `_validate_sharding` stage — extract that one
first.

### 4.5 `connito/shared/dataloader.py:176 get_tokenised_dataset` — 219 lines, 8 args

**Highest consensus risk in this document.** This builds the eval stream; if two
validators build different streams from the same seed, weight consensus breaks
every round. It has one test module.

**Existing structure.** A per-source loop (≈300–383) that resolves a shard
pick, selects columns, maps, filters and appends to `dataset_splits`; then a
sequence of stream-level transforms, each behind its own `if`: shuffle buffer
(410–418), skip (433–442), single-vs-interleave (444–450), dedup (461–462),
fraction (466–478), shard-by-rank (482–486).

**Proposal — extract, but do not reorder.** The per-source body becomes
`_prepare_source(...) -> tuple[Dataset, float]` and the loop becomes a
comprehension. The stream-level transforms become small named functions applied
in the *same order*.

The ordering is load-bearing and is not obvious from reading: `shuffle` before
`skip` before `interleave` before `fraction` before `shard` produces different
rows from any other permutation. Whoever does this should add a comment fixing
the order as a contract, and validate by materialising N batches at a fixed
seed before and after and asserting byte-equality — `test_eval_source_skip.py`
already contains the machinery to do that.

### 4.6 `connito/validator/evaluator.py:990 evaluate_foreground_round` — 192 lines, 12 args

Twelve parameters, and the body mixes three concerns: building the eval model,
looping miners with a deadline, and recording results. Suggested seams:
`_prepare_foreground_model(...)`, `_evaluate_miner_batch(...)`,
`_record_foreground_results(...)`. The deadline logic (`foreground_timeout_sec`,
computed by the caller in `run.py` at 1540) should move in with the loop rather
than being passed down — it is the only consumer.

### 4.7 `connito/shared/checkpoints.py:354 ChainCheckpoints.filter_checkpoints` — 140 lines, depth 5

Three independent gates applied in sequence, each with its own
"all checkpoints excluded by X gate" log: completeness (≈380–412), version
range (≈420–472), majority hash (≈480–520).

**Proposal.** One predicate function per gate, then:

```
for gate in (_gate_completeness, _gate_version_range, _gate_majority_hash):
    filtered, dropped = gate(filtered, ...)
    if dropped: logger.info("filter_checkpoints: excluded", gate=gate.__name__, n=dropped)
```

This also removes the repeated "everything was excluded" logging, which is
currently copy-pasted three times with slightly different wording. Note the
miner-role exemption pinned by `test_filter_checkpoints_miner_role.py` lives in
the version-range gate and must stay there.

### 4.8 `connito/validator/evaluator.py:163 finalize_round_scores` — 168 lines, 6 `try` blocks

Six `try` blocks in one function is the signal here: each is a separate failure
domain being individually defended, which is what a sequence of steps looks like
before it has been named. The concerns are rank→score mapping, failure
attribution, aggregator writes, journal finalisation and telemetry emission.

Good coverage (4 modules, including the tie-handling and recovery-replay paths),
so extraction is comparatively safe. Suggested seams: `_rank_scores(...)`,
`_apply_failure_attribution(...)`, `_persist_scores(...)`,
`_emit_round_telemetry(...)`. Keep the ordering: telemetry must be emitted after
the aggregator write, because `republish_telemetry_from_journal` depends on
that ordering for restart recovery.

---

## 5. The remaining functions, by pattern

The other 40 fall into four recurring shapes. None individually justifies a
dedicated section; all are worth fixing opportunistically when the file is open
for another reason.

**A. Setup/teardown ceremony** — `run.py:376 setup_training` (70),
`train.py:106 setup_training` (118), `dht_init.py:18 init_dht_and_peer_id` (73),
`mycelia.py:145 get_base_model` (132), `custom_deepseek_v2_lite.py:340
CustomDeepseekV2Moe.__init__` (104). Long but flat (depth 1–4) and mostly
sequential construction. Lowest priority: they read top-to-bottom and splitting
them buys little. If touched, extract the *validation* prologue rather than the
construction body.

**B. Retry/timeout wrappers** — `background_download_worker.py:211 _download_one`
(167, 5 `try`), `hf_distribute.py:495 download_checkpoint_from_hf_subprocess`
(74), `model.py:314 fetch_model_from_chain_validator` (132),
`model.py:488 reload_model_inplace` (122, 7 returns). The length is error
taxonomy, not logic. The shared fix is a small retry helper parameterised by
which exception types are retryable, terminal, or attributable to the miner —
`_download_one`'s miner-fault attribution (pinned by
`test_repo_unavailable_miner_fault.py`) is the delicate part and should be
extracted as a named classifier, not inlined into a generic helper.

**C. Worker loops** — `background_eval_worker.py:149 _loop` (70) and `:381
_evaluate_one` (100), `background_download_worker.py:90 _loop` (77),
`background_eval_worker.py:271 _load_round_snapshot` (75). These carry the
`gpu_eval_lock` invariant ("never held across an `await`, an `Event.wait`, or an
iteration boundary"). Extraction risks moving a lock acquire/release across a
seam. **Refactor these last**, and only alongside a test that asserts the lock
is unheld at iteration boundaries — `test_background_submission_validation.py`
already has `test_lock_unheld_at_iteration_boundary` to extend.

**D. Pure selection/serialisation** — `round_groups.py:401 build_cohort_groups`
(68, depth 0) and `:505 maybe_advance_cohort` (87, 13 args),
`chain.py:520 _submit_fallback_weights` (69) and its async twin `:617
_asubmit_fallback_weights` (69), `checkpoints.py:627 build_chain_checkpoints`
(82) and `:717 build_chain_checkpoints_from_previous_phase` (74). These are the
best-tested and lowest-risk. Note the two pairs are near-duplicates of each
other — `_submit_fallback_weights`/`_asubmit_fallback_weights` differ only in
`await`, and the two `build_chain_checkpoints*` share most of their body. Both
pairs are better addressed by de-duplication than by splitting: 138 and 156
lines respectively collapse to roughly half.

---

## 6. Suggested sequencing

Ordered by value per unit of risk, not by function size.

| # | Action | Why first / why not |
|---|---|---|
| 1 | Delete the dead diagnostic block at `train.py:443–451` | Not a refactor. Removes real per-step cost. Independent of everything else. |
| 2 | De-duplicate `_submit_fallback_weights` / `_asubmit_fallback_weights` and the two `build_chain_checkpoints*` | Pure win, well tested, removes ~150 lines without designing anything. |
| 3 | `save_state_dict_by_expert_group` → 5 stages (§4.4) | Cleanest seams in the tree; each stage's output feeds only the next. |
| 4 | `filter_checkpoints` → 3 gates (§4.7) | Mechanical, and collapses triplicated logging. |
| 5 | `Round.freeze`: extract the 166-line group overlay (§4.3) | Biggest single-block win; the covering test already exists. |
| 6 | Introduce `ValidatorSession` + `CycleState` in `run()` — **no extraction yet** (§4.1) | The enabling step. Reviewable as a pure rename. |
| 7 | Extract `run()` phases smallest-first | Only after 6, one phase per commit. |
| 8 | Add `train_worker` tests, then decompose (§4.2) | Tests are the prerequisite, not a follow-up. |
| — | Worker loops (pattern C) | Deliberately last — lock invariants. |
| — | `get_tokenised_dataset` (§4.5) | Only with byte-equality validation at a fixed seed. |

A reasonable stopping point is step 5. Steps 1–5 remove roughly 400 lines from
the four worst offenders using seams that already exist and tests that already
cover them, without introducing a single new abstraction. Steps 6–8 are a
larger design commitment and deserve their own discussion.

### What this document does not claim

Length is a smell, not a defect. Nothing here is a bug, and none of these
functions is wrong — several are long precisely because they handle failure
cases carefully, which is the opposite of a problem. `run()` in particular is
long because the phase machine is genuinely sequential, and a bad split would
make it *harder* to follow, not easier. Treat this as a menu, not a backlog.
