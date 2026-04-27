# Validator subtensor topology

Reference for how the validator stack uses `bittensor.Subtensor` and `bittensor.AsyncSubtensor` connections — which threads own which instances, which calls need an archive node, and the rules a new caller has to follow to avoid the recurring `cannot call recv while another thread is already running recv` failure mode.

## TL;DR

- **Two networks**: `config.chain.network` (archive, defaults to `"archive"`) and `config.chain.lite_network` (defaults to `"finney"`). Archive holds historical state and serves `block=N` queries; lite is a public head-only node.
- **Two main-thread instances**: `subtensor` (archive) and `lite_subtensor` (lite). Anything that hits `block=<int>` must use `subtensor`; everything else uses `lite_subtensor`.
- **One Subtensor object per thread, no sharing.** The `substrate-interface` WebSocket is not thread-safe — concurrent `recv()` from two threads raises `websockets.exceptions.ConcurrencyError`. Every long-lived thread (chain submitter, bg-download, bg-eval, telemetry poller) opens its own connection.
- **Non-blocking writes go through `ChainSubmitter`** (`connito/validator/chain_submitter.py`). It owns its own `AsyncSubtensor` on a private `AsyncRunner` thread so `set_commitment` and `set_weights` never block the main loop and never share the WS with sync callers.

## Networks

Defined in `connito/shared/config.py:160-161`:

| Field | Default | Used for |
|---|---|---|
| `config.chain.network` | `archive` | Historical `block=N` queries; **must** be an archive node |
| `config.chain.lite_network` | `finney` | Current-head reads, commits, weight submissions |

If `lite_network` is unset or equals `network`, `setup_chain_worker` collapses both into the same `Subtensor` instance (`connito/shared/chain.py:336-345`). In production they should differ — `finney` is pruned, so calling a historical `block=N` op on it will raise *"State discarded for 0x… too old, use an archive node"* and fall back to head, silently changing the data the caller sees (`connito/shared/chain.py:217-229`).

## Instances by component

| Component | Instance | Network | Owning thread | File:line |
|---|---|---|---|---|
| Validator main loop | `subtensor` (archive) | `chain.network` | Main | `run.py:566` |
| Validator main loop | `lite_subtensor` (sync) | `chain.lite_network` | Main | `run.py:566` |
| `ChainSubmitter` | `_async_subtensor` (`AsyncSubtensor`) | `chain.lite_network` | Private `AsyncRunner` | `chain_submitter.py:60-62` |
| `BackgroundDownloadWorker` | `_subtensor` | `chain.network` (archive) | Worker thread | `background_download_worker.py:70-72` |
| `_eval_model_factory` (BackgroundEvalWorker) | `eval_subtensor` (archive) | `chain.network` | Worker thread (created via `asyncio.to_thread`) | `run.py:683` |
| `SystemStatePoller` | `_local_subtensor` | mirrors caller | Poller thread (lazy on first poll) | `telemetry.py:214-216` |
| Wallet UID resolution | one-shot `Subtensor` (discarded) | `chain.network` | Init thread | `config.py:528-532` |
| Miner `model_io.py` | per-worker `Subtensor` (×2) | `chain.network` | download/commit worker threads | `model_io.py:131,344` |

There are intentionally no module-level singletons. Every long-running thread that needs chain access constructs its own connection at startup; one-shot helpers (e.g. config UID lookup) build and discard.

## Head-only vs archive-required operations

The tax is: a bittensor sync `Subtensor` will happily route a historical `block=N` request to a pruned `finney` endpoint, log "State discarded", and silently fall back to head. To callers, it looks like the data just changed cycle-to-cycle.

**Archive-required** (issues `block=<int>` internally):

| Operation | Origin | Used by |
|---|---|---|
| `subtensor.get_all_commitments(netuid, block=<int>)` | `chain.py:212` | `get_chain_commits()` whenever a historical block is passed |
| `subtensor.metagraph(netuid, block=<int>)` | `chain.py:215` | `get_chain_commits()` |
| `build_chain_checkpoints_from_previous_phase` | `checkpoints.py:743-747` | `Round.freeze`, `setup_training`/`load_model` (resume), `reload_model_inplace` (peer sync), `evaluate_foreground_round` (`resolve_miner_hf_target`) |

In `run.py` the archive `subtensor` is therefore reserved for exactly these callers (`run.py:592, 728, 826, 845`). Everything else uses `lite_subtensor`.

**Head-only** (any node, archive or lite):

| Operation | Used by |
|---|---|
| `subtensor.block` / `get_current_block()` | phase-expiry checks, idle/stale logging, weight-age checks |
| `subtensor.metagraph(netuid)` (no `block`) | weight-stale check, validator-miner assignment, `connect_with_peers` discovery |
| `subtensor.neuron_for_uid(uid, netuid)` | fallback weight reconstruction |

**Mutations** (head only, but must not race other writes on the same WS):

| Operation | Path |
|---|---|
| `set_commitment(...)` | `chain_submitter.async_commit(...)` → `acommit_status` → `await async_subtensor.set_commitment(...)` |
| `set_weights(...)` | `chain_submitter.async_submit_weight(...)` / `async_submit_fallback_weights()` → `submit_weights_async` / `_asubmit_fallback_weights` |

The sync paths (`commit_status`, `submit_weights`, `_submit_fallback_weights`) still exist for miners and tests, and serialize through the global `_subtensor_lock` (`chain.py:35,632`). Validator code should never call them directly; it goes through `ChainSubmitter`.

### Suggested simplification (not yet applied)

The current call chain has three layers per write:
`ChainSubmitter` method → free coroutine in `chain.py` (`acommit_status` / `submit_weights_async` / `_asubmit_fallback_weights`) → `bittensor.AsyncSubtensor` RPC.

The middle layer is a leftover from when `acommit_status` etc. were called directly from `run.py` via `async_runner.submit(...)`. Now that every validator write goes through `ChainSubmitter`, those free coroutines have a single caller and no other reason to exist as separate functions.

Options to flatten:

1. **Inline the free coroutines into `ChainSubmitter` methods.** `acommit_status`, `submit_weights_async` (validator-side use only), and `_asubmit_fallback_weights` collapse into private methods on `ChainSubmitter`, eliminating one hop and the cross-module import. The sync `commit_status` / `submit_weights` / `_submit_fallback_weights` stay in `chain.py` for miners and tests.
2. **Keep them in `chain.py` but make them `ChainSubmitter`-private by convention** (rename with a leading underscore, mark as not-for-callers). Cheaper change, doesn't move logic, but keeps the indirection.
3. **Merge `submit_weights_async` and `_asubmit_fallback_weights`** into one method: callers pass a "fallback ok if no valid weights" flag instead of choosing between two entry points. The current split exists because the fallback path needs a different on-chain probe before it picks miners; that probe could be a private helper.

Option 1 is the cleanest endpoint. Option 2 is the lowest-risk intermediate step. Either should be done in a single PR scoped to writes only — leaving the read path (`get_chain_commits`, `metagraph`, etc.) untouched, since those still have multiple sync callers (miners, server, tests).

## Concurrency rules

The substrate WebSocket is not safe to use from two threads at once — calling `recv()` while another thread is mid-`recv()` raises `websockets.exceptions.ConcurrencyError`. The rules below are mechanical, not negotiable:

1. **One Subtensor per thread.** If a thread needs to query the chain, it constructs its own `Subtensor` (or you hand it a Subtensor that nothing else will touch).
2. **Never hand a Subtensor to `asyncio.to_thread(...)` while another thread might still call into it.** `_eval_model_factory` was an example: it ran on a worker thread, but originally reused the main loop's `lite_subtensor` and immediately raced it. The fix (`run.py:683`) is for the factory to open its own connection on first call.
3. **All non-blocking validator writes go through `ChainSubmitter`.** It owns one `AsyncSubtensor` and one `AsyncRunner` (a daemon thread running a private asyncio loop). Coroutines submitted via `async_commit`, `async_submit_weight`, and `async_submit_fallback_weights` are serialized on that loop, so commit and weight calls never race each other on the WS.
4. **Sync `set_weights` is protected by a process-wide lock.** `_subtensor_lock` in `chain.py:35` covers `submit_weights` (`chain.py:632`). Required because miners and tests still call the sync path; not required on the validator path because `ChainSubmitter` already serializes.
5. **`SystemStatePoller` lazy-creates its own subtensor** (`telemetry.py:214-216`). Even though the constructor accepts a `subtensor` argument, the poller never actually uses it — it reads `subtensor.network` once and opens its own connection. The argument is effectively a network selector.

## Adding a new caller

Decision tree:

1. **Does the call need a historical `block=<int>`?** Use the archive `subtensor` (validator main loop) or open a new `Subtensor(network=config.chain.network)` if you're on a different thread.
2. **Is it a write (`set_commitment`, `set_weights`)?** Don't add a new path. Use `chain_submitter.async_commit(...)` / `async_submit_weight(...)`. If the existing API doesn't fit, extend `ChainSubmitter` rather than spinning up a parallel async runner.
3. **Is it head-only and on the main thread?** Use `lite_subtensor` directly. No lock needed; nothing else races it.
4. **Is it head-only and on a worker thread you own?** Open your own `Subtensor` in the thread body (mirror `BackgroundDownloadWorker._loop` at `background_download_worker.py:70-72`). Never reuse a Subtensor that the main loop or another worker holds.
5. **Is it a one-shot in init code (e.g. resolving a UID at startup)?** Construct, use, discard. Don't keep the reference around to "save a connection."

## Known violations (de facto safe today, worth fixing)

The current code has two technical violations of the rules above. Neither is causing observed failures, but both weaken the invariant and could break under future edits.

### V1 — Background download Subtensor is bound to the executor pool, not the worker

`background_download_worker.py:70-72`:

```python
self._subtensor = await asyncio.to_thread(
    bittensor.Subtensor, network=self.config.chain.network,
)
```

Subsequent uses (`resolve_miner_hf_target`, the HF download helper, `self._subtensor.block`) also go through `await asyncio.to_thread(...)`. `asyncio.to_thread` dispatches to `loop.default_executor`, a `ThreadPoolExecutor` with multiple threads, so the Subtensor's "owning thread" is whichever pool thread happened to run a given call. Different calls may land on different threads.

It's safe today only because every Subtensor touch is `await`ed — exactly one thread accesses the WS at a time, so there's no concurrent `recv()`. The substrate library tolerates "different thread per call as long as no concurrency."

What would break it: any concurrent `to_thread` on `self._subtensor` (e.g. a `gather` over multiple miners), or a direct `self._subtensor.block` from the asyncio loop without `to_thread`.

Suggested fix: have the worker's run loop be a real OS thread that owns the Subtensor for life, and call sync substrate methods directly. Concretely, replace `asyncio.run(self._loop())` + `await asyncio.to_thread(...)` with a synchronous worker body, or pin the worker to a single-threaded executor (`loop.set_default_executor(ThreadPoolExecutor(max_workers=1))` scoped to this worker's loop). Either restores strict "one Subtensor per thread."

### V2 — `_submit_fallback_weights` calls `set_weights` without `_subtensor_lock`

`chain.py:489` (sync path) writes weights directly:

```python
result = subtensor.set_weights(
    wallet=wallet, netuid=config.chain.netuid,
    uids=miner_uids, weights=[weight] * len(miner_uids),
    ...
)
```

…and the metagraph + `neuron_for_uid` reads earlier in the function (`chain.py:383-385, 410, 433`) are also unlocked. The lock at `chain.py:632` covers only the happy path inside `submit_weights`; when `submit_weights` falls through to `_submit_fallback_weights` at `chain.py:624`, the lock is never acquired for that branch.

Safe today because validator writes go through `ChainSubmitter` (async path, separate connection) and miners never call `_submit_fallback_weights` — so nothing concurrent races it. The doc rule "sync `set_weights` is protected by a process-wide lock" is still violated by inspection.

Suggested fix: wrap the body of `_submit_fallback_weights` (the metagraph reads, neuron lookups, and `set_weights` call) in `with _subtensor_lock:`. ~3 lines. Has to release before any logging that itself touches the chain.

## Known pitfalls / past incidents

- **`finney` doesn't have archive history.** A caller that internally hits `get_chain_commits(block=…)` will log `Historical chain state unavailable on current node; retrying with latest head` and silently return current-block data. Currently affected callers all use the archive `subtensor` deliberately. If you change one to use `lite_subtensor`, expect this warning followed by mysteriously shifting commits.
- **`asyncio.to_thread` does not give you a new thread per call** — it reuses the default executor's thread pool. So a Subtensor created with `await asyncio.to_thread(bittensor.Subtensor, ...)` is bound to whichever executor thread happened to run that call, and *only* coroutines that subsequently `await asyncio.to_thread(method_call, …)` for the same Subtensor are safe. The two background workers handle this by serializing per-worker (one in-flight call at a time per worker).
- **`AsyncSubtensor.initialize()` must be driven on its target loop.** `ChainSubmitter` does this in its constructor: `self._runner.run(init())` (`chain_submitter.py:65-70`). Skipping `initialize()` works on some bittensor builds and silently no-ops on others — it's worth keeping the explicit call.
- **`metric_logger.log` failures kill the cycle.** Unrelated to subtensor itself, but the same pattern: an exception in the main loop's tail (`run.py:1164`) pushes through to the `except Exception:` handler, which signals every worker to stop. In-flight HF downloads or evals continue (uninterruptible) until either they finish or the join timeout (30s) expires; daemon threads then die when `run()` returns.

## File map

| Concern | Where to look |
|---|---|
| Connection setup, sync RPC helpers, `_subtensor_lock` | `connito/shared/chain.py` |
| Async RPC helpers (`acommit_status`, `submit_weights_async`, `_asubmit_fallback_weights`) | `connito/shared/chain.py:138-162, 505-589, 720-…` |
| `ChainSubmitter` (validator non-blocking writes) | `connito/validator/chain_submitter.py` |
| `AsyncRunner` (persistent asyncio loop) | `connito/shared/async_runner.py` |
| Validator main loop wiring | `connito/validator/run.py:556-580` (setup), per-call sites in the cycle body |
| Background workers | `connito/validator/background_download_worker.py`, `connito/validator/background_eval_worker.py` |
| Telemetry poller | `connito/shared/telemetry.py:180-245` |
