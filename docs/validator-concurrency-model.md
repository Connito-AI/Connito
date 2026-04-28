# Validator concurrency model

This document maps every thread, event loop, and shared-state access in
`connito/validator/run.py` and assesses the chance of collision (race
conditions, deadlocks, GPU contention, chain-RPC interleaving) across
them. It is meant as a quick reference for anyone touching the validator
loop or the background workers.

The branch this targets is `claude/feature/background-submission-validation`,
where the validator was split into a foreground top-N pass plus background
download and eval workers.

---

## 1. Threads and event loops

The validator process runs five long-lived concurrent components plus
a handful of short-lived helpers. Every async coroutine in this section
belongs to exactly one of the event loops listed below; conflating them
is the most common source of bugs.

| # | Thread / loop                          | Created at                              | Owns                                                              | Lifetime |
|---|----------------------------------------|------------------------------------------|-------------------------------------------------------------------|----------|
| 1 | **Main thread**                        | process start                            | `global_model` (GPU), outer optimizer, train loop, phase progression | process |
| 2 | **AsyncRunner** (daemon)               | `run.py:568`                             | persistent asyncio loop; all `lite_subtensor` (AsyncSubtensor) RPCs | process |
| 3 | **BackgroundDownloadWorker** (daemon)  | `run.py:689` (gated by config)           | own asyncio loop, own sync `bittensor.Subtensor`, HF downloads     | until `stop()` |
| 4 | **BackgroundEvalWorker** (daemon)      | `run.py:695` (gated by config)           | own asyncio loop, own GPU model (`_eval_base_model`), bg evals     | until `stop()` |
| 5 | **BackgroundWeightSubmitter** (proxy)  | `run.py:714`                             | *no thread of its own* — schedules coroutines on AsyncRunner       | process |
| 6 | **SystemStatePoller** (daemon)         | `run.py:629`                             | telemetry sidecar polling `subtensor` / `PhaseManager` every 12 s  | until `stop()` |
| 7 | Hivemind/DHT internal threads          | `connect_with_peers` / `group_averagers` | gradient allreduce during Merge                                    | process |

Two more transient event loops are created on the **main thread** via
`asyncio.run(...)`:

- `evaluate_foreground_round` at `run.py:864`
- `aggregate_miner_gradient_change` at `run.py:914`

Each call spins up a fresh loop, runs to completion, and tears it down.
That loop only lives on the main thread, so it never collides with
AsyncRunner or the bg-worker loops — they are different OS threads.

---

## 2. Shared state and how it is protected

| State                                    | Writers                                                           | Readers                                                          | Protection |
|------------------------------------------|-------------------------------------------------------------------|------------------------------------------------------------------|------------|
| `round_ref: RoundRef`                    | main (swap once per cycle)                                        | main, bg-download, bg-eval                                       | `RoundRef._lock` on swap; workers re-read `current` each iteration |
| `Round` claim sets / pools               | bg-eval, bg-download, foreground (via `evaluate_foreground_round`) | same + main                                                      | `Round._lock` (every mutator/reader) |
| `score_aggregator`                       | foreground (main), bg-eval, weight-submitter (read only)          | main (build `uid_score_pairs`)                                   | `MinerScoreAggregator._lock` (RLock); `persist_atomic` is tmpfile + `os.replace` |
| `score_path` on disk                     | main and bg-eval call `persist_atomic`                             | startup loader                                                   | atomic rename inside the aggregator's RLock |
| `global_model` (GPU)                     | main only (load, foreground eval, merge sync, optimizer step, save) | main only                                                       | implicit — bg-eval never touches it; uses its own model |
| bg-eval's `_eval_base_model` (GPU)       | bg-eval only                                                      | bg-eval only                                                     | confined to that thread |
| `lite_subtensor` (`AsyncSubtensor`)      | every chain RPC site                                              | n/a                                                              | **All coroutines must run on AsyncRunner's loop** so the single WebSocket is owned by exactly one loop. Main calls `async_runner.run/submit`; weight-submitter does the same. |
| sync `subtensor` (archive endpoint)      | main, SystemStatePoller, bg-download (its own instance)           | same                                                             | each thread holds its own `Subtensor` instance — no shared connection |
| `pending_round.weights_submitted`        | weight-submitter coroutine on AsyncRunner thread                  | main (next cycle)                                                | one writer, one reader; cycle boundary is long enough that flag is settled |

---

## 3. Synchronization primitives

Defined at `run.py:671-674`:

| Name                 | Type            | Set by                                                                                        | Cleared by                                                                | Effect |
|----------------------|-----------------|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|--------|
| `merge_phase_active` | `Event`         | main, around Merge / global-optimizer / save **and** around HF upload (`run.py:1096-1116`)    | main, in `finally`                                                        | both bg workers paused (DHT/GPU/HF) |
| `eval_window_active` | `Event`         | main, after Merge completes (`run.py:1040`)                                                   | main, at top of next cycle's MinerCommit1 (`run.py:763`)                  | bg-eval may run; cleared so step (4) sees a stable score set |
| `download_window_closed` | `Event`     | main, before `wait_till(MinerCommit1, block_offset=-20)`                                      | main, after the next `Round.freeze`                                       | bg-download paused while round-K downloads are dead weight (post-MinerCommit1(K+1)) |
| `gpu_eval_lock`      | `Lock`          | bg-eval, briefly around `load_state_dict` and `evaluate_one_miner`                            | bg-eval, in `finally`                                                     | the *worker's* contract for serializing its own GPU work; **foreground does not take it** |

bg-download intentionally does NOT pause on the foreground eval pass.
Foreground reads from `miner_submission_path`, which bg-download is
responsible for filling, so the two MUST run concurrently or foreground
never finds anything to evaluate. Foreground does not pull from HF
itself, so there is no bandwidth contention between the two.

Workers re-check the gate Events only **between iterations**, never mid-call.

---

## 4. RPC ordering on the chain

There is exactly one `AsyncSubtensor` (`lite_subtensor`) and exactly one
loop owns it (AsyncRunner). Submission ordering is therefore the order
of `async_runner.submit/run` calls from the producing thread:

- main thread: `acommit_status`, `_asubmit_fallback_weights`, `metagraph()`, `get_current_block()`
- BackgroundWeightSubmitter: `submit_weights_async`

Because they all queue onto a single loop, two RPCs can be in flight
concurrently as coroutines but **the WebSocket is multiplexed by
`bittensor`**, not by the validator. The risk surface is therefore the
SDK's own multiplexing; everything above the SDK is single-loop and
serialized in submission order.

---

## 5. Collision evaluation

For each plausible interaction, an estimated **likelihood** of a
real-world collision and the consequence if it happens.

### 5.1 GPU contention between foreground eval and bg-eval — **medium**

- Foreground does not acquire `gpu_eval_lock`. It relies on
  `eval_window_active` (set only after Merge completes) to keep bg-eval
  out — bg-eval is gated on this and on `merge_phase_active`, so it
  never runs during the Submission window when foreground is active.
- Bg-eval re-checks gates only at iteration boundaries. If the worker
  is mid-`_evaluate_one` when the window closes, the in-flight
  evaluation finishes on its own clock — bounded by
  `per_miner_eval_timeout_sec`.
- Two CUDA streams will run in parallel for at most one bg-eval miner.
  Correctness is fine (separate models on the same device); throughput
  drops for both during the overlap.
- **Mitigation if it ever bites**: have foreground also acquire
  `gpu_eval_lock` for its inner-most `evaluate_one_miner` call, the same
  way bg-eval does. That would force serialisation through a single
  `Lock`. Today this is not done.

### 5.2 GPU contention between Merge/optimizer and bg workers — **low**

- `merge_phase_active` is held across `sync_grad_across_validators`,
  `run_global_optimization`, and `save_checkpoint` (`run.py:964-1035`).
- Same iteration-boundary caveat as 5.1 applies — at most one in-flight
  bg-eval can overlap the start of merge.
- `sync_grad_across_validators` uses DHT bandwidth, not GPU; the heavy
  GPU step is the optimizer. Overlap window is therefore the very start
  of the merge phase, which is the part that does *not* contend on GPU.
  In practice the risk is negligible.

### 5.3 HF bandwidth contention during validator HF upload — **low**

- Main re-sets `merge_phase_active` around `upload_checkpoint_to_hf`
  (`run.py:1096`). But `eval_window_active.set()` happens before this
  (`run.py:1040`), so there is a window between Merge clearing the
  Event and HF upload re-setting it during which bg-eval can resume.
  Bg-download is also ungated in that window.
- During HF upload, bg-download could pull HF in parallel; throughput
  competes for outbound NIC. Correctness is not at risk. If contention
  matters, set `merge_phase_active` once at the start of merge and clear
  it once after HF upload — collapsing the gap.

### 5.4 score_aggregator concurrent writers — **negligible**

- Three writers: main (foreground eval), bg-eval (per miner), main
  again (penalty pass + post-eval persist).
- Aggregator uses an internal RLock for both `add_score` and
  `persist_atomic`; on-disk writes are tmpfile + `os.replace`.
- Worst case: a `persist_atomic` from one thread overlays the snapshot
  of another that was a few hundred milliseconds older. Acceptable
  because each scored miner triggers another flush.

### 5.5 lite_subtensor multi-thread access — **already mitigated**

- The reason `AsyncRunner` exists: every coroutine that touches
  `lite_subtensor` is scheduled on the same loop. Bg-download uses a
  separate sync `Subtensor` of its own (`background_download_worker.py:70`),
  so it cannot accidentally drive a coroutine on the wrong loop.
- The remaining risk is if someone in the future adds an
  `await lite_subtensor.something()` on the bg-download or bg-eval
  loop — that would race the WebSocket. Code review must keep
  `lite_subtensor` calls inside `async_runner.run/submit`.

### 5.6 round_ref swap mid-iteration — **negligible**

- Main calls `round_ref.swap(...)` once per cycle at the start of the
  Submission phase. Workers read `current` once per iteration.
- bg-eval guards by checking `round_obj.round_id != self._loaded_round_id`
  before evaluating, so a stale UID from the previous round will not
  be evaluated against the new round's snapshot.
- bg-download uses `next_for_download()` which pulls the current round's
  `background_uids` — if the round changes between two iterations, the
  next iteration just sees the new pool.

### 5.7 weights_submitted flag visibility — **negligible**

- `weight_submitter.submit(pending_round, uid_weights)` returns a
  `concurrent.futures.Future`. The flag flips on the AsyncRunner thread
  after the chain RPC returns; main reads it at the top of the *next*
  cycle.
- Cycle length (~1.5 h) >> chain RPC latency, so the read always sees
  the settled value.

### 5.8 Three concurrent `asyncio.run` loops on the same process — **safe**

- `asyncio.run(...)` requires no loop on the calling thread; main has
  none, so the two transient loops it creates (foreground eval,
  gradient aggregation) are fine.
- AsyncRunner / bg-download / bg-eval each `asyncio.run` on their own
  thread. Different threads, different `set_event_loop` registries. No
  conflict.
- The pitfall: `asyncio.run` from the **main thread** while the main
  thread is *also* in the middle of an `async_runner.run(...)` call.
  That cannot happen with the current code because `async_runner.run`
  is a blocking shim that goes through `run_coroutine_threadsafe`,
  not `asyncio.run`. Keep it that way.

### 5.9 Shutdown ordering (KeyboardInterrupt / Exception) — **low risk, watch for hangs**

- Order: stop download → stop eval → join both (timeout=30s) → stop
  async_runner → stop poller. (`run.py:1200-1209`, `1217-1226`).
- If a bg worker is mid-HF download with a 30-min timeout, the join
  will time out at 30 s and shutdown continues anyway (daemon threads
  die with the process). Acceptable.
- AsyncRunner's `stop()` cancels in-flight tasks via `loop.stop()` →
  pending coroutines get cancelled in the loop's `finally`. Bittensor
  WebSocket disconnects cleanly.

### 5.10 Deadlocks — **none plausible**

- Locks held: `RoundRef._lock` (only during swap), `Round._lock`
  (atomic, fine-grained), `MinerScoreAggregator._lock` (RLock, atomic),
  `gpu_eval_lock` (only by bg-eval, briefly).
- No code path acquires two of these in opposite orders. The bg-eval
  invariant (`_assert_lock_unheld_by_us`) makes a deadlock from "worker
  blocks on a gate while holding the lock" detectable.

---

## 6. Quick rules for future changes

1. Anything that talks to `lite_subtensor` must go through
   `async_runner.run` or `async_runner.submit`. Never `asyncio.run`
   a coroutine that calls into `lite_subtensor` from the main thread,
   and never schedule one on the bg-download or bg-eval loops.
2. Bg workers re-check gating Events only between iterations. If a new
   gate is added, set it **before** the protected work begins and clear
   it **after**, with a `try/finally` — same pattern as
   `merge_phase_active` and `download_window_closed`.
3. Any new GPU code path that runs concurrently with bg-eval must
   acquire `gpu_eval_lock`. Foreground today does not, which is OK only
   because `eval_window_active` is unset during the Submission window
   so bg-eval never runs alongside foreground. Don't add a new producer
   that doesn't gate or lock.
4. If you add a new shared mutable state, decide up front: thread-local,
   lock-protected (and which lock), or atomic flag. Don't rely on
   "the GIL will sort it out" — it won't for compound updates.
5. Always update gates with `set()`/`clear()` inside `try/finally`
   (`run.py:862-879`, `run.py:964-1035`). Leaving a gate set after a
   raised exception silently halts background workers.
