# How the `Round` object is built

The `Round` (`connito/validator/round.py`) is the unit of work for steps
(0)..(4) of the lifecycle in `_specs/background-submission-validation.md`.
It is constructed once per cycle, at the start of the Submission phase,
by `Round.freeze(...)`. After construction, the *frozen* fields are
immutable; the *mutable* per-worker fields are guarded by an internal
`threading.Lock` and updated by foreground eval, bg-download, and bg-eval.

This document explains exactly what `Round.freeze` does, in the order it
does it, and which inputs determine which fields.

---

## Call site

`Round.freeze(...)` is called once per cycle from the validator main
loop after `wait_till(PhaseNames.submission)` returns
(`connito/validator/run.py:998`). The freshly built `Round` is then
published to the workers via `round_ref.swap(new_current=new_round)`
(`run.py:1037`).

```python
new_round = Round.freeze(
    config=config,
    subtensor=subtensor,
    metagraph=metagraph,
    global_model=global_model,
    round_id=phase_response.phase_start_block,
    submission_block_range=(
        phase_response.phase_start_block,
        phase_response.phase_end_block,
    ),
    last_evaluated=score_aggregator.last_evaluated_per_uid(),
    prior_avg_scores=score_aggregator.uid_score_pairs(how="avg"),
)
```

The metagraph is fetched by the caller (sync or async, depending on the
validator's subtensor type) and passed in. `freeze` itself never issues
an `AsyncSubtensor` call, so it has no opinion on the connection model.

---

## Step-by-step construction

### 1. Single chain-commits fetch, shared across helpers

`get_chain_commits(config, subtensor)` is called **once**, then handed
to both `get_combined_validator_seed(commits=...)` and
`get_validator_miner_assignment(commits=..., metagraph=...)`. Without
this, each helper would issue a duplicate `get_all_commitments` +
`metagraph()` pair against the archive endpoint, serialized through the
global subtensor lock.

Outputs:
- `seed: str` — the combined validator seed used by all eval workers.
- `assignment_result.assignment: dict[str, list[str]]` — full
  validator→miner-hotkey mapping for this cycle.
- `my_assignment_set: set[str]` — this validator's own slice
  (`assignment[config.chain.hotkey_ss58]`).

### 2. Split the roster into foreground vs background

`assignment_result.miners_with_checkpoint` is **already incentive-ranked**
by `get_validator_miner_assignment`. `freeze` walks it once and splits
each hotkey into:

- **`foreground_uids`** — UIDs in `my_assignment_set` (this validator's
  assigned slice). Kept in incentive order; the foreground eval budget
  covers them in that order.
- **`background_uids`** — every other UID with a chain checkpoint this
  cycle. Reordered in step 4 below.

Along the way it builds:
- `uid_to_hotkey: dict[int, str]` — covers the union, so workers don't
  need to hold a metagraph reference to translate UID→hotkey.
- `uid_to_chain_checkpoint: dict[int, ChainCheckpoint]` — per-UID
  `ChainCheckpoint` snapshot captured at freeze time so the eval path
  can run `validate(expert_group_assignment=...)` (signature, hash,
  expert-group ownership, NaN/Inf scan) without re-issuing chain RPCs.
- `assigned_with_valid_ckpt: set[str]` — hotkeys whose chain checkpoint
  has both an `hf_repo_id` and an `hf_revision`. Used by step 3.

A hotkey not present in `metagraph.hotkeys` is logged and skipped.

### 3. Record freeze-time invalid-checkpoint penalties

For every hotkey in `metagraph.hotkeys` **not** in
`assigned_with_valid_ckpt`, freeze records a penalty:

- `freeze_zero_uids: set[int]` — the UID itself.
- `freeze_zero_hotkeys: dict[int, str]` — hotkey map captured alongside
  because these UIDs may not appear in `uid_to_hotkey` at all (which
  only covers roster miners with a valid checkpoint).

`finalize_round_scores` later stamps each of these UIDs with `score=0`
in the aggregator. Catching them on the main thread also covers miners
with no commit at all — those never appear in `miners_with_checkpoint`
and would otherwise be invisible to the eval workers entirely.

### 4. Reorder `background_uids` (chain-weight prepend + top-scored prepend + staleness tail)

Background UIDs are split into three segments and concatenated. The
construction guarantees:

1. Every UID in `foreground_uids` is excluded from all three segments —
   the foreground slice has its own priority queue and must not be
   re-scheduled here.
2. No UID appears in more than one segment. Earlier segments win:
   any UID admitted into (a) is removed from the candidate pool before
   (b) is built; any UID admitted into (a) or (b) is removed before
   (c) is built.

In code this is enforced by maintaining a running `placed: set[int]`
that starts as `set(foreground_uids)` and is extended with each
segment's output before the next segment is built.

**(a) UIDs receiving non-zero weight from a qualified validator.**
Pull `meta.weights` (the on-chain weight matrix) and, for every validator
hotkey in `assignment_result.assignment` (the set of *qualified*
validators this cycle — i.e. those the assignment helper recognized as
weight-setters), collect the UIDs that validator is rewarding with a
non-zero weight. Restrict the union to UIDs in `background_uids` *and
not in `placed`* (i.e. not already foreground), then order them by
stake-weighted total weight (descending). Ties are broken **randomly**
(no UID-asc fallback) so a low-numbered UID has no systematic head
start when stake-weighted weights are equal. After (a) is built, its
UIDs are added to `placed`.

Why: a UID that *other* qualified validators are already rewarding is a
strong consensus signal that this validator should re-score it early —
even if its prior local avg is still low (e.g. a fresh miner that
landed an evaluation on the rest of the subnet but not yet here). This
segment runs before (b) so chain consensus can pull a previously
unseen miner ahead of the validator's own EMA leaders.

Source: `meta.weights` (fetched alongside `metagraph` at freeze) +
`assignment_result.assignment` keys. The qualified-validator filter
is what makes this trustworthy — unqualified rows in `meta.weights`
(stale, deregistered, or unrecognized validators) are ignored.

**(b) Top-N by prior-round avg score**, capped at
`BG_TOP_SCORED_PREPEND_COUNT = 5`. Source: the `prior_avg_scores`
argument (the same metric that drives weight submission).

Selection: walk `prior_avg_scores` in descending score order and admit
the first up-to-N UIDs that are in `background_uids` *and not in
`placed`* (i.e. not already foreground and not already in (a)). Ties
are broken **randomly** (no UID-asc fallback). After (b) is built, its
UIDs are added to `placed`.

Why: re-evaluating the current leaders first protects the top of the
leaderboard against a stale EMA. Without this, a strong miner that
hasn't been picked recently could keep its lead even after submitting a
*worse* checkpoint, because staleness alone defers their re-eval. The
cap exists so this segment cannot crowd out the staleness rotation.
Excluding foreground and (a) before counting against the cap means the
N slots always go to *new* leaders the earlier segments didn't already
cover.

**(c) Everyone else, sorted by staleness** — every UID in
`background_uids` not in `placed`, longest-since-last-evaluated first.
Never-evaluated UIDs are treated as infinitely stale
(`datetime.min` in UTC). Source: the `last_evaluated` argument.

Why: rotates the long tail through bg-eval instead of always favoring
the same incentive-ranked head. Each validator has different
`last_evaluated`, so the tail spreads naturally across the subnet.

The final `background_uids` is `(*a, *b, *c)` — disjoint by
construction, disjoint from `foreground_uids` by construction.

`foreground_uids` itself stays in incentive order — that's the priority
set by design and the per-round eval budget covers it.

### 5. CPU snapshot of `global_model.state_dict()`

```python
snapshot = {
    k: v.detach().clone().cpu()
    for k, v in global_model.state_dict().items()
}
```

The `detach + clone + cpu` chain is deliberate: subsequent in-place
mutations of `global_model` (Merge, optimizer step) cannot leak into the
snapshot. Bg-eval loads this into its own `_eval_base_model` via
`load_state_dict(..., strict=False)`, so its evaluations always run
against the round-K base model regardless of what the main thread does
to `global_model` afterwards.

### 6. Compute `round_id`

`rid = int(round_id) if round_id is not None else int(subtensor.block)`.
The caller passes `phase_response.phase_start_block`, so the round id
is the Submission-phase start block — stable across all workers and
log lines for that round.

### 7. Construct the dataclass

The `Round` is returned with all frozen fields populated and all
mutable fields at their defaults:

| Frozen field                     | Source                                                         |
|----------------------------------|----------------------------------------------------------------|
| `round_id`                       | `phase_response.phase_start_block` (or override)              |
| `seed`                           | `get_combined_validator_seed(commits=...)`                    |
| `validator_miner_assignment`     | `get_validator_miner_assignment(...).assignment`              |
| `foreground_uids`                | this validator's assignment slice, incentive-ordered          |
| `background_uids`                | top-scored prepend + staleness tail (step 4)                  |
| `uid_to_hotkey`                  | union of foreground + background                              |
| `uid_to_chain_checkpoint`        | per-UID `ChainCheckpoint` from `assignment_result`            |
| `model_snapshot_cpu`             | CPU clone of `global_model.state_dict()`                      |
| `submission_block_range`         | `(phase_start_block, phase_end_block)` from caller            |
| `freeze_zero_uids`               | metagraph UIDs lacking a valid chain checkpoint                |
| `freeze_zero_hotkeys`            | hotkey map for `freeze_zero_uids`                              |

Mutable fields (all `field(default_factory=...)`, lock-guarded):

| Mutable field             | Writer(s)                                          |
|---------------------------|----------------------------------------------------|
| `downloaded_pool`         | bg-download (`publish_download`); pop by eval      |
| `scored_uids` / `scores`  | foreground eval, bg-eval (`mark_scored`)           |
| `claimed_uids`            | foreground + bg-eval (`claim_for_*`/`release_claim`) |
| `failed_uids`             | foreground + bg-eval (`mark_failed`)               |
| `validation_failed_uids`  | bg-eval on validate failure (`mark_validation_failed`) |
| `weights_submitted`       | `BackgroundWeightSubmitter` after chain RPC        |

---

## Why this shape

- **One construction site, one swap point.** `Round.freeze` is the only
  place that can produce a `Round`, and `RoundRef.swap` is the only way
  workers see a new one. Workers re-read `round_ref.current` each
  iteration, so a swap takes effect without restarting any thread.
- **Frozen fields are *truly* frozen.** `freeze` builds them from
  inputs the caller has already pinned (metagraph, head-block commits,
  `global_model.state_dict()` clone). Nothing in the round depends on
  re-querying chain or GPU state later.
- **Mutable fields are narrowly scoped.** Every getter/setter that
  touches them holds `self._lock`. The `_assert_lock_unheld_by_us`
  invariant in bg-eval prevents the "worker blocks on a gate while
  holding the round lock" deadlock pattern.
- **`uid_to_chain_checkpoint` is captured at freeze.** Eval validates
  signature / expert-group / hash / NaN-Inf against this snapshot
  rather than re-issuing chain RPCs per miner. A miner that re-commits
  mid-round is not re-validated against the new commit until next
  round's freeze.
- **`submission_block_range` is part of the round.** Bg-download uses
  it to filter `_existing_submission` reuse so a stale `.pt` left over
  from a previous cycle cannot short-circuit a fresh fetch and get
  published into `downloaded_pool`.

---

## Related docs

- `validator-cycle-phases.md` — phase-by-phase walkthrough of one cycle.
- `validator-concurrency-model.md` — every thread / loop / lock that
  reads or writes the `Round` after construction.
- `_specs/background-submission-validation.md` — the original spec that
  defines steps (0)..(4) and the round lifecycle.
