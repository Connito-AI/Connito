# Miner promotion through validation Groups A → B → C

Connito's tiered scheme places every miner into one of three validation
groups each cycle. The group determines whether the miner is evaluated,
by which validators, and how much chain weight they can earn. This doc
describes the conditions under which a miner enters each tier and the
promotion path from new arrival (Group C) up to the heavily rewarded
consensus tier (Group A).

The construction logic lives in `connito/validator/round_groups.py` and
is invoked by `Round.freeze()` (`connito/validator/round.py`) at the
start of every cycle's Submission phase.

---

## The three validation groups

| Group | Role | Size | Where it comes from |
|-------|------|------|---------------------|
| **A** | Consensus tier — miners already heavily rewarded by the rest of the subnet | up to 13 (shares budget with B) | Chain-set top-3 tally (`metagraph.weights`) |
| **B** | Candidate tier — miners on the chain-set top-5 but not yet hitting Group A's gates | `13 − \|A\|` | Chain-set top-5 tally (`cfg.weight_group_2_size`) |
| **C** | Exploration tier — every other registered miner, partitioned across validators | up to `validation_group_c_size` (default 17) per validator | Seeded `assign_miners_to_validators` over `(all miners) \\ (A ∪ B)` |

`|A| + |B|` is capped at `validation_group_ab_total` (default 13)
globally; Group A absorbs every miner clearing both gates and Group B
fills the remainder. Group C is per-validator, so two validators see
different Group C rosters.

Source helpers (all in `round_groups.py`):

- Group A: `compute_group_a` (line 157)
- Group B: `compute_group_b` (line 193)
- Group C: `compute_group_c` (line 260)
- Single entry point: `build_cohort_groups` (line 401)

---

## Conditions for entering each group

### Group C — the entry point

A miner lands in some validator's Group C as soon as it:

1. Is registered on the subnet (present in `metagraph.hotkeys`).
2. Is **not** in the global `A ∪ B` set this cycle.
3. Is assigned to that validator by the seeded round-robin partition
   (`connito.shared.cycle.assign_miners_to_validators`).

Group C is the default placement; no positive signal is required. The
Group C roster is capped at `cfg.validation_group_c_size` miners per
validator.

**Why it matters:** miners in Group C are evaluated by their assigned
validator and earn local score points. Without Group C exposure a
fresh miner accumulates no history and cannot clear the upper tiers'
weight-emission gates.

### Group B — top-`(13 − |A|)` of the chain-set top-5 tally

A miner enters Group B when:

1. At least one **qualified validator** placed them in their on-chain
   weight Group 2 (top-`cfg.weight_group_2_size`, default 5) last
   cycle. "Qualified" means the validator hotkey is recognized by
   `validator_miner_assignment` — stale, deregistered, or
   unrecognized validators are ignored.
2. They are not already in Group A (Group A drains first; Group B
   takes only the remaining `13 − |A|` slots).
3. After excluding Group A, they rank in the top of the chain-set
   tally by **total weight received**.

Tiebreaker: validator count desc, then UID asc. There is no
per-validator weight floor for Group B — this is the candidate tier.

The tally is computed by `read_chain_set_top_k(k=cfg.weight_group_2_size)`
reading directly from `metagraph.weights`.

### Group A — top of consensus, with a 3 % per-validator floor

A miner enters Group A when **both** gates clear:

1. **Count gate.** At least `cfg.group_a_min_consensus` qualified
   validators (default `1`) placed them in their on-chain weight
   Group 1 (top-3) last cycle.
2. **Per-validator weight gate.** At least one of those validators
   emitted **strictly more than `cfg.group_a_min_weight_per_validator`**
   of their total weight to this miner — default **3 %**. (Implemented
   as `max_weight_from_one_validator > min_weight_per_validator` in the
   chain-set tally.)

The cap is `cfg.validation_group_ab_total` (13). If more than 13 miners
clear both gates, the highest aggregate `total_weight_received` win;
ties break on validator count desc, then UID asc.

Source: `read_chain_set_top_k(k=cfg.weight_group_1_size)` (top-3
ballots) followed by `compute_group_a`.

---

## The promotion path: C → B → A

A miner can climb the tiers only by accumulating chain-set votes —
**what other validators emit on chain**, not what the local validator
has scored:

1. **C → B (chain-set Group 2).** Receive non-zero weight from any
   qualified validator's top-5 ballot. Each validator builds that
   ballot from its **local rolling-average** score history — see
   "Local weight emission" below — so in practice scoring well in
   Group C against multiple validators is the path: each validator's
   local avg pushes you into its on-chain top-5, and the chain-set
   tally then pulls you into the global Group B next cycle.

2. **B → A (chain-set Group 1).** Receive **> 3 %** weight from at
   least one qualified validator's top-3 ballot. This requires being
   one of the very best miners in some validator's local rolling avg
   AND clearing the per-validator weight floor. Group B exposure is
   where you collect those votes — multiple validators' top-3 ballots
   accumulate via the chain-set tally and pull you into A.

Cohort timing: as of the current `master`, the cohort is rebuilt
**every cycle** — `maybe_advance_cohort` always re-runs the election
and rebuilds the groups (`round_groups.py:505`). Validation groups can
shift cycle-by-cycle as chain weights change; there is no 8-cycle hold
even though `cohort_window_cycles` still appears in the config.

---

## Local weight emission — how this validator votes

Validation groups (A/B/C) decide *who gets evaluated*. The local
weight ballots decide *who this validator rewards on chain*, and they
have additional history gates layered on top.

The two ballots are emitted from
`connito/validator/run.py` right after `finalize_round_scores`:

**Weight Group 1 — `cfg.weight_group_1_share` (default 98 %), top-3 of A ∪ B by aggregator avg:**

- Local score-aggregator avg ranking, restricted to `A ∪ B`.
- Miner must have **≥ 3** score records in the aggregator.
- Miner must have a score recorded in **both** of the last 2 rounds:
  one tagged with `round_id = current_round_id` and one tagged with
  `round_id = current_round_id − cycle_length`. Missing either round
  drops the miner off the Group 1 ballot for this cycle.
- **Empty-Group-1 guard:** if no UID clears the gates, the 98 % share
  is redirected to `uid = 0` (subnet owner) rather than dropped.
  This keeps the validator's total emission at 100 % so its
  consensus signal is not diluted while it waits for a miner to
  clear the recency gate.

**Weight Group 2 — `cfg.weight_group_2_share` (default 2 %), top-`cfg.weight_group_2_size` (default 5) of A ∪ B ∪ C \\ G1 by aggregator avg:**

- Local score-aggregator avg ranking, restricted to
  `A ∪ B ∪ C` minus the miners already on this validator's
  Group 1 ballot.
- Miner must have **≥ 2** score records in the aggregator.
- No recency requirement — Group 2 is the slow-rotating reward tier
  and a miner with two old records still qualifies.

These two ballots are what other validators see next cycle when
computing **their** chain-set tallies — the loop that drives the
C → B → A promotion above.

Score history older than `8 × cycle_length` blocks is pruned every
cycle (`MinerScoreAggregator.prune_before_round`). The avg never
reflects scores beyond that 8-cycle window, so a miner that goes idle
for 8+ cycles drops out of the upper tiers automatically.

---

## The non-A/B/C tail — eval coverage for unranked miners

A miner that is not selected into this validator's `A ∪ B ∪ C` for the
cycle is **not excluded** from evaluation. `Round.freeze`
(`connito/validator/round.py`) appends a tail to `background_uids`
containing every miner with a chain checkpoint that did not land in
the cohort roster.

Construction:

- **Pool** — `(miners with chain checkpoint) \\ (A ∪ B ∪ C ∪ foreground)`.
  Foreground is already a subset of A ∪ B, so this is effectively
  "everyone with a checkpoint outside the cohort".
- **Order** — staleness desc (longest-since-last-evaluated first),
  random tiebreak. Same ordering used for the legacy background
  staleness tail.
- **Placement** — appended after the A → B → C background segments.
  Background workers (download + bg-eval) walk the queue in order, so
  the tail runs only when the cohort roster is fully covered and there
  is spare capacity left in the round.

Effect on promotion: tail miners earn score records exactly like
Group C miners do, which is what feeds the ≥ 2 / ≥ 3 record thresholds
above and the rolling avg that drives the next cycle's chain-set
ballots. Without the tail, a miner that drops out of every validator's
A ∪ B ∪ C for a cycle would have no opportunity to accumulate history
and would be stuck — the tail keeps the C → B → A path open even when
the seeded Group C partition does not pick them.

---

## What keeps a miner out of every group (or out of weight emission)

- **No chain commitment / invalid checkpoint at freeze time.**
  UIDs without a usable `(hf_repo_id, hf_revision)` land in
  `freeze_zero_uids` and `finalize_round_scores` writes
  `score = 0` for that round. They still appear in the metagraph but
  are not evaluated and earn no reward (`Round.freeze` step 3).
- **Hotkey rotation.** When a UID's hotkey changes,
  `MinerScoreAggregator.add_score` resets that UID's history. The
  miner must re-accumulate the ≥ 3 / ≥ 2 record thresholds before it
  can re-appear in weight Group 1 / 2.
- **Validation failure.** Hash, signature, expert-group, or NaN/Inf
  failures during eval flag the UID in `validation_failed_uids`;
  `finalize_round_scores` writes `score = 0` for that round.
  Repeated failures pull the avg down and eventually push the miner
  out of the upper tiers via the chain-set tally.
- **Stale aggregator.** A miner that misses **either** of the last
  2 rounds fails the Group 1 recency gate and drops off this
  validator's top-3 ballot for the cycle. Group 2 has no recency gate —
  it only requires ≥ 2 records.

---

## Related docs

- `validator-round-construction.md` — how `Round.freeze` consumes the
  cohort groups to build foreground / background eval queues.
- `validator-cycle-phases.md` — phase-by-phase walkthrough of one cycle.
- `validator-concurrency-model.md` — every thread / loop / lock that
  reads or writes the `Round` after construction.
