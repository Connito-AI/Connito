# Dead code and comment cleanup

Scope: `connito/`, `expert_groups/`, `observability/`. Behaviour-preserving
except where noted under "Dead code removed" — nothing was refactored, renamed,
or re-typed.

Totals: 46 files, +708 / −1988 lines. Comments in changed files: 3282 → 2248
(−32%).

Verification:
- `ruff check` (0.6.3, repo config): 466 → 393 findings, **zero new**, compared
  rule-by-rule against a `master` worktree.
- `python -m compileall` clean over the whole tree.
- AST-equivalence check (docstrings blanked) against `master` for the
  comment-only portion of the change.
- **Tests were not run.** The repo's `.con-venv` is unusable — its base
  interpreter `/usr/bin/python3.12` no longer exists on this host (system Python
  is 3.14) and the dependencies are not installed outside it. `pytest` needs a
  rebuilt environment.

---

## Dead code removed

### Functions and classes with no callers

Verified by cross-referencing every identifier against all `.py`, `.ipynb`,
`.md`, `.yaml` and `.json` files in the repo.

| Symbol | File | Why it is dead |
|---|---|---|
| `_cuda_mem_report` | `validator/run.py` | Never called, **and calls an undefined `log_phase`** — it would have raised `NameError` on the first call. |
| `PhaseResponseLite` | `shared/cycle.py` | Superseded by `PhaseResponse`. |
| `search_model_submission_destination` | `shared/cycle.py` | Resolved a validator axon for the HTTP submission path, removed in the HF-only migration. |
| `SignedModelSubmitMessage` | `shared/schema.py` | Never constructed. |
| `_normalize_hash`, `_hash_bytes` | `shared/checkpoints.py` | Private helpers, no callers. |
| `hex_to_byte` | `shared/helper.py` | One-line wrapper over `bytes.fromhex`. |
| `names_for_expert`, `iter_named_grads` | `validator/inter_validator_connection.py` | No callers (`iter_named_params`, which they sit beside, is used). |

### Unused imports and duplicate imports

42 removed across 23 files via `ruff --select F401,F811 --fix`. Checked first
that none were re-exported: no other module imports any of them from the module
that held them. Includes two genuine duplicate imports
(`checkpoints.get_layer_expert_id`, `inter_validator_connection.dataclass`).

### Unused local variables

- **`miner/train.py`** — a per-step diagnostic block (`grad_total`,
  `sample_grads`, `p_norm`, `grad_norm`) whose results were computed and never
  read. This is the only removal with a runtime effect: it drops a
  `sum_model_gradients()` walk and up to five `.norm().item()` GPU syncs per
  training step. No observable output changes.
- **`shared/model.py`** `is_3d_expert_block`, **`modeling/mycelia.py`**
  `model_path`, **`sn_owner/cycle.py`** `phase_end`, **`shared/cycle.py`**
  `reason` — computed, never read.
- **`validator/evaluator.py`** — dropped the unused binding from
  `incompatible = model.load_state_dict(...)`; the call itself is kept.
- **`miner/train_helper.py`** — `except Exception as e: pass` → `except Exception:`.

### Commented-out code

~330 lines. The largest blocks: a dead `ModelCheckpoint` class and four dead
checkpoint helpers in `shared/checkpoint_helper.py` (all have live
implementations in `shared/checkpoints.py`), a superseded `load_model_from_path`
in `validator/evaluator.py`, a `broadcast_weights` stub in
`shared/expert_manager.py`, and commented-out `torch.distributed` calls,
debug prints and superseded assignments in `miner/train.py`.

---

## Comment cleanup

### Excessive comments simplified

The dominant pattern was a config field or a two-line call carrying a
15–45 line essay: the full derivation, the memory arithmetic, the measured
statistics from a one-off probe, and the rollout plan. These were compressed to
the operative constraint, typically 2–4 lines. The heaviest cases:

- `shared/config.py` `DataCfg` eval-sampling knobs — ~75 lines → ~24, with one
  shared "must be rolled out fleet-wide or weight consensus breaks" line instead
  of the same warning repeated per field.
- `shared/dataloader.py` — the HF timeout rationale (18 lines → 5) and the
  shuffle/skip derivations (30 lines → 10).
- `shared/eval_shard_pick.py` — the per-source policy contract (40 lines → 20),
  and the C4 shard-count arithmetic reduced to the invariant an editor has to
  respect.
- `modeling/mycelia.py` partial-load strategy (22 lines → 12).
- `validator/run.py`, `validator/round.py`, `validator/evaluator.py`,
  `shared/telemetry.py`, both background workers — dozens of 6–11 line blocks
  reduced to 2–4.

### Development history removed

- Incident logs and timestamps used as evidence (`validator_A6000_v0.1.38.log`,
  "observed 2026-07-31: two Watchtower restarts 25 min apart", "round 8081470
  sat ~27 min", specific staging UIDs and block numbers).
- "The previous implementation did X" narratives where only the current
  behaviour matters.
- Rollout-window commentary ("during the auto-upgrade rollout", "kept for the
  migration window", "see PR description").
- Measured statistics that were evidence for a past decision rather than a
  constraint on future edits (38% empty rows, 75% shared prefixes, ~2000
  distinct samples over a 50-seed probe, ~76 GB peak RAM of a since-replaced
  approach).

The underlying *constraint* was kept in every case — e.g. "a small enough pool
for a miner to memorize" survived; the probe that measured it did not.

### Code narration removed

`# Compute missing keys`, `# Sort all scores descending`, `# Build loader`,
`# 3. Verify`, `# Load raw JSON assignment`, and section banners that only
restated the `logger.debug` on the next line (`# === optimizers ===` above
`logger.debug("init - optimizer")`).

### Stale comments fixed

| Comment | Reality |
|---|---|
| Two references to a `/v1/state.json` API | Removed from the codebase. |
| `submission_period: int = 80 # 4 mins`, `validate_period: int = 10 # 10 mins` | 16 min and 2 min at 12 s/block. |
| `vali_fraction, # use ~20% of the dataset` | Default is 0.1. |
| `HfCfg`: advertised repo "derived as `{owner}/cycle`" | `advertised_repo_id` returns the upload repo unchanged. |
| "Wait until 30 blocks before the next MinerCommit1" | Code passes `block_offset=-15`. |
| `# get the base config from qwen model` | It is DeepSeek. |
| `# Remove .pt extension` above `Path(filename).stem` | Strips any suffix; `.safetensors` is now the primary format. |
| `# llm_weightnet/shared/logging.py` file header | Wrong package name. |
| `# Module-level logger you can import directly` above `structlog.configure_once(...)` | It is configuration, not a logger. |
| "category (2)", "PR 3", `# === Comit to chain ===`, `benchamrk`, `trian`, `vlaidators`, `COMISSION`, `mocated` | Dangling references and typos. |
| `_specs/background-submission-validation.md`, `_specs/round-group-construction-scheme.md` (6 sites) | Not in this repo; retargeted to `docs/validator-round-construction.md` and `docs/miner-validation-group-promotion.md`. **See "Needs a decision" below.** |

### Invariants deliberately preserved

Compressed but never dropped:

- **Consensus** — fleet-wide rollout coupling for every `eval_source_*` knob;
  revision pinning because `"main"` moves; the seeded-shard-pick policy registry
  contract; "never fall back to a constant seed".
- **Security** — `.safetensors` has no pickle path and `.pt` is gated by
  `weights_only=True`; `trust_remote_code` is opt-in per source; the NaN-batch
  divisor exploit in `shared/evaluate.py`; the unauthenticated second-opinion
  probe before attributing a download failure to a miner.
- **Concurrency and ordering** — `gpu_eval_lock` is acquired inside the eval
  thread and never held across an iteration boundary; never free a
  `threading.Lock` this thread does not own; one `Subtensor` per thread; the
  chain-extrinsic serialization that prevents "Priority is too low"; refusing a
  fresh gradient sync while an orphan may still be writing `model.grad`.
- **Data-format and protocol** — the 128-byte chain commit budget; `pubkey(32)
  || block(u64 big-endian)`; URL-safe base64 without padding; per-expert slice
  serialization of the stacked expert tensors.
- **Anti-footgun** — why `asyncio.run` is not used for foreground eval; why HF
  uploads run in a subprocess; why the miner-role version filter is skipped.

### Long comments deliberately kept

- `shared/telemetry.py` `EVAL_STATUS_CODES` / `COHORT_GROUP_CODES` contracts —
  these are mirrored by an external gateway and renumbering silently
  reinterprets historical Prometheus samples. Length is the warning.
- `validator/background_eval_worker.py` GPU-lock ownership comment (~10 lines
  after compression) — it explains a cancellation-semantics trap that has
  produced real wedges and cannot be shortened without losing the mechanism.
- `expert_groups/*/dataset.py` "Customer Extension Point" headers — these are
  operator-facing documentation of a supported extension point.
- `shared/chain.py` back-compat wrapper note (see below).

---

## Deliberately not removed

- **`shared/chain.py:validate_miner_chain_commit_payload`** — unused in-repo,
  but explicitly documented as a back-compat wrapper for external importers.
  Removing it would break the documented contract; left alone.
- **`miner/train.py:398`** — `aux_loss = torch.tensor(0.0)` with the real
  `outputs.aux_loss` computation commented out directly above. The commented
  line is the only explanation for why the MoE aux loss is hard-zeroed, so it
  was kept rather than deleted. Needs a decision (below).
- **`miner/train.py:477`** — `# dist.all_reduce(p.grad, ...)` above a bare
  `p.grad.div_(world_size)`. Same reasoning: dividing by `world_size` with no
  all-reduce looks wrong, and the commented line is the only context for it.
- **Unused locals in `connito/test/`** (6 of them) — left alone; tests were
  otherwise untouched apart from the two `_specs/` pointers and ruff's import
  fixes.

---

## Needs a decision

1. **`inter_validator_connection.py:validate_response` never verifies
   anything.** It decodes the signature into `sig_b64url` (unused), rebuilds
   the signing payload into `msg` (unused), then `return True`, followed by an
   unreachable `return False`. `sign_request` / `sign_response` are implemented;
   the response-side check is not. This is the inter-validator auth handshake,
   so it is reported rather than "cleaned" — deleting the unused locals would
   make the function look intentionally trivial and hide the gap.

2. **`_specs/*.md` is not in this repo.** Six sites cited it. They now point at
   the `docs/` equivalents. If those specs live in a private repo or another
   branch, revert those six one-line changes.

3. **`checkpoints.py:archive_top_miner_submissions` ranking looks inverted.**
   It scores off `uid_score_pairs(how="avg")`, where higher is better, then
   sorts ascending and labels `ranked_hotkeys[0]` "best". The comments said
   "lower val_loss = better", which is not what the value is. The comments are
   now factually neutral; the ordering itself is untouched.

4. **Two `logger.info("reached barrier, ...")` calls in `miner/train.py`** now
   log a barrier that does not exist — the `dist.barrier` they described was
   commented-out code and was removed. Changing log strings is out of scope for
   this pass.

5. **`shared/cycle.py:33`** — `dict[str, "ChainCheckpoint"]` is a forward
   reference with no import (ruff F821, pre-existing). Harmless at runtime under
   `from __future__ import annotations`, but `get_type_hints()` on
   `ValidatorMinerAssignment` would fail. A `TYPE_CHECKING` import fixes it; not
   done here because it is neither dead code nor a comment.

6. **`pyproject.toml` is partly stale** (not touched by this pass):
   `[project.scripts]` points at `connito.miner.cli` / `connito.validator.cli`,
   neither of which exists; `testpaths = ["tests"]` names a directory that is
   not there, so a bare `pytest` collects nothing; `[tool.connito.defaults]`
   (Llama-3-8B, Redis, S3) is read by nothing; `ruff` and `black` disagree on
   line length (120 vs 100).
