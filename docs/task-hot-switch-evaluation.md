# Evaluation: switching the active task without a restart

> **Status: evaluation only — no such path exists in the code today.** This
> document answers "if the active expert group (`task.expert_group_name`) had
> to change in a live process, which objects and processes would need updating,
> and to what extent — a property assignment, or a full rebuild?"
>
> **Revised against `f565df6`** (post-`staging_v3`: no merge/averaging, no
> foreground eval, round-baseline adoption). Validator internals moved enough
> that the earlier revision of this document is wrong in three places — see
> [§6](#6-what-changed-in-staging_v3).
>
> For the *restart-based* fleet migration (the supported path), see
> [exp-legal-migration-plan.md](exp-legal-migration-plan.md). That document's
> flag-day mechanics apply here unchanged; this one covers only the in-process
> mechanics a hot switch would additionally need.

## Summary

The only mutator of the active task is `WorkerConfig._update_by_task()`
(`connito/shared/config.py:697`), called exactly once from `model_post_init`.
Every downstream consumer falls into one of four tiers:

| Tier | Extent of update | Examples |
|---|---|---|
| 0 | The mutation itself + two hand re-derivations | `_update_by_task`, grad-accum |
| 1 | Nothing — reads config lazily by reference | bg-download, per-cycle commits, gauges |
| 2 | One attribute assignment / one discard | `BackgroundEvalWorker.expert_group_assignment`, `baseline_ref` |
| 3 | Object reconstruction | model, dataloader, `ExpertManager`, (miner) optimizer |

Whether the **model** needs a full rebuild or only a weight reload depends on
the target group's `expert_assignment.json` — see [§3](#3-full-rebuild--object-reconstruction).

## 0. The switch itself — a property change, but not a plain assignment

```python
config._update_by_task(expert_group_name="exp_X")  # name → _refresh_paths → reload task.exp → _refresh_paths
config._ensure_runtime_dirs()                      # new task.path + group-scoped ckpt dir
```

Assigning `config.task.expert_group_name` directly is the exact production bug
observed on the pioneer validator on 2026-07-11, now regression-tested in
`connito/test/test_locked_task_rederive.py`: the name flipped, but
`task.path`, `task.exp` and `ckpt.checkpoint_path` stayed derived from the old
group, so the process kept training `exp_math` while committing as if it had
switched.

Two things `_update_by_task` does **not** re-derive, because they live in a
`model_validator(mode="after")` that runs only at construction
(`config.py:995-1011`) and only fires on sentinel values:

- `local_par.gradient_accumulation_steps` — derived from `task.exp.data.batch_size`
- `ckpt.checkpoint_interval`, `ckpt.full_validation_interval`, `log.metric_interval`

If the target group's `batch_size` / `per_device_train_batch_size` differ, these
must be recomputed by hand.

## 1. Free — no action needed

These read `config.task.exp.*` at call time and hold the same config object by
reference, so an in-place mutation propagates on its own:

| Consumer | Reads at |
|---|---|
| bg-download filenames + group filter | `background_download_worker.py:224-231` |
| bg-eval dataloader (rebuilt per round) | `background_eval_worker.py:311-316` |
| per-cycle chain commits (`expert_group`) | `run.py:1184,1576`; `model_io.py:340` |
| checkpoint sharding / `active_expert_group_id` | `run.py:1488`; `train.py:700` |
| `eval_shard_pick` LRU caches | keyed by `(repo_id, name, revision)` — no stale entries |
| Prometheus gauges | labeled `expert_group=...` at emit time; the old label series just freezes at its last value |

`cohort_state.json`, `score_aggregator.json` and the round journals all live
under `ckpt.checkpoint_path`, which **is** group-scoped
(`config.py:655-661`), so they self-isolate.

> The migration plan's "delete the validator's `cohort_state.json` after the
> group switch" step (`exp-legal-migration-plan.md` §1.4) is obsolete for the
> same reason — the file path itself moves with the group.

## 2. Property-level updates — one attribute assignment or discard each

- **`BackgroundEvalWorker.expert_group_assignment`** — captured once at
  construction (`run.py:932`, `background_eval_worker.py:95`). Must be
  reassigned from the new `ExpertManager`, or the worker validates submissions
  against the old group.
  (`_eval_base_model` self-heals: the main loop re-seeds it at every round
  freeze via `set_eval_base_model`, `run.py:1373`.)
- **`baseline_ref`** (`run.py:697`) — the dict the publish thread fills at
  MinerCommit1 and the Merge window consumes to advance the model. It holds a
  path to an *old-group* shard, so it must be cleared as part of the switch;
  the cost is one cycle with no model advance (`run.py:1416` already handles the
  empty case).
- **`score_path` / `score_aggregator`** — the path moves with
  `checkpoint_path` (`run.py:708`); reload from the new file rather than
  carrying the old group's rolling scores forward.
- **In-flight `RoundRef` / `current_cohort_state` locals** — dropped, not
  migrated (see [§4](#4-where-in-the-cycle-the-switch-is-safe)).

## 3. Full rebuild — object reconstruction

- **`ExpertManager`** — always. Cheap (JSON load), but it re-resolves the
  active + helper task folders (`expert_manager.py:152`).
- **Dataloader** — always. The `StatefulDataLoader` wraps a streaming iterator
  bound to the old `dataset_sources` / `dataset_class` (`dataloader.py:501`).
- **Optimizer + scheduler + `GradScaler`** — miner only; the validator no
  longer has any (see §6). On the miner: always, in practice. Param objects
  change on any model rebuild, and even on a same-topology switch the AdamW
  moment estimates are stale with respect to a different data distribution.
- **Model** — *conditional:*
  - The expert `ModuleDict` is keyed by global expert id and materializes only
    the ids the group assignment names
    (`custom_deepseek_v2_lite.py:105-180`), and `max_position_embeddings` is
    baked from `task.exp.data.sequence_length`
    (`custom_deepseek_v2_lite.py:487`). If the target group's assignment or
    sequence length differs, the existing object **cannot host** the new group
    → full `get_base_model` → `from_pretrained` rebuild (`mycelia.py:131`),
    then `freeze_parameters` (`model.py:97`) + `apply_from_config`.
  - Concretely: `exp_math` (group 0) and `exp_nemotron_c4` (group 4) have
    **byte-identical** assignments (26 layers, the same 8 org expert ids per
    layer, identical `(slot, org)` pairs) and the same `sequence_length: 1024`
    — that pair is a *same-topology* switch where the module tree can be kept.
    `exp_legal` (group 3, 5 experts/layer, different ids) and the `*_p02`
    groups (12 experts/layer, seq 4096) are hard rebuilds.
  - Either way the **weights** must be replaced: state is per-group on
    chain/HF (`fetch_model_from_chain_validator(expert_group_ids=[group_id])`).
    The live in-place loader — the Merge-window baseline adopt at
    `run.py:1435` — is `load_state_dict(..., strict=False)` and cannot change
    topology; a cross-group file there matches zero keys.
    (`reload_model_inplace`, `model.py:457`, is no longer called from the run
    loop at all — tests only.)
- **Tokenizer** — no change. It comes from `config.model.model_path`, not the
  task.

The miner already has the whole of tier 3 behind a single call: the
"(5) Reload Model" branch does `free_cuda_models(...)` (`train.py:733`) then
`setup_training(...)` (`train.py:750`), which returns a fresh
model / optimizer / scaler / scheduler / expert_manager / dataloader from
config. A hot switch on the miner is therefore: mutate config, then take that
branch. The validator's `setup_training` (`run.py:551`) now returns only
`(global_model, start_step, expert_manager, train_dataloader)` — the background
workers, chain submitter, poller and score aggregator all sit *outside* it.

## 4. Where in the cycle the switch is safe

The validator's model no longer advances by merging gradients; it advances once
per cycle in the Merge window by adopting the round baseline. That moves the
safe seam. Phase order in the current loop:

| Phase | What happens | Task-coupled state created |
|---|---|---|
| MinerCommit1 −5 | `eval_window_active.clear()`, finalize round K, spawn `publish_round_baseline` + `publish_round_podium` (daemon threads) | `baseline_ref` ← old-group shard path |
| Submission | `Round.freeze` for K+1, `set_eval_base_model(deepcopy(global_model))`, `eval_window_active.set()` | round snapshot, per-round batches, baseline loss |
| Merge | adopt `baseline_ref["path"]` in place, save `globalver_*` | new checkpoint under the group-scoped dir |
| ValidatorCommit1/2 | advertise baseline coords, `baseline_ref.clear()` (`run.py:1539`) | — |

- **Validator:** switch after `baseline_ref.clear()` and before the next
  cycle's `Round.freeze`. The eval window for round K+1 is still open across
  that span, so the switch must first `eval_window_active.clear()` and finalize
  (or abandon) that round's journal — a `Round` carries
  `model_snapshot_cpu`, a `baseline_loss` and per-round batches from the old
  dataset, and scoring a new-group miner against them is meaningless. Re-emit
  `ValidatorChainCommit(expert_group=...)` immediately; the startup commit at
  `run.py:879` is the only one outside the loop.
- **Miner:** at `PhaseNames.distribute`; the existing
  `wait_till(config, phase_name=PhaseNames.distribute)` re-entry idiom is the
  right hook. Switching mid-train means committing a new-group model with
  almost no training on it.

## 5. Sharp edges

1. **The publish-thread race is the dangerous one.**
   `publish_round_baseline` runs on an unjoined daemon thread
   (`run.py:1021`) and reads the group id *late*, when it stages the upload:
   `os.link(src, stage / f"model_expgroup_{config.task.exp.group_id}.safetensors")`
   (`distribute.py:98`). A switch landing inside that window uploads the **old
   group's weights under the new group's shard name**. For the
   `exp_math` ↔ `exp_nemotron_c4` pair the expert ids are identical, so the
   file loads cleanly on the other side and the corruption is *silent* — no
   missing keys, no `matched_keys == 0` warning. Any hot-switch implementation
   must join or cancel that thread (and `publish-podium`) before mutating the
   config.
2. **Two directories are not group-scoped**, unlike `checkpoint_path`:
   `validator_checkpoint_path` (`config.py:663`, still the miner's download dir
   for the validator's global checkpoint) and `miner_submission_path`
   (`config.py:678`). After a switch, `select_best_checkpoint` can pick a stale
   `globalver_*` directory there; the group filter in
   `compile_full_state_dict_from_path` (`checkpoint_helper.py:590`) then skips
   every file in it, so `load_checkpoint` silently overlays **nothing** while
   `current_model_meta` advances. The retained-baseline dir
   `<base_ckpt>/baseline/round_<rid>.safetensors` (`distribute.py:111`) has the
   same problem and carries no group marker in its name. Clear or group-scope
   all three as part of the switch.
3. **A hot switch does not survive a restart.** `config.write()`
   (`config.py:965`) targets `<checkpoint_path>/config.yaml` — i.e. the *new*
   group's directory — while the file the process booted from (and the docker
   `CONFIG_PATH`, `config.py:1090`) still names the old group. And
   `task.expert_group_name` is a locked field (`config.py:562-565`), so
   `auto_update_config` resets any non-default value back to the fleet default
   on the next load. A no-restart switch is only coherent *toward* whatever the
   locked default is; anything else is drift that gets reverted.
4. **The switch is a consensus event, not a local one.** Validators score only
   miners whose chain commit carries a matching `expert_group`, so a unilateral
   hot switch makes the operator invisible to the rest of the fleet — including
   the deregistration hazard recorded in `exp-legal-migration-plan.md` §1.4.
   Removing the restart does not remove the flag-day requirement.
5. **First cycle after the switch has a meaningless `baseline_loss`** — the
   adopted model has never seen the new data. Same warm-up caveat as the
   restart path, and it now also lands in the
   `validator_baseline_loss` series (`background_eval_worker.py:367`).
6. If the new group's dataset source is not registered in `_KNOWN_SOURCES`
   (`eval_shard_pick.py:168`), the first seeded shard-pick after the switch
   raises. Either register the source or set
   `data.eval_source_seeded_shard_pick: false` in that group's `config.yaml`
   before switching.
7. Minor: podium archive folders are named `cycle_<n>` with files
   `rank<k>_uid<n>.safetensors` and a manifest carrying no expert group
   (`distribute.py:186-289`), so an `hf.archive_repo` spanning a switch cannot
   be told apart by group after the fact.

## 6. What changed in `staging_v3`

Three items from the previous revision of this document are now dead, and one
is new. Recorded because they were the *heaviest* rebuild items:

- **Hivemind averagers / DHT — gone.** The validator no longer merges
  (`a80a7a7`, `b353379`); `connect_with_peers`, `build_grad_buff_from_model`
  and `build_averagers_from_buff` are no longer called from `run.py`, so there
  is nothing to tear down and rejoin under a new
  `expert_averaging-group{id}` prefix. `SystemStatePoller` still accepts
  `group_averagers` but `run.py:865` no longer passes it.
- **Validator optimizer + `GradScaler` — gone.** `setup_training` returns no
  optimizer (`run.py:551`); only the miner still builds one. (The
  `setup_training` docstring at `run.py:566` is stale and still mentions
  optimizers/scheduler/scaler.)
- **Foreground eval — gone** (`cb7cc9c`, `841babd`). Every miner is scored
  through `BackgroundEvalWorker`, so its construction-time
  `expert_group_assignment` is now the *only* group-assignment snapshot on the
  eval path, not a secondary one.
- **New: in-place baseline adoption.** `global_model.load_state_dict(sd,
  strict=False)` at `run.py:1435` is now how the validator's model advances,
  which is what creates both the `baseline_ref` discard (§2) and the publish
  race (§5.1).

Unchanged since the previous revision: `connito/miner/`, `dataloader.py`,
`expert_manager.py`, `modeling/`, and `expert_groups/` — so tier 0, the miner
switch path, and the whole model-rebuild rule in §3 carry over verbatim.
