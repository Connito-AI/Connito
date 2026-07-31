# Connito subnet architecture (miner's view)

How the subnet runs end-to-end: who does what, how a cycle moves, and where the
code lives. Written for a miner running a GPU box, not a contributor. Plain
English first; code pointers at the end of every section so you can verify
against the source.

## The cycle

The subnet operates on a fixed-length cycle (default **448 blocks ≈ 1.5 hours**
at 12 s/block). Every cycle walks through nine named phases in this order:

```
Distribute → Train → MinerCommit1 → MinerCommit2 → Submission
          → Validate → Merge → ValidatorCommit1 → ValidatorCommit2
```

The phase boundary is enforced by chain block height — not wall-clock. Both
miners and validators block on `wait_till(config, <phase_name>)`, which polls
the owner phase service (`https://cycle-api.connito.ai/get_phase`) and sleeps
until the requested phase starts. After every cycle the loop restarts at
`Distribute`.

A miner only acts during three phases: **Distribute** (download the latest
global model), **Train** (run local training on its expert group), and
**MinerCommit1 / MinerCommit2** (commit the trained checkpoint to chain and
HuggingFace). Everything else belongs to validators.

Code: `connito/sn_owner/cycle.py:PhaseManager`,
`connito/shared/cycle.py:wait_till`, `connito/shared/cycle.py:PhaseNames`.

## Validator vs. miner responsibilities

**Miner** (one GPU, no DHT, no peering):

1. **Distribute** — pull the latest global checkpoint from the HF repo whose
   coordinates the active validators committed to chain last cycle. The
   `model_expgroup_{N}.safetensors` shard for the miner's own expert group is
   the only file fetched; the backbone is rebuilt from
   `deepseek-ai/DeepSeek-V2-Lite`.
2. **Train** — run AdamW inner-optimizer steps on the miner's expert-group
   parameters only. Non-expert parameters stay frozen.
3. **MinerCommit1** — sign the freshly trained checkpoint's hash with the
   miner's hotkey and commit `signed_model_hash` to chain.
4. **MinerCommit2** — upload the checkpoint shard to the miner's HF repo,
   then commit `(model_hash, global_ver, expert_group, hf_repo_id,
   hf_revision)` to chain. After this, the miner is "submitted" for the
   cycle.

Code: `connito/miner/train.py:train_worker`,
`connito/miner/model_io.py:commit_worker`,
`connito/miner/model_io.py:_upload_checkpoint_to_hf_safe`.

**Validator** (one GPU, DHT peer, archive subtensor):

1. **Submission** — freeze the round (build the eval roster), download each
   assigned miner's HF shard, and run foreground evaluation: compute
   `val_loss` on a held-out slice and compare to the validator's own
   baseline.
2. **Validate** — finalize per-miner scores (rank-based, top-3 only) and
   write zero-scores for missing or invalid submissions.
3. **Merge** — average the top-K miners' gradients into the global model
   via Hivemind `DecentralizedAverager`, then run an outer SGD step.
4. **ValidatorCommit1 / ValidatorCommit2** — upload the new global model to
   HF and commit the repo+revision to chain so the next cycle's miners
   know where to download from.
5. **Post-cycle** — submit chain weights derived from the rolling-average
   score history.

Code: `connito/validator/run.py` (the single-loop main function),
`connito/validator/evaluator.py:evaluate_one_miner_sync`,
`connito/validator/round.py:Round.freeze`.

## Expert-group sharding

The base model (DeepSeek-V2-Lite) has 64 routed experts per MoE layer, indexed
0–63, on MoE layers 1–26 (layer 0 is dense). Connito assigns a subset of those
experts to each **worker group**. A miner trains exactly one group's experts and
uploads only that group's shard. The active group is **`exp_nemotron_c4`**
(`group_id = 4`); other directories under `expert_groups/` are inactive or serve
as frozen helpers.

The active group is **not** a free choice. `config.task.expert_group_name` is a
locked field: loading a config with `auto_update_config` resets any other value
back to the built-in default and rewrites it to your YAML, logging a one-time
reset warning. Setting it to something else does not switch groups — it is
overwritten on the next load. This is deliberate, because validators only
evaluate miners whose group matches their own.

The validator filters all chain commits by `expert_group == config.task.exp.group_id`
when building its roster, so a validator on `exp_nemotron_c4` (`group_id = 4`)
only evaluates miners that also committed under `group_id = 4`. A miner still
committing under a previous group is invisible to the fleet and scores nothing —
so group changes are coordinated subnet-wide, announced in advance, and take
effect when you upgrade to the release that carries them.

Code: `connito/shared/expert_manager.py:ExpertManager.load_expert_group_assignment`,
`connito/shared/config.py:TaskCfg` (`_LOCKED_FIELDS`),
`connito/shared/cycle.py:get_miners_from_commit`,
`expert_groups/exp_nemotron_c4/expert_assignment.json`.

## Foundation / global checkpoint flow

The "global checkpoint" is the merged model that validators produce at the
end of each cycle. Its path through the system:

1. **Validator builds it** during the Merge phase by averaging top-K miner
   gradients and running an outer SGD step on the previous cycle's model.
2. **Validator uploads it** to its own HF repo during ValidatorCommit1, then
   commits the `(hf_repo_id, hf_revision)` short SHA to chain in
   ValidatorCommit2. The revision is truncated to 7 characters to fit the
   128-byte chain commit budget.
3. **Miner downloads it** during the next cycle's Distribute by reading
   validator commits from chain, picking one whose `signed_model_hash` is
   valid and whose `global_ver` matches what other validators advertised,
   and pulling `model_expgroup_{N}.safetensors` from `repo@revision`.

The model_hash on chain is verified against the downloaded bytes — a
tampered file is rejected before training resumes.

Code: `connito/validator/run.py` (ValidatorCommit1/2 block, ~line 929-1007),
`connito/shared/model.py:fetch_model_from_chain_validator`,
`connito/shared/hf_distribute.py:download_checkpoint_from_hf`.

## Baseline model and eval data

The backbone architecture is `deepseek-ai/DeepSeek-V2-Lite` and is loaded
fresh on every miner and validator start — it is never serialized into the
miner's HF shard.

Validators evaluate miners against two HuggingFace streaming datasets, mixed
50/50:

- `allenai/c4` (config `en`)
- `nvidia/Nemotron-CC-Math-v1` (config `4plus`)

**`nvidia/Nemotron-CC-Math-v1` is a gated dataset.** Your `HF_TOKEN` must
belong to a HuggingFace account that has accepted the license on that dataset's
page. Dataset metadata is readable without it, so the failure does not appear at
startup — file reads fail later with a `GatedRepoError` (HTTP 403) when the
dataloader is built. If your miner cannot stream training data, check this
first. Both datasets are pinned to a specific commit SHA via
`eval_source_revision_pin` so every validator reads identical rows.

The eval slice is chosen by **seeded shard-pick**: the round's seed selects
which shard to open and a hash-derived offset into that shard, so across
rotating seeds every shard and every row is eventually reachable and no fixed
window can be memorized. Rows shorter than 200 characters are dropped and rows
repeating an already-seen 200-character prefix are skipped, so duplicated
boilerplate cannot inflate a score. Documents longer than the 1024-token
sequence length contribute a content-hash-derived window rather than always
their opening tokens.

All validators in a cycle use the same `combined_seed`, derived from the block
hash of the last MinerCommit2 block, so scores are reproducible across
validators but unpredictable to miners during their commit window.

Code: `connito/shared/cycle.py:get_combined_validator_seed`,
`connito/shared/dataloader.py:get_dataloader`,
`connito/shared/eval_shard_pick.py:pick_shard_for_source`,
`connito/shared/dataloader.py:tokenize_windowed`,
`expert_groups/exp_nemotron_c4/config.yaml` (dataset definition).

## What miners can and cannot influence

Miners control: the trained weights uploaded each cycle, the expert group they
target (via local config), their HF repo destination, training
hyperparameters (learning rate, batch size, accumulation steps).

Miners do **not** control: the cycle phase boundaries (chain-driven), the
eval data slice (seeded from chain state validators cannot forge), which
validator scores them in a given cycle (seeded round-robin partition), or the
rank → score mapping that converts a `val_loss` into a chain weight (hard-coded
geometric progression — see `scoring.md`).

Code: `connito/shared/cycle.py:assign_miners_to_validators`,
`connito/validator/evaluator.py:_RANK_TO_SCORE`.
