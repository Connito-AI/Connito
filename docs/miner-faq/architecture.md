# Connito subnet architecture (miner's view)

How the subnet runs end-to-end: who does what, how a cycle moves, and where the
code lives. Written for a miner running a GPU box, not a contributor. Plain
English first; code pointers at the end of every section so you can verify
against the source.

## The cycle

The subnet operates on a fixed-length cycle measured in chain blocks (12 s per
block). Every cycle walks through nine named phases in this order:

```
Distribute → Train → MinerCommit1 → MinerCommit2 → Submission
          → Validate → Merge → ValidatorCommit1 → ValidatorCommit2
```

The phase boundary is enforced by chain block height — not wall-clock. Both
miners and validators block on `wait_till(config, <phase_name>)`, which polls
the owner phase service (`https://cycle-api.connito.ai/get_phase`) and sleeps
until the requested phase starts. The phase lengths are served by that API;
ask `/get_phase` rather than computing them yourself. After every cycle the
loop restarts at `Distribute`.

A miner only acts during three phases: **Distribute** (download the round's
published winner, if any), **Train** (train the task's residual experts on top
of the frozen base model), and **MinerCommit1 / MinerCommit2** (commit the
trained shard to chain and HuggingFace). Everything else belongs to
validators.

Code: `connito/shared/cycle.py:PhaseManager`,
`connito/shared/cycle.py:wait_till`, `connito/shared/cycle.py:PhaseNames`.

## Validator vs. miner responsibilities

**Miner** (one GPU, no peering):

1. **Distribute** — read the validators' chain commits from last cycle. If
   they advertise a published round winner, download that residual shard
   (`model_expgroup_{group_id}.safetensors`) from the HF repo@revision they
   committed, check it, and restart training from it. The frozen base model,
   `Qwen/Qwen3.6-35B-A3B` at a pinned revision, is loaded from the local
   Hugging Face cache at startup and never comes from a validator.
2. **Train** — run AdamW steps on the residual experts and their routers
   only. Every base weight stays frozen.
3. **MinerCommit1** — sign the freshly trained shard's hash with the miner's
   hotkey and commit `signed_model_hash` to chain.
4. **MinerCommit2** — upload the shard to the miner's HF repo, then commit
   `(model_hash, global_ver, expert_group, hf_repo_id, hf_revision)` to
   chain. After this, the miner is "submitted" for the cycle.

Code: `connito/miner/train.py:train_worker`,
`connito/miner/model_io.py:commit_worker`,
`connito/miner/model_io.py:_upload_checkpoint_to_hf_safe`.

**Validator** (one GPU, archive subtensor):

1. **Submission** — freeze the round (build the eval roster), download each
   assigned miner's HF shard, verify it, and score it: load the miner's
   residual onto the frozen base, measure `val_loss` on the round's task eval
   split and the loss on a c4 replay split, and compute the round score: the
   task gain over the pretrained base minus penalties for a c4 rise and for
   where the residual's gate fires. A c4 rise above a cutoff scores `−inf`.
2. **Validate** — persist the score history. The round's scores are finalized
   (rank-based, top-3 only) 5 blocks before the next cycle's MinerCommit1,
   just before weights are submitted.
3. **Merge** — nothing is merged. The phase stays in the schedule because the
   phase API owns its boundaries; validators do no work in it.
4. **ValidatorCommit1 / ValidatorCommit2** — commit the hash and HF
   coordinates of the round winner this validator published, so the next
   cycle's miners know where to download from. With no winner, the commit
   carries a hash and no HF coordinates.
5. **5 blocks before the next MinerCommit1** — finalize the round's scores,
   publish the winner's shard to HF, and submit chain weights derived from the
   rolling-average score history.

Code: `connito/validator/run.py` (the single-loop main function),
`connito/validator/evaluator.py:evaluate_one_miner_sync`,
`connito/validator/round.py:Round.freeze`,
`connito/validator/distribute.py:publish_round_baseline`.

## Residual experts: what miners train

The base model is `Qwen/Qwen3.6-35B-A3B` at revision
`995ad96eacd98c81ed38be0c5b274b04031597b0`, used as a text-only language
model: 40 layers, each with a mixture-of-experts (MoE) block. **The whole base
is frozen.** Miners train small **residual experts** that sit beside every MoE
block and add to its output:

    h = h_base(x) + Σ_j q_j(x) · E_j(x),   q_j = sigmoid((sigmoid(w_j · x) − A) / τ)

- Each residual expert `E_j` is a small SwiGLU MLP (2048 → 512 → 2048) with its
  own router row `w_j`. The gate `q_j` decides, per token, how much the expert
  contributes.
- **K** (experts per block), **A** (gate threshold) and **τ** (gate
  temperature) are set by the task, not by the miner. The task
  `exp_pmc_residual` uses K = 1, A = 0.8, τ = 0.05.
- A fresh residual starts with `down_proj` = 0, so at step 0 the model is
  exactly the pretrained base.

**What is in the shard** (`model_expgroup_{group_id}.safetensors`), for every
layer L in 0–39 and expert j in 0..K−1:

- `model.layers.{L}.mlp.residual_experts.{j}.{gate,up,down}_proj.weight` in
  bf16;
- `model.layers.{L}.mlp.residual_router.weight`, shape [K, 2048], in fp32.

At K = 1 that is 160 tensors and about 252 MB. Nothing else may be in the file:
a validator refuses any shard whose keys, shapes or dtypes differ from this
spec (eval status 14, `bad_residual_shard`).

**The task is served, not chosen.** The owner's phase service tells every node
which task is active; its `group_id` (9 for `exp_pmc_residual`) names the
shard file and goes into every chain commit. Validators only score miners
whose commit carries their own `group_id`, so a miner on another group is
invisible to them. `exp_pmc_residual` is also the built-in fallback when the
API cannot be reached.

Code: `connito/shared/modeling/residual.py:install_residual_experts`,
`connito/shared/modeling/residual.py:residual_shapes`,
`connito/shared/config.py:TaskCfg`, `connito/shared/config.py:ResidualCfg`,
`connito/shared/task_sync.py:sync_active_task`,
`connito/shared/cycle.py:get_miners_from_commit`,
`expert_groups/exp_pmc_residual/config.yaml`.

## Round baseline flow

Each round, validators pick a winner, and every miner restarts from the
winner's residual. Its path through the system:

1. **The validator picks a winner** when it finalizes the round. Only a miner
   with a round score above 0 can win: better than the pretrained base after
   the c4 and gate-usage penalties, which also means it passed the c4 cutoff.
   Among those, a miner with a top-3 track record is preferred; otherwise the
   highest score wins. If no miner qualifies, nothing is published.
2. **The validator publishes it** to its own HF repo, then commits the shard's
   hash and the HF `(repo_id, revision)` to chain in ValidatorCommit1/2. The
   revision is truncated to 7 characters to fit the 128-byte chain commit.
   With no winner, it commits a hash with no HF coordinates, so miners have
   nothing to fetch.
3. **The miner downloads it** in the next Distribute, verifies the bytes
   against the committed hash, and checks the shard against the residual spec.
   A shard that passes becomes the miner's starting point.

**Where a miner starts when nothing is published** (the first round of a task,
or a round where nobody beat the base): it keeps its own last trained
checkpoint, or, with none, its own freshly drawn residual. A miner never
starts from an untrained residual (all-zero `down_proj`, or gate and up both
zero), because such a residual gets no gradient; the miner refuses one as a
start just as validators refuse it as a submission.

Code: `connito/validator/round.py:select_baseline_uid`,
`connito/validator/distribute.py:publish_round_baseline`,
`connito/validator/distribute.py:unadvertised_model_hash`,
`connito/shared/model.py:fetch_model_from_chain_validator`,
`connito/shared/model.py:get_model_from_checkpoint`,
`connito/shared/checkpoint_helper.py:verify_residual_shard`.

## Baseline model and eval data

The base model is `Qwen/Qwen3.6-35B-A3B` at the pinned revision
`995ad96eacd98c81ed38be0c5b274b04031597b0`. Every miner and validator loads it
from its own Hugging Face cache at startup; it is never part of a miner's
shard. The checkpoint is 67 GiB (72 GB) on disk.

The task `exp_pmc_residual` trains on two streaming datasets:

- `TomTBT/pmc_open_access_section` (config `commercial`), sampling weight 0.8:
  the task data;
- `allenai/c4` (config `en`), sampling weight 0.2, marked `replay: true`:
  general text that keeps the residual from drifting.

Validators score each miner twice per round, on the same loaded model:

- **The task eval** uses the non-replay sources only (pmc). Its `val_loss`
  against the base's gives your task gain.
- **The c4 eval** uses the replay source, sized by real tokens (65,536 per
  round). A c4 rise is subtracted from your score, and a rise above the
  cutoff (0.02 by default) scores `−inf`.
- Both evals also measure how much your residual's gate fires; the score
  asks for about 0.2 on task tokens and as little as possible on c4. See
  `scoring.md`.

Both datasets are pinned to a commit SHA (`eval_source_revision_pin`), so every
validator reads identical rows. The eval slice is chosen by **seeded
shard-pick**: the round's seed selects which shard to open and a hash-derived
offset into it, so no fixed window can be memorized. Rows shorter than 200
characters are dropped, and rows repeating an already-seen 200-character
prefix are skipped. Sequences are 4096 tokens long.

All validators in a cycle use the same `combined_seed`, derived from the block
hash of the last MinerCommit2 block, so scores are reproducible across
validators but unpredictable to miners during their commit window.

Code: `connito/shared/cycle.py:get_combined_validator_seed`,
`connito/shared/dataloader.py:get_dataloader`,
`connito/shared/dataloader.py:materialize_by_tokens`,
`connito/shared/eval_shard_pick.py:pick_shard_for_source`,
`expert_groups/exp_pmc_residual/config.yaml` (dataset definition).

## What miners can and cannot influence

Miners control: the trained residual uploaded each cycle, **which data they
train on and in what order** (varying it every round matters; see
`miner-config.md`, "Vary your training data every round"), their HF repo
destination, and training settings such as the learning rate,
`local_par.micro_batch_size`, `local_par.gradient_accumulation_steps` and
`model.quantization_miner`.

Miners do **not** control: the task, its K / A / τ and its `group_id` (served by
the owner API), the base model, the cycle phase boundaries (chain-driven), the
eval data slice (seeded from chain state validators cannot forge), which
validator scores them in a given cycle (seeded partition), the score's
weights, gate-usage target and c4 cutoff (locked validator config), or the
rank → score mapping that converts a round score into a chain weight
(hard-coded; see `scoring.md`).

Code: `connito/shared/cycle.py:assign_miners_to_validators`,
`connito/validator/evaluator.py:_RANK_TO_SCORE`.
