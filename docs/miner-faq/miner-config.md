# Miner configuration

The miner config lives in a YAML file passed via `--path` to
`python -m connito.miner.model_io` (and `connito.miner.train`). The settings
below are the ones that actually affect whether your miner gets evaluated
and rewarded. Everything else is either a derived path or a locked field
that you should not touch — `check_and_prompt_locked` resets a locked field
to the release default on the next startup anyway.

The class definition: `connito/shared/config.py:MinerConfig`.

## Identity (must be set)

```yaml
chain:
  coldkey_name: my-coldkey       # bittensor wallet coldkey
  hotkey_name:  my-hotkey        # bittensor wallet hotkey
  netuid:       102              # LOCKED — do not change
  network:      archive          # LOCKED — needed for chain-commit history
```

`hotkey_ss58` and `coldkey_ss58` are filled in by `_fill_wallet_data` on
first run by reading the wallet files; you do not need to set them.

Code: `connito/shared/config.py:ChainCfg`.

## Task (served, not chosen)

```yaml
task:
  expert_group_name: exp_pmc_residual   # set from the owner API at startup
  # exp.group_id and exp.residual are loaded from expert_groups/<name>/config.yaml
```

- **The active task comes from the owner's phase service**, not from your
  YAML. At startup, and again at each Distribute, the miner asks the API which
  task is active, fetches its bundle (checked against a hash) into
  `expert_groups/<name>/`, and loads it. The value in your YAML is only the
  fallback used when the API is unreachable.
- The task's `config.yaml` defines its `group_id` (9 for `exp_pmc_residual`),
  its datasets, and its residual settings: K experts per MoE block, gate
  threshold A and temperature τ (`residual: {k: 1, threshold: 0.8,
  temperature: 0.05}`).
- **A miner committing under a different group is never evaluated.** The
  validator filters its roster by `expert_group == config.task.exp.group_id`,
  so a stale group means no evaluation and no rewards — not a partial score.
  A miner that follows the API switches tasks together with the validators.
- Changing tasks **does not** reset your aggregator history if the hotkey
  stays the same, but your accumulated scores age out of the rolling window
  within a few rounds because the loss scale on a new dataset differs.

Code: `connito/shared/config.py:TaskCfg`, `connito/shared/config.py:ResidualCfg`,
`connito/shared/task_sync.py:sync_active_task`,
`connito/shared/cycle.py:get_miners_from_commit`.

## HuggingFace (must be writable; validators pull from here)

```yaml
hf:
  checkpoint_repo: my-org/my-miner-repo   # optional; if unset, derived as {hf_user}/co
  default_repo_name: co                   # used only if checkpoint_repo is unset
  token_env_var: HF_TOKEN                 # validator/miner read HF_TOKEN env var
```

Environment requirement:

```bash
export HF_TOKEN="hf_..."                  # must have WRITE access to checkpoint_repo
```

- The HF repo must be **public** (or accessible to every validator's HF
  token — in practice that means public). Validators only have read access
  with the stock `HF_TOKEN`.
- `checkpoint_repo` is capped at **32 characters**
  (`CHAIN_COMMIT_MAX_HF_REPO_ID_CHARS`) because the full repo string has to
  fit in the 128-byte chain commit alongside other fields. A longer repo id
  will raise `ValueError` at commit time.

Code: `connito/shared/config.py:HfCfg`,
`connito/shared/chain.py:CHAIN_COMMIT_MAX_HF_REPO_ID_CHARS`,
`connito/miner/model_io.py:_upload_checkpoint_to_hf_safe`.

## Checkpoint format: `.safetensors` only

A submission is one file, `model_expgroup_{group_id}.safetensors`, holding
exactly the task's residual experts and routers:

- `model.layers.{L}.mlp.residual_experts.{j}.{gate,up,down}_proj.weight`,
  bf16, for every layer L in 0–39 and expert j in 0..K−1;
- `model.layers.{L}.mlp.residual_router.weight`, shape [K, 2048], fp32.

At K = 1 that is 160 tensors, about 252 MB. The stock miner writes it with
`save_residual_shard` and uploads only that file.

**Validators refuse anything else** with eval status 14,
`bad_residual_shard`:

1. **A `.pt` file** — a `.pt` is a Python pickle and is never loaded.
2. **Any other key, shape or dtype** — extra tensors (base weights, old
   expert-group `mlp.experts.N.*` keys), missing tensors, a bf16 router, or a
   shard built for another K. This is checked from the file header, before
   any tensor is read.
3. **A NaN or Inf tensor.**
4. **An untrained expert** — `down_proj` all zero, or gate and up both all
   zero. A residual saved before the first optimizer step is refused.

The same check runs on the miner side: a miner never starts training from a
local checkpoint or downloaded winner that fails it.

Code: `connito/shared/checkpoint_helper.py:save_residual_shard`,
`connito/shared/checkpoint_helper.py:verify_residual_shard`,
`connito/shared/modeling/residual.py:residual_shapes`,
`connito/shared/helper.py:expert_group_shard_name`.

## Hardware and environment

The miner trains the residual on top of `Qwen/Qwen3.6-35B-A3B` (pinned
revision `995ad96eacd98c81ed38be0c5b274b04031597b0`), with the base's routed
experts stored in fp8 and every base weight frozen. Measured at the default
settings (sequence length 4096, micro-batch 2 × accumulation 2):

- **GPU: one 48 GB card.** On an RTX A6000, a 300-step run peaked at
  39.5 GiB allocated / 41.2 GiB reserved, and the miners in a full rehearsal
  at 39.53 GiB allocated / 39.8–40.0 GiB reserved. The L40S (44.4 GiB usable)
  is the measured floor: micro-batch 2 fits there. One optimizer step (4 rows,
  16k tokens) takes about 12.5 s on an A6000.
- **Only one GPU.** The model must sit on a single GPU. Split over three GPUs
  with transformers `device_map="auto"`, Qwen returned wrong logits with no
  error (loss 12.42 instead of 2.04), so the loader refuses a model spread
  over more than one device (`the base model is split over devices …`).
- **Host RAM:** a dedicated 62 GB box works. Loading peaked at about 33 GiB of
  anonymous memory. Resident memory (RSS) read 62–63 GB, but most of that is
  the memory-mapped checkpoint, which the kernel can drop under pressure.
- **Disk:** the checkpoint is 67 GiB (72 GB) in your Hugging Face cache.

Plan for about 100 GB of free disk for the model: a margin over the 67 GiB
checkpoint, not a measurement.

The settings that move GPU memory:

```yaml
local_par:
  micro_batch_size: 2              # rows per forward; lower to 1 on OOM
  gradient_accumulation_steps: 2   # micro-batches per optimizer step
model:
  quantization_miner: all          # "all" (default) or "experts"
```

- **On OOM, lower `local_par.micro_batch_size`**, and raise
  `gradient_accumulation_steps` to keep 4 rows per step. Do not change
  `sequence_length`: it is locked at the task's 4096.
- `model.quantization_miner: experts` stores only the routed experts in fp8
  and keeps every other base weight in bf16; it
  uses about 1.3 GiB more GPU memory than `all` (34.68 vs 33.37 GiB after
  load) and fits micro-batch 2 on the L40S. It is not locked: it only affects
  your own box.
- Training runs in bf16 only.
- **Upgrading an old YAML:** a config written before `micro_batch_size`
  existed sets `gradient_accumulation_steps` to 4. On load that value is dropped
  with the warning "Ignoring local_par.gradient_accumulation_steps from before
  micro_batch_size", so you get 2 × 2. Add `local_par.micro_batch_size` to
  keep your own accumulation.

Code: `connito/shared/config.py:ParallelismCfg`,
`connito/shared/config.py:ModelCfg`, `connito/miner/train.py:setup_training`,
`connito/shared/modeling/mycelia.py:get_base_model`.

## Checkpoint persistence

```yaml
ckpt:
  base_checkpoint_path: checkpoints/miner   # derived: root_path / this / coldkey / hotkey / run_name
  checkpoint_topk: 2                        # how many local checkpoints to keep
  checkpoint_interval: 20                   # save every N inner_opt_step (default: 20% of global_opt_interval)
  resume_from_ckpt: true                    # LOCKED — resume from a local checkpoint on restart
```

The miner saves local checkpoints during Train at every
`checkpoint_interval` inner_opt_step. Only the most recent
`checkpoint_topk` are kept on disk. Each holds the residual shard only (about
252 MB at K = 1), not the base model.

On restart the miner walks its local checkpoints and downloaded winners,
newest first, and resumes from the first one that passes the residual check.
If none does, it keeps a freshly drawn residual and logs "Keeping the fresh
residual: no local checkpoint is a start".

The path that ends up serving MinerCommit2 is `select_best_checkpoint(
config.ckpt.checkpoint_path)` — the newest local checkpoint. If you delete
the local checkpoint dir between Train and MinerCommit2, the commit phase
will fail with `FileNotReadyError`.

Code: `connito/shared/config.py:CheckpointCfg`,
`connito/shared/checkpoint_helper.py:save_checkpoint`,
`connito/shared/checkpoints.py:select_best_checkpoint`,
`connito/shared/model.py:get_model_from_checkpoint`.

## Cycle / phase API endpoint

```yaml
cycle:
  owner_url:        https://cycle-api.connito.ai:443     # LOCKED
  api_timeout_sec:  10
  api_retries:      5
  api_backoff_sec:  2
  version_range_cycles: 3                                # how old a validator checkpoint can be and still be accepted
```

The `owner_url` is hard-pinned. Do not point this at a different host
unless you are running a private testnet — the miner will refuse commits
that don't agree with the chain-side phase boundaries enforced by all
other workers. The phase lengths are served by this API; the
`cycle.*_period` and `cycle.cycle_length` fields in your YAML are locked
fallback defaults and do not set the real schedule.

If you see HTTP 403 from `cycle-api.connito.ai`, see
`troubleshooting.md` — it is almost always a Cloudflare WAF rule.

Code: `connito/shared/config.py:CycleCfg`.

## Common pitfalls

1. **Wallet permission.** Miners sign and commit using the **hotkey**;
   the coldkey is only for registration. Make sure
   `~/.bittensor/wallets/<coldkey>/hotkeys/<hotkey>` exists and is readable
   by the user running the miner.
2. **Stale checkpoints in `checkpoints/`.** When the cycle dies mid-Train,
   the local checkpoint dir can keep checkpoints from a previous run.
   `_prepare_checkpoint_for_commit` picks `select_best_checkpoint`, which
   orders by `global_ver` then by `inner_opt`, so a stale checkpoint with a
   higher `global_ver` than your current training run will be committed
   instead of the freshly trained one. A checkpoint from another task (a
   different K) is refused by validators as `bad_residual_shard`. Solution:
   clear the local checkpoint dir between runs if you reset training.
3. **HF private repo.** If `hf.checkpoint_repo` is private, validators cannot
   download your shard, and you are never scored. If the repo or revision is
   gone for good, you get rank score 0 for the round (`repo_unavailable`).
   Make the repo public.
4. **Run name reuse.** `run_name` is part of the local checkpoint path. If
   you rename the run between cycles, miner restart will not find the
   previous checkpoint and will start from the published winner or a fresh
   residual instead of your own work.
5. **A Python without headers.** If `Python.h` is missing, the fla Triton
   kernels cannot build and training silently falls back to a slower path.
   See "Software environment".

## Telemetry / observability

```yaml
log:
  log_wandb: false
  metric_path: metrics/<run_name>.csv
  metric_interval: 20             # default: 20% of global_opt_interval
```

Miners expose a Prometheus endpoint at port `8100 + rank` (so rank 0 → port
8100). Default metrics include `miner_training_loss`, `miner_gradient_norm`,
`miner_learning_rate`, `miner_local_step_rate`, `miner_tokens_per_sec`,
`miner_perplexity` and `miner_param_sum` (the sum of the residual
parameters, which moves while training works). Use these to confirm training
is actually progressing before debugging score issues.

Code: `connito/shared/telemetry.py` (the `MINER_*` gauges),
`connito/shared/metrics.py:MetricLogger`.

## Vary your training data every round

**The stock miner trains on the same data every round, and that trips the c4
cutoff.** It builds its training loader with no seed, so the stream starts at
the same rows in the same order every time. It rebuilds the loader at every
`setup_training`, including the restart from each round's published winner,
so each round replays the same data. And every one-GPU miner reads the same
slice of it (rank 0 of the task's 10 data shards), so every miner on the stock
code sees identical batches.

**What that does, measured** (two miners, both starting from round 1's
published winner, scored on the same eval set, under the scoring of that run:
task gain plus the c4 cutoff, before the gate-usage penalties):

| training data | task loss vs start | c4 loss vs base | result |
|---|---|---|---|
| stock (same stream again) | worse | +0.064 / +0.066 | both stopped by the c4 cutoff (status 15), nothing published |
| each source shuffled per (hotkey, round) | −0.0014 / −0.0047 | −0.024 / −0.025 | cutoff passed, winner published |

In a full two-round rehearsal on the stock code, round 2 went the same way:
c4 loss +0.061 / +0.059, both miners stopped (status 15, `replay_loss_increase`),
nothing published.

The cutoff is what varied data protects you from. Under the current score,
passing it is necessary but not enough: the score also subtracts gate-usage
penalties, and reference-miner shards have measured below 0 even with fresh
data (see `scoring.md`, "Gate usage: why a better loss can still score below
0"). With repeated data, 13–28% of micro-batches were also
skipped as non-finite; the shuffled run skipped none.

**What to do.** Give each round, and each hotkey, its own data order. The
measured fix shuffles every streaming source with a seed derived from the
hotkey and the round before the sources are interleaved, e.g.
`ds.shuffle(seed=<sha256 of "hotkey:round">, buffer_size=1000)`. Any round
identifier works, such as the Train phase's `phase_start_block`. The place
to do it is where each source is loaded:
`connito/shared/dataloader.py:DefaultStreamingTorchDataset.get_tokenised_dataset`,
before `interleave_datasets`. You can also use other data from the task's
domain; validators only score the result.

**Do not pass `seed=` to `get_dataloader` for training.** A seed switches the
loader to the validators' eval path (seeded shard-pick plus a fractional
subsample). In the same experiment that left about 223 training sequences,
and the miners ran out of data within about 20 minutes.

**Two reference miners score 0 together.** Two miners running the stock code
from the same published winner read identical batches. If they also run the
same number of steps, they produce the same shard and exactly equal scores,
and `finalize_round_scores` gives an exact tie 0 for both, as a duplicate
submission.

Code: `connito/miner/train.py:setup_training`,
`connito/shared/dataloader.py:get_dataloader`,
`connito/validator/evaluator.py:finalize_round_scores`.

## Software environment

The model needs a recent CUDA stack and a fresh Python environment; an
environment built for transformers 4.x cannot load it.

- **NVIDIA driver ≥ 570** (CUDA 12.8 wheels).
- **Python 3.10 with its headers (`Python.h`).** The model's Gated DeltaNet
  layers run on flash-linear-attention's Triton kernels, which compile at
  first use. Without the headers, fla only prints the warning "Triton is not
  supported on current platform, roll back to CPU" and training falls back to
  a slower path. A uv-managed Python ships the headers; a system Python needs
  its `-dev` package (on Ubuntu 22.04, `python3.10-dev`).
- **A new virtual environment**, installed from the repo's pins:
  ```bash
  uv venv --seed --managed-python -p 3.10 .venv && source .venv/bin/activate
  pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
  ```
  `--managed-python` makes uv use its own Python build, which ships the
  headers, even when a system Python 3.10 is installed. On a uv release
  without that flag, run `uv python install 3.10` first and pass
  `--python-preference only-managed` instead.
  This brings torch 2.10.0 (cu128), transformers 5.18.0, huggingface_hub 1.x,
  datasets 4.8.5, safetensors 0.8.0 and flash-linear-attention 0.5.2.
- **Pre-download the model** at the pinned revision, so the first start does
  not spend a round downloading 72 GB:
  ```bash
  hf download Qwen/Qwen3.6-35B-A3B --revision 995ad96eacd98c81ed38be0c5b274b04031597b0
  ```
  Set `HF_HOME` to the cache the miner uses. To remove a model, run
  `hf cache rm model/<org>/<name>` against the same cache (`hf cache ls`
  lists the ids; this one is `model/Qwen/Qwen3.6-35B-A3B`), or delete the
  whole cache directory. Remove the previous release's base model the same
  way once you have upgraded. Deleting only the model's
  folder does not free the disk: its files stay in the shared `blobs/`
  directory.

Code: `requirements.txt`, `connito/shared/modeling/families.py` (model id and
revision).
