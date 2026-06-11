# Miner configuration

The miner config lives in a YAML file passed via `--path` to
`python -m connito.miner.model_io` (and `connito.miner.train`). The settings
below are the ones that actually affect whether your miner gets evaluated
and rewarded. Everything else is either a derived path or a locked field
that you should not touch — the validator's `check_and_prompt_locked` will
reset it on next startup anyway.

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

## Expert group (the most consequential miner choice)

```yaml
task:
  expert_group_name: exp_math    # or exp_dummy — must match a dir under expert_groups/
  # exp.group_id is loaded from expert_groups/<expert_group_name>/config.yaml
```

- The directory under `expert_groups/` defines the `group_id`, the dataset,
  and the routing assignment.
- **Choose a group where at least one validator is running.** A miner
  committing to a group no validator covers will never be evaluated and
  will never earn rewards. The current production groups are
  `exp_math` (`group_id = 0`) and `exp_dummy` (`group_id = 1`).
- Switching expert groups **does not** reset your aggregator history if the
  hotkey stays the same — but the validator only evaluates miners whose
  chain-committed `expert_group` matches its own active group, so a config
  switch only takes effect after MinerCommit2 in the cycle of the change.

Code: `connito/shared/config.py:TaskCfg`,
`connito/shared/expert_manager.py:ExpertManager.load_expert_group_assignment`,
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
- `checkpoint_repo` is capped at **32 characters** (`CHAIN_COMMIT_MAX_HF_REPO_ID_CHARS`)
  because the full repo string has to fit in the 128-byte chain commit
  alongside other fields. A longer repo id will raise `ValueError` at
  commit time.

Code: `connito/shared/config.py:HfCfg`,
`connito/shared/chain.py:CHAIN_COMMIT_MAX_HF_REPO_ID_CHARS`,
`connito/miner/model_io.py:_upload_checkpoint_to_hf_safe`.

## Checkpoint format: `.safetensors` vs `.pt`

```python
# connito/shared/helper.py
MINER_CHECKPOINT_SUFFIXES: tuple[str, ...] = (".safetensors", ".pt")
```

**You should be uploading `.safetensors`.** The reasons:

1. **Security on the validator side.** `.pt` is a Python pickle and could
   execute arbitrary code on load. Validators load `.pt` with
   `torch.load(..., weights_only=True)` to mitigate this, but that gates
   off some legitimate `.pt` files too (anything wrapping the
   state_dict in a non-trivial container).
2. **The default miner upload path emits both.** Stock
   `_upload_checkpoint_to_hf_safe` uploads
   `model_expgroup_{N}.safetensors` and `model_expgroup_{N}.pt` if both
   exist in the checkpoint directory. New checkpoints saved with
   `save_checkpoint(save_model_by_expert_group=True)` should be
   `.safetensors`.
3. **The `.pt` 404 trap.** When the *validator* uploads a global
   checkpoint, it currently writes `.safetensors`. If your miner downloader
   is hardcoded to fetch `model_expgroup_{N}.pt`, the download will 404
   and the miner trains from the base architecture every cycle. The fix
   `connito/shared/model.py:_build_download_targets` should be using
   `.safetensors` filenames; if you forked the miner, double-check this.

> Lultime's Discord note on the `.pt` vs `.safetensors` issue: validator
> uploads switched to `.safetensors` first; miners on the old downloader
> looking for `.pt` started getting `404` from HF and falling back to the
> base architecture. The signal in the logs is "Starting batch loss" near
> 20 instead of <1 (untrained base) and a `download` HTTP 404 in the
> Distribute phase. See `troubleshooting.md` entry on "Miner trains from
> base every cycle."
>
> **TODO: code in
> `connito/shared/model.py:_build_download_targets` still uses
> `f"model_expgroup_{expert_group_id}.pt"` (line 51). Whether the validator
> upload path now consistently produces `.safetensors` is worth
> double-checking against current production logs before recommending
> miners hard-pin `.safetensors`.**

Code: `connito/shared/helper.py:MINER_CHECKPOINT_SUFFIXES`,
`connito/shared/helper.py:load_state_dict_from_path`.

## GPU expectations

Hardcoded in `connito/miner/train.py:train_worker`:

```yaml
model:
  precision:        fp16-mixed       # bf16-mixed is supported but falls back to fp16 if BF16 not available
  attn_implementation: sdpa
  torch_compile:    false            # do not enable; not tested
local_par:
  gradient_accumulation_steps: 4
  global_opt_interval: 100
  world_size: 1                      # always 1 for now — single-GPU only
```

- **GPU memory:** the validator runs DeepSeek-V2-Lite at fp16-mixed on a
  40 GB GPU (A100/H100 class) with ~24 GB allocated during eval. Miners run
  partial models (only their expert group is trainable), so memory is
  smaller — but a 24 GB consumer card is a reasonable minimum.
- **Precision:** `fp16-mixed` is the default. `bf16-mixed` is supported but
  the code path auto-falls back if the device doesn't report BF16 support
  (warned, not crashed).
- **DDP / multi-GPU:** `local_par.world_size > 1` *exists* in the code path
  but is not part of the current single-GPU miner profile. Do not set
  unless you know what you're doing.

Code: `connito/shared/config.py:ParallelismCfg`,
`connito/miner/train.py:setup_training`.

## Checkpoint persistence

```yaml
ckpt:
  base_checkpoint_path: checkpoints/miner   # derived: root_path / this / coldkey / hotkey / run_name
  checkpoint_topk: 2                        # how many local checkpoints to keep
  checkpoint_interval: 20                   # save every N inner_opt_step (default: 20% of global_opt_interval)
  resume_from_ckpt: true                    # resume from local checkpoint on restart
```

The miner saves local checkpoints during Train at every
`checkpoint_interval` inner_opt_step. Only the most recent
`checkpoint_topk` are kept on disk.

The path that ends up serving MinerCommit2 is `select_best_checkpoint(
config.ckpt.checkpoint_path)` — the newest local checkpoint. If you delete
the local checkpoint dir between Train and MinerCommit2, the commit phase
will fail with `FileNotReadyError`.

Code: `connito/shared/config.py:CheckpointCfg`,
`connito/shared/checkpoint_helper.py:save_checkpoint`,
`connito/shared/checkpoints.py:select_best_checkpoint`.

## Cycle / phase API endpoint

```yaml
cycle:
  owner_url:        https://cycle-api.connito.ai:443     # LOCKED
  cycle_length:     448                                  # LOCKED — re-derived from PhaseManager
  api_timeout_sec:  10
  api_retries:      5
  api_backoff_sec:  2
  version_range_cycles: 3                                # how old a validator checkpoint can be and still be accepted
```

The `owner_url` is hard-pinned. Do not point this at a different host
unless you are running a private testnet — the miner will refuse commits
that don't agree with the chain-side phase boundaries enforced by all
other workers.

If you see HTTP 403 from `cycle-api.connito.ai`, see
`troubleshooting.md` — it is almost always a Cloudflare WAF rule.

Code: `connito/shared/config.py:CycleCfg`.

## Common pitfalls

1. **Wallet permission.** Validators sign and commit using the **hotkey**;
   the coldkey is only for registration. Make sure
   `~/.bittensor/wallets/<coldkey>/hotkeys/<hotkey>` exists and is readable
   by the user running the miner.
2. **Stale `.pt` files in `checkpoints/`.** When the cycle dies mid-Train,
   the local checkpoint dir can keep `.pt` files from a previous run.
   Miner `_prepare_checkpoint_for_commit` picks `select_best_checkpoint`,
   which orders by `global_ver` then by `inner_opt`, so a stale checkpoint
   with a higher `global_ver` than your current training run will be
   committed instead of the freshly trained one. Solution: nuke the local
   checkpoint dir between runs if you reset training.
3. **HF private repo.** If `hf.checkpoint_repo` is private, validators will
   get 401 on download and your `eval_status_label` will be
   `download_failed`. Make the repo public.
4. **Run name reuse.** `run_name` is part of the local checkpoint path. If
   you rename the run between cycles, miner restart will not find the
   previous checkpoint and will train from base — the same symptom as the
   `.pt` 404 trap, but caused by local config drift, not HF.
5. **Setting `expert_group_name` to a directory that doesn't exist.** Will
   fail at startup in `_update_by_task` — make sure `task.expert_group_name`
   matches a directory under `expert_groups/`.

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
`miner_perplexity`. Use these to confirm training is actually progressing
before debugging score issues.

Code: `connito/shared/telemetry.py` (MINER_* gauges, line 233-251).
