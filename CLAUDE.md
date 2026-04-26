# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Connito SN102 — Bittensor subnet for decentralized Mixture-of-Experts LLM training. Validators coordinate, miners train expert groups, and the sn_owner runs a phase-clock service. All three coordinate through Bittensor block heights and a Hivemind DHT.

## Commands

Python 3.10+. Install deps with the CUDA-12.4 PyTorch index:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
pip install -e .
```

### Generating configs

Configs are pydantic-validated YAML. Generate from defaults rather than hand-writing:

```bash
# Validator config + Docker .env (under checkpoints/validator/<coldkey>/<hotkey>/<run_name>/config.yaml)
python connito/shared/config.py create_config --role validator --coldkey_name <ck> --hotkey_name <hk>

# Miner config
python connito/shared/config.py create_config --role miner --coldkey_name <ck> --hotkey_name <hk>

# Regenerate docker .env from an existing validator config
python connito/shared/config.py create_docker_env --path <path/to/config.yaml>
```

`--auto_update_config` (default on) silently resets locked fields to defaults instead of prompting; pass `--no-auto_update_config` to be prompted.

### Running

```bash
# Miner — uses torch.multiprocessing.spawn when world_size > 1
python -m connito.miner.train --path <miner_config.yaml>

# Validator main loop
python -m connito.validator.run --path <validator_config.yaml>

# Validator HTTP server (FastAPI; serves checkpoints to miners, accepts submissions)
python -m connito.validator.server --path <validator_config.yaml>

# Subnet owner phase clock + DHT bootstrap (FastAPI on :8080)
python -m connito.sn_owner.phase_service --path <owner_config.yaml>
```

### Tests, lint, type-check

Tests live under `connito/test/` (not `tests/` — `pyproject.toml`'s `testpaths` is wrong, point pytest at the real dir):

```bash
pytest connito/test                          # full suite
pytest connito/test/test_checkpoint_retention.py -v
pytest connito/test -k "not integration"     # skip GPU-heavy integration tests (marked `pytest.mark.integration`)

ruff check connito                           # 120-col, target py310
black --line-length 100 connito              # note: ruff uses 120, black uses 100 — black wins on format
mypy connito
```

### Validator container (validators only — miners are not containerized)

```bash
cd connito/validator/docker
docker compose up -d                         # validator + server + watchtower
docker compose logs -f validator
```

Releases: `git tag vX.Y.Z && git push --tags` triggers `.github/workflows/docker-publish.yml`, which publishes `ghcr.io/connito-ai/connito-validator:stable` (and `:vX.Y.Z`). Operators run `:stable`; Watchtower auto-pulls. Pushes to `master`/`staging` build `:master-<sha>` / `:staging-<sha>` artifacts but do not move `:stable`.

## Architecture

### The cycle (read this first)

Everything is driven by a fixed-length block-height cycle, defined in `connito/sn_owner/cycle.py::PhaseManager`. Phase names are in `connito/shared/cycle.py::PhaseNames`:

```
Distribute → Train → MinerCommit1 → MinerCommit2 → Submission
          → Validate → Merge → ValidatorCommit1 → ValidatorCommit2
```

Miners and validators block on `connito.shared.cycle.wait_till(config, phase_name)` to enter each phase synchronously. Phase boundaries — not wall time — are when chain commits happen, when checkpoints are written, and when the validator can safely restart without losing state. The sn_owner's FastAPI service (`/get_phase`, `/blocks_until_next_phase`) is the authoritative clock when it's reachable; otherwise workers compute phase locally from `subtensor.block`. If you're touching state-machine code, mentally place every action inside one of these phases.

For the per-phase breakdown of what `connito/validator/run.py` actually does in each phase, see [`docs/validator-cycle-phases.md`](docs/validator-cycle-phases.md).

### Three roles, one package

- **`connito/miner/`** — `train.py` is the entry point. Trains the assigned expert group, commits a hash on-chain, then uploads weights to the validator.
- **`connito/validator/`** — `run.py` is the main loop (eval + score + chain weight commit + Hivemind averaging). `server.py` is a separate FastAPI service that serves checkpoints to miners and ingests submissions; it runs as its own container (`server` service in compose) sharing the same image. `evaluator.py` does the per-miner scoring; `aggregator.py` keeps a rolling `MinerScoreAggregator` (window size set by `EvalCfg.score_window`).
- **`connito/sn_owner/`** — subnet owner only. `phase_service.py` is the public phase clock + DHT bootstrap; `validator_whitelist.json` lists hotkeys force-permitted as validators.

### `connito/shared/` is the heart

Shared between roles, and most non-trivial logic lives here:

- `config.py` — pydantic config tree. `WorkerConfig` is the base; `MinerConfig`/`ValidatorConfig`/`OwnerConfig` extend it. Fields marked in `_LOCKED_FIELDS` are network-consensus parameters and cannot diverge between workers — `from_path` will reset them on load (with `auto_update_config`) or prompt. Inside Docker, `model_post_init` rewrites `run.root_path` to `/data` and `task.base_path` to `/app/expert_groups`. Run names auto-bump (`-v2`, `-v3`) when a config diff is detected, unless `--no_bump`.
- `cycle.py` — phase synchronization, block-time math (`BITTENSOR_BLOCK_TIME_SECONDS = 12`), and HF checkpoint hydration helpers. The phase clock falls back from the sn_owner API to local computation if the API is unreachable.
- `chain.py` — Bittensor chain commits (`MinerChainCommit`, `ValidatorChainCommit`, `SignedModelHashChainCommit`). Validator commits to chain include an HF repo revision (`HF_CHAIN_REVISION_LENGTH = 7` chars, capped by `VALIDATOR_COMMIT_MAX_HF_REPO_ID_CHARS`).
- `model.py` + `modeling/mycelia.py` — model loading and the MoE wrapper. **Backend selection is a module-level constant**: `MODEL_BACKEND = "deepseek_v2"` (default) or `"qwen3_next"` in `connito/shared/modeling/mycelia.py`. To swap, edit the constant — there is no config flag.
- `expert_manager.py` — maps expert group ids to the slice of the model each miner owns; populates global gradients from local averaged ones.
- `checkpoints.py` + `checkpoint_helper.py` — `globalver_<N>/` directory layout, retention via `delete_old_checkpoints(topk=…)`, and the `score_aggregator.json` sidecar that lives alongside checkpoints and **must be preserved** when pruning (regression-tested in `test_checkpoint_retention.py`).
- `hf_distribute.py` — Hugging Face is the **preferred** checkpoint transport. The validator HTTP `/get-checkpoint` path on `connito.validator.server` is the fallback. Both are kept working; don't delete one assuming the other is sufficient.

### Distributed training & DHT

Hivemind's `DecentralizedAverager` does parameter averaging across miners. Group formation is bounded by `ValidatorRunCfg.averager_step_timeout_sec`. The DHT is bootstrapped by the sn_owner's `dht_init.py` and peer ids are persisted to `init_peer_ids.json` (gitignored). DHT runs on TCP+UDP **6000** by default — both must be open for hivemind to discover peers.

### Telemetry

Prometheus client serves on `8000 + rank` (validator) or `8100 + local_rank` (miner). Disable with `ENABLE_TELEMETRY=false`. The repo includes a Grafana+Prometheus stack under `observability/` for local viewing. Metrics naming convention: `validator_*` for validator, `miner_*` for miner, `subnet_*` for chain/cycle. The validator container exposes `:8200` (different from the bare-metal default — `network_mode: host`).

## Conventions worth knowing

- **Dual-formatted code**: ruff at 120 cols and black at 100 cols are both configured. Run black if either disagrees — black wins.
- **`mypy` is strict** (`disallow_untyped_defs`, `strict_equality`, `warn_unused_ignores`). New code in `connito/` must be fully typed.
- **Locked config fields are network-consensus parameters.** If you find yourself wanting to make one configurable per-worker, you almost certainly don't — diverging values will desync the network. Add to `_LOCKED_FIELDS` instead.
- **Tests with `pytestmark = pytest.mark.integration`** require a GPU and real model downloads. Skip them in normal local runs.
- **Build version provenance**: the validator prints `version=… git_sha=…` at startup. In Docker, those come from `CONNITO_GIT_VERSION` / `CONNITO_GIT_SHA` env vars baked in by CI; in a source checkout, they come from `git describe` / `git rev-parse`. If you change `_get_build_version` in `connito/validator/run.py`, preserve both code paths.
- **Always create new commits** rather than amending — when a pre-commit hook fails, the commit didn't happen, and `--amend` would silently rewrite the previous one.
