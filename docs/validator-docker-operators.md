# Running the Connito SN102 validator with Docker + Watchtower

> **Audience:** people running validators on SN102.
> **Not for:** miners (miners do not use this stack), or repo maintainers
> (see [validator-docker-maintainers.md](validator-docker-maintainers.md)
> for build/release internals).

This is the operator runbook. It complements the standard guide at
<https://www.connito.ai/docs/validator/running> — read that first to
understand wallets, registration, expert groups, and weight setting.
This doc only covers the **containerization layer**: how to run the
validator as a container that auto-updates whenever a new release
lands.

## What you get

```
GitHub push to master
        │
        ▼
GitHub Actions builds + pushes the image
        │
        ▼
ghcr.io/connito-ai/connito-validator:latest
        │
        ▼
Watchtower on your validator host
        │  polls every 5 min, sees new digest
        ▼
docker pull + recreate `connito-validator`
        │  bind mounts (wallets, /data) preserved
        ▼
Validator resumes from the latest checkpoint
```

You don't have to `git pull && pip install && pkill && python -m ...`
ever again. Watchtower does that for you.

## Files you'll touch

All under [connito/validator/docker/](../connito/validator/docker/):

| File | Purpose |
|------|---------|
| [`Dockerfile`](../connito/validator/docker/Dockerfile) | Validator image definition. You normally don't edit this — the registry build is canonical. |
| [`docker-compose.yml`](../connito/validator/docker/docker-compose.yml) | Two services: `validator` and `watchtower`. You normally don't edit this either. |
| [`.env.example`](../connito/validator/docker/.env.example) | Copy to `.env`, edit the values. **This is the only file you need to edit.** |

## One-time host setup

The container needs three things from the host: a GPU, a Bittensor
wallet, and a writable data directory.

### 1. Install Docker Engine + Compose plugin

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker
docker compose version    # confirm the v2 plugin is installed
```

### 2. Install the NVIDIA Container Toolkit

The validator needs CUDA inside the container. The base image ships the
CUDA runtime, but the host driver and the toolkit must be installed
**outside** the container.

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

You should see your GPU(s) listed. If not, fix this **before** continuing
— nothing else will work.

### 3. Authenticate to GHCR (only if the image is private)

If `ghcr.io/connito-ai/connito-validator` is **public**, skip this. If
it is private:

```bash
echo "<github_pat_with_read:packages>" \
  | docker login ghcr.io -u <your-github-username> --password-stdin
```

This writes credentials to `~/.docker/config.json`, which the compose
file bind-mounts into Watchtower so it inherits the same auth.

### 4. Lay out the data directory

Watchtower destroys and recreates the validator container on every
update, so anything you want to keep **must** live on a host bind mount.

You have two choices:

#### Option A — quick start (use the in-repo skeleton)

The repo ships an empty skeleton at
[`connito/validator/docker/data/`](../connito/validator/docker/data/)
with the right subdirectories already created. The compose file's
defaults point at it, so you can skip the `mkdir` step entirely. Just
copy the example config:

```bash
cd connito/validator/docker
cp data/configs/validator.example.yaml data/configs/validator.yaml
$EDITOR data/configs/validator.yaml
```

**Use this for:** local dev, smoke tests, demo runs.

**Don't use this for:** production. Anything inside a git checkout is
one careless `git clean -fdx` away from being deleted, and re-cloning
the repo wipes your checkpoints.

#### Option B — production (stable host path outside the repo)

```bash
sudo mkdir -p /opt/connito/data/{checkpoints,expert_groups,hf,wandb,logs,configs}
sudo chown -R $USER:$USER /opt/connito

# Copy the example config out of the repo into your stable location:
cp connito/validator/docker/data/configs/validator.example.yaml \
   /opt/connito/data/configs/validator.yaml
$EDITOR /opt/connito/data/configs/validator.yaml
```

Then in your `.env`:

```
DATA_DIR=/opt/connito/data
CONFIG_PATH=/opt/connito/data/configs/validator.yaml
```

Layout:

```
/opt/connito/
└── data/
    ├── checkpoints/        # validator checkpoints (large, persistent)
    ├── expert_groups/      # expert group artifacts
    ├── hf/                 # HuggingFace cache (HF_HOME)
    ├── wandb/              # W&B run logs
    ├── logs/               # validator metric logs
    └── configs/
        └── validator.yaml  # your validator config (see step 5)
```

**Use this for:** any validator that's actually earning weight on
mainnet.

### 5. Create your `validator.yaml`

This is the same YAML the standard guide tells you to write — it's read
by `connito.shared.config.ValidatorConfig.from_path`. Drop it at
`/opt/connito/data/configs/validator.yaml`. Required fields:

| Field | Default | Notes |
|-------|---------|-------|
| `chain.netuid` | `102` | Locked to SN102. |
| `chain.network` | `archive` | Or `finney`. |
| `chain.hotkey_name` / `chain.coldkey_name` | — | Must match a wallet under `~/.bittensor/wallets`. |
| `chain.port` | `8000` | Bittensor `serve` port; must be reachable from the public internet. |
| `dht.port` | `6000` | Hivemind DHT — open both **TCP and UDP** in your firewall. |
| `task.exp.group_id` | — | Your assigned expert group. |

**All paths in this YAML must use the container view of the world.** The
compose file mounts your host data dir at `/data` inside the container,
so use:

```yaml
ckpt:
  base_checkpoint_path: /data/checkpoints
  checkpoint_path: /data/checkpoints/active
  miner_submission_path: /data/checkpoints/miner_submissions
  miner_submission_archive_path: /data/checkpoints/miner_submission_archive
log:
  base_metric_path: /data/logs
task:
  base_path: /data/expert_groups
```

If you point any of these outside `/data` they will be lost on the next
Watchtower upgrade.

### 6. Create the compose `.env` file

```bash
cd /path/to/repo/connito/validator/docker
cp .env.example .env
$EDITOR .env
```

Required values:

- `WALLET_NAME` / `HOTKEY_NAME` — must match `btcli wallet list`.
- `BITTENSOR_WALLET_PATH` — full path to the parent dir of your coldkey
  (default `~/.bittensor/wallets`).
- `DATA_DIR` — set to `/opt/connito/data` (or wherever you laid it out).
- `CONFIG_PATH` — set to `/opt/connito/data/configs/validator.yaml`.

Optional:

- `WANDB_API_KEY` if you want metrics in Weights & Biases.
- `WATCHTOWER_NOTIFICATIONS=slack` and `WATCHTOWER_NOTIFICATION_SLACK_HOOK_URL`
  if you want a Slack ping every time Watchtower upgrades the container.
- `IMAGE` — leave on `:latest` for auto-upgrades, or pin to
  `ghcr.io/connito-ai/connito-validator:master-<sha>` to freeze a
  specific build (Watchtower will not upgrade off a pinned tag).

### 7. Open firewall ports

Same ports as the bare-metal guide. With `network_mode: host` in the
compose file the container binds directly to the host interface.

```bash
sudo ufw allow 8000/tcp     # bittensor serve
sudo ufw allow 6000/tcp     # hivemind DHT
sudo ufw allow 6000/udp     # hivemind DHT (QUIC)
# 8200/tcp telemetry — leave closed unless you scrape it remotely
```

## Running

From `connito/validator/docker/`:

```bash
docker compose up -d              # pull + start both services
docker compose logs -f validator  # watch validator logs
docker compose logs -f watchtower # watch upgrade decisions
```

You should see the validator hit `(0) Commit new seed for next validation`
within a few minutes if your wallet is registered and the chain is
reachable.

## Verifying the auto-update loop

1. Wait for the next push to `master` (or coordinate with the maintainers
   to push a trivial change).
2. Within `WATCHTOWER_POLL_INTERVAL` seconds (default 5 min) after the
   GitHub Actions build finishes:
   ```bash
   docker compose logs --since 10m watchtower | grep -i "found new"
   ```
   You should see Watchtower detect the new digest, stop the old
   container, pull the new image, and start the new one.
3. The validator resumes from the latest checkpoint in `/data/checkpoints`.

## Operational notes

- **Mid-cycle restarts.** Watchtower stops the container with `SIGTERM`
  and waits up to `WATCHTOWER_TIMEOUT` (`120s`). `connito.validator.run`
  catches `KeyboardInterrupt`, shuts the averagers + telemetry poller
  down cleanly, and exits. If an upgrade lands mid-validation phase the
  validator restarts and re-enters the cycle at the next phase boundary;
  no chain commit is lost because commits happen at phase transitions,
  not continuously.
- **Pinning a version during incidents.** Set
  `IMAGE=ghcr.io/connito-ai/connito-validator:master-<sha>` in `.env`
  and `docker compose up -d`. Watchtower will not upgrade off a tag it
  doesn't recognize as newer.
- **Stopping auto-updates entirely.** `docker compose stop watchtower`,
  or remove the `com.centurylinklabs.watchtower.enable` label from the
  validator service.
- **Multi-GPU hosts.** Replace `count: all` in `docker-compose.yml` with
  `device_ids: ["0"]` to pin a specific GPU. Run separate compose
  projects (`COMPOSE_PROJECT_NAME=val2 docker compose ...`) per
  validator if you want multiple validators on one box.

## Things you have to do manually (not automated)

1. **GHCR access.** If the image is private, run `docker login ghcr.io`
   on the host. The compose file bind-mounts `~/.docker/config.json`
   into Watchtower so it inherits the same auth — no extra config.
2. **`validator.yaml` is host-specific** (wallet names, expert group,
   ports). Source-control it in a private repo or your secret store, not
   in the public one.
3. **Wallet bind mount is read-only.** The compose file does this for
   you (`:ro`). Confirm `BITTENSOR_WALLET_PATH` actually points at a
   directory containing the coldkey + hotkey you registered with on
   SN102. To double-check from inside the container:
   ```bash
   docker compose exec validator btcli wallet list
   ```
4. **Backup `/data/checkpoints` independently.** Watchtower upgrades are
   safe, but a corrupt checkpoint after an upgrade is not a Watchtower
   problem — it's a "you need a restore path" problem. `rsync` the
   checkpoints dir to S3/GCS on a cron.
5. **NVIDIA driver ≥ 550.x on the host.** The image ships CUDA 12.4
   runtime, which requires NVIDIA driver 550 or newer. If you're on an
   older driver, ask the maintainers to publish a CUDA 12.1 / 12.2
   variant, or upgrade your driver.
6. **Prometheus scraping for `:8200/metrics`.** The container exposes the
   same telemetry surface as the bare-metal validator. Point the
   existing `observability/` stack at `localhost:8200` (the validator
   uses `network_mode: host`, so it's on the host's loopback).
7. **Slack notifications.** Set `WATCHTOWER_NOTIFICATIONS=slack` and the
   webhook URL in `.env` if you want to know the moment an auto-upgrade
   lands. Highly recommended — silent upgrades are great until they
   aren't.
8. **HuggingFace cache hygiene.** The compose file sets `HF_HOME=/data/hf`
   so model weights survive restarts. To reclaim disk:
   `rm -rf /opt/connito/data/hf` and let the next run re-download.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `could not select device driver "" with capabilities: [[gpu]]` | NVIDIA Container Toolkit not installed or Docker not restarted. Re-run step 2. |
| `permission denied while trying to connect to the Docker daemon socket` | You forgot `usermod -aG docker $USER` + `newgrp docker` from step 1. |
| Validator logs spam `wallet not found` | `BITTENSOR_WALLET_PATH` in `.env` doesn't point at the right host path, or `WALLET_NAME` / `HOTKEY_NAME` don't match what's actually under that dir. |
| Validator can't reach hivemind peers | Firewall step 7 not done, or `dht.port` in `validator.yaml` doesn't match what you opened. Both **TCP and UDP** must be open. |
| Watchtower never upgrades | Either the registry is private and Watchtower has no creds (mount `~/.docker/config.json`), or you pinned `IMAGE` to a sha tag (Watchtower only follows moving tags like `latest`). |
| Image pull fails with `denied` | GHCR package is private and you haven't run `docker login ghcr.io` on the host. |
| `CUDA error: no kernel image is available for execution on the device` | Your GPU is too old for the CUDA runtime in the image, or your host driver is < 550. |

If you're stuck, capture:

```bash
docker compose ps
docker compose logs --tail 200 validator
docker compose logs --tail 200 watchtower
nvidia-smi
docker info | grep -i runtime
```

…and open an issue.
