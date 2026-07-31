# syntax=docker/dockerfile:1.6
#
# Connito SN102 validator image.
#
# IMPORTANT: build context is the repo root, not this directory.
#   docker build -f connito/validator/docker/Dockerfile -t connito-validator .
#
# Built and published from master via .github/workflows/docker-publish.yml.
# Run via connito/validator/docker/docker-compose.yml on a GPU host with
# NVIDIA Container Toolkit installed. Watchtower in the same compose file
# will pull new tags and restart the validator automatically.
#
# Miners do NOT use this image — only validators are containerized.

ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${CUDA_IMAGE} AS base

# Build-time provenance, baked into the image so `run.py` can report which
# commit/tag this image was built from. Set by CI (docker-publish.yml) or via
#   docker build \
#     --build-arg GIT_SHA=$(git rev-parse HEAD) \
#     --build-arg GIT_VERSION=$(git describe --tags --always) \
#     ...
# GIT_SHA = immutable commit id; GIT_VERSION = human-readable (e.g. v1.2.3,
# master-abc1234) — operators see the latter in startup logs.
ARG GIT_SHA=unknown
ARG GIT_VERSION=unknown

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    # Base port for the Prometheus exporter. Read by
    # `connito.validator.run.resolve_telemetry_port`; the effective port is
    # `CONNITO_TELEMETRY_PORT + rank` (rank is 0 for the validator). Override
    # via the compose `.env` when 8200 is already taken on the host — and
    # remember to open the new port in your firewall.
    #
    # NOTE: the Hivemind/DHT port is NOT an env var. It is configured by
    # `dht.port` in validator.yaml, which is the single source of truth —
    # that port is announced to peers, so a second competing setting could
    # make a validator advertise one port while listening on another. The
    # former CONNITO_DHT_PORT / CONNITO_STATE_API_PORT vars were never read
    # by any code and have been removed (the /v1/state.json API they
    # referenced no longer exists).
    CONNITO_TELEMETRY_PORT=8200 \
    CONNITO_GIT_SHA=${GIT_SHA} \
    CONNITO_GIT_VERSION=${GIT_VERSION}

# System packages: python 3.10, build essentials (some wheels still need them),
# git for the VCS-installed deps in requirements.txt (hivemind, unsloth),
# and curl for healthchecks.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-venv python3-pip \
        build-essential git curl ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# Install dependencies first so the layer caches across code changes.
# Torch is pinned in requirements.txt; we add the CUDA 12.4 wheel index
# so the right binary wheel is selected.
COPY requirements.txt ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
        -r requirements.txt

# Now copy the source. .dockerignore at the repo root keeps venvs, wandb
# runs, checkpoints, notebooks, and the giant mycelia_final.pt out of the
# build context.
COPY pyproject.toml README.md LICENSE ./
COPY connito ./connito
# expert_groups/ ships with the code (per-task config.yaml + expert_assignment.json).
# Baking them into the image guarantees the validator has the exact same
# assignments as the connito/ package version it's paired with, so
# `task.helper_group_id` lookups by group_id always resolve without
# depending on a host-side checkout. Compose can still bind-mount over
# /app/expert_groups if a developer wants to iterate locally.
COPY expert_groups ./expert_groups

# Install the package itself (editable not needed inside the image).
RUN pip install --no-deps -e .

# Persistent state goes into bind-mounted directories under /data so that
# upgrades via Watchtower never lose checkpoints, wallets, or expert groups.
RUN mkdir -p /data/checkpoints /data/expert_groups /data/wandb /data/logs \
    && mkdir -p /root/.bittensor/wallets

ENV CHECKPOINTS_DIR=/data/checkpoints \
    EXPERT_GROUPS_DIR=/data/expert_groups \
    WANDB_DIR=/data/wandb

# 6000/tcp + 6000/udp: hivemind DHT (config.dht.port)
# 8000/tcp:            chain serve port (config.chain.port)
# 8200/tcp:            prometheus telemetry (CONNITO_TELEMETRY_PORT + rank) -> /metrics
# Documentation only — the compose file uses `network_mode: host`, so these
# are not published mappings. 8300 (the removed /v1/state.json API) is gone.
EXPOSE 6000/tcp 6000/udp 8000/tcp 8200/tcp

# Default command — override `--path` to point at your config in compose.
ENTRYPOINT ["python", "-m", "connito.validator.run"]
CMD ["--path", "/data/configs/validator.yaml"]
