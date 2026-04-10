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

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    # Default Hivemind/DHT + telemetry ports — overridable at runtime
    CONNITO_DHT_PORT=6000 \
    CONNITO_TELEMETRY_PORT=8200

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
# 8200/tcp:            prometheus telemetry (8200 + rank)
EXPOSE 6000/tcp 6000/udp 8000/tcp 8200/tcp

# Default command — override `--path` to point at your config in compose.
ENTRYPOINT ["python", "-m", "connito.validator.run"]
CMD ["--path", "/data/configs/validator.yaml"]
