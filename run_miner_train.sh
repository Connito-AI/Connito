#!/usr/bin/env bash
# Babysitter launcher for the miner training run on GPU 2.
# Logs live to logs/train-<ts>.log and repoints logs/train-latest.log.
# Any extra env vars can be exported before calling (e.g. PYTORCH_CUDA_ALLOC_CONF).
#
# NOTE: 32-bit AdamW OOMs at the first optimizer step on a 48GB A6000 with the
# current train.py (~200-350 MiB over the edge). The 8-bit bitsandbytes AdamW
# state (~4x smaller) that clears it is now selected via config.opt.adamw_optim_bits
# (default 8) — no longer an env var. expandable_segments stays on as insurance.
set -uo pipefail
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF

cd "$(dirname "$0")"                 # -> ConnitoAI repo root
mkdir -p logs
TS="$(date +%Y%m%d-%H%M%S)"
LOG="logs/train-${TS}.log"
ln -sfn "$(pwd)/${LOG}" logs/train-latest.log

CONFIG="$HOME/crucible/ConnitoAI/checkpoints/miner/connito-puppet/hk2/pr-188/config.yaml"

{
  echo "=== RUN START ${TS} ==="
  echo "=== host=$(hostname) pid=$$ ==="
  echo "=== env: TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=${TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS:-1} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-<unset>} (adamw_optim_bits via config.opt) ==="
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/=== gpu: /'
} >>"${LOG}" 2>&1

env TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS="${TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS:-1}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" \
    ${PYTORCH_CUDA_ALLOC_CONF:+PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"} \
    "$(pwd)/.miner-venv/bin/python" -u connito/miner/train.py --path "${CONFIG}" \
    >>"${LOG}" 2>&1
RC=$?

echo "=== RUN EXIT rc=${RC} ts=$(date +%Y%m%d-%H%M%S) ===" >>"${LOG}"
exit ${RC}
