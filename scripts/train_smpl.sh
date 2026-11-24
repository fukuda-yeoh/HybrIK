#!/usr/bin/env bash
# Usage: ./scripts/train_smpl.sh EXP_ID CONFIG [PORT]
set -euo pipefail

EXPID="${1:?usage: $0 EXP_ID CONFIG [PORT]}"
CONFIG="${2:?usage: $0 EXP_ID CONFIG [PORT]}"
PORT="${3:-23456}"
HOST="${HOST:-127.0.0.1}"

uv run python ./scripts/train_smpl.py \
    --nThreads "${NTHREADS:-16}" \
    --launcher pytorch --rank 0 \
    --dist-url "tcp://${HOST}:${PORT}" \
    --exp-id "${EXPID}" \
    --cfg "${CONFIG}" --seed "${SEED:-123123}"
