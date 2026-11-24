#!/usr/bin/env bash
# Usage: GPUS=0,1 WORLD_SIZE=2 ./scripts/validate_smpl.sh CONFIG CHECKPOINT [PORT]
set -euo pipefail

CONFIG="${1:?usage: $0 CONFIG CHECKPOINT [PORT]}"
CKPT="${2:?usage: $0 CONFIG CHECKPOINT [PORT]}"
PORT="${3:-23456}"
HOST="${HOST:-127.0.0.1}"
GPUS="${GPUS:-0}"
WORLD_SIZE="${WORLD_SIZE:-$(awk -F, '{print NF}' <<< "${GPUS}")}"

uv run python ./scripts/validate_smpl.py \
    --batch "${BATCH:-32}" \
    --gpus "${GPUS}" \
    --world-size "${WORLD_SIZE}" \
    --flip-test \
    --launcher pytorch --rank 0 \
    --dist-url "tcp://${HOST}:${PORT}" \
    --cfg "${CONFIG}" \
    --checkpoint "${CKPT}"
