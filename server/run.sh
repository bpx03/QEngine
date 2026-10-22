#!/usr/bin/env bash
# Start the QEngine server (OpenAI + Anthropic API) on three GPUs (tensor parallel 3).
# usage: server/run.sh <port> [extra args for server/app.py]
# CUDA_VISIBLE_DEVICES defaults to 0,1,2; the engine expects exactly three visible GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${1:-8080}"
shift || true
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
IFS=, read -ra _gpus <<< "$CUDA_VISIBLE_DEVICES"
if [ "${#_gpus[@]}" -ne 3 ]; then
  echo "QEngine needs exactly 3 GPUs, but CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; unset it or list three GPUs" >&2
  exit 1
fi
exec "${PYTHON:-python3}" server/app.py --host 127.0.0.1 --port "$PORT" "$@"
