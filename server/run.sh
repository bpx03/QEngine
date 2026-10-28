#!/usr/bin/env bash
# Start the QEngine server (OpenAI + Anthropic API), tensor parallel over the selected GPUs.
# usage: server/run.sh <port> [extra args for server/app.py]
#
# GPU selection (default: GPUs 0,1,2, tensor parallel 3; 2 GPUs are supported too):
#   QE_DEVICES=0,2 server/run.sh 8080     # physical GPU ids; sets CUDA_VISIBLE_DEVICES=0,2, --devices 0,1
#   CUDA_VISIBLE_DEVICES=0,2 server/run.sh 8080   # same, all visible GPUs are used
#   server/run.sh 8080 --devices 0,1      # explicit --devices (indices within CUDA_VISIBLE_DEVICES) is passed through
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${1:-8080}"
shift || true

if [ -n "${QE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$QE_DEVICES"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
fi
IFS=, read -ra _vis <<< "$CUDA_VISIBLE_DEVICES"

# --devices on the command line wins; otherwise every visible GPU is used
_devices=""
for ((i = 1; i <= $#; i++)); do
  a="${!i}"
  case "$a" in
    --devices=*) _devices="${a#--devices=}" ;;
    --devices) j=$((i + 1)); _devices="${!j:-}" ;;
  esac
done
_extra=()
if [ -z "$_devices" ]; then
  _devices=$(seq -s, 0 $((${#_vis[@]} - 1)))
  _extra=(--devices "$_devices")
fi
IFS=, read -ra _req <<< "$_devices"

# sanity check: enough visible GPUs for the requested devices
_avail=${#_vis[@]}
if command -v nvidia-smi >/dev/null 2>&1; then
  _phys=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)
  [ "${_phys:-0}" -gt 0 ] && [ "$_phys" -lt "$_avail" ] && _avail=$_phys
fi
if [ "${#_req[@]}" -gt "$_avail" ]; then
  echo "QEngine: ${#_req[@]} devices requested (--devices $_devices) but only $_avail GPU(s) visible (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)" >&2
  exit 1
fi
if [ "${#_req[@]}" -lt 2 ] || [ "${#_req[@]}" -gt 3 ]; then
  echo "QEngine: warning: ${#_req[@]} GPU(s) requested; the engines are tested on 2 and 3 GPUs (Flash-Next needs 2 or 3)" >&2
fi
exec "${PYTHON:-python3}" server/app.py --host 127.0.0.1 --port "$PORT" "${_extra[@]}" "$@"
