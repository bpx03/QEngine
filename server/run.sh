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
# (an index may repeat: Flash-Next pipeline stages sharing a card, e.g. --devices 0,1,2,0 = 4 stages on 3 cards)
_uniq=$(printf '%s\n' "${_req[@]}" | sort -u | wc -l)
_max=$(printf '%s\n' "${_req[@]}" | sort -n | tail -1)
if [ "$_uniq" -gt "$_avail" ] || [ "${_max:-0}" -ge "$_avail" ]; then
  echo "QEngine: ${_uniq} devices requested (--devices $_devices) but only $_avail GPU(s) visible (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)" >&2
  exit 1
fi
if [ "${#_req[@]}" -lt 2 ] || [ "${#_req[@]}" -gt 3 ]; then
  echo "QEngine: warning: ${#_req[@]} GPU(s) requested; the engines are tested on 2 and 3 GPUs (Flash-Next: 2 or 3 cards: tp; 4 cards: tp with peer access between all of them, else pp; 5+ cards: pp, pipeline stages; see the README)" >&2
fi
# no core dumps unless QE_CORE_DUMP=1 (issue #3): a crashed engine holds 50+ GB of pinned host memory and the GPU
# mappings; writing its core kept the GPUs busy for minutes (a restart failed meanwhile) and is rarely usable. The
# crash report (native + Python stacks) is in the log either way.
[ "${QE_CORE_DUMP:-0}" = 1 ] || ulimit -c 0
exec "${PYTHON:-python3}" server/app.py --host 127.0.0.1 --port "$PORT" "${_extra[@]}" "$@"
