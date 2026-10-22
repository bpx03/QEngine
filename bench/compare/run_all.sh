#!/usr/bin/env bash
# compare_2026-10: engine by engine (one server at a time), each block = classic, goodput, prefill;
# then the drift check (short c=1 and c=32 again on a freshly started server).
# env: see driver.py (STAGE, OUT, VLLM, QENGINE_REF, SHAREGPT). Afterwards: python3 bench/compare/build_summary.py
set -u
cd "$(dirname "$0")"
STAGE="${STAGE:-$PWD/stage}"
export STAGE
mkdir -p "$STAGE"
block() {
  python3 driver.py "$1" "$2" >> "$STAGE/driver_$1_$2.log" 2>&1
  echo "$(date -Iseconds) DRIVER_EXIT $1 $2 rc=$?" >> "$STAGE/progress.log"
}
for e in qengine-sunday vllm-nospec qengine-now vllm-mtp; do block "$e" classic,goodput,prefill; done
for e in vllm-mtp qengine-sunday vllm-nospec qengine-now; do block "$e" drift; done
echo "$(date -Iseconds) ALL_DONE" >> "$STAGE/progress.log"
