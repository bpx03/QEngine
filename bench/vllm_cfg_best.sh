#!/usr/bin/env bash
# club-3090 "vllm/qwen38-27b-dual-fast" (the most-starred community 3090 recipe), ported flag-for-flag
# to stock vLLM 0.28.0 and the RedHatAI W4A16 compressed-tensors checkpoint. TP=2 on two GPUs
# (CUDA_VISIBLE_DEVICES, default 0,2).
# Source: https://github.com/noonghunna/club-3090 models/qwen3.8-27b/vllm/compose/dual/autoround-int4/mtp.yml
#         sweep: https://github.com/noonghunna/club-3090/discussions/1092 (1024 in / 512 out, MAX_NUM_SEQS=32)
# Deviations: our checkpoint instead of Frozenlock AutoRound;
#   no club-3090 W4A8 patches (W4A8 is stock for compressed-tensors); --disable-custom-all-reduce (patched P2P driver);
#   the two club-3090 runtime patches for MTP under concurrency were applied to the vLLM install
#   (see bench/results/compare_2026-10/README.md).
#   no --reasoning-parser / tool parser / custom chat template (they do not change throughput, but they can change
#   how the bench counts output tokens).
# VLLM=<vllm CLI> (default: vllm on PATH).
# Knobs (same names as club-3090): SPEC_N=4 (0 = no drafter; their c=32 peak of 308 was measured this way), MAX_NUM_SEQS=32,
#   W4A8=1 (0 = W4A16), MAX_MODEL_LEN=262144, GPU_MEMORY_UTILIZATION=0.90, NCCL_P2P_DISABLE (unset = NCCL default).
# usage: bench/vllm_cfg_best.sh [port]
#   then: bench/serving_suite.py --engine-name vllm-club3090 --base-url http://127.0.0.1:8182 --model vllm-int4
cd "$(dirname "$0")/.."
SPEC_N="${SPEC_N:-4}"
SPEC_ARGS=()
if [ "$SPEC_N" -gt 0 ]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_N,\"draft_sample_method\":\"probabilistic\"}")
fi
W4A8_ENV=()
[ "${W4A8:-1}" != "0" ] && W4A8_ENV=(VLLM_MARLIN_INPUT_DTYPE=int8)
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}" "${W4A8_ENV[@]}" \
  VLLM_WORKER_MULTIPROC_METHOD=spawn NCCL_CUMEM_ENABLE=0 VLLM_NO_USAGE_STATS=1 OMP_NUM_THREADS=1 \
  VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${VLLM:-vllm}" serve "$PWD/models/Qwen3.8-27B-INT4" \
  --served-model-name vllm-int4 --host 127.0.0.1 --port "${1:-8182}" \
  --dtype bfloat16 --tensor-parallel-size 2 --disable-custom-all-reduce \
  --max-model-len "${MAX_MODEL_LEN:-262144}" --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --max-num-seqs "${MAX_NUM_SEQS:-32}" --max-num-batched-tokens 8192 --long-prefill-token-threshold 4096 \
  --kv-cache-dtype fp8_e4m3 --enable-prefix-caching --enable-chunked-prefill \
  --mamba-cache-mode align --prefix-match-unit 16 --mamba-ssm-cache-dtype bfloat16 \
  --trust-remote-code --enable-prompt-tokens-details \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
  "${SPEC_ARGS[@]}"
