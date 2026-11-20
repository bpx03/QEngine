#pragma once
// Release prep [dsv4-rel-sample]: the DSV4 engine's sampling (dsv4_sample.cu, over the shared src/kernels/sample.cu)
#include <cstddef>

#include <cuda_runtime.h>

namespace dsv4 {
constexpr int DSS_MAXR = 9;   // verify rows (MAXR 8) + 1
// device block: parameters + counter, the acceptance state, the candidate pairs
size_t samp_bytes();
// parameters (temperature > 0; top_k 0 or > 64 = 64, the sampler's candidate count); the counter restarts at 0
void samp_set(void* dev, float temperature, float top_p, int top_k, float min_p, unsigned long long seed, cudaStream_t st);
// m rows of logits [m][v] -> one sampled token per row (device ints)
void samp_rows(void* dev, const float* logits, int v, int m, int* tokens_dev, cudaStream_t st);
// a verify pass of k drafts: rows 0..k of logits, in_tokens = [t, d1..dk]; emit_host (pinned) <- [count, tokens...]
// (count = accepted drafts + 1); the stream must be synchronized before emit_host is read
void samp_accept(void* dev, const float* logits, int v, const int* in_tokens_host, int k, int pos, int eos0, int eos1,
                 int* emit_host, cudaStream_t st);
}  // namespace dsv4
