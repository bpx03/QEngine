#pragma once
#include "../common.h"
#include "spec.cuh"

// Sampling over a vocab split across GPUs: each GPU extracts its local top-kTopK logits per row,
// the candidates are all-gathered, and every GPU merges them identically (same seed and
// counter), so all GPUs pick the same tokens without host round trips.
constexpr int kTopK = 64;

struct SampleParams {  // lives in device memory so captured graphs pick up new values
    float temperature;   // > 0
    float top_p;         // (0, 1]
    int top_k;           // 1..kTopK
    float min_p;         // 0 disables
    unsigned long long seed;
    unsigned long long counter;  // advanced by every sampling kernel
};

// Grammar mask: rows [r0, r0 + rows) of this GPU's logits shard ([rows][v], global ids from
// vocab_off) keep only the tokens whose bit is set in their mask bits + j * stride (masks of
// (vocab + 31) / 32 words; stride 0: the same mask for every row); others -> -inf.
void mask_logits(float* logits, int v, int vocab_off, const unsigned* bits, int r0, int rows,
                 size_t stride, cudaStream_t st);

// pairs[m][kTopK][2] = (logit, global index as float bits), sorted descending
void topk_local(const float* logits, int v, int m, int vocab_off, float* pairs, cudaStream_t st);

// Plain sampling: tokens[mi] ~ p(row mi) from gathered candidates [ngpu][m][kTopK][2].
void sample_rows(const float* gathered, int ngpu, int m, SampleParams* p, int* tokens,
                 cudaStream_t st);
// batched rows of several sequences: row mi samples with base[rowslot[mi]], greedy (top candidate) when !on[slot]
void sample_rows_slots(const float* gathered, int ngpu, int m, SampleParams* base, const int* rowslot,
                       const int* on, int* tokens, cudaStream_t st);

// Speculative verification with sampling (lossless for greedy drafts): row j accepts draft
// tokens[j+1] with probability p_j(draft), otherwise samples from p_j without the draft and
// stops. Writes s.out[0..a], s.acc, s.emit, s.mtp_tok, s.mtp_pos, s.hrow like spec_accept.
// mtot / r0: the step's candidate rows may hold several sequences; this one uses rows
// [r0, r0 + k + 1) of mtot
void spec_accept_sampled(const float* gathered, int ngpu, int k, SampleParams* p,
                         const SpecState& s, cudaStream_t st, int mtot = -1, int r0 = 0);
// all sequences of a batched step in one launch (same per-sequence arithmetic as spec_accept /
// spec_accept_sampled): p[z] null = greedy
constexpr int kAcceptBatch = 32;
struct AcceptBatch {
    int n = 0, m = 0;                 // sequences, candidate rows of the step
    SpecState s[kAcceptBatch];
    SampleParams* p[kAcceptBatch] = {};
    int k[kAcceptBatch] = {}, r0[kAcceptBatch] = {};
};
void spec_accept_batch(const float* gathered, int ngpu, const AcceptBatch& b, cudaStream_t st);
