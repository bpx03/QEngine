#pragma once
#include <cstddef>

#include "../common.h"
#include "spec.cuh"

// Sampling over a vocab split across GPUs: each GPU extracts its local top-kTopK logits per row,
// the candidates are all-gathered, and every GPU merges them identically (same seed and
// counter), so all GPUs pick the same tokens without host round trips.
constexpr int kTopK = 64;

// Presence / frequency penalty state of one sequence slot (device, one per slot and card, fixed address): how often
// each token occurs in the reply so far (saturating), plus the tokens committed at the last positions so a step that
// is undone and redone (launch-ahead) replaces its predecessor's tokens instead of counting them twice.
constexpr int kPenRing = 128;  // > the most tokens one step emits
struct PenState {
    int cpos;                 // position after the last committed token (-1: nothing committed since the reset)
    int pad;
    int ring[kPenRing];       // token at position p: ring[p % kPenRing] (positions [cpos - kPenRing, cpos))
    unsigned char cnt[1];     // [vocab]: occurrences (seed tokens of the reset count once)
};
inline size_t pen_state_bytes(int vocab) { return (offsetof(PenState, cnt) + (size_t)vocab + 15) & ~(size_t)15; }

struct SampleParams {  // lives in device memory so captured graphs pick up new values
    float temperature;   // > 0
    float top_p;         // (0, 1]
    int top_k;           // 1..kTopK
    float min_p;         // 0 disables
    unsigned long long seed;
    unsigned long long counter;  // advanced by every sampling kernel
    // OpenAI-style penalties on the logits of the merged top-kTopK candidates before temperature / top-k / top-p:
    // logit -= frequency * count + presence * (count > 0), count = occurrences in the reply so far (pen) plus the
    // earlier rows of the same sequence in this launch (verify rows: their drafts are the reply if row i is used)
    float presence = 0.f, frequency = 0.f;
    PenState* pen = nullptr;     // this slot's (null: no penalty)
};
__host__ __device__ inline bool pen_active(const SampleParams& p) {
    return p.pen != nullptr && (p.presence != 0.f || p.frequency != 0.f);
}

// After acceptance: the tokens toks[1 .. toks[0]] end at position *end_pos (the last one sits there) in the slot
// whose parameters are *p; added to its penalty counts (no-op without an active penalty). A position below the last
// committed one first takes back the tokens committed from there on (the step was undone and redone).
void pen_commit(const SampleParams* p, const int* toks, const int* end_pos, cudaStream_t st);
// batched: segment s (slot segs[s]) emitted toks + s * stride ([count, tokens...]) ending at position svp[slot * 2]
// (after k_accept_b advanced it); base[slot] = the slot's parameters
void pen_commit_b(const SampleParams* base, const int* segs, int nseg, const int* toks, int stride, const int* svp,
                  cudaStream_t st);
// host-known tokens (the first token after the prompt, steps without drafts) at positions pos, pos + 1, ...; n <= 16
void pen_commit_host(const SampleParams* p, const int* toks, int n, int pos, cudaStream_t st);
// reset the slot's counts (cpos = -1), then count each distinct seed token once (seed: device, n tokens)
void pen_reset(PenState* pen, int vocab, const int* seed, int n, cudaStream_t st);

// Grammar mask: rows [r0, r0 + rows) of this GPU's logits shard ([rows][v], global ids from
// vocab_off) keep only the tokens whose bit is set in their mask bits + j * stride (masks of
// (vocab + 31) / 32 words; stride 0: the same mask for every row); others -> -inf.
void mask_logits(float* logits, int v, int vocab_off, const unsigned* bits, int r0, int rows,
                 size_t stride, cudaStream_t st);

// pairs[m][kTopK][2] = (logit, global index as float bits), sorted descending
void topk_local(const float* logits, int v, int m, int vocab_off, float* pairs, cudaStream_t st);

// Plain sampling: tokens[mi] ~ p(row mi) from gathered candidates [ngpu][m][kTopK][2].
// chain: the rows are consecutive positions of one sequence being verified (sample-then-match): for the penalty
// row mi sees the tokens sampled at rows < mi as generated (they equal the drafts whenever row mi is used)
void sample_rows(const float* gathered, int ngpu, int m, SampleParams* p, int* tokens,
                 cudaStream_t st, bool chain = false);
// batched rows of several sequences: row mi samples with base[rowslot[mi]], greedy (top candidate) when !on[slot];
// rows of one slot are its verify rows (penalty: as sample_rows with chain)
void sample_rows_slots(const float* gathered, int ngpu, int m, SampleParams* base, const int* rowslot,
                       const int* on, int* tokens, cudaStream_t st);
// the largest per-thread stack (local memory) of the sampling kernels: the biggest frames of the library
// (sample_rows_slots_kernel 1056 B, above the context's default stack limit of 1024 B; tests/kernel_stack_check.sh)
size_t sample_stack_need();

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
