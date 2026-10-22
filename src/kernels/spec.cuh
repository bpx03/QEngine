#pragma once
#include "../common.h"

// Device-side speculative decoding bookkeeping (MTP drafts). Every GPU runs the same kernels
// on identical inputs, so all GPUs agree on acceptance without host round trips.
struct SpecState {
    int* tokens = nullptr;    // [k+1] main-model input: last token + k drafts (== Worker::tokens_)
    int* pos = nullptr;       // main-model position of tokens[0]
    int* out = nullptr;       // [k+1] main-model argmax per input row
    int* emit = nullptr;      // [1 + k+1]: count, tokens   (read by the host)
    int* acc = nullptr;       // accepted draft count a
    int* mtp_tok = nullptr;   // [k+1] MTP input tokens
    int* mtp_pos = nullptr;   // MTP position of mtp_tok[0]
    int* mtp_out = nullptr;   // [k+1] MTP argmax per row
    int* hrow = nullptr;      // row of the hidden-state source the next MTP pass starts from
    int* drafts = nullptr;    // [k]
    int eos0 = -1, eos1 = -1; // end-of-turn ids: acceptance stops after one
};

// MTP input: out[mi] = [rmsnorm(embed(tok[mi])) * (1+we) | rmsnorm(h[hrow + mi]) * (1+wh)]
void set_image_table_spec(const bf16* table);  // see set_image_table (ops.cuh)
void mtp_prep(const bf16* table, const int* tok, const bf16* h, const int* hrow, const bf16* we,
              const bf16* wh, float eps, int m, int hidden, bf16* out, cudaStream_t st);

// After the verify pass: a = #leading drafts matching the model (never past an emitted end-of-turn
// token: the tokens after it would enter the state), emit out[0..a],
// MTP tokens = out[0..k], mtp_pos = pos.
void spec_accept(const SpecState& s, int k, cudaStream_t st);
// After the k+1-row MTP pass: drafts[0] = mtp_out[a]; next MTP step uses row a at pos+a+1.
void spec_select(const SpecState& s, cudaStream_t st);
// After a 1-row recursive MTP step: drafts[r] = mtp_out[0]; advance.
void spec_append(const SpecState& s, int r, cudaStream_t st);
// tokens = [out[a], drafts...], pos += a + 1
void spec_finish(const SpecState& s, int k, cudaStream_t st);
// the same for every sequence of a batched step in one launch (thread z = sequence z)
struct SpecBatch {
    int n = 0;
    SpecState s[32];
};
void spec_select_batch(const SpecBatch& b, cudaStream_t st);
void spec_finish_batch(const SpecBatch& b, int k, cudaStream_t st);

// Prompt end: drafts[0] = mtp_out[m-1], tokens[0] = out[m-1], next MTP step at position n
// from hidden row m-1; *acc = -1 so the caller's later commit logic is unaffected.
void spec_prefill_select(const SpecState& s, int m, int n, cudaStream_t st);
// mtp_tok[m-1] = out[m-1] (the token sampled after the prompt)
void spec_copy_last(const SpecState& s, int m, cudaStream_t st);
// tokens[1..k] = drafts (pos unchanged)
void spec_finish_prefill(const SpecState& s, int k, cudaStream_t st);
