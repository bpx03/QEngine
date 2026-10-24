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

// ---- lookup drafts on the GPU (in the speculative step graph)
// Device history hd[0, *hn) = the sequence's tokens (prompt + emitted, the pending one last). After a step its emitted
// tokens (emit[0] count, emit[1..]) are appended; the longest earlier match (<= 32) of the history's suffix gives up to
// lmax drafts: the tokens that followed it. Policy: match >= mmin -> min(lmax, available) drafts, else 0 (MTP step).
struct LookupDev {
    int* hd = nullptr;                 // [cap] history
    int* hn = nullptr;                 // length
    unsigned long long* best = nullptr;
    int* lk = nullptr;                 // [32]: L, drafts...
    int cap = 0;
    int mmin = 6, lmax = 7;
};
void lookup_append(const LookupDev& d, const int* emit, cudaStream_t st);
void lookup_scan_emit(const LookupDev& d, cudaStream_t st, int* out = nullptr);  // -> d.lk (+ *out = L); resets best
void lookup_to_tokens(const LookupDev& d, int* tokens, int L, cudaStream_t st);  // tokens[1..L] = drafts
// conditional-graph helper: sets h to (d.lk[0] == 0) (the MTP draft chain runs only when no lookup drafts follow)
void lookup_cond(const LookupDev& d, cudaGraphConditionalHandle h, cudaStream_t st);
// *dst = *src + d (device positions of the second 8-row half of a long verification)
void spec_add_pos(int* dst, const int* src, int d, cudaStream_t st);

// Lookup drafts in batched steps: one history per slot (sequence z of the step).
// fill (start of the step): slots with rows[z] > 0 verify rows[z] drafts: the lookup drafts found by the slot's last
// step (padded with the last one) or, when it found none, its kav[z] MTP drafts padded; used[z] = 1 if lookup drafts.
// pos8[z] (non-null: more than 8 verification rows): *pos[z] + 8 (the second attention segment).
// step (end of the step): append the emitted tokens of every slot, scan, drafts for the next step; L at emit[z][31].
struct LkBatch {
    int n = 0;
    LookupDev d[32];
    int* tokens[32];
    int* emit[32];
    int rows[32];
    int kav[32];
    int* pos8[32];
    const int* pos[32];
    int* used = nullptr;
};
void lookup_fill_batch(const LkBatch& b, cudaStream_t st);
void lookup_pos8_batch(const LkBatch& b, cudaStream_t st);  // only *pos8 = *pos + 8
void lookup_step_batch(const LkBatch& b, cudaStream_t st);
// history length := n (the host's committed tokens), then scan (drafts for the slot's next step)
void lookup_set_len_scan(const LookupDev& d, int n, cudaStream_t st);
