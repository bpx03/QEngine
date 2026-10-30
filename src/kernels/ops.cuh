#pragma once
#include "../common.h"
#include <cfloat>

// All kernels operate on one GPU's slice for m <= kMaxGemvTokens tokens. `pos` is a device
// pointer holding the absolute position of the first token of the step, so captured CUDA
// graphs can be replayed without re-recording.

constexpr int kConvRing = 16;     // delta-net conv input history ring (>= conv_k - 1 + max m)
// attn_dec8d row split pays while the ~38-split rule holds (one wave of row-tile blocks); beyond it 82 splits
// x row tiles are several waves
constexpr int kD8RsMax = 53248;
constexpr int kAttnSplits = 82;   // flash-decoding sequence splits per kv head (one per 3090 SM)
constexpr int kMaxGroup = 48;
#ifdef __CUDACC__
// Keys per split of the fp8 decode attention for a query block starting at position p0 (depends
// only on p0, so a token's attention does not depend on how many drafts are verified with it):
// at least 1/82 of the keys (one split per SM), and at least ~1/24 of them clamped to [64, 256]
// so short contexts do not pay for many tiny splits in the combine.
// win < 0 (MTP default): a 32K window beyond 144K keys only (measured: at 180K the drafts lose
// little acceptance and the step is ~1 ms shorter; at <= 120K full attention drafts better)
__device__ __forceinline__ int dec8_first(int p0, int win) {  // first key of the window
    if (win < 0) return p0 > 147456 ? p0 - 32768 : 0;
    return win > 0 && p0 > win ? p0 - win : 0;
}
__device__ __forceinline__ int dec8_chunk(int p0, int cmin = -1, int max38 = 53248) {  // p0: keys before the query block
    // Depends only on the length (and QE_DEC_CHUNK_MIN) -> MTP == greedy and batched == alone stay bit-exact.
    // cmin < 0 (default): ~38 splits up to 52K keys: 2 kv heads x 38 = 76 blocks = one wave on 82 SMs at n = 1,
    // and far less combine / better GB/s for batches (42-53 splits leave a near-empty second wave; beyond ~64K
    // 38 splits starts to hurt n = 1, so the old ~82 splits stay there). cmin = 0: the old L/82 rule;
    // cmin > 0: old rule with at least cmin keys per split.
    const int L = p0 + 8;
    int lo = (L + 23) / 24;
    lo = lo < 64 ? 64 : (lo > 256 ? 256 : lo);
    const int dv = (cmin < 0 && L <= max38) ? 38 : 82;  // max38: QE_DEC38_MAX (default 53248)
    int c = (L + dv - 1) / dv;
    c = c < lo ? lo : c;
    c = c < cmin ? cmin : c;
    return (c + 15) / 16 * 16;
}
#endif     // max (q heads per kv head) * m handled by one attention block

// resid[m][hidden] = float(table[tokens[i]])   (table is host-mapped pinned memory)
// Token ids >= kImgToken select rows of the image embedding table (vision encoder output for
// the current prompt, host-mapped) instead of the vocabulary.
void set_image_table(const bf16* table);  // current device
void embed_gather(const bf16* table, const int* tokens, int m, int hidden, float* resid,
                  cudaStream_t st);

// out = bf16(x * rsqrt(mean(x^2) + eps) * (1 + w))
// xh (optional): the rows also in the XH layout of gemv_h16.cuh (stride n)
void rmsnorm(const float* x, const bf16* w, int m, int n, float eps, bf16* out, cudaStream_t st,
             __half* xh = nullptr);

// Position ownership of a kv head on this GPU (branch kv-pos-split): the keys of a kv head may be
// split between two GPUs by absolute position with period P: owned(k) <=> ((k + O) mod P) < A.
// A == P: the whole head lives here. rank(k) = owned keys below k = the local row of key k in a
// region whose first key is 0 (a region of keys [split, ..) holds key k at rank(k) - rank(split));
// unrank(l) = the l-th owned key; cap(c) = rows any window of c keys can need (the array size of
// a region of c keys). Ownership is a function of the absolute position only, so attention over
// the same tokens gives the same result in every slot / layout, and splits / batches cannot
// change it.
struct PosMap { int P = 1, A = 1, O = 0; };
constexpr int kMaxLocalKv = 4;  // kv heads per GPU with position maps (kernel parameters stay under 4 KB)
constexpr int kPattnKscRows = 262144 + 4096;  // local key rows of the per-key int8 K scale scratch (max context 256K)
#ifdef __CUDACC__
#define QE_HD __host__ __device__ __forceinline__
#else
#define QE_HD inline
#endif
QE_HD int pm_R(int x, int P, int A) { const int r = x % P; return (x / P) * A + (r < A ? r : A); }
QE_HD int pm_rank(const PosMap& m, int k) { return m.A == m.P ? k : pm_R(k + m.O, m.P, m.A) - pm_R(m.O, m.P, m.A); }
QE_HD int pm_unrank(const PosMap& m, int l) {
    if (m.A == m.P) return l;
    const int y = l + pm_R(m.O, m.P, m.A);
    return (y / m.A) * m.P + y % m.A - m.O;
}
QE_HD int pm_cap(const PosMap& m, int c) { return m.A == m.P ? c : pm_R(c, m.P, m.A); }
QE_HD bool pm_full(const PosMap& m) { return m.A == m.P; }
QE_HD bool pm_owned(const PosMap& m, int k) { return m.A == m.P || (k + m.O) % m.P < m.A; }
#ifdef __CUDACC__
// system-scope flag ops of the stats exchange (same as the all-reduce barrier in comm.cu)
__device__ __forceinline__ void qe_st_release_sys(unsigned* p, unsigned v) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned qe_ld_acquire_sys(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
#endif

// Scratch of the persistent windowed prefill attention (attn_pkp_kernel), carved out of buffers that are
// idle during a full-attention layer: two window buffers of w key rows per local kv head (K int8 [w][256],
// V fp16 [w][256] in attn_pf8's dim permutation), Q int8 MMA fragments + row scales of the call
// ([qt][head][warp][kk][lane] uint4, [qt][head][128] float; null: quantized per item), and the
// work-queue counters (kPkpCtrInts ints, zeroed before every call).
constexpr int kPkpCtrInts = 4096;
struct PattnScratch {
    uint8_t* kq[2][kMaxLocalKv] = {};
    uint4* vh[2][kMaxLocalKv] = {};
    int w = 0;
    uint4* qf = nullptr;
    float* qsq = nullptr;
    int qt_max = 0;
    int* ctr = nullptr;
};

struct AttnShape {
    int nq, nkv;          // local heads
    int kv_qb[8], kv_qe[8];  // local q-head range of each local kv head
    int q2kv[32];
    int max_t;            // kv cache capacity (tokens) of the slot's own region
    // kv-pos-split: ownership map of each local kv head; its array in a region of cap keys has
    // pm_cap(cap) rows, arrays of a region follow each other in local head order
    PosMap kv_pm[kMaxLocalKv];
    // Softmax-stats exchange of a position-split head (kv_x == 3): both cards of the pair compute
    // the unnormalized numerator N_i and the stats (max m_i, sum l_i) of the head's q heads over
    // their own keys; they exchange only (m_i, l_i) (8 B per row and q head: xo / xflag = this
    // card's receive rows and flag, xpo / xpflag = the peer's) and each scales its own N_i by
    // e^(m_i - M) / D with the global M, D. o_proj is linear and its columns for these heads live
    // on both cards, so the all-reduce after o_proj sums the two contributions - no partial-O
    // transfer, no cross-card combine. kv_x == 0: a whole head. xepoch / xidx: the flag value of
    // this exchange (monotone), xdone: block arrival counter of the stats kernel.
    int kv_x[kMaxLocalKv] = {};
    float* xo[kMaxLocalKv] = {};
    unsigned* xflag[kMaxLocalKv] = {};
    float* xpo[kMaxLocalKv] = {};
    unsigned* xpflag[kMaxLocalKv] = {};
    const unsigned* xepoch = nullptr;  // launch-unit epoch (CommCounters::epoch): flag value = *xepoch * 4096 + xidx + 1
    int xidx = 0;                      // index of this exchange within the launch unit
    unsigned* xdone = nullptr;
    // local q heads [ob, ob + no) are this GPU's o_proj heads: only they are combined into `out`
    // (rows of no * 256); no = 0: all nq heads
    int ob = 0, no = 0;
    int xrow0 = 0;        // row of this call's first token in the exchange buffers (single-sequence path)
    // 1: a sender skips the merge / fence when it holds no keys in the window and a receiver does
    // not wait then (both decide from the same positions); 0 (QE_KV_XSKIP=0): always send and wait
    int xskip = 1;
    // position-split heads: flash-decoding chunk rule (dec8_chunk_pm): 0 = the chunk of a full head
    // of the same absolute length (default), > 0 = at most ~xdiv splits, < 0 = rule on the local length
    int xdiv = 0;
    // RoPE table of the step's rows (rope_table): [row][32] (cos, sin) in double, row = xrow0 + mi
    // (seg kernels: the sequence's first row + mi); null: attn_prep computes the trig itself
    const double2* rtab = nullptr;
    // Shared prefix (branch cache-priority): keys [0, split) are read from kc2 / vc2
    // ([nkv][split][256], a region shared by several slots), keys >= split from the slot's own
    // region (kc / vc) at row key - split; split = 0: everything in the own region.
    int split = 0;
    int pf_splits = 1;    // prefill attention: split-K count of attn_pf8_kernel's 1-D grid (set by the launcher)
    int pf_p0 = -1;       // prefill attention: position of the call's first query if the host knows it (-1: unknown)
    // prefill attention scratch for the per-key int8 K scales ([nkv][ksc_ld] (inv half2, scale)); null: computed in-kernel
    uint2* ksc = nullptr;
    int ksc_ld = 0;
    float* kmu = nullptr;  // [nkv][256] per-channel K sums for K smoothing (QE_PATTN_KSMOOTH), with ksc
    // windowed precomputed K / V of the prefill attention: [nkv][kvw][256] int8 and [nkv][kvw][256] fp16
    uint8_t* kvq = nullptr;
    uint4* kvh = nullptr;
    int kvw = 0;
    // persistent windowed prefill attention (attn_pkp_kernel): double-buffered K / V windows, Q scratch,
    // counters (host pointer; the kernel gets the struct by value)
    const struct PattnScratch* pscr = nullptr;
    const void* kc2 = nullptr;
    const void* vc2 = nullptr;
    int fp8 = 0;          // cache holds e4m3 values x / scale (else bf16)
    float ks = 1.f, vs = 1.f;  // per-tensor k / v scales (fp8 only)
    float* amax = nullptr;     // KV scale calibration (QE_KV_AMAX): max |k| at [0], |v| at [1] before the cast
    int dchunk_min = -1;  // QE_DEC_CHUNK_MIN: fp8 decode split rule (-1 = 38 splits up to 52K, 0 = old L/82)
    // attn_dec8d row split: 1 = one 16-row tile per block (decode of one sequence below
    // kD8RsMax keys, set by the engine from the host-side length); results are bit-identical either way
    int d8rs = 0;
    // keys up to which the ~38-split rule holds (QE_DEC38_MAX; length-only -> MTP == greedy, batched == alone);
    // the engine's row split (d8rs) uses the same bound
    int d38max = 53248;
    int win = 0;          // decode (fp8 kernel): attend only to the last `win` keys (0 = all)
    // RoPE position of cache index p: rpos3 rows (multimodal prompt chunk: [0] rows, [1] first
    // cache index, then (t, h, w) per row) if p falls in them, else p + *rdelta (text after
    // images is shifted, Qwen M-RoPE); both null = p
    const int* rdelta = nullptr;
    const int* rpos3 = nullptr;
};

// RoPE (cos, sin) of every rotary frequency for rows [xrow0, xrow0 + m) of the step at positions
// pos[0] + mi (rdelta / rpos3 of the shape apply); the attention preps of all layers read it
// instead of computing the double-precision trig per head.
void rope_table(const AttnShape& s, const int* pos, int m, const float* inv_freq, double2* tab, cudaStream_t st);
struct AttnSegs;
void rope_table_seg(const AttnSegs& g, const AttnShape& s, const float* inv_freq, double2* tab, cudaStream_t st);
// The per-sequence part of AttnShape (a batched kernel reads it from its AttnSeg instead of
// patching the by-value shape parameter, which would copy the whole shape to local memory in
// every thread)
struct SeqView {
    int max_t, split;
    const void *kc2, *vc2;
    const int *rdelta, *rpos3;
};
QE_HD SeqView seq_view(const AttnShape& s) { return SeqView{s.max_t, s.split, s.kc2, s.vc2, s.rdelta, s.rpos3}; }
// proj rows per token: [nq x (q 256 | gate 256)] [nkv x k 256] [nkv x v 256]
// Writes normalized+roped q (bf16) and appends k/v to the cache at pos..pos+m-1.
void attn_prep(const float* proj, int ld, const AttnShape& s, const bf16* qn_w, const bf16* kn_w,
               float eps, const float* inv_freq, const int* pos, int m, bf16* q_out, void* kc,
               void* vc, cudaStream_t st);

// Causal attention of the m new queries over the cache, then out = bf16(o * sigmoid(gate)).
// work must hold attn_work_floats(s) floats.
size_t attn_work_floats(const AttnShape& s);
void attention_init();  // once per device (opt-in shared memory)
// Tensor-core split kernel used by attention() (partials for attn_combine).
void attention_decode_tc_init();
// true unless QE_ATTN_DEC8=0: fp8 caches use attn_dec8_kernel
bool attention_dec8_enabled();
void attention_decode_tc(const bf16* q, const void* kc, const void* vc, const AttnShape& s,
                         const int* pos, int m, float* work, int G, cudaStream_t st);
// Flash-attention path for m > kMaxGemvTokens queries (prefill chunks); same semantics.
void attention_prefill_init();
// work (optional): scratch for split-K partials when there are few query tiles
void attention_prefill(const bf16* q, const void* kc, const void* vc, const AttnShape& s,
                       const float* proj, int ld, const int* pos, int m, bf16* out,
                       cudaStream_t st, float* work = nullptr, size_t work_floats = 0);
size_t attention_prefill_work_floats(int max_rows_x_splits, int nq);
// microbench / precision helpers: deterministic random fill (kind 0: fp8 e4m3
// ~N(0, sd) with sd = scale, kind 1: bf16 ~N(0, scale)) and an fp32 reference of the prefill
// attention output (before the gate) for rows [r0, r0 + nr) of all nq heads: ref[(row * nq + h) * 256]
void pattn_fill(void* p, size_t n, int kind, float scale, unsigned seed, cudaStream_t st);
void pattn_reference(const bf16* q, const void* kc, const void* vc, const AttnShape& s, const int* pos, int m,
                     int r0, int nr, float* ref, cudaStream_t st);
// attention_prefill in two steps (prefill overlap with a position split: the receiver combines
// after the other micro-batch's attention): the split-K count for m rows, the kernel, the combine
int attention_prefill_splits(const AttnShape& s, int m, size_t work_floats, bool have_work);
void attention_prefill_kernel(const bf16* q, const void* kc, const void* vc, const AttnShape& s, const float* proj,
                              int ld, const int* pos, int m, bf16* out, cudaStream_t st, float* work, int splits);
void attention_prefill_combine(const AttnShape& s, const float* proj, int ld, const int* pos, int m, bf16* out,
                               cudaStream_t st, float* work, int splits);
// true with the default fp8 decode kernels (attn_dec8d, QE_ATTN_DEC8D=2), which the position split needs
bool attention_dec8d_fused();
// xh (optional): out also in the XH layout (stride nq * 256)
void attention(const bf16* q, const void* kc, const void* vc, const AttnShape& s,
               const float* proj, int ld, const int* pos, int m, float* work, bf16* out,
               cudaStream_t st, __half* xh = nullptr);

// Depthwise causal conv (k=4) + SiLU over the delta-net q|k|v channels.
// proj rows per token start with the ch conv channels. hist: [kConvRing][ch] bf16.
void dn_conv(const float* proj, int ld, int ch, const bf16* conv_w, bf16* hist, const int* pos,
             int m, float* out, cudaStream_t st);

struct DeltaShape {
    int nk, nv;     // local key / value heads
    int v2k[64];    // local v head -> local k head
    int ab_pad;     // ab rows: [b: nv .. pad][a: nv .. pad]
    int ab_splits;  // ab holds this many split-K partial slices of [m][2*ab_pad]
    int ab_rows;    // row capacity of each slice
    int lazy_t;     // deferred commit: write the state once this many rows are pending (<= 1: always)
};
constexpr int kAbSplits = 16;

// Gated delta rule recurrence over m tokens. Each delta-net layer keeps kStateSlots = 2 state
// slots of [nv][128 v][128 k] fp32 (ping-pong); the current one is slot *cur. A normal pass
// writes its final state to slot (cur + m) % 2 and dn_commit advances cur by m.
// write_all (speculative verification) writes no state at all: the per-token inputs stay in
// the caller's per-layer buffers and dn_replay rebuilds the state of the accepted prefix.
constexpr int kStateSlots = 2;
constexpr int kMaxSpecK = 7;  // verification rows k + 1 <= 8 = kMaxGemvTokens
// slot_off: start from state slot (*cur + slot_off) (second micro-batch of a prefill chunk)
// Decode front end fused: conv + q/k normalization + decay/beta (QE_DN_FUSE=0: separate kernels)
bool dn_fuse_on();
void dn_convqk(const float* proj, int ld, int ch, const bf16* conv_w, bf16* hist, const int* pos,
               int m, float* cv, const DeltaShape& s, const float* ab, const float* A,
               const float* dt_bias, float* qk, float* gb, cudaStream_t st);
// pn / pend (optional, write_all only): deferred commit, see dn_stash_seg
void dn_core_only(const float* conv_out, int ch, const DeltaShape& s, float* S, size_t slot_floats,
                  const int* cur, int m, bool write_all, const float* qk, const float* gb, float* o,
                  cudaStream_t st, const int* pn = nullptr, const float* pend = nullptr);
// in_proj rows + conv / gate inputs for the chunk prep that computes the front end itself
struct DnConvIn {
    const float* proj = nullptr;
    int ld = 0;
    const bf16* w = nullptr;
    const bf16* hist = nullptr;
    const int* pos = nullptr;
    const float* ab = nullptr;
    const float* A = nullptr;
    const float* dt_bias = nullptr;
};
// Chunked delta rule (dn_chunk.cu) for prompts of at least dn_chunk_min() tokens (0 = off)
int dn_chunk_min();
// the sequential recurrence regardless of dn_chunk_min (reference for tests)
void dn_core_seq(const float* conv_out, int ch, const DeltaShape& s, const float* ab, const float* A,
                 const float* dt_bias, float* S, size_t slot_floats, const int* cur, int m,
                 float* qk, float* gb, float* o, cudaStream_t st);
void dn_chunk_init();
// load time: the chunk scratch for nv value heads and chunks of up to m_max rows, reserved per device (see qe_ws_alloc)
void dn_chunk_reserve(int nv, int m_max);
void dn_core_chunked(const float* conv_out, int ch, const DeltaShape& s, const float* qk,
                     const float* gb, float* S, size_t slot_floats, const int* cur, int m,
                     float* o, cudaStream_t st, int slot_off, bool h16 = false, const struct DnConvIn* ci = nullptr);
// Prefill chunks (m >= dn_chunk_min, QE_DN_CQK=0: dn_conv + dn_core as before): fused conv + q/k
// normalization + decay/beta (only the v part of cv is written), history update, chunked rule
bool dn_prefill_fused(int m);
// gate projection of a prefill micro-batch into one ab slice (dn_ab.cu; consumers then use
// ab_splits = 1): ab[t][c] = W x_t for the 32 rows of the W16 matrix (QE_DN_AB=0: gemm_w16 split-K)
struct W16;
bool dn_ab_prefill_ok(const W16& w);
void dn_ab_prefill(const W16& w, const bf16* x, int ldx, float* out, int ldo, int m, cudaStream_t st);
void dn_prefill(const float* proj, int ld, int ch, const bf16* conv_w, bf16* hist, const int* pos,
                const DeltaShape& s, const float* ab, const float* A, const float* dt_bias, float* S,
                size_t slot_floats, const int* cur, int m, float* cv, float* qk, float* gb, float* o,
                cudaStream_t st, int slot_off);
void dn_core(const float* conv_out, int ch, const DeltaShape& s, const float* ab, const float* A,
             const float* dt_bias, float* S, size_t slot_floats, const int* cur, int m,
             bool write_all, float* qk, float* gb, float* o, cudaStream_t st, int slot_off = 0,
             const int* pn = nullptr, const float* pend = nullptr);
// Replay of a speculative step for every delta-net layer at once: state(cur) + tokens 0..*acc
// (inputs kept by the verification pass) -> slot cur ^ 1. Follow with dn_commit(cur, null, 1).
constexpr int kMaxLinLayers = 64;
struct DnReplay {
    int n;                       // layers
    float* S[kMaxLinLayers];     // state slots base
    const float* cv[kMaxLinLayers];  // conv outputs [m][ch] (v section used)
    const float* qk[kMaxLinLayers];  // [m][2][nk][128]
    const float* gb[kMaxLinLayers];  // [m][2][nv]
};
void dn_replay(const DnReplay& r, int ch, const DeltaShape& s, size_t slot_floats, const int* cur,
               const int* acc, cudaStream_t st);

// Batched decode: several sequences (segments of the rows) in one launch.
constexpr int kMaxSegs = 32;  // sequences of one batched step (one decode row each beyond 8)
struct AttnSeg {        // one sequence's attention call (rows start at the given pointers)
    const float* proj;  // q/gate/k/v projections of its rows
    bf16* q;            // roped q (written by prep, read by attention)
    void* kc;
    void* vc;           // its KV cache
    const int* pos;
    const int* rdelta;  // its RoPE position shift (AttnShape::rdelta)
    int m, max_t;
    float* work;        // its own split partials (attn_work_floats)
    bf16* out;
    __half* xh;         // optional: out also in the XH layout (stride nq * 256)
    const void* kc2 = nullptr;  // shared prefix of this sequence (AttnShape::split)
    const void* vc2 = nullptr;
    int split = 0;
};
struct AttnSegs {
    int n, max_m;
    AttnSeg c[kMaxSegs];
};
QE_HD SeqView seq_view(const AttnSeg& c) { return SeqView{c.max_t, c.split, c.kc2, c.vc2, c.rdelta, nullptr}; }
// kv-pos-split: one (row, q head) record of the stats exchange: (max, sum)
constexpr int kXchgFloats = 2;
// first row of sequence z of a batched step (rows of the sequences follow each other)
QE_HD int seg_row0(const AttnSegs& g, int z) {
    int r = 0;
    for (int i = 0; i < z; i++) r += g.c[i].m;
    return r;
}
// row offset of local kv head kv's array in a region of cap keys (the arrays of the local heads
// follow each other, each pm_cap(cap) rows); the legacy layout is kv * cap
QE_HD size_t head_rows(const AttnShape& s, int kv, int cap) {
    size_t o = 0;
    for (int j = 0; j < kv; j++) o += pm_cap(s.kv_pm[j], cap);
    return o;
}
// local rows of a kv head in the window [k0w, L) of a sequence / the local split rule, shared by
// the attention kernels, the combine and the sender so all agree bit for bit
#ifdef __CUDACC__
// Flag value of exchange xidx of the launch unit with epoch *xepoch (bumped by comm_begin_unit),
// compared by the receiver as (int)(flag - expected) < 0. The sender raises the flag on EVERY
// exchange (also when it has no keys to send), so flag and expected never drift apart by more
// than one exchange and the wrap-around of the unsigned sequence is harmless.
__device__ __forceinline__ unsigned xchg_seq(const AttnShape& s) { return *s.xepoch * 4096u + (unsigned)s.xidx + 1u; }
// keys per flash-decoding split of kv head kv over `len` local keys (len_abs absolute keys) before
// the query block. A position-split head takes the chunk a full head of the same absolute length
// would use (xdiv = 0, default): its blocks stream as many keys as the full heads' blocks, so the
// split count follows the share (fewer partials) without longer blocks; xdiv > 0: at most ~xdiv
// splits over the local keys; xdiv < 0: the rule on the local length only.
__device__ __forceinline__ int dec8_chunk_pm(const AttnShape& s, int kv, int len, int len_abs) {
    const bool full = pm_full(s.kv_pm[kv]);
    int c = dec8_chunk(full || s.xdiv < 0 ? len : len_abs, s.dchunk_min, s.d38max);
    if (s.xdiv > 0 && !full) {
        int c2 = (len + 8 + s.xdiv - 1) / s.xdiv;
        c2 = (c2 + 15) / 16 * 16;
        c = c < c2 ? c2 : c;
    }
    return c;
}
__device__ __forceinline__ bool xchg_needed(const AttnShape& s, int kv, int p0, int mm, int win) {
    // the PEER of this head (owning the complement of the positions) has keys in the window
    if (!s.xskip) return true;
    const PosMap& m = s.kv_pm[kv];
    const int k0w = dec8_first(p0, win), L = p0 + mm;
    return (L - pm_rank(m, L)) - (k0w - pm_rank(m, k0w)) > 0;
}
__device__ __forceinline__ bool xchg_has(const AttnShape& s, int kv, int p0, int mm, int win) {
    // THIS GPU has keys of the head in the window (what the peer's xchg_needed sees: the maps of
    // a pair are complementary)
    if (!s.xskip) return true;
    const PosMap& m = s.kv_pm[kv];
    const int k0w = dec8_first(p0, win), L = p0 + mm;
    return pm_rank(m, L) - pm_rank(m, k0w) > 0;
}
// the q heads of the position-split kv heads (kv_x == 3), in head order; returns their count
__device__ __forceinline__ int xchg_heads(const AttnShape& s, int* xh) {
    int nx = 0;
    for (int h = 0; h < s.nq; h++)
        if (s.kv_x[s.q2kv[h]] == 3) xh[nx++] = h;
    return nx;
}
#endif

// attn_prep + attention (decode path) for several sequences, one launch per kernel
void attention_dec8_seg(const AttnSegs& g, const AttnShape& s, int G, cudaStream_t st);
void attention_seg(const AttnSegs& g, const AttnShape& s, int ld, const bf16* qn_w,
                   const bf16* kn_w, float eps, const float* inv_freq, cudaStream_t st);
struct DnSegs {
    int n;                       // segments
    int r0[kMaxSegs], m[kMaxSegs];  // first row, rows
    const int* pos[kMaxSegs];    // sequence position (conv history ring)
    bf16* hist[kMaxSegs];        // this layer's conv history of each sequence
    float* S[kMaxSegs];          // this layer's state slots of each sequence
    const int* cur[kMaxSegs];
    const int* pn[kMaxSegs];     // deferred commit (null: off): pending rows of each sequence
    const float* pend[kMaxSegs]; // their stash for this layer
    unsigned char fin[kMaxSegs]; // 1: single row (no drafts, always accepted): write the state now
};
void dn_conv_seg(const DnSegs& g, const float* proj, int ld, int ch, const bf16* conv_w, float* out,
                 cudaStream_t st);
void dn_convqk_seg(const DnSegs& g, const float* proj, int ld, int ch, const bf16* conv_w, float* cv,
                   const DeltaShape& s, const float* ab, const float* A, const float* dt_bias,
                   float* qk, float* gb, cudaStream_t st);
void dn_qk_rows(const float* conv_out, int ch, const DeltaShape& s, const float* ab, const float* A,
                const float* dt_bias, int m, float* qk, float* gb, cudaStream_t st);
void dn_core_seg(const DnSegs& g, const float* conv_out, int ch, const DeltaShape& s,
                 const float* qk, const float* gb, size_t slot_floats, float* o, cudaStream_t st);
struct DnReplaySegs {           // replay + commit of several sequences (accepted rows of each)
    int n;
    float* S[kMaxSegs][kMaxLinLayers];
    const float* cv[kMaxLinLayers];
    const float* qk[kMaxLinLayers];
    const float* gb[kMaxLinLayers];
    int r0[kMaxSegs];
    const int* cur[kMaxSegs];
    const int* acc[kMaxSegs];
};
void dn_replay_seg(const DnReplaySegs& r, int nlayers, int ch, const DeltaShape& s,
                   size_t slot_floats, cudaStream_t st);
// Deferred commit (QE_DN_DEFER, default on): instead of a replay pass after each verification
// step (one more read + write of every state), the accepted rows' inputs (k | v | decay | beta,
// kPendRows x dn_pend_row_floats per layer) stay in the sequence's stash with their count *pn.
// The next verification pass of that sequence replays them from S[cur] in registers, writes the
// state to S[cur ^ 1] and continues with its own rows (same arithmetic and order as dn_replay: bit
// identical); dn_stash_seg then flips cur and stashes the new accepted rows. Anything else that
// reads the state first runs dn_flush. Per step: one state read + one write instead of 2 + 1.
// pn points at 2 ints: pending rows, and the pending rows to restore when the step is undone.
// Single-row sequences (fin) write the state directly (nothing pending afterwards).
// verification rows of a single-sequence step: 1 + up to 15 lookup drafts (MTP steps: k + 1 <= 8); the callers'
// output buffers hold 16 tokens
constexpr int kMaxLookupK = 12;  // rows <= 13: the conv history ring (kConvRing 16 >= conv_k - 1 + m)
constexpr int kPendRows = kMaxLookupK + 1;
__host__ __device__ inline int dn_pend_row_floats(const DeltaShape& s) {
    return s.nk * 128 + s.nv * 128 + 2 * s.nv;
}
// Lazy writes (QE_DN_LAZY = T): a verification pass writes the state only when at least T rows are
// pending, else the new accepted rows are appended to the stash (one state read per step, a write
// every ~T / accepted steps; the arithmetic is the same either way). Stash capacity T - 1 + 8 rows.
__host__ __device__ inline int dn_lazy_t(const DeltaShape& s) { return s.lazy_t > 1 ? s.lazy_t : 1; }
__host__ __device__ inline int dn_pcap(const DeltaShape& s) { return kPendRows + dn_lazy_t(s) - 1; }
// threshold of a pass with m rows: lazy only for single-row (draft-free) steps, where the write is
// half of the state traffic; with drafts the replay of more pending rows costs more than it saves
__host__ __device__ inline int dn_lazy_of(const DeltaShape& s, int m) { return m == 1 ? dn_lazy_t(s) : 1; }
bool dn_defer_on();
// single-sequence steps: deferred commit only with QE_DN_DEFER1=1 (default 0: n = 1 4K 12.86 ->
// 13.23 ms/step, the pending rows lengthen each of 48 latency-bound layers more than the one
// parallel replay costs). Otherwise run_spec flushes the pending rows of an earlier batched step
// and replays its own rows as on master; dn_single_begin sets the undo of that (save = cur, no
// pending rows).
bool dn_defer1_on();
void dn_single_begin(int* cur, int* save, int* pn, cudaStream_t st);
// keep only the first r accepted rows of a step that stashed its rows (deferred commit): r = 0 is
// the step's undo (cur := save, pn[0] := pn[1]), else pn[0] := pn[1] + r (Engine::trim)
void dn_trim(int* cur, const int* save, int* pn, int r, cudaStream_t st);
// decode recurrence kernel: lanes per value column 8 / 4 / 2 (dn_core2), 0 = dn_core_dev (QE_DN_CORE2)
void dn_core2_set(int lanes);
bool dn_direct_on();  // QE_DN_DIRECT (default 1): fin for single-row sequences of a batched step
struct DnStashSegs {
    int n;
    const float* cv[kMaxLinLayers];
    const float* qk[kMaxLinLayers];
    const float* gb[kMaxLinLayers];
    int r0[kMaxSegs];
    int* cur[kMaxSegs];
    int* pn[kMaxSegs];
    float* pend[kMaxSegs];       // stash base [layers][kPendRows][dn_pend_row_floats]
    const int* acc[kMaxSegs];
    int* save[kMaxSegs];         // undo of the step: cur := *save, pn[0] := pn[1]
    int m[kMaxSegs];             // rows of the pass (dn_lazy_of)
    unsigned char fin[kMaxSegs]; // as DnSegs::fin
};
void dn_stash_seg(const DnStashSegs& r, int nlayers, int ch, const DeltaShape& s, cudaStream_t st);
struct DnFlush {
    float* S[kMaxLinLayers];
    int* cur;
    int* pn;
    const float* pend;
};
void dn_flush(const DnFlush& r, int nlayers, const DeltaShape& s, size_t slot_floats, cudaStream_t st);
// *dst = *src + off
void add_int(int* dst, const int* src, int off, cudaStream_t st);
// cur = (cur + (adv ? *adv + 1 : adv_const)) % kStateSlots
void dn_commit(int* cur, const int* adv, int adv_const, cudaStream_t st);
// *dst = *src (saves the current slot index so a speculative step can be undone)
void copy_int(int* dst, const int* src, cudaStream_t st);

// out = bf16(w * rmsnorm(o) * silu(z)), per 128-wide value head.
// xq / xsc (prefill, gemm_i8_prequant_out): write the int8 input of the W4A8 out_proj instead of out
void dn_out(const float* o, const float* z, int ldz, const bf16* w, int nv, float eps, int m,
            bf16* out, cudaStream_t st, __half* xh = nullptr, int8_t* xq = nullptr, float* xsc = nullptr, int G = 0);

// out[m][I] = bf16(silu(gu[m][i]) * gu[m][I + i]), gu row stride ld
void silu_mul(const float* gu, int ld, int inter, int m, bf16* out, cudaStream_t st, __half* xh = nullptr);

// Per token: (max logit, global index) over this GPU's vocab slice.
// map (optional): local row -> token id (pruned draft vocabulary); otherwise id = row + off
void argmax_local(const float* logits, int v, int m, int vocab_off, float* out_pairs,
                  cudaStream_t st, const int* map = nullptr);
// Reduce gathered pairs [ngpu][m][2] -> tokens[m]
void argmax_final(const float* pairs, int ngpu, int m, int* tokens, cudaStream_t st);
