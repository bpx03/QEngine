#pragma once
// Latent Sparse Attention (D-LSA flat, arXiv 2610.42147; design ideas only, own implementation) for the 27B
// full-attention layers: index keys / queries in a 64-dim space (released maps over the RoPE-free dims 64..255 of
// the 4 kv heads), 512-key cluster blocks of 4-key pools merged by Ward into 64 buckets (cap 16 keys), each query
// attends to its top-p buckets' keys (k long-range keys, best first) + a window + sinks.
#include "common.h"
#include "ops.cuh"

constexpr int kLsaDim = 64;        // index dims
constexpr int kLsaIn = 768;        // map input: 4 kv heads x 192 RoPE-free dims
constexpr int kLsaBlk = 512;       // keys per cluster block = window
constexpr int kLsaPool = 4;        // keys per pool
constexpr int kLsaPools = kLsaBlk / kLsaPool;  // 128 pools per block
constexpr int kLsaBpb = 64;        // buckets per block (8 keys on average)
constexpr int kLsaCap = 16;        // keys per bucket at most (4 pools)
constexpr int kLsaK = 4096;        // long-range keys per query
constexpr int kLsaP = 1024;        // buckets per query
constexpr int kLsaSink = 4;
constexpr int kLsaListMax = kLsaK / kLsaPool + 8;  // pool entries of a row's long-range list (partial pools at the cut)
constexpr int kLsaBpbMax = 128;  // buckets per block: 64 (D-LSA, Ward) or 128 (QE_LSA_MODE=block: one 4-key pool each)
// runtime budget (QE_LSA_MODE): buckets per block, long-range keys, buckets per query (<= the k / p constants above)
struct LsaCfg {
    int bpb, k, p;
};
extern LsaCfg g_lsa;
#ifdef __CUDA_ARCH__
extern __constant__ LsaCfg c_lsa;
#define LSA_BPB c_lsa.bpb
#define LSA_K c_lsa.k
#define LSA_P c_lsa.p
#else
#define LSA_BPB g_lsa.bpb
#define LSA_K g_lsa.k
#define LSA_P g_lsa.p
#endif
void lsa_config(const LsaCfg& c);
constexpr int kLsaTileRows = 8;  // rows sharing one union in the attention
constexpr int kLsaUnionMax = kLsaTileRows * kLsaListMax + (kLsaBlk + kChunk) / kLsaPool + 8;  // lists + a lagging window
// the union kernel's dynamic smem: a bitmap of W = (max_pos / 4 + 32) / 32 words + kLsaUnionMax masks
constexpr int kLsaUnionSmem = 92 * 1024;
constexpr int lsa_union_words(long long max_pos) { return (int)((max_pos / 4 + 32) / 32); }
// longest context the bitmap holds (derived; QE_LSA is refused above it)
constexpr int kLsaMaxPos = (int)((((long long)kLsaUnionSmem / 4 - kLsaUnionMax + 1) * 32 - 33) * 4 + 3);
static_assert(((long long)lsa_union_words(kLsaMaxPos) + kLsaUnionMax) * 4 <= kLsaUnionSmem, "LSA context limit");
static_assert(((long long)lsa_union_words(kLsaMaxPos + 1) + kLsaUnionMax) * 4 > kLsaUnionSmem, "LSA context limit tight");
constexpr int kLsaXRows = 8192;    // rows of one exchange (index queries: a micro-batch; keys: 16 blocks of a catch-up)

// Bucket cache of one layer: centroid c[b][64] (bf16, unit), members m[b] = 4 pool ids (u8, position order, 0xFF
// pad) inside the block b / 64.
struct LsaLayer {
    bf16* cent = nullptr;
    unsigned* mem = nullptr;
    const bf16* wq = nullptr;  // map [768][64] bf16
    const bf16* wk = nullptr;
};

// What this card adds to the index partials: q heads (local index) whose q it counts (each global q head on exactly
// one card) and, per local kv head, whether it counts the positions it owns (split head: its share; a whole head: only
// on the lowest-rank card that holds it).
struct LsaRoles {
    unsigned qmask = 0;   // bit j: local q head j counted
    int qg[32] = {};      // global kv group of local q head j
    int kcount[kMaxLocalKv] = {};
    int kg[kMaxLocalKv] = {};  // global kv head of local kv head j
};

// Exchange of index partials between the cards (P2P stores into the peers' receive rows, flags with a device
// sequence; every card sums the three partials in rank order: bit-identical on all cards).
struct LsaXchg {
    int rank = 0, ngpu = 1;
    float* rx = nullptr;               // own receive rows [2 slots][ngpu senders][kLsaXRows][64]
    unsigned* rf = nullptr;            // own flags [2 slots][ngpu senders]
    float* prx[4] = {};                // peers' rx (P2P pointers)
    unsigned* prf[4] = {};
    unsigned* seq = nullptr;           // device sequence counter (bumped per exchange)
    unsigned* done = nullptr;          // block arrival counter of the send kernel
    unsigned* stats = nullptr;         // [2] rows that took the exact fallback / rows past its capacity (QE_LSA_STATS)
};

// Index query partials of m rows from this card's q (bf16 [m][nq][256], after attn_prep): out [m][64] fp32
void lsa_iq_partial(const bf16* q, int nq, int m, const LsaRoles& r, const bf16* wq, float* out, cudaStream_t st);
// Index key partials of keys [k0, k1) from this card's fp8 K cache (owned rows): out [k1 - k0][64] fp32
void lsa_ik_partial(const void* kc, const AttnShape& s, float ks, int k0, int k1, const LsaRoles& r, const bf16* wk,
                    float* out, cudaStream_t st);
// sum of the cards' partials [n][64] (n <= kLsaXRows), normalised, bf16 -> out [n][64]
void lsa_exchange(const LsaXchg& x, const float* part, int n, bf16* out, cudaStream_t st);
// Ward clustering of the cluster blocks [b0, b1): index keys ik[(b - b0) * 512 + i][64] (unit bf16) -> cent / mem
void lsa_cluster(const bf16* ik, int b0, int b1, const LsaLayer& L, cudaStream_t st);
// Per row i of m (position pos[0] + i, `closed` = clustered blocks of the layer, device): exact top-p buckets of the
// visible ones (min(p / 512, closed) blocks), expanded best first into pools of long-range keys:
// list[i][kLsaListMax] = pool id << 4 | key mask (4 bits), count n[i]. vmax: upper bound of the visible buckets (grid
// size, score pitch). scr: lsa_select_scratch_bytes. Graph-safe (no host value depends on the position).
size_t lsa_select_scratch_bytes(int max_pos, int rows, int bpb);  // bpb: the mode's buckets per block (max_pos 0: the fixed part)
void lsa_select(const bf16* iq, int m, int vmax, const int* pos, const int* closed, const LsaLayer& L, void* scr,
                size_t scr_bytes, unsigned* list, int* n, cudaStream_t st, unsigned* stats = nullptr);
// Sparse attention of m rows over list + window (min(p - 511, closed * 512) .. p) + sinks (this card's heads, owned
// keys); output like attention_prefill: whole heads final (gated) into out, position-split heads as split-K
// partials into work, then the stats exchange + combine (attention_prefill_combine, splits 1).
size_t lsa_union_bytes(int max_rows, int nkv);  // scratch of the fast path's tile unions
void lsa_attend(const bf16* q, const void* kc, const void* vc, const AttnShape& s, const float* proj, int ld,
                const int* pos, const int* closed, int m, int max_pos, const unsigned* list, const int* n, void* uscr,
                bf16* out, float* work, size_t work_floats, cudaStream_t st);
void lsa_set_int(int* p, int v, cudaStream_t st);  // *p = v on the stream
