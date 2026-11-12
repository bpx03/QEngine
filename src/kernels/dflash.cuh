#pragma once
#include "../common.h"

// DFlash2 block drafter of the 27B: 5 Qwen3-style layers (hidden 5120, 32 q / 8 kv heads x 128,
// MLP 17408), block of 8 rows (anchor + 7 mask rows, non-causal), its own context of keys / values projected from
// residual taps of 5 target layers, sliding window 2048, two-tap grouped dynamic convolutions (group 16) around
// attention and MLP, top-16 candidates per row re-ranked by a rank-256 predecessor / successor codebook selector.
// Our kernels for its small parts; the weight passes use the INT4 / bf16 GEMVs (gemv.cuh).
constexpr int kDflHid = 5120, kDflNq = 32, kDflNkv = 8, kDflHd = 128, kDflLayers = 5, kDflTaps = 5;
constexpr int kDflBlock = 8, kDflDrafts = kDflBlock - 1, kDflWin = 2048, kDflTopK = 16, kDflRank = 256;
constexpr int kDflGs = 16, kDflG = kDflHid / kDflGs;  // conv groups
constexpr int kDflKp = 4 * kDflG;                    // kernel projection rows: [conv 2][tap 2][group]
constexpr int kDflRing = kDflWin + 64;               // context ring rows (> window + the rows of a step)
constexpr int kDflSplits = 32;                       // context splits of the drafter attention
constexpr int kDflMask = 248070;

// one context ring (per sequence slot that runs block steps): K / V of every drafter layer for position p at row
// p % kDflRing and the position each row holds (-1: none)
struct DflRing {
    bf16* k[kDflLayers];
    bf16* v[kDflLayers];
    int* tag;
};

// tap j of rows [0, m): tap[(row0 + r) * 5H + j H + c] = bf16(resid[r H + c])
void dfl_tap(const float* resid, int m, int j, bf16* tap, int row0, cudaStream_t st);
// out = bf16(bf16(x * rsqrt(mean(x^2) + eps)) * w) (Qwen3 norm), x fp32 rows of stride ldx
void dfl_rms(const float* x, int ldx, const bf16* w, int m, float eps, bf16* out, cudaStream_t st);
// two-tap grouped dynamic conv over the block rows: y[t] = sum_o (base[o] + dyn[t][c][o][g]) * x[t - o] (x[-1] = 0);
// dyn: fp32 [m][kDflKp] (the kernel projection), conv c; x fp32 (rounded to bf16) or bf16.
// add != null: add[t] += y[t] (fp32 residual), else out[t] = bf16(y[t])
void dfl_conv(const float* xf, const bf16* xb, const float* dyn, int c, const bf16* base, int m, bf16* out,
              float* add, cudaStream_t st);
// block rows: proj fp32 [m][6144] (q 4096 | k 1024 | v 1024) -> per-head norm + RoPE at *pos + t ->
// q [m][32][128], k [m][8][128], v [m][8][128] bf16
void dfl_qkv_block(const float* proj, const bf16* qn, const bf16* kn, const float* inv, const int* pos, int m, float eps,
                   bf16* q, bf16* k, bf16* v, cudaStream_t st);
// context rows: kv fp32 [m][2048] (k | v) of layer l at positions *pos + row0 + t -> the ring (tags written when l == last)
void dfl_kv_ctx(const float* kv, const bf16* kn, const float* inv, const int* pos, int row0, int m, float eps,
                const DflRing* ring, int l, bool tags, cudaStream_t st);
// attention of the block (queries at *pos + t) over the ring (tag < *pos, *pos + t - tag < window) and the block keys
// -> out bf16 [m][4096]; work: fp32 [(kDflSplits + 1)][32][8][130]
void dfl_attn(const bf16* q, const bf16* kb, const bf16* vb, const DflRing* ring, int l, const int* pos, float* work,
              bf16* out, cudaStream_t st);
size_t dfl_attn_work_floats();
// rows [lo, kDflRing) of the ring with a position >= *from (or from_host >= 0) -> -1; all = every row
void dfl_ring_inval(const DflRing* ring, const int* from, int from_host, cudaStream_t st);
// top-16 of each of m rows of logits [m][v] (row id -> token id: map[i] or off + i) -> pairs [m][16][2] (value, id bits)
void dfl_topk_local(const float* logits, int v, int m, const int* map, int off, float* chunk, float* pairs, cudaStream_t st);
size_t dfl_topk_chunk_floats(int v, int m);
// merge ngpu x 16 candidates per row (gathered [ngpu][m][16][2]) -> unary [m][16] (descending), cand [m][16]
void dfl_topk_merge(const float* gathered, int ngpu, int m, float* unary, int* cand, cudaStream_t st);
// codebook rows of the anchor and every candidate (host-mapped tables [V][256] bf16) -> device rows:
// pc [1 + m * 16][256] predecessor rows (row 0: the anchor), sc [m * 16][256] successor rows
void dfl_cb_fetch(const bf16* pred_cb, const bf16* succ_cb, const int* anchor, const int* cand, int m, bf16* pc,
                  bf16* sc, int V, cudaStream_t st);  // ids clamped to [0, V)
// sequential selector: per row j the candidate maximizing unary + sum_r P[pred][r] hp[j][r] S[c][r]; drafts -> tokens[1 + j]
void dfl_select(const float* unary, const int* cand, const float* hp, const bf16* pc, const bf16* sc, int m, int* tokens,
                int V, cudaStream_t st);  // a candidate id outside [0, V): the anchor instead
// TP: every card takes rank 0's drafts (allgather of tokens[0..7] as float bits, then row 0)
void dfl_take_row0(const float* gathered, int* tokens, cudaStream_t st);
// tok[0] = *src (the anchor), tok[1..7] = mask id
void dfl_set_block_tokens(int* tok, const int* src, cudaStream_t st);
