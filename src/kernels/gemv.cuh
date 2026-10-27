#pragma once
#include <string>
#include <algorithm>

#include "../common.h"

// Weight matrices are stored "fragment-packed" so that one 16-byte load per lane yields
// ready-to-use mma.m16n8k16 A fragments. Activations (M <= 8 tokens) are the B operand,
// so for small batches the tensor cores do the math while the kernel streams weights at
// full memory bandwidth.
//
// W4 layout (symmetric INT4, group 128 along K, stored nibble u = w + 8):
//   qweight: [N/16][K/64][32 lanes][4 x u32]   (u32 j = k-step j of the 64-wide chunk)
//   scales : [N/16][K/128][16] bf16
// W16 layout (bf16):
//   weight : [N/16][K/16][32 lanes][4 x u32]   (one k-step per 16 bytes)
struct W4 {
    const uint4* q = nullptr;
    const bf16* s = nullptr;
    int n = 0, k = 0;
    // GLU layout (MLP gate_up): each 16-row tile holds 8 gate rows then the 8 matching up rows,
    // and the output is act[t][i] = bf16(silu(gate_i) * up_i) with n / 2 columns.
    bool glu = false;
};
struct W16 {
    const uint4* w = nullptr;
    int n = 0, k = 0;
};
// W8 layout (symmetric INT8, group 128 along K):
//   q: [N/16][K/32][32 lanes][16 bytes]  (two k-steps per 16-byte load, 8 bytes = a0..a7 each)
//   s: [N/16][K/128][16] bf16
struct W8 {
    const uint4* q = nullptr;
    const bf16* s = nullptr;
    int n = 0, k = 0;
    bool glu = false;  // GLU row order as in W4 (gate_up): gemv_w8 / gemm_w8 write silu(gate) * up (bf16)
};
void gemv_w8(const W8& w, const bf16* x, int ldx, float* y, int ldy, int m, cudaStream_t st);
void gemv_w8_set_p1_min(int m);  // (bench) run-time QE_W8P1_MIN, 0 = environment / default
// RTN INT8 (group 128) quantization of a row-major bf16 matrix straight into the W8 layout.
void quantize_w8(const bf16* src, int n, int k, uint4* dst_q, bf16* dst_s, cudaStream_t st);
inline size_t w8_qbytes(int n, int k) { return (size_t)n * k; }
// Repack a row-major compressed-tensors INT8 matrix (int32 [N][K/4], bytes LSB-first, stored value
// u = w+128, so each row is K plain bytes) plus bf16 scales [N][K/128] into the W8 layout.
void repack_w8(const int32_t* src_q, const bf16* src_s, int n, int k, uint4* dst_q, bf16* dst_s,
               cudaStream_t st);

// y[m][n] = sum_k x[m][k] * W[n][k]   (fp32 out, row stride ldy; x bf16 with row stride ldx)
// For a GLU matrix y is a bf16 activation buffer [m][ldy] (ldy >= n / 2).
// per-device scratch for the wide (> 8 token) INT4 GEMV; call once per device before use
void gemv_workspace_init();
void gemv_occupancy_report(int threads);  // debug
void gemv_set_carveout(int pct, int which);  // smem carveout of the decode GEMV kernels (gemv.cu)
void gemv_w4(const W4& w, const bf16* x, int ldx, float* y, int ldy, int m, cudaStream_t st);
// fp16-accumulate variant (QE_GEMV_H16=1; see gemv_h16.cuh): x in the XH layout (row stride ldx halves,
// 128-byte aligned rows); for a GLU matrix the activation is also written in XH form to yh (stride
// ldyh) when given. Same results for every m (bit-exact batch == alone).
bool gemv_h16_on();
void gemv_w4h(const W4& w, const __half* xh, int ldx, float* y, int ldy, int m, cudaStream_t st,
              __half* yh = nullptr, int ldyh = 0);
// XH packing of a bf16 [m][K] matrix (row stride ldx) into xh (row stride ldxh): fallback for inputs
// whose producer does not write the XH layout itself.
void xh_pack(const bf16* x, int ldx, int K, int m, __half* xh, int ldxh, cudaStream_t st);
// Skinny INT4 GEMM for 9..64 rows (bit-identical to gemv_w4; see skinny.cu). Returns false when the
// shape is not supported (caller falls back to gemv_w4). ldx must be a multiple of 8.
bool skinny_w4(const W4& w, const bf16* x, int ldx, float* y, int ldy, int m, cudaStream_t st);
// Experimental: gemv_w4h (m <= 8) launched before its input exists; stages `depth % 10` weight groups
// per warp in registers (+ `depth / 10` in smem), then waits for ready[idx] == *epoch * 1024 + idx + 1
// (engine) or flag != 0 (bench, epoch null). Bit-identical to gemv_w4h.
void gemv_w4h_early(const W4& w, const __half* xh, int ldx, float* y, int ldy, int m, int depth,
                    const unsigned* flag, int nw_override, cudaStream_t st, const unsigned* epoch = nullptr,
                    int idx = 0, __half* yh = nullptr, int ldyh = 0, const struct FusedAR* far = nullptr);

// fp16-accumulate skinny GEMM on XH activations (bit-identical to gemv_w4h for every m)
bool skinny_w4h(const W4& w, const __half* xh, int ldx, float* y, int ldy, int m, cudaStream_t st,
                __half* yh = nullptr, int ldyh = 0);
void skinny_init();            // once per device, before any stream capture
// Wide H16 GEMV for 9..64 rows (gemv_x.cu): x staged in smem per K slice, slices reduced across blocks;
// bit-identical to gemv_w4h for every m. false: not supported (no workspace / shape).
bool gemv_w4hx(const W4& w, const __half* xh, int ldx, float* y, int ldy, int m, cudaStream_t st,
               __half* yh = nullptr, int ldyh = 0);
void gemv_x_init();            // once per device (workspace), before any stream capture
// K slices (warps) per 16-row tile of the GEMV, shared with skinny; w4: INT4 decode matrices (the
// QE_GEMV_NW_SMALL rule applies only to them, not to the split-K W16 projections)
int pick_warps(int tiles, int units, bool w4 = false);
bool skinny_use(int n, int m); // engine policy (QE_SKINNY, QE_SKINNY_MIN_M, QE_SKINNY_MIN_TILES)
// skinny GEMM for INT8 layers, 1..32 rows (bit-identical to gemv_w8 for every m); false when unsupported
bool skinny_w8(const W8& w, const bf16* x, int ldx, float* y, int ldy, int m, cudaStream_t st);
bool skinny_w8_use(int n, int m);  // engine policy (QE_SKINNY8, QE_SKINNY8_MIN_M, QE_SKINNY8_MIN_TILES)
// ksplit > 1 splits K across blocks (for tiny N); partial sums go to y + s * ystride and the
// consumer adds the ksplit slices.
void gemv_w16(const W16& w, const bf16* x, int ldx, float* y, int ldy, int m, cudaStream_t st,
              int ksplit = 1, size_t ystride = 0);
// gemv_w16 on XH activations (gemv_h16.cuh) with an fp16 copy of the weights (w16_to_h): fp32
// accumulation as gemv_w16, so the numerics match the bf16 path up to fp16 sub-normals of x / 16
void gemv_w16h(const W16& w, const __half* xh, int ldx, float* y, int ldy, int m, cudaStream_t st,
               int ksplit = 1, size_t ystride = 0);
void w16_to_h(const W16& src, uint4* dst, cudaStream_t st);  // dst: w16_bytes(src.n, src.k)

// Tensor-core GEMM for t > kMaxGemvTokens tokens (prefill), same weight layouts and output
// convention as the GEMV. W4 weights are dequantized to bf16 (scale applied, rounded) first.
void gemm_w4(const W4& w, const bf16* x, int ldx, float* c, int ldc, int t, cudaStream_t st);
// on: following non-GLU gemm_w4 calls of this thread (device) write bf16 outputs (round to nearest, as
// __floats2bfloat162_rn) to c reinterpreted as bf16 [t][ldc]; GLU calls are unaffected. Default off.
void gemm_set_out_bf16(bool on);
// lazily grown GEMM workspaces (gemm.cu tl_alloc): on an out-of-memory growth fn(ctx) is called once (it frees device
// memory, returns a count for the log) and the allocation retried; reserve grows them at load time
void gemm_set_oom_hook(int (*fn)(void*), void* ctx);
void gemm_reserve_ws(int t_max, int k_max, size_t skws_bytes);
// the same for other lazily grown device workspaces (named blocks per device): qe_ws_reserve at load, qe_ws_alloc
// for a growth (takes the reserved block of that name if it is big enough; else cudaMalloc, out of memory: the oom
// hook + one retry). Returns the bytes of the block (>= bytes)
size_t qe_ws_alloc(void** p, size_t bytes, const char* what);
void qe_ws_reserve(const char* what, size_t bytes);
void gemm_set_sk_maxparts(int n);  // (bench) cap of the stream-K split count, 0 = default
// Persisted GEMM tuning table (gemm.cu TuneStore): model = fingerprint of the weights; enables it
// (QE_TUNE_FILE=path / off). defer on: this thread's new choices are saved once, when it is turned off.
void gemm_tune_set_model(const std::string& model);
void gemm_tune_defer(bool on);
void gemm_w16(const W16& w, const bf16* x, int ldx, float* c, int ldc, int t, cudaStream_t st,
              int ksplit = 1, size_t cstride = 0);
// W4A8 prefill GEMM with an explicit configuration (experiments; gemm_w4 selects it by
// QE_PREFILL_INT8): activation group G (32/64/128), tile candidate ci (0..3), kernel variant
// var (0: gemm_i8_kernel, 1: gemm_i8t_kernel).
void gemm_w4_i8(const W4& w, const bf16* x, int ldx, float* c, int ldc, int t, cudaStream_t st,
                int G, int ci, int var);
// A producer of x (t rows, k = hidden) may quantize it for the W4A8 GEMM that consumes it next
// (glu_next: the MLP gate_up, else the mixer qkv / in_proj): returns the int8 and scale
// buffers to fill (layouts of x_to_i8_kernel, group *G) and registers them for the next
// gemm_w4 call with this x pointer. false: the consumer does not run W4A8.
bool gemm_i8_prequant(const bf16* x, int t, int k, int row0, bool glu_next, int8_t** xq, float** xsc, int* G,
                      int** ctr = nullptr);  // ctr: the producer zeroes the GEMM's stream-K counter (no memset)
int gemm_i8_prequant_group(int k, bool glu_next);
void gemm_i8_prequant_ext(const bf16* x, int t, int k, int G, int8_t* xq, float* xsc);
int* gemm_i8_sk_ctr();
// dn_out writes the int8 input of the next W4A8 out_proj (k = 2048) itself (QE_DNO_I8=0: off)
bool gemm_i8_prequant_out(const bf16* x, int t, int k, int row0, int8_t** xq, float** xsc, int* G);
// W4A8 kernel variant (QE_PREFILL_INT8_KERNEL): 1 = gemm_i8t_kernel (compact scale records
// {sx0, C0, sx1, C1}), 0 = gemm_i8_kernel of master ({sx, -M sx} pairs)
int gemm_i8_variant();

// Pull the first `bytes` of a weight matrix into L2 (evict_last) - run on a side stream while an
// all-reduce leaves DRAM idle, so the next GEMV starts from L2.
void prefetch_l2(const void* p, size_t bytes, cudaStream_t st);

// Repack a row-major compressed-tensors INT4 matrix (int32 [N][K/8], nibbles LSB-first,
// stored value u = w+8) plus bf16 scales [N][K/128] into the W4 layout.
void repack_w4(const int32_t* src_q, const bf16* src_s, int n, int k, uint4* dst_q, bf16* dst_s,
               cudaStream_t st);
// Round-to-nearest symmetric INT4 (group 128) quantization of a row-major bf16 matrix into the
// compressed-tensors layout consumed by repack_w4 (used for draft-only weights).
void quantize_w4_rtn(const bf16* src, int n, int k, int32_t* dst_q, bf16* dst_s, cudaStream_t st);
// Reorder rows [gate 0..I) [up 0..I) of a compressed-tensors int4 (bits 4) or int8 (bits 8) matrix
// (+ scales) into the GLU tile order (8 gate rows, 8 up rows, ...), in place via the scratch copies.
void glu_interleave_rows(const int32_t* src_q, const bf16* src_s, int n, int k, int32_t* dst_q,
                         bf16* dst_s, cudaStream_t st, int bits = 4);
// Repack a row-major bf16 matrix [N][K] into the W16 layout.
void repack_w16(const bf16* src, int n, int k, uint4* dst, cudaStream_t st);

inline size_t w4_qbytes(int n, int k) { return (size_t)n * k / 2; }
inline size_t w4_sbytes(int n, int k) { return (size_t)n * (k / 128) * sizeof(bf16); }
inline size_t w16_bytes(int n, int k) { return (size_t)n * k * sizeof(bf16); }
inline void prefetch_l2(const W4& w, size_t cap, cudaStream_t st) {
    prefetch_l2(w.q, std::min(cap, w4_qbytes(w.n, w.k)), st);
}

// A linear layer stored as INT4, INT8 (decode-only) or BF16.
struct Lin {
    W4 q;
    W16 f;
    W8 q8;
    bool is4 = false, is8 = false;
};
inline void gemv(const Lin& w, const bf16* x, int ldx, float* y, int ldy, int m, cudaStream_t st) {
    if (w.is4) gemv_w4(w.q, x, ldx, y, ldy, m, st);
    else if (w.is8) gemv_w8(w.q8, x, ldx, y, ldy, m, st);
    else gemv_w16(w.f, x, ldx, y, ldy, m, st);
}
// Picks GEMV (m <= kMaxGemvTokens) or GEMM.
inline void matmul(const W4& w, const bf16* x, int ldx, float* y, int ldy, int m, cudaStream_t st) {
    if (m <= kMaxGemvW4Tokens) gemv_w4(w, x, ldx, y, ldy, m, st);
    else gemm_w4(w, x, ldx, y, ldy, m, st);
}
inline void matmul(const W16& w, const bf16* x, int ldx, float* y, int ldy, int m, cudaStream_t st) {
    if (m <= kMaxBatchRows) gemv_w16(w, x, ldx, y, ldy, m, st);
    else gemm_w16(w, x, ldx, y, ldy, m, st);
}
inline void matmul(const Lin& w, const bf16* x, int ldx, float* y, int ldy, int m, cudaStream_t st) {
    CHECK(!w.is8 || m <= kMaxBatchRows, "W8 layers are decode-only");
    if (w.is8) gemv_w8(w.q8, x, ldx, y, ldy, m, st);
    else if (w.is4) matmul(w.q, x, ldx, y, ldy, m, st);
    else matmul(w.f, x, ldx, y, ldy, m, st);
}
