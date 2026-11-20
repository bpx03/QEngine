// R22 [dsv4-r22-dmoe8]: the DSpark drafter's cb3 expert GEMV in int8 (drafter outputs are proposals only; the verify
// pass is exact, so only the acceptance can move).
// - The drafter's cb3 experts are repacked once at load (k_cb3_rp, in place, same bytes): per 32 weights of a 128-group
//   three words of nibble selectors. Word A nibble j = code of weight j (bits 0-2) + bit 0 of weight 24 + j's code
//   (bit 3); B = weights 8 .. 15 + bit 1 of 24 + j; C = weights 16 .. 23 + bit 2 of 24 + j. Group = chunks 0 .. 3
//   (words 3 f .. 3 f + 2), the fp16 group scales unchanged.
// - The 8-entry codebook as int8 (scale max |c| / 127) in two words: one prmt (__byte_perm) turns 4 selectors into
//   4 int8 weights, one dp4a dots them with 4 int8 x (x per 32 from k_q8x, the R21 pool GEMV's quantiser).
// - About 1 integer instruction per weight (old kernel: ~8 incl. the shared-memory codebook lookups).
#pragma once
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>
#include "dsv4.h"

namespace dsv4 {

// one thread per 128-group (48 B): old cb3 layout (src) -> R22 selector layout (dst, another buffer), planar per row:
// the row's 3 G uint4 as [chunk word triple c][group g] (uint4 index c G + g), so the lanes of a row that take
// consecutive groups read consecutive 16 B (the group-contiguous layout spread a warp's 16 B loads over 48 B strides)
__global__ void k_cb3_rp(const uint8_t* src, uint8_t* dst, size_t ng, int G) {
    const size_t gi = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gi >= ng) return;
    const uint4* p = (const uint4*)(src + gi * 48);
    uint4 v0 = p[0], v1 = p[1], vh = p[2];
    const uint32_t fw[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    const uint32_t hw[4] = {vh.x, vh.y, vh.z, vh.w};
    uint32_t o[12];
#pragma unroll
    for (int f = 0; f < 4; f++) {
        uint32_t cd[32];
#pragma unroll
        for (int k = 0; k < 32; k++) {   // element 32 f + k = 32 f + 4 j4 + i
            const int j4 = k >> 2, i = k & 3;
            cd[k] = ((fw[j4] >> (8 * i + 2 * f)) & 3u) | (((hw[f] >> (4 * j4 + i)) & 1u) << 2);
        }
        uint32_t A = 0, B = 0, C = 0;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            A |= (cd[j] | ((cd[24 + j] & 1u) << 3)) << (4 * j);
            B |= (cd[8 + j] | (((cd[24 + j] >> 1) & 1u) << 3)) << (4 * j);
            C |= (cd[16 + j] | (((cd[24 + j] >> 2) & 1u) << 3)) << (4 * j);
        }
        o[3 * f] = A, o[3 * f + 1] = B, o[3 * f + 2] = C;
    }
    const size_t row = gi / G, g = gi % G;
    uint4* q = (uint4*)(dst + row * G * 48);
    q[g] = make_uint4(o[0], o[1], o[2], o[3]);
    q[G + g] = make_uint4(o[4], o[5], o[6], o[7]);
    q[2 * G + g] = make_uint4(o[8], o[9], o[10], o[11]);
}
// host: repack the codes of n experts at base (stride xbytes), in place through a scratch of one matrix (tmp >= 
// INTER * D / 128 * 48 bytes, device)
inline void cb3_repack(uint8_t* base, int n, size_t xbytes, uint8_t* tmp, cudaStream_t st) {
    for (int e = 0; e < n; e++) {
        MatView mv[3];
        expert_views(XF_CB3, base + (size_t)e * xbytes, mv);
        const int N[3] = {INTER, INTER, D}, K[3] = {D, D, INTER};
        for (int i = 0; i < 3; i++) {
            const size_t ng = (size_t)N[i] * (K[i] / 128);
            k_cb3_rp<<<(unsigned)((ng + 255) / 256), 256, 0, st>>>(mv[i].q, tmp, ng, K[i] / 128);
            cudaMemcpyAsync((uint8_t*)mv[i].q, tmp, ng * 48, cudaMemcpyDeviceToDevice, st);
        }
    }
}
// the int8 codebook: words lo (entries 0-3), hi (4-7), scale
struct Cb8 { uint32_t lo, hi; float s; };
__device__ __forceinline__ Cb8 cb8_make(const float* cb) {
    float m = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) m = fmaxf(m, fabsf(cb[i]));
    const float id = m > 0.f ? 127.f / m : 0.f;
    uint32_t w[2] = {0, 0};
#pragma unroll
    for (int i = 0; i < 8; i++) w[i >> 2] |= (uint32_t)(uint8_t)(int8_t)__float2int_rn(cb[i] * id) << (8 * (i & 3));
    return {w[0], w[1], m / 127.f};
}
// 32 weights (chunk words A, B, C) . 32 int8 x (8 words)
__device__ __forceinline__ int cb8_dot32(uint32_t A, uint32_t B, uint32_t C, const int* xw, uint32_t lo, uint32_t hi) {
    const uint32_t Dw = ((A >> 3) & 0x11111111u) | ((B >> 2) & 0x22222222u) | ((C >> 1) & 0x44444444u);
    int s = 0;
    s = __dp4a((int)__byte_perm(lo, hi, A & 0x7777u), xw[0], s);
    s = __dp4a((int)__byte_perm(lo, hi, (A >> 16) & 0x7777u), xw[1], s);
    s = __dp4a((int)__byte_perm(lo, hi, B & 0x7777u), xw[2], s);
    s = __dp4a((int)__byte_perm(lo, hi, (B >> 16) & 0x7777u), xw[3], s);
    s = __dp4a((int)__byte_perm(lo, hi, C & 0x7777u), xw[4], s);
    s = __dp4a((int)__byte_perm(lo, hi, (C >> 16) & 0x7777u), xw[5], s);
    s = __dp4a((int)__byte_perm(lo, hi, Dw & 0xFFFFu), xw[6], s);
    s = __dp4a((int)__byte_perm(lo, hi, Dw >> 16), xw[7], s);
    return s;
}
// drafter expert GEMV, int8: which = 0: gate / up rows of [w1; w3] (K = 5120, x row r), 1: down (K = 2304, x row
// r * TOPK + k). x8 / xs8: k_q8x output (int8 per value, float scale per 32) of the fp32 input rows. LPR lanes per
// output row, a lane takes whole 128-groups g = sub + LPR u (u < NG), all its loads issued first.
// PX: grid (pair, row band) instead of (row band, pair): the choices of one row band run in the same wave, so rows that
// share an expert read it from L2 (the drafter's 5 rows pick ~7.6 distinct experts of 15 choices per stage)
template <int LPR, int NG, bool XS = true, bool SKEL = false, bool PX = false>
__global__ void __launch_bounds__(256) k_dmoe8(const int8_t* __restrict__ x8, const float* __restrict__ xs8, int which,
                                               const int* ids, const int* dslot, const uint8_t* pool, size_t xbytes,
                                               const float* cb, float* out) {
    const int bx = PX ? blockIdx.y : blockIdx.x, by = PX ? blockIdx.x : blockIdx.y;
    __shared__ __align__(16) int8_t xsm[XS ? D : 16];
    __shared__ float ssm[XS ? D / 32 : 1];
    const int k = by % TOPK, r = by / TOPK;
    const int sl = dslot[ids[r * TOPK + k]];
    if (sl < 0) return;
    const int K = which ? INTER : D;
    const size_t xr = which ? (size_t)r * TOPK + k : (size_t)r;
    constexpr int RPB = 8 * (32 / LPR);
    const int row = bx * RPB + threadIdx.x / LPR, sub = threadIdx.x % LPR;
    const int NR = which ? D : 2 * INTER;
    const bool live = row < NR;
    const int G = K / 128;
    // the weight loads first (they do not depend on x): in flight while x is staged
    uint4 w[NG][3];
    float sw[NG];
    {
        // cb3 expert = w1 | w3 | w2, each [N][K / 128] x 48 B codes then [N][K / 128] fp16 scales (expert_views without
        // its local-memory arrays)
        constexpr size_t MB = (size_t)INTER * (D / 128) * 50;   // one matrix (all three have N K / 128 groups)
        constexpr size_t QB = (size_t)INTER * (D / 128) * 48;
        const int m = which ? 2 : (row < INTER ? 0 : 1), rr = live ? (which ? row : row % INTER) : 0;
        const uint8_t* mb = pool + (size_t)sl * xbytes + m * MB;
        const uint4* q = (const uint4*)(mb + (size_t)rr * G * 48);
        const __half* sc = (const __half*)(mb + QB) + (size_t)rr * G;
#pragma unroll
        for (int u = 0; u < NG; u++) {
            const int g = sub + LPR * u;
            if (live && g < G) {
                w[u][0] = q[g], w[u][1] = q[G + g], w[u][2] = q[2 * G + g];
                sw[u] = __half2float(sc[g]);
            }
        }
    }
    const int8_t* xb = x8 + xr * K;
    const float* sb = xs8 + xr * (K / 32);
    if (XS) {
        const uint4* xg = (const uint4*)xb;
        for (int t = threadIdx.x; t < K / 16; t += blockDim.x) ((uint4*)xsm)[t] = xg[t];
        for (int t = threadIdx.x; t < K / 32; t += blockDim.x) ssm[t] = sb[t];
        __syncthreads();
        xb = xsm;
        sb = ssm;
    }
    const Cb8 c8 = cb8_make(cb);
    if (!live) return;
    float acc = 0.f;
#pragma unroll
    for (int u = 0; u < NG; u++) {
        const int g = sub + LPR * u;
        if (g >= G) break;
        const uint32_t* wd = (const uint32_t*)w[u];
        if (SKEL) {   // loads-only skeleton (microbench diagnostic)
            acc += __int_as_float((wd[0] ^ wd[5] ^ wd[11]) & 0x3fffffff) * sw[u];
            continue;
        }
        const int4* xv = (const int4*)(xb + g * 128);
        float ag = 0.f;
#pragma unroll
        for (int f = 0; f < 4; f++) {
            const int4 a = xv[2 * f], b = xv[2 * f + 1];
            const int xw[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
            ag += (float)cb8_dot32(wd[3 * f], wd[3 * f + 1], wd[3 * f + 2], xw, c8.lo, c8.hi) * sb[g * 4 + f];
        }
        acc += ag * sw[u];
    }
    for (int off = LPR / 2; off; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (sub == 0) {
        float* o = which ? out + ((size_t)r * TOPK + k) * D + row : out + ((size_t)r * TOPK + k) * 2 * INTER + row;
        *o = acc * c8.s;
    }
}


// R22 [dsv4-r22-di8]: the drafter's FP8 linear weights (its own mtp.* tensors: attention, shared expert; main_proj stays
// FP8) as int8 with a float scale per row and 128 columns, converted once at load in place over the e4m3 bytes (the
// e8m0 scales are freed after it); x int8 per 32
// (k_q8r); dp4a. Drafter only (proposals): the target's FP8 GEMVs are untouched.
// one warp per (row, 128 columns): e4m3 x e8m0 (32 x 32 blocks) -> int8 (round(v / s), s = max |v| / 127)
__global__ void k_f8_to_i8(uint8_t* w, const uint8_t* s8, float* si, int N, int K) {
    const size_t wg = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wg >= (size_t)N * (K / 128)) return;
    const int row = (int)(wg / (K / 128)), g = (int)(wg % (K / 128));
    uint8_t* p = w + (size_t)row * K + g * 128 + lane * 4;
    float v[4];
    for (int j = 0; j < 4; j++) {
        const int col = g * 128 + lane * 4 + j;
        const float sc = exp2f((float)s8[(size_t)(row / 32) * (K / 32) + col / 32] - 127.f);
        v[j] = float(*(const __nv_fp8_e4m3*)&p[j]) * sc;
    }
    float m = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const float id = m > 0.f ? 127.f / m : 0.f;
    for (int j = 0; j < 4; j++) p[j] = (uint8_t)(int8_t)__float2int_rn(v[j] * id);
    if (lane == 0) si[wg] = m / 127.f;
}
// x [n] rows (stride ldx) of K columns -> int8 [n][K] + float scale per 32; one warp per 32-group
__global__ void k_q8r(const float* __restrict__ x, int ldx, int n, int K, int8_t* __restrict__ q, float* __restrict__ s) {
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t gi = t >> 5;
    const int lane = t & 31;
    if (gi >= (size_t)n * (K / 32)) return;
    const int r = (int)(gi / (K / 32)), c = (int)(gi % (K / 32)) * 32 + lane;
    const float v = x[(size_t)r * ldx + c];
    float m = fabsf(v);
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    // R23 (review r22 L7): fmaxf drops NaN; a non-finite value makes the group's scale NaN (as k_q8x), not a silent 0
    const bool bad = __any_sync(0xffffffffu, !isfinite(v));
    const float id = !bad && m > 1e-30f ? 127.f / m : 0.f;
    q[(size_t)r * K + c] = (int8_t)__float2int_rn(v * id);
    if (lane == 0) s[gi] = bad ? __int_as_float(0x7fc00000) : m / 127.f;
}
// [dsv4-dec2-x2q] k_q8r with output strides: q row r at q + r ldq, scales at s + r lds (the same per-32 values as k_q8r:
// a card quantizes its o2 columns in place in the wo_b input layout and sends the int8 + scales instead of fp32)
__global__ void k_q8rs(const float* __restrict__ x, int ldx, int n, int K, int8_t* __restrict__ q, int ldq,
                       float* __restrict__ s, int lds) {
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t gi = t >> 5;
    const int lane = t & 31;
    if (gi >= (size_t)n * (K / 32)) return;
    const int r = (int)(gi / (K / 32)), g = (int)(gi % (K / 32)), c = g * 32 + lane;
    const float v = x[(size_t)r * ldx + c];
    float m = fabsf(v);
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const bool bad = __any_sync(0xffffffffu, !isfinite(v));
    const float id = !bad && m > 1e-30f ? 127.f / m : 0.f;
    q[(size_t)r * ldq + c] = (int8_t)__float2int_rn(v * id);
    if (lane == 0) s[(size_t)r * lds + g] = bad ? __int_as_float(0x7fc00000) : m / 127.f;
}
// y[r][row] = sum over K of W_i8[row] . x_i8[r] (x columns per row group: grp rows share the x segment
// (row / grp) x K of a row of Kx = K x N / grp columns; grp 0 = all rows one segment). LPR lanes per row, a lane takes
// 32-weight chunks c = sub + LPR u (two 16 B loads), x rows staged in shared memory (int8 + scales)
// release prep [dsv4-rel-q8]: H32 = Q8_0 weights (GGUF): an fp16 scale per row and 32 columns (S is __half [N][K / 32])
// instead of a float per row and 128; ACC: y += (a K slice of a wider row, the caller splits K for the shared memory)
template <int R, int LPR, int RW = 4, bool H32 = false, bool ACC = false>
__global__ void __launch_bounds__(256) k_gemv_i8o(const int8_t* __restrict__ W, const void* __restrict__ Sv, int N, int K,
                                                 const int8_t* __restrict__ xq, const float* __restrict__ xs, int Kx,
                                                 float* y, int ldy, int grp, int ldw = 0) {
    // ldw: W / S row stride in columns (0 = K; a K slice of a wider matrix)
    const float* __restrict__ S = (const float*)Sv;
    const __half* __restrict__ S16 = (const __half*)Sv;
    const int LW = ldw ? ldw : K;
    // RW row groups per block (x staged once for RPB x RW rows; grp must be a multiple of RPB x RW)
    extern __shared__ __align__(16) int8_t xsm[];   // [R][K] then float [R][K / 32]
    float* ssm = (float*)(xsm + R * K);
    constexpr int RPB = 8 * (32 / LPR), PF = 4;
    const int blk0 = blockIdx.x * RPB * RW, sub = threadIdx.x % LPR;
    const int NC = K / 32;
    const int xo = grp ? (blk0 / grp) * K : 0;
    uint4 w0[PF], w1[PF];
    {   // the first row group's first chunks in flight before the x staging
        const int row = blk0 + threadIdx.x / LPR;
        const uint4* wr = (const uint4*)(W + (size_t)(row < N ? row : 0) * LW);
#pragma unroll
        for (int u = 0; u < PF; u++) {
            const int c = sub + LPR * u;
            if (row < N && c < NC) w0[u] = wr[2 * c], w1[u] = wr[2 * c + 1];
        }
    }
    for (int t = threadIdx.x; t < R * K / 16; t += blockDim.x) {
        const int r = t / (K / 16), k = t % (K / 16);
        ((uint4*)xsm)[t] = ((const uint4*)(xq + (size_t)r * Kx + xo))[k];
    }
    for (int t = threadIdx.x; t < R * NC; t += blockDim.x) {
        const int r = t / NC, k = t % NC;
        ssm[t] = xs[(size_t)r * (Kx / 32) + xo / 32 + k];
    }
    __syncthreads();
    for (int rw = 0; rw < RW; rw++) {
        const int row = blk0 + rw * RPB + threadIdx.x / LPR;
        if (row >= N) break;
        const uint4* wr = (const uint4*)(W + (size_t)row * LW);
        const float* sr = S + (size_t)row * (LW / 128);
        const __half* sr16 = S16 + (size_t)row * (LW / 32);
        if (rw) {
#pragma unroll
            for (int u = 0; u < PF; u++) {
                const int c = sub + LPR * u;
                if (c < NC) w0[u] = wr[2 * c], w1[u] = wr[2 * c + 1];
            }
        }
        float acc[R];
#pragma unroll
        for (int r = 0; r < R; r++) acc[r] = 0.f;
        for (int u0 = 0; sub + LPR * u0 < NC; u0 += PF) {
#pragma unroll
            for (int u = 0; u < PF; u++) {
                const int c = sub + LPR * (u0 + u);
                if (c >= NC) break;
                const uint4 a = w0[u], b = w1[u];
                const int cn = c + LPR * PF;   // the same slot's next chunk
                if (cn < NC) w0[u] = wr[2 * cn], w1[u] = wr[2 * cn + 1];
                const float sw = H32 ? __half2float(sr16[c]) : sr[c >> 2];
#pragma unroll
                for (int r = 0; r < R; r++) {
                    const int4* xp = (const int4*)(xsm + r * K + c * 32);
                    const int4 xa = xp[0], xb = xp[1];
                    int d = __dp4a((int)a.x, xa.x, 0);
                    d = __dp4a((int)a.y, xa.y, d); d = __dp4a((int)a.z, xa.z, d); d = __dp4a((int)a.w, xa.w, d);
                    d = __dp4a((int)b.x, xb.x, d); d = __dp4a((int)b.y, xb.y, d); d = __dp4a((int)b.z, xb.z, d);
                    d = __dp4a((int)b.w, xb.w, d);
                    acc[r] += (float)d * (ssm[r * NC + c] * sw);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
            float v = acc[r];
            for (int off = LPR / 2; off; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
            if (sub == 0) {
                if (ACC) y[(size_t)r * ldy + row] += v;
                else y[(size_t)r * ldy + row] = v;
            }
        }
    }
}
// [dsv4-dec2-i8p] k_gemv_i8 v2 (default; DSV4_NOI8P=1: k_gemv_i8o, the R22 kernel): the weight scale of a chunk is
// loaded with the chunk (it was loaded at its use: one exposed DRAM latency per chunk), and a slot whose row has no
// more chunks loads its first chunk of the block's next row group (the old kernel restarted the loads per row group:
// RW latency rounds per warp). PF 2 chunks per slot (fewer registers). Same lane / chunk assignment and the same
// accumulation order as k_gemv_i8o: bit-identical (src/dsv4/tests/i8_rows_bench.cu, var 11).
template <int R, int LPR, int RW = 4, bool H32 = false, bool ACC = false>
__global__ void __launch_bounds__(256) k_gemv_i8(const int8_t* __restrict__ W, const void* __restrict__ Sv, int N, int K,
                                                 const int8_t* __restrict__ xq, const float* __restrict__ xs, int Kx,
                                                 float* y, int ldy, int grp, int ldw = 0) {
    using ST = typename std::conditional<H32, __half, float>::type;   // H32: fp16 per 32 columns, else float per 128
    const ST* __restrict__ S = (const ST*)Sv;
    const int LW = ldw ? ldw : K;
    constexpr int SD = H32 ? 32 : 128;   // columns per scale
    extern __shared__ __align__(16) int8_t xsm[];   // [R][K] then float [R][K / 32]
    float* ssm = (float*)(xsm + R * K);
    constexpr int RPB = 8 * (32 / LPR), PF = 2;
    const int blk0 = blockIdx.x * RPB * RW, sub = threadIdx.x % LPR;
    const int NC = K / 32;
    const int xo = grp ? (blk0 / grp) * K : 0;
    uint4 w0[PF], w1[PF];
    ST sc[PF];
    {   // the first row group's first chunks (+ scales) in flight before the x staging
        const int row = blk0 + threadIdx.x / LPR;
        const uint4* wr = (const uint4*)(W + (size_t)(row < N ? row : 0) * LW);
        const ST* sr = S + (size_t)(row < N ? row : 0) * (LW / SD);
#pragma unroll
        for (int u = 0; u < PF; u++) {
            const int c = sub + LPR * u;
            if (row < N && c < NC) w0[u] = wr[2 * c], w1[u] = wr[2 * c + 1], sc[u] = sr[c * 32 / SD];
        }
    }
    for (int t = threadIdx.x; t < R * K / 16; t += 256) {
        const int r = t / (K / 16), k = t % (K / 16);
        ((uint4*)xsm)[t] = ((const uint4*)(xq + (size_t)r * Kx + xo))[k];
    }
    for (int t = threadIdx.x; t < R * NC; t += 256) {
        const int r = t / NC, k = t % NC;
        ssm[t] = xs[(size_t)r * (Kx / 32) + xo / 32 + k];
    }
    __syncthreads();
    for (int rw = 0; rw < RW; rw++) {
        const int row = blk0 + rw * RPB + threadIdx.x / LPR;
        if (row >= N) break;
        const int rown = row + RPB;
        const bool hasn = rw + 1 < RW && rown < N;
        const uint4* wr = (const uint4*)(W + (size_t)row * LW);
        const ST* sr = S + (size_t)row * (LW / SD);
        const uint4* wrn = (const uint4*)(W + (size_t)(hasn ? rown : row) * LW);
        const ST* srn = S + (size_t)(hasn ? rown : row) * (LW / SD);
        float acc[R];
#pragma unroll
        for (int r = 0; r < R; r++) acc[r] = 0.f;
        for (int u0 = 0; sub + LPR * u0 < NC; u0 += PF) {
#pragma unroll
            for (int u = 0; u < PF; u++) {
                const int c = sub + LPR * (u0 + u);
                if (c >= NC) break;
                const uint4 a = w0[u], b = w1[u];
                const float sw = (float)sc[u];
                const int cn = c + LPR * PF;   // the same slot's next chunk: this row, else the next row group's first
                if (cn < NC) w0[u] = wr[2 * cn], w1[u] = wr[2 * cn + 1], sc[u] = sr[cn * 32 / SD];
                else if (hasn) {
                    const int c0 = sub + LPR * u;
                    w0[u] = wrn[2 * c0], w1[u] = wrn[2 * c0 + 1], sc[u] = srn[c0 * 32 / SD];
                }
#pragma unroll
                for (int r = 0; r < R; r++) {
                    const int4* xp = (const int4*)(xsm + r * K + c * 32);
                    const int4 xa = xp[0], xb = xp[1];
                    int d = __dp4a((int)a.x, xa.x, 0);
                    d = __dp4a((int)a.y, xa.y, d); d = __dp4a((int)a.z, xa.z, d); d = __dp4a((int)a.w, xa.w, d);
                    d = __dp4a((int)b.x, xb.x, d); d = __dp4a((int)b.y, xb.y, d); d = __dp4a((int)b.z, xb.z, d);
                    d = __dp4a((int)b.w, xb.w, d);
                    acc[r] += (float)d * (ssm[r * NC + c] * sw);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
            float v = acc[r];
            for (int off = LPR / 2; off; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
            if (sub == 0) {
                if (ACC) y[(size_t)r * ldy + row] += v;
                else y[(size_t)r * ldy + row] = v;
            }
        }
    }
}
template <int R, int RW, bool H, bool A>
inline void i8_smem_set3() {
    DCK(cudaFuncSetAttribute(k_gemv_i8<R, 8, RW, H, A>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
    DCK(cudaFuncSetAttribute(k_gemv_i8<R, 16, RW, H, A>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
    DCK(cudaFuncSetAttribute(k_gemv_i8<R, 32, RW, H, A>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
    DCK(cudaFuncSetAttribute(k_gemv_i8o<R, 8, RW, H, A>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
    DCK(cudaFuncSetAttribute(k_gemv_i8o<R, 16, RW, H, A>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
    DCK(cudaFuncSetAttribute(k_gemv_i8o<R, 32, RW, H, A>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
}
template <int R, int RW>
inline void i8_smem_set2() {   // (R23, review r22 L5: checked)
    i8_smem_set3<R, RW, false, false>();
    i8_smem_set3<R, RW, true, false>();
    i8_smem_set3<R, RW, true, true>();
}
static const bool g_i8p_off = getenv("DSV4_NOI8P") != nullptr;   // [dsv4-dec2-i8p] off: k_gemv_i8o (R22 kernel)
// R23 (review r22 L5): DSV4_LCHECK=1 checks the drafter's own launches (debug; any earlier error is cleared first)
static const bool g_lchk = getenv("DSV4_LCHECK") != nullptr;
inline void lchk_pre() {   // (R23, review r23 L6: a pending earlier error is reported, not silently dropped)
    if (!g_lchk) return;
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) fprintf(stderr, "DSV4_LCHECK: error pending before a drafter launch: %s\n", cudaGetErrorString(e));
}
inline void lchk(const char* what) {
    if (!g_lchk) return;
    const cudaError_t e = cudaGetLastError();
    DCHECK(e == cudaSuccess, "launch %s: %s", what, cudaGetErrorString(e));
}
template <int R>
inline void i8_smem_set() { i8_smem_set2<R, 1>(); i8_smem_set2<R, 4>(); }
inline void i8_smem_init() {   // per device
    i8_smem_set<1>(); i8_smem_set<2>(); i8_smem_set<3>(); i8_smem_set<4>();
    i8_smem_set<5>(); i8_smem_set<6>(); i8_smem_set<7>(); i8_smem_set<8>();
}
template <int R, int RW, bool H, bool A>
inline void i8_gemv_rwh(cudaStream_t st, const int8_t* W, const void* S, int N, int K, const int8_t* xq, const float* xs,
                        int Kx, float* y, int ldy, int grp, int ldw) {
    const int lpr = K >= 4096 ? 32 : K >= 2048 ? 16 : 8;
    const size_t sm = (size_t)R * K + (size_t)R * (K / 32) * 4;
    const int rb = RW * 8 * (32 / lpr);
    // R23 (review r22 L4): whole warps per row range (full-mask shuffles), a block's rows inside one x group
    DCHECK(N % (32 / lpr) == 0 && (!grp || grp % rb == 0) && K % 128 == 0 && sm <= 96 * 1024,
           "int8 GEMV shape: N %d K %d grp %d (rows per block %d)", N, K, grp, rb);
    const unsigned g = (unsigned)((N + rb - 1) / rb);
    lchk_pre();
    if (g_i8p_off) {
    if (lpr == 32) k_gemv_i8o<R, 32, RW, H, A><<<g, 256, sm, st>>>(W, S, N, K, xq, xs, Kx, y, ldy, grp, ldw);
    else if (lpr == 16) k_gemv_i8o<R, 16, RW, H, A><<<g, 256, sm, st>>>(W, S, N, K, xq, xs, Kx, y, ldy, grp, ldw);
    else k_gemv_i8o<R, 8, RW, H, A><<<g, 256, sm, st>>>(W, S, N, K, xq, xs, Kx, y, ldy, grp, ldw);
    } else {
    if (lpr == 32) k_gemv_i8<R, 32, RW, H, A><<<g, 256, sm, st>>>(W, S, N, K, xq, xs, Kx, y, ldy, grp, ldw);
    else if (lpr == 16) k_gemv_i8<R, 16, RW, H, A><<<g, 256, sm, st>>>(W, S, N, K, xq, xs, Kx, y, ldy, grp, ldw);
    else k_gemv_i8<R, 8, RW, H, A><<<g, 256, sm, st>>>(W, S, N, K, xq, xs, Kx, y, ldy, grp, ldw);
    }
    lchk("k_gemv_i8");
}
template <int R, int RW>
inline void i8_gemv_rw(cudaStream_t st, const int8_t* W, const void* S, int N, int K, const int8_t* xq, const float* xs,
                       int Kx, float* y, int ldy, int grp, bool h32 = false, bool acc = false, int ldw = 0) {
    if (!h32) i8_gemv_rwh<R, RW, false, false>(st, W, S, N, K, xq, xs, Kx, y, ldy, grp, ldw);
    else if (!acc) i8_gemv_rwh<R, RW, true, false>(st, W, S, N, K, xq, xs, Kx, y, ldy, grp, ldw);
    else i8_gemv_rwh<R, RW, true, true>(st, W, S, N, K, xq, xs, Kx, y, ldy, grp, ldw);
}
// RW 4 (x staged once per 4 row groups: 1.1-1.25x faster) except for small N (wq_a, wkv: too few blocks);
// [dsv4-dec2-i8p] the v2 kernel: RW 4 for every N (its loads run across the row groups: faster also at N <= 2048)
template <int R>
inline void i8_gemv_r(cudaStream_t st, const int8_t* W, const void* S, int N, int K, const int8_t* xq, const float* xs,
                      int Kx, float* y, int ldy, int grp, bool h32 = false, bool acc = false, int ldw = 0) {
    const int rb4 = 4 * 8 * (32 / (K >= 4096 ? 32 : K >= 2048 ? 16 : 8));   // (review dec2 L5: RW 1 when grp is no multiple)
    if ((N <= 2048 && g_i8p_off) || (grp && grp % rb4)) i8_gemv_rw<R, 1>(st, W, S, N, K, xq, xs, Kx, y, ldy, grp, h32, acc, ldw);
    else i8_gemv_rw<R, 4>(st, W, S, N, K, xq, xs, Kx, y, ldy, grp, h32, acc, ldw);
}
template <int R>
inline void i8_gemv_k(cudaStream_t st, const int8_t* W, const void* S, int N, int K, const int8_t* xq, const float* xs,
                      int Kx, float* y, int ldy, int grp, bool h32) {
    // [dsv4-rel-q8]: a K too wide for the x staging (main_proj 15360 at R >= 6) in K slices (H32 only: the first
    // slice stores, the rest add; x / scales offset per slice)
    if ((size_t)R * K * 9 / 8 <= 96 * 1024 || !h32) { i8_gemv_r<R>(st, W, S, N, K, xq, xs, Kx, y, ldy, grp, h32); return; }
    DCHECK(!grp && Kx == K, "int8 GEMV K split: grouped rows");
    int ks = K;
    while ((size_t)R * ks * 9 / 8 > 96 * 1024 || ks % 128) ks = ks % 2 == 0 && (ks / 2) % 128 == 0 ? ks / 2 : ks - 128;
    for (int k0 = 0; k0 < K; k0 += ks) {
        const int kn = std::min(ks, K - k0);
        i8_gemv_r<R>(st, W + k0, (const __half*)S + k0 / 32, N, kn, xq + k0, xs + k0 / 32, Kx, y, ldy, 0, true, k0 > 0, K);
    }
}
// y[n][N] = W_i8 . q8(x)^T for n <= 8 rows (x [n] rows of Kx fp32 at stride ldx; q / qs scratch >= n Kx bytes / floats / 32)
// h32: S = Q8_0 scales (__half per 32), else float per 128
inline void i8_linear(cudaStream_t st, const int8_t* W, const void* S, int N, int K, int grp, const float* x, int ldx,
                      int n, float* y, int ldy, int8_t* q, float* qs, size_t qcap, bool h32 = false) {
    const int Kx = grp ? K * (N / grp) : K;
    DCHECK((size_t)Kx * n <= qcap, "int8 linear: Kx %d x n %d past the scratch (%zu)", Kx, n, qcap);   // (R23 review L3 / L7)
    if (x) k_q8r<<<(unsigned)(((size_t)n * Kx + 255) / 256), 256, 0, st>>>(x, ldx, n, Kx, q, qs);
    // ([dsv4-dec2-x2q] x == nullptr: q / qs already hold the int8 input)
    switch (n) {
        case 1: i8_gemv_k<1>(st, W, S, N, K, q, qs, Kx, y, ldy, grp, h32); break;
        case 2: i8_gemv_k<2>(st, W, S, N, K, q, qs, Kx, y, ldy, grp, h32); break;
        case 3: i8_gemv_k<3>(st, W, S, N, K, q, qs, Kx, y, ldy, grp, h32); break;
        case 4: i8_gemv_k<4>(st, W, S, N, K, q, qs, Kx, y, ldy, grp, h32); break;
        case 5: i8_gemv_k<5>(st, W, S, N, K, q, qs, Kx, y, ldy, grp, h32); break;
        case 6: i8_gemv_k<6>(st, W, S, N, K, q, qs, Kx, y, ldy, grp, h32); break;
        case 7: i8_gemv_k<7>(st, W, S, N, K, q, qs, Kx, y, ldy, grp, h32); break;
        default: i8_gemv_k<8>(st, W, S, N, K, q, qs, Kx, y, ldy, grp, h32); break;
    }
}

// R22 [dsv4-r22-mk8]: the Markov bias + argmax of a draft row in one launch, the Markov head as int8 (a float scale per
// vocabulary row) and emb[tok] as int8 (one scale), dp4a. A warp per vocabulary row (grid-stride), the row's best
// (logit + bias, smallest id on ties) reduced per block, one 64-bit atomicMax per block into key[i]; the next row reads
// its token from key[i - 1]. Logits updated in place (the R20 selector reads them).
// bf16 [V][256] -> int8 [V][256] + scale [V]; one warp per row
__global__ void k_mk_to_i8(const bf16* __restrict__ w, int8_t* q, float* s, int nrow) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= nrow) return;
    float v[8], m = 0.f;
    for (int j = 0; j < 8; j++) { v[j] = __bfloat162float(w[(size_t)row * 256 + lane * 8 + j]); m = fmaxf(m, fabsf(v[j])); }
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const float id = m > 0.f ? 127.f / m : 0.f;
    for (int j = 0; j < 8; j++) q[(size_t)row * 256 + lane * 8 + j] = (int8_t)__float2int_rn(v[j] * id);
    if (lane == 0) s[row] = m / 127.f;
}
__device__ __forceinline__ unsigned long long mk_key(float v, int id) {
    uint32_t u = __float_as_uint(v);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    if (v != v) u = 0u;   // R23 (review r22 L7): NaN below every number (-inf included), still a valid id (key > 0)
    return ((unsigned long long)u << 32) | (0xFFFFFFFFu - (uint32_t)id);
}
__device__ __forceinline__ int mk_tok(unsigned long long k) { return (int)(0xFFFFFFFFu - (uint32_t)(k & 0xFFFFFFFFu)); }
// tok = i == 0 ? dtok[0] : the argmax of row i - 1 (key[i - 1]); dtok[i] = tok; embf[i] = emb[tok] (fp32)
__global__ void __launch_bounds__(256) k_markov8(const int8_t* __restrict__ Wq, const float* __restrict__ Ws,
                                                 const bf16* __restrict__ emb, int i, int* dtok,
                                                 unsigned long long* key, float* logits, float* embf, int V_) {
    __shared__ __align__(16) int8_t eq[256];   // (R23, review r22 L6: read as int2)
    __shared__ float red[8];
    __shared__ unsigned long long bk[8];
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int tk = i == 0 ? dtok[0] : mk_tok(key[i - 1]);
    const float ev = __bfloat162float(emb[(size_t)tk * 256 + tid]);
    if (blockIdx.x == 0) {
        embf[i * 256 + tid] = ev;
        if (tid == 0 && i > 0) dtok[i] = tk;
    }
    float m = fabsf(ev);
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) red[w] = m;
    __syncthreads();
    m = fmaxf(fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3])), fmaxf(fmaxf(red[4], red[5]), fmaxf(red[6], red[7])));
    const float es = m / 127.f, id = m > 0.f ? 127.f / m : 0.f;
    eq[tid] = (int8_t)__float2int_rn(ev * id);
    __syncthreads();
    const int2 x = ((const int2*)eq)[lane];
    unsigned long long best = 0;
    for (int row = blockIdx.x * 8 + w; row < V_; row += gridDim.x * 8) {
        const int2 q = ((const int2*)(Wq + (size_t)row * 256))[lane];
        int d = __dp4a(q.x, x.x, __dp4a(q.y, x.y, 0));
        for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        if (lane == 0) {
            const float v = logits[row] + (float)d * Ws[row] * es;
            logits[row] = v;
            const unsigned long long k = mk_key(v, row);
            best = k > best ? k : best;
        }
    }
    if (lane == 0) bk[w] = best;
    __syncthreads();
    if (tid == 0) {
        unsigned long long b = bk[0];
        for (int j = 1; j < 8; j++) b = bk[j] > b ? bk[j] : b;
        atomicMax(key + i, b);
    }
}
// dtok[0] = tok, key[0 .. 7] = 0 (before the rows); after them: dtok[5] = argmax of the last row
__global__ void k_mk_init(int* dtok, int tok, unsigned long long* key) {
    if (threadIdx.x == 0) dtok[0] = tok;
    if (threadIdx.x < 8) key[threadIdx.x] = 0ull;
}
__global__ void k_mk_last(int* dtok, const unsigned long long* key, int B) {
    if (threadIdx.x == 0) dtok[B] = mk_tok(key[B - 1]);
}

// R22 [dsv4-r22-hmma]: the drafter's output head (bf16 [N][5120] per card, R <= 8 rows) on the tensor cores:
// mma.m16n8k16 bf16 -> fp32, a warp per 16 vocabulary rows over the whole K, A fragments loaded from global in
// fragment order (per k-step a warp reads 16 rows x 32 B = full sectors), x rows as bf16 in shared memory (rows >= R
// zero). Drafter only (its logits are proposals; the verify head keeps k_gemv_bf16).
template <int R>
__global__ void __launch_bounds__(256) k_head_mma(const bf16* __restrict__ W, int N, const float* __restrict__ x,
                                                  float* y, int ldy) {
    // k is summed in a permuted order (the same for A and B): per 32 columns, lane tq of a row owns the physical columns
    // 8 tq .. 8 tq + 7 (one 16 B load per row), used as the logical k pairs of two consecutive k-steps
    constexpr int K = D, SS = K / 32, U = 4, LD = K + 8;   // +8: bank spread of the x loads
    extern __shared__ __align__(16) bf16 xs[];   // [R][LD] (dynamic: R x 10.3 KB)
    for (int t = threadIdx.x; t < R * K; t += blockDim.x) {
        const int r = t / K, k = t % K;
        xs[r * LD + k] = __float2bfloat16(x[(size_t)r * K + k]);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, tq = lane & 3;
    const uint4* xg = (const uint4*)(xs + g * LD) + tq;
    // persistent: the block's warps stride over 16-row tiles (x staged once per block)
    for (int row0 = (blockIdx.x * 8 + warp) * 16; row0 < N; row0 += gridDim.x * 8 * 16) {
        const int ra = min(row0 + g, N - 1), rb = min(row0 + g + 8, N - 1);
        const uint4* wa = (const uint4*)(W + (size_t)ra * K) + tq;
        const uint4* wb = (const uint4*)(W + (size_t)rb * K) + tq;
        float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
        for (int s0 = 0; s0 < SS; s0 += U) {
            uint4 A[U], B[U];
#pragma unroll
            for (int u = 0; u < U; u++) A[u] = wa[(s0 + u) * 4], B[u] = wb[(s0 + u) * 4];
#pragma unroll
            for (int u = 0; u < U; u++) {
                const uint4 xv = g < R ? xg[(s0 + u) * 4] : make_uint4(0, 0, 0, 0);
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                             "{%0,%1,%2,%3};\n"
                             : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                             : "r"(A[u].x), "r"(B[u].x), "r"(A[u].y), "r"(B[u].y), "r"(xv.x), "r"(xv.y));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                             "{%0,%1,%2,%3};\n"
                             : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                             : "r"(A[u].z), "r"(B[u].z), "r"(A[u].w), "r"(B[u].w), "r"(xv.z), "r"(xv.w));
            }
        }
        // C fragment: c0, c1 = row g, columns (x rows) 2 tq, 2 tq + 1; c2, c3 = row g + 8
        const int n0 = 2 * tq, n1 = 2 * tq + 1;
        if (row0 + g < N) {
            if (n0 < R) y[(size_t)n0 * ldy + row0 + g] = c0;
            if (n1 < R) y[(size_t)n1 * ldy + row0 + g] = c1;
        }
        if (row0 + g + 8 < N) {
            if (n0 < R) y[(size_t)n0 * ldy + row0 + g + 8] = c2;
            if (n1 < R) y[(size_t)n1 * ldy + row0 + g + 8] = c3;
        }
    }
}
#ifndef HMMA_GRID
#define HMMA_GRID 82
#endif
template <int R>
inline void head_mma_launch(const bf16* W, int N, const float* x, float* y, int ldy, cudaStream_t st) {
    const size_t sm = (size_t)R * (D + 8) * 2;
    static thread_local int setdev[8] = {0, 0, 0, 0, 0, 0, 0, 0};   // attribute per device (one host thread per engine)
    int dev = 0;
    cudaGetDevice(&dev);
    DCHECK(dev >= 0 && dev < 8, "head_mma: device %d", dev);   // (R23, review r22 L5)
    if (!setdev[dev]) {
        DCK(cudaFuncSetAttribute(k_head_mma<R>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
        setdev[dev] = 1;
    }
    lchk_pre();
    k_head_mma<R><<<HMMA_GRID, 256, sm, st>>>(W, N, x, y, ldy);
    lchk("k_head_mma");
}

}  // namespace dsv4
