#pragma once
// R16 [dsv4-gguf-iq]: GPU kernels for the GGUF i-quant matrix formats (gguf_iq.h engine layout):
//   k_moe_gg     decode GEMV on dynamic-pool slots (as k_moe_m23): rows of the routed choices, fp32 activations and
//                fp32 math (exact products, fp32 sums); a warp step = one 256-weight block of RPW rows, a lane = 8
//                consecutive weights; x read from global memory (coalesced, L1-resident), U blocks x RPW rows of
//                raw bytes loaded before any decode
//   k_deq_gg     one matrix -> bf16 [N][K] (prefill GEMM operand, as k_deq_cb8); a thread = one 32-weight sub-block
// The format's grid is copied into shared memory per block (1-8 KB: IQ2_S 1024 x 8 B is the largest), the lookups
// then hit LDS; signs = 7-bit fields with even parity (computed, no table).
// Measured variants before this one (R16): v1 (LPR lanes per row, x per lane from
// global), v2 (x staged in shared memory per 32 rows), v4 (fp16 HFMA2 dot with fp16 / sign-mask tables): all slower.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "dsv4.h"

namespace dsv4 {

// device copy of the four grids, back to back (gg_grids_upload): byte offsets
constexpr int GG_OFF_IQ2XXS = 0, GG_OFF_IQ3XXS = 2048, GG_OFF_IQ2S = 3072, GG_OFF_IQ3S = 11264, GG_GRID_BYTES = 13312;
__host__ __device__ constexpr int gg_grid_off(int mf) {
    return mf == XF_IQ2XXS ? GG_OFF_IQ2XXS : mf == XF_IQ3XXS ? GG_OFF_IQ3XXS : mf == XF_IQ2S ? GG_OFF_IQ2S : GG_OFF_IQ3S;
}
__host__ __device__ constexpr int gg_grid_bytes(int mf) {
    return mf == XF_IQ2XXS ? 2048 : mf == XF_IQ3XXS ? 1024 : mf == XF_IQ2S ? 8192 : 2048;
}

__device__ __forceinline__ uint32_t gg_sign8(uint32_t v7) { return v7 | ((__popc(v7) & 1) << 7); }
__device__ __forceinline__ float gg_sgn(float v, uint32_t s, int j) { return (s >> j) & 1 ? -v : v; }
// R20 [dsv4-r20-u8f]: byte q of w as an exact float without I2F (a quarter-rate pipe on sm_86): the byte placed in
// the mantissa of 2^23 (byte_perm with 0x4B000000), minus 2^23. Same value as (float)((w >> 8q) & 255)
__device__ __forceinline__ float gg_u8f(uint32_t w, int q) {
#ifndef GG_U8F   // R20 [dsv4-r20-u8f] ❌: the byte_perm form measured 0-9 % slower (I2F is not the limit): opt-in build flag
    return (float)((w >> (8 * q)) & 255);
#else
    return __int_as_float((int)__byte_perm(w, 0x4B000000u, 0x7540u | (unsigned)q)) - 8388608.f;
#endif
}

// sub-block k (0..7) of a super-block: 32 weights -> w[32] (unscaled by d), grid in shared memory
template <int MF>
__device__ __forceinline__ void gg_sub(const uint8_t* c, int k, const void* grid, float* w) {
    if constexpr (MF == XF_IQ2XXS) {
        const uint2 v = *(const uint2*)(c + 8 * k);
        const float db = (0.5f + (float)(v.y >> 28)) * 0.25f;
        const uint2* G = (const uint2*)grid;
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint2 g = G[(v.x >> (8 * l)) & 255];
            const uint32_t s = gg_sign8((v.y >> (7 * l)) & 127);
#pragma unroll
            for (int j = 0; j < 8; j++)
                w[8 * l + j] = gg_sgn(db * gg_u8f(j < 4 ? g.x : g.y, j & 3), s, j);
        }
    } else if constexpr (MF == XF_IQ3XXS) {
        const uint2 v = *(const uint2*)(c + 8 * k);
        const uint32_t aux = *(const uint32_t*)(c + 64 + 4 * k);
        const float db = (0.5f + (float)(aux >> 28)) * 0.5f;
        const uint32_t* G = (const uint32_t*)grid;
#pragma unroll
        for (int i = 0; i < 8; i++) {   // entry i: 4 weights
            const uint32_t g = G[((i < 4 ? v.x : v.y) >> (8 * (i & 3))) & 255];
            const uint32_t s = gg_sign8((aux >> (7 * (i / 2))) & 127) >> (4 * (i & 1));
#pragma unroll
            for (int j = 0; j < 4; j++) w[4 * i + j] = gg_sgn(db * gg_u8f(g, j), s, j);
        }
    } else if constexpr (MF == XF_IQ2S) {
        const uint32_t qs = *(const uint32_t*)(c + 4 * k), sg = *(const uint32_t*)(c + 32 + 4 * k);
        const uint32_t qh = c[64 + k], sc = c[72 + k];
        const uint2* G = (const uint2*)grid;
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint2 g = G[((qs >> (8 * l)) & 255) | (((qh >> (2 * l)) & 3) << 8)];
            const float db = (0.5f + (float)(l < 2 ? sc & 15 : sc >> 4)) * 0.25f;
            const uint32_t s = (sg >> (8 * l)) & 255;
#pragma unroll
            for (int j = 0; j < 8; j++)
                w[8 * l + j] = gg_sgn(db * gg_u8f(j < 4 ? g.x : g.y, j & 3), s, j);
        }
    } else {   // XF_IQ3S
        const uint32_t q0 = *(const uint32_t*)(c + 8 * k), q1 = *(const uint32_t*)(c + 8 * k + 4);
        const uint32_t qh = c[64 + k], sg = *(const uint32_t*)(c + 72 + 4 * k);
        const float db = (float)(1 + 2 * ((c[104 + k / 2] >> (4 * (k & 1))) & 15));
        const uint32_t* G = (const uint32_t*)grid;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const uint32_t g = G[(((i < 4 ? q0 : q1) >> (8 * (i & 3))) & 255) | (((qh >> i) & 1) << 8)];
            const uint32_t s = (sg >> (4 * i)) & 15;
#pragma unroll
            for (int j = 0; j < 4; j++) w[4 * i + j] = gg_sgn(db * gg_u8f(g, j), s, j);
        }
    }
}

template <int MF>
__device__ __forceinline__ void gg_load_grid(const uint8_t* grids, uint32_t* sg) {
    constexpr int NB = MF == XF_IQ2XXS ? 2048 : MF == XF_IQ3XXS ? 1024 : MF == XF_IQ2S ? 8192 : 2048;
    const uint32_t* g = (const uint32_t*)(grids + (MF == XF_IQ2XXS ? GG_OFF_IQ2XXS : MF == XF_IQ3XXS ? GG_OFF_IQ3XXS
                                                   : MF == XF_IQ2S ? GG_OFF_IQ2S : GG_OFF_IQ3S));
    for (int i = threadIdx.x; i < NB / 4; i += blockDim.x) sg[i] = g[i];
    __syncthreads();
}

// decode GEMV on the GGUF dynamic pool (format MF for the launched matrices; gate and up share it).
// which 0: rows 0 .. 2 INTER (gate then up) of x [R][D] -> out [R][TOPK][2 INTER]; which 1: down rows of
// x [R][TOPK][INTER] -> out [R][TOPK][D]. Slot of expert e = dslot[e] (-1: not here); its bytes at
// pool + (slot - s0) * xbytes (the layer's region of the card pool).
template <int MF>
struct GgRaw {   // the bytes lane i needs from one block (and the block's d)
    uint32_t a = 0, b = 0;
    float d = 0.f;
    __device__ __forceinline__ void load(const uint8_t* c, const __half* ds, int i) {
        d = __half2float(*ds);
        if constexpr (MF == XF_IQ2XXS) {
            const uint2 v = *(const uint2*)(c + 8 * (i >> 2));
            a = v.x;
            b = v.y;
        } else if constexpr (MF == XF_IQ3XXS) {
            a = *(const uint16_t*)(c + 2 * i);
            b = *(const uint32_t*)(c + 64 + 4 * (i >> 2));
        } else if constexpr (MF == XF_IQ2S) {
            const int k = i >> 2;
            a = (uint32_t)c[i] | ((uint32_t)c[32 + i] << 8) | ((uint32_t)c[64 + k] << 16) | ((uint32_t)c[72 + k] << 24);
        } else {
            const int k = i >> 2;
            a = (uint32_t)*(const uint16_t*)(c + 2 * i) | ((uint32_t)c[64 + k] << 16) | ((uint32_t)c[72 + i] << 24);
            b = c[104 + (k >> 1)];
        }
    }
    __device__ __forceinline__ void decode(int i, const void* grid, float* w) const {
        if constexpr (MF == XF_IQ2XXS) {
            const int l = i & 3;
            const float db = (0.5f + (float)(b >> 28)) * 0.25f * d;
            const uint2 g = ((const uint2*)grid)[(a >> (8 * l)) & 255];
            const uint32_t s = gg_sign8((b >> (7 * l)) & 127);
#pragma unroll
            for (int j = 0; j < 8; j++) w[j] = gg_sgn(db * gg_u8f(j < 4 ? g.x : g.y, j & 3), s, j);
        } else if constexpr (MF == XF_IQ3XXS) {
            const float db = (0.5f + (float)(b >> 28)) * 0.5f * d;
            const uint32_t s = gg_sign8((b >> (7 * (i & 3))) & 127);
            const uint32_t* G = (const uint32_t*)grid;
            const uint32_t g0 = G[a & 255], g1 = G[a >> 8];
#pragma unroll
            for (int j = 0; j < 8; j++) w[j] = gg_sgn(db * gg_u8f(j < 4 ? g0 : g1, j & 3), s, j);
        } else if constexpr (MF == XF_IQ2S) {
            const int l = i & 3;
            const uint32_t sc = a >> 24, qh = (a >> 16) & 255, s = (a >> 8) & 255;
            const float db = (0.5f + (float)(l < 2 ? sc & 15 : sc >> 4)) * 0.25f * d;
            const uint2 g = ((const uint2*)grid)[(a & 255) | (((qh >> (2 * l)) & 3) << 8)];
#pragma unroll
            for (int j = 0; j < 8; j++) w[j] = gg_sgn(db * gg_u8f(j < 4 ? g.x : g.y, j & 3), s, j);
        } else {
            const int k = i >> 2, e = 2 * (i & 3);
            const uint32_t qh = (a >> 16) & 255, s = a >> 24;
            const float db = (float)(1 + 2 * ((b >> (4 * (k & 1))) & 15)) * d;
            const uint32_t* G = (const uint32_t*)grid;
            const uint32_t g0 = G[(a & 255) | (((qh >> e) & 1) << 8)], g1 = G[((a >> 8) & 255) | (((qh >> (e + 1)) & 1) << 8)];
#pragma unroll
            for (int j = 0; j < 8; j++) w[j] = gg_sgn(db * gg_u8f(j < 4 ? g0 : g1, j & 3), s, j);
        }
    }
};

template <int MF, int RPW, int U>
__global__ void __launch_bounds__(256) k_moe_gg(const float* x, int which, const int* ids, const int* dslot,
                                                 const uint8_t* pool, int s0, size_t xbytes, uint8_t xfmt,
                                                 const uint8_t* grids, float* out) {
    __shared__ __align__(16) uint32_t sgrid[gg_grid_bytes(MF) / 4];   // only the format's grid (occupancy)
    const int k = blockIdx.y % TOPK, r = blockIdx.y / TOPK;
    const int sl = dslot[ids[r * TOPK + k]];
    if (sl < 0) return;   // uniform over the block
    gg_load_grid<MF>(grids, sgrid);
    const int K = which ? INTER : D;
    const float* xr = which ? x + ((size_t)r * TOPK + k) * INTER : x + (size_t)r * D;
    const int NR = which ? D : 2 * INTER;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * 8 * RPW + warp * RPW;
    if (row0 >= NR) return;
    MatView mv[3];
    expert_views((XFmt)xfmt, pool + (size_t)(sl - s0) * xbytes, mv);
    constexpr int CB = MF == XF_IQ2XXS ? 64 : MF == XF_IQ3XXS ? 96 : MF == XF_IQ2S ? 80 : 108;
    const int nb = K / 256;
    const uint8_t* q[RPW];
    const __half* ds[RPW];
#pragma unroll
    for (int j = 0; j < RPW; j++) {
        const int row = min(row0 + j, NR - 1);
        const int m = which ? 2 : (row < INTER ? 0 : 1), rr = which ? row : row % INTER;
        q[j] = mv[m].q + (size_t)rr * nb * CB;
        ds[j] = (const __half*)mv[m].s + (size_t)rr * nb;
    }
    float acc[RPW] = {};
    for (int b0 = 0; b0 < nb; b0 += U) {
        GgRaw<MF> raw[U][RPW];
#pragma unroll
        for (int u = 0; u < U; u++)
#pragma unroll
            for (int j = 0; j < RPW; j++)
                if (b0 + u < nb) raw[u][j].load(q[j] + (size_t)(b0 + u) * CB, ds[j] + b0 + u, lane);
#pragma unroll
        for (int u = 0; u < U; u++) {
            if (b0 + u >= nb) break;
            const float4* xp = (const float4*)(xr + (b0 + u) * 256 + lane * 8);
            const float4 x0 = __ldg(xp), x1 = __ldg(xp + 1);
#pragma unroll
            for (int j = 0; j < RPW; j++) {
                float w[8];
                raw[u][j].decode(lane, sgrid, w);
                acc[j] += w[0] * x0.x + w[1] * x0.y + w[2] * x0.z + w[3] * x0.w + w[4] * x1.x + w[5] * x1.y + w[6] * x1.z +
                          w[7] * x1.w;
            }
        }
    }
#pragma unroll
    for (int j = 0; j < RPW; j++) {
        float a = acc[j];
#pragma unroll
        for (int off = 16; off; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
        const int row = row0 + j;
        if (lane == 0 && row < NR) {
            float* o = which ? out + ((size_t)r * TOPK + k) * D + row : out + ((size_t)r * TOPK + k) * 2 * INTER + row;
            *o = a;
        }
    }
}
// R17: NM > 1 = verify passes: the block of the FIRST routed choice (r, k) of an expert reads its weights once and
// multiplies them with every row that chose it (up to NM rows); the other choices of that expert return. Each row's
// sum is formed in the same order as with NM = 1 (bit-identical outputs), the expert's bytes are read once per pass
// instead of once per row
template <int MF, int RPW, int U, int NM>
__global__ void __launch_bounds__(256) k_moe_ggm(const float* x, int which, const int* ids, const int* dslot,
                                                 const uint8_t* pool, int s0, size_t xbytes, uint8_t xfmt,
                                                 const uint8_t* grids, float* out, int R) {
    __shared__ __align__(16) uint32_t sgrid[gg_grid_bytes(MF) / 4];   // only the format's grid (occupancy)
    const int k = blockIdx.y % TOPK, r = blockIdx.y / TOPK;
    const int e = ids[r * TOPK + k];
    const int sl = dslot[e];
    if (sl < 0) return;   // uniform over the block
    int mo[NM];           // output choice index (r2 * TOPK + k2) of each matched row
    int nm = 1;
    mo[0] = r * TOPK + k;
    if constexpr (NM > 1) {
        for (int q = 0; q < r * TOPK; q++)
            if (ids[q] == e) return;   // an earlier row's block does this expert
        for (int q = (r + 1) * TOPK; q < R * TOPK && nm < NM; q++)
            if (ids[q] == e) mo[nm++] = q;
    }
    gg_load_grid<MF>(grids, sgrid);
    const int K = which ? INTER : D;
    const int NR = which ? D : 2 * INTER;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * 8 * RPW + warp * RPW;
    if (row0 >= NR) return;
    const float* xr[NM];
#pragma unroll
    for (int m = 0; m < NM; m++) {
        const int q = m < nm ? mo[m] : mo[0];
        xr[m] = which ? x + (size_t)q * INTER : x + (size_t)(q / TOPK) * D;
    }
    MatView mv[3];
    expert_views((XFmt)xfmt, pool + (size_t)(sl - s0) * xbytes, mv);
    constexpr int CB = MF == XF_IQ2XXS ? 64 : MF == XF_IQ3XXS ? 96 : MF == XF_IQ2S ? 80 : 108;
    const int nb = K / 256;
    const uint8_t* q[RPW];
    const __half* ds[RPW];
#pragma unroll
    for (int j = 0; j < RPW; j++) {
        const int row = min(row0 + j, NR - 1);
        const int mi = which ? 2 : (row < INTER ? 0 : 1), rr = which ? row : row % INTER;
        q[j] = mv[mi].q + (size_t)rr * nb * CB;
        ds[j] = (const __half*)mv[mi].s + (size_t)rr * nb;
    }
    float acc[NM][RPW] = {};
    for (int b0 = 0; b0 < nb; b0 += U) {
        GgRaw<MF> raw[U][RPW];
#pragma unroll
        for (int u = 0; u < U; u++)
#pragma unroll
            for (int j = 0; j < RPW; j++)
                if (b0 + u < nb) raw[u][j].load(q[j] + (size_t)(b0 + u) * CB, ds[j] + b0 + u, lane);
#pragma unroll
        for (int u = 0; u < U; u++) {
            if (b0 + u >= nb) break;
            float4 x0[NM], x1[NM];
#pragma unroll
            for (int m = 0; m < NM; m++) {
                if (m >= nm) break;
                const float4* xp = (const float4*)(xr[m] + (b0 + u) * 256 + lane * 8);
                x0[m] = __ldg(xp);
                x1[m] = __ldg(xp + 1);
            }
#pragma unroll
            for (int j = 0; j < RPW; j++) {
                float w[8];
                raw[u][j].decode(lane, sgrid, w);
#pragma unroll
                for (int m = 0; m < NM; m++) {
                    if (m >= nm) break;
                    acc[m][j] += w[0] * x0[m].x + w[1] * x0[m].y + w[2] * x0[m].z + w[3] * x0[m].w + w[4] * x1[m].x +
                                 w[5] * x1[m].y + w[6] * x1[m].z + w[7] * x1[m].w;
                }
            }
        }
    }
#pragma unroll
    for (int m = 0; m < NM; m++) {
        if (m >= nm) break;
#pragma unroll
        for (int j = 0; j < RPW; j++) {
            float a = acc[m][j];
#pragma unroll
            for (int off = 16; off; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
            const int row = row0 + j;
            if (lane == 0 && row < NR) {
                float* o = which ? out + (size_t)mo[m] * D + row : out + (size_t)mo[m] * 2 * INTER + row;
                *o = a;
            }
        }
    }
}
// engine choice of (RPW, U): set from the R16 measurement pass, not tuned per shape
constexpr int GG_RPW = 4, GG_U = 2;
template <int RPW, int U, int NM>
inline void gg_moe_launch_nm(int which, int mf, const float* x, const int* ids, const int* dslot, const uint8_t* pool,
                             int s0, size_t xbytes, uint8_t xf, const uint8_t* grids, float* out, int R, cudaStream_t st) {
    const int NR = which ? D : 2 * INTER;
    const dim3 g((NR + 8 * RPW - 1) / (8 * RPW), TOPK * R);
    switch (mf) {
        case XF_IQ2XXS: k_moe_ggm<XF_IQ2XXS, RPW, U, NM><<<g, 256, 0, st>>>(x, which, ids, dslot, pool, s0, xbytes, xf, grids, out, R); break;
        case XF_IQ3XXS: k_moe_ggm<XF_IQ3XXS, RPW, U, NM><<<g, 256, 0, st>>>(x, which, ids, dslot, pool, s0, xbytes, xf, grids, out, R); break;
        case XF_IQ2S: k_moe_ggm<XF_IQ2S, RPW, U, NM><<<g, 256, 0, st>>>(x, which, ids, dslot, pool, s0, xbytes, xf, grids, out, R); break;
        default: k_moe_ggm<XF_IQ3S, RPW, U, NM><<<g, 256, 0, st>>>(x, which, ids, dslot, pool, s0, xbytes, xf, grids, out, R); break;
    }
}
static const bool g_gg_rows_off = getenv("DSV4_GG_ROWS") == nullptr;   // R17: grouped rows opt-in (measured slower: R17-4)
template <int RPW, int U>
inline void gg_moe_launch(int which, int mf, const float* x, const int* ids, const int* dslot, const uint8_t* pool, int s0,
                           size_t xbytes, uint8_t xf, const uint8_t* grids, float* out, int R, cudaStream_t st) {
    if (R <= 1 || g_gg_rows_off) {   // the R16 kernel (one block per routed choice)
        const int NR = which ? D : 2 * INTER;
        const dim3 g((NR + 8 * RPW - 1) / (8 * RPW), TOPK * R);
        switch (mf) {
            case XF_IQ2XXS: k_moe_gg<XF_IQ2XXS, RPW, U><<<g, 256, 0, st>>>(x, which, ids, dslot, pool, s0, xbytes, xf, grids, out); break;
            case XF_IQ3XXS: k_moe_gg<XF_IQ3XXS, RPW, U><<<g, 256, 0, st>>>(x, which, ids, dslot, pool, s0, xbytes, xf, grids, out); break;
            case XF_IQ2S: k_moe_gg<XF_IQ2S, RPW, U><<<g, 256, 0, st>>>(x, which, ids, dslot, pool, s0, xbytes, xf, grids, out); break;
            default: k_moe_gg<XF_IQ3S, RPW, U><<<g, 256, 0, st>>>(x, which, ids, dslot, pool, s0, xbytes, xf, grids, out); break;
        }
        return;
    }
    if (R <= 2) return gg_moe_launch_nm<RPW, U, 2>(which, mf, x, ids, dslot, pool, s0, xbytes, xf, grids, out, R, st);
    if (R <= 4) return gg_moe_launch_nm<RPW, U, 4>(which, mf, x, ids, dslot, pool, s0, xbytes, xf, grids, out, R, st);
    return gg_moe_launch_nm<RPW, U, MAXR>(which, mf, x, ids, dslot, pool, s0, xbytes, xf, grids, out, R, st);
}

// R21 [dsv4-r21-dp4a]: the pool GEMV with int8 activations. x is quantized per 32 consecutive values (absmax / 127,
// round to nearest) by k_q8x; a lane takes its 8 weights as 8 unsigned grid bytes + 8 sign bits + one float scale
// (the format's sub-block scale x the block's d), turns the signs into byte masks (vcmpne4), applies them as a two's
// complement negate (XOR, vsub4) and forms the 8-weight dot with two dp4a; one float multiply-add per 8 weights.
// Not bit-identical to k_moe_gg (x rounded to int8 per 32, the CPU lane's activation level); the weights are exact.
struct GgI8 { uint32_t lo, hi, s; };   // magnitudes of weights 0-3 / 4-7, sign bits
// signed 8-weight dot with 8 int8 activations (xl: values 0-3, xh: 4-7). Sign bit j -> 0x01 in byte j by one multiply
// (bits 0-3 times 0x204081 land on bits 0 / 8 / 16 / 24 with no overlaps or carries), mask = that x 255; the signed
// weight byte is (g ^ mask) + 1 = -g. No byte carries: grid bytes are never 0 (smallest 1, IQ3_S). The byte-SIMD
// intrinsics (__vcmpne4, __vsub4) are emulated on sm_86 and measured no faster than the fp32 path (R21 b1)
__device__ __forceinline__ int gg_dot8(const GgI8& w, int xl, int xh) {
    const uint32_t bl = ((w.s & 15u) * 0x204081u) & 0x01010101u, bh = (((w.s >> 4) & 15u) * 0x204081u) & 0x01010101u;
    const int wl = (int)((w.lo ^ (bl * 255u)) + bl), wh = (int)((w.hi ^ (bh * 255u)) + bh);
    return __dp4a(wh, xh, __dp4a(wl, xl, 0));
}
// x [n] fp32 -> int8 [n] + float scale per 32 (n a multiple of 256); a thread = 8 values, 4 threads = one group.
// review r21 L1: a NaN / Inf in the group makes the group's scale NaN (the output shows it, as the fp32 path would);
// a group max below 1e-30 quantizes to 0 (127 / m would overflow and flip signs)
__global__ void k_q8x(const float* __restrict__ x, int n, int8_t* __restrict__ q, float* __restrict__ s) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const bool ok = t * 8 < n;
    float4 a = make_float4(0, 0, 0, 0), b = a;
    if (ok) { a = ((const float4*)x)[2 * t]; b = ((const float4*)x)[2 * t + 1]; }
    float m = fmaxf(fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(a.z), fabsf(a.w))),
                    fmaxf(fmaxf(fabsf(b.x), fabsf(b.y)), fmaxf(fabsf(b.z), fabsf(b.w))));
    int bad = !(isfinite(a.x) && isfinite(a.y) && isfinite(a.z) && isfinite(a.w) && isfinite(b.x) && isfinite(b.y) &&
                isfinite(b.z) && isfinite(b.w));   // fmaxf drops NaN: test every value
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
    bad |= __shfl_xor_sync(0xffffffffu, bad, 1);
    bad |= __shfl_xor_sync(0xffffffffu, bad, 2);
    if (!ok) return;
    const float d = bad ? __int_as_float(0x7fc00000) : m / 127.f, id = !bad && m > 1e-30f ? 127.f / m : 0.f;
    auto c = [&](float v) { return (uint32_t)(uint8_t)(int8_t)__float2int_rn(v * id); };
    uint2 o;
    o.x = c(a.x) | c(a.y) << 8 | c(a.z) << 16 | c(a.w) << 24;
    o.y = c(b.x) | c(b.y) << 8 | c(b.z) << 16 | c(b.w) << 24;
    ((uint2*)q)[t] = o;
    if ((t & 3) == 0) s[t >> 2] = d;
}
inline void gg_q8x_launch(const float* x, int n, int8_t* q, float* s, cudaStream_t st) {
    const int nt = n / 8;
    k_q8x<<<(nt + 255) / 256, 256, 0, st>>>(x, n, q, s);
}
// R21 [dsv4-r21-dp4a] v2: a lane owns one whole 32-weight sub-block (its x scale group): one to four loads per 32
// weights instead of per 8 (the loads-only skeleton of k_moe_gq measured 12.8 us per expert against a 7.4 us read
// floor: the load instructions, not the decode, set the time). Sub-block t = round * 32 + lane of the row
// (block t / 8, sub-block t % 8); x as two 16-byte loads, one float scale. Integer sum per sub-block (two for IQ2_S,
// whose scales are per 16), one float multiply-add per scale.
template <int MF>
__device__ __forceinline__ float gg_sub_dot(const uint32_t a, const uint32_t b, const uint32_t c, const uint32_t qh3,
                                            const float d, const int* xw, const void* grid) {
    if constexpr (MF == XF_IQ2XXS) {
        const uint2* G = (const uint2*)grid;
        int s = 0;
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint2 g = G[(a >> (8 * l)) & 255];
            const GgI8 w{g.x, g.y, gg_sign8((b >> (7 * l)) & 127)};
            s += gg_dot8(w, xw[2 * l], xw[2 * l + 1]);
        }
        return (0.5f + (float)(b >> 28)) * 0.25f * d * (float)s;
    } else if constexpr (MF == XF_IQ3XXS) {
        const uint32_t* G = (const uint32_t*)grid;
        int s = 0;
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint32_t q = l < 2 ? a : b;
            const GgI8 w{G[(q >> (16 * (l & 1))) & 255], G[(q >> (16 * (l & 1) + 8)) & 255], gg_sign8((c >> (7 * l)) & 127)};
            s += gg_dot8(w, xw[2 * l], xw[2 * l + 1]);
        }
        return (0.5f + (float)(c >> 28)) * 0.5f * d * (float)s;
    } else if constexpr (MF == XF_IQ2S) {
        const uint2* G = (const uint2*)grid;
        const uint32_t qh = c & 255, sc = c >> 8;
        int s0 = 0, s1 = 0;
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint2 g = G[((a >> (8 * l)) & 255) | (((qh >> (2 * l)) & 3) << 8)];
            const GgI8 w{g.x, g.y, (b >> (8 * l)) & 255};
            (l < 2 ? s0 : s1) += gg_dot8(w, xw[2 * l], xw[2 * l + 1]);
        }
        return 0.25f * d * ((0.5f + (float)(sc & 15)) * (float)s0 + (0.5f + (float)(sc >> 4)) * (float)s1);
    } else {
        const uint32_t* G = (const uint32_t*)grid;
        int s = 0;
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint32_t q = l < 2 ? a : b;
            const uint32_t i0 = ((q >> (16 * (l & 1))) & 255) | (((qh3 >> (2 * l)) & 1) << 8);
            const uint32_t i1 = ((q >> (16 * (l & 1) + 8)) & 255) | (((qh3 >> (2 * l + 1)) & 1) << 8);
            const GgI8 w{G[i0], G[i1], (c >> (8 * l)) & 255};
            s += gg_dot8(w, xw[2 * l], xw[2 * l + 1]);
        }
        return d * (float)s;   // d holds the sub-block scale already
    }
}
template <int MF, int RPW, int U>
__global__ void __launch_bounds__(256) k_moe_gq2(const int8_t* xq, const float* xs, int which, const int* ids,
                                                  const int* dslot, const uint8_t* pool, int s0, size_t xbytes,
                                                  uint8_t xfmt, const uint8_t* grids, float* out) {
    __shared__ __align__(16) uint32_t sgrid[gg_grid_bytes(MF) / 4];
    const int k = blockIdx.y % TOPK, r = blockIdx.y / TOPK;
    const int sl = dslot[ids[r * TOPK + k]];
    if (sl < 0) return;
    gg_load_grid<MF>(grids, sgrid);
    const int K = which ? INTER : D;
    const size_t xo = which ? ((size_t)r * TOPK + k) * INTER : (size_t)r * D;
    const int8_t* xr = xq + xo;
    const float* sr = xs + xo / 32;
    const int NR = which ? D : 2 * INTER;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * 8 * RPW + warp * RPW;
    if (row0 >= NR) return;
    MatView mv[3];
    expert_views((XFmt)xfmt, pool + (size_t)(sl - s0) * xbytes, mv);
    constexpr int CB = MF == XF_IQ2XXS ? 64 : MF == XF_IQ3XXS ? 96 : MF == XF_IQ2S ? 80 : 108;
    const int nb = K / 256, ns = nb * 8;   // sub-blocks per row
    const uint8_t* q[RPW];
    const __half* ds[RPW];
#pragma unroll
    for (int j = 0; j < RPW; j++) {
        const int row = min(row0 + j, NR - 1);
        const int m = which ? 2 : (row < INTER ? 0 : 1), rr = which ? row : row % INTER;
        q[j] = mv[m].q + (size_t)rr * nb * CB;
        ds[j] = (const __half*)mv[m].s + (size_t)rr * nb;
    }
    float acc[RPW] = {};
    for (int t0 = 0; t0 < ns; t0 += 32 * U) {
        uint32_t ra[U][RPW], rb[U][RPW], rc[U][RPW], rq[U][RPW];
        float rd[U][RPW];
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int t = t0 + 32 * u + lane, bb = t >> 3, kk = t & 7;
#pragma unroll
            for (int j = 0; j < RPW; j++) {
                ra[u][j] = rb[u][j] = rc[u][j] = rq[u][j] = 0; rd[u][j] = 0.f;
                if (t < ns) {
                    const uint8_t* c = q[j] + (size_t)bb * CB;
                    rd[u][j] = __half2float(ds[j][bb]);
                    if constexpr (MF == XF_IQ2XXS) {
                        const uint2 v = *(const uint2*)(c + 8 * kk); ra[u][j] = v.x; rb[u][j] = v.y;
                    } else if constexpr (MF == XF_IQ3XXS) {
                        const uint2 v = *(const uint2*)(c + 8 * kk); ra[u][j] = v.x; rb[u][j] = v.y;
                        rc[u][j] = *(const uint32_t*)(c + 64 + 4 * kk);
                    } else if constexpr (MF == XF_IQ2S) {
                        ra[u][j] = *(const uint32_t*)(c + 4 * kk);
                        rb[u][j] = *(const uint32_t*)(c + 32 + 4 * kk);
                        rc[u][j] = (uint32_t)c[64 + kk] | ((uint32_t)c[72 + kk] << 8);
                    } else {
                        ra[u][j] = *(const uint32_t*)(c + 8 * kk);   // 108-byte blocks: 4-byte aligned only
                        rb[u][j] = *(const uint32_t*)(c + 8 * kk + 4);
                        rc[u][j] = *(const uint32_t*)(c + 72 + 4 * kk);
                        rq[u][j] = (uint32_t)c[64 + kk] | ((uint32_t)c[104 + kk / 2] << 8);
                    }
                }
            }
        }
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int t = t0 + 32 * u + lane;
            if (t >= ns) break;
            const uint4 x0 = __ldg((const uint4*)(xr + t * 32)), x1 = __ldg((const uint4*)(xr + t * 32 + 16));
            const int xw[8] = {(int)x0.x, (int)x0.y, (int)x0.z, (int)x0.w, (int)x1.x, (int)x1.y, (int)x1.z, (int)x1.w};
            const float dx = __ldg(sr + t);
#pragma unroll
            for (int j = 0; j < RPW; j++) {
                float dd = rd[u][j];
                if constexpr (MF == XF_IQ3S) dd *= (float)(1 + 2 * ((rq[u][j] >> (8 + 4 * (t & 1))) & 15));
                acc[j] = fmaf(dx, gg_sub_dot<MF>(ra[u][j], rb[u][j], rc[u][j], rq[u][j] & 255, dd, xw, sgrid), acc[j]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < RPW; j++) {
        float a = acc[j];
#pragma unroll
        for (int off = 16; off; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
        const int row = row0 + j;
        if (lane == 0 && row < NR) {
            float* o = which ? out + ((size_t)r * TOPK + k) * D + row : out + ((size_t)r * TOPK + k) * 2 * INTER + row;
            *o = a;
        }
    }
}
template <int RPW, int U>
inline void gg_moe_launch_q2(int which, int mf, const int8_t* xq, const float* xs, const int* ids, const int* dslot,
                             const uint8_t* pool, int s0, size_t xbytes, uint8_t xf, const uint8_t* grids, float* out,
                             int R, cudaStream_t st) {
    const int NR = which ? D : 2 * INTER;
    const dim3 g((NR + 8 * RPW - 1) / (8 * RPW), TOPK * R);
    switch (mf) {
        case XF_IQ2XXS: k_moe_gq2<XF_IQ2XXS, RPW, U><<<g, 256, 0, st>>>(xq, xs, which, ids, dslot, pool, s0, xbytes, xf, grids, out); break;
        case XF_IQ3XXS: k_moe_gq2<XF_IQ3XXS, RPW, U><<<g, 256, 0, st>>>(xq, xs, which, ids, dslot, pool, s0, xbytes, xf, grids, out); break;
        case XF_IQ2S: k_moe_gq2<XF_IQ2S, RPW, U><<<g, 256, 0, st>>>(xq, xs, which, ids, dslot, pool, s0, xbytes, xf, grids, out); break;
        default: k_moe_gq2<XF_IQ3S, RPW, U><<<g, 256, 0, st>>>(xq, xs, which, ids, dslot, pool, s0, xbytes, xf, grids, out); break;
    }
}

// one matrix [N][K] (engine layout) -> bf16 [N][K]; a thread = one 32-weight sub-block
template <int MF>
__global__ void __launch_bounds__(256) k_deq_gg(const uint8_t* q, const __half* ds, int N, int K, const uint8_t* grids,
                                                bf16* out) {
    __shared__ __align__(16) uint32_t sgrid[2048];
    gg_load_grid<MF>(grids, sgrid);
    constexpr int CB = MF == XF_IQ2XXS ? 64 : MF == XF_IQ3XXS ? 96 : MF == XF_IQ2S ? 80 : 108;
    const size_t u = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= (size_t)N * K / 32) return;
    const size_t b = u >> 3;
    float w[32];
    gg_sub<MF>(q + b * CB, (int)(u & 7), sgrid, w);
    const float d = __half2float(ds[b]);
    bf162* o = (bf162*)(out + u * 32);
#pragma unroll
    for (int j = 0; j < 16; j++) o[j] = __floats2bfloat162_rn(w[2 * j] * d, w[2 * j + 1] * d);
}

// [dsv4-pf-deq4]: weights 8h .. 8h + 7 of sub-block k (h 0..3): the same operations as gg_sub for those weights
template <int MF>
__device__ __forceinline__ void gg_sub8(const uint8_t* c, int k, int h, const void* grid, float* w) {
    if constexpr (MF == XF_IQ2XXS) {
        const uint2 v = *(const uint2*)(c + 8 * k);
        const float db = (0.5f + (float)(v.y >> 28)) * 0.25f;
        const uint2 g = ((const uint2*)grid)[(v.x >> (8 * h)) & 255];
        const uint32_t s = gg_sign8((v.y >> (7 * h)) & 127);
#pragma unroll
        for (int j = 0; j < 8; j++) w[j] = gg_sgn(db * gg_u8f(j < 4 ? g.x : g.y, j & 3), s, j);
    } else if constexpr (MF == XF_IQ3XXS) {
        const uint2 v = *(const uint2*)(c + 8 * k);
        const uint32_t aux = *(const uint32_t*)(c + 64 + 4 * k);
        const float db = (0.5f + (float)(aux >> 28)) * 0.5f;
        const uint32_t* G = (const uint32_t*)grid;
#pragma unroll
        for (int ii = 0; ii < 2; ii++) {
            const int i = 2 * h + ii;
            const uint32_t g = G[((i < 4 ? v.x : v.y) >> (8 * (i & 3))) & 255];
            const uint32_t s = gg_sign8((aux >> (7 * (i / 2))) & 127) >> (4 * (i & 1));
#pragma unroll
            for (int j = 0; j < 4; j++) w[4 * ii + j] = gg_sgn(db * gg_u8f(g, j), s, j);
        }
    } else if constexpr (MF == XF_IQ2S) {
        const uint32_t qs = *(const uint32_t*)(c + 4 * k), sg = *(const uint32_t*)(c + 32 + 4 * k);
        const uint32_t qh = c[64 + k], sc = c[72 + k];
        const uint2 g = ((const uint2*)grid)[((qs >> (8 * h)) & 255) | (((qh >> (2 * h)) & 3) << 8)];
        const float db = (0.5f + (float)(h < 2 ? sc & 15 : sc >> 4)) * 0.25f;
        const uint32_t s = (sg >> (8 * h)) & 255;
#pragma unroll
        for (int j = 0; j < 8; j++) w[j] = gg_sgn(db * gg_u8f(j < 4 ? g.x : g.y, j & 3), s, j);
    } else {   // XF_IQ3S
        const uint32_t q0 = *(const uint32_t*)(c + 8 * k), q1 = *(const uint32_t*)(c + 8 * k + 4);
        const uint32_t qh = c[64 + k], sg = *(const uint32_t*)(c + 72 + 4 * k);
        const float db = (float)(1 + 2 * ((c[104 + k / 2] >> (4 * (k & 1))) & 15));
        const uint32_t* G = (const uint32_t*)grid;
#pragma unroll
        for (int ii = 0; ii < 2; ii++) {
            const int i = 2 * h + ii;
            const uint32_t g = G[(((i < 4 ? q0 : q1) >> (8 * (i & 3))) & 255) | (((qh >> i) & 1) << 8)];
            const uint32_t s = (sg >> (4 * i)) & 15;
#pragma unroll
            for (int j = 0; j < 4; j++) w[4 * ii + j] = gg_sgn(db * gg_u8f(g, j), s, j);
        }
    }
}
// [dsv4-pf-deq4]: the same values as k_deq_gg (bit-exact), written coalesced: 4 threads per 32-weight sub-block, one
// 16 B store each (a warp writes 512 contiguous bytes; k_deq_gg's 64 B per thread in 4 B stores ran at ~190 GB/s),
// grid-stride blocks so that each block loads the grid into shared memory once
template <int MF>
__global__ void __launch_bounds__(256) k_deq_gg4(const uint8_t* q, const __half* ds, int N, int K, const uint8_t* grids,
                                                 bf16* out) {
    __shared__ __align__(16) uint32_t sgrid[2048];
    gg_load_grid<MF>(grids, sgrid);
    constexpr int CB = MF == XF_IQ2XXS ? 64 : MF == XF_IQ3XXS ? 96 : MF == XF_IQ2S ? 80 : 108;
    const size_t nt = (size_t)N * K / 8;   // 8 weights per thread
    for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < nt; t += (size_t)gridDim.x * blockDim.x) {
        const size_t u = t >> 2, b = u >> 3;
        float w[8];
        gg_sub8<MF>(q + b * CB, (int)(u & 7), (int)(t & 3), sgrid, w);
        const float d = __half2float(ds[b]);
        uint4 o;
        bf162 p0 = __floats2bfloat162_rn(w[0] * d, w[1] * d), p1 = __floats2bfloat162_rn(w[2] * d, w[3] * d);
        bf162 p2 = __floats2bfloat162_rn(w[4] * d, w[5] * d), p3 = __floats2bfloat162_rn(w[6] * d, w[7] * d);
        o.x = *(uint32_t*)&p0; o.y = *(uint32_t*)&p1; o.z = *(uint32_t*)&p2; o.w = *(uint32_t*)&p3;
        *(uint4*)(out + t * 8) = o;
    }
}

// host: dispatch by matrix format
inline void gg_deq_launch(int mf, const uint8_t* q, const void* s, int N, int K, const uint8_t* grids, bf16* out,
                          cudaStream_t st) {
    const unsigned nbk = (unsigned)(((size_t)N * K / 32 + 255) / 256);
    const __half* ds = (const __half*)s;
    static const bool old = getenv("DSV4_NODEQ4") != nullptr;   // [dsv4-pf-deq4] off-switch: the old kernel
    if (!old) {
        const unsigned nb4 = (unsigned)std::min<size_t>(((size_t)N * K / 8 + 255) / 256, 82 * 6);
        switch (mf) {
            case XF_IQ2XXS: k_deq_gg4<XF_IQ2XXS><<<nb4, 256, 0, st>>>(q, ds, N, K, grids, out); break;
            case XF_IQ3XXS: k_deq_gg4<XF_IQ3XXS><<<nb4, 256, 0, st>>>(q, ds, N, K, grids, out); break;
            case XF_IQ2S: k_deq_gg4<XF_IQ2S><<<nb4, 256, 0, st>>>(q, ds, N, K, grids, out); break;
            default: k_deq_gg4<XF_IQ3S><<<nb4, 256, 0, st>>>(q, ds, N, K, grids, out); break;
        }
        return;
    }
    switch (mf) {
        case XF_IQ2XXS: k_deq_gg<XF_IQ2XXS><<<nbk, 256, 0, st>>>(q, ds, N, K, grids, out); break;
        case XF_IQ3XXS: k_deq_gg<XF_IQ3XXS><<<nbk, 256, 0, st>>>(q, ds, N, K, grids, out); break;
        case XF_IQ2S: k_deq_gg<XF_IQ2S><<<nbk, 256, 0, st>>>(q, ds, N, K, grids, out); break;
        default: k_deq_gg<XF_IQ3S><<<nbk, 256, 0, st>>>(q, ds, N, K, grids, out); break;
    }
}
}  // namespace dsv4
