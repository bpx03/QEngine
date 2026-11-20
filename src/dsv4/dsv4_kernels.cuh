#pragma once
// DSV4 M1 GPU kernels. Numerics follow tools/dsv4/dsv4_model.py (DeepSeek kernel.py semantics): every FP8 / FP4
// weight linear quantizes its input to FP8 e4m3 per 32 with a power-of-two scale; weights dequantize exactly; math
// fp32. Built WITHOUT --use_fast_math (exp / division / rsqrt as IEEE as the reference needs).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "dsv4.h"

namespace dsv4 {

// ---------------------------------------------------------------- scalar helpers

__device__ __forceinline__ float pow2ceil_f(float v) {   // 2^ceil(log2 v), v > 0 (torch.frexp semantics)
    int e;
    float m = frexpf(v, &e);
    if (m == 0.5f) e -= 1;
    return ldexpf(1.f, e);
}
__device__ __forceinline__ float e4m3_rne(float v) {     // v within +-448
    __nv_fp8_storage_t q = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    __half_raw h = __nv_cvt_fp8_to_halfraw(q, __NV_E4M3);
    return __half2float(__half(h));
}
__device__ __forceinline__ float e8m0f(uint8_t e) { return e ? __int_as_float((int)e << 23) : ldexpf(1.f, -127); }
__device__ __forceinline__ float rne_fp4_mag(float m) {  // |x| in [0, 6] to the e2m1 grid, ties to even
    return m < 2.f ? rintf(m * 2.f) * 0.5f : m < 4.f ? rintf(m) : rintf(m * 0.5f) * 2.f;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float siluf_(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float softplusf_(float x) { return x > 20.f ? x : log1pf(expf(x)); }

// ---------------------------------------------------------------- activation quantizers (one warp per 32 values)

// q_fp8: x [n][K] (row stride ldx) -> FP8-rounded values; fp32 out (ldo) and/or bf16 out (ldb), in place allowed
// [dsv4-rel2-p1] ob row of input row r = (brow + r) & bmask (a ring of window rows; defaults: row r)
__global__ void k_q_fp8(const float* x, int ldx, int n, int K, float* o, int ldo, bf16* ob, int ldb, int brow = 0,
                        int bmask = -1) {
    int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    int nb = K / 32;
    if (gw >= n * nb) return;
    int r = gw / nb, b = gw % nb;
    float v = x[(size_t)r * ldx + b * 32 + lane];
    float am = fmaxf(warp_max(fabsf(v)), 1e-4f);
    float s = pow2ceil_f(am * (1.f / 448.f));
    float y = e4m3_rne(fminf(fmaxf(v / s, -448.f), 448.f)) * s;
    if (o) o[(size_t)r * ldo + b * 32 + lane] = y;
    if (ob) ob[(size_t)((brow + r) & bmask) * ldb + b * 32 + lane] = __float2bfloat16(y);   // exact (3-bit mantissa)
}
// [dsv4-pf8-xqh] k_q_fp8 (bf16 out, the same expressions) + the packed form of the same values: pc [n][K] E4M3 codes,
// pe [n][K / 32] the power-of-2 block scale's exponent (s = 2^pe). k_fp8_unpk rebuilds the bf16 rows bit-exactly.
__global__ void k_q_fp8_pk(const float* x, int ldx, int n, int K, bf16* ob, int ldb, uint8_t* pc, int8_t* pe) {
    int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    int nb = K / 32;
    if (gw >= n * nb) return;
    int r = gw / nb, b = gw % nb;
    float v = x[(size_t)r * ldx + b * 32 + lane];
    float am = fmaxf(warp_max(fabsf(v)), 1e-4f);
    float s = pow2ceil_f(am * (1.f / 448.f));
    __nv_fp8_storage_t q = __nv_cvt_float_to_fp8(fminf(fmaxf(v / s, -448.f), 448.f), __NV_SATFINITE, __NV_E4M3);
    __half_raw h = __nv_cvt_fp8_to_halfraw(q, __NV_E4M3);
    float y = __half2float(__half(h)) * s;   // = e4m3_rne(..) * s
    ob[(size_t)r * ldb + b * 32 + lane] = __float2bfloat16(y);
    pc[(size_t)r * K + b * 32 + lane] = (uint8_t)q;
    if (lane == 0) {
        int e;
        frexpf(s, &e);   // s = 2^(e - 1) exactly
        pe[(size_t)r * nb + b] = (int8_t)(e - 1);
    }
}
// [dsv4-pf8-xqh] packed (pc, pe) -> bf16 [n][K] (8 values per thread; the k_q_fp8_pk value expression)
__global__ void k_fp8_unpk(const uint8_t* pc, const int8_t* pe, int n, int K, bf16* ob) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t * 8 >= (size_t)n * K) return;
    const size_t i0 = t * 8;
    const size_t r = i0 / K, col = i0 % K;
    const float s = ldexpf(1.f, (int)pe[r * (K / 32) + col / 32]);
    const uint2 c8 = *(const uint2*)(pc + i0);
    const uint8_t* cb = (const uint8_t*)&c8;
    __align__(16) bf16 o[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)cb[j], __NV_E4M3);
        o[j] = __float2bfloat16(__half2float(__half(h)) * s);
    }
    *(uint4*)(ob + i0) = *(const uint4*)o;
}
// q_fp4 per blk (16 or 32) along K, UE8M0 scale (e4m3s = 0) or E4M3 scale (compressed KV); in place on x;
// optional bf16 copy
__global__ void k_q_fp4(float* x, int ldx, int n, int K, int blk, int e4m3s, bf16* ob, int ldb) {
    int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    int per = 32 / blk;                       // groups per warp
    int nb = K / 32;
    if (gw >= n * nb) return;
    int r = gw / nb, b = gw % nb;
    float* p = x + (size_t)r * ldx + b * 32;
    float v = p[lane];
    float am = fabsf(v);
    for (int o = blk / 2; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));   // within blk lanes
    (void)per;
    float s;
    if (e4m3s) {
        am = fmaxf(am, 6.f * 0.001953125f);   // 6 * 2^-9
        s = e4m3_rne(am / 6.0f);
    } else {
        am = fmaxf(am, 6.f * 1.1754943508222875e-38f);
        s = pow2ceil_f(am * (1.f / 6.f));
    }
    float q = fminf(fmaxf(v / s, -6.f), 6.f);
    float y = (q > 0.f ? 1.f : q < 0.f ? -1.f : 0.f) * rne_fp4_mag(fabsf(q)) * s;
    p[lane] = y;
    if (ob) ob[(size_t)r * ldb + b * 32 + lane] = __float2bfloat16(y);
}

// ---------------------------------------------------------------- norms / rope / elementwise

// y = x * rsqrt(mean(x^2) + eps) * w, rows of length K (block per row)
__global__ void k_rms(const float* x, int ldx, const float* w, float* y, int ldy, int K) {
    __shared__ float red[32];
    const float* xr = x + (size_t)blockIdx.x * ldx;
    float s = 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) s += xr[k] * xr[k];
    s = warp_sum(s);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
        t = warp_sum(t);
        if (threadIdx.x == 0) red[0] = rsqrtf(t / (float)K + EPS);
    }
    __syncthreads();
    float r = red[0];
    float* yr = y + (size_t)blockIdx.x * ldy;
    for (int k = threadIdx.x; k < K; k += blockDim.x) yr[k] = xr[k] * r * w[k];
}
// rope on the last 64 dims of each head: x [n][nh][hd] (row stride ldx per token), positions pos0 + i or pos[i]
// (pos != nullptr); tables cos/sin [maxpos][32]; inverse = conjugate
__global__ void k_rope(float* x, int ldx, int n, int nh, int hd, const float* cosv, const float* sinv, int pos0,
                       int pstride, int inverse) {
    int i = blockIdx.x;
    int p = pos0 + i * pstride;
    for (int t = threadIdx.x; t < nh * 32; t += blockDim.x) {
        int h = t >> 5, j = t & 31;
        float* q = x + (size_t)i * ldx + (size_t)h * hd + (hd - RD) + 2 * j;
        float c = cosv[(size_t)p * 32 + j], s = sinv[(size_t)p * 32 + j];
        if (inverse) s = -s;
        float a = q[0], b = q[1];
        q[0] = a * c - b * s;
        q[1] = a * s + b * c;
    }
}

// ---------------------------------------------------------------- GEMV (rows <= MAXR), warp per output row

// FP8 weight [N][K] (e8m0 per 32x32), x fp32 [n][ldx] (already FP8-rounded), y [n][ldy] (+= when acc).
// grp: rows per input group (wo_a: 1024 rows read x columns [g*K, g*K + K) of group g); 0 = none
// FP8 GEMV: a group of LPR lanes per output row (LPR = 32 / 16 / 8 by K, so every lane streams >= ~8 x 16 B),
// 8 warps per block; x staged as bf16 in shared memory when R * Kx fits (exact: x is FP8-rounded)
// R17: x staged in shared memory up to 96 KB (opt-in dynamic shared memory, set per instantiation by f8_smem_init):
// the verify passes' 5-8 rows of K = 5120 / 8192 inputs used the fp32 global path (same products and order, slower)
#ifndef F8_SMEM_MAX
#define F8_SMEM_MAX (96 * 1024)
#endif
template <int R, int LPR = 32>
__global__ void __launch_bounds__(256) k_gemv_f8(const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, int N,
                                                 int K, const float* __restrict__ x, int ldx, float* y, int ldy, int grp,
                                                 int qx = 0) {
    extern __shared__ __align__(16) bf16 xs[];
    constexpr int RPB = 8 * (32 / LPR);                 // rows per block
    const int sub = threadIdx.x % LPR;
    const int row0 = blockIdx.x * RPB;
    const int xo_blk = grp ? (row0 / grp) * K : 0;      // all rows of a block in one group (grp % RPB == 0)
    const bool sm = R * K * 2 <= F8_SMEM_MAX;
    if (sm) {
        // qx: the reference's input quantization (FP8 e4m3 per 32, power-of-two scale) done here, one warp per 32
        if (qx == 2) {   // x already FP8-rounded bf16 (one q_fp8 for all blocks): 16 B copies
            const bf16* xb = (const bf16*)x;
            for (int t = threadIdx.x * 8; t < R * K; t += blockDim.x * 8) {
                int r = t / K, k = t % K;
                *(uint4*)(xs + t) = *(const uint4*)(xb + (size_t)r * ldx + xo_blk + k);
            }
        } else
        for (int t = threadIdx.x; t < R * K; t += blockDim.x) {
            int r = t / K, k = t % K;
            float v = x[(size_t)r * ldx + xo_blk + k];
            if (qx) {   // t advances by 256 = 8 x 32: every warp holds whole 32-blocks
                float am = fmaxf(warp_max(fabsf(v)), 1e-4f);
                float s8 = pow2ceil_f(am * (1.f / 448.f));
                v = e4m3_rne(fminf(fmaxf(v / s8, -448.f), 448.f)) * s8;
            }
            xs[t] = __float2bfloat16(v);
        }
        __syncthreads();
    }
    const int row = row0 + threadIdx.x / LPR;
    if (row >= N) return;
    const uint8_t* wr = W + (size_t)row * K;
    const uint8_t* sr = S + (size_t)(row / 32) * (K / 32);
    const int xo = grp ? (row / grp) * K : 0;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.f;
    const int nch = K / 16;
    for (int c0 = sub; c0 < nch; c0 += 4 * LPR) {
        uint4 v[4];
#pragma unroll
        for (int u = 0; u < 4; u++)
            v[u] = c0 + LPR * u < nch ? *(const uint4*)(wr + (c0 + LPR * u) * 16) : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int u = 0; u < 4; u++) {
            int c = c0 + LPR * u;
            if (c >= nch) break;
            // e4m3 -> fp16 bits by shifts (value / 256; the 2^8 is folded into the scale): exact incl. subnormals
            const uint32_t* w32 = (const uint32_t*)&v[u];
            float w[16];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                uint32_t q = w32[j];
                uint32_t lo = ((q & 0x007F007Fu) << 7) | ((q & 0x00800080u) << 8);   // bytes 0, 2
                uint32_t hi = ((q & 0x7F007F00u) >> 1) | (q & 0x80008000u);          // bytes 1, 3
                float2 f0 = __half22float2(*(const __half2*)&lo), f1 = __half22float2(*(const __half2*)&hi);
                w[4 * j] = f0.x;
                w[4 * j + 1] = f1.x;
                w[4 * j + 2] = f0.y;
                w[4 * j + 3] = f1.y;
            }
            float sc = e8m0f(sr[c >> 1]) * 256.f;
#pragma unroll
            for (int r = 0; r < R; r++) {
                float d = 0.f;
                if (sm) {
                    const uint4* xp = (const uint4*)(xs + (size_t)r * K + c * 16);
                    uint4 xa = xp[0], xb = xp[1];
                    const bf162* p0 = (const bf162*)&xa;
                    const bf162* p1 = (const bf162*)&xb;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        float2 f0 = __bfloat1622float2(p0[j]), f1 = __bfloat1622float2(p1[j]);
                        d += w[2 * j] * f0.x + w[2 * j + 1] * f0.y + w[8 + 2 * j] * f1.x + w[9 + 2 * j] * f1.y;
                    }
                } else {
                    // same products and summation order as the shared-memory path (a pass of several rows must
                    // give each row exactly what a one-row step gives)
                    const float4* xp = (const float4*)(x + (size_t)r * ldx + xo + c * 16);
                    float xv[16];
#pragma unroll
                    for (int j = 0; j < 4; j++) *(float4*)&xv[4 * j] = xp[j];
#pragma unroll
                    for (int j = 0; j < 4; j++)
                        d += w[2 * j] * xv[2 * j] + w[2 * j + 1] * xv[2 * j + 1] + w[8 + 2 * j] * xv[8 + 2 * j] +
                             w[9 + 2 * j] * xv[9 + 2 * j];
                }
                acc[r] += d * sc;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
        float t = acc[r];
        for (int o = LPR / 2; o; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
        if (sub == 0) y[(size_t)r * ldy + row] = t;
    }
}
// R18: k_gemv_f8 with x staged per K tile (64 * LPR columns = one c0 step of every lane) instead of whole rows: 2-8
// rows at K 4096-8192 needed up to 96 KB per block (1 block / SM, latency bound). Same chunks per lane in the same
// order, same products and sums: each row's result = the 1-row k_gemv_f8 bitwise (tests/gemv_rows_bench.cu).
template <int R, int LPR = 32>
__global__ void __launch_bounds__(256) k_gemv_f8t(const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, int N,
                                                  int K, const float* __restrict__ x, int ldx, float* y, int ldy, int grp,
                                                  int qx = 0) {
    extern __shared__ __align__(16) bf16 xs[];          // [R][TW]
    constexpr int RPB = 8 * (32 / LPR), TW = 64 * LPR;
    const int sub = threadIdx.x % LPR;
    const int row0 = blockIdx.x * RPB;
    const int xo_blk = grp ? (row0 / grp) * K : 0;
    const int row = row0 + threadIdx.x / LPR;
    const bool live = row < N;
    const uint8_t* wr = W + (size_t)(live ? row : 0) * K;
    const uint8_t* sr = S + (size_t)((live ? row : 0) / 32) * (K / 32);
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.f;
    const int nch = K / 16;
    for (int t0 = 0; t0 < K; t0 += TW) {
        const int tw = min(TW, K - t0);
        __syncthreads();
        if (qx == 2) {
            const bf16* xb = (const bf16*)x;
            for (int t = threadIdx.x * 8; t < R * tw; t += blockDim.x * 8) {
                int r = t / tw, k = t % tw;
                *(uint4*)(xs + r * TW + k) = *(const uint4*)(xb + (size_t)r * ldx + xo_blk + t0 + k);
            }
        } else
        for (int t = threadIdx.x; t < R * tw; t += blockDim.x) {
            int r = t / tw, k = t % tw;
            float v = x[(size_t)r * ldx + xo_blk + t0 + k];
            if (qx) {   // whole 32-blocks per warp (tw and K are multiples of 32)
                float am = fmaxf(warp_max(fabsf(v)), 1e-4f);
                float s8 = pow2ceil_f(am * (1.f / 448.f));
                v = e4m3_rne(fminf(fmaxf(v / s8, -448.f), 448.f)) * s8;
            }
            xs[r * TW + k] = __float2bfloat16(v);
        }
        __syncthreads();
        if (!live) continue;
        const int c0 = t0 / 16 + sub;
        uint4 v[4];
#pragma unroll
        for (int u = 0; u < 4; u++)
            v[u] = c0 + LPR * u < nch ? *(const uint4*)(wr + (c0 + LPR * u) * 16) : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int u = 0; u < 4; u++) {
            int c = c0 + LPR * u;
            if (c >= nch) break;
            const uint32_t* w32 = (const uint32_t*)&v[u];
            float w[16];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                uint32_t q = w32[j];
                uint32_t lo = ((q & 0x007F007Fu) << 7) | ((q & 0x00800080u) << 8);
                uint32_t hi = ((q & 0x7F007F00u) >> 1) | (q & 0x80008000u);
                float2 f0 = __half22float2(*(const __half2*)&lo), f1 = __half22float2(*(const __half2*)&hi);
                w[4 * j] = f0.x;
                w[4 * j + 1] = f1.x;
                w[4 * j + 2] = f0.y;
                w[4 * j + 3] = f1.y;
            }
            float sc = e8m0f(sr[c >> 1]) * 256.f;
#pragma unroll
            for (int r = 0; r < R; r++) {
                float d = 0.f;
                const uint4* xp = (const uint4*)(xs + (size_t)r * TW + (c * 16 - t0));
                uint4 xa = xp[0], xb = xp[1];
                const bf162* p0 = (const bf162*)&xa;
                const bf162* p1 = (const bf162*)&xb;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    float2 f0 = __bfloat1622float2(p0[j]), f1 = __bfloat1622float2(p1[j]);
                    d += w[2 * j] * f0.x + w[2 * j + 1] * f0.y + w[8 + 2 * j] * f1.x + w[9 + 2 * j] * f1.y;
                }
                acc[r] += d * sc;
            }
        }
    }
    if (!live) return;
#pragma unroll
    for (int r = 0; r < R; r++) {
        float t = acc[r];
        for (int o = LPR / 2; o; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
        if (sub == 0) y[(size_t)r * ldy + row] = t;
    }
}
template <int R>
inline void f8_smem_init_r() {
    cudaFuncSetAttribute(k_gemv_f8<R, 32>, cudaFuncAttributeMaxDynamicSharedMemorySize, F8_SMEM_MAX);
    cudaFuncSetAttribute(k_gemv_f8<R, 16>, cudaFuncAttributeMaxDynamicSharedMemorySize, F8_SMEM_MAX);
    cudaFuncSetAttribute(k_gemv_f8<R, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, F8_SMEM_MAX);
}
inline void f8_smem_init() {   // per device (call with each card current)
    f8_smem_init_r<1>(); f8_smem_init_r<2>(); f8_smem_init_r<3>(); f8_smem_init_r<4>();
    f8_smem_init_r<5>(); f8_smem_init_r<6>(); f8_smem_init_r<7>(); f8_smem_init_r<8>();
}
inline int gemv_lpr(int K) { return K >= 4096 ? 32 : K >= 2048 ? 16 : 8; }

// R10: 4 weights at float4 index c of a row as fp32: fp32 weights, or BF16 weights (the checkpoint's router /
// compressor / indexer projections) widened exactly (same products: bit-identical to an fp32 copy, half the bytes)
__device__ __forceinline__ float4 ldw4(const float* wr, int c) { return ((const float4*)wr)[c]; }
__device__ __forceinline__ float4 ldw4(const bf16* wr, int c) {
    const uint2 u = ((const uint2*)wr)[c];
    return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xFFFF0000u), __uint_as_float(u.y << 16),
                       __uint_as_float(u.y & 0xFFFF0000u));
}
// fp32 (or BF16) weight [N][K] (K % 4 == 0)
template <int R, typename WT = float>
__global__ void k_gemv_f32(const WT* __restrict__ W, int N, int K, const float* __restrict__ x, int ldx, float* y,
                           int ldy) {
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= N) return;
    const WT* wr = W + (size_t)row * K;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.f;
    for (int c = lane; c < K / 4; c += 32) {
        float4 w = ldw4(wr, c);
#pragma unroll
        for (int r = 0; r < R; r++) {
            float4 xv = ((const float4*)(x + (size_t)r * ldx))[c];
            acc[r] += w.x * xv.x + w.y * xv.y + w.z * xv.z + w.w * xv.w;
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
        float t = warp_sum(acc[r]);
        if (lane == 0) y[(size_t)r * ldy + row] = t;
    }
}
// fp32 weight [N][K], small N (router, compressor, indexer): KS warps split K per row (8 / KS rows per block), 4 loads
// in flight per lane; the one-warp-per-row kernel above left most SMs idle at N = 384 (48 blocks)
template <int R, int KS, typename WT = float>
__global__ void __launch_bounds__(256) k_gemv_f32k(const WT* __restrict__ W, int N, int K, const float* __restrict__ x,
                                                   int ldx, float* y, int ldy) {
    __shared__ float red[8][R];
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * (8 / KS) + w / KS, ks = w % KS;
    const int K4 = K / 4, seg = (K4 + KS - 1) / KS, c0 = ks * seg, c1 = min(K4, c0 + seg);
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.f;
    if (row < N) {
        const WT* wr = W + (size_t)row * K;
        int c = c0 + lane;
        for (; c + 96 < c1; c += 128) {
            float4 wv[4];
#pragma unroll
            for (int u = 0; u < 4; u++) wv[u] = ldw4(wr, c + 32 * u);
#pragma unroll
            for (int u = 0; u < 4; u++)
#pragma unroll
                for (int r = 0; r < R; r++) {
                    float4 xv = ((const float4*)(x + (size_t)r * ldx))[c + 32 * u];
                    acc[r] += wv[u].x * xv.x + wv[u].y * xv.y + wv[u].z * xv.z + wv[u].w * xv.w;
                }
        }
        for (; c < c1; c += 32) {
            float4 wv = ldw4(wr, c);
#pragma unroll
            for (int r = 0; r < R; r++) {
                float4 xv = ((const float4*)(x + (size_t)r * ldx))[c];
                acc[r] += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
        float t = warp_sum(acc[r]);
        if (lane == 0) red[w][r] = t;
    }
    __syncthreads();
    if (ks == 0 && lane < R && row < N) {
        float t = 0.f;
        for (int j = 0; j < KS; j++) t += red[w + j][lane];   // fixed order: deterministic
        y[(size_t)lane * ldy + row] = t;
    }
}
// bf16 -> fp32 (prefill: the BF16 projections widened for the fp32 GEMM)
__global__ void k_bf16_f32(const bf16* a, size_t n, float* o) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) o[t] = __bfloat162float(a[t]);
}
// R23 (review r23 L3): the head GEMVs' 8-term dot (k_gemv_bf16 / k_gemv_bf16s / k_gemv_bf16t): one expression, so
// every head kernel's row is bitwise the one-row head's by construction
__device__ __forceinline__ void bf16x8(const uint4& v, float w[8]) {
    const bf16* b = (const bf16*)&v;
#pragma unroll
    for (int j = 0; j < 8; j++) w[j] = __bfloat162float(b[j]);
}
__device__ __forceinline__ float dot8(const float w[8], const float4 a, const float4 b) {
    return w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w + w[4] * b.x + w[5] * b.y + w[6] * b.z + w[7] * b.w;
}
// bf16 weight [N][K] (head), K % 8 == 0
template <int R>
__global__ void k_gemv_bf16(const bf16* __restrict__ W, int N, int K, const float* __restrict__ x, int ldx, float* y,
                            int ldy) {
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= N) return;
    const uint4* wr = (const uint4*)(W + (size_t)row * K);
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.f;
    for (int c = lane; c < K / 8; c += 32) {
        const uint4 v = wr[c];
        float w[8];
        bf16x8(v, w);
#pragma unroll
        for (int r = 0; r < R; r++) {
            const float4* xp = (const float4*)(x + (size_t)r * ldx + c * 8);
            const float4 a = xp[0], bq = xp[1];
            acc[r] += dot8(w, a, bq);
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
        float t = warp_sum(acc[r]);
        if (lane == 0) y[(size_t)r * ldy + row] = t;
    }
}

// R23 [dsv4-r23-hstore]: k_gemv_bf16 for a y in another card's memory (the head split's slices go straight into the
// last card's logits over PCIe): the block's 8 rows are staged in shared memory and stored as R runs of 8 consecutive
// floats (32 B writes) instead of one 4 B remote store per (row, x row) from lane 0 of each warp, which capped the
// shard at ~2.3 ns per store (cards 0 / 1 at R 3-6: 1.25-1.5x card 2). Loads, lane order and expression are the
// ones of k_gemv_bf16: every row bitwise equal to it (and so to the one-row head).
// RPW rows per warp (x loads shared by the warp's rows; each row's own loads, lane order and expression unchanged):
// 8 RPW rows per block, stored as R runs of 8 RPW floats
template <int R, int RPW = 1>
__global__ void __launch_bounds__(256) k_gemv_bf16s(const bf16* __restrict__ W, int N, int K, const float* __restrict__ x,
                                                    int ldx, float* y, int ldy) {
    constexpr int RB = 8 * RPW;
    __shared__ float ys[R][RB];
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, row0 = blockIdx.x * RB + w * RPW;
    if (row0 < N) {
        const uint4* wr[RPW];
#pragma unroll
        for (int q = 0; q < RPW; q++) wr[q] = (const uint4*)(W + (size_t)min(row0 + q, N - 1) * K);
        float acc[RPW][R];
#pragma unroll
        for (int q = 0; q < RPW; q++)
#pragma unroll
            for (int r = 0; r < R; r++) acc[q][r] = 0.f;
        for (int c = lane; c < K / 8; c += 32) {
            uint4 v[RPW];
#pragma unroll
            for (int q = 0; q < RPW; q++) v[q] = wr[q][c];
#pragma unroll
            for (int r = 0; r < R; r++) {
                const float4* xp = (const float4*)(x + (size_t)r * ldx + c * 8);
                const float4 a = xp[0], bq = xp[1];
#pragma unroll
                for (int q = 0; q < RPW; q++) {
                    float wf[8];
                    bf16x8(v[q], wf);
                    acc[q][r] += dot8(wf, a, bq);
                }
            }
        }
#pragma unroll
        for (int q = 0; q < RPW; q++)
#pragma unroll
            for (int r = 0; r < R; r++) {
                float t = warp_sum(acc[q][r]);
                if (lane == 0) ys[r][w * RPW + q] = t;
            }
    }
    __syncthreads();
    for (int t = threadIdx.x; t < R * RB; t += blockDim.x) {
        const int r = t / RB, j = t % RB, rr = blockIdx.x * RB + j;
        if (rr < N) y[(size_t)r * ldy + rr] = ys[r][j];
    }
}

// R20 [dsv4-r20-headt]: k_gemv_bf16 with x staged in shared memory per K tile (TW = 1024 columns = 4 chunks of 8 per
// lane) and the 4 weight loads of a tile issued together. Each lane takes the same chunks c = lane + 32 i in the same
// order with the same expression: every row = k_gemv_bf16 bitwise. For R >= 2 (the drafter's 5 rows, verify heads):
// k_gemv_bf16 re-read 2 float4 of x per row per 16-byte weight load from L1.
template <int R>
__global__ void __launch_bounds__(256) k_gemv_bf16t(const bf16* __restrict__ W, int N, int K, const float* __restrict__ x,
                                                    int ldx, float* y, int ldy) {
    constexpr int TW = 1024;
    __shared__ __align__(16) float xs[R][TW];
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const bool live = row < N;
    const uint4* wr = (const uint4*)(W + (size_t)(live ? row : 0) * K);
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.f;
    for (int t0 = 0; t0 < K; t0 += TW) {
        const int tw = min(TW, K - t0);
        __syncthreads();
        for (int t = threadIdx.x * 4; t < R * tw; t += blockDim.x * 4) {
            const int r = t / tw, k = t % tw;
            *(float4*)&xs[r][k] = *(const float4*)(x + (size_t)r * ldx + t0 + k);
        }
        __syncthreads();
        if (!live) continue;
        const int c0 = t0 / 8 + lane, nc = (t0 + tw) / 8;
        uint4 v[4];
#pragma unroll
        for (int u = 0; u < 4; u++) v[u] = c0 + 32 * u < nc ? wr[c0 + 32 * u] : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const int c = c0 + 32 * u;
            if (c >= nc) break;
            float w[8];
            bf16x8(v[u], w);
#pragma unroll
            for (int r = 0; r < R; r++) {
                const float4* xp = (const float4*)&xs[r][c * 8 - t0];
                acc[r] += dot8(w, xp[0], xp[1]);
            }
        }
    }
    if (!live) return;
#pragma unroll
    for (int r = 0; r < R; r++) {
        float t = warp_sum(acc[r]);
        if (lane == 0) y[(size_t)r * ldy + row] = t;
    }
}

// ---------------------------------------------------------------- dequant to bf16 (prefill GEMM path)

// FP8 [N][K] rows [r0, r0 + nr) -> bf16 [nr][K]
__global__ void k_deq_f8(const uint8_t* W, const uint8_t* S, int K, int r0, int nr, bf16* out) {
    size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i >= (size_t)nr * K) return;
    int r = (int)(i / K) + r0, k = (int)(i % K);
    __half2_raw h2 = __nv_cvt_fp8x2_to_halfraw2(*(const __nv_fp8x2_storage_t*)(W + (size_t)r * K + k), __NV_E4M3);
    float2 f = __half22float2(__half2(h2));
    float sc = e8m0f(S[(size_t)(r / 32) * (K / 32) + k / 32]);
    out[i] = __float2bfloat16(f.x * sc);
    out[i + 1] = __float2bfloat16(f.y * sc);
}
// release prep [dsv4-rel-q8]: Q8_0 matrix (int8 [N][K] + fp16 scale per row and 32 columns, split from the GGUF
// blocks at load) rows [r0, r0 + nr) -> bf16 for the prefill GEMM (int8 x fp16 rounded once to bf16, relative 2^-9)
__global__ void k_deq_q8(const int8_t* W, const __half* S, int K, int r0, int nr, bf16* out) {
    size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i >= (size_t)nr * K) return;
    const int r = (int)(i / K) + r0, k = (int)(i % K);
    const float sc = __half2float(S[(size_t)r * (K / 32) + k / 32]);
    const char2 w = *(const char2*)(W + (size_t)r * K + k);
    out[i] = __float2bfloat16(sc * (float)w.x);
    out[i + 1] = __float2bfloat16(sc * (float)w.y);
}
// [dsv4-rel-q8]: x [n] rows at stride ldx -> bf16 [n][K] at stride ldy
__global__ void k_to_bf16s(const float* x, int ldx, int n, int K, bf16* y, int ldy) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)n * K) return;
    int i = (int)(t / K), k = (int)(t % K);
    y[(size_t)i * ldy + k] = __float2bfloat16(x[(size_t)i * ldx + k]);
}
// release prep [dsv4-rel-q6k]: Engram rows in Q6_K (the GGUF's tables: one 210 B super-block of 256 values per row:
// ql[128] | qh[64] | scales int8[16] | d fp16), each row in its 264 B staging slot -> fp32 [nrows][256]; the value
// order and the fp32 arithmetic of ggml / gguf-py dequantize ((d * scale) * q): bitwise equal
__global__ void k_eg_decode_q6k(const uint8_t* rows, int nrows, float* emb) {
    const int r = blockIdx.x, t = threadIdx.x;   // 256 threads: one output value each
    const uint8_t* p = rows + (size_t)r * 264;
    const int half = t >> 7, j = t & 127, quad = j >> 5, l = j & 31;   // y[half * 128 + quad * 32 + l]
    const uint8_t* ql = p + half * 64;
    const uint8_t* qh = p + 128 + half * 32;
    const int8_t* sc = (const int8_t*)(p + 192) + half * 8;
    const float d = __half2float(*(const __half*)(p + 208));
    const int lo = quad & 1 ? ql[l + 32] : ql[l];
    const int q = ((quad >> 1 ? lo >> 4 : lo & 15) | (((qh[l] >> (2 * quad)) & 3) << 4)) - 32;
    emb[(size_t)r * EG_HD + t] = (d * (float)sc[l / 16 + 2 * quad]) * (float)q;
}
// [dsv4-dec5-egz] the same decode with the rows read in place from mapped pinned host memory (decode batches: no H2D
// copy, which queued behind the tier's admission copies on the copy engine): each block stages its 264 B slot in
// shared memory with 4 B loads first; values and arithmetic = k_eg_decode_q6k (bitwise)
__global__ void k_eg_decode_q6k_h(const uint8_t* rows, int nrows, float* emb) {
    __shared__ uint32_t sm[66];
    const int r = blockIdx.x, t = threadIdx.x;
    if (t < 66) sm[t] = ((const volatile uint32_t*)(rows + (size_t)r * 264))[t];
    __syncthreads();
    const uint8_t* p = (const uint8_t*)sm;
    const int half = t >> 7, j = t & 127, quad = j >> 5, l = j & 31;
    const uint8_t* ql = p + half * 64;
    const uint8_t* qh = p + 128 + half * 32;
    const int8_t* sc = (const int8_t*)(p + 192) + half * 8;
    const float d = __half2float(*(const __half*)(p + 208));
    const int lo = quad & 1 ? ql[l + 32] : ql[l];
    const int q = ((quad >> 1 ? lo >> 4 : lo & 15) | (((qh[l] >> (2 * quad)) & 3) << 4)) - 32;
    emb[(size_t)r * EG_HD + t] = (d * (float)sc[l / 16 + 2 * quad]) * (float)q;
}
__constant__ float c_fp4[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
// FP4 matrix [N][K] (bytes K/2 per row, lo nibble = even element; e8m0 per 32) -> bf16 [N][K]; one thread per byte
__global__ void k_deq_fp4(const uint8_t* q, const uint8_t* s, int N, int K, bf16* out) {
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = c_fp4[threadIdx.x];
    __syncthreads();
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)N * K / 2) return;
    int r = (int)(i / (K / 2)), j = (int)(i % (K / 2));
    uint8_t b = q[i];
    float sc = e8m0f(s[(size_t)r * (K / 32) + (2 * j) / 32]);
    out[(size_t)r * K + 2 * j] = __float2bfloat16(lut[b & 15] * sc);
    out[(size_t)r * K + 2 * j + 1] = __float2bfloat16(lut[b >> 4] * sc);
}
// codebook matrix (cb3g128: 48 B per 128 group = 32 B of 2-bit fields + 16 B high bits; cb2g128: 32 B), fp16 scale
// per 128 -> bf16. Element k of a group: field byte k & 31, bits 2 (k >> 5); high bit = bit k of bytes 32..47.
// codebook values in cb[] (float, as dsv4_quant.py)
__global__ void k_deq_cb(const uint8_t* q, const __half* s, int N, int K, int bits, const float* cb, bf16* out) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)N * K) return;
    int r = (int)(i / K), k = (int)(i % K);
    int g = k >> 7, kk = k & 127;
    int gb = bits == 3 ? 48 : 32;
    const uint8_t* p = q + ((size_t)r * (K / 128) + g) * gb;
    int code = (p[kk & 31] >> (2 * (kk >> 5))) & 3;
    if (bits == 3) code |= ((p[32 + (kk >> 3)] >> (kk & 7)) & 1) << 2;
    float sc = __half2float(s[(size_t)r * (K / 128) + g]);
    out[i] = __float2bfloat16(cb[code] * sc);
}


// vectorized prefill dequant: one thread per 8 consecutive elements of a row (one 16 B bf16 store)
__global__ void k_deq_cb8(const uint8_t* q, const __half* s, int N, int K, int bits, const float* cb, bf16* out) {
    __shared__ float c[8];
    if (threadIdx.x < 8) c[threadIdx.x] = cb[threadIdx.x < (1 << bits) ? threadIdx.x : 0];
    __syncthreads();
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)N * K / 8) return;
    size_t e0 = t * 8;
    int r = (int)(e0 / K), k = (int)(e0 % K);
    int g = k >> 7, kk = k & 127;
    int gb = bits == 3 ? 48 : 32;
    const uint8_t* p = q + ((size_t)r * (K / 128) + g) * gb;
    uint2 f = *(const uint2*)(p + (kk & 31));          // 8 field bytes (kk & 31 is a multiple of 8)
    const uint8_t* fb = (const uint8_t*)&f;
    int sh = 2 * (kk >> 5);
    uint32_t hb = bits == 3 ? p[32 + (kk >> 3)] : 0;
    float sc = __half2float(s[(size_t)r * (K / 128) + g]);
    __align__(16) bf16 o[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        int code = ((fb[j] >> sh) & 3) | (((hb >> j) & 1) << 2);
        o[j] = __float2bfloat16(c[code] * sc);
    }
    *(uint4*)(out + e0) = *(const uint4*)o;
}
// FP4: one thread per 8 bytes (16 elements, two 16 B stores)
__global__ void k_deq_fp4v(const uint8_t* q, const uint8_t* s, int N, int K, bf16* out) {
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = c_fp4[threadIdx.x];
    __syncthreads();
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)N * K / 16) return;
    size_t e0 = t * 16;
    int r = (int)(e0 / K), k = (int)(e0 % K);
    uint2 b = *(const uint2*)(q + e0 / 2);
    const uint8_t* bb = (const uint8_t*)&b;
    float sc = e8m0f(s[(size_t)r * (K / 32) + k / 32]);
    __align__(16) bf16 o[16];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        o[2 * j] = __float2bfloat16(lut[bb[j] & 15] * sc);
        o[2 * j + 1] = __float2bfloat16(lut[bb[j] >> 4] * sc);
    }
    *(uint4*)(out + e0) = *(const uint4*)o;
    *(uint4*)(out + e0 + 8) = *(const uint4*)(o + 8);
}

// ---------------------------------------------------------------- mHC

// mixes for decode rows: m[i][j] = <x_i, fn_j> (x_i = 4 streams flat, 20480), ss[i] = sum x_i^2; block per (row, j)
__global__ void k_hc_dot(const float* x, const float* fn, float* m, float* ss) {
    __shared__ float red[32];
    int i = blockIdx.y, j = blockIdx.x;   // j in 0..24 (24 = sum of squares)
    const float4* xr = (const float4*)(x + (size_t)i * HC * D);
    const float4* fr = (const float4*)(fn + (size_t)(j < NMIX ? j : 0) * HC * D);
    float s = 0.f;
    for (int k = threadIdx.x; k < HC * D / 4; k += blockDim.x) {
        float4 a = xr[k];
        float4 b = j < NMIX ? fr[k] : a;
        s += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    s = warp_sum(s);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
        t = warp_sum(t);
        if (threadIdx.x == 0) {
            if (j < NMIX) m[(size_t)i * NMIX + j] = t;
            else ss[i] = t;
        }
    }
}
// sum of squares per row (prefill path, the dots come from a GEMM)
__global__ void k_sumsq(const float* x, int K, float* ss) {
    __shared__ float red[32];
    const float4* xr = (const float4*)(x + (size_t)blockIdx.x * K);
    float s = 0.f;
    for (int k = threadIdx.x; k < K / 4; k += blockDim.x) {
        float4 a = xr[k];
        s += a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
    }
    s = warp_sum(s);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
        t = warp_sum(t);
        if (threadIdx.x == 0) ss[blockIdx.x] = t;
    }
}
// pre [4], post [4], comb [4][4] (comb[j][k]: from stream j to stream k), as hc_split_sinkhorn; one warp per row,
// lane j < 16 holds comb entry (a, b) = (j / 4, j % 4); row sums over lanes 4a..4a+3, column sums over b, b+4, ..
__global__ void k_hc_finish(const float* m, const float* ss, const float* scale, const float* base, int n, float* pre,
                            float* post, float* comb) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (i >= n) return;
    float r = rsqrtf(ss[i] / (float)(HC * D) + EPS);
    const float* mi = m + (size_t)i * NMIX;
    if (lane < HC) {
        pre[i * HC + lane] = sigmoidf_(mi[lane] * r * scale[0] + base[lane]) + HC_EPS;
        post[i * HC + lane] = 2.f * sigmoidf_(mi[HC + lane] * r * scale[1] + base[HC + lane]);
    }
    const int j = lane & 15;
    float c = mi[2 * HC + j] * r * scale[2] + base[2 * HC + j];
    // softmax over b within row a (lanes 4a .. 4a + 3)
    float mx = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    float e = expf(c - mx);
    float se = e + __shfl_xor_sync(0xffffffffu, e, 1);
    se += __shfl_xor_sync(0xffffffffu, se, 2);
    c = e / se + HC_EPS;
    auto rowsum = [&](float v) {
        float t = v + __shfl_xor_sync(0xffffffffu, v, 1);
        return t + __shfl_xor_sync(0xffffffffu, t, 2);
    };
    auto colsum = [&](float v) {   // over a: lanes j, j ^ 4, j ^ 8, j ^ 12 (same b), fixed order a = 0..3
        float v0 = __shfl_sync(0xffffffffu, v, (j & 3));
        float v1 = __shfl_sync(0xffffffffu, v, (j & 3) + 4);
        float v2 = __shfl_sync(0xffffffffu, v, (j & 3) + 8);
        float v3 = __shfl_sync(0xffffffffu, v, (j & 3) + 12);
        return v0 + v1 + v2 + v3;
    };
    for (int it = 0; it < HC_ITERS; it++) {
        if (it > 0) {
            // row sum in the reference order c0 + c1 + c2 + c3
            int a4 = j & ~3;
            float r0 = __shfl_sync(0xffffffffu, c, a4), r1 = __shfl_sync(0xffffffffu, c, a4 + 1),
                  r2 = __shfl_sync(0xffffffffu, c, a4 + 2), r3 = __shfl_sync(0xffffffffu, c, a4 + 3);
            c = c / (r0 + r1 + r2 + r3 + HC_EPS);
        }
        c = c / (colsum(c) + HC_EPS);
    }
    (void)rowsum;
    if (lane < 16) comb[(size_t)i * 16 + j] = c;
}
// x[i] = sum_j pre[i][j] h[i][j]
__global__ void k_hc_pre(const float* h, const float* pre, int n, float* x) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)n * D) return;
    int i = (int)(t / D), d = (int)(t % D);
    const float* hi = h + (size_t)i * HC * D + d;
    const float* p = pre + i * HC;
    x[t] = p[0] * hi[0] + p[1] * hi[D] + p[2] * hi[2 * D] + p[3] * hi[3 * D];
}

// fused hc_pre + rms: x[i] = rms(sum_j pre[i][j] h[i][j]) * w; one block (1024 threads) per row (round 3)
__global__ void __launch_bounds__(1024) k_hc_pre_rms(const float* h, const float* pre, const float* w, int n, float* x) {
    __shared__ float red[32];
    const int i = blockIdx.x;
    const float* hi = h + (size_t)i * HC * D;
    const float* p = pre + i * HC;
    float p0 = p[0], p1 = p[1], p2 = p[2], p3 = p[3];
    float v[5];
    float ss = 0.f;
#pragma unroll
    for (int u = 0; u < 5; u++) {
        int d = threadIdx.x + 1024 * u;
        v[u] = p0 * hi[d] + p1 * hi[D + d] + p2 * hi[2 * D + d] + p3 * hi[3 * D + d];
        ss += v[u] * v[u];
    }
    ss = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = warp_sum(red[threadIdx.x]);
        if (threadIdx.x == 0) red[0] = rsqrtf(t / (float)D + EPS);
    }
    __syncthreads();
    float r = red[0];
#pragma unroll
    for (int u = 0; u < 5; u++) {
        int d = threadIdx.x + 1024 * u;
        x[(size_t)i * D + d] = v[u] * r * w[d];
    }
}

// h[i][k] = post[i][k] out[i] + sum_j comb[i][j][k] h[i][j]   (in place)
__global__ void k_hc_post(float* h, const float* out, const float* post, const float* comb, int n) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)n * D) return;
    int i = (int)(t / D), d = (int)(t % D);
    float* hi = h + (size_t)i * HC * D + d;
    float r[4] = {hi[0], hi[D], hi[2 * D], hi[3 * D]};
    float o = out[t];
    const float* c = comb + (size_t)i * 16;
    for (int k = 0; k < 4; k++)
        hi[(size_t)k * D] = post[i * HC + k] * o + (c[0 * 4 + k] * r[0] + c[1 * 4 + k] * r[1] + c[2 * 4 + k] * r[2] +
                                                    c[3 * 4 + k] * r[3]);
}

// R10 fused mHC boundary for one decode row (bit-identical to k_hc_post + k_hc_dot + k_hc_finish + k_hc_pre_rms: every
// output uses the same arithmetic and the same reduction order). Grid NMIX + 2 blocks of 1024 threads:
//   POST: h = post * out + comb^T h (elementwise, all blocks), then a grid barrier (all 26 blocks are co-resident);
//   blocks 0 .. NMIX: the mix dots of h with fn rows (threads < 256, k_hc_dot's order; block NMIX = sum of squares),
//   the last of them to finish runs the Sinkhorn finish (k_hc_finish's warp) -> pre_o, post_o, comb_o;
//   block NMIX + 1: x = rms(pre_x . h) * w (k_hc_pre_rms's order; pre_x is an earlier pre, independent of the mixes).
// bar: 3 counters of this card (zero at rest). h is read with ld.cg after the barrier (other blocks wrote it).
template <bool POST>
__global__ void __launch_bounds__(1024) k_hc_fused(float* h, const float* out, const float* post_in, const float* comb_in,
                                                   const float* fn, const float* scale, const float* base, float* mix,
                                                   float* ssq, float* pre_o, float* post_o, float* comb_o,
                                                   const float* pre_x, const float* w, float* x, unsigned* bar,
                                                   bf16* xq = nullptr) {
    __shared__ float red[32];
    __shared__ int last;
    const int b = blockIdx.x, tid = threadIdx.x;
    if (POST) {
        const int t = b * 1024 + tid;
        if (t < D) {
            float* hi = h + t;
            float r[4] = {hi[0], hi[D], hi[2 * D], hi[3 * D]};
            float o = out[t];
            for (int k = 0; k < 4; k++)
                hi[(size_t)k * D] = post_in[k] * o + (comb_in[0 * 4 + k] * r[0] + comb_in[1 * 4 + k] * r[1] +
                                                      comb_in[2 * 4 + k] * r[2] + comb_in[3 * 4 + k] * r[3]);
        }
        __syncthreads();
        if (tid == 0) {
            volatile unsigned* gen = bar + 1;
            const unsigned g0 = *gen;
            __threadfence();
            if (atomicAdd(bar, 1u) == gridDim.x - 1) {
                bar[0] = 0;
                __threadfence();
                atomicAdd(bar + 1, 1u);
            } else {
                while (*gen == g0) __nanosleep(32);
            }
            __threadfence();
        }
        __syncthreads();
    }
    if (b == NMIX + 1) {   // pre_rms (k_hc_pre_rms, row 0)
        float p0 = pre_x[0], p1 = pre_x[1], p2 = pre_x[2], p3 = pre_x[3];
        float v[5];
        float ss = 0.f;
#pragma unroll
        for (int u = 0; u < 5; u++) {
            int d = tid + 1024 * u;
            v[u] = p0 * __ldcg(h + d) + p1 * __ldcg(h + D + d) + p2 * __ldcg(h + 2 * D + d) + p3 * __ldcg(h + 3 * D + d);
            ss += v[u] * v[u];
        }
        ss = warp_sum(ss);
        if ((tid & 31) == 0) red[tid >> 5] = ss;
        __syncthreads();
        if (tid < 32) {
            float t = warp_sum(red[tid]);
            if (tid == 0) red[0] = rsqrtf(t / (float)D + EPS);
        }
        __syncthreads();
        float r = red[0];
#pragma unroll
        for (int u = 0; u < 5; u++) {
            int d = tid + 1024 * u;
            const float y = v[u] * r * w[d];
            x[d] = y;
            if (xq) {   // R10: q_fp8 of x (a warp holds 32 consecutive d: k_q_fp8's blocks), bf16 (exact)
                float am = fmaxf(warp_max(fabsf(y)), 1e-4f);
                float s8 = pow2ceil_f(am * (1.f / 448.f));
                xq[d] = __float2bfloat16(e4m3_rne(fminf(fmaxf(y / s8, -448.f), 448.f)) * s8);
            }
        }
        return;
    }
    {   // dots (k_hc_dot, row 0, mix j = b; j = NMIX: sum of squares), 256 threads
        const int j = b;
        const float4* xr = (const float4*)h;
        const float4* fr = (const float4*)(fn + (size_t)(j < NMIX ? j : 0) * HC * D);
        float s = 0.f;
        if (tid < 256) {
            for (int k = tid; k < HC * D / 4; k += 256) {
                float4 a = __ldcg(xr + k);
                float4 bb = j < NMIX ? fr[k] : a;
                s += a.x * bb.x + a.y * bb.y + a.z * bb.z + a.w * bb.w;
            }
            s = warp_sum(s);
            if ((tid & 31) == 0) red[tid >> 5] = s;
        }
        __syncthreads();
        if (tid < 32) {
            float t = tid < 8 ? red[tid] : 0.f;
            t = warp_sum(t);
            if (tid == 0) {
                if (j < NMIX) mix[j] = t;
                else ssq[0] = t;
                __threadfence();
                last = atomicAdd(bar + 2, 1u) == NMIX;
            }
        }
        __syncthreads();
        if (!last) return;
    }
    if (tid >= 32) return;
    if (tid == 0) bar[2] = 0;
    __threadfence();
    {   // finish (k_hc_finish, row 0): lane = tid
        const int lane = tid;
        float r = rsqrtf(__ldcg(ssq) / (float)(HC * D) + EPS);
        if (lane < HC) {
            pre_o[lane] = sigmoidf_(__ldcg(mix + lane) * r * scale[0] + base[lane]) + HC_EPS;
            post_o[lane] = 2.f * sigmoidf_(__ldcg(mix + HC + lane) * r * scale[1] + base[HC + lane]);
        }
        const int j = lane & 15;
        float c = __ldcg(mix + 2 * HC + j) * r * scale[2] + base[2 * HC + j];
        float mx = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        float e = expf(c - mx);
        float se = e + __shfl_xor_sync(0xffffffffu, e, 1);
        se += __shfl_xor_sync(0xffffffffu, se, 2);
        c = e / se + HC_EPS;
        auto colsum = [&](float v) {
            float v0 = __shfl_sync(0xffffffffu, v, (j & 3));
            float v1 = __shfl_sync(0xffffffffu, v, (j & 3) + 4);
            float v2 = __shfl_sync(0xffffffffu, v, (j & 3) + 8);
            float v3 = __shfl_sync(0xffffffffu, v, (j & 3) + 12);
            return v0 + v1 + v2 + v3;
        };
        for (int it = 0; it < HC_ITERS; it++) {
            if (it > 0) {
                int a4 = j & ~3;
                float r0 = __shfl_sync(0xffffffffu, c, a4), r1 = __shfl_sync(0xffffffffu, c, a4 + 1),
                      r2 = __shfl_sync(0xffffffffu, c, a4 + 2), r3 = __shfl_sync(0xffffffffu, c, a4 + 3);
                c = c / (r0 + r1 + r2 + r3 + HC_EPS);
            }
            c = c / (colsum(c) + HC_EPS);
        }
        if (lane < 16) comb_o[j] = c;
    }
}

// R20 [dsv4-r20-hcfr]: k_hc_fused for n rows (2 .. MAXR, verify passes): the same 26 blocks, each looping over the
// rows; every output of row i uses the arithmetic and reduction order of the one-row kernel (= k_hc_post + k_hc_dot +
// k_hc_finish + k_hc_pre_rms of that row): bit-identical per row. The finish runs one warp per row in the last block.
template <bool POST>
__global__ void __launch_bounds__(1024) k_hc_fused_r(float* h, const float* out, const float* post_in, const float* comb_in,
                                                     const float* fn, const float* scale, const float* base, float* mix,
                                                     float* ssq, float* pre_o, float* post_o, float* comb_o,
                                                     const float* pre_x, const float* w, float* x, unsigned* bar, int n,
                                                     bf16* xq = nullptr) {
    __shared__ float red[32];
    __shared__ int last;
    const int b = blockIdx.x, tid = threadIdx.x;
    if (POST) {
        for (int t = b * 1024 + tid; t < n * D; t += gridDim.x * 1024) {
            const int i = t / D, d = t % D;
            float* hi = h + (size_t)i * HC * D + d;
            const float* pi = post_in + i * HC;
            const float* ci = comb_in + i * 16;
            float r[4] = {hi[0], hi[D], hi[2 * D], hi[3 * D]};
            float o = out[t];
            for (int k = 0; k < 4; k++)
                hi[(size_t)k * D] = pi[k] * o + (ci[0 * 4 + k] * r[0] + ci[1 * 4 + k] * r[1] + ci[2 * 4 + k] * r[2] +
                                                 ci[3 * 4 + k] * r[3]);
        }
        __syncthreads();
        if (tid == 0) {
            volatile unsigned* gen = bar + 1;
            const unsigned g0 = *gen;
            __threadfence();
            if (atomicAdd(bar, 1u) == gridDim.x - 1) {
                bar[0] = 0;
                __threadfence();
                atomicAdd(bar + 1, 1u);
            } else {
                while (*gen == g0) __nanosleep(32);
            }
            __threadfence();
        }
        __syncthreads();
    }
    if (b == NMIX + 1) {   // pre_rms (k_hc_pre_rms), row by row
        for (int i = 0; i < n; i++) {
            const float* hi = h + (size_t)i * HC * D;
            const float* px = pre_x + i * HC;
            float p0 = px[0], p1 = px[1], p2 = px[2], p3 = px[3];
            float v[5];
            float ss = 0.f;
#pragma unroll
            for (int u = 0; u < 5; u++) {
                int d = tid + 1024 * u;
                v[u] = p0 * __ldcg(hi + d) + p1 * __ldcg(hi + D + d) + p2 * __ldcg(hi + 2 * D + d) + p3 * __ldcg(hi + 3 * D + d);
                ss += v[u] * v[u];
            }
            ss = warp_sum(ss);
            __syncthreads();   // (red reused per row)
            if ((tid & 31) == 0) red[tid >> 5] = ss;
            __syncthreads();
            if (tid < 32) {
                float t = warp_sum(red[tid]);
                if (tid == 0) red[0] = rsqrtf(t / (float)D + EPS);
            }
            __syncthreads();
            float r = red[0];
#pragma unroll
            for (int u = 0; u < 5; u++) {
                int d = tid + 1024 * u;
                const float y = v[u] * r * w[d];
                x[(size_t)i * D + d] = y;
                if (xq) {
                    float am = fmaxf(warp_max(fabsf(y)), 1e-4f);
                    float s8 = pow2ceil_f(am * (1.f / 448.f));
                    xq[(size_t)i * D + d] = __float2bfloat16(e4m3_rne(fminf(fmaxf(y / s8, -448.f), 448.f)) * s8);
                }
            }
        }
        return;
    }
    {   // dots (k_hc_dot's 256-thread order), mix j = b of every row (j = NMIX: sums of squares)
        const int j = b;
        const float4* fr = (const float4*)(fn + (size_t)(j < NMIX ? j : 0) * HC * D);
        for (int i = 0; i < n; i++) {
            const float4* xr = (const float4*)(h + (size_t)i * HC * D);
            float s = 0.f;
            if (tid < 256) {
                for (int k = tid; k < HC * D / 4; k += 256) {
                    float4 a = __ldcg(xr + k);
                    float4 bb = j < NMIX ? fr[k] : a;
                    s += a.x * bb.x + a.y * bb.y + a.z * bb.z + a.w * bb.w;
                }
                s = warp_sum(s);
            }
            __syncthreads();   // (red reused per row)
            if (tid < 256 && (tid & 31) == 0) red[tid >> 5] = s;
            __syncthreads();
            if (tid < 32) {
                float t = tid < 8 ? red[tid] : 0.f;
                t = warp_sum(t);
                if (tid == 0) {
                    if (j < NMIX) mix[(size_t)i * NMIX + j] = t;
                    else ssq[i] = t;
                }
            }
        }
        if (tid == 0) {
            __threadfence();
            last = atomicAdd(bar + 2, 1u) == NMIX;
        }
        __syncthreads();
        if (!last) return;
    }
    const int i = tid >> 5;
    if (tid == 0) bar[2] = 0;
    __threadfence();
    if (i >= n) return;
    {   // finish (k_hc_finish), warp i = row i
        const int lane = tid & 31;
        const float* mi = mix + (size_t)i * NMIX;
        float r = rsqrtf(__ldcg(ssq + i) / (float)(HC * D) + EPS);
        if (lane < HC) {
            pre_o[i * HC + lane] = sigmoidf_(__ldcg(mi + lane) * r * scale[0] + base[lane]) + HC_EPS;
            post_o[i * HC + lane] = 2.f * sigmoidf_(__ldcg(mi + HC + lane) * r * scale[1] + base[HC + lane]);
        }
        const int j = lane & 15;
        float c = __ldcg(mi + 2 * HC + j) * r * scale[2] + base[2 * HC + j];
        float mx = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        float e = expf(c - mx);
        float se = e + __shfl_xor_sync(0xffffffffu, e, 1);
        se += __shfl_xor_sync(0xffffffffu, se, 2);
        c = e / se + HC_EPS;
        auto colsum = [&](float v) {
            float v0 = __shfl_sync(0xffffffffu, v, (j & 3));
            float v1 = __shfl_sync(0xffffffffu, v, (j & 3) + 4);
            float v2 = __shfl_sync(0xffffffffu, v, (j & 3) + 8);
            float v3 = __shfl_sync(0xffffffffu, v, (j & 3) + 12);
            return v0 + v1 + v2 + v3;
        };
        for (int it = 0; it < HC_ITERS; it++) {
            if (it > 0) {
                int a4 = j & ~3;
                float r0 = __shfl_sync(0xffffffffu, c, a4), r1 = __shfl_sync(0xffffffffu, c, a4 + 1),
                      r2 = __shfl_sync(0xffffffffu, c, a4 + 2), r3 = __shfl_sync(0xffffffffu, c, a4 + 3);
                c = c / (r0 + r1 + r2 + r3 + HC_EPS);
            }
            c = c / (colsum(c) + HC_EPS);
        }
        if (lane < 16) comb_o[(size_t)i * 16 + j] = c;
    }
}

// [dsv4-dec2-hcfr2] k_hc_fused_r4: 256-thread blocks (two earlier layouts were slower: test history in the plan). Block j < NMIX + 1 = mix j for every row (k_hc_dot's block):
// thread t's 20 fn float4 are loaded at once before the rows (the old kernels' loop chained ~20 DRAM latencies), then per
// row its 20 h float4 at once and the same ordered sum (s += ...), warp sums, warp i reduces row i's 8 warp values with
// lanes 0..7 (k_hc_dot's final step). pre_rms: one block per row emulating the 1024-thread block (virtual thread
// q 256 + tid, q < 4; a virtual warp = 32 consecutive d as before). The last dot block runs the finish (warp i = row i).
// Bit-identical to k_hc_fused_r / k_hc_fused (n = 1); grid NMIX + 1 + n blocks (co-resident for the POST barrier).
inline unsigned hcfr4_grid(int n) { return (unsigned)(NMIX + 1 + n); }
template <bool POST>
__global__ void __launch_bounds__(256) k_hc_fused_r4(float* h, const float* out, const float* post_in, const float* comb_in,
                                                     const float* fn, const float* scale, const float* base, float* mix,
                                                     float* ssq, float* pre_o, float* post_o, float* comb_o,
                                                     const float* pre_x, const float* w, float* x, unsigned* bar, int n,
                                                     bf16* xq = nullptr) {
    __shared__ float red[MAXR * 8];
    __shared__ int last;
    const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
    constexpr int NK = HC * D / 4 / 256;   // 20 float4 per thread
    float4 f[NK];
    if (!POST && b < NMIX) {   // (POST: h changes first; fn does not, but its loads would hold registers across the barrier)
        const float4* fr = (const float4*)(fn + (size_t)b * HC * D);
#pragma unroll
        for (int m = 0; m < NK; m++) f[m] = fr[tid + 256 * m];
    }
    if (POST) {
        for (int t = b * 256 + tid; t < n * D; t += gridDim.x * 256) {
            const int i = t / D, d = t % D;
            float* hi = h + (size_t)i * HC * D + d;
            const float* pi = post_in + i * HC;
            const float* ci = comb_in + i * 16;
            float r[4] = {hi[0], hi[D], hi[2 * D], hi[3 * D]};
            float o = out[t];
            for (int k = 0; k < 4; k++)
                hi[(size_t)k * D] = pi[k] * o + (ci[0 * 4 + k] * r[0] + ci[1 * 4 + k] * r[1] + ci[2 * 4 + k] * r[2] +
                                                 ci[3 * 4 + k] * r[3]);
        }
        if (b < NMIX) {   // fn loads in flight across the barrier wait
            const float4* fr = (const float4*)(fn + (size_t)b * HC * D);
#pragma unroll
            for (int m = 0; m < NK; m++) f[m] = fr[tid + 256 * m];
        }
        __syncthreads();
        if (tid == 0) {
            volatile unsigned* gen = bar + 1;
            const unsigned g0 = *gen;
            __threadfence();
            if (atomicAdd(bar, 1u) == gridDim.x - 1) {
                bar[0] = 0;
                __threadfence();
                atomicAdd(bar + 1, 1u);
            } else {
                while (*gen == g0) __nanosleep(32);
            }
            __threadfence();
        }
        __syncthreads();
    }
    if (b >= NMIX + 1) {   // pre_rms (k_hc_pre_rms) of row b - NMIX - 1, 1024 virtual threads
        const int i = b - NMIX - 1;
        const float* hi = h + (size_t)i * HC * D;
        const float* px = pre_x + i * HC;
        float p0 = px[0], p1 = px[1], p2 = px[2], p3 = px[3];
        float v[4][5];
        float ssv[4];
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const int vt = q * 256 + tid;
            float ss = 0.f;
#pragma unroll
            for (int u = 0; u < 5; u++) {
                int d = vt + 1024 * u;
                v[q][u] = p0 * __ldcg(hi + d) + p1 * __ldcg(hi + D + d) + p2 * __ldcg(hi + 2 * D + d) + p3 * __ldcg(hi + 3 * D + d);
                ss += v[q][u] * v[q][u];
            }
            ssv[q] = ss;
        }
        __shared__ float red32[32];
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const float ss = warp_sum(ssv[q]);
            if (lane == 0) red32[(q * 256 + tid) >> 5] = ss;
        }
        __syncthreads();
        if (tid < 32) {
            float t = warp_sum(red32[tid]);
            if (tid == 0) red32[0] = rsqrtf(t / (float)D + EPS);
        }
        __syncthreads();
        float r = red32[0];
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const int vt = q * 256 + tid;
#pragma unroll
            for (int u = 0; u < 5; u++) {
                int d = vt + 1024 * u;
                const float y = v[q][u] * r * w[d];
                x[(size_t)i * D + d] = y;
                if (xq) {
                    float am = fmaxf(warp_max(fabsf(y)), 1e-4f);
                    float s8 = pow2ceil_f(am * (1.f / 448.f));
                    xq[(size_t)i * D + d] = __float2bfloat16(e4m3_rne(fminf(fmaxf(y / s8, -448.f), 448.f)) * s8);
                }
            }
        }
        return;
    }
    {   // dots: block j, every row (j = NMIX: the sums of squares)
        const int j = b;
        for (int i = 0; i < n; i++) {
            const float4* xr = (const float4*)(h + (size_t)i * HC * D);
            float4 a[NK];
#pragma unroll
            for (int m = 0; m < NK; m++) a[m] = __ldcg(xr + tid + 256 * m);
            float s = 0.f;
#pragma unroll
            for (int m = 0; m < NK; m++) {
                const float4 bb = j < NMIX ? f[m] : a[m];
                // k_hc_fused_r's compiled form, spelled out (FMUL x, FFMA y, z, w, FADD): no other contraction
                const float d = __fmaf_rn(a[m].w, bb.w, __fmaf_rn(a[m].z, bb.z, __fmaf_rn(a[m].y, bb.y, __fmul_rn(a[m].x, bb.x))));
                s = __fadd_rn(s, d);
            }
            s = warp_sum(s);
            if (lane == 0) red[i * 8 + wp] = s;
        }
        __syncthreads();
        if (wp < n) {   // warp i: row i's 8 warp sums (lanes 0..7)
            const int i = wp;
            float t = lane < 8 ? red[i * 8 + lane] : 0.f;
            t = warp_sum(t);
            if (lane == 0) {
                if (j < NMIX) mix[(size_t)i * NMIX + j] = t;
                else ssq[i] = t;
                __threadfence();
            }
        }
        __syncthreads();
        if (tid == 0) last = atomicAdd(bar + 2, 1u) == NMIX;
        __syncthreads();
        if (!last) return;
    }
    const int i = wp;
    if (tid == 0) bar[2] = 0;
    __threadfence();
    if (i >= n) return;
    {   // finish (k_hc_finish), warp i = row i
        const float* mi = mix + (size_t)i * NMIX;
        float r = rsqrtf(__ldcg(ssq + i) / (float)(HC * D) + EPS);
        if (lane < HC) {
            pre_o[i * HC + lane] = sigmoidf_(__ldcg(mi + lane) * r * scale[0] + base[lane]) + HC_EPS;
            post_o[i * HC + lane] = 2.f * sigmoidf_(__ldcg(mi + HC + lane) * r * scale[1] + base[HC + lane]);
        }
        const int jj = lane & 15;
        float c = __ldcg(mi + 2 * HC + jj) * r * scale[2] + base[2 * HC + jj];
        float mx = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        float e = expf(c - mx);
        float se = e + __shfl_xor_sync(0xffffffffu, e, 1);
        se += __shfl_xor_sync(0xffffffffu, se, 2);
        c = e / se + HC_EPS;
        auto colsum = [&](float v) {
            float v0 = __shfl_sync(0xffffffffu, v, (jj & 3));
            float v1 = __shfl_sync(0xffffffffu, v, (jj & 3) + 4);
            float v2 = __shfl_sync(0xffffffffu, v, (jj & 3) + 8);
            float v3 = __shfl_sync(0xffffffffu, v, (jj & 3) + 12);
            return v0 + v1 + v2 + v3;
        };
        for (int it = 0; it < HC_ITERS; it++) {
            if (it > 0) {
                int a4 = jj & ~3;
                float r0 = __shfl_sync(0xffffffffu, c, a4), r1 = __shfl_sync(0xffffffffu, c, a4 + 1),
                      r2 = __shfl_sync(0xffffffffu, c, a4 + 2), r3 = __shfl_sync(0xffffffffu, c, a4 + 3);
                c = c / (r0 + r1 + r2 + r3 + HC_EPS);
            }
            c = c / (colsum(c) + HC_EPS);
        }
        if (lane < 16) comb_o[(size_t)i * 16 + jj] = c;
    }
}

// ---------------------------------------------------------------- attention (latent MQA, K = V)

// q [n][64][512] fp32, window keys kvw [pos][512] bf16 (positions p - nw + 1 .. p), selected compressed keys
// ckv [Tc][512] bf16 at sel[i][0..nsel) (-1 = none). Block = (query, group of 8 heads, key split); 256 threads.
// nsplit == 1: final output o [n][64][512]; else partials to part (m, l per head, O [8][512]) for k_attn_comb.
// R10: q_fp8 of a pair (2t, 2t + 1) of a 512-dim head row: a 32-dim block = 16 consecutive threads (k_q_fp8's blocks)
__device__ __forceinline__ void q8_pair(float a, float b, bf16* out) {
    float am = fmaxf(fabsf(a), fabsf(b));
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    am = fmaxf(am, 1e-4f);
    const float s8 = pow2ceil_f(am * (1.f / 448.f));
    out[0] = __float2bfloat16(e4m3_rne(fminf(fmaxf(a / s8, -448.f), 448.f)) * s8);
    out[1] = __float2bfloat16(e4m3_rne(fminf(fmaxf(b / s8, -448.f), 448.f)) * s8);
}
constexpr int AT_KT = 32, AT_HG = 8;
constexpr int AT_SMEM = AT_HG * HD * 4 + AT_KT * HD * 2;
// [dsv4-dec2-attl] k_attn_o = the R10 kernel (DSV4_NOATTL=1, attn_kern() in dsv4.cu); k_attn below: same arithmetic and order, fewer
// shared-memory loads (q pairs as float2, P of 4 keys as float4): bit-identical
__global__ void __launch_bounds__(256) k_attn_o(const float* q, int pos0, const bf16* kvw, const bf16* ckv,
                                              const int* sel, int nsel, int ldsel, const float* sink, float* o,
                                              int nsplit, float* part, int ratio, int kend, int wlen, int hg0 = 0,
                                              int kmask = -1, const float* rcos = nullptr, const float* rsin = nullptr,
                                              bf16* oq = nullptr) {
    // R10 (rcos != null): q rope on load and the inverse rope on the output (nsplit == 1), the same expressions as
    // k_rope (bit-identical); the caller drops its two k_rope launches
    // hg0 (R8 attention split): this launch covers head groups hg0 .. hg0 + gridDim.y - 1; kmask: kvw is a ring of
    // kmask + 1 rows (a peer card's copy of the window), position p at row p & kmask (-1: plain)
    // ratio > 0: the row's selection holds at most (p + 1) / ratio valid entries (a prefix; review: a pass of
    // several rows must split each row exactly as a one-row decode step would). kend >= 0 (DSpark block): every
    // row sees the keys kend - wlen + 1 .. kend of kvw (window + the whole block, not causal inside the block).
    extern __shared__ __align__(16) char at_sm[];   // dynamic: AT_SMEM bytes
    float (*qs)[HD] = (float (*)[HD])at_sm;
    bf16 (*ks)[HD] = (bf16 (*)[HD])(at_sm + AT_HG * HD * 4);
    __shared__ float ps[AT_HG][AT_KT];
    __shared__ float alpha_s[AT_HG];
    const int i = blockIdx.x, hg = blockIdx.y + hg0, sp = blockIdx.z;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int p = pos0 + i;
    const int pe = kend >= 0 ? kend : p;
    const int nw = min(kend >= 0 ? wlen : WIN, pe + 1);
    const int* si = sel ? sel + (size_t)i * ldsel : nullptr;
    const int nsr = si ? (ratio > 0 ? min(nsel, (p + 1) / ratio) : nsel) : 0;
    const int nk = nw + nsr;
    const int ntile = (nk + AT_KT - 1) / AT_KT;
    // the row's own split (what a one-row launch would use); extra splits of the grid write empty partials
    const int nsp = nsplit == 1 ? 1 : min(nsplit, max(1, min(32, ntile)));
    const int t0 = sp < nsp ? (int)((long long)ntile * sp / nsp) : 0;
    const int t1 = sp < nsp ? (int)((long long)ntile * (sp + 1) / nsp) : 0;
    for (int k = t; k < AT_HG * HD; k += 256) {
        const float* qr = q + ((size_t)i * NH + hg * AT_HG) * HD;
        const int dd = k % HD;
        float v = qr[k];
        if (rcos && dd >= HD - RD) {
            const int jj = (dd - (HD - RD)) >> 1, k0 = k & ~1;
            const float c = rcos[(size_t)p * 32 + jj], sn = rsin[(size_t)p * 32 + jj];
            const float a = qr[k0], b = qr[k0 + 1];
            v = (dd & 1) ? a * sn + b * c : a * c - b * sn;
        }
        qs[k / HD][dd] = v;
    }
    const float scale = 0.044194173824159216f;   // 512^-0.5
    float m_run = -INFINITY, l_run = 0.f;        // per head (warp's head) kept by lane 0 logic; all lanes same
    float acc[AT_HG][2];
#pragma unroll
    for (int h = 0; h < AT_HG; h++) acc[h][0] = acc[h][1] = 0.f;
    for (int tile = t0; tile < t1; tile++) {
        __syncthreads();
        // load 32 keys: thread t -> key t / 8, 64 dims (8 x uint4)
        {
            int kk = t >> 3, part8 = t & 7;
            int j = tile * AT_KT + kk;
            const bf16* src = nullptr;
            if (j < nw) src = kvw + (size_t)((pe - nw + 1 + j) & kmask) * HD;
            else if (j < nk) {
                int c = si[j - nw];
                if (c >= 0) src = ckv + (size_t)c * HD;
            }
            uint4* dst = (uint4*)&ks[kk][part8 * 64];
            if (src) {
                const uint4* s4 = (const uint4*)(src + part8 * 64);
#pragma unroll
                for (int u = 0; u < 8; u++) dst[u] = s4[u];
            } else {
#pragma unroll
                for (int u = 0; u < 8; u++) dst[u] = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();
        // scores: warp = head, lane = key; rotated dims (conflict-free)
        int j = tile * AT_KT + lane;
        bool valid = j < nk && (j < nw || si[j - nw] >= 0);
        float s = 0.f;
        for (int d0 = 0; d0 < HD; d0 += 2) {
            int d = (d0 + 2 * lane) & (HD - 1);
            float2 kv = __bfloat1622float2(*(const bf162*)&ks[lane][d]);
            s += qs[warp][d] * kv.x + qs[warp][d + 1] * kv.y;
        }
        s = valid ? s * scale : -INFINITY;
        float tm = warp_max(s);
        float mn = fmaxf(m_run, tm);
        float pv = mn == -INFINITY ? 0.f : expf(s - mn);
        float al = mn == -INFINITY ? 1.f : expf(m_run - mn);
        l_run = l_run * al + warp_sum(pv);
        m_run = mn;
        ps[warp][lane] = pv;
        if (lane == 0) alpha_s[warp] = al;
        __syncthreads();
        // O[h][2t, 2t+1] = alpha * O + sum_k P[h][k] K[k][2t..]
#pragma unroll
        for (int h = 0; h < AT_HG; h++) {
            acc[h][0] *= alpha_s[h];
            acc[h][1] *= alpha_s[h];
        }
        for (int k = 0; k < AT_KT; k++) {
            float2 kv = __bfloat1622float2(*(const bf162*)&ks[k][2 * t]);
#pragma unroll
            for (int h = 0; h < AT_HG; h++) {
                float pp = ps[h][k];
                acc[h][0] += pp * kv.x;
                acc[h][1] += pp * kv.y;
            }
        }
    }
    // share m, l of every head
    __shared__ float ms[AT_HG], ls[AT_HG];
    __syncthreads();
    if (lane == 0) {
        ms[warp] = m_run;
        ls[warp] = l_run;
    }
    __syncthreads();
    if (nsplit == 1) {
#pragma unroll
        for (int h = 0; h < AT_HG; h++) {
            int hh = hg * AT_HG + h;
            float den = ls[h] + expf(sink[hh] - ms[h]);
            float* op = o + ((size_t)i * NH + hh) * HD + 2 * t;
            float a = acc[h][0] / den, b = acc[h][1] / den;
            if (rcos && 2 * t >= HD - RD) {   // inverse rope (k_rope, inverse = 1)
                const int jj = t - (HD - RD) / 2;
                const float c = rcos[(size_t)p * 32 + jj], sn = -rsin[(size_t)p * 32 + jj];
                const float a2 = a * c - b * sn, b2 = a * sn + b * c;
                a = a2;
                b = b2;
            }
            op[0] = a;
            op[1] = b;
            if (oq) q8_pair(a, b, oq + ((size_t)i * NH + hh) * HD + 2 * t);
        }
    } else {
        // part layout per (i, head, split): [m, l, O[512]]
#pragma unroll
        for (int h = 0; h < AT_HG; h++) {
            int hh = hg * AT_HG + h;
            float* pp = part + (((size_t)i * NH + hh) * nsplit + sp) * (HD + 2);
            if (t == 0) {
                pp[0] = ms[h];
                pp[1] = ls[h];
            }
            pp[2 + 2 * t] = acc[h][0];
            pp[2 + 2 * t + 1] = acc[h][1];
        }
    }
}

__global__ void __launch_bounds__(256) k_attn(const float* q, int pos0, const bf16* kvw, const bf16* ckv,
                                              const int* sel, int nsel, int ldsel, const float* sink, float* o,
                                              int nsplit, float* part, int ratio, int kend, int wlen, int hg0 = 0,
                                              int kmask = -1, const float* rcos = nullptr, const float* rsin = nullptr,
                                              bf16* oq = nullptr) {
    // R10 (rcos != null): q rope on load and the inverse rope on the output (nsplit == 1), the same expressions as
    // k_rope (bit-identical); the caller drops its two k_rope launches
    // hg0 (R8 attention split): this launch covers head groups hg0 .. hg0 + gridDim.y - 1; kmask: kvw is a ring of
    // kmask + 1 rows (a peer card's copy of the window), position p at row p & kmask (-1: plain)
    // ratio > 0: the row's selection holds at most (p + 1) / ratio valid entries (a prefix; review: a pass of
    // several rows must split each row exactly as a one-row decode step would). kend >= 0 (DSpark block): every
    // row sees the keys kend - wlen + 1 .. kend of kvw (window + the whole block, not causal inside the block).
    extern __shared__ __align__(16) char at_sm[];   // dynamic: AT_SMEM bytes
    float (*qs)[HD] = (float (*)[HD])at_sm;
    bf16 (*ks)[HD] = (bf16 (*)[HD])(at_sm + AT_HG * HD * 4);
    __shared__ __align__(16) float ps[AT_HG][AT_KT];
    __shared__ float alpha_s[AT_HG];
    const int i = blockIdx.x, hg = blockIdx.y + hg0, sp = blockIdx.z;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int p = pos0 + i;
    const int pe = kend >= 0 ? kend : p;
    const int nw = min(kend >= 0 ? wlen : WIN, pe + 1);
    const int* si = sel ? sel + (size_t)i * ldsel : nullptr;
    const int nsr = si ? (ratio > 0 ? min(nsel, (p + 1) / ratio) : nsel) : 0;
    const int nk = nw + nsr;
    const int ntile = (nk + AT_KT - 1) / AT_KT;
    // the row's own split (what a one-row launch would use); extra splits of the grid write empty partials
    const int nsp = nsplit == 1 ? 1 : min(nsplit, max(1, min(32, ntile)));
    const int t0 = sp < nsp ? (int)((long long)ntile * sp / nsp) : 0;
    const int t1 = sp < nsp ? (int)((long long)ntile * (sp + 1) / nsp) : 0;
    for (int k = t; k < AT_HG * HD; k += 256) {
        const float* qr = q + ((size_t)i * NH + hg * AT_HG) * HD;
        const int dd = k % HD;
        float v = qr[k];
        if (rcos && dd >= HD - RD) {
            const int jj = (dd - (HD - RD)) >> 1, k0 = k & ~1;
            const float c = rcos[(size_t)p * 32 + jj], sn = rsin[(size_t)p * 32 + jj];
            const float a = qr[k0], b = qr[k0 + 1];
            v = (dd & 1) ? a * sn + b * c : a * c - b * sn;
        }
        qs[k / HD][dd] = v;
    }
    const float scale = 0.044194173824159216f;   // 512^-0.5
    float m_run = -INFINITY, l_run = 0.f;        // per head (warp's head) kept by lane 0 logic; all lanes same
    float acc[AT_HG][2];
#pragma unroll
    for (int h = 0; h < AT_HG; h++) acc[h][0] = acc[h][1] = 0.f;
    for (int tile = t0; tile < t1; tile++) {
        __syncthreads();
        // load 32 keys: thread t -> key t / 8, 64 dims (8 x uint4)
        {
            int kk = t >> 3, part8 = t & 7;
            int j = tile * AT_KT + kk;
            const bf16* src = nullptr;
            if (j < nw) src = kvw + (size_t)((pe - nw + 1 + j) & kmask) * HD;
            else if (j < nk) {
                int c = si[j - nw];
                if (c >= 0) src = ckv + (size_t)c * HD;
            }
            uint4* dst = (uint4*)&ks[kk][part8 * 64];
            if (src) {
                const uint4* s4 = (const uint4*)(src + part8 * 64);
#pragma unroll
                for (int u = 0; u < 8; u++) dst[u] = s4[u];
            } else {
#pragma unroll
                for (int u = 0; u < 8; u++) dst[u] = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();
        // scores: warp = head, lane = key; rotated dims (conflict-free)
        int j = tile * AT_KT + lane;
        bool valid = j < nk && (j < nw || si[j - nw] >= 0);
        float s = 0.f;
        for (int d0 = 0; d0 < HD; d0 += 2) {
            int d = (d0 + 2 * lane) & (HD - 1);
            float2 kv = __bfloat1622float2(*(const bf162*)&ks[lane][d]);
            const float2 qv = *(const float2*)&qs[warp][d];   // [dsv4-dec2-attl] one 8-byte load (was two)
            s += qv.x * kv.x + qv.y * kv.y;
        }
        s = valid ? s * scale : -INFINITY;
        float tm = warp_max(s);
        float mn = fmaxf(m_run, tm);
        float pv = mn == -INFINITY ? 0.f : expf(s - mn);
        float al = mn == -INFINITY ? 1.f : expf(m_run - mn);
        l_run = l_run * al + warp_sum(pv);
        m_run = mn;
        ps[warp][lane] = pv;
        if (lane == 0) alpha_s[warp] = al;
        __syncthreads();
        // O[h][2t, 2t+1] = alpha * O + sum_k P[h][k] K[k][2t..]
#pragma unroll
        for (int h = 0; h < AT_HG; h++) {
            acc[h][0] *= alpha_s[h];
            acc[h][1] *= alpha_s[h];
        }
        for (int k0 = 0; k0 < AT_KT; k0 += 4) {   // [dsv4-dec2-attl] P of 4 keys per 16-byte load (was one per key)
            float2 kv[4];
#pragma unroll
            for (int u = 0; u < 4; u++) kv[u] = __bfloat1622float2(*(const bf162*)&ks[k0 + u][2 * t]);
#pragma unroll
            for (int h = 0; h < AT_HG; h++) {
                const float4 p4 = *(const float4*)&ps[h][k0];
                const float pp[4] = {p4.x, p4.y, p4.z, p4.w};
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    acc[h][0] += pp[u] * kv[u].x;
                    acc[h][1] += pp[u] * kv[u].y;
                }
            }
        }
    }
    // share m, l of every head
    __shared__ float ms[AT_HG], ls[AT_HG];
    __syncthreads();
    if (lane == 0) {
        ms[warp] = m_run;
        ls[warp] = l_run;
    }
    __syncthreads();
    if (nsplit == 1) {
#pragma unroll
        for (int h = 0; h < AT_HG; h++) {
            int hh = hg * AT_HG + h;
            float den = ls[h] + expf(sink[hh] - ms[h]);
            float* op = o + ((size_t)i * NH + hh) * HD + 2 * t;
            float a = acc[h][0] / den, b = acc[h][1] / den;
            if (rcos && 2 * t >= HD - RD) {   // inverse rope (k_rope, inverse = 1)
                const int jj = t - (HD - RD) / 2;
                const float c = rcos[(size_t)p * 32 + jj], sn = -rsin[(size_t)p * 32 + jj];
                const float a2 = a * c - b * sn, b2 = a * sn + b * c;
                a = a2;
                b = b2;
            }
            op[0] = a;
            op[1] = b;
            if (oq) q8_pair(a, b, oq + ((size_t)i * NH + hh) * HD + 2 * t);
        }
    } else {
        // part layout per (i, head, split): [m, l, O[512]]
#pragma unroll
        for (int h = 0; h < AT_HG; h++) {
            int hh = hg * AT_HG + h;
            float* pp = part + (((size_t)i * NH + hh) * nsplit + sp) * (HD + 2);
            if (t == 0) {
                pp[0] = ms[h];
                pp[1] = ls[h];
            }
            pp[2 + 2 * t] = acc[h][0];
            pp[2 + 2 * t + 1] = acc[h][1];
        }
    }
}

// [dsv4-dec3-atts] k_attn_s: k_attn with no static shared memory (P / alpha / m / l in the dead q region, q staged per
// tile), so two blocks fit an SM (k_attn: 48 KB dynamic + 1.1 KB static + 1 KB reserve = one block); same arithmetic
__global__ void __launch_bounds__(256, 2) k_attn_s(const float* q, int pos0, const bf16* kvw, const bf16* ckv,
                                              const int* sel, int nsel, int ldsel, const float* sink, float* o,
                                              int nsplit, float* part, int ratio, int kend, int wlen, int hg0 = 0,
                                              int kmask = -1, const float* rcos = nullptr, const float* rsin = nullptr,
                                              bf16* oq = nullptr) {
    // R10 (rcos != null): q rope on load and the inverse rope on the output (nsplit == 1), the same expressions as
    // k_rope (bit-identical); the caller drops its two k_rope launches
    // hg0 (R8 attention split): this launch covers head groups hg0 .. hg0 + gridDim.y - 1; kmask: kvw is a ring of
    // kmask + 1 rows (a peer card's copy of the window), position p at row p & kmask (-1: plain)
    // ratio > 0: the row's selection holds at most (p + 1) / ratio valid entries (a prefix; review: a pass of
    // several rows must split each row exactly as a one-row decode step would). kend >= 0 (DSpark block): every
    // row sees the keys kend - wlen + 1 .. kend of kvw (window + the whole block, not causal inside the block).
    extern __shared__ __align__(16) char at_sm[];   // dynamic: AT_SMEM bytes
    float (*qs)[HD] = (float (*)[HD])at_sm;
    bf16 (*ks)[HD] = (bf16 (*)[HD])(at_sm + AT_HG * HD * 4);
    // [dsv4-dec3-atts]: P, alpha, m, l live in the q region (dead after a tile's scores): no static shared memory
    float (*ps)[AT_KT] = (float (*)[AT_KT])at_sm;
    float* alpha_s = (float*)at_sm + AT_HG * AT_KT;
    const int i = blockIdx.x, hg = blockIdx.y + hg0, sp = blockIdx.z;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int p = pos0 + i;
    const int pe = kend >= 0 ? kend : p;
    const int nw = min(kend >= 0 ? wlen : WIN, pe + 1);
    const int* si = sel ? sel + (size_t)i * ldsel : nullptr;
    const int nsr = si ? (ratio > 0 ? min(nsel, (p + 1) / ratio) : nsel) : 0;
    const int nk = nw + nsr;
    const int ntile = (nk + AT_KT - 1) / AT_KT;
    // the row's own split (what a one-row launch would use); extra splits of the grid write empty partials
    const int nsp = nsplit == 1 ? 1 : min(nsplit, max(1, min(32, ntile)));
    const int t0 = sp < nsp ? (int)((long long)ntile * sp / nsp) : 0;
    const int t1 = sp < nsp ? (int)((long long)ntile * (sp + 1) / nsp) : 0;
    const float scale = 0.044194173824159216f;   // 512^-0.5
    float m_run = -INFINITY, l_run = 0.f;        // per head (warp's head) kept by lane 0 logic; all lanes same
    float acc[AT_HG][2];
#pragma unroll
    for (int h = 0; h < AT_HG; h++) acc[h][0] = acc[h][1] = 0.f;
    for (int tile = t0; tile < t1; tile++) {
        __syncthreads();
        // q staged per tile (decode: one tile per block; the same values every time)
        for (int k = t; k < AT_HG * HD; k += 256) {
            const float* qr = q + ((size_t)i * NH + hg * AT_HG) * HD;
            const int dd = k % HD;
            float v = qr[k];
            if (rcos && dd >= HD - RD) {
                const int jj = (dd - (HD - RD)) >> 1, k0 = k & ~1;
                const float c = rcos[(size_t)p * 32 + jj], sn = rsin[(size_t)p * 32 + jj];
                const float a = qr[k0], b = qr[k0 + 1];
                v = (dd & 1) ? a * sn + b * c : a * c - b * sn;
            }
            qs[k / HD][dd] = v;
        }
        // load 32 keys: thread t -> key t / 8, 64 dims (8 x uint4)
        {
            int kk = t >> 3, part8 = t & 7;
            int j = tile * AT_KT + kk;
            const bf16* src = nullptr;
            if (j < nw) src = kvw + (size_t)((pe - nw + 1 + j) & kmask) * HD;
            else if (j < nk) {
                int c = si[j - nw];
                if (c >= 0) src = ckv + (size_t)c * HD;
            }
            uint4* dst = (uint4*)&ks[kk][part8 * 64];
            if (src) {
                const uint4* s4 = (const uint4*)(src + part8 * 64);
#pragma unroll
                for (int u = 0; u < 8; u++) dst[u] = s4[u];
            } else {
#pragma unroll
                for (int u = 0; u < 8; u++) dst[u] = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();
        // scores: warp = head, lane = key; rotated dims (conflict-free)
        int j = tile * AT_KT + lane;
        bool valid = j < nk && (j < nw || si[j - nw] >= 0);
        float s = 0.f;
        for (int d0 = 0; d0 < HD; d0 += 2) {
            int d = (d0 + 2 * lane) & (HD - 1);
            float2 kv = __bfloat1622float2(*(const bf162*)&ks[lane][d]);
            const float2 qv = *(const float2*)&qs[warp][d];   // [dsv4-dec2-attl] one 8-byte load (was two)
            s += qv.x * kv.x + qv.y * kv.y;
        }
        s = valid ? s * scale : -INFINITY;
        __syncthreads();   // every warp's scores read q before P overwrites it
        float tm = warp_max(s);
        float mn = fmaxf(m_run, tm);
        float pv = mn == -INFINITY ? 0.f : expf(s - mn);
        float al = mn == -INFINITY ? 1.f : expf(m_run - mn);
        l_run = l_run * al + warp_sum(pv);
        m_run = mn;
        ps[warp][lane] = pv;
        if (lane == 0) alpha_s[warp] = al;
        __syncthreads();
        // O[h][2t, 2t+1] = alpha * O + sum_k P[h][k] K[k][2t..]
#pragma unroll
        for (int h = 0; h < AT_HG; h++) {
            acc[h][0] *= alpha_s[h];
            acc[h][1] *= alpha_s[h];
        }
        for (int k0 = 0; k0 < AT_KT; k0 += 4) {   // [dsv4-dec2-attl] P of 4 keys per 16-byte load (was one per key)
            float2 kv[4];
#pragma unroll
            for (int u = 0; u < 4; u++) kv[u] = __bfloat1622float2(*(const bf162*)&ks[k0 + u][2 * t]);
#pragma unroll
            for (int h = 0; h < AT_HG; h++) {
                const float4 p4 = *(const float4*)&ps[h][k0];
                const float pp[4] = {p4.x, p4.y, p4.z, p4.w};
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    acc[h][0] += pp[u] * kv[u].x;
                    acc[h][1] += pp[u] * kv[u].y;
                }
            }
        }
    }
    // share m, l of every head
    float* ms = (float*)at_sm + AT_HG * AT_KT + AT_HG;
    float* ls = ms + AT_HG;
    __syncthreads();
    if (lane == 0) {
        ms[warp] = m_run;
        ls[warp] = l_run;
    }
    __syncthreads();
    if (nsplit == 1) {
#pragma unroll
        for (int h = 0; h < AT_HG; h++) {
            int hh = hg * AT_HG + h;
            float den = ls[h] + expf(sink[hh] - ms[h]);
            float* op = o + ((size_t)i * NH + hh) * HD + 2 * t;
            float a = acc[h][0] / den, b = acc[h][1] / den;
            if (rcos && 2 * t >= HD - RD) {   // inverse rope (k_rope, inverse = 1)
                const int jj = t - (HD - RD) / 2;
                const float c = rcos[(size_t)p * 32 + jj], sn = -rsin[(size_t)p * 32 + jj];
                const float a2 = a * c - b * sn, b2 = a * sn + b * c;
                a = a2;
                b = b2;
            }
            op[0] = a;
            op[1] = b;
            if (oq) q8_pair(a, b, oq + ((size_t)i * NH + hh) * HD + 2 * t);
        }
    } else {
        // part layout per (i, head, split): [m, l, O[512]]
#pragma unroll
        for (int h = 0; h < AT_HG; h++) {
            int hh = hg * AT_HG + h;
            float* pp = part + (((size_t)i * NH + hh) * nsplit + sp) * (HD + 2);
            if (t == 0) {
                pp[0] = ms[h];
                pp[1] = ls[h];
            }
            pp[2 + 2 * t] = acc[h][0];
            pp[2 + 2 * t + 1] = acc[h][1];
        }
    }
}

// ---------------------------------------------------------------- prefill attention on tensor cores (MQA: the 64
// heads of one query are the M dimension). fp16 mma.sync m16n8k16, fp32 accumulate. Keys / values are FP8- or
// FP4-derived (exact in fp16); q and P are rounded to fp16 (2^-11 relative). Block = one query, 8 warps:
// S: warp (mt = w & 3, kh = w >> 2) -> heads 16 mt .. +15, keys 16 kh .. +15 of the 32-key tile;
// O: warp (mt, dh) -> heads 16 mt .. +15, dims 256 dh .. +255 (32 n8 tiles, 128 fp32 accumulators per thread).
constexpr int TK = 16;   // keys per tile (smem: Q 64 KB + K 16 KB + P 2 KB)
constexpr int TC_SMEM = NH * HD * 2 + TK * HD * 2 + NH * TK * 2;
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__global__ void __launch_bounds__(256, 1) k_attn_tc(const float* q, int pos0, const bf16* kvw, const bf16* ckv,
                                                    const int* sel, int nsel, int ldsel, const float* sink, float* o) {
    extern __shared__ __align__(16) char tsm[];
    __half* Qs = (__half*)tsm;                                  // [64][512]
    __half* Ks = Qs + NH * HD;                                  // [16][512]
    __half* Ps = Ks + TK * HD;                                  // [64][16]
    __shared__ float mrow[NH], lrow[NH];
    __shared__ float red[2][NH];                                // per key-half: tile max, then tile sum
    __shared__ float alpha_s[NH];
    const int i = blockIdx.x, t = threadIdx.x, w = t >> 5, lane = t & 31;
    const int p = pos0 + i;
    const int nw = min(WIN, p + 1);
    const int* si = sel ? sel + (size_t)i * ldsel : nullptr;
    const int nk = nw + (si ? nsel : 0);
    const int ntile = (nk + TK - 1) / TK;
    for (int k = t; k < NH * HD / 2; k += 256) {
        float2 v = ((const float2*)(q + (size_t)i * NH * HD))[k];
        ((__half2*)Qs)[k] = __floats2half2_rn(v.x, v.y);
    }
    if (t < NH) {
        mrow[t] = -INFINITY;
        lrow[t] = 0.f;
    }
    const float scale = 0.044194173824159216f;
    const int mt = w & 3, kh = w >> 2, dh = w >> 2;
    const int r0 = 16 * mt + (lane >> 2);                       // fragment rows r0, r0 + 8
    float acc[32][4];
#pragma unroll
    for (int a = 0; a < 32; a++) acc[a][0] = acc[a][1] = acc[a][2] = acc[a][3] = 0.f;
    for (int tile = 0; tile < ntile; tile++) {
        __syncthreads();
        {   // keys -> fp16 smem: thread t -> key t / 16, 32 dims
            int kk = t >> 4, part = t & 15;
            int j = tile * TK + kk;
            const bf16* src = nullptr;
            if (j < nw) src = kvw + (size_t)(p - nw + 1 + j) * HD;
            else if (j < nk) {
                int c = si[j - nw];
                if (c >= 0) src = ckv + (size_t)c * HD;
            }
            __half2* dst = (__half2*)(Ks + kk * HD + part * 32);
            if (src) {
                const bf162* s2 = (const bf162*)(src + part * 32);
#pragma unroll 8
                for (int u = 0; u < 16; u++) {
                    float2 f = __bfloat1622float2(s2[u]);
                    dst[u] = __floats2half2_rn(f.x, f.y);
                }
            } else {
#pragma unroll 8
                for (int u = 0; u < 16; u++) dst[u] = __floats2half2_rn(0.f, 0.f);
            }
        }
        __syncthreads();
        // S = Q K^T for heads 16 mt.., keys 8 kh .. 8 kh + 7
        float sc[1][4] = {{0, 0, 0, 0}};
        for (int ks = 0; ks < HD / 16; ks++) {
            int kc = ks * 16 + (lane & 3) * 2;
            uint32_t a[4];
            a[0] = *(const uint32_t*)(Qs + r0 * HD + kc);
            a[1] = *(const uint32_t*)(Qs + (r0 + 8) * HD + kc);
            a[2] = *(const uint32_t*)(Qs + r0 * HD + kc + 8);
            a[3] = *(const uint32_t*)(Qs + (r0 + 8) * HD + kc + 8);
#pragma unroll
            for (int nt = 0; nt < 1; nt++) {
                int key = 8 * kh + (lane >> 2);
                uint32_t b[2];
                b[0] = *(const uint32_t*)(Ks + key * HD + kc);
                b[1] = *(const uint32_t*)(Ks + key * HD + kc + 8);
                mma16816(sc[nt], a, b);
            }
        }
        // scale + mask; fragment (row r0 / r0 + 8, key 8 kh + 2 (lane & 3) + {0, 1})
        float rmax[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int nt = 0; nt < 1; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int key = tile * TK + 8 * kh + 2 * (lane & 3) + (e & 1);
                bool ok = key < nk && (key < nw || si[key - nw] >= 0);
                float v = ok ? sc[nt][e] * scale : -INFINITY;
                sc[nt][e] = v;
                rmax[e >> 1] = fmaxf(rmax[e >> 1], v);
            }
        for (int h = 0; h < 2; h++) {
            rmax[h] = fmaxf(rmax[h], __shfl_xor_sync(0xffffffffu, rmax[h], 1));
            rmax[h] = fmaxf(rmax[h], __shfl_xor_sync(0xffffffffu, rmax[h], 2));
        }
        if ((lane & 3) == 0) {
            red[kh][r0] = rmax[0];
            red[kh][r0 + 8] = rmax[1];
        }
        __syncthreads();
        float mnew[2], rsum[2] = {0.f, 0.f};
        for (int h = 0; h < 2; h++) {
            int r = r0 + 8 * h;
            float mo = mrow[r];
            mnew[h] = fmaxf(mo, fmaxf(red[0][r], red[1][r]));
        }
#pragma unroll
        for (int nt = 0; nt < 1; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int h = e >> 1;
                float pv = mnew[h] == -INFINITY ? 0.f : expf(sc[nt][e] - mnew[h]);
                rsum[h] += pv;
                int key = 8 * kh + 2 * (lane & 3) + (e & 1);
                Ps[(r0 + 8 * h) * TK + key] = __float2half_rn(pv);
            }
        for (int h = 0; h < 2; h++) {
            rsum[h] += __shfl_xor_sync(0xffffffffu, rsum[h], 1);
            rsum[h] += __shfl_xor_sync(0xffffffffu, rsum[h], 2);
        }
        __syncthreads();   // all reads of red (max) done
        if ((lane & 3) == 0) {
            red[kh][r0] = rsum[0];
            red[kh][r0 + 8] = rsum[1];
        }
        __syncthreads();
        if (kh == 0 && (lane & 3) == 0) {   // warps 0..3 own the row stats of their 16 rows
            for (int h = 0; h < 2; h++) {
                int r = r0 + 8 * h;
                float mo = mrow[r];
                float al = mnew[h] == -INFINITY ? 1.f : expf(mo - mnew[h]);
                lrow[r] = lrow[r] * al + red[0][r] + red[1][r];
                mrow[r] = mnew[h];
                alpha_s[r] = al;
            }
        }
        __syncthreads();
        // O = alpha O + P V: rows 16 mt.., dims 256 dh + 8 n + ...
        {
            float al0 = alpha_s[r0], al1 = alpha_s[r0 + 8];
#pragma unroll
            for (int a = 0; a < 32; a++) {
                acc[a][0] *= al0;
                acc[a][1] *= al0;
                acc[a][2] *= al1;
                acc[a][3] *= al1;
            }
#pragma unroll
            for (int ks = 0; ks < TK / 16; ks++) {
                int kc = ks * 16 + (lane & 3) * 2;
                uint32_t a[4];
                a[0] = *(const uint32_t*)(Ps + r0 * TK + kc);
                a[1] = *(const uint32_t*)(Ps + (r0 + 8) * TK + kc);
                a[2] = *(const uint32_t*)(Ps + r0 * TK + kc + 8);
                a[3] = *(const uint32_t*)(Ps + (r0 + 8) * TK + kc + 8);
#pragma unroll
                for (int nt = 0; nt < 32; nt++) {
                    int dim = 256 * dh + 8 * nt + (lane >> 2);
                    __half2 b0 = __halves2half2(Ks[kc * HD + dim], Ks[(kc + 1) * HD + dim]);
                    __half2 b1 = __halves2half2(Ks[(kc + 8) * HD + dim], Ks[(kc + 9) * HD + dim]);
                    uint32_t b[2] = {*(uint32_t*)&b0, *(uint32_t*)&b1};
                    mma16816(acc[nt], a, b);
                }
            }
        }
    }
    __syncthreads();
    for (int h = 0; h < 2; h++) {
        int r = r0 + 8 * h;
        float den = lrow[r] + expf(sink[r] - mrow[r]);
        float* op = o + ((size_t)i * NH + r) * HD + 256 * dh;
#pragma unroll
        for (int nt = 0; nt < 32; nt++) {
            int c = 8 * nt + 2 * (lane & 3);
            op[c] = acc[nt][2 * h] / den;
            op[c + 1] = acc[nt][2 * h + 1] / den;
        }
    }
}

// [dsv4-pf-attn2] k_attn_tc with padded shared rows (no 8-way bank conflicts on the fragment loads), V fragments by
// ldmatrix.trans and 16 B key loads; the same tiles, mma sequence and softmax order: bit-identical output
constexpr int QS2 = HD + 8, PS2 = TK + 8;
constexpr int TC2_SMEM = NH * QS2 * 2 + TK * QS2 * 2 + NH * PS2 * 2;
__global__ void __launch_bounds__(256, 1) k_attn_tc2(const float* q, int pos0, const bf16* kvw, const bf16* ckv,
                                                    const int* sel, int nsel, int ldsel, const float* sink, float* o) {
    extern __shared__ __align__(16) char tsm[];
    __half* Qs = (__half*)tsm;                                  // [64][QS2]
    __half* Ks = Qs + NH * QS2;                                 // [16][QS2]
    __half* Ps = Ks + TK * QS2;                                 // [64][PS2]
    __shared__ float mrow[NH], lrow[NH];
    __shared__ float red[2][NH];                                // per key-half: tile max, then tile sum
    __shared__ float alpha_s[NH];
    const int i = blockIdx.x, t = threadIdx.x, w = t >> 5, lane = t & 31;
    const int p = pos0 + i;
    const int nw = min(WIN, p + 1);
    const int* si = sel ? sel + (size_t)i * ldsel : nullptr;
    const int nk = nw + (si ? nsel : 0);
    const int ntile = (nk + TK - 1) / TK;
    for (int k = t; k < NH * HD / 2; k += 256) {
        float2 v = ((const float2*)(q + (size_t)i * NH * HD))[k];
        *(__half2*)(Qs + (k / (HD / 2)) * QS2 + 2 * (k % (HD / 2))) = __floats2half2_rn(v.x, v.y);
    }
    if (t < NH) {
        mrow[t] = -INFINITY;
        lrow[t] = 0.f;
    }
    const float scale = 0.044194173824159216f;
    const int mt = w & 3, kh = w >> 2, dh = w >> 2;
    const int r0 = 16 * mt + (lane >> 2);                       // fragment rows r0, r0 + 8
    float acc[32][4];
#pragma unroll
    for (int a = 0; a < 32; a++) acc[a][0] = acc[a][1] = acc[a][2] = acc[a][3] = 0.f;
    for (int tile = 0; tile < ntile; tile++) {
        __syncthreads();
        {   // keys -> fp16 smem: thread t -> key t / 16, 32 dims
            int kk = t >> 4, part = t & 15;
            int j = tile * TK + kk;
            const bf16* src = nullptr;
            if (j < nw) src = kvw + (size_t)(p - nw + 1 + j) * HD;
            else if (j < nk) {
                int c = si[j - nw];
                if (c >= 0) src = ckv + (size_t)c * HD;
            }
            uint4* dst = (uint4*)(Ks + kk * QS2 + part * 32);
            if (src) {
                const uint4* s4 = (const uint4*)(src + part * 32);
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    uint4 v = s4[u], o4;
                    const bf162* b2 = (const bf162*)&v;
                    __half2* h2 = (__half2*)&o4;
#pragma unroll
                    for (int z = 0; z < 4; z++) {
                        float2 f = __bfloat1622float2(b2[z]);
                        h2[z] = __floats2half2_rn(f.x, f.y);
                    }
                    dst[u] = o4;
                }
            } else {
#pragma unroll
                for (int u = 0; u < 4; u++) dst[u] = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();
        // S = Q K^T for heads 16 mt.., keys 8 kh .. 8 kh + 7
        float sc[1][4] = {{0, 0, 0, 0}};
        for (int ks = 0; ks < HD / 16; ks++) {
            int kc = ks * 16 + (lane & 3) * 2;
            uint32_t a[4];
            a[0] = *(const uint32_t*)(Qs + r0 * QS2 + kc);
            a[1] = *(const uint32_t*)(Qs + (r0 + 8) * QS2 + kc);
            a[2] = *(const uint32_t*)(Qs + r0 * QS2 + kc + 8);
            a[3] = *(const uint32_t*)(Qs + (r0 + 8) * QS2 + kc + 8);
#pragma unroll
            for (int nt = 0; nt < 1; nt++) {
                int key = 8 * kh + (lane >> 2);
                uint32_t b[2];
                b[0] = *(const uint32_t*)(Ks + key * QS2 + kc);
                b[1] = *(const uint32_t*)(Ks + key * QS2 + kc + 8);
                mma16816(sc[nt], a, b);
            }
        }
        // scale + mask; fragment (row r0 / r0 + 8, key 8 kh + 2 (lane & 3) + {0, 1})
        float rmax[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int nt = 0; nt < 1; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int key = tile * TK + 8 * kh + 2 * (lane & 3) + (e & 1);
                bool ok = key < nk && (key < nw || si[key - nw] >= 0);
                float v = ok ? sc[nt][e] * scale : -INFINITY;
                sc[nt][e] = v;
                rmax[e >> 1] = fmaxf(rmax[e >> 1], v);
            }
        for (int h = 0; h < 2; h++) {
            rmax[h] = fmaxf(rmax[h], __shfl_xor_sync(0xffffffffu, rmax[h], 1));
            rmax[h] = fmaxf(rmax[h], __shfl_xor_sync(0xffffffffu, rmax[h], 2));
        }
        if ((lane & 3) == 0) {
            red[kh][r0] = rmax[0];
            red[kh][r0 + 8] = rmax[1];
        }
        __syncthreads();
        float mnew[2], rsum[2] = {0.f, 0.f};
        for (int h = 0; h < 2; h++) {
            int r = r0 + 8 * h;
            float mo = mrow[r];
            mnew[h] = fmaxf(mo, fmaxf(red[0][r], red[1][r]));
        }
#pragma unroll
        for (int nt = 0; nt < 1; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int h = e >> 1;
                float pv = mnew[h] == -INFINITY ? 0.f : expf(sc[nt][e] - mnew[h]);
                rsum[h] += pv;
                int key = 8 * kh + 2 * (lane & 3) + (e & 1);
                Ps[(r0 + 8 * h) * PS2 + key] = __float2half_rn(pv);
            }
        for (int h = 0; h < 2; h++) {
            rsum[h] += __shfl_xor_sync(0xffffffffu, rsum[h], 1);
            rsum[h] += __shfl_xor_sync(0xffffffffu, rsum[h], 2);
        }
        __syncthreads();   // all reads of red (max) done
        if ((lane & 3) == 0) {
            red[kh][r0] = rsum[0];
            red[kh][r0 + 8] = rsum[1];
        }
        __syncthreads();
        if (kh == 0 && (lane & 3) == 0) {   // warps 0..3 own the row stats of their 16 rows
            for (int h = 0; h < 2; h++) {
                int r = r0 + 8 * h;
                float mo = mrow[r];
                float al = mnew[h] == -INFINITY ? 1.f : expf(mo - mnew[h]);
                lrow[r] = lrow[r] * al + red[0][r] + red[1][r];
                mrow[r] = mnew[h];
                alpha_s[r] = al;
            }
        }
        __syncthreads();
        // O = alpha O + P V: rows 16 mt.., dims 256 dh + 8 n + ...
        {
            float al0 = alpha_s[r0], al1 = alpha_s[r0 + 8];
#pragma unroll
            for (int a = 0; a < 32; a++) {
                acc[a][0] *= al0;
                acc[a][1] *= al0;
                acc[a][2] *= al1;
                acc[a][3] *= al1;
            }
#pragma unroll
            for (int ks = 0; ks < TK / 16; ks++) {
                int kc = ks * 16 + (lane & 3) * 2;
                uint32_t a[4];
                a[0] = *(const uint32_t*)(Ps + r0 * PS2 + kc);
                a[1] = *(const uint32_t*)(Ps + (r0 + 8) * PS2 + kc);
                a[2] = *(const uint32_t*)(Ps + r0 * PS2 + kc + 8);
                a[3] = *(const uint32_t*)(Ps + (r0 + 8) * PS2 + kc + 8);
                // V fragments by ldmatrix.trans (the same values as the scalar half pairs of k_attn_tc): matrices
                // (keys 0-7 | 8-15) x (dims d .. d + 7 | d + 8 .. d + 15) of two n8 tiles
                const int lk = ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7), ld8 = (lane >> 4) * 8;
#pragma unroll
                for (int nt = 0; nt < 32; nt += 2) {
                    const uint32_t ad = (uint32_t)__cvta_generic_to_shared(Ks + lk * QS2 + 256 * dh + 8 * nt + ld8);
                    uint32_t b[4];
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(ad));
                    mma16816(acc[nt], a, b);
                    mma16816(acc[nt + 1], a, b + 2);
                }
            }
        }
    }
    __syncthreads();
    for (int h = 0; h < 2; h++) {
        int r = r0 + 8 * h;
        float den = lrow[r] + expf(sink[r] - mrow[r]);
        float* op = o + ((size_t)i * NH + r) * HD + 256 * dh;
#pragma unroll
        for (int nt = 0; nt < 32; nt++) {
            int c = 8 * nt + 2 * (lane & 3);
            op[c] = acc[nt][2 * h] / den;
            op[c + 1] = acc[nt][2 * h + 1] / den;
        }
    }
}

// [dsv4-pf4-asplit] k_attn_tc2 on a head range [h0, h0 + NHR) of every query (prefill attention split over the cards):
// MT = row tiles of 16 (rows >= NHR are zero Q, never stored). S = Q K^T and the row statistics on 2 MT warps (mt, kh)
// exactly as k_attn_tc2's warps (the same 8-key halves, the same max / sum order); O = alpha O + P V on all 8 warps
// (mt, dim part of 512 / (8 / MT)). Every stored element has k_attn_tc2's arithmetic: bit-identical per head.
template <int MT, int NHR>
constexpr int tc3_smem() { return NHR * QS2 * 2 + TK * QS2 * 2 + 16 * MT * PS2 * 2 + ITOPK * 4; }
// [dsv4-pf13-attnsrc] S fragments by ldmatrix (the same registers as the 32-bit loads): A x4 (x2 + zero registers for
// a row tile whose upper 8 rows are >= NHR), B x4 = two k-steps of one 8-key half
__device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((uint32_t)__cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void ldsm_x2(uint32_t* r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"((uint32_t)__cvta_generic_to_shared(p)));
}
template <int MT, int NHR>
__global__ void __launch_bounds__(256, 2) k_attn_tc3(const float* q, int pos0, const bf16* kvw, const bf16* ckv,
                                                    const int* sel, int nsel, int ldsel, const float* sink, float* o,
                                                    int h0) {
    constexpr int R = 16 * MT, ND = 8 / MT, DW = HD / ND, NT = DW / 8;
    static_assert(NHR <= R && NHR > R - 16 && NHR % 8 == 0, "rows");
    extern __shared__ __align__(16) char tsm[];
    __half* Qs = (__half*)tsm;                                  // [NHR][QS2]
    __half* Ks = Qs + NHR * QS2;                                // [16][QS2]
    __half* Ps = Ks + TK * QS2;                                 // [R][PS2]
    int* Ss = (int*)(Ps + R * PS2);                             // [nsel] the query's selection row
    __shared__ float mrow[R], lrow[R];
    __shared__ float red[2][R];
    __shared__ float alpha_s[R];
    const int i = blockIdx.x, t = threadIdx.x, w = t >> 5, lane = t & 31;
    const int p = pos0 + i;
    const int nw = min(WIN, p + 1);
    const int* sg = sel ? sel + (size_t)i * ldsel : nullptr;
    const int nk = nw + (sg ? nsel : 0);
    const int* si = Ss;
    if (sg)
        for (int k = t; k < nsel; k += 256) Ss[k] = sg[k];
    const int ntile = (nk + TK - 1) / TK;
    for (int k = t; k < NHR * HD / 2; k += 256) {
        float2 v = ((const float2*)(q + ((size_t)i * NH + h0) * HD))[k];
        *(__half2*)(Qs + (k / (HD / 2)) * QS2 + 2 * (k % (HD / 2))) = __floats2half2_rn(v.x, v.y);
    }
    if (t < R) {
        mrow[t] = -INFINITY;
        lrow[t] = 0.f;
    }
    const float scale = 0.044194173824159216f;
    const bool sw = w < 2 * MT;                                 // S warps
    const int mt = w % MT, kh = w / MT;                         // S role (kh valid for sw)
    const int r0 = 16 * mt + (lane >> 2);                       // fragment rows r0, r0 + 8 (both roles: mt = w % MT)
    const int dq = w / MT;                                      // PV dim part
    // ldmatrix row addresses: A rows 16 mt + (lane & 7) + 8 ((lane >> 3) & 1), dims + 8 (lane >> 4); B keys 8 kh +
    // (lane & 7), dims + 8 ((lane >> 3) & 1) + 16 (lane >> 4)
    const bool a_hi = 16 * mt + 8 < NHR;                        // the tile's upper 8 rows are real heads
    const __half* qrow = Qs + (16 * mt + (lane & 7) + (a_hi ? 8 * ((lane >> 3) & 1) : 0)) * QS2 +
                         (a_hi ? 8 * (lane >> 4) : 8 * ((lane >> 3) & 1));
    const __half* krow = Ks + (8 * kh + (lane & 7)) * QS2 + 8 * ((lane >> 3) & 1) + 16 * (lane >> 4);
    float acc[NT][4];
#pragma unroll
    for (int a = 0; a < NT; a++) acc[a][0] = acc[a][1] = acc[a][2] = acc[a][3] = 0.f;
    float mnew[2] = {0.f, 0.f};
    // [dsv4-pf13-attnsrc] each thread's 64 B of the next tile's key rows are loaded into registers while this tile
    // computes (raw bf16; zero rows for masked keys), converted to fp16 and stored at the top of the next tile
    const int kk = t >> 4, part = t & 15;
    uint4 kr[4];
    auto kload = [&](int tile) {
        int j = tile * TK + kk;
        const bf16* src = nullptr;
        if (j < nw) src = kvw + (size_t)(p - nw + 1 + j) * HD;
        else if (j < nk) {
            int c = si[j - nw];
            if (c >= 0) src = ckv + (size_t)c * HD;
        }
        if (src) {
            const uint4* s4 = (const uint4*)(src + part * 32);
#pragma unroll
            for (int u = 0; u < 4; u++) kr[u] = s4[u];
        } else {
#pragma unroll
            for (int u = 0; u < 4; u++) kr[u] = make_uint4(0, 0, 0, 0);
        }
    };
    __syncthreads();                                            // Ss
    kload(0);
    for (int tile = 0; tile < ntile; tile++) {
        __syncthreads();
        {
            uint4* dst = (uint4*)(Ks + kk * QS2 + part * 32);
#pragma unroll
            for (int u = 0; u < 4; u++) {
                uint4 o4;
                const bf162* b2 = (const bf162*)&kr[u];
                __half2* h2 = (__half2*)&o4;
#pragma unroll
                for (int z = 0; z < 4; z++) {
                    float2 f = __bfloat1622float2(b2[z]);
                    h2[z] = __floats2half2_rn(f.x, f.y);
                }
                dst[u] = o4;
            }
            if (tile + 1 < ntile) kload(tile + 1);
        }
        __syncthreads();
        float sc[4] = {0, 0, 0, 0};
        float rsum[2] = {0.f, 0.f};
        if (sw) {
#pragma unroll 4
            for (int ks = 0; ks < HD / 16; ks += 2) {
                uint32_t a0[4], a1[4], b[4];
                if (a_hi) {
                    ldsm_x4(a0, qrow + ks * 16);
                    ldsm_x4(a1, qrow + ks * 16 + 16);
                } else {
                    uint32_t x[2];
                    ldsm_x2(x, qrow + ks * 16);
                    a0[0] = x[0], a0[2] = x[1], a0[1] = a0[3] = 0u;
                    ldsm_x2(x, qrow + ks * 16 + 16);
                    a1[0] = x[0], a1[2] = x[1], a1[1] = a1[3] = 0u;
                }
                ldsm_x4(b, krow + ks * 16);
                mma16816(sc, a0, b);
                mma16816(sc, a1, b + 2);
            }
            float rmax[2] = {-INFINITY, -INFINITY};
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int key = tile * TK + 8 * kh + 2 * (lane & 3) + (e & 1);
                bool ok = key < nk && (key < nw || si[key - nw] >= 0);
                float v = ok ? sc[e] * scale : -INFINITY;
                sc[e] = v;
                rmax[e >> 1] = fmaxf(rmax[e >> 1], v);
            }
            for (int h = 0; h < 2; h++) {
                rmax[h] = fmaxf(rmax[h], __shfl_xor_sync(0xffffffffu, rmax[h], 1));
                rmax[h] = fmaxf(rmax[h], __shfl_xor_sync(0xffffffffu, rmax[h], 2));
            }
            if ((lane & 3) == 0) {
                red[kh][r0] = rmax[0];
                red[kh][r0 + 8] = rmax[1];
            }
        }
        __syncthreads();
        if (sw) {
            for (int h = 0; h < 2; h++) {
                int r = r0 + 8 * h;
                float mo = mrow[r];
                mnew[h] = fmaxf(mo, fmaxf(red[0][r], red[1][r]));
            }
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int h = e >> 1;
                float pv = mnew[h] == -INFINITY ? 0.f : expf(sc[e] - mnew[h]);
                rsum[h] += pv;
                int key = 8 * kh + 2 * (lane & 3) + (e & 1);
                Ps[(r0 + 8 * h) * PS2 + key] = __float2half_rn(pv);
            }
            for (int h = 0; h < 2; h++) {
                rsum[h] += __shfl_xor_sync(0xffffffffu, rsum[h], 1);
                rsum[h] += __shfl_xor_sync(0xffffffffu, rsum[h], 2);
            }
        }
        __syncthreads();
        if (sw && (lane & 3) == 0) {
            red[kh][r0] = rsum[0];
            red[kh][r0 + 8] = rsum[1];
        }
        __syncthreads();
        if (sw && kh == 0 && (lane & 3) == 0) {
            for (int h = 0; h < 2; h++) {
                int r = r0 + 8 * h;
                float mo = mrow[r];
                float al = mnew[h] == -INFINITY ? 1.f : expf(mo - mnew[h]);
                lrow[r] = lrow[r] * al + red[0][r] + red[1][r];
                mrow[r] = mnew[h];
                alpha_s[r] = al;
            }
        }
        __syncthreads();
        {
            float al0 = alpha_s[r0], al1 = alpha_s[r0 + 8];
#pragma unroll
            for (int a = 0; a < NT; a++) {
                acc[a][0] *= al0;
                acc[a][1] *= al0;
                acc[a][2] *= al1;
                acc[a][3] *= al1;
            }
            int kc = (lane & 3) * 2;
            uint32_t a[4];
            a[0] = *(const uint32_t*)(Ps + r0 * PS2 + kc);
            a[1] = *(const uint32_t*)(Ps + (r0 + 8) * PS2 + kc);
            a[2] = *(const uint32_t*)(Ps + r0 * PS2 + kc + 8);
            a[3] = *(const uint32_t*)(Ps + (r0 + 8) * PS2 + kc + 8);
            const int lk = ((lane >> 3) & 1) * 8 + (lane & 7), ld8 = (lane >> 4) * 8;
#pragma unroll
            for (int nt = 0; nt < NT; nt += 2) {
                const uint32_t ad = (uint32_t)__cvta_generic_to_shared(Ks + lk * QS2 + DW * dq + 8 * nt + ld8);
                uint32_t b[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(ad));
                mma16816(acc[nt], a, b);
                mma16816(acc[nt + 1], a, b + 2);
            }
        }
    }
    __syncthreads();
    for (int h = 0; h < 2; h++) {
        int r = r0 + 8 * h;
        if (r >= NHR) continue;
        float den = lrow[r] + expf(sink[h0 + r] - mrow[r]);
        float* op = o + ((size_t)i * NH + h0 + r) * HD + DW * dq;
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            int c = 8 * nt + 2 * (lane & 3);
            op[c] = acc[nt][2 * h] / den;
            op[c + 1] = acc[nt][2 * h + 1] / den;
        }
    }
}

// [dsv4-pf13-attnsrc] k_attn_tc3 transposed, bitwise equal per element: the mma runs with keys / dims as M and the
// share's heads as N (m16n8k16 is symmetric per element: same products, same order; tests/mmat_test.cu), so a share of
// NHR = 8 NN heads needs no padded rows (96 instead of 128 + 128 mma per 16-key tile at 24 heads). S^T = K Q^T on NN
// warps (one 8-head column each, the 32-step chain as before); the row statistics stay in that warp (max: any order;
// the per-half sums by xor 4 / 8 / 16 = k_attn_tc3's xor 1 / 2 tree over the key pairs; l * al + sum0 + sum1), so one
// barrier passes P and alpha to the PV step: O^T = V^T P^T on all 8 warps (64 dims x NHR heads each).
template <int NHR>
constexpr int tc5_smem() { return NHR * QS2 * 2 + TK * QS2 * 2 + NHR * PS2 * 2 + ITOPK * 4; }
template <int NHR>
__global__ void __launch_bounds__(256, 2) k_attn_tc5(const float* q, int pos0, const bf16* kvw, const bf16* ckv,
                                                    const int* sel, int nsel, int ldsel, const float* sink, float* o,
                                                    int h0) {
    constexpr int NN = NHR / 8, MW = HD / 8 / 16;               // head columns; m16 dim tiles per PV warp
    static_assert(NHR % 8 == 0 && NN <= 8, "heads");
    extern __shared__ __align__(16) char tsm[];
    __half* Qs = (__half*)tsm;                                  // [NHR][QS2]
    __half* Ks = Qs + NHR * QS2;                                // [16][QS2]
    __half* Ps = Ks + TK * QS2;                                 // [NHR][PS2] (head-major, as k_attn_tc3)
    int* Ss = (int*)(Ps + NHR * PS2);                           // [nsel]
    __shared__ float alpha_s[NHR], den_s[NHR];
    const int i = blockIdx.x, t = threadIdx.x, w = t >> 5, lane = t & 31, g = lane >> 2, qd = lane & 3;
    const int p = pos0 + i;
    const int nw = min(WIN, p + 1);
    const int* sg = sel ? sel + (size_t)i * ldsel : nullptr;
    const int nk = nw + (sg ? nsel : 0);
    const int* si = Ss;
    const int ntile = (nk + TK - 1) / TK;
    for (int k = t; k < NHR * HD / 2; k += 256) {
        float2 v = ((const float2*)(q + ((size_t)i * NH + h0) * HD))[k];
        *(__half2*)(Qs + (k / (HD / 2)) * QS2 + 2 * (k % (HD / 2))) = __floats2half2_rn(v.x, v.y);
    }
    if (sg)
        for (int k = t; k < nsel; k += 256) Ss[k] = sg[k];
    const float scale = 0.044194173824159216f;
    // S role (w < NN): heads 8 w + 2 qd + c (c = e & 1), keys g + 8 (e >> 1); running statistics per head column
    float mrun[2] = {-INFINITY, -INFINITY}, lrun[2] = {0.f, 0.f};
    const __half* qb = Qs + (8 * w + (lane & 7)) * QS2 + 8 * ((lane >> 3) & 1) + 16 * (lane >> 4);   // B = Q^T, x4 = 2 k-steps
    const __half* ka = Ks + ((lane & 7) + 8 * ((lane >> 3) & 1)) * QS2 + 8 * (lane >> 4);         // A = K, x4 = 1 k-step
    // PV role: dims 64 w + 16 m + g (+ 8), heads 8 n + 2 qd (+ 1)
    const __half* va = Ks + ((lane & 7) + 8 * (lane >> 4)) * QS2 + 64 * w + 8 * ((lane >> 3) & 1);  // A = V^T (.trans)
    float acc[MW][NN][4];
#pragma unroll
    for (int m = 0; m < MW; m++)
#pragma unroll
        for (int n = 0; n < NN; n++) acc[m][n][0] = acc[m][n][1] = acc[m][n][2] = acc[m][n][3] = 0.f;
    const int kk = t >> 4, part = t & 15;
    uint4 kr[4];
    auto kload = [&](int tile) {
        int j = tile * TK + kk;
        const bf16* src = nullptr;
        if (j < nw) src = kvw + (size_t)(p - nw + 1 + j) * HD;
        else if (j < nk) {
            int c = si[j - nw];
            if (c >= 0) src = ckv + (size_t)c * HD;
        }
        if (src) {
            const uint4* s4 = (const uint4*)src;                // 16 B pieces u * 16 + part: 256 B runs per u
#pragma unroll
            for (int u = 0; u < 4; u++) kr[u] = s4[u * 16 + part];
        } else {
#pragma unroll
            for (int u = 0; u < 4; u++) kr[u] = make_uint4(0, 0, 0, 0);
        }
    };
    __syncthreads();                                            // Ss, Qs
    kload(0);
    for (int tile = 0; tile < ntile; tile++) {
        __syncthreads();                                        // the previous tile's PV reads of Ks / Ps / alpha_s
        {
            uint4* dst = (uint4*)(Ks + kk * QS2);
#pragma unroll
            for (int u = 0; u < 4; u++) {
                uint4 o4;
                const bf162* b2 = (const bf162*)&kr[u];
                __half2* h2 = (__half2*)&o4;
#pragma unroll
                for (int z = 0; z < 4; z++) {
                    float2 f = __bfloat1622float2(b2[z]);
                    h2[z] = __floats2half2_rn(f.x, f.y);
                }
                dst[u * 16 + part] = o4;
            }
            if (tile + 1 < ntile) kload(tile + 1);
        }
        __syncthreads();
        if (w < NN) {
            float sc[4] = {0, 0, 0, 0};
#pragma unroll 4
            for (int ks = 0; ks < HD / 16; ks += 2) {
                uint32_t a0[4], a1[4], b[4];
                ldsm_x4(b, qb + ks * 16);
                ldsm_x4(a0, ka + ks * 16);
                ldsm_x4(a1, ka + ks * 16 + 16);
                mma16816(sc, a0, b);
                mma16816(sc, a1, b + 2);
            }
            float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int key = tile * TK + g + 8 * (e >> 1);
                bool ok = key < nk && (key < nw || si[key - nw] >= 0);
                float v = ok ? sc[e] * scale : -INFINITY;
                sc[e] = v;
                mx[e & 1] = fmaxf(mx[e & 1], v);
            }
#pragma unroll
            for (int c = 0; c < 2; c++) {
                mx[c] = fmaxf(mx[c], __shfl_xor_sync(0xffffffffu, mx[c], 4));
                mx[c] = fmaxf(mx[c], __shfl_xor_sync(0xffffffffu, mx[c], 8));
                mx[c] = fmaxf(mx[c], __shfl_xor_sync(0xffffffffu, mx[c], 16));
            }
            // branch-free (k_attn_tc3's values): expf is computed for every element and selected after (mnew = -inf:
            // pv 0); expf(mrun - 0) = 0 there -> al = 0 + 1 = 1; otherwise x + 0 = x
            float mnew[2], mz[2], pv[4], hs[2][2];
#pragma unroll
            for (int c = 0; c < 2; c++) {
                mnew[c] = fmaxf(mrun[c], mx[c]);
                mz[c] = mnew[c] == -INFINITY ? 0.f : mnew[c];
            }
#pragma unroll
            for (int e = 0; e < 4; e++) {
                pv[e] = expf(sc[e] - mz[e & 1]);
                pv[e] = mnew[e & 1] == -INFINITY ? 0.f : pv[e];   // review pf13 L1: a NaN score gives 0 there, as tc2
                Ps[(8 * w + 2 * qd + (e & 1)) * PS2 + g + 8 * (e >> 1)] = __float2half_rn(pv[e]);
            }
#pragma unroll
            for (int e = 0; e < 4; e++) pv[e] += __shfl_xor_sync(0xffffffffu, pv[e], 4);
#pragma unroll
            for (int e = 0; e < 4; e++) pv[e] += __shfl_xor_sync(0xffffffffu, pv[e], 8);
#pragma unroll
            for (int e = 0; e < 4; e++) {
                pv[e] += __shfl_xor_sync(0xffffffffu, pv[e], 16);
                hs[e >> 1][e & 1] = pv[e];
            }
#pragma unroll
            for (int c = 0; c < 2; c++) {
                float al = expf(mrun[c] - mz[c]) + (mnew[c] == -INFINITY ? 1.f : 0.f);
                lrun[c] = lrun[c] * al + hs[0][c] + hs[1][c];
                mrun[c] = mnew[c];
                if (g == 0) alpha_s[8 * w + 2 * qd + c] = al;
            }
        }
        __syncthreads();
        {
            uint32_t b[NN][2];
            float al[NN][2];
#pragma unroll
            for (int n = 0; n < NN; n++) {
                b[n][0] = *(const uint32_t*)(Ps + (8 * n + g) * PS2 + 2 * qd);
                b[n][1] = *(const uint32_t*)(Ps + (8 * n + g) * PS2 + 2 * qd + 8);
                al[n][0] = alpha_s[8 * n + 2 * qd];
                al[n][1] = alpha_s[8 * n + 2 * qd + 1];
            }
#pragma unroll
            for (int m = 0; m < MW; m++) {
                uint32_t a[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                             : "r"((uint32_t)__cvta_generic_to_shared(va + 16 * m)));
#pragma unroll
                for (int n = 0; n < NN; n++) {
                    acc[m][n][0] *= al[n][0];
                    acc[m][n][1] *= al[n][1];
                    acc[m][n][2] *= al[n][0];
                    acc[m][n][3] *= al[n][1];
                    mma16816(acc[m][n], a, b[n]);
                }
            }
        }
    }
    if (w < NN && g == 0)
#pragma unroll
        for (int c = 0; c < 2; c++) {
            const int hh = 8 * w + 2 * qd + c;
            den_s[hh] = lrun[c] + expf(sink[h0 + hh] - mrun[c]);
        }
    __syncthreads();
#pragma unroll
    for (int m = 0; m < MW; m++)
#pragma unroll
        for (int n = 0; n < NN; n++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int hh = 8 * n + 2 * qd + (e & 1), d = 64 * w + 16 * m + g + 8 * (e >> 1);
                o[((size_t)i * NH + h0 + hh) * HD + d] = acc[m][n][e] / den_s[hh];
            }
}

// combine split partials: block per (query, head), 256 threads (2 dims each)
__global__ void k_attn_comb(const float* part, int nsplit, const float* sink, float* o, int h0 = 0, int pos0 = 0,
                            const float* rcos = nullptr, const float* rsin = nullptr, bf16* oq = nullptr) {
    int i = blockIdx.x, hh = blockIdx.y + h0, t = threadIdx.x;
    const float* pp = part + ((size_t)i * NH + hh) * nsplit * (HD + 2);
    float m = -INFINITY;
    for (int s = 0; s < nsplit; s++) m = fmaxf(m, pp[s * (HD + 2)]);
    float l = 0.f, a0 = 0.f, a1 = 0.f;
    for (int s = 0; s < nsplit; s++) {
        float ms = pp[s * (HD + 2)];
        float f = ms == -INFINITY ? 0.f : expf(ms - m);
        l += pp[s * (HD + 2) + 1] * f;
        a0 += pp[s * (HD + 2) + 2 + 2 * t] * f;
        a1 += pp[s * (HD + 2) + 2 + 2 * t + 1] * f;
    }
    float den = l + expf(sink[hh] - m);
    float* op = o + ((size_t)i * NH + hh) * HD + 2 * t;
    float a = a0 / den, b = a1 / den;
    if (rcos && 2 * t >= HD - RD) {   // R10: inverse rope (k_rope, inverse = 1)
        const int jj = t - (HD - RD) / 2, p = pos0 + i;
        const float c = rcos[(size_t)p * 32 + jj], sn = -rsin[(size_t)p * 32 + jj];
        const float a2 = a * c - b * sn, b2 = a * sn + b * c;
        a = a2;
        b = b2;
    }
    op[0] = a;
    op[1] = b;
    if (oq) q8_pair(a, b, oq + ((size_t)i * NH + hh) * HD + 2 * t);   // R10: the wo_a input, FP8-rounded bf16
}

// fp32 [n][K] -> bf16 rows into a strided buffer (window cache store)
__global__ void k_to_bf16(const float* x, int n, int K, bf16* y, int ldy) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)n * K) return;
    int i = (int)(t / K), k = (int)(t % K);
    y[(size_t)i * ldy + k] = __float2bfloat16(x[t]);
}

// ---------------------------------------------------------------- compressor (ratio 2: pool groups of 2 tokens)

// kv, sc [m][512] (m even, rows 2g, 2g+1 = group g) -> lat [m/2][512] = sum_r softmax_r(sc) kv
__global__ void k_comp_pool2(const float* kv, const float* sc, int g, float* lat) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)g * HD) return;
    int gi = (int)(t / HD), d = (int)(t % HD);
    float s0 = sc[(size_t)(2 * gi) * HD + d], s1 = sc[(size_t)(2 * gi + 1) * HD + d];
    float mx = fmaxf(s0, s1);
    float e0 = expf(s0 - mx), e1 = expf(s1 - mx);
    float den = e0 + e1;
    lat[t] = kv[(size_t)(2 * gi) * HD + d] * (e0 / den) + kv[(size_t)(2 * gi + 1) * HD + d] * (e1 / den);
}

// ---------------------------------------------------------------- indexer

// S [nb][32][T] (q.k), w [nb][32] -> score [nb][T] = sum_h relu(S) w_h, -inf where t >= vis(i) = (p + 1) / r
// [dsv4-rel2-p1b] j0: the first key of this tile of T keys (S [nb * IH][T], score columns j0 .. via the pointer)
__global__ void k_idx_reduce(const float* S, const float* w, int nb, int T, int pos0, int r, float* score, int lds,
                             int j0 = 0) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)nb * T) return;
    int i = (int)(t / T), j = (int)(t % T);
    int vis = (pos0 + i + 1) / r;
    if (j0 + j >= vis) {   // [dsv4-pf5-idx] masked: S not read (the hoisted score GEMM stops at the block's visible count)
        score[(size_t)i * lds + j] = -INFINITY;
        return;
    }
    float s = 0.f;
    const float* Si = S + (size_t)i * IH * T + j;
    for (int h = 0; h < IH; h++) s += fmaxf(Si[(size_t)h * T], 0.f) * (w[i * IH + h] * 0.015625f);   // 128^-.5 32^-.5
    score[(size_t)i * lds + j] = s;
}
// candidate source (layer 20): block scores (max over 8), newest visible block pinned +inf -> bscore [nb][nbk]
__global__ void k_cand_blocks(const float* score, int lds, int nb, int T, int pos0, int r, float* bs, int nbk) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)nb * nbk) return;
    int i = (int)(t / nbk), b = (int)(t % nbk);
    float m = -INFINITY;
    for (int j = b * CAND_BS; j < min(T, b * CAND_BS + CAND_BS); j++) m = fmaxf(m, score[(size_t)i * lds + j]);
    int vis = (pos0 + i + 1) / r;
    int last = max((vis - 1) / CAND_BS, 0);
    bs[t] = b == last ? INFINITY : m;
}
// mask positions outside the selected candidate blocks: cand [nb][nbk] (1 = selected)
__global__ void k_cand_mask(float* score, int lds, int nb, int T, const uint8_t* cand, int nbk) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)nb * T) return;
    int i = (int)(t / T), j = (int)(t % T);
    int b = j / CAND_BS;
    if (b >= nbk || !cand[(size_t)i * nbk + b]) score[(size_t)i * lds + j] = -INFINITY;
}

// R20 [dsv4-r20-idxskip]: the selection when every visible compressed row is taken (T <= ITOPK): row i of the pass
// selects 0 .. vis - 1 in index order (= k_topk's order for kk == total), -1 up to k; at the candidate source the
// block mask k_topk would write (a block is taken iff it holds a visible row)
__global__ void k_idx_all(int nb, int pos0, int r, int k, int* sel, int ldsel, uint8_t* cand, int ldc, int nbk) {
    const int i = blockIdx.x;
    const int vis = (pos0 + i + 1) / r;
    for (int j = threadIdx.x; j < k; j += blockDim.x) sel[(size_t)i * ldsel + j] = j < vis ? j : -1;
    if (cand)
        for (int b = threadIdx.x; b < nbk; b += blockDim.x) cand[(size_t)i * ldc + b] = b * CAND_BS < vis ? 1 : 0;
}
// top-k per row (deterministic): one block of 1024 threads per row; keys = order-preserving uint of the float;
// radix select of the k-th largest (4 passes x 8 bits), then a stable compaction (index order) of all > threshold
// plus the first equal ones. -inf entries are never taken (out = -1 for the rest). mode 1: out = selection mask
// (uint8 [T]) instead of indices.
#ifndef DSV4_TOPK_OLD_DEF
#define DSV4_TOPK_OLD_DEF
__device__ bool g_topk_old = true;   // R20: the R19 order by default (review r20 L1); DSV4_IDXSKIP=1 writes false
#endif
__device__ __forceinline__ uint32_t fkey(float f) {
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
// [dsv4-ctx1-cand] bmap (out mode): entry j of row i stands for key bmap[i][j / CAND_BS] * CAND_BS + j % CAND_BS (compact
// candidates; row stride T / CAND_BS)
__global__ void __launch_bounds__(1024) k_topk(const float* sc, int lds, int T, int k, int* out, int ldo,
                                               uint8_t* mask, int ldm, const int* bmap = nullptr) {
    __shared__ unsigned hist[256];
    __shared__ uint32_t pref_s, need_s;
    __shared__ int cnt_s[1024];
    const float* s = sc + (size_t)blockIdx.x * lds;
    const int t = threadIdx.x;
    const uint32_t NEG = fkey(-INFINITY);
    // count finite entries
    int nfin = 0;
    for (int j = t; j < T; j += 1024) nfin += fkey(s[j]) > NEG;
    cnt_s[t] = nfin;
    __syncthreads();
    for (int o = 512; o; o >>= 1) {
        if (t < o) cnt_s[t] += cnt_s[t + o];
        __syncthreads();
    }
    int total = cnt_s[0];
    __syncthreads();
    int kk = min(k, total);
    uint32_t thr = 0;   // take keys > thr and up to (kk - count(> thr)) keys == thr
    int need_eq = 0;
    // R20 [dsv4-r20-idxskip]: every finite entry taken (kk == total): all of them in index order (the threshold's equal
    // keys used to go last); the indexer's short-context skip writes this order without scores
    if (kk > 0 && kk == total && !g_topk_old) {
        thr = NEG;   // "> NEG" = finite, none "== thr" needed
    } else
    if (kk > 0) {
        if (t == 0) {
            pref_s = 0;
            need_s = kk;
        }
        __syncthreads();
        for (int pass = 0; pass < 4; pass++) {
            int sh = 24 - 8 * pass;
            for (int b = t; b < 256; b += 1024) hist[b] = 0;
            __syncthreads();
            uint32_t pref = pref_s;
            uint32_t pmask = pass == 0 ? 0u : (0xFFFFFFFFu << (sh + 8));
            for (int j = t; j < T; j += 1024) {
                uint32_t u = fkey(s[j]);
                if (u > NEG && (u & pmask) == pref) atomicAdd(&hist[(u >> sh) & 255], 1u);
            }
            __syncthreads();
            if (t == 0) {
                uint32_t need = need_s, acc = 0;
                int b = 255;
                for (; b > 0; b--) {
                    if (acc + hist[b] >= need) break;
                    acc += hist[b];
                }
                pref_s = pref | ((uint32_t)b << sh);
                need_s = need - acc;
            }
            __syncthreads();
        }
        thr = pref_s;
        need_eq = need_s;
    }
    // stable compaction: thread t owns the contiguous range [t * per, (t + 1) * per)
    int per = (T + 1023) / 1024;
    int j0 = t * per, j1 = min(T, j0 + per);
    int ngt = 0, neq = 0;
    if (kk > 0)
        for (int j = j0; j < j1; j++) {
            uint32_t u = fkey(s[j]);
            ngt += u > thr;
            neq += u == thr;
        }
    // exclusive scans of ngt and neq (two passes through cnt_s)
    __shared__ int base_gt[1024], base_eq[1024];
    cnt_s[t] = ngt;
    __syncthreads();
    for (int o = 1; o < 1024; o <<= 1) {
        int v = t >= o ? cnt_s[t - o] : 0;
        __syncthreads();
        cnt_s[t] += v;
        __syncthreads();
    }
    base_gt[t] = cnt_s[t] - ngt;
    int ngt_all = cnt_s[1023];
    __syncthreads();
    cnt_s[t] = neq;
    __syncthreads();
    for (int o = 1; o < 1024; o <<= 1) {
        int v = t >= o ? cnt_s[t - o] : 0;
        __syncthreads();
        cnt_s[t] += v;
        __syncthreads();
    }
    base_eq[t] = cnt_s[t] - neq;
    __syncthreads();
    if (mask) {
        uint8_t* mk = mask + (size_t)blockIdx.x * ldm;
        int e = base_eq[t];
        for (int j = j0; j < j1; j++) {
            uint32_t u = kk > 0 ? fkey(s[j]) : 0;
            uint8_t v = 0;
            if (kk > 0 && u > thr) v = 1;
            else if (kk > 0 && u == thr) v = e++ < need_eq;
            mk[j] = v;
        }
        return;
    }
    int* oi = out + (size_t)blockIdx.x * ldo;
    // order in the output: all "> thr" in index order, then the taken "== thr" in index order
    int g = base_gt[t], e = base_eq[t];
    for (int j = j0; j < j1 && kk > 0; j++) {
        uint32_t u = fkey(s[j]);
        const int jo = bmap ? bmap[(size_t)blockIdx.x * (T / CAND_BS) + j / CAND_BS] * CAND_BS + j % CAND_BS : j;
        if (u > thr) oi[g++] = jo;
        else if (u == thr) {
            if (e < need_eq) oi[ngt_all + e] = jo;
            e++;
        }
    }
    for (int j = kk + t; j < k; j += 1024) oi[j] = -1;
}

// ---------------------------------------------------------------- router

// logits [n][384] -> ids [n][6] (top-6 of sqrt(softplus) + bias, descending), w [n][6] = s / sum * 1.5;
// warp per row
// slot / margin (R7 cache-aware routing, decode): a chosen expert without a VRAM slot (slot[e] < 0) is replaced by the
// best resident expert not chosen yet if its choice score (score + bias) is within `margin` of the chosen one's
__global__ void k_route(const float* logits, const float* bias, int n, int* ids, float* w, const int* slot = nullptr,
                        float margin = -1.f, const int* dslot = nullptr, float dropw = 0.f) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (i >= n) return;
    float sv[NE / 32], cv[NE / 32];
    for (int j = 0; j < NE / 32; j++) {
        int e = j * 32 + lane;
        sv[j] = sqrtf(softplusf_(logits[(size_t)i * NE + e]));
        cv[j] = sv[j] + bias[e];
    }
    int sel[TOPK];
    float ss[TOPK], cs[TOPK], sum = 0.f;
    for (int k = 0; k < TOPK; k++) {
        float best = -INFINITY;
        int be = 1 << 30;
        for (int j = 0; j < NE / 32; j++) {
            int e = j * 32 + lane;
            if (cv[j] > best || (cv[j] == best && e < be)) {
                best = cv[j];
                be = e;
            }
        }
        for (int o = 16; o; o >>= 1) {
            float ob = __shfl_xor_sync(0xffffffffu, best, o);
            int oe = __shfl_xor_sync(0xffffffffu, be, o);
            if (ob > best || (ob == best && oe < be)) {
                best = ob;
                be = oe;
            }
        }
        if (be >= NE) {   // review r8 L1: NaN / -inf logits row: a valid id (k), weight from sv (no out-of-bounds reads)
            be = k;
            best = -INFINITY;
        }
        sel[k] = be;
        cs[k] = best;
        int jj = be / 32;
        float sc = __shfl_sync(0xffffffffu, sv[jj], be & 31);
        ss[k] = sc;
        if ((be & 31) == lane) cv[jj] = -INFINITY;
    }
    if (slot && margin >= 0.f) {
        margin = fminf(margin, 1e30f);   // review r7 L9: +inf would let an already chosen (-inf) expert through
        bool res[NE / 32];   // resident: an FP4 slot, or a cb3 pool slot (review r7 L7)
        for (int j = 0; j < NE / 32; j++) res[j] = slot[j * 32 + lane] >= 0 || (dslot && dslot[j * 32 + lane] >= 0);
        for (int k = 0; k < TOPK; k++) {
            if (sel[k] >= NE) continue;   // review r7 L9: degenerate (all NaN / -inf) row
            if (slot[sel[k]] >= 0 || (dslot && dslot[sel[k]] >= 0)) continue;
            float best = -INFINITY;
            int be = 1 << 30;
            for (int j = 0; j < NE / 32; j++) {
                int e = j * 32 + lane;
                if (res[j] && (cv[j] > best || (cv[j] == best && e < be))) {
                    best = cv[j];
                    be = e;
                }
            }
            for (int o = 16; o; o >>= 1) {
                float ob = __shfl_xor_sync(0xffffffffu, best, o);
                int oe = __shfl_xor_sync(0xffffffffu, be, o);
                if (ob > best || (ob == best && oe < be)) {
                    best = ob;
                    be = oe;
                }
            }
            if (be >= NE || best < cs[k] - margin) continue;
            sel[k] = be;
            int jj = be / 32;
            ss[k] = __shfl_sync(0xffffffffu, sv[jj], be & 31);
            if ((be & 31) == lane) cv[jj] = -INFINITY;
        }
    }
    for (int k = 0; k < TOPK; k++) sum += ss[k];
    if (lane == 0) {
        float wk[TOPK];
        for (int k = 0; k < TOPK; k++) wk[k] = ss[k] / (sum + 1e-20f) * RSF;
        if (dropw > 0.f && wk[5] < dropw) {   // R8 adaptive top-k: the 6th (then the 5th) choice left out (weight 0)
            wk[5] = 0.f;
            if (wk[4] < dropw) wk[4] = 0.f;
        }
        for (int k = 0; k < TOPK; k++) {
            ids[i * TOPK + k] = sel[k];
            w[i * TOPK + k] = wk[k];
        }
    }
}

// R10: the decode routing tail in one launch (rows n <= MAXR) instead of k_route + q_fp8 + a D2H copy + k_flag.
// Blocks 0 .. n-1: routing of row i with 384 threads: the rank of every expert under the total order (choice score
// desc, id asc) is counted directly, which selects exactly what k_route's six argmax rounds select (same values, same
// tie rule); cache-aware swaps take the residents in the same order; weights summed in the same order (bit-identical).
// Blocks n ..: q_fp8 of x (as k_q_fp8). Each output goes to the device pointer and, when given, the host-mapped one
// (the CPU lane reads it there: no copy). The last block to finish raises flag = seq after a system fence.
__device__ __forceinline__ unsigned long long warp_max64(unsigned long long v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        unsigned long long w = __shfl_xor_sync(0xffffffffu, v, o);
        v = w > v ? w : v;
    }
    return v;
}
// top TOPK keys of a block's 384 keys (one per thread, 0 = none) -> out[0..TOPK) in order (0 = none); keys unique
__device__ __forceinline__ void topk_keys(unsigned long long key, unsigned long long* s_w, unsigned long long* out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    unsigned long long kk = key;
    for (int r = 0; r < TOPK; r++) {   // warp-local top 6
        const unsigned long long m = warp_max64(kk);
        if (lane == 0) s_w[warp * TOPK + r] = m;
        if (kk == m) kk = 0;
    }
    __syncthreads();
    if (warp == 0) {   // the global top 6 is among the 12 x 6 local ones
        unsigned long long a = s_w[lane], b = s_w[lane + 32], c = lane + 64 < 12 * TOPK ? s_w[lane + 64] : 0ull;
        for (int r = 0; r < TOPK; r++) {
            unsigned long long l = a > b ? a : b;
            l = l > c ? l : c;
            const unsigned long long m = warp_max64(l);
            if (lane == 0) out[r] = m;
            if (a == m) a = 0;
            if (b == m) b = 0;
            if (c == m) c = 0;
        }
    }
    __syncthreads();
}
__global__ void __launch_bounds__(384) k_route_pub(const float* logits, const float* bias, int n, const float* x,
                                                   int* ids_d, float* w_d, float* xq_d, int* ids_h, float* w_h,
                                                   float* xq_h, int* counter, volatile int* flag, int seq,
                                                   const int* slot, float margin, const int* dslot, float dropw,
                                                   int x16 = 0, const int* early = nullptr, int* edls = nullptr,
                                                   volatile int* edec = nullptr) {
    __shared__ float s_cv[NE], s_sv[NE];
    __shared__ unsigned long long s_w[12 * TOPK], s_top[TOPK], s_res[TOPK];
    if ((int)blockIdx.x < n) {
        const int i = blockIdx.x, e = threadIdx.x;
        const float sv = sqrtf(softplusf_(logits[(size_t)i * NE + e]));
        const float cv = sv + bias[e];
        s_cv[e] = cv;
        s_sv[e] = sv;
        // order key (score desc, id asc; -0 as +0, NaN last): orderable float bits, then NE - id (keys >= 1, unique)
        const float c0 = cv + 0.f;
        unsigned u = __float_as_uint(c0);
        u = c0 != c0 ? 0u : (u & 0x80000000u) ? ~u : (u | 0x80000000u);
        const unsigned long long key = ((unsigned long long)u << 32) | (unsigned)(NE - e);
        topk_keys(key, s_w, s_top);   // (syncs: s_cv / s_sv visible after it)
        const bool car = slot && margin >= 0.f;
        if (car) {   // the best residents outside the initial choice, in the same order
            bool chosen = false;
            for (int k = 0; k < TOPK; k++) chosen |= s_top[k] == key;
            const bool cand = !chosen && (slot[e] >= 0 || (dslot && dslot[e] >= 0));
            topk_keys(cand ? key : 0ull, s_w, s_res);
        }
        if (e == 0) {
            int sel[TOPK];
            float ss[TOPK], cs[TOPK], sum = 0.f;
            for (int k = 0; k < TOPK; k++) {
                sel[k] = NE - (int)(s_top[k] & 0xFFFFu);
                cs[k] = s_cv[sel[k]];
                ss[k] = s_sv[sel[k]];
            }
            if (car) {
                const float mg = fminf(margin, 1e30f);
                int pnext = 0;
                for (int k = 0; k < TOPK; k++) {
                    if (slot[sel[k]] >= 0 || (dslot && dslot[sel[k]] >= 0)) continue;
                    if (pnext >= TOPK || s_res[pnext] == 0ull) continue;
                    const int be = NE - (int)(s_res[pnext] & 0xFFFFu);
                    if (s_cv[be] < cs[k] - mg) continue;
                    sel[k] = be;
                    ss[k] = s_sv[be];
                    pnext++;
                }
            }
            for (int k = 0; k < TOPK; k++) sum += ss[k];
            float wk[TOPK];
            for (int k = 0; k < TOPK; k++) wk[k] = ss[k] / (sum + 1e-20f) * RSF;
            if (dropw > 0.f && wk[5] < dropw) {
                wk[5] = 0.f;
                if (wk[4] < dropw) wk[4] = 0.f;
            }
            for (int k = 0; k < TOPK; k++) {
                if (ids_d) ids_d[i * TOPK + k] = sel[k];
                if (w_d) w_d[i * TOPK + k] = wk[k];
                if (ids_h) ((volatile int*)ids_h)[i * TOPK + k] = sel[k];
                if (w_h) ((volatile float*)w_h)[i * TOPK + k] = wk[k];
            }
        }
    } else {
        // [dsv4-dec2-rpq] the q_fp8 blocks loop over the 32-value groups (grid-stride over the q blocks): fewer blocks,
        // so fewer system fences below (86 of them cost 10 us quiet, ~54 us while the CPU streams DRAM)
        const int lane = threadIdx.x & 31, nqb = (int)gridDim.x - n;
        for (int gw = ((int)blockIdx.x - n) * (384 / 32) + (threadIdx.x >> 5); gw < n * (D / 32); gw += nqb * (384 / 32)) {
            const int r = gw / (D / 32), b = gw % (D / 32);
            float v = x[(size_t)r * D + b * 32 + lane];
            float am = fmaxf(warp_max(fabsf(v)), 1e-4f);
            float sc = pow2ceil_f(am * (1.f / 448.f));
            float y = e4m3_rne(fminf(fmaxf(v / sc, -448.f), 448.f)) * sc;
            xq_d[(size_t)r * D + b * 32 + lane] = y;
            if (xq_h) {   // R20 [dsv4-r20-xh16]: x16 = bf16 to the host (exact for e4m3 x 2^k), half the mapped bytes
                if (x16) ((volatile unsigned short*)xq_h)[(size_t)r * D + b * 32 + lane] = (unsigned short)(__float_as_uint(y) >> 16);
                else ((volatile float*)xq_h)[(size_t)r * D + b * 32 + lane] = y;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        if (atomicAdd(counter, 1) == (int)gridDim.x - 1) {
            *counter = 0;
            if (edec) {   // dsv4-fut [dsv4-fut-early]: the early copy landed for this pass -> the expert is resident (owner
                // table, read by the MoE kernels after this one in stream order); the decision goes to the host with the flag
                const volatile int* ev = early;
                int took = 0;
                if (ev[2] == seq) {
                    __threadfence();
                    edls[ev[0]] = ev[1];
                    took = 1;
                }
                edec[1] = took;
                edec[0] = seq;
            }
            __threadfence_system();
            *flag = seq;
            __threadfence_system();
        }
    }
}
// dsv4-fut [dsv4-fut-early]: after the early copy (same stream): the request (expert, local slot) and then its seq
__global__ void k_early_ready(int* d, int e, int lidx, int seq) {
    d[0] = e;
    d[1] = lidx;
    __threadfence();
    ((volatile int*)d)[2] = seq;
}
// grid of k_route_pub for n rows (with the q_fp8 blocks when x is given)
// [dsv4-dec2-rpq]: at most RPQ_B q_fp8 blocks (DSV4_NORPQ=1: one per 12 groups, as before; same values either way)
constexpr int RPQ_B = 6;
static const bool g_rpq_off = getenv("DSV4_NORPQ") != nullptr;
inline unsigned route_pub_grid(int n, bool xq) {
    const int qb = (n * (D / 32) + 11) / 12;
    return (unsigned)(n + (xq ? (g_rpq_off ? qb : std::min(qb, RPQ_B)) : 0));
}

// ---------------------------------------------------------------- MoE (decode: hot FP4 experts from VRAM slots)

// gate/up GEMV of the resident experts among the row's 6: x [5120] FP8-rounded fp32 (row r of a step of R rows),
// out gu [R][6][4608] (gate 0..2303, up 2304..4607). grid (4608 / 8, 6 * R), warp per output row.
__global__ void k_moe_gu(const float* x, const int* ids, const int* slot, const uint8_t* slots, size_t xbytes,
                         float* gu) {
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = c_fp4[threadIdx.x];
    __syncthreads();
    int k = blockIdx.y % TOPK, r = blockIdx.y / TOPK;
    int sl = slot[ids[r * TOPK + k]];
    if (sl < 0) return;
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const uint8_t* base = slots + (size_t)sl * xbytes;
    MatView mv[3];
    expert_views(XF_FP4, base, mv);
    int which = row < INTER ? 0 : 1, rr = row % INTER;
    const uint8_t* q = mv[which].q + (size_t)rr * (D / 2);
    const uint8_t* s = (const uint8_t*)mv[which].s + (size_t)rr * (D / 32);
    const float* xr = x + (size_t)r * D;
    float acc = 0.f;
    for (int c = lane; c < D / 32; c += 32) {   // 16 B = 32 values = one scale group
        uint4 v = *(const uint4*)(q + c * 16);
        const uint8_t* b8 = (const uint8_t*)&v;
        const float4* xp = (const float4*)(xr + c * 32);
        float d = 0.f;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            float4 xv = xp[j];
            uint8_t b0 = b8[2 * j], b1 = b8[2 * j + 1];
            d += lut[b0 & 15] * xv.x + lut[b0 >> 4] * xv.y + lut[b1 & 15] * xv.z + lut[b1 >> 4] * xv.w;
        }
        acc += d * e8m0f(s[c]);
    }
    acc = warp_sum(acc);
    if (lane == 0) gu[((size_t)r * TOPK + k) * 2 * INTER + row] = acc;
}
// h = w * silu(min(g, 10)) * clamp(u, +-10) for resident experts (or all when slot == nullptr), h [R][6][2304]
__global__ void k_moe_glu(const float* gu, const float* w, const int* ids, const int* slot, int R, float* h,
                          const int* dslot = nullptr) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)R * TOPK * INTER) return;
    int rk = (int)(t / INTER), c = (int)(t % INTER);
    if (slot && slot[ids[rk]] < 0 && !(dslot && dslot[ids[rk]] >= 0)) {
        h[t] = 0.f;
        return;
    }
    float g = fminf(gu[(size_t)rk * 2 * INTER + c], CLAMP);
    float u = fminf(fmaxf(gu[(size_t)rk * 2 * INTER + INTER + c], -CLAMP), CLAMP);
    h[t] = w[rk] * (siluf_(g) * u);
}
// R10: k_moe_glu + q_fp8 of h in one launch (a warp = 32 consecutive c of one choice: k_q_fp8's blocks; a choice
// without a slot is all zero, which q_fp8 keeps zero)
__global__ void k_moe_glu_q(const float* gu, const float* w, const int* ids, const int* slot, int R, float* h,
                            const int* dslot = nullptr) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)R * TOPK * INTER) return;
    int rk = (int)(t / INTER), c = (int)(t % INTER);
    if (slot && slot[ids[rk]] < 0 && !(dslot && dslot[ids[rk]] >= 0)) {
        h[t] = 0.f;
        return;
    }
    float g = fminf(gu[(size_t)rk * 2 * INTER + c], CLAMP);
    float u = fminf(fmaxf(gu[(size_t)rk * 2 * INTER + INTER + c], -CLAMP), CLAMP);
    float v = w[rk] * (siluf_(g) * u);
    float am = fmaxf(warp_max(fabsf(v)), 1e-4f);
    float s8 = pow2ceil_f(am * (1.f / 448.f));
    h[t] = e4m3_rne(fminf(fmaxf(v / s8, -448.f), 448.f)) * s8;
}
// R10: decode window KV tail in one launch: rms(kv) * w, rope of the last 64 dims at p = pos0 + i, q_fp8 -> bf16 row
// of the window cache (k_rms with 256 threads, k_rope, k_q_fp8: the same expressions and orders). Block per row.
__global__ void __launch_bounds__(256) k_kv_tail(const float* kv, int ldkv, const float* w, const float* cosv,
                                                 const float* sinv, int pos0, bf16* out, int n = 0,
                                                 const float* qa = nullptr, const float* qw = nullptr, float* qr = nullptr,
                                                 bf16* qrb = nullptr, int orow = 0, int omask = -1) {
    // [dsv4-rel2-p1] the window row of block i goes to out row (orow + i) & omask (a ring; defaults: row i)
    __shared__ float red[32];
    __shared__ float y[HD];
    const int t = threadIdx.x;
    if (qa && (int)blockIdx.x >= n) {   // R10: blocks n .. 2n - 1: q low-rank norm, k_rms (256 threads) of row i
        const int i = blockIdx.x - n;
        const float* xr = qa + (size_t)i * ldkv;
        float s = 0.f;
        for (int k = t; k < QLR; k += 256) s += xr[k] * xr[k];
        s = warp_sum(s);
        if ((t & 31) == 0) red[t >> 5] = s;
        __syncthreads();
        if (t < 32) {
            float tt = t < 8 ? red[t] : 0.f;
            tt = warp_sum(tt);
            if (t == 0) red[0] = rsqrtf(tt / (float)QLR + EPS);
        }
        __syncthreads();
        const float r = red[0];
        for (int k = t; k < QLR; k += 256) {   // (a warp holds 32 consecutive k: one FP8 block)
            const float v = xr[k] * r * qw[k];
            qr[(size_t)i * QLR + k] = v;
            if (qrb) {   // R15: k_q_fp8's expressions on the same value (exact bf16 of the FP8-rounded qr)
                const float am = fmaxf(warp_max(fabsf(v)), 1e-4f);
                const float s8 = pow2ceil_f(am * (1.f / 448.f));
                qrb[(size_t)i * QLR + k] = __float2bfloat16(e4m3_rne(fminf(fmaxf(v / s8, -448.f), 448.f)) * s8);
            }
        }
        return;
    }
    const int i = blockIdx.x;
    const float* xr = kv + (size_t)i * ldkv;
    float s = 0.f;
    for (int k = t; k < HD; k += 256) s += xr[k] * xr[k];
    s = warp_sum(s);
    if ((t & 31) == 0) red[t >> 5] = s;
    __syncthreads();
    if (t < 32) {
        float tt = t < 8 ? red[t] : 0.f;
        tt = warp_sum(tt);
        if (t == 0) red[0] = rsqrtf(tt / (float)HD + EPS);
    }
    __syncthreads();
    const float r = red[0];
    for (int k = t; k < HD; k += 256) y[k] = xr[k] * r * w[k];
    __syncthreads();
    if (t < 32) {
        const int p = pos0 + i;
        float* q = y + (HD - RD) + 2 * t;
        float c = cosv[(size_t)p * 32 + t], sn = sinv[(size_t)p * 32 + t];
        float a = q[0], b = q[1];
        q[0] = a * c - b * sn;
        q[1] = a * sn + b * c;
    }
    __syncthreads();
    for (int k = t; k < HD; k += 256) {
        float v = y[k];
        float am = fmaxf(warp_max(fabsf(v)), 1e-4f);
        float s8 = pow2ceil_f(am * (1.f / 448.f));
        out[(size_t)((orow + i) & omask) * HD + k] = __float2bfloat16(e4m3_rne(fminf(fmaxf(v / s8, -448.f), 448.f)) * s8);
    }
}
// down GEMV: hq [R][6][2304] FP8-rounded -> y [R][6][5120]; grid (5120 / 8, 6 R)
__global__ void k_moe_down(const float* hq, const int* ids, const int* slot, const uint8_t* slots, size_t xbytes,
                           float* y) {
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = c_fp4[threadIdx.x];
    __syncthreads();
    int k = blockIdx.y % TOPK, r = blockIdx.y / TOPK;
    int sl = slot[ids[r * TOPK + k]];
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (sl < 0) {
        if (lane == 0) y[((size_t)r * TOPK + k) * D + row] = 0.f;
        return;
    }
    const uint8_t* base = slots + (size_t)sl * xbytes;
    MatView mv[3];
    expert_views(XF_FP4, base, mv);
    const uint8_t* q = mv[2].q + (size_t)row * (INTER / 2);
    const uint8_t* s = (const uint8_t*)mv[2].s + (size_t)row * (INTER / 32);
    const float* xr = hq + ((size_t)r * TOPK + k) * INTER;
    float acc = 0.f;
    for (int c = lane; c < INTER / 32; c += 32) {
        uint4 v = *(const uint4*)(q + c * 16);
        const uint8_t* b8 = (const uint8_t*)&v;
        const float4* xp = (const float4*)(xr + c * 32);
        float d = 0.f;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            float4 xv = xp[j];
            uint8_t b0 = b8[2 * j], b1 = b8[2 * j + 1];
            d += lut[b0 & 15] * xv.x + lut[b0 >> 4] * xv.y + lut[b1 & 15] * xv.z + lut[b1 >> 4] * xv.w;
        }
        acc += d * e8m0f(s[c]);
    }
    acc = warp_sum(acc);
    if (lane == 0) y[((size_t)r * TOPK + k) * D + row] = acc;
}

// hot-expert GEMV, round 5: e2m1 -> fp16 bits by shifts on pairs of nibbles (exact; value / 2^14, the 2^14 folded
// into the scale) instead of a shared-memory LUT per element; same layout / grid / semantics as k_moe_gemv2
__device__ __forceinline__ float2 fp4pair(uint32_t t) {   // t: two nibbles in bits 0-3 and 16-19
    uint32_t h = ((t & 0x00070007u) << 9) | ((t & 0x00080008u) << 12);
    return __half22float2(*(const __half2*)&h);
}
template <int LPR, int NLD>
__global__ void __launch_bounds__(256) k_moe_gemv4(const float* x, int which, const int* ids, const int* slot,
                                                   const uint8_t* slots, size_t xbytes, float* out) {
    const int k = blockIdx.y % TOPK, r = blockIdx.y / TOPK;
    const int sl = slot[ids[r * TOPK + k]];
    constexpr int RPB = 8 * (32 / LPR);
    const int row = blockIdx.x * RPB + threadIdx.x / LPR, sub = threadIdx.x % LPR;
    const int K = which ? INTER : D;
    const int NR = which ? D : 2 * INTER;
    if (row >= NR) return;
    float* o = which ? out + ((size_t)r * TOPK + k) * D + row : out + ((size_t)r * TOPK + k) * 2 * INTER + row;
    if (sl < 0) {
        if (which && sub == 0) *o = 0.f;
        return;
    }
    MatView mv[3];
    expert_views(XF_FP4, slots + (size_t)sl * xbytes, mv);
    const int m = which ? 2 : (row < INTER ? 0 : 1), rr = which ? row : row % INTER;
    const uint8_t* q = mv[m].q + (size_t)rr * (K / 2);
    const uint8_t* s = (const uint8_t*)mv[m].s + (size_t)rr * (K / 32);
    const float* xr = which ? x + ((size_t)r * TOPK + k) * INTER : x + (size_t)r * D;
    const int nch = K / 32;
    uint4 v[NLD];
    uint8_t sc8[NLD];
#pragma unroll
    for (int u = 0; u < NLD; u++) {
        int c = sub + LPR * u;
        v[u] = c < nch ? *(const uint4*)(q + c * 16) : make_uint4(0, 0, 0, 0);
        sc8[u] = c < nch ? s[c] : 0;
    }
    float acc = 0.f;
#pragma unroll
    for (int u = 0; u < NLD; u++) {
        int c = sub + LPR * u;
        if (c >= nch) break;
        const uint32_t* w32 = (const uint32_t*)&v[u];
        const float4* xp = (const float4*)(xr + c * 32);
        float d = 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++) {   // word j = bytes 4j..4j+3 = elements 8j..8j+7 (low nibble = even element)
            uint32_t w = w32[j];
            float2 e04 = fp4pair(w), e26 = fp4pair(w >> 8), e15 = fp4pair(w >> 4), e37 = fp4pair(w >> 12);
            float4 xa = xp[2 * j], xb = xp[2 * j + 1];   // elements 8j..8j+3, 8j+4..8j+7
            d += e04.x * xa.x + e15.x * xa.y + e26.x * xa.z + e37.x * xa.w;
            d += e04.y * xb.x + e15.y * xb.y + e26.y * xb.z + e37.y * xb.w;
        }
        acc += d * (e8m0f(sc8[u]) * 16384.f);
    }
    for (int off = LPR / 2; off; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (sub == 0) *o = acc;
}

// hot-expert GEMV, round 5b: coalesced. A warp walks one row at a time, each lane 8 B (16 elements) per step, so a
// warp load is 256 contiguous bytes; x (FP8-rounded, exact in bf16) staged once per block in shared memory, read
// 32 B per lane (conflict-light). RW rows per warp, 8 warps per block. Same outputs as k_moe_gemv2 / 4 (sum order
// differs). which = 0: gate/up (2 x 2304 rows, K 5120), 1: down (5120 rows, K 2304)
template <int RW>
__global__ void __launch_bounds__(256) k_moe_gemv5(const float* x, int which, const int* ids, const int* slot,
                                                   const uint8_t* slots, size_t xbytes, float* out) {
    __shared__ __align__(16) bf16 xs[D];
    const int k = blockIdx.y % TOPK, r = blockIdx.y / TOPK;
    const int sl = slot[ids[r * TOPK + k]];
    const int K = which ? INTER : D;
    const int NR = which ? D : 2 * INTER;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * 8 + warp) * RW;
    if (sl < 0) {
        if (which && lane < RW)
            for (int i = lane; i < RW; i += 32)
                if (row0 + i < NR) out[((size_t)r * TOPK + k) * D + row0 + i] = 0.f;
        return;
    }
    const float* xr = which ? x + ((size_t)r * TOPK + k) * INTER : x + (size_t)r * D;
    for (int t = threadIdx.x * 4; t < K; t += 256 * 4) {
        float4 v = *(const float4*)(xr + t);
        *(bf162*)(xs + t) = __floats2bfloat162_rn(v.x, v.y);
        *(bf162*)(xs + t + 2) = __floats2bfloat162_rn(v.z, v.w);
    }
    __syncthreads();
    MatView mv[3];
    expert_views(XF_FP4, slots + (size_t)sl * xbytes, mv);
    const int nst = (K + 511) / 512;   // steps of 32 lanes x 16 elements (down: K = 2304, the last step half full)
    for (int i = 0; i < RW; i++) {
        const int row = row0 + i;
        if (row >= NR) break;
        const int m = which ? 2 : (row < INTER ? 0 : 1), rr = which ? row : row % INTER;
        const uint8_t* q = mv[m].q + (size_t)rr * (K / 2);
        const uint8_t* s = (const uint8_t*)mv[m].s + (size_t)rr * (K / 32);
        float acc = 0.f;
#pragma unroll 5
        for (int st = 0; st < nst; st++) {
            const int e0 = st * 512 + lane * 16;           // first element of this lane
            if (e0 >= K) break;
            uint2 w = *(const uint2*)(q + e0 / 2);
            float sc = e8m0f(s[e0 >> 5]) * 16384.f;
            const uint4* xp = (const uint4*)(xs + e0);
            uint4 xa = xp[0], xb = xp[1];
            const bf162* pa = (const bf162*)&xa;
            const bf162* pb = (const bf162*)&xb;
            float d = 0.f;
            uint32_t ww[2] = {w.x, w.y};
#pragma unroll
            for (int j = 0; j < 2; j++) {   // word j: elements 8j..8j+7
                float2 e04 = fp4pair(ww[j]), e26 = fp4pair(ww[j] >> 8), e15 = fp4pair(ww[j] >> 4), e37 = fp4pair(ww[j] >> 12);
                const bf162* px = j ? pb : pa;
                float2 x01 = __bfloat1622float2(px[0]), x23 = __bfloat1622float2(px[1]);
                float2 x45 = __bfloat1622float2(px[2]), x67 = __bfloat1622float2(px[3]);
                d += e04.x * x01.x + e15.x * x01.y + e26.x * x23.x + e37.x * x23.y;
                d += e04.y * x45.x + e15.y * x45.y + e26.y * x67.x + e37.y * x67.y;
            }
            acc += d * sc;
        }
        acc = warp_sum(acc);
        if (lane == 0) {
            float* o = which ? out + ((size_t)r * TOPK + k) * D + row : out + ((size_t)r * TOPK + k) * 2 * INTER + row;
            *o = acc;
        }
    }
}

// faster decode hot-expert GEMV (round 2): LPR lanes per output row, all of a lane's 16 B loads issued before use.
// which = 0: gate/up (rows 0..4607 of [w1; w3], K = 5120, out gu), 1: down (rows 0..5119, K = 2304, out y)
template <int LPR, int NLD>
__global__ void __launch_bounds__(256) k_moe_gemv2(const float* x, int which, const int* ids, const int* slot,
                                                   const uint8_t* slots, size_t xbytes, float* out) {
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = c_fp4[threadIdx.x];
    __syncthreads();
    const int k = blockIdx.y % TOPK, r = blockIdx.y / TOPK;
    const int sl = slot[ids[r * TOPK + k]];
    constexpr int RPB = 8 * (32 / LPR);
    const int row = blockIdx.x * RPB + threadIdx.x / LPR, sub = threadIdx.x % LPR;
    const int K = which ? INTER : D;
    const int NR = which ? D : 2 * INTER;
    if (row >= NR) return;
    float* o = which ? out + ((size_t)r * TOPK + k) * D + row : out + ((size_t)r * TOPK + k) * 2 * INTER + row;
    if (sl < 0) {
        if (which && sub == 0) *o = 0.f;
        return;
    }
    MatView mv[3];
    expert_views(XF_FP4, slots + (size_t)sl * xbytes, mv);
    const int m = which ? 2 : (row < INTER ? 0 : 1), rr = which ? row : row % INTER;
    const uint8_t* q = mv[m].q + (size_t)rr * (K / 2);
    const uint8_t* s = (const uint8_t*)mv[m].s + (size_t)rr * (K / 32);
    const float* xr = which ? x + ((size_t)r * TOPK + k) * INTER : x + (size_t)r * D;
    const int nch = K / 32;   // 16 B = 32 values = one scale group
    uint4 v[NLD];
#pragma unroll
    for (int u = 0; u < NLD; u++) {
        int c = sub + LPR * u;
        v[u] = c < nch ? *(const uint4*)(q + c * 16) : make_uint4(0, 0, 0, 0);
    }
    float acc = 0.f;
#pragma unroll
    for (int u = 0; u < NLD; u++) {
        int c = sub + LPR * u;
        if (c >= nch) break;
        const uint8_t* b8 = (const uint8_t*)&v[u];
        const float4* xp = (const float4*)(xr + c * 32);
        float d = 0.f;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            float4 xv = xp[j];
            uint8_t b0 = b8[2 * j], b1 = b8[2 * j + 1];
            d += lut[b0 & 15] * xv.x + lut[b0 >> 4] * xv.y + lut[b1 & 15] * xv.z + lut[b1 >> 4] * xv.w;
        }
        acc += d * e8m0f(s[c]);
    }
    for (int off = LPR / 2; off; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (sub == 0) *o = acc;
}


// dynamic-tier decode GEMV on cb3g128 slots (host copies admitted at run time): LPR lanes per output row, each lane
// one 128-group at a time (48 B codes + fp16 scale); which = 0: gate/up rows of [w1; w3] (K = 5120), 1: down (K = 2304)
template <int LPR>
__global__ void __launch_bounds__(256) k_moe_cb3(const float* x, int which, const int* ids, const int* dslot,
                                                 const uint8_t* pool, size_t xbytes, const float* cb, float* out) {
    __shared__ float c[8];
    if (threadIdx.x < 8) c[threadIdx.x] = cb[threadIdx.x];
    __syncthreads();
    const int k = blockIdx.y % TOPK, r = blockIdx.y / TOPK;
    const int sl = dslot[ids[r * TOPK + k]];
    if (sl < 0) return;
    constexpr int RPB = 8 * (32 / LPR);
    const int row = blockIdx.x * RPB + threadIdx.x / LPR, sub = threadIdx.x % LPR;
    const int K = which ? INTER : D;
    const int NR = which ? D : 2 * INTER;
    if (row >= NR) return;
    MatView mv[3];
    expert_views(XF_CB3, pool + (size_t)sl * xbytes, mv);
    const int m = which ? 2 : (row < INTER ? 0 : 1), rr = which ? row : row % INTER;
    const int G = K / 128;
    const uint8_t* q = mv[m].q + (size_t)rr * G * 48;
    const __half* sc = (const __half*)mv[m].s + (size_t)rr * G;
    const float* xr = which ? x + ((size_t)r * TOPK + k) * INTER : x + (size_t)r * D;
    float acc = 0.f;
    // next group's 48 B issued before the current one is decoded (memory-level parallelism)
    uint4 f0 = make_uint4(0, 0, 0, 0), f1 = f0, hb = f0;
    if (sub < G) {
        const uint4* p4 = (const uint4*)(q + (size_t)sub * 48);
        f0 = p4[0], f1 = p4[1], hb = p4[2];
    }
    for (int g = sub; g < G; g += LPR) {
        uint4 c0 = f0, c1 = f1, ch = hb;
        if (g + LPR < G) {
            const uint4* p4 = (const uint4*)(q + (size_t)(g + LPR) * 48);
            f0 = p4[0], f1 = p4[1], hb = p4[2];
        }
        const uint32_t* fw0 = (const uint32_t*)&c0;
        const uint32_t* fw1 = (const uint32_t*)&c1;
        const uint32_t* h32 = (const uint32_t*)&ch;
        const float4* xg = (const float4*)(xr + g * 128);
        float d = 0.f;
#pragma unroll
        for (int f = 0; f < 4; f++) {
            uint32_t hw = h32[f];
#pragma unroll
            for (int j4 = 0; j4 < 8; j4++) {   // elements 32 f + 4 j4 .. + 3: field bytes 4 j4 .. 4 j4 + 3
                uint32_t fw = j4 < 4 ? fw0[j4] : fw1[j4 - 4];
                float4 xv = xg[8 * f + j4];
                int b0 = 4 * j4;
                d += c[((fw >> (2 * f)) & 3) | (((hw >> b0) & 1) << 2)] * xv.x +
                     c[((fw >> (8 + 2 * f)) & 3) | (((hw >> (b0 + 1)) & 1) << 2)] * xv.y +
                     c[((fw >> (16 + 2 * f)) & 3) | (((hw >> (b0 + 2)) & 1) << 2)] * xv.z +
                     c[((fw >> (24 + 2 * f)) & 3) | (((hw >> (b0 + 3)) & 1) << 2)] * xv.w;
            }
        }
        acc += d * __half2float(sc[g]);
    }
    for (int off = LPR / 2; off; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (sub == 0) {
        float* o = which ? out + ((size_t)r * TOPK + k) * D + row : out + ((size_t)r * TOPK + k) * 2 * INTER + row;
        *o = acc;
    }
}

// R19 [dsv4-r19-dmoe] v2: k_moe_cb3 with the block's x row staged in shared memory as bf16 (x is FP8-rounded by q_fp8,
// so bf16 holds it exactly): the fp32 x was read from L1 / L2 once per output row (512 B per 48 B of codes), the
// kernel's real cost. Same values, same expression order: bit-identical to k_moe_cb3
template <int LPR>
__global__ void __launch_bounds__(256) k_moe_cb3s(const float* x, int which, const int* ids, const int* dslot,
                                                  const uint8_t* pool, size_t xbytes, const float* cb, float* out) {
    __shared__ float c[8];
    __shared__ __align__(16) bf16 xs[D];
    const int k = blockIdx.y % TOPK, r = blockIdx.y / TOPK;
    const int sl = dslot[ids[r * TOPK + k]];
    if (sl < 0) return;
    if (threadIdx.x < 8) c[threadIdx.x] = cb[threadIdx.x];
    const int K = which ? INTER : D;
    const float* xr = which ? x + ((size_t)r * TOPK + k) * INTER : x + (size_t)r * D;
    for (int t = threadIdx.x; t < K; t += blockDim.x) xs[t] = __float2bfloat16(xr[t]);
    __syncthreads();
    constexpr int RPB = 8 * (32 / LPR);
    const int row = blockIdx.x * RPB + threadIdx.x / LPR, sub = threadIdx.x % LPR;
    const int NR = which ? D : 2 * INTER;
    if (row >= NR) return;
    MatView mv[3];
    expert_views(XF_CB3, pool + (size_t)sl * xbytes, mv);
    const int m = which ? 2 : (row < INTER ? 0 : 1), rr = which ? row : row % INTER;
    const int G = K / 128;
    const uint8_t* q = mv[m].q + (size_t)rr * G * 48;
    const __half* sc = (const __half*)mv[m].s + (size_t)rr * G;
    float acc = 0.f;
    uint4 f0 = make_uint4(0, 0, 0, 0), f1 = f0, hb = f0;
    if (sub < G) {
        const uint4* p4 = (const uint4*)(q + (size_t)sub * 48);
        f0 = p4[0], f1 = p4[1], hb = p4[2];
    }
    for (int g = sub; g < G; g += LPR) {
        uint4 c0 = f0, c1 = f1, ch = hb;
        if (g + LPR < G) {
            const uint4* p4 = (const uint4*)(q + (size_t)(g + LPR) * 48);
            f0 = p4[0], f1 = p4[1], hb = p4[2];
        }
        const uint32_t* fw0 = (const uint32_t*)&c0;
        const uint32_t* fw1 = (const uint32_t*)&c1;
        const uint32_t* h32 = (const uint32_t*)&ch;
        const uint2* xg = (const uint2*)(xs + g * 128);   // 4 bf16 per uint2
        float d = 0.f;
#pragma unroll
        for (int f = 0; f < 4; f++) {
            uint32_t hw = h32[f];
#pragma unroll
            for (int j4 = 0; j4 < 8; j4++) {
                uint32_t fw = j4 < 4 ? fw0[j4] : fw1[j4 - 4];
                const uint2 xb2 = xg[8 * f + j4];
                const float2 xa = __bfloat1622float2(*(const bf162*)&xb2.x), xc = __bfloat1622float2(*(const bf162*)&xb2.y);
                float4 xv = make_float4(xa.x, xa.y, xc.x, xc.y);
                int b0 = 4 * j4;
                d += c[((fw >> (2 * f)) & 3) | (((hw >> b0) & 1) << 2)] * xv.x +
                     c[((fw >> (8 + 2 * f)) & 3) | (((hw >> (b0 + 1)) & 1) << 2)] * xv.y +
                     c[((fw >> (16 + 2 * f)) & 3) | (((hw >> (b0 + 2)) & 1) << 2)] * xv.z +
                     c[((fw >> (24 + 2 * f)) & 3) | (((hw >> (b0 + 3)) & 1) << 2)] * xv.w;
            }
        }
        acc += d * __half2float(sc[g]);
    }
    for (int off = LPR / 2; off; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (sub == 0) {
        float* o = which ? out + ((size_t)r * TOPK + k) * D + row : out + ((size_t)r * TOPK + k) * 2 * INTER + row;
        *o = acc;
    }
}

// R16: the same GEMV on M23 pool slots (Maya's byte budget): gate / up rows cb2g128 (32 B per 128, no high bits),
// down rows cb3g128 (as k_moe_cb3). cb = the card's codebooks (cb3 at 0, cb2 at 16)
template <int LPR, int BITS>
__global__ void __launch_bounds__(256) k_moe_m23(const float* x, int which, const int* ids, const int* dslot,
                                                 const uint8_t* pool, size_t xbytes, const float* cb, float* out) {
    __shared__ float c[8];
    if (threadIdx.x < 8) c[threadIdx.x] = cb[(BITS == 3 ? 0 : 16) + (threadIdx.x < (1 << BITS) ? threadIdx.x : 0)];
    __syncthreads();
    const int k = blockIdx.y % TOPK, r = blockIdx.y / TOPK;
    const int sl = dslot[ids[r * TOPK + k]];
    if (sl < 0) return;
    constexpr int RPB = 8 * (32 / LPR);
    constexpr int GB = BITS == 3 ? 48 : 32;
    const int row = blockIdx.x * RPB + threadIdx.x / LPR, sub = threadIdx.x % LPR;
    const int K = which ? INTER : D;
    const int NR = which ? D : 2 * INTER;
    if (row >= NR) return;
    MatView mv[3];
    expert_views(XF_M23, pool + (size_t)sl * xbytes, mv);
    const int m = which ? 2 : (row < INTER ? 0 : 1), rr = which ? row : row % INTER;
    const int G = K / 128;
    const uint8_t* q = mv[m].q + (size_t)rr * G * GB;
    const __half* sc = (const __half*)mv[m].s + (size_t)rr * G;
    const float* xr = which ? x + ((size_t)r * TOPK + k) * INTER : x + (size_t)r * D;
    float acc = 0.f;
    uint4 f0 = make_uint4(0, 0, 0, 0), f1 = f0, hb = f0;
    if (sub < G) {
        const uint4* p4 = (const uint4*)(q + (size_t)sub * GB);
        f0 = p4[0], f1 = p4[1];
        if (BITS == 3) hb = p4[2];
    }
    for (int g = sub; g < G; g += LPR) {
        uint4 c0 = f0, c1 = f1, ch = hb;
        if (g + LPR < G) {
            const uint4* p4 = (const uint4*)(q + (size_t)(g + LPR) * GB);
            f0 = p4[0], f1 = p4[1];
            if (BITS == 3) hb = p4[2];
        }
        const uint32_t* fw0 = (const uint32_t*)&c0;
        const uint32_t* fw1 = (const uint32_t*)&c1;
        const uint32_t* h32 = (const uint32_t*)&ch;
        const float4* xg = (const float4*)(xr + g * 128);
        float d = 0.f;
#pragma unroll
        for (int f = 0; f < 4; f++) {
            uint32_t hw = BITS == 3 ? h32[f] : 0u;
#pragma unroll
            for (int j4 = 0; j4 < 8; j4++) {
                uint32_t fw = j4 < 4 ? fw0[j4] : fw1[j4 - 4];
                float4 xv = xg[8 * f + j4];
                int b0 = 4 * j4;
                d += c[((fw >> (2 * f)) & 3) | (((hw >> b0) & 1) << 2)] * xv.x +
                     c[((fw >> (8 + 2 * f)) & 3) | (((hw >> (b0 + 1)) & 1) << 2)] * xv.y +
                     c[((fw >> (16 + 2 * f)) & 3) | (((hw >> (b0 + 2)) & 1) << 2)] * xv.z +
                     c[((fw >> (24 + 2 * f)) & 3) | (((hw >> (b0 + 3)) & 1) << 2)] * xv.w;
            }
        }
        acc += d * __half2float(sc[g]);
    }
    for (int off = LPR / 2; off; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (sub == 0) {
        float* o = which ? out + ((size_t)r * TOPK + k) * D + row : out + ((size_t)r * TOPK + k) * 2 * INTER + row;
        *o = acc;
    }
}

// R16 v2: M23 pool GEMV, warps over output rows, the block's input row staged once in shared memory. A lane reads 4
// code bytes (one 32-bit word) of a 128-group's 32 field bytes per step (8 lanes = one group, a warp = 4 groups = 128 B
// coalesced), i.e. elements b + 32 f (b = its 4 bytes, f = 0..3); cb3 rows also take the group's high-bit words. A warp
// works on RPW rows at once (RPW independent loads in flight per step) and the block covers M23_RPB rows, so the input
// row is read from L2 once per M23_RPB rows. which = 0: gate / up rows (cb2, K = 5120), 1: down rows (cb3, K = 2304).
// grid (rows / M23_RPB, TOPK * R), 256 threads.
constexpr int M23_RPW = 4, M23_RPB = 8 * M23_RPW * 2;   // 64 rows per block
template <int BITS>
__global__ void __launch_bounds__(256) k_moe_m23w(const float* x, int which, const int* ids, const int* dslot,
                                                  const uint8_t* pool, size_t xbytes, const float* cb, float* out) {
    __shared__ __align__(16) float xs[D];
    __shared__ float c[8];   // codebook (broadcast reads; a register array would be indexed dynamically)
    const int k = blockIdx.y % TOPK, r = blockIdx.y / TOPK;
    const int sl = dslot[ids[r * TOPK + k]];
    if (sl < 0) return;
    const int K = which ? INTER : D;
    const float* xr = which ? x + ((size_t)r * TOPK + k) * INTER : x + (size_t)r * D;
    for (int i = threadIdx.x; i < K / 4; i += blockDim.x) ((float4*)xs)[i] = ((const float4*)xr)[i];
    if (threadIdx.x < 8) c[threadIdx.x] = cb[(BITS == 3 ? 0 : 16) + (threadIdx.x < (1 << BITS) ? threadIdx.x : 0)];
    __syncthreads();
    const int NR = which ? D : 2 * INTER;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    MatView mv[3];
    expert_views(XF_M23, pool + (size_t)sl * xbytes, mv);
    constexpr int GB = BITS == 3 ? 48 : 32;
    const int G = K / 128;
    const int gl = lane >> 3, wb = (lane & 7) * 4;   // group within the step, first field byte
    const int sh = (wb >> 3) * 8 + (wb & 7);
    for (int ch = 0; ch < M23_RPB / (8 * M23_RPW); ch++) {
        const int row0 = blockIdx.x * M23_RPB + (ch * 8 + warp) * M23_RPW;
        if (row0 >= NR) break;
        const uint8_t* q[M23_RPW];
        const __half* sc[M23_RPW];
#pragma unroll
        for (int j = 0; j < M23_RPW; j++) {
            const int row = row0 + j;
            const int m = which ? 2 : (row < INTER ? 0 : 1), rr = which ? row : row % INTER;
            q[j] = mv[m].q + (size_t)rr * G * GB;
            sc[j] = (const __half*)mv[m].s + (size_t)rr * G;
        }
        float acc[M23_RPW] = {};
        for (int g0 = 0; g0 < G; g0 += 4) {
            const int g = g0 + gl;
            if (g >= G) break;
            uint32_t fw[M23_RPW], hw[M23_RPW];
            float s[M23_RPW];
#pragma unroll
            for (int j = 0; j < M23_RPW; j++) {
                const uint8_t* p = q[j] + (size_t)g * GB;
                fw[j] = *(const uint32_t*)(p + wb);
                hw[j] = 0;
                if (BITS == 3) {   // high bits of elements wb + i + 32 f: byte 32 + (wb >> 3) + 4 f, bit (wb & 7) + i
                    const uint4 h = *(const uint4*)(p + 32);
                    hw[j] = ((h.x >> sh) & 15) | (((h.y >> sh) & 15) << 4) | (((h.z >> sh) & 15) << 8) |
                            (((h.w >> sh) & 15) << 12);
                }
                s[j] = __half2float(sc[j][g]);
            }
            const float* xg = xs + g * 128 + wb;
#pragma unroll
            for (int f = 0; f < 4; f++) {
                const float4 xv = *(const float4*)(xg + 32 * f);
#pragma unroll
                for (int j = 0; j < M23_RPW; j++) {
                    float d = 0.f;
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        int code = (fw[j] >> (8 * i + 2 * f)) & 3;
                        if (BITS == 3) code |= ((hw[j] >> (4 * f + i)) & 1) << 2;
                        const float xe = i == 0 ? xv.x : i == 1 ? xv.y : i == 2 ? xv.z : xv.w;
                        d += c[code] * xe;
                    }
                    acc[j] += d * s[j];
                }
            }
        }
#pragma unroll
        for (int j = 0; j < M23_RPW; j++) {
            const float a = warp_sum(acc[j]);
            const int row = row0 + j;
            if (lane == 0 && row < NR) {
                float* o = which ? out + ((size_t)r * TOPK + k) * D + row : out + ((size_t)r * TOPK + k) * 2 * INTER + row;
                *o = a;
            }
        }
    }
}

// out[r] = sum over the 6 (in routing order; cold ones are zero here) + ycold[r] + shared[r]
__global__ void k_set_int(int* p, int v) { *p = v; }   // device table entry (no H2D copy: review r4 M4)
__global__ void k_moe_comb(const float* y, const float* ycold, const float* shared, int R, float* out) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)R * D) return;
    int r = (int)(t / D), d = (int)(t % D);
    float s = 0.f;
    for (int k = 0; k < TOPK; k++) s += y[((size_t)r * TOPK + k) * D + d];
    out[t] = s + ycold[t] + shared[t];
}


// run-ahead decode handshake (round 3): the GPU raises a host-mapped flag when a layer's routing pack has reached
// the host, and the combine kernel waits on a host-mapped flag for the CPU lane's result (read over PCIe)
// publish the predicted expert ids of a decode layer to host-mapped memory, then raise its flag (L3 prefetch)
__global__ void k_pub_ids(const int* ids, int n, volatile int* hids, volatile int* f, int seq) {
    for (int k = 0; k < n * TOPK; k++) hids[k] = ids[k];
    __threadfence_system();
    *f = seq;
    __threadfence_system();
}
// R10 step graph: wait (one thread) for a host-mapped flag set by the decode thread (Engram rows staged), ~60 s bound
__global__ void k_hwait(const volatile int* f, int seq) {
    uint64_t t0, t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t0));
    for (unsigned it = 0; *f != seq; it++) {
        __nanosleep(100);
        if ((it & 4095) == 4095) {
            asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
            if (t - t0 > 30000000000ull) __trap();
        }
    }
    __threadfence_system();
}
__global__ void k_flag(volatile int* f, int seq) {
    __threadfence_system();
    *f = seq;
    __threadfence_system();
}
__global__ void k_moe_comb_wait(const float* y, const volatile float* ycold, const volatile int* yflag, int seq,
                                const float* shared, int R, float* out) {
    if (threadIdx.x == 0) {   // review r9 M4: a lane that never answers traps after ~60 s of %globaltimer
        uint64_t t0, t;
        asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t0));
        for (unsigned it = 0; *yflag != seq; it++) {
            __nanosleep(200);
            if ((it & 4095) == 4095) {
                asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
                if (t - t0 > 60000000000ull) __trap();
            }
        }
    }
    __syncthreads();
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)R * D) return;
    int r = (int)(t / D), d = (int)(t % D);
    float s = 0.f;
    for (int k = 0; k < TOPK; k++) s += y[((size_t)r * TOPK + k) * D + d];
    out[t] = s + ycold[t] + shared[t];
}

// R10: k_moe_comb_wait + k_hc_post in one launch (elementwise, the same arithmetic): out = experts + lane + shared,
// then h = post * out + comb^T h for the row
__global__ void k_moe_comb_post(const float* y, const volatile float* ycold, const volatile int* yflag, int seq,
                                const float* shared, int R, float* out, float* h, const float* post, const float* comb) {
    if (threadIdx.x == 0) {
        uint64_t t0, t;
        asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t0));
        for (unsigned it = 0; *yflag != seq; it++) {
            __nanosleep(200);
            if ((it & 4095) == 4095) {
                asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
                if (t - t0 > 60000000000ull) __trap();
            }
        }
    }
    __syncthreads();
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)R * D) return;
    int r = (int)(t / D), d = (int)(t % D);
    float s = 0.f;
    for (int k = 0; k < TOPK; k++) s += y[((size_t)r * TOPK + k) * D + d];
    const float o = s + ycold[t] + shared[t];
    out[t] = o;
    float* hi = h + (size_t)r * HC * D + d;
    float rr[4] = {hi[0], hi[D], hi[2 * D], hi[3 * D]};
    const float* c = comb + (size_t)r * 16;
    for (int k = 0; k < 4; k++)
        hi[(size_t)k * D] = post[r * HC + k] * o + (c[0 * 4 + k] * rr[0] + c[1 * 4 + k] * rr[1] + c[2 * 4 + k] * rr[2] +
                                                    c[3 * 4 + k] * rr[3]);
}

// [dsv4-dec7-combh] k_moe_comb_post with C columns per thread (the engine uses C = 4): the lane's rows (host-mapped) as
// C / 4 16-byte volatile loads per thread and C x fewer blocks polling the lane flag over PCIe (the old kernel's
// post-flag time grew ~10 us per row); the per-element expression is the old one (bit-identical)
template <int C>
__global__ void k_moe_comb_post_v(const float* y, const volatile float* ycold, const volatile int* yflag, int seq,
                                  const float* shared, int R, float* out, float* h, const float* post, const float* comb) {
    if (threadIdx.x == 0) {
        uint64_t t0, t;
        asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t0));
        for (unsigned it = 0; *yflag != seq; it++) {
            __nanosleep(200);
            if ((it & 4095) == 4095) {
                asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
                if (t - t0 > 60000000000ull) __trap();
            }
        }
    }
    static_assert(C % 4 == 0 && D % C == 0, "a thread's C columns: whole 16-byte loads inside one row");
    __syncthreads();
    const size_t t0 = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * C;   // D % C == 0: a thread stays in one row
    if (t0 >= (size_t)R * D) return;
    float yc[C];
#pragma unroll
    for (int u = 0; u < C / 4; u++)
        asm volatile("ld.volatile.global.v4.f32 {%0,%1,%2,%3}, [%4];"
                     : "=f"(yc[4 * u]), "=f"(yc[4 * u + 1]), "=f"(yc[4 * u + 2]), "=f"(yc[4 * u + 3])
                     : "l"((const float*)ycold + t0 + 4 * u) : "memory");
    const int r = (int)(t0 / D);
    const float* c = comb + (size_t)r * 16;
#pragma unroll
    for (int q = 0; q < C; q++) {
        const size_t t = t0 + q;
        const int d = (int)(t % D);
        float s = 0.f;
        for (int k = 0; k < TOPK; k++) s += y[((size_t)r * TOPK + k) * D + d];
        const float o = s + yc[q] + shared[t];
        out[t] = o;
        float* hi = h + (size_t)r * HC * D + d;
        float rr[4] = {hi[0], hi[D], hi[2 * D], hi[3 * D]};
        for (int k = 0; k < 4; k++)
            hi[(size_t)k * D] = post[r * HC + k] * o + (c[0 * 4 + k] * rr[0] + c[1 * 4 + k] * rr[1] + c[2 * 4 + k] * rr[2] +
                                                        c[3 * 4 + k] * rr[3]);
    }
}

// R21 [dsv4-r21-defer] KL emulation (opt-in DSV4_DEFER=1): as k_moe_comb_post without the CPU lane's rows; their
// stream terms post_k * ycold go to ydef [R][HC][D] and join at the next layer after its attention post (k_def_add)
__global__ void k_moe_comb_post_def(const float* y, const volatile float* ycold, const volatile int* yflag, int seq,
                                    const float* shared, int R, float* out, float* h, const float* post, const float* comb,
                                    float* ydef) {
    if (threadIdx.x == 0) {
        uint64_t t0, t;
        asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t0));
        for (unsigned it = 0; *yflag != seq; it++) {
            __nanosleep(200);
            if ((it & 4095) == 4095) {
                asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
                if (t - t0 > 60000000000ull) __trap();
            }
        }
    }
    __syncthreads();
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)R * D) return;
    int r = (int)(t / D), d = (int)(t % D);
    float s = 0.f;
    for (int k = 0; k < TOPK; k++) s += y[((size_t)r * TOPK + k) * D + d];
    const float o = s + shared[t];
    const float yc = ycold[t];
    out[t] = o;
    float* hi = h + (size_t)r * HC * D + d;
    float rr[4] = {hi[0], hi[D], hi[2 * D], hi[3 * D]};
    const float* c = comb + (size_t)r * 16;
    for (int k = 0; k < 4; k++) {
        hi[(size_t)k * D] = post[r * HC + k] * o + (c[0 * 4 + k] * rr[0] + c[1 * 4 + k] * rr[1] + c[2 * 4 + k] * rr[2] +
                                                    c[3 * 4 + k] * rr[3]);
        ydef[((size_t)r * HC + k) * D + d] = post[r * HC + k] * yc;
    }
}
// h[r][k] += sum_j comb[r][j][k] * ydef[r][j]: the deferred lane terms through the attention post's stream mix
__global__ void k_def_add(float* h, const float* ydef, const float* comb, int n) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)n * D) return;
    int r = (int)(t / D), d = (int)(t % D);
    const float* q = ydef + (size_t)r * HC * D + d;
    float v[4] = {q[0], q[D], q[2 * D], q[3 * D]};
    const float* c = comb + (size_t)r * 16;
    float* hi = h + (size_t)r * HC * D + d;
    for (int k = 0; k < 4; k++) hi[(size_t)k * D] += c[0 * 4 + k] * v[0] + c[1 * 4 + k] * v[1] + c[2 * 4 + k] * v[2] + c[3 * 4 + k] * v[3];
}

// R10 diagnostics: order-independent bit hash of n u32 words (sum of mixed words) into *out
__global__ void k_bhash(const uint32_t* p, int n, unsigned long long* out) {
    __shared__ unsigned long long red[32];
    unsigned long long h = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        unsigned long long v = (unsigned long long)p[i] * 0x9E3779B97F4A7C15ull + (unsigned long long)i;
        h += v ^ (v >> 29);
    }
    for (int o = 16; o; o >>= 1) h += __shfl_xor_sync(0xffffffffu, h, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = h;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long t = 0;
        for (int w = 0; w < (int)(blockDim.x >> 5); w++) t += red[w];
        *out = t;
    }
}
// ---------------------------------------------------------------- MoE prefill helpers

// gather rows of a bf16 matrix: dst[i] = src[rows[i]] (K columns)
__global__ void k_gather_bf16(const bf16* src, int K, const int* rows, int m, bf16* dst) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)m * K / 8) return;
    int i = (int)(t / (K / 8)), c = (int)(t % (K / 8));
    ((uint4*)(dst + (size_t)i * K))[c] = ((const uint4*)(src + (size_t)rows[i] * K))[c];
}
// prefill GLU: H [m][4608] fp32 -> h [m][2304] fp32 (weighted), then q_fp8 separately
__global__ void k_glu_rows(const float* H, const float* wt, int m, float* h) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)m * INTER) return;
    int i = (int)(t / INTER), c = (int)(t % INTER);
    float g = fminf(H[(size_t)i * 2 * INTER + c], CLAMP);
    float u = fminf(fmaxf(H[(size_t)i * 2 * INTER + INTER + c], -CLAMP), CLAMP);
    h[t] = wt[i] * (siluf_(g) * u);
}
// out[rows[i]] += Y[i] (one expert at a time in stream order: no two rows of an expert share a token)
__global__ void k_scatter_add(const float* Y, const int* rows, int m, float* out) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)m * D) return;
    int i = (int)(t / D), d = (int)(t % D);
    out[(size_t)rows[i] * D + d] += Y[t];
}
__global__ void k_cmp_bits(const float* a, const float* b, size_t n, unsigned* cnt) {   // R11 check
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n && __float_as_uint(a[t]) != __float_as_uint(b[t])) atomicAdd(cnt, 1u);
}
__global__ void k_add4(float4* a, const float4* b, size_t n) {   // [dsv4-pf9-hsum] a += b, 16 B per thread
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    float4 x = a[t];
    const float4 y = b[t];
    x.x += y.x; x.y += y.y; x.z += y.z; x.w += y.w;
    a[t] = x;
}
__global__ void k_add(float* a, const float* b, size_t n) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) a[t] += b[t];
}

// ---------------------------------------------------------------- engram

// packed rows -> emb [n * 24][256] fp32. Row format egB: 4 fp16 scales (per 64) then 256 B-bit codes (element i at
// bits [B i, B i + B) little endian). fmt[r] picks the codebook (bits) of row r.
__global__ void k_eg_decode(const uint8_t* rows, const int* roff, const uint8_t* rbits, int nrows,
                            const float* cb2, const float* cb3, const float* cb4, float* emb) {
    int r = blockIdx.x, t = threadIdx.x;   // 256 threads
    if (r >= nrows) return;
    const uint8_t* p = rows + roff[r];
    int bits = rbits[r];
    const float* cb = bits == 2 ? cb2 : bits == 3 ? cb3 : cb4;
    float sc = __half2float(((const __half*)p)[t >> 6]);
    const uint8_t* c = p + 8;
    int bit = bits * t;
    uint32_t w = c[bit >> 3] | ((uint32_t)c[(bit >> 3) + 1] << 8);
    int code = (w >> (bit & 7)) & ((1 << bits) - 1);
    emb[(size_t)r * EG_HD + t] = cb[code] * sc;
}
// FP8 engram rows (native, check mode): 256 e4m3 + 8 e8m0 scales per row
__global__ void k_eg_decode_fp8(const uint8_t* rows, int nrows, float* emb) {
    int r = blockIdx.x, t = threadIdx.x;
    const uint8_t* p = rows + (size_t)r * 264;
    __half_raw h = __nv_cvt_fp8_to_halfraw(p[t], __NV_E4M3);
    emb[(size_t)r * EG_HD + t] = __half2float(__half(h)) * e8m0f(p[256 + (t >> 5)]);
}
// gate: kv [n][25600] (keys 4 x 5120, value 5120); h [n][4][5120] += sigmoid(ssqrt(dot)) * value
// block per (row, stream)
__global__ void k_eg_gate(const float* kv, const float* qk, float* h) {
    __shared__ float red[3][32];
    int i = blockIdx.x, s = blockIdx.y, t = threadIdx.x;
    const float* key = kv + (size_t)i * (HC + 1) * D + (size_t)s * D;
    const float* val = kv + (size_t)i * (HC + 1) * D + (size_t)HC * D;
    float* hs = h + ((size_t)i * HC + s) * D;
    const float* q = qk + (size_t)s * D;
    float a = 0.f, b = 0.f, c = 0.f;
    for (int d = t; d < D; d += blockDim.x) {
        a += hs[d] * hs[d];
        b += key[d] * key[d];
        c += hs[d] * q[d] * key[d];
    }
    a = warp_sum(a);
    b = warp_sum(b);
    c = warp_sum(c);
    if ((t & 31) == 0) {
        red[0][t >> 5] = a;
        red[1][t >> 5] = b;
        red[2][t >> 5] = c;
    }
    __syncthreads();
    __shared__ float gate_s;
    if (t < 32) {
        int nw = blockDim.x >> 5;
        float A = warp_sum(t < nw ? red[0][t] : 0.f), B = warp_sum(t < nw ? red[1][t] : 0.f),
              C = warp_sum(t < nw ? red[2][t] : 0.f);
        if (t == 0) {
            float rstd = rsqrtf(A / (float)D + EPS) * rsqrtf(B / (float)D + EPS);
            float dot = C * rstd * 0.013975424859373685f;   // 5120^-0.5
            float sq = sqrtf(fmaxf(fabsf(dot), 1e-6f));
            gate_s = sigmoidf_(copysignf(sq, dot));
        }
    }
    __syncthreads();
    float g = gate_s;
    for (int d = t; d < D; d += blockDim.x) hs[d] += g * val[d];
}

// ---------------------------------------------------------------- misc

// embedding rows (bf16 host-pinned table rows copied to dev) -> h [n][4][5120] = row (all 4 streams), pre = e0
__global__ void k_embed(const bf16* rows, int n, float* h, float* pre) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)n * D) return;
    int i = (int)(t / D), d = (int)(t % D);
    float v = __bfloat162float(rows[t]);
    for (int s = 0; s < HC; s++) h[((size_t)i * HC + s) * D + d] = v;
    if (d < HC) pre[i * HC + d] = d == 0 ? 1.f : 0.f;
}
// argmax over V (first max), block of 1024 per row
__global__ void k_argmax(const float* x, int V_, int* out) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    const float* r = x + (size_t)blockIdx.x * V_;
    float best = -INFINITY;
    int b = 0;
    for (int j = threadIdx.x; j < V_; j += 1024)
        if (r[j] > best) {
            best = r[j];
            b = j;
        }
    bv[threadIdx.x] = best;
    bi[threadIdx.x] = b;
    __syncthreads();
    for (int o = 512; o; o >>= 1) {
        if (threadIdx.x < o) {
            float ov = bv[threadIdx.x + o];
            int oi = bi[threadIdx.x + o];
            if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = ov;
                bi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = bi[0];
}
// log-softmax NLL of the target per row (ppl check): block per row
__global__ void k_nll(const float* x, int V_, const int* tgt, float* nll) {
    __shared__ float red[32];
    __shared__ float mx_s;
    const float* r = x + (size_t)blockIdx.x * V_;
    float m = -INFINITY;
    for (int j = threadIdx.x; j < V_; j += blockDim.x) m = fmaxf(m, r[j]);
    m = warp_max(m);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : -INFINITY;
        t = warp_max(t);
        if (threadIdx.x == 0) mx_s = t;
    }
    __syncthreads();
    float M_ = mx_s, s = 0.f;
    for (int j = threadIdx.x; j < V_; j += blockDim.x) s += expf(r[j] - M_);
    s = warp_sum(s);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int w = 0; w < (int)(blockDim.x >> 5); w++) t += red[w];
        nll[blockIdx.x] = logf(t) + M_ - r[tgt[blockIdx.x]];
    }
}

// ---------------------------------------------------------------- DSpark (draft head) and verify helpers

// mean over the hc streams of rows [row0, row0 + n) of h -> out [n][ldo] (DSpark main hidden of a target layer)
__global__ void k_hc_mean(const float* h, int row0, int n, float* out, int ldo) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)n * D) return;
    int i = (int)(t / D), d = (int)(t % D);
    const float* r = h + (size_t)(row0 + i) * HC * D + d;
    out[(size_t)i * ldo + d] = (r[0] + r[D] + r[2 * D] + r[3 * D]) * 0.25f;
}
// DSpark router: logits [n][128] -> top-3 of sqrt(softplus) + bias in the TOPK-wide id layout; ids 3..5 = 128
// (a table entry that is always -1: the expert kernels write zeros for it); warp per row
__global__ void k_route_d(const float* logits, const float* bias, int n, int* ids, float* w) {
    constexpr int NEd = 128, Kd = 3;
    int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (i >= n) return;
    float sv[NEd / 32], cv[NEd / 32];
    for (int j = 0; j < NEd / 32; j++) {
        int e = j * 32 + lane;
        sv[j] = sqrtf(softplusf_(logits[(size_t)i * NEd + e]));
        cv[j] = sv[j] + bias[e];
    }
    int sel[Kd];
    float ss[Kd], sum = 0.f;
    for (int k = 0; k < Kd; k++) {
        float best = -INFINITY;
        int be = 1 << 30;
        for (int j = 0; j < NEd / 32; j++) {
            int e = j * 32 + lane;
            if (cv[j] > best || (cv[j] == best && e < be)) { best = cv[j]; be = e; }
        }
        for (int o = 16; o; o >>= 1) {
            float ob = __shfl_xor_sync(0xffffffffu, best, o);
            int oe = __shfl_xor_sync(0xffffffffu, be, o);
            if (ob > best || (ob == best && oe < be)) { best = ob; be = oe; }
        }
        if (be >= NEd) be = k;   // review r8 L1
        sel[k] = be;
        int jj = be / 32;
        ss[k] = __shfl_sync(0xffffffffu, sv[jj], be & 31);
        if ((be & 31) == lane) cv[jj] = -INFINITY;
    }
    for (int k = 0; k < Kd; k++) sum += ss[k];
    if (lane == 0)
        for (int k = 0; k < TOPK; k++) {
            ids[i * TOPK + k] = k < Kd ? sel[k] : NEd;
            w[i * TOPK + k] = k < Kd ? ss[k] / (sum + 1e-20f) * RSF : 0.f;
        }
}
// Markov bias of the draft head, one row: logits[v] += Wm[v] . emb[tok]; also emb[tok] -> embf [256] (fp32);
// warp per vocabulary row (256 bf16 = 32 lanes x 8)
__global__ void k_markov(const bf16* __restrict__ Wm, const bf16* __restrict__ emb, const int* tok, float* logits,
                         float* embf) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int tk = *tok;
    uint4 ev = *(const uint4*)(emb + (size_t)tk * 256 + lane * 8);
    const bf16* eb = (const bf16*)&ev;
    if (blockIdx.x == 0 && threadIdx.x < 32)
        for (int j = 0; j < 8; j++) embf[lane * 8 + j] = __bfloat162float(eb[j]);
    if (row >= V) return;
    uint4 wv = *(const uint4*)(Wm + (size_t)row * 256 + lane * 8);
    const bf16* wb = (const bf16*)&wv;
    float a = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) a += __bfloat162float(wb[j]) * __bfloat162float(eb[j]);
    a = warp_sum(a);
    if (lane == 0) logits[row] += a;
}
// confidence of each draft row: proj [5120 + 256] . cat(x [5120], embf [256]); block per row
// R20 diagnostic: spin for ns nanoseconds (one thread): a known delay on a stream (DSV4_SPIN_US)
__global__ void k_spin(long long ns) {
    const long long t0 = clock64();
    const long long cyc = ns * 17 / 10;   // ~1.7 GHz
    while (clock64() - t0 < cyc) {}
}
// R20 [dsv4-r20-dssel]: per draft row: top-1 logit, top-2 logit (the second entry in value order, ties count), log-sum-exp
__device__ __forceinline__ void top2_merge(float& a, float& b, float a2, float b2) {
    const float hi = fmaxf(a, a2), lo = fmaxf(fminf(a, a2), fmaxf(b, b2));
    a = hi;
    b = lo;
}
__global__ void __launch_bounds__(1024) k_dstats(const float* lg, int V_, float* o) {
    __shared__ float s1[32], s2[32], s3[32];
    __shared__ float m_s;
    const float* x = lg + (size_t)blockIdx.x * V_;
    float a = -INFINITY, b = -INFINITY;
    for (int j = threadIdx.x; j < V_; j += blockDim.x) {
        const float v = x[j];
        if (v > a) { b = a; a = v; } else if (v > b) b = v;
    }
    for (int off = 16; off; off >>= 1) top2_merge(a, b, __shfl_xor_sync(0xffffffffu, a, off), __shfl_xor_sync(0xffffffffu, b, off));
    const int w = threadIdx.x >> 5, ln = threadIdx.x & 31;
    if (!ln) { s1[w] = a; s2[w] = b; }
    __syncthreads();
    if (w == 0) {
        a = s1[ln];
        b = s2[ln];
        for (int off = 16; off; off >>= 1) top2_merge(a, b, __shfl_xor_sync(0xffffffffu, a, off), __shfl_xor_sync(0xffffffffu, b, off));
        if (!ln) {
            m_s = a;
            o[blockIdx.x * 3] = a;
            o[blockIdx.x * 3 + 1] = b;
        }
    }
    __syncthreads();
    const float m = m_s;
    float se = 0.f;
    for (int j = threadIdx.x; j < V_; j += blockDim.x) se += __expf(x[j] - m);
    se = warp_sum(se);
    if (!ln) s3[w] = se;
    __syncthreads();
    if (w == 0) {
        float t = warp_sum(s3[ln]);
        if (!ln) o[blockIdx.x * 3 + 2] = m + logf(t);
    }
}
__global__ void k_conf(const float* x, const float* embf, const float* proj, float* conf) {
    __shared__ float red[32];
    const int i = blockIdx.x;
    float a = 0.f;
    for (int j = threadIdx.x; j < D + 256; j += blockDim.x)
        a += proj[j] * (j < D ? x[(size_t)i * D + j] : embf[i * 256 + j - D]);
    a = warp_sum(a);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = a;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int w = 0; w < (int)(blockDim.x >> 5); w++) t += red[w];
        conf[i] = t;
    }
}
// indexer scores for a few query rows in a fixed order (does not depend on the number of rows, unlike a cuBLAS
// GEMM): S[q][t] = q_b [q][128] . k [t][128] (bf16 in, fp32 sums); warp per key, lane = 4 dims, tree sum
__global__ void k_idx_score(const bf16* __restrict__ qb, int nq, const bf16* __restrict__ ik, int T, float* S) {
    const int t = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (t >= T) return;
    uint2 kv = *(const uint2*)(ik + (size_t)t * IDIM + lane * 4);
    const bf16* kb = (const bf16*)&kv;
    float k0 = __bfloat162float(kb[0]), k1 = __bfloat162float(kb[1]), k2 = __bfloat162float(kb[2]),
          k3 = __bfloat162float(kb[3]);
    for (int q = 0; q < nq; q++) {
        uint2 qv = *(const uint2*)(qb + (size_t)q * IDIM + lane * 4);
        const bf16* b = (const bf16*)&qv;
        float a = __bfloat162float(b[0]) * k0 + __bfloat162float(b[1]) * k1 + __bfloat162float(b[2]) * k2 +
                  __bfloat162float(b[3]) * k3;
        a = warp_sum(a);
        if (lane == 0) S[(size_t)q * T + t] = a;
    }
}

// [dsv4-ctx1-sr] decode index scores fused with their reduce (no S tensor), bit-equal to k_idx_score + k_idx_reduce:
// score[i][c] = sum_h relu(q_{i,h} . k) * (w_{i,h} / 64), -inf where the key is not visible to row i. One thread per
// key: the key in registers (fp32), each 128-dim dot as warp_sum's butterfly (32 partials of 4 dims in k_idx_score's
// expression, then pairs at distance 16, 8, 4, 2, 1: the value every lane of the warp form holds), the heads summed in
// order as k_idx_reduce; the row's 32 queries staged in shared memory as fp32 (exact widening).
// CAND false: c = key t (0 .. T-1), rows 0 .. nb-1 looped inside the block (the key read once).
// [dsv4-ctx1-cand] CAND true: one row per blockIdx.y; entry c of row i is key blist[i][c / 8] * 8 + c % 8 (layer 20's
// candidate blocks in index order, -1 = none): score [nb][ldb * 8] compact, every entry written.
constexpr int IDXSR_NT = 128;
template <bool CAND>
__global__ void __launch_bounds__(IDXSR_NT) k_idx_sr(const bf16* __restrict__ qb, const float* __restrict__ w, int nb,
                                                     const bf16* __restrict__ ik, int T, int pos0, int r, float* score,
                                                     int lds, const int* __restrict__ blist, int ldb) {
    __shared__ __align__(16) float qs[IH * IDIM];
    __shared__ float ws[IH];
    const int c = blockIdx.x * IDXSR_NT + threadIdx.x;
    const int i0 = CAND ? blockIdx.y : 0, i1 = CAND ? i0 + 1 : nb;
    float kf[IDIM];
    int have = -1;
    for (int i = i0; i < i1; i++) {
        __syncthreads();   // the previous row's queries are no longer read
        for (int j = threadIdx.x; j < IH * IDIM; j += IDXSR_NT) qs[j] = __bfloat162float(qb[(size_t)i * IH * IDIM + j]);
        if (threadIdx.x < IH) ws[threadIdx.x] = w[i * IH + threadIdx.x] * 0.015625f;   // 128^-.5 32^-.5
        __syncthreads();
        int t = c;
        if (CAND) {
            if (c >= ldb * CAND_BS) continue;
            const int b = blist[(size_t)i * ldb + c / CAND_BS];
            t = b < 0 ? -1 : b * CAND_BS + c % CAND_BS;
        } else if (c >= T) {
            continue;
        }
        const int vis = (pos0 + i + 1) / r;
        if (t < 0 || t >= T || t >= vis) {
            score[(size_t)i * lds + c] = -INFINITY;
            continue;
        }
        if (have != t) {
            const uint4* kp = (const uint4*)(ik + (size_t)t * IDIM);
#pragma unroll
            for (int v = 0; v < IDIM / 8; v++) {
                uint4 u = kp[v];
                const bf16* kb = (const bf16*)&u;
#pragma unroll
                for (int e = 0; e < 8; e++) kf[v * 8 + e] = __bfloat162float(kb[e]);
            }
            have = t;
        }
        float s = 0.f;
        for (int h = 0; h < IH; h++) {
            const float4* q4 = (const float4*)(qs + h * IDIM);
            float p[32];
#pragma unroll
            for (int L = 0; L < 32; L++) {
                const float4 q = q4[L];
                p[L] = q.x * kf[4 * L] + q.y * kf[4 * L + 1] + q.z * kf[4 * L + 2] + q.w * kf[4 * L + 3];
            }
#pragma unroll
            for (int L = 0; L < 16; L++) p[L] += p[L + 16];
#pragma unroll
            for (int L = 0; L < 8; L++) p[L] += p[L + 8];
#pragma unroll
            for (int L = 0; L < 4; L++) p[L] += p[L + 4];
            p[0] += p[2];
            p[1] += p[3];
            p[0] += p[1];
            s += fmaxf(p[0], 0.f) * ws[h];
        }
        score[(size_t)i * lds + c] = s;
    }
}
// [dsv4-ctx1-cand] candidate blocks of each row (mask cand [nb][ldc], at most ldb set) -> blist [nb][ldb] in index
// order, -1 after the last; one block of 1024 threads per row
__global__ void __launch_bounds__(1024) k_cand_list(const uint8_t* cand, int ldc, int nbk, int* blist, int ldb) {
    __shared__ int cnt[1024];
    const uint8_t* m = cand + (size_t)blockIdx.x * ldc;
    int* o = blist + (size_t)blockIdx.x * ldb;
    const int t = threadIdx.x, per = (nbk + 1023) / 1024, j0 = t * per, j1 = min(nbk, j0 + per);
    int n = 0;
    for (int j = j0; j < j1; j++) n += m[j] != 0;
    cnt[t] = n;
    __syncthreads();
    for (int d = 1; d < 1024; d <<= 1) {
        int v = t >= d ? cnt[t - d] : 0;
        __syncthreads();
        cnt[t] += v;
        __syncthreads();
    }
    int b = cnt[t] - n;
    const int total = cnt[1023];
    for (int j = j0; j < j1; j++)
        if (m[j] && b < ldb) o[b++] = j;
    for (int j = total + t; j < ldb; j += 1024) o[j] = -1;
}

// [dsv4-ctx2-idxf] prefill index scores on tensor cores fused with the head reduce (no S tensor, which cost 32 fp32 per
// (query, key) pair written by the GEMM and read back by k_idx_reduce): block = IPF_R query rows x every key up to the
// block's last visible one, in tiles of IPF_TK keys (cp.async double buffer); warp (row w / 2, key half w % 2): M = the
// row's 32 heads (2 m16 tiles, A fragments in registers for the whole key loop), N = 32 keys (4 n8 tiles), K = 128 (8
// k16 steps); bf16 mma m16n8k16, fp32 accumulate; epilogue relu(s) x w / 64 over the thread's 4 heads, then xor 4 / 8 /
// 16 lanes. Order-only against cuBLAS + k_idx_reduce (other summation order of the dots and of the heads). Writes
// score[i][j] for j < T: the sum, -inf where j >= vis(i) = (pos0 + i + 1) / r.
constexpr int IPF_R = 4, IPF_TK = 64, IPF_LD = IDIM + 8;   // smem key pitch 272 B: conflict-free ldmatrix rows
__device__ __forceinline__ void mma16816_bf(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void ipf_ldsm_x4(uint32_t* r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((uint32_t)__cvta_generic_to_shared(p)));
}
__global__ void __launch_bounds__(256, 2) k_idx_pf(const bf16* __restrict__ qb, const float* __restrict__ w, int nb,
                                                   const bf16* __restrict__ ik, int T, int pos0, int r, float* score,
                                                   int lds) {
    __shared__ __align__(16) bf16 ks[2][IPF_TK * IPF_LD];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
    const int rb = blockIdx.x * IPF_R, i = rb + (warp >> 1), kh = warp & 1;
    const int last = min(nb, rb + IPF_R) - 1;
    const int kend = min(T, (pos0 + last + 1) / r);   // keys any row of the block sees
    const int ntile = (kend + IPF_TK - 1) / IPF_TK;
    uint32_t a[2][8][4];
    if (i < nb) {
        const bf16* q = qb + (size_t)i * IH * IDIM;
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int k = 0; k < 8; k++) {
                const int h = 16 * mt + g, d = 16 * k + 2 * t4;
                a[mt][k][0] = *(const uint32_t*)(q + (size_t)h * IDIM + d);
                a[mt][k][1] = *(const uint32_t*)(q + (size_t)(h + 8) * IDIM + d);
                a[mt][k][2] = *(const uint32_t*)(q + (size_t)h * IDIM + d + 8);
                a[mt][k][3] = *(const uint32_t*)(q + (size_t)(h + 8) * IDIM + d + 8);
            }
    } else {
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int k = 0; k < 8; k++) a[mt][k][0] = a[mt][k][1] = a[mt][k][2] = a[mt][k][3] = 0u;
    }
    float wv[4] = {0.f, 0.f, 0.f, 0.f};
    if (i < nb)
        for (int e = 0; e < 4; e++) wv[e] = w[(size_t)i * IH + 8 * e + g] * 0.015625f;   // heads g, g + 8, g + 16, g + 24
    const int vis = i < nb ? (pos0 + i + 1) / r : 0;
    auto load = [&](int tile, int buf) {   // 64 keys x 256 B = 1024 chunks of 16 B, 4 per thread
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const int ch = tid + 256 * u, kr = ch >> 4, c16 = ch & 15, key = tile * IPF_TK + kr;
            bf16* dst = &ks[buf][kr * IPF_LD + c16 * 8];
            if (key < T) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned)__cvta_generic_to_shared(dst)),
                             "l"(ik + (size_t)key * IDIM + c16 * 8));
            } else {
                *(uint4*)dst = make_uint4(0, 0, 0, 0);
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    if (ntile > 0) load(0, 0);
    for (int tile = 0; tile < ntile; tile++) {
        const int buf = tile & 1;
        if (tile + 1 < ntile) {
            load(tile + 1, buf ^ 1);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        float acc[2][4][4];
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int nt = 0; nt < 4; nt++) acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f;
        const bf16* kb = &ks[buf][0];
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int k2 = 0; k2 < 4; k2++) {
                uint32_t b[4];   // ks 2 k2: {b0, b1}, ks 2 k2 + 1: {b2, b3}
                ipf_ldsm_x4(b, kb + (32 * kh + 8 * nt + (lane & 7)) * IPF_LD + 32 * k2 + 8 * (lane >> 3));
#pragma unroll
                for (int mt = 0; mt < 2; mt++) {
                    mma16816_bf(acc[mt][nt], a[mt][2 * k2], b);
                    mma16816_bf(acc[mt][nt], a[mt][2 * k2 + 1], b + 2);
                }
            }
        __syncthreads();   // this buffer is refilled by the next iteration's load
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                float p = fmaxf(acc[0][nt][e], 0.f) * wv[0] + fmaxf(acc[0][nt][2 + e], 0.f) * wv[1] +
                          fmaxf(acc[1][nt][e], 0.f) * wv[2] + fmaxf(acc[1][nt][2 + e], 0.f) * wv[3];
                p += __shfl_xor_sync(0xffffffffu, p, 4);
                p += __shfl_xor_sync(0xffffffffu, p, 8);
                p += __shfl_xor_sync(0xffffffffu, p, 16);
                const int j = tile * IPF_TK + 32 * kh + 8 * nt + 2 * t4 + e;
                if (g == 0 && i < nb && j < T) score[(size_t)i * lds + j] = j < vis ? p : -INFINITY;
            }
    }
    // keys past the block's last visible one: -inf (as k_idx_reduce writes them)
    for (int rr = 0; rr < IPF_R && rb + rr < nb; rr++)
        for (int j = ntile * IPF_TK + tid; j < T; j += 256) score[(size_t)(rb + rr) * lds + j] = -INFINITY;
}

}  // namespace dsv4

// ---------------------------------------------------------------- R8: decode exchanges between the cards (attention
// split). LL12 words (as the GLM engine's all-reduce): 16-byte stores of 3 payload u32 + a tag,
// written straight into the receiver's memory over P2P and polled there; a word is valid when its tag matches, so no
// flags / fences. Payload of a row = up to 3 segments of u32 (base, row stride, width, ring mask): rows r0 + r.
struct XSeg {
    uint32_t* p[5];
    int ld[5], w[5], r0[5], rmask[5], nr[5];   // R11: nr = rows of the segment (rows >= nr: zeros sent, nothing stored)
    int nseg, wpr, tot;   // words per row = ceil(sum w / 3)
};
__device__ __forceinline__ uint32_t* xseg_at(const XSeg& g, int r, int k) {   // k-th u32 of row r (k < sum w)
    int s = 0;
    while (s < g.nseg - 1 && k >= g.w[s]) k -= g.w[s++];
    if (r >= g.nr[s]) return nullptr;
    return g.p[s] + (size_t)((g.r0[s] + r) & g.rmask[s]) * g.ld[s] + k;
}
// poll one LL12 word until its tag arrives; a peer that never arrives traps after ~20 s of %globaltimer (review r8
// M3: the old bound of 2^34 polls was hours)
__device__ __forceinline__ uint64_t gtimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ uint4 ll_wait(const uint4* p, unsigned tag) {
    uint4 x;
    uint64_t t0 = 0;
    for (unsigned it = 0;; it++) {
        asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w) : "l"(p) : "memory");
        if (x.w == tag) return x;
        if ((it & 1023) == 1023) {
            uint64_t t = gtimer();
            if (!t0) t0 = t;
            else if (t - t0 > 20000000000ull) __trap();
        }
    }
}
// [dsv4-rel-pollbo]: with the words in host memory (no P2P) one thread per block waits for the block's first word, the
// others start polling after it: the long wait (during the CPU lane) costs one PCIe poller per block instead of every
// thread's (microbench: CPU read 141 -> 99 GB/s with every thread polling, 137 with one per block). Each word is still
// taken only with its own tag (exact)
__constant__ int c_llgate = 0;
__device__ __forceinline__ void ll_gate(const uint4* first, bool have, unsigned tag) {
    if (!c_llgate) return;
    if (threadIdx.x == 0 && have) (void)ll_wait(first, tag);
    __syncthreads();
}
__device__ __forceinline__ void ll_store(uint4* p, uint32_t a, uint32_t b, uint32_t c, unsigned tag) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(a), "r"(b), "r"(c), "r"(tag) : "memory");
}
__global__ void k_xpush(const __grid_constant__ XSeg g, int R, uint4* d0, uint4* d1, unsigned tag) {
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= R * g.wpr) return;
    const int r = w / g.wpr, k0 = (w % g.wpr) * 3;
    const int tot = g.tot;
    uint32_t v[3];
#pragma unroll
    for (int j = 0; j < 3; j++) {
        const uint32_t* q = k0 + j < tot ? xseg_at(g, r, k0 + j) : nullptr;
        v[j] = q ? *q : 0u;
    }
    asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(d0 + w), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(tag) : "memory");
    if (d1) asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(d1 + w), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(tag) : "memory");
}
__global__ void k_xpull(const __grid_constant__ XSeg g, int R, const uint4* src, unsigned tag) {
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int w0 = blockIdx.x * blockDim.x;
    ll_gate(src + w0, w0 < R * g.wpr, tag);
    if (w >= R * g.wpr) return;
    const int r = w / g.wpr, k0 = (w % g.wpr) * 3;
    const int tot = g.tot;
    const uint4 x = ll_wait(src + w, tag);
    const uint32_t v[3] = {x.x, x.y, x.z};
#pragma unroll
    for (int j = 0; j < 3; j++)
        if (k0 + j < tot)
            if (uint32_t* q = xseg_at(g, r, k0 + j)) *q = v[j];
}

// R10: two pulls (one per peer) in one launch: blocks [0, nb1) pull g1 from src1, the rest g2 from src2
__global__ void k_xpull2(const __grid_constant__ XSeg g1, const uint4* src1, const __grid_constant__ XSeg g2,
                         const uint4* src2, int R, int nb1, unsigned tag) {
    const bool two = (int)blockIdx.x >= nb1;
    const XSeg& g = two ? g2 : g1;
    const uint4* src = two ? src2 : src1;
    const int w = ((int)blockIdx.x - (two ? nb1 : 0)) * blockDim.x + threadIdx.x;
    const int w0 = ((int)blockIdx.x - (two ? nb1 : 0)) * blockDim.x;
    ll_gate(src + w0, w0 < R * g.wpr, tag);
    if (w >= R * g.wpr) return;
    const int r = w / g.wpr, k0 = (w % g.wpr) * 3;
    const int tot = g.tot;
    const uint4 x = ll_wait(src + w, tag);
    const uint32_t v[3] = {x.x, x.y, x.z};
#pragma unroll
    for (int j = 0; j < 3; j++)
        if (k0 + j < tot)
            if (uint32_t* q = xseg_at(g, r, k0 + j)) *q = v[j];
}

// R10 per-card head split: the argmax of a vocab slice [n0, n1) of one row (k_argmax's rule: max, then the lowest
// index; NaN never wins), published as one LL12 word {value bits, index, 0, tag} to dst (another card), or combined
// on the last card with the two peers' words (src0 / src1) into out
__device__ __forceinline__ void argmax_slice(const float* r, int n0, int n1, float* bv, int* bi, float& best, int& b) {
    best = -INFINITY;
    b = 1 << 30;
    for (int j = n0 + threadIdx.x; j < n1; j += 1024)
        if (r[j] > best || (r[j] == best && j < b)) {
            best = r[j];
            b = j;
        }
    bv[threadIdx.x] = best;
    bi[threadIdx.x] = b;
    __syncthreads();
    for (int o = 512; o; o >>= 1) {
        if (threadIdx.x < o) {
            float ov = bv[threadIdx.x + o];
            int oi = bi[threadIdx.x + o];
            if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = ov;
                bi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    best = bv[0];
    b = bi[0];
}
__global__ void __launch_bounds__(1024) k_argmax_put(const float* r, int n0, int n1, uint4* dst, unsigned tag) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    float best;
    int b;
    argmax_slice(r, n0, n1, bv, bi, best, b);
    if (threadIdx.x == 0) ll_store(dst, __float_as_uint(best), (uint32_t)b, 0u, tag);
}
__global__ void __launch_bounds__(1024) k_argmax_comb(const float* r, int n0, int n1, const uint4* src0, const uint4* src1,
                                                      unsigned tag, int* out) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    float best;
    int b;
    argmax_slice(r, n0, n1, bv, bi, best, b);
    if (threadIdx.x == 0) {
        for (const uint4* sp : {src0, src1}) {
            if (!sp) continue;
            const uint4 w = ll_wait(sp, tag);
            const float v = __uint_as_float(w.x);
            const int i = (int)w.y;
            if (v > best || (v == best && i < b)) {
                best = v;
                b = i;
            }
        }
        out[0] = b >= (1 << 30) ? 0 : b;   // all -inf / NaN: k_argmax's 0
    }
}
// ---------------------------------------------------------------- R9: hot experts of a layer split over the cards.
// A card's rows of the layer's routed choices go to the owner as LL12 words: hdr[rk] = {present, 0, 0, tag}, row rk =
// YW words of y[rk][0..5119]; the owner overwrites its (zero) rows of the remote choices, so the combine sums the same
// values in the same order as unsplit (bit-identical).
constexpr int YW = (dsv4::D + 2) / 3;   // 1707
__global__ void k_ypush(const int* ids, const int* slot, const float* y, uint4* rows, uint4* hdr, unsigned tag) {
    const int rk = blockIdx.x;
    const int e = ids[rk];
    const bool have = e >= 0 && e < dsv4::NE && slot[e] >= 0;
    if (threadIdx.x == 0) ll_store(hdr + rk, have ? 1u : 0u, 0u, 0u, tag);
    if (!have) return;
    const float* yr = y + (size_t)rk * dsv4::D;
    for (int w = threadIdx.x; w < YW; w += blockDim.x) {
        const int k0 = 3 * w;
        uint32_t v[3];
#pragma unroll
        for (int j = 0; j < 3; j++) v[j] = k0 + j < dsv4::D ? __float_as_uint(yr[k0 + j]) : 0u;
        ll_store(rows + (size_t)rk * YW + w, v[0], v[1], v[2], tag);
    }
}
// owner: for every choice rk, the senders' headers (rx areas of senders s0, s1; -1 = none), then the present row
__global__ void k_ypull(const uint4* rows, const uint4* hdr, int stride, int s0, int s1, unsigned tag, float* y) {
    const int rk = blockIdx.x;
    __shared__ int src;
    if (threadIdx.x == 0) {
        src = -1;
        if (s0 >= 0 && ll_wait(hdr + (size_t)s0 * stride + rk, tag).x) src = s0;
        if (s1 >= 0 && ll_wait(hdr + (size_t)s1 * stride + rk, tag).x) src = s1;
    }
    __syncthreads();
    if (src < 0) return;
    const uint4* rr = rows + ((size_t)src * stride + rk) * YW;
    float* yr = y + (size_t)rk * dsv4::D;
    for (int w = threadIdx.x; w < YW; w += blockDim.x) {
        const uint4 x = ll_wait(rr + w, tag);
        const int k0 = 3 * w;
        const uint32_t v[3] = {x.x, x.y, x.z};
#pragma unroll
        for (int j = 0; j < 3; j++)
            if (k0 + j < dsv4::D) yr[k0 + j] = __uint_as_float(v[j]);
    }
}

