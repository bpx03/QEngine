#pragma once
// Flash-Next specific kernels (first version: correctness first, decode rows m <= MAXM processed together
// where rows are independent and sequentially where they are not: conv / recurrent state / indexer tail).
#include <cuda_fp8.h>

#include "../kernels/gemv_h16.cuh"
#include "flash.h"
#include "../kernels/watchdog.cuh"  // QE_WD_POLL in the exchange waits (a stranded rank traps instead of hanging)

#ifndef FLASH_HCU_H16
#define FLASH_HCU_H16 1  // prefill HC up: fp16-accumulate MMA per 64-k group (0: fp32 accumulate)
#endif
namespace flash {

__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ unsigned fkey(float f) {  // order-preserving float -> uint (larger float, larger key)
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float kfloat(unsigned k) {
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}


template <int NT>
__device__ __forceinline__ float block_sum(float v, float* sh) {  // NT threads, sh >= 32 floats
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    float r = 0.f;
    for (int i = 0; i < NT / 32; i++) r += sh[i];
    return r;
}
template <int NT>
__device__ __forceinline__ float block_max(float v, float* sh) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    float r = -INFINITY;
    for (int i = 0; i < NT / 32; i++) r = fmaxf(r, sh[i]);
    return r;
}

// ------------------------------------------------------------------ embedding
// R[t][i * H + d] = embed[tok][d] for the 4 streams (embed: mapped host memory)
__global__ void k_embed(const bf16* __restrict__ emb, const int* __restrict__ tok, float* __restrict__ R) {
    const int t = blockIdx.x;
    const bf16* e = emb + (size_t)tok[t] * H;
    for (int d = threadIdx.x; d < H; d += blockDim.x) {
        const float v = __bfloat162float(e[d]);
#pragma unroll
        for (int i = 0; i < HC; i++) R[(size_t)t * RW + i * H + d] = v;
    }
}

// ------------------------------------------------------------------ MTP input
// blockIdx.y == 0: en[t] = bf16(rms(embed[tok]) * (1 + we)); y = 1..4: hn[t * 4 + s] = bf16(rms(h[t][s]) * (1 + wh[s]))
__global__ void k_mtp_prep(const bf16* __restrict__ emb, const int* __restrict__ tok, const float* __restrict__ h,
                           const bf16* __restrict__ we, const bf16* __restrict__ wh, bf16* __restrict__ en,
                           bf16* __restrict__ hn) {
    __shared__ float sh[32];
    const int t = blockIdx.x, y = blockIdx.y;
    float v[10];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 10; j++) {
        const int d = threadIdx.x + j * 256;
        v[j] = y == 0 ? __bfloat162float(emb[(size_t)tok[t] * H + d]) : h[(size_t)t * RW + (y - 1) * H + d];
        ss += v[j] * v[j];
    }
    ss = block_sum<256>(ss, sh);
    const float r = rsqrtf(ss / H + EPS);
#pragma unroll
    for (int j = 0; j < 10; j++) {
        const int d = threadIdx.x + j * 256;
        if (y == 0) en[(size_t)t * H + d] = __float2bfloat16(v[j] * r * (1.f + __bfloat162float(we[d])));
        else hn[((size_t)t * HC + y - 1) * H + d] = __float2bfloat16(v[j] * r * (1.f + __bfloat162float(wh[(y - 1) * H + d])));
    }
}
// R[t][s] = fc_embedding(en[t]) + fc_hidden(hn[t][s])
__global__ void k_mtp_comb(const float* __restrict__ fe, const float* __restrict__ fh, float* __restrict__ R) {
    const int t = blockIdx.x;
    for (int d = threadIdx.x; d < H; d += blockDim.x)
#pragma unroll
        for (int s = 0; s < HC; s++) R[(size_t)t * RW + s * H + d] = fe[(size_t)t * H + d] + fh[((size_t)t * HC + s) * H + d];
}

// ------------------------------------------------------------------ hyper-connections
// hn[t][i][d] = bf16(R * rsqrt(mean_i(R^2) + eps) * (1 + w)), grid (HC, m), 256 threads
__global__ void k_hc_norm(const float* __restrict__ R, const bf16* __restrict__ w, bf16* __restrict__ hn) {
    __shared__ float sh[32];
    const int i = blockIdx.x, t = blockIdx.y;
    const float* r = R + (size_t)t * RW + i * H;
    float ss = 0.f;
    for (int d = threadIdx.x; d < H; d += 256) ss += r[d] * r[d];
    ss = block_sum<256>(ss, sh);
    const float inv = rsqrtf(ss / H + EPS);
    for (int d = threadIdx.x; d < H; d += 256)
        hn[(size_t)t * RW + i * H + d] = __float2bfloat16(r[d] * inv * (1.f + __bfloat162float(w[i * H + d])));
}
// zb = bf16(silu(z / HC)) (320 columns, zero pad to 384), inj = 2 sigmoid(z[320 + i] / HC)
__global__ void k_hc_mid(const float* __restrict__ z, bf16* __restrict__ zb, float* __restrict__ inj, int m) {
    const int t = blockIdx.x;
    for (int j = threadIdx.x; j < HCU_K; j += blockDim.x)
        zb[(size_t)t * HCU_K + j] = __float2bfloat16(j < LOWRANK ? silu(z[(size_t)t * HCD_ROWS + j] / HC) : 0.f);
    if (threadIdx.x < HC) inj[t * HC + threadIdx.x] = 2.f * sigm(z[(size_t)t * HCD_ROWS + LOWRANK + threadIdx.x] / HC);
}
// x[t][d] = mean_i(sigmoid(g[t][i][d]) * hn[t][i][d]) -> bf16 x and XH xh (row stride H)
__global__ void k_hc_mix(const float* __restrict__ g, const bf16* __restrict__ hn, bf16* __restrict__ x,
                         __half* __restrict__ xh) {
    const int t = blockIdx.x;
    for (int d = threadIdx.x; d < H; d += blockDim.x) {
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < HC; i++)
            s += sigm(g[(size_t)t * RW + i * H + d]) * __bfloat162float(hn[(size_t)t * RW + i * H + d]);
        s *= 1.f / HC;
        x[(size_t)t * H + d] = __float2bfloat16(s);
        if (xh) store_xh(xh + (size_t)t * H, d, s);
    }
}
// R[t][i][d] += inj[t][i] * y[t][d]; y is zeroed after use (the next all-reduce adds into it)
__global__ void k_hc_write(float* __restrict__ R, const float* __restrict__ inj, float* __restrict__ y) {
    const int t = blockIdx.x;
    for (int d = threadIdx.x; d < H; d += blockDim.x) {
        const float v = y[(size_t)t * H + d];
        y[(size_t)t * H + d] = 0.f;
#pragma unroll
        for (int i = 0; i < HC; i++) R[(size_t)t * RW + i * H + d] += inj[t * HC + i] * v;
    }
}

// Activation steering (QE_STEER / QE_STEER_DIR) at a layer output, the point llama.cpp names "l_last" (the residual
// after the FFN hyper-connection combine), on each of the HC streams. Loaded vectors share one buffer
// w[vec][mode][layer][H] (strength folded in: add alpha v, project sqrt(alpha) unit v); every row picks
// {vector, mode, lo, hi} and the scale of its slot (sl[slot * 4], vector < 0: off) - per slot in batched steps (rslot),
// the bound slot's copy (index 0) otherwise. These live at fixed addresses: changing them changes no graph.
// add: R_i += scale w; project: R_i -= scale (R_i . w) w. dw[vec][mode][layer][stream][HCD_ROWS]: the next HC read's
// fused down weights applied to w (computed by the decode down kernel itself at load), the correction of the fused path.
// the row's w index (vector * 2 + mode, or -1) with its scale and mode
__device__ __forceinline__ int steer_row(const SteerArg& a, int t, float& sc, int& mode) {
    if (a.li < 0) return -1;
    const int slot = a.rslot ? a.rslot[t] : 0;
    const int* q = a.sl + slot * 4;
    const int v = q[0];
    if (v < 0 || a.li < q[2] || a.li > q[3]) return -1;
    sc = a.sc[slot];
    mode = q[1];
    return v * 2 + mode;
}
// Separate steering kernel (layer 0 / 47 outputs, prefill, unfused HC paths): one block per row; an unsteered row
// exits before touching memory. y != nullptr: the layer's residual update is still pending (R += inj * y): applied
// here first (the same fma as k_hc_write) and y cleared for the row (the next HC read then adds inj * 0, exact).
constexpr int STEER_T = 512, STEER_PT = H / STEER_T;
static_assert(H % STEER_T == 0, "steer: H / 512");
__global__ void __launch_bounds__(STEER_T) k_steer(float* __restrict__ R, const float* __restrict__ inj, float* __restrict__ y,
                                                   SteerArg a) {
    const int t = blockIdx.x, tid = threadIdx.x;
    float sc = 0.f;
    int mode = 0;
    const int vi = steer_row(a, t, sc, mode);
    if (vi < 0) return;
    const float* w = a.w + ((size_t)vi * LAYERS + a.li) * H;
    float* rr = R + (size_t)t * RW;
    float io[HC];
#pragma unroll
    for (int i = 0; i < HC; i++) io[i] = y ? inj[t * HC + i] : 0.f;
    float v[HC][STEER_PT], wv[STEER_PT];
#pragma unroll
    for (int j = 0; j < STEER_PT; j++) {
        const int d = tid + j * STEER_T;
        wv[j] = w[d];
        float yv = 0.f;
        if (y) { yv = y[(size_t)t * H + d]; y[(size_t)t * H + d] = 0.f; }
#pragma unroll
        for (int i = 0; i < HC; i++) {
            float x = rr[i * H + d];
            if (y) x += io[i] * yv;
            v[i][j] = x;
        }
    }
    float c[HC];
    if (mode == 1) {  // project
        __shared__ float red[STEER_T / 32][HC];
        __shared__ float dot[HC];
#pragma unroll
        for (int i = 0; i < HC; i++) {
            float q = 0.f;
#pragma unroll
            for (int j = 0; j < STEER_PT; j++) q += v[i][j] * wv[j];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
            if ((tid & 31) == 0) red[tid >> 5][i] = q;
        }
        __syncthreads();
        if (tid < HC) {
            float s2 = 0.f;
            for (int k = 0; k < STEER_T / 32; k++) s2 += red[k][tid];
            dot[tid] = s2;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < HC; i++) c[i] = -sc * dot[i];
    } else {
#pragma unroll
        for (int i = 0; i < HC; i++) c[i] = sc;
    }
#pragma unroll
    for (int j = 0; j < STEER_PT; j++)
#pragma unroll
        for (int i = 0; i < HC; i++) rr[i * H + tid + j * STEER_T] = v[i][j] + c[i] * wv[j];
}
// ------------------------------------------------------------------ fused hyper-connection read
// int8 pair -> fp16x2 (exact integers), as gemv.cu
__device__ __forceinline__ uint32_t s8x2_f16(uint32_t t) {
    uint32_t v = ((t ^ 0x00800080u) & 0x00FF00FFu) | 0x64006400u;
    __half2 h = __hsub2(*reinterpret_cast<__half2*>(&v), __float2half2_rn(1152.f));
    return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ void mma16(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void a_from_w8(uint32_t lo, uint32_t hi, uint32_t a[4]) {
    a[0] = s8x2_f16(__byte_perm(lo, 0, 0x4140));
    a[1] = s8x2_f16(__byte_perm(lo, 0, 0x4342));
    a[2] = s8x2_f16(__byte_perm(hi, 0, 0x4140));
    a[3] = s8x2_f16(__byte_perm(hi, 0, 0x4342));
}
__device__ __forceinline__ uint32_t f2h2(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}
// Down part: u[t][i][j] = sum_d D'[j][i, d] R[t][i, d] per stream i (D' = D (1 + w_norm) folded, W8 layout over
// K = 10240, rows 0..319 down, 320..323 inject, pad to 336), raw residual (the norm is a per-stream scalar applied
// later); blocks of tile 0 also write ss[t][i] = sum_d R^2. grid (21 tiles, 4 streams), 8 warps split the 20 groups.
template <int NT>
__global__ void __launch_bounds__(256) k_hcd(const uint4* __restrict__ q, const bf16* __restrict__ s, const float* __restrict__ R,
                                             float* __restrict__ u, float* __restrict__ ss, int m,
                                             const float* __restrict__ y = nullptr, const float* __restrict__ injp = nullptr,
                                             float* __restrict__ Rn = nullptr) {
    // y != nullptr: the residual update of the previous block is fused in: x = R + inj[br] * y, and the blocks of
    // tile 0 write the updated stream to Rn (ping-pong buffer) and its sums of squares
    const int r = blockIdx.x, br = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t4 = lane & 3;
    constexpr int K = RW, groups = K >> 7;   // 80 groups over the 4 streams
    const int gb = br * 20 + warp * 20 / 8, ge = br * 20 + (warp + 1) * 20 / 8;
    const uint4* qp = q + (size_t)r * (K >> 5) * 32 + lane;
    const bf16* sp = s + (size_t)r * groups * 16;
    bool act[NT];
    const float* xr[NT];
    const float* yr[NT];
    float ij[NT];
#pragma unroll
    for (int j = 0; j < NT; j++) {
        act[j] = 8 * j + g < m;
        const int tok = act[j] ? 8 * j + g : 0;
        xr[j] = R + (size_t)tok * RW + 2 * t4;
        yr[j] = y ? y + (size_t)tok * H + 2 * t4 - br * H : nullptr;
        ij[j] = y ? injp[tok * HC + br] : 0.f;
    }
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    uint4 w[3][4];
#pragma unroll
    for (int a = 0; a < 3; a++)
        if (gb + a < ge)
#pragma unroll
            for (int i = 0; i < 4; i++) w[a][i] = __ldg(qp + (size_t)((gb + a) * 4 + i) * 32);
#pragma unroll
    for (int a = 0; a < 3; a++) {
        const int gi = gb + a;
        if (gi >= ge) break;
        const float sg = __bfloat162float(sp[gi * 16 + g]), sg8 = __bfloat162float(sp[gi * 16 + g + 8]);
        float c[NT][4];
#pragma unroll
        for (int j = 0; j < NT; j++) c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0.f;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const uint32_t ws[4] = {w[a][i].x, w[a][i].y, w[a][i].z, w[a][i].w};
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int k0 = gi * 128 + (i * 2 + h) * 16;
                uint32_t aa[4];
                a_from_w8(ws[h * 2], ws[h * 2 + 1], aa);
#pragma unroll
                for (int j = 0; j < NT; j++) {
                    float2 x0 = act[j] ? *reinterpret_cast<const float2*>(xr[j] + k0) : make_float2(0.f, 0.f);
                    float2 x1 = act[j] ? *reinterpret_cast<const float2*>(xr[j] + k0 + 8) : make_float2(0.f, 0.f);
                    if (y && act[j]) {
                        const float2 y0 = *reinterpret_cast<const float2*>(yr[j] + k0), y1 = *reinterpret_cast<const float2*>(yr[j] + k0 + 8);
                        x0.x += ij[j] * y0.x; x0.y += ij[j] * y0.y; x1.x += ij[j] * y1.x; x1.y += ij[j] * y1.y;
                    }
                    mma16(c[j], aa, f2h2(x0.x, x0.y), f2h2(x1.x, x1.y));
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NT; j++) {
            acc[j][0] += c[j][0] * sg; acc[j][1] += c[j][1] * sg; acc[j][2] += c[j][2] * sg8; acc[j][3] += c[j][3] * sg8;
        }
    }
    __shared__ float red[8][NT][32][4];
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
        for (int i = 0; i < 4; i++) red[warp][j][lane][i] = acc[j][i];
    if (r == 0) {  // sum of squares of stream br, all rows at once (one reduction)
        __shared__ float red3[8][8 * NT];
        float q2[8 * NT];
#pragma unroll
        for (int t = 0; t < 8 * NT; t++) q2[t] = 0.f;
        for (int d = threadIdx.x; d < H; d += 256)
#pragma unroll
            for (int t = 0; t < 8 * NT; t++)
                if (t < m) {
                    float v = R[(size_t)t * RW + br * H + d];
                    if (y) {
                        v += injp[t * HC + br] * y[(size_t)t * H + d];
                        Rn[(size_t)t * RW + br * H + d] = v;
                    }
                    q2[t] += v * v;
                }
#pragma unroll
        for (int t = 0; t < 8 * NT; t++) {
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) q2[t] += __shfl_xor_sync(0xffffffffu, q2[t], o);
            if (lane == 0) red3[warp][t] = q2[t];
        }
        __syncthreads();
        if (threadIdx.x < m) {
            float a = 0.f;
            for (int w2 = 0; w2 < 8; w2++) a += red3[w2][threadIdx.x];
            ss[threadIdx.x * HC + br] = a;
        }
    }
    __syncthreads();
    if (warp != 0) return;
#pragma unroll
    for (int j = 0; j < NT; j++) {
        for (int w2 = 1; w2 < 8; w2++)
#pragma unroll
            for (int i = 0; i < 4; i++) acc[j][i] += red[w2][j][lane][i];
        const int n0 = 8 * j + 2 * t4, n1 = n0 + 1, row0 = r * 16;
        float* ub = u + br * HCD_ROWS;
        if (n0 < m) { ub[(size_t)n0 * HC * HCD_ROWS + row0 + g] = acc[j][0]; ub[(size_t)n0 * HC * HCD_ROWS + row0 + g + 8] = acc[j][2]; }
        if (n1 < m) { ub[(size_t)n1 * HC * HCD_ROWS + row0 + g] = acc[j][1]; ub[(size_t)n1 * HC * HCD_ROWS + row0 + g + 8] = acc[j][3]; }
    }
}
// Decode HC down split by groups (no wave tail): one warp per (16-row tile r, 128-k group gi), blocks of 4 warps,
// grid (21 tiles, 20). Partial u of the group -> P[t][stream][group in stream][HCD_ROWS] (summed in a fixed order by
// k_hc_mid_dec: deterministic, no atomics). Same per-group arithmetic as k_hcd.
constexpr int HCD_GPS = 20;  // 128-k groups per stream
// GPW groups per warp (partials per stream: HCD_GPS / GPW); y != nullptr: the pending residual update is applied on
// the fly (x = R + inj[stream] * y; k_hc_mid_dec writes the updated R)
constexpr int HCD_GPW = 2, HCD_NP = HCD_GPS / HCD_GPW;
template <int NT>
__global__ void __launch_bounds__(128) k_hcd_s(const uint4* __restrict__ q, const bf16* __restrict__ s, const float* __restrict__ R,
                                               float* __restrict__ P, int m, const float* __restrict__ y = nullptr,
                                               const float* __restrict__ injp = nullptr) {
    const int r = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wi = blockIdx.y * 4 + warp, gi0 = wi * HCD_GPW, br = gi0 / HCD_GPS, pq = (gi0 % HCD_GPS) / HCD_GPW;
    const int g = lane >> 2, t4 = lane & 3;
    constexpr int K = RW, groups = K >> 7;
    const uint4* qp = q + (size_t)r * (K >> 5) * 32 + lane;
    uint4 w[HCD_GPW][4];
#pragma unroll
    for (int a = 0; a < HCD_GPW; a++)
#pragma unroll
        for (int i = 0; i < 4; i++) w[a][i] = __ldg(qp + (size_t)((gi0 + a) * 4 + i) * 32);
    const bf16* sp = s + (size_t)r * groups * 16;
    float ij[NT];
#pragma unroll
    for (int j = 0; j < NT; j++) { const int tok = 8 * j + g; ij[j] = y && tok < m ? injp[tok * HC + br] : 0.f; }
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
#pragma unroll
    for (int a = 0; a < HCD_GPW; a++) {
        const int gi = gi0 + a;
        const float sg = __bfloat162float(sp[gi * 16 + g]), sg8 = __bfloat162float(sp[gi * 16 + g + 8]);
        float c[NT][4];
#pragma unroll
        for (int j = 0; j < NT; j++) c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0.f;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const uint32_t ws[4] = {w[a][i].x, w[a][i].y, w[a][i].z, w[a][i].w};
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int k0 = gi * 128 + (i * 2 + h) * 16;
                uint32_t aa[4];
                a_from_w8(ws[h * 2], ws[h * 2 + 1], aa);
#pragma unroll
                for (int j = 0; j < NT; j++) {
                    const int tok = 8 * j + g;
                    const bool act = tok < m;
                    const float* xr = R + (size_t)(act ? tok : 0) * RW + 2 * t4;
                    float2 x0 = act ? *reinterpret_cast<const float2*>(xr + k0) : make_float2(0.f, 0.f);
                    float2 x1 = act ? *reinterpret_cast<const float2*>(xr + k0 + 8) : make_float2(0.f, 0.f);
                    if (y && act) {
                        const float* yr = y + (size_t)tok * H + 2 * t4 - br * H;
                        const float2 y0 = *reinterpret_cast<const float2*>(yr + k0), y1 = *reinterpret_cast<const float2*>(yr + k0 + 8);
                        x0.x += ij[j] * y0.x; x0.y += ij[j] * y0.y; x1.x += ij[j] * y1.x; x1.y += ij[j] * y1.y;
                    }
                    mma16(c[j], aa, f2h2(x0.x, x0.y), f2h2(x1.x, x1.y));
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NT; j++) {
            acc[j][0] += c[j][0] * sg; acc[j][1] += c[j][1] * sg; acc[j][2] += c[j][2] * sg8; acc[j][3] += c[j][3] * sg8;
        }
    }
#pragma unroll
    for (int j = 0; j < NT; j++) {
        const int n0 = 8 * j + 2 * t4, n1 = n0 + 1, row0 = r * 16;
        if (n0 < m) {
            float* pb = P + (((size_t)n0 * HC + br) * HCD_NP + pq) * HCD_ROWS + row0;
            pb[g] = acc[j][0]; pb[g + 8] = acc[j][2];
        }
        if (n1 < m) {
            float* pb = P + (((size_t)n1 * HC + br) * HCD_NP + pq) * HCD_ROWS + row0;
            pb[g] = acc[j][1]; pb[g + 8] = acc[j][3];
        }
    }
}
// k_hcd_s for batched steps (m > 8): block = (group pair wi, 7 of the 21 row tiles), one warp per tile. The pair's
// 256 inputs of every row (R + inj * y, fp16) are staged in shared memory once per block instead of being loaded and
// converted by each of the 21 tile warps. Per-warp arithmetic and outputs as k_hcd_s (bit-identical).
constexpr int HCDX_TW = 7, HCDX_LD = 2 * 128 + 8;  // tiles per block (21 = 3 x 7), padded row (halves)
template <int NT>
__global__ void __launch_bounds__(32 * HCDX_TW) k_hcd_x(const uint4* __restrict__ q, const bf16* __restrict__ s, const float* __restrict__ R,
                                                        float* __restrict__ P, int m, const float* __restrict__ y = nullptr,
                                                        const float* __restrict__ injp = nullptr) {
    __shared__ __align__(16) __half xs[8 * NT][HCDX_LD];
    const int wi = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.y * HCDX_TW + warp;
    const int gi0 = wi * HCD_GPW, br = gi0 / HCD_GPS, pq = (gi0 % HCD_GPS) / HCD_GPW;
    const int g = lane >> 2, t4 = lane & 3;
    constexpr int K = RW, groups = K >> 7;
    const uint4* qp = q + (size_t)r * (K >> 5) * 32 + lane;
    uint4 w[HCD_GPW][4];
#pragma unroll
    for (int a = 0; a < HCD_GPW; a++)
#pragma unroll
        for (int i = 0; i < 4; i++) w[a][i] = __ldg(qp + (size_t)((gi0 + a) * 4 + i) * 32);
#pragma unroll 4
    for (int idx = threadIdx.x; idx < 8 * NT * 128; idx += 32 * HCDX_TW) {  // pairs of columns
        const int tok = idx >> 7, kk = (idx & 127) * 2;
        float2 v = make_float2(0.f, 0.f);
        if (tok < m) {
            const int k = gi0 * 128 + kk;
            v = *reinterpret_cast<const float2*>(R + (size_t)tok * RW + k);
            if (y) {
                const float ij = injp[tok * HC + br];
                const float2 yv = *reinterpret_cast<const float2*>(y + (size_t)tok * H + k - br * H);
                v.x += ij * yv.x; v.y += ij * yv.y;
            }
        }
        *reinterpret_cast<uint32_t*>(&xs[tok][kk]) = f2h2(v.x, v.y);
    }
    __syncthreads();
    const bf16* sp = s + (size_t)r * groups * 16;
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
#pragma unroll
    for (int a = 0; a < HCD_GPW; a++) {
        const int gi = gi0 + a;
        const float sg = __bfloat162float(sp[gi * 16 + g]), sg8 = __bfloat162float(sp[gi * 16 + g + 8]);
        float c[NT][4];
#pragma unroll
        for (int j = 0; j < NT; j++) c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0.f;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const uint32_t ws[4] = {w[a][i].x, w[a][i].y, w[a][i].z, w[a][i].w};
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int k0 = a * 128 + (i * 2 + h) * 16;
                uint32_t aa[4];
                a_from_w8(ws[h * 2], ws[h * 2 + 1], aa);
#pragma unroll
                for (int j = 0; j < NT; j++) {
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&xs[8 * j + g][k0 + 2 * t4]);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&xs[8 * j + g][k0 + 8 + 2 * t4]);
                    mma16(c[j], aa, b0, b1);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NT; j++) {
            acc[j][0] += c[j][0] * sg; acc[j][1] += c[j][1] * sg; acc[j][2] += c[j][2] * sg8; acc[j][3] += c[j][3] * sg8;
        }
    }
#pragma unroll
    for (int j = 0; j < NT; j++) {
        const int n0 = 8 * j + 2 * t4, n1 = n0 + 1, row0 = r * 16;
        if (n0 < m) {
            float* pb = P + (((size_t)n0 * HC + br) * HCD_NP + pq) * HCD_ROWS + row0;
            pb[g] = acc[j][0]; pb[g + 8] = acc[j][2];
        }
        if (n1 < m) {
            float* pb = P + (((size_t)n1 * HC + br) * HCD_NP + pq) * HCD_ROWS + row0;
            pb[g] = acc[j][1]; pb[g + 8] = acc[j][3];
        }
    }
}
// per row: u = the 20 group partials summed in order, ss = sum of squares of each stream, rinv, h = silu(z / 4) (fp16),
// inj (as k_hc_mid_pf). grid m, 512 threads.
__global__ void __launch_bounds__(512) k_hc_mid_dec(const float* __restrict__ P, float* __restrict__ R,
                                                    __half* __restrict__ hh, float* __restrict__ rinv, float* __restrict__ inj,
                                                    const float* __restrict__ y = nullptr, const float* __restrict__ injo = nullptr,
                                                    SteerArg sa = SteerArg()) {
    // y != nullptr: R += inj_old * y here (k_hcd_s used the same values); inj is overwritten only after the barrier
    const int t = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    __shared__ float red[16][HC];
    __shared__ float ri[HC];
    __shared__ float sdot[HC];
    float q2[HC] = {0.f, 0.f, 0.f, 0.f};
    float* rr = R + (size_t)t * RW;
    float io[HC];
#pragma unroll
    for (int i = 0; i < HC; i++) io[i] = y ? injo[t * HC + i] : 0.f;
    // steering of the previous layer's output fused in (SteerArg): R' = R + c_i w per stream, the down sums of the
    // unsteered R (k_hcd_s) corrected by c_i dw_i (linear in R); c_i = scale (add) or -scale (R_i . w) (project)
    float ssc = 0.f;
    int mode = 0;
    const int svi = steer_row(sa, t, ssc, mode);
    float cst[HC] = {0.f, 0.f, 0.f, 0.f};
    if (svi >= 0) {
        const float* w = sa.w + ((size_t)svi * LAYERS + sa.li) * H;
        float v[HC][H / 512], wv[H / 512];
#pragma unroll
        for (int j = 0; j < H / 512; j++) {
            const int d = tid + j * 512;
            wv[j] = w[d];
            const float yv = y ? y[(size_t)t * H + d] : 0.f;
#pragma unroll
            for (int i = 0; i < HC; i++) {
                float x = rr[i * H + d];
                if (y) x += io[i] * yv;
                v[i][j] = x;
            }
        }
        if (mode == 1) {
#pragma unroll
            for (int i = 0; i < HC; i++) {
                float q = 0.f;
#pragma unroll
                for (int j = 0; j < H / 512; j++) q += v[i][j] * wv[j];
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
                if (lane == 0) red[warp][i] = q;
            }
            __syncthreads();
            if (tid < HC) {
                float a = 0.f;
                for (int w2 = 0; w2 < 16; w2++) a += red[w2][tid];
                sdot[tid] = a;
            }
            __syncthreads();
#pragma unroll
            for (int i = 0; i < HC; i++) cst[i] = -ssc * sdot[i];
        } else {
#pragma unroll
            for (int i = 0; i < HC; i++) cst[i] = ssc;
        }
#pragma unroll
        for (int j = 0; j < H / 512; j++)
#pragma unroll
            for (int i = 0; i < HC; i++) {
                const float x = v[i][j] + cst[i] * wv[j];
                rr[i * H + tid + j * 512] = x;
                q2[i] += x * x;
            }
    } else
    for (int d = tid; d < H; d += 512) {
        const float yv = y ? y[(size_t)t * H + d] : 0.f;
#pragma unroll
        for (int i = 0; i < HC; i++) {
            float v = rr[i * H + d];
            if (y) { v += io[i] * yv; rr[i * H + d] = v; }
            q2[i] += v * v;
        }
    }
#pragma unroll
    for (int i = 0; i < HC; i++) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) q2[i] += __shfl_xor_sync(0xffffffffu, q2[i], o);
        if (lane == 0) red[warp][i] = q2[i];
    }
    __syncthreads();
    if (tid < HC) {
        float a = 0.f;
        for (int w2 = 0; w2 < 16; w2++) a += red[w2][tid];
        ri[tid] = rsqrtf(a / H + EPS);
        rinv[t * HC + tid] = ri[tid];
    }
    __syncthreads();
    if (tid < LOWRANK + HC) {
        float z = 0.f;
#pragma unroll
        for (int i = 0; i < HC; i++) {
            const float* pb = P + ((size_t)t * HC + i) * HCD_NP * HCD_ROWS + tid;
            float u = 0.f;
#pragma unroll
            for (int gq = 0; gq < HCD_NP; gq++) u += pb[(size_t)gq * HCD_ROWS];
            if (svi >= 0) u += cst[i] * sa.dw[(((size_t)svi * LAYERS + sa.li) * HC + i) * HCD_ROWS + tid];
            z += u * ri[i];
        }
        if (tid < LOWRANK) hh[(size_t)t * LOWRANK + tid] = __float2half_rn(silu(z / HC));
        else if (inj) inj[t * HC + tid - LOWRANK] = 2.f * sigm(z / HC);
    }
}
// activation capture (make_vector, qe_capture): acc[d] += sum over rows t >= skip of the layer-output residual's
// stream mean (R + inj * y when the update is pending)
__global__ void k_capture(const float* __restrict__ R, const float* __restrict__ inj, const float* __restrict__ y, int m, int skip,
                          float* __restrict__ acc) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= H) return;
    float s = 0.f;
    for (int t = skip; t < m; t++) {
        const float yv = y ? y[(size_t)t * H + d] : 0.f;
#pragma unroll
        for (int i = 0; i < HC; i++) s += R[(size_t)t * RW + i * H + d] + (y ? inj[t * HC + i] * yv : 0.f);
    }
    acc[d] += s * (1.f / HC);
}
// dw of one (vector, layer): sum of the down kernel's group partials of row s, stream s (the order of k_hc_mid_dec)
__global__ void k_steer_dw(const float* __restrict__ P, float* __restrict__ dw) {
    const int s = blockIdx.x, r = threadIdx.x;
    if (r >= HCD_ROWS) return;
    const float* pb = P + ((size_t)s * HC + s) * HCD_NP * HCD_ROWS + r;
    float u = 0.f;
    for (int gq = 0; gq < HCD_NP; gq++) u += pb[(size_t)gq * HCD_ROWS];
    dw[s * HCD_ROWS + r] = u;
}

// Up part + mix. Prologue (every block): z_j = sum_i u[t][i][j] * rinv_i, h = silu(z / 4) (fp16, smem); block 0
// writes inj = 2 sigmoid(z[320 + i] / 4). U rows are permuted so a 16-row tile = 4 hidden dims x 4 streams
// (row dd * 4 + i of tile T <-> U row i * 2560 + 4T + dd), INT8 with g64 scales; one warp per tile, K = 320.
// x[t][d] = mean_i sigmoid(U h)[i, d] * R[t][i, d] * rinv_i * (1 + w[i, d]) -> bf16 x and XH xh.
constexpr int HCU_TILES = RW / 16, HCU_KS = LOWRANK / 16, HCU_G = LOWRANK / 64;
template <int NT>
__global__ void __launch_bounds__(256) k_hcu(const uint4* __restrict__ q, const __half* __restrict__ s, const float* __restrict__ u,
                                             const float* __restrict__ ss, const float* __restrict__ R, const bf16* __restrict__ nw,
                                             float* __restrict__ inj, bf16* __restrict__ x, __half* __restrict__ xh, int m,
                                             float* __restrict__ yzero = nullptr, const __half* __restrict__ hhg = nullptr,
                                             const float* __restrict__ rinvg = nullptr) {
    if (yzero) {  // the fused residual update consumed y: clear it for the next all-reduce (spread over the blocks)
        if (m > 8) {
            for (int i = (blockIdx.x * 256 + threadIdx.x) * 4; i < m * H; i += gridDim.x * 256 * 4)
                *reinterpret_cast<float4*>(yzero + i) = make_float4(0.f, 0.f, 0.f, 0.f);
        } else if (blockIdx.x == gridDim.x - 1)
            for (int i = threadIdx.x; i < m * H; i += 256) yzero[i] = 0.f;
    }
    __shared__ __half hs[8 * NT][LOWRANK + 8];
    __shared__ float rinv[8 * NT][HC];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int T = blockIdx.x * 8 + warp;
    uint4 w[HCU_KS / 2];  // weights first: their DRAM latency overlaps the prologue
    if (T < HCU_TILES) {
        const uint4* qp = q + (size_t)T * (HCU_KS / 2) * 32 + lane;
#pragma unroll
        for (int i = 0; i < HCU_KS / 2; i++) w[i] = __ldg(qp + (size_t)i * 32);
    }
    if (hhg) {  // h and rinv precomputed (k_hc_mid_dec)
        if (tid < m * HC) rinv[tid / HC][tid % HC] = rinvg[tid];
        for (int idx = tid; idx < m * LOWRANK / 2; idx += 256) {
            const int t = idx / (LOWRANK / 2), j = idx % (LOWRANK / 2);
            reinterpret_cast<__half2*>(&hs[t][0])[j] = reinterpret_cast<const __half2*>(hhg + (size_t)t * LOWRANK)[j];
        }
    } else {
    if (tid < m * HC) rinv[tid / HC][tid % HC] = rsqrtf(ss[tid] / H + EPS);
    __syncthreads();
    for (int idx = tid; idx < m * (LOWRANK + HC); idx += 256) {
        const int t = idx / (LOWRANK + HC), j = idx % (LOWRANK + HC);
        float z = 0.f;
#pragma unroll
        for (int i = 0; i < HC; i++) z += u[((size_t)t * HC + i) * HCD_ROWS + j] * rinv[t][i];
        if (j < LOWRANK) hs[t][j] = __float2half_rn(silu(z / HC));
        else if (blockIdx.x == 0 && inj) inj[t * HC + j - LOWRANK] = 2.f * sigm(z / HC);
    }
    }
    for (int idx = tid; idx < (8 * NT - m) * (LOWRANK + 8); idx += 256) hs[m + idx / (LOWRANK + 8)][idx % (LOWRANK + 8)] = __float2half_rn(0.f);
    __syncthreads();
    if (T >= HCU_TILES) return;
    const int g = lane >> 2, t4 = lane & 3;
    const __half* sp = s + (size_t)T * HCU_G * 16;
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
#pragma unroll
    for (int gi = 0; gi < HCU_G; gi++) {
        float c[NT][4];
#pragma unroll
        for (int j = 0; j < NT; j++) c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0.f;
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const uint4 wv = w[gi * 2 + i];
            const uint32_t ws[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int k0 = gi * 64 + (i * 2 + h) * 16;
                uint32_t aa[4];
                a_from_w8(ws[h * 2], ws[h * 2 + 1], aa);
#pragma unroll
                for (int j = 0; j < NT; j++) {
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&hs[8 * j + g][k0 + 2 * t4]);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&hs[8 * j + g][k0 + 8 + 2 * t4]);
                    mma16(c[j], aa, b0, b1);
                }
            }
        }
        const float sg = __half2float(sp[gi * 16 + g]), sg8 = __half2float(sp[gi * 16 + g + 8]);
#pragma unroll
        for (int j = 0; j < NT; j++) {
            acc[j][0] += c[j][0] * sg; acc[j][1] += c[j][1] * sg; acc[j][2] += c[j][2] * sg8; acc[j][3] += c[j][3] * sg8;
        }
    }
    // rows g (dd = g / 4, stream g % 4) and g + 8 (dd = 2 + g / 4); tokens 8 j + 2 t4 (+1)
    const int i = g & 3;
#pragma unroll
    for (int j = 0; j < NT; j++) {
        float v[4];
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int dd = (e >> 1) * 2 + (g >> 2), tok = 8 * j + 2 * t4 + (e & 1);
            const int d = T * 4 + dd;
            const float gate = sigm(acc[j][(e >> 1) * 2 + (e & 1)]);
            v[e] = tok < m ? gate * R[(size_t)tok * RW + i * H + d] * rinv[tok][i] * (1.f + __bfloat162float(nw[i * H + d])) : 0.f;
        }
#pragma unroll
        for (int e = 0; e < 4; e++) {
            v[e] += __shfl_xor_sync(0xffffffffu, v[e], 4);
            v[e] += __shfl_xor_sync(0xffffffffu, v[e], 8);
        }
        if (i == 0) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int dd = (e >> 1) * 2 + (g >> 2), tok = 8 * j + 2 * t4 + (e & 1);
                if (tok >= m) continue;
                const int d = T * 4 + dd;
                const float xv = v[e] * (1.f / HC);
                x[(size_t)tok * H + d] = __float2bfloat16(xv);
                if (xh) store_xh(xh + (size_t)tok * H, d, xv);
            }
        }
    }
}

// ------------------------------------------------------------------ PLE (layer 1)
// per (row, stream): key_n = norm_key(key_i), q_n = norm_query(R_i), gate = sgnsqrt(<key_n, q_n> / sqrt(H)),
// gv_i = sigmoid(gate) * value, gvn_i = norm_conv(gv_i). grid (HC, m), 256 threads
__global__ void k_ple_gate(const float* __restrict__ key, const float* __restrict__ val, const float* __restrict__ R,
                           const bf16* __restrict__ nk, const bf16* __restrict__ nq, const bf16* __restrict__ nc,
                           float* __restrict__ gv, float* __restrict__ gvn) {
    __shared__ float sh[32];
    const int i = blockIdx.x, t = blockIdx.y;
    const float* k = key + (size_t)t * RW + i * H;
    const float* r = R + (size_t)t * RW + i * H;
    float sk = 0.f, sq = 0.f;
    for (int d = threadIdx.x; d < H; d += 256) { sk += k[d] * k[d]; sq += r[d] * r[d]; }
    sk = block_sum<256>(sk, sh);
    sq = block_sum<256>(sq, sh);
    const float ik = rsqrtf(sk / H + EPS), iq = rsqrtf(sq / H + EPS);
    float dot = 0.f;
    for (int d = threadIdx.x; d < H; d += 256)
        dot += (k[d] * ik * (1.f + __bfloat162float(nk[i * H + d]))) * (r[d] * iq * (1.f + __bfloat162float(nq[i * H + d])));
    dot = block_sum<256>(dot, sh) / sqrtf((float)H);
    const float g = copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot);
    const float s = sigm(g);
    const float* v = val + (size_t)t * H;
    float sc = 0.f;
    for (int d = threadIdx.x; d < H; d += 256) { const float a = s * v[d]; sc += a * a; }
    sc = block_sum<256>(sc, sh);
    const float ic = rsqrtf(sc / H + EPS);
    for (int d = threadIdx.x; d < H; d += 256) {
        const float a = s * v[d];
        gv[(size_t)t * RW + i * H + d] = a;
        gvn[(size_t)t * RW + i * H + d] = a * ic * (1.f + __bfloat162float(nc[i * H + d]));
    }
}
// dilated causal conv (k = 4, dilation 3) over gvn with a 9-row history, R += gv + silu(conv); rows in order.
// Deferred commit as k_dn_conv_d: hist holds the committed history, pend [MAXM][RW] the last forward's rows.
__global__ void k_ple_conv(const float* __restrict__ gv, const float* __restrict__ gvn, const bf16* __restrict__ cw,
                           float* __restrict__ hist, float* __restrict__ pend, const int* __restrict__ pn,
                           float* __restrict__ R, int m, bool now = false) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= RW) return;
    float h[9];
#pragma unroll
    for (int j = 0; j < 9; j++) h[j] = hist[j * RW + c];  // h[0] = oldest (t - 9)
    const int n = *pn;
    for (int r = 0; r < n; r++) {
#pragma unroll
        for (int j = 0; j < 8; j++) h[j] = h[j + 1];
        h[8] = pend[(size_t)r * RW + c];
    }
#pragma unroll
    for (int j = 0; j < 9; j++) hist[j * RW + c] = h[j];
    const float w0 = __bfloat162float(cw[c * 4 + 0]), w1 = __bfloat162float(cw[c * 4 + 1]);
    const float w2 = __bfloat162float(cw[c * 4 + 2]), w3 = __bfloat162float(cw[c * 4 + 3]);
    for (int t = 0; t < m; t++) {
        const float cur = gvn[(size_t)t * RW + c];
        if (!now) pend[(size_t)t * RW + c] = cur;
        const float o = w0 * h[0] + w1 * h[3] + w2 * h[6] + w3 * cur;
        R[(size_t)t * RW + c] += gv[(size_t)t * RW + c] + silu(o);
#pragma unroll
        for (int j = 0; j < 8; j++) h[j] = h[j + 1];
        h[8] = cur;
    }
    if (now)
#pragma unroll
        for (int j = 0; j < 9; j++) hist[j * RW + c] = h[j];
}

// ------------------------------------------------------------------ DeltaNet
// causal conv (k = 4) + silu over ch channels, rows in order; history [3][ch] (oldest first)
__global__ void k_dn_conv(const float* __restrict__ proj, int ld, int ch, const bf16* __restrict__ cw,
                          float* __restrict__ hist, float* __restrict__ out, int m) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    float h0 = hist[c], h1 = hist[ch + c], h2 = hist[2 * ch + c];
    const float w0 = __bfloat162float(cw[c * 4]), w1 = __bfloat162float(cw[c * 4 + 1]);
    const float w2 = __bfloat162float(cw[c * 4 + 2]), w3 = __bfloat162float(cw[c * 4 + 3]);
    for (int t = 0; t < m; t++) {
        const float x = proj[(size_t)t * ld + c];
        out[(size_t)t * ch + c] = silu(w0 * h0 + w1 * h1 + w2 * h2 + w3 * x);
        h0 = h1; h1 = h2; h2 = x;
    }
    hist[c] = h0; hist[ch + c] = h1; hist[2 * ch + c] = h2;
}
// one block per local value head, 128 threads (thread j = value column j); S [128 k][128 v] in smem.
// cv rows: [q kl*128][k kl*128][v vl*128]; ab rows [b vl][a vl]; o [m][vl][128]
__global__ void __launch_bounds__(128) k_dn_rec(const float* __restrict__ cv, int ch, int kl, int vl, int k_first,
                                                int v_first, const float* __restrict__ ab, const float* __restrict__ A,
                                                const float* __restrict__ dtb, float* __restrict__ S,
                                                float* __restrict__ o, int m) {
    extern __shared__ float sm[];
    float* Ss = sm;               // 128 x 128
    float* qs = sm + 128 * 128;   // 128
    float* ks = qs + 128;         // 128
    __shared__ float red[32];
    const int hv = blockIdx.x, j = threadIdx.x;
    const int kh = (v_first + hv) / (LNV / LNK) - k_first;  // local key head
    float* Sg = S + (size_t)hv * 128 * 128;
    for (int i = j; i < 128 * 128; i += 128) Ss[i] = Sg[i];
    const float Ah = A[hv], dt = dtb[hv];
    for (int t = 0; t < m; t++) {
        const float* row = cv + (size_t)t * ch;
        const float q = row[kh * 128 + j], k = row[kl * 128 + kh * 128 + j], v = row[2 * kl * 128 + hv * 128 + j];
        const float qq = block_sum<128>(q * q, red), kk = block_sum<128>(k * k, red);
        qs[j] = q * rsqrtf(qq + 1e-6f) * rsqrtf(128.f);
        ks[j] = k * rsqrtf(kk + 1e-6f);
        const float b = ab[(size_t)t * 2 * vl + hv], a = ab[(size_t)t * 2 * vl + vl + hv];
        const float beta = sigm(b);
        const float xa = a + dt;
        const float sp = xa > 20.f ? xa : log1pf(__expf(xa));
        const float decay = __expf(Ah * sp);
        __syncthreads();
        float mem = 0.f;
        for (int kk2 = 0; kk2 < 128; kk2++) {
            const float s = Ss[kk2 * 128 + j] * decay;
            Ss[kk2 * 128 + j] = s;
            mem += s * ks[kk2];
        }
        const float delta = (v - mem) * beta;
        float out = 0.f;
        for (int kk2 = 0; kk2 < 128; kk2++) {
            const float s = Ss[kk2 * 128 + j] + ks[kk2] * delta;
            Ss[kk2 * 128 + j] = s;
            out += s * qs[kk2];
        }
        o[((size_t)t * vl + hv) * 128 + j] = out;
        __syncthreads();
    }
    for (int i = j; i < 128 * 128; i += 128) Sg[i] = Ss[i];
}
// Parallel variant: grid (vl heads, 16 column groups of 8 value columns), 128 threads; thread t holds S rows
// [8 (t % 16), +8) of column cg * 8 + t / 16 in registers. State layout ST[head][v][k] (k contiguous).
__global__ void __launch_bounds__(128) k_dn_rec2(const float* __restrict__ cv, int ch, int kl, int vl, int k_first,
                                                 int v_first, const float* __restrict__ ab, const float* __restrict__ A,
                                                 const float* __restrict__ dtb, float* __restrict__ ST,
                                                 float* __restrict__ o, int m) {
    __shared__ float qs[128], ks[128];
    __shared__ float red[32];
    const int hv = blockIdx.x, cg = blockIdx.y, t = threadIdx.x;
    const int col = cg * 8 + (t >> 4), kb = (t & 15) * 8;
    const int kh = (v_first + hv) / (LNV / LNK) - k_first;
    float* sp = ST + ((size_t)hv * 128 + col) * 128 + kb;
    float s[8];
    {
        const float4 a = *reinterpret_cast<const float4*>(sp), b = *reinterpret_cast<const float4*>(sp + 4);
        s[0] = a.x; s[1] = a.y; s[2] = a.z; s[3] = a.w; s[4] = b.x; s[5] = b.y; s[6] = b.z; s[7] = b.w;
    }
    const float Ah = A[hv], dt = dtb[hv];
    for (int r = 0; r < m; r++) {
        const float* row = cv + (size_t)r * ch;
        const float q = row[kh * 128 + t], k = row[kl * 128 + kh * 128 + t];
        const float qq = block_sum<128>(q * q, red), kk = block_sum<128>(k * k, red);
        qs[t] = q * rsqrtf(qq + 1e-6f) * rsqrtf(128.f);
        ks[t] = k * rsqrtf(kk + 1e-6f);
        __syncthreads();
        const float b = ab[(size_t)r * 2 * vl + hv], a = ab[(size_t)r * 2 * vl + vl + hv];
        const float beta = sigm(b);
        const float xa = a + dt;
        const float spl = xa > 20.f ? xa : log1pf(__expf(xa));
        const float decay = __expf(Ah * spl);
        float mem = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) { s[i] *= decay; mem += s[i] * ks[kb + i]; }
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) mem += __shfl_xor_sync(0xffffffffu, mem, off);
        const float v = row[2 * kl * 128 + hv * 128 + col];
        const float delta = (v - mem) * beta;
        float out = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) { s[i] += ks[kb + i] * delta; out += s[i] * qs[kb + i]; }
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) out += __shfl_xor_sync(0xffffffffu, out, off);
        if ((t & 15) == 0) o[((size_t)r * vl + hv) * 128 + col] = out;
    }
    *reinterpret_cast<float4*>(sp) = make_float4(s[0], s[1], s[2], s[3]);
    *reinterpret_cast<float4*>(sp + 4) = make_float4(s[4], s[5], s[6], s[7]);
}
// ---- deferred commit (speculative verify): a forward never commits its rows to the recurrent state; the next
// forward first commits the `*pn` accepted rows of the previous one from their saved inputs, then computes its
// own rows from that state. Plain decoding commits every row one step late (no cost).
// conv: hist [3][ch] = state; pend [MAXM][ch] raw inputs of the last forward (per channel: same thread -> no race)
__global__ void k_dn_conv_d(const float* __restrict__ proj, int ld, int ch, const bf16* __restrict__ cw,
                            float* __restrict__ hist, float* __restrict__ pend, const int* __restrict__ pn,
                            float* __restrict__ out, int m, bool now = false) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    float h0 = hist[c], h1 = hist[ch + c], h2 = hist[2 * ch + c];
    const int n = *pn;
    for (int r = 0; r < n; r++) {
        const float x = pend[(size_t)r * ch + c];
        h0 = h1; h1 = h2; h2 = x;
    }
    hist[c] = h0; hist[ch + c] = h1; hist[2 * ch + c] = h2;
    const float w0 = __bfloat162float(cw[c * 4]), w1 = __bfloat162float(cw[c * 4 + 1]);
    const float w2 = __bfloat162float(cw[c * 4 + 2]), w3 = __bfloat162float(cw[c * 4 + 3]);
    for (int t = 0; t < m; t++) {
        const float x = proj[(size_t)t * ld + c];
        if (!now) pend[(size_t)t * ch + c] = x;
        out[(size_t)t * ch + c] = silu(w0 * h0 + w1 * h1 + w2 * h2 + w3 * x);
        h0 = h1; h1 = h2; h2 = x;
    }
    if (now) { hist[c] = h0; hist[ch + c] = h1; hist[2 * ch + c] = h2; }  // prefill: rows committed at once
}
// row-parallel k_dn_conv_d for prefill chunks (rows committed at once): grid (ceil(ch / 256), ceil(m / 16)); the
// history (after the pending rows of a previous forward) is only read here, k_dn_conv_hist writes the new one after
__device__ __forceinline__ void dn_hist_after_pend(const float* hist, const float* pend, int n, int ch, int c, float& h0,
                                                   float& h1, float& h2) {
    h0 = hist[c]; h1 = hist[ch + c]; h2 = hist[2 * ch + c];
    for (int r = 0; r < n; r++) { const float x = pend[(size_t)r * ch + c]; h0 = h1; h1 = h2; h2 = x; }
}
__global__ void k_dn_conv_par(const float* __restrict__ proj, int ld, int ch, const bf16* __restrict__ cw,
                              const float* __restrict__ hist, const float* __restrict__ pend, const int* __restrict__ pn,
                              float* __restrict__ out, int m) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    const int t0 = blockIdx.y * 16, t1 = min(m, t0 + 16);
    float h0, h1, h2;
    if (t0 >= 3) {
        h0 = proj[(size_t)(t0 - 3) * ld + c]; h1 = proj[(size_t)(t0 - 2) * ld + c]; h2 = proj[(size_t)(t0 - 1) * ld + c];
    } else {
        float a0, a1, a2;
        dn_hist_after_pend(hist, pend, *pn, ch, c, a0, a1, a2);
        float w[6] = {a0, a1, a2, 0.f, 0.f, 0.f};
        for (int t = 0; t < t0; t++) w[3 + t] = proj[(size_t)t * ld + c];
        h0 = w[t0]; h1 = w[t0 + 1]; h2 = w[t0 + 2];
    }
    const float w0 = __bfloat162float(cw[c * 4]), w1 = __bfloat162float(cw[c * 4 + 1]);
    const float w2 = __bfloat162float(cw[c * 4 + 2]), w3 = __bfloat162float(cw[c * 4 + 3]);
    for (int t = t0; t < t1; t++) {
        const float x = proj[(size_t)t * ld + c];
        out[(size_t)t * ch + c] = silu(w0 * h0 + w1 * h1 + w2 * h2 + w3 * x);
        h0 = h1; h1 = h2; h2 = x;
    }
}
__global__ void k_dn_conv_hist(const float* __restrict__ proj, int ld, int ch, float* __restrict__ hist,
                               const float* __restrict__ pend, const int* __restrict__ pn, int m) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    float h0, h1, h2;
    if (m >= 3) {
        h0 = proj[(size_t)(m - 3) * ld + c]; h1 = proj[(size_t)(m - 2) * ld + c]; h2 = proj[(size_t)(m - 1) * ld + c];
    } else {
        dn_hist_after_pend(hist, pend, *pn, ch, c, h0, h1, h2);
        for (int t = 0; t < m; t++) { const float x = proj[(size_t)t * ld + c]; h0 = h1; h1 = h2; h2 = x; }
    }
    hist[c] = h0; hist[ch + c] = h1; hist[2 * ch + c] = h2;
}
// inputs of the chunked delta rule (as dn_qk_kernel of the 27B engine, same arithmetic as k_dn_rec_d): per row
// L2-normalized q (/ sqrt(128)) and k of the local key heads -> qk [m][2][kl][128], decay and beta of the local value
// heads -> gb [m][2][vl]. grid (m, kl + 1), 128 threads
__global__ void k_dn_qk(const float* __restrict__ cv, int ch, int kl, int vl, int k_first, int v_first,
                        const float* __restrict__ ab, const float* __restrict__ A, const float* __restrict__ dtb,
                        float* __restrict__ qk, float* __restrict__ gb) {
    __shared__ float red[32];
    const int r = blockIdx.x, kh = blockIdx.y, t = threadIdx.x;
    if (kh == kl) {
        for (int hv = t; hv < vl; hv += blockDim.x) {
            const float b = ab[(size_t)r * 2 * vl + hv], a = ab[(size_t)r * 2 * vl + vl + hv];
            const float xa = a + dtb[hv];
            const float spl = xa > 20.f ? xa : log1pf(__expf(xa));
            gb[(size_t)r * 2 * vl + hv] = __expf(A[hv] * spl);
            gb[(size_t)r * 2 * vl + vl + hv] = sigm(b);
        }
        return;
    }
    const float* row = cv + (size_t)r * ch;
    const float q = row[kh * 128 + t], k = row[kl * 128 + kh * 128 + t];
    const float qq = block_sum<128>(q * q, red), kk = block_sum<128>(k * k, red);
    float* o = qk + (size_t)r * 2 * kl * 128;
    o[kh * 128 + t] = q * rsqrtf(qq + 1e-6f) * rsqrtf(128.f);
    o[kl * 128 + kh * 128 + t] = k * rsqrtf(kk + 1e-6f);
}
// recurrence with deferred commit. pend: 2 parity buffers [MAXM][vl][DREC] (k 128, v 128, beta, decay) ; par = this
// forward's parity (the previous forward wrote 1 - par). Blocks of one head share pend, hence the parity buffers.
constexpr int DREC = 2 * 128 + 2;
__global__ void __launch_bounds__(128) k_dn_rec_d(const float* __restrict__ cv, int ch, int kl, int vl, int k_first,
                                                  int v_first, const float* __restrict__ ab, const float* __restrict__ A,
                                                  const float* __restrict__ dtb, float* __restrict__ ST,
                                                  float* __restrict__ pend, const int* __restrict__ pn,
                                                  const unsigned* __restrict__ step, float* __restrict__ o, int m,
                                                  bool now = false) {
    __shared__ float qs[128], ks[128];
    __shared__ float red[32];
    const int hv = blockIdx.x, cg = blockIdx.y, t = threadIdx.x;
    const int col = cg * 8 + (t >> 4), kb = (t & 15) * 8;
    const int kh = (v_first + hv) / (LNV / LNK) - k_first;
    const unsigned par = *step & 1u;
    const float* pold = pend + (size_t)(1u - par) * MAXP * vl * DREC;
    float* pnew = pend + (size_t)par * MAXP * vl * DREC;
    float* sp = ST + ((size_t)hv * 128 + col) * 128 + kb;
    float s[8];
    {
        const float4 a = *reinterpret_cast<const float4*>(sp), b = *reinterpret_cast<const float4*>(sp + 4);
        s[0] = a.x; s[1] = a.y; s[2] = a.z; s[3] = a.w; s[4] = b.x; s[5] = b.y; s[6] = b.z; s[7] = b.w;
    }
    // commit the accepted rows of the previous forward
    const int n = *pn;
    for (int r = 0; r < n; r++) {
        const float* pr = pold + ((size_t)r * vl + hv) * DREC;
        const float beta = pr[256], decay = pr[257];
        float mem = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) { s[i] *= decay; mem += s[i] * pr[kb + i]; }
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) mem += __shfl_xor_sync(0xffffffffu, mem, off);
        const float delta = (pr[128 + col] - mem) * beta;
#pragma unroll
        for (int i = 0; i < 8; i++) s[i] += pr[kb + i] * delta;
    }
    *reinterpret_cast<float4*>(sp) = make_float4(s[0], s[1], s[2], s[3]);
    *reinterpret_cast<float4*>(sp + 4) = make_float4(s[4], s[5], s[6], s[7]);
    const float Ah = A[hv], dt = dtb[hv];
    for (int r = 0; r < m; r++) {
        const float* row = cv + (size_t)r * ch;
        const float q = row[kh * 128 + t], k = row[kl * 128 + kh * 128 + t];
        const float qq = block_sum<128>(q * q, red), kk = block_sum<128>(k * k, red);
        qs[t] = q * rsqrtf(qq + 1e-6f) * rsqrtf(128.f);
        ks[t] = k * rsqrtf(kk + 1e-6f);
        __syncthreads();
        const float b = ab[(size_t)r * 2 * vl + hv], a = ab[(size_t)r * 2 * vl + vl + hv];
        const float beta = sigm(b);
        const float xa = a + dt;
        const float spl = xa > 20.f ? xa : log1pf(__expf(xa));
        const float decay = __expf(Ah * spl);
        const float v = row[2 * kl * 128 + hv * 128 + col];
        if (!now) {
            float* pw = pnew + ((size_t)r * vl + hv) * DREC;
            if (cg == 0) {
                pw[t] = ks[t];
                if (t == 0) { pw[256] = beta; pw[257] = decay; }
            }
            if ((t & 15) == 0) pw[128 + col] = v;
        }
        float mem = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) { s[i] *= decay; mem += s[i] * ks[kb + i]; }
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) mem += __shfl_xor_sync(0xffffffffu, mem, off);
        const float delta = (v - mem) * beta;
        float out = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) { s[i] += ks[kb + i] * delta; out += s[i] * qs[kb + i]; }
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) out += __shfl_xor_sync(0xffffffffu, out, off);
        if ((t & 15) == 0) o[((size_t)r * vl + hv) * 128 + col] = out;
        __syncthreads();
    }
    if (now) {  // prefill: the rows are committed at once
        *reinterpret_cast<float4*>(sp) = make_float4(s[0], s[1], s[2], s[3]);
        *reinterpret_cast<float4*>(sp + 4) = make_float4(s[4], s[5], s[6], s[7]);
    }
}
__global__ void k_step_bump(unsigned* step) { *step += 1u; }
// gated RMSNorm per value head: bf16(w * bf16(rms(o))) * sigmoid(z) -> XH (row stride ldo halves)
__global__ void k_dn_out(const float* __restrict__ o, const float* __restrict__ z, int ldz, const bf16* __restrict__ w,
                         int vl, __half* __restrict__ out, int ldo, bf16* __restrict__ outb = nullptr) {
    __shared__ float red[32];
    const int hv = blockIdx.x, t = blockIdx.y, j = threadIdx.x;
    const float v = o[((size_t)t * vl + hv) * 128 + j];
    const float ss = block_sum<128>(v * v, red);
    const float n = __bfloat162float(__float2bfloat16(v * rsqrtf(ss / 128.f + EPS)));
    const float r = __bfloat162float(w[j]) * n * sigm(z[(size_t)t * ldz + hv * 128 + j]);
    if (outb) outb[(size_t)t * ldo + hv * 128 + j] = __float2bfloat16(r);
    else store_xh(out + (size_t)t * ldo, hv * 128 + j, r);
}

// ------------------------------------------------------------------ QSA
__device__ __forceinline__ int own_row(const Plan& p, int pos) {  // -1: not owned
    const int r = pos % p.period - p.own0;
    if (r < 0 || r >= p.own) return -1;
    return pos / p.period * p.own + r;
}
// q (24 heads), k, v (2 heads) of row t: RMSNorm (1 + w) + partial RoPE (64 dims, rotate_half) -> qb [m][24 + 4][256]
// (q heads, k0, k1, v0, v1); owned rows also go to the fp8 cache. grid (28, m), 256 threads
// proj rows [q nqp heads (global qh0..)][gate local][k 2][v 2]; grid (nqp + 4, m)
__global__ void k_qsa_prep(const float* __restrict__ proj, int ld, const bf16* __restrict__ qn, const bf16* __restrict__ kn,
                           const int* __restrict__ pos0, float* __restrict__ qb, uint8_t* __restrict__ kc,
                           uint8_t* __restrict__ vc, float* __restrict__ ks, float* __restrict__ vs, Plan p, int nqp, int qh0,
                           const int* __restrict__ rpos = nullptr, const int* __restrict__ rbase = nullptr) {
    __shared__ float red[32];
    __shared__ float xs[HD];
    const int hb = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int h = hb < nqp ? qh0 + hb : NQ + hb - nqp;  // q head (global) or NQ + {k0, k1, v0, v1}
    // batched rows (rpos / rbase, flash_batch.cuh): the row's own position and KV region (first row of its slot)
    const int pos = rpos ? rpos[t] : *pos0 + t;
    if (rbase) { const size_t rb = rbase[t]; kc += rb * NKV * HD; vc += rb * NKV * HD; ks += rb * NKV; vs += rb * NKV; }
    const int kvo = (nqp + (p.q1 - p.q0)) * HD;
    const float* src;
    const bf16* w = nullptr;
    if (h < NQ) { src = proj + (size_t)t * ld + hb * HD; w = qn; }
    else if (h < NQ + 2) { src = proj + (size_t)t * ld + kvo + (h - NQ) * HD; w = kn; }
    else src = proj + (size_t)t * ld + kvo + 2 * HD + (h - NQ - 2) * HD;
    float x = src[d];
    if (w) {
        const float ss = block_sum<256>(x * x, red);
        x = x * rsqrtf(ss / HD + EPS) * (1.f + __bfloat162float(w[d]));
        xs[d] = x;
        __syncthreads();
        if (d < ROT) {
            const int i = d & 31;
            const float f = (float)pos * powf((float)THETA, -2.f * i / ROT);
            float sn, cs;
            sincosf(f, &sn, &cs);
            x = d < 32 ? xs[d] * cs - xs[d + 32] * sn : xs[d] * cs + xs[d - 32] * sn;
        }
    }
    qb[((size_t)t * (NQ + 4) + h) * HD + d] = x;
    if (h >= NQ) {  // k / v -> cache if owned (fp8 e4m3, scale per (row, head))
        const int row = own_row(p, pos);
        if (row < 0) return;
        const float mx = block_max<256>(fabsf(x), red);
        const float sc = mx > 0.f ? mx / 448.f : 1.f;
        const int kvh = (h - NQ) & 1;
        const bool isk = h < NQ + 2;
        __nv_fp8_e4m3 q8(x / sc);
        (isk ? kc : vc)[((size_t)row * NKV + kvh) * HD + d] = *reinterpret_cast<uint8_t*>(&q8);
        if (d == 0) (isk ? ks : vs)[row * NKV + kvh] = sc;
    }
}
// indexer: q heads (layernorm 1 + w, RoPE at pos), raw key into the tail ring; a completed block of 4 is pooled
// (mean, layernorm, RoPE at the block start) and stored by its owner. Rows in order: one block, 128 threads.
__global__ void k_idx_prep(const float* __restrict__ ip, int ld, const bf16* __restrict__ qn, const bf16* __restrict__ kn,
                           const int* __restrict__ pos0, float* __restrict__ iq, float* __restrict__ tail,
                           bf16* __restrict__ ikb, Plan p, int m, const int* __restrict__ kvb0 = nullptr) {
    __shared__ float red[32];
    __shared__ float xs[IDX_HD];
    const int d = threadIdx.x;
    if (kvb0) ikb += (size_t)(*kvb0 / IDX_R) * IDX_HD;  // slot region (rows)
    auto norm_rope = [&](float x, const bf16* w, int pos) {
        const float ss = block_sum<128>(x * x, red);
        x = x * rsqrtf(ss / IDX_HD + EPS) * (1.f + __bfloat162float(w[d]));
        __syncthreads();
        xs[d] = x;
        __syncthreads();
        if (d < ROT) {
            const int i = d & 31;
            const float f = (float)pos * powf((float)THETA, -2.f * i / ROT);
            float sn, cs;
            sincosf(f, &sn, &cs);
            x = d < 32 ? xs[d] * cs - xs[d + 32] * sn : xs[d] * cs + xs[d - 32] * sn;
        }
        return x;
    };
    for (int t = 0; t < m; t++) {
        const int pos = *pos0 + t;
        for (int h = 0; h < IDX_NH; h++) iq[((size_t)t * IDX_NH + h) * IDX_HD + d] = norm_rope(ip[(size_t)t * ld + h * IDX_HD + d], qn, pos);
        const float rk = __bfloat162float(__float2bfloat16(ip[(size_t)t * ld + IDX_NH * IDX_HD + d]));
        tail[(pos & (IDX_TAIL - 1)) * IDX_HD + d] = rk;
        __syncthreads();
        if ((pos & 3) == 3) {
            float s = 0.f;
            for (int i = 0; i < 4; i++) s += tail[((pos - i) & (IDX_TAIL - 1)) * IDX_HD + d];
            const float pooled = __bfloat162float(__float2bfloat16(s * 0.25f));
            const float kr = norm_rope(pooled, kn, pos - 3);
            const int row = own_row(p, pos - 3);
            if (row >= 0) ikb[(size_t)(row >> 2) * IDX_HD + d] = __float2bfloat16(kr);
        }
        __syncthreads();
    }
}
// Row-parallel k_idx_prep for prefill chunks: grid m, block t does the q heads of row t and, when pos_t closes a
// block of 4, its pooled key (raw keys of earlier rows from ip or, before the chunk, from tail). The tail update
// (last raw keys of the chunk) is k_idx_tail, launched after.
__global__ void k_idx_prep_rows(const float* __restrict__ ip, int ld, const bf16* __restrict__ qn, const bf16* __restrict__ kn,
                                const int* __restrict__ pos0, float* __restrict__ iq, const float* __restrict__ tail,
                                bf16* __restrict__ ikb, Plan p, const int* __restrict__ kvb0 = nullptr) {
    __shared__ float red[32];
    __shared__ float xs[IDX_HD];
    const int d = threadIdx.x, t = blockIdx.x;
    const int pos = *pos0 + t;
    if (kvb0) ikb += (size_t)(*kvb0 / IDX_R) * IDX_HD;  // slot region (rows)
    auto norm_rope = [&](float x, const bf16* w, int ps) {
        const float ss = block_sum<128>(x * x, red);
        x = x * rsqrtf(ss / IDX_HD + EPS) * (1.f + __bfloat162float(w[d]));
        __syncthreads();
        xs[d] = x;
        __syncthreads();
        if (d < ROT) {
            const int i = d & 31;
            const float f = (float)ps * powf((float)THETA, -2.f * i / ROT);
            float sn, cs;
            sincosf(f, &sn, &cs);
            x = d < 32 ? xs[d] * cs - xs[d + 32] * sn : xs[d] * cs + xs[d - 32] * sn;
        }
        return x;
    };
    for (int h = 0; h < IDX_NH; h++) iq[((size_t)t * IDX_NH + h) * IDX_HD + d] = norm_rope(ip[(size_t)t * ld + h * IDX_HD + d], qn, pos);
    if ((pos & 3) == 3) {
        float s = 0.f;
        for (int i = 0; i < 4; i++) {
            const int tt = t - i;
            s += tt >= 0 ? __bfloat162float(__float2bfloat16(ip[(size_t)tt * ld + IDX_NH * IDX_HD + d])) : tail[((pos - i) & (IDX_TAIL - 1)) * IDX_HD + d];
        }
        const float pooled = __bfloat162float(__float2bfloat16(s * 0.25f));
        const float kr = norm_rope(pooled, kn, pos - 3);
        const int row = own_row(p, pos - 3);
        if (row >= 0) ikb[(size_t)(row >> 2) * IDX_HD + d] = __float2bfloat16(kr);
    }
}
__global__ void k_idx_tail(const float* __restrict__ ip, int ld, const int* __restrict__ pos0, float* __restrict__ tail, int m) {
    const int d = threadIdx.x;
    // every row (a verify step's accepted rows can be any prefix of them), the last IDX_TAIL of a prefill chunk
    for (int t = max(0, m - IDX_TAIL); t < m; t++)
        tail[((*pos0 + t) & (IDX_TAIL - 1)) * IDX_HD + d] = __bfloat162float(__float2bfloat16(ip[(size_t)t * ld + IDX_NH * IDX_HD + d]));
}
// Partial attention of all 24 q heads over the keys this card owns with position <= pos_t (short context:
// every key selected). out row [24][HD + 2]: unnormalized sum, max, sum of exp. grid (24, m), 256 threads.
__global__ void k_qsa_attn(const float* __restrict__ qb, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
                           const float* __restrict__ ks, const float* __restrict__ vs, const int* __restrict__ pos0,
                           float* __restrict__ out, Plan p, const int* __restrict__ kvb0 = nullptr) {
    __shared__ float qs[HD];
    __shared__ float wm[8], wl[8];
    __shared__ float wacc[8][HD];
    const int h = blockIdx.x, t = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (kvb0) { const size_t rb = *kvb0; kc += rb * NKV * HD; vc += rb * NKV * HD; ks += rb * NKV; vs += rb * NKV; }  // slot region
    const int kvh = h / (NQ / NKV);
    const int pos = *pos0 + t;
    qs[threadIdx.x] = qb[((size_t)t * (NQ + 4) + h) * HD + threadIdx.x] * (1.f / 16.f);
    __syncthreads();
    float q8[8];
#pragma unroll
    for (int e = 0; e < 8; e++) q8[e] = qs[lane * 8 + e];
    float mx = -INFINITY, l = 0.f, acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    // owned positions <= pos: rows 0..nrows-1 where row r <-> position (r / own) * period + own0 + r % own
    const int nper = pos / p.period, rem = pos % p.period - p.own0;
    const int nrows = nper * p.own + (rem < 0 ? 0 : min(rem + 1, p.own));
    for (int r = warp; r < nrows; r += 8) {
        const uint8_t* kr = kc + ((size_t)r * NKV + kvh) * HD + lane * 8;
        const uint2 kw = *reinterpret_cast<const uint2*>(kr);
        const uint8_t* kb = reinterpret_cast<const uint8_t*>(&kw);
        float s = 0.f;
#pragma unroll
        for (int e = 0; e < 8; e++) s += q8[e] * float(*reinterpret_cast<const __nv_fp8_e4m3*>(kb + e));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        s *= ks[r * NKV + kvh];
        const float nm = fmaxf(mx, s);
        const float c = __expf(mx - nm), e1 = __expf(s - nm);
        l = l * c + e1;
        const uint2 vw = *reinterpret_cast<const uint2*>(vc + ((size_t)r * NKV + kvh) * HD + lane * 8);
        const uint8_t* vb = reinterpret_cast<const uint8_t*>(&vw);
        const float vsc = vs[r * NKV + kvh] * e1;
#pragma unroll
        for (int e = 0; e < 8; e++) acc[e] = acc[e] * c + vsc * float(*reinterpret_cast<const __nv_fp8_e4m3*>(vb + e));
        mx = nm;
    }
    if (lane == 0) { wm[warp] = mx; wl[warp] = l; }
#pragma unroll
    for (int e = 0; e < 8; e++) wacc[warp][lane * 8 + e] = acc[e];
    __syncthreads();
    float M = -INFINITY;
    for (int w = 0; w < 8; w++) M = fmaxf(M, wm[w]);
    float* o = out + ((size_t)t * NQ + h) * (HD + 2);
    float a = 0.f, L = 0.f;
    for (int w = 0; w < 8; w++) {
        const float c = wm[w] == -INFINITY ? 0.f : __expf(wm[w] - M);
        a += c * wacc[w][threadIdx.x];
        L += c * wl[w];
    }
    o[threadIdx.x] = a;
    if (threadIdx.x == 0) { o[HD] = M; o[HD + 1] = L; }
}
// P2P exchange of the partial attention: every card sends each peer the rows of that peer's 8 q heads
// ([MAXM][8][HD + 2]) into the peer's receive buffer [2 slots][ngpu senders][MAXM][8][HD + 2], then raises
// the peer's flag[sender] = s (release, system scope). One block per peer.
struct Xchg {
    float* recv[4];       // receive buffer of every rank (device pointers, peer access)
    unsigned* flags[4];   // [ngpu senders] flag words of every rank
    unsigned* seq;        // local exchange counter
    unsigned* done;       // local arrival counter of k_qsa_comb2 blocks
    unsigned* scnt;       // local per-peer arrival counters of the sending blocks
};
constexpr int XROWS = PFM;  // rows of the exchange buffers (a whole prefill chunk)
// slots of the QSA exchange receive ring (q / partials / indexer); the slot index is s & (XSLOT - 1). 2 slots are
// enough: the consumer accepts flag >= s, and the per-layer all-reduce keeps the cards within one exchange of each
// other. (prefill-r4 tried 4 slots and a consumer -> sender credit handshake while hunting the 2-GPU garbage: no
// effect, the cause was a load-time copy race; bench/patches/qfn_r4_garbage_hunt_full.patch)
constexpr int XSLOT = 2;
// per-row floats of a receive slot: the QSA q heads of one card (3 cards: 8, 2 cards: 12) x (HD + 2) partial records;
// the q exchange uses qper (x HD). Allocations: the plan's value (3 cards: the former constant 8 x (HD + 2))
__host__ __device__ __forceinline__ int QPC(const Plan& p) { return NQ / p.ngpu; }
__host__ __device__ __forceinline__ int xper(const Plan& p) { return QPC(p) * (HD + 2); }
__host__ __device__ __forceinline__ int qper(const Plan& p) { return QPC(p) * HD; }
// grid (peers, blocks per peer): the last block of a peer raises its flag
__device__ __forceinline__ void xsend_done(Xchg& xc, int peer, unsigned s, const Plan& p) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        if (atomicAdd(xc.scnt + peer, 1u) == gridDim.y - 1) {
            xc.scnt[peer] = 0u;
            __threadfence_system();
            qe_st_release_sys(xc.flags[peer] + p.rank, s);
        }
    }
}
// Row all-gather of the sequence-parallel HC prefill: every card copies its own rows [r0, r1) of up to 3 arrays
// (x bf16, xh fp16, or the residual R) into the same rows of every peer's array (P2P stores), then raises the peer's
// flag[sender] = s; k_row_ag_wait waits for the flags of all peers and advances the local counter.
struct RowAg {
    char* dst[3][4];      // [array][rank] base pointers (device, peer access)
    int rowb[3];          // bytes per row (multiple of 16; 0: array unused)
    unsigned* flags[4];   // [rank] flag words [ngpu senders]
    unsigned* seq;        // local counter
    unsigned* scnt;       // local per-peer block counters
    unsigned* acks[4];    // pull mode: [rank] ack words [ngpu readers] (a reader is done with the owner's rows)
    int rr[8];            // pull mode: rows [rr[2g], rr[2g + 1]) of rank g
    unsigned pm;          // Comm::pullm (bit 4 i + j: rank i reads rank j's rows)
};
// Pull variant of k_row_ag (prefill-r4): SM stores into the peers' memory stall this card's memory system for the
// kernels next to them (tools/mb/mb_gdn.cu), loads of the same bytes from the peer barely. Per directed pair (a.pm bit
// 4 i + j: rank i reads rank j's rows, Comm::pullm): k_row_ag_mix stores this card's rows into the pushed peers' arrays
// (+ their flag after the rows) and raises the "rows ready" flag of the pulling peers; k_row_ag_pull (block x = peer
// slot) waits for a pulled peer's flag, copies its rows into the local arrays (P2P loads) and acks in the peer's
// memory; k_row_ag_fin waits for the pushed peers' rows and for the acks of the peers that read this card's rows (the
// MoE quantizes them in place next), then advances the sequence.
__device__ __forceinline__ bool ag_pull(unsigned pm, int reader, int owner) { return (pm >> (4 * reader + owner)) & 1u; }
__global__ void k_row_ag_mix(RowAg a, int r0, int r1, int rank, int ngpu, unsigned amask) {
    const int peer = blockIdx.x >= (unsigned)rank ? blockIdx.x + 1 : blockIdx.x;
    const unsigned s = *a.seq + 1;
    if (ag_pull(a.pm, peer, rank)) {
        if (blockIdx.y == 0 && threadIdx.x == 0) { __threadfence_system(); qe_st_release_sys(a.flags[peer] + rank, s); }
        return;
    }
    for (int k = 0; k < 3; k++) {
        if (!((amask >> k) & 1) || r1 <= r0) continue;
        const size_t n16 = (size_t)(r1 - r0) * a.rowb[k] / 16;
        const uint4* src = reinterpret_cast<const uint4*>(a.dst[k][rank] + (size_t)r0 * a.rowb[k]);
        uint4* dst = reinterpret_cast<uint4*>(a.dst[k][peer] + (size_t)r0 * a.rowb[k]);
        for (size_t i = (size_t)blockIdx.y * blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.y * blockDim.x) dst[i] = src[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        if (atomicAdd(a.scnt + peer, 1u) == gridDim.y - 1) {
            a.scnt[peer] = 0u;
            __threadfence_system();
            qe_st_release_sys(a.flags[peer] + rank, s);
        }
    }
}
__global__ void k_row_ag_pull(RowAg a, int rank, int ngpu, unsigned amask) {
    const int peer = blockIdx.x >= (unsigned)rank ? blockIdx.x + 1 : blockIdx.x;
    if (!ag_pull(a.pm, rank, peer)) return;
    const unsigned s = *a.seq + 1;
    if (threadIdx.x == 0)
        for (unsigned long long it = 0, ws_ = 0; (int)(qe_ld_acquire_sys(a.flags[rank] + peer) - s) < 0; it++) {
            __nanosleep(20);
            QE_WD_POLL(ws_, kWdRowAg, peer, s, qe_ld_acquire_sys(a.flags[rank] + peer));
            if (it == (1ull << 27)) printf("[flash] stall: rank %d row-AG pull wait for %d: seq %u\n", rank, peer, s);
        }
    __syncthreads();
    const int r0 = a.rr[2 * peer], r1 = a.rr[2 * peer + 1];
    for (int k = 0; k < 3; k++) {
        if (!((amask >> k) & 1) || r1 <= r0) continue;
        const size_t n16 = (size_t)(r1 - r0) * a.rowb[k] / 16;
        const uint4* src = reinterpret_cast<const uint4*>(a.dst[k][peer] + (size_t)r0 * a.rowb[k]);
        uint4* dst = reinterpret_cast<uint4*>(a.dst[k][rank] + (size_t)r0 * a.rowb[k]);
        size_t i = (size_t)blockIdx.y * blockDim.x + threadIdx.x;
        const size_t st = (size_t)gridDim.y * blockDim.x;
        for (; i + st < n16; i += 2 * st) {  // two loads in flight per thread
            const uint4 v0 = __ldcv(src + i), v1 = __ldcv(src + i + st);
            dst[i] = v0;
            dst[i + st] = v1;
        }
        if (i < n16) dst[i] = __ldcv(src + i);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(a.scnt + 4 + peer, 1u) == gridDim.y - 1) {
            a.scnt[4 + peer] = 0u;
            qe_st_release_sys(a.acks[peer] + rank, s);
        }
    }
}
__global__ void k_row_ag_fin(RowAg a, int rank, int ngpu) {
    if (threadIdx.x != 0) return;
    const unsigned s = *a.seq + 1;
    for (int g = 0; g < ngpu; g++) {
        if (g == rank) continue;
        if (!ag_pull(a.pm, rank, g))  // g pushed its rows: its flag lands after them
            for (unsigned long long it = 0, ws_ = 0; (int)(qe_ld_acquire_sys(a.flags[rank] + g) - s) < 0; it++) {
                __nanosleep(20);
                QE_WD_POLL(ws_, kWdRowAg, g, s, qe_ld_acquire_sys(a.flags[rank] + g));
                if (it == (1ull << 27)) printf("[flash] stall: rank %d row-AG wait for %d: seq %u\n", rank, g, s);
            }
        if (ag_pull(a.pm, g, rank))  // g reads this card's rows: wait until it is done
            for (unsigned long long it = 0, ws_ = 0; (int)(qe_ld_acquire_sys(a.acks[rank] + g) - s) < 0; it++) {
                __nanosleep(20);
                QE_WD_POLL(ws_, kWdRowAg, g, s, qe_ld_acquire_sys(a.acks[rank] + g));
                if (it == (1ull << 27)) printf("[flash] stall: rank %d row-AG ack wait for %d: seq %u\n", rank, g, s);
            }
    }
    __threadfence_system();
    *a.seq = s;
}
__global__ void k_row_ag(RowAg a, int r0, int r1, int rank, int ngpu, unsigned amask) {
    const int peer = blockIdx.x >= (unsigned)rank ? blockIdx.x + 1 : blockIdx.x;
    const unsigned s = *a.seq + 1;
    for (int k = 0; k < 3; k++) {
        if (!((amask >> k) & 1) || r1 <= r0) continue;
        const size_t n16 = (size_t)(r1 - r0) * a.rowb[k] / 16;
        const uint4* src = reinterpret_cast<const uint4*>(a.dst[k][rank] + (size_t)r0 * a.rowb[k]);
        uint4* dst = reinterpret_cast<uint4*>(a.dst[k][peer] + (size_t)r0 * a.rowb[k]);
        for (size_t i = (size_t)blockIdx.y * blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.y * blockDim.x) dst[i] = src[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        if (atomicAdd(a.scnt + peer, 1u) == gridDim.y - 1) {
            a.scnt[peer] = 0u;
            __threadfence_system();
            qe_st_release_sys(a.flags[peer] + rank, s);
        }
    }
}
// x (bf16) of gathered rows from their fp16 copy: only xh crosses the links (5 KB per row, as the all-reduce's sums);
// every card converts all rows of the chunk, own ones included, so all cards hold the same x
__global__ void k_xh_to_x(const __half* __restrict__ xh, bf16* __restrict__ x, size_t n2) {
    // n2 = rows * H / 2; xh rows are in the XH column order (xh_pos) and scaled by 1 / kXhScale (clamped at kXhClamp)
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n2; i += (size_t)gridDim.x * blockDim.x) {
        const size_t e = 2 * i, r = e / H;
        const int k = (int)(e % H);
        const __half* row = xh + r * H;
        const float a = __half2float(row[xh_pos(k)]) * kXhScale, b = __half2float(row[xh_pos(k + 1)]) * kXhScale;
        reinterpret_cast<bf162*>(x)[i] = __floats2bfloat162_rn(a, b);
    }
}
// copy-engine variant: the rows went by cudaMemcpyAsync (peer copies on the same stream, complete before this
// kernel starts); raise every peer's flag[rank] = s
__global__ void k_row_ag_flag(RowAg a, int rank, int ngpu) {
    if (threadIdx.x != 0) return;
    const unsigned s = *a.seq + 1;
    __threadfence_system();
    for (int g = 0; g < ngpu; g++)
        if (g != rank) qe_st_release_sys(a.flags[g] + rank, s);
}
__global__ void k_row_ag_wait(RowAg a, int rank, int ngpu) {
    if (threadIdx.x != 0) return;
    const unsigned s = *a.seq + 1;
    for (int g = 0; g < ngpu; g++)
        if (g != rank)
            for (unsigned long long it = 0, ws_ = 0; (int)(qe_ld_acquire_sys(a.flags[rank] + g) - s) < 0; it++) {
                __nanosleep(20);
                QE_WD_POLL(ws_, kWdRowAg, g, s, qe_ld_acquire_sys(a.flags[rank] + g));
                if (it == (1ull << 27)) printf("[flash] stall: rank %d row-AG wait for %d: seq %u, seen %u\n", rank, g, s, qe_ld_acquire_sys(a.flags[rank] + g));
            }
    __threadfence_system();
    *a.seq = s;
}
// ---- staged prefill transfers (comm_select_path: FLASH_COMM=ce / host, or auto on slow SM stores)
// host-staged row all-gather: the sender waits until every peer has copied its previous rows out of its pinned staging
// rows (ack words, raised after the peer's H2D), so the staging rows can be reused
struct AckP {
    unsigned* w[4];  // [rank] the ack words [4 senders] of every card for this channel
};
__global__ void k_rag_ackwait(const unsigned* __restrict__ mine, const unsigned* seq, int rank, int ngpu) {
    if (threadIdx.x != 0) return;
    const unsigned s = *seq;
    for (int g = 0; g < ngpu; g++)
        if (g != rank)
            for (unsigned long long it = 0, ws_ = 0; (int)(qe_ld_acquire_sys(mine + g) - s) < 0; it++) {
                __nanosleep(64);
                QE_WD_POLL(ws_, kWdRowAg, g, s, qe_ld_acquire_sys(mine + g));
                if (it == (1ull << 26)) printf("[flash] stall: rank %d row-AG ack wait for %d: seq %u, seen %u\n", rank, g, s, qe_ld_acquire_sys(mine + g));
            }
}
__global__ void k_rag_ack(AckP a, const unsigned* seq, int rank, int ngpu) {
    if (threadIdx.x != 0) return;
    const unsigned s = *seq;
    __threadfence_system();
    for (int g = 0; g < ngpu; g++)
        if (g != rank) qe_st_release_sys(a.w[g] + rank, s);
}
// staged QSA exchange: the send kernel wrote into the local outbox, the copies follow; raise the receivers' flags with
// the exchange's sequence number (flags: [rank] the flag words [4 senders] of every card)
struct FlagP {
    unsigned* f[4];
};
__global__ void k_xs_flag(FlagP p, const unsigned* seq, int rank, int ngpu) {
    if (threadIdx.x != 0) return;
    const unsigned s = *seq + 1;
    __threadfence_system();
    for (int g = 0; g < ngpu; g++)
        if (g != rank) qe_st_release_sys(p.f[g] + rank, s);
}
// host-staged exchange: the peers' rows are in the receive slot (H2D before this kernel): raise this card's own flag
// words of every sender, which the consumer kernel waits for
// copy-engine staged transfers: the transfer's flag value (*seq + 1) into a local word, which the stream's copies then
// write into the peers' flag words after the rows
__global__ void k_seq_word(unsigned* w, const unsigned* seq) { *w = *seq + 1; }
__global__ void k_xs_local(unsigned* mine, const unsigned* seq, int rank, int ngpu) {
    if (threadIdx.x != 0) return;
    const unsigned s = *seq + 1;
    __threadfence();
    for (int g = 0; g < ngpu; g++)
        if (g != rank) qe_st_release_sys(mine + g, s);
}
// wait for the peers' flags of the exchange (host words), without advancing the counter (the consumer kernel does)
__global__ void k_xs_wait(const unsigned* __restrict__ mine, const unsigned* seq, int rank, int ngpu) {
    if (threadIdx.x != 0) return;
    const unsigned s = *seq + 1;
    for (int g = 0; g < ngpu; g++)
        if (g != rank)
            for (unsigned long long it = 0, ws_ = 0; (int)(qe_ld_acquire_sys(mine + g) - s) < 0; it++) {
                __nanosleep(64);
                QE_WD_POLL(ws_, kWdRowAg, g, s, qe_ld_acquire_sys(mine + g));
                if (it == (1ull << 26)) printf("[flash] stall: rank %d staged exchange wait for %d: seq %u, seen %u\n", rank, g, s, qe_ld_acquire_sys(mine + g));
            }
    __threadfence_system();
}
// receive slots of the QSA exchanges by an even counter at the start of a prefill batch: the host then knows the slot
// parity of every staged exchange of the batch (its copies are addressed on the host, also in a replayed graph)
// staged batches: align the exchange counters to a multiple of XSLOT, so exchange k of the batch (host counter k = 1,
// 2, ...) uses slot k & (XSLOT - 1), the slot the consumers read (s & (XSLOT - 1), s = seq + 1); was parity (2 slots)
__global__ void k_xpar_align(unsigned* a, unsigned* b) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *a += (XSLOT - *a % XSLOT) % XSLOT;
        *b += (XSLOT - *b % XSLOT) % XSLOT;
    }
}
__global__ void k_xsend(const float* __restrict__ part, int m, Xchg xc, Plan p) {
    const int peer = blockIdx.x >= (unsigned)p.rank ? blockIdx.x + 1 : blockIdx.x;
    const unsigned s = *xc.seq + 1;
    float* dst = xc.recv[peer] + ((size_t)(s & (XSLOT - 1)) * p.ngpu + p.rank) * XROWS * xper(p);
    const int h0 = peer * QPC(p);
    for (int idx = blockIdx.y * blockDim.x + threadIdx.x; idx < m * xper(p) / 2; idx += gridDim.y * blockDim.x) {
        const int t = idx / (xper(p) / 2), r = idx % (xper(p) / 2);
        reinterpret_cast<float2*>(dst)[idx] = reinterpret_cast<const float2*>(part + ((size_t)t * NQ + h0) * (HD + 2))[r];
    }
    xsend_done(xc, peer, s, p);
}
// wait for the peers' partials, combine with the own one, * sigmoid(gate) -> bf16 input of o_proj
__global__ void k_qsa_comb2(const float* __restrict__ part, int m, const float* __restrict__ proj, int ld,
                            bf16* __restrict__ out, int ldo, Xchg xc, Plan p, int gate_off) {
    __shared__ unsigned s_s;
    if (threadIdx.x == 0) {
        const unsigned s = *xc.seq + 1;
        for (int g = 0; g < p.ngpu; g++)
            if (g != p.rank)
                for (unsigned ws_ = 0; (int)(qe_ld_acquire_sys(xc.flags[p.rank] + g) - s) < 0;) { __nanosleep(20); QE_WD_POLL(ws_, kWdQsaX, g, s, qe_ld_acquire_sys(xc.flags[p.rank] + g)); }
        s_s = s;
    }
    __syncthreads();
    const unsigned s = s_s;
    const int hl = blockIdx.x, t = blockIdx.y, d = threadIdx.x, h = p.q0 + hl;
    const float* rb = xc.recv[p.rank] + (size_t)(s & (XSLOT - 1)) * p.ngpu * XROWS * xper(p);
    float M = -INFINITY;
    for (int g = 0; g < p.ngpu; g++) {
        const float* o = g == p.rank ? part + ((size_t)t * NQ + h) * (HD + 2) : rb + ((size_t)g * XROWS + t) * xper(p) + hl * (HD + 2);
        M = fmaxf(M, __ldcg(o + HD));
    }
    float a = 0.f, L = 0.f;
    for (int g = 0; g < p.ngpu; g++) {
        const float* o = g == p.rank ? part + ((size_t)t * NQ + h) * (HD + 2) : rb + ((size_t)g * XROWS + t) * xper(p) + hl * (HD + 2);
        const float mg = __ldcg(o + HD);
        if (mg == -INFINITY) continue;
        const float c = __expf(mg - M);
        a += c * __ldcg(o + d);
        L += c * __ldcg(o + HD + 1);
    }
    const float gate = proj[(size_t)t * ld + gate_off + hl * HD + d];
    out[(size_t)t * ldo + hl * HD + d] = __float2bfloat16(a / L * sigm(gate));
    // last block: the exchange is complete on this card
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(xc.done, 1u) == gridDim.x * gridDim.y - 1) {
            *xc.done = 0;
            *xc.seq = s;
            __threadfence();
        }
    }
}
// ------------------------------------------------------------------ QSA block selection (indexer)
// owned positions <= pos (rows of this card's cache)
__device__ __forceinline__ int owned_upto(const Plan& p, int pos) {
    if (pos < 0) return 0;
    const int nper = pos / p.period, rem = pos % p.period - p.own0;
    return nper * p.own + (rem < 0 ? 0 : min(rem + 1, p.own));
}
// scores[t][lb] = sum_h relu(iq_h . K_lb) / sqrt(128) of this card's complete visible blocks (when the row selects:
// more than IDX_TOPB complete blocks visible). grid (ceil(lbmax / 256), m)
__global__ void k_idx_score(const bf16* __restrict__ ikb, const float* __restrict__ iq, const int* __restrict__ pos0,
                            Plan p, int lbmax, float* __restrict__ scores, const int* __restrict__ rpos = nullptr,
                            const int* __restrict__ rbase = nullptr) {
    __shared__ float qs[IDX_NH * IDX_HD];
    const int t = blockIdx.y;
    const int pos = rpos ? rpos[t] : *pos0 + t, nbg = (pos + 1) / IDX_R;
    if (rbase) ikb += (size_t)(rbase[t] / IDX_R) * IDX_HD;
    if (nbg <= IDX_TOPB) return;
    for (int i = threadIdx.x; i < IDX_NH * IDX_HD; i += blockDim.x) qs[i] = iq[(size_t)t * IDX_NH * IDX_HD + i];
    __syncthreads();
    const int nlb = owned_upto(p, IDX_R * nbg - 1) / IDX_R;
    const int lb = blockIdx.x * blockDim.x + threadIdx.x;
    if (lb >= nlb) return;
    const uint4* kp = reinterpret_cast<const uint4*>(ikb + (size_t)lb * IDX_HD);
    float acc[IDX_NH] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
    for (int c = 0; c < IDX_HD / 8; c++) {
        const uint4 v = kp[c];
        const bf16* kb = reinterpret_cast<const bf16*>(&v);
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const float kf = __bfloat162float(kb[e]);
#pragma unroll
            for (int h = 0; h < IDX_NH; h++) acc[h] += qs[h * IDX_HD + c * 8 + e] * kf;
        }
    }
    float sc = 0.f;
#pragma unroll
    for (int h = 0; h < IDX_NH; h++) sc += fmaxf(acc[h], 0.f);
    scores[(size_t)t * lbmax + lb] = sc * rsqrtf((float)IDX_HD);
}
// the same scores for a tile of 16 rows x 64 blocks per block: the 16 rows' queries and the 64 blocks' keys staged in
// shared memory once (keys read once per 16 rows instead of once per row). grid (ceil(lbmax / 64), ceil(m / 16)),
// 256 threads, dynamic smem IS_SMEM. Same arithmetic per score as k_idx_score (fp32, heads summed in order).
constexpr int IS_ROWS = 16, IS_BLK = 64;
constexpr size_t IS_SMEM = (size_t)(IS_ROWS * IDX_NH * IDX_HD + IDX_HD * (IS_BLK + 1)) * 4;
__global__ void __launch_bounds__(256) k_idx_score_t(const bf16* __restrict__ ikb, const float* __restrict__ iq,
                                                     const int* __restrict__ pos0, Plan p, int lbmax, int m,
                                                     float* __restrict__ scores, const int* __restrict__ kvb0 = nullptr) {
    extern __shared__ float is_sm[];
    if (kvb0) ikb += (size_t)(*kvb0 / IDX_R) * IDX_HD;  // slot region (rows, uniform)
    float* qs = is_sm;                                // [16 rows][4 heads][128]
    float* ks = is_sm + IS_ROWS * IDX_NH * IDX_HD;    // [128][65] (transposed, padded)
    const int r0 = blockIdx.y * IS_ROWS, b0 = blockIdx.x * IS_BLK, tid = threadIdx.x;
    // the tile's largest block count (its last valid row)
    const int rl = min(m, r0 + IS_ROWS) - 1;
    const int posl = *pos0 + rl, nbgl = (posl + 1) / IDX_R;
    const int nlbl = nbgl > IDX_TOPB ? owned_upto(p, IDX_R * nbgl - 1) / IDX_R : 0;
    if (b0 >= nlbl) return;
    for (int i = tid; i < IS_ROWS * IDX_NH * IDX_HD; i += 256) {
        const int r = i / (IDX_NH * IDX_HD);
        qs[i] = r0 + r < m ? iq[(size_t)(r0 + r) * IDX_NH * IDX_HD + i % (IDX_NH * IDX_HD)] : 0.f;
    }
    for (int i = tid; i < IS_BLK * IDX_HD / 8; i += 256) {
        const int b = i / (IDX_HD / 8), c = i % (IDX_HD / 8);
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (b0 + b < lbmax) v = *reinterpret_cast<const uint4*>(ikb + (size_t)(b0 + b) * IDX_HD + c * 8);
        const bf16* kb = reinterpret_cast<const bf16*>(&v);
#pragma unroll
        for (int e = 0; e < 8; e++) ks[(c * 8 + e) * (IS_BLK + 1) + b] = __bfloat162float(kb[e]);
    }
    __syncthreads();
    const int r = tid / 16, bq = tid % 16;  // row r, blocks bq + 16 j
    float acc[4][IDX_NH];
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
        for (int h = 0; h < IDX_NH; h++) acc[j][h] = 0.f;
    const float* qr = qs + r * IDX_NH * IDX_HD;
#pragma unroll 4
    for (int k = 0; k < IDX_HD; k++) {
        float kk[4];
#pragma unroll
        for (int j = 0; j < 4; j++) kk[j] = ks[k * (IS_BLK + 1) + bq + 16 * j];
#pragma unroll
        for (int h = 0; h < IDX_NH; h++) {
            const float qv = qr[h * IDX_HD + k];
#pragma unroll
            for (int j = 0; j < 4; j++) acc[j][h] += qv * kk[j];
        }
    }
    const int t = r0 + r;
    if (t >= m) return;
    const int pos = *pos0 + t, nbg = (pos + 1) / IDX_R;
    if (nbg <= IDX_TOPB) return;
    const int nlb = owned_upto(p, IDX_R * nbg - 1) / IDX_R;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int lb = b0 + bq + 16 * j;
        if (lb >= nlb) continue;
        float sc = 0.f;
#pragma unroll
        for (int h = 0; h < IDX_NH; h++) sc += fmaxf(acc[j][h], 0.f);
        scores[(size_t)t * lbmax + lb] = sc * rsqrtf((float)IDX_HD);
    }
}
// k-th largest key (1-based) of n keys in smem via 4 passes of 8-bit radix (block-wide); returns the key
// the same over keys in global memory written by this block (k_idx_local<true>): coherent (L2) loads
__device__ unsigned radix_kth_g(const unsigned* keys, int n, int k, unsigned* hist) {
    unsigned prefix = 0u, mask = 0u;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0u;
        __syncthreads();
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const unsigned kk = __ldcg(keys + i);
            if ((kk & mask) == prefix) atomicAdd(&hist[(kk >> shift) & 255u], 1u);
        }
        __syncthreads();
        __shared__ unsigned s_digit, s_rem;
        if (threadIdx.x == 0) {
            unsigned acc = 0u;
            int d = 255;
            for (; d > 0; d--) {
                if (acc + hist[d] >= (unsigned)k) break;
                acc += hist[d];
            }
            s_digit = d;
            s_rem = k - acc;
        }
        __syncthreads();
        prefix |= s_digit << shift;
        mask |= 255u << shift;
        k = s_rem;
        __syncthreads();
    }
    return prefix;
}
__device__ unsigned radix_kth(const unsigned* __restrict__ keys, int n, int k, unsigned* hist) {
    unsigned prefix = 0u, mask = 0u;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0u;
        __syncthreads();
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const unsigned kk = keys[i];
            if ((kk & mask) == prefix) atomicAdd(&hist[(kk >> shift) & 255u], 1u);
        }
        __syncthreads();
        __shared__ unsigned s_digit, s_rem;
        if (threadIdx.x == 0) {
            unsigned acc = 0u;
            int d = 255;
            for (; d > 0; d--) {
                if (acc + hist[d] >= (unsigned)k) break;
                acc += hist[d];
            }
            s_digit = d;
            s_rem = k - acc;
        }
        __syncthreads();
        prefix |= s_digit << shift;
        mask |= 255u << shift;
        k = s_rem;
        __syncthreads();
    }
    return prefix;
}
// local top-IDX_TOPB scores of row t -> send buffer (as keys; fewer blocks: all of them, pad 0 = -inf).
// One block per row, dynamic smem lbmax keys; GK (the keys of a card's whole context do not fit shared memory: 2 cards
// own 1/2 of the positions, 262144 -> 32768 blocks = 128 KB): the keys replace the row's scores in place (global
// memory), k_idx_select then reads them as keys (skeys)
template <bool GK = false>
__global__ void k_idx_local(float* __restrict__ scores, const int* __restrict__ pos0, Plan p, int lbmax,
                            unsigned* __restrict__ out, const int* __restrict__ rpos = nullptr) {
    extern __shared__ unsigned keys_sm[];
    __shared__ unsigned hist[256];
    __shared__ int s_cnt;
    const int t = blockIdx.x;
    unsigned* keys = GK ? reinterpret_cast<unsigned*>(scores + (size_t)t * lbmax) : keys_sm;
    const int pos = rpos ? rpos[t] : *pos0 + t, nbg = (pos + 1) / IDX_R;
    unsigned* o = out + (size_t)t * IDX_TOPB;
    if (nbg <= IDX_TOPB) {
        for (int i = threadIdx.x; i < IDX_TOPB; i += blockDim.x) o[i] = 0u;
        return;
    }
    const int nlb = owned_upto(p, IDX_R * nbg - 1) / IDX_R;
    for (int i = threadIdx.x; i < nlb; i += blockDim.x) keys[i] = fkey(scores[(size_t)t * lbmax + i]);
    if (GK) __threadfence_block();
    if (threadIdx.x == 0) s_cnt = 0;
    __syncthreads();
    if (nlb <= IDX_TOPB) {
        for (int i = threadIdx.x; i < IDX_TOPB; i += blockDim.x) o[i] = i < nlb ? (GK ? __ldcg(keys + i) : keys[i]) : 0u;
        return;
    }
    if (GK) {
        const unsigned kth = radix_kth_g(keys, nlb, IDX_TOPB, hist);
        for (int i = threadIdx.x; i < nlb; i += blockDim.x) {
            const unsigned kk = __ldcg(keys + i);
            if (kk > kth) o[atomicAdd(&s_cnt, 1)] = kk;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < nlb; i += blockDim.x)
            if (__ldcg(keys + i) == kth) { const int j = atomicAdd(&s_cnt, 1); if (j < IDX_TOPB) o[j] = kth; }
        return;
    }
    const unsigned kth = radix_kth(keys, nlb, IDX_TOPB, hist);
    // keys > kth first, then == kth until IDX_TOPB
    for (int i = threadIdx.x; i < nlb; i += blockDim.x)
        if (keys[i] > kth) o[atomicAdd(&s_cnt, 1)] = keys[i];
    __syncthreads();
    for (int i = threadIdx.x; i < nlb; i += blockDim.x)
        if (keys[i] == kth) { const int j = atomicAdd(&s_cnt, 1); if (j < IDX_TOPB) o[j] = kth; }
}
// every card sends its local top keys of all rows to both peers ([2 slots][ngpu][MAXM][IDX_TOPB] keys)
__global__ void k_idx_send(const unsigned* __restrict__ loc, int m, Xchg xc, Plan p) {
    const int peer = blockIdx.x >= (unsigned)p.rank ? blockIdx.x + 1 : blockIdx.x;
    const unsigned s = *xc.seq + 1;
    unsigned* dst = reinterpret_cast<unsigned*>(xc.recv[peer]) + ((size_t)(s & (XSLOT - 1)) * p.ngpu + p.rank) * XROWS * IDX_TOPB;
    for (int idx = threadIdx.x; idx < m * IDX_TOPB; idx += blockDim.x) dst[idx] = loc[idx];
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        qe_st_release_sys(xc.flags[peer] + p.rank, s);
    }
}
// global threshold = IDX_TOPB-th largest of the 3 x IDX_TOPB gathered keys; compact this card's selected blocks
// (score key >= threshold) into sel[t][0..nsel[t]). One block per row; the last block bumps the exchange.
__global__ void k_idx_select(const float* __restrict__ scores, const unsigned* __restrict__ loc, const int* __restrict__ pos0,
                             Plan p, int lbmax, Xchg xc, int* __restrict__ sel, int* __restrict__ nsel,
                             const int* __restrict__ rpos = nullptr, bool skeys = false) {
    __shared__ unsigned keys[3 * IDX_TOPB];  // ngpu <= 3
    __shared__ unsigned hist[256];
    __shared__ unsigned s_s;
    __shared__ int s_cnt;
    const int t = blockIdx.x;
    if (threadIdx.x == 0) {
        const unsigned s = *xc.seq + 1;
        for (int g = 0; g < p.ngpu; g++)
            if (g != p.rank)
                for (unsigned ws_ = 0; (int)(qe_ld_acquire_sys(xc.flags[p.rank] + g) - s) < 0;) { __nanosleep(20); QE_WD_POLL(ws_, kWdQsaX, g, s, qe_ld_acquire_sys(xc.flags[p.rank] + g)); }
        s_s = s;
        s_cnt = 0;
    }
    __syncthreads();
    const int pos = rpos ? rpos[t] : *pos0 + t, nbg = (pos + 1) / IDX_R;
    if (nbg > IDX_TOPB) {
        const unsigned* rb = reinterpret_cast<const unsigned*>(xc.recv[p.rank]) + (size_t)(s_s & (XSLOT - 1)) * p.ngpu * XROWS * IDX_TOPB;
        for (int i = threadIdx.x; i < p.ngpu * IDX_TOPB; i += blockDim.x) {
            const int g = i / IDX_TOPB, j = i % IDX_TOPB;
            keys[i] = g == p.rank ? loc[(size_t)t * IDX_TOPB + j] : __ldcg(rb + ((size_t)g * XROWS + t) * IDX_TOPB + j);
        }
        __syncthreads();
        const unsigned thr = radix_kth(keys, p.ngpu * IDX_TOPB, IDX_TOPB, hist);
        const int nlb = owned_upto(p, IDX_R * nbg - 1) / IDX_R;
        // ordered compaction (ascending block index: the same list every run, no atomic order)
        __shared__ int wcnt[32];
        const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
        for (int b0 = 0; b0 < nlb; b0 += blockDim.x) {
            const int i = b0 + threadIdx.x;
            const float sv = i < nlb ? scores[(size_t)t * lbmax + i] : 0.f;
            const bool f = i < nlb && (skeys ? __float_as_uint(sv) : fkey(sv)) >= thr;
            const unsigned bal = __ballot_sync(0xffffffffu, f);
            if (lane == 0) wcnt[wid] = __popc(bal);
            __syncthreads();
            int base = s_cnt;
            for (int w = 0; w < wid; w++) base += wcnt[w];
            if (f) {
                const int j = base + __popc(bal & ((1u << lane) - 1u));
                if (j < lbmax) sel[(size_t)t * lbmax + j] = i;
            }
            __syncthreads();
            if (threadIdx.x == 0) for (int w = 0; w < nw; w++) s_cnt += wcnt[w];
            __syncthreads();
        }
        if (threadIdx.x == 0) nsel[t] = min(s_cnt, lbmax);
    } else if (threadIdx.x == 0) {
        nsel[t] = -1;  // no selection: every visible key
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(xc.done, 1u) == gridDim.x - 1) {
            *xc.done = 0;
            *xc.seq = s_s;
            __threadfence();
        }
    }
}

// QSA variant B: q is computed for the local heads only; every card sends its 8 heads' q rows ([MAXM][8][HD]) to
// both peers (receive buffer [2 slots][ngpu][MAXM][8][HD]); the attention waits for the peer's flag per block.
__global__ void k_qxsend(const float* __restrict__ qb, int m, Xchg xc, Plan p) {
    const int peer = blockIdx.x >= (unsigned)p.rank ? blockIdx.x + 1 : blockIdx.x;
    const unsigned s = *xc.seq + 1;
    float* dst = xc.recv[peer] + ((size_t)(s & (XSLOT - 1)) * p.ngpu + p.rank) * XROWS * qper(p);
    for (int idx = blockIdx.y * blockDim.x + threadIdx.x; idx < m * qper(p) / 4; idx += gridDim.y * blockDim.x) {
        const int t = idx / (qper(p) / 4), r = idx % (qper(p) / 4);
        reinterpret_cast<float4*>(dst)[idx] = reinterpret_cast<const float4*>(qb + ((size_t)t * (NQ + 4) + p.q0) * HD)[r];
    }
    xsend_done(xc, peer, s, p);
}
// fp16 exchange formats (prefill chunks, tensor-core attention only): q heads as fp16 * 1/16 ([t][8][HD] halves in the
// slot), partials as the normalized output in fp16 + (max, sum) fp32: a record of XP16 floats per (row, head)
constexpr int XP16 = HD / 2 + 4;  // 128 floats of halves, m, l, pad (16-byte aligned records)
__global__ void k_qxsend16(const float* __restrict__ qb, int m, Xchg xc, Plan p) {
    const int peer = blockIdx.x >= (unsigned)p.rank ? blockIdx.x + 1 : blockIdx.x;
    const unsigned s = *xc.seq + 1;
    __half* dst = reinterpret_cast<__half*>(xc.recv[peer] + ((size_t)(s & (XSLOT - 1)) * p.ngpu + p.rank) * XROWS * qper(p));
    for (int idx = blockIdx.y * blockDim.x + threadIdx.x; idx < m * qper(p) / 8; idx += gridDim.y * blockDim.x) {
        const int t = idx / (qper(p) / 8), r = idx % (qper(p) / 8);
        const float4* src = reinterpret_cast<const float4*>(qb + ((size_t)t * (NQ + 4) + p.q0) * HD) + 2 * r;
        const float4 a = src[0], b = src[1];
        constexpr float k = 1.f / 16.f;
        reinterpret_cast<uint4*>(dst)[idx] = make_uint4(f2h2(a.x * k, a.y * k), f2h2(a.z * k, a.w * k), f2h2(b.x * k, b.y * k), f2h2(b.z * k, b.w * k));
    }
    xsend_done(xc, peer, s, p);
}
// ahead: exchanges of this kind sent but not yet consumed here (pipelined QSA chunks, prefill-r3)
__global__ void k_xsend16(const float* __restrict__ part, int m, Xchg xc, Plan p, int ahead = 0) {
    const int peer = blockIdx.x >= (unsigned)p.rank ? blockIdx.x + 1 : blockIdx.x;
    const unsigned s = *xc.seq + 1 + ahead;
    float* dst = xc.recv[peer] + ((size_t)(s & (XSLOT - 1)) * p.ngpu + p.rank) * XROWS * xper(p);
    const int qpc = QPC(p), h0 = peer * qpc;
    for (int idx = blockIdx.y * blockDim.x + threadIdx.x; idx < m * qpc * 32; idx += gridDim.y * blockDim.x) {
        const int rec = idx >> 5, l8 = idx & 31, t = rec / qpc, hl = rec - t * qpc;
        const float* src = part + ((size_t)t * NQ + h0 + hl) * (HD + 2);
        const float L = src[HD + 1], inv = L > 0.f ? 1.f / L : 0.f;
        const float2* s2 = reinterpret_cast<const float2*>(src) + 4 * l8;  // records of HD + 2 floats: 8-byte aligned
        const float2 a0 = s2[0], a1 = s2[1], b0 = s2[2], b1 = s2[3];
        const float4 a = make_float4(a0.x, a0.y, a1.x, a1.y), b = make_float4(b0.x, b0.y, b1.x, b1.y);
        float* rd = dst + (size_t)rec * XP16;
        reinterpret_cast<uint4*>(rd)[l8] = make_uint4(f2h2(a.x * inv, a.y * inv), f2h2(a.z * inv, a.w * inv), f2h2(b.x * inv, b.y * inv), f2h2(b.z * inv, b.w * inv));
        if (l8 == 0) *reinterpret_cast<float2*>(rd + HD / 2) = make_float2(src[HD], L);
    }
    xsend_done(xc, peer, s, p);
}
// as k_qsa_comb2 with the peers' partials in the fp16 format
__global__ void k_qsa_comb16(const float* __restrict__ part, int m, const float* __restrict__ proj, int ld,
                             bf16* __restrict__ out, int ldo, Xchg xc, Plan p, int gate_off) {
    __shared__ unsigned s_s;
    if (threadIdx.x == 0) {
        const unsigned s = *xc.seq + 1;
        for (int g = 0; g < p.ngpu; g++)
            if (g != p.rank)
                for (unsigned ws_ = 0; (int)(qe_ld_acquire_sys(xc.flags[p.rank] + g) - s) < 0;) { __nanosleep(20); QE_WD_POLL(ws_, kWdQsaX, g, s, qe_ld_acquire_sys(xc.flags[p.rank] + g)); }
        s_s = s;
    }
    __syncthreads();
    const unsigned s = s_s;
    const int hl = blockIdx.x, t = blockIdx.y, d = threadIdx.x, h = p.q0 + hl;
    const float* rb = xc.recv[p.rank] + (size_t)(s & (XSLOT - 1)) * p.ngpu * XROWS * xper(p);
    const float* own = part + ((size_t)t * NQ + h) * (HD + 2);
    float M = __ldcg(own + HD);
    for (int g = 0; g < p.ngpu; g++)
        if (g != p.rank) M = fmaxf(M, __ldcg(rb + ((size_t)g * XROWS * xper(p)) + ((size_t)t * QPC(p) + hl) * XP16 + HD / 2));
    float a = 0.f, L = 0.f;
    {
        const float mg = __ldcg(own + HD);
        if (mg != -INFINITY) {
            const float c = __expf(mg - M);
            a += c * __ldcg(own + d);
            L += c * __ldcg(own + HD + 1);
        }
    }
    for (int g = 0; g < p.ngpu; g++) {
        if (g == p.rank) continue;
        const float* rec = rb + ((size_t)g * XROWS * xper(p)) + ((size_t)t * QPC(p) + hl) * XP16;
        const float mg = __ldcg(rec + HD / 2);
        if (mg == -INFINITY) continue;
        const float w = __expf(mg - M) * __ldcg(rec + HD / 2 + 1);
        a += w * __half2float(reinterpret_cast<const __half*>(rec)[d]);
        L += w;
    }
    const float gate = proj[(size_t)t * ld + gate_off + hl * HD + d];
    out[(size_t)t * ldo + hl * HD + d] = __float2bfloat16(a / L * sigm(gate));
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(xc.done, 1u) == gridDim.x * gridDim.y - 1) {
            *xc.done = 0;
            *xc.seq = s;
            __threadfence();
        }
    }
}
// as k_qsa_attn, q of a peer's head from the receive buffer (after its flag); the last block bumps the q exchange
__global__ void k_qsa_attn_b(const float* __restrict__ qb, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
                             const float* __restrict__ ks, const float* __restrict__ vs, const int* __restrict__ pos0,
                             float* __restrict__ out, Plan p, Xchg xq, const int* __restrict__ sel,
                             const int* __restrict__ nsel, int lbmax, const int* __restrict__ kvb0 = nullptr) {
    __shared__ float qs[HD];
    __shared__ float wm[8], wl[8];
    __shared__ float wacc[8][HD];
    __shared__ unsigned s_s;
    if (kvb0) { const size_t rb = *kvb0; kc += rb * NKV * HD; vc += rb * NKV * HD; ks += rb * NKV; vs += rb * NKV; }  // slot region
    const int h = blockIdx.x, t = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int owner = h / QPC(p);
    if (threadIdx.x == 0) {
        const unsigned s = *xq.seq + 1;
        if (owner != p.rank)
            for (unsigned ws_ = 0; (int)(qe_ld_acquire_sys(xq.flags[p.rank] + owner) - s) < 0;) { __nanosleep(20); QE_WD_POLL(ws_, kWdQsaX, owner, s, qe_ld_acquire_sys(xq.flags[p.rank] + owner)); }
        s_s = s;
    }
    __syncthreads();
    const float* qsrc = owner == p.rank ? qb + ((size_t)t * (NQ + 4) + h) * HD
                                        : xq.recv[p.rank] + (((size_t)(s_s & (XSLOT - 1)) * p.ngpu + owner) * XROWS + t) * qper(p) + (h - owner * QPC(p)) * HD;
    qs[threadIdx.x] = __ldcg(qsrc + threadIdx.x) * (1.f / 16.f);
    __syncthreads();
    const int kvh = h / (NQ / NKV);
    const int pos = *pos0 + t;
    float q8[8];
#pragma unroll
    for (int e = 0; e < 8; e++) q8[e] = qs[lane * 8 + e];
    float mx = -INFINITY, l = 0.f, acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    // keys: all owned rows <= pos (no selection), else the rows of the selected blocks + the owned tail rows
    const int ns = sel ? nsel[t] : -1;
    const int nrows_all = owned_upto(p, pos);
    const int rc = ns < 0 ? nrows_all : owned_upto(p, IDX_R * ((pos + 1) / IDX_R) - 1);  // rows of complete blocks
    const int nit = ns < 0 ? nrows_all : ns * IDX_R + (nrows_all - rc);
    for (int it = warp; it < nit; it += 8) {
        const int r = ns < 0 ? it : (it < ns * IDX_R ? sel[(size_t)t * lbmax + it / IDX_R] * IDX_R + it % IDX_R : rc + it - ns * IDX_R);
        const uint2 kw = *reinterpret_cast<const uint2*>(kc + ((size_t)r * NKV + kvh) * HD + lane * 8);
        const uint8_t* kb = reinterpret_cast<const uint8_t*>(&kw);
        float sc = 0.f;
#pragma unroll
        for (int e = 0; e < 8; e++) sc += q8[e] * float(*reinterpret_cast<const __nv_fp8_e4m3*>(kb + e));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sc += __shfl_xor_sync(0xffffffffu, sc, o);
        sc *= ks[r * NKV + kvh];
        const float nm = fmaxf(mx, sc);
        const float c = __expf(mx - nm), e1 = __expf(sc - nm);
        l = l * c + e1;
        const uint2 vw = *reinterpret_cast<const uint2*>(vc + ((size_t)r * NKV + kvh) * HD + lane * 8);
        const uint8_t* vb = reinterpret_cast<const uint8_t*>(&vw);
        const float vsc = vs[r * NKV + kvh] * e1;
#pragma unroll
        for (int e = 0; e < 8; e++) acc[e] = acc[e] * c + vsc * float(*reinterpret_cast<const __nv_fp8_e4m3*>(vb + e));
        mx = nm;
    }
    if (lane == 0) { wm[warp] = mx; wl[warp] = l; }
#pragma unroll
    for (int e = 0; e < 8; e++) wacc[warp][lane * 8 + e] = acc[e];
    __syncthreads();
    float M = -INFINITY;
    for (int w = 0; w < 8; w++) M = fmaxf(M, wm[w]);
    float* o = out + ((size_t)t * NQ + h) * (HD + 2);
    float a = 0.f, L = 0.f;
    for (int w = 0; w < 8; w++) {
        const float c = wm[w] == -INFINITY ? 0.f : __expf(wm[w] - M);
        a += c * wacc[w][threadIdx.x];
        L += c * wl[w];
    }
    o[threadIdx.x] = a;
    if (threadIdx.x == 0) { o[HD] = M; o[HD + 1] = L; }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(xq.done, 1u) == gridDim.x * gridDim.y - 1) {
            *xq.done = 0;
            *xq.seq = s_s;
            __threadfence();
        }
    }
}
// Prefill attention: block = (q head, 8 query rows), one warp per row walks all its keys (K/V of the block's rows are
// shared through L1); 2 keys per iteration. Same outputs as k_qsa_attn_b (unnormalized sum, max, sum of exp).
__global__ void __launch_bounds__(256) k_qsa_attn_rows(const float* __restrict__ qb, const uint8_t* __restrict__ kc,
                                                       const uint8_t* __restrict__ vc, const float* __restrict__ ks,
                                                       const float* __restrict__ vs, const int* __restrict__ pos0,
                                                       float* __restrict__ out, Plan p, Xchg xq, const int* __restrict__ sel,
                                                       const int* __restrict__ nsel, int lbmax, int m,
                                                       const int* __restrict__ kvb0 = nullptr) {
    __shared__ unsigned s_s;
    if (kvb0) { const size_t rb = *kvb0; kc += rb * NKV * HD; vc += rb * NKV * HD; ks += rb * NKV; vs += rb * NKV; }  // slot region
    const int h = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int t = blockIdx.y * 8 + warp;
    const int owner = h / QPC(p);
    if (threadIdx.x == 0) {
        const unsigned s = *xq.seq + 1;
        if (owner != p.rank)
            for (unsigned ws_ = 0; (int)(qe_ld_acquire_sys(xq.flags[p.rank] + owner) - s) < 0;) { __nanosleep(20); QE_WD_POLL(ws_, kWdQsaX, owner, s, qe_ld_acquire_sys(xq.flags[p.rank] + owner)); }
        s_s = s;
    }
    __syncthreads();
    if (t < m) {
        const float* qsrc = owner == p.rank ? qb + ((size_t)t * (NQ + 4) + h) * HD
                                            : xq.recv[p.rank] + (((size_t)(s_s & (XSLOT - 1)) * p.ngpu + owner) * XROWS + t) * qper(p) + (h - owner * QPC(p)) * HD;
        float q8[8];
#pragma unroll
        for (int e = 0; e < 8; e++) q8[e] = __ldcg(qsrc + lane * 8 + e) * (1.f / 16.f);
        const int kvh = h / (NQ / NKV);
        const int pos = *pos0 + t;
        const int ns = sel ? nsel[t] : -1;
        const int nrows_all = owned_upto(p, pos);
        const int rc = ns < 0 ? nrows_all : owned_upto(p, IDX_R * ((pos + 1) / IDX_R) - 1);
        const int nit = ns < 0 ? nrows_all : ns * IDX_R + (nrows_all - rc);
        float mx = -INFINITY, l = 0.f, acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int it = 0; it < nit; it++) {
            const int r = ns < 0 ? it : (it < ns * IDX_R ? sel[(size_t)t * lbmax + it / IDX_R] * IDX_R + it % IDX_R : rc + it - ns * IDX_R);
            const uint2 kw = *reinterpret_cast<const uint2*>(kc + ((size_t)r * NKV + kvh) * HD + lane * 8);
            const uint2 vw = *reinterpret_cast<const uint2*>(vc + ((size_t)r * NKV + kvh) * HD + lane * 8);
            const float ksc = ks[r * NKV + kvh], vsc0 = vs[r * NKV + kvh];
            const uint8_t* kb = reinterpret_cast<const uint8_t*>(&kw);
            float sc = 0.f;
#pragma unroll
            for (int e = 0; e < 8; e++) sc += q8[e] * float(*reinterpret_cast<const __nv_fp8_e4m3*>(kb + e));
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) sc += __shfl_xor_sync(0xffffffffu, sc, o);
            sc *= ksc;
            const float nm = fmaxf(mx, sc);
            const float c = __expf(mx - nm), e1 = __expf(sc - nm);
            l = l * c + e1;
            const uint8_t* vb = reinterpret_cast<const uint8_t*>(&vw);
            const float vsc = vsc0 * e1;
#pragma unroll
            for (int e = 0; e < 8; e++) acc[e] = acc[e] * c + vsc * float(*reinterpret_cast<const __nv_fp8_e4m3*>(vb + e));
            mx = nm;
        }
        float* o = out + ((size_t)t * NQ + h) * (HD + 2);
#pragma unroll
        for (int e = 0; e < 8; e++) o[lane * 8 + e] = acc[e];
        if (lane == 0) { o[HD] = mx; o[HD + 1] = l; }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(xq.done, 1u) == gridDim.x * gridDim.y - 1) {
            *xq.done = 0;
            *xq.seq = s_s;
            __threadfence();
        }
    }
}
// Prefill attention on tensor cores (no block selection: all owned rows <= pos). Block = (q head, 64 query rows),
// 4 warps x 16 rows; key tiles of 32 owned rows. K stays raw fp8 widened exactly to fp16 (its per-row scale applied to
// the scores), V widened with its per-row scale; fp16 mma, fp32 online softmax. Same outputs as k_qsa_attn_rows.
constexpr int TA_ROWS = 64, TA_KEYS = 32, TA_SROW = HD + 8;
constexpr size_t TA_SMEM = (size_t)(TA_ROWS + 2 * TA_KEYS) * TA_SROW * 2 + TA_KEYS * 4;
__device__ __forceinline__ void ta_ldsm4(uint32_t r[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((unsigned)__cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void ta_ldsm4t(uint32_t r[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((unsigned)__cvta_generic_to_shared(p)));
}
__device__ __forceinline__ uint32_t ta_f8x2(unsigned short v) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)v, __NV_E4M3);
    return (uint32_t)h.x | ((uint32_t)h.y << 16);
}
__global__ void __launch_bounds__(128) k_qsa_attn_tc(const float* __restrict__ qb, const uint8_t* __restrict__ kc,
                                                     const uint8_t* __restrict__ vc, const float* __restrict__ ks,
                                                     const float* __restrict__ vs, const int* __restrict__ pos0,
                                                     float* __restrict__ out, Plan p, Xchg xq, int m,
                                                     const int* __restrict__ kvb0 = nullptr) {
    extern __shared__ __align__(16) unsigned char ta_sm[];
    if (kvb0) { const size_t rb = *kvb0; kc += rb * NKV * HD; vc += rb * NKV * HD; ks += rb * NKV; vs += rb * NKV; }  // slot region
    __half* Qs = reinterpret_cast<__half*>(ta_sm);
    __half* Ks = Qs + TA_ROWS * TA_SROW;
    __half* Vs = Ks + TA_KEYS * TA_SROW;
    float* kss = reinterpret_cast<float*>(Vs + TA_KEYS * TA_SROW);
    __shared__ unsigned s_s;
    const int h = blockIdx.x, tb = blockIdx.y * TA_ROWS, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int owner = h / QPC(p), kvh = h / (NQ / NKV);
    if (threadIdx.x == 0) {
        const unsigned s = *xq.seq + 1;
        if (owner != p.rank)
            for (unsigned ws_ = 0; (int)(qe_ld_acquire_sys(xq.flags[p.rank] + owner) - s) < 0;) { __nanosleep(20); QE_WD_POLL(ws_, kWdQsaX, owner, s, qe_ld_acquire_sys(xq.flags[p.rank] + owner)); }
        s_s = s;
    }
    __syncthreads();
    const int pos = *pos0;
    for (int idx = threadIdx.x; idx < TA_ROWS * (HD / 4); idx += blockDim.x) {
        const int r = idx / (HD / 4), c4 = idx % (HD / 4), t = tb + r;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (t < m) {
            const float* qsrc = owner == p.rank ? qb + ((size_t)t * (NQ + 4) + h) * HD
                                                : xq.recv[p.rank] + (((size_t)(s_s & (XSLOT - 1)) * p.ngpu + owner) * XROWS + t) * qper(p) + (h - owner * QPC(p)) * HD;
            v = __ldcg(reinterpret_cast<const float4*>(qsrc) + c4);
        }
        __half2* d = reinterpret_cast<__half2*>(Qs + r * TA_SROW + c4 * 4);
        d[0] = __floats2half2_rn(v.x * (1.f / 16.f), v.y * (1.f / 16.f));
        d[1] = __floats2half2_rn(v.z * (1.f / 16.f), v.w * (1.f / 16.f));
    }
    const int kend = owned_upto(p, pos + min(tb + TA_ROWS, m) - 1);
    const int r0 = warp * 16 + (lane >> 2), r1 = r0 + 8;
    const int nr0 = tb + r0 < m ? owned_upto(p, pos + tb + r0) : 0;
    const int nr1 = tb + r1 < m ? owned_upto(p, pos + tb + r1) : 0;
    const int wmax = __reduce_max_sync(0xffffffffu, max(nr0, nr1));
    float o[HD / 8][4];
#pragma unroll
    for (int j = 0; j < HD / 8; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    for (int k0 = 0; k0 < kend; k0 += TA_KEYS) {
        __syncthreads();
        for (int c = threadIdx.x; c < TA_KEYS * (HD / 16); c += blockDim.x) {
            const int key = c / (HD / 16), ch = c % (HD / 16), r = k0 + key;
            uint4 kw = make_uint4(0, 0, 0, 0), vw = make_uint4(0, 0, 0, 0);
            float vsc = 0.f;
            if (r < kend) {
                kw = *reinterpret_cast<const uint4*>(kc + ((size_t)r * NKV + kvh) * HD + ch * 16);
                vw = *reinterpret_cast<const uint4*>(vc + ((size_t)r * NKV + kvh) * HD + ch * 16);
                vsc = vs[r * NKV + kvh];
            }
            const unsigned short* kb = reinterpret_cast<const unsigned short*>(&kw);
            const unsigned short* vb = reinterpret_cast<const unsigned short*>(&vw);
            uint32_t kh[8], vh[8];
#pragma unroll
            for (int e = 0; e < 8; e++) {
                kh[e] = ta_f8x2(kb[e]);
                const uint32_t v2 = ta_f8x2(vb[e]);
                const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&v2));
                const __half2 hv = __floats2half2_rn(f.x * vsc, f.y * vsc);
                vh[e] = *reinterpret_cast<const uint32_t*>(&hv);
            }
            uint4* kd = reinterpret_cast<uint4*>(Ks + key * TA_SROW + ch * 16);
            uint4* vd = reinterpret_cast<uint4*>(Vs + key * TA_SROW + ch * 16);
            kd[0] = make_uint4(kh[0], kh[1], kh[2], kh[3]);
            kd[1] = make_uint4(kh[4], kh[5], kh[6], kh[7]);
            vd[0] = make_uint4(vh[0], vh[1], vh[2], vh[3]);
            vd[1] = make_uint4(vh[4], vh[5], vh[6], vh[7]);
        }
        if (threadIdx.x < TA_KEYS) kss[threadIdx.x] = k0 + threadIdx.x < kend ? ks[(k0 + threadIdx.x) * NKV + kvh] : 0.f;
        __syncthreads();
        if (wmax <= k0) continue;  // all rows of this warp end before this tile
        float s[4][4];
#pragma unroll
        for (int j = 0; j < 4; j++) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < HD / 16; kk++) {
            uint32_t a[4];
            ta_ldsm4(a, Qs + (warp * 16 + (lane & 15)) * TA_SROW + kk * 16 + (lane >> 4) * 8);
#pragma unroll
            for (int np = 0; np < 2; np++) {
                uint32_t b[4];
                ta_ldsm4(b, Ks + (np * 16 + (lane & 7) + ((lane >> 4) << 3)) * TA_SROW + kk * 16 + ((lane >> 3) & 1) * 8);
                mma16(s[2 * np], a, b[0], b[1]);
                mma16(s[2 * np + 1], a, b[2], b[3]);
            }
        }
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int key = j * 8 + (lane & 3) * 2 + e, kg = k0 + key;
                const float sc = kss[key];
                s[j][e] = kg < nr0 ? s[j][e] * sc : -INFINITY;
                s[j][2 + e] = kg < nr1 ? s[j][2 + e] * sc : -INFINITY;
                mx0 = fmaxf(mx0, s[j][e]);
                mx1 = fmaxf(mx1, s[j][2 + e]);
            }
#pragma unroll
        for (int x = 1; x <= 2; x <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, x));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, x));
        }
        const float nm0 = fmaxf(m0, mx0), nm1 = fmaxf(m1, mx1);
        const float c0 = nm0 == -INFINITY ? 1.f : __expf(m0 - nm0), c1 = nm1 == -INFINITY ? 1.f : __expf(m1 - nm1);
        float ls0 = 0.f, ls1 = 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                s[j][e] = nm0 == -INFINITY ? 0.f : __expf(s[j][e] - nm0);
                s[j][2 + e] = nm1 == -INFINITY ? 0.f : __expf(s[j][2 + e] - nm1);
                ls0 += s[j][e];
                ls1 += s[j][2 + e];
            }
        l0 = l0 * c0 + ls0;
        l1 = l1 * c1 + ls1;
        m0 = nm0;
        m1 = nm1;
#pragma unroll
        for (int j = 0; j < HD / 8; j++) { o[j][0] *= c0; o[j][1] *= c0; o[j][2] *= c1; o[j][3] *= c1; }
#pragma unroll
        for (int s2 = 0; s2 < 2; s2++) {
            uint32_t a[4];
            a[0] = f2h2(s[2 * s2][0], s[2 * s2][1]);
            a[1] = f2h2(s[2 * s2][2], s[2 * s2][3]);
            a[2] = f2h2(s[2 * s2 + 1][0], s[2 * s2 + 1][1]);
            a[3] = f2h2(s[2 * s2 + 1][2], s[2 * s2 + 1][3]);
#pragma unroll
            for (int dp = 0; dp < HD / 16; dp++) {
                uint32_t b[4];
                ta_ldsm4t(b, Vs + (s2 * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * TA_SROW + dp * 16 + (lane >> 4) * 8);
                mma16(o[2 * dp], a, b[0], b[1]);
                mma16(o[2 * dp + 1], a, b[2], b[3]);
            }
        }
    }
#pragma unroll
    for (int x = 1; x <= 2; x <<= 1) {
        l0 += __shfl_xor_sync(0xffffffffu, l0, x);
        l1 += __shfl_xor_sync(0xffffffffu, l1, x);
    }
    if (tb + r0 < m) {
        float* ob = out + ((size_t)(tb + r0) * NQ + h) * (HD + 2);
#pragma unroll
        for (int j = 0; j < HD / 8; j++) *reinterpret_cast<float2*>(ob + j * 8 + (lane & 3) * 2) = make_float2(o[j][0], o[j][1]);
        if ((lane & 3) == 0) { ob[HD] = m0; ob[HD + 1] = l0; }
    }
    if (tb + r1 < m) {
        float* ob = out + ((size_t)(tb + r1) * NQ + h) * (HD + 2);
#pragma unroll
        for (int j = 0; j < HD / 8; j++) *reinterpret_cast<float2*>(ob + j * 8 + (lane & 3) * 2) = make_float2(o[j][2], o[j][3]);
        if ((lane & 3) == 0) { ob[HD] = m1; ob[HD + 1] = l1; }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(xq.done, 1u) == gridDim.x * gridDim.y - 1) {
            *xq.done = 0;
            *xq.seq = s_s;
            __threadfence();
        }
    }
}
// v2: block = 4 warps = 2 q heads of one kv head x 2 groups of 16 rows (32 rows), Q fragments in registers, K / V tiles
// of 32 owned rows widened once for both heads; the keys are split into gridDim.z ranges (more blocks for short
// prefill chunks), split z writes its partial to dst.o[z]; k_attn_merge combines the splits (z > 1).
struct AttnDst { float* o[8]; };
// 4 e4m3 bytes -> 4 fp16 values / 256 (exact, subnormals included): the e4m3 bits shifted into an fp16 word are the value
// times 2^-8 (exponent bias 7 vs 15); the 2^8 goes into the scales
__device__ __forceinline__ void f8x4_h(uint32_t w, uint32_t& lo, uint32_t& hi) {
    const uint32_t a = __byte_perm(w, 0, 0x4140), b = __byte_perm(w, 0, 0x4342);
    lo = ((a & 0x007F007Fu) << 7) | ((a & 0x00800080u) << 8);
    hi = ((b & 0x007F007Fu) << 7) | ((b & 0x00800080u) << 8);
}
constexpr int TB_KEYS = 32;
constexpr size_t TB_SMEM = (size_t)2 * TB_KEYS * TA_SROW * 2 + TB_KEYS * 4 + 2 * TB_KEYS * HD + 3 * TB_KEYS * 4;
template <int RG, bool Q16 = false>  // 16-row groups per block (block = 2 heads x RG warps); Q16: peers' q in fp16
__global__ void __launch_bounds__(64 * RG) k_qsa_attn_tc2(const float* __restrict__ qb, const uint8_t* __restrict__ kc,
                                                      const uint8_t* __restrict__ vc, const float* __restrict__ ks,
                                                      const float* __restrict__ vs, const int* __restrict__ pos0,
                                                      AttnDst dst, Plan p, Xchg xq, int m,
                                                      const int* __restrict__ kvb0 = nullptr) {
    extern __shared__ __align__(16) unsigned char ta_sm[];
    if (kvb0) { const size_t rb = *kvb0; kc += rb * NKV * HD; vc += rb * NKV * HD; ks += rb * NKV; vs += rb * NKV; }  // slot region
    __half* Ks = reinterpret_cast<__half*>(ta_sm);
    __half* Vs = Ks + TB_KEYS * TA_SROW;
    float* kss = reinterpret_cast<float*>(Vs + TB_KEYS * TA_SROW);
    __shared__ unsigned s_s;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = blockIdx.x * 2 + warp / RG, tb = blockIdx.y * 16 * RG, rg = warp % RG;
    const int kvh = (blockIdx.x * 2) / (NQ / NKV);
    if (threadIdx.x == 0) {
        const unsigned s = *xq.seq + 1;
        for (int hh = blockIdx.x * 2; hh < blockIdx.x * 2 + 2; hh++) {
            const int ow = hh / QPC(p);
            if (ow != p.rank)
                for (unsigned ws_ = 0; (int)(qe_ld_acquire_sys(xq.flags[p.rank] + ow) - s) < 0;) { __nanosleep(20); QE_WD_POLL(ws_, kWdQsaX, ow, s, qe_ld_acquire_sys(xq.flags[p.rank] + ow)); }
        }
        s_s = s;
    }
    __syncthreads();
    const int owner = h / QPC(p);
    const int pos = *pos0;
    const int g = lane >> 2, t4 = lane & 3;
    const int r0 = rg * 16 + g, r1 = r0 + 8;  // rows of this lane within the block
    // Q fragments (fp16, * 1/16) of rows r0 / r1
    uint32_t qa[HD / 16][4];
    {
        auto qrow = [&](int t) -> const float* {
            if (tb + t >= m) return nullptr;
            const int tt = tb + t;
            return owner == p.rank ? qb + ((size_t)tt * (NQ + 4) + h) * HD
                                   : xq.recv[p.rank] + (((size_t)(s_s & (XSLOT - 1)) * p.ngpu + owner) * XROWS + tt) * qper(p) + (h - owner * QPC(p)) * HD;
        };
        if (Q16 && owner != p.rank) {
            auto qrow16 = [&](int t) -> const __half* {
                if (tb + t >= m) return nullptr;
                return reinterpret_cast<const __half*>(xq.recv[p.rank] + ((size_t)(s_s & (XSLOT - 1)) * p.ngpu + owner) * XROWS * qper(p)) +
                       (size_t)(tb + t) * qper(p) + (h - owner * QPC(p)) * HD;
            };
            const __half* h0 = qrow16(r0);
            const __half* h1 = qrow16(r1);
#pragma unroll
            for (int kk = 0; kk < HD / 16; kk++) {
                const int c = kk * 16 + 2 * t4;
                qa[kk][0] = h0 ? __ldcg(reinterpret_cast<const unsigned*>(h0 + c)) : 0u;
                qa[kk][1] = h1 ? __ldcg(reinterpret_cast<const unsigned*>(h1 + c)) : 0u;
                qa[kk][2] = h0 ? __ldcg(reinterpret_cast<const unsigned*>(h0 + c + 8)) : 0u;
                qa[kk][3] = h1 ? __ldcg(reinterpret_cast<const unsigned*>(h1 + c + 8)) : 0u;
            }
        } else {
        const float* q0 = qrow(r0);
        const float* q1 = qrow(r1);
#pragma unroll
        for (int kk = 0; kk < HD / 16; kk++) {
            const int c = kk * 16 + 2 * t4;
            float2 a = q0 ? __ldcg(reinterpret_cast<const float2*>(q0 + c)) : make_float2(0.f, 0.f);
            float2 b = q1 ? __ldcg(reinterpret_cast<const float2*>(q1 + c)) : make_float2(0.f, 0.f);
            float2 a8 = q0 ? __ldcg(reinterpret_cast<const float2*>(q0 + c + 8)) : make_float2(0.f, 0.f);
            float2 b8 = q1 ? __ldcg(reinterpret_cast<const float2*>(q1 + c + 8)) : make_float2(0.f, 0.f);
            qa[kk][0] = f2h2(a.x * (1.f / 16.f), a.y * (1.f / 16.f));
            qa[kk][1] = f2h2(b.x * (1.f / 16.f), b.y * (1.f / 16.f));
            qa[kk][2] = f2h2(a8.x * (1.f / 16.f), a8.y * (1.f / 16.f));
            qa[kk][3] = f2h2(b8.x * (1.f / 16.f), b8.y * (1.f / 16.f));
        }
        }
    }
    const int kend = owned_upto(p, pos + min(tb + 16 * RG, m) - 1);
    const int nr0 = tb + r0 < m ? owned_upto(p, pos + tb + r0) : 0;
    const int nr1 = tb + r1 < m ? owned_upto(p, pos + tb + r1) : 0;
    const int wmax = __reduce_max_sync(0xffffffffu, max(nr0, nr1));
    const int ntile = (kend + TB_KEYS - 1) / TB_KEYS, S = gridDim.z, z = blockIdx.z;
    const int tile0 = z * ntile / S, tile1 = (z + 1) * ntile / S;
    float o[HD / 8][4];
#pragma unroll
    for (int j = 0; j < HD / 8; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    // raw fp8 tiles + scales staged by cp.async one tile ahead
    uint8_t* Kr = reinterpret_cast<uint8_t*>(kss + TB_KEYS);
    uint8_t* Vr = Kr + TB_KEYS * HD;
    float* ksr = reinterpret_cast<float*>(Vr + TB_KEYS * HD);
    float* vsr = ksr + TB_KEYS;
    float* vss = vsr + TB_KEYS;  // V scales * 256 of the current tile (applied to P)
    auto issue = [&](int tile) {
        const int k0 = tile * TB_KEYS;
        for (int c = threadIdx.x; c < TB_KEYS * (HD / 16); c += blockDim.x) {
            const int key = c / (HD / 16), ch = c % (HD / 16), r = k0 + key;
            const bool ok = r < kend;
            const size_t go = ((size_t)(ok ? r : 0) * NKV + kvh) * HD + ch * 16;
            const unsigned sk = (unsigned)__cvta_generic_to_shared(Kr + key * HD + ch * 16);
            const unsigned sv = (unsigned)__cvta_generic_to_shared(Vr + key * HD + ch * 16);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sk), "l"(kc + go), "r"(ok ? 16 : 0));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sv), "l"(vc + go), "r"(ok ? 16 : 0));
        }
        if (threadIdx.x < TB_KEYS) {
            const int r = k0 + threadIdx.x;
            const bool ok = r < kend;
            const unsigned a = (unsigned)__cvta_generic_to_shared(ksr + threadIdx.x);
            const unsigned b = (unsigned)__cvta_generic_to_shared(vsr + threadIdx.x);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(a), "l"(ks + (ok ? r : 0) * NKV + kvh), "r"(ok ? 4 : 0));
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(b), "l"(vs + (ok ? r : 0) * NKV + kvh), "r"(ok ? 4 : 0));
        }
        asm volatile("cp.async.commit_group;");
    };
    if (tile0 < tile1) issue(tile0);
    for (int tile = tile0; tile < tile1; tile++) {
        const int k0 = tile * TB_KEYS;
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();  // staging complete; every warp is done with the previous fp16 tile
        for (int c = threadIdx.x; c < TB_KEYS * (HD / 16); c += blockDim.x) {
            const int key = c / (HD / 16), ch = c % (HD / 16);
            const uint4 kw = *reinterpret_cast<const uint4*>(Kr + key * HD + ch * 16);
            const uint4 vw = *reinterpret_cast<const uint4*>(Vr + key * HD + ch * 16);
            uint32_t kh[8], vh[8];
            f8x4_h(kw.x, kh[0], kh[1]); f8x4_h(kw.y, kh[2], kh[3]); f8x4_h(kw.z, kh[4], kh[5]); f8x4_h(kw.w, kh[6], kh[7]);
            f8x4_h(vw.x, vh[0], vh[1]); f8x4_h(vw.y, vh[2], vh[3]); f8x4_h(vw.z, vh[4], vh[5]); f8x4_h(vw.w, vh[6], vh[7]);
            uint4* kd = reinterpret_cast<uint4*>(Ks + key * TA_SROW + ch * 16);
            uint4* vd = reinterpret_cast<uint4*>(Vs + key * TA_SROW + ch * 16);
            kd[0] = make_uint4(kh[0], kh[1], kh[2], kh[3]);
            kd[1] = make_uint4(kh[4], kh[5], kh[6], kh[7]);
            vd[0] = make_uint4(vh[0], vh[1], vh[2], vh[3]);
            vd[1] = make_uint4(vh[4], vh[5], vh[6], vh[7]);
        }
        if (threadIdx.x < TB_KEYS) { kss[threadIdx.x] = ksr[threadIdx.x] * 256.f; vss[threadIdx.x] = vsr[threadIdx.x] * 256.f; }
        __syncthreads();  // fp16 tile ready, staging free
        if (tile + 1 < tile1) issue(tile + 1);
        if (wmax <= k0) continue;
        float s[4][4];
#pragma unroll
        for (int j = 0; j < 4; j++) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < HD / 16; kk++) {
#pragma unroll
            for (int np = 0; np < 2; np++) {
                uint32_t b[4];
                ta_ldsm4(b, Ks + (np * 16 + (lane & 7) + ((lane >> 4) << 3)) * TA_SROW + kk * 16 + ((lane >> 3) & 1) * 8);
                mma16(s[2 * np], qa[kk], b[0], b[1]);
                mma16(s[2 * np + 1], qa[kk], b[2], b[3]);
            }
        }
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int key = j * 8 + t4 * 2 + e, kg = k0 + key;
                const float sc = kss[key];
                s[j][e] = kg < nr0 ? s[j][e] * sc : -INFINITY;
                s[j][2 + e] = kg < nr1 ? s[j][2 + e] * sc : -INFINITY;
                mx0 = fmaxf(mx0, s[j][e]);
                mx1 = fmaxf(mx1, s[j][2 + e]);
            }
#pragma unroll
        for (int x = 1; x <= 2; x <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, x));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, x));
        }
        const float nm0 = fmaxf(m0, mx0), nm1 = fmaxf(m1, mx1);
        const float c0 = nm0 == -INFINITY ? 1.f : __expf(m0 - nm0), c1 = nm1 == -INFINITY ? 1.f : __expf(m1 - nm1);
        float ls0 = 0.f, ls1 = 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                s[j][e] = nm0 == -INFINITY ? 0.f : __expf(s[j][e] - nm0);
                s[j][2 + e] = nm1 == -INFINITY ? 0.f : __expf(s[j][2 + e] - nm1);
                ls0 += s[j][e];
                ls1 += s[j][2 + e];
            }
        l0 = l0 * c0 + ls0;
        l1 = l1 * c1 + ls1;
        m0 = nm0;
        m1 = nm1;
#pragma unroll
        for (int j = 0; j < HD / 8; j++) { o[j][0] *= c0; o[j][1] *= c0; o[j][2] *= c1; o[j][3] *= c1; }
#pragma unroll
        for (int j = 0; j < 4; j++) {  // P * V scale of its key (V is widened unscaled)
            const float va = vss[j * 8 + t4 * 2], vb = vss[j * 8 + t4 * 2 + 1];
            s[j][0] *= va; s[j][1] *= vb; s[j][2] *= va; s[j][3] *= vb;
        }
#pragma unroll
        for (int s2 = 0; s2 < 2; s2++) {
            uint32_t a[4];
            a[0] = f2h2(s[2 * s2][0], s[2 * s2][1]);
            a[1] = f2h2(s[2 * s2][2], s[2 * s2][3]);
            a[2] = f2h2(s[2 * s2 + 1][0], s[2 * s2 + 1][1]);
            a[3] = f2h2(s[2 * s2 + 1][2], s[2 * s2 + 1][3]);
#pragma unroll
            for (int dp = 0; dp < HD / 16; dp++) {
                uint32_t b[4];
                ta_ldsm4t(b, Vs + (s2 * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * TA_SROW + dp * 16 + (lane >> 4) * 8);
                mma16(o[2 * dp], a, b[0], b[1]);
                mma16(o[2 * dp + 1], a, b[2], b[3]);
            }
        }
    }
#pragma unroll
    for (int x = 1; x <= 2; x <<= 1) {
        l0 += __shfl_xor_sync(0xffffffffu, l0, x);
        l1 += __shfl_xor_sync(0xffffffffu, l1, x);
    }
    float* out = dst.o[z];
    if (tb + r0 < m) {
        float* ob = out + ((size_t)(tb + r0) * NQ + h) * (HD + 2);
#pragma unroll
        for (int j = 0; j < HD / 8; j++) *reinterpret_cast<float2*>(ob + j * 8 + t4 * 2) = make_float2(o[j][0], o[j][1]);
        if (t4 == 0) { ob[HD] = m0; ob[HD + 1] = l0; }
    }
    if (tb + r1 < m) {
        float* ob = out + ((size_t)(tb + r1) * NQ + h) * (HD + 2);
#pragma unroll
        for (int j = 0; j < HD / 8; j++) *reinterpret_cast<float2*>(ob + j * 8 + t4 * 2) = make_float2(o[j][2], o[j][3]);
        if (t4 == 0) { ob[HD] = m1; ob[HD + 1] = l1; }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(xq.done, 1u) == gridDim.x * gridDim.y * gridDim.z - 1) {
            *xq.done = 0;
            *xq.seq = s_s;
            __threadfence();
        }
    }
}
// merge the key splits 1..S-1 into split 0 (dst.o[0]): grid (NQ, m), HD threads
__global__ void k_attn_merge(AttnDst dst, int S) {
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const size_t off = ((size_t)t * NQ + h) * (HD + 2);
    float M = -INFINITY;
#pragma unroll
    for (int z = 0; z < 8; z++) if (z < S) M = fmaxf(M, dst.o[z][off + HD]);
    float a = 0.f, L = 0.f;
#pragma unroll
    for (int z = 0; z < 8; z++) {
        if (z >= S) break;
        const float mz = dst.o[z][off + HD];
        if (mz == -INFINITY) continue;
        const float c = __expf(mz - M);
        a += c * dst.o[z][off + d];
        L += c * dst.o[z][off + HD + 1];
    }
    __syncthreads();
    dst.o[0][off + d] = a;
    if (d == 0) { dst.o[0][off + HD] = M; dst.o[0][off + HD + 1] = L; }
}
// Decode / verify attention (m <= 32 rows): block = (kv head, row t, key split z) with 4 warps; the 12 q heads of the
// kv head are the 16-row MMA tile (rows 12..15 zero), so every K / V tile serves all 12 heads. Keys: all owned rows
// <= pos_t (sel == nullptr) or the row's selected blocks + its tail (as k_qsa_attn_b). Each warp walks its own
// 16-key tiles of the block's slice (fp8 -> fp16 in its smem), the 4 warps merge at the end; split z writes its
// partial to dst.o[z] (k_attn_merge combines the splits).
constexpr int QD_KEYS = 16, QD_SROW = HD + 8;
constexpr size_t QD_SMEM = (size_t)4 * 2 * QD_KEYS * QD_SROW * 2 + 4 * 2 * QD_KEYS * 4;
__global__ void __launch_bounds__(128) k_qsa_dec(const float* __restrict__ qb, const uint8_t* __restrict__ kc,
                                                 const uint8_t* __restrict__ vc, const float* __restrict__ ks,
                                                 const float* __restrict__ vs, const int* __restrict__ pos0, AttnDst dst,
                                                 Plan p, Xchg xq, const int* __restrict__ sel, const int* __restrict__ nsel,
                                                 int lbmax, const int* __restrict__ rpos = nullptr,
                                                 const int* __restrict__ rbase = nullptr, bool q16 = false) {
    // q16: the peers' q rows arrived as fp16 (k_qxsend16: q / 16, the same values this kernel converts to)
    extern __shared__ __align__(16) unsigned char qd_sm[];
    __shared__ unsigned s_s;
    __shared__ float wm[4][16], wl[4][16];
    const int kvh = blockIdx.x, t = blockIdx.y, z = blockIdx.z, S = gridDim.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    __half* Ks = reinterpret_cast<__half*>(qd_sm) + (size_t)warp * 2 * QD_KEYS * QD_SROW;
    __half* Vs = Ks + QD_KEYS * QD_SROW;
    float* kss = reinterpret_cast<float*>(reinterpret_cast<__half*>(qd_sm) + (size_t)4 * 2 * QD_KEYS * QD_SROW) + warp * 2 * QD_KEYS;
    float* vss = kss + QD_KEYS;
    constexpr int HPG = NQ / NKV;  // 12 q heads per kv head
    if (threadIdx.x == 0) {
        const unsigned s = *xq.seq + 1;
        for (int hh = kvh * HPG; hh < kvh * HPG + HPG; hh += QPC(p)) {
            const int ow = hh / QPC(p);
            if (ow != p.rank) for (unsigned ws_ = 0; (int)(qe_ld_acquire_sys(xq.flags[p.rank] + ow) - s) < 0;) { __nanosleep(20); QE_WD_POLL(ws_, kWdQsaX, ow, s, qe_ld_acquire_sys(xq.flags[p.rank] + ow)); }
        }
        const int ow2 = (kvh * HPG + HPG - 1) / QPC(p);
        if (ow2 != p.rank) for (unsigned ws_ = 0; (int)(qe_ld_acquire_sys(xq.flags[p.rank] + ow2) - s) < 0;) { __nanosleep(20); QE_WD_POLL(ws_, kWdQsaX, ow2, s, qe_ld_acquire_sys(xq.flags[p.rank] + ow2)); }
        s_s = s;
    }
    __syncthreads();
    // Q fragments of heads kvh * 12 + g (rows g) and + g + 8 (rows g + 8 < 12)
    uint32_t qa[HD / 16][4];
    {
        auto qrow = [&](int h) -> const float* {
            const int owner = h / QPC(p);
            return owner == p.rank ? qb + ((size_t)t * (NQ + 4) + h) * HD
                                   : xq.recv[p.rank] + (((size_t)(s_s & (XSLOT - 1)) * p.ngpu + owner) * XROWS + t) * qper(p) + (h - owner * QPC(p)) * HD;
        };
        auto qrow16 = [&](int h) -> const __half* {  // fp16 rows of a peer (nullptr: own head, fp32)
            const int owner = h / QPC(p);
            return owner == p.rank || !q16 ? nullptr
                 : reinterpret_cast<const __half*>(xq.recv[p.rank] + ((size_t)(s_s & (XSLOT - 1)) * p.ngpu + owner) * XROWS * qper(p)) +
                       (size_t)t * qper(p) + (h - owner * QPC(p)) * HD;
        };
        const int h0 = kvh * HPG + g, h1 = kvh * HPG + g + 8;
        const float* q0 = qrow(h0);
        const float* q1 = g + 8 < HPG ? qrow(h1) : nullptr;
        const __half* q0h = qrow16(h0);
        const __half* q1h = g + 8 < HPG ? qrow16(h1) : nullptr;
        auto ld2 = [&](const float* f, const __half* hp, int c) -> uint32_t {
            if (hp) return __ldcg(reinterpret_cast<const unsigned*>(hp + c));
            if (!f) return 0u;
            const float2 a = __ldcg(reinterpret_cast<const float2*>(f + c));
            return f2h2(a.x * (1.f / 16.f), a.y * (1.f / 16.f));
        };
#pragma unroll
        for (int kk = 0; kk < HD / 16; kk++) {
            const int c = kk * 16 + 2 * t4;
            qa[kk][0] = ld2(q0, q0h, c);
            qa[kk][1] = ld2(q1, q1h, c);
            qa[kk][2] = ld2(q0, q0h, c + 8);
            qa[kk][3] = ld2(q1, q1h, c + 8);
        }
    }
    // the row's key list (batched rows: the row's own position and KV region)
    const int pos = rpos ? rpos[t] : *pos0 + t;
    if (rbase) { const size_t rb = rbase[t]; kc += rb * NKV * HD; vc += rb * NKV * HD; ks += rb * NKV; vs += rb * NKV; }
    const int ns = sel ? nsel[t] : -1;
    const int nrows_all = owned_upto(p, pos);
    const int rc = ns < 0 ? nrows_all : owned_upto(p, IDX_R * ((pos + 1) / IDX_R) - 1);
    const int nit = ns < 0 ? nrows_all : ns * IDX_R + (nrows_all - rc);
    auto key_row = [&](int it) { return ns < 0 ? it : (it < ns * IDX_R ? sel[(size_t)t * lbmax + it / IDX_R] * IDX_R + it % IDX_R : rc + it - ns * IDX_R); };
    const int ntile = (nit + QD_KEYS - 1) / QD_KEYS;
    const int tile0 = z * ntile / S, tile1 = (z + 1) * ntile / S;
    float o[HD / 8][4];
#pragma unroll
    for (int j = 0; j < HD / 8; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    for (int tile = tile0 + warp; tile < tile1; tile += 4) {
        const int k0 = tile * QD_KEYS;
        // 16 keys x 256 fp8 for K and V: lane owns key (lane >> 1), half (lane & 1) of the row = 128 bytes = 8 uint4
        {
            const int key = lane >> 1, hf = lane & 1, it = k0 + key;
            const bool ok = it < nit;
            const int r = ok ? key_row(it) : 0;
            const uint4* kp = reinterpret_cast<const uint4*>(kc + ((size_t)r * NKV + kvh) * HD + hf * 128);
            const uint4* vp = reinterpret_cast<const uint4*>(vc + ((size_t)r * NKV + kvh) * HD + hf * 128);
#pragma unroll
            for (int c = 0; c < 8; c++) {
                const uint4 kw = ok ? __ldg(kp + c) : make_uint4(0, 0, 0, 0), vw = ok ? __ldg(vp + c) : make_uint4(0, 0, 0, 0);
                uint32_t kh[8], vh[8];
                f8x4_h(kw.x, kh[0], kh[1]); f8x4_h(kw.y, kh[2], kh[3]); f8x4_h(kw.z, kh[4], kh[5]); f8x4_h(kw.w, kh[6], kh[7]);
                f8x4_h(vw.x, vh[0], vh[1]); f8x4_h(vw.y, vh[2], vh[3]); f8x4_h(vw.z, vh[4], vh[5]); f8x4_h(vw.w, vh[6], vh[7]);
                uint4* kd = reinterpret_cast<uint4*>(Ks + key * QD_SROW + hf * 128 + c * 16);
                uint4* vd = reinterpret_cast<uint4*>(Vs + key * QD_SROW + hf * 128 + c * 16);
                kd[0] = make_uint4(kh[0], kh[1], kh[2], kh[3]); kd[1] = make_uint4(kh[4], kh[5], kh[6], kh[7]);
                vd[0] = make_uint4(vh[0], vh[1], vh[2], vh[3]); vd[1] = make_uint4(vh[4], vh[5], vh[6], vh[7]);
            }
            if (hf == 0) {
                kss[key] = ok ? ks[r * NKV + kvh] * 256.f : 0.f;
                vss[key] = ok ? vs[r * NKV + kvh] * 256.f : 0.f;
            }
        }
        __syncwarp();
        float sc[2][4];
#pragma unroll
        for (int j = 0; j < 2; j++) sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < HD / 16; kk++) {
            uint32_t b[4];
            ta_ldsm4(b, Ks + ((lane & 7) + ((lane >> 4) << 3)) * QD_SROW + kk * 16 + ((lane >> 3) & 1) * 8);
            mma16(sc[0], qa[kk], b[0], b[1]);
            mma16(sc[1], qa[kk], b[2], b[3]);
        }
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int key = j * 8 + t4 * 2 + e;
                const bool ok = k0 + key < nit;
                const float kf = kss[key];
                sc[j][e] = ok ? sc[j][e] * kf : -INFINITY;
                sc[j][2 + e] = ok ? sc[j][2 + e] * kf : -INFINITY;
                mx0 = fmaxf(mx0, sc[j][e]);
                mx1 = fmaxf(mx1, sc[j][2 + e]);
            }
#pragma unroll
        for (int x = 1; x <= 2; x <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, x));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, x));
        }
        const float nm0 = fmaxf(m0, mx0), nm1 = fmaxf(m1, mx1);
        const float c0 = nm0 == -INFINITY ? 1.f : __expf(m0 - nm0), c1 = nm1 == -INFINITY ? 1.f : __expf(m1 - nm1);
        float ls0 = 0.f, ls1 = 0.f;
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                sc[j][e] = nm0 == -INFINITY ? 0.f : __expf(sc[j][e] - nm0);
                sc[j][2 + e] = nm1 == -INFINITY ? 0.f : __expf(sc[j][2 + e] - nm1);
                ls0 += sc[j][e];
                ls1 += sc[j][2 + e];
            }
        l0 = l0 * c0 + ls0;
        l1 = l1 * c1 + ls1;
        m0 = nm0;
        m1 = nm1;
#pragma unroll
        for (int j = 0; j < HD / 8; j++) { o[j][0] *= c0; o[j][1] *= c0; o[j][2] *= c1; o[j][3] *= c1; }
#pragma unroll
        for (int j = 0; j < 2; j++) {
            const float va = vss[j * 8 + t4 * 2], vb = vss[j * 8 + t4 * 2 + 1];
            sc[j][0] *= va; sc[j][1] *= vb; sc[j][2] *= va; sc[j][3] *= vb;
        }
        uint32_t a[4];
        a[0] = f2h2(sc[0][0], sc[0][1]);
        a[1] = f2h2(sc[0][2], sc[0][3]);
        a[2] = f2h2(sc[1][0], sc[1][1]);
        a[3] = f2h2(sc[1][2], sc[1][3]);
#pragma unroll
        for (int dp = 0; dp < HD / 16; dp++) {
            uint32_t b[4];
            ta_ldsm4t(b, Vs + ((lane & 7) + ((lane >> 3) & 1) * 8) * QD_SROW + dp * 16 + (lane >> 4) * 8);
            mma16(o[2 * dp], a, b[0], b[1]);
            mma16(o[2 * dp + 1], a, b[2], b[3]);
        }
        __syncwarp();
    }
#pragma unroll
    for (int x = 1; x <= 2; x <<= 1) {
        l0 += __shfl_xor_sync(0xffffffffu, l0, x);
        l1 += __shfl_xor_sync(0xffffffffu, l1, x);
    }
    // merge the 4 warps: rows g (head g) and g + 8 (head g + 8 < 12)
    if (t4 == 0) { wm[warp][g] = m0; wm[warp][g + 8] = m1; wl[warp][g] = l0; wl[warp][g + 8] = l1; }
    __syncthreads();
    float M0 = -INFINITY, M1 = -INFINITY;
#pragma unroll
    for (int w = 0; w < 4; w++) { M0 = fmaxf(M0, wm[w][g]); M1 = fmaxf(M1, wm[w][g + 8]); }
    const float f0 = m0 == -INFINITY ? 0.f : __expf(m0 - M0), f1 = m1 == -INFINITY ? 0.f : __expf(m1 - M1);
    float L0 = 0.f, L1 = 0.f;
#pragma unroll
    for (int w = 0; w < 4; w++) {
        L0 += wm[w][g] == -INFINITY ? 0.f : __expf(wm[w][g] - M0) * wl[w][g];
        L1 += wm[w][g + 8] == -INFINITY ? 0.f : __expf(wm[w][g + 8] - M1) * wl[w][g + 8];
    }
    // sum the warps' scaled accumulators through shared memory (reuses the K / V area: [4 warps][16 rows][HD])
    __syncthreads();
    float* acc_s = reinterpret_cast<float*>(qd_sm);
#pragma unroll
    for (int j = 0; j < HD / 8; j++) {
        const int c = j * 8 + t4 * 2;
        acc_s[((size_t)warp * 16 + g) * HD + c] = o[j][0] * f0;
        acc_s[((size_t)warp * 16 + g) * HD + c + 1] = o[j][1] * f0;
        acc_s[((size_t)warp * 16 + g + 8) * HD + c] = o[j][2] * f1;
        acc_s[((size_t)warp * 16 + g + 8) * HD + c + 1] = o[j][3] * f1;
    }
    __syncthreads();
    float* out = dst.o[z];
    for (int idx = threadIdx.x; idx < HPG * HD; idx += 128) {
        const int row = idx / HD, d = idx % HD;
        const float v = acc_s[(size_t)row * HD + d] + acc_s[((size_t)16 + row) * HD + d] + acc_s[((size_t)32 + row) * HD + d] +
                        acc_s[((size_t)48 + row) * HD + d];
        out[((size_t)t * NQ + kvh * HPG + row) * (HD + 2) + d] = v;
    }
    if (threadIdx.x < HPG) {
        const int row = threadIdx.x;
        float M = -INFINITY, L = 0.f;
        for (int w = 0; w < 4; w++) M = fmaxf(M, wm[w][row]);
        for (int w = 0; w < 4; w++) if (wm[w][row] != -INFINITY) L += __expf(wm[w][row] - M) * wl[w][row];
        out[((size_t)t * NQ + kvh * HPG + row) * (HD + 2) + HD] = M;
        out[((size_t)t * NQ + kvh * HPG + row) * (HD + 2) + HD + 1] = L;
    }
    (void)L0; (void)L1;
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(xq.done, 1u) == gridDim.x * gridDim.y * gridDim.z - 1) {
            *xq.done = 0;
            *xq.seq = s_s;
            __threadfence();
        }
    }
}
// Many-row GEMV for prefill chunks: y[t][n] = sum_k x[t][k] W[n][k] over m rows in one launch, grid (N / 16 tiles,
// ceil(m / 32) row groups), 4 warps split K; W8 (int8 g128, fp16 mma on exact int8 values, x widened bf16 -> fp16)
// or W16 (bf16 mma). The decode GEMVs cover <= 32 rows with N / 16 blocks; here the row groups fill the GPU.
__device__ __forceinline__ uint32_t bf2_to_h2(uint32_t v) {
    const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v));
    const __half2 h = __floats2half2_rn(f.x, f.y);
    return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ void mma_bf(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
template <bool W8M, int NT = 4, int NW = 4>
__global__ void __launch_bounds__(NW * 32) k_rows_gemv(const uint4* __restrict__ wq, const bf16* __restrict__ ws, int K,
                                                   const bf16* __restrict__ x, int ldx, float* __restrict__ y, int ldy, int m) {
    const int r = blockIdx.x, t0 = blockIdx.y * 8 * NT, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t4 = lane & 3;
    const int nks = K >> 4;  // k-steps
    const int kb = warp * nks / NW, ke = (warp + 1) * nks / NW;
    const uint32_t* xr[NT];
#pragma unroll
    for (int j = 0; j < NT; j++) {
        const int t = t0 + 8 * j + g;
        xr[j] = reinterpret_cast<const uint32_t*>(x + (size_t)(t < m ? t : t0) * ldx) + t4;
    }
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    if constexpr (W8M) {
        // [N/16][K/32][32][16 B]: one load = two k-steps; scales per 128-k group (8 k-steps)
        const uint4* qp = wq + (size_t)r * (K >> 5) * 32 + lane;
        const bf16* sp = ws + (size_t)r * (K >> 7) * 16;
        for (int k2 = kb >> 1; k2 < (ke >> 1); k2 += 4) {  // one 128-k group per iteration (kb, ke multiples of 8)
            const int gi = k2 >> 2;
            float c[NT][4];
#pragma unroll
            for (int j = 0; j < NT; j++) c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0.f;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const uint4 w = __ldg(qp + (size_t)(k2 + i) * 32);
                const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    uint32_t a[4];
                    a_from_w8(wv[h * 2], wv[h * 2 + 1], a);
                    const int k0 = ((k2 + i) * 2 + h) * 8;  // in bf16 pairs
#pragma unroll
                    for (int j = 0; j < NT; j++) mma16(c[j], a, bf2_to_h2(__ldg(xr[j] + k0)), bf2_to_h2(__ldg(xr[j] + k0 + 4)));
                }
            }
            const float sg = __bfloat162float(sp[gi * 16 + g]), sg8 = __bfloat162float(sp[gi * 16 + g + 8]);
#pragma unroll
            for (int j = 0; j < NT; j++) { acc[j][0] += c[j][0] * sg; acc[j][1] += c[j][1] * sg; acc[j][2] += c[j][2] * sg8; acc[j][3] += c[j][3] * sg8; }
        }
    } else {
        const uint4* wp = wq + (size_t)r * nks * 32 + lane;
        for (int ks = kb; ks < ke; ks++) {
            const uint4 w = __ldg(wp + (size_t)ks * 32);
            const uint32_t a[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
            for (int j = 0; j < NT; j++) mma_bf(acc[j], a, __ldg(xr[j] + ks * 8), __ldg(xr[j] + ks * 8 + 4));
        }
    }
    __shared__ float red[NW][NT][4][32];
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
        for (int i = 0; i < 4; i++) red[warp][j][i][lane] = acc[j][i];
    __syncthreads();
    if (warp != 0) return;
#pragma unroll
    for (int j = 0; j < NT; j++) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            float v = 0.f;
#pragma unroll
            for (int w = 0; w < NW; w++) v += red[w][j][i][lane];
            acc[j][i] = v;
        }
        const int n0 = t0 + 8 * j + 2 * t4, n1 = n0 + 1, row0 = r * 16;
        if (n0 < m) { y[(size_t)n0 * ldy + row0 + g] = acc[j][0]; y[(size_t)n0 * ldy + row0 + g + 8] = acc[j][2]; }
        if (n1 < m) { y[(size_t)n1 * ldy + row0 + g] = acc[j][1]; y[(size_t)n1 * ldy + row0 + g + 8] = acc[j][3]; }
    }
}
// Prefill GEMM for W8 / W16 matrices: block = 4 warps x one 16-row weight tile each (64 outputs), 32 rows; the x rows are
// staged per 128-k chunk in shared memory (cp.async, 2 stages) and shared by the 4 warps; bf16 mma (int8 weights widened
// exactly to bf16). grid (N / 64, ceil(m / 32)). N must be a multiple of 64, K of 128, ldx of 8.
constexpr int RG_XS = 128 + 8;  // smem row (bf16)
__device__ __forceinline__ uint32_t s8x2_bf16(uint32_t v) {  // two int8 (bytes 0, 1 of v) -> bf16x2 (exact)
    const int a = (int)(signed char)(v & 0xFF), b = (int)(signed char)((v >> 8) & 0xFF);
    const __nv_bfloat162 h = __floats2bfloat162_rn((float)a, (float)b);
    return *reinterpret_cast<const uint32_t*>(&h);
}
template <bool W8M, int RM = 32>  // RM rows per block (32 / 64)
__global__ void __launch_bounds__(128) k_rows_gemm(const uint4* __restrict__ wq, const bf16* __restrict__ ws, int K, int N,
                                                   const bf16* __restrict__ x, int ldx, float* __restrict__ y, int ldy, int m) {
    __shared__ __align__(16) bf16 xs[2][RM][RG_XS];
    constexpr int NTT = RM / 8;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    const int Tr = blockIdx.x * 4 + warp, t0 = blockIdx.y * RM;
    const bool live = Tr * 16 < N;  // N % 64 != 0: the last block's extra warps only help staging
    const int T = live ? Tr : 0;
    const int nchunk = K / 128;
    auto stage = [&](int ck, int buf) {
        for (int i = threadIdx.x; i < RM * 16; i += 128) {  // RM rows x 16 chunks of 16 bytes
            const int r = i >> 4, c = i & 15, t = t0 + r;
            const bool ok = t < m;
            const unsigned sa = (unsigned)__cvta_generic_to_shared(&xs[buf][r][c * 8]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sa),
                         "l"(x + (size_t)(ok ? t : t0) * ldx + ck * 128 + c * 8), "r"(ok ? 16 : 0));
        }
        asm volatile("cp.async.commit_group;");
    };
    float acc[NTT][4];
#pragma unroll
    for (int j = 0; j < NTT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    stage(0, 0);
    for (int ck = 0; ck < nchunk; ck++) {
        const int buf = ck & 1;
        // this warp's weights of the chunk (issued before the wait: their latency overlaps it)
        uint4 w[W8M ? 4 : 8];
        if constexpr (W8M) {
#pragma unroll
            for (int i = 0; i < 4; i++) w[i] = __ldg(wq + ((size_t)T * (K >> 5) + ck * 4 + i) * 32 + lane);
        } else {
#pragma unroll
            for (int i = 0; i < 8; i++) w[i] = __ldg(wq + ((size_t)T * (K >> 4) + ck * 8 + i) * 32 + lane);
        }
        if (ck + 1 < nchunk) { stage(ck + 1, buf ^ 1); asm volatile("cp.async.wait_group 1;"); }
        else asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        float c[NTT][4];
#pragma unroll
        for (int j = 0; j < NTT; j++) c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0.f;
#pragma unroll
        for (int kp = 0; kp < 4; kp++) {  // pairs of k-steps (32 k)
            uint32_t b[NTT][4];
#pragma unroll
            for (int j = 0; j < NTT; j++) ta_ldsm4(b[j], &xs[buf][j * 8 + (lane & 7)][kp * 32 + (lane >> 3) * 8]);
#pragma unroll
            for (int h = 0; h < 2; h++) {
                uint32_t a[4];
                if constexpr (W8M) {
                    const uint4 wv = w[kp];
                    const uint32_t lo = h ? wv.z : wv.x, hi = h ? wv.w : wv.y;
                    a[0] = s8x2_bf16(lo); a[1] = s8x2_bf16(lo >> 16); a[2] = s8x2_bf16(hi); a[3] = s8x2_bf16(hi >> 16);
                } else {
                    const uint4 wv = w[kp * 2 + h];
                    a[0] = wv.x; a[1] = wv.y; a[2] = wv.z; a[3] = wv.w;
                }
#pragma unroll
                for (int j = 0; j < NTT; j++) mma_bf(c[j], a, b[j][2 * h], b[j][2 * h + 1]);
            }
        }
        if constexpr (W8M) {
            const float sg = __bfloat162float(ws[((size_t)T * nchunk + ck) * 16 + g]), sg8 = __bfloat162float(ws[((size_t)T * nchunk + ck) * 16 + g + 8]);
#pragma unroll
            for (int j = 0; j < NTT; j++) { acc[j][0] += c[j][0] * sg; acc[j][1] += c[j][1] * sg; acc[j][2] += c[j][2] * sg8; acc[j][3] += c[j][3] * sg8; }
        } else {
#pragma unroll
            for (int j = 0; j < NTT; j++) { acc[j][0] += c[j][0]; acc[j][1] += c[j][1]; acc[j][2] += c[j][2]; acc[j][3] += c[j][3]; }
        }
        __syncthreads();  // the buffer is restaged two chunks later
    }
    if (!live) return;
    const int row0 = T * 16;
#pragma unroll
    for (int j = 0; j < NTT; j++) {
        const int n0 = t0 + 8 * j + 2 * t4, n1 = n0 + 1;
        if (n0 < m) { y[(size_t)n0 * ldy + row0 + g] = acc[j][0]; y[(size_t)n0 * ldy + row0 + g + 8] = acc[j][2]; }
        if (n1 < m) { y[(size_t)n1 * ldy + row0 + g] = acc[j][1]; y[(size_t)n1 * ldy + row0 + g + 8] = acc[j][3]; }
    }
}
// combine the cards' partials of the local q heads, * sigmoid(gate) -> XH input of o_proj. grid (local heads, m)
__global__ void k_qsa_comb(const float* __restrict__ all, int ngpu, int m, const float* __restrict__ proj, int ld,
                           bf16* __restrict__ out, int ldo, Plan p) {
    const int hl = blockIdx.x, t = blockIdx.y, d = threadIdx.x, h = p.q0 + hl;
    // all: per row t the gathered [ngpu][24][HD + 2] (one all-gather per row)
    const size_t per = (size_t)NQ * (HD + 2);
    float M = -INFINITY;
    for (int g = 0; g < ngpu; g++) M = fmaxf(M, all[((size_t)t * ngpu + g) * per + (size_t)h * (HD + 2) + HD]);
    float a = 0.f, L = 0.f;
    for (int g = 0; g < ngpu; g++) {
        const float* o = all + ((size_t)t * ngpu + g) * per + (size_t)h * (HD + 2);
        if (o[HD] == -INFINITY) continue;
        const float c = __expf(o[HD] - M);
        a += c * o[d];
        L += c * o[HD + 1];
    }
    const float gate = proj[(size_t)t * ld + NQ * HD + hl * HD + d];
    out[(size_t)t * ldo + hl * HD + d] = __float2bfloat16(a / L * sigm(gate));
}

// ------------------------------------------------------------------ MoE
// Router: softmax over 512, top-10 (value desc, index asc), renormalized (norm_topk_prob); shared gate sigmoid.
// Jobs = local 128-wide intermediate groups of every expert hit by any row (+ the shared expert's local groups),
// in (expert, group) order; jw[job][t] = routing weight of row t (0 if not routed). One block of 512 threads.
__global__ void k_route(const float* __restrict__ lg, int m, const int* __restrict__ gmap, const char* __restrict__ groups,
                        MoeJob* __restrict__ jobs, float* __restrict__ jw, int* __restrict__ njobs) {
    __shared__ float red[32];
    __shared__ float wsel[MAXM][TOPK];
    __shared__ int isel[MAXM][TOPK];
    __shared__ float bv[16];
    __shared__ int bi[16];
    __shared__ int cnt[NE + 1];
    const int e = threadIdx.x;
    bool hit = false;
    for (int t = 0; t < m; t++) {
        const float x = lg[t * 528 + e];
        const float mx = block_max<512>(x, red);
        const float ex = __expf(x - mx);
        const float den = block_sum<512>(ex, red);
        const float pr = ex / den;
        bool taken = false;
        float tot = 0.f;
        for (int k = 0; k < TOPK; k++) {
            float v = taken ? -1.f : pr;
            int idx = e;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float v2 = __shfl_xor_sync(0xffffffffu, v, o);
                const int i2 = __shfl_xor_sync(0xffffffffu, idx, o);
                if (v2 > v || (v2 == v && i2 < idx)) { v = v2; idx = i2; }
            }
            __syncthreads();
            if ((e & 31) == 0) { bv[e >> 5] = v; bi[e >> 5] = idx; }
            __syncthreads();
            if (e == 0) {
                float V = bv[0]; int I = bi[0];
                for (int w = 1; w < 16; w++) if (bv[w] > V || (bv[w] == V && bi[w] < I)) { V = bv[w]; I = bi[w]; }
                wsel[t][k] = V; isel[t][k] = I;
            }
            __syncthreads();
            if (isel[t][k] == e) taken = true;
            tot += wsel[t][k];
        }
        hit |= taken;
        __syncthreads();
        if (e < TOPK) wsel[t][e] /= tot;
        __syncthreads();
    }
    // local groups per expert (gmap[e * 5 + g] = local index or -1); expert NE = shared (always)
    auto ngr = [&](int ex) { int c = 0; for (int g = 0; g < 5; g++) c += gmap[ex * 5 + g] >= 0; return c; };
    cnt[e] = hit ? ngr(e) : 0;
    if (e == 0) cnt[NE] = ngr(NE);
    __syncthreads();
    // exclusive prefix over 513 entries (thread e owns e; thread 0 also NE): simple serial-chunk scan
    __shared__ int pre[NE + 1];
    if (e == 0) {
        int a = 0;
        for (int i = 0; i <= NE; i++) { pre[i] = a; a += cnt[i]; }
        *njobs = a;
    }
    __syncthreads();
    auto emit = [&](int ex, bool shared) {
        int j = pre[ex];
        for (int g = 0; g < 5; g++) {
            const int li = gmap[ex * 5 + g];
            if (li < 0) continue;
            const char* b = groups + (size_t)li * GRP_BYTES;
            MoeJob jb;
            jb.gq = reinterpret_cast<const uint4*>(b);
            jb.gs = reinterpret_cast<const __half*>(b + GG_Q);
            jb.dq = reinterpret_cast<const uint4*>(b + GG_Q + GG_S);
            jb.ds = reinterpret_cast<const __half*>(b + GG_Q + GG_S + GD_Q);
            jobs[j] = jb;
            for (int t = 0; t < m; t++) {
                float w = 0.f;
                if (shared) w = sigm(lg[t * 528 + NE]);
                else for (int k = 0; k < TOPK; k++) if (isel[t][k] == ex) w = wsel[t][k];
                jw[j * MAXM + t] = w;
            }
            j++;
        }
    };
    if (cnt[e]) emit(e, false);
    if (e == 0) emit(NE, true);
}
// Faster router: top-10 on the logits (softmax is monotonic; with norm_topk_prob the full denominator cancels:
// w_i = exp(l_i - l_max) / sum_sel exp(l_j - l_max)). Per warp a local top-10 by warp argmax, warp 0 merges the
// 16 x 10 candidates. Ties: value desc, index asc. Jobs as k_route. 512 threads.
__device__ __forceinline__ void warp_argmax(float& v, int& i) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float v2 = __shfl_xor_sync(0xffffffffu, v, o);
        const int i2 = __shfl_xor_sync(0xffffffffu, i, o);
        if (v2 > v || (v2 == v && i2 < i)) { v = v2; i = i2; }
    }
}
// lg: nsplit K-slices of the router logits [nsplit][MAXM][528] (summed here)
#ifdef ROUTE_PROF
__device__ long long g_rprof[16];
#define RP(k) do { __syncthreads(); if (threadIdx.x == 0) g_rprof[k] = clock64(); } while (0)
#else
#define RP(k) __syncthreads()  // the phase barriers are needed either way
#endif
__global__ void __launch_bounds__(512) k_route2(const float* __restrict__ lg, int m, const int* __restrict__ gmap,
                                                const char* __restrict__ groups, MoeJob* __restrict__ jobs,
                                                float* __restrict__ jw, int* __restrict__ njobs, int nsplit,
                                                int* __restrict__ stats = nullptr, ColdTab ct = nullptr,
                                                int ndev = 1 << 30, unsigned* __restrict__ coldcnt = nullptr) {
    __shared__ float cv[16][TOPK];
    __shared__ int ci[16][TOPK];
    __shared__ float wsel[MAXM][TOPK];
    __shared__ int isel[MAXM][TOPK];
    __shared__ int hitm[NE];
    __shared__ int wtot[16];
    const int e = threadIdx.x, warp = e >> 5, lane = e & 31;
    int gm[NGRP];
#pragma unroll
    for (int g = 0; g < NGRP; g++) gm[g] = gmap[e * NGRP + g];
    float xs[MAXM];
#pragma unroll
    for (int t = 0; t < MAXM; t++) {
        xs[t] = 0.f;
        if (t < m)
            for (int sp = 0; sp < nsplit; sp++) xs[t] += lg[((size_t)sp * MAXM + t) * 528 + e];
    }
    __shared__ float shw[MAXM];
    __shared__ int shg[NGRP];
    if (e < m) {  // shared expert gate of row e
        float x = 0.f;
        for (int sp = 0; sp < nsplit; sp++) x += lg[((size_t)sp * MAXM + e) * 528 + NE];
        shw[e] = sigm(x);
    }
    if (e >= 32 && e < 32 + NGRP) shg[e - 32] = gmap[NE * NGRP + e - 32];
    hitm[e] = 0;
    RP(0);
    for (int t = 0; t < m; t++) {
        float x = 0.f;
#pragma unroll
        for (int tt = 0; tt < MAXM; tt++) if (tt == t) x = xs[tt];
        // per warp: top-10 by redux.sync max on order-preserving keys; ties -> lowest lane (= lowest index)
        unsigned key = fkey(x);
        for (int k = 0; k < TOPK; k++) {
            const unsigned M = __reduce_max_sync(0xffffffffu, key);
            const unsigned b = __ballot_sync(0xffffffffu, key == M);
            const int l = __ffs(b) - 1;
            if (lane == l) key = 0u;
            if (lane == 0) { cv[warp][k] = __uint_as_float(M); ci[warp][k] = warp * 32 + l; }
        }
        RP(1);
        if (warp == 0) {  // merge 16 x 10 candidates (keys), ties by expert index
            unsigned lk[5];
            int li[5];
#pragma unroll
            for (int j = 0; j < 5; j++) { const int c = lane * 5 + j; lk[j] = __float_as_uint(cv[c / TOPK][c % TOPK]); li[j] = ci[c / TOPK][c % TOPK]; }
            float mx = 0.f, tot = 0.f;
            for (int k = 0; k < TOPK; k++) {
                unsigned bk = 0u;
                int bi = 0x7fffffff;
#pragma unroll
                for (int j = 0; j < 5; j++)
                    if (lk[j] > bk || (lk[j] == bk && li[j] < bi)) { bk = lk[j]; bi = li[j]; }
                const unsigned M = __reduce_max_sync(0xffffffffu, bk);
                const int I = (int)__reduce_min_sync(0xffffffffu, bk == M ? (unsigned)bi : 0xffffffffu);
#pragma unroll
                for (int j = 0; j < 5; j++) if (lk[j] == M && li[j] == I) lk[j] = 0u;
                const float v = kfloat(M);
                if (k == 0) mx = v;
                const float w = __expf(v - mx);
                tot += w;
                if (lane == 0) { wsel[t][k] = w; isel[t][k] = I; }
            }
            __syncwarp();
            if (lane < TOPK) {
                wsel[t][lane] /= tot;
                hitm[isel[t][lane]] = 1;
                if (stats) atomicAdd(stats + isel[t][lane], 1);
            }
        }
        __syncthreads();
    }
    RP(2);
    // jobs: count local groups of hit experts, block exclusive scan, emit
    int c = 0;
    if (hitm[e])
#pragma unroll
        for (int g = 0; g < NGRP; g++) c += gm[g] >= 0;
    int inc = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int n = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += n;
    }
    if (lane == 31) wtot[warp] = inc;
    __syncthreads();
    int base = 0;
    for (int w = 0; w < warp; w++) base += wtot[w];
    int total = 0;
    for (int w = 0; w < 16; w++) total += wtot[w];
    auto emit = [&](int ex, int j, bool shared) {
        for (int g = 0; g < NGRP; g++) {
            const int li = shared ? shg[g] : gm[g];
            if (li < 0) continue;
            const char* b = grp_ptr(groups, li, ndev, ct, coldcnt);
            MoeJob jb;
            jb.gq = reinterpret_cast<const uint4*>(b);
            jb.gs = reinterpret_cast<const __half*>(b + GG_Q);
            jb.dq = reinterpret_cast<const uint4*>(b + GG_Q + GG_S);
            jb.ds = reinterpret_cast<const __half*>(b + GG_Q + GG_S + GD_Q);
            jobs[j] = jb;
            for (int t = 0; t < m; t++) {
                float w = 0.f;
                if (shared) w = shw[t];
                else for (int k = 0; k < TOPK; k++) if (isel[t][k] == ex) w = wsel[t][k];
                jw[j * MAXM + t] = w;
            }
            j++;
        }
    };
    RP(3);
    if (c) emit(e, base + inc - c, false);
    RP(4);
    if (e == 0) {
        int ns = 0;
        for (int g = 0; g < NGRP; g++) ns += shg[g] >= 0;
        emit(NE, total, true);
        *njobs = total + ns;
    }
}
// Router v3: one block per row computes the row's top-10 (per-warp redux.sync top-10, then the 160 candidates are
// ranked in parallel: rank = #candidates with a larger key or an equal key and a lower index); the last block to
// finish builds the jobs of all rows (as k_route2). sel: global scratch [MAXM][TOPK] (idx, weight), cnt: arrival.
__global__ void __launch_bounds__(512) k_route3(const float* __restrict__ lg, int m, const int* __restrict__ gmap,
                                                const char* __restrict__ groups, MoeJob* __restrict__ jobs,
                                                float* __restrict__ jw, int* __restrict__ njobs, int nsplit,
                                                int* __restrict__ gsel, float* __restrict__ gw, unsigned* __restrict__ cnt) {
    __shared__ unsigned ck[16 * TOPK];
    __shared__ int ci2[16 * TOPK];
    __shared__ float wv[TOPK];
    __shared__ int wi[TOPK];
    __shared__ unsigned s_last;
    const int e = threadIdx.x, warp = e >> 5, lane = e & 31, t = blockIdx.x;
    float x = 0.f;
    for (int sp = 0; sp < nsplit; sp++) x += lg[((size_t)sp * MAXM + t) * 528 + e];
    unsigned key = fkey(x);
    for (int k = 0; k < TOPK; k++) {
        const unsigned M = __reduce_max_sync(0xffffffffu, key);
        const unsigned b = __ballot_sync(0xffffffffu, key == M);
        const int l = __ffs(b) - 1;
        if (lane == l) key = 0u;
        if (lane == 0) { ck[warp * TOPK + k] = M; ci2[warp * TOPK + k] = warp * 32 + l; }
    }
    __syncthreads();
    if (e < 16 * TOPK) {
        const unsigned mk = ck[e];
        const int mi = ci2[e];
        int rank = 0;
        for (int j = 0; j < 16 * TOPK; j++) {
            const unsigned kj = ck[j];
            rank += kj > mk || (kj == mk && ci2[j] < mi);
        }
        if (rank < TOPK) { wv[rank] = kfloat(mk); wi[rank] = mi; }
    }
    __syncthreads();
    if (e < 32) {
        const float v = e < TOPK ? __expf(wv[e] - wv[0]) : 0.f;
        float tot = v;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
        if (e < TOPK) { gsel[t * TOPK + e] = wi[e]; gw[t * TOPK + e] = v / tot; }
    }
    // last row block builds the jobs
    __syncthreads();
    if (e == 0) {
        __threadfence();
        s_last = atomicAdd(cnt, 1u) == (unsigned)m - 1;
    }
    __syncthreads();
    if (!s_last) return;
    __shared__ int isel[MAXM][TOPK];
    __shared__ float wsel[MAXM][TOPK];
    __shared__ int hitm[NE];
    __shared__ float shw[MAXM];
    __shared__ int shg[NGRP];
    __shared__ int wtot[16];
    hitm[e] = 0;
    if (e < m) {
        float xx = 0.f;
        for (int sp = 0; sp < nsplit; sp++) xx += lg[((size_t)sp * MAXM + e) * 528 + NE];
        shw[e] = sigm(xx);
    }
    if (e >= 32 && e < 32 + NGRP) shg[e - 32] = gmap[NE * NGRP + e - 32];
    int gm[NGRP];
#pragma unroll
    for (int g = 0; g < NGRP; g++) gm[g] = gmap[e * NGRP + g];
    __syncthreads();
    if (e < m * TOPK) {
        const int tt = e / TOPK, k = e % TOPK;
        const int id = __ldcg(gsel + tt * TOPK + k);
        isel[tt][k] = id;
        wsel[tt][k] = __ldcg(gw + tt * TOPK + k);
        hitm[id] = 1;
    }
    __syncthreads();
    int c = 0;
    if (hitm[e])
#pragma unroll
        for (int g = 0; g < NGRP; g++) c += gm[g] >= 0;
    int inc = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int n = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += n;
    }
    if (lane == 31) wtot[warp] = inc;
    __syncthreads();
    int base = 0, total = 0;
    for (int w = 0; w < 16; w++) { if (w < warp) base += wtot[w]; total += wtot[w]; }
    auto emit = [&](int ex, int j, bool shared) {
        for (int g = 0; g < NGRP; g++) {
            const int li = shared ? shg[g] : gm[g];
            if (li < 0) continue;
            const char* b = groups + (size_t)li * GRP_BYTES;
            MoeJob jb;
            jb.gq = reinterpret_cast<const uint4*>(b);
            jb.gs = reinterpret_cast<const __half*>(b + GG_Q);
            jb.dq = reinterpret_cast<const uint4*>(b + GG_Q + GG_S);
            jb.ds = reinterpret_cast<const __half*>(b + GG_Q + GG_S + GD_Q);
            jobs[j] = jb;
            for (int tt = 0; tt < m; tt++) {
                float w = 0.f;
                if (shared) w = shw[tt];
                else for (int k = 0; k < TOPK; k++) if (isel[tt][k] == ex) w = wsel[tt][k];
                jw[j * MAXM + tt] = w;
            }
            j++;
        }
    };
    if (c) emit(e, base + inc - c, false);
    if (e == 0) {
        int ns = 0;
        for (int g = 0; g < NGRP; g++) ns += shg[g] >= 0;
        emit(NE, total, true);
        *njobs = total + ns;
        *cnt = 0u;
    }
}
// Router v4: (a) k_topk_rows: one block per row, top-10 by per-warp redux.sync + parallel ranking of the 160
// candidates (ties by index), renormalized weights -> gsel/gw [MAXM][10] and the shared gate -> gsh[MAXM];
// (b) k_jobs: one block builds the jobs of all rows (hit experts' local groups in (expert, group) order + shared).
__global__ void __launch_bounds__(512) k_topk_rows(const float* __restrict__ lg, int nsplit, int* __restrict__ gsel,
                                                   float* __restrict__ gw, float* __restrict__ gsh, int* __restrict__ stats,
                                                   int split_rows = MAXM) {
    __shared__ unsigned ck[16 * TOPK];
    __shared__ int ci2[16 * TOPK];
    __shared__ float wv[TOPK];
    __shared__ int wi[TOPK];
    const int e = threadIdx.x, warp = e >> 5, lane = e & 31, t = blockIdx.x;
    float x = 0.f;
    for (int sp = 0; sp < nsplit; sp++) x += lg[((size_t)sp * split_rows + t) * 528 + e];
    if (e == 0) {
        float xs = 0.f;
        for (int sp = 0; sp < nsplit; sp++) xs += lg[((size_t)sp * split_rows + t) * 528 + NE];
        gsh[t] = sigm(xs);
    }
    // 10 rounds of a block-wide argmax: per warp redux.sync max + lowest lane, the 16 warp winners (double-buffered
    // smem) compared by every thread (ties: lower index) - one barrier per round
    unsigned key = fkey(x);
    for (int k = 0; k < TOPK; k++) {
        const unsigned M = __reduce_max_sync(0xffffffffu, key);
        const int l = __ffs(__ballot_sync(0xffffffffu, key == M)) - 1;
        if (lane == 0) { ck[(k & 1) * 16 + warp] = M; ci2[(k & 1) * 16 + warp] = warp * 32 + l; }
        __syncthreads();
        unsigned bk = ck[(k & 1) * 16];
        int bi = ci2[(k & 1) * 16];
#pragma unroll
        for (int w2 = 1; w2 < 16; w2++) {
            const unsigned k2 = ck[(k & 1) * 16 + w2];
            if (k2 > bk) { bk = k2; bi = ci2[(k & 1) * 16 + w2]; }  // same key: the lower warp (lower index) wins
        }
        if (bi == e) key = 0u;
        if (e == 0) { wv[k] = kfloat(bk); wi[k] = bi; }
    }
    __syncthreads();
    if (e < 32) {
        const float v = e < TOPK ? __expf(wv[e] - wv[0]) : 0.f;
        float tot = v;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
        if (e < TOPK) {
            gsel[t * TOPK + e] = wi[e];
            gw[t * TOPK + e] = v / tot;
            if (stats) atomicAdd(stats + wi[e], 1);
        }
    }
}
__device__ unsigned long long g_jobstat[MAXM + 1][2];  // debug FLASH_JOBSTAT: per row count {jobs, calls} of k_jobs
__global__ void __launch_bounds__(512) k_jobs(int m, const int* __restrict__ gmap, const char* __restrict__ groups,
                                              ColdTab ct, int ndev, const int* __restrict__ gsel,
                                              const float* __restrict__ gw, const float* __restrict__ gsh,
                                              MoeJob* __restrict__ jobs, float* __restrict__ jw, int* __restrict__ njobs,
                                              unsigned* __restrict__ coldcnt, int* __restrict__ jrow = nullptr) {
    __shared__ int isel[MAXM][TOPK];
    __shared__ float wsel[MAXM][TOPK];
    __shared__ int hitm[NE];
    __shared__ float shw[MAXM];
    __shared__ int shg[NGRP];
    __shared__ int wtot[16];
    constexpr int KSL = MAXM + 4;  // padded: thread e reads row e (9-word stride, conflict-free)
    __shared__ __align__(4) signed char ksel[NE][KSL];  // ksel[expert][row] = its top-k slot or -1
    const int e = threadIdx.x, warp = e >> 5, lane = e & 31;
    int gm[NGRP];
#pragma unroll
    for (int g = 0; g < NGRP; g++) gm[g] = gmap[e * NGRP + g];
    hitm[e] = 0;
    for (int i = 0; i < KSL / 4; i++) reinterpret_cast<int*>(ksel[e])[i] = -1;
    if (e < m) shw[e] = gsh[e];
    if (e >= 32 && e < 32 + NGRP) shg[e - 32] = gmap[NE * NGRP + e - 32];
    __syncthreads();
    if (e < m * TOPK) {
        const int tt = e / TOPK, k = e % TOPK;
        const int id = gsel[tt * TOPK + k];
        isel[tt][k] = id;
        wsel[tt][k] = gw[tt * TOPK + k];
        hitm[id] = 1;
        ksel[id][tt] = (signed char)k;
    }
    __syncthreads();
    int c = 0;
    if (hitm[e])
#pragma unroll
        for (int g = 0; g < NGRP; g++) c += gm[g] >= 0;
    int inc = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int n = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += n;
    }
    if (lane == 31) wtot[warp] = inc;
    __syncthreads();
    int base = 0, total = 0;
    for (int w = 0; w < 16; w++) { if (w < warp) base += wtot[w]; total += wtot[w]; }
    auto emit = [&](int ex, int j, bool shared) {
        for (int g = 0; g < NGRP; g++) {
            const int li = shared ? shg[g] : gm[g];
            if (li < 0) continue;
            const char* b = grp_ptr(groups, li, ndev, ct, coldcnt);
            MoeJob jb;
            jb.gq = reinterpret_cast<const uint4*>(b);
            jb.gs = reinterpret_cast<const __half*>(b + GG_Q);
            jb.dq = reinterpret_cast<const uint4*>(b + GG_Q + GG_S);
            jb.ds = reinterpret_cast<const __half*>(b + GG_Q + GG_S + GD_Q);
            jobs[j] = jb;
            int nr = 0;  // grouped MoE (jrow): the rows that picked the expert, in row order, weights compacted
            for (int tt = 0; tt < m; tt++) {
                float w = 0.f;
                bool hit = shared;
                if (shared) w = shw[tt];
                else {
                    const int k = ksel[ex][tt];
                    if (k >= 0) { w = wsel[tt][k]; hit = true; }
                }
                if (!jrow) jw[j * MAXM + tt] = w;
                else if (hit) { jrow[j * (MAXM + 1) + 1 + nr] = tt; jw[j * MAXM + nr] = w; nr++; }
            }
            if (jrow) jrow[j * (MAXM + 1)] = nr;
            j++;
        }
    };
    if (c) emit(e, base + inc - c, false);
    if (e == 0) {
        int ns = 0;
        for (int g = 0; g < NGRP; g++) ns += shg[g] >= 0;
        emit(NE, total, true);
        *njobs = total + ns;
        atomicAdd(&g_jobstat[m][0], (unsigned long long)(total + ns));
        atomicAdd(&g_jobstat[m][1], 1ull);
    }
}

// Router v5 (decode, m <= 8, INT8 router): router GEMV + top-10 + job list in one kernel (was 3 launches:
// gemv_w8 8.0 + k_topk_rows 5.5 + k_jobs 4.8 us per layer at m = 4). GEMV: grid (9 tile quads, 5 K splits) x 16
// warps, warp = (16-row tile, 128-k group); the 4 groups of a split are summed by one warp in gemv_w8_kernel's
// order (its 5 warps x 4 groups, acc += c * sg, then warps 1..4 added in order): logits bit-identical to gemv_w8.
// The last block to finish adds the 5 splits, takes the top-10 of each row with one warp per row (lowest index on
// ties, as k_topk_rows) and builds the jobs (k_jobs).
constexpr int RF_TILES = 33, RF_SPLIT = 5;
__global__ void __launch_bounds__(512) k_route_f(const uint4* __restrict__ q, const bf16* __restrict__ s,
                                                 const bf16* __restrict__ x, int m, float* __restrict__ part,
                                                 unsigned* __restrict__ cnt, float* __restrict__ lg,
                                                 int* __restrict__ gsel, float* __restrict__ gw, float* __restrict__ gsh,
                                                 int* __restrict__ stats, const int* __restrict__ gmap,
                                                 const char* __restrict__ groups, ColdTab ct, int ndev,
                                                 MoeJob* __restrict__ jobs, float* __restrict__ jw,
                                                 int* __restrict__ njobs, unsigned* __restrict__ coldcnt) {
    constexpr int K = H, groupsK = K >> 7;
    __shared__ float cs[16][32][4];
    __shared__ float sgs[16][32][2];
    __shared__ unsigned s_last;
    const int e = threadIdx.x, warp = e >> 5, lane = e & 31;
    const int g = lane >> 2, t = lane & 3;
    {
        const int tile = blockIdx.x * 4 + (warp >> 2), gi = blockIdx.y * 4 + (warp & 3);
        if (tile < RF_TILES) {
            const uint4* qp = q + (size_t)tile * (K >> 5) * 32 + lane;
            const bf16* sp = s + (size_t)tile * groupsK * 16;
            const bool act = g < m;
            const bf16* xr = x + (size_t)(act ? g : 0) * H + 2 * t;
            uint4 w[4];
#pragma unroll
            for (int i = 0; i < 4; i++) w[i] = __ldg(qp + (size_t)(gi * 4 + i) * 32);
            float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const uint32_t ws[4] = {w[i].x, w[i].y, w[i].z, w[i].w};
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int k0 = gi * 128 + (i * 2 + h) * 16;
                    uint32_t a[4];
                    a_from_w8(ws[h * 2], ws[h * 2 + 1], a);
                    uint32_t b0 = 0u, b1 = 0u;
                    if (act) {
                        const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(xr + k0));
                        const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(xr + k0 + 8));
                        b0 = f2h2(f0.x, f0.y);
                        b1 = f2h2(f1.x, f1.y);
                    }
                    mma16(c, a, b0, b1);
                }
            }
#pragma unroll
            for (int i = 0; i < 4; i++) cs[warp][lane][i] = c[i];
            sgs[warp][lane][0] = __bfloat162float(sp[gi * 16 + g]);
            sgs[warp][lane][1] = __bfloat162float(sp[gi * 16 + g + 8]);
        }
        __syncthreads();
        if (tile < RF_TILES && (warp & 3) == 0) {
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float sg = sgs[warp + j][lane][0], sg8 = sgs[warp + j][lane][1];
                acc[0] += cs[warp + j][lane][0] * sg;
                acc[1] += cs[warp + j][lane][1] * sg;
                acc[2] += cs[warp + j][lane][2] * sg8;
                acc[3] += cs[warp + j][lane][3] * sg8;
            }
            *reinterpret_cast<float4*>(part + (((size_t)blockIdx.y * RF_TILES + tile) * 32 + lane) * 4) =
                make_float4(acc[0], acc[1], acc[2], acc[3]);
        }
    }
    __syncthreads();
    if (e == 0) {
        __threadfence();
        s_last = atomicAdd(cnt, 1u) == gridDim.x * gridDim.y - 1;
    }
    __syncthreads();
    if (!s_last) return;
    // ---- last block: logits, top-10 per row, jobs
    __shared__ float lgs[8][528];
    __shared__ float wsel[8][TOPK];
    __shared__ int hitm[NE];
    __shared__ float shw[8];
    __shared__ int shg[NGRP];
    __shared__ int wtot[16];
    __shared__ __align__(8) signed char ksel[NE][8];
    int gm[NGRP];
#pragma unroll
    for (int gg = 0; gg < NGRP; gg++) gm[gg] = gmap[e * NGRP + gg];
    hitm[e] = 0;
    *reinterpret_cast<unsigned long long*>(ksel[e]) = ~0ull;
    if (e >= 32 && e < 32 + NGRP) shg[e - 32] = gmap[NE * NGRP + e - 32];
    constexpr int RF_IT = (RF_TILES * 32 + 511) / 512;
    float4 pv[RF_IT][RF_SPLIT];  // all partial loads in flight at once
#pragma unroll
    for (int u = 0; u < RF_IT; u++) {
        const int it = e + u * 512;
        if (it < RF_TILES * 32)
#pragma unroll
            for (int ks = 0; ks < RF_SPLIT; ks++)
                pv[u][ks] = __ldcg(reinterpret_cast<const float4*>(part + ((size_t)ks * RF_TILES * 32 + it) * 4));
    }
#pragma unroll
    for (int u = 0; u < RF_IT; u++) {
        const int it = e + u * 512;
        if (it >= RF_TILES * 32) continue;
        const int tile = it >> 5, ln = it & 31;
        float4 a = pv[u][0];
#pragma unroll
        for (int ks = 1; ks < RF_SPLIT; ks++) {
            const float4 b = pv[u][ks];
            a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
        }
        const int gg = ln >> 2, tt = ln & 3, n0 = 2 * tt, n1 = 2 * tt + 1, r0 = tile * 16 + gg;
        if (n0 < m) { lgs[n0][r0] = a.x; lgs[n0][r0 + 8] = a.z; lg[(size_t)n0 * 528 + r0] = a.x; lg[(size_t)n0 * 528 + r0 + 8] = a.z; }
        if (n1 < m) { lgs[n1][r0] = a.y; lgs[n1][r0 + 8] = a.w; lg[(size_t)n1 * 528 + r0] = a.y; lg[(size_t)n1 * 528 + r0 + 8] = a.w; }
    }
    __syncthreads();
    if (warp < m) {  // top-10 of row `warp`: lane holds experts lane + 32 i; 10 rounds of warp max + lowest index
        const int tr = warp;
        unsigned key[16];
#pragma unroll
        for (int i = 0; i < 16; i++) key[i] = fkey(lgs[tr][lane + 32 * i]);
        unsigned lm = key[0];
        int li = 0;
#pragma unroll
        for (int i = 1; i < 16; i++) if (key[i] > lm) { lm = key[i]; li = i; }
        float myv = 0.f, top = 0.f;
        int myi = 0;
        for (int k = 0; k < TOPK; k++) {
            const unsigned M = __reduce_max_sync(0xffffffffu, lm);
            const unsigned I = __reduce_min_sync(0xffffffffu, lm == M ? (unsigned)(lane + 32 * li) : 0xffffffffu);
            if (k == 0) top = kfloat(M);
            if (lane == k) { myv = kfloat(M); myi = (int)I; }
            if (lane == (int)(I & 31)) {
#pragma unroll
                for (int i = 0; i < 16; i++) if (i == li) key[i] = 0u;
                lm = key[0]; li = 0;
#pragma unroll
                for (int i = 1; i < 16; i++) if (key[i] > lm) { lm = key[i]; li = i; }
            }
        }
        const float v = lane < TOPK ? __expf(myv - top) : 0.f;
        float tot = v;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
        if (lane < TOPK) {
            const float wgt = v / tot;
            gsel[tr * TOPK + lane] = myi;
            gw[tr * TOPK + lane] = wgt;
            wsel[tr][lane] = wgt;
            hitm[myi] = 1;
            ksel[myi][tr] = (signed char)lane;
            if (stats) atomicAdd(stats + myi, 1);
        }
        if (lane == 0) {
            const float sh = sigm(0.f + lgs[tr][NE]);
            gsh[tr] = sh;
            shw[tr] = sh;
        }
    }
    __syncthreads();
    int c = 0;
    if (hitm[e])
#pragma unroll
        for (int gg = 0; gg < NGRP; gg++) c += gm[gg] >= 0;
    int inc = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int n = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += n;
    }
    if (lane == 31) wtot[warp] = inc;
    __syncthreads();
    int base = 0, total = 0;
    for (int w2 = 0; w2 < 16; w2++) { if (w2 < warp) base += wtot[w2]; total += wtot[w2]; }
    auto emit = [&](int ex, int j, bool shared) {
        for (int gg = 0; gg < NGRP; gg++) {
            const int li = shared ? shg[gg] : gm[gg];
            if (li < 0) continue;
            const char* b = grp_ptr(groups, li, ndev, ct, coldcnt);
            MoeJob jb;
            jb.gq = reinterpret_cast<const uint4*>(b);
            jb.gs = reinterpret_cast<const __half*>(b + GG_Q);
            jb.dq = reinterpret_cast<const uint4*>(b + GG_Q + GG_S);
            jb.ds = reinterpret_cast<const __half*>(b + GG_Q + GG_S + GD_Q);
            jobs[j] = jb;
            for (int tt = 0; tt < m; tt++) {
                float w = 0.f;
                if (shared) w = shw[tt];
                else if (ksel[ex][tt] >= 0) w = wsel[tt][ksel[ex][tt]];
                jw[j * MAXM + tt] = w;
            }
            j++;
        }
    };
    if (c) emit(e, base + inc - c, false);
    if (e == 0) {
        int ns = 0;
        for (int gg = 0; gg < NGRP; gg++) ns += shg[gg] >= 0;
        emit(NE, total, true);
        *njobs = total + ns;
        *cnt = 0u;
    }
}

// gate_up of every job: one block per (16-row tile of the group's 256 GLU-interleaved rows, job), 4 warps split
// K = 2560; NT 8-token tiles. Epilogue: act = silu(gate) * up * jw[job][t] -> XH acth[job][t][128]. H16 numerics.
template <int NT>
__global__ void __launch_bounds__(128) k_moe_gu(const MoeJob* __restrict__ jobs, const int* __restrict__ njobs,
                                                const float* __restrict__ jw, const __half* __restrict__ x,
                                                __half* __restrict__ acth, int m) {
    const int job = blockIdx.y;
    if (job >= *njobs) return;
    const MoeJob jb = jobs[job];
    const int r = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarps = 4;
    const int g = lane >> 2, t4 = lane & 3;
    constexpr int groups = H >> 7, chunks = H >> 6;
    const int gb = warp * groups / nwarps, ge = (warp + 1) * groups / nwarps;
    const uint4* qp = jb.gq + (size_t)r * chunks * 32 + lane;
    const __half* sp = jb.gs + (size_t)r * groups * 16;
    const uint4* xr[NT];
#pragma unroll
    for (int j = 0; j < NT; j++) {
        const int tok = 8 * j + g < m ? 8 * j + g : 0;
        xr[j] = reinterpret_cast<const uint4*>(x + (size_t)tok * H + t4 * 16);
    }
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    // all of the warp's weights and scales first (5 groups): one memory round trip, which matters for cold groups
    // read over PCIe from host memory
    constexpr int GW = groups / 4;
    uint4 wv[GW][2];
    float sgv[GW][2];
#pragma unroll
    for (int a2 = 0; a2 < GW; a2++) {
        const int gi = gb + a2;
        wv[a2][0] = __ldg(qp + (size_t)(2 * gi) * 32);
        wv[a2][1] = __ldg(qp + (size_t)(2 * gi + 1) * 32);
        sgv[a2][0] = __half2float(sp[gi * 16 + g]) * kXhScale;
        sgv[a2][1] = __half2float(sp[gi * 16 + g + 8]) * kXhScale;
    }
#pragma unroll
    for (int a2 = 0; a2 < GW; a2++) {
        const int gi = gb + a2;
        const float sg = sgv[a2][0], sg8 = sgv[a2][1];
        const uint4 w0 = wv[a2][0], w1 = wv[a2][1];
#pragma unroll
        for (int c = 0; c < 2; c++) {
            const int ch = 2 * gi + c;
            const uint4 wc = c ? w1 : w0;
            uint32_t hacc[NT][2];
            uint4 b0[NT], b1[NT];
#pragma unroll
            for (int j = 0; j < NT; j++) { hacc[j][0] = hacc[j][1] = 0u; b0[j] = xr[j][(size_t)ch * 8]; b1[j] = xr[j][(size_t)ch * 8 + 1]; }
            const uint32_t words[4] = {wc.x, wc.y, wc.z, wc.w};
#pragma unroll
            for (int q = 0; q < 4; q++) {
                uint32_t a[4];
                xh_dequant(words[q], a);
#pragma unroll
                for (int j = 0; j < NT; j++) {
                    const uint4& bb = (q >> 1) ? b1[j] : b0[j];
                    xh_mma(hacc[j], a, (q & 1) ? bb.z : bb.x, (q & 1) ? bb.w : bb.y);
                }
            }
#pragma unroll
            for (int j = 0; j < NT; j++) xh_flush(acc[j], hacc[j], sg, sg8);
        }
    }
    __shared__ float red[4][NT][32][4];
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
        for (int i = 0; i < 4; i++) red[warp][j][lane][i] = acc[j][i];
    __syncthreads();
    if (warp != 0) return;
    __half* ab = acth + (size_t)job * MAXM * GRP;
    const int col = r * 8 + g;
#pragma unroll
    for (int j = 0; j < NT; j++) {
        for (int w2 = 1; w2 < nwarps; w2++)
#pragma unroll
            for (int i = 0; i < 4; i++) acc[j][i] += red[w2][j][lane][i];
        const int n0 = 8 * j + 2 * t4, n1 = n0 + 1;
        if (n0 < m) store_xh(ab + (size_t)n0 * GRP, col, silu(acc[j][0]) * acc[j][2] * jw[job * MAXM + n0]);
        if (n1 < m) store_xh(ab + (size_t)n1 * GRP, col, silu(acc[j][1]) * acc[j][3] * jw[job * MAXM + n1]);
    }
}
// down of all jobs as ONE GEMV: N = 2560, K = 128 x njobs (job j's K slice = its 128 activations, already weighted).
// One block per 16-row tile, 8 warps split the jobs; partial sums -> part[t][2560] directly.
// debug FLASH_MOE_DN_1W=1: warp 0 sums every job in list order (a row's sum independent of the other rows' jobs)
__device__ int g_moe_dn_1w = 0;
template <int NT>
__global__ void __launch_bounds__(256) k_moe_dn(const MoeJob* __restrict__ jobs, const int* __restrict__ njobs,
                                                const __half* __restrict__ acth, float* __restrict__ part, int m) {
    const int r = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarps = 8;
    const int g = lane >> 2, t4 = lane & 3;
    const int nj = *njobs;
    const int nwj = g_moe_dn_1w ? 1 : nwarps;
    const int jb0 = warp < nwj ? warp * nj / nwj : 0, jb1 = warp < nwj ? (warp + 1) * nj / nwj : 0;
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    // software pipeline over the jobs: the next job's weights / scales are loaded while this one computes
    uint4 nw0 = make_uint4(0, 0, 0, 0), nw1 = nw0;
    float ns0 = 0.f, ns8 = 0.f;
    if (jb0 < jb1) {
        const MoeJob jb = jobs[jb0];
        const uint4* qp = jb.dq + (size_t)r * 2 * 32 + lane;
        nw0 = __ldg(qp); nw1 = __ldg(qp + 32);
        ns0 = __half2float(jb.ds[r * 16 + g]) * kXhScale; ns8 = __half2float(jb.ds[r * 16 + g + 8]) * kXhScale;
    }
    for (int jj = jb0; jj < jb1; jj++) {
        const uint4 w0 = nw0, w1 = nw1;
        const float sg = ns0, sg8 = ns8;
        if (jj + 1 < jb1) {
            const MoeJob jn = jobs[jj + 1];
            const uint4* qn = jn.dq + (size_t)r * 2 * 32 + lane;
            nw0 = __ldg(qn); nw1 = __ldg(qn + 32);
            ns0 = __half2float(jn.ds[r * 16 + g]) * kXhScale; ns8 = __half2float(jn.ds[r * 16 + g + 8]) * kXhScale;
        }
#pragma unroll
        for (int c = 0; c < 2; c++) {
            const uint4 wc = c ? w1 : w0;
            uint32_t hacc[NT][2];
            uint4 b0[NT], b1[NT];
#pragma unroll
            for (int j = 0; j < NT; j++) {
                const int tok = 8 * j + g < m ? 8 * j + g : 0;
                const uint4* xr = reinterpret_cast<const uint4*>(acth + ((size_t)jj * MAXM + tok) * GRP + t4 * 16);
                hacc[j][0] = hacc[j][1] = 0u;
                b0[j] = xr[c * 8];
                b1[j] = xr[c * 8 + 1];
            }
            const uint32_t words[4] = {wc.x, wc.y, wc.z, wc.w};
#pragma unroll
            for (int q = 0; q < 4; q++) {
                uint32_t a[4];
                xh_dequant(words[q], a);
#pragma unroll
                for (int j = 0; j < NT; j++) {
                    const uint4& bb = (q >> 1) ? b1[j] : b0[j];
                    xh_mma(hacc[j], a, (q & 1) ? bb.z : bb.x, (q & 1) ? bb.w : bb.y);
                }
            }
#pragma unroll
            for (int j = 0; j < NT; j++) xh_flush(acc[j], hacc[j], sg, sg8);
        }
    }
    __shared__ float red[8][NT][32][4];
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
        for (int i = 0; i < 4; i++) red[warp][j][lane][i] = acc[j][i];
    __syncthreads();
    if (warp != 0) return;
    const int row0 = r * 16;
#pragma unroll
    for (int j = 0; j < NT; j++) {
        for (int w2 = 1; w2 < nwarps; w2++)
#pragma unroll
            for (int i = 0; i < 4; i++) acc[j][i] += red[w2][j][lane][i];
        const int n0 = 8 * j + 2 * t4, n1 = n0 + 1;
        if (n0 < m) { part[(size_t)n0 * H + row0 + g] = acc[j][0]; part[(size_t)n0 * H + row0 + g + 8] = acc[j][2]; }
        if (n1 < m) { part[(size_t)n1 * H + row0 + g] = acc[j][1]; part[(size_t)n1 * H + row0 + g + 8] = acc[j][3]; }
    }
}

// Grouped MoE for batched steps (m > 8): a job computes only the rows that picked its expert (jrow[job] = {count,
// rows...}), in 8-row tiles (up to 4, the tiles past the count skipped), instead of all m rows with zero weights.
// acth[job][slot][128] holds the job's rows compacted (slot = index in its row list). Per row the arithmetic is the
// same as k_moe_gu / k_moe_dn (same k order, same per-chunk fp16 sums and fp32 flushes, jobs in list order; a skipped
// job added +0 there), so with FLASH_MOE_DN_1W=1 a row's result does not depend on the other rows.
__global__ void __launch_bounds__(128) k_moe_gu_g(const MoeJob* __restrict__ jobs, const int* __restrict__ njobs,
                                                  const int* __restrict__ jrow, const float* __restrict__ jw,
                                                  const __half* __restrict__ x, __half* __restrict__ acth) {
    constexpr int NT = 4;
    const int nj = *njobs;
    for (int job = blockIdx.y; job < nj; job += gridDim.y) {  // grid y below the job bound: blocks loop
    const MoeJob jb = jobs[job];
    const int* rl = jrow + job * (MAXM + 1);
    const int cnt = rl[0], nt = (cnt + 7) >> 3;
    const int r = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarps = 4;
    const int g = lane >> 2, t4 = lane & 3;
    constexpr int groups = H >> 7, chunks = H >> 6;
    const int gb = warp * groups / nwarps;
    const uint4* qp = jb.gq + (size_t)r * chunks * 32 + lane;
    const __half* sp = jb.gs + (size_t)r * groups * 16;
    const uint4* xr[NT];
#pragma unroll
    for (int j = 0; j < NT; j++) {
        const int sl = 8 * j + g;
        const int tok = sl < cnt ? rl[1 + sl] : rl[1];
        xr[j] = reinterpret_cast<const uint4*>(x + (size_t)tok * H + t4 * 16);
    }
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    constexpr int GW = groups / 4;
    uint4 wv[GW][2];
    float sgv[GW][2];
#pragma unroll
    for (int a2 = 0; a2 < GW; a2++) {
        const int gi = gb + a2;
        wv[a2][0] = __ldg(qp + (size_t)(2 * gi) * 32);
        wv[a2][1] = __ldg(qp + (size_t)(2 * gi + 1) * 32);
        sgv[a2][0] = __half2float(sp[gi * 16 + g]) * kXhScale;
        sgv[a2][1] = __half2float(sp[gi * 16 + g + 8]) * kXhScale;
    }
#pragma unroll
    for (int a2 = 0; a2 < GW; a2++) {
        const int gi = gb + a2;
        const float sg = sgv[a2][0], sg8 = sgv[a2][1];
        const uint4 w0 = wv[a2][0], w1 = wv[a2][1];
#pragma unroll
        for (int c = 0; c < 2; c++) {
            const int ch = 2 * gi + c;
            const uint4 wc = c ? w1 : w0;
            uint32_t hacc[NT][2];
            uint4 b0[NT], b1[NT];
#pragma unroll
            for (int j = 0; j < NT; j++) {
                hacc[j][0] = hacc[j][1] = 0u;
                if (j < nt) { b0[j] = xr[j][(size_t)ch * 8]; b1[j] = xr[j][(size_t)ch * 8 + 1]; }
            }
            const uint32_t words[4] = {wc.x, wc.y, wc.z, wc.w};
#pragma unroll
            for (int q = 0; q < 4; q++) {
                uint32_t a[4];
                xh_dequant(words[q], a);
#pragma unroll
                for (int j = 0; j < NT; j++) {
                    if (j < nt) {
                        const uint4& bb = (q >> 1) ? b1[j] : b0[j];
                        xh_mma(hacc[j], a, (q & 1) ? bb.z : bb.x, (q & 1) ? bb.w : bb.y);
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < NT; j++) if (j < nt) xh_flush(acc[j], hacc[j], sg, sg8);
        }
    }
    __shared__ float red[4][NT][32][4];
#pragma unroll
    for (int j = 0; j < NT; j++)
        if (j < nt)
#pragma unroll
            for (int i = 0; i < 4; i++) red[warp][j][lane][i] = acc[j][i];
    __syncthreads();
    if (warp == 0) {
    __half* ab = acth + (size_t)job * MAXM * GRP;
    const int col = r * 8 + g;
#pragma unroll
    for (int j = 0; j < NT; j++) {
        if (j >= nt) break;
        for (int w2 = 1; w2 < nwarps; w2++)
#pragma unroll
            for (int i = 0; i < 4; i++) acc[j][i] += red[w2][j][lane][i];
        const int n0 = 8 * j + 2 * t4, n1 = n0 + 1;
        if (n0 < cnt) store_xh(ab + (size_t)n0 * GRP, col, silu(acc[j][0]) * acc[j][2] * jw[job * MAXM + n0]);
        if (n1 < cnt) store_xh(ab + (size_t)n1 * GRP, col, silu(acc[j][1]) * acc[j][3] * jw[job * MAXM + n1]);
    }
    }
    __syncthreads();
    }
}
// down: one block per 16 output rows, 8 warps split the job list (as k_moe_dn); each job's tiles cover only its rows,
// the fp32 sums live in shared memory per (warp, batch row) and are added over the warps in order at the end
__global__ void __launch_bounds__(256) k_moe_dn_g(const MoeJob* __restrict__ jobs, const int* __restrict__ njobs,
                                                  const int* __restrict__ jrow, const __half* __restrict__ acth,
                                                  float* __restrict__ part, int m) {
    __shared__ float sacc[8][MAXM][16];
    const int r = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarps = 8;
    const int g = lane >> 2, t4 = lane & 3;
    for (int i = threadIdx.x; i < 8 * MAXM * 16; i += 256) (&sacc[0][0][0])[i] = 0.f;
    __syncthreads();
    const int nj = *njobs;
    const int nwj = g_moe_dn_1w ? 1 : nwarps;
    const int jb0 = warp < nwj ? warp * nj / nwj : 0, jb1 = warp < nwj ? (warp + 1) * nj / nwj : 0;
    float (*sa)[16] = sacc[warp];
    // software pipeline: the next job's weights, scales, row count, first-tile rows and activations are loaded while
    // this one computes (the activation slot g is read even past the count: those MMA columns are discarded)
    struct Pre { uint4 w0, w1, b[4]; float s0, s8; int cnt, r0, r1; };
    auto load = [&](int jj, Pre& p) {
        const MoeJob jb = jobs[jj];
        const uint4* qp = jb.dq + (size_t)r * 2 * 32 + lane;
        p.w0 = __ldg(qp); p.w1 = __ldg(qp + 32);
        p.s0 = __half2float(jb.ds[r * 16 + g]) * kXhScale; p.s8 = __half2float(jb.ds[r * 16 + g + 8]) * kXhScale;
        const int* rl = jrow + jj * (MAXM + 1);
        p.cnt = rl[0]; p.r0 = rl[1 + 2 * t4]; p.r1 = rl[2 + 2 * t4];
        const uint4* xr = reinterpret_cast<const uint4*>(acth + ((size_t)jj * MAXM + g) * GRP + t4 * 16);
        p.b[0] = xr[0]; p.b[1] = xr[1]; p.b[2] = xr[8]; p.b[3] = xr[9];
    };
    auto tile = [&](const uint4& w0, const uint4& w1, float sg, float sg8, const uint4* bb4, int cnt, int t0, int row0, int row1) {
        const int n0 = t0 + 2 * t4, n1 = n0 + 1;
        if (n0 >= cnt) row0 = -1;
        if (n1 >= cnt) row1 = -1;
        float acc[4];
        acc[0] = row0 >= 0 ? sa[row0][g] : 0.f;
        acc[2] = row0 >= 0 ? sa[row0][g + 8] : 0.f;
        acc[1] = row1 >= 0 ? sa[row1][g] : 0.f;
        acc[3] = row1 >= 0 ? sa[row1][g + 8] : 0.f;
#pragma unroll
        for (int c = 0; c < 2; c++) {
            const uint4 wc = c ? w1 : w0;
            uint32_t hacc[2] = {0u, 0u};
            const uint4 b0 = bb4[c * 2], b1 = bb4[c * 2 + 1];
            const uint32_t words[4] = {wc.x, wc.y, wc.z, wc.w};
#pragma unroll
            for (int q = 0; q < 4; q++) {
                uint32_t a[4];
                xh_dequant(words[q], a);
                const uint4& bb = (q >> 1) ? b1 : b0;
                xh_mma(hacc, a, (q & 1) ? bb.z : bb.x, (q & 1) ? bb.w : bb.y);
            }
            xh_flush(acc, hacc, sg, sg8);
        }
        if (row0 >= 0) { sa[row0][g] = acc[0]; sa[row0][g + 8] = acc[2]; }
        if (row1 >= 0) { sa[row1][g] = acc[1]; sa[row1][g + 8] = acc[3]; }
        __syncwarp();
    };
    Pre cur, nxt;
    if (jb0 < jb1) load(jb0, nxt);
    for (int jj = jb0; jj < jb1; jj++) {
        cur = nxt;
        if (jj + 1 < jb1) load(jj + 1, nxt);
        tile(cur.w0, cur.w1, cur.s0, cur.s8, cur.b, cur.cnt, 0, cur.r0, cur.r1);
        for (int t0 = 8; t0 < cur.cnt; t0 += 8) {  // further tiles (shared expert, popular experts): direct loads
            const int* rl = jrow + jj * (MAXM + 1);
            const int sl = t0 + g < cur.cnt ? t0 + g : 0;
            const uint4* xr = reinterpret_cast<const uint4*>(acth + ((size_t)jj * MAXM + sl) * GRP + t4 * 16);
            uint4 bb4[4] = {xr[0], xr[1], xr[8], xr[9]};
            const int n0 = t0 + 2 * t4;
            tile(cur.w0, cur.w1, cur.s0, cur.s8, bb4, cur.cnt, t0, n0 < cur.cnt ? rl[1 + n0] : -1, n0 + 1 < cur.cnt ? rl[2 + n0] : -1);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < m * 16; i += 256) {
        const int t = i >> 4, c = i & 15;
        float v = sacc[0][t][c];
        for (int w2 = 1; w2 < nwarps; w2++) v += sacc[w2][t][c];
        part[(size_t)t * H + r * 16 + c] = v;
    }
}

// ------------------------------------------------------------------ device-side speculative step
// n-gram embedding rows of the verify rows (hash as HF Qwen4ExpTextNGramEmbedding, table in mapped host memory)
__device__ __forceinline__ void ngram_gather_q8(const uint8_t* q, int sub, bf16* __restrict__ dst) {
    __half_raw hr;
    hr.x = (unsigned short)(q[PLE_DIM] | (q[PLE_DIM + 1] << 8));
    const float sc = __half2float(__half(hr));
#pragma unroll
    for (int e = 0; e < 10; e++) dst[sub * 10 + e] = __float2bfloat16((float)(int8_t)q[sub * 10 + e] * sc);
}
__device__ __forceinline__ void ngram_gather(const NgramDev& ng, long long row, int sub, bf16* __restrict__ dst) {
    if (ng.is8) {  // INT8 table: 160 int8 + fp16 scale per row
        const uint8_t* q = ng.q8[row / NGRAM_ROWS] + (size_t)(row % NGRAM_ROWS) * NGRAM_Q8_ROW;
        __half_raw hr;
        hr.x = (unsigned short)(q[PLE_DIM] | (q[PLE_DIM + 1] << 8));
        const float sc = __half2float(__half(hr));
#pragma unroll
        for (int e = 0; e < 10; e++) dst[sub * 10 + e] = __float2bfloat16((float)(int8_t)q[sub * 10 + e] * sc);
        return;
    }
    const bf16* src = ng.shard[row / NGRAM_ROWS] + (size_t)(row % NGRAM_ROWS) * PLE_DIM;
#pragma unroll
    for (int e = 0; e < 10; e++) dst[sub * 10 + e] = src[sub * 10 + e];
}
__global__ void k_ngram(const int* __restrict__ vtok, const int* __restrict__ hist2, NgramDev ng, bf16* __restrict__ out) {
    const int j = blockIdx.x, tid = threadIdx.x;
    const long long t0 = vtok[j];
    const long long t1 = j >= 1 ? vtok[j - 1] : hist2[1];
    const long long t2r = j >= 2 ? vtok[j - 2] : (j == 1 ? hist2[1] : hist2[0]);
    const long long s1 = t1, s2 = t1 == EOS ? (long long)EOS : t2r;
    auto mul = [](long long a, long long b) { return (long long)((unsigned long long)a * (unsigned long long)b); };
    const long long big = mul(t0, ng.mult[0]) ^ mul(s1, ng.mult[1]);
    const long long tri = big ^ mul(s2, ng.mult[2]);
    const int hd = tid >> 4, sub = tid & 15;
    const long long mixed = hd < 8 ? big : tri;
    long long r = mixed % ng.vsz[hd];
    if (r < 0) r += ng.vsz[hd];
    const long long row = r + ng.voff[hd];
    if (ng.phase == 1) { if (sub == 0) ng.req[j * PLE_HEADS + hd] = row; return; }
    if (ng.phase == 2) { ngram_gather_q8(ng.stage + (size_t)(j * PLE_HEADS + hd) * NGRAM_Q8_ROW, sub, out + (size_t)j * H + hd * PLE_DIM); return; }
    ngram_gather(ng, row, sub, out + (size_t)j * H + hd * PLE_DIM);
}
// staged n-gram rows: after the row-id pass, announce n rows to the host thread and wait until it has copied them
__global__ void k_ngram_handoff(NgramDev ng, int n) {
    if (threadIdx.x != 0) return;
    __threadfence_system();
    const unsigned s = ng.flags[0] + 1;
    ng.flags[2] = n;
    __threadfence_system();
    qe_st_release_sys(ng.flags, s);
    for (unsigned ws_ = 0; (int)(qe_ld_acquire_sys(ng.flags + 1) - s) < 0;) { __nanosleep(64); QE_WD_POLL(ws_, kWdQsaX, 1, s, qe_ld_acquire_sys(ng.flags + 1)); }
}
__global__ void k_accept(const int* __restrict__ vtok, const int* __restrict__ otok, int k, int* __restrict__ posd,
                         int* __restrict__ mposd, int* __restrict__ acc, int* __restrict__ hist2, int* __restrict__ em) {
    if (threadIdx.x != 0) return;
    int a = 0;
    while (a < k && vtok[a + 1] == otok[a] && otok[a] != EOS && otok[a] != EOS + 2) a++;
    em[0] = a + 1;
    for (int i = 0; i <= a; i++) em[1 + i] = otok[i];
    *mposd = posd[0];
    int h0 = hist2[0], h1 = hist2[1];
    for (int i = 0; i <= a; i++) { h0 = h1; h1 = vtok[i]; }
    hist2[0] = h0;
    hist2[1] = h1;
    posd[0] += a + 1;
    posd[1] = a + 1;
    *acc = a;
}
__global__ void k_draft0(const int* __restrict__ motok, const int* __restrict__ acc, int* __restrict__ dr) {
    if (threadIdx.x == 0) dr[0] = motok[*acc];
}
__global__ void k_setpos(int* __restrict__ mposd, const int* __restrict__ posd, int off) { *mposd = posd[0] + off; }
__global__ void k_set_last_tok(int* __restrict__ toks, int n, const int* __restrict__ src) { toks[n - 1] = *src; }
__global__ void k_copy_row(float* __restrict__ dst, const float* __restrict__ src, const int* __restrict__ rowp, int defrow) {
    const int row = rowp ? *rowp : defrow;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < RW) dst[i] = src[(size_t)row * RW + i];
}
__global__ void k_next(int* __restrict__ vtok, const int* __restrict__ otok, const int* __restrict__ acc,
                       const int* __restrict__ dr, int k, int* __restrict__ em) {
    if (threadIdx.x != 0) return;
    vtok[0] = otok[*acc];
    for (int i = 0; i < k; i++) vtok[1 + i] = dr[i];
}

// ------------------------------------------------------------------ lookup drafts (device side, in the step graph)
// hd: token history (prompt + every emitted token); the emitted tokens of the step (em) are appended, then the
// longest earlier match of the history's suffix (<= 32 tokens) gives the drafts: the tokens that followed it
// region of the text so far (lookup policy): rg[0] inside <tool_call> .. </tool_call>, rg[1] odd number of ``` fences
constexpr int TOK_TOOL_OPEN = 248058, TOK_TOOL_CLOSE = 248059, TOK_FENCE = 71093, TOK_FENCE_SP = 52451;
__host__ __device__ inline void lk_region_tok(int t, int* rg) {
    if (t == TOK_TOOL_OPEN) rg[0] = 1;
    else if (t == TOK_TOOL_CLOSE) rg[0] = 0;
    else if (t == TOK_FENCE || t == TOK_FENCE_SP) rg[1] ^= 1;
}
__global__ void k_hist_add(int* __restrict__ hd, int* __restrict__ hn, const int* __restrict__ em, int* __restrict__ rg) {
    if (threadIdx.x != 0) return;
    const int n = *hn, c = em[0];
    for (int i = 0; i < c; i++) { hd[n + i] = em[1 + i]; lk_region_tok(em[1 + i], rg); }
    *hn = n + c;
}
// candidate end position j < n - 1: match length of hd[.. j] with the suffix hd[.. n - 1] -> max (len, j)
__global__ void k_lk_scan(const int* __restrict__ hd, const int* __restrict__ hn, unsigned long long* __restrict__ best) {
    const int n = *hn;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long key = 0ull;
    if (j < n - 1 && hd[j] == hd[n - 1]) {
        int s = 1;
        while (s < 32 && j - s >= 0 && hd[j - s] == hd[n - 1 - s]) s++;
        key = ((unsigned long long)s << 32) | (unsigned)j;
    }
    // block max, one atomic per block
    for (int o = 16; o > 0; o >>= 1) { const unsigned long long t = __shfl_xor_sync(0xffffffffu, key, o); key = t > key ? t : key; }
    __shared__ unsigned long long wmax[8];
    if ((threadIdx.x & 31) == 0) wmax[threadIdx.x >> 5] = key;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long b = 0ull;
        for (int w = 0; w < (int)(blockDim.x >> 5); w++) b = wmax[w] > b ? wmax[w] : b;
        if (b) atomicMax(best, b);
    }
}
// global answer corpus (SuffixDecoding style, qe_lookup_corpus): earlier answers separated by -1; candidate end j of
// the corpus: match length with the history's suffix (<= 32, never across a separator) -> max (len, j). Grid-stride
// over the filled part (cn), so the cost follows the corpus size
__global__ void k_lk_scan_c(const int* __restrict__ cd, const int* __restrict__ cn, const int* __restrict__ hd,
                            const int* __restrict__ hn, unsigned long long* __restrict__ best) {
    const int n = *hn, nc = *cn;
    if (n < 1) return;
    const int last = hd[n - 1];
    unsigned long long key = 0ull;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < nc - 1; j += gridDim.x * blockDim.x)
        if (cd[j] == last && cd[j + 1] >= 0) {
            int s = 1;
            while (s < 32 && j - s >= 0 && n - 1 - s >= 0 && cd[j - s] == hd[n - 1 - s]) s++;
            const unsigned long long k2 = ((unsigned long long)s << 32) | (unsigned)j;
            key = k2 > key ? k2 : key;
        }
    for (int o = 16; o > 0; o >>= 1) { const unsigned long long t = __shfl_xor_sync(0xffffffffu, key, o); key = t > key ? t : key; }
    __shared__ unsigned long long wmax[8];
    if ((threadIdx.x & 31) == 0) wmax[threadIdx.x >> 5] = key;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long b = 0ull;
        for (int w = 0; w < (int)(blockDim.x >> 5); w++) b = wmax[w] > b ? wmax[w] : b;
        if (b) atomicMax(best, b);
    }
}
// drafts: L tokens after the match: lmax when the match has >= hi tokens, llo when >= mmin, else 0 (an MTP step);
// out (host-mapped): [L, drafts...]; lk (device): the same for the next step's verify rows. best is reset.
// cd (corpus, may be null): its match (bestc) is taken when strictly longer than the history's and >= cmin tokens
// (another request's text: a short match there rarely continues); the drafts stop at the answer's end
__global__ void k_lk_emit(const int* __restrict__ hd, const int* __restrict__ hn, unsigned long long* __restrict__ best,
                          int mmin, int llo, int hi, int lmax, int* __restrict__ lk, int* __restrict__ out,
                          const int* __restrict__ rg, const int* __restrict__ rp, const int* __restrict__ cd = nullptr,
                          unsigned long long* __restrict__ bestc = nullptr, int cmin = 12) {
    if (threadIdx.x != 0) return;
    const unsigned long long b = *best;
    *best = 0ull;
    int s = (int)(b >> 32), j = (int)(b & 0xffffffffu), n = *hn;
    if (cd) {
        const unsigned long long bc = *bestc;
        *bestc = 0ull;
        const int sc = (int)(bc >> 32);
        if (sc > s && sc >= cmin) {  // drafts from the corpus: point hd at it, n - 1 - j = the tokens before the separator
            const int jc = (int)(bc & 0xffffffffu);
            int run = 0;
            while (run < 32 && cd[jc + 1 + run] >= 0) run++;
            hd = cd;
            s = sc;
            j = jc;
            n = jc + 1 + run;
        }
    }
    int L = 0;
    if (rp) {  // region policy rp = {code: nmin, beta x4, lmin, lmax; text: nmin, beta x4, lmin, lmax}
        const int* q = rp + ((rg[0] || rg[1]) ? 0 : 4);
        if (s >= q[0]) L = min(min(max(s * q[1] / 4, q[2]), q[3]), n - 1 - j);
    } else if (s >= mmin) L = min(s >= hi ? lmax : llo, n - 1 - j);
    lk[0] = L;
    out[0] = L;
    out[31] = s;  // raw match length (statistics / policy tuning)
    for (int i = 0; i < L; i++) { lk[1 + i] = hd[j + 1 + i]; out[1 + i] = hd[j + 1 + i]; }
}
// guard: count non-finite values of the lm_head input (m rows of h values) into a host-mapped counter (blocks that saw
// one) and flag the rows (host-mapped, row index clamped to rows - 1): which sequence of a batch to fail
__global__ void k_nonfinite(const bf16* __restrict__ x, int m, int h, unsigned* __restrict__ cnt, unsigned* __restrict__ rows,
                            int nrows) {
    int bad = 0;
    const int n = m * h;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        if (!isfinite(__bfloat162float(x[i]))) {
            bad = 1;
            if (rows) rows[min(i / h, nrows - 1)] = 1u;
        }
    if (__syncthreads_or(bad) && threadIdx.x == 0) atomicAdd(cnt, 1u);
}
// MTP draft chain after a lookup step only when the next step is an MTP step (no lookup drafts)
__global__ void k_lk_cond(cudaGraphConditionalHandle h, const int* __restrict__ lk) {
    if (threadIdx.x == 0) cudaGraphSetConditional(h, lk[0] == 0 ? 1u : 0u);
}
// next verify rows = [last emitted token, the L lookup drafts]
__global__ void k_vtok_lk(int* __restrict__ vtok, const int* __restrict__ lk, int L) {
    const int i = threadIdx.x;
    if (i < L) vtok[1 + i] = lk[1 + i];
}
// ------------------------------------------------------------------ prefill (C rows per chunk)
// HC down for C rows: k_hcd with the block's weights held in registers across 32-row token groups
__global__ void __launch_bounds__(256) k_hcd_pf(const uint4* __restrict__ q, const bf16* __restrict__ s, const float* __restrict__ R,
                                                float* __restrict__ u, float* __restrict__ ss, int C) {
    constexpr int NT = 4;
    const int r = blockIdx.x, br = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t4 = lane & 3;
    constexpr int K = RW, groups = K >> 7;
    const int gb = br * 20 + warp * 20 / 8, ge = br * 20 + (warp + 1) * 20 / 8;
    const uint4* qp = q + (size_t)r * (K >> 5) * 32 + lane;
    const bf16* sp = s + (size_t)r * groups * 16;
    uint4 w[3][4];
    float sgv[3], sg8v[3];
#pragma unroll
    for (int a = 0; a < 3; a++)
        if (gb + a < ge) {
#pragma unroll
            for (int i = 0; i < 4; i++) w[a][i] = __ldg(qp + (size_t)((gb + a) * 4 + i) * 32);
            sgv[a] = __bfloat162float(sp[(gb + a) * 16 + g]);
            sg8v[a] = __bfloat162float(sp[(gb + a) * 16 + g + 8]);
        }
    __shared__ float red[8][NT][32][4];
    for (int t0 = 0; t0 < C; t0 += 8 * NT) {
        const int m = min(8 * NT, C - t0);
        bool act[NT];
        const float* xr[NT];
#pragma unroll
        for (int j = 0; j < NT; j++) {
            act[j] = 8 * j + g < m;
            xr[j] = R + (size_t)(t0 + (act[j] ? 8 * j + g : 0)) * RW + 2 * t4;
        }
        float acc[NT][4];
#pragma unroll
        for (int j = 0; j < NT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
#pragma unroll
        for (int a = 0; a < 3; a++) {
            const int gi = gb + a;
            if (gi >= ge) break;
            float c[NT][4];
#pragma unroll
            for (int j = 0; j < NT; j++) c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0.f;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const uint32_t ws[4] = {w[a][i].x, w[a][i].y, w[a][i].z, w[a][i].w};
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int k0 = gi * 128 + (i * 2 + h) * 16;
                    uint32_t aa[4];
                    a_from_w8(ws[h * 2], ws[h * 2 + 1], aa);
#pragma unroll
                    for (int j = 0; j < NT; j++) {
                        const float2 x0 = act[j] ? *reinterpret_cast<const float2*>(xr[j] + k0) : make_float2(0.f, 0.f);
                        const float2 x1 = act[j] ? *reinterpret_cast<const float2*>(xr[j] + k0 + 8) : make_float2(0.f, 0.f);
                        mma16(c[j], aa, f2h2(x0.x, x0.y), f2h2(x1.x, x1.y));
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < NT; j++) {
                acc[j][0] += c[j][0] * sgv[a]; acc[j][1] += c[j][1] * sgv[a]; acc[j][2] += c[j][2] * sg8v[a]; acc[j][3] += c[j][3] * sg8v[a];
            }
        }
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
            for (int i = 0; i < 4; i++) red[warp][j][lane][i] = acc[j][i];
        __syncthreads();
        if (warp == 0) {
#pragma unroll
            for (int j = 0; j < NT; j++) {
                for (int w2 = 1; w2 < 8; w2++)
#pragma unroll
                    for (int i = 0; i < 4; i++) acc[j][i] += red[w2][j][lane][i];
                const int n0 = 8 * j + 2 * t4, n1 = n0 + 1, row0 = r * 16;
                float* ub = u + br * HCD_ROWS + (size_t)t0 * HC * HCD_ROWS;
                if (n0 < m) { ub[(size_t)n0 * HC * HCD_ROWS + row0 + g] = acc[j][0]; ub[(size_t)n0 * HC * HCD_ROWS + row0 + g + 8] = acc[j][2]; }
                if (n1 < m) { ub[(size_t)n1 * HC * HCD_ROWS + row0 + g] = acc[j][1]; ub[(size_t)n1 * HC * HCD_ROWS + row0 + g + 8] = acc[j][3]; }
            }
        }
        __syncthreads();
    }
    if (r == 0) {  // sums of squares of stream br: one warp per row
        for (int t = warp; t < C; t += 8) {
            float q2 = 0.f;
            const float* rr = R + (size_t)t * RW + br * H;
            for (int d = lane; d < H; d += 32) q2 += rr[d] * rr[d];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) q2 += __shfl_xor_sync(0xffffffffu, q2, o);
            if (lane == 0) ss[t * HC + br] = q2;
        }
    }
}
// ---- prefill HC v2: activations staged in shared memory (fp16), K split into slices, partial sums by atomics
// grid (3 tile groups of 7 tiles, 4 streams x 4 K-slices of 640, token groups of 32), 7 warps (one 16-row tile each)
constexpr int HKS = 640;
// deterministic mode (FLASH_DETERMINISTIC=1): float sums by integer atomics on a 2^-30 fixed-point grid
constexpr double FX_SCALE = 1073741824.0;
__device__ __forceinline__ unsigned long long fx_of(float v) { return (unsigned long long)__double2ll_rn((double)v * FX_SCALE); }
__global__ void k_fx_to_f(unsigned long long* __restrict__ ip, float* __restrict__ out, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        out[i] = (float)((double)(long long)ip[i] / FX_SCALE);
        ip[i] = 0ull;
    }
}
// debug: order-independent hash of the selected block set of row blockIdx.x (+ its count)
__global__ void k_dbg_selhash(const int* __restrict__ isel, const int* __restrict__ insel, int stride, unsigned long long* __restrict__ out) {
    const int r = blockIdx.x, n = insel[r];
    unsigned long long h = threadIdx.x == 0 ? (unsigned long long)(n + 7) * 0x9E3779B97F4A7C15ull : 0ull;
    for (int i = threadIdx.x; i < n; i += blockDim.x) h += ((unsigned long long)(isel[(size_t)r * stride + i] + 1) * 0xBF58476D1CE4E5B9ull) ^ 0x94D049BB133111EBull;
    for (int o = 16; o > 0; o >>= 1) h += __shfl_xor_sync(0xffffffffu, h, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, h);
}
__global__ void __launch_bounds__(224) k_hcd_pf2(const uint4* __restrict__ q, const bf16* __restrict__ s, const float* __restrict__ R,
                                                 float* __restrict__ u, int C, float* __restrict__ ss = nullptr,
                                                 unsigned long long* __restrict__ fxu = nullptr,
                                                 unsigned long long* __restrict__ fxss = nullptr) {
    // ss (optional, zeroed): per (row, stream) sum of squares of R, added by the blocks of output group 0
    // fxu / fxss (deterministic mode): the same sums as exact fixed point (k_fx_to_f -> u / ss after)
    __shared__ __half act[32][HKS + 8];
    const int tg = blockIdx.x, br = blockIdx.y >> 2, ks = blockIdx.y & 3, t0 = blockIdx.z * 32;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    const int m = min(32, C - t0);
    const int kbase = br * H + ks * HKS;
    for (int r = warp; r < 32; r += 7) {
        float a = 0.f;
        for (int c4 = lane; c4 < HKS / 4; c4 += 32) {
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (r < m) v = *reinterpret_cast<const float4*>(R + (size_t)(t0 + r) * RW + kbase + c4 * 4);
            *reinterpret_cast<__half2*>(&act[r][c4 * 4]) = __floats2half2_rn(v.x, v.y);
            *reinterpret_cast<__half2*>(&act[r][c4 * 4 + 2]) = __floats2half2_rn(v.z, v.w);
            a += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        if (ss && tg == 0) {
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
            if (lane == 0 && r < m) {
                if (fxss) atomicAdd(fxss + (size_t)(t0 + r) * HC + br, fx_of(a));
                else atomicAdd(ss + (size_t)(t0 + r) * HC + br, a);
            }
        }
    }
    __syncthreads();
    const int T = tg * 7 + warp;
    if (T >= HCD_ROWS / 16) return;
    constexpr int K = RW, groups = K >> 7;
    const uint4* qp = q + (size_t)T * (K >> 5) * 32 + lane;
    const bf16* sp = s + (size_t)T * groups * 16;
    float acc[4][4];
#pragma unroll
    for (int j = 0; j < 4; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    const int g0 = kbase >> 7;
#pragma unroll 1
    for (int gi = g0; gi < g0 + HKS / 128; gi++) {
        uint4 w[4];
#pragma unroll
        for (int i = 0; i < 4; i++) w[i] = __ldg(qp + (size_t)(gi * 4 + i) * 32);
        const float sg = __bfloat162float(sp[gi * 16 + g]), sg8 = __bfloat162float(sp[gi * 16 + g + 8]);
        float c[4][4];
#pragma unroll
        for (int j = 0; j < 4; j++) c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0.f;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const uint32_t ws[4] = {w[i].x, w[i].y, w[i].z, w[i].w};
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int kl = (gi - g0) * 128 + (i * 2 + h) * 16;  // within the slice
                uint32_t aa[4];
                a_from_w8(ws[h * 2], ws[h * 2 + 1], aa);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&act[8 * j + g][kl + 2 * t4]);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&act[8 * j + g][kl + 8 + 2 * t4]);
                    mma16(c[j], aa, b0, b1);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < 4; j++) {
            acc[j][0] += c[j][0] * sg; acc[j][1] += c[j][1] * sg; acc[j][2] += c[j][2] * sg8; acc[j][3] += c[j][3] * sg8;
        }
    }
    const int row0 = T * 16;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int n0 = 8 * j + 2 * t4, n1 = n0 + 1;
        float* ub = u + br * HCD_ROWS;
        if (fxu) {
            unsigned long long* fb = fxu + br * HCD_ROWS;
            if (n0 < m) {
                atomicAdd(fb + (size_t)(t0 + n0) * HC * HCD_ROWS + row0 + g, fx_of(acc[j][0]));
                atomicAdd(fb + (size_t)(t0 + n0) * HC * HCD_ROWS + row0 + g + 8, fx_of(acc[j][2]));
            }
            if (n1 < m) {
                atomicAdd(fb + (size_t)(t0 + n1) * HC * HCD_ROWS + row0 + g, fx_of(acc[j][1]));
                atomicAdd(fb + (size_t)(t0 + n1) * HC * HCD_ROWS + row0 + g + 8, fx_of(acc[j][3]));
            }
            continue;
        }
        if (n0 < m) {
            atomicAdd(ub + (size_t)(t0 + n0) * HC * HCD_ROWS + row0 + g, acc[j][0]);
            atomicAdd(ub + (size_t)(t0 + n0) * HC * HCD_ROWS + row0 + g + 8, acc[j][2]);
        }
        if (n1 < m) {
            atomicAdd(ub + (size_t)(t0 + n1) * HC * HCD_ROWS + row0 + g, acc[j][1]);
            atomicAdd(ub + (size_t)(t0 + n1) * HC * HCD_ROWS + row0 + g + 8, acc[j][3]);
        }
    }
}
// k_hcd_pf2 with all 21 output tiles in one block of 21 warps (the R slice is staged once instead of 3 times) and the
// per (row, stream) sum of squares of R accumulated during the staging (ss must be zeroed; 4 K quarters add into it).
// grid (16 = 4 streams x 4 K quarters, ceil(C / 32)), 672 threads
__global__ void __launch_bounds__(672) k_hcd_pf3(const uint4* __restrict__ q, const bf16* __restrict__ s, const float* __restrict__ R,
                                                 float* __restrict__ u, float* __restrict__ ss, int C) {
    __shared__ __half act[32][HKS + 8];
    const int br = blockIdx.x >> 2, ks = blockIdx.x & 3, t0 = blockIdx.y * 32;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    const int m = min(32, C - t0);
    const int kbase = br * H + ks * HKS;
    for (int r = warp; r < 32; r += blockDim.x >> 5) {  // a warp per row: the sum of squares reduces in-warp
        float a = 0.f;
        for (int c4 = lane; c4 < HKS / 4; c4 += 32) {
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (r < m) v = *reinterpret_cast<const float4*>(R + (size_t)(t0 + r) * RW + kbase + c4 * 4);
            *reinterpret_cast<__half2*>(&act[r][c4 * 4]) = __floats2half2_rn(v.x, v.y);
            *reinterpret_cast<__half2*>(&act[r][c4 * 4 + 2]) = __floats2half2_rn(v.z, v.w);
            a += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0 && r < m) atomicAdd(ss + (size_t)(t0 + r) * HC + br, a);
    }
    __syncthreads();
    const int T = warp;
    if (T >= HCD_ROWS / 16) return;
    constexpr int K = RW, groups = K >> 7;
    const uint4* qp = q + (size_t)T * (K >> 5) * 32 + lane;
    const bf16* sp = s + (size_t)T * groups * 16;
    float acc[4][4];
#pragma unroll
    for (int j = 0; j < 4; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    const int g0 = kbase >> 7;
#pragma unroll 1
    for (int gi = g0; gi < g0 + HKS / 128; gi++) {
        uint4 w[4];
#pragma unroll
        for (int i = 0; i < 4; i++) w[i] = __ldg(qp + (size_t)(gi * 4 + i) * 32);
        const float sg = __bfloat162float(sp[gi * 16 + g]), sg8 = __bfloat162float(sp[gi * 16 + g + 8]);
        float c[4][4];
#pragma unroll
        for (int j = 0; j < 4; j++) c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0.f;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const uint32_t ws[4] = {w[i].x, w[i].y, w[i].z, w[i].w};
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int kl = (gi - g0) * 128 + (i * 2 + h) * 16;
                uint32_t aa[4];
                a_from_w8(ws[h * 2], ws[h * 2 + 1], aa);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    if (8 * j >= m) break;
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&act[8 * j + g][kl + 2 * t4]);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&act[8 * j + g][kl + 8 + 2 * t4]);
                    mma16(c[j], aa, b0, b1);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < 4; j++) {
            acc[j][0] += c[j][0] * sg; acc[j][1] += c[j][1] * sg; acc[j][2] += c[j][2] * sg8; acc[j][3] += c[j][3] * sg8;
        }
    }
    const int row0 = T * 16;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int n0 = 8 * j + 2 * t4, n1 = n0 + 1;
        float* ub = u + br * HCD_ROWS;
        if (n0 < m) {
            atomicAdd(ub + (size_t)(t0 + n0) * HC * HCD_ROWS + row0 + g, acc[j][0]);
            atomicAdd(ub + (size_t)(t0 + n0) * HC * HCD_ROWS + row0 + g + 8, acc[j][2]);
        }
        if (n1 < m) {
            atomicAdd(ub + (size_t)(t0 + n1) * HC * HCD_ROWS + row0 + g, acc[j][1]);
            atomicAdd(ub + (size_t)(t0 + n1) * HC * HCD_ROWS + row0 + g + 8, acc[j][3]);
        }
    }
}
// per (row, stream) sum of squares of R. grid (C, 4), 256 threads
__global__ void k_ss_pf(const float* __restrict__ R, float* __restrict__ ss) {
    __shared__ float sh[32];
    const int t = blockIdx.x, br = blockIdx.y;
    float a = 0.f;
    for (int d = threadIdx.x; d < H; d += 256) { const float v = R[(size_t)t * RW + br * H + d]; a += v * v; }
    a = block_sum<256>(a, sh);
    if (threadIdx.x == 0) ss[t * HC + br] = a;
}
// per row: rinv, h = silu(z / 4) (fp16) and inj. grid C, 352 threads
__global__ void k_hc_mid_pf(const float* __restrict__ u, const float* __restrict__ ss, __half* __restrict__ hh,
                            float* __restrict__ rinv, float* __restrict__ inj) {
    const int t = blockIdx.x, j = threadIdx.x;
    float ri[HC];
#pragma unroll
    for (int i = 0; i < HC; i++) ri[i] = rsqrtf(ss[t * HC + i] / H + EPS);
    if (j < HC) rinv[t * HC + j] = ri[j];
    if (j < LOWRANK + HC) {
        float z = 0.f;
#pragma unroll
        for (int i = 0; i < HC; i++) z += u[((size_t)t * HC + i) * HCD_ROWS + j] * ri[i];
        if (j < LOWRANK) hh[(size_t)t * LOWRANK + j] = __float2half_rn(silu(z / HC));
        else if (inj) inj[t * HC + j - LOWRANK] = 2.f * sigm(z / HC);
    }
}
// HC up + mix for C rows from precomputed h / rinv (as k_hcu_pf without the per-block prologue)
__global__ void __launch_bounds__(256) k_hcu_pf2(const uint4* __restrict__ q, const __half* __restrict__ s, const __half* __restrict__ hh,
                                                 const float* __restrict__ rinvg, const float* __restrict__ R, const bf16* __restrict__ nw,
                                                 bf16* __restrict__ x, __half* __restrict__ xh, int C) {
    constexpr int NT = 4;
    __shared__ __align__(16) __half hsb[2][8 * NT][LOWRANK + 8];  // double-buffered h rows (cp.async one group ahead)
    __shared__ __align__(16) float rinvb[2][8 * NT][HC];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int T = blockIdx.x * 8 + warp;
    const int g = lane >> 2, t4 = lane & 3;
    uint4 w[HCU_KS / 2];
    float sgv[HCU_G], sg8v[HCU_G];
    if (T < HCU_TILES) {
        const uint4* qp = q + (size_t)T * (HCU_KS / 2) * 32 + lane;
#pragma unroll
        for (int i = 0; i < HCU_KS / 2; i++) w[i] = __ldg(qp + (size_t)i * 32);
        const __half* sp = s + (size_t)T * HCU_G * 16;
#pragma unroll
        for (int gi = 0; gi < HCU_G; gi++) { sgv[gi] = __half2float(sp[gi * 16 + g]); sg8v[gi] = __half2float(sp[gi * 16 + g + 8]); }
    }
    auto stage = [&](int t0, int buf) {
        const int m = min(8 * NT, C - t0);
        for (int i = tid; i < 8 * NT * LOWRANK / 8; i += 256) {  // 16-byte pieces
            const int r = i / (LOWRANK / 8), c8 = i % (LOWRANK / 8);
            const bool ok = r < m;
            const unsigned sa = (unsigned)__cvta_generic_to_shared(&hsb[buf][r][c8 * 8]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sa),
                         "l"(hh + (size_t)(t0 + (ok ? r : 0)) * LOWRANK + c8 * 8), "r"(ok ? 16 : 0));
        }
        if (tid < 8 * NT * HC) {
            const bool ok = tid / HC < m;
            const unsigned sa = (unsigned)__cvta_generic_to_shared(&rinvb[buf][tid / HC][tid % HC]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(sa), "l"(rinvg + (size_t)t0 * HC + (ok ? tid : 0)),
                         "r"(ok ? 4 : 0));
        }
        asm volatile("cp.async.commit_group;");
    };
    const int step = gridDim.y * 8 * NT;
    if (blockIdx.y * 8 * NT < C) stage(blockIdx.y * 8 * NT, 0);
    int buf = 0;
    for (int t0 = blockIdx.y * 8 * NT; t0 < C; t0 += step, buf ^= 1) {
        const int m = min(8 * NT, C - t0);
        if (t0 + step < C) { stage(t0 + step, buf ^ 1); asm volatile("cp.async.wait_group 1;"); }
        else asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        __half (*hs)[LOWRANK + 8] = hsb[buf];
        float (*rinv)[HC] = rinvb[buf];
        if (T >= HCU_TILES) { __syncthreads(); continue; }
        // residual values of the epilogue first (independent of the MMAs: their latency overlaps them)
        float rv[NT][4];
        {
            const int i = g & 3;
#pragma unroll
            for (int j = 0; j < NT; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int dd = (e >> 1) * 2 + (g >> 2), tok = 8 * j + 2 * t4 + (e & 1);
                    rv[j][e] = tok < m ? __ldg(R + (size_t)(t0 + tok) * RW + i * H + T * 4 + dd) : 0.f;
                }
        }
        float acc[NT][4];
#pragma unroll
        for (int j = 0; j < NT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
#pragma unroll
        for (int gi = 0; gi < HCU_G; gi++) {
#if FLASH_HCU_H16  // fp16-accumulate MMA per 64-k group (2x tensor rate), flushed to fp32 with the group scale
            uint32_t c[NT][2];
#pragma unroll
            for (int j = 0; j < NT; j++) c[j][0] = c[j][1] = 0u;
#else
            float c[NT][4];
#pragma unroll
            for (int j = 0; j < NT; j++) c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0.f;
#endif
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const uint4 wv = w[gi * 2 + i];
                const uint32_t ws[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int k0 = gi * 64 + (i * 2 + h) * 16;
                    uint32_t aa[4];
                    a_from_w8(ws[h * 2], ws[h * 2 + 1], aa);
#pragma unroll
                    for (int j = 0; j < NT; j++) {
                        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&hs[8 * j + g][k0 + 2 * t4]);
                        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&hs[8 * j + g][k0 + 8 + 2 * t4]);
#if FLASH_HCU_H16
                        xh_mma(c[j], aa, b0, b1);
#else
                        mma16(c[j], aa, b0, b1);
#endif
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < NT; j++) {
#if FLASH_HCU_H16
            {
                const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&c[j][0]));
                const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&c[j][1]));
                acc[j][0] += lo.x * sgv[gi]; acc[j][1] += lo.y * sgv[gi]; acc[j][2] += hi.x * sg8v[gi]; acc[j][3] += hi.y * sg8v[gi];
            }
#else
                acc[j][0] += c[j][0] * sgv[gi]; acc[j][1] += c[j][1] * sgv[gi]; acc[j][2] += c[j][2] * sg8v[gi]; acc[j][3] += c[j][3] * sg8v[gi];
#endif
            }
        }
        const int i = g & 3;
#pragma unroll
        for (int j = 0; j < NT; j++) {
            float v[4];
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int dd = (e >> 1) * 2 + (g >> 2), tok = 8 * j + 2 * t4 + (e & 1);
                const int d = T * 4 + dd;
                const float gate = sigm(acc[j][(e >> 1) * 2 + (e & 1)]);
                v[e] = tok < m ? gate * rv[j][e] * rinv[tok][i] * (1.f + __bfloat162float(nw[i * H + d])) : 0.f;
            }
#pragma unroll
            for (int e = 0; e < 4; e++) {
                v[e] += __shfl_xor_sync(0xffffffffu, v[e], 4);
                v[e] += __shfl_xor_sync(0xffffffffu, v[e], 8);
            }
            if (i == 0) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int dd = (e >> 1) * 2 + (g >> 2), tok = 8 * j + 2 * t4 + (e & 1);
                    if (tok >= m) continue;
                    const int d = T * 4 + dd;
                    const float xv = v[e] * (1.f / HC);
                    x[(size_t)(t0 + tok) * H + d] = __float2bfloat16(xv);
                    if (xh) store_xh(xh + (size_t)(t0 + tok) * H, d, xv);
                }
            }
        }
        __syncthreads();  // this buffer is restaged by the next iteration's prefetch
    }
}
// HC up + mix for C rows, weights in registers across 32-row groups (prologue per group as k_hcu)
__global__ void __launch_bounds__(256) k_hcu_pf(const uint4* __restrict__ q, const __half* __restrict__ s, const float* __restrict__ u,
                                                const float* __restrict__ ss, const float* __restrict__ R, const bf16* __restrict__ nw,
                                                float* __restrict__ inj, bf16* __restrict__ x, __half* __restrict__ xh, int C) {
    constexpr int NT = 4;
    __shared__ __half hs[8 * NT][LOWRANK + 8];
    __shared__ float rinv[8 * NT][HC];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int T = blockIdx.x * 8 + warp;
    const int g = lane >> 2, t4 = lane & 3;
    uint4 w[HCU_KS / 2];
    float sgv[HCU_G], sg8v[HCU_G];
    if (T < HCU_TILES) {
        const uint4* qp = q + (size_t)T * (HCU_KS / 2) * 32 + lane;
#pragma unroll
        for (int i = 0; i < HCU_KS / 2; i++) w[i] = __ldg(qp + (size_t)i * 32);
        const __half* sp = s + (size_t)T * HCU_G * 16;
#pragma unroll
        for (int gi = 0; gi < HCU_G; gi++) { sgv[gi] = __half2float(sp[gi * 16 + g]); sg8v[gi] = __half2float(sp[gi * 16 + g + 8]); }
    }
    for (int t0 = 0; t0 < C; t0 += 8 * NT) {
        const int m = min(8 * NT, C - t0);
        __syncthreads();
        if (tid < m * HC) rinv[tid / HC][tid % HC] = rsqrtf(ss[(size_t)t0 * HC + tid] / H + EPS);
        __syncthreads();
        for (int idx = tid; idx < m * (LOWRANK + HC); idx += 256) {
            const int t = idx / (LOWRANK + HC), j = idx % (LOWRANK + HC);
            float z = 0.f;
#pragma unroll
            for (int i = 0; i < HC; i++) z += u[((size_t)(t0 + t) * HC + i) * HCD_ROWS + j] * rinv[t][i];
            if (j < LOWRANK) hs[t][j] = __float2half_rn(silu(z / HC));
            else if (blockIdx.x == 0 && inj) inj[(t0 + t) * HC + j - LOWRANK] = 2.f * sigm(z / HC);
        }
        for (int idx = tid; idx < (8 * NT - m) * (LOWRANK + 8); idx += 256) hs[m + idx / (LOWRANK + 8)][idx % (LOWRANK + 8)] = __float2half_rn(0.f);
        __syncthreads();
        if (T >= HCU_TILES) continue;
        float acc[NT][4];
#pragma unroll
        for (int j = 0; j < NT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
#pragma unroll
        for (int gi = 0; gi < HCU_G; gi++) {
            float c[NT][4];
#pragma unroll
            for (int j = 0; j < NT; j++) c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0.f;
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const uint4 wv = w[gi * 2 + i];
                const uint32_t ws[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int k0 = gi * 64 + (i * 2 + h) * 16;
                    uint32_t aa[4];
                    a_from_w8(ws[h * 2], ws[h * 2 + 1], aa);
#pragma unroll
                    for (int j = 0; j < NT; j++) {
                        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&hs[8 * j + g][k0 + 2 * t4]);
                        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&hs[8 * j + g][k0 + 8 + 2 * t4]);
                        mma16(c[j], aa, b0, b1);
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < NT; j++) {
                acc[j][0] += c[j][0] * sgv[gi]; acc[j][1] += c[j][1] * sgv[gi]; acc[j][2] += c[j][2] * sg8v[gi]; acc[j][3] += c[j][3] * sg8v[gi];
            }
        }
        const int i = g & 3;
#pragma unroll
        for (int j = 0; j < NT; j++) {
            float v[4];
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int dd = (e >> 1) * 2 + (g >> 2), tok = 8 * j + 2 * t4 + (e & 1);
                const int d = T * 4 + dd;
                const float gate = sigm(acc[j][(e >> 1) * 2 + (e & 1)]);
                v[e] = tok < m ? gate * R[(size_t)(t0 + tok) * RW + i * H + d] * rinv[tok][i] * (1.f + __bfloat162float(nw[i * H + d])) : 0.f;
            }
#pragma unroll
            for (int e = 0; e < 4; e++) {
                v[e] += __shfl_xor_sync(0xffffffffu, v[e], 4);
                v[e] += __shfl_xor_sync(0xffffffffu, v[e], 8);
            }
            if (i == 0) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int dd = (e >> 1) * 2 + (g >> 2), tok = 8 * j + 2 * t4 + (e & 1);
                    if (tok >= m) continue;
                    const int d = T * 4 + dd;
                    const float xv = v[e] * (1.f / HC);
                    x[(size_t)(t0 + tok) * H + d] = __float2bfloat16(xv);
                    if (xh) store_xh(xh + (size_t)(t0 + tok) * H, d, xv);
                }
            }
        }
    }
}
// MoE prefill jobs: per expert the rows routed to it (token list + weight), jobs = local groups of hit experts
// (+ the shared expert: all rows). One block of 512 threads.
__global__ void __launch_bounds__(512) k_jobs_pf(int C, const int* __restrict__ gmap, const char* __restrict__ groups,
                                                 ColdTab ct, int ndev, const int* __restrict__ gsel,
                                                 const float* __restrict__ gw, const float* __restrict__ gsh,
                                                 PfJob* __restrict__ jobs, int* __restrict__ tl, float* __restrict__ wl,
                                                 int* __restrict__ njobs, int* __restrict__ inv = nullptr) {
    __shared__ int cnt[NE + 1], off[NE + 1], fill[NE + 1], jbase[NE + 1], abase[NE + 1], nloc[NE];
    __shared__ int wtot[16], wtot2[16], wtot3[16];
    const int e = threadIdx.x, warp = e >> 5, lane = e & 31;
    cnt[e] = 0;
    fill[e] = 0;
    if (e == 0) { cnt[NE] = C; fill[NE] = 0; }
    __syncthreads();
    for (int i = e; i < C * TOPK; i += 512) atomicAdd(&cnt[gsel[i]], 1);
    __syncthreads();
    int ng = 0;
    int gm[NGRP];
#pragma unroll
    for (int g = 0; g < NGRP; g++) { gm[g] = gmap[e * NGRP + g]; ng += (gm[g] >= 0 && cnt[e] > 0); }
    nloc[e] = ng;
    // exclusive scans: token-list offsets (count) and job indices (local groups)
    const int ar = ng * cnt[e];  // activation rows of the expert's local groups
    const int nq = (cnt[e] + PF_SHJ - 1) / PF_SHJ;  // row chunks of the expert (jobs = chunks x local groups)
    int a = cnt[e], b = ng * nq, c3 = ar;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int na = __shfl_up_sync(0xffffffffu, a, o), nb = __shfl_up_sync(0xffffffffu, b, o), nc = __shfl_up_sync(0xffffffffu, c3, o);
        if (lane >= o) { a += na; b += nb; c3 += nc; }
    }
    if (lane == 31) { wtot[warp] = a; wtot2[warp] = b; wtot3[warp] = c3; }
    __syncthreads();
    int ba = 0, bb = 0, bc = 0, ta = 0, tb = 0, tc = 0;
    for (int w2 = 0; w2 < 16; w2++) {
        if (w2 < warp) { ba += wtot[w2]; bb += wtot2[w2]; bc += wtot3[w2]; }
        ta += wtot[w2]; tb += wtot2[w2]; tc += wtot3[w2];
    }
    off[e] = ba + a - cnt[e];
    jbase[e] = bb + b - ng * nq;
    abase[e] = bc + c3 - ar;
    if (e == 0) { off[NE] = ta; jbase[NE] = tb; abase[NE] = tc; }
    __syncthreads();
    for (int i = e; i < C * TOPK; i += 512) {
        const int ex = gsel[i];
        const int slot = off[ex] + atomicAdd(&fill[ex], 1);
        tl[slot] = i / TOPK;
        wl[slot] = gw[i];
        // slot of (row, k) for the slot-buffer down's sums (k_moe_sum); -1: the expert has no group on this card (no
        // job writes the slot, it holds another expert's output from an earlier batch)
        if (inv) inv[i] = nloc[ex] > 0 ? slot : -1;
    }
    for (int i = e; i < C; i += 512) { tl[ta + i] = i; wl[ta + i] = gsh[i]; }
    __syncthreads();
    auto mk = [&](int li) {
        const char* bp = grp_ptr(groups, li, ndev, ct, nullptr);
        MoeJob jb;
        jb.gq = reinterpret_cast<const uint4*>(bp);
        jb.gs = reinterpret_cast<const __half*>(bp + GG_Q);
        jb.dq = reinterpret_cast<const uint4*>(bp + GG_Q + GG_S);
        jb.ds = reinterpret_cast<const __half*>(bp + GG_Q + GG_S + GD_Q);
        return jb;
    };
    if (cnt[e] > 0) {
        int j = jbase[e];
        for (int q = 0; q < nq; q++) {
            int k = 0;
            for (int g = 0; g < NGRP; g++)
                if (gm[g] >= 0) {
                    PfJob pj; pj.w = mk(gm[g]); pj.off = off[e] + q * PF_SHJ; pj.n = min(PF_SHJ, cnt[e] - q * PF_SHJ);
                    pj.aoff = abase[e] + k * cnt[e] + q * PF_SHJ; pj.ng = k == 0 ? ng : 0;
                    jobs[j++] = pj; k++;
                }
        }
    }
    if (e == 0) {
        // shared expert: every row; jobs of PF_SHJ rows (many blocks instead of a few long ones at the end of the grid),
        // the local groups of one row chunk consecutive (k_moe_dn_pf merges them: ng on the chunk's first job)
        int nl = 0, lis[NGRP];
        for (int g = 0; g < NGRP; g++) {
            const int li = gmap[NE * NGRP + g];
            if (li >= 0) lis[nl++] = li;
        }
        int j = tb;
        for (int q = 0; q * PF_SHJ < C; q++)
            for (int k = 0; k < nl; k++) {
                PfJob pj; pj.w = mk(lis[k]); pj.off = ta + q * PF_SHJ; pj.n = min(PF_SHJ, C - q * PF_SHJ);
                pj.aoff = tc + k * C + q * PF_SHJ; pj.ng = k == 0 ? nl : 0; jobs[j++] = pj;
            }
        *njobs = j;
    }
}
// gate_up of a prefill job over its rows (gathered from x XH), 32-row groups; act[off + i][128] = silu(g) * u * w
__global__ void __launch_bounds__(128) k_moe_gu_pf(const PfJob* __restrict__ jobs, const int* __restrict__ njobs,
                                                   const int* __restrict__ tl, const float* __restrict__ wl,
                                                   const __half* __restrict__ x, __half* __restrict__ act) {
    constexpr int NT = 4;
    const int job = blockIdx.y;
    if (job >= *njobs) return;
    const PfJob pj = jobs[job];
    const int r = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarps = 4;
    const int g = lane >> 2, t4 = lane & 3;
    constexpr int groups = H >> 7, chunks = H >> 6;
    const int gb = warp * groups / nwarps, ge = (warp + 1) * groups / nwarps;
    const uint4* qp = pj.w.gq + (size_t)r * chunks * 32 + lane;
    const __half* sp = pj.w.gs + (size_t)r * groups * 16;
    __shared__ float red[4][NT][32][4];
    for (int t0 = 0; t0 < pj.n; t0 += 8 * NT) {
        const int m = min(8 * NT, pj.n - t0);
        const uint4* xr[NT];
#pragma unroll
        for (int j = 0; j < NT; j++) {
            const int li = 8 * j + g < m ? 8 * j + g : 0;
            xr[j] = reinterpret_cast<const uint4*>(x + (size_t)tl[pj.off + t0 + li] * H + t4 * 16);
        }
        float acc[NT][4];
#pragma unroll
        for (int j = 0; j < NT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
        for (int gi = gb; gi < ge; gi++) {
            const float sg = __half2float(sp[gi * 16 + g]) * kXhScale, sg8 = __half2float(sp[gi * 16 + g + 8]) * kXhScale;
            const uint4 w0 = __ldg(qp + (size_t)(2 * gi) * 32), w1 = __ldg(qp + (size_t)(2 * gi + 1) * 32);
#pragma unroll
            for (int c = 0; c < 2; c++) {
                const int ch = 2 * gi + c;
                const uint4 wc = c ? w1 : w0;
                uint32_t hacc[NT][2];
                uint4 b0[NT], b1[NT];
#pragma unroll
                for (int j = 0; j < NT; j++) { hacc[j][0] = hacc[j][1] = 0u; b0[j] = xr[j][(size_t)ch * 8]; b1[j] = xr[j][(size_t)ch * 8 + 1]; }
                const uint32_t words[4] = {wc.x, wc.y, wc.z, wc.w};
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    uint32_t a[4];
                    xh_dequant(words[q], a);
#pragma unroll
                    for (int j = 0; j < NT; j++) {
                        if (8 * j >= m) break;  // empty 8-row tiles of the last group (warp-uniform)
                        const uint4& bb = (q >> 1) ? b1[j] : b0[j];
                        xh_mma(hacc[j], a, (q & 1) ? bb.z : bb.x, (q & 1) ? bb.w : bb.y);
                    }
                }
#pragma unroll
                for (int j = 0; j < NT; j++) xh_flush(acc[j], hacc[j], sg, sg8);
            }
        }
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
            for (int i = 0; i < 4; i++) red[warp][j][lane][i] = acc[j][i];
        __syncthreads();
        if (warp == 0) {
            const int col = r * 8 + g;
#pragma unroll
            for (int j = 0; j < NT; j++) {
                for (int w2 = 1; w2 < nwarps; w2++)
#pragma unroll
                    for (int i = 0; i < 4; i++) acc[j][i] += red[w2][j][lane][i];
                const int n0 = 8 * j + 2 * t4, n1 = n0 + 1;
                if (n0 < m) store_xh(act + (size_t)(pj.aoff + t0 + n0) * GRP, col, silu(acc[j][0]) * acc[j][2] * wl[pj.off + t0 + n0]);
                if (n1 < m) store_xh(act + (size_t)(pj.aoff + t0 + n1) * GRP, col, silu(acc[j][1]) * acc[j][3] * wl[pj.off + t0 + n1]);
            }
        }
        __syncthreads();
    }
}
// v2 (FLASH_MOE_GU2): one block per job covers all 256 gate_up rows (8 warps x 2 16-row tiles) for 32-token tiles;
// the tokens' x K-slices (64 columns) are staged once per block in shared memory by cp.async (double buffer) instead
// of every 16-row block reading the rows from L2 (16 x the x bytes). Per element: the same 64-column fp16 chunks
// flushed with the group scale, the K order sequential (v1 summed 4 K-quarters at the end).
__global__ void __launch_bounds__(256) k_moe_gu_pf2(const PfJob* __restrict__ jobs, const int* __restrict__ njobs,
                                                    const int* __restrict__ tl, const float* __restrict__ wl,
                                                    const __half* __restrict__ x, __half* __restrict__ act) {
    constexpr int NT = 4, TT = 8 * NT, XP = 72;  // tokens per tile, padded halves per staged row (64 + 8)
    constexpr int groups = H >> 7, chunks = H >> 6;
    __shared__ __align__(16) __half xs[2][TT][XP];
    const int nj = *njobs;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t4 = lane & 3;
    for (int job = blockIdx.x; job < nj; job += gridDim.x) {
        const PfJob pj = jobs[job];
        const uint4* qp0 = pj.w.gq + (size_t)(2 * warp) * chunks * 32 + lane;
        const uint4* qp1 = pj.w.gq + (size_t)(2 * warp + 1) * chunks * 32 + lane;
        const __half* sp0 = pj.w.gs + (size_t)(2 * warp) * groups * 16;
        const __half* sp1 = pj.w.gs + (size_t)(2 * warp + 1) * groups * 16;
        for (int t0 = 0; t0 < pj.n; t0 += TT) {
            const int m = min(TT, pj.n - t0), nt = (m + 7) >> 3;
            // staging: thread i -> token row i / 8, 16-byte piece i % 8 of the 64-column slice
            const int st_r = threadIdx.x >> 3, st_p = threadIdx.x & 7;
            const __half* xrow = x + (size_t)tl[pj.off + t0 + (st_r < m ? st_r : 0)] * H + st_p * 8;
            auto stage = [&](int ch, int b) {
                const unsigned dst = (unsigned)__cvta_generic_to_shared(&xs[b][st_r][st_p * 8]);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(xrow + (size_t)ch * 64));
                asm volatile("cp.async.commit_group;");
            };
            float acc[2][NT][4];
#pragma unroll
            for (int a = 0; a < 2; a++)
#pragma unroll
                for (int j = 0; j < NT; j++) acc[a][j][0] = acc[a][j][1] = acc[a][j][2] = acc[a][j][3] = 0.f;
            stage(0, 0);
            uint4 wn0 = __ldg(qp0), wn1 = __ldg(qp1);
            for (int ch = 0; ch < chunks; ch++) {
                const int b = ch & 1;
                const uint4 w0 = wn0, w1 = wn1;
                if (ch + 1 < chunks) {
                    stage(ch + 1, b ^ 1);
                    wn0 = __ldg(qp0 + (size_t)(ch + 1) * 32);
                    wn1 = __ldg(qp1 + (size_t)(ch + 1) * 32);
                    asm volatile("cp.async.wait_group 1;");
                } else {
                    asm volatile("cp.async.wait_group 0;");
                }
                __syncthreads();
                const int gi = ch >> 1;
                const float s00 = __half2float(sp0[gi * 16 + g]) * kXhScale, s08 = __half2float(sp0[gi * 16 + g + 8]) * kXhScale;
                const float s10 = __half2float(sp1[gi * 16 + g]) * kXhScale, s18 = __half2float(sp1[gi * 16 + g + 8]) * kXhScale;
                uint4 b0[NT], b1[NT];
#pragma unroll
                for (int j = 0; j < NT; j++) {
                    if (j < nt) {
                        b0[j] = *reinterpret_cast<const uint4*>(&xs[b][8 * j + g][t4 * 16]);
                        b1[j] = *reinterpret_cast<const uint4*>(&xs[b][8 * j + g][t4 * 16 + 8]);
                    }
                }
                uint32_t h0[NT][2], h1[NT][2];
#pragma unroll
                for (int j = 0; j < NT; j++) h0[j][0] = h0[j][1] = h1[j][0] = h1[j][1] = 0u;
                const uint32_t wa[4] = {w0.x, w0.y, w0.z, w0.w}, wb[4] = {w1.x, w1.y, w1.z, w1.w};
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    uint32_t a0[4], a1[4];
                    xh_dequant(wa[q], a0);
                    xh_dequant(wb[q], a1);
#pragma unroll
                    for (int j = 0; j < NT; j++) {
                        if (j < nt) {
                            const uint4& bb = (q >> 1) ? b1[j] : b0[j];
                            const uint32_t x0 = (q & 1) ? bb.z : bb.x, x1 = (q & 1) ? bb.w : bb.y;
                            xh_mma(h0[j], a0, x0, x1);
                            xh_mma(h1[j], a1, x0, x1);
                        }
                    }
                }
#pragma unroll
                for (int j = 0; j < NT; j++) {
                    if (j < nt) {
                        xh_flush(acc[0][j], h0[j], s00, s08);
                        xh_flush(acc[1][j], h1[j], s10, s18);
                    }
                }
                __syncthreads();  // buffer b is restaged two chunks later
            }
#pragma unroll
            for (int a = 0; a < 2; a++) {
                const int col = (2 * warp + a) * 8 + g;
#pragma unroll
                for (int j = 0; j < NT; j++) {
                    const int n0 = 8 * j + 2 * t4, n1 = n0 + 1;
                    if (n0 < m) store_xh(act + (size_t)(pj.aoff + t0 + n0) * GRP, col, silu(acc[a][j][0]) * acc[a][j][2] * wl[pj.off + t0 + n0]);
                    if (n1 < m) store_xh(act + (size_t)(pj.aoff + t0 + n1) * GRP, col, silu(acc[a][j][1]) * acc[a][j][3] * wl[pj.off + t0 + n1]);
                }
            }
        }
    }
}
// down of a prefill job: part[tok] += W_down(job) . act rows (atomics; 32-row groups). Block = 4 warps, warp = TT
// 16-row output tiles (TT * 16 output dims): the act fragments of a token group are loaded once for its TT tiles.
// debug (FLASH_DBGHASH=1): order-independent 64-bit hash of n 32-bit words added into *out
__global__ void k_dbg_hash(const unsigned* __restrict__ p, size_t n, unsigned long long* __restrict__ out) {
    unsigned long long h = 0ull;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        h += (unsigned long long)(p[i] + 0x9e3779b9u) * (2654435761ull * (i + 1) | 1ull);
    for (int o = 16; o > 0; o >>= 1) h += __shfl_xor_sync(0xffffffffu, h, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, h);
}
template <int TT>
__global__ void __launch_bounds__(128) k_moe_dn_pf(const PfJob* __restrict__ jobs, const int* __restrict__ njobs,
                                                   const int* __restrict__ tl, const __half* __restrict__ act,
                                                   float* __restrict__ part, unsigned long long* __restrict__ ipart = nullptr) {
    // jobs of one expert are consecutive and share the token list: the first one (ng > 0) runs all its local groups
    // and adds the sum once per output (atomics / ng)
    constexpr int NT = 4;
    const int job = blockIdx.y;
    if (job >= *njobs) return;
    const int ngj = jobs[job].ng;
    if (ngj == 0) return;
    const int pn = jobs[job].n, poff = jobs[job].off;
    const int rb = (blockIdx.x * 4 + (threadIdx.x >> 5)) * TT;  // first 16-row output tile of this warp
    const int lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    for (int t0 = 0; t0 < pn; t0 += 8 * NT) {
        const int m = min(8 * NT, pn - t0);
        float acc[TT][NT][4];
#pragma unroll
        for (int tt = 0; tt < TT; tt++)
#pragma unroll
            for (int j = 0; j < NT; j++) acc[tt][j][0] = acc[tt][j][1] = acc[tt][j][2] = acc[tt][j][3] = 0.f;
        for (int k = 0; k < ngj; k++) {
            const PfJob pj = jobs[job + k];
            uint4 b0[2][NT], b1[2][NT];
#pragma unroll
            for (int c = 0; c < 2; c++)
#pragma unroll
                for (int j = 0; j < NT; j++) {
                    const int li = 8 * j + g < m ? 8 * j + g : 0;
                    const uint4* xr = reinterpret_cast<const uint4*>(act + (size_t)(pj.aoff + t0 + li) * GRP + t4 * 16);
                    b0[c][j] = xr[c * 8];
                    b1[c][j] = xr[c * 8 + 1];
                }
#pragma unroll
            for (int tt = 0; tt < TT; tt++) {
                const int r = rb + tt;
                const uint4* qp = pj.w.dq + (size_t)r * 2 * 32 + lane;
                const uint4 w0 = __ldg(qp), w1 = __ldg(qp + 32);
                const float sg = __half2float(pj.w.ds[r * 16 + g]) * kXhScale, sg8 = __half2float(pj.w.ds[r * 16 + g + 8]) * kXhScale;
#pragma unroll
                for (int c = 0; c < 2; c++) {
                    const uint4 wc = c ? w1 : w0;
                    uint32_t hacc[NT][2];
#pragma unroll
                    for (int j = 0; j < NT; j++) hacc[j][0] = hacc[j][1] = 0u;
                    const uint32_t words[4] = {wc.x, wc.y, wc.z, wc.w};
#pragma unroll
                    for (int q = 0; q < 4; q++) {
                        uint32_t a[4];
                        xh_dequant(words[q], a);
#pragma unroll
                        for (int j = 0; j < NT; j++) {
                            if (8 * j >= m) break;
                            const uint4& bb = (q >> 1) ? b1[c][j] : b0[c][j];
                            xh_mma(hacc[j], a, (q & 1) ? bb.z : bb.x, (q & 1) ? bb.w : bb.y);
                        }
                    }
#pragma unroll
                    for (int j = 0; j < NT; j++) xh_flush(acc[tt][j], hacc[j], sg, sg8);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NT; j++) {
            const int n0 = 8 * j + 2 * t4;
            const int tk0 = n0 < m ? tl[poff + t0 + n0] : -1, tk1 = n0 + 1 < m ? tl[poff + t0 + n0 + 1] : -1;
#pragma unroll
            for (int tt = 0; tt < TT; tt++) {
                const int row0 = (rb + tt) * 16;
#ifdef FLASH_DN_NOATOM  // bench only: atomics cost
                if (acc[tt][j][0] == 123.f) part[0] = acc[tt][j][1] + acc[tt][j][2] + acc[tt][j][3];
                continue;
#endif
                if (ipart) {  // deterministic mode: exact fixed-point sums (order independent), k_fx_to_f after
                    if (tk0 >= 0) {
                        atomicAdd(ipart + (size_t)tk0 * H + row0 + g, fx_of(acc[tt][j][0]));
                        atomicAdd(ipart + (size_t)tk0 * H + row0 + g + 8, fx_of(acc[tt][j][2]));
                    }
                    if (tk1 >= 0) {
                        atomicAdd(ipart + (size_t)tk1 * H + row0 + g, fx_of(acc[tt][j][1]));
                        atomicAdd(ipart + (size_t)tk1 * H + row0 + g + 8, fx_of(acc[tt][j][3]));
                    }
                    continue;
                }
                if (tk0 >= 0) {
                    atomicAdd(part + (size_t)tk0 * H + row0 + g, acc[tt][j][0]);
                    atomicAdd(part + (size_t)tk0 * H + row0 + g + 8, acc[tt][j][2]);
                }
                if (tk1 >= 0) {
                    atomicAdd(part + (size_t)tk1 * H + row0 + g, acc[tt][j][1]);
                    atomicAdd(part + (size_t)tk1 * H + row0 + g + 8, acc[tt][j][3]);
                }
            }
        }
    }
}
// down of a prefill job into the slot buffer (FLASH_MOE_SLOT): the expert output of every (row, expert) slot of the
// token lists is stored (fp16, x 1/16) instead of added to the row's partial by fp32 atomics (the atomics bound the
// kernel: ~28K per row); k_moe_sum adds a row's 11 slots in a fixed order. Same tiles and arithmetic as k_moe_dn_pf.
constexpr float kSlotScale = 16.f;
struct SlotBuf {  // slot rows [0, n1) in a, the rest in b (the big-batch extension)
    __half* a; int n1; __half* b;
    __device__ __forceinline__ __half* row(int s) const { return s < n1 ? a + (size_t)s * H : b + (size_t)(s - n1) * H; }
};
template <int TT>
__global__ void __launch_bounds__(128) k_moe_dn_slot(const PfJob* __restrict__ jobs, const int* __restrict__ njobs,
                                                     const __half* __restrict__ act, SlotBuf slots) {
    constexpr int NT = 4;
    const int job = blockIdx.y;
    if (job >= *njobs) return;
    const int ngj = jobs[job].ng;
    if (ngj == 0) return;
    const int pn = jobs[job].n, poff = jobs[job].off;
    const int rb = (blockIdx.x * 4 + (threadIdx.x >> 5)) * TT;
    const int lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    for (int t0 = 0; t0 < pn; t0 += 8 * NT) {
        const int m = min(8 * NT, pn - t0);
        float acc[TT][NT][4];
#pragma unroll
        for (int tt = 0; tt < TT; tt++)
#pragma unroll
            for (int j = 0; j < NT; j++) acc[tt][j][0] = acc[tt][j][1] = acc[tt][j][2] = acc[tt][j][3] = 0.f;
        for (int k = 0; k < ngj; k++) {
            const PfJob pj = jobs[job + k];
            uint4 b0[2][NT], b1[2][NT];
#pragma unroll
            for (int c = 0; c < 2; c++)
#pragma unroll
                for (int j = 0; j < NT; j++) {
                    const int li = 8 * j + g < m ? 8 * j + g : 0;
                    const uint4* xr = reinterpret_cast<const uint4*>(act + (size_t)(pj.aoff + t0 + li) * GRP + t4 * 16);
                    b0[c][j] = xr[c * 8];
                    b1[c][j] = xr[c * 8 + 1];
                }
#pragma unroll
            for (int tt = 0; tt < TT; tt++) {
                const int r = rb + tt;
                const uint4* qp = pj.w.dq + (size_t)r * 2 * 32 + lane;
                const uint4 w0 = __ldg(qp), w1 = __ldg(qp + 32);
                const float sg = __half2float(pj.w.ds[r * 16 + g]) * kXhScale, sg8 = __half2float(pj.w.ds[r * 16 + g + 8]) * kXhScale;
#pragma unroll
                for (int c = 0; c < 2; c++) {
                    const uint4 wc = c ? w1 : w0;
                    uint32_t hacc[NT][2];
#pragma unroll
                    for (int j = 0; j < NT; j++) hacc[j][0] = hacc[j][1] = 0u;
                    const uint32_t words[4] = {wc.x, wc.y, wc.z, wc.w};
#pragma unroll
                    for (int q = 0; q < 4; q++) {
                        uint32_t a[4];
                        xh_dequant(words[q], a);
#pragma unroll
                        for (int j = 0; j < NT; j++) {
                            if (8 * j >= m) break;
                            const uint4& bb = (q >> 1) ? b1[c][j] : b0[c][j];
                            xh_mma(hacc[j], a, (q & 1) ? bb.z : bb.x, (q & 1) ? bb.w : bb.y);
                        }
                    }
#pragma unroll
                    for (int j = 0; j < NT; j++) xh_flush(acc[tt][j], hacc[j], sg, sg8);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NT; j++) {
            const int n0 = 8 * j + 2 * t4;
#pragma unroll
            for (int tt = 0; tt < TT; tt++) {
                const int row0 = (rb + tt) * 16;
                if (n0 < m) {
                    __half* o = slots.row(poff + t0 + n0) + row0 + g;
                    o[0] = __float2half_rn(acc[tt][j][0] * (1.f / kSlotScale));
                    o[8] = __float2half_rn(acc[tt][j][2] * (1.f / kSlotScale));
                }
                if (n0 + 1 < m) {
                    __half* o = slots.row(poff + t0 + n0 + 1) + row0 + g;
                    o[0] = __float2half_rn(acc[tt][j][1] * (1.f / kSlotScale));
                    o[8] = __float2half_rn(acc[tt][j][3] * (1.f / kSlotScale));
                }
            }
        }
    }
}
// part[t] = shared slot + the row's TOPK routed slots (k order): every element written, no memset
__global__ void __launch_bounds__(320) k_moe_sum(SlotBuf slots, const int* __restrict__ inv, int shoff,
                                                 const int* __restrict__ gmap, float* __restrict__ part) {
    const int t = blockIdx.x, i = threadIdx.x;  // 8 columns per thread (H = 2560 = 320 x 8)
    __shared__ int sl[TOPK + 1];
    if (i < TOPK) sl[i] = inv[t * TOPK + i];  // -1: no group of that expert on this card
    if (i == TOPK) {  // the shared expert's slot, if this card holds a group of it
        bool any = false;
        for (int g = 0; g < NGRP; g++) any |= gmap[NE * NGRP + g] >= 0;
        sl[TOPK] = any ? shoff + t : -1;
    }
    __syncthreads();
    float a[8];
#pragma unroll
    for (int q = 0; q < 8; q++) a[q] = 0.f;
    if (sl[TOPK] >= 0) {
        const uint4 v = reinterpret_cast<const uint4*>(slots.row(sl[TOPK]))[i];
        const __half2* h = reinterpret_cast<const __half2*>(&v);
#pragma unroll
        for (int q = 0; q < 4; q++) { const float2 f = __half22float2(h[q]); a[2 * q] = f.x; a[2 * q + 1] = f.y; }
    }
#pragma unroll
    for (int k = 0; k < TOPK; k++) {
        if (sl[k] < 0) continue;
        const uint4 v = reinterpret_cast<const uint4*>(slots.row(sl[k]))[i];
        const __half2* h = reinterpret_cast<const __half2*>(&v);
#pragma unroll
        for (int q = 0; q < 4; q++) { const float2 f = __half22float2(h[q]); a[2 * q] += f.x; a[2 * q + 1] += f.y; }
    }
    float4* o = reinterpret_cast<float4*>(part + (size_t)t * H) + 2 * i;
    o[0] = make_float4(a[0] * kSlotScale, a[1] * kSlotScale, a[2] * kSlotScale, a[3] * kSlotScale);
    o[1] = make_float4(a[4] * kSlotScale, a[5] * kSlotScale, a[6] * kSlotScale, a[7] * kSlotScale);
}
__global__ void k_spin(unsigned ns) {  // debug: delay a stream (race stress tests)
    const unsigned long long t0 = clock64();
    while (clock64() - t0 < (unsigned long long)ns * 2ull) {}
}
__global__ void k_possub(int* __restrict__ out, const int* __restrict__ pos, int n) {
    const int i = threadIdx.x;
    if (i < n) out[i] = *pos + i * 32;
}

}  // namespace flash
