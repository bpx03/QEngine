#pragma once
// [dsv4-pf2-fx] Prefill routed expert on the IQ bytes in two launches (no bf16 weight round trip):
//   k_fx13  w1 / w3 rows c0 .. c0 + 31 (a block) for TOK = 8 NT tokens of the expert (blockIdx.y = token pass):
//           codes staged per 256-weight block by cp.async (warp-private, S stages), decoded in registers by gg_sub8
//           (the same values as k_deq_gg4: bf16 rn of w * d), mma.sync m16n8k16 bf16 -> fp32 with the weights as the
//           M operand (16 rows) and tokens as N (8 per tile). K is permuted inside a 32-weight sub-block so that a
//           lane's 8 decoded weights (8t .. 8t + 7, gg_sub8 h = t) and its 16 B x load cover the same k (the dot is
//           over all k: the permutation only changes the fp32 summation order). 8 warps = 4 M tiles (w1 lo / hi,
//           w3 lo / hi) x 2 K halves; the halves add in shared memory; epilogue = k_glu_rows + k_q_fp8 (same
//           expressions) -> hq bf16 [m][INTER].
//   k_fx2   w2: 64 output rows per block (4 M tiles x 2 K halves), hq -> Y fp32 [m][D] (row stride D).
// Summation order differs from cuBLAS: KL gate (not bit-exact with the old path).
#include <atomic>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "dsv4.h"
#include "dsv4_kernels.cuh"
#include "gguf_gpu.cuh"

namespace dsv4 {

template <int MF>
struct FxF {
    static constexpr int CB = MF == XF_IQ2XXS ? 64 : MF == XF_IQ3XXS ? 96 : MF == XF_IQ2S ? 80 : 108;
    static constexpr int NW = CB / 4;              // 32-bit words of one 256-weight block (codes, no d)
    static constexpr int W = (NW + 3) / 4;         // words per lane of a quad (lane t: words t W .. t W + W - 1)
    static constexpr int GB = MF == XF_IQ2XXS ? 2048 : MF == XF_IQ3XXS ? 1024 : MF == XF_IQ2S ? 8192 : 2048;
};
constexpr int FX_P = 3;   // blocks of codes in flight per warp (register ring)

__device__ __forceinline__ void fx_mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                       uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t fx_pk(float a, float b) {
    bf162 p = __floats2bfloat162_rn(a, b);
    return *(uint32_t*)&p;
}
// word wd (a compile-time constant after unrolling) of the quad's block: held by quad lane wd / W, register wd % W
template <int MF>
__device__ __forceinline__ uint32_t fx_word(const uint32_t* R, int wd) {
    constexpr int W = FxF<MF>::W;
    return __shfl_sync(0xffffffffu, R[wd % W], (threadIdx.x & 28) | (wd / W));
}
// gg_sub8 (weights 8h .. 8h + 7 of sub-block k) on the quad's registers: the same operations, the bytes by shuffles
template <int MF>
__device__ __forceinline__ void fx_dec(const uint32_t* R, int k, int h, const void* grid, float* w) {
    if constexpr (MF == XF_IQ2XXS) {
        const uint32_t vx = fx_word<MF>(R, 2 * k), vy = fx_word<MF>(R, 2 * k + 1);
        const float db = (0.5f + (float)(vy >> 28)) * 0.25f;
        const uint2 g = ((const uint2*)grid)[(vx >> (8 * h)) & 255];
        const uint32_t s = gg_sign8((vy >> (7 * h)) & 127);
#pragma unroll
        for (int j = 0; j < 8; j++) w[j] = gg_sgn(db * gg_u8f(j < 4 ? g.x : g.y, j & 3), s, j);
    } else if constexpr (MF == XF_IQ3XXS) {
        const uint32_t vx = fx_word<MF>(R, 2 * k), vy = fx_word<MF>(R, 2 * k + 1), aux = fx_word<MF>(R, 16 + k);
        const float db = (0.5f + (float)(aux >> 28)) * 0.5f;
        const uint32_t* G = (const uint32_t*)grid;
#pragma unroll
        for (int ii = 0; ii < 2; ii++) {
            const int i = 2 * h + ii;
            const uint32_t g = G[((i < 4 ? vx : vy) >> (8 * (i & 3))) & 255];
            const uint32_t s = gg_sign8((aux >> (7 * (i / 2))) & 127) >> (4 * (i & 1));
#pragma unroll
            for (int j = 0; j < 4; j++) w[4 * ii + j] = gg_sgn(db * gg_u8f(g, j), s, j);
        }
    } else if constexpr (MF == XF_IQ2S) {
        const uint32_t qs = fx_word<MF>(R, k), sg = fx_word<MF>(R, 8 + k);
        const uint32_t qh = (fx_word<MF>(R, 16 + k / 4) >> (8 * (k & 3))) & 255;
        const uint32_t sc = (fx_word<MF>(R, 18 + k / 4) >> (8 * (k & 3))) & 255;
        const uint2 g = ((const uint2*)grid)[((qs >> (8 * h)) & 255) | (((qh >> (2 * h)) & 3) << 8)];
        const float db = (0.5f + (float)(h < 2 ? sc & 15 : sc >> 4)) * 0.25f;
        const uint32_t s = (sg >> (8 * h)) & 255;
#pragma unroll
        for (int j = 0; j < 8; j++) w[j] = gg_sgn(db * gg_u8f(j < 4 ? g.x : g.y, j & 3), s, j);
    } else {   // XF_IQ3S
        const uint32_t q0 = fx_word<MF>(R, 2 * k), q1 = fx_word<MF>(R, 2 * k + 1);
        const uint32_t qh = (fx_word<MF>(R, 16 + k / 4) >> (8 * (k & 3))) & 255, sg = fx_word<MF>(R, 18 + k);
        const uint32_t scb = (fx_word<MF>(R, 26) >> (8 * (k / 2))) & 255;
        const float db = (float)(1 + 2 * ((scb >> (4 * (k & 1))) & 15));
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

template <int NT>
__host__ __device__ constexpr int fx_xs() { return NT >= 8 ? 2 : 3; }   // x stages (64 k each) per K part (2 at NT 8: two blocks per SM)
constexpr int FX_XR = 72;         // bf16 per staged token row (64 + 8 pad: 144 B, conflict-free 16 B reads)
__device__ __forceinline__ void fx_cp16(void* s, const void* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned)__cvta_generic_to_shared(s)), "l"(g));
}
__device__ __forceinline__ void fx_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void fx_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void fx_bar(int id) { asm volatile("bar.sync %0, 128;\n" ::"r"(id)); }

// one warp of a K part (4 warps = 128 threads, named barrier 1 + kh): 16 weight rows row0 .. row0 + 15 of a matrix
// [N][K] (codes q, fp16 d per 256-block ds), blocks kb0 .. kb1 - 1, against TOK = 8 NT tokens whose x rows are
// x + srow[i] * ldx (i < TOK; nt tiles valid); acc[j] = C fragment of tile j.
// Codes: lane t of the quad of row r loads words t W .. of the row's block (coalesced per row), FX_P blocks ahead in
// registers. x: the K part's 4 warps stage 64 k x TOK rows per step in shared memory (cp.async, XS stages).
template <int MF, int NT>
__device__ __forceinline__ void fx_warp(const uint8_t* __restrict__ q, const __half* __restrict__ ds, int K, int row0,
                                        int kb0, int kb1, const bf16* __restrict__ x, int ldx, const int* srow, int mp,
                                        bf16* xs, int kh, const void* sgrid, float (&acc)[NT][4]) {
    using F = FxF<MF>;
    constexpr int W = F::W, TOK = 8 * NT, XS = fx_xs<NT>();
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, gt = threadIdx.x & 127;
    const int KB = K / 256, nkb = kb1 - kb0, nst = 4 * nkb, nt = (mp + 7) / 8;
    uint32_t R[FX_P][2][W];
    float dd[FX_P][2];
    auto load = [&](int p, int kb) {
#pragma unroll
        for (int r = 0; r < 2; r++) {
            const size_t bi = (size_t)(row0 + g + 8 * r) * KB + kb;
            const uint32_t* src = (const uint32_t*)(q + bi * F::CB) + t * W;
#pragma unroll
            for (int i = 0; i < W; i++) R[p][r][i] = (kb < kb1 && t * W + i < F::NW) ? __ldg(src + i) : 0u;
            dd[p][r] = kb < kb1 ? __half2float(ds[bi]) : 0.f;
        }
    };
    auto xissue = [&](int st) {   // x of step st (k = kb0 * 256 + 64 st .. + 63) into stage st % XS
        if (st < nst) {
            bf16* dst = xs + (st % XS) * TOK * FX_XR;
            const int k0 = kb0 * 256 + 64 * st;
            for (int u = gt; u < mp * 8; u += 128) {   // valid rows only (columns of other tokens are unused)
                const int i = u >> 3, c = u & 7;
                fx_cp16(dst + i * FX_XR + 8 * c, x + (size_t)srow[i] * ldx + k0 + 8 * c);
            }
        }
        fx_commit();
    };
#pragma unroll
    for (int p = 0; p < FX_P; p++) load(p, kb0 + p);
#pragma unroll
    for (int s2 = 0; s2 < XS - 1; s2++) xissue(s2);
    for (int kk = 0; kk < nkb; kk++) {
        const float dA = dd[0][0], dB = dd[0][1];
#pragma unroll
        for (int sp = 0; sp < 4; sp++) {   // 64-k steps of this block
            const int st = 4 * kk + sp;
            fx_wait<XS - 2>();
            fx_bar(1 + kh);
            xissue(st + XS - 1);
            const bf16* xb = xs + (st % XS) * TOK * FX_XR + g * FX_XR + 8 * t;
#pragma unroll
            for (int k2 = 0; k2 < 2; k2++) {
                const int k = 2 * sp + k2;
                float wa[8], wb[8];
                fx_dec<MF>(R[0][0], k, t, sgrid, wa);
                fx_dec<MF>(R[0][1], k, t, sgrid, wb);
                const uint32_t a0 = fx_pk(wa[0] * dA, wa[1] * dA), a1 = fx_pk(wb[0] * dB, wb[1] * dB);
                const uint32_t a2 = fx_pk(wa[2] * dA, wa[3] * dA), a3 = fx_pk(wb[2] * dB, wb[3] * dB);
                const uint32_t e0 = fx_pk(wa[4] * dA, wa[5] * dA), e1 = fx_pk(wb[4] * dB, wb[5] * dB);
                const uint32_t e2 = fx_pk(wa[6] * dA, wa[7] * dA), e3 = fx_pk(wb[6] * dB, wb[7] * dB);
#pragma unroll
                for (int j = 0; j < NT; j++)
                    if (j < nt) {
                        const uint4 xv = *(const uint4*)(xb + j * 8 * FX_XR + 32 * k2);
                        fx_mma(acc[j], a0, a1, a2, a3, xv.x, xv.y);
                        fx_mma(acc[j], e0, e1, e2, e3, xv.z, xv.w);
                    }
            }
        }
#pragma unroll
        for (int p = 0; p + 1 < FX_P; p++) {
#pragma unroll
            for (int r = 0; r < 2; r++) {
#pragma unroll
                for (int i = 0; i < W; i++) R[p][r][i] = R[p + 1][r][i];
                dd[p][r] = dd[p + 1][r];
            }
        }
        load(FX_P - 1, kb0 + kk + FX_P);
    }
    fx_wait<0>();
}

template <int MF, int NT, int KS>
struct FxSmem {
    static constexpr int TOK = 8 * NT;
    static constexpr int GRID = FxF<MF>::GB;
    static constexpr int ROWS = TOK * 4, WT = TOK * 4;
    static constexpr int ES = TOK + 1;
    static constexpr int EPI = (KS - 1) * 64 * ES * 4;
    static constexpr int XS = fx_xs<NT>();
    static constexpr int XSB = KS * XS * TOK * FX_XR * 2;   // x stages; the epilogue slots alias them
    static constexpr int BIG = XSB > EPI ? XSB : EPI;
    static constexpr int BYTES = GRID + ROWS + WT + BIG;
};
// split-K reduction: warps of K part kh > 0 store their fragments to slot kh - 1, the kh = 0 warps add the slots in
// order (p0 + p1 + p2 + p3); returns with acc = the sum in the kh = 0 warps
template <int NT, int KS>
__device__ __forceinline__ void fx_reduce(float* E, int ES, int mt, int kh, int nt, float (&acc)[NT][4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    if (kh) {
        float* e0 = E + (size_t)(kh - 1) * 64 * ES + (mt * 16 + g) * ES + 2 * t;
#pragma unroll
        for (int j = 0; j < NT; j++)
            if (j < nt) {
                float* e = e0 + j * 8;
                e[0] = acc[j][0]; e[1] = acc[j][1]; e[8 * ES] = acc[j][2]; e[8 * ES + 1] = acc[j][3];
            }
    }
    __syncthreads();
    if (!kh) {
#pragma unroll
        for (int s = 0; s < KS - 1; s++) {
            const float* e0 = E + (size_t)s * 64 * ES + (mt * 16 + g) * ES + 2 * t;
#pragma unroll
            for (int j = 0; j < NT; j++)
                if (j < nt) {
                    const float* e = e0 + j * 8;
                    acc[j][0] += e[0]; acc[j][1] += e[1]; acc[j][2] += e[8 * ES]; acc[j][3] += e[8 * ES + 1];
                }
        }
    }
}

// w1 / w3 + GLU + q_fp8. xq bf16 [*][D] (FP8-rounded rows), rows[i] = the expert's i-th token row of xq, wt[i] its
// routing weight; hq bf16 [m][INTER]. Block = 4 M tiles (w1 lo / hi, w3 lo / hi of 32 c) x KS parts of K.
template <int MF, int NT, int KS>
__global__ void __launch_bounds__(128 * KS) k_fx13(const uint8_t* __restrict__ w1q, const __half* __restrict__ w1s,
                                                   const uint8_t* __restrict__ w3q, const __half* __restrict__ w3s,
                                                   const bf16* __restrict__ xq, const int* __restrict__ rows,
                                                   const float* __restrict__ wt, int m, const uint8_t* grids, bf16* hq) {
    using SM = FxSmem<MF, NT, KS>;
    constexpr int TOK = SM::TOK, ES = SM::ES;
    extern __shared__ __align__(16) uint8_t fx_sm[];
    uint32_t* sgrid = (uint32_t*)fx_sm;
    int* srow = (int*)(fx_sm + SM::GRID);
    float* swt = (float*)(fx_sm + SM::GRID + SM::ROWS);
    float* E = (float*)(fx_sm + SM::GRID + SM::ROWS + SM::WT);
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int c0 = blockIdx.x * 32, tok0 = blockIdx.y * TOK, mp = min(TOK, m - tok0), nt = (mp + 7) / 8;
    for (int i = threadIdx.x; i < TOK; i += blockDim.x) {
        srow[i] = rows[tok0 + min(i, mp - 1)];
        swt[i] = i < mp ? wt[tok0 + i] : 0.f;
    }
    gg_load_grid<MF>(grids, sgrid);   // (syncs the block)
    const int mt = w & 3, kh = w >> 2;
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    constexpr int KB = D / 256;
    fx_warp<MF, NT>(mt < 2 ? w1q : w3q, mt < 2 ? w1s : w3s, D, c0 + 16 * (mt & 1), kh * KB / KS, (kh + 1) * KB / KS,
                    xq, D, srow, mp, (bf16*)E + kh * SM::XS * TOK * FX_XR, kh, sgrid, acc);
    __syncthreads();   // x stages done: the epilogue slots alias them
    fx_reduce<NT, KS>(E, ES, mt, kh, nt, acc);
    __syncthreads();   // all slot reads done: slot 0 takes the sums
    if (!kh) {
        float* e0 = E + (mt * 16 + g) * ES + 2 * t;
#pragma unroll
        for (int j = 0; j < NT; j++)
            if (j < nt) {
                float* e = e0 + j * 8;
                e[0] = acc[j][0]; e[1] = acc[j][1]; e[8 * ES] = acc[j][2]; e[8 * ES + 1] = acc[j][3];
            }
    }
    __syncthreads();
    // GLU (k_glu_rows) + q_fp8 (k_q_fp8: the 32 values of a warp = one block of 32 c)
    const int c = lane;
    for (int i = w; i < mp; i += 4 * KS) {
        const float h1 = E[((c >> 4) * 16 + (c & 15)) * ES + i], h3 = E[((2 + (c >> 4)) * 16 + (c & 15)) * ES + i];
        const float gg = fminf(h1, CLAMP);
        const float u = fminf(fmaxf(h3, -CLAMP), CLAMP);
        const float v = swt[i] * (siluf_(gg) * u);
        const float am = fmaxf(warp_max(fabsf(v)), 1e-4f);
        const float s = pow2ceil_f(am * (1.f / 448.f));
        const float y = e4m3_rne(fminf(fmaxf(v / s, -448.f), 448.f)) * s;
        hq[(size_t)(tok0 + i) * INTER + c0 + c] = __float2bfloat16(y);
    }
}

// w2: hq bf16 [m][INTER] -> Y fp32 [m][D]; block = 64 output rows (4 M tiles) x KS parts of K
template <int MF, int NT, int KS>
__global__ void __launch_bounds__(128 * KS) k_fx2(const uint8_t* __restrict__ w2q, const __half* __restrict__ w2s,
                                                  const bf16* __restrict__ hq, int m, const uint8_t* grids, float* Y) {
    using SM = FxSmem<MF, NT, KS>;
    constexpr int TOK = SM::TOK, ES = SM::ES;
    extern __shared__ __align__(16) uint8_t fx_sm[];
    uint32_t* sgrid = (uint32_t*)fx_sm;
    int* srow = (int*)(fx_sm + SM::GRID);
    float* E = (float*)(fx_sm + SM::GRID + SM::ROWS + SM::WT);
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int n0 = blockIdx.x * 64, tok0 = blockIdx.y * TOK, mp = min(TOK, m - tok0), nt = (mp + 7) / 8;
    for (int i = threadIdx.x; i < TOK; i += blockDim.x) srow[i] = tok0 + min(i, mp - 1);
    gg_load_grid<MF>(grids, sgrid);
    const int mt = w & 3, kh = w >> 2;
    constexpr int KB = INTER / 256;
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    fx_warp<MF, NT>(w2q, w2s, INTER, n0 + 16 * mt, kh * KB / KS, (kh + 1) * KB / KS, hq, INTER, srow, mp,
                    (bf16*)E + kh * SM::XS * TOK * FX_XR, kh, sgrid, acc);
    __syncthreads();
    fx_reduce<NT, KS>(E, ES, mt, kh, nt, acc);
    if (!kh) {
        const int r0 = n0 + 16 * mt + g;
#pragma unroll
        for (int j = 0; j < NT; j++)
            if (j < nt) {
                const int i0 = j * 8 + 2 * t;
                if (i0 < mp) {
                    Y[(size_t)(tok0 + i0) * D + r0] = acc[j][0];
                    Y[(size_t)(tok0 + i0) * D + r0 + 8] = acc[j][2];
                }
                if (i0 + 1 < mp) {
                    Y[(size_t)(tok0 + i0 + 1) * D + r0] = acc[j][1];
                    Y[(size_t)(tok0 + i0 + 1) * D + r0 + 8] = acc[j][3];
                }
            }
    }
}

// once per device (current one): load both kernels (lazy module loading) and opt in to > 48 KB dynamic shared memory
template <int MF, int NT, int KS>
// review pf11 L7: false (error cleared) when the device refuses (e.g. less opt-in shared memory): the load-time warm-up
// then turns the fused path off instead of failing the load
inline bool fx_attr() {
    constexpr int B = FxSmem<MF, NT, KS>::BYTES;
    static std::atomic<bool> set[16] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (set[dev & 15].load(std::memory_order_acquire)) return true;
    cudaFuncAttributes fa;
    bool ok = cudaFuncGetAttributes(&fa, k_fx13<MF, NT, KS>) == cudaSuccess &&
              cudaFuncGetAttributes(&fa, k_fx2<MF, NT, KS>) == cudaSuccess;
    if (ok && B > 48 * 1024)
        ok = cudaFuncSetAttribute(k_fx13<MF, NT, KS>, cudaFuncAttributeMaxDynamicSharedMemorySize, B) == cudaSuccess &&
             cudaFuncSetAttribute(k_fx2<MF, NT, KS>, cudaFuncAttributeMaxDynamicSharedMemorySize, B) == cudaSuccess;
    if (!ok) {
        cudaGetLastError();
        return false;
    }
    set[dev & 15].store(true, std::memory_order_release);
    return true;
}
template <int MF, int NT, int KS>
inline void fx_launch_nt(int which, const MatView* mv, const bf16* xq, const int* rows, const float* wt, int m,
                         const uint8_t* grids, bf16* hq, float* Y, cudaStream_t st) {
    constexpr int B = FxSmem<MF, NT, KS>::BYTES, TOK = 8 * NT;
    static_assert(B <= 99 * 1024, "fx: shared memory limit");
    DCHECK((fx_attr<MF, NT, KS>()), "fx: kernel attributes (the load-time warm-up passed)");
    const unsigned np = (unsigned)((m + TOK - 1) / TOK);
    if (which == 0)
        k_fx13<MF, NT, KS><<<dim3(INTER / 32, np), 128 * KS, B, st>>>(mv[0].q, (const __half*)mv[0].s, mv[1].q,
                                                                     (const __half*)mv[1].s, xq, rows, wt, m, grids, hq);
    else
        k_fx2<MF, NT, KS><<<dim3(D / 64, np), 128 * KS, B, st>>>(mv[2].q, (const __half*)mv[2].s, hq, m, grids, Y);
    DCK(cudaGetLastError());
}
template <int MF>
inline void fx_launch_mf(int which, const MatView* mv, const bf16* xq, const int* rows, const float* wt, int m,
                         const uint8_t* grids, bf16* hq, float* Y, cudaStream_t st) {
    static const bool fx16 = getenv("DSV4_FX16") != nullptr;   // (review pf2 L3) m > 32: 16 token tiles per pass
    if (m <= 16) fx_launch_nt<MF, 2, 4>(which, mv, xq, rows, wt, m, grids, hq, Y, st);
    else if (m <= 32) fx_launch_nt<MF, 4, 4>(which, mv, xq, rows, wt, m, grids, hq, Y, st);
    else if (!fx16) fx_launch_nt<MF, 8, 2>(which, mv, xq, rows, wt, m, grids, hq, Y, st);
    else fx_launch_nt<MF, 16, 1>(which, mv, xq, rows, wt, m, grids, hq, Y, st);
}
// one routed expert (GGUF expert format f, views mv): Y [m][D] = w2 . q_fp8(wt * glu(w1 x, w3 x)), x = xq rows
// which 0: k_fx13 (hq), 1: k_fx2 (Y)
inline void fx_part(XFmt f, int which, const MatView* mv, const bf16* xq, const int* rows, const float* wt, int m,
                    const uint8_t* grids, bf16* hq, float* Y, cudaStream_t st) {
    {
        const int mf = mat_fmt(f, which ? 2 : 0);
        switch (mf) {
            case XF_IQ2XXS: fx_launch_mf<XF_IQ2XXS>(which, mv, xq, rows, wt, m, grids, hq, Y, st); break;
            case XF_IQ3XXS: fx_launch_mf<XF_IQ3XXS>(which, mv, xq, rows, wt, m, grids, hq, Y, st); break;
            case XF_IQ2S: fx_launch_mf<XF_IQ2S>(which, mv, xq, rows, wt, m, grids, hq, Y, st); break;
            default: fx_launch_mf<XF_IQ3S>(which, mv, xq, rows, wt, m, grids, hq, Y, st); break;
        }
    }
}
inline void fx_expert(XFmt f, const MatView* mv, const bf16* xq, const int* rows, const float* wt, int m,
                      const uint8_t* grids, bf16* hq, float* Y, cudaStream_t st) {
    fx_part(f, 0, mv, xq, rows, wt, m, grids, hq, Y, st);
    fx_part(f, 1, mv, xq, rows, wt, m, grids, hq, Y, st);
}
template <int MF> inline bool fx_warm_mf() { return fx_attr<MF, 2, 4>() && fx_attr<MF, 4, 4>() && fx_attr<MF, 8, 2>() && fx_attr<MF, 16, 1>(); }
// at load, per card (current device): every instantiation loaded before the first prefill; false: not usable here
inline bool fx_warm() { return fx_warm_mf<XF_IQ2XXS>() && fx_warm_mf<XF_IQ3XXS>() && fx_warm_mf<XF_IQ2S>() && fx_warm_mf<XF_IQ3S>(); }
// usable when all three matrices are i-quants and w1 / w3 share a format (Maya-S: gate = up)
inline bool fx_ok(XFmt f) {
    return is_gg(f) && is_iq(mat_fmt(f, 0)) && mat_fmt(f, 0) == mat_fmt(f, 1) && is_iq(mat_fmt(f, 2));
}

}  // namespace dsv4
