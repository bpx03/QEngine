#pragma once
// [dsv4-pf10-fxl] Prefill routed expert on the IQ bytes for many rows (tiled): one block = 256 weight rows x up to 64
// tokens, 8 warps, each warp 32 weight rows (2 M tiles of mma m16n8k16) x 8 token tiles, the whole K.
//   - codes of a 256-k block for the block's 256 rows staged by cp.async in shared memory (double buffered, row
//     stride == 16 mod 32 B: conflict-free 4 / 8 B reads of 8 rows), the block's x rows (64 k per stage, 3 stages,
//     128 B rows, 16 B chunks XOR-swizzled by the row parity: conflict-free 16 B reads of a quarter warp);
//   - every weight decoded once per block in registers to the k_deq_gg4 value bf16_rn(w * d): the grid byte u placed in
//     the mantissa of 2^23 (byte_perm) and one FFMA (2^23 + u) dd - 2^23 dd = u dd with dd = db * d (u, db, d have
//     <= 8 + 5 + 11 significant bits: every product is exact in fp32, so the order of the multiplications does not
//     matter); the sign applied to the packed bf16 pair by XOR (bf16 rounding is symmetric) from a 256-entry table;
//   - K permuted inside a 32-weight sub-block as in k_fx13 (a lane's 8 weights 8t .. 8t + 7 and its 16 B x load share
//     one order): the dot is over all k, only the fp32 summation order differs from cuBLAS / k_fx.
// k_fxl13: warp w holds w1 rows c0 + 16 w .. + 15 (tile 0) and w3 rows of the same c (tile 1): GLU in registers, the
// q_fp8 block max (32 c = warps 2p, 2p + 1) through shared memory; same expressions as k_glu_rows + k_q_fp8.
// k_fxl2: warp w holds w2 rows n0 + 32 w .. + 31 -> Y fp32 [m][D].
#include "gguf_fx.cuh"

namespace dsv4 {

template <int MF>
struct FxlF {
    static constexpr int CB = FxF<MF>::CB;
    static constexpr int CH = CB % 16 == 0 ? 16 : 4;                                  // cp.async chunk of a code row
    static constexpr int CS0 = (CB + 15) / 16 * 16;
    static constexpr int CS = CS0 % 32 == 0 ? CS0 + 16 : CS0;                          // smem stride of a code row
    static constexpr int GB = FxF<MF>::GB;
};
constexpr int FXL_ROWS = 256, FXL_TOK = 64, FXL_XS = 2, FXL_XST = FXL_TOK * 256;   // x stage: 128 k x 64 tokens
template <int MF>
struct FxlSmem {
    using F = FxlF<MF>;
    static constexpr int GRID = 0, SGN = F::GB, CODE = SGN + 4096, X = CODE + 2 * FXL_ROWS * F::CS,
                         ROWS = X + FXL_XS * FXL_XST, WT = ROWS + FXL_TOK * 4, MAX = WT + FXL_TOK * 4,
                         BYTES = MAX + 8 * FXL_TOK * 4;
};
__device__ __forceinline__ void fxl_cp4(void* s, const void* g) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((unsigned)__cvta_generic_to_shared(s)), "l"(g));
}
// the block scale as fxl_dec takes it (review pf11 M1 follow-up, fxl_test fmt 3): d = -0 would give +0 from the FFMA
// (-0 + +0) where k_deq_gg4's (db * u) * d keeps the sign; -2^-146 instead makes every u * db * d an exact negative fp32
// subnormal below 2^-134 in magnitude (u * db < 2^15 for all four grids, db >= 1 / 8), which the bf16 RN turns into -0:
// bitwise the reference. Applied once per super-block load, not in the decode.
__device__ __forceinline__ float fxl_dscale(__half h) {
    const float d = __half2float(h);
    return __float_as_uint(d) == 0x80000000u ? -0x1p-146f : d;
}
// sub-block k of a code row (smem) for lane quarter h: the 8 grid bytes (j = 0..3 in g.x, 4..7 in g.y), sign8, db
template <int MF>
__device__ __forceinline__ void fxl_sub(const uint8_t* cr, int k, int h, const uint32_t* sgrid, uint2& gv, uint32_t& s,
                                        float& db) {
    if constexpr (MF == XF_IQ2XXS) {
        const uint2 v = *(const uint2*)(cr + 8 * k);
        db = (0.5f + (float)(v.y >> 28)) * 0.25f;
        gv = ((const uint2*)sgrid)[(v.x >> (8 * h)) & 255];
        s = gg_sign8((v.y >> (7 * h)) & 127);
    } else if constexpr (MF == XF_IQ3XXS) {
        const uint32_t ix = *(const uint16_t*)(cr + 8 * k + 2 * h), aux = *(const uint32_t*)(cr + 64 + 4 * k);
        db = (0.5f + (float)(aux >> 28)) * 0.5f;
        gv.x = sgrid[ix & 255];
        gv.y = sgrid[ix >> 8];
        s = gg_sign8((aux >> (7 * h)) & 127);
    } else if constexpr (MF == XF_IQ2S) {
        const uint32_t qs = *(const uint32_t*)(cr + 4 * k), sg = *(const uint32_t*)(cr + 32 + 4 * k);
        const uint32_t qh = cr[64 + k], sc = cr[72 + k];
        gv = ((const uint2*)sgrid)[((qs >> (8 * h)) & 255) | (((qh >> (2 * h)) & 3) << 8)];
        db = (0.5f + (float)(h < 2 ? sc & 15 : sc >> 4)) * 0.25f;
        s = (sg >> (8 * h)) & 255;
    } else {   // XF_IQ3S
        const uint32_t ix = *(const uint16_t*)(cr + 8 * k + 2 * h), qh = cr[64 + k];
        const uint32_t sg = *(const uint32_t*)(cr + 72 + 4 * k), scb = cr[104 + k / 2];
        db = (float)(1 + 2 * ((scb >> (4 * (k & 1))) & 15));
        gv.x = sgrid[(ix & 255) | (((qh >> (2 * h)) & 1) << 8)];
        gv.y = sgrid[(ix >> 8) | (((qh >> (2 * h + 1)) & 1) << 8)];
        s = (sg >> (8 * h)) & 255;
    }
}
// the 4 packed bf16 pairs (weights 2p, 2p + 1) of the lane's 8 weights of one row and sub-block
template <int MF>
__device__ __forceinline__ void fxl_dec(const uint8_t* cr, int k, int h, const uint32_t* sgrid, const uint4* sgn,
                                        float d, uint32_t (&P)[4]) {
    uint2 gv;
    uint32_t s;
    float db;
    fxl_sub<MF>(cr, k, h, sgrid, gv, s, db);
    const float dd = db * d, c = dd * -8388608.f;
#ifdef FXL_SGNTAB
    const uint4 sm = sgn[s];
    const uint32_t smk[4] = {sm.x, sm.y, sm.z, sm.w};
#else   // sign bits 2p, 2p + 1 -> bits 15, 31 of pair p: ((s >> 2p) * 0x40008000) & 0x80008000 (the higher bits of s only
        // reach bits 17 .. 22, no carry into 15 / 31); no shared-memory table (its random 16 B reads conflicted)
    uint32_t smk[4];
#pragma unroll
    for (int p = 0; p < 4; p++) smk[p] = ((s >> (2 * p)) * 0x40008000u) & 0x80008000u;
#endif
#pragma unroll
    for (int p = 0; p < 4; p++) {
        const uint32_t w = p < 2 ? gv.x : gv.y;
        const int j0 = (2 * p) & 3;
        const float v0 = fmaf(__int_as_float((int)__byte_perm(w, 0x4B000000u, 0x7440u | j0)), dd, c);
        const float v1 = fmaf(__int_as_float((int)__byte_perm(w, 0x4B000000u, 0x7440u | (j0 + 1))), dd, c);
        P[p] = fx_pk(v0, v1) ^ smk[p];
    }
}

// exact fast forms of the q_fp8 helpers (same values as pow2ceil_f, e4m3_rne, v / s; fxl_test checks them bitwise)
__device__ __forceinline__ float fxl_pow2ceil(float v) {   // v > 0 finite normal: 2^ceil(log2 v)
    const uint32_t b = __float_as_uint(v);
    return (b & 0x7fffffu) ? __uint_as_float((b & 0x7f800000u) + 0x00800000u) : v;
}
__device__ __forceinline__ float fxl_e4m3(float v) {   // v within +-448: RNE to e4m3 (subnormals below 2^-6)
    if (fabsf(v) < 0.015625f) return rintf(v * 512.f) * (1.f / 512.f);
    const uint32_t b = __float_as_uint(v);
    return __uint_as_float((b + 0x7ffffu + ((b >> 20) & 1u)) & 0xfff00000u);
}
__device__ __forceinline__ float fxl_q8(float v, float am) {   // = e4m3_rne(clamp(v / s)) * s, s = pow2ceil(am / 448)
    const float s = fxl_pow2ceil(am * (1.f / 448.f));
    const float r = __uint_as_float(0x7f000000u - __float_as_uint(s));   // 1 / s (exact: s a power of 2, normal)
    return fxl_e4m3(fminf(fmaxf(v * r, -448.f), 448.f)) * s;
}

// MODE 0: k_fxl13 (w1 / w3 + GLU + q_fp8 -> hq), MODE 1: k_fxl2 (w2: hq -> Y). NT token tiles of 8 per block
template <int MF, int MODE, int NT>
__global__ void __launch_bounds__(256, 1) k_fxl(const uint8_t* __restrict__ qa, const __half* __restrict__ sa,
                                                const uint8_t* __restrict__ qb, const __half* __restrict__ sb,
                                                const bf16* __restrict__ x, const int* __restrict__ rows,
                                                const float* __restrict__ wt, int m, const uint8_t* grids, bf16* hq,
                                                float* Y) {
    using F = FxlF<MF>;
    using SM = FxlSmem<MF>;
    constexpr int TS = 8 * NT;
    constexpr int K = MODE == 0 ? D : INTER, KB = K / 256, NST = K / 128, CS = F::CS, CB = F::CB, CH = F::CH;
    extern __shared__ __align__(16) uint8_t fxl_sm[];
    uint32_t* sgrid = (uint32_t*)(fxl_sm + SM::GRID);
    uint4* sgn = (uint4*)(fxl_sm + SM::SGN);
    uint8_t* scode = fxl_sm + SM::CODE;
    uint8_t* sx = fxl_sm + SM::X;
    int* srow = (int*)(fxl_sm + SM::ROWS);
    float* swt = (float*)(fxl_sm + SM::WT);
    float* smax = (float*)(fxl_sm + SM::MAX);
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    // grid x = row blocks (fast: the concurrent blocks share a few token tiles of x in L2), y = token tiles
    const int tok0 = blockIdx.y * TS, mp = min(TS, m - tok0);
    const int r0 = blockIdx.x * (MODE == 0 ? 128 : 256);   // c0 (MODE 0) or n0 (MODE 1)
    if (tid < FXL_TOK) {
        if (MODE == 0) {
            srow[tid] = tid < mp ? rows[tok0 + tid] : 0;
            swt[tid] = tid < mp ? wt[tok0 + tid] : 0.f;
        } else srow[tid] = tok0 + min(tid, mp - 1);
    }
    {
        uint32_t v[4];
#pragma unroll
        for (int p = 0; p < 4; p++) v[p] = (((tid >> (2 * p)) & 1) << 15) | (((tid >> (2 * p + 1)) & 1) << 31);
        sgn[tid] = make_uint4(v[0], v[1], v[2], v[3]);
    }
    gg_load_grid<MF>(grids, sgrid);   // (syncs the block: srow / swt / sgn visible)
    // global row of smem code row sr (warp sr / 32, tile (sr / 16) & 1, row sr & 15)
    auto grow = [&](int sr, const uint8_t*& q) -> int {
        if (MODE == 0) {
            q = (sr >> 4) & 1 ? qb : qa;
            return r0 + 16 * (sr >> 5) + (sr & 15);
        }
        q = qa;
        return r0 + sr;
    };
    // issue state fixed per thread: code row tid (L1-allocating copies: a row's chunks share sectors); x chunk tid % 16
    // (16 B of the 256 B row) of tokens tid / 16 + 16 i
    const uint8_t* qrow;
    {
        const uint8_t* q;
        const int gr = grow(tid, q);
        qrow = q + (size_t)gr * KB * CB;
    }
    uint8_t* const cdst = scode + tid * CS;
    const int xi0 = tid >> 4, xc = tid & 15;
    const bf16* xs[4];
#pragma unroll
    for (int i = 0; i < 4; i++) xs[i] = x + (size_t)srow[xi0 + 16 * i] * K + 8 * xc;
    const int xd0 = xi0 * 256 + ((xc ^ ((xi0 & 1) << 2)) << 4);
    auto cissue = [&](int kb) {   // codes of 256-k block kb -> buffer kb & 1
        uint8_t* dst = cdst + (kb & 1) * FXL_ROWS * CS;
        const uint8_t* src = qrow + (size_t)kb * CB;
#pragma unroll
        for (int c = 0; c < CB / CH; c++) {
            if constexpr (CH == 16)
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"((unsigned)__cvta_generic_to_shared(dst + 16 * c)), "l"(src + 16 * c));
            else fxl_cp4(dst + 4 * c, src + 4 * c);
        }
    };
    auto xissue = [&](int st) {   // x of step st (128 k) -> stage st & 1
        uint8_t* dst = sx + (st & 1) * FXL_XST + xd0;
#pragma unroll
        for (int i = 0; i < 4; i++)
            if (16 * i < TS && xi0 + 16 * i < mp) fx_cp16(dst + 16 * i * 256, xs[i] + 128 * st);
    };
    // this lane's 4 rows (smem code rows 32 w + 8 ri + g) and their d per block
    const __half* srcd[4];
    int gr4[4];
#pragma unroll
    for (int ri = 0; ri < 4; ri++) {
        const uint8_t* q;
        gr4[ri] = grow(32 * w + 8 * ri + g, q);
        srcd[ri] = MODE == 0 ? ((ri >> 1) ? sb : sa) : sa;
    }
    // d of the next block kept raw (converted at its first use, a block later: the load latency never stalls)
    __half dn[4];
    float dc[4];
#pragma unroll
    for (int ri = 0; ri < 4; ri++) dn[ri] = srcd[ri][(size_t)gr4[ri] * KB];
    float acc[2][NT][4];
#pragma unroll
    for (int a = 0; a < 2; a++)
#pragma unroll
        for (int j = 0; j < NT; j++) acc[a][j][0] = acc[a][j][1] = acc[a][j][2] = acc[a][j][3] = 0.f;
    cissue(0);
    xissue(0);
    fx_commit();
    for (int st = 0; st < NST; st++) {
        const int kb = st >> 1, hs = st & 1;
        if (hs == 0) {
#pragma unroll
            for (int ri = 0; ri < 4; ri++) {
                dc[ri] = fxl_dscale(dn[ri]);
                if (kb + 1 < KB) dn[ri] = srcd[ri][(size_t)gr4[ri] * KB + kb + 1];
            }
        }
        fx_wait<0>();
        __syncthreads();   // stage st and the codes of block kb landed; stage st + 1's buffer and code buffer kb + 1 free
        if (st + 1 < NST) {
#ifndef FXL_NOX   // timing experiments only
            xissue(st + 1);
#endif
#ifndef FXL_NOC
            if (hs == 0 && kb + 1 < KB) cissue(kb + 1);
#endif
        }
        fx_commit();
        const uint8_t* cbuf = scode + (kb & 1) * FXL_ROWS * CS + (32 * w + g) * CS;
        const uint8_t* xb = sx + (st & 1) * FXL_XST + g * 256;
#pragma unroll
        for (int q2 = 0; q2 < 2; q2++) {   // two pairs of sub-blocks (4 hs + 2 q2 + k2)
            // both sub-blocks of the pair decoded first (their mmas overlap the next decode), mmas ordered so that two
            // on one accumulator are 2 NT apart
            uint32_t P[2][4][4];
#pragma unroll
            for (int k2 = 0; k2 < 2; k2++)
#pragma unroll
                for (int ri = 0; ri < 4; ri++) {
#ifdef FXL_NODEC   // timing experiment only: no decode (wrong results)
                    for (int p = 0; p < 4; p++) P[k2][ri][p] = __float_as_uint(dc[ri]) + p + 4 * q2 + 16 * k2;
#else
                    fxl_dec<MF>(cbuf + 8 * ri * CS, 4 * hs + 2 * q2 + k2, t, sgrid, sgn, dc[ri], P[k2][ri]);
#endif
                }
#pragma unroll
            for (int k2 = 0; k2 < 2; k2++) {
                const int xo = ((4 * (2 * q2 + k2) + t) ^ ((g & 1) << 2)) << 4;
                uint4 xv[NT];
#pragma unroll
                for (int j = 0; j < NT; j++) xv[j] = *(const uint4*)(xb + j * 8 * 256 + xo);
#pragma unroll
                for (int j = 0; j < NT; j++)
#pragma unroll
                    for (int a = 0; a < 2; a++)
                        fx_mma(acc[a][j], P[k2][2 * a][0], P[k2][2 * a + 1][0], P[k2][2 * a][1], P[k2][2 * a + 1][1], xv[j].x, xv[j].y);
#pragma unroll
                for (int j = 0; j < NT; j++)
#pragma unroll
                    for (int a = 0; a < 2; a++)
                        fx_mma(acc[a][j], P[k2][2 * a][2], P[k2][2 * a + 1][2], P[k2][2 * a][3], P[k2][2 * a + 1][3], xv[j].z, xv[j].w);
            }
        }
    }
    if (MODE == 1) {
#pragma unroll
        for (int a = 0; a < 2; a++) {
            const int rr = r0 + 32 * w + 16 * a + g;
#pragma unroll
            for (int j = 0; j < NT; j++) {
                const int i0 = j * 8 + 2 * t;
                if (i0 < mp) {
                    Y[(size_t)(tok0 + i0) * D + rr] = acc[a][j][0];
                    Y[(size_t)(tok0 + i0) * D + rr + 8] = acc[a][j][2];
                }
                if (i0 + 1 < mp) {
                    Y[(size_t)(tok0 + i0 + 1) * D + rr] = acc[a][j][1];
                    Y[(size_t)(tok0 + i0 + 1) * D + rr + 8] = acc[a][j][3];
                }
            }
        }
        return;
    }
    // GLU (k_glu_rows) in registers: v[j][e] for c = c0 + 16 w + g + 8 (e >> 1), token j * 8 + 2 t + (e & 1)
    float v[NT][4];
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const float gg = fminf(acc[0][j][e], CLAMP);
            const float u = fminf(fmaxf(acc[1][j][e], -CLAMP), CLAMP);
            v[j][e] = swt[j * 8 + 2 * t + (e & 1)] * (siluf_(gg) * u);
        }
    // |v| max per token over the warp's 16 c, then over the 32-c block (warps 2p, 2p + 1)
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
        for (int e1 = 0; e1 < 2; e1++) {
            float a = fmaxf(fabsf(v[j][e1]), fabsf(v[j][e1 + 2]));
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 4));
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 8));
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 16));
            if (g == 0) smax[w * FXL_TOK + j * 8 + 2 * t + e1] = a;
        }
    __syncthreads();
    const int c = r0 + 16 * w + g;
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int i = j * 8 + 2 * t + (e & 1);
            if (i < mp) {
                const float am = fmaxf(fmaxf(smax[w * FXL_TOK + i], smax[(w ^ 1) * FXL_TOK + i]), 1e-4f);
                float y;
                if (am < 1e30f) y = fxl_q8(v[j][e], am);   // (finite: the fast exact form)
                else {
                    const float s = pow2ceil_f(am * (1.f / 448.f));
                    y = e4m3_rne(fminf(fmaxf(v[j][e] / s, -448.f), 448.f)) * s;
                }
                hq[(size_t)(tok0 + i) * INTER + c + 8 * (e >> 1)] = __float2bfloat16(y);
            }
        }
}

template <int MF, int MODE, int NT>
inline bool fxl_attr() {
    constexpr int B = FxlSmem<MF>::BYTES;
    static_assert(B <= 99 * 1024, "fxl: shared memory limit");
    static std::atomic<bool> set[16] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (set[dev & 15].load(std::memory_order_acquire)) return true;
    if (cudaFuncSetAttribute(k_fxl<MF, MODE, NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, B) != cudaSuccess) {
        cudaGetLastError();   // review pf11 L7: the warm-up turns the path off instead of failing the load
        return false;
    }
    set[dev & 15].store(true, std::memory_order_release);
    return true;
}
// token tiles of 8 per block (4..8) for m rows on RB row blocks: fewest waves x (tile + decode overhead), 1 block / SM
// SMs of the current device (review pf11 L2 / L8: one race-free initialiser, used by both wave models; the cards are
// one model)
inline int fxl_nsm() {
    static const int n = [] {
        int dev = 0, v = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev);
        return v > 0 ? v : 82;
    }();
    return n;
}
inline int fxl_nt(int m, int RB) {
    const int nsm = fxl_nsm();
    static const int ov = getenv("DSV4_FXLOV") ? atoi(getenv("DSV4_FXLOV")) : 16;
    static const int fnt = getenv("DSV4_FXLNT") ? std::clamp(atoi(getenv("DSV4_FXLNT")), 4, 8) : 0;
    if (fnt) return fnt;
    int best = 8;
    double bc = 1e30;
    for (int nt = 8; nt >= 4; nt--) {
        const long nb = (long)((m + 8 * nt - 1) / (8 * nt)) * RB;
        const double cost = (double)((nb + nsm - 1) / nsm) * (8 * nt + ov);
        if (cost < bc - 1e-9) bc = cost, best = nt;
    }
    return best;
}
template <int MF, int MODE, int NT>
inline void fxl_go(const MatView* mv, const bf16* xq, const int* rows, const float* wt, int m, const uint8_t* grids,
                   bf16* hq, float* Y, cudaStream_t st) {
    DCHECK((fxl_attr<MF, MODE, NT>()), "fxl: kernel attributes (the load-time warm-up passed)");
    const unsigned nb = (unsigned)((m + 8 * NT - 1) / (8 * NT));
    if (MODE == 0)
        k_fxl<MF, 0, NT><<<dim3(INTER / 128, nb), 256, FxlSmem<MF>::BYTES, st>>>(
            mv[0].q, (const __half*)mv[0].s, mv[1].q, (const __half*)mv[1].s, xq, rows, wt, m, grids, hq, Y);
    else
        k_fxl<MF, 1, NT><<<dim3(D / 256, nb), 256, FxlSmem<MF>::BYTES, st>>>(
            mv[2].q, (const __half*)mv[2].s, nullptr, nullptr, hq, nullptr, nullptr, m, grids, hq, Y);
    DCK(cudaGetLastError());
}
template <int MF, int MODE>
inline void fxl_go_nt(int nt, const MatView* mv, const bf16* xq, const int* rows, const float* wt, int m,
                      const uint8_t* grids, bf16* hq, float* Y, cudaStream_t st) {
    switch (nt) {
        case 4: fxl_go<MF, MODE, 4>(mv, xq, rows, wt, m, grids, hq, Y, st); break;
        case 5: fxl_go<MF, MODE, 5>(mv, xq, rows, wt, m, grids, hq, Y, st); break;
        case 6: fxl_go<MF, MODE, 6>(mv, xq, rows, wt, m, grids, hq, Y, st); break;
        case 7: fxl_go<MF, MODE, 7>(mv, xq, rows, wt, m, grids, hq, Y, st); break;
        default: fxl_go<MF, MODE, 8>(mv, xq, rows, wt, m, grids, hq, Y, st); break;
    }
}

// [dsv4-pf10-fxl] v2 k_fxm: warp = one M tile (16 weight rows) x NT token tiles (<= 16: 128 tokens), block = 8 warps =
// 128 rows; a decoded weight feeds NT mmas (2x k_fxl's). MODE 0: rows g / g + 8 of warp w's tile = w1 / w3 rows of the
// same c = c0 + 8 w + g (GLU in registers; the 32-c q_fp8 block = warps 4p .. 4p + 3). MODE 1: w2 rows n0 + 16 w + ..
constexpr int FXM_ROWS = 128, FXM_TOK = 128, FXM_XS = 3, FXM_XST = FXM_TOK * 128;   // x stage: 64 k x 128 tokens
template <int MF>
struct FxmSmem {
    using F = FxlF<MF>;
    static constexpr int GRID = 0, SGN = F::GB, CODE = SGN + 4096, X = CODE + 2 * FXM_ROWS * F::CS,
                         ROWS = X + FXM_XS * FXM_XST, WT = ROWS + FXM_TOK * 4, MAX = WT + FXM_TOK * 4,
                         BYTES = MAX + 8 * FXM_TOK * 4;
};
template <int MF, int MODE, int NT>
__global__ void __launch_bounds__(256, 1) k_fxm(const uint8_t* __restrict__ qa, const __half* __restrict__ sa,
                                                const uint8_t* __restrict__ qb, const __half* __restrict__ sb,
                                                const bf16* __restrict__ x, const int* __restrict__ rows,
                                                const float* __restrict__ wt, int m, const uint8_t* grids, bf16* hq,
                                                float* Y) {
    using F = FxlF<MF>;
    using SM = FxmSmem<MF>;
    constexpr int TS = 8 * NT;
    constexpr int K = MODE == 0 ? D : INTER, KB = K / 256, NST = K / 64, CS = F::CS, CB = F::CB, CH = F::CH;
    extern __shared__ __align__(16) uint8_t fxl_sm[];
    uint32_t* sgrid = (uint32_t*)(fxl_sm + SM::GRID);
    uint4* sgn = (uint4*)(fxl_sm + SM::SGN);
    uint8_t* scode = fxl_sm + SM::CODE;
    uint8_t* sx = fxl_sm + SM::X;
    int* srow = (int*)(fxl_sm + SM::ROWS);
    float* swt = (float*)(fxl_sm + SM::WT);
    float* smax = (float*)(fxl_sm + SM::MAX);
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int tok0 = blockIdx.y * TS, mp = min(TS, m - tok0);
    const int r0 = blockIdx.x * (MODE == 0 ? 64 : 128);   // c0 (MODE 0) or n0 (MODE 1)
    if (tid < FXM_TOK) {
        if (MODE == 0) {
            srow[tid] = tid < mp ? rows[tok0 + tid] : 0;
            swt[tid] = tid < mp ? wt[tok0 + tid] : 0.f;
        } else srow[tid] = tok0 + min(tid, mp - 1);
    }
    {
        uint32_t v[4];
#pragma unroll
        for (int p = 0; p < 4; p++) v[p] = (((tid >> (2 * p)) & 1) << 15) | (((tid >> (2 * p + 1)) & 1) << 31);
        sgn[tid] = make_uint4(v[0], v[1], v[2], v[3]);
    }
    gg_load_grid<MF>(grids, sgrid);   // (syncs the block)
    // smem code row sr = 16 w + i (i < 8: w1 / w2 row, i >= 8: w3 row in MODE 0)
    auto grow = [&](int sr, const uint8_t*& q, const __half*& sc) -> int {
        if (MODE == 0) {
            const bool up = sr & 8;
            q = up ? qb : qa;
            sc = up ? sb : sa;
            return r0 + 8 * (sr >> 4) + (sr & 7);
        }
        q = qa;
        sc = sa;
        return r0 + sr;
    };
    // copies fixed per thread: code row tid / 2 (chunks of parity tid & 1); x chunk tid % 8 of tokens tid / 8 + 32 i
    const uint8_t* qrow;
    {
        const uint8_t* q;
        const __half* sc;
        const int gr = grow(tid >> 1, q, sc);
        qrow = q + (size_t)gr * KB * CB;
    }
    uint8_t* const cdst = scode + (tid >> 1) * CS;
    const int xi0 = tid >> 3, xc = tid & 7;
    const bf16* xs[4];
#pragma unroll
    for (int i = 0; i < 4; i++) xs[i] = x + (size_t)srow[xi0 + 32 * i] * K + 8 * xc;
    const int xd0 = xi0 * 128 + ((xc ^ ((xi0 & 1) << 2)) << 4);
    auto cissue = [&](int kb) {
        uint8_t* dst = cdst + (kb & 1) * FXM_ROWS * CS;
        const uint8_t* src = qrow + (size_t)kb * CB;
#pragma unroll
        for (int c = 0; c < CB / CH; c++)
            if ((c & 1) == (tid & 1)) {
                if constexpr (CH == 16)
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"((unsigned)__cvta_generic_to_shared(dst + 16 * c)), "l"(src + 16 * c));
                else fxl_cp4(dst + 4 * c, src + 4 * c);
            }
    };
    auto xissue = [&](int st) {
        if (st < NST) {
            uint8_t* dst = sx + (st % FXM_XS) * FXM_XST + xd0;
#pragma unroll
            for (int i = 0; i < 4; i++)
                if (32 * i < TS && xi0 + 32 * i < mp) fx_cp16(dst + 32 * i * 128, xs[i] + 64 * st);
        }
    };
    const __half* srcd[2];
    int gr2[2];
#pragma unroll
    for (int ri = 0; ri < 2; ri++) {
        const uint8_t* q;
        gr2[ri] = grow(16 * w + 8 * ri + g, q, srcd[ri]);
    }
    __half dn[2];
    float dc[2];
#pragma unroll
    for (int ri = 0; ri < 2; ri++) dn[ri] = srcd[ri][(size_t)gr2[ri] * KB];
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    cissue(0);
    xissue(0);
    fx_commit();
    xissue(1);
    fx_commit();
    // software pipeline: the A fragments of step st + 1 are decoded between the mmas of step st (its codes and d
    // are in place: codes of block kb + 1 land with x stage 4 kb + 2, d of block kb + 1 was loaded a block ahead)
    uint32_t P[2][2][4], Pn[2][2][4];
    fx_wait<1>();
    __syncthreads();   // codes of block 0 and x stage 0 landed
#pragma unroll
    for (int ri = 0; ri < 2; ri++) {
        dc[ri] = fxl_dscale(dn[ri]);
        if (1 < KB) dn[ri] = srcd[ri][(size_t)gr2[ri] * KB + 1];
    }
#pragma unroll
    for (int k2 = 0; k2 < 2; k2++)
#pragma unroll
        for (int ri = 0; ri < 2; ri++)
            fxl_dec<MF>(scode + (16 * w + g + 8 * ri) * CS, k2, t, sgrid, sgn, dc[ri], P[k2][ri]);
    for (int st = 0; st < NST; st++) {
        const int sp = st & 3;
        if (st) {
            fx_wait<1>();
            __syncthreads();
        }
#ifndef FXL_NOX
        xissue(st + 2);
#endif
#ifndef FXL_NOC
        if (sp == 0 && (st >> 2) + 1 < KB) cissue((st >> 2) + 1);
#endif
        fx_commit();
        // next step's sub-blocks: block kn, sub-blocks 2 sn, 2 sn + 1
        const int sn = (st + 1) & 3, kn = (st + 1) >> 2;
        const bool more = st + 1 < NST;
        if (more && sn == 0) {
#pragma unroll
            for (int ri = 0; ri < 2; ri++) {
                dc[ri] = fxl_dscale(dn[ri]);
                if (kn + 1 < KB) dn[ri] = srcd[ri][(size_t)gr2[ri] * KB + kn + 1];
            }
        }
        const uint8_t* cbn = scode + (kn & 1) * FXM_ROWS * CS + (16 * w + g) * CS;
        const uint8_t* xb = sx + (st % FXM_XS) * FXM_XST + g * 128;
#pragma unroll
        for (int k2 = 0; k2 < 2; k2++) {
            const int xo = ((4 * k2 + t) ^ ((g & 1) << 2)) << 4;
#pragma unroll
            for (int j = 0; j < NT; j++) {
                const uint4 xv = *(const uint4*)(xb + j * 8 * 128 + xo);
                fx_mma(acc[j], P[k2][0][0], P[k2][1][0], P[k2][0][1], P[k2][1][1], xv.x, xv.y);
                fx_mma(acc[j], P[k2][0][2], P[k2][1][2], P[k2][0][3], P[k2][1][3], xv.z, xv.w);
            }
            if (more) {
#pragma unroll
                for (int ri = 0; ri < 2; ri++) {
#ifdef FXL_NODEC
                    for (int p = 0; p < 4; p++) Pn[k2][ri][p] = 0x3c003c00u + p + st;
#else
                    fxl_dec<MF>(cbn + 8 * ri * CS, 2 * sn + k2, t, sgrid, sgn, dc[ri], Pn[k2][ri]);
#endif
                }
            }
        }
#pragma unroll
        for (int k2 = 0; k2 < 2; k2++)
#pragma unroll
            for (int ri = 0; ri < 2; ri++)
#pragma unroll
                for (int p = 0; p < 4; p++) P[k2][ri][p] = Pn[k2][ri][p];
    }
    if (MODE == 1) {
        const int rr = r0 + 16 * w + g;
#pragma unroll
        for (int j = 0; j < NT; j++) {
            const int i0 = j * 8 + 2 * t;
            if (i0 < mp) {
                Y[(size_t)(tok0 + i0) * D + rr] = acc[j][0];
                Y[(size_t)(tok0 + i0) * D + rr + 8] = acc[j][2];
            }
            if (i0 + 1 < mp) {
                Y[(size_t)(tok0 + i0 + 1) * D + rr] = acc[j][1];
                Y[(size_t)(tok0 + i0 + 1) * D + rr + 8] = acc[j][3];
            }
        }
        return;
    }
    float v[NT][2];
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const float gg = fminf(acc[j][e], CLAMP);
            const float u = fminf(fmaxf(acc[j][e + 2], -CLAMP), CLAMP);
            v[j][e] = swt[j * 8 + 2 * t + e] * (siluf_(gg) * u);
        }
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
            float a = fabsf(v[j][e]);
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 4));
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 8));
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 16));
            if (g == 0) smax[w * FXM_TOK + j * 8 + 2 * t + e] = a;
        }
    __syncthreads();
    const int c = r0 + 8 * w + g, w4 = w & ~3;
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const int i = j * 8 + 2 * t + e;
            if (i < mp) {
                const float am = fmaxf(fmaxf(fmaxf(smax[w4 * FXM_TOK + i], smax[(w4 + 1) * FXM_TOK + i]),
                                             fmaxf(smax[(w4 + 2) * FXM_TOK + i], smax[(w4 + 3) * FXM_TOK + i])), 1e-4f);
                float y;
                if (am < 1e30f) y = fxl_q8(v[j][e], am);
                else {
                    const float s = pow2ceil_f(am * (1.f / 448.f));
                    y = e4m3_rne(fminf(fmaxf(v[j][e] / s, -448.f), 448.f)) * s;
                }
                hq[(size_t)(tok0 + i) * INTER + c] = __float2bfloat16(y);
            }
        }
}
template <int MF, int MODE, int NT>
inline void fxm_go(const MatView* mv, const bf16* xq, const int* rows, const float* wt, int m, const uint8_t* grids,
                   bf16* hq, float* Y, cudaStream_t st) {
    constexpr int B = FxmSmem<MF>::BYTES;
    static_assert(B <= 99 * 1024, "fxm: shared memory limit");
    static std::atomic<bool> set[16] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (!set[dev & 15].load(std::memory_order_acquire)) {
        DCK(cudaFuncSetAttribute(k_fxm<MF, MODE, NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, B));   // warm-up passed
        set[dev & 15].store(true, std::memory_order_release);
    }
    const unsigned nb = (unsigned)((m + 8 * NT - 1) / (8 * NT));
    if (MODE == 0)
        k_fxm<MF, 0, NT><<<dim3(INTER / 64, nb), 256, B, st>>>(mv[0].q, (const __half*)mv[0].s, mv[1].q,
                                                               (const __half*)mv[1].s, xq, rows, wt, m, grids, hq, Y);
    else
        k_fxm<MF, 1, NT><<<dim3(D / 128, nb), 256, B, st>>>(mv[2].q, (const __half*)mv[2].s, nullptr, nullptr, hq,
                                                            nullptr, nullptr, m, grids, hq, Y);
    DCK(cudaGetLastError());
}
// review pf11 L3: DSV4_FXMNT forced NT rounded down to an even value in 8 .. 16 (0 / unset / unparsable: the wave model)
inline int fxm_fnt() {
    static const int v = [] {
        const char* e = getenv("DSV4_FXMNT");
        char* end = nullptr;
        const long x = e ? strtol(e, &end, 10) : 0;
        if (!e || end == e || x <= 0) return 0;
        return (int)std::clamp<long>(x, 8, 16) & ~1;
    }();
    return v;
}
inline int fxm_nt(int m, int RB) {   // NT in {8, 10, 12, 14, 16}: fewest waves x (tile + decode overhead)
    static const int ov = getenv("DSV4_FXLOV") ? atoi(getenv("DSV4_FXLOV")) : 16;
    const int fnt = fxm_fnt(), nsm = fxl_nsm();
    if (fnt) return fnt;
    int best = 16;
    double bc = 1e30;
    for (int nt = 16; nt >= 8; nt -= 2) {
        const long nb = (long)((m + 8 * nt - 1) / (8 * nt)) * RB;
        const double cost = (double)((nb + nsm - 1) / nsm) * (8 * nt + ov);
        if (cost < bc - 1e-9) bc = cost, best = nt;
    }
    return best;
}
template <int MF, int MODE>
inline void fxm_go_nt(int nt, const MatView* mv, const bf16* xq, const int* rows, const float* wt, int m,
                      const uint8_t* grids, bf16* hq, float* Y, cudaStream_t st) {
    switch (nt) {
        case 8: fxm_go<MF, MODE, 8>(mv, xq, rows, wt, m, grids, hq, Y, st); break;
        case 10: fxm_go<MF, MODE, 10>(mv, xq, rows, wt, m, grids, hq, Y, st); break;
        case 12: fxm_go<MF, MODE, 12>(mv, xq, rows, wt, m, grids, hq, Y, st); break;
        case 14: fxm_go<MF, MODE, 14>(mv, xq, rows, wt, m, grids, hq, Y, st); break;
        default: fxm_go<MF, MODE, 16>(mv, xq, rows, wt, m, grids, hq, Y, st); break;
    }
}
template <int MF>
inline void fxl_launch_mf(int which, const MatView* mv, const bf16* xq, const int* rows, const float* wt, int m,
                          const uint8_t* grids, bf16* hq, float* Y, cudaStream_t st) {
    static const bool v1 = getenv("DSV4_FXLV1") != nullptr;   // the 256-row x 64-token variant (comparison)
    if (v1) {
        if (which == 0) fxl_go_nt<MF, 0>(fxl_nt(m, INTER / 128), mv, xq, rows, wt, m, grids, hq, Y, st);
        else fxl_go_nt<MF, 1>(fxl_nt(m, D / 256), mv, xq, rows, wt, m, grids, hq, Y, st);
        return;
    }
    if (which == 0) fxm_go_nt<MF, 0>(fxm_nt(m, INTER / 64), mv, xq, rows, wt, m, grids, hq, Y, st);
    else fxm_go_nt<MF, 1>(fxm_nt(m, D / 128), mv, xq, rows, wt, m, grids, hq, Y, st);
}
inline void fxl_part(XFmt f, int which, const MatView* mv, const bf16* xq, const int* rows, const float* wt, int m,
                     const uint8_t* grids, bf16* hq, float* Y, cudaStream_t st) {
    switch (mat_fmt(f, which ? 2 : 0)) {
        case XF_IQ2XXS: fxl_launch_mf<XF_IQ2XXS>(which, mv, xq, rows, wt, m, grids, hq, Y, st); break;
        case XF_IQ3XXS: fxl_launch_mf<XF_IQ3XXS>(which, mv, xq, rows, wt, m, grids, hq, Y, st); break;
        case XF_IQ2S: fxl_launch_mf<XF_IQ2S>(which, mv, xq, rows, wt, m, grids, hq, Y, st); break;
        default: fxl_launch_mf<XF_IQ3S>(which, mv, xq, rows, wt, m, grids, hq, Y, st); break;
    }
}
// [dsv4-pf10-fxl] the prefill fused expert dispatch: k_fxm for m > DSV4_FXLMIN rows (default 112: k_fx is faster
// below ~100), k_fx up to it; DSV4_NOFXL=1: k_fx for every fused expert (the old path)
// review pf11 L3: DSV4_FXLMIN parsed strictly (unparsable or negative: the default); L7: g_fxm_off = the load-time warm-up
// failed on a card (k_fx then runs every fused expert, as with DSV4_NOFXL=1)
inline std::atomic<bool> g_fxm_off{false};
inline int fxl_min() {
    static const int v = [] {
        if (getenv("DSV4_NOFXL")) return 1 << 30;
        const char* e = getenv("DSV4_FXLMIN");
        char* end = nullptr;
        const long x = e ? strtol(e, &end, 10) : -1;
        return e && end != e && x >= 0 ? (int)std::min<long>(x, 1 << 30) : 112;
    }();
    return g_fxm_off.load(std::memory_order_relaxed) ? 1 << 30 : v;
}
inline void fxd_part(XFmt f, int which, const MatView* mv, const bf16* xq, const int* rows, const float* wt, int m,
                     const uint8_t* grids, bf16* hq, float* Y, cudaStream_t st) {
    // review pf11 L1: k_fxm's cp.async needs 16 B aligned code rows (k_fx needs 4 B): any other base takes k_fx
    const bool al = !(((uintptr_t)mv[0].q | (uintptr_t)mv[1].q | (uintptr_t)mv[2].q) & 15);
    if (m > fxl_min() && al) fxl_part(f, which, mv, xq, rows, wt, m, grids, hq, Y, st);
    else fx_part(f, which, mv, xq, rows, wt, m, grids, hq, Y, st);
}
template <int MF, int MODE, int NT>
inline bool fxm_attr1() {
    if (cudaFuncSetAttribute(k_fxm<MF, MODE, NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, FxmSmem<MF>::BYTES) == cudaSuccess)
        return true;
    cudaGetLastError();
    return false;
}
template <int MF, int MODE>
inline bool fxm_warm_mode() {
    return fxm_attr1<MF, MODE, 8>() && fxm_attr1<MF, MODE, 10>() && fxm_attr1<MF, MODE, 12>() && fxm_attr1<MF, MODE, 14>() &&
           fxm_attr1<MF, MODE, 16>();
}
template <int MF, int MODE>
inline bool fxl_warm_mode() {
    return fxl_attr<MF, MODE, 4>() && fxl_attr<MF, MODE, 5>() && fxl_attr<MF, MODE, 6>() && fxl_attr<MF, MODE, 7>() &&
           fxl_attr<MF, MODE, 8>();
}
// at load, per card: every k_fxm instance loaded (lazy module loading) before the first prefill, and the k_fxl ones when
// DSV4_FXLV1 selects them (review pf11 L7). false: the device refused one (error cleared): the caller turns k_fxm off
inline bool fxm_warm() {
    bool ok = fxm_warm_mode<XF_IQ2XXS, 0>() && fxm_warm_mode<XF_IQ3XXS, 1>() && fxm_warm_mode<XF_IQ2S, 0>() &&
              fxm_warm_mode<XF_IQ3S, 1>() && fxm_warm_mode<XF_IQ2XXS, 1>() && fxm_warm_mode<XF_IQ3XXS, 0>() &&
              fxm_warm_mode<XF_IQ2S, 1>() && fxm_warm_mode<XF_IQ3S, 0>();
    if (ok && getenv("DSV4_FXLV1"))
        ok = fxl_warm_mode<XF_IQ2XXS, 0>() && fxl_warm_mode<XF_IQ3XXS, 1>() && fxl_warm_mode<XF_IQ2S, 0>() &&
             fxl_warm_mode<XF_IQ3S, 1>() && fxl_warm_mode<XF_IQ2XXS, 1>() && fxl_warm_mode<XF_IQ3XXS, 0>() &&
             fxl_warm_mode<XF_IQ2S, 1>() && fxl_warm_mode<XF_IQ3S, 0>();
    return ok;
}
inline void fxl_expert(XFmt f, const MatView* mv, const bf16* xq, const int* rows, const float* wt, int m,
                       const uint8_t* grids, bf16* hq, float* Y, cudaStream_t st) {
    fxl_part(f, 0, mv, xq, rows, wt, m, grids, hq, Y, st);
    fxl_part(f, 1, mv, xq, rows, wt, m, grids, hq, Y, st);
}

}  // namespace dsv4
